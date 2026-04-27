#!/usr/bin/env python3
"""
Main experiment runner.

Usage:
    python -m src.run                        # all DGPs (Experiment 1)
    python -m src.run --dgp gbm             # single DGP
    python -m src.run --weekly               # Experiment 4 (daily vs weekly, Heston)
    python -m src.run --sensitivity          # sensitivity check (ρ sweep, Heston)
"""

import argparse
import json
import multiprocessing
import os
import time
import numpy as np
import torch

from src_2.config import (
    GBMParams, HestonParams, MertonParams,
    SimConfig, TrainConfig,
)
from src_2.simulate import simulate_gbm, simulate_heston, simulate_merton
from src_2.dataset import build_dataset
from src_2.pricing import HestonPricer
from src_2.models import BSDeltaHedge, HestonDeltaHedge, StructuredVolNet, MLPHedge
from src_2.train import train_nn, train_ridge, train_volnet_naive
from src_2.evaluate import evaluate


RESULTS_DIR = "results"   # default; overridden by --results-dir

# =====================================================================
# Helpers
# =====================================================================

DGP_CONFIG = {
    "gbm":    GBMParams(),
    "heston": HestonParams(),
    "merton": MertonParams(),
}

SIMULATORS = {
    "gbm":    simulate_gbm,
    "heston": simulate_heston,
    "merton": simulate_merton,
}


def _generate_data(dgp_name, sim_cfg, dgp_params=None,
                   heston_cache="cache/heston_grid.pkl"):
    """Simulate paths and build train/val/test datasets for one DGP."""
    params = dgp_params if dgp_params is not None else DGP_CONFIG[dgp_name]
    sim_fn = SIMULATORS[dgp_name]

    print(f"\n{'='*60}")
    print(f"  DGP: {dgp_name.upper()}")
    print(f"{'='*60}")

    # Simulate path pools
    splits = {}
    for split, n, seed in [
        ("train", sim_cfg.n_train, sim_cfg.seed_train),
        ("val",   sim_cfg.n_val,   sim_cfg.seed_val),
        ("test",  sim_cfg.n_test,  sim_cfg.seed_test),
    ]:
        print(f"\n  Simulating {split} paths (n={n}) …")
        paths = sim_fn(n, sim_cfg.path_length, params, seed)
        splits[split] = paths

    # Heston pricer (grid interpolation)
    heston_pricer = None
    if dgp_name == "heston":
        heston_pricer = HestonPricer(
            params.kappa, params.theta, params.xi, params.rho, params.r,
            cache_path=heston_cache,
        )

    # Build datasets
    datasets = {}
    for split in ["train", "val", "test"]:
        print(f"\n  Building {split} dataset …")
        datasets[split] = build_dataset(
            splits[split], dgp_name, params, sim_cfg,
            heston_pricer=heston_pricer,
        )
        print(f"    → {len(datasets[split]['moneyness']):,} observations")

    return datasets, heston_pricer


def _run_methods(datasets, dgp_name, train_cfg, dgp_params=None, device="cpu",
                 heston_pricer=None):
    """Train and evaluate all methods on one DGP."""
    train_d = datasets["train"]
    val_d   = datasets["val"]
    test_d  = datasets["test"]
    params  = dgp_params if dgp_params is not None else DGP_CONFIG[dgp_name]

    results = {}
    hedges = {}                              # method_name → h array

    # ---- Method 1: BS Delta ----
    print("\n  Method 1: BS Delta")
    true_sigma = params.sigma if dgp_name == "gbm" else None
    m1 = BSDeltaHedge(r=params.r, true_sigma=true_sigma)
    h1 = m1.predict(test_d)
    results["1_BS_Delta"] = evaluate(h1, test_d, train_cfg.alpha)
    hedges["1_BS_Delta"] = h1

    # ---- Method 1b: Heston Oracle Delta (Heston only) ----
    if dgp_name == "heston" and heston_pricer is not None:
        print("  Method 1b: Heston Oracle Delta")
        m1b = HestonDeltaHedge(heston_pricer)
        h1b = m1b.predict(test_d)
        results["1b_Heston_Delta"] = evaluate(h1b, test_d, train_cfg.alpha)
        hedges["1b_Heston_Delta"] = h1b

    # ---- Method 2: Structured Vol-Net ----
    print("  Method 2: Structured Vol-Net")
    m2 = StructuredVolNet(hidden_dims=train_cfg.vol_net_hidden, r=params.r)
    m2 = train_nn(m2, train_d, val_d, train_cfg,
                  loss_type="mshe", select_type="mshe", device=device)
    with torch.no_grad():
        X_test = torch.tensor(
            np.column_stack([test_d["moneyness"], test_d["tau"], test_d["real_vol"]]),
            dtype=torch.float32,
        )
        h2 = m2(X_test).numpy()
    results["2_VolNet"] = evaluate(h2, test_d, train_cfg.alpha)
    hedges["2_VolNet"] = h2

    # ---- Method 2b: Naive VolNet (two-stage) ----
    print("  Method 2b: Naive VolNet (two-stage)")
    m2b = StructuredVolNet(hidden_dims=train_cfg.vol_net_hidden, r=params.r)
    m2b = train_volnet_naive(m2b, train_d, val_d, train_cfg, device=device)
    with torch.no_grad():
        h2b = m2b(X_test).numpy()
    results["2b_VolNet_naive"] = evaluate(h2b, test_d, train_cfg.alpha)
    hedges["2b_VolNet_naive"] = h2b

    # ---- Method 3: Linear Ridge ----
    print("  Method 3: Linear Ridge")
    m3 = train_ridge(train_d, val_d, train_cfg, select_type="mshe")
    h3 = m3.predict_h(test_d)
    results["3_Ridge"] = evaluate(h3, test_d, train_cfg.alpha)
    hedges["3_Ridge"] = h3

    # ---- Method 4: MLP (MSHE / MSHE) ----
    print("  Method 4: MLP (MSHE/MSHE)")
    m4 = MLPHedge(hidden_dims=train_cfg.hidden_dims,
                  hedge_lower=train_cfg.hedge_lower,
                  hedge_upper=train_cfg.hedge_upper)
    m4 = train_nn(m4, train_d, val_d, train_cfg,
                  loss_type="mshe", select_type="mshe", device=device)
    with torch.no_grad():
        h4 = m4(X_test).numpy()
    results["4_MLP_MSHE"] = evaluate(h4, test_d, train_cfg.alpha)
    hedges["4_MLP_MSHE"] = h4

    # ---- Method 5: MLP (MSHE / CVaR selection) ----
    print("  Method 5: MLP (MSHE/CVaR sel.)")
    m5 = MLPHedge(hidden_dims=train_cfg.hidden_dims,
                  hedge_lower=train_cfg.hedge_lower,
                  hedge_upper=train_cfg.hedge_upper)
    m5 = train_nn(m5, train_d, val_d, train_cfg,
                  loss_type="mshe", select_type="cvar", device=device)
    with torch.no_grad():
        h5 = m5(X_test).numpy()
    results["5_MLP_CVaR_sel"] = evaluate(h5, test_d, train_cfg.alpha)
    hedges["5_MLP_CVaR_sel"] = h5

    # ---- Method 6: MLP (CVaR end-to-end) ----
    print("  Method 6: MLP (CVaR e2e)")
    m6 = MLPHedge(hidden_dims=train_cfg.hidden_dims,
                  hedge_lower=train_cfg.hedge_lower,
                  hedge_upper=train_cfg.hedge_upper)
    m6 = train_nn(m6, train_d, val_d, train_cfg,
                  loss_type="cvar", select_type="cvar", device=device)
    with torch.no_grad():
        h6 = m6(X_test).numpy()
    results["6_MLP_CVaR_e2e"] = evaluate(h6, test_d, train_cfg.alpha)
    hedges["6_MLP_CVaR_e2e"] = h6

    # ---- Save artifacts for analysis plots ----
    artifacts = {
        "dC": test_d["dC"],
        "dS": test_d["dS"],
        "moneyness": test_d["moneyness"],
        "tau": test_d["tau"],
        "real_vol": test_d["real_vol"],
    }
    artifacts.update({f"h_{k}": v for k, v in hedges.items()})
    return results, artifacts, m2  # return VolNet model for smile plot


# =====================================================================
# Results I/O
# =====================================================================

def _save_results(all_results, filename):
    """Save results dict to JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved → {path}")


def _print_results(all_results):
    """Pretty-print the 6 × N results table."""
    print("\n" + "=" * 72)
    print("  RESULTS")
    print("=" * 72)

    dgps = list(all_results.keys())
    # Collect all methods across DGPs (some methods may be DGP-specific)
    methods_set = []
    for dgp in dgps:
        for m in all_results[dgp]:
            if m not in methods_set:
                methods_set.append(m)
    methods = methods_set

    # Header
    header = f"{'Method':<22}"
    for dgp in dgps:
        header += f" | {'MSHE':>8} {'CVaR':>8}"
    print(header)
    print("-" * len(header))

    for m in methods:
        row = f"{m:<22}"
        for dgp in dgps:
            r = all_results[dgp].get(m)
            if r is not None:
                row += f" | {r['mshe']:8.4f} {r['cvar']:8.4f}"
            else:
                row += f" | {'---':>8} {'---':>8}"
        print(row)

    # Column headers
    print("\nColumns:", "  |  ".join(
        f"{d.upper()} (MSHE / CVaR₀.₉₅)" for d in dgps
    ))


# =====================================================================
# Multiprocessing worker
# =====================================================================

def _run_single_dgp(args_tuple):
    """Worker function for one DGP (top-level for pickling)."""
    dgp_name, sim_cfg, train_cfg, device = args_tuple
    t0 = time.time()
    datasets, hp = _generate_data(dgp_name, sim_cfg)
    results, artifacts, volnet_model = _run_methods(
        datasets, dgp_name, train_cfg, device=device, heston_pricer=hp)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez_compressed(
        os.path.join(RESULTS_DIR, f"artifacts_{dgp_name}.npz"), **artifacts)
    torch.save(volnet_model.state_dict(),
               os.path.join(RESULTS_DIR, f"volnet_{dgp_name}.pt"))
    elapsed = time.time() - t0
    print(f"\n  [{dgp_name.upper()}] done in {elapsed:.0f}s")
    return dgp_name, results


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dgp", nargs="+", default=["gbm", "heston", "merton"])
    parser.add_argument("--weekly", action="store_true",
                        help="Run Experiment 4 (daily vs weekly rebalancing under Heston)")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Sensitivity check: Heston with ρ ∈ {-0.3, -0.5, -0.7}")
    parser.add_argument("--merton-sigma-sweep", action="store_true",
                        help="Sensitivity: Merton with σ_J ∈ {0.05, 0.10, 0.15, 0.20}")
    parser.add_argument("--merton-lambda-sweep", action="store_true",
                        help="Sensitivity: Merton with λ ∈ {0.25, 0.5, 1.0, 2.0}")
    parser.add_argument("--tc", type=float, nargs="+", default=None,
                        help="Transaction cost experiment (Heston). E.g. --tc 0 0.001 0.005")
    parser.add_argument("--merton-tc", type=float, nargs="+", default=None,
                        help="Transaction cost experiment (Merton). E.g. --merton-tc 0 0.001 0.005")
    parser.add_argument("--merton-weekly", action="store_true",
                        help="Daily vs weekly rebalancing under Merton")
    parser.add_argument("--multi-seed", action="store_true",
                        help="Run Experiment 1 with multiple NN seeds for uncertainty")
    parser.add_argument("--no-realvol", action="store_true",
                        help="Ablation: replace realized vol feature with constant 0.20")
    parser.add_argument("--arch-sweep", action="store_true",
                        help="Architecture sensitivity: vary MLP width on Heston")
    parser.add_argument("--parallel", action="store_true",
                        help="Run DGPs in parallel using multiprocessing")
    parser.add_argument("--results-dir", default="results",
                        help="Directory to write results into (default: results)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    global RESULTS_DIR
    RESULTS_DIR = args.results_dir

    sim_cfg = SimConfig()
    train_cfg = TrainConfig()
    device = args.device

    # ================================================================
    # Experiment 1: Main method matrix
    # ================================================================
    all_results = {}
    if args.parallel and len(args.dgp) > 1:
        worker_args = [(dgp, sim_cfg, train_cfg, device) for dgp in args.dgp]
        with multiprocessing.Pool(processes=len(args.dgp)) as pool:
            results_list = pool.map(_run_single_dgp, worker_args)
        all_results = {dgp: res for dgp, res in results_list}
    else:
        for dgp_name in args.dgp:
            t0 = time.time()
            datasets, hp = _generate_data(dgp_name, sim_cfg)
            results, artifacts, volnet_model = _run_methods(
                datasets, dgp_name, train_cfg, device=device,
                heston_pricer=hp)
            all_results[dgp_name] = results

            # Save artifacts for distribution / smile plots
            os.makedirs(RESULTS_DIR, exist_ok=True)
            np.savez_compressed(
                os.path.join(RESULTS_DIR, f"artifacts_{dgp_name}.npz"),
                **artifacts)
            torch.save(volnet_model.state_dict(),
                       os.path.join(RESULTS_DIR, f"volnet_{dgp_name}.pt"))
            print(f"\n  [{dgp_name.upper()}] done in {time.time()-t0:.0f}s")

    _print_results(all_results)
    _save_results(all_results, "experiment1_main.json")

    # ================================================================
    # Experiment 4: Weekly rebalancing (Heston, same paths)
    # ================================================================
    if args.weekly:
        print("\n\n" + "=" * 72)
        print("  EXPERIMENT 4: Daily vs Weekly Rebalancing (Heston)")
        print("=" * 72)
        # Use SAME seeds as main experiment to isolate rebalancing effect
        weekly_cfg = SimConfig(hedge_step=5)
        datasets_w, hp_w = _generate_data("heston", weekly_cfg,
                                    heston_cache="cache/heston_grid.pkl")
        results_w, _, _ = _run_methods(datasets_w, "heston", train_cfg,
                                       device=device, heston_pricer=hp_w)

        exp4 = {}
        if "heston" in all_results:
            exp4["heston_daily"] = all_results["heston"]
        else:
            # Run daily too if not already done
            datasets_d, hp_d = _generate_data("heston", sim_cfg,
                                        heston_cache="cache/heston_grid.pkl")
            exp4["heston_daily"], _, _ = _run_methods(
                datasets_d, "heston", train_cfg, device=device,
                heston_pricer=hp_d)
        exp4["heston_weekly"] = results_w

        _print_results(exp4)
        _save_results(exp4, "experiment4_weekly.json")

    # ================================================================
    # Sensitivity: Heston ρ sweep
    # ================================================================
    if args.sensitivity:
        print("\n\n" + "=" * 72)
        print("  SENSITIVITY: Heston ρ sweep")
        print("=" * 72)
        rho_values = [-0.3, -0.5, -0.7]
        sens_results = {}

        for rho in rho_values:
            label = f"heston_rho{rho:.1f}"
            print(f"\n  --- ρ = {rho} ---")
            t0 = time.time()

            params_rho = HestonParams(rho=rho)
            cache_path = f"cache/heston_grid_rho{rho:.1f}.pkl"

            datasets, hp_rho = _generate_data(
                "heston", sim_cfg,
                dgp_params=params_rho,
                heston_cache=cache_path,
            )
            results, _, _ = _run_methods(
                datasets, "heston", train_cfg,
                dgp_params=params_rho, device=device,
                heston_pricer=hp_rho,
            )
            sens_results[label] = results
            print(f"\n  [{label}] done in {time.time()-t0:.0f}s")

        _print_results(sens_results)
        _save_results(sens_results, "sensitivity_rho.json")

    # ================================================================
    # Sensitivity: Merton σ_J sweep (jump-size / tail-thickness)
    # ================================================================
    if args.merton_sigma_sweep:
        print("\n\n" + "=" * 72)
        print("  SENSITIVITY: Merton σ_J sweep")
        print("=" * 72)
        sigma_J_values = [0.05, 0.10, 0.15, 0.20]
        sens_results = {}

        for sJ in sigma_J_values:
            label = f"merton_sigmaJ{sJ:.2f}"
            print(f"\n  --- σ_J = {sJ} ---")
            t0 = time.time()

            params_m = MertonParams(sigma_J=sJ)
            datasets, _ = _generate_data(
                "merton", sim_cfg, dgp_params=params_m,
            )
            results, _, _ = _run_methods(
                datasets, "merton", train_cfg,
                dgp_params=params_m, device=device,
            )
            sens_results[label] = results
            print(f"\n  [{label}] done in {time.time()-t0:.0f}s")

        _print_results(sens_results)
        _save_results(sens_results, "sensitivity_merton_sigmaJ.json")

    # ================================================================
    # Sensitivity: Merton λ sweep (jump-frequency)
    # ================================================================
    if args.merton_lambda_sweep:
        print("\n\n" + "=" * 72)
        print("  SENSITIVITY: Merton λ sweep")
        print("=" * 72)
        lam_values = [0.25, 0.5, 1.0, 2.0]
        sens_results = {}

        for lam in lam_values:
            label = f"merton_lam{lam:.2f}"
            print(f"\n  --- λ = {lam} ---")
            t0 = time.time()

            params_m = MertonParams(lam=lam)
            datasets, _ = _generate_data(
                "merton", sim_cfg, dgp_params=params_m,
            )
            results, _, _ = _run_methods(
                datasets, "merton", train_cfg,
                dgp_params=params_m, device=device,
            )
            sens_results[label] = results
            print(f"\n  [{label}] done in {time.time()-t0:.0f}s")

        _print_results(sens_results)
        _save_results(sens_results, "sensitivity_merton_lambda.json")

    # ================================================================
    # Transaction Cost Experiment (Heston, specified tc levels)
    # ================================================================
    if args.tc is not None:
        print("\n\n" + "=" * 72)
        print("  TRANSACTION COST EXPERIMENT (Heston)")
        print("=" * 72)
        tc_results = {}
        datasets_tc, hp_tc = _generate_data("heston", sim_cfg,
                                            heston_cache="cache/heston_grid.pkl")
        # Train once — TC is not in the training loss, so models are the
        # same regardless of tc level.  Evaluate at each tc level.
        results, artifacts, _ = _run_methods(
            datasets_tc, "heston", train_cfg, device=device,
            heston_pricer=hp_tc)

        from src_2.losses import compute_mshe_tc_np, compute_cvar_tc_np
        test_d = datasets_tc["test"]
        # h_prev = 0: each observation is an independent one-step hedge
        # starting from a flat position (consistent with proposal §2).
        h_prev = np.zeros(len(test_d["dC"]))
        S = test_d["moneyness"] * 100.0       # approximate S from moneyness

        for tc_val in args.tc:
            label = f"heston_tc{tc_val:.4f}"
            print(f"\n  --- tc = {tc_val} ---")
            tc_res = {}
            for mname in results:
                h = artifacts[f"h_{mname}"]
                tc_res[mname] = {
                    "mshe": results[mname]["mshe"],
                    "cvar": results[mname]["cvar"],
                    "mshe_tc": compute_mshe_tc_np(
                        h, test_d["dC"], test_d["dS"], h_prev, S, tc_val),
                    "cvar_tc": compute_cvar_tc_np(
                        h, test_d["dC"], test_d["dS"], h_prev, S, tc_val),
                }
            tc_results[label] = tc_res

        _print_results(tc_results)
        _save_results(tc_results, "experiment_tc.json")

    # ================================================================
    # Transaction Cost Experiment (Merton)
    # ================================================================
    if args.merton_tc is not None:
        print("\n\n" + "=" * 72)
        print("  TRANSACTION COST EXPERIMENT (Merton)")
        print("=" * 72)
        tc_results = {}
        datasets_tc, _ = _generate_data("merton", sim_cfg)
        results, artifacts, _ = _run_methods(
            datasets_tc, "merton", train_cfg, device=device)

        from src_2.losses import compute_mshe_tc_np, compute_cvar_tc_np
        test_d = datasets_tc["test"]
        h_prev = np.zeros(len(test_d["dC"]))
        S = test_d["moneyness"] * 100.0

        for tc_val in args.merton_tc:
            label = f"merton_tc{tc_val:.4f}"
            print(f"\n  --- tc = {tc_val} ---")
            tc_res = {}
            for mname in results:
                h = artifacts[f"h_{mname}"]
                tc_res[mname] = {
                    "mshe": results[mname]["mshe"],
                    "cvar": results[mname]["cvar"],
                    "mshe_tc": compute_mshe_tc_np(
                        h, test_d["dC"], test_d["dS"], h_prev, S, tc_val),
                    "cvar_tc": compute_cvar_tc_np(
                        h, test_d["dC"], test_d["dS"], h_prev, S, tc_val),
                }
            tc_results[label] = tc_res

        _print_results(tc_results)
        _save_results(tc_results, "experiment_tc_merton.json")

    # ================================================================
    # Daily vs Weekly Rebalancing (Merton, same paths)
    # ================================================================
    if args.merton_weekly:
        print("\n\n" + "=" * 72)
        print("  Daily vs Weekly Rebalancing (Merton)")
        print("=" * 72)
        weekly_cfg = SimConfig(hedge_step=5)
        datasets_w, _ = _generate_data("merton", weekly_cfg)
        results_w, _, _ = _run_methods(datasets_w, "merton", train_cfg,
                                       device=device)

        ew = {}
        if "merton" in all_results:
            ew["merton_daily"] = all_results["merton"]
        else:
            datasets_d, _ = _generate_data("merton", sim_cfg)
            ew["merton_daily"], _, _ = _run_methods(
                datasets_d, "merton", train_cfg, device=device)
        ew["merton_weekly"] = results_w

        _print_results(ew)
        _save_results(ew, "experiment4_weekly_merton.json")

    # ================================================================
    # Multi-Seed Uncertainty (Experiment 1 with multiple NN seeds)
    # ================================================================
    if args.multi_seed:
        print("\n\n" + "=" * 72)
        print("  MULTI-SEED UNCERTAINTY")
        print("=" * 72)
        SEEDS = [42, 7, 2024]
        ms_results = {}

        for dgp_name in args.dgp:
            print(f"\n  DGP: {dgp_name.upper()}")
            datasets_ms, hp_ms = _generate_data(dgp_name, sim_cfg)
            all_seed_results = []
            for seed in SEEDS:
                print(f"\n    --- seed = {seed} ---")
                torch.manual_seed(seed)
                np.random.seed(seed)
                results, _, _ = _run_methods(
                    datasets_ms, dgp_name, train_cfg, device=device,
                    heston_pricer=hp_ms)
                all_seed_results.append(results)

            aggregated = {}
            method_names = []
            for r in all_seed_results:
                for m in r:
                    if m not in method_names:
                        method_names.append(m)
            for m in method_names:
                aggregated[m] = {}
                for metric in ["mshe", "cvar"]:
                    vals = [r[m][metric] for r in all_seed_results if m in r]
                    aggregated[m][metric] = float(np.mean(vals))
                    aggregated[m][f"{metric}_std"] = float(np.std(vals))
            ms_results[dgp_name] = aggregated

        _save_results(ms_results, "experiment1_multi_seed.json")
        # Print with std
        print("\n" + "=" * 72)
        print("  MULTI-SEED RESULTS (mean ± std)")
        print("=" * 72)
        for dgp_name in ms_results:
            print(f"\n  {dgp_name.upper()}:")
            for m in ms_results[dgp_name]:
                r = ms_results[dgp_name][m]
                s_mshe = r.get("mshe_std", 0)
                s_cvar = r.get("cvar_std", 0)
                print(f"    {m:<22} MSHE={r['mshe']:.4f}±{s_mshe:.4f}  "
                      f"CVaR={r['cvar']:.4f}±{s_cvar:.4f}")

    # ================================================================
    # Realized Vol Ablation
    # ================================================================
    if args.no_realvol:
        print("\n\n" + "=" * 72)
        print("  ABLATION: Without realized vol feature")
        print("=" * 72)
        abl_results = {}
        for dgp_name in args.dgp:
            t0 = time.time()
            datasets_abl, hp_abl = _generate_data(dgp_name, sim_cfg)
            # Replace real_vol with constant
            for split in ["train", "val", "test"]:
                datasets_abl[split]["real_vol"] = np.full_like(
                    datasets_abl[split]["real_vol"], 0.20)
            results, _, _ = _run_methods(
                datasets_abl, dgp_name, train_cfg, device=device,
                heston_pricer=hp_abl)
            abl_results[dgp_name] = results
            print(f"\n  [{dgp_name.upper()}] done in {time.time()-t0:.0f}s")

        _print_results(abl_results)
        _save_results(abl_results, "experiment1_no_realvol.json")

    # ================================================================
    # Architecture Sensitivity (Heston only)
    # ================================================================
    if args.arch_sweep:
        print("\n\n" + "=" * 72)
        print("  ARCHITECTURE SENSITIVITY (Heston)")
        print("=" * 72)
        arch_configs = [
            ((32, 16), (16, 8)),
            ((64, 32), (32, 16)),   # baseline
            ((128, 64), (64, 32)),
        ]
        arch_results = {}
        datasets_arch, hp_arch = _generate_data("heston", sim_cfg)

        for mlp_h, vol_h in arch_configs:
            label = f"mlp{mlp_h}_vol{vol_h}"
            print(f"\n  --- {label} ---")
            t0 = time.time()
            tc = TrainConfig(hidden_dims=mlp_h, vol_net_hidden=vol_h)
            results, _, _ = _run_methods(
                datasets_arch, "heston", tc, device=device,
                heston_pricer=hp_arch)
            arch_results[label] = results
            print(f"\n  [{label}] done in {time.time()-t0:.0f}s")

        _print_results(arch_results)
        _save_results(arch_results, "architecture_sensitivity.json")


if __name__ == "__main__":
    main()
