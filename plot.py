"""Generate figures from saved experiment results."""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

RESULTS_DIR = "results"
FIGURES_DIR = "figures"


def _load(filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path) as f:
        return json.load(f)


# =====================================================================
# Experiment 1: Main 6×3 heatmap + grouped bar
# =====================================================================

def plot_experiment1():
    """Grouped bar chart: MSHE and CVaR across methods and DGPs."""
    data = _load("experiment1_main.json")
    dgps = list(data.keys())
    # Collect all methods across DGPs
    methods_set = []
    for dgp in dgps:
        for m in data[dgp]:
            if m not in methods_set:
                methods_set.append(m)
    methods = methods_set
    method_labels = [m.replace("_", " ") for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in zip(axes, ["mshe", "cvar"],
                                  ["MSHE", r"CVaR$_{0.95}$"]):
        x = np.arange(len(methods))
        width = 0.25
        for i, dgp in enumerate(dgps):
            vals = [data[dgp].get(m, {}).get(metric, 0) for m in methods]
            ax.bar(x + i * width, vals, width, label=dgp.upper())
        ax.set_xticks(x + width)
        ax.set_xticklabels(method_labels, rotation=35, ha="right")
        ax.set_ylabel(title)
        ax.set_title(f"{title} by Method and DGP")
        ax.legend()

    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, "experiment1_bars.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "experiment1_bars.png"))
    plt.close(fig)
    print("  → figures/experiment1_bars.pdf")


# =====================================================================
# Experiment 2 & 3: Pairwise comparison plots
# =====================================================================

def plot_experiment2_3():
    """
    Exp 2: Method 5 vs 6 (light-touch vs e2e).
    Exp 3: Method 2 vs 2b vs 4 (architecture + decision-aware vs naive).
    Dot-line plots across DGPs.
    """
    data = _load("experiment1_main.json")
    dgps = list(data.keys())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Exp 2: light-touch (5) vs e2e (6)
    ax = axes[0]
    for metric, marker, ls in [("mshe", "o", "-"), ("cvar", "s", "--")]:
        v5 = [data[d]["5_MLP_CVaR_sel"][metric] for d in dgps]
        v6 = [data[d]["6_MLP_CVaR_e2e"][metric] for d in dgps]
        ax.plot(dgps, v5, marker=marker, ls=ls, label=f"M5 ({metric.upper()})")
        ax.plot(dgps, v6, marker=marker, ls=ls, label=f"M6 ({metric.upper()})")
    ax.set_title("Exp 2: Light-touch (M5) vs End-to-end (M6)")
    ax.set_ylabel("Metric value")
    ax.legend()

    # Exp 3: architecture (2 vs 2b vs 4) — decision-aware vs naive vs MLP
    ax = axes[1]
    for metric, marker, ls in [("mshe", "o", "-"), ("cvar", "s", "--")]:
        v2 = [data[d]["2_VolNet"][metric] for d in dgps]
        v4 = [data[d]["4_MLP_MSHE"][metric] for d in dgps]
        ax.plot(dgps, v2, marker=marker, ls=ls, label=f"M2 ({metric.upper()})")
        ax.plot(dgps, v4, marker=marker, ls=ls, label=f"M4 ({metric.upper()})")
        # Include M2b if present
        if all("2b_VolNet_naive" in data[d] for d in dgps):
            v2b = [data[d]["2b_VolNet_naive"][metric] for d in dgps]
            ax.plot(dgps, v2b, marker=marker, ls=":",
                    label=f"M2b ({metric.upper()})")
    ax.set_title("Exp 3: VolNet (M2) vs Naive (M2b) vs MLP (M4)")
    ax.set_ylabel("Metric value")
    ax.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, "experiment2_3.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "experiment2_3.png"))
    plt.close(fig)
    print("  → figures/experiment2_3.pdf")


# =====================================================================
# Experiment 4: Daily vs Weekly
# =====================================================================

def plot_experiment4():
    """Grouped bar: daily vs weekly Heston."""
    data = _load("experiment4_weekly.json")
    labels = list(data.keys())
    # Collect all methods across both conditions
    methods_set = []
    for lbl in labels:
        for m in data[lbl]:
            if m not in methods_set:
                methods_set.append(m)
    methods = methods_set
    method_labels = [m.replace("_", " ") for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in zip(axes, ["mshe", "cvar"],
                                  ["MSHE", r"CVaR$_{0.95}$"]):
        x = np.arange(len(methods))
        width = 0.35
        for i, lbl in enumerate(labels):
            vals = [data[lbl].get(m, {}).get(metric, 0) for m in methods]
            ax.bar(x + i * width, vals, width, label=lbl.replace("_", " ").title())
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(method_labels, rotation=30, ha="right")
        ax.set_ylabel(title)
        ax.set_title(f"Exp 4: {title} — Daily vs Weekly (Heston)")
        ax.legend()

    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, "experiment4_weekly.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "experiment4_weekly.png"))
    plt.close(fig)
    print("  → figures/experiment4_weekly.pdf")


# =====================================================================
# Sensitivity: ρ sweep
# =====================================================================

def plot_sensitivity():
    """Line plot of MSHE and CVaR across ρ values. Excludes Ridge to avoid axis distortion."""
    data = _load("sensitivity_rho.json")
    labels = sorted(data.keys())
    rho_vals = [float(l.split("rho")[1]) for l in labels]
    methods = list(data[labels[0]].keys())
    # Exclude Ridge — its values can be orders of magnitude larger
    methods = [m for m in methods if "ridge" not in m.lower()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in zip(axes, ["mshe", "cvar"],
                                  ["MSHE", r"CVaR$_{0.95}$"]):
        for m in methods:
            vals = [data[l][m][metric] for l in labels if m in data[l]]
            ax.plot(rho_vals[:len(vals)], vals, marker="o",
                    label=m.replace("_", " "))
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(title)
        ax.set_title(f"Sensitivity: {title} vs Heston ρ")
        ax.legend(fontsize=7)

    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, "sensitivity_rho.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "sensitivity_rho.png"))
    plt.close(fig)
    print("  → figures/sensitivity_rho.pdf")


# =====================================================================
# Loss distribution histograms
# =====================================================================

def plot_loss_distributions():
    """Overlay histograms of hedging loss L = ΔC − h·ΔS for key methods."""
    dgp_names = ["gbm", "heston", "merton"]
    # Focus on representative methods (include M2b if present)
    method_keys = ["1_BS_Delta", "2_VolNet", "4_MLP_MSHE", "6_MLP_CVaR_e2e"]
    method_labels = ["BS Delta", "VolNet (M2)", "MLP MSHE", "MLP CVaR e2e"]
    colors = ["#4C72B0", "#C44E52", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

    for ax, dgp in zip(axes, dgp_names):
        path = os.path.join(RESULTS_DIR, f"artifacts_{dgp}.npz")
        if not os.path.exists(path):
            ax.set_title(f"{dgp.upper()} (no data)")
            continue

        art = np.load(path)
        dC = art["dC"]
        dS = art["dS"]

        for mk, ml, c in zip(method_keys, method_labels, colors):
            key = f"h_{mk}"
            if key not in art:
                continue
            h = art[key]
            losses = dC - h * dS
            ax.hist(losses, bins=80, alpha=0.45, label=ml, color=c,
                    density=True, edgecolor="none")

        ax.set_xlabel("Hedging loss $L = \\Delta C - h \\cdot \\Delta S$")
        ax.set_ylabel("Density")
        ax.set_title(f"{dgp.upper()}")
        ax.legend(fontsize=8)
        ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)

    fig.suptitle("Hedging Loss Distributions by DGP", y=1.02, fontsize=13)
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, "loss_distributions.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_DIR, "loss_distributions.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("  → figures/loss_distributions.pdf")


# =====================================================================
# VolNet learned implied volatility smile
# =====================================================================

def plot_volnet_smile():
    """Heatmap of σ̂(moneyness, τ) learned by StructuredVolNet per DGP."""
    from src_2.models import StructuredVolNet
    from src_2.config import TrainConfig, GBMParams

    train_cfg = TrainConfig()
    dgp_names = ["gbm", "heston", "merton"]
    r = GBMParams().r  # same across DGPs

    # Evaluation grid
    moneyness = np.linspace(0.85, 1.15, 80)
    tau = np.linspace(0.02, 0.30, 60)
    M, T = np.meshgrid(moneyness, tau)
    # Use median realized vol as constant for the grid
    default_real_vol = 0.20

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, dgp in zip(axes, dgp_names):
        pt_path = os.path.join(RESULTS_DIR, f"volnet_{dgp}.pt")
        if not os.path.exists(pt_path):
            ax.set_title(f"{dgp.upper()} (no model)")
            continue

        model = StructuredVolNet(hidden_dims=train_cfg.vol_net_hidden, r=r)
        model.load_state_dict(torch.load(pt_path, weights_only=True))
        model.eval()

        X_grid = torch.tensor(
            np.column_stack([
                M.ravel(),
                T.ravel(),
                np.full(M.size, default_real_vol),
            ]),
            dtype=torch.float32,
        )

        with torch.no_grad():
            sigma_hat = (
                torch.nn.functional.softplus(model.net(X_grid).squeeze(-1))
                + 1e-4
            ).numpy()

        sigma_grid = sigma_hat.reshape(T.shape)

        im = ax.contourf(M, T, sigma_grid, levels=25, cmap="RdYlBu_r")
        fig.colorbar(im, ax=ax, label=r"$\hat\sigma$")
        ax.set_xlabel("Moneyness $S/K$")
        ax.set_ylabel(r"TTM $\tau$ (years)")
        ax.set_title(f"{dgp.upper()}: Learned $\\hat\\sigma$")

    fig.suptitle("VolNet Implied Volatility Surface", y=1.02, fontsize=13)
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, "volnet_smile.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_DIR, "volnet_smile.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("  → figures/volnet_smile.pdf")


# =====================================================================
# Transaction cost sweep
# =====================================================================

def plot_tc_sweep():
    """Line plot of MSHE and CVaR across TC levels. Excludes Ridge."""
    data = _load("experiment_tc.json")
    labels = sorted(data.keys())
    tc_vals = [float(l.split("tc")[1]) for l in labels]
    methods = list(data[labels[0]].keys())
    methods = [m for m in methods if "ridge" not in m.lower()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in zip(axes, ["mshe", "cvar"],
                                  ["MSHE", r"CVaR$_{0.95}$"]):
        for m in methods:
            vals = [data[l][m][metric] for l in labels if m in data[l]]
            ax.plot(tc_vals[:len(vals)], vals, marker="o",
                    label=m.replace("_", " "))
        ax.set_xlabel(r"Transaction cost $\kappa$")
        ax.set_ylabel(title)
        ax.set_title(f"TC Sweep (Heston): {title}")
        ax.legend(fontsize=7)

    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, "tc_sweep.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "tc_sweep.png"))
    plt.close(fig)
    print("  → figures/tc_sweep.pdf")


# =====================================================================
# Multi-seed: Experiment 1 with error bars
# =====================================================================

def plot_multi_seed():
    """Grouped bar chart with error bars from multi-seed experiment."""
    data = _load("experiment1_multi_seed.json")
    dgps = list(data.keys())
    methods_set = []
    for dgp in dgps:
        for m in data[dgp]:
            if m not in methods_set:
                methods_set.append(m)
    methods = methods_set
    # Exclude Ridge for readability (off-scale)
    methods_no_ridge = [m for m in methods if "ridge" not in m.lower()]
    method_labels = [m.replace("_", " ") for m in methods_no_ridge]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, metric, title in zip(axes, ["mshe", "cvar"],
                                  ["MSHE", r"CVaR$_{0.95}$"]):
        x = np.arange(len(methods_no_ridge))
        width = 0.25
        for i, dgp in enumerate(dgps):
            vals = [data[dgp].get(m, {}).get(metric, 0) for m in methods_no_ridge]
            errs = [data[dgp].get(m, {}).get(f"{metric}_std", 0) for m in methods_no_ridge]
            ax.bar(x + i * width, vals, width, yerr=errs, capsize=3,
                   label=dgp.upper(), alpha=0.85)
        ax.set_xticks(x + width)
        ax.set_xticklabels(method_labels, rotation=35, ha="right")
        ax.set_ylabel(title)
        ax.set_title(f"{title} by Method and DGP (Multi-Seed ± 1 SD)")
        ax.legend()

    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, "multi_seed_bars.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "multi_seed_bars.png"))
    plt.close(fig)
    print("  → figures/multi_seed_bars.pdf")


# =====================================================================
# No-realvol ablation
# =====================================================================

def plot_no_realvol():
    """Side-by-side comparison: with vs without realised-vol feature."""
    data_with = _load("experiment1_main.json")
    data_without = _load("experiment1_no_realvol.json")
    dgps = list(data_with.keys())
    methods = [m for m in data_with[dgps[0]].keys() if "ridge" not in m.lower()]
    method_labels = [m.replace("_", " ") for m in methods]

    fig, axes = plt.subplots(len(dgps), 2, figsize=(13, 4 * len(dgps)),
                             squeeze=False)

    for row, dgp in enumerate(dgps):
        for col, (metric, title) in enumerate(
                [("mshe", "MSHE"), ("cvar", r"CVaR$_{0.95}$")]):
            ax = axes[row, col]
            x = np.arange(len(methods))
            width = 0.35
            v_with = [data_with[dgp].get(m, {}).get(metric, 0) for m in methods]
            v_without = [data_without.get(dgp, {}).get(m, {}).get(metric, 0)
                         for m in methods]
            ax.bar(x - width / 2, v_with, width, label="With RealVol", alpha=0.85)
            ax.bar(x + width / 2, v_without, width, label="No RealVol", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(method_labels, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel(title)
            ax.set_title(f"{dgp.upper()} — {title}")
            ax.legend(fontsize=8)

    fig.suptitle("Ablation: With vs Without Realised Volatility Feature",
                 y=1.01, fontsize=13)
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, "no_realvol_ablation.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_DIR, "no_realvol_ablation.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("  → figures/no_realvol_ablation.pdf")


# =====================================================================
# Architecture sensitivity
# =====================================================================

def plot_arch_sweep():
    """Grouped bar comparing architecture sizes (Heston)."""
    data = _load("architecture_sensitivity.json")
    archs = list(data.keys())
    methods = [m for m in data[archs[0]].keys() if "ridge" not in m.lower()]
    method_labels = [m.replace("_", " ") for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric, title in zip(axes, ["mshe", "cvar"],
                                  ["MSHE", r"CVaR$_{0.95}$"]):
        x = np.arange(len(methods))
        width = 0.8 / len(archs)
        for i, arch in enumerate(archs):
            vals = [data[arch].get(m, {}).get(metric, 0) for m in methods]
            ax.bar(x + i * width, vals, width, label=arch, alpha=0.85)
        ax.set_xticks(x + width * (len(archs) - 1) / 2)
        ax.set_xticklabels(method_labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(title)
        ax.set_title(f"Architecture Sweep (Heston): {title}")
        ax.legend(fontsize=7)

    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, "arch_sweep.pdf"))
    fig.savefig(os.path.join(FIGURES_DIR, "arch_sweep.png"))
    plt.close(fig)
    print("  → figures/arch_sweep.pdf")


# =====================================================================
# Experiment 1 heatmap (compact table-style view)
# =====================================================================

def plot_experiment1_heatmap():
    """Heatmap-style table of MSHE and CVaR across methods × DGPs."""
    data = _load("experiment1_main.json")
    dgps = list(data.keys())
    methods_set = []
    for dgp in dgps:
        for m in data[dgp]:
            if m not in methods_set:
                methods_set.append(m)
    methods = methods_set
    method_labels = [m.replace("_", " ") for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, title in zip(axes, ["mshe", "cvar"],
                                  ["MSHE", r"CVaR$_{0.95}$"]):
        matrix = np.zeros((len(methods), len(dgps)))
        for j, dgp in enumerate(dgps):
            for i, m in enumerate(methods):
                matrix[i, j] = data[dgp].get(m, {}).get(metric, np.nan)

        # Clip Ridge for better color scale
        vmax = np.nanpercentile(matrix[~np.isnan(matrix)], 90)
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=vmax)
        ax.set_xticks(range(len(dgps)))
        ax.set_xticklabels([d.upper() for d in dgps])
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(method_labels, fontsize=9)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Annotate cells
        for i in range(len(methods)):
            for j in range(len(dgps)):
                val = matrix[i, j]
                if not np.isnan(val):
                    txt = f"{val:.4f}" if val < 1 else f"{val:.2f}"
                    color = "white" if val > vmax * 0.6 else "black"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=7, color=color)

    fig.suptitle("Experiment 1: Method × DGP Performance", y=1.01, fontsize=13)
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, "experiment1_heatmap.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_DIR, "experiment1_heatmap.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("  → figures/experiment1_heatmap.pdf")


# =====================================================================
# Decision-aware gap: CVaR improvement of M6 over M4 across DGPs
# =====================================================================

def plot_decision_gap():
    """Bar chart showing % CVaR improvement of M6 over M4 by DGP."""
    data = _load("experiment1_main.json")
    dgps = list(data.keys())

    comparisons = [
        ("6_MLP_CVaR_e2e vs 4_MLP_MSHE", "6_MLP_CVaR_e2e", "4_MLP_MSHE"),
        ("5_MLP_CVaR_sel vs 4_MLP_MSHE", "5_MLP_CVaR_sel", "4_MLP_MSHE"),
        ("2_VolNet vs 2b_VolNet_naive", "2_VolNet", "2b_VolNet_naive"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric, title in zip(axes, ["mshe", "cvar"],
                                  ["MSHE", r"CVaR$_{0.95}$"]):
        x = np.arange(len(dgps))
        width = 0.25
        for i, (label, better, baseline) in enumerate(comparisons):
            pct_diff = []
            for dgp in dgps:
                b_val = data[dgp].get(better, {}).get(metric, np.nan)
                base_val = data[dgp].get(baseline, {}).get(metric, np.nan)
                if base_val > 0:
                    pct_diff.append((base_val - b_val) / base_val * 100)
                else:
                    pct_diff.append(0)
            ax.bar(x + i * width, pct_diff, width, label=label, alpha=0.85)

        ax.set_xticks(x + width)
        ax.set_xticklabels([d.upper() for d in dgps])
        ax.set_ylabel(f"% Improvement in {title}")
        ax.set_title(f"Decision-Aware Gap: {title}")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.legend(fontsize=7)

    fig.suptitle("How Much Does Decision-Awareness Help?", y=1.01, fontsize=13)
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, "decision_gap.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_DIR, "decision_gap.png"),
                bbox_inches="tight")
    plt.close(fig)
    print("  → figures/decision_gap.pdf")


# =====================================================================
# CLI
# =====================================================================

def main():
    """Generate all available plots from results/ directory."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    available = os.listdir(RESULTS_DIR) if os.path.isdir(RESULTS_DIR) else []

    if "experiment1_main.json" in available:
        print("Plotting Experiment 1 bars …")
        plot_experiment1()
        print("Plotting Experiment 1 heatmap …")
        plot_experiment1_heatmap()
        print("Plotting Experiments 2 & 3 pairwise …")
        plot_experiment2_3()
        print("Plotting decision-aware gap …")
        plot_decision_gap()

    # Artifact-based plots (need .npz and .pt files)
    has_artifacts = any(f.startswith("artifacts_") and f.endswith(".npz")
                        for f in available)
    has_volnet = any(f.startswith("volnet_") and f.endswith(".pt")
                     for f in available)

    if has_artifacts:
        print("Plotting loss distributions …")
        plot_loss_distributions()

    if has_volnet:
        print("Plotting VolNet smile surface …")
        plot_volnet_smile()

    if "experiment4_weekly.json" in available:
        print("Plotting Experiment 4 (weekly) …")
        plot_experiment4()

    if "sensitivity_rho.json" in available:
        print("Plotting sensitivity (ρ) …")
        plot_sensitivity()

    if "experiment_tc.json" in available:
        print("Plotting TC sweep …")
        plot_tc_sweep()

    if "experiment1_multi_seed.json" in available:
        print("Plotting multi-seed bars …")
        plot_multi_seed()

    if "experiment1_no_realvol.json" in available and "experiment1_main.json" in available:
        print("Plotting no-realvol ablation …")
        plot_no_realvol()

    if "architecture_sensitivity.json" in available:
        print("Plotting architecture sweep …")
        plot_arch_sweep()

    if not available:
        print("No result files found in results/. Run experiments first.")


if __name__ == "__main__":
    main()
