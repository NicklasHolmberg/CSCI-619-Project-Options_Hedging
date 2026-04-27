"""Build hedging datasets from simulated paths + option pricing."""

import numpy as np
from scipy.optimize import brentq
from tqdm import trange

from src_2.config import SimConfig, GBMParams, HestonParams, MertonParams
from src_2.pricing import bs_call_price, merton_call_price, HestonPricer


def _trailing_realized_vol(S, t, window=20):
    """Annualised trailing realised vol at day *t* from daily prices."""
    if t < window:
        raise ValueError(f"Need at least {window} days of history, got t={t}")
    log_ret = np.log(S[:, t - window + 1 : t + 1] / S[:, t - window : t])
    return np.std(log_ret, axis=1, ddof=1) * np.sqrt(252)


def _bs_implied_vol_vec(S, K, tau, C_market, r, lo=0.01, hi=3.0):
    """Vectorised BS implied-vol inversion via Brent's method.

    For each observation, find σ such that C_BS(S,K,τ,σ) = C_market.
    Falls back to realised vol proxy if root-finding fails.
    """
    n = len(S)
    ivol = np.full(n, 0.20)  # fallback
    for i in range(n):
        s, k, t, c = float(S[i]), float(K[i]), float(tau[i]), float(C_market[i])
        if t < 1e-10 or c <= 0:
            continue
        intrinsic = max(s - k * np.exp(-r * t), 0.0)
        if c <= intrinsic + 1e-10:
            ivol[i] = lo
            continue
        try:
            def obj(sig):
                return float(bs_call_price(s, k, t, sig, r)) - c
            ivol[i] = brentq(obj, lo, hi, xtol=1e-6, maxiter=100)
        except (ValueError, RuntimeError):
            pass  # keep fallback
    return ivol


def build_dataset(paths: dict, dgp_name: str, dgp_params, sim_cfg: SimConfig,
                  heston_pricer: HestonPricer = None) -> dict:
    """
    Construct the hedging dataset from simulated paths.

    For each path, contracts are created at t = burn_in with strikes set by
    the moneyness grid.  Each contract is tracked through its life, producing
    one-step hedging observations with naturally evolving moneyness.

    Returns dict with arrays:
        moneyness  (N,)   S_t / K
        tau        (N,)   remaining time to maturity (years)
        dC         (N,)   C_{t+1} - C_t
        dS         (N,)   S_{t+1} - S_t
        real_vol   (N,)   trailing 20-day realised vol
    """
    S = paths["S"]                          # (n_paths, n_steps+1)
    v = paths.get("v", None)                # (n_paths, n_steps+1) or None
    n_paths = S.shape[0]
    dt = sim_cfg.dt
    r = dgp_params.r
    burn = sim_cfg.burn_in

    all_moneyness, all_tau, all_dC, all_dS, all_rvol, all_ivol, all_true_var = [], [], [], [], [], [], []

    # Price function dispatcher
    def _price(S_arr, K_arr, tau_arr, v_arr=None):
        if dgp_name == "gbm":
            return bs_call_price(S_arr, K_arr, tau_arr, dgp_params.sigma, r)
        elif dgp_name == "heston":
            return heston_pricer.price(S_arr, K_arr, tau_arr, v_arr)
        elif dgp_name == "merton":
            return merton_call_price(
                S_arr, K_arr, tau_arr,
                dgp_params.sigma, r,
                dgp_params.lam, dgp_params.mu_J, dgp_params.sigma_J,
            )
        else:
            raise ValueError(dgp_name)

    # Create contracts at t = burn for each (moneyness, ttm) pair
    contracts = []
    for m_target in sim_cfg.moneyness_levels:
        for ttm_d in sim_cfg.ttm_days:
            contracts.append((m_target, ttm_d))

    print(f"  Building dataset: {n_paths} paths × {len(contracts)} contracts …")

    # Process in path batches for memory efficiency
    batch = min(500, n_paths)
    for b_start in trange(0, n_paths, batch, desc="  paths", leave=False):
        b_end = min(b_start + batch, n_paths)
        S_b = S[b_start:b_end]              # (B, T+1)
        v_b = v[b_start:b_end] if v is not None else None
        B = S_b.shape[0]

        step = sim_cfg.hedge_step
        for m_target, ttm_d in contracts:
            # Strike set at burn-in day
            K = S_b[:, burn] / m_target                     # (B,)

            for d in range(0, ttm_d, step):
                t = burn + d
                if t + step >= S_b.shape[1]:
                    break
                tau_t = (ttm_d - d) * dt        # remaining TTM at t
                tau_t1 = (ttm_d - d - step) * dt

                if tau_t1 < 1e-10:
                    continue

                S_t = S_b[:, t]
                S_t1 = S_b[:, t + step]
                K_rep = K

                v_t = v_b[:, t] if v_b is not None else None
                v_t1 = v_b[:, t + step] if v_b is not None else None

                C_t = _price(S_t, K_rep, np.full(B, tau_t), v_t)
                C_t1 = _price(S_t1, K_rep, np.full(B, tau_t1), v_t1)

                dC = C_t1 - C_t
                dS = S_t1 - S_t
                moneyness = S_t / K_rep

                rvol = _trailing_realized_vol(S_b, t, window=20)

                # BS-implied vol (for M2b naive training target)
                ivol = _bs_implied_vol_vec(S_t, K_rep, np.full(B, tau_t), C_t, r)

                # Store true instantaneous variance (for Heston oracle)
                if v_t is not None:
                    all_true_var.append(v_t)
                else:
                    all_true_var.append(np.full(B, np.nan))

                all_moneyness.append(moneyness)
                all_tau.append(np.full(B, tau_t))
                all_dC.append(dC)
                all_dS.append(dS)
                all_rvol.append(rvol)
                all_ivol.append(ivol)

    return {
        "moneyness": np.concatenate(all_moneyness),
        "tau": np.concatenate(all_tau),
        "dC": np.concatenate(all_dC),
        "dS": np.concatenate(all_dS),
        "real_vol": np.concatenate(all_rvol),
        "implied_vol": np.concatenate(all_ivol),
        "true_var": np.concatenate(all_true_var),
    }
