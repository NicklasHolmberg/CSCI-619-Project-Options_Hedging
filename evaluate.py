"""Evaluation: compute MSHE and CVaR metrics for a method on a dataset."""

import numpy as np
from src_2.losses import compute_mshe_np, compute_cvar_np


def evaluate(h: np.ndarray, data: dict, alpha: float = 0.95) -> dict:
    """
    Given hedge ratios *h* and a dataset, compute test-set metrics.

    Returns dict with keys 'mshe' and 'cvar'.
    """
    dC = data["dC"]
    dS = data["dS"]
    return {
        "mshe": compute_mshe_np(h, dC, dS),
        "cvar": compute_cvar_np(h, dC, dS, alpha),
    }


def evaluate_by_moneyness(h: np.ndarray, data: dict, alpha: float = 0.95) -> dict:
    """Break down MSHE and CVaR by moneyness bucket.

    Buckets:
        OTM: S/K < 0.95
        ATM: 0.95 ≤ S/K ≤ 1.05
        ITM: S/K > 1.05
    """
    m = data["moneyness"]
    dC = data["dC"]
    dS = data["dS"]
    buckets = {
        "OTM": m < 0.95,
        "ATM": (m >= 0.95) & (m <= 1.05),
        "ITM": m > 1.05,
    }
    result = {}
    for label, mask in buckets.items():
        if mask.sum() == 0:
            result[label] = {"mshe": float("nan"), "cvar": float("nan"), "n": 0}
            continue
        result[label] = {
            "mshe": compute_mshe_np(h[mask], dC[mask], dS[mask]),
            "cvar": compute_cvar_np(h[mask], dC[mask], dS[mask], alpha),
            "n": int(mask.sum()),
        }
    return result


def bootstrap_ci(h: np.ndarray, data: dict, metric: str = "mshe",
                 alpha: float = 0.95, n_boot: int = 1000,
                 ci: float = 0.95, seed: int = 0) -> dict:
    """Bootstrap confidence interval for MSHE or CVaR.

    Returns dict with 'mean', 'lo', 'hi', 'std'.
    """
    rng = np.random.RandomState(seed)
    dC = data["dC"]
    dS = data["dS"]
    n = len(dC)
    vals = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if metric == "mshe":
            vals[b] = compute_mshe_np(h[idx], dC[idx], dS[idx])
        else:
            vals[b] = compute_cvar_np(h[idx], dC[idx], dS[idx], alpha)

    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 + ci) / 2 * 100
    return {
        "mean": float(np.mean(vals)),
        "lo": float(np.percentile(vals, lo_pct)),
        "hi": float(np.percentile(vals, hi_pct)),
        "std": float(np.std(vals)),
    }
