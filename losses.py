"""Loss functions: MSHE and CVaR (Rockafellar-Uryasev)."""

import torch
import numpy as np


def hedging_loss(h, dC, dS):
    """One-step hedging loss  L = ΔC − h ΔS."""
    return dC - h * dS


def hedging_loss_with_tc(h, dC, dS, h_prev, S, tc):
    """One-step hedging loss with proportional transaction costs.

    L = ΔC − h ΔS − tc · |h − h_prev| · S
    """
    return dC - h * dS - tc * torch.abs(h - h_prev) * S


def mshe_loss(h, dC, dS):
    """Mean squared hedging error."""
    L = hedging_loss(h, dC, dS)
    return torch.mean(L ** 2)


def cvar_loss(h, dC, dS, nu, alpha=0.95):
    """
    CVaR via Rockafellar-Uryasev reformulation:
        CVaR_α(L) = min_ν [ ν + 1/(1−α) E[(L−ν)⁺] ]
    nu is a learnable scalar parameter (jointly optimised).
    """
    L = hedging_loss(h, dC, dS)
    return nu + (1.0 / (1.0 - alpha)) * torch.mean(torch.relu(L - nu))


def mshe_loss_tc(h, dC, dS, h_prev, S, tc):
    """MSHE with transaction costs."""
    L = hedging_loss_with_tc(h, dC, dS, h_prev, S, tc)
    return torch.mean(L ** 2)


def cvar_loss_tc(h, dC, dS, h_prev, S, tc, nu, alpha=0.95):
    """CVaR with transaction costs."""
    L = hedging_loss_with_tc(h, dC, dS, h_prev, S, tc)
    return nu + (1.0 / (1.0 - alpha)) * torch.mean(torch.relu(L - nu))


# ---- numpy versions for evaluation ----

def compute_mshe_np(h, dC, dS):
    L = dC - h * dS
    return float(np.mean(L ** 2))


def compute_cvar_np(h, dC, dS, alpha=0.95):
    L = dC - h * dS
    threshold = np.quantile(L, alpha)
    tail = L[L >= threshold]
    return float(np.mean(tail)) if len(tail) > 0 else float(threshold)


def compute_mshe_tc_np(h, dC, dS, h_prev, S, tc):
    L = dC - h * dS - tc * np.abs(h - h_prev) * S
    return float(np.mean(L ** 2))


def compute_cvar_tc_np(h, dC, dS, h_prev, S, tc, alpha=0.95):
    L = dC - h * dS - tc * np.abs(h - h_prev) * S
    threshold = np.quantile(L, alpha)
    tail = L[L >= threshold]
    return float(np.mean(tail)) if len(tail) > 0 else float(threshold)
