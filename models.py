"""All six hedging methods."""

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm as sp_norm
from sklearn.linear_model import Ridge


# =====================================================================
# Method 1 — Black-Scholes Delta (analytic, no learning)
# =====================================================================

class BSDeltaHedge:
    """Compute BS delta using trailing realised vol (or true vol for GBM)."""

    def __init__(self, r: float = 0.02, true_sigma: float = None):
        self.r = r
        self.true_sigma = true_sigma          # if set, overrides realised vol

    def predict(self, data: dict) -> np.ndarray:
        m = data["moneyness"]
        tau = data["tau"]
        sigma = (
            np.full_like(m, self.true_sigma)
            if self.true_sigma is not None
            else data["real_vol"]
        )
        sigma = np.maximum(sigma, 1e-6)
        d1 = (np.log(m) + (self.r + 0.5 * sigma**2) * tau) / (
            sigma * np.sqrt(np.maximum(tau, 1e-10))
        )
        return sp_norm.cdf(d1)


# =====================================================================
# Method 1b — Heston Oracle Delta (numerical, uses true DGP params)
# =====================================================================

class HestonDeltaHedge:
    """Oracle Heston delta using true parameters and current variance."""

    def __init__(self, heston_pricer):
        self.pricer = heston_pricer

    def predict(self, data: dict) -> np.ndarray:
        moneyness = data["moneyness"]
        tau = data["tau"]
        v_t = data["true_var"]
        return self.pricer.delta_from_grid(moneyness, tau, v_t)


# =====================================================================
# Method 2 — Structured Volatility Network (predict σ̂ → BS Δ)
# =====================================================================

class StructuredVolNet(nn.Module):
    """Small network outputs σ̂, which feeds through differentiable BS Δ."""

    def __init__(self, hidden_dims=(32, 16), r=0.02):
        super().__init__()
        self.r = r
        layers = []
        in_dim = 3                            # moneyness, tau, real_vol
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        """X: (B, 3)  →  h: (B,)"""
        sigma_hat = nn.functional.softplus(self.net(X).squeeze(-1)) + 1e-4
        m = X[:, 0]                            # moneyness
        tau = X[:, 1]                          # TTM
        d1 = (torch.log(m) + (self.r + 0.5 * sigma_hat**2) * tau) / (
            sigma_hat * torch.sqrt(tau.clamp(min=1e-6))
        )
        # Φ(d1) via the error function
        delta = 0.5 * (1 + torch.erf(d1 / np.sqrt(2)))
        return delta


# =====================================================================
# Method 3 — Linear Ridge Hedge
# =====================================================================

class LinearRidgeHedge:
    """
    Ridge regression:  h = w^T x.
    Training trick: minimise  E[(ΔC - h ΔS)²] by regressing ΔC on x*ΔS.
    """

    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha, fit_intercept=False)

    def fit(self, data: dict):
        X = np.column_stack([data["moneyness"], data["tau"], data["real_vol"]])
        Z = X * data["dS"][:, None]           # features × ΔS
        self.model.fit(Z, data["dC"])

    def predict(self, data: dict) -> np.ndarray:
        X = np.column_stack([data["moneyness"], data["tau"], data["real_vol"]])
        return self.model.predict(X * data["dS"][:, None]) / (
            data["dS"] + 1e-30
        )

    def predict_h(self, data: dict) -> np.ndarray:
        """Return hedge ratios h directly (for evaluation)."""
        X = np.column_stack([data["moneyness"], data["tau"], data["real_vol"]])
        # h = w^T x + b  (from the regression  ΔC ≈ (w^T x) ΔS)
        return X @ self.model.coef_


# =====================================================================
# Methods 4, 5, 6 — MLP Hedge
# =====================================================================

class MLPHedge(nn.Module):
    """
    Direct MLP:  X → h ∈ [hedge_lower, hedge_upper].
    Used for Methods 4 (MSHE/MSHE), 5 (MSHE/CVaR sel.), 6 (CVaR/CVaR).
    """

    def __init__(self, hidden_dims=(64, 32), hedge_lower=0.0, hedge_upper=1.5):
        super().__init__()
        self.lo = hedge_lower
        self.hi = hedge_upper
        layers = []
        in_dim = 3
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        """X: (B, 3)  →  h: (B,)"""
        raw = self.net(X).squeeze(-1)
        return self.lo + (self.hi - self.lo) * torch.sigmoid(raw)
