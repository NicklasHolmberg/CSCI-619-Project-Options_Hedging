"""Configuration: DGP parameters, simulation settings, training hyperparameters."""

from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
# Data-generating process parameters
# ---------------------------------------------------------------------------

@dataclass
class GBMParams:
    S0: float = 100.0
    sigma: float = 0.20
    r: float = 0.02          # risk-free rate (also used as drift under Q)


@dataclass
class HestonParams:
    S0: float = 100.0
    v0: float = 0.04         # initial variance
    kappa: float = 1.5       # mean-reversion speed
    theta: float = 0.04      # long-run variance
    xi: float = 0.3          # vol-of-vol
    rho: float = -0.7        # correlation
    r: float = 0.02


@dataclass
class MertonParams:
    S0: float = 100.0
    sigma: float = 0.15      # diffusion volatility
    lam: float = 1.0         # jump intensity (jumps/year)
    mu_J: float = -0.05      # mean of log-jump size
    sigma_J: float = 0.10    # std of log-jump size
    r: float = 0.02


# ---------------------------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    n_train: int = 5000
    n_val: int = 1500
    n_test: int = 1500
    path_length: int = 80                # trading days per path
    dt: float = 1.0 / 252.0
    moneyness_levels: Tuple[float, ...] = (0.90, 0.95, 1.00, 1.05, 1.10)
    ttm_days: Tuple[int, ...] = (14, 30, 60)
    burn_in: int = 20                    # days reserved for trailing vol
    hedge_step: int = 1                  # rebalancing frequency (1=daily, 5=weekly)
    seed_train: int = 42
    seed_val: int = 123
    seed_test: int = 456


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 200
    batch_size: int = 8192
    patience: int = 15
    hidden_dims: Tuple[int, ...] = (64, 32)
    vol_net_hidden: Tuple[int, ...] = (32, 16)
    alpha: float = 0.95                  # CVaR level
    hedge_lower: float = 0.0
    hedge_upper: float = 1.5
    weight_decay: float = 1e-4
    ridge_alphas: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    r: float = 0.02                      # risk-free rate (for vol-net delta)
