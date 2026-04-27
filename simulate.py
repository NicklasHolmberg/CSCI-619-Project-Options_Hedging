"""Monte Carlo path simulation under the risk-neutral measure (Q)."""

import numpy as np
from src_2.config import GBMParams, HestonParams, MertonParams


def simulate_gbm(n_paths: int, n_steps: int, params: GBMParams,
                 seed: int) -> dict:
    """Simulate GBM paths.  dS = r S dt + σ S dW."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    Z = rng.standard_normal((n_paths, n_steps))

    S = np.empty((n_paths, n_steps + 1))
    S[:, 0] = params.S0

    drift = (params.r - 0.5 * params.sigma**2) * dt
    diff = params.sigma * np.sqrt(dt)

    for t in range(n_steps):
        S[:, t + 1] = S[:, t] * np.exp(drift + diff * Z[:, t])

    return {"S": S}


def simulate_heston(n_paths: int, n_steps: int, params: HestonParams,
                    seed: int) -> dict:
    """Simulate Heston paths (truncated Euler for the variance process)."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0

    Z1 = rng.standard_normal((n_paths, n_steps))
    Z2 = rng.standard_normal((n_paths, n_steps))
    # Correlated Brownians
    W_S = Z1
    W_v = params.rho * Z1 + np.sqrt(1 - params.rho**2) * Z2

    S = np.empty((n_paths, n_steps + 1))
    v = np.empty((n_paths, n_steps + 1))
    S[:, 0] = params.S0
    v[:, 0] = params.v0

    for t in range(n_steps):
        v_pos = np.maximum(v[:, t], 0.0)
        sqrt_v = np.sqrt(v_pos)

        S[:, t + 1] = S[:, t] * np.exp(
            (params.r - 0.5 * v_pos) * dt + sqrt_v * np.sqrt(dt) * W_S[:, t]
        )
        v[:, t + 1] = (
            v[:, t]
            + params.kappa * (params.theta - v_pos) * dt
            + params.xi * sqrt_v * np.sqrt(dt) * W_v[:, t]
        )
        v[:, t + 1] = np.maximum(v[:, t + 1], 0.0)

    return {"S": S, "v": v}


def simulate_merton(n_paths: int, n_steps: int, params: MertonParams,
                    seed: int) -> dict:
    """Simulate Merton jump-diffusion paths."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0

    k_bar = np.exp(params.mu_J + 0.5 * params.sigma_J**2) - 1
    compensator = params.lam * k_bar

    Z = rng.standard_normal((n_paths, n_steps))
    N_jumps = rng.poisson(params.lam * dt, (n_paths, n_steps))

    S = np.empty((n_paths, n_steps + 1))
    S[:, 0] = params.S0

    for t in range(n_steps):
        # Sum of log-jump sizes for this step
        jump_log = np.zeros(n_paths)
        for i in range(n_paths):
            nj = N_jumps[i, t]
            if nj > 0:
                jump_log[i] = rng.normal(params.mu_J, params.sigma_J, nj).sum()

        S[:, t + 1] = S[:, t] * np.exp(
            (params.r - compensator - 0.5 * params.sigma**2) * dt
            + params.sigma * np.sqrt(dt) * Z[:, t]
            + jump_log
        )

    return {"S": S}


# convenience dispatcher
SIMULATORS = {
    "gbm": simulate_gbm,
    "heston": simulate_heston,
    "merton": simulate_merton,
}
