"""Option pricing via QuantLib (BS, Heston) and NumPy (Merton series)."""

import os
import pickle
import numpy as np
from scipy.stats import norm
from scipy.interpolate import RegularGridInterpolator
import QuantLib as ql


# =====================================================================
# QuantLib helpers
# =====================================================================

_QL_DAY_COUNT = ql.Actual365Fixed()
_QL_CALENDAR = ql.NullCalendar()


def _ql_setup(S0, K_val, tau_years, r):
    """Create common QuantLib objects for a single-option pricing call."""
    today = ql.Settings.instance().evaluationDate
    # maturity from tau (years → days, at least 1 day)
    n_days = max(int(round(tau_years * 365)), 1)
    maturity = today + ql.Period(n_days, ql.Days)

    spot = ql.QuoteHandle(ql.SimpleQuote(float(S0)))
    rfr = ql.YieldTermStructureHandle(
        ql.FlatForward(today, float(r), _QL_DAY_COUNT)
    )
    div = ql.YieldTermStructureHandle(
        ql.FlatForward(today, 0.0, _QL_DAY_COUNT)
    )
    exercise = ql.EuropeanExercise(maturity)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, float(K_val))
    option = ql.VanillaOption(payoff, exercise)
    return option, spot, rfr, div, maturity


# =====================================================================
# Black-Scholes (QuantLib AnalyticEuropeanEngine)
# =====================================================================

def _ql_bs_price_single(S0, K_val, tau, sigma, r):
    """Price one European call via QuantLib Black-Scholes."""
    if tau < 1e-10:
        return max(S0 - K_val, 0.0)
    option, spot, rfr, div, _ = _ql_setup(S0, K_val, tau, r)
    vol = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(
            ql.Settings.instance().evaluationDate,
            _QL_CALENDAR, float(sigma), _QL_DAY_COUNT,
        )
    )
    process = ql.BlackScholesMertonProcess(spot, div, rfr, vol)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return option.NPV()


def bs_call_price(S, K, tau, sigma, r):
    """Vectorized Black-Scholes European call price (closed-form numpy)."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    tau = np.asarray(tau, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    S, K, tau, sigma = np.broadcast_arrays(S, K, tau, sigma)

    price = np.zeros_like(S)
    valid = tau > 1e-10
    s, k, t, sig = S[valid], K[valid], tau[valid], sigma[valid]
    d1 = (np.log(s / k) + (r + 0.5 * sig**2) * t) / (sig * np.sqrt(t))
    d2 = d1 - sig * np.sqrt(t)
    price[valid] = s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
    price[~valid] = np.maximum(S[~valid] - K[~valid], 0.0)
    return price


def bs_delta(S, K, tau, sigma, r):
    """Vectorized Black-Scholes delta (closed-form, fast)."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    tau = np.asarray(tau, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    S, K, tau, sigma = np.broadcast_arrays(S, K, tau, sigma)

    delta = np.zeros_like(S)
    valid = tau > 1e-10

    s, k, t, sig = S[valid], K[valid], tau[valid], sigma[valid]
    d1 = (np.log(s / k) + (r + 0.5 * sig**2) * t) / (sig * np.sqrt(t))
    delta[valid] = norm.cdf(d1)
    delta[~valid] = np.where(S[~valid] > K[~valid], 1.0, 0.0)
    return delta


# =====================================================================
# Heston — QuantLib AnalyticHestonEngine with grid interpolation
# =====================================================================

def _ql_heston_price_single(S0, K_val, tau, v0, kappa, theta, xi, rho, r):
    """Price one European call via QuantLib Heston."""
    if tau < 1e-10:
        return max(S0 - K_val, 0.0)
    option, spot, rfr, div, _ = _ql_setup(S0, K_val, tau, r)
    process = ql.HestonProcess(rfr, div, spot,
                               float(v0), float(kappa), float(theta),
                               float(xi), float(rho))
    model = ql.HestonModel(process)
    option.setPricingEngine(ql.AnalyticHestonEngine(model))
    return option.NPV()


class HestonPricer:
    """Grid-interpolation Heston pricer: QuantLib builds grid, interp for speed."""

    def __init__(self, kappa, theta, xi, rho, r, cache_path=None):
        self.kappa, self.theta, self.xi, self.rho, self.r = (
            kappa, theta, xi, rho, r,
        )
        # Set a fixed evaluation date for QuantLib
        today = ql.Date(15, 1, 2025)
        ql.Settings.instance().evaluationDate = today

        if cache_path and os.path.exists(cache_path):
            self._load_grid(cache_path)
        else:
            self._build_grid()
            if cache_path:
                self._save_grid(cache_path)

    def _build_grid(self):
        self.m_grid = np.linspace(0.70, 1.40, 50)
        self.tau_grid = np.linspace(0.5 / 252, 65 / 252, 40)
        self.v_grid = np.linspace(0.002, 0.25, 40)

        M, T, V = np.meshgrid(self.m_grid, self.tau_grid, self.v_grid,
                               indexing="ij")
        m_flat = M.ravel()
        tau_flat = T.ravel()
        v_flat = V.ravel()

        total = len(m_flat)
        print(f"Building Heston pricing grid ({total:,} points) via QuantLib …")

        results = np.empty(total)
        S_ref = 100.0  # reference spot; normalised prices scale with S
        for i in range(total):
            K_i = S_ref / m_flat[i]
            results[i] = _ql_heston_price_single(
                S_ref, K_i, tau_flat[i], v_flat[i],
                self.kappa, self.theta, self.xi, self.rho, self.r,
            ) / S_ref  # store normalised C/S
            if (i + 1) % 10000 == 0 or i + 1 == total:
                print(f"  {i+1:,}/{total:,}")

        self._prices = results.reshape(M.shape)
        self._build_interp()
        print("Heston grid complete.")

    def _build_interp(self):
        self._interp = RegularGridInterpolator(
            (self.m_grid, self.tau_grid, self.v_grid),
            self._prices,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def price(self, S, K, tau, v):
        """Return European call price C(S,K,tau,v)."""
        S = np.asarray(S, dtype=float)
        K = np.asarray(K, dtype=float)
        tau = np.asarray(tau, dtype=float)
        v = np.asarray(v, dtype=float)

        m = np.clip(S / K, self.m_grid[0], self.m_grid[-1])
        tau_c = np.clip(tau, self.tau_grid[0], self.tau_grid[-1])
        v_c = np.clip(v, self.v_grid[0], self.v_grid[-1])

        pts = np.column_stack([m.ravel(), tau_c.ravel(), v_c.ravel()])
        norm_price = self._interp(pts).reshape(S.shape)
        return np.maximum(S * norm_price, 0.0)

    def _save_grid(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                dict(m=self.m_grid, tau=self.tau_grid,
                     v=self.v_grid, prices=self._prices),
                f,
            )

    def _load_grid(self, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.m_grid = d["m"]
        self.tau_grid = d["tau"]
        self.v_grid = d["v"]
        self._prices = d["prices"]
        self._build_interp()
        print("Loaded cached Heston grid.")

    def delta_from_grid(self, moneyness, tau, v_t, dm=0.005):
        """Finite-difference Heston delta from cached grid.

        The grid stores normalised prices f(m, τ, v) = C/S as a function
        of moneyness m = S/K.  Since C = S · f(m, τ, v):

            ∂C/∂S = f(m) + m · ∂f/∂m

        where ∂f/∂m ≈ [f(m+dm) − f(m−dm)] / (2·dm).
        """
        moneyness = np.asarray(moneyness, dtype=float)
        tau = np.asarray(tau, dtype=float)
        v_t = np.asarray(v_t, dtype=float)

        m_up = np.clip(moneyness + dm, self.m_grid[0], self.m_grid[-1])
        m_dn = np.clip(moneyness - dm, self.m_grid[0], self.m_grid[-1])
        m_c = np.clip(moneyness, self.m_grid[0], self.m_grid[-1])
        tau_c = np.clip(tau, self.tau_grid[0], self.tau_grid[-1])
        v_c = np.clip(v_t, self.v_grid[0], self.v_grid[-1])

        pts_up = np.column_stack([m_up.ravel(), tau_c.ravel(), v_c.ravel()])
        pts_dn = np.column_stack([m_dn.ravel(), tau_c.ravel(), v_c.ravel()])
        pts_mid = np.column_stack([m_c.ravel(), tau_c.ravel(), v_c.ravel()])

        f_up = self._interp(pts_up).reshape(moneyness.shape)
        f_dn = self._interp(pts_dn).reshape(moneyness.shape)
        f_mid = self._interp(pts_mid).reshape(moneyness.shape)

        actual_dm = m_up - m_dn
        actual_dm = np.where(actual_dm < 1e-10, 1.0, actual_dm)
        df_dm = (f_up - f_dn) / actual_dm
        return f_mid + m_c * df_dm


# =====================================================================
# Merton jump-diffusion (series of weighted BS prices)
#
# QuantLib's JumpDiffusionEngine is not exposed in the Python SWIG
# bindings, so we use the standard analytic series expansion.  This is
# numerically identical to the QuantLib C++ implementation.
# =====================================================================

def _bs_call_numpy(S, K, tau, sigma, r):
    """Fast vectorised BS for the Merton series (pure NumPy)."""
    price = np.zeros_like(S)
    valid = tau > 1e-10
    s, k, t, sig = S[valid], K[valid], tau[valid], sigma[valid]
    # r may be a scalar or array; index by [valid] when array
    r_v = r[valid] if np.ndim(r) > 0 else r
    d1 = (np.log(s / k) + (r_v + 0.5 * sig**2) * t) / (sig * np.sqrt(t))
    d2 = d1 - sig * np.sqrt(t)
    price[valid] = s * norm.cdf(d1) - k * np.exp(-r_v * t) * norm.cdf(d2)
    price[~valid] = np.maximum(S[~valid] - K[~valid], 0.0)
    return price


def merton_call_price(S, K, tau, sigma, r, lam, mu_J, sigma_J, n_terms=20):
    """Vectorized Merton jump-diffusion European call price."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    tau = np.asarray(tau, dtype=float)

    k_bar = np.exp(mu_J + 0.5 * sigma_J**2) - 1
    lam_prime = lam * (1 + k_bar)

    price = np.zeros_like(S)
    log_factorial = 0.0

    for n in range(n_terms):
        if n > 0:
            log_factorial += np.log(n)

        log_w = -lam_prime * tau + n * np.log(lam_prime * tau + 1e-30) - log_factorial
        weight = np.exp(log_w)

        sigma_n = np.sqrt(sigma**2 + n * sigma_J**2 / np.maximum(tau, 1e-10))
        r_n = r - lam * k_bar + n * np.log(1 + k_bar) / np.maximum(tau, 1e-10)

        price += weight * _bs_call_numpy(S, K, tau, sigma_n, r_n)

    return price
