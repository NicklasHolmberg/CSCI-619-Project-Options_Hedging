# When Does Decision-Aware Hedging Matter?

**CSCI 619 — Final Project, University of Southern California (Spring 2026)**

This repository benchmarks eight option-hedging methods across three data-generating processes (DGPs) to investigate when end-to-end, decision-aware training yields meaningful improvements over classical model-based and supervised approaches.

---

## Methods

| ID | Name | Description |
|----|------|-------------|
| M1 | BS Delta | Black-Scholes delta using trailing realised vol |
| M1b | Heston Oracle Delta | Numerical delta under true Heston parameters (oracle) |
| M2 | StructuredVolNet (e2e) | Small net predicts σ̂, fed into differentiable BS Δ; trained end-to-end on MSHE |
| M2b | Naive VolNet (two-stage) | Same architecture, but trained on vol-prediction MSE first, then frozen |
| M3 | Ridge | Linear regression of features → delta, model-selected on MSHE |
| M4 | MLP (MSHE/MSHE) | MLP hedger trained and selected on MSHE |
| M5 | MLP (MSHE/CVaR-sel) | MLP hedger trained on MSHE, selected on CVaR |
| M6 | MLP (CVaR e2e) | MLP hedger trained and selected on CVaR end-to-end |

## Data-Generating Processes

| DGP | Description |
|-----|-------------|
| GBM | Geometric Brownian Motion (constant vol; BS is correct model) |
| Heston | Stochastic vol with vol-of-vol and leverage effect |
| Merton | Jump-diffusion (GBM + compound Poisson jumps) |

## Metrics

- **MSHE** — Mean Squared Hedging Error: $\frac{1}{N}\sum_i (dC_i - h_i\, dS_i)^2$
- **CVaR₀.₉₅** — Conditional Value-at-Risk of the hedging P&L at the 95th percentile tail

---

## Repository Structure

```
.                            # repo root (this folder)
├── __init__.py
├── config.py                # DGP parameters, simulation & training hyperparams
├── simulate.py              # Path simulators (GBM, Heston, Merton)
├── pricing.py               # Heston pricer (grid interpolation via QuantLib)
├── dataset.py               # Feature engineering, dataset builder
├── models.py                # All eight hedging model classes
├── losses.py                # MSHE and CVaR loss functions (NumPy + PyTorch)
├── train.py                 # Training loops (NN, Ridge, two-stage)
├── evaluate.py              # Evaluation metrics, bootstrap CI, moneyness buckets
├── run.py                   # Experiment runner (CLI entry point)
├── plot.py                  # Figure generation utilities
├── results/                 # JSON output files (git-ignored by default)
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **QuantLib** (`QuantLib>=1.30`) is required for the Heston pricer. On macOS you may need to install it via conda:
> ```bash
> conda install -c conda-forge quantlib
> ```

---

## Running Experiments

All experiments are run from the **repo root** (i.e., this folder):

```bash
# Experiment 1 — All three DGPs (main benchmark table)
python -m run

# Single DGP
python -m run --dgp gbm
python -m run --dgp heston
python -m run --dgp merton

# Experiment 4 — Weekly rebalancing vs daily (Heston)
python -m run --weekly

# Sensitivity — ρ sweep (Heston leverage effect)
python -m run --sensitivity

# Transaction cost sweep (Heston)
python -m run --tc-sweep

# Multi-seed robustness check
python -m run --multi-seed
```

Results are written to `results/` as JSON files (e.g., `experiment1_main.json`).

---

## Key Findings (Summary)

| Setting | Winner | Takeaway |
|---------|--------|----------|
| GBM | M1 BS Delta ≈ all ML | BS is the correct model; learning adds nothing |
| Heston | M6 MLP CVaR e2e | Stochastic vol + leverage effect creates exploitable structure |
| Merton | M2b Naive VolNet | Jump risk is hard to learn; vol-based features still help |
| Stronger leverage (ρ → −0.7) | M6 widens lead | More distributional skew → more payoff for CVaR-aware training |
| Transaction costs (tc 0–0.005) | No change in ranking | Single-step formulation means all methods trade equally; TC-awareness requires path-dependent training |

---

## Notes for Code Reviewers

- **`src_2/`** is the active codebase. `src/` (if present) is a frozen earlier version kept for reference only.
- The Heston Oracle Delta (M1b) is intentionally an *oracle* — it uses true latent variance and true model parameters and therefore sets an upper bound for model-based hedging under Heston.
- The QuantLib Heston pricer builds a grid on first run and caches it to `cache/heston_grid.pkl`. Subsequent runs load from cache (much faster).
- All random seeds are fixed in `config.py` (`SimConfig.seed_train/val/test`). Results are deterministic given the same PyTorch/NumPy versions.
- Training uses CPU by default. Pass `--device cuda` (if GPU is available) to accelerate MLP training.
