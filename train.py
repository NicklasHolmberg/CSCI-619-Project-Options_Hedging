"""Training loops for neural-net methods and ridge regression."""

import copy
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src_2.config import TrainConfig
from src_2.losses import mshe_loss, cvar_loss, compute_cvar_np, compute_mshe_np
from src_2.models import MLPHedge, StructuredVolNet, LinearRidgeHedge


# =====================================================================
# Helpers
# =====================================================================

def _to_tensors(data: dict, device: str = "cpu"):
    X = torch.tensor(
        np.column_stack([data["moneyness"], data["tau"], data["real_vol"]]),
        dtype=torch.float32, device=device,
    )
    dC = torch.tensor(data["dC"], dtype=torch.float32, device=device)
    dS = torch.tensor(data["dS"], dtype=torch.float32, device=device)
    return X, dC, dS


def _to_tensors_with_ivol(data: dict, device: str = "cpu"):
    """Same as _to_tensors but also returns implied vol target tensor."""
    X, dC, dS = _to_tensors(data, device)
    ivol = torch.tensor(data["implied_vol"], dtype=torch.float32, device=device)
    return X, dC, dS, ivol


def _loader(X, dC, dS, batch_size, shuffle=True):
    ds = TensorDataset(X, dC, dS)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# =====================================================================
# Neural net training (Methods 2, 4, 5, 6)
# =====================================================================

def train_nn(
    model: torch.nn.Module,
    train_data: dict,
    val_data: dict,
    cfg: TrainConfig,
    loss_type: str = "mshe",          # "mshe" or "cvar"
    select_type: str = "mshe",        # validation metric for early stopping
    device: str = "cpu",
) -> torch.nn.Module:
    """
    Train a PyTorch hedge model.

    loss_type   — training objective ("mshe" for Methods 2/4/5, "cvar" for 6)
    select_type — validation selection criterion ("mshe" for 2/4, "cvar" for 5/6)
    """
    model = model.to(device)
    X_tr, dC_tr, dS_tr = _to_tensors(train_data, device)
    X_va, dC_va, dS_va = _to_tensors(val_data, device)

    params = list(model.parameters())

    # Auxiliary variable ν for CVaR training
    nu = None
    if loss_type == "cvar":
        nu = torch.nn.Parameter(torch.tensor(0.0, device=device))
        params.append(nu)

    optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_state = None
    patience_ctr = 0

    loader = _loader(X_tr, dC_tr, dS_tr, cfg.batch_size)

    for epoch in range(cfg.epochs):
        # ---- train ----
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_b, dC_b, dS_b in loader:
            optimizer.zero_grad()
            h = model(X_b)
            if loss_type == "mshe":
                loss = mshe_loss(h, dC_b, dS_b)
            else:
                loss = cvar_loss(h, dC_b, dS_b, nu, cfg.alpha)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # ---- validate ----
        model.eval()
        with torch.no_grad():
            h_val = model(X_va).cpu().numpy()
        dC_v = val_data["dC"]
        dS_v = val_data["dS"]

        if select_type == "mshe":
            val_metric = compute_mshe_np(h_val, dC_v, dS_v)
        else:
            val_metric = compute_cvar_np(h_val, dC_v, dS_v, cfg.alpha)

        improved = val_metric < best_val
        if improved:
            best_val = val_metric
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 10 == 0 or improved or patience_ctr >= cfg.patience:
            star = "*" if improved else " "
            print(f"    {star} epoch {epoch:3d}  train_loss={epoch_loss/n_batches:.6f}  "
                  f"val_{select_type}={val_metric:.6f}  patience={patience_ctr}/{cfg.patience}")

        if patience_ctr >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


# =====================================================================
# Naive VolNet training — M2b (predict implied vol, then freeze + apply BS Δ)
# =====================================================================

def train_volnet_naive(
    model: StructuredVolNet,
    train_data: dict,
    val_data: dict,
    cfg: TrainConfig,
    device: str = "cpu",
) -> StructuredVolNet:
    """Train VolNet Stage 1: minimise MSE(σ̂, implied_vol).

    The network learns to predict BS-implied volatility accurately,
    disconnected from the downstream hedging objective.  At inference
    the full forward pass (σ̂ → BS Δ) is used with frozen weights.
    """
    model = model.to(device)
    X_tr, _, _, ivol_tr = _to_tensors_with_ivol(train_data, device)
    X_va, _, _, ivol_va = _to_tensors_with_ivol(val_data, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_state = None
    patience_ctr = 0

    ds = TensorDataset(X_tr, ivol_tr)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_b, vol_b in loader:
            optimizer.zero_grad()
            sigma_hat = torch.nn.functional.softplus(
                model.net(X_b).squeeze(-1)) + 1e-4
            loss = torch.mean((sigma_hat - vol_b) ** 2)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            sigma_hat_va = torch.nn.functional.softplus(
                model.net(X_va).squeeze(-1)) + 1e-4
            val_mse = torch.mean((sigma_hat_va - ivol_va) ** 2).item()

        improved = val_mse < best_val
        if improved:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 10 == 0 or improved or patience_ctr >= cfg.patience:
            star = "*" if improved else " "
            print(f"    {star} epoch {epoch:3d}  train_vol_mse="
                  f"{epoch_loss/n_batches:.6f}  val_vol_mse={val_mse:.6f}"
                  f"  patience={patience_ctr}/{cfg.patience}")

        if patience_ctr >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


# =====================================================================
# Ridge regression (Method 3)
# =====================================================================

def train_ridge(
    train_data: dict,
    val_data: dict,
    cfg: TrainConfig,
    select_type: str = "mshe",
) -> LinearRidgeHedge:
    """Fit ridge with cross-validated α, selecting on MSHE or CVaR."""
    best_val = float("inf")
    best_model = None

    for alpha in cfg.ridge_alphas:
        m = LinearRidgeHedge(alpha=alpha)
        m.fit(train_data)
        h_val = m.predict_h(val_data)

        dC_v = val_data["dC"]
        dS_v = val_data["dS"]
        if select_type == "mshe":
            metric = compute_mshe_np(h_val, dC_v, dS_v)
        else:
            metric = compute_cvar_np(h_val, dC_v, dS_v, cfg.alpha)

        if metric < best_val:
            best_val = metric
            best_model = m

    return best_model
