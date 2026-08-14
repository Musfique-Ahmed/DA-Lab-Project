"""PyTorch MLP for the credit-risk default-prediction task.

Two pieces:
  - `MLP`        : a tiny `nn.Module` with 2-3 hidden layers, ReLU, dropout.
  - `MLPWrapper` : a sklearn-style shim that holds the trained `MLP` plus the
                   train-only `StandardScaler` and exposes `predict_proba` /
                   `predict`, so the rest of the pipeline can treat it like
                   any other classifier.
  - `train_loop` : a small training loop that early-stops on val AUC.

Design notes:
  - We do NOT use `torch.utils.data.DataLoader`. For 215k × 30 floats, the
    per-batch Python overhead of DataLoader dominates the actual tensor ops.
    Materializing a single big train tensor and slicing is faster in our
    regime and lets us write a clean early-stop loop without workers.
  - The output is a single logit (binary classification). We use
    `BCEWithLogitsLoss` (numerically stable; `pos_weight` lets us handle
    the class imbalance without SMOTE).
  - We always pass `MLP` to a single GPU-or-CPU device determined at fit
    time; no hard-coded `.cuda()`.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn

# Determinism knobs for reproducibility. Used by callers (train_mlp) too.
DEFAULT_RANDOM_STATE: int = 42


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    """A small feed-forward MLP for binary classification.

    Parameters
    ----------
    n_features : int
        Input feature count.
    hidden : tuple[int, ...]
        Hidden layer widths (e.g. (128, 64) means 2 hidden layers of 128
        and 64 units, then a single output logit).
    dropout : float
        Dropout probability after each hidden activation.
    """

    def __init__(self, n_features: int, hidden: Sequence[int] = (128, 64), dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))  # single logit
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape (batch,)."""
        return self.net(x).squeeze(-1)


class MLPWrapper:
    """Carries a trained `MLP` plus the train-only `StandardScaler`.

    Mirrors the sklearn classifier API (`predict_proba`, `predict`) so the
    rest of the codebase can treat it as just another classifier.

    Persistence:
      - `state_dict()` returns the model weights (save with `torch.save`).
      - `meta()` returns a small dict (scaler stats + arch) for the
        sidecar `models/best_model_meta.json` file.
    """

    def __init__(
        self,
        model: MLP,
        scaler: StandardScaler,
        imputer,
        hidden: Sequence[int],
        dropout: float,
    ):
        self.model = model
        self.scaler = scaler
        self.imputer = imputer
        self.hidden = tuple(hidden)
        self.dropout = float(dropout)
        self._device = next(model.parameters()).device

    @classmethod
    def fit(
        cls,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        *,
        hidden: Sequence[int] = (128, 64),
        dropout: float = 0.3,
        epochs: int = 15,
        lr: float = 1e-3,
        batch_size: int = 2048,
        patience: int = 3,
        random_state: int = DEFAULT_RANDOM_STATE,
        pos_weight: float | None = None,
    ) -> "MLPWrapper":
        """Fit an MLP on (X_tr, y_tr) with early stopping on val AUC.

        Returns a fitted `MLPWrapper` whose `.model` is in eval mode on
        the best epoch (lowest val loss) — never the last epoch.
        """
        _set_seed(random_state)
        # Impute any NaN features (the engineered ratio columns have ~185k
        # NaN each, where the original denominator was zero). Use train-only
        # medians to avoid leakage. The fitted imputer is then persisted in
        # `meta()` so the production scorer can repeat it.
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy="median").fit(X_tr)
        X_tr_i = imputer.transform(X_tr)
        X_val_i = imputer.transform(X_val)
        scaler = StandardScaler().fit(X_tr_i)
        X_tr_s = scaler.transform(X_tr_i).astype(np.float32)
        X_val_s = scaler.transform(X_val_i).astype(np.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MLP(n_features=X_tr_s.shape[1], hidden=hidden, dropout=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # `pos_weight` is None -> no re-weighting; otherwise BCEWithLogitsLoss
        # up-weights the positive class.
        if pos_weight is not None:
            pw = torch.tensor([pos_weight], dtype=torch.float32, device=device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            loss_fn = nn.BCEWithLogitsLoss()

        X_tr_t = torch.from_numpy(X_tr_s).to(device)
        y_tr_t = torch.from_numpy(y_tr.astype(np.float32)).to(device)
        X_val_t = torch.from_numpy(X_val_s).to(device)
        y_val_np = y_val.astype(np.float32)

        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_val_auc = -np.inf
        best_epoch = 0
        epochs_no_improve = 0
        history: list[dict] = []

        n = X_tr_t.shape[0]
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                logits = model(X_tr_t[idx])
                loss = loss_fn(logits, y_tr_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Val AUC at end of epoch.
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t).detach().cpu().numpy()
            val_proba = 1.0 / (1.0 + np.exp(-val_logits))
            val_auc = float(roc_auc_score(y_val_np, val_proba))
            history.append({"epoch": epoch, "val_auc": val_auc})
            if val_auc > best_val_auc + 1e-6:
                best_val_auc = val_auc
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        # Restore best epoch's weights.
        model.load_state_dict(best_state)
        model.eval()
        wrapper = cls(model=model, scaler=scaler, imputer=imputer,
                     hidden=hidden, dropout=dropout)
        wrapper.history_ = history  # for callers who want it
        wrapper.best_epoch_ = best_epoch
        wrapper.best_val_auc_ = best_val_auc
        return wrapper

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict P(class=1) for each row of X. Returns shape (n, 2)."""
        X_i = self.imputer.transform(X)
        X_s = self.scaler.transform(X_i).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(X_s).to(self._device)
            logits = self.model(x_t).detach().cpu().numpy()
        p1 = 1.0 / (1.0 + np.exp(-logits))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict hard class labels at the given threshold."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def state_dict(self):
        return self.model.state_dict()

    def meta(self) -> dict:
        """Return a small dict for the sidecar meta.json file."""
        return {
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "imputer_median": self.imputer.statistics_.tolist(),
            "hidden": list(self.hidden),
            "dropout": self.dropout,
        }


__all__ = ["MLP", "MLPWrapper", "DEFAULT_RANDOM_STATE"]