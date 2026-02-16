"""Train a TF-IDF + MLP classifier for phishing-email detection.

Usage::

    python -m src.train_mlp
    python -m src.train_mlp --epochs 20 --hidden1 512 --hidden2 256 --lr 5e-4

Outputs saved under ``results/``:
    mlp_tfidf.joblib               – fitted TfidfVectorizer
    mlp_model.pt                   – best model checkpoint (by val F1)
    mlp_history.json               – per-epoch train/val metrics
    figures/mlp_loss_curve.png     – train & val loss plot
    figures/mlp_val_f1_curve.png   – val F1 over epochs
"""

from __future__ import annotations

import argparse
import pathlib
from typing import List

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from src.data import get_splits
from src.model import MLPClassifier
from src.utils import log, save_json, set_seed

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_RESULTS_DIR = _PROJECT_ROOT / "results"
_FIGURES_DIR = _RESULTS_DIR / "figures"


# ---------------------------------------------------------------------------
# Sparse TF-IDF Dataset + collate_fn
# ---------------------------------------------------------------------------

class SparseTfidfDataset(Dataset):
    """Wraps a scipy sparse matrix + label array for use with DataLoader.

    Individual rows are kept sparse; densification happens per-batch inside
    the collate function so memory stays low.
    """

    def __init__(self, X_sparse: sp.csr_matrix, labels: np.ndarray) -> None:
        self.X = X_sparse
        self.y = labels.astype(np.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        # Return sparse row and scalar label
        return self.X[idx], self.y[idx]


def sparse_collate(batch):
    """Collate sparse rows into a dense float32 batch tensor."""
    rows, labels = zip(*batch)
    # Stack sparse rows -> sparse matrix -> dense
    X_batch = sp.vstack(rows).toarray()
    X_tensor = torch.tensor(X_batch, dtype=torch.float32)
    y_tensor = torch.tensor(np.array(labels), dtype=torch.float32)
    return X_tensor, y_tensor


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_curves(history: dict, save_dir: pathlib.Path) -> None:
    """Save loss and val-F1 curves."""
    epochs = list(range(1, len(history["train_loss"]) + 1))

    # Loss curve
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["train_loss"], label="train loss")
    ax.plot(epochs, history["val_loss"], label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("MLP – Loss Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(save_dir / "mlp_loss_curve.png"), dpi=150)
    plt.close(fig)
    log(f"Saved loss curve -> {save_dir / 'mlp_loss_curve.png'}")

    # Val F1 curve
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["val_f1"], marker="o", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.set_title("MLP – Validation F1")
    fig.tight_layout()
    fig.savefig(str(save_dir / "mlp_val_f1_curve.png"), dpi=150)
    plt.close(fig)
    log(f"Saved val-F1 curve -> {save_dir / 'mlp_val_f1_curve.png'}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # ---- data -------------------------------------------------------------
    log("Loading data …")
    tr_t, tr_l, va_t, va_l, _te_t, _te_l = get_splits(
        data_path=args.data_path, seed=args.seed,
    )

    # ---- TF-IDF (fit on train only) --------------------------------------
    log(f"Fitting TF-IDF (max_features={args.max_features}, "
        f"ngram_range=(1,{args.ngram_max}), min_df={args.min_df})")
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(tr_t)
    X_val = vectorizer.transform(va_t)
    input_dim = X_train.shape[1]
    log(f"TF-IDF vocabulary size: {input_dim}")

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tfidf_path = _RESULTS_DIR / "mlp_tfidf.joblib"
    joblib.dump(vectorizer, str(tfidf_path))
    log(f"Saved TF-IDF vectorizer -> {tfidf_path}")

    y_train = np.array(tr_l)
    y_val = np.array(va_l)

    # ---- DataLoaders ------------------------------------------------------
    train_ds = SparseTfidfDataset(X_train, y_train)
    val_ds = SparseTfidfDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=sparse_collate, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=sparse_collate, num_workers=0,
    )

    # ---- model ------------------------------------------------------------
    model = MLPClassifier(
        input_dim=input_dim,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout=args.dropout,
    ).to(device)
    log(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # pos_weight for class imbalance
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    log(f"pos_weight = {pos_weight.item():.4f}  (neg={n_neg}, pos={n_pos})")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- training ---------------------------------------------------------
    history: dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "val_f1": [],
    }
    best_f1 = -1.0
    patience_counter = 0
    best_state = None

    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _RESULTS_DIR / "mlp_model.pt"

    for epoch in range(1, args.epochs + 1):
        # -- train ----------------------------------------------------------
        model.train()
        running_loss = 0.0
        n_samples = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(y_batch)
            n_samples += len(y_batch)
        train_loss = running_loss / n_samples

        # -- validate -------------------------------------------------------
        model.eval()
        val_running_loss = 0.0
        val_n = 0
        all_preds: List[int] = []
        all_true: List[int] = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_running_loss += loss.item() * len(y_batch)
                val_n += len(y_batch)
                preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
                all_preds.extend(preds.tolist())
                all_true.extend(y_batch.long().cpu().numpy().tolist())
        val_loss = val_running_loss / val_n
        val_f1 = f1_score(all_true, all_preds, zero_division=0)

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_f1"].append(round(val_f1, 6))

        improved = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            improved = "  *best*"
        else:
            patience_counter += 1

        log(f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_f1={val_f1:.4f}{improved}")

        if patience_counter >= args.patience:
            log(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break

    # ---- save best model --------------------------------------------------
    if best_state is not None:
        checkpoint = {
            "model_state_dict": best_state,
            "input_dim": input_dim,
            "hidden1": args.hidden1,
            "hidden2": args.hidden2,
            "dropout": args.dropout,
            "best_val_f1": best_f1,
        }
        torch.save(checkpoint, str(model_path))
        log(f"Saved best model (val_f1={best_f1:.4f}) -> {model_path}")

    # ---- save history & plots ---------------------------------------------
    history_path = _RESULTS_DIR / "mlp_history.json"
    save_json(str(history_path), history)
    log(f"Saved training history -> {history_path}")

    _plot_curves(history, _FIGURES_DIR)
    log("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TF-IDF + MLP classifier")
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden1", type=int, default=256)
    p.add_argument("--hidden2", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--max_features", type=int, default=50_000)
    p.add_argument("--ngram_max", type=int, default=2, choices=[1, 2, 3])
    p.add_argument("--min_df", type=int, default=2)
    p.add_argument("--patience", type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())


# ---------------------------------------------------------------------------
# Quick-run commands:
#
#   python -m src.train_mlp
#   python -m src.train_mlp --epochs 20 --hidden1 512 --hidden2 256
#   python -m src.train_mlp --lr 5e-4 --dropout 0.5 --patience 5
# ---------------------------------------------------------------------------
