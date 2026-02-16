"""Evaluate the trained MLP model on the test split.

Usage::

    python -m src.eval_mlp
    python -m src.eval_mlp --threshold 0.4

Loads ``results/mlp_model.pt`` and ``results/mlp_tfidf.joblib``, evaluates on
the test set, and saves:
    results/mlp_metrics.json
    results/figures/mlp_confusion_matrix.png
    results/mlp_examples.csv
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
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

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
# Confusion-matrix plot (same style as baseline)
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    save_path: pathlib.Path,
    title: str = "Confusion Matrix",
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    log(f"Saved confusion matrix plot -> {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    # ---- load artefacts ---------------------------------------------------
    tfidf_path = _RESULTS_DIR / "mlp_tfidf.joblib"
    model_path = _RESULTS_DIR / "mlp_model.pt"
    if not tfidf_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            "Model artefacts not found. Run  python -m src.train_mlp  first."
        )

    vectorizer = joblib.load(str(tfidf_path))
    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
    log("Loaded TF-IDF vectorizer and model checkpoint.")

    # ---- data (test split) ------------------------------------------------
    log("Loading data …")
    _tr_t, _tr_l, _va_t, _va_l, te_t, te_l = get_splits(
        data_path=args.data_path, seed=42,
    )
    X_test = vectorizer.transform(te_t)
    y_test = np.array(te_l)

    # ---- reconstruct model ------------------------------------------------
    model = MLPClassifier(
        input_dim=checkpoint["input_dim"],
        hidden1=checkpoint["hidden1"],
        hidden2=checkpoint["hidden2"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    log(f"Model loaded (best val_f1={checkpoint.get('best_val_f1', '?')})")

    # ---- predict (batched to keep memory low) ----------------------------
    batch_size = 512
    all_probs: List[float] = []
    n = X_test.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X_batch = torch.tensor(
                X_test[start:end].toarray(), dtype=torch.float32
            ).to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            all_probs.extend(probs)

    test_probs = np.array(all_probs)
    test_preds = (test_probs >= args.threshold).astype(int)

    # ---- metrics ----------------------------------------------------------
    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, zero_division=0)
    rec = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)
    try:
        auc = roc_auc_score(y_test, test_probs)
    except ValueError:
        auc = float("nan")

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "threshold": args.threshold,
        "support": int(len(y_test)),
    }

    log("--- Test metrics ---")
    for k, v in metrics.items():
        log(f"  {k:12s}: {v}")
    log(f"\n{classification_report(y_test, test_preds, target_names=['legit', 'phishing'], zero_division=0)}")

    # ---- save metrics -----------------------------------------------------
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = _RESULTS_DIR / "mlp_metrics.json"
    save_json(str(metrics_path), metrics)
    log(f"Saved metrics -> {metrics_path}")

    # ---- confusion matrix ------------------------------------------------
    cm = confusion_matrix(y_test, test_preds)
    _plot_confusion_matrix(
        cm,
        labels=["legit", "phishing"],
        save_path=_FIGURES_DIR / "mlp_confusion_matrix.png",
        title="MLP – Test Confusion Matrix",
    )

    # ---- example predictions ---------------------------------------------
    n_examples = min(10, len(te_t))
    examples = pd.DataFrame({
        "text_snippet": [t[:120] for t in te_t[:n_examples]],
        "true_label": y_test[:n_examples].tolist(),
        "pred_label": test_preds[:n_examples].tolist(),
        "probability_phishing": [round(float(p), 4) for p in test_probs[:n_examples]],
    })
    examples_path = _RESULTS_DIR / "mlp_examples.csv"
    examples.to_csv(str(examples_path), index=False)
    log(f"Saved example predictions -> {examples_path}")

    log("Evaluation complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained MLP on test set")
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())


# ---------------------------------------------------------------------------
# Quick-run commands:
#
#   python -m src.eval_mlp
#   python -m src.eval_mlp --threshold 0.4
# ---------------------------------------------------------------------------
