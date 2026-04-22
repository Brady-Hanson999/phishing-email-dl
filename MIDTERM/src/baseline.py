"""TF-IDF + Logistic Regression baseline for phishing-email classification.

Usage::

    python -m src.baseline                       # auto-find CSV in data/
    python -m src.baseline --data_path data/x.csv
    python -m src.baseline --max_features 30000 --ngram_max 1

Outputs saved under ``results/``:
    baseline_metrics.json          – accuracy, precision, recall, f1, roc_auc
    baseline_confusion_matrix.png  – confusion-matrix heatmap  (in figures/)
    baseline_examples.csv          – 10 sample predictions
    baseline_tfidf.joblib          – fitted TfidfVectorizer
    baseline_logreg.joblib         – fitted LogisticRegression model
"""

import argparse
import pathlib
from typing import List

import joblib
import matplotlib
matplotlib.use("Agg")                     # non-interactive backend (CI / SSH safe)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
from src.utils import log, save_json, set_seed

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_RESULTS_DIR = _PROJECT_ROOT / "results"
_FIGURES_DIR = _RESULTS_DIR / "figures"


# ---------------------------------------------------------------------------
# Confusion-matrix plot helper
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    save_path: pathlib.Path,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save a confusion-matrix heatmap using matplotlib only."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels)

    # Annotate cells with counts
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
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute and print classification metrics; return them as a dict."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "threshold": threshold,
        "support": int(len(y_true)),
    }

    log(f"--- {name} metrics ---")
    for k, v in metrics.items():
        log(f"  {k:12s}: {v}")
    log(f"\n{classification_report(y_true, y_pred, target_names=['legit', 'phishing'], zero_division=0)}")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TF-IDF + LogReg baseline")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to CSV (default: auto-find in data/)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_features", type=int, default=50_000)
    parser.add_argument("--ngram_max", type=int, default=2, choices=[1, 2, 3],
                        help="Upper bound of ngram_range (default 2)")
    parser.add_argument("--min_df", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold on P(phishing)")
    args = parser.parse_args()

    set_seed(args.seed)

    # ---- data -------------------------------------------------------------
    log("Loading data …")
    tr_t, tr_l, va_t, va_l, te_t, te_l = get_splits(
        data_path=args.data_path, seed=args.seed,
    )

    # ---- TF-IDF -----------------------------------------------------------
    log(f"Fitting TF-IDF  (max_features={args.max_features}, "
        f"ngram_range=(1,{args.ngram_max}), min_df={args.min_df})")
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(tr_t)
    X_val = vectorizer.transform(va_t)
    X_test = vectorizer.transform(te_t)
    log(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")

    y_train = np.array(tr_l)
    y_val = np.array(va_l)
    y_test = np.array(te_l)

    # ---- Logistic Regression ----------------------------------------------
    log("Training Logistic Regression …")
    model = LogisticRegression(
        solver="liblinear",
        max_iter=3000,
        class_weight="balanced",
        C=1.0,
        random_state=args.seed,
    )
    model.fit(X_train, y_train)
    log("Training complete.")

    # ---- predictions (threshold-aware) ------------------------------------
    def predict_with_threshold(X, threshold):
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= threshold).astype(int)
        return preds, probs

    val_preds, val_probs = predict_with_threshold(X_val, args.threshold)
    test_preds, test_probs = predict_with_threshold(X_test, args.threshold)

    # ---- evaluate ---------------------------------------------------------
    val_metrics = evaluate("Validation", y_val, val_preds, val_probs, args.threshold)
    test_metrics = evaluate("Test", y_test, test_preds, test_probs, args.threshold)

    # ---- save metrics JSON ------------------------------------------------
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = {
        "validation": val_metrics,
        "test": test_metrics,
        "config": {
            "max_features": args.max_features,
            "ngram_range": [1, args.ngram_max],
            "min_df": args.min_df,
            "threshold": args.threshold,
            "seed": args.seed,
            "solver": "liblinear",
            "class_weight": "balanced",
            "C": 1.0,
            "max_iter": 3000,
        },
    }
    metrics_path = _RESULTS_DIR / "baseline_metrics.json"
    save_json(str(metrics_path), all_metrics)
    log(f"Saved metrics -> {metrics_path}")

    # ---- confusion matrix plot -------------------------------------------
    cm = confusion_matrix(y_test, test_preds)
    plot_confusion_matrix(
        cm,
        labels=["legit", "phishing"],
        save_path=_FIGURES_DIR / "baseline_confusion_matrix.png",
        title="Baseline – Test Confusion Matrix",
    )

    # ---- example predictions CSV -----------------------------------------
    n_examples = min(10, len(te_t))
    examples = pd.DataFrame({
        "text_snippet": [t[:120] for t in te_t[:n_examples]],
        "true_label": y_test[:n_examples].tolist(),
        "pred_label": test_preds[:n_examples].tolist(),
        "probability_phishing": [round(float(p), 4) for p in test_probs[:n_examples]],
    })
    examples_path = _RESULTS_DIR / "baseline_examples.csv"
    examples.to_csv(str(examples_path), index=False)
    log(f"Saved example predictions -> {examples_path}")

    # ---- save model artefacts --------------------------------------------
    tfidf_path = _RESULTS_DIR / "baseline_tfidf.joblib"
    model_path = _RESULTS_DIR / "baseline_logreg.joblib"
    joblib.dump(vectorizer, str(tfidf_path))
    joblib.dump(model, str(model_path))
    log(f"Saved TF-IDF vectorizer -> {tfidf_path}")
    log(f"Saved LogReg model      -> {model_path}")

    log("Baseline complete.")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Quick-run commands (copy-paste into terminal):
#
#   python -m src.baseline
#   python -m src.baseline --data_path data/phishing_emails.csv
#   python -m src.baseline --max_features 30000 --ngram_max 1 --threshold 0.4
# ---------------------------------------------------------------------------
