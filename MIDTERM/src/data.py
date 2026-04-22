"""Data loading, column detection, preprocessing, and stratified splitting.

Usage (module mode)::

    python -m src.data                     # auto-find CSV in data/
    python -m src.data path/to/file.csv    # explicit CSV path
"""

import os
import pathlib
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocess import clean_texts
from src.utils import log, set_seed

# ---------------------------------------------------------------------------
# Column-name candidates (case-insensitive matching)
# ---------------------------------------------------------------------------
TEXT_CANDIDATES: List[str] = [
    "text", "email", "body", "content", "message",
    "email_text", "email text",
]

LABEL_CANDIDATES: List[str] = [
    "label", "class", "target", "spam", "phishing", "category",
]

# Project root  ── repo top-level directory (parent of src/)
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_csv(data_dir: pathlib.Path = _DATA_DIR) -> pathlib.Path:
    """Return the path of the single CSV inside *data_dir*.

    Raises
    ------
    FileNotFoundError
        If no CSV is found or more than one CSV exists.
    """
    csvs = sorted(data_dir.glob("*.csv"))
    if len(csvs) == 0:
        raise FileNotFoundError(
            f"No CSV file found in {data_dir}. "
            "Download the dataset and place the CSV in the data/ folder."
        )
    if len(csvs) > 1:
        names = [c.name for c in csvs]
        raise FileNotFoundError(
            f"Multiple CSVs in {data_dir}: {names}. "
            "Pass the exact path via the data_path argument."
        )
    return csvs[0]


def _detect_column(
    columns: pd.Index, candidates: List[str], role: str
) -> str:
    """Return the first column name in *columns* that matches a candidate (case-insensitive)."""
    col_lower = {c.strip().lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in col_lower:
            return col_lower[cand.lower()]
    # Nothing matched — give the user a helpful error
    raise KeyError(
        f"Could not auto-detect the {role} column.\n"
        f"  Columns in CSV : {list(columns)}\n"
        f"  Tried candidates: {candidates}\n"
        f"Edit TEXT_CANDIDATES / LABEL_CANDIDATES in src/data.py "
        f"to add the correct column name."
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_splits(
    data_path: Optional[str] = None,
    seed: int = 42,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
) -> Tuple[
    List[str], List[int],
    List[str], List[int],
    List[str], List[int],
]:
    """Load, preprocess, and split the phishing-email dataset.

    Parameters
    ----------
    data_path : str or None
        Explicit path to the CSV.  If *None*, the single CSV in ``data/``
        is used automatically.
    seed : int
        Random seed for reproducible splits.
    val_frac, test_frac : float
        Fraction of the full dataset used for validation / test.

    Returns
    -------
    tuple of six lists
        ``(train_texts, train_labels, val_texts, val_labels,
          test_texts, test_labels)``
    """
    set_seed(seed)

    # ---- locate & load CSV ------------------------------------------------
    csv_path = pathlib.Path(data_path) if data_path else _find_csv()
    log(f"Loading {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace")
    log(f"Raw rows: {len(df)},  columns: {list(df.columns)}")

    # ---- detect columns ---------------------------------------------------
    text_col = _detect_column(df.columns, TEXT_CANDIDATES, "text")
    label_col = _detect_column(df.columns, LABEL_CANDIDATES, "label")
    log(f"Using text column: '{text_col}',  label column: '{label_col}'")

    # ---- drop rows with missing text or label -----------------------------
    before = len(df)
    df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)
    if len(df) < before:
        log(f"Dropped {before - len(df)} rows with missing text/label")

    # ---- encode labels to ints -------------------------------------------
    raw_labels = df[label_col]
    unique_labels = sorted(raw_labels.unique(), key=str)
    label_map = {v: i for i, v in enumerate(unique_labels)}
    labels: np.ndarray = raw_labels.map(label_map).values.astype(int)
    log(f"Label mapping: {label_map}")

    # ---- preprocess text --------------------------------------------------
    texts: List[str] = clean_texts(df[text_col].astype(str).tolist())

    # ---- class balance ----------------------------------------------------
    for lbl_name, lbl_id in label_map.items():
        count = int((labels == lbl_id).sum())
        pct = 100.0 * count / len(labels)
        log(f"  class {lbl_id} ({lbl_name}): {count}  ({pct:.1f}%)")

    # ---- stratified split -------------------------------------------------
    holdout_frac = val_frac + test_frac
    train_texts, hold_texts, train_labels, hold_labels = train_test_split(
        texts, labels,
        test_size=holdout_frac,
        random_state=seed,
        stratify=labels,
    )
    relative_test = test_frac / holdout_frac
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        hold_texts, hold_labels,
        test_size=relative_test,
        random_state=seed,
        stratify=hold_labels,
    )

    # Convert numpy label arrays to plain Python lists
    train_labels = train_labels.tolist()
    val_labels = val_labels.tolist()
    test_labels = test_labels.tolist()

    log(f"Split sizes  ->  train: {len(train_texts)},  "
        f"val: {len(val_texts)},  test: {len(test_texts)}")

    return (
        train_texts, train_labels,
        val_texts, val_labels,
        test_texts, test_labels,
    )


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Accept an optional CLI argument for the CSV path
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    tr_t, tr_l, va_t, va_l, te_t, te_l = get_splits(data_path=csv_arg)

    print("\n--- Sample train texts ---")
    for t, l in zip(tr_t[:3], tr_l[:3]):
        print(f"  [{l}] {t[:120]}...")

    print("\n--- Sample val texts ---")
    for t, l in zip(va_t[:2], va_l[:2]):
        print(f"  [{l}] {t[:120]}...")

    print("\nDone.")