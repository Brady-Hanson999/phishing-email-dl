# Phishing Email Classifier (Deep Learning)

Binary classifier for phishing vs. legitimate emails.

## Setup

```bash
pip install -r requirements.txt
```

Place your Kaggle phishing-email CSV in `data/` (a single CSV file).

## Data pipeline

```bash
python -m src.data                       # auto-find CSV in data/
python -m src.data data/my_file.csv      # explicit path
```

Prints dataset stats (rows, class balance, split sizes) and runs a quick sanity check.

## Baseline: TF-IDF + Logistic Regression

```bash
python -m src.baseline
```

### CLI options

| Flag              | Default | Description                         |
|-------------------|---------|-------------------------------------|
| `--data_path`     | *auto*  | Path to CSV (auto-finds in `data/`) |
| `--seed`          | 42      | Random seed                         |
| `--max_features`  | 50000   | TF-IDF vocabulary cap               |
| `--ngram_max`     | 2       | Upper bound of ngram range (1–3)    |
| `--min_df`        | 2       | Minimum document frequency          |
| `--threshold`     | 0.5     | Classification probability cutoff   |

### Example with custom settings

```bash
python -m src.baseline --max_features 30000 --ngram_max 1 --threshold 0.4
```

### Outputs

All outputs are saved under `results/`:

| File | Description |
|------|-------------|
| `baseline_metrics.json` | Accuracy, precision, recall, F1, ROC-AUC for val & test |
| `figures/baseline_confusion_matrix.png` | Test-set confusion matrix heatmap |
| `baseline_examples.csv` | 10 sample predictions with snippet, labels, probability |
| `baseline_tfidf.joblib` | Fitted TfidfVectorizer (for reuse / inference) |
| `baseline_logreg.joblib` | Fitted LogisticRegression model |

## Project structure

```
data/               <- put dataset CSV here
notebooks/          <- exploratory notebooks
results/
  figures/          <- saved plots
scripts/
  run_train.sh
src/
  __init__.py
  data.py           <- data loading, column detection, splitting
  preprocess.py     <- text cleaning (URL/email replacement, etc.)
  utils.py          <- set_seed, save_json, logging helpers
  baseline.py       <- TF-IDF + LogReg baseline
  model.py          <- (deep learning model – coming soon)
  train.py          <- (training loop – coming soon)
  eval.py           <- (evaluation – coming soon)
```
