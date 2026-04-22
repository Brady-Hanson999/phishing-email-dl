# Phishing Email Classifier (Deep Learning)

Binary classifier for phishing vs. legitimate emails.

**Author**: Brady Hanson
Computer Science and AI Engineering
Penn State University

**Project Type:** Individual Project

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
  model.py          <- MLPClassifier (PyTorch)
  train_mlp.py      <- train TF-IDF + MLP, early stopping on val F1
  eval_mlp.py       <- evaluate trained MLP on test set
  train.py          <- (reserved for future use)
  eval.py           <- (reserved for future use)
```

## MLP: TF-IDF + PyTorch MLP

### Train

```bash
python -m src.train_mlp
```

### Train CLI options

| Flag             | Default | Description                          |
|------------------|---------|--------------------------------------|
| `--data_path`    | *auto*  | Path to CSV (auto-finds in `data/`)  |
| `--seed`         | 42      | Random seed                          |
| `--epochs`       | 10      | Max training epochs                  |
| `--batch_size`   | 256     | Batch size                           |
| `--lr`           | 1e-3    | Learning rate (Adam)                 |
| `--hidden1`      | 256     | First hidden layer size              |
| `--hidden2`      | 128     | Second hidden layer size             |
| `--dropout`      | 0.3     | Dropout rate                         |
| `--max_features` | 50000   | TF-IDF vocabulary cap                |
| `--ngram_max`    | 2       | Upper bound of ngram range (1–3)     |
| `--min_df`       | 2       | Minimum document frequency           |
| `--patience`     | 3       | Early-stopping patience (val F1)     |

### Example

```bash
python -m src.train_mlp --epochs 20 --hidden1 512 --hidden2 256 --lr 5e-4
```

### Evaluate (test only)

```bash
python -m src.eval_mlp
python -m src.eval_mlp --threshold 0.4
```

| Flag          | Default | Description                         |
|---------------|---------|-------------------------------------|
| `--data_path` | *auto*  | Path to CSV                         |
| `--threshold` | 0.5     | Classification probability cutoff   |

### MLP outputs

| File | Description |
|------|-------------|
| `mlp_tfidf.joblib` | Fitted TfidfVectorizer |
| `mlp_model.pt` | Best model checkpoint (by val F1) |
| `mlp_history.json` | Per-epoch train loss, val loss, val F1 |
| `figures/mlp_loss_curve.png` | Train & val loss plot |
| `figures/mlp_val_f1_curve.png` | Val F1 over epochs |
| `mlp_metrics.json` | Test accuracy, precision, recall, F1, ROC-AUC |
| `figures/mlp_confusion_matrix.png` | Test confusion matrix |
| `mlp_examples.csv` | 10 sample predictions with snippet, labels, prob |



