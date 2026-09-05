# Model Benchmark — Proactive HPC Node Health Monitoring

## Goal

Systematically evaluate a list of classification models over a shared feature set produced by the data pipeline described in the TCC slides.

---

## Data pipeline

**Reference specification (authoritative):** `D:\_fcall\UNIFEI\tcc\hpc_workload\TCC_Dados_ML.pdf`
The pipeline **must follow exactly** what is described in this PDF: 8 metrics (node + GPU) based on Skrzeczek 2024 and Cai et al. 2025, 2h pre-failure labeling window, temporal split by job `end_date` (80/20), 40-snapshot sliding window with min/max/mean/std aggregations (32 features per sample), majority-vote window labels, GPU nodes only (`gpu_node=1`), `CANCELLED` excluded.

**Reference implementation (example only, do not follow blindly):** `D:\_fcall\UNIFEI\tcc\hpc_workload\dataset\ml_pipeline.ipynb`
This notebook contains a working example of the pipeline and may be useful for understanding the data structures and the parquet loading logic. However, if it diverges from the PDF specification in any way, **the PDF wins**. Re-implement the pipeline from scratch following the PDF; use the notebook only as a reference for library idioms, column names, and file paths.

---

## Data pipeline outputs (compute once, reuse across all models)

Materialize and save to disk:

- `X_train.npy`, `X_test.npy` — `float32`, shape `(n_windows, 32)`
- `y_train.npy`, `y_test.npy` — `int8`
- `states_train.npy`, `states_test.npy` — original job state per window (`TIMEOUT` / `FAILED` / `OOM` / `NODE_FAIL` / `COMPLETED`), used for per-state metrics
- `job_ids_train.npy`, `job_ids_test.npy` — `slurm_id` per window
- `feature_names.json` — the 32 feature names in column order
- `scaler.joblib` — `StandardScaler` fit on `X_train` only (used by linear models and MLP; ignored by tree-based models)

All models must consume these artifacts directly. **No model recomputes the sliding window.**

---

## Full-dataset training

Every model is trained on the **entire training set** and evaluated on the **entire test set**. No subsampling.

### Memory management (16 GB RAM budget)

The laptop running this benchmark has 16 GB RAM. The pipeline and benchmark must be designed to fit within this limit. Guidance:

- Materialize `X_train` and `X_test` as `float32` (half the memory of `float64`, negligible impact on metrics).
- During the parquet load, select only the columns actually needed for the 8 metrics + job metadata; do not load the full 100-column table into a single pandas DataFrame.
- Use PyArrow to load the parquet with column projection and, if needed, filter by `gpu_node=1` at the Arrow level before materializing to pandas.
- Free intermediate objects explicitly (`del df; gc.collect()`) after the windowed arrays are saved.
- For each model, load the `.npy` artifacts fresh at the start of the model's training function and free them after evaluation, so peak memory does not stack across models.
- If a specific model implementation requires materializing a dense copy of the training set (e.g., some scikit-learn estimators internally copy to `float64`), prefer the estimator variant that supports `float32` inputs natively. LightGBM, XGBoost, and HistGradientBoosting all handle `float32` without internal copies.
- If a single model still exceeds memory during training despite the above, log the failure in `results.csv` and continue with the next model — do not silently subsample.

---

## Metrics (computed for every model)

Single evaluation function taking `(y_true, y_score, states)` and returning:

- `pr_auc` (`average_precision_score`) — primary metric
- `roc_auc`
- `recall@0.5` global
- `recall@0.5` per state (`TIMEOUT`, `FAILED`, `OOM`, `NODE_FAIL`)
- `mean_proba` per state (diagnostic — shows whether the model ranks positives correctly even when threshold is not crossed)
- `train_time_s`, `predict_time_s`

**Incremental logging:** each model appends its row to `results.csv` immediately after evaluation. If the process crashes, completed runs are preserved.

---

## Execution protocol

1. **Pass 1 — base configurations**: run every model listed below with the single base configuration. Purpose: fast ranking.
2. **Pass 2 — grid search**: only for the top 3–4 models by PR-AUC from Pass 1. Use `GridSearchCV` with `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` **on the training set only**. Selection metric: `average_precision`. Evaluate the best estimator on the full test set.

Fixed seed everywhere: `random_state=42`. Use `n_jobs=-1` where supported.

Wrap each model in `try/except` so a single failure (e.g., a missing dependency or OOM) does not abort the whole benchmark.

Compute at the start:
```python
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
```

---

## Models — Pass 1 (single base configuration)

### Tier 0 — Baselines

| # | Model | Base configuration |
|---|---|---|
| 1 | `DummyClassifier` | `strategy='prior'` |
| 2 | `GaussianNB` | defaults (scaled X) |
| 3 | `LogisticRegression` | `class_weight='balanced'`, `max_iter=2000`, `solver='lbfgs'` (scaled X) |

### Tier 1 — Trees and ensembles

| # | Model | Base configuration |
|---|---|---|
| 4 | `DecisionTreeClassifier` | `class_weight='balanced'`, `max_depth=10` |
| 5 | `RandomForestClassifier` | `n_estimators=300`, `class_weight='balanced'`, `n_jobs=-1` |
| 6 | `ExtraTreesClassifier` | `n_estimators=300`, `class_weight='balanced'`, `n_jobs=-1` |
| 7 | `HistGradientBoostingClassifier` | `class_weight='balanced'`, `max_iter=300` |
| 8 | `LightGBM` | `n_estimators=300`, `num_leaves=31`, `learning_rate=0.1`, `scale_pos_weight=<computed>` |
| 9 | `XGBoost` | `n_estimators=300`, `max_depth=6`, `learning_rate=0.1`, `scale_pos_weight=<computed>`, `tree_method='hist'` |
| 10 | `CatBoost` | `iterations=300`, `depth=6`, `learning_rate=0.1`, `auto_class_weights='Balanced'`, `verbose=0` |

### Tier 2 — Imbalance-aware methods

| # | Model | Base configuration |
|---|---|---|
| 11 | `BalancedRandomForestClassifier` (imblearn) | `n_estimators=300`, `sampling_strategy='auto'`, `replacement=True`, `bootstrap=False`, `n_jobs=-1` |
| 12 | `EasyEnsembleClassifier` (imblearn) | `n_estimators=20`, `n_jobs=-1` |
| 13 | `RUSBoostClassifier` (imblearn) | `n_estimators=200`, `learning_rate=0.1` |

### Tier 3 — Non-tree models

| # | Model | Base configuration |
|---|---|---|
| 14 | `MLPClassifier` | `hidden_layer_sizes=(64, 32)`, `alpha=1e-4`, `early_stopping=True`, `max_iter=200` (scaled X) |
| 15 | `SGDClassifier` | `loss='log_loss'`, `class_weight='balanced'`, `max_iter=1000` (scaled X) |

---

## Grids — Pass 2 (top 3–4 models from Pass 1 only)

### LightGBM
```python
{
    'num_leaves': [15, 31, 63],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [200, 500],
    'min_child_samples': [20, 100, 500],
    'scale_pos_weight': [scale_pos_weight],
}
```

### XGBoost
```python
{
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [200, 500],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'scale_pos_weight': [scale_pos_weight],
    'tree_method': ['hist'],
}
```

### RandomForest
```python
{
    'n_estimators': [200, 500],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 20, 100],
    'max_features': ['sqrt', 0.5],
    'class_weight': ['balanced', 'balanced_subsample'],
}
```

### HistGradientBoosting
```python
{
    'max_leaf_nodes': [15, 31, 63],
    'learning_rate': [0.05, 0.1],
    'max_iter': [200, 500],
    'l2_regularization': [0.0, 1.0],
    'min_samples_leaf': [20, 100],
}
```

### BalancedRandomForest
```python
{
    'n_estimators': [200, 500],
    'max_depth': [None, 10, 20],
    'sampling_strategy': [0.1, 0.5, 1.0],
    'min_samples_leaf': [1, 20, 100],
}
```

### MLP
```python
{
    'hidden_layer_sizes': [(64,), (64, 32), (128, 64)],
    'alpha': [1e-4, 1e-3, 1e-2],
    'learning_rate_init': [1e-3, 1e-4],
}
```

---

## Output

`results.csv` with one row per (model, configuration), columns:

```
model, family, pass, params, train_time_s, predict_time_s,
pr_auc, roc_auc,
recall_global,
recall_TIMEOUT, recall_FAILED, recall_OOM, recall_NODE_FAIL,
mean_proba_TIMEOUT, mean_proba_FAILED, mean_proba_OOM, mean_proba_NODE_FAIL, mean_proba_COMPLETED
```

At the end, print a table sorted by `pr_auc` descending and highlight the top 3 for Pass 2.

---

## Project organization

Design the project folder structure to make the benchmark reproducible, debuggable, and easy to extend. Suggested layout (adapt as needed):

```
hpc_workload/
├── benchmark/
│   ├── data_pipeline.py       # loads parquet, labels, splits, windows, saves .npy artifacts
│   ├── evaluation.py          # single eval function returning all metrics
│   ├── models.py              # base configs for Pass 1
│   ├── grids.py               # parameter grids for Pass 2
│   ├── run_pass1.py           # loops through all models, appends to results.csv
│   ├── run_pass2.py           # runs GridSearchCV for the top-K models
│   └── utils.py               # logging, timing, seed setup
├── artifacts/
│   ├── X_train.npy, X_test.npy, y_train.npy, y_test.npy
│   ├── states_train.npy, states_test.npy
│   ├── job_ids_train.npy, job_ids_test.npy
│   ├── feature_names.json
│   └── scaler.joblib
├── results/
│   ├── results.csv            # incremental log, one row per model run
│   └── pass2_grid_results/    # per-model grid search details
└── README.md                  # how to reproduce
```

Additional organizational requirements:
- Each script should be runnable independently (e.g., `python -m benchmark.run_pass1`).
- The data pipeline script should be idempotent: if the artifacts already exist, skip regeneration unless a `--force` flag is passed.
- Fixed random seed (`42`) set at the top of every script.
- Log every model's start time, end time, peak memory (via `resource.getrusage` or `psutil`), and any exception traceback to a plain-text log file alongside `results.csv`.
- Requirements pinned in `requirements.txt`: `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `imbalanced-learn`, `pyarrow`, `pandas`, `numpy`, `joblib`, `psutil`.
