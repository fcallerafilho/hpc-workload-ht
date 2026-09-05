# 01_benchmark — Model Benchmark (Session 1, Aug 6–7)

Systematic comparison of ~15 classifiers on a shared windowed feature set built
from the SURF Lisa `prom_slurm_joined` dataset. See `benchmark_models.md` for the
protocol and `../docs/references/TCC_Dados_ML.pdf` for the authoritative
data-pipeline spec.

## Layout

```
benchmark/
  config.py         paths, seed, the 8 features, states, constants
  data_pipeline.py  parquet -> label -> temporal split -> sliding window -> .npy
  evaluation.py     evaluate(y_true, y_score, state_codes) -> all metrics
  models.py         Pass-1 base configs (15 models)
  grids.py          Pass-2 parameter grids (6 grid-able models)
  run_pass1.py      train every base model once, rank by PR-AUC
  run_pass2.py      GridSearchCV for the top-K from Pass 1
  utils.py          seeding, timing, peak-RAM sampling, logging, CSV append
notebooks/          a parallel, step-by-step view of the same benchmark
artifacts/          windowed .npy + feature_names.json + state_names.json + scaler.joblib
results/            results.csv (incremental), benchmark.log, pass2_grid_results/
```

`artifacts/` is gitignored (1.4 GB) and rebuilt by `data_pipeline`. Session 2
(`../02_cv_strategies/`) reuses these artifacts frozen, and imports
`benchmark.config` read-only, so the two can never drift.

## Reproduce

Run from this folder (`01_benchmark/`):

```bash
pip install -r ../requirements.txt

python -m benchmark.data_pipeline      # build artifacts once (--force to rebuild)
python -m benchmark.run_pass1          # 15 base models -> results/results.csv
# inspect the Pass-1 ranking, then:
python -m benchmark.run_pass2 --top 4  # or --models LightGBM,XGBoost,...
```

## Key design points

- **Compute-once artifacts.** The windowed matrix is materialized as `.npy` and
  reused by every model. `run_pass1` and `run_pass2` are separate processes, so
  the data must persist to disk; `X` is loaded read-only via `mmap_mode='r'`.
- **16 GB RAM budget.** Only 14 of 100 columns are read (PyArrow projection +
  `gpu_node=1` predicate); `float32` throughout; each model is isolated and freed
  before the next; Pass-2 grid search runs `n_jobs=1` (estimators use the cores)
  so the training matrix is never duplicated across processes.
- **Reproducibility.** `random_state=42` everywhere. Aggregations use `np.std`
  (ddof=0) and majority window labels (`>=20 -> 1`), matching the reference
  notebook where the PDF is silent on mechanics.
- **States.** `states_*.npy` store `int8` codes; `state_names.json` maps codes to
  raw Slurm states. `OUT_OF_MEMORY` is displayed as `OOM` in output columns only.
  (In this project "OOM" = the OUT_OF_MEMORY *job state*, not machine RAM.)

## Metrics (results.csv)

`pr_auc` (primary), `roc_auc`, `recall_global`, per-state `recall@0.5`
(TIMEOUT/FAILED/OOM/NODE_FAIL) and per-state `mean_proba` (adds COMPLETED),
plus `train_time_s`, `predict_time_s`, `peak_rss_mb`, `status`.
