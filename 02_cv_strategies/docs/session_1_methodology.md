# Session 1 — Methodology (the initial benchmark)

*What we built first, and why. This is the state of the work before the Session-2
changes. Companion: `session_1_findings.md`.*

---

## 1. The task

Proactive health monitoring of HPC **GPU compute nodes**: from a short, rolling
window of a node's recent metrics, predict whether the running job is heading for a
**failure** in the near future, early enough for Slurm to drain the node and requeue
the job. The benchmark covers the **ML component only** — training and evaluating
classifiers on real historical data.

- **Input.** A sliding window of **40 consecutive snapshots** (15 s each ≈ a 10-minute
  window) of one job, summarized as **min / max / mean / std** over **8 selected
  metrics → 32 features** per window.
- **Label.** `1` if the window lies within the **2 hours before the end** of a job
  that ended in a failure state; `0` otherwise.
- **Positive (failure) states.** `FAILED`, `TIMEOUT`, `OUT_OF_MEMORY`, `NODE_FAIL`.
  `COMPLETED` is the healthy class. `CANCELLED` is excluded (user-driven, not a health
  signal).

> **Vocabulary.** `OUT_OF_MEMORY` (shown as **OOM**) is a **Slurm job state**. It is
> *not* the 16 GB laptop running out of RAM. The two are kept strictly separate
> throughout.

## 2. Data

`prom_slurm_joined` from the **SURF Lisa** academic cluster (Chu et al., ICPADS 2024),
~13.5 M rows joining Prometheus node metrics with Slurm job metadata. We use **GPU
nodes only** (`gpu_node = 1`): **11,144,842 snapshots across 23,618 jobs**.

**The 8 metrics** (feature selection from **Skrzeczek 2024**): node memory active,
disk bytes written, TCP in-errors, forks; GPU temperature, power, memory used, fan
speed. Column order fixes the 32-feature order (`[min×8 | max×8 | mean×8 | std×8]`).

## 3. Pipeline (authoritative spec: `TCC_Dados_ML.pdf`)

1. **Load** only the 14 needed columns via PyArrow with a `gpu_node=1` predicate (never
   the full 100-column table) — a memory decision (see §7).
2. **Label** each snapshot: `1` iff its job is a failure job **and** the snapshot is in
   the 2 h before `end_date`.
3. **Temporal split.** Sort jobs by `end_date`; the first **80 %** are train, the last
   **20 %** test. *Why temporal:* a random split would let a model peek at the future;
   splitting by job end date mimics deployment (train on the past, predict the future).
4. **Window.** Per `slurm_id`, a size-40 stride-1 sliding window (39/40 overlap).
   Jobs with < 40 snapshots are dropped (~55 % of windows survive).
5. **Window label.** Majority vote — `1` if ≥ 20 of the 40 snapshots are positive.
6. **Aggregate.** min/max/mean/std (std `ddof=0`), reductions done in float64 then cast
   to float32.

Result: **X_train 9,125,945 × 32**, **X_test 1,411,809 × 32**;
`scale_pos_weight = (y==0)/(y==1) ≈ 30.26`.

## 4. Why PR-AUC

The positive class is rare, so **ROC-AUC is misleading** (a model can look strong while
being useless at the rare-event decision). **PR-AUC (average precision)** is the primary
metric. **Per-state recall** (TIMEOUT/FAILED/OOM/NODE_FAIL) is tracked separately,
because a single global number hides that some failure modes are gradual and learnable
while others are instantaneous and essentially unpredictable from prior metrics.

## 5. Models (Pass 1 — one base config each)

15 classifiers spanning: baselines (Dummy, GaussianNB, LogisticRegression); trees and
ensembles (DecisionTree, RandomForest, ExtraTrees, HistGradientBoosting, **LightGBM**,
**XGBoost**, **CatBoost**); imbalance-aware methods (BalancedRandomForest,
EasyEnsemble, RUSBoost); and non-tree models (MLP, SGD). Linear/NB/MLP consume
StandardScaler-transformed inputs; tree/boosting models are scale-invariant and use the
raw features. Class imbalance is handled per family (`class_weight='balanced'`,
`scale_pos_weight`, or `auto_class_weights`).

## 6. Protocol

> **Terminology.** "Session 1" is this whole benchmark, and it runs in **two stages**:
> **Pass 1** (train every model once to rank them) followed by **Pass 2** (the
> **hyperparameter grid search** for the best models). Whenever this project says
> **"Pass 2" it means the grid search.**

- **Pass 1** — every model trained **once** with its base config, ranked by PR-AUC (fast
  triage of which family wins).
- **Pass 2 (the grid search)** — for the top models, **`GridSearchCV`** tries many
  hyperparameter combinations using
  **`StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`**, scoring
  `average_precision`; the best estimator is then scored on the full temporal test set.
  *(This cross-validation choice is the one Session 2 shows to be the bug.)*
- Fixed **seed 42** everywhere; each model wrapped in `try/except`; one row appended to
  `results.csv` immediately after each run.

> This Pass-2 CV choice is exactly what Session 2 revises — see `session_1_findings.md`
> §3 and `session_2_methodology.md`.

## 7. Memory design (16 GB laptop)

Every model trains on the **full** dataset — no subsampling. To fit in 16 GB: `float32`
matrices; the windowed arrays are materialized **once** as `.npy` and reused by every
model; `X_*` are loaded as a **read-only memmap** (shared across processes, never
duplicated); states stored as `int8` codes + a name map, job ids as `int32`; per-model
`gc`; and Pass-2 grids run `n_jobs=1` so a self-threaded estimator never forks copies of
the 1 GB matrix.

## 8. Reference / inspiration — Skrzeczek 2024

Skrzeczek's Utrecht MSc (`Ignacy_Master_Thesis-5.pdf`) is the **source of the 8-feature
selection** and the informal comparison target. Its evaluation is **not directly
comparable** to ours — a point developed in `session_2_methodology.md` §9 — because it
classifies **whole jobs** (aggregated), on a small, mildly imbalanced set, with the
**successful** job as the positive class, reporting accuracy/F1/ROC-AUC rather than
PR-AUC.
