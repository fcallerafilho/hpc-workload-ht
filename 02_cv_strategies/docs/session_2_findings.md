# Session 2 — Findings (results and recommendation)

*What the Session-2 work produced. Companion: `session_2_methodology.md`. New reader? Start with
`session_2_overview.md`.*

> **Terminology.** **Session 1** = the benchmark = **Pass 1** (15 models ranked once) **+
> Pass 2** (the **hyperparameter grid search**). **"Pass 2" = that grid search**, which is what
> regressed and what Session 2 re-did correctly.

---

## 1. Exact imbalance and the corrected baseline

| Split | Windows | Positives | Fraction | Note |
|---|---:|---:|---:|---|
| Train | 9,125,945 | 291,936 | **0.0320** | training prior (`scale_pos_weight = 30.26`) |
| Test | 1,411,809 | 56,831 | **0.0403** | **← PR-AUC baseline** |

Verified by assertion: test fraction **= Dummy `pr_auc`** (0.0403) and train fraction **=
Dummy `mean_proba`** (0.0320). **ROC-AUC baseline = 0.5.**

*A "positive" is a **window label** (majority vote of its 40 snapshots), i.e. a window in the
final 2 h of a job that failed — not a job state and not a single snapshot. The baseline counts
positive **windows** because that is the unit the model predicts and is scored on.*

**Per-state positives (test):**

| State | Windows | Positives | Positive within state | Share of all positives |
|---|---:|---:|---:|---:|
| COMPLETED | 926,422 | 0 | 0.000 | 0.000 |
| TIMEOUT | 408,660 | 39,857 | 0.098 | **0.701** |
| FAILED | 51,962 | 12,862 | 0.248 | 0.226 |
| OOM | 1,606 | 1,592 | **0.991** | 0.028 |
| NODE_FAIL | 23,159 | 2,520 | 0.109 | 0.044 |

Two things jump out: **TIMEOUT supplies 70 %** of all positive windows (it dominates the
objective), while **OOM is tiny but almost entirely positive** (1,592 / 1,606) — rare yet
very learnable, which is why per-state OOM recall is the highest despite OOM being a sliver
of the data.

## 2. Session-1 results, re-read against the correct baseline (×0.0403)

| Model | PR-AUC | ×-lift | ROC-AUC | Verdict |
|---|---:|---:|---:|---|
| CatBoost | 0.0795 | **1.97×** | 0.608 | weak (1.2–2×) |
| LightGBM | 0.0793 | 1.97× | 0.618 | weak |
| HistGradientBoosting | 0.0752 | 1.87× | 0.620 | weak |
| EasyEnsemble | 0.0704 | 1.75× | 0.566 | weak |
| LogisticRegression | 0.0626 | 1.55× | 0.577 | weak |
| XGBoost | 0.0595 | 1.48× | 0.603 | weak |

So the honest headline is **~2× a correct random baseline**, not the ~2.5× implied by the
old 0.032 anchor. "Good" for extreme-imbalance PR-AUC is read as ×-lift; these are a real
but weak signal — consistent with ROC-AUC ~0.62.

## 3. CV diagnosis — the leakage, made visible

Same small LightGBM grid, six CV strategies; CV-AP is what the grid *believes*, test-PR-AUC
is the truth.

| Strategy | folds | CV-AP (believed) | test PR-AUC (truth) | inflation |
|---|---:|---:|---:|---:|
| stratified (leaky) | 3 | **0.463** | 0.081 | **5.7×** |
| group | 3 | 0.215 | 0.081 | 2.7× |
| stratified_group | 3 | 0.216 | 0.077 | 2.8× |
| logo_weekly | 9 | 0.136 | 0.077 | 1.8× |
| purged | 5 | 0.120 | 0.081 | **1.5×** |
| tscv | 5 | 0.088 | 0.081 | **1.1×** |

The leaky `stratified` CV believes **0.463** while the model truly scores **0.081** — a
**5.7× overstatement**. That inflated number is what led Pass-2 to pick over-complex models
and regress. **Grouping alone still overstates by ~2.7×** (it removes same-job overlap but not
concurrent-job / temporal leakage); only **temporal** CV is honest — `tscv` is within **1.1×**
of the truth and `purged` within **1.5×**. This is the quantitative case for the advisor's
time-aware CV: grouping is necessary but **not sufficient**.

## 4. Key finding — temporal validation is required to pick complexity

Selecting the number of boosting rounds for one LightGBM, two leak-free monitors disagree:

| Monitor | rounds it selects | test PR-AUC there | where test actually peaks |
|---|---:|---:|---|
| grouped-random (same period) | ~260 | **0.066** (worst) | — |
| **temporal** (latest 15 %) | ~120 | **~0.083** (best) | 100–200 rounds ≈ 0.082–0.084 |

Test PR-AUC by round (same model): `20→0.077, 40→0.083, 100→0.082, 150→0.083, 300→0.077`.
More trees keep improving the same-period AP but **degrade** the future/test AP. **Grouping
removes same-job leakage but not temporal over-optimism; only a temporal monitor tracks the
future.** This both fixes the Pass-2 regression and slightly **beats** the Session-1 base.

## 5. Honest tuned results

Structure + rounds selected on the **temporal holdout** (test-blind), validated under the
**purged** CV, refit on full train, scored on the temporal test set.

| Model | tuned PR-AUC | base | Δ vs base | ×-lift | purged-CV AP | rounds |
|---|---:|---:|---:|---:|---:|---:|
| **CatBoost** | **0.0860** | 0.0795 | **+0.0066** | **2.13×** | 0.112 | 18 |
| LightGBM | 0.0782 | 0.0793 | −0.0011 | 1.94× | 0.122 | 92 |
| HistGradientBoosting | 0.0699 | 0.0752 | −0.0053 | 1.73× | 0.117 | 50 |

**CatBoost wins** (depth 8, lr 0.1, `auto_class_weights='Balanced'`, **18 rounds**): a
genuine, test-blind **+0.0066 over the Session-1 best (0.0795)**, reaching **2.13×** baseline
and matching the project's prior-iteration reference (**~0.088**). Note the honest contrast
with Session 1's Pass-2, which *regressed* to 0.0697/0.0612 — the same models, tuned under a
valid CV, now **improve** instead. (LightGBM's temporal holdout happened to prefer
`num_leaves=63`; a config that scores lower on test than `31` would — but re-picking on the
test set is leakage, so 0.0782 is the honest number.)

## 6. Operating point (CatBoost)

PR-AUC is threshold-free; deployment needs a threshold. For proactive draining a missed
failure costs more than a false drain → recall-favoring **F2**.

| Operating point | threshold | precision | recall | F2 | flagged |
|---|---:|---:|---:|---:|---:|
| default @0.5 | 0.500 | 0.057 | 0.627 | 0.208 | 627,422 |
| target-recall @0.80 | 0.381 | 0.046 | 0.819 | 0.187 | 1,020,369 |
| best F2 | 0.542 | 0.062 | 0.560 | 0.214 | 514,816 |

At the F2 point the model catches **56 %** of failures at **6.2 %** precision — the honest
precision/recall trade of an extreme-imbalance early-warning task. Precision is low by
construction (positives are ~4 % of windows); the value is catching failures early enough to
drain, not perfect precision.

## 7. Ensemble and calibration

- **Ensemble does not help here.** Mean = 0.0840, rank-mean = 0.0818 — both **below** the best
  single (CatBoost 0.0860), because the weaker HGB drags the average down. **Recommend the best
  single model, not the ensemble.**
- **Calibration** (best model, temporal-holdout fit): the class-weighted scores are badly
  mis-calibrated — Brier **0.276**. Calibration cuts it **~7×**: **sigmoid 0.038**, isotonic
  0.040. **Sigmoid (Platt) preserves PR-AUC exactly** (0.0823, strictly monotonic); **isotonic
  shifts it slightly** (→0.0702) because it maps ranges to tied values. → **Use sigmoid
  calibration**: same ranking, trustworthy probabilities.

## 8. Recommendation and honest expectation

- **Recommended thesis model:** **CatBoost** (depth 8, lr 0.1, `auto_class_weights='Balanced'`,
  ~18 rounds), selected by temporal early stopping, reported at the **F2** operating point with
  **sigmoid-calibrated** probabilities. **PR-AUC 0.0860 = 2.13× the 0.0403 baseline.**
- **Fixing CV was the real win:** it removed the Pass-2 regression and turned tuning from
  −0.018 into **+0.0066**. The corrected baseline makes the numbers honestly interpretable.
- **Temporal validation** is the decisive methodological point — grouping alone still overstates
  by 2.7× (§3), and same-period round-selection overfits the future (§4).
- **Honest ceiling:** all model/eval work lands the leaders at **~0.078–0.086** (≈ the prior
  ~0.088). Beating that meaningfully needs **pipeline / feature** changes, deliberately left for
  a later, clearly-separated session.
