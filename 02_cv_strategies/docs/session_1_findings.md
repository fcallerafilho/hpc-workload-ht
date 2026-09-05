# Session 1 — Findings (the initial benchmark results)

*What the benchmark produced, and the problems that motivated Session 2. Companion:
`session_1_methodology.md`.*

---

## 1. Pass-1 ranking (base configs, test PR-AUC)

| Rank | Model | PR-AUC | ROC-AUC |
|---|---|---:|---:|
| 1 | **CatBoost** | 0.0795 | 0.608 |
| 2 | **LightGBM** | 0.0793 | 0.618 |
| 3 | **HistGradientBoosting** | 0.0752 | 0.620 |
| 4 | EasyEnsemble | 0.0704 | 0.566 |
| 5 | LogisticRegression | 0.0626 | 0.577 |
| 6 | XGBoost | 0.0595 | 0.603 |
| … | … | … | … |
| 15 | DummyClassifier | 0.0403 | 0.500 |

**Histogram gradient boosting wins** (CatBoost / LightGBM / HGB lead decisively). All
strong models cluster around **PR-AUC ≈ 0.075–0.080** with **ROC-AUC ≈ 0.60–0.62**.

## 2. Per-state recall (base models, @0.5)

| State | Recall | Reading |
|---|---:|---|
| **OOM** | ~0.66 | most predictable — memory pressure builds gradually |
| **TIMEOUT** | ~0.52 | gradual — often a slow hang |
| **FAILED** | ~0.42 | mixed |
| **NODE_FAIL** | ~0.40 | hardest — often instantaneous, no pre-failure signature |

This matches the physics: **gradual** failures leave a trail in the metrics; **abrupt**
ones do not. It is the main reason a single global score understates the model — it is
genuinely good at some failure modes and near-blind to others.

## 3. The Pass-2 regression (the problem)

*(Reminder: **Pass 2 is the hyperparameter grid search** — the second stage of the Session-1
benchmark, after the 15 models were ranked in Pass 1.)*

Grid search was expected to help. It did the opposite:

| Model | Pass-1 base | Pass-2 tuned | Δ |
|---|---:|---:|---:|
| LightGBM | 0.0793 | **0.0612** | −0.0181 |
| CatBoost | 0.0795 | **0.0697** | −0.0098 |

**Cause — CV leakage.** With `StratifiedKFold(shuffle=True)`, the 39/40-overlapping
windows of the *same job* are near-duplicates that get scattered across folds. Every
validation fold then contains rows almost identical to rows the model just trained on,
so the cross-validation average precision is **wildly optimistic** (~0.6–0.7) while the
true temporal test PR-AUC is ~0.06. The grid, trusting the inflated CV, selects the
**most complex** configurations (e.g. LightGBM `num_leaves=63`, 500 trees), which then
**overfit** the temporal test set. This is the central defect Session 2 fixes.

## 4. How the baseline was (mis)read

At this stage the baseline was taken informally as the **Dummy's `mean_proba` ≈ 0.032**
and models were described as "~2.5× baseline." Session 2 shows this is the **training
prior**, not the PR-AUC baseline: the correct anchor is the **test prevalence ≈ 0.0403**
(the Dummy's `pr_auc`), which puts the best model at **~1.97×** baseline. See
`session_2_findings.md` §1.

## 5. Honest read of Session 1

- The leaders are **~2× a correctly-computed random baseline** with **ROC-AUC ~0.62** —
  a **weak-but-real** signal, not a strong classifier.
- Tuning **regressed** because the CV was not valid for overlapping windowed data.
- There was **no baseline-relative interpretation** exposed anywhere, so "is 0.079 good?"
  had no answer on the page.

These three gaps — wrong baseline, invalid CV, no interpretation — are the entire remit
of Session 2.
