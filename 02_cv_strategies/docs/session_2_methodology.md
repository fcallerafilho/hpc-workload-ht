# Session 2 — Methodology (honest CV, exact baseline, and improvements)

*Every decision made in this session and why. Companion: `session_2_findings.md`.
All work lives in `cv_strategies/` and reuses the frozen Session-1 artifacts without
modifying any Session-1 file.*

---

> **Terminology recap.** **Session 1** = the original benchmark = **Pass 1** (train 15 models
> once to rank them) **+ Pass 2** (the **hyperparameter grid search** on the top models).
> **"Pass 2" always means that grid search.** **Session 2** (this document) changed no models
> and no data — it corrected how we *judge* the scores and re-ran the **Pass 2** search with an
> honest, time-aware cross-validation.

## 0. Where Session 1 left us (the initial state)

- Best model **PR-AUC ≈ 0.079** (CatBoost/LightGBM), **ROC-AUC ≈ 0.62** — weak but real.
- **Pass 2 (the grid search) regressed** (0.0793 → 0.0612 for LightGBM) due to CV leakage.
- The baseline had been eyeballed as ~0.032; **no baseline-relative interpretation** was
  exposed.
- The thesis advisor asked specifically for time-series/grouped cross-validation
  (`TimeSeriesSplit(gap)`, `GroupKFold`, `LeaveOneGroupOut`, `PurgedKFold`).

The remit: make the metrics **interpretable**, and **exhaust every model/evaluation/CV
improvement before changing the data pipeline** (window size, features, labeling stay
frozen).

## 1. The exact PR-AUC baseline (fixing the anchor)

The PR-AUC of a random/constant classifier equals the **positive prevalence of the set it
is evaluated on**. Our models are scored on the **test** set, so the honest baseline is
the **test** positive fraction.

The Session-1 Dummy row proves the distinction:
- Dummy **`mean_proba` = 0.0320** — `strategy='prior'` emits the **train** prior for every
  row, so this is the *training* positive fraction.
- Dummy **`pr_auc` = 0.0403** — average precision of a constant predictor equals the
  prevalence of the *evaluated* set, i.e. the **test** positive fraction.

We compute both directly from `y_train`/`y_test` and **assert** they match the Dummy
(`imbalance.py`). The baseline is **0.0403**; every "×-lift" downstream uses it.
Consequence: the best model is **~1.97×** baseline, not the ~2.5× implied by the wrong
0.032 anchor. **ROC-AUC baseline = 0.5.**

*Why it matters:* a lift is only meaningful against the right zero. Anchoring to the
train prior flatters the results and would misinform the model choice.

## 2. Why `StratifiedKFold(shuffle)` leaks here

The sliding window has **39/40 overlap**, so consecutive windows of one job are
near-duplicate rows. Shuffling scatters a job's windows across folds; each validation
fold is then full of rows nearly identical to training rows → **optimistic CV**. The fix
is CV that respects **which job** a row belongs to and **when** it occurred.

## 3. The six CV strategies and what each controls (`cv.py`)

| Strategy | Guarantee | Removes |
|---|---|---|
| `stratified` (baseline) | class-balanced random folds | *nothing* — the leaky control |
| `group` (GroupKFold, groups=`slurm_id`) | a whole job is in one fold | same-job overlap leakage |
| `stratified_group` | group integrity **+** class balance | same-job overlap, with stable fold prevalence |
| `tscv` (TimeSeriesSplit, gap=39) | train strictly precedes validation in time | look-ahead; gap purges the boundary overlap |
| `purged` (PurgedGroupKFold) | time-blocked folds, jobs kept whole, train jobs overlapping the validation time span (+2 h embargo) are **purged** | same-job **and** concurrent-job temporal leakage |
| `logo_weekly` (LeaveOneGroupOut over ISO-weeks) | a whole calendar week held out; each job assigned to one week | temporal leakage at week granularity |

**Why weekly LOGO.** Per-job `LeaveOneGroupOut` would mean ~18 k folds — infeasible. We
realize the advisor's LOGO idea at **week** granularity (each job assigned to its
end-time's week, so jobs are never split), giving ~9 folds that hold out whole weeks.

Each splitter carries a **self-test** (`python -m cv_strategies.cv`) asserting its
guarantee: no job on both sides (group/purged), `max(train_time)+gap ≤ min(val_time)`
(tscv), and no train row inside the embargoed validation span (purged).

## 4. The window-time artifact (`timekey.py`)

`TimeSeriesSplit` and `PurgedGroupKFold` need a per-window time key, which the frozen
artifacts do not contain, and `X_train` is job-blocked rather than globally time-sorted.
We re-read **only** `[slurm_id, timestamp, end_date]` (same `gpu_node=1` + CANCELLED
filters, same sort, same 80/20 split), window the timestamp identically, and take each
window's **last** snapshot time as its end-time → `win_time_{train,test}.npy` (int64 ns).
This changes **no** feature, window size, label, or `X`. Alignment is **proven**, not
assumed: we rebuild the job-id-per-window sequence and assert it equals the saved
`job_ids_*` element-for-element.

## 5. The key finding — temporal vs. same-period validation

*(This is the most important methodological result of the session; numbers in
`session_2_findings.md` §3.)*

When selecting **model complexity** (number of boosting rounds), we compared two leak-free
monitors on the *same* LightGBM:
- a **grouped-random** holdout (whole jobs, from the same time period), and
- a **temporal** holdout (the latest 15 % of train jobs).

The grouped-random monitor's average precision keeps **rising** with more trees and points
to ~260 rounds — which is exactly where the **temporal test** performance is **worst**.
The temporal monitor points to ~120 rounds — right where the **test** performance
**peaks** (and *beats* the Session-1 base).

**Conclusion:** removing same-job overlap (grouping) is **necessary but not sufficient**.
A leak-free *same-period* validator still **overfits the future**; only a **temporal**
validator selects complexity that generalizes forward. This is precisely why the advisor's
*time-aware* CV matters, not merely grouping.

## 6. The honest tuning method (`tuning.py`)

Driven by §5:
1. **Select structure + rounds** by fitting each candidate with **early stopping on the
   temporal holdout** — one fit covers all round counts, and the round count is chosen to
   generalize forward. (LightGBM/CatBoost use their eval-set early stopping; HGB is
   staged via `warm_start`.) The eval metric is average precision throughout; we found and
   fixed a plumbing bug where LightGBM was watching `binary_logloss` and stopping at round
   1 under `scale_pos_weight=30`.
2. **Robustness check** the chosen config under the advisor's **purged** CV (report its
   mean ± std AP across time-blocked folds).
3. **Refit** on the full training set at the chosen structure + rounds and score on the
   temporal test set. Success criterion: tuned **≥** base (the Session-1 regression
   reversed).

Separately, a cheap **diagnosis** (`--diagnostic`) runs a small LightGBM grid under **all
six** strategies and records CV-AP vs test-AP, making the leakage visible as a table.

## 7. The operating point (`evaluation_ext.py`)

PR-AUC is threshold-free, but deployment needs a threshold. For proactive node draining a
**missed failure costs more than a false drain**, so we report, on the best model:
- **target-recall @ 0.80** — the highest-precision threshold that still catches 80 % of
  failures, and
- **best F-β with β = 2** — the recall-favoring operating point,
with global precision/recall and **per-state recall** at each. This turns a ranking score
into an actionable policy.

## 8. Ensemble and calibration (`ensemble_calib.py`)

- **Ensemble.** Plain-mean and rank-mean of the tuned LightGBM/CatBoost/HGB probabilities
  — cheap diversity that often adds a little PR-AUC.
- **Calibration.** The class-weighted boosters rank well but are **mis-calibrated**
  (probabilities are inflated — raw Brier ≈ 0.28). We fit isotonic and sigmoid calibrators on
  the temporal holdout and apply them to the test scores; Brier drops ~7× (≈ 0.038). A
  **strictly**-monotonic map (**sigmoid / Platt**) preserves PR-AUC/ROC-AUC exactly; **isotonic**
  is monotonic *with ties*, so it can shift PR-AUC slightly (it maps score ranges to a single
  value). We therefore recommend **sigmoid** calibration: identical ranking, trustworthy
  probabilities, so the chosen threshold means what it says.

## 9. Why our numbers look lower than Skrzeczek's — and why that's expected

Skrzeczek reports **accuracy 0.95 / F1 0.97 / ROC-AUC 0.98**. That is **not** the same
task:

| | Skrzeczek | This work |
|---|---|---|
| Unit | whole **job** (aggregated) | rolling 40-snapshot **window** |
| Size | ~920 jobs (667/253) | ~11.1 M windows / 23.6 k jobs |
| Positive class | **successful** job (majority) | **pre-failure** window (minority) |
| Imbalance | ~2.6 : 1 | ~30 : 1 |
| Metrics | accuracy/F1/ROC-AUC | **PR-AUC**, per-state recall |
| Timing | post-hoc outcome | **2 h before end**, deployable |

His 0.98 ROC-AUC reflects a much **easier, majority-positive, post-hoc** problem, not a
better model. Our task — extreme-imbalance, per-window, **early** prediction of the rare
failure — is harder by construction, and PR-AUC against the correct 0.0403 baseline is the
honest way to judge it. This framing is a **defensible contribution**, not a shortfall.

## 10. What we deliberately did **not** touch

Features, the 8-metric selection, the 40-snapshot window, the 2 h labeling horizon, the
majority-vote rule, and the 80/20 temporal split are **unchanged**. Everything here is
evaluation/model-side. If the results plateau (they do; see §5 and the findings), the next
lever is the **pipeline** — which is left for a subsequent, clearly-separated session.
