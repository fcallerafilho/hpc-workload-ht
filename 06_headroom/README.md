# headroom/ — Where is the missing performance, and can any of it be recovered?

`experiments/` established that node sharing is *not* what limited the earlier models:
the Bayes-optimal ceiling is `AP ≈ 0.90` while the best measured model reached `0.125`.
This folder finds out what the remaining gap actually is, tests the cheapest levers
against it, and ends with a configuration that has a usable operating point.

**The result.** A 30-minute horizon with **seven columns of job context and no telemetry
at all** reaches **job-level `AP = 0.5017`** (95 % CI [0.457, 0.546], 3.58× baseline).
Flagging 50 jobs gives **96 % precision**. It is a walltime-exhaustion detector, not a
hardware-failure detector — and that distinction is the finding, not a caveat.

## The notebooks

| notebook | what it does | runtime |
|---|---|---|
| `01_headroom.ipynb` | Capacity ladder, label-permutation control, learning curve. Decides between capacity / variance / signal. | ~4 min |
| `02_context.ipynb` | Telemetry vs job context vs both, crossed with failure mode. | ~1 min |
| `03_horizon.ipynb` | Horizon sweep and job-level evaluation. | ~3 min |
| `04_short_horizon.ipynb` | Three-way temporal split; selection on window-level lift. **Produces a bad answer, kept deliberately.** | ~5 min |
| `05_protocol_v2.ipynb` | Same split, corrected criterion and a minimum-sample guard. The final result. | ~4 min |
| `06_oom_precursor.ipynb` | Mechanistic check on raw snapshots: does memory drift before an OOM kill? | ~1 min |
| `07_delta_features.ipynb` | Can the pipeline absorb per-job baseline features? | ~2 min |

Each opens with a **pre-registration** cell. Depends on `eda/data/{jobs,windows}.parquet`.
`data/` is gitignored, `figures/` is committed.

## Findings

### 1. The effective sample size is jobs, not windows

Test AP is flat across the capacity ladder (0.0797 → 0.0852) while train AP climbs
0.578 → 0.995 and the train/test gap widens to 15.2×. The permutation control explains
it: at a **matched positive rate**, shuffled labels still reach train AP **0.944**. The
training score is ~95 % memorisation — windows inside a job are near-duplicates, so the
model stores job signatures.

**Effective training set ≈ 10,265 jobs (~1,700 failing), not 915,508 windows.** This
retires "train on a denser stride": more windows are correlated rows, not new examples.
The learning curve is not resolvable (span 0.013 < seed spread 0.020).

### 2. Job metadata beats node telemetry

At the 2 h horizon, window level: telemetry (56 cols) `AP 0.0797` = 1.95×; context
(9 cols) `AP 0.1903` = 4.65×; both = 4.74×.

Telemetry wins in exactly one failure mode — **`OUT_OF_MEMORY`, 8.07× vs 4.15×** — where
the memory metrics measure the thing that actually fails. On hardware failures generally,
*adding* telemetry to context makes it worse (2.26× → 1.98×).

The predicted mechanism was wrong: `sec_into_hour` and `sec_to_next_hour` contribute
0.5 % and 0.3 % of split gain. The work is done by `elapsed_hours_floor` (28 %),
`numcores` (20 %), `queue_code` (15 %).

### 3. The calendar columns were hurting, not helping

`dow` and `hour_of_day` were suspected of encoding *when in the dataset we are*. They do
worse than that — they are pure overfitting fuel. Dropping them wins at **7 of 7**
horizons on window lift and **6 of 6** on job-level AP. The honest feature set is also
the better one.

### 4. Selecting on lift picks the worst model — a demonstrated methodological result

`04` selected on validation window-level lift and chose a 10-minute horizon at
**117.97×**, which collapsed to **14.16×** on test, with an operating point of 0.7 %
precision catching **1 failing job out of 127**.

The cause is structural, not bad luck. **Lift is `AP / base_rate`, so shortening the
horizon shrinks the denominator and inflates lift mechanically even as the model gets
worse.** Compounding it, the winning cell rested on 152 validation positives, and taking
a maximum over 28 such cells is a textbook winner's curse.

`04` is kept in the repository unchanged. The contrast with `05` is the point.

### 5. The corrected protocol converges

`05` changes two things and nothing else: a horizon must yield **≥ 500 validation
positives** to be eligible, and selection is on **validation job-level average
precision** — which, unlike lift, measures the same quantity at every horizon because a
job's label does not move when the horizon does.

| | validation | test | shrinkage |
|---|---|---|---|
| `04` window lift | 117.97× | 14.16× | **8.3×** |
| `05` job AP | 0.5760 | **0.5017** | **1.15×** |

Selected: **30 minutes, `C7` context, 7 columns, no telemetry.**

| metric | value | 95 % CI |
|---|---|---|
| job-level AP | **0.5017** | [0.4570, 0.5461] |
| job-level lift | 3.58× | |
| window-level AP | 0.1494 | [0.1114, 0.1906] |
| window-level lift | 14.19× | |

Operating points on test (2,694 scorable jobs, 378 failing):

| jobs flagged | precision | recall | true alerts |
|---|---|---|---|
| 10 | 100.0 % | 2.6 % | 10 |
| 25 | 100.0 % | 6.6 % | 25 |
| **50** | **96.0 %** | 12.7 % | 48 |
| 100 | 72.0 % | 19.0 % | 72 |
| 200 | 60.5 % | 32.0 % | 121 |
| 400 | 42.8 % | 45.2 % | 171 |

### 6. It is a walltime detector

At the selected configuration: **TIMEOUT 29.82×**, hardware failures **2.06×**
(FAILED 2.07×, NODE_FAIL 2.30×, OUT_OF_MEMORY 3.29×).

The honest claim is *"a job's queue, size and elapsed runtime predict imminent walltime
exhaustion well"*. It is a scheduling result. Node telemetry contributes nothing to it —
the selected model contains no telemetry column at all.

### 7. OOM has a real memory signal, and it is not gradual

`06` works on **raw snapshots**, not windows — the median OOM job runs 11 minutes, so 73
of 130 never produce a 20-minute window at all. Features are **baseline-relative**:
change from the median of each series' first 10 snapshots, a family never tried before
(session 3 tested per-*node* normalisation, not per-*job*).

Node active memory, baseline-relative, separates OOM from matched COMPLETED jobs at
**AUC 0.871, 95 % CI [0.802, 0.923]** (58 OOM jobs). For scale, the best single telemetry
feature anywhere in `eda/` reached |AUC − 0.5| = 0.12; this is 0.371.

* **Not job size in disguise.** Stratified by core count, where `numcores` is constant:
  AUC 0.722 (3 cores), 0.913 (6), 0.933 (12). `numcores` alone reaches only 0.756.
* **Not a co-tenancy artefact.** Exclusive queues only: AUC 0.793 [0.598, 0.923], n = 11.
* **Not a death signature, and not degradation either.** |AUC − 0.5| by lead time:
  0.372 / 0.326 / 0.315 / 0.373 / 0.365 at 0–5, 5–15, 15–30, 30–60, 60–120 minutes before
  death. **Flat.** OOM-bound jobs are not deteriorating — they occupy an elevated-memory
  state from early on and stay in it. Operationally that is favourable: full signal
  strength with two hours of lead time.

Ceiling: **130 OOM jobs exist in the entire dataset** (0.55 % of GPU jobs). A strong
feature cannot fix a prevalence that low.

### 8. The pipeline can represent that signal — and still cannot learn it

`07` adds 14 per-job delta columns (each window's mean minus its series' first window).

The signal survives aggregation almost intact: `mem_active__delta` separates at **AUC
0.890 on windows** against 0.927 on raw snapshots. So windowing is *not* the culprit.

But the model cannot use it. At the 2 h horizon, T → T+D: window AP 0.0797 → 0.0740
(−7 %), OOM lift 8.07× → 5.21× (−35 %), while job AP rises 0.3938 → 0.4385 (+11 %; +38 %
at 30 min). Adding 14 correlated columns to 56 degrades the pooled metric and helps the
max-over-windows one — the signature of a variance-limited model.

**None of the four pre-registered outcomes describes this.** The architecture is not
wrong and the features are not wrong; the binding constraint is the one from finding 1
and it has not moved: ~1,700 effective failing jobs.

### 9. Correction — telemetry does *not* win on OOM

Finding 2 above reports telemetry beating context on `OUT_OF_MEMORY` (8.07× vs 4.15×).
That used the **9-column** context set including the two calendar columns which finding 3
showed to be harmful. With the clean 7-column set, context scores **10.64×** on OOM
against telemetry's 8.07×. **The claim that OOM is the one mode where node telemetry
beats job metadata does not survive and should not be repeated.**

What does survive is narrower: the stratified analysis in `06` shows memory deltas
separating OOM jobs *within* fixed core-count strata, where metadata cannot be doing the
work. The telemetry information exists; it is not extractable by a model at this sample
size.

## Honest limitations

1. **Two denominators, and both belong in any quoted figure.** Job-level numbers cover
   only jobs that produced at least one complete 20-minute window. `eda/` measured that
   two thirds of failing jobs never do — 378 of roughly 1,158 failing test jobs are
   scorable. So "50 alerts, 96 % precision, 48 caught" is 12.7 % of *observable*
   failures and about **4 % of all failures**.
2. **This is the third selection pass.** `03` explored, `04` selected badly, `05`
   selected properly. All three are reported. The final number comes from one refit and
   one evaluation, but the sequence is part of the record.
3. **100 % precision at 10–25 flagged jobs rests on small counts** and should be quoted
   with them.
4. **56 features, not 144.** Absolute levels are not comparable to `pipeline_v2`'s 3.12×.
5. **No hyper-parameter was tuned anywhere** — one fixed configuration everywhere,
   deliberately, after session 3 showed tuning on adjacent folds cost 27 % of PR-AUC.
6. **`deterministic=True` is pinned** in `04`, `05` and `07`; earlier notebooks used
   default threading, which wobbles ~6 % between runs.
7. **`06` has no train/test split.** It is a descriptive statistic over the whole 68-day
   period — appropriate for a mechanistic claim, but not a held-out prediction. The
   by-core-count AUCs are point estimates on 18–20 OOM jobs per stratum, without intervals.
8. **Two findings here correct earlier ones in this same folder** (findings 3 and 9). The
   superseded versions are left in place with the correction attached, rather than
   quietly edited.

## Requirements

Python 3.13 with `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `lightgbm`,
`matplotlib`, `jupyter`. Peak memory ~3 GB.
