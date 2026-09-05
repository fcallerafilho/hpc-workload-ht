# Session 3 — findings

*Pipeline rebuilt from scratch in `pipeline_v2/`. Nothing from Sessions 1–2 was reused; the time
invariants were.*

## Headline

| | PR-AUC | random baseline | lift |
|---|---|---|---|
| Session 1 (best of ~15 classifiers) | 0.0795 | 0.0403 | 1.97x |
| Session 2 (honest CV + tuning) | 0.0860 | 0.0403 | 2.13x |
| **Session 3 (this pipeline)** | **0.1253** | 0.0402 | **3.12x** |

95 % cluster-bootstrap interval, resampling test *jobs* rather than windows (windows inside a job are not
independent): PR-AUC [0.0946, 0.1726] =
[2.35x, 4.30x].

Same test population as Sessions 1–2 — the same 80/20 job split by end date, every test window kept at
stride 1 — which is why the random baselines agree to four decimals and the comparison is legitimate.

**Say "about three times better than chance", not a precise multiple.** The same configuration fitted on
85 % of train instead of 100 % scores 0.1382 (3.44x), and a paired
bootstrap cannot separate the two (-0.0097, 95 % CI
[-0.0258, +0.0114]). The figure reported above is the pre-registered one —
refit on all of the training data — and quoting the other would be picking the luckier of two fits this
experiment cannot tell apart.

## What produced the gain

1. **Measuring more of the node.** The old pipeline used 8 metrics. This one uses 36. In notebook 03's
   ablation ladder that single step is worth **+0.059 PR-AUC** (0.0788 -> 0.1382), and +0.066 counted
   together with the rate conversion below — essentially the entire improvement.
2. **Rates instead of raw counters.** A cumulative counter carries the job's *age*, not the node's health;
   converting the three counters to per-second rates was worth +0.006 on its own.
3. **Not much else helped.** Trajectory columns (delta, slope) carry under 1 % of model gain. Per-node
   normalisation scored 0.0837 and node identity 0.1172, both well below the plain expanded set's 0.1382.
   Hyperparameter tuning made things worse (below).

## The methodological result

Four purged, expanding-window folds can **reject** but cannot **choose**. Of twenty challengers, four came
out resolvably worse than the incumbent (all four folds agreeing, |t| >= 2) — the old 8-metric feature set
by a wide margin — so the folds are not blind. But not one candidate came out resolvably *better*: no
top-K column set, none of eight hyperparameter configurations, neither CatBoost nor HistGradientBoosting.
The fold-to-fold range of a *single* candidate (0.082 to 0.191) is larger than the spread across *all
twenty-one* candidates within one fold.

Obeying the cross-validated argmax anyway cost 30.7 % of PR-AUC (3.44x -> 2.38x; both figures come from
the 85 %-train protocol notebooks 04 and 05 use throughout, which is why they are quoted against 3.44x and
not against the headline). Notebook 04 takes that
apart: the argmax's 50 columns were harmless (-0.6 %), but its deeper trees made early stopping halt at 26
rounds instead of ~150 (-27.4 %), and matched-round refits show only a quarter of that loss is the trees
themselves. The temporal holdout early stopping watches barely registered the damage (AP 0.164 vs 0.170)
because the holdout is adjacent in time while the test set is further away.

**The rules that follow:** do not inherit a round budget from early stopping on a near holdout; do not
adopt a candidate the folds cannot separate from the incumbent; and report a paired interval before
claiming any difference at all.

## The pipeline decisions, and what the alternatives cost

| decision | kept | alternatives | note |
|---|---|---|---|
| window | 40 snapshots (20 min) | 20 -> 2.94x, 10 -> 3.14x | W=40 wins on lift and lead time, W=10 on coverage |
| horizon | 2 h | 0.5 h -> 2.59x, 1 h -> 3.31x, 4 h -> 2.59x | raw PR-AUC rises with horizon only because the baseline does |
| label rule | majority | any -> 2.91x | |
| positive class | all four failure states | hardware only -> 1.94x | see the caveat below |
| train stride | 5 | 10 -> 3.23x, 20 -> 2.89x | curve still rising: denser stride untested |
| split | by job end date, 80/20 | embargo already implied by the protocol | |

## The caveat that matters most

TIMEOUT is 70 % of the positives, and a TIMEOUT is usually a user under-estimating walltime rather than a
sick node. Restrict the positive class to genuine hardware failures (FAILED / OUT_OF_MEMORY / NODE_FAIL)
and the result collapses from 3.44x to 1.94x (both on the 85 %-train
protocol, so the ratio between them is the meaningful part).

The job-level table says the same thing more concretely:

| state | failed jobs | had a window | alerted | operational recall |
|---|---|---|---|---|
| TIMEOUT | 298 | 224 (75%) | 177 | 59.4% |
| FAILED | 772 | 113 (15%) | 84 | 10.9% |
| OOM | 25 | 19 (76%) | 16 | 64.0% |
| NODE_FAIL | 63 | 22 (35%) | 16 | 25.4% |

FAILED jobs are 67 % of all test failures, and only 15 % of them last long enough to produce a single
20-minute window. **Any claim that this model predicts node health has to be made at
1.94x, not 3.12x.**

## Operating characteristics

At the holdout-chosen F2 threshold the model flags 28.1% of all
windows at 0.070 precision — recall-heavy and, on its own, not an
operable setting. The alert-budget view is the useful one:

| budget | windows | precision | recall | precision lift |
|---|---|---|---|---|
| top 0.1% | 1,412 | 0.900 | 0.022 | 22.4x |
| top 0.5% | 7,162 | 0.343 | 0.043 | 8.5x |
| top 1.0% | 14,291 | 0.228 | 0.058 | 5.7x |
| top 2.0% | 28,144 | 0.230 | 0.114 | 5.7x |
| top 5.0% | 70,363 | 0.155 | 0.192 | 3.8x |
| top 10.0% | 140,719 | 0.113 | 0.282 | 2.8x |

The top 0.1 % of windows are 90% true positives. That is the number to
take to an operator: a small, high-purity queue, not a threshold that flags a quarter of the cluster.

Calibration: the raw output is a ranking, not a probability — its mean is 0.42 against an actual positive
rate of 0.0402, for a Brier score of 0.2278. An isotonic
map fitted on the holdout brings Brier to 0.0466, at a cost of
-12.7% of PR-AUC — isotonic is
monotone *non-decreasing*, so it collapses score ranges into ties and PR-AUC charges for the lost ordering.
Rank and alert with the raw scores; calibrate only at the point where a probability is actually read.

Round budget is not load-bearing: across 40–600 rounds test PR-AUC stays within
0.1145–0.1253, a spread entirely inside the bootstrap interval above.

## The job-level number, which is the honest one

Of **1158 test jobs that failed**, only 378
(32.6%) ran long enough to produce even one 20-minute window, and
293 were alerted — **25.3% of all failures** —
at a false-alarm rate of 50.3% on completed jobs. Median warning:
152 minutes overall, but only 36 minutes on
hardware failures; the long tail belongs to TIMEOUT jobs the model recognised early as long-running, which
is not the same as seeing a failure coming.

Window coverage, not model quality, is the binding constraint. Notebook 05's window sweep is the first look
at buying it back: W=10 covers 50% of failing jobs instead of
33% and alerts 29.3% of all failures instead
of 24.4%, at the cost of lift and of a much shorter warning.

## Not comparable with Skrzeczek

That thesis reports ROC-AUC ~0.98 by aggregating each job's entire time series into one row and then
classifying it — the features include the failure. This work predicts from a 20-minute window that ends
before a 2-hour horizon opens. Different question, different unit, different metric.
