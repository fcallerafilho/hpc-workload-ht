# allnodes/ — What happens when the failing-job shortage is removed

Every earlier folder in this project measured 51 GPU nodes. `headroom/` ended by naming the
binding constraint: **~1,700 failing jobs**, not the model, not the features, not
co-tenancy. That diagnosis makes a falsifiable prediction, and the dataset can test it —
the same recording holds 257,012 non-GPU jobs this project had never touched.

This folder rebuilds the pipeline over **all 338 nodes** and varies only the amount of
data. Same window definition, same labelling rule, same temporal split, same model
settings.

**The result in one line.** The diagnosis was right, the remedy worked, and it bought
nothing for the telemetry: **8.5× more failing jobs, and the sensor model's score does not
move.** What it did buy is the ability to ask a question that was previously unanswerable —
one detector per failure mode — and the answer to that is the useful part.

## The notebooks

| notebook | what it does | runtime |
|---|---|---|
| `01_build.ipynb` | Cluster-wide census, job table, chunked window build, invariants. | ~5 min |
| `02_scale.ipynb` | The scale test: learning curve over 1,000 → 60,380 training jobs. | ~5 min |
| `03_oom.ipynb` | The memory signal replicated, and a dedicated OOM detector. | ~6 min |
| `04_verdict.ipynb` | One detector per failure mode; the pooled protocol. | ~8 min |
| `05_controls.ipynb` | Four controls that try to break `02` and `03`. | ~3 min |
| `06_timeout.ipynb` | **Correction:** `04`'s round-hour measure was broken. | <1 min |

Each opens with a **pre-registration** cell. `01` reads the raw parquet; the rest read
`data/windows_all.parquet`. `data/` is gitignored, `figures/` is committed.

## What was built

8,045,769 windows × 36 features, 826 MB, in 230 s. The build streams the 200 parts in
groups of eight and appends each group's windows straight to the output, because 85 million
snapshots do not fit in this machine's free memory. The parts turn out to be **sorted by
node**, so a `(job, node)` series can only straddle a group boundary at the last node of a
group; those rows are carried forward. The assert that the recovered series count equals
the census count — 312,885 = 312,885, and 85,440,430 snapshots on both sides — is what
proves no series was silently cut in half.

**The feature set had to shrink to 36 columns.** The five `nvidia_gpu_*` metrics are
identically zero on the 287 non-GPU nodes, so the shared set is 9 node-level metrics ×
{min, max, mean, std}. This is deliberate: if "more data" and "more features" moved
together, neither could be attributed. It also means **no number here is comparable to
`headroom/`'s** 56-column results, so the GPU-only baseline is re-measured rather than
quoted.

| | jobs | failing | OUT_OF_MEMORY | failing **and** scorable |
|---|---|---|---|---|
| GPU (all prior work) | 23,618 | 6,312 | 130 | 2,097 |
| whole cluster | 280,630 | 60,450 | 4,499 | **17,808** |

Two things the build got wrong in advance:

* **The window positive rate went *up*, not down** (3.37 % GPU → 4.10 % non-GPU), against
  pre-registration B2. Non-GPU jobs are far shorter (median 3 min against 32 min), but
  short jobs produce no window at all and never enter the denominator. Among jobs that do
  produce one, non-GPU failures spend a larger share of their telemetry inside the horizon.
* **The observation period doubled.** Non-GPU telemetry starts 30 June, GPU telemetry
  25 August, so the record covers 123 days rather than 68 and the split lands at
  2022-10-17. That pushes most GPU jobs into the late folds and handicaps every GPU-only
  arm — which is why `05` control B exists.

## Findings

### 1. More data helps the job paperwork and does nothing at all for the telemetry

Fixed protocol, three seeds per rung, a test set that never changes:

| training jobs | failing jobs in it | job context (7 cols) | node telemetry (36 cols) |
|---|---|---|---|
| 1,000 | 138 | 0.3852 | 0.3858 |
| 5,000 | 723 | 0.4856 | 0.4002 |
| 20,000 | 2,872 | 0.5133 | 0.3851 |
| 40,000 | 5,706 | 0.5217 | 0.3939 |
| 60,380 | 8,638 | **0.5216** | **0.3842** |

Context rises 35 % and flattens around 20,000 jobs. **Telemetry is flat from the first rung
to the last** — 0.3858 at 1,000 jobs, 0.3842 at 60,380, well inside the seed spread.

Sixty times the data changes nothing for the sensors. That is no longer a sample-size
result; it is a statement about what the columns contain. Whatever node-level telemetry
knows about an impending failure, it already knows after a thousand jobs.

### 2. The memorisation `headroom/` diagnosed was real, and it dissolves at scale

The label-permutation control at a matched positive rate — same test, same fix for the
marginal, 60,380 jobs instead of 10,265:

| | train AP on shuffled labels |
|---|---|
| GPU nodes (`headroom/`) | 0.9439 |
| whole cluster | **0.2070** |

The train/test ratio falls with it, 2.59 → 1.49. `headroom/` finding 1 identified the
mechanism correctly: the training score was job fingerprinting, and eight times as many
fingerprints makes it collapse. **The diagnosis was right and the remedy worked — it simply
did not buy what it was expected to buy.**

### 3. Telemetry transfers across hardware; job context does not

`05` control B, both arms trained on every job ending before the split, both evaluated on
the same 3,043 GPU test jobs:

| | trained on GPU jobs | trained on the whole cluster |
|---|---|---|
| node telemetry (36) | 0.4444 | **0.4905** (+10 %) |
| job context (7) | **0.5022** | 0.4760 (−5 %) |

Training on machines that are *not* the ones being predicted makes the **sensors** better
and the **paperwork** worse. Node-level telemetry describes physical behaviour that carries
from one hardware type to another; queue names, job sizes and elapsed-time patterns are
specific to the population they were learned on.

**This corrects `02`.** `02` reported 0.5056 against 0.4404 and concluded telemetry
overtakes context on GPU jobs. Its GPU-only arm was fitted on 606 failing GPU jobs because
of the split described above. With the fair arm (1,618 failing GPU jobs) the honest reading
is narrower: cluster training helps telemetry, but **context is still the better single
model on GPU jobs** (0.5022 against 0.4905).

### 4. One detector per failure mode — the table `eda/` could not produce

At 107 `NODE_FAIL` and 130 `OUT_OF_MEMORY` jobs, three of these four rows were noise. At
1,426 and 4,499 they are measurable. Validation job-level AP, 2 h horizon, and each
detector trained against **one** mechanism:

| failure mode | jobs | base rate | job context | node telemetry | both | winner |
|---|---|---|---|---|---|---|
| `TIMEOUT` | 3,376 | 22.4 % | 0.7574 | 0.7200 | **0.7948** | context |
| `FAILED` | 618 | 4.09 % | 0.0369 | **0.0666** | 0.0765 | **telemetry** |
| `NODE_FAIL` | 251 | 1.66 % | **0.0591** | 0.0329 | 0.0306 | context |
| `OUT_OF_MEMORY` | 120 | 0.79 % | 0.0330 | **0.0384** | 0.0532 | **telemetry** |

Three readings, in order of how much they matter:

* **Telemetry wins where the mechanism is physical.** On `FAILED` the job paperwork scores
  **below chance** (0.90× its base rate) while telemetry reaches 1.63× — the sensors are
  carrying all of it. On `OUT_OF_MEMORY` telemetry beats context 4.84× to 4.15×. These are
  the two modes about what the program does to the machine.
* **`NODE_FAIL` is predicted by job size, which is not prediction.** Context wins there,
  and the reason is visible in the raw rates: the `NODE_FAIL` rate is 0.52 % for one-node
  jobs, 9.8 % at 5–8 nodes and **23.1 % above 16 nodes**. A job spread over more machines
  has more chances to meet a broken one. The model is measuring **exposure**, not detecting
  an impending hardware fault.
* **`TIMEOUT` is easy and worth nothing.** Job AP 0.79 against a 22 % base rate. It is also
  what any pooled model ends up detecting — see finding 5.

### 5. The pooled model reaches a genuinely usable operating point, and 97 % of its alerts
are walltime exhaustion

Protocol identical to `headroom/05`. Selected: **4 h horizon, 7 context columns, no
telemetry**. Validation job AP 0.7546 → **test 0.4641, 95 % CI [0.4511, 0.4772]**, against a
28.97 % base rate (1.60×). Shrinkage 1.63×.

| jobs flagged | precision | recall | true alerts |
|---|---|---|---|
| 100 | **100.0 %** | 2.1 % | 100 |
| 200 | 98.5 % | 4.1 % | 197 |
| 400 | 95.3 % | 7.9 % | 381 |
| 800 | 90.8 % | 15.1 % | 726 |
| 1,600 | 58.6 % | 19.5 % | 938 |

That is a far better operating point than the GPU-only result (50 alerts at 96 %). And the
top 200 alerts are **194 `TIMEOUT`, 3 `NODE_FAIL`, 3 `COMPLETED`**. The tool predicts that
a job is about to exhaust its walltime, which is a scheduling result and not a hardware one.

Pre-registration V1 asked whether `TIMEOUT` lift would still exceed hardware-failure lift by
5× and got 2.56× against 1.24× — **REJECTED, and the verdict should be ignored.** Lift
divides by a base rate, the base rates differ between the two populations being compared,
and this project has already documented what happens when lift is used across populations
(`headroom/04`). The alert composition above answers the same question without that defect.

### 6. `OUT_OF_MEMORY`: the mechanism replicates, in the opposite form

Matched controls (same queues, same runtime band), raw snapshots, final five minutes,
1,863 OOM jobs against 3,000 controls:

| | GPU nodes, 58 jobs | whole cluster, 1,863 jobs |
|---|---|---|
| absolute node active memory | not the headline | **0.8566 [0.8463, 0.8657]** |
| baseline-relative | **0.871 [0.802, 0.923]** | 0.6115 [0.5918, 0.6296] |

The interval is six times narrower and it lands somewhere else. **A memory signal is
unambiguously there — it is in the absolute level, not in the change from a job's own
baseline.** `headroom/` finding 7 was measured on 58 jobs and its specific claim, that the
baseline-relative form carries the signal, **does not survive**.

Two controls make the absolute result quotable:

* **Not a node-class artefact** (`05` control C). Computed inside each queue separately,
  where the machine type is essentially fixed: 0.844 (`shared`, 828 OOM jobs), 0.817
  (`shared_52c_384g`, 32), 0.775 (`normal`, 325). Median 0.817.
* **Available two hours ahead** (`05` control D, matched controls compared at equal
  distance from their own job's end): AUC 0.858 / 0.847 / 0.844 / 0.822 / 0.787 at 0–5,
  5–15, 15–30, 30–60 and 60–120 minutes before death. It decays gently rather than
  appearing at the last moment.

The baseline-relative form stays at 0.43–0.52 in every band even after matching, so the
"jobs drift away from their own baseline" story is not what is happening. The coherent
reading is that OOM-bound jobs **start** at a high memory level and grow little.

### 7. And the OOM detector is still not an alarm anyone would switch on

Best configuration: 2 h horizon, telemetry + per-job delta columns. Telemetry beats context
1.41× and the delta columns add 11 % — both of them reversals of what `headroom/` measured
at n = 57. Test job AP **0.0560, 95 % CI [0.0475, 0.0669]**, 3.3× a 1.67 % base rate.

| jobs flagged | true OOM jobs caught | precision |
|---|---|---|
| 50 | 1 | 2 % |
| 200 | 5 | 2.5 % |
| 800 | 73 | 9.1 % |

A 3.3× lift on a 1.7 % event is a real signal and a useless operating point. Reporting the
lift without this table would be exactly the mistake `headroom/04` is kept in the repository
to illustrate.

The gap between finding 6 and finding 7 is the interesting part: a single sensor separates
OOM jobs from healthy ones at AUC 0.86, and a trained model over 45 columns cannot turn that
into a usable ranking. The separation is between OOM jobs and *matched* controls; the
ranking task is against **all** 16,301 healthy jobs, and at a 1.7 % prevalence an AUC of
0.86 simply does not deliver precision.

### 8. Correction — `04`'s `TIMEOUT` round-hour measure was broken

`04` reported that the round-hour signature failed to replicate off the GPU nodes: 53.9 % of
non-GPU `TIMEOUT` jobs ran within 60 seconds of a whole number of hours, but so did 34.7 %
of `COMPLETED` jobs.

**Zero is a whole number of hours.** Any job shorter than a minute passes the test
automatically, and the median non-GPU job runs three minutes. `06` applies a 10-minute
runtime floor:

| | `TIMEOUT` | `COMPLETED` | ratio |
|---|---|---|---|
| GPU nodes | **92.6 %** | 1.9 % | 48× |
| non-GPU nodes | 58.1 % | 2.3 % | 26× |

The GPU figures reproduce `eda/`'s 92.6 % and 1.9 % exactly, from a second implementation on
a second pass over the data. The signature **does** replicate cluster-wide, an order of
magnitude weaker in absolute terms. By queue it tracks who chooses the walltime:
`shared_jupyter` 100 %, the GPU queues 84–99 %, `normal` 64.6 %, `shared` 47.6 %.

`04`'s V2 verdict stands as rejected **by a broken measure, not by the data**. It is left in
place unedited with this correction attached.

## Honest limitations

1. **Every model here trains on thinned data** — one window every 20 minutes rather than
   every 5. `02` pre-registered that this costs under 0.02 job AP and measured 0.048 on the
   GPU subset, so `05` control A re-measured it at cluster scale: **−0.032 at 2,500 jobs,
   −0.001 at 10,000, +0.006 at 40,000**. At the scale everything here is trained at the
   penalty is indistinguishable from zero, and at small samples thinning actually helps.
   The pre-registered *shape* (a monotonically shrinking penalty) is wrong; the protocol it
   was defending is fine.
2. **36 features, not 56.** Absolute levels are not comparable to `headroom/` or
   `pipeline_v2`.
3. **The split is cluster-native**, which handicaps GPU-only arms. `02`'s S4 was confounded
   by this and `05` control B corrects it; quote control B.
4. **`03` step 4's lead-time profile used unmatched controls** and is confounded by position
   inside the job. `05` control D is the matched version.
5. **Two `04` verdicts are unsafe**: V1 uses lift across populations with different base
   rates, V2 uses a broken measure. Both are reported here with what to read instead.
6. **No hyper-parameter was tuned anywhere**, deliberately, as in every folder since
   session 3.
7. **The observability ceiling has not moved.** 60,450 jobs failed and 17,808 are scorable;
   4,499 hit `OUT_OF_MEMORY` and 1,303 are scorable. More data raises the number of
   observable failures, never the share.

## Requirements

Python 3.13 with `pandas`, `numpy`, `pyarrow`, `scipy`, `scikit-learn`, `lightgbm`,
`matplotlib`, `jupyter`. Peak memory ~2.2 GB; the build itself streams and stays under
~600 MB.
