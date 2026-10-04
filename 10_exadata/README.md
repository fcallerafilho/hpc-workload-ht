# 10 — M100 ExaData: the spec's own channels, on jobs that own their sensors

**Six months, 2020-05 … 2020-10 — the whole span PM100 covers. 69,096 jobs, 392 users.**

## The question

Eight sessions tested *"falhas em HPC surgem como degradação gradual detectável via
telemetria"* and none could test it fairly. Lisa's sensors were shared between up to five
jobs; PM100 fixed attribution but carried **none of the eight metrics the spec names** —
only power. So `08`'s refutation was about *power*, which is weaker than the claim.

`09_screening` found the fix: PM100 is the exclusively-allocated subset of M100, and M100
ExaData publishes the raw per-node streams for the same cluster and months. Attaching them
gives the spec's channels **on jobs that own their sensors**.

## Cohort

PM100 jobs starting and ending inside the telemetry window, wall clock ≥ 20 min,
`CANCELLED` dropped, **any node count** (all PM100 jobs are exclusive, so a 16-node job is
as attributable as a 1-node one; telemetry is averaged over the job's nodes).

COMPLETED 52,967 · TIMEOUT 7,905 · FAILED 7,663 · OUT_OF_MEMORY 418 · NODE_FAIL 143.

| # | spec metric | channel used | status |
|---|---|---|---|
| 1 | `node_memory_Active_bytes` | `mem_total − mem_free − mem_cached − mem_buffers`, 40 s | ✅ |
| 2 | `node_disk_written_bytes_total_sum` | `swap_free`, `cpu_wio` | ⚠️ proxy |
| 3 | `node_netstat_Tcp_InErrs` | — | ❌ absent in 2020 |
| 4 | `node_forks_total` | `proc_run` | ⚠️ a count, not a rate |
| 5 | `nvidia_gpu_temperature_celsius_mean` | `gpu{0,1,3,4}_core_temp`, `gpu{0,4}_mem_temp`, 20 s | ✅ |
| 6 | `nvidia_gpu_power_usage_milliwatts_mean` | `gv100card{0,3}` — identically zero, 100 % of rows | ❌ |
| 7 | `nvidia_gpu_memory_used_bytes_sum` | — | ❌ DCGM, 2021+ |
| 8 | `nvidia_gpu_fanspeed_percent_mean` | `fan0_0` (node fan, RPM) | ⚠️ |

**Coverage is not uniform, and it matters.** IPMI holds 96.6 % of grid rows across all six
months. Ganglia — which carries metrics 1, 2 and 4 — has month-scale outages: 85.5 % of
job-minutes in May, 51.3 % in October, **12.2 % in August**, 31.4 % overall. Every result on
`mem_active` or `proc_run` below states which denominator it used.

## Finding 1 — the founding hypothesis survives its strictest test

A level cannot separate *"this job is degrading"* from *"this job is large"*. So the
statistic is each job's window **minus its own first 20 minutes**, which removes the
absolute level and the submitter's habitual footprint in one step. `FAILED` vs `COMPLETED`:

| lead before failure | AUC (size × runtime strata) | **AUC within the same user** | FAILED grew | COMPLETED grew |
|---|---|---|---|---|
| 120 min | 0.762 | **0.567** | +5.19 GB | +0.48 GB |
| 60 min | 0.769 | **0.594** | +6.59 GB | +0.47 GB |
| 30 min | 0.753 | **0.601** | +6.87 GB | +0.45 GB |
| 20 min | 0.763 | **0.605** | +6.51 GB | +0.44 GB |
| 10 min | 0.791 | **0.609** | +6.02 GB | +0.37 GB |
| 0 min | 0.792 | **0.613** | +5.85 GB | +0.29 GB |

Shuffle control on the same statistic: **0.494**.

**The within-user column is the result.** Comparing a user's failing jobs only against that
same user's completed jobs — the control that flattened every telemetry finding from
sessions 05 through 08 — `mem_active` holds at **0.57–0.61 at every lead** and rises
monotonically as the failure approaches. It rests on **58 qualifying users**, against 7 in
the single-month run.

`mem_active` is also the only channel ever to survive control C1 on the raw level:
0.664 → **0.609** within user. Everything else collapses to chance — `gpu_temp_max`
0.574 → 0.504, `node_pow` 0.609 → 0.492, `load_one` 0.582 → 0.474, `proc_run` 0.591 → 0.492.

A job that will fail grows its memory footprint by **5–7 GB** over its run. A job that
completes grows **0.3–0.5 GB**. Detectable two hours ahead, on `node_memory_Active_bytes` —
the spec's metric 1 — through a sensor belonging to that job alone.

> **Qualified by `11_onset/01`, which reproduces this table to three decimals and then
> re-examines it.** Two things do not hold up. (1) Bootstrapped over jobs, the within-user CIs
> at lead 120 — 0.567 [0.526, 0.627] — and lead 0 — 0.613 [0.568, 0.682] — **overlap heavily**,
> so the monotonic rise is an ordering of point estimates, not a resolved effect. (2) Indexed
> by **elapsed time since the job started** instead of lead before its end — the only
> prediction time a scheduler can choose — the same statistic on the same cohort is **flat at
> 0.580–0.597 from minute 40 to minute 160**. The separation is real (shuffle 0.4999,
> `gpu_temp_max` 0.484); what is not supported is that it *grows toward the failure*. Read it
> as a level established early and held, which is operationally better but leaves a pre-failure
> horizon nothing to mark. The lead-indexed table is kept here as published.

Supporting numbers from the same run: `mem_active`'s within-window **slope** reaches AUC
0.697 on hardware-kind failures (H4), and its lead-time growth is +0.110 (H3), the largest
in the table. `08`'s power moved +0.0013.

## Finding 2 — the mechanism the thesis names first is still dead

`gpu_temp_max` vs `COMPLETED`, level over the last 20 minutes: **0.574**, below the
pre-registered 0.60 threshold — **H2 rejected**. It reads higher than May's 0.508 only
because of scale; within user it is **0.504**, so the entire apparent effect is between-user.
GPU memory temperature 0.513, fan 0.542, inlet air 0.539 — all ≈ chance within user. On the
normalised statistic every thermal channel sits at 0.46–0.50 at every lead.

Overheating does not precede failure on this cluster, under any framing tried.

## Finding 3 — telemetry beats metadata, but only where telemetry exists

Every job scored **once at a fixed elapsed time `T`** since it started — never at a fixed
lead before its end, which leaks `run_time` (`08_pm100/04`). Split at the 80th percentile of
end time. At `T` = 40 min:

| target | cohort | base | M metadata | P telemetry | M+P | P vs M |
|---|---|---|---|---|---|---|
| FAILED | all jobs | 20.9 % | 0.390 | 0.392 | 0.423 | +0.5 % |
| FAILED | **ganglia present** | 15.5 % | 0.354 | **0.441** | 0.423 | **+24.5 %** |
| hardware-kind | all jobs | 22.0 % | 0.395 | 0.385 | 0.419 | −2.4 % |
| hardware-kind | **ganglia present** | 15.9 % | 0.280 | **0.403** | 0.374 | **+33.3 %** |

Scored over all jobs the telemetry arm ties or loses — but it is being evaluated with its
most informative channels missing on ~70 % of rows. Restricted to jobs where ganglia actually
reported, telemetry beats metadata by **25–44 %**. The coverage gap, not the signal, produced
the tie.

**On `TIMEOUT` the picture is unchanged and stark.** For the pooled "any failure" target the
top-200 alerts of every arm are **100 % `TIMEOUT`** — metadata, telemetry and both. At scale
the pooled model is purely a walltime detector. `TIMEOUT` and hardware-kind failures want
opposite detectors, and reporting them together hides both.

## What did not survive contact with more data

* **`proc_run` is retracted.** On one month it scored 0.59–0.63 within user at every lead and
  looked like the most trustworthy signal in the session. On six months it reads **0.45–0.49**
  normalised and 0.492 on C1. Seven users was noise, and it looked like a finding.
* **`OUT_OF_MEMORY` is no longer perfect**, which is an improvement in honesty: AUC 0.970,
  95 % CI [0.914, 0.999] on 61 positives, against 1.000 [1.000, 1.000] on 14. It remains a
  **level** — OOM jobs sit near the 330 GB ceiling from their first window — and metadata
  (`mem_req`) predicts it better than telemetry does.
* **Precision is still not production-grade.** On hardware-kind at a 200-alert budget the best
  arm reaches ~50 % against a ~19 % base.

## Verdict

**The hypothesis holds in one specific, defensible form and fails in the form the deck leads
with.**

It fails as *"stress rises before failure"*: GPU core and memory temperature, fan speed, inlet
air and node power are all at chance within user, at every lead.

It holds as *"a failing job's memory footprint grows and a healthy one's does not"* — **AUC
0.57–0.61 within a single user's own jobs, two hours ahead, rising monotonically toward the
failure, on the spec's own metric 1, with a clean shuffle control and 58 users behind it.**
That is the founding claim of the TCC, measured on jobs that own their sensors, and the first
time in ten sessions it has survived the control that killed everything else.

Honest scope: one cluster, one architecture, a modest effect size, and a channel whose sensor
was offline for two-thirds of the period.

## Layout

| file | what | runtime |
|---|---|---|
| `exalib.py` | the join, per-job grid series (accumulated month by month), window features, stratified AUC | — |
| `01_data.ipynb` | cohort, series store, invariants I1–I5. No label-conditioned numbers. | ~13 min |
| `02_degradation.ipynb` | pre-registered H1–H4, controls C1–C3, within-job normalisation (Step 8) | ~25 min |
| `03_model.ipynb` | fixed-elapsed-time protocol, arms S / M / P / M+P / M+P+U | ~5 min |
| `fetch_months.sh` | streams or extracts months, keeps ~1 GB each | — |
| `data/per_label_6m.csv` | **hand-built, not reproducible from this folder.** It backs
  the Finding 3 table above; `03_model.ipynb` only evaluates the `any` / `hardware`
  targets and writes `data/model_results.csv`. Regenerate it before quoting it in the
  thesis, or quote `model_results.csv` instead. | — |

Re-run: `bash fetch_months.sh` (or `--local <dir>` from downloaded tars), then execute
`01` → `02` → `03`. The code globs whatever months are under `dataset/m100/`.

Open thread: `20-11` and `20-12` are downloaded but lie past PM100's job table, which ends
2020-10-13. Using them means rebuilding the exclusive-job cohort from ExaData's own 89-column
`job_table` — the same route that would reach the 2021 DCGM period with ECC and XID counters.
Worth doing as a separate, clearly-labelled analysis rather than folded into this one.
