# experiments/ — Does node sharing explain the plateau?

A pre-registered test of the strongest claim to come out of `eda/`: that because
several jobs share one node's telemetry, the labels are ambiguous, and that this
ambiguity is what held the earlier models to ~3x baseline.

**It does not.** The ambiguity is real and exactly as large as `eda/` measured, but the
ceiling it imposes sits far above where the models actually are. That is the result.

## The notebooks

Run in order. `01` writes what `02` reads.

| notebook | what it does | runtime |
|---|---|---|
| `01_bound.ipynb` | Enumerates **every** potential window (no stride), measures co-tenancy multiplicity per `(node, instant)`, and computes the exact Bayes-optimal precision–recall bound per queue class. | ~40 s |
| `02_experiment.ipynb` | Four arms of LightGBM under one fixed protocol, bootstrap intervals over test jobs, measured performance against the bound. | ~70 s |

Both open with a **pre-registration** cell stating the hypotheses and the falsifier
before any result appears. The protocol (features, rounds, split, metric) is fixed in
the setup cell and applied identically to every arm; nothing was tuned.

Depends on `eda/data/{jobs,windows}.parquet`, so run the `eda/` notebooks first.
`data/` is gitignored; `figures/` is committed.

## What was predicted, and what happened

| | prediction | result |
|---|---|---|
| **H1** | exclusive queues have `k = 1` | **confirmed** — all 993,002 windows, no exceptions |
| **H2** | shared queues bound below `AP = 1` | **confirmed** — `AP = 0.8675` on test |
| **H3** | the bound explains the ~0.125 plateau | **rejected** — the bound is `0.902`, 7.2x higher |
| **P1** | exclusive scores higher than shared | **not resolved** — 1.201, 95 % CI [0.510, 2.253] |
| **P2** | the gap matches the bound (1.153) | consistent, but **the test is not discriminating** |
| **P3** | more data beats cleaner labels | **confirmed** — `C/D = 1.145`, `B/D = 0.777` |

## The findings

- **The natural experiment is clean.** `gpu` and `gpu_titanrtx` have co-tenancy
  multiplicity exactly 1 for every one of their 993,002 windows; shared queues run 2–4
  jobs per node. There is no grey zone to argue about.

- **Node sharing costs 13.3 % of achievable average precision** on shared queues
  (19.3 % on train) — the Bayes-optimal `AP` falls from 1.000 to 0.867. Real, and much
  smaller than the share of *positives* affected (54.2 %). Those two quantities are
  different and only the first one bounds a model.

- **The ceiling is not what is binding.** `pipeline_v2` measured `AP = 0.1253` against
  a ceiling of `0.902`. About seven eighths of the distance to a perfect classifier is
  due to something other than label ambiguity, so the `eda/` interpretation — that
  co-tenancy explains the plateau — is wrong and should not be repeated.

- **Both populations are equally hard relative to their own ceilings.** One model
  trained on all queues attains 9.7 % of the ceiling on exclusive queues and 9.3 % on
  shared ones. The populations differ in how much is *achievable*, not in how hard it
  is to approach what is achievable — which is what the bound predicts.

- **Training on clean labels is worse than training on more data.** Arm B (exclusive
  train, `AP = 0.0589`) loses to a size-matched shared model (arm D, `0.0758`) and to
  the full shared model (arm C, `0.0867`). Restricting the pipeline to exclusive queues
  is not a good idea.

## Honest limitations

1. **The primary test is underpowered.** 48 positive jobs on the exclusive side give a
   ratio interval of roughly [0.5, 2.3]. The point estimate is close to the prediction,
   but an interval that wide would have accepted almost any prediction. The notebook
   prints this rather than claiming a confirmation.
2. **This is a natural experiment.** Queues differ in job length, workload and users as
   well as in co-tenancy; the bound says how much of the gap co-tenancy *should*
   explain, and the remainder is confounded.
3. **Arm D matches windows, not jobs** — arm B trains on ~399 jobs against D's ~830, so
   `B/D < 1` is not attributable to label quality alone.
4. **Exclusive training data contains no NODE_FAIL positives** while its test set has 66.
5. **Bootstrap clusters are jobs, and co-resident jobs are not independent**, so the
   intervals are slightly optimistic — the same caveat applies to session 3's.
6. **56 features, not 144.** Absolute numbers are below the 3.12x headline in
   `pipeline_v2` by construction. Every arm shares the same 56 columns, so the
   contrasts are valid; the levels are not comparable to that figure.

## Requirements

Python 3.13 with `pandas`, `numpy`, `pyarrow`, `scipy`, `scikit-learn`, `lightgbm`,
`matplotlib`, `jupyter`. Peak memory ~3 GB in `01`. Nothing is sampled: the full GPU
population is read at fidelity.
