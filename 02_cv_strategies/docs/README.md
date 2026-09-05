# `cv_strategies/` — Session-2 work

Follow-up to the Session-1 benchmark (`../benchmark/`). Goal: make the results
**interpretable** (correct baseline, ×-lift) and **exhaust every model / evaluation /
cross-validation improvement before touching the data pipeline**. Nothing in Session 1 is
modified; this folder reuses only the frozen `../artifacts/*.npy`.

## Documents (read in this order)

0. **`session_2_overview.md`** — one-page plain recap + the diagram + a cross-validation
   primer. **New readers / presentation audience start here.** (Diagram: `session_2_overview.svg`.)
1. `session_1_methodology.md` — how the benchmark was built, and why (the initial state).
2. `session_1_findings.md` — what it produced (Pass-1 ranking, the Pass-2 = grid-search regression).
3. `session_2_methodology.md` — every decision in this session and why (**the "why"**).
4. `session_2_findings.md` — the numbers, evidence, and recommendation (**the "what"**).

The visual companion is `../results/report.html` (open in a browser) — it now opens with the
same diagram and a cross-validation explainer.

**Terminology:** *Session 1* = the benchmark = *Pass 1* (rank 15 models) + *Pass 2* (the
hyperparameter **grid search**). *Session 2* = this work (fix judging + redo the grid search
with honest CV).

## Code (each runnable as `python -m cv_strategies.<module>`)

| Module | Phase | Output |
|---|---|---|
| `config.py` | — | paths + read-only reuse of `benchmark.config` |
| `imbalance.py` | 1 | `results/imbalance.json` (exact baseline) |
| `timekey.py` | 2 | `artifacts/win_time_{train,test}.npy` |
| `cv.py` | 3 | CV splitters (`python -m cv_strategies.cv` runs self-tests) |
| `tuning.py --diagnostic` | 4 | `results/cv_comparison.csv` |
| `tuning.py --retune` | 5 | `results/tuning_results.csv`, tuned models |
| `evaluation_ext.py` | 6 | `results/operating_points.csv` |
| `ensemble_calib.py` | 7 | `results/ensemble_calibration.csv`, `calibration.json` |
| `report.py` | 8 | `results/report.html` |
| `run_all.py` | — | runs phases 1–8 in order |

## Reproduce

```
python -m cv_strategies.run_all           # everything, skipping finished steps
python -m cv_strategies.run_all --force    # rebuild from scratch
```

Requires the Session-1 artifacts in `../artifacts/`. Fixed seed 42. Designed for the
16 GB machine (float32, memmap, `n_jobs=1` in the grids).
