"""Session-2 follow-up work: honest cross-validation strategies, the exact
PR-AUC baseline, honest hyperparameter tuning, operating-point selection,
ensembling and calibration.

This package is fully separable from the session-1 `benchmark/` package: it does
NOT modify any session-1 file. It reuses only (a) the frozen windowed artifacts
in `../artifacts/*.npy` and (b) a read-only import of `benchmark.config` for the
constants that *describe* those artifacts (so the two can never drift).
"""
