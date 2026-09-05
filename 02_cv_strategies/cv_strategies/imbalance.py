"""Phase 1 - Dataset imbalance and the EXACT PR-AUC baseline.

The PR-AUC of a random/constant classifier equals the positive prevalence of the
set it is *evaluated* on. Our models are evaluated on the TEST set, so the honest
PR-AUC baseline is the **test** positive fraction -- not the training prior.

The session-1 Dummy row makes the distinction concrete:
  * Dummy `mean_proba` == the TRAIN positive fraction  (strategy='prior' emits the
    training prior for every sample), and
  * Dummy `pr_auc`     == the TEST  positive fraction   (AP of a constant predictor
    == prevalence of the evaluated set).

This module computes both fractions directly from the frozen `y_*`/`states_*`
artifacts, asserts they match the Dummy row, breaks positives down per failure
state, and writes `results/imbalance.json` -- the single ~0.0403 number every
downstream lift/verdict uses.

    python -m cv_strategies.imbalance
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C


def _split_stats(y: np.ndarray, states: np.ndarray) -> dict:
    """Prevalence + per-state positive breakdown for one split."""
    n = int(y.shape[0])
    pos = int((y == 1).sum())
    frac = pos / n

    per_state = {}
    for code in range(len(C.CODE_TO_STATE)):
        name = C.CODE_TO_STATE[code]
        disp = C.STATE_DISPLAY[name]
        mask = states == code
        w = int(mask.sum())
        p = int((y[mask] == 1).sum()) if w else 0
        per_state[disp] = {
            "windows": w,
            "positives": p,
            "pos_frac_within_state": (p / w) if w else 0.0,
            "share_of_all_positives": (p / pos) if pos else 0.0,
        }

    return {
        "n_windows": n,
        "n_positive": pos,
        "n_negative": n - pos,
        "positive_fraction": frac,
        "scale_pos_weight": (n - pos) / pos if pos else float("inf"),
        "per_state": per_state,
    }


def _dummy_reference() -> dict:
    """Pull the session-1 Dummy row (train prior + test-prevalence check)."""
    df = pd.read_csv(C.BENCH_RESULTS_CSV)
    d = df[(df["model"] == "DummyClassifier") & (df["pass"] == 1)]
    if d.empty:
        return {}
    r = d.drop_duplicates("model", keep="last").iloc[0]
    return {
        "dummy_pr_auc": float(r["pr_auc"]),               # == test prevalence
        "dummy_mean_proba": float(r["mean_proba_COMPLETED"]),  # == train prior
        "dummy_roc_auc": float(r["roc_auc"]),             # == 0.5
    }


def _session1_lift(baseline: float) -> list[dict]:
    """Express every session-1 Pass-1 model as absolute PR-AUC and x-lift over the
    correct (test-prevalence) baseline, so 'is this good?' has an answer."""
    df = pd.read_csv(C.BENCH_RESULTS_CSV)
    df = df[(df["pass"] == 1) & (df["status"] == "ok")].drop_duplicates("model", keep="last")
    df = df.sort_values("pr_auc", ascending=False)
    return [
        {
            "model": r["model"],
            "pr_auc": float(r["pr_auc"]),
            "roc_auc": float(r["roc_auc"]),
            "lift_over_baseline": float(r["pr_auc"]) / baseline,
        }
        for _, r in df.iterrows()
    ]


def compute() -> dict:
    C.ensure_dirs()
    a = C.load_frozen(mmap=True)
    train = _split_stats(np.asarray(a["y_train"]), np.asarray(a["states_train"]))
    test = _split_stats(np.asarray(a["y_test"]), np.asarray(a["states_test"]))
    dummy = _dummy_reference()

    baseline = test["positive_fraction"]        # THE PR-AUC baseline
    out = {
        "pr_auc_baseline": baseline,             # test prevalence ~0.0403
        "train_prior": train["positive_fraction"],  # ~0.0320
        "roc_auc_baseline": 0.5,
        "scale_pos_weight_train": train["scale_pos_weight"],
        "train": train,
        "test": test,
        "dummy_reference": dummy,
        "session1_lift": _session1_lift(baseline),
        "note": (
            "PR-AUC baseline = TEST positive fraction (prevalence of the evaluated "
            "set). The TRAIN prior (~0.032) is a different number; do not use it to "
            "judge test PR-AUC. ROC-AUC baseline = 0.5."
        ),
    }

    # --- reconciliation asserts (this is the whole point of the phase) -------- #
    if dummy:
        assert abs(baseline - dummy["dummy_pr_auc"]) < 1e-4, (
            f"test prevalence {baseline:.6f} != Dummy pr_auc {dummy['dummy_pr_auc']:.6f}")
        assert abs(train["positive_fraction"] - dummy["dummy_mean_proba"]) < 1e-4, (
            f"train prior {train['positive_fraction']:.6f} != Dummy mean_proba "
            f"{dummy['dummy_mean_proba']:.6f}")
        out["reconciliation"] = "OK: baseline==Dummy.pr_auc and train_prior==Dummy.mean_proba"

    with open(C.IMBALANCE_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return out


def _print(out: dict) -> None:
    b = out["pr_auc_baseline"]
    print("=" * 74)
    print("DATASET IMBALANCE & EXACT BASELINE")
    print("=" * 74)
    print(f"  TRAIN  windows={out['train']['n_windows']:>10,}  "
          f"pos={out['train']['n_positive']:>8,}  "
          f"prior(frac)={out['train_prior']:.4f}  "
          f"scale_pos_weight={out['scale_pos_weight_train']:.4f}")
    print(f"  TEST   windows={out['test']['n_windows']:>10,}  "
          f"pos={out['test']['n_positive']:>8,}  "
          f"prevalence={b:.4f}   <-- PR-AUC baseline")
    if "reconciliation" in out:
        print(f"  {out['reconciliation']}")
    print(f"  ROC-AUC baseline = {out['roc_auc_baseline']}")
    print("-" * 74)
    print("  Per-state positives (TEST):")
    for disp, s in out["test"]["per_state"].items():
        if s["windows"]:
            print(f"    {disp:<10} windows={s['windows']:>9,}  pos={s['positives']:>7,}  "
                  f"within-state={s['pos_frac_within_state']:.3f}  "
                  f"share_of_pos={s['share_of_all_positives']:.3f}")
    print("-" * 74)
    print(f"  Session-1 PR-AUC as x-lift over baseline ({b:.4f}):")
    for row in out["session1_lift"][:6]:
        print(f"    {row['model']:<24} PR-AUC={row['pr_auc']:.4f}  "
              f"({row['lift_over_baseline']:.2f}x)   ROC-AUC={row['roc_auc']:.3f}")
    print("=" * 74)
    print(f"  written: {C.IMBALANCE_JSON}")


def main() -> None:
    _print(compute())


if __name__ == "__main__":
    main()
