"""Generate the five fresh Session-2 notebooks (nbformat), to be executed with
nbconvert. They LOAD the results produced by the pipeline and VISUALIZE them
(reusing the exact report figures for consistency) -- no heavy recompute -- so the
analysis can be traced step by step.

    python -m cv_strategies.build_notebooks          # write the .ipynb files
    jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
"""
from __future__ import annotations

import nbformat as nbf

from . import config as C

BOOT = (
    "import sys, pathlib\n"
    "ROOT = pathlib.Path.cwd().resolve().parents[1]  # cv_strategies/notebooks -> project root\n"
    "sys.path.insert(0, str(ROOT))\n"
    "import warnings; warnings.filterwarnings('ignore')\n"
    "import json, numpy as np, pandas as pd\n"
    "from IPython.display import HTML, display\n"
    "from cv_strategies import config as C, evaluation_ext as E, report as R\n"
    "pd.set_option('display.float_format', lambda v: f'{v:.4f}')\n"
    "print('session folder:', ROOT)"
)


def md(t): return nbf.v4.new_markdown_cell(t)
def code(t): return nbf.v4.new_code_cell(t)
def img(expr): return code(f"display(HTML(f'<img src=\"{{{expr}}}\">'))")


def nb(*cells):
    n = nbf.v4.new_notebook(); n.cells = list(cells)
    n.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}}
    return n


def build() -> None:
    C.ensure_dirs()
    D = C.NOTEBOOKS_DIR

    # 00 - imbalance & exact baseline ------------------------------------------
    nb00 = nb(
        md("# 00 · Dataset imbalance & the exact PR-AUC baseline\n"
           "The PR-AUC of a random classifier equals the **positive prevalence of the set it "
           "is evaluated on**. Our models are scored on the **test** set, so the baseline is the "
           "**test** prevalence — *not* the training prior. We verify this against the Session-1 "
           "Dummy row."),
        code(BOOT),
        code("imb = json.loads(C.IMBALANCE_JSON.read_text())\n"
             "print('PR-AUC baseline (test prevalence):', round(imb['pr_auc_baseline'],4))\n"
             "print('train prior               :', round(imb['train_prior'],4))\n"
             "print('scale_pos_weight (train)   :', round(imb['scale_pos_weight_train'],4))\n"
             "print('reconciliation             :', imb.get('reconciliation'))"),
        md("### Per-state positives (test)"),
        code("ps = imb['test']['per_state']\n"
             "pd.DataFrame([{'state':k, **v} for k,v in ps.items()])"),
        md("### Session-1 models against the correct baseline"),
        img("R.fig_lift(pd.DataFrame(imb['session1_lift'])[['model','pr_auc']][:8], imb['pr_auc_baseline'])"),
        md("**Takeaway.** Best model ≈ **1.97×** the correct baseline (0.0403), not the ~2.5× the "
           "train prior (0.0320) would suggest. TIMEOUT supplies ~70% of positives; OOM is tiny but "
           "almost entirely positive."),
    )

    # 01 - CV diagnosis --------------------------------------------------------
    nb01 = nb(
        md("# 01 · Cross-validation diagnosis\n"
           "`StratifiedKFold(shuffle)` leaks because the 39/40-overlapping windows of one job are "
           "near-duplicates scattered across folds. We compare what each CV **believes** (CV-AP) with "
           "the **truth** (temporal test PR-AUC)."),
        code(BOOT),
        code("cv = pd.read_csv(C.CV_COMPARISON_CSV)\n"
             "cv['inflation_x'] = cv['cv_ap_mean']/cv['test_pr_auc']\n"
             "cv[['strategy','n_splits','cv_ap_mean','test_pr_auc','inflation_x']]"),
        img("R.fig_cv_diagnosis(cv, E.baseline())"),
        md("### The splitters keep whole jobs together — a cheap structural check\n"
           "No model fitting: just verify fold membership on the train split."),
        code("from cv_strategies import cv as CV\n"
             "a = C.load_frozen(mmap=True); y=np.asarray(a['y_train']); g=np.asarray(a['job_ids_train'])\n"
             "tt,_ = C.load_win_time()\n"
             "for name in ['stratified','group','purged','tscv','logo_weekly']:\n"
             "    sp = CV.make_splits(name, y=y, groups=g, times=tt)\n"
             "    shared = max(len(set(g[tr]) & set(g[te])) for tr,te in sp)\n"
             "    print(f'{name:<14} folds={len(sp):<2} jobs shared between any train/val fold: {shared}')"),
        md("**Takeaway.** Leaky `stratified` overstates AP ~5.7×; **grouping** halves it but is still "
           "~2.7×; only **temporal** CV (`tscv` ~1.1×, `purged` ~1.5×) tracks the truth. Grouping is "
           "necessary but not sufficient."),
    )

    # 02 - tuning --------------------------------------------------------------
    nb02 = nb(
        md("# 02 · Honest re-tuning\n"
           "**Key finding:** a leak-free *same-period* monitor overfits the future; only a **temporal** "
           "monitor picks the number of rounds that generalizes. We select structure + rounds by early "
           "stopping on a temporal holdout, then validate under the purged CV and score on the test set."),
        code(BOOT),
        img("R.fig_temporal_select()"),
        md("### Tuned vs. Session-1 base"),
        code("t = pd.read_csv(C.TUNING_RESULTS_CSV)\n"
             "t = t[t['status']=='ok']\n"
             "t['delta_vs_base'] = t['test_pr_auc']-t['base_pr_auc']\n"
             "t[['model','test_pr_auc','base_pr_auc','delta_vs_base','purged_cv_ap','final_n_rounds','best_params']]"),
        img("R.fig_tuned_vs_base(pd.read_csv(C.TUNING_RESULTS_CSV), E.baseline())"),
        md("**Takeaway.** Honest tuning **reverses the −0.018 Pass-2 regression**, landing at the base "
           "level (~0.078–0.09). It does not dramatically exceed the prior ~0.088 — as expected, that "
           "needs pipeline/feature work, deliberately deferred."),
    )

    # 03 - thresholds, ensemble, calibration -----------------------------------
    nb03 = nb(
        md("# 03 · Operating point, ensemble & calibration\n"
           "PR-AUC is threshold-free; deployment needs a threshold. For proactive draining a missed "
           "failure costs more than a false drain → recall-favoring **F2**."),
        code(BOOT),
        md("### Operating points (best tuned model)"),
        code("op = pd.read_csv(C.OPERATING_POINTS_CSV); op"),
        md("### Ensemble of the tuned top-3"),
        code("pd.read_csv(C.ENSEMBLE_CALIB_CSV)"),
        md("### Calibration — Brier improves, PR-AUC is unchanged (monotonic map)"),
        code("cal = json.loads((C.RESULTS_DIR/'calibration.json').read_text())\n"
             "for k in ('raw','isotonic','sigmoid'): print(f\"{k:<9} Brier={cal[k]['brier']:.5f}  PR-AUC={cal[k]['pr_auc']:.4f}\")\n"
             "display(HTML(f'<img src=\"{R.fig_reliability(cal)}\">'))"),
        md("**Takeaway.** The F2 threshold turns the ranking score into an actionable drain policy; "
           "calibration makes the probability behind that threshold trustworthy."),
    )

    # 04 - metric interpretation -----------------------------------------------
    nb04 = nb(
        md("# 04 · How to read PR-AUC & ROC-AUC at this imbalance\n"
           "A score is only meaningful against the right baseline. PR-AUC baseline = test prevalence "
           "(0.0403); ROC-AUC baseline = 0.5. We express every model as **×-lift** and a coarse verdict."),
        code(BOOT),
        code("base = E.baseline(); print('baseline', round(base,4))\n"
             "rows=[]\n"
             "for r in json.loads(C.IMBALANCE_JSON.read_text())['session1_lift']:\n"
             "    rows.append({'model':r['model'],'pr_auc':r['pr_auc'],'x_lift':r['pr_auc']/base,\n"
             "                 'verdict':E.pr_verdict(r['pr_auc']),'roc_auc':r['roc_auc'],\n"
             "                 'roc_verdict':E.roc_verdict(r['roc_auc'])})\n"
             "pd.DataFrame(rows)"),
        md("### Best tuned model — PR and ROC curves with baseline references"),
        code("t = pd.read_csv(C.TUNING_RESULTS_CSV); t=t[t['status']=='ok'].sort_values('test_pr_auc',ascending=False)\n"
             "best=t.iloc[0]['model']; proba=np.load(C.GRID_DIR/f'{best}_tuned_test_proba.npy')\n"
             "y=np.load(C.frozen_artifact_path('y_test'))\n"
             "display(HTML(f'<img src=\"{R.fig_pr_roc(y, proba, base, best)}\">'))"),
        md("### Why our numbers look lower than Skrzeczek's (0.98 ROC-AUC) — and why that is expected\n"
           "Skrzeczek classifies **whole jobs** (aggregated), on ~920 jobs at ~2.6:1 imbalance, with the "
           "**successful** job as the positive class, reporting accuracy/F1/ROC-AUC. Ours is **per-window, "
           "extreme-imbalance, early** prediction of the **rare failure** — harder by construction. PR-AUC "
           "against the 0.0403 baseline is the honest yardstick. See `docs/session_2_methodology.md` §9."),
    )

    for name, n in [("00_imbalance", nb00), ("01_cv_diagnosis", nb01), ("02_tuning", nb02),
                    ("03_thresholds_ensemble", nb03), ("04_metric_interpretation", nb04)]:
        p = D / f"{name}.ipynb"
        nbf.write(n, p)
        print("wrote", p)


if __name__ == "__main__":
    build()
