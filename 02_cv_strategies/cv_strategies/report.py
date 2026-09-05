"""Phase 8 - regenerate the self-contained HTML report.

Reads whatever result files exist (imbalance.json, cv_comparison.csv,
tuning_results.csv, operating_points.csv, ensemble_calibration.csv,
calibration.json) and renders a single theme-aware `results/report.html` with
embedded (base64) matplotlib figures -- no external assets, opens straight in a
browser. Charts use the validated data-viz palette; figures render on a fixed light
card so they read in both page themes.

    python -m cv_strategies.report
"""
from __future__ import annotations

import base64
import io
import json

import numpy as np
import pandas as pd

from . import config as C
from . import evaluation_ext as E

# ---- validated palette (dataviz skill: references/palette.md) -------------- #
BLUE = "#2a78d6"      # slot 1  -- CV-AP "believed"
ORANGE = "#eb6834"    # slot 8  -- test "truth"
AQUA = "#1baf7a"      # slot 2
VIOLET = "#4a3aa7"    # slot 5
RED = "#d03b3b"       # status critical -- regression / inflation
GOOD = "#0ca30c"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

# Measured once (session_2_methodology.md 5): LightGBM test PR-AUC vs #rounds, and
# where each leak-free monitor's AP argmaxes. Illustrative constant for the figure.
TEMPORAL_SELECT = {
    "rounds": [20, 40, 60, 100, 150, 200, 300],
    "test_ap": [0.0773, 0.0831, 0.0806, 0.0820, 0.0834, 0.0822, 0.0769],
    "temporal_argmax": 122,   # temporal holdout -> ~120 rounds (near the test peak)
    "random_argmax": 260,     # grouped-random holdout -> ~260 rounds (test is worst here)
}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.size": 11,
        "font.family": "sans-serif", "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _overview_svg() -> str:
    p = C.DOCS_DIR / "session_2_overview.svg"
    return p.read_text(encoding="utf-8") if p.exists() else ""


# Simple 3-fold illustration for the cross-validation explainer (light card).
FOLD_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 740 200" font-family="system-ui,sans-serif">
  <rect x="0" y="0" width="740" height="200" fill="#fcfcfb"/>
  <text x="80" y="24" font-size="12.5" fill="#52514e">3-fold cross-validation: rotate which block is held out to &#8220;validate&#8221;</text>
  __ROWS__
  <text x="80" y="176" font-size="11.5" fill="#52514e">Keep the setting that scores best on the held-out block, averaged over folds. Blue = train, dark = validate.</text>
</svg>
"""


def _fold_svg() -> str:
    rows = []
    ys = [40, 82, 124]
    valpos = [2, 1, 0]     # fold1 validates block3, fold2 block2, fold3 block1
    xs = [80, 281, 482]
    for i, (y, vp) in enumerate(zip(ys, valpos), 1):
        rows.append(f'<text x="10" y="{y+20}" font-size="12" fill="#52514e">Fold {i}</text>')
        for b, x in enumerate(xs):
            val = (b == vp)
            fill = "#2a78d6" if val else "#cde2fb"
            txt = "#ffffff" if val else "#0b0b0b"
            label = "validate" if val else "train"
            rows.append(f'<rect x="{x}" y="{y}" width="195" height="30" rx="6" fill="{fill}" '
                        f'stroke="#2a78d6" stroke-width="1"/>')
            rows.append(f'<text x="{x+97}" y="{y+20}" text-anchor="middle" font-size="12.5" '
                        f'fill="{txt}">{label}</text>')
    return FOLD_SVG.replace("__ROWS__", "\n  ".join(rows))


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_cv_diagnosis(cv: pd.DataFrame, baseline: float) -> str:
    plt = _mpl()
    cv = cv.sort_values("cv_ap_mean", ascending=True)
    y = np.arange(len(cv)); h = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.barh(y + h/2, cv["cv_ap_mean"], height=h, color=BLUE, label="CV-AP (believed)")
    ax.barh(y - h/2, cv["test_pr_auc"], height=h, color=ORANGE, label="test PR-AUC (truth)")
    for yi, v in zip(y + h/2, cv["cv_ap_mean"]):
        ax.text(v + .006, yi, f"{v:.3f}", va="center", fontsize=9, color=INK)
    for yi, v in zip(y - h/2, cv["test_pr_auc"]):
        ax.text(v + .006, yi, f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax.axvline(baseline, color=MUTED, ls="--", lw=1)
    ax.text(baseline, len(cv)-.4, f" baseline {baseline:.3f}", color=MUTED, fontsize=8, va="top")
    ax.set_yticks(y); ax.set_yticklabels(cv["strategy"])
    ax.set_xlabel("average precision"); ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title("Cross-validation believes vs. reality", color=INK, fontsize=12, loc="left")
    return _b64(fig)


def fig_lift(base_df: pd.DataFrame, baseline: float) -> str:
    plt = _mpl()
    d = base_df.sort_values("pr_auc")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.barh(y, d["pr_auc"], color=BLUE, height=0.62)
    for yi, v in zip(y, d["pr_auc"]):
        ax.text(v + .0012, yi, f"{v:.3f} ({v/baseline:.1f}x)", va="center", fontsize=8.5, color=INK)
    ax.axvline(baseline, color=RED, ls="--", lw=1.2)
    ax.text(baseline, -0.7, f"random baseline {baseline:.3f}", color=RED, fontsize=8.5, ha="left")
    ax.set_yticks(y); ax.set_yticklabels(d["model"], fontsize=9)
    ax.set_xlabel("test PR-AUC"); ax.set_xlim(0, max(d["pr_auc"])*1.25)
    ax.set_title("Session-1 models vs. the correct baseline", color=INK, fontsize=12, loc="left")
    return _b64(fig)


def fig_tuned_vs_base(t: pd.DataFrame, baseline: float) -> str:
    plt = _mpl()
    t = t[t["status"] == "ok"].copy()
    x = np.arange(len(t)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(x - w/2, t["base_pr_auc"], width=w, color=MUTED, label="Session-1 base")
    ax.bar(x + w/2, t["test_pr_auc"], width=w, color=BLUE, label="Session-2 tuned")
    for xi, v in zip(x - w/2, t["base_pr_auc"]):
        ax.text(xi, v + .001, f"{v:.3f}", ha="center", fontsize=8, color=INK)
    for xi, v in zip(x + w/2, t["test_pr_auc"]):
        ax.text(xi, v + .001, f"{v:.3f}", ha="center", fontsize=8, color=INK)
    ax.axhline(baseline, color=RED, ls="--", lw=1)
    ax.text(len(t)-.5, baseline, f" baseline {baseline:.3f}", color=RED, fontsize=8, va="bottom", ha="right")
    ax.set_xticks(x); ax.set_xticklabels([m.replace("Classifier","") for m in t["model"]], fontsize=9)
    ax.set_ylabel("test PR-AUC"); ax.legend(frameon=False, fontsize=9)
    ax.set_title("Honest tuning vs. base", color=INK, fontsize=12, loc="left")
    return _b64(fig)


def fig_pr_roc(y_te, proba, baseline: float, name: str) -> str:
    plt = _mpl()
    pr = E.pr_points(y_te, proba); rc = E.roc_points(y_te, proba)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.9))
    a1.plot(pr["recall"], pr["precision"], color=BLUE, lw=2)
    a1.axhline(baseline, color=MUTED, ls="--", lw=1)
    a1.text(0.02, baseline, f" random {baseline:.3f}", color=MUTED, fontsize=8, va="bottom")
    a1.set_xlabel("recall"); a1.set_ylabel("precision")
    a1.set_title(f"PR curve  (AP={pr['ap']:.3f})", color=INK, fontsize=11, loc="left")
    a2.plot(rc["fpr"], rc["tpr"], color=BLUE, lw=2)
    a2.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    a2.set_xlabel("false positive rate"); a2.set_ylabel("true positive rate")
    a2.set_title(f"ROC curve  (AUC={rc['auc']:.3f})", color=INK, fontsize=11, loc="left")
    fig.suptitle(f"Best tuned model: {name}", color=INK, fontsize=12, x=0.02, ha="left")
    return _b64(fig)


def fig_perstate(y_te, proba, states_te, thr: float, thr_name: str) -> str:
    plt = _mpl()
    rec = E.per_state_recall(y_te, proba, states_te, thr)
    labels = list(rec); vals = [rec[k] for k in labels]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(labels, vals, color=[AQUA, BLUE, VIOLET, ORANGE])
    for i, v in enumerate(vals):
        ax.text(i, v + .01, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_ylim(0, 1); ax.set_ylabel("recall")
    ax.set_title(f"Per-state recall @ {thr_name} (thr={thr:.3f})", color=INK, fontsize=11, loc="left")
    return _b64(fig)


def fig_temporal_select() -> str:
    plt = _mpl()
    d = TEMPORAL_SELECT
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.plot(d["rounds"], d["test_ap"], color=BLUE, lw=2, marker="o", ms=5, label="test PR-AUC")
    ax.axvline(d["temporal_argmax"], color=GOOD, ls="-", lw=1.6, label="temporal monitor -> 120")
    ax.axvline(d["random_argmax"], color=RED, ls="--", lw=1.6, label="same-period monitor -> 260")
    ax.set_xlabel("boosting rounds"); ax.set_ylabel("test PR-AUC")
    ax.legend(frameon=False, fontsize=9, loc="lower center")
    ax.set_title("Only a temporal monitor picks rounds that generalize", color=INK, fontsize=12, loc="left")
    return _b64(fig)


def fig_reliability(cal: dict) -> str:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1, label="perfect")
    for key, col in (("raw", RED), ("isotonic", BLUE), ("sigmoid", AQUA)):
        c = cal[key]
        ax.plot(c["prob_pred"], c["prob_true"], color=col, lw=1.8, marker="o", ms=4,
                label=f"{key} (Brier={c['brier']:.4f})")
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Reliability (calibration) curve", color=INK, fontsize=12, loc="left")
    return _b64(fig)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def _table(df: pd.DataFrame, floats: int = 4) -> str:
    fmt = lambda v: (f"{v:.{floats}f}" if isinstance(v, float) and not pd.isna(v)
                     else ("" if pd.isna(v) else str(v)))
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = "".join("<tr>" + "".join(f"<td>{fmt(v)}</td>" for v in r) + "</tr>"
                   for r in df.itertuples(index=False))
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


def _img(src: str, cap: str = "") -> str:
    c = f'<figcaption>{cap}</figcaption>' if cap else ""
    return f'<figure class="card"><img src="{src}" alt="{cap}"/>{c}</figure>'


CSS = """
:root{--bg:#f9f9f7;--fg:#0b0b0b;--mut:#52514e;--line:#e1e0d9;--card:#fcfcfb;--accent:#2a78d6;--bad:#d03b3b;--good:#006300}
@media (prefers-color-scheme:dark){:root{--bg:#0d0d0d;--fg:#f4f4f2;--mut:#c3c2b7;--line:#2c2c2a;--card:#161615;--accent:#3987e5;--bad:#e66767;--good:#0ca30c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:40px 22px 80px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:38px 0 10px;padding-top:14px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:22px 0 8px}p,li{color:var(--fg)}.sub{color:var(--mut);margin:0 0 8px}
.kpis{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0}
.kpi{flex:1 1 150px;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.kpi .n{font-size:26px;font-weight:650;font-variant-numeric:tabular-nums}.kpi .l{color:var(--mut);font-size:12.5px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin:14px 0}
figure.card{background:#fcfcfb}figure.card img{width:100%;height:auto;display:block;border-radius:6px}
.svgwrap{overflow-x:auto}.svgwrap svg{width:100%;height:auto;min-width:560px;display:block}
figcaption{color:#52514e;font-size:12.5px;margin-top:8px}
ul{margin:10px 0;padding-left:22px}li{margin:4px 0}
.tw{overflow-x:auto;margin:12px 0}table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums;white-space:nowrap}
th:first-child,td:first-child{text-align:left}thead th{color:var(--mut);font-weight:600;border-bottom:2px solid var(--line)}
.bad{color:var(--bad);font-weight:600}.good{color:var(--good);font-weight:600}
.callout{background:var(--card);border-left:3px solid var(--accent);border-radius:8px;padding:12px 16px;margin:14px 0}
.callout.warn{border-left-color:var(--bad)}
code{background:var(--card);border:1px solid var(--line);border-radius:5px;padding:1px 5px;font-size:13px}
"""


def build() -> None:
    C.ensure_dirs()
    baseline = E.baseline()
    imb = json.loads(C.IMBALANCE_JSON.read_text()) if C.IMBALANCE_JSON.exists() else {}
    figs, S = [], []

    S.append(f"""<div class="wrap"><h1>Honest cross-validation, exact baseline &amp; model improvements</h1>
    <p class="sub">Session-2 report &middot; <code>cv_strategies/</code> &middot; task: proactive HPC GPU-node failure prediction (per-window, extreme imbalance)</p>""")

    # KPI row
    best_base = imb.get("session1_lift", [{}])[0].get("pr_auc", float("nan")) if imb else float("nan")
    tuned_best = float("nan")
    if C.TUNING_RESULTS_CSV.exists():
        td = pd.read_csv(C.TUNING_RESULTS_CSV)
        td = td[td["status"] == "ok"]
        if len(td):
            tuned_best = float(td["test_pr_auc"].max())
    S.append('<div class="kpis">')
    S.append(f'<div class="kpi"><div class="n">{baseline:.4f}</div><div class="l">PR-AUC baseline (test prevalence)</div></div>')
    S.append(f'<div class="kpi"><div class="n">{best_base:.4f}</div><div class="l">Session-1 best (&times;{best_base/baseline:.2f})</div></div>')
    if tuned_best == tuned_best:  # not NaN
        S.append(f'<div class="kpi"><div class="n">{tuned_best:.4f}</div><div class="l">Session-2 tuned best (&times;{tuned_best/baseline:.2f})</div></div>')
    S.append(f'<div class="kpi"><div class="n">30.3</div><div class="l">scale_pos_weight (train)</div></div>')
    S.append('</div>')

    # terminology
    S.append("""<div class="callout"><b>Words used here.</b> <b>Session&nbsp;1</b> = the original
    benchmark, in two stages: <b>Pass&nbsp;1</b> trained 15 models once (default settings) to rank
    them, then <b>Pass&nbsp;2</b> = the <b>grid search</b> that hunts for each top model's best
    hyperparameters. <b>Session&nbsp;2</b> (this report) added no models and did not change the
    data &mdash; it fixed how we <i>judge</i> the scores and re-did the <b>Pass&nbsp;2</b> search
    with an honest cross-validation.</div>""")

    # at-a-glance diagram
    ov = _overview_svg()
    if ov:
        S.append("<h2>At a glance</h2>")
        S.append(f'<figure class="card"><div class="svgwrap">{ov}</div>'
                 '<figcaption>What Session 2 did, end to end.</figcaption></figure>')

    # cross-validation explainer
    S.append("<h2>What is cross-validation (and what are these &ldquo;time-aware splitters&rdquo;)?</h2>")
    S.append("""<p>To choose a model's settings we must estimate <b>how well it will do on data it
    hasn't seen</b> &mdash; without touching the final test set. <b>Cross-validation (CV)</b> does
    this by splitting the <b>training</b> data into a few <b>folds</b>: train on some, check on the
    held-out fold, rotate, average. We keep the setting that scores best.</p>""")
    S.append(f'<figure class="card"><div class="svgwrap">{_fold_svg()}</div>'
             '<figcaption>One round of 3-fold cross-validation.</figcaption></figure>')
    S.append("""<p><b>When does this happen in our project?</b> <b>Only inside Pass&nbsp;2</b>
    (the grid search). Pass&nbsp;1 trains each model once and tests once &mdash; no CV. In
    Pass&nbsp;2, the grid search splits the <b>training</b> set into folds to decide which
    hyperparameters are best, then refits the winner and scores it on the test set. The final
    <b>test set is never part of the CV</b> (it's a later, separate set of jobs), so it stays
    honest &mdash; which is exactly how we can <i>see</i> the leak: CV claimed 0.46, the test
    delivered 0.08.</p>""")
    S.append("""<p><b>The catch for us.</b> Our rows are <b>overlapping 10-minute windows</b> of the
    same job (each shares 39 of its 40 snapshots with the next). A <b>random</b> split drops
    near-identical windows of one job on <i>both</i> sides &mdash; the model effectively sees the
    answer, so the CV score is fake-high. Pass&nbsp;2 trusted that inflated score and picked
    over-complex models &rarr; the regression.</p>
    <p><b>The fix is better ways to split</b> &mdash; note these are <b>splitting rules, not
    models</b> (the models stay LightGBM / CatBoost / HGB):</p>
    <ul>
      <li><b>GroupKFold</b> &mdash; put every window of a given <b>job</b> entirely on one side, so
      the overlap can't leak.</li>
      <li><b>TimeSeriesSplit / PurgedKFold</b> &mdash; split by <b>time</b>: train on the past,
      validate on the <b>future</b> (with a small gap so nothing bleeds across) &mdash; exactly how
      the model is used in deployment.</li>
    </ul>
    <p>Section&nbsp;2 below measures how much each splitter &ldquo;cheats&rdquo;, and section&nbsp;3
    shows the twist: grouping alone isn't enough &mdash; the split must respect <b>time</b>.</p>""")

    # 1. baseline / imbalance
    S.append("<h2>1 &middot; The exact baseline</h2>")
    S.append("""<p><b>Why the baseline is the failure rate.</b> A random (constant) classifier has,
    at every recall, a precision equal to the share of positives among all rows &mdash; so its
    precision&ndash;recall curve is a flat line at <code>y = prevalence</code>, and its average
    precision (PR-AUC) <b>equals the positive prevalence</b>. Any real model must clear that bar to
    be worth anything.</p>""")
    S.append(f"""<div class="callout">Because PR-AUC is scored on the <b>test</b> windows, the baseline
    is the fraction of positive <b>test</b> windows = <b>{baseline:.4f}</b>. Concretely: the Dummy
    <i>predicts</i> the train prior 0.0320 (its <code>mean_proba</code>), but the score it
    <i>achieves</i> &mdash; its <code>pr_auc</code> = {baseline:.4f} &mdash; is the test prevalence.
    <b>That</b> is the baseline, so the best model is &times;{best_base/baseline:.2f} it, not the
    &times;{best_base/0.032:.1f} the train prior would suggest. ROC-AUC baseline = 0.5.</div>""")
    S.append("""<p><b>What counts as a &ldquo;positive&rdquo;?</b> Three things are easy to confuse
    &mdash; only the last is the target:</p>
    <ul>
      <li><b>Job state</b> (COMPLETED / TIMEOUT / FAILED / OOM / NODE_FAIL) &mdash; used only for the
      per-state breakdown.</li>
      <li><b>Snapshot label</b> (0/1 per 15-s snapshot: is it in the 2&nbsp;h before a failure job's
      end?) &mdash; an intermediate step.</li>
      <li><b>Window label</b> (0/1 per 40-snapshot window, by <b>majority vote</b>: &ge;20 of 40
      snapshots positive) &mdash; <b>this is what the model predicts and is scored on.</b></li>
    </ul>
    <p>So a positive = <b>a window in the final 2&nbsp;h of a job that ultimately failed</b>, <i>not</i>
    &ldquo;any window of a failed job&rdquo; and <i>not</i> a snapshot. That is why a long TIMEOUT job
    is only ~10% positive windows (just its tail), while short OOM jobs are ~99% positive. The
    baseline counts positive <b>windows</b>, matching the unit the model predicts.</p>""")
    if imb:
        ps = imb["test"]["per_state"]
        dfps = pd.DataFrame([{"state": k, "windows": v["windows"], "positives": v["positives"],
                              "within_state": v["pos_frac_within_state"],
                              "share_of_positives": v["share_of_all_positives"]}
                             for k, v in ps.items()])
        S.append("<h3>Per-state positives (test)</h3>" + _table(dfps, 3))
        lift = pd.DataFrame(imb["session1_lift"])[:8]
        lift["x_lift"] = lift["pr_auc"] / baseline
        figs.append(fig_lift(lift.rename(columns={})[["model", "pr_auc"]], baseline))
        S.append(_img(figs[-1], "Session-1 test PR-AUC against the correct random baseline (with x-lift)."))

    # 2. CV diagnosis
    if C.CV_COMPARISON_CSV.exists():
        cv = pd.read_csv(C.CV_COMPARISON_CSV)
        cv["inflation_x"] = cv["cv_ap_mean"] / cv["test_pr_auc"]
        S.append("<h2>2 &middot; Cross-validation diagnosis</h2>")
        S.append(f"""<div class="callout warn">The spec's <code>StratifiedKFold(shuffle)</code> believes
        AP=<b class="bad">{cv[cv.strategy=='stratified']['cv_ap_mean'].iloc[0]:.3f}</b> while the model truly
        scores <b>{cv[cv.strategy=='stratified']['test_pr_auc'].iloc[0]:.3f}</b> on the temporal test set &mdash;
        a <b class="bad">{cv[cv.strategy=='stratified']['inflation_x'].iloc[0]:.1f}&times;</b> overstatement.
        That inflated number is what drove Pass-2 to pick over-complex models and regress. Grouping halves the
        inflation; only <b>temporal</b> CV (tscv, purged) is honest.</div>""")
        figs.append(fig_cv_diagnosis(cv, baseline))
        S.append(_img(figs[-1], "Believed CV-AP vs. true test PR-AUC by strategy. Closer to the orange bar = more honest."))
        show = cv[["strategy", "n_splits", "cv_ap_mean", "test_pr_auc", "inflation_x"]]
        S.append(_table(show, 3))

    # 3. temporal selection finding
    S.append("<h2>3 &middot; Why temporal validation is required</h2>")
    S.append("""<div class="callout">Selecting boosting rounds, a leak-free <b>same-period</b> monitor points to
    ~260 rounds &mdash; exactly where the <b>future/test</b> score is worst. A <b>temporal</b> monitor points to
    ~120 rounds &mdash; the test peak. Grouping removes same-job leakage but not temporal over-optimism.</div>""")
    figs.append(fig_temporal_select())
    S.append(_img(figs[-1], "LightGBM test PR-AUC vs. rounds; the temporal monitor (green) lands on the peak, the same-period monitor (red) on the worst point."))

    # 4. tuned results
    if C.TUNING_RESULTS_CSV.exists() and len(td):
        S.append("<h2>4 &middot; Honest tuned results</h2>")
        figs.append(fig_tuned_vs_base(td, baseline))
        S.append(_img(figs[-1], "Session-2 temporal-holdout tuning vs. Session-1 base."))
        cols = ["model", "test_pr_auc", "base_pr_auc", "purged_cv_ap", "final_n_rounds", "test_roc_auc"]
        S.append(_table(td[cols], 4))

        # best model curves + per-state
        best = td.sort_values("test_pr_auc", ascending=False).iloc[0]
        pth = C.GRID_DIR / f"{best['model']}_tuned_test_proba.npy"
        if pth.exists():
            y_te = np.load(C.frozen_artifact_path("y_test"))
            st = np.load(C.frozen_artifact_path("states_test"))
            proba = np.load(pth)
            figs.append(fig_pr_roc(y_te, proba, baseline, best["model"]))
            S.append(_img(figs[-1], f"PR and ROC curves for {best['model']} (tuned)."))
            if C.OPERATING_POINTS_CSV.exists():
                op = pd.read_csv(C.OPERATING_POINTS_CSV)
                f2 = op[op["operating_point"] == "best_F2"]
                if len(f2):
                    thr = float(f2["threshold"].iloc[0])
                    figs.append(fig_perstate(y_te, proba, st, thr, "best F2"))
                    S.append(_img(figs[-1], "Per-state recall at the F2 operating point."))

    # 5. operating points
    if C.OPERATING_POINTS_CSV.exists():
        op = pd.read_csv(C.OPERATING_POINTS_CSV)
        S.append("<h2>5 &middot; Operating point</h2>")
        S.append("<p class='sub'>PR-AUC is threshold-free; deployment needs a threshold. For proactive draining a missed failure costs more than a false drain &rarr; recall-favoring F2.</p>")
        keep = ["operating_point", "threshold", "precision", "recall", "f2", "flagged",
                "recall_TIMEOUT", "recall_FAILED", "recall_OOM", "recall_NODE_FAIL"]
        S.append(_table(op[[c for c in keep if c in op.columns]], 3))

    # 6. ensemble + calibration
    if C.ENSEMBLE_CALIB_CSV.exists():
        en = pd.read_csv(C.ENSEMBLE_CALIB_CSV)
        S.append("<h2>6 &middot; Ensemble &amp; calibration</h2>")
        S.append(_table(en[["name", "kind", "test_pr_auc", "test_roc_auc", "recall_global"]], 4))
    calp = C.RESULTS_DIR / "calibration.json"
    if calp.exists():
        cal = json.loads(calp.read_text())
        figs.append(fig_reliability(cal))
        S.append(_img(figs[-1], "Calibration cuts Brier ~7x (0.28 to 0.04). Sigmoid preserves PR-AUC exactly (strictly monotonic); isotonic shifts it slightly via ties."))

    S.append("""<h2>7 &middot; Bottom line</h2>
    <div class="callout"><b>Fixing CV was the win:</b> it removes the Pass-2 regression and makes selection
    trustworthy; the corrected 0.0403 baseline makes the numbers honestly interpretable. <b>Temporal</b>
    validation &mdash; not merely grouping &mdash; is the decisive point. A large jump beyond ~0.088 will need
    pipeline/feature changes, deliberately left for a later session.</div>""")
    S.append("<p class='sub'>See <code>docs/session_2_methodology.md</code> (why) and <code>docs/session_2_findings.md</code> (what) for the full narrative.</p>")
    S.append("</div>")

    html = f"<!--report--><style>{CSS}</style>" + "".join(S)
    C.REPORT_HTML.write_text(html, encoding="utf-8")
    print("wrote", C.REPORT_HTML, f"({len(figs)} figures embedded)")


if __name__ == "__main__":
    build()
