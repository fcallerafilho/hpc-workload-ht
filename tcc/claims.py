"""Check a computed number against the number printed in the paper.

    claim("GPU jobs in the cohort", len(cohort), "66,017")

The paper value is written exactly as it appears in main.tex.  The computed value is rounded to
the same number of decimals before comparing; a value written with "%" is compared as a
percentage.  Every notebook ends with save_claims(), which writes results/claims/<name>.csv, so
the whole list of checks can be read in one place.
"""
import pandas as pd

from .config import RESULTS

_rows = []


def _parse(paper: str):
    s = paper.replace(",", "").replace("$", "").replace("−", "-").strip()
    pct = s.endswith("%")
    s = s.rstrip("%").strip()
    decimals = len(s.split(".")[1]) if "." in s else 0
    return float(s), decimals, pct


def claim(what: str, value, paper: str, note: str = ""):
    expected, decimals, pct = _parse(paper)
    v = float(value) * (100 if pct else 1)
    got = round(v, decimals)
    ok = abs(got - expected) < 1e-9 or (got == 0 and expected == 0)
    shown = f"{v:.{decimals}f}" + ("%" if pct else "")
    print(f"{'OK  ' if ok else 'DIFF'}  {what}: {shown}   (paper: {paper})" + (f"   [{note}]" if note else ""))
    _rows.append({"claim": what, "computed": shown, "paper": paper, "ok": ok, "note": note})
    return ok


def save_claims(notebook: str):
    df = pd.DataFrame(_rows)
    df.to_csv(RESULTS / "claims" / f"{notebook}.csv", index=False)
    n_bad = int((~df.ok).sum()) if len(df) else 0
    print(f"{len(df)} claims checked, {n_bad} differ from the paper -> results/claims/{notebook}.csv")
    _rows.clear()
    return df
