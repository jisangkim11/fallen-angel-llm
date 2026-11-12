import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as PR

from robust import models as md
from robust.models import walk_forward

ROOT = Path(__file__).resolve().parents[2]
rng = np.random.default_rng(1)
PIL = ["llm_bus", "llm_fin", "llm_liq", "llm_pol"]


def boot_p(df, y, a, b, n=1000):
    firms = df["cik"].unique(); g = df.groupby("cik").indices
    d0 = PR(df[y], df[b]) - PR(df[y], df[a]); v = []
    for _ in range(n):
        d = df.iloc[np.concatenate([g[c] for c in rng.choice(firms, len(firms))])]
        if d[y].nunique() == 2: v.append(PR(d[y], d[b]) - PR(d[y], d[a]))
    v = np.array(v)
    return d0, np.percentile(v, 2.5), np.percentile(v, 97.5), float((v <= 0).mean())


def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * ps[i]); adj[i] = min(1, run)
    return adj


def main():
    p = pd.read_parquet(ROOT / "data" / "interim" / "panel_final.parquet")
    num = p.select_dtypes("number").columns
    p[num] = p[num].replace([np.inf, -np.inf], np.nan)
    p = p.sort_values(["cik", "mi"])
    raw = {"llm_bus": "raw_business", "llm_fin": "raw_financial", "llm_liq": "raw_liquidity", "llm_pol": "raw_policy"}
    for c, r in raw.items():
        lag = p.groupby("cik")[r].shift(12)
        p["d_" + c] = (p[r] - lag).groupby(p["month"]).rank(pct=True)
    p["unc"] = p["llm_unc"].groupby(p["month"]).rank(pct=True)
    D = ["d_" + c for c in PIL]
    md.LADDER["M5d"] = md.LADDER["M4"] + D
    md.LADDER["M5u"] = md.LADDER["M4"] + ["unc"]
    md.LADDER["M5du"] = md.LADDER["M4"] + D + ["unc"]
    md.LADDER["M5dns"] = md.LADDER["M4ns"] + D
    print("non-missing d_llm_bus:", round(p["d_llm_bus"].notna().mean(), 3))

    a = p[p["spread"].notna() & p["year"].between(2014, 2024)].reset_index(drop=True)
    pl, _ = walk_forward(a, "y_fa12", models=("M3", "M4", "M5d", "M5u", "M5du"))
    pb, _ = walk_forward(a, "y_fa12", models=("M4",), boost=True)
    pl["M4cb"] = pb["M4"].values
    pl["ENS"] = (pl.groupby("year")["M4"].rank(pct=True) + pl.groupby("year")["M4cb"].rank(pct=True)) / 2
    b = p[p["llm_bus"].notna() & p["year"].between(2014, 2024)].reset_index(drop=True)
    pn, _ = walk_forward(b, "y_fa12", models=("M3ns", "M4ns", "M5dns"))

    def samp(df, which):
        nk = ~((df["y_fa12"] == 1) & (df["known"] == 1))
        return df[nk & ((df["warn"] == 0) if which == "no_warn" else (df["warn"] == 1) if which == "warned" else True)].reset_index(drop=True)

    rows = []
    for which in ("no_warn", "warned", "all"):
        s = samp(pl, which)
        for m in ("M4", "M5d", "M5u", "M5du", "M4cb", "ENS"):
            d0, lo, hi, pv = boot_p(s, "y_fa12", "M3", m)
            rows.append({"sample": "spread/" + which, "base": "M3", "model": m, "PR_base": PR(s["y_fa12"], s["M3"]), "PR_model": PR(s["y_fa12"], s[m]),
                         "diff": d0, "ci_lo": lo, "ci_hi": hi, "p_one": pv, "events": int(s["y_fa12"].sum())})
        s = samp(pn, which)
        for m in ("M4ns", "M5dns"):
            d0, lo, hi, pv = boot_p(s, "y_fa12", "M3ns", m)
            rows.append({"sample": "nospread/" + which, "base": "M3ns", "model": m, "PR_base": PR(s["y_fa12"], s["M3ns"]), "PR_model": PR(s["y_fa12"], s[m]),
                         "diff": d0, "ci_lo": lo, "ci_hi": hi, "p_one": pv, "events": int(s["y_fa12"].sum())})
    r = pd.DataFrame(rows)
    r["p_holm"] = holm(r["p_one"].values)
    r.to_csv(ROOT / "results" / "full" / "ext_models.csv", index=False)
    pd.set_option("display.width", 220)
    print(r.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
