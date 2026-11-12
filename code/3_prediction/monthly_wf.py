import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as PR, roc_auc_score as AUC

ROOT = Path(__file__).resolve().parents[2]
from robust import models as md
from robust import stats_tests as st

OUT = ROOT / "results" / "full"
md.LADDER["M2L"] = md.LADDER["M2"] + md.LLM
MODELS = ("M1", "M2", "M4", "M2L")


def mdd(r): w = np.cumprod(1 + np.asarray(r)); return float((1 - w / np.maximum.accumulate(w)).max())
def sharpe(r): r = np.asarray(r); return r.mean() / r.std(ddof=1) * np.sqrt(12)


def main():
    p = pd.read_parquet(ROOT / "data" / "interim" / "panel_final.parquet")
    num = p.select_dtypes("number").columns
    p[num] = p[num].replace([np.inf, -np.inf], np.nan)
    a = p[p["spread"].notna() & p["year"].between(2014, 2024)].reset_index(drop=True)
    cols = {m: [c for c in md.LADDER[m] if c in a.columns] for m in MODELS}
    test_months = sorted(a.loc[a["year"].between(2016, 2024), "mi"].unique())
    out = []
    cache = OUT / "mwf_preds.parquet"
    if cache.exists():
        pred = pd.read_parquet(cache).merge(a.drop(columns=["month", "year", "warn", "known", "y_fa12"]), on=["cik", "mi"])
        test_months = []
    for i, t in enumerate(test_months):
        tr = a[(a["mi"] <= t - 12) & (a["year"] >= 2014)]
        te = a[a["mi"] == t].copy()
        for m in MODELS:
            te[m] = md.LogitModel(cols[m], 0.5).fit(tr, "y_fa12").predict(te)
        out.append(te)
        if i % 12 == 0:
            print("month", t, "train rows", len(tr), "events", int(tr["y_fa12"].sum()), flush=True)
    if out:
        pred = pd.concat(out).reset_index(drop=True)
        pred[["cik", "mi", "month", "year", "warn", "known", "y_fa12", *MODELS]].to_parquet(cache, index=False)
    nk = ~((pred["y_fa12"] == 1) & (pred["known"] == 1))

    rows = []
    for which, mask in [("경고 없는 구간 (주 분석)", nk & (pred["warn"] == 0)), ("S&P 경고 기업", nk & (pred["warn"] == 1)), ("전체", nk)]:
        s = pred[mask]
        for c in MODELS:
            q = s.groupby("month")[c].rank(pct=True) >= 0.9
            rows.append({"sample": which, "model": c, "n": len(s), "events": int(s["y_fa12"].sum()), "PR": PR(s["y_fa12"], s[c]),
                         "AUC": AUC(s["y_fa12"], s[c]), "top10_capture": s.loc[q, "y_fa12"].sum() / s["y_fa12"].sum()})
    a1 = pd.DataFrame(rows); a1.to_csv(OUT / "mwf_A1.csv", index=False)

    s = pred[nk]; rows = []
    for c in ("M2", "M2L"):
        pct = s.groupby("month")[c].rank(pct=True)
        for top in (0.05, 0.10, 0.20):
            f = pct >= 1 - top
            rows.append({"model": c, "top": top, "precision": s.loc[f, "y_fa12"].mean()})
    rows.append({"model": "base", "top": None, "precision": s["y_fa12"].mean()})
    pd.DataFrame(rows).to_csv(OUT / "mwf_A2.csv", index=False)

    s = pred[nk & (pred["warn"] == 0)]; yr = []
    for y, g in s.groupby("year"):
        yr.append({"year": y, "PR_M2": PR(g["y_fa12"], g["M2"]), "PR_M2L": PR(g["y_fa12"], g["M2L"])})
    yr = pd.DataFrame(yr); yr["better"] = yr["PR_M2L"] > yr["PR_M2"]; yr.to_csv(OUT / "mwf_yearly.csv", index=False)

    pred2 = pred.copy()
    pred2["r"] = pred2["credit_ret"].clip(-0.5, 0.5); pred2["hc"] = pred2["half_cost"].fillna(pred2["half_cost"].median())
    pred2["pct"] = pred2.groupby("mi")["M2L"].rank(pct=True)
    prev, rec = {}, []
    for t in sorted(pred2["mi"].unique()):
        d = pred2[pred2["mi"] == t].set_index("cik"); r = {"mi": t + 1}
        for nm, keep in {"IDX": np.ones(len(d), bool), "D_M2L": (d["pct"] < 0.9).values}.items():
            w = d["amt"].where(keep, 0.0); w = w / w.sum(); wp = prev.get(nm, pd.Series(dtype=float)).reindex(w.index).fillna(0)
            r[nm] = float((w * d["r"].fillna(0)).sum()) - float(((w - wp).abs() * d["hc"]).sum()); prev[nm] = w
        rec.append(r)
    m = pd.DataFrame(rec)
    life = pd.read_csv(OUT / "life_monthly.csv")[["mi", "B_SEL"]]
    m = m.merge(life, on="mi", how="left"); m["AB"] = 0.9 * m["D_M2L"] + 0.1 * m["B_SEL"]
    dfz = pd.DataFrame([{"strategy": k, "ann_ret": np.prod(1 + m[k]) ** (12 / len(m)) - 1, "sharpe": sharpe(m[k]), "cvar95": st.cvar(m[k].values, 0.05),
                         "max_dd": mdd(m[k])} for k in ("IDX", "D_M2L", "AB")])
    dfz.to_csv(OUT / "mwf_defense.csv", index=False)

    pd.set_option("display.width", 200)
    print(a1.round(4).to_string(index=False)); print(pd.read_csv(OUT / "mwf_A2.csv").round(4).to_string(index=False))
    print(yr.round(4).to_string(index=False), "wins", int(yr["better"].sum())); print(dfz.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
