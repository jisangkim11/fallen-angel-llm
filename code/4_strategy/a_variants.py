import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
from robust import models as md
from robust import stats_tests as st

OUT = ROOT / "results" / "full"
md.LADDER["M2L"] = md.LADDER["M2"] + md.LLM


def stats_of(r):
    r = np.asarray(r); w = np.cumprod(1 + r)
    return {"ann_ret": w[-1] ** (12 / len(r)) - 1, "sharpe": r.mean() / r.std(ddof=1) * np.sqrt(12),
            "cvar95": st.cvar(r, 0.05), "max_dd": float((1 - w / np.maximum.accumulate(w)).max())}


def main():
    p = pd.read_parquet(ROOT / "data" / "interim" / "panel_final.parquet")
    num = p.select_dtypes("number").columns
    p[num] = p[num].replace([np.inf, -np.inf], np.nan)
    a = p[p["spread"].notna() & p["year"].between(2014, 2024)].reset_index(drop=True)
    pred, _ = md.walk_forward(a, "y_fa12", models=("M2L",))
    pred = pred[pred["year"].between(2016, 2024)].copy()
    pred["pct"] = pred.groupby("mi")["M2L"].rank(pct=True)
    pred["sp_pct"] = pred.groupby("mi")["spread"].rank(pct=True)
    pred["r"] = pred["credit_ret"].clip(-0.5, 0.5)
    pred["hc"] = pred["half_cost"].fillna(pred["half_cost"].median())
    books = {
        "IDX": lambda d: np.ones(len(d)),
        "A_top10 (기존)": lambda d: np.where(d["pct"] >= 0.90, 0.0, 1.0),
        "A_top5": lambda d: np.where(d["pct"] >= 0.95, 0.0, 1.0),
        "A_top2": lambda d: np.where(d["pct"] >= 0.98, 0.0, 1.0),
        "A_top1": lambda d: np.where(d["pct"] >= 0.99, 0.0, 1.0),
        "A_half10": lambda d: np.where(d["pct"] >= 0.90, 0.5, 1.0),
        "A_unpriced10": lambda d: np.where((d["pct"] >= 0.90) & (d["sp_pct"] < 0.5), 0.0, 1.0),
    }
    prev, rec = {}, []
    for t in sorted(pred["mi"].unique()):
        d = pred[pred["mi"] == t].set_index("cik"); row = {"mi": t + 1}
        for nm, f in books.items():
            w = d["amt"] * f(d); w = w / w.sum(); wp = prev.get(nm, pd.Series(dtype=float)).reindex(w.index).fillna(0)
            row[nm] = float((w * d["r"].fillna(0)).sum()) - float(((w - wp).abs() * d["hc"]).sum())
            row[nm + "_excl"] = 1 - float((d["amt"] * f(d)).sum() / d["amt"].sum()); prev[nm] = w
        rec.append(row)
    m = pd.DataFrame(rec)
    b = pd.read_csv(OUT / "life_monthly.csv")[["mi", "month", "B_SEL"]]
    m = m.merge(b, on="mi")
    m.to_csv(OUT / "a_variants_monthly.csv", index=False)
    rows = []
    for nm in books:
        for tag, mask in [("전체", np.ones(len(m), bool)), ("2020 제외", ~m["month"].str.startswith("2020").values)]:
            mm = m[mask]
            rows.append({"variant": nm, "period": tag, "excluded_amt": m[nm + "_excl"].mean(), "sleeve": "A 단독", **stats_of(mm[nm].values)})
            rows.append({"variant": nm, "period": tag, "excluded_amt": m[nm + "_excl"].mean(), "sleeve": "A 90 + B 10", **stats_of((0.9 * mm[nm] + 0.1 * mm["B_SEL"]).values)})
    r = pd.DataFrame(rows); r.to_csv(OUT / "a_variants.csv", index=False)
    pd.set_option("display.width", 220)
    print(r.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
