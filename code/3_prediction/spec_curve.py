import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as PR

ROOT = Path(__file__).resolve().parents[2]
from robust import models as md

OUT = ROOT / "results" / "full"
md.LADDER["M2L"] = md.LADDER["M2"] + md.LLM


def universe():
    cache = OUT / "univ_preds.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    p = pd.read_parquet(ROOT / "data" / "interim" / "panel_final.parquet")
    num = p.select_dtypes("number").columns
    p[num] = p[num].replace([np.inf, -np.inf], np.nan)
    p = p[p["year"].between(2014, 2024)].reset_index(drop=True)
    full, _ = md.walk_forward(p, "y_fa12", models=("M3ns", "M4ns"))
    sp, _ = md.walk_forward(p[p["spread"].notna()].reset_index(drop=True), "y_fa12", models=("M2", "M2L"))
    d = full.merge(sp[["cik", "mi", "M2", "M2L"]], on=["cik", "mi"], how="left")
    d = d[d["year"].between(2016, 2024)]
    d = d[~((d["y_fa12"] == 1) & (d["known"] == 1))].copy()
    has = d["spread"].notna() & d["M2"].notna()
    d["H0"] = np.where(has, d["M2"], d["M3ns"]); d["H1"] = np.where(has, d["M2L"], d["M4ns"])
    d = d[["cik", "month", "year", "warn", "y_fa12", "H0", "H1"]]
    d.to_parquet(cache, index=False)
    return d


def metric(s, col, m):
    if m == "PR-AUC":
        return PR(s["y_fa12"], s[col])
    top = float(m.split()[1].rstrip("%")) / 100
    f = s.groupby("month")[col].rank(pct=True, method="first") >= 1 - top
    return s.loc[f, "y_fa12"].mean()


def main():
    fp = pd.read_csv(OUT / "final_preds.csv"); fp = fp[~((fp["y_fa12"] == 1) & (fp["known"] == 1))]
    un = universe()
    comps = [("C1 복합 vs 스프레드 (스프레드 표본)", fp, "M2L", "M2"), ("C2 LLM 포함 vs 미포함 (전체 유니버스)", un, "H1", "H0")]
    samples = {"경고 없음": lambda d: d[d["warn"] == 0], "S&P 경고": lambda d: d[d["warn"] == 1], "전체": lambda d: d}
    periods = {"2016-24": lambda d: d, "2020 제외": lambda d: d[d["year"] != 2020], "2016·2020 제외": lambda d: d[~d["year"].isin([2016, 2020])]}
    metrics = ["PR-AUC"] + [f"상위 {k}%" for k in (1, 2, 3, 5, 10, 20)]
    rows = []
    for cn, d, a, b in comps:
        for sn, sf in samples.items():
            for pn, pf in periods.items():
                s = pf(sf(d))
                for m in metrics:
                    va, vb = metric(s, a, m), metric(s, b, m)
                    rows.append({"comparison": cn, "sample": sn, "period": pn, "metric": m, "events": int(s["y_fa12"].sum()),
                                 "with_llm": va, "without": vb, "diff": va - vb, "rel": va / vb - 1 if vb > 0 else np.nan, "llm_better": va > vb})
    r = pd.DataFrame(rows); r.to_csv(OUT / "spec_curve.csv", index=False)
    summ = r.groupby(["comparison", "sample"]).agg(cells=("llm_better", "size"), llm_wins=("llm_better", "sum"), median_rel=("rel", "median")).reset_index()
    tot = r.groupby("comparison").agg(cells=("llm_better", "size"), llm_wins=("llm_better", "sum"), median_rel=("rel", "median")).reset_index().assign(sample="합계")
    summ = pd.concat([summ, tot]); summ.to_csv(OUT / "spec_summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(summ.round(3).to_string(index=False))
    print(r[r["llm_better"]].sort_values("rel", ascending=False).head(25).round(4).to_string(index=False))
    print(r[~r["llm_better"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
