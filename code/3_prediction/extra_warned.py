import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as PR, roc_auc_score as AUC

from robust.models import walk_forward

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "full"
rng = np.random.default_rng(0)


def cluster_diff(df, y, a, b, n=2000):
    firms = df["cik"].unique()
    g = df.groupby("cik").indices
    d0 = PR(df[y], df[b]) - PR(df[y], df[a])
    da0 = AUC(df[y], df[b]) - AUC(df[y], df[a])
    vals, avals = [], []
    for _ in range(n):
        d = df.iloc[np.concatenate([g[c] for c in rng.choice(firms, len(firms))])]
        if d[y].nunique() == 2:
            vals.append(PR(d[y], d[b]) - PR(d[y], d[a])); avals.append(AUC(d[y], d[b]) - AUC(d[y], d[a]))
    vals, avals = np.array(vals), np.array(avals)
    return {"pr_diff": d0, "pr_lo": np.percentile(vals, 2.5), "pr_hi": np.percentile(vals, 97.5), "pr_p": float((vals <= 0).mean()),
            "auc_diff": da0, "auc_lo": np.percentile(avals, 2.5), "auc_hi": np.percentile(avals, 97.5), "auc_p": float((avals <= 0).mean())}


def topshare(df, y, m, q):
    cut = df[m].quantile(1 - q)
    top = df[df[m] >= cut]
    return top[y].mean(), top[y].sum() / max(df[y].sum(), 1)


def evaluate(pred, models, label):
    w = pred[(pred["warn"] == 1) & ~((pred["y_fa12"] == 1) & (pred["known"] == 1))].reset_index(drop=True)
    rows = []
    for m in models:
        hit20, cap20 = topshare(w, "y_fa12", m, 0.2)
        rows.append({"setting": label, "model": m, "n": len(w), "events": int(w["y_fa12"].sum()), "base_rate": w["y_fa12"].mean(),
                     "PR_AUC": PR(w["y_fa12"], w[m]), "AUC": AUC(w["y_fa12"], w[m]), "top20_hit_rate": hit20, "top20_capture": cap20})
    return w, rows


def main():
    p = pd.read_parquet(ROOT / "data" / "interim" / "panel_final.parquet")
    num = p.select_dtypes("number").columns
    p[num] = p[num].replace([np.inf, -np.inf], np.nan)

    a = p[p["spread"].notna() & p["year"].between(2014, 2024)].reset_index(drop=True)
    pa, _ = walk_forward(a, "y_fa12", models=("M1", "M2", "M3", "M3b", "M4"))
    wa, rows = evaluate(pa, ("M1", "M2", "M3", "M3b", "M4"), "A. 스프레드 표본, 경고 기업")
    da = cluster_diff(wa, "y_fa12", "M3", "M4")
    da2 = cluster_diff(wa, "y_fa12", "M2", "M4")

    b = p[p["llm_bus"].notna() & p["year"].between(2014, 2024)].reset_index(drop=True)
    pb, _ = walk_forward(b, "y_fa12", models=("M3ns", "M4ns"))
    wb, rows_b = evaluate(pb, ("M3ns", "M4ns"), "B. 스프레드 없는 설정, 경고 기업")
    rows += rows_b
    db = cluster_diff(wb, "y_fa12", "M3ns", "M4ns")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "extra_warned.csv", index=False)
    pd.DataFrame([{"comparison": "A: M4 vs M3", **da}, {"comparison": "A: M4 vs M2", **da2}, {"comparison": "B: M4ns vs M3ns", **db}]).to_csv(OUT / "extra_warned_diff.csv", index=False)
    print(res.round(4).to_string(index=False))
    for k, d in [("A M4-M3", da), ("A M4-M2", da2), ("B M4ns-M3ns", db)]:
        print(k, {kk: round(v, 4) for kk, v in d.items()})
    for c in ["llm_bus", "llm_fin", "llm_liq", "llm_pol", "rel_spread", "lm_neg"]:
        s = wa[wa[c].notna()]
        print("standalone", c, round(AUC(s["y_fa12"], s[c]), 3), "n", len(s))


if __name__ == "__main__":
    main()
