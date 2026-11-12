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

OUT = ROOT / "results" / "full"
md.LADDER["M2L"] = md.LADDER["M2"] + md.LLM
rng = np.random.default_rng(0)


def main():
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
    d["S0"] = np.where(has, d["M2"], -1.0)
    d["H0"] = np.where(has, d["M2"], d["M3ns"])
    d["H1"] = np.where(has, d["M2L"], d["M4ns"])
    d["has_spread"] = has
    rows = []
    for which, s in [("경고 없는 구간 (주 분석)", d[d["warn"] == 0]), ("전체", d)]:
        ev = s["y_fa12"].sum()
        for c, nm in [("S0", "스프레드 모델만 (없으면 신호 없음)"), ("H0", "통합, LLM 없음"), ("H1", "통합, LLM 포함")]:
            r = {"sample": which, "signal": nm, "n": len(s), "events": int(ev), "share_no_spread_events": float(s.loc[~s["has_spread"], "y_fa12"].sum() / ev),
                 "PR": PR(s["y_fa12"], s[c]), "AUC": AUC(s["y_fa12"], s[c])}
            pct = s.groupby("month")[c].rank(pct=True, method="first")
            for top in (0.05, 0.10):
                f = pct >= 1 - top
                r[f"prec_top{int(top*100)}"] = s.loc[f, "y_fa12"].mean(); r[f"recall_top{int(top*100)}"] = s.loc[f, "y_fa12"].sum() / ev
            rows.append(r)
        r = {"sample": which, "signal": "무작위", "n": len(s), "events": int(ev), "PR": s["y_fa12"].mean(), "AUC": 0.5}
        rows.append(r)
    res = pd.DataFrame(rows); res.to_csv(OUT / "univ_metrics.csv", index=False)

    s = d[d["warn"] == 0]; firms = s["cik"].unique(); g = {c: s.index[s["cik"] == c] for c in firms}
    obs = PR(s["y_fa12"], s["H1"]) - PR(s["y_fa12"], s["H0"]); bs = []
    for _ in range(1000):
        idx = np.concatenate([g[c] for c in rng.choice(firms, len(firms))]); b = s.loc[idx]
        if b["y_fa12"].sum() == 0:
            continue
        bs.append(PR(b["y_fa12"], b["H1"]) - PR(b["y_fa12"], b["H0"]))
    bs = np.array(bs)
    pd.DataFrame([{"test": "H1 - H0 PR-AUC (경고 없는 구간)", "diff": obs, "ci_low": np.percentile(bs, 2.5), "ci_high": np.percentile(bs, 97.5),
                   "p_one": float((bs <= 0).mean())}]).to_csv(OUT / "univ_test.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(4).to_string(index=False)); print(pd.read_csv(OUT / "univ_test.csv").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
