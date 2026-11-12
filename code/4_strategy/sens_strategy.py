import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import build_panel_base as bpb
from robust import models as md
from robust import stats_tests as st

OUT = ROOT / "results" / "full"
md.LADDER["M2L"] = md.LADDER["M2"] + md.LLM


def stats_of(r):
    r = np.asarray(r); w = np.cumprod(1 + r)
    return {"ann_ret": w[-1] ** (12 / len(r)) - 1, "ann_vol": r.std(ddof=1) * np.sqrt(12), "sharpe": r.mean() / r.std(ddof=1) * np.sqrt(12),
            "cvar95": st.cvar(r, 0.05), "max_dd": float((1 - w / np.maximum.accumulate(w)).max())}


def main():
    p = pd.read_parquet(ROOT / "data" / "interim" / "panel_final.parquet")
    num = p.select_dtypes("number").columns
    p[num] = p[num].replace([np.inf, -np.inf], np.nan)
    a = p[p["spread"].notna() & p["year"].between(2014, 2024)].reset_index(drop=True)
    pred, _ = md.walk_forward(a, "y_fa12", models=("M2L",))
    pred = pred[pred["year"].between(2016, 2024)].copy()
    pred["pct"] = pred.groupby("month")["M2L"].rank(pct=True)
    pred["r"] = pred["credit_ret"].clip(-0.5, 0.5)
    pred["hc"] = pred["half_cost"].fillna(pred["half_cost"].median())
    uni = pd.read_csv(ROOT / "data" / "interim" / "universe_sec.csv")
    iss = bpb.issuer_spreads(uni); iss["mi"] = iss["m"].apply(lambda q: q.ordinal); iss = iss.sort_values(["cik", "mi"])
    iss["ret_cur"] = iss.groupby("cik")["credit_ret"].shift(1).clip(-0.5, 0.5)
    evp = p[p["fa_month"].notna()].copy(); evp["fmi"] = evp["fa_month"].map(lambda x: pd.Period(x, freq="M").ordinal)
    last = evp[evp["mi"] == evp["fmi"] - 1].drop_duplicates(["cik", "fa_month"])
    sleeve = []
    for _, e in last.iterrows():
        g = iss[(iss["cik"] == e["cik"]) & iss["mi"].between(e["fmi"] + 1, e["fmi"] + 18)][["mi", "ret_cur", "amt"]]
        for _, x in g.iterrows():
            sleeve.append({"t": int(x["mi"]), "ret": x["ret_cur"], "amt": x["amt"], "k": int(x["mi"] - e["fmi"])})
    sl = pd.DataFrame(sleeve).dropna(subset=["ret"])
    months = sorted(pred["mi"].unique())
    rows = []
    for q in (0.05, 0.10, 0.20):
        for cm in (1.0, 2.0):
            prev = {}; recs = []
            for t1 in months:
                d = pred[pred["mi"] == t1].set_index("cik"); rec = {"mi": t1 + 1}
                for nm, keep in {"IDX": np.ones(len(d), bool), "DEF": (d["pct"] < 1 - q).values}.items():
                    w = d["amt"].where(keep, 0.0); w = w / w.sum(); wp = prev.get(nm, pd.Series(dtype=float)).reindex(w.index).fillna(0)
                    rec[nm] = float((w * d["r"].fillna(0)).sum()) - cm * float(((w - wp).abs() * d["hc"]).sum())
                    rec[nm + "_amt"] = float(d["amt"].where(keep, 0.0).sum()); rec[nm + "_to"] = float((w - wp).abs().sum()); prev[nm] = w
                for H in (6, 12):
                    z = sl[(sl["t"] == t1 + 1) & (sl["k"] <= H)]
                    if len(z):
                        wz = z["amt"] / z["amt"].sum()
                        rec[f"FA{H}"] = float((wz * z["ret"]).sum()) - cm * 0.0025 * float((wz * ((z["k"] == 1).astype(float) + (z["k"] == H).astype(float))).sum())
                        rec[f"FA{H}_amt"] = float(z["amt"].sum())
                    else:
                        rec[f"FA{H}"], rec[f"FA{H}_amt"] = 0.0, 0.0
                recs.append(rec)
            m = pd.DataFrame(recs)
            base = stats_of(m["IDX"].values)
            rows.append({"threshold": f"top {int(q*100)}%", "cost_x": cm, "strategy": "BBB 지수", **base, "turnover": m["IDX_to"].mean()})
            rows.append({"threshold": f"top {int(q*100)}%", "cost_x": cm, "strategy": "방어", **stats_of(m["DEF"].values), "turnover": m["DEF_to"].mean(),
                         "excluded_share": 1 - (m["DEF_amt"] / m["IDX_amt"]).mean()})
            for H in (6, 12):
                mix = (m["DEF_amt"] * m["DEF"] + m[f"FA{H}_amt"] * m[f"FA{H}"]) / (m["DEF_amt"] + m[f"FA{H}_amt"])
                rows.append({"threshold": f"top {int(q*100)}%", "cost_x": cm, "strategy": f"방어 + 매수 {H}개월", **stats_of(mix.values),
                             "fa_share": (m[f"FA{H}_amt"] / (m["DEF_amt"] + m[f"FA{H}_amt"])).mean()})
            print("done", q, cm, flush=True)
    r = pd.DataFrame(rows); r.to_csv(OUT / "final_sens.csv", index=False)
    pd.set_option("display.width", 220); print(r.round(4).to_string(index=False))
    ex = {"n_fa_events_sleeve": int(last.shape[0]), "sleeve_issuer_months": int(len(sl)), "n_months": len(months),
          "avg_bbb_issuers": float(pred.groupby("mi").size().mean()), "avg_alarmed_issuers_top10": float((pred["pct"] >= 0.9).groupby(pred["mi"]).sum().mean())}
    pd.Series(ex).to_csv(OUT / "final_sens_extra.csv"); print(ex)


if __name__ == "__main__":
    main()
