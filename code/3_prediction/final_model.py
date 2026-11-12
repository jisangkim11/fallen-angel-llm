import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as PR, roc_auc_score as AUC

ROOT = Path(__file__).resolve().parents[2]
import build_panel_base as bpb
from robust import models as md
from robust import stats_tests as st

OUT = ROOT / "results" / "full"
FA_COST = 0.0025
rng = np.random.default_rng(0)
NB = 2000
md.LADDER["M2L"] = md.LADDER["M2"] + md.LLM
MODELS = ("M1", "M2", "M3", "M4", "M2L")
SIG = {"S&P 경고": None, "스프레드 모델 (M2)": "M2", "복합 모델 (M2+LLM)": "M2L"}


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj


def sb_idx(n, L=6):
    idx = np.empty(n, int); i = rng.integers(n)
    for t in range(n):
        if t > 0:
            i = rng.integers(n) if rng.random() < 1 / L else (i + 1) % n
        idx[t] = i
    return idx


def sharpe(r): return r.mean() / r.std(ddof=1) * np.sqrt(12)


def mdd(r):
    w = np.cumprod(1 + r); return float((1 - w / np.maximum.accumulate(w)).max())


def stats_of(r):
    r = np.asarray(r); w = np.cumprod(1 + r)
    return {"ann_ret": w[-1] ** (12 / len(r)) - 1, "ann_vol": r.std(ddof=1) * np.sqrt(12), "sharpe": sharpe(r),
            "cvar95": st.cvar(r, 0.05), "max_dd": mdd(r)}


def main():
    p = pd.read_parquet(ROOT / "data" / "interim" / "panel_final.parquet")
    num = p.select_dtypes("number").columns
    p[num] = p[num].replace([np.inf, -np.inf], np.nan)
    a = p[p["spread"].notna() & p["year"].between(2014, 2024)].reset_index(drop=True)
    pred, _ = md.walk_forward(a, "y_fa12", models=MODELS)
    pred = pred[pred["year"].between(2016, 2024)].copy()
    for c in MODELS:
        pred[c + "_pct"] = pred.groupby("month")[c].rank(pct=True)
    nk = ~((pred["y_fa12"] == 1) & (pred["known"] == 1))

    rows = []
    for which, mask in [("경고 없는 구간 (주 분석)", nk & (pred["warn"] == 0)), ("S&P 경고 기업", nk & (pred["warn"] == 1)), ("전체", nk)]:
        s = pred[mask]
        for c in MODELS:
            q = s.groupby("month")[c].rank(pct=True) >= 0.9
            rows.append({"sample": which, "model": c, "n": len(s), "events": int(s["y_fa12"].sum()), "PR": PR(s["y_fa12"], s[c]),
                         "AUC": AUC(s["y_fa12"], s[c]), "top10_capture": s.loc[q, "y_fa12"].sum() / s["y_fa12"].sum()})
    a1 = pd.DataFrame(rows); a1.to_csv(OUT / "final_A1_metrics.csv", index=False)
    print("== A1 =="); print(a1.round(4).to_string(index=False))

    rows = []
    for which, mask in [("전체", nk), ("S&P 경고 기업", nk & (pred["warn"] == 1))]:
        s = pred[mask]
        if which == "전체":
            f = s["warn"] == 1
            rows.append({"sample": which, "signal": "S&P 경고", "top": "경고 여부", "flagged": int(f.sum()), "precision": s.loc[f, "y_fa12"].mean(),
                         "recall": s.loc[f, "y_fa12"].sum() / s["y_fa12"].sum()})
        for nm, c in list(SIG.items())[1:]:
            for top in (0.05, 0.10, 0.20):
                f = s[c + "_pct"] >= 1 - top if which == "전체" else s.groupby("month")[c].rank(pct=True) >= 1 - top
                rows.append({"sample": which, "signal": nm, "top": f"상위 {int(top * 100)}%", "flagged": int(f.sum()),
                             "precision": s.loc[f, "y_fa12"].mean(), "recall": s.loc[f, "y_fa12"].sum() / s["y_fa12"].sum()})
        rows.append({"sample": which, "signal": "기준 (평균 강등률)", "top": "-", "flagged": len(s), "precision": s["y_fa12"].mean(), "recall": 1.0})
    a2 = pd.DataFrame(rows); a2.to_csv(OUT / "final_A2_alarm.csv", index=False)
    print("== A2 =="); print(a2.round(4).to_string(index=False))

    pred["fa_mi"] = pred["fa_month"].map(lambda x: pd.Period(x, freq="M").ordinal if isinstance(x, str) else np.nan)
    pred["k"] = pred["mi"] - pred["fa_mi"]
    ev = pred[pred["k"].between(-12, -1) & ~(pred["known"] == 1)].sort_values(["cik", "fa_month", "k"])
    path = ev.groupby("k").agg(M2=("M2_pct", "mean"), M2L=("M2L_pct", "mean"), M1=("M1_pct", "mean"), warn=("warn", "mean"), n=("cik", "size")).reset_index()
    path.to_csv(OUT / "final_A3_path.csv", index=False)
    print("== A3 path =="); print(path.round(3).to_string(index=False))

    s = pred[nk & (pred["warn"] == 0)]
    a4 = s.groupby("year").apply(lambda g: pd.Series({"events": int(g["y_fa12"].sum()), "PR_M2": PR(g["y_fa12"], g["M2"]),
                                                       "PR_M2L": PR(g["y_fa12"], g["M2L"])}), include_groups=False).reset_index()
    a4["better"] = a4["PR_M2L"] > a4["PR_M2"]
    a4.to_csv(OUT / "final_A4_yearly.csv", index=False)
    print("== A4 =="); print(a4.round(4).to_string(index=False))

    per = []
    for (cik, fam), g in ev.groupby(["cik", "fa_month"]):
        r = g["credit_ret"].fillna(0).clip(-0.5, 0.5).values
        rec = {"cik": cik, "fa_month": fam, "months": len(g), "loss_hold": np.prod(1 + r) - 1}
        for nm, c in SIG.items():
            flag = (g["warn"] == 1).values if c is None else (g[c + "_pct"] >= 0.9).values
            if flag.any():
                i = int(np.argmax(flag)); rec[nm + "_lead"] = -g["k"].values[i]; rec[nm + "_av"] = -(np.prod(1 + r[i:]) - 1)
            else:
                rec[nm + "_lead"], rec[nm + "_av"] = np.nan, 0.0
        per.append(rec)
    pe = pd.DataFrame(per); pe.to_csv(OUT / "final_B1_per_event.csv", index=False)
    b1 = []
    for nm, c in SIG.items():
        fl = pred[pred["warn"] == 1] if c is None else pred[pred[c + "_pct"] >= 0.9]
        al = pe[nm + "_lead"].notna()
        b1.append({"signal": nm, "events": len(pe), "alarm_rate": al.mean(), "median_lead": pe.loc[al, nm + "_lead"].median(),
                   "avg_avoided": pe[nm + "_av"].mean(), "precision": fl["y_fa12"].mean()})
    b1 = pd.DataFrame(b1); b1.to_csv(OUT / "final_B1_exit.csv", index=False)
    print("== B1 =="); print(b1.round(4).to_string(index=False)); print("hold loss", round(pe["loss_hold"].mean(), 4))
    eqt = pred[pred["cik"] == 33213][["month", "M2_pct", "M2L_pct", "warn", "rel_spread"]]
    eqt.to_csv(OUT / "final_eqt.csv", index=False)

    pred["r"] = pred["credit_ret"].clip(-0.5, 0.5)
    pred["hc"] = pred["half_cost"].fillna(pred["half_cost"].median())
    uni = pd.read_csv(ROOT / "data" / "interim" / "universe_sec.csv")
    iss = bpb.issuer_spreads(uni)
    iss["mi"] = iss["m"].apply(lambda q: q.ordinal)
    iss = iss.sort_values(["cik", "mi"])
    iss["ret_cur"] = iss.groupby("cik")["credit_ret"].shift(1).clip(-0.5, 0.5)
    evp = p[p["fa_month"].notna()].copy()
    evp["fmi"] = evp["fa_month"].map(lambda x: pd.Period(x, freq="M").ordinal)
    last = evp[evp["mi"] == evp["fmi"] - 1].drop_duplicates(["cik", "fa_month"])
    sleeve = []
    for _, e in last.iterrows():
        g = iss[(iss["cik"] == e["cik"]) & iss["mi"].between(e["fmi"] + 1, e["fmi"] + 12)][["mi", "ret_cur", "amt"]]
        for _, x in g.iterrows():
            sleeve.append({"t": int(x["mi"]), "ret": x["ret_cur"], "amt": x["amt"], "first": x["mi"] == e["fmi"] + 1, "lastm": x["mi"] == e["fmi"] + 12})
    sl = pd.DataFrame(sleeve).dropna(subset=["ret"])
    rows, prev = [], {}
    for t1 in sorted(pred["mi"].unique()):
        d = pred[pred["mi"] == t1].set_index("cik")
        rec = {"mi": t1 + 1}
        books = {"IDX": np.ones(len(d), bool), "D_SP": (d["warn"] != 1).values, "D_M2": (d["M2_pct"] < 0.9).values, "D_M2L": (d["M2L_pct"] < 0.9).values}
        for nm, keep in books.items():
            w = d["amt"].where(keep, 0.0); w = w / w.sum()
            wp = prev.get(nm, pd.Series(dtype=float)).reindex(w.index).fillna(0)
            rec[nm + "_to"] = float((w - wp).abs().sum())
            rec[nm] = float((w * d["r"].fillna(0)).sum()) - float(((w - wp).abs() * d["hc"]).sum())
            rec[nm + "_amt"] = float(d["amt"].where(keep, 0.0).sum())
            prev[nm] = w
        z = sl[sl["t"] == t1 + 1]
        if len(z):
            wz = z["amt"] / z["amt"].sum()
            rec["FA"] = float((wz * z["ret"]).sum()) - FA_COST * float((wz * (z["first"].astype(float) + z["lastm"].astype(float))).sum())
            rec["FA_amt"] = float(z["amt"].sum())
        else:
            rec["FA"], rec["FA_amt"] = 0.0, 0.0
        rows.append(rec)
    m = pd.DataFrame(rows)
    mix = lambda b: (m[b + "_amt"] * m[b] + m["FA_amt"] * m["FA"]) / (m[b + "_amt"] + m["FA_amt"])
    m["IDX_FA"], m["D_M2_FA"], m["D_M2L_FA"] = mix("IDX"), mix("D_M2"), mix("D_M2L")
    m["month"] = m["mi"].apply(lambda q: str(pd.Period(ordinal=int(q), freq="M")))
    m.to_csv(OUT / "final_B_monthly.csv", index=False)
    names = {"IDX": "BBB 지수", "D_SP": "방어: S&P 경고 제외", "D_M2": "방어: 스프레드 모델 상위 10% 제외", "D_M2L": "방어: 복합 모델 상위 10% 제외",
             "IDX_FA": "지수 + 폴른 엔젤 보유", "D_M2_FA": "스프레드 모델 방어 + 매수", "D_M2L_FA": "복합 모델 방어 + 매수"}
    res = []
    for k, nm in names.items():
        res.append({"key": k, "name": nm, **stats_of(m[k].values), "turnover": m[k + "_to"].mean() if k + "_to" in m else np.nan})
    rs = pd.DataFrame(res); rs.to_csv(OUT / "final_B_summary.csv", index=False)
    print("== B2/B3 =="); print(rs.round(4).to_string(index=False))
    sub = []
    for lab, mask in [("2016~2019", m["month"] < "2020-01"), ("2020~2024", m["month"] >= "2020-01")]:
        for k, nm in names.items():
            sub.append({"period": lab, "key": k, "name": nm, **stats_of(m.loc[mask, k].values)})
    pd.DataFrame(sub).to_csv(OUT / "final_B_subperiod.csv", index=False)

    T = []
    for lab, base in [("T1 선제 매도: 복합 vs S&P 경고", "S&P 경고"), ("T2 선제 매도: 복합 vs 스프레드 모델", "스프레드 모델 (M2)")]:
        d = pe["복합 모델 (M2+LLM)_av"].values - pe[base + "_av"].values; n = len(d)
        bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(NB)])
        T.append({"test": lab, "diff": d.mean(), "ci_low": np.percentile(bs, 2.5), "ci_high": np.percentile(bs, 97.5), "p_one": (bs <= 0).mean(),
                  "better": int((d > 0).sum()), "worse": int((d < 0).sum())})
    for lab, a_, b_ in [("T3 방어 CVaR: 복합 vs 지수", "IDX", "D_M2L"), ("T4 방어 CVaR: 복합 vs 스프레드 모델", "D_M2", "D_M2L")]:
        xa, xb = m[a_].values, m[b_].values; bb = []
        for _ in range(NB):
            i = sb_idx(len(xa)); bb.append(st.cvar(xb[i]) - st.cvar(xa[i]))
        bb = np.array(bb)
        T.append({"test": lab, "diff": st.cvar(xb) - st.cvar(xa), "ci_low": np.percentile(bb, 2.5), "ci_high": np.percentile(bb, 97.5), "p_one": (bb >= 0).mean()})
    x0, x3 = m["IDX"].values, m["D_M2L_FA"].values; n = len(x0); bsh, bmd = [], []
    for _ in range(NB):
        i = sb_idx(n); bsh.append(sharpe(x3[i]) - sharpe(x0[i])); bmd.append(mdd(x0[i]) - mdd(x3[i]))
    bsh, bmd = np.array(bsh), np.array(bmd)
    T.append({"test": "T5 통합 샤프: 복합 방어+매수 vs 지수", "diff": sharpe(x3) - sharpe(x0), "ci_low": np.percentile(bsh, 2.5), "ci_high": np.percentile(bsh, 97.5), "p_one": (bsh <= 0).mean()})
    T.append({"test": "T6 통합 최대낙폭 감소: 지수 - 복합 방어+매수", "diff": mdd(x0) - mdd(x3), "ci_low": np.percentile(bmd, 2.5), "ci_high": np.percentile(bmd, 97.5), "p_one": (bmd <= 0).mean()})
    tt = pd.DataFrame(T)
    tt["p_holm"] = holm(tt["p_one"].fillna(1).values)
    tt.to_csv(OUT / "final_tests.csv", index=False)
    print("== tests =="); print(tt.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
