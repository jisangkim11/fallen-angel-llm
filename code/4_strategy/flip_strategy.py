import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
import build_panel_base as bpb
from robust import models as md
from robust import stats_tests as st

OUT = ROOT / "results" / "full"
md.LADDER["M2L"] = md.LADDER["M2"] + md.LLM
COST, NB = 0.0025, 2000
rng = np.random.default_rng(0)


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


def sharpe(r): r = np.asarray(r); return r.mean() / r.std(ddof=1) * np.sqrt(12) if r.std(ddof=1) > 0 else np.nan
def mdd(r): w = np.cumprod(1 + np.asarray(r)); return float((1 - w / np.maximum.accumulate(w)).max())


def stats_of(r):
    r = np.asarray(r); w = np.cumprod(1 + r)
    return {"ann_ret": w[-1] ** (12 / len(r)) - 1, "ann_vol": r.std(ddof=1) * np.sqrt(12), "sharpe": sharpe(r),
            "cvar95": st.cvar(r, 0.05), "max_dd": mdd(r), "hit_month": float((r > 0).mean())}


def logit(x): x = np.clip(x, 1e-6, 1 - 1e-6); return np.log(x / (1 - x))


def predict_calibrated(a):
    cols = [c for c in md.LADDER["M2L"] if c in a.columns]
    out, oos = [], []
    for Y in range(2016, 2025):
        tr = a[(a["year"] >= 2014) & (a["year"] <= Y - 2) & (a["censored"] == 0)]
        te = a[a["year"] == Y].copy()
        m = md.LogitModel(cols, 0.5).fit(tr, "y_fa12")
        te["raw"] = m.predict(te)
        hist = [o for o in oos if o["year"].iloc[0] <= Y - 2]
        cal = pd.concat(hist) if hist else tr.assign(raw=m.predict(tr))
        cal = cal.dropna(subset=["raw", "y_fa12"])
        pl = LogisticRegression(C=1e6).fit(logit(cal["raw"].values).reshape(-1, 1), cal["y_fa12"].astype(int))
        te["prob"] = pl.predict_proba(logit(te["raw"].values).reshape(-1, 1))[:, 1]
        te["cal_src"] = "oos" if hist else "train"
        out.append(te); oos.append(te[["year", "raw", "y_fa12"]])
    return pd.concat(out).reset_index(drop=True)


def capped(w, cap):
    w = pd.Series(w, dtype=float)
    if w.empty or w.sum() <= 0:
        return w * 0
    w = w / w.sum()
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any():
            break
        free = (~over) & (w > 0)
        extra = (w[over] - cap).sum(); w[over] = cap
        if free.any():
            w[free] += extra * w[free] / w[free].sum()
    return w.clip(upper=cap)


def main():
    p = pd.read_parquet(ROOT / "data" / "interim" / "panel_final.parquet")
    num = p.select_dtypes("number").columns
    p[num] = p[num].replace([np.inf, -np.inf], np.nan)
    p = p.sort_values(["cik", "mi"])
    p["sp3"] = p["spread"] - p.groupby("cik")["spread"].shift(3)
    p.loc[p.groupby("cik")["mi"].diff(3) != 3, "sp3"] = np.nan
    a = p[p["spread"].notna() & p["year"].between(2014, 2024)].reset_index(drop=True)
    pred = predict_calibrated(a)
    pred["r"] = pred["credit_ret"].clip(-0.5, 0.5)
    pred["prank"] = pred.groupby("mi")["prob"].rank(pct=True)
    pred["srank"] = pred.groupby("mi")["spread"].rank(pct=True)
    pred["top"] = pred.groupby("mi")["prob"].rank(ascending=False, method="first")

    uni = pd.read_csv(ROOT / "data" / "interim" / "universe_sec.csv")
    iss = bpb.issuer_spreads(uni); iss["mi"] = iss["m"].apply(lambda q: q.ordinal)
    iss["rn"] = iss["credit_ret"].clip(-0.5, 0.5)
    RN = iss.set_index(["cik", "mi"])["rn"].to_dict()
    SP = iss.set_index(["cik", "mi"])["spread"].to_dict()
    rm, _ = bpb.ratings_monthly(uni); rm["mi"] = rm["m"].apply(lambda q: q.ordinal)
    NOTCH = rm.set_index(["cik", "mi"])["notch"].to_dict()

    ev = p[p["fa_month"].notna()].copy(); ev["fmi"] = ev["fa_month"].map(lambda x: pd.Period(x, freq="M").ordinal)
    ev = ev[ev["mi"] < ev["fmi"]].sort_values("mi").groupby(["cik", "fmi"]).tail(1)
    ev = ev[["cik", "fmi", "llm_bus", "spread"]].sort_values("fmi").reset_index(drop=True)
    FA_AT = {}
    for r in ev.itertuples():
        FA_AT.setdefault(int(r.fmi), []).append(r)
    FA_SET = {(int(r.cik), int(r.fmi)) for r in ev.itertuples()}

    months = sorted(pred["mi"].unique())
    first, last = months[0], months[-1]

    def run(n_top=10, borrow=0.005, months_mask=None):
        shorts, longs_llm, longs_all = {}, {}, {}
        prev_w = {"S": {}, "LL": {}, "LA": {}}
        rec, trades = [], []
        for t in months:
            d = pred[pred["mi"] == t].set_index("cik")
            for c in list(shorts):
                pos = shorts[c]; reason = None
                if (c, t) in FA_SET: reason = "downgrade"
                elif t - pos["entry"] >= 12: reason = "12m"
                elif c not in d.index: reason = "left data"
                elif d.loc[c, "prank"] < 0.8: reason = "rank out"
                elif SP.get((c, t), np.nan) <= 0.7 * pos["sp0"]: reason = "spread -30%"
                if reason:
                    trades.append({"leg": "short", "cik": c, "entry": pos["entry"], "exit": t, "reason": reason, "pnl": pos["pnl"]})
                    del shorts[c]
            cand = d[(d["top"] <= n_top) & (d["sp3"] > 0) & (d["srank"] <= 0.8)]
            for c in cand.index:
                if c not in shorts:
                    shorts[c] = {"entry": t, "sp0": d.loc[c, "spread"], "pnl": 0.0}
            for book, use_llm in ((longs_llm, True), (longs_all, False)):
                for c in list(book):
                    pos = book[c]; nt = NOTCH.get((c, t), np.nan); reason = None
                    if t - pos["entry"] >= 12: reason = "12m"
                    elif nt >= 14: reason = "B+ or lower"
                    elif nt <= 10: reason = "back to IG"
                    elif (c, t) not in RN or np.isnan(RN[(c, t)]): reason = "left data"
                    if reason:
                        trades.append({"leg": "long_llm" if use_llm else "long_all", "cik": c, "entry": pos["entry"], "exit": t, "reason": reason, "pnl": pos["pnl"]})
                        del book[c]
                prior = ev[ev["fmi"] < t]["llm_bus"].dropna()
                med = prior.median() if len(prior) else np.nan
                for e in FA_AT.get(t, []):
                    c = int(e.cik)
                    if (c, t) not in RN or np.isnan(RN[(c, t)]):
                        continue
                    if use_llm and not (pd.notna(e.llm_bus) and pd.notna(med) and e.llm_bus <= med):
                        continue
                    book[c] = {"entry": t, "pnl": 0.0}
            ws = capped({c: d.loc[c, "prob"] if c in d.index else 0.0 for c in shorts}, 0.15)
            n_l, n_a = len(longs_llm), len(longs_all)
            wl = pd.Series({c: min(1 / n_l, 0.10) for c in longs_llm}, dtype=float) if n_l else pd.Series(dtype=float)
            wa = pd.Series({c: min(1 / n_a, 0.10) for c in longs_all}, dtype=float) if n_a else pd.Series(dtype=float)
            widx = d["amt"] / d["amt"].sum(); r_idx = float((widx * d["r"].fillna(0)).sum())

            def leg_ret(w, sign, key):
                rr = 0.0
                for c, x in w.items():
                    ri = RN.get((c, t), np.nan); ri = 0.0 if np.isnan(ri) else ri
                    rr += sign * x * ri
                    pos = (shorts if key == "S" else longs_llm if key == "LL" else longs_all)[c]
                    pos["pnl"] = (1 + pos["pnl"]) * (1 + sign * ri) - 1
                pw = prev_w[key]; names = set(w.index) | set(pw)
                to = sum(abs(w.get(c, 0.0) - pw.get(c, 0.0)) for c in names)
                prev_w[key] = dict(w)
                return rr - COST * to, to

            rs, tos = leg_ret(ws, -1, "S")
            gross_s = float(ws.sum())
            rs = rs + gross_s * r_idx - borrow / 12 * gross_s
            rl, tol = leg_ret(wl, +1, "LL"); ra, toa = leg_ret(wa, +1, "LA")
            rec.append({"mi": t + 1, "IDX": r_idx, "SHORT": rs, "LONG_LLM": rl, "LONG_ALL": ra, "FLIP": 0.5 * rs + 0.5 * rl,
                        "n_short": len(ws), "gross_short": gross_s, "n_long_llm": n_l, "n_long_all": n_a, "to_s": tos, "to_l": tol})
        m = pd.DataFrame(rec); m["month"] = m["mi"].apply(lambda q: str(pd.Period(ordinal=int(q), freq="M")))
        return m, pd.DataFrame(trades)

    NAMES = {"IDX": "BBB 지수", "SHORT": "숏만 (지수 헤지)", "LONG_LLM": "롱만 (LLM 선별)", "LONG_ALL": "롱만 (전부 매수)", "FLIP": "숏→롱 전환"}
    base, trades = run()
    base.to_csv(OUT / "flip_monthly.csv", index=False); trades.to_csv(OUT / "flip_trades.csv", index=False)

    rows = []
    def add(tag, m):
        for k, nm in NAMES.items():
            rows.append({"case": tag, "strategy": nm, **stats_of(m[k].values)})
    add("기본 (상위 10, 차입 0.5%)", base)
    add("2020년 제외", base[~base["month"].str.startswith("2020")])
    for b in (0.0, 0.01):
        mb, _ = run(borrow=b); add(f"차입 {b*100:.1f}%", mb)
    m5, _ = run(n_top=5); add("상위 5곳", m5)
    summ = pd.DataFrame(rows); summ.to_csv(OUT / "flip_summary.csv", index=False)

    tr = trades.copy()
    tstat = []
    for leg, g in tr.groupby("leg"):
        tstat.append({"leg": leg, "n_trades": len(g), "win_rate": float((g["pnl"] > 0).mean()), "avg_pnl": g["pnl"].mean(), "med_pnl": g["pnl"].median(),
                      "avg_hold": float((g["exit"] - g["entry"]).mean()), **{f"exit_{k}": int(v) for k, v in g["reason"].value_counts().items()}})
    tstat = pd.DataFrame(tstat); tstat.to_csv(OUT / "flip_trade_stats.csv", index=False)
    ex = {"months": len(base), "avg_short_names": base["n_short"].mean(), "avg_short_gross": base["gross_short"].mean(),
          "months_no_short": int((base["n_short"] == 0).sum()), "avg_long_llm": base["n_long_llm"].mean(), "avg_long_all": base["n_long_all"].mean(),
          "months_no_long_llm": int((base["n_long_llm"] == 0).sum()), "short_downgrade_hit": float((tr[tr.leg == "short"]["reason"] == "downgrade").mean()),
          "cal_src_train_years": "2016-2017"}
    pd.Series(ex).to_csv(OUT / "flip_extra.csv")

    x0, xf, xs, xl, xa = (base[k].values for k in ("IDX", "FLIP", "SHORT", "LONG_LLM", "LONG_ALL"))
    n = len(x0); b1, b2, b3 = [], [], []
    for _ in range(NB):
        i = sb_idx(n); b1.append(sharpe(xf[i]) - sharpe(x0[i])); b2.append(xs[i].mean()); b3.append((xl[i] - xa[i]).mean())
    T = pd.DataFrame([
        {"test": "1 숏→롱 전환 샤프 > 지수 샤프", "diff": sharpe(xf) - sharpe(x0), "p_one": float((np.array(b1) <= 0).mean())},
        {"test": "2 숏 수익 > 0 (월평균)", "diff": xs.mean(), "p_one": float((np.array(b2) <= 0).mean())},
        {"test": "3 LLM 선별 롱 > 전부 매수 롱 (월평균 차이)", "diff": (xl - xa).mean(), "p_one": float((np.array(b3) <= 0).mean())}])
    T["p_holm"] = holm(T["p_one"].values); T.to_csv(OUT / "flip_tests.csv", index=False)

    pd.set_option("display.width", 250)
    print(summ.round(4).to_string(index=False)); print(tstat.round(4).to_string(index=False)); print(ex); print(T.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
