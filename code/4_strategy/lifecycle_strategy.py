import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import build_panel_base as bpb
from robust import stats_tests as st

OUT = ROOT / "results" / "full"
NB = 2000
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


def sharpe(r): r = np.asarray(r); return r.mean() / r.std(ddof=1) * np.sqrt(12)
def mdd(r): w = np.cumprod(1 + np.asarray(r)); return float((1 - w / np.maximum.accumulate(w)).max())


def stats_of(r):
    r = np.asarray(r); w = np.cumprod(1 + r)
    return {"ann_ret": w[-1] ** (12 / len(r)) - 1, "ann_vol": r.std(ddof=1) * np.sqrt(12), "sharpe": sharpe(r),
            "cvar95": st.cvar(r, 0.05), "max_dd": mdd(r), "hit_month": float((r > 0).mean())}


def capped(score, cap):
    w = pd.Series(score, dtype=float)
    if w.empty:
        return w
    w = w.clip(lower=1e-9); w = w / w.sum(); cap = max(cap, 1 / len(w))
    for _ in range(50):
        over = w > cap + 1e-12
        if not over.any():
            break
        free = ~over; extra = (w[over] - cap).sum(); w[over] = cap
        w[free] += extra * w[free] / w[free].sum()
    return w


def main():
    p = pd.read_parquet(ROOT / "data" / "interim" / "panel_final.parquet")
    num = p.select_dtypes("number").columns
    p[num] = p[num].replace([np.inf, -np.inf], np.nan)
    A = pd.read_csv(OUT / "final_B_monthly.csv")[["mi", "month", "IDX", "D_M2L"]]
    months = [int(x) - 1 for x in A["mi"]]

    uni = pd.read_csv(ROOT / "data" / "interim" / "universe_sec.csv")
    iss = bpb.issuer_spreads(uni); iss["mi"] = iss["m"].apply(lambda q: q.ordinal)
    RN = iss.set_index(["cik", "mi"])["credit_ret"].clip(-0.5, 0.5).to_dict()
    rm, _ = bpb.ratings_monthly(uni); rm["mi"] = rm["m"].apply(lambda q: q.ordinal)
    NOTCH = rm.set_index(["cik", "mi"])["notch"].to_dict()

    ev = p[p["fa_month"].notna()].copy(); ev["fmi"] = ev["fa_month"].map(lambda x: pd.Period(x, freq="M").ordinal)
    ev = ev[ev["mi"] < ev["fmi"]].sort_values("mi").groupby(["cik", "fmi"]).tail(1)
    ev = ev[["cik", "fmi", "rel_spread", "llm_bus"]].sort_values("fmi").reset_index(drop=True)

    def pct_upto(col, F, x):
        hist = ev.loc[ev["fmi"] <= F, col].dropna()
        return 0.5 if (pd.isna(x) or len(hist) < 5) else float((hist <= x).mean())

    ev["depth_pct"] = [pct_upto("rel_spread", r.fmi, r.rel_spread) for r in ev.itertuples()]
    ev["llm_pct"] = [pct_upto("llm_bus", r.fmi, r.llm_bus) for r in ev.itertuples()]
    ev["selected"] = (ev["depth_pct"] >= 1 / 3) & (ev["llm_pct"] <= 0.8)

    def sleeve(lag=0, cost=0.0025, select=True):
        book, prev, rec, trades = {}, {}, [], []
        starts = {}
        for r in ev.itertuples():
            starts.setdefault(int(r.fmi) + lag, []).append(r)
        for t in months:
            for c in list(book):
                pos = book[c]; nt = NOTCH.get((c, t), np.nan); reason = None
                if t - pos["entry"] >= 12: reason = "12m"
                elif nt >= 14: reason = "B+ or lower"
                elif nt <= 10: reason = "back to IG"
                elif pd.isna(RN.get((c, t), np.nan)): reason = "no data"
                if reason:
                    trades.append({"cik": c, "fmi": pos["fmi"], "entry": pos["entry"], "exit": t, "reason": reason, "pnl": pos["pnl"],
                                   "depth_pct": pos["d"], "llm_pct": pos["l"], "selected": pos["sel"]})
                    del book[c]
            for e in starts.get(t, []):
                c = int(e.cik)
                if c in book or pd.isna(RN.get((c, t), np.nan)):
                    continue
                if lag and NOTCH.get((c, t), 11) <= 10:
                    continue
                if select and not e.selected:
                    continue
                book[c] = {"entry": t, "fmi": int(e.fmi), "pnl": 0.0, "d": e.depth_pct, "l": e.llm_pct, "sel": bool(e.selected)}
            sc = {c: (book[c]["d"] if select else 1.0) for c in book}
            w = capped(sc, 0.10) if sc else pd.Series(dtype=float)
            rr = 0.0
            for c, x in w.items():
                ri = RN.get((c, t), 0.0); ri = 0.0 if pd.isna(ri) else ri
                rr += x * ri; book[c]["pnl"] = (1 + book[c]["pnl"]) * (1 + ri) - 1
            names = set(w.index) | set(prev); to = sum(abs(w.get(c, 0) - prev.get(c, 0)) for c in names); prev = dict(w)
            rec.append({"mi": t + 1, "ret": rr - cost * to, "n": len(w), "to": to})
        return pd.DataFrame(rec), pd.DataFrame(trades)

    base_s, tr_s = sleeve()
    base_a, tr_a = sleeve(select=False)
    m = A.copy()
    m["B_SEL"], m["B_ALL"] = base_s["ret"].values, base_a["ret"].values
    m["n_sel"], m["n_all"] = base_s["n"].values, base_a["n"].values
    m["AB"] = 0.9 * m["D_M2L"] + 0.1 * m["B_SEL"]
    m["AB_ALL"] = 0.9 * m["D_M2L"] + 0.1 * m["B_ALL"]
    m["IDXB"] = 0.9 * m["IDX"] + 0.1 * m["B_ALL"]
    m.to_csv(OUT / "life_monthly.csv", index=False)
    tr = pd.concat([tr_s.assign(book="selected"), tr_a.assign(book="all")]); tr.to_csv(OUT / "life_trades.csv", index=False)

    NAMES = {"IDX": "BBB 지수", "D_M2L": "A 방어 (복합 모델)", "B_ALL": "B 폴른 엔젤 전부 매수", "B_SEL": "B 폴른 엔젤 선별 매수",
             "IDXB": "지수 90 + 전부 매수 10", "AB_ALL": "A 90 + 전부 매수 10", "AB": "A 90 + B 선별 10 (생애주기)"}
    rows = []
    def add(tag, mm):
        for k, nm in NAMES.items():
            if k in mm:
                rows.append({"case": tag, "strategy": nm, **stats_of(mm[k].values)})
    add("기본", m)
    add("2020년 제외", m[~m["month"].str.startswith("2020")])
    for lag in (1, 3):
        s, _ = sleeve(lag=lag); a, _ = sleeve(lag=lag, select=False)
        mm = A.copy(); mm["B_SEL"], mm["B_ALL"] = s["ret"].values, a["ret"].values; mm["AB"] = 0.9 * mm["D_M2L"] + 0.1 * mm["B_SEL"]
        add(f"진입 {lag}개월 뒤", mm[["month", "B_ALL", "B_SEL", "AB"]])
    s, _ = sleeve(cost=0.005); a, _ = sleeve(cost=0.005, select=False)
    mm = A.copy(); mm["B_SEL"], mm["B_ALL"] = s["ret"].values, a["ret"].values; mm["AB"] = 0.9 * mm["D_M2L"] + 0.1 * mm["B_SEL"]
    add("비용 2배 (B만)", mm[["month", "B_ALL", "B_SEL", "AB"]])
    mm = m.copy(); mm["AB"] = 0.8 * mm["D_M2L"] + 0.2 * mm["B_SEL"]; mm["AB_ALL"] = 0.8 * mm["D_M2L"] + 0.2 * mm["B_ALL"]
    add("결합 80/20", mm[["month", "AB", "AB_ALL"]])
    summ = pd.DataFrame(rows); summ.to_csv(OUT / "life_summary.csv", index=False)

    ts = []
    for bk, g in tr.groupby("book"):
        ts.append({"book": bk, "n_trades": len(g), "win_rate": float((g["pnl"] > 0).mean()), "avg_pnl": g["pnl"].mean(), "med_pnl": g["pnl"].median(),
                   "avg_hold": float((g["exit"] - g["entry"]).mean()), **{f"exit_{k}": int(v) for k, v in g["reason"].value_counts().items()}})
    g = tr[tr["book"] == "all"]
    for lab, sub in [("all_but_not_selected", g[~g["selected"]]), ("all_selected", g[g["selected"]])]:
        ts.append({"book": lab, "n_trades": len(sub), "win_rate": float((sub["pnl"] > 0).mean()), "avg_pnl": sub["pnl"].mean(), "med_pnl": sub["pnl"].median()})
    ts = pd.DataFrame(ts); ts.to_csv(OUT / "life_trade_stats.csv", index=False)

    x0, xs, xa, xab = (m[k].values for k in ("IDX", "B_SEL", "B_ALL", "AB"))
    n = len(x0); b1, b2, b3 = [], [], []
    for _ in range(NB):
        i = sb_idx(n); b1.append(sharpe(xs[i]) - sharpe(xa[i])); b2.append(sharpe(xs[i]) - sharpe(x0[i])); b3.append(sharpe(xab[i]) - sharpe(x0[i]))
    T = pd.DataFrame([
        {"test": "1 B 선별 샤프 > B 전부 매수 샤프", "diff": sharpe(xs) - sharpe(xa), "p_one": float((np.array(b1) <= 0).mean())},
        {"test": "2 B 선별 샤프 > BBB 지수 샤프", "diff": sharpe(xs) - sharpe(x0), "p_one": float((np.array(b2) <= 0).mean())},
        {"test": "3 A+B 샤프 > BBB 지수 샤프", "diff": sharpe(xab) - sharpe(x0), "p_one": float((np.array(b3) <= 0).mean())}])
    T["p_holm"] = holm(T["p_one"].values); T.to_csv(OUT / "life_tests.csv", index=False)
    ex = {"events": len(ev), "selected_events": int(ev["selected"].sum()), "avg_names_sel": m["n_sel"].mean(), "avg_names_all": m["n_all"].mean(),
          "months_empty_sel": int((m["n_sel"] == 0).sum()), "months_empty_all": int((m["n_all"] == 0).sum())}
    pd.Series(ex).to_csv(OUT / "life_extra.csv")
    pd.set_option("display.width", 250)
    print(summ.round(4).to_string(index=False)); print(ts.round(4).to_string(index=False)); print(T.round(4).to_string(index=False)); print(ex)


if __name__ == "__main__":
    main()
