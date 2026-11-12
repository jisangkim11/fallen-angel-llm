import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as PR
from sklearn.metrics import roc_auc_score as AUC

from robust.evaluate import primary_sample
from robust.models import LogitModel, LADDER, walk_forward

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "full"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.dpi": 150})
rng = np.random.default_rng(0)
NAMES = {"llm_bus": "LLM 사업위험", "llm_fin": "LLM 재무위험", "llm_liq": "LLM 유동성", "llm_pol": "LLM 재무정책",
         "llm_avg": "LLM 4개 평균", "lm_neg": "감성사전(LM 부정어)", "rel_spread": "채권 상대 스프레드"}


def cluster_ci(df, y, s, fn=AUC, n=500):
    firms = df["cik"].unique()
    g = {c: i for c, i in df.groupby("cik").indices.items()}
    vals = []
    for _ in range(n):
        idx = np.concatenate([g[c] for c in rng.choice(firms, len(firms))])
        d = df.iloc[idx]
        if d[y].nunique() == 2:
            vals.append(fn(d[y], d[s]))
    return np.percentile(vals, [2.5, 97.5])


def compare(pred, y, a, b, label):
    rows = []
    for m in (a, b):
        rows.append({"setting": label, "model": m, "n": len(pred), "events": int(pred[y].sum()),
                     "PR_AUC": PR(pred[y], pred[m]), "AUC": AUC(pred[y], pred[m])})
    return rows


def main():
    p = pd.read_parquet(ROOT / "data" / "interim" / "panel_final.parquet")
    num = p.select_dtypes("number").columns
    p[num] = p[num].replace([np.inf, -np.inf], np.nan)
    p["llm_avg"] = p[["llm_bus", "llm_fin", "llm_liq", "llm_pol"]].mean(axis=1)
    p["llm_avg_k"] = p[["llm_bus_k", "llm_fin_k", "llm_liq_k", "llm_pol_k"]].mean(axis=1)
    bt = p[p["year"].between(2016, 2024) & (p["censored"] == 0)]
    base = bt[bt["spread"].notna() & (bt["warn"] == 0)]

    t1 = []
    for c in ["llm_bus", "llm_fin", "llm_liq", "llm_pol", "llm_avg", "lm_neg", "rel_spread"]:
        s = base[base[c].notna()]
        lo, hi = cluster_ci(s, "y_fa12", c)
        t1.append({"signal": c, "name": NAMES[c], "AUC": AUC(s["y_fa12"], s[c]), "ci_low": lo, "ci_high": hi,
                   "n": len(s), "events": int(s["y_fa12"].sum())})
    t1 = pd.DataFrame(t1)
    t1.to_csv(OUT / "T1_standalone_auc.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    col = ["#2f6db5" if c.startswith("llm") else "#9aa4b1" for c in t1["signal"]]
    ax.barh(t1["name"], t1["AUC"], xerr=[t1["AUC"] - t1["ci_low"], t1["ci_high"] - t1["AUC"]], color=col, capsize=3)
    ax.axvline(0.5, color="k", lw=0.8, ls="--")
    ax.set_xlim(0.45, 0.9)
    ax.invert_yaxis()
    ax.set_xlabel("폴른엔젤 12개월 예측 AUC (S&P 경고 없는 구간, 2016-2024)")
    for i, (v, h) in enumerate(zip(t1["AUC"], t1["ci_high"])):
        ax.text(h + 0.005, i, f"{v:.3f}", va="center", fontsize=8, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_auc_bars.png")

    rows = []
    cov = p[p["llm_bus"].notna() & p["spread"].notna()].reset_index(drop=True)
    pr2, _ = walk_forward(cov, "y_fa12", models=("M2", "M3", "M4"))
    pr2 = primary_sample(pr2)
    rows += compare(pr2, "y_fa12", "M2", "M3", "T2 LLM 점수 있는 기업만")
    rows += compare(pr2, "y_fa12", "M4", "M4", "T2 LLM 점수 있는 기업만")[:1]
    lo2 = cluster_diff(pr2, "y_fa12", "M3", "M4")

    ns = p[p["llm_bus"].notna()].reset_index(drop=True)
    pr3, _ = walk_forward(ns, "y_fa12", models=("M3ns", "M4ns"))
    pr3 = primary_sample(pr3)
    rows += compare(pr3, "y_fa12", "M3ns", "M4ns", "T3 스프레드 없는 상황")
    lo3 = cluster_diff(pr3, "y_fa12", "M3ns", "M4ns")

    t4 = []
    for a, b in [("llm_avg_k", "llm_avg"), ("llm_fin_k", "llm_fin"), ("llm_liq_k", "llm_liq")]:
        s = base[base[a].notna()]
        t4.append({"pillar": b, "AUC_10K_only": AUC(s["y_fa12"], s[a]), "AUC_10K_plus_8K": AUC(s["y_fa12"], s[b])})
    pd.DataFrame(t4).to_csv(OUT / "T4_8k_ablation.csv", index=False)

    p["fa_mi"] = p["fa_month"].map(lambda x: pd.Period(x, freq="M").ordinal if isinstance(x, str) else np.nan)
    p["y_fa6"] = ((p["fa_mi"] - p["mi"]).between(1, 6)).astype(int)
    tr = p[p["year"].between(2014, 2023) & p["llm_bus"].notna() & (p["censored"] == 0)]
    te = p[(p["month"].astype(str) >= "2025-02") & (p["month"].astype(str) <= "2025-06") & p["llm_bus"].notna()].copy()
    for m in ("M3ns", "M4ns"):
        te[m] = LogitModel([c for c in LADDER[m] if c in p]).fit(tr, "y_fa6").predict(te)
    te5 = te
    if te5["y_fa6"].nunique() == 2:
        rows += compare(te5, "y_fa6", "M3ns", "M4ns", "T5 깨끗한 기간 2025-02~06 (6개월)")
        print("T5 events:", te5.loc[te5["y_fa6"] == 1, "cik"].map(load_names()).unique())
    pd.DataFrame(rows).to_csv(OUT / "T2_T3_T5_model_compare.csv", index=False)

    last = p["month"].max()
    trall = p[(p["censored"] == 0) & p["llm_bus"].notna() & (p["year"] >= 2014)]
    cur = p[(p["month"] == last) & p["llm_bus"].notna() & p["rating"].astype(str).str.startswith("BBB")].copy()
    cur["risk"] = LogitModel([c for c in LADDER["M4ns"] if c in p]).fit(trall, "y_fa12").predict(cur)
    names = load_names()
    cur["name"] = cur["cik"].map(names)
    w = cur.sort_values("risk", ascending=False)[["cik", "name", "rating", "outlook", "risk", "llm_bus", "llm_fin",
                                                   "llm_liq", "llm_pol"]].head(20)
    w.to_csv(OUT / "T6_watchlist.csv", index=False)

    p["to_fa"] = p["fa_mi"] - p["mi"]
    evm = p[p["to_fa"].between(1, 12)].groupby("to_fa")["llm_avg"].mean()
    ctrl = p[p["fa_month"].isna() & p["llm_avg"].notna()]["llm_avg"].mean()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(-evm.index, evm.values, marker="o", ms=3, color="#2f6db5", label="폴른엔젤이 되는 기업")
    ax.axhline(ctrl, color="#9aa4b1", ls="--", label="나머지 BBB 기업 평균")
    ax.set_xlabel("폴른엔젤 강등까지 남은 개월 (0 = 강등 월)")
    ax.set_ylabel("LLM 위험 백분위 (4개 평균)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_event_time.png")

    print(t1.round(3).to_string(index=False))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("T2 PR-AUC diff (M4-M3):", {k: round(v, 4) for k, v in lo2.items()})
    print("T3 PR-AUC diff (M4ns-M3ns):", {k: round(v, 4) for k, v in lo3.items()})
    print(pd.DataFrame(t4).round(3).to_string(index=False))
    print("watchlist month", last)
    print(w.round(3).to_string(index=False))
    print("event-time: 24m before", round(evm.get(24, np.nan), 3), "1m before", round(evm.get(1, np.nan), 3),
          "others", round(ctrl, 3))


def cluster_diff(df, y, a, b, n=500):
    firms = df["cik"].unique()
    g = {c: i for c, i in df.groupby("cik").indices.items()}
    d0 = PR(df[y], df[b]) - PR(df[y], df[a])
    vals = []
    for _ in range(n):
        d = df.iloc[np.concatenate([g[c] for c in rng.choice(firms, len(firms))])]
        if d[y].nunique() == 2:
            vals.append(PR(d[y], d[b]) - PR(d[y], d[a]))
    vals = np.array(vals)
    return {"diff": d0, "ci_low": np.percentile(vals, 2.5), "ci_high": np.percentile(vals, 97.5),
            "p_one_sided": float((vals <= 0).mean())}


def load_names():
    for f in [ROOT / "data" / "interim" / "universe_sec.csv", ROOT / "data" / "interim" / "universe_candidates.csv"]:
        if f.exists():
            u = pd.read_csv(f)
            col = next((c for c in u.columns if "name" in c.lower()), None)
            if col and "cik" in u:
                return u.drop_duplicates("cik").set_index("cik")[col].to_dict()
    return {}


if __name__ == "__main__":
    main()


