import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
RESULTS = ROOT / "results"
PILLARS = {"business": "llm_bus", "financial": "llm_fin", "liquidity": "llm_liq", "policy": "llm_pol"}
HALF_LIFE = 3


def month_index(dates):
    return pd.to_datetime(dates).dt.to_period("M").apply(lambda p: p.ordinal)


def asof(panel, df, cols):
    df = df.dropna(subset=["filing_date"]).copy()
    df["fmi"] = month_index(df["filing_date"])
    df = df.sort_values("fmi")[["cik", "fmi"] + cols]
    out = pd.merge_asof(panel.sort_values("mi"), df, left_on="mi", right_on="fmi", by="cik", direction="backward")
    return out


def llm_10k():
    files = glob.glob(str(RESULTS / "scores_shard10-K_*.csv"))
    if not files:
        return None
    s = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for p in PILLARS:
        ev, sc = f"{p}_ev", f"{p}_score"
        s[p] = s[ev] if ev in s else np.nan
        s[p] = s[p].fillna(pd.to_numeric(s.get(sc), errors="coerce"))
    agg = s.groupby(["key", "cik", "filing_date"])[list(PILLARS)].mean().reset_index()
    agg = agg.rename(columns={p: f"raw_{p}" for p in PILLARS})
    unc = [c for c in s.columns if c.endswith("_unc")]
    if unc:
        agg = agg.merge(s.groupby("key")[unc].mean().mean(axis=1).rename("llm_unc").reset_index(), on="key", how="left")
    print(f"LLM 10-K scores: {len(agg):,} filings")
    return agg


def llm_8k_decay(panel):
    files = glob.glob(str(RESULTS / "scores_shard8-K_*.csv"))
    if not files:
        return panel
    e = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    e = e.drop_duplicates(["key", "impact_no"])
    credit = set()
    for f in glob.glob(str(ROOT / "data" / "kaggle_upload" / "shard_8-K_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            if any(i in (d.get("items") or "") for i in ("2.01", "2.03", "2.05", "2.06")):
                credit.add(d["key"])
    e = e[e["key"].isin(credit)]
    e = e[e["pillar"].isin(PILLARS)].copy()
    e["rc"] = pd.to_numeric(e["risk_change"], errors="coerce").fillna(0)
    e["emi"] = month_index(e["filing_date"])
    ev = e.groupby(["cik", "emi", "pillar"])["rc"].sum().reset_index()
    print(f"LLM 8-K impacts: {len(e):,} rows from {e['key'].nunique():,} filings")
    for p, col in PILLARS.items():
        sub = ev[ev["pillar"] == p]
        by = {c: g[["emi", "rc"]].values for c, g in sub.groupby("cik")}
        vals = np.zeros(len(panel))
        for i, (c, mi) in enumerate(zip(panel["cik"].values, panel["mi"].values)):
            a = by.get(c)
            if a is None:
                continue
            d = mi - a[:, 0]
            k = (d >= 0) & (d <= 12)
            vals[i] = float(np.sum(a[k, 1] * 0.5 ** (d[k] / HALF_LIFE)))
        panel[f"ev8k_{p}"] = vals
    return panel


def known_events():
    p = RESULTS / "memory_answers.csv"
    if not p.exists():
        return set()
    q = pd.read_csv(p)
    if "p_yes" in q:
        thr = max(0.5, float(q.loc[q["type"] == "control", "p_yes"].quantile(0.95)))
        k = q[(q["type"] == "event") & (q["p_yes"] >= thr)]
        print(f"memory: threshold {thr:.2f}, known events {len(k)} of {(q['type'] == 'event').sum()}")
    else:
        k = q[(q["type"] == "event") & (q["answer"] == "yes")]
        k = k[k["answer_month"].astype(str).str[:4] == k["year"].astype(str)]
    return set(zip(k["cik"], k["year"]))


def main():
    panel = pd.read_parquet(INTERIM / "panel_base.parquet")
    panel = panel[(panel["financial"] == 0) & (panel["year"] >= 2013)].copy()

    fin = pd.read_csv(INTERIM / "fin_xbrl.csv")
    panel = asof(panel, fin, ["size_log", "lev_ind_adj", "lev_chg", "mat_ratio", "payout_assets"]).drop(columns="fmi")

    lm = INTERIM / "lm_dict.csv"
    if lm.exists():
        panel = asof(panel, pd.read_csv(lm), ["lm_neg", "lm_pol"]).drop(columns="fmi")

    k = llm_10k()
    if k is not None:
        panel = asof(panel, k, [f"raw_{p}" for p in PILLARS] + (["llm_unc"] if "llm_unc" in k else []))
        panel["llm_age"] = panel["mi"] - panel["fmi"]
        panel = panel.drop(columns="fmi")
        panel = llm_8k_decay(panel)
        for p, col in PILLARS.items():
            raw10k = panel[f"raw_{p}"]
            raw = raw10k + (panel[f"ev8k_{p}"] if f"ev8k_{p}" in panel else 0)
            panel[col] = raw.groupby(panel["month"]).rank(pct=True)
            panel[col + "_k"] = raw10k.groupby(panel["month"]).rank(pct=True)
    known = known_events()
    panel["known"] = [int((c, int(str(f)[:4])) in known) if isinstance(f, str) else 0
                      for c, f in zip(panel["cik"], panel["fa_month"])]
    panel = panel[panel["nr_exit"] == 0]
    panel.to_parquet(INTERIM / "panel_final.parquet", index=False)

    s = panel[panel["spread"].notna()]
    print(f"panel rows {len(panel):,}; with spread {len(s):,}; issuers {s['cik'].nunique()}")
    print("non-missing share:", panel[["size_log", "lev_ind_adj", "mat_ratio", "payout_assets"]
                                      + [c for c in ("lm_neg", "llm_bus") if c in panel]].notna().mean().round(2).to_dict())
    ev = s[s["fa_month"].notna()].drop_duplicates(["cik", "fa_month"])
    print("fallen-angel events (with spread):", len(ev), "| known to LLM:", int(ev["known"].sum()))


if __name__ == "__main__":
    main()
