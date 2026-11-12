import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
RAW = ROOT / "data" / "raw" / "sec"
END = pd.Period("2025-12", "M")


def ratings_monthly(uni):
    r = pd.read_csv(INTERIM / "sp_obligor_ratings.csv", parse_dates=["date"])
    r = r[r["obligor"].isin(uni["obligor"])].drop(columns=["cik"]).merge(uni[["obligor", "cik"]], on="obligor")
    r["m"] = r["date"].dt.to_period("M")
    r = r.sort_values(["cik", "date"])
    last = r.groupby(["cik", "m"]).tail(1)
    months = pd.period_range("2012-01", "2025-12", freq="M")
    rows = []
    last = last.assign(outlook=last["outlook"].fillna(""), watch=last["watch"].fillna(""))
    for cik, g in last.groupby("cik"):
        g = g.set_index("m")[["rating", "notch", "outlook", "watch"]].infer_objects()
        g = g[~g.index.duplicated(keep="last")].reindex(months).ffill()
        g["cik"] = cik
        rows.append(g.reset_index(names="m"))
    return pd.concat(rows), r


def labels(panel, actions):
    acts = actions[["cik", "m", "notch", "rating"]].copy()
    acts["mi"] = acts["m"].apply(lambda p: p.ordinal)
    panel["mi"] = panel["m"].apply(lambda p: p.ordinal)
    out = {k: np.zeros(len(panel)) for k in ("y_fa12", "y_dg12", "nr_exit")}
    fa_m = np.full(len(panel), np.nan)
    by = {c: g for c, g in acts.groupby("cik")}
    for i, (cik, mi, notch) in enumerate(zip(panel["cik"].values, panel["mi"].values, panel["notch"].values)):
        g = by.get(cik)
        if g is None or np.isnan(notch):
            continue
        w = g[(g["mi"] > mi) & (g["mi"] <= mi + 12)]
        fa = w[w["notch"] >= 11]
        if len(fa):
            out["y_fa12"][i] = 1
            fa_m[i] = fa["mi"].iloc[0]
        if (w["notch"] > notch).any():
            out["y_dg12"][i] = 1
        if not len(fa) and (w["rating"] == "NR").any():
            out["nr_exit"][i] = 1
    for k, v in out.items():
        panel[k] = v.astype(int)
    panel["fa_month"] = [str(pd.Period(ordinal=int(x), freq="M")) if not np.isnan(x) else np.nan for x in fa_m]
    panel["censored"] = (panel["mi"] + 12 > END.ordinal).astype(int)
    return panel


def last_downgrade_gap(panel, actions):
    a = actions.sort_values(["cik", "date"]).copy()
    a["prev"] = a.groupby("cik")["notch"].shift()
    dg = a[a["notch"] > a["prev"]][["cik", "m"]]
    dg["mi"] = dg["m"].apply(lambda p: p.ordinal)
    panel = panel.sort_values("mi")
    dg = dg.sort_values("mi").rename(columns={"mi": "dg_mi"})[["cik", "dg_mi"]]
    panel = pd.merge_asof(panel, dg, left_on="mi", right_on="dg_mi", by="cik", direction="backward")
    panel["months_since_dg"] = panel["mi"] - panel["dg_mi"]
    return panel.drop(columns="dg_mi"), dg


def issuer_spreads(uni):
    bm = pd.read_parquet(INTERIM / "bond_monthly.parquet")
    lk = pd.read_csv(INTERIM / "bond_issuer_link.csv")
    cand = pd.read_csv(INTERIM / "universe_candidates.csv")
    lei2ob = cand.dropna(subset=["lei"]).drop_duplicates("lei").set_index("lei")["obligor"]
    lk["obligor"] = lk["issuer_lei"].map(lei2ob)
    lk = lk.merge(uni[["obligor", "cik"]], on="obligor")[["cusip_id", "cik"]]
    b = bm.merge(lk, on="cusip_id")
    b = b[(b["mod_dur"] > 1) & b["credit_spread"].between(-0.02, 0.5)]
    b["w"] = b["bond_amt_outstanding"].clip(lower=1)
    g = b.groupby(["cik", "month"])
    iss = pd.DataFrame({
        "spread": g.apply(lambda d: np.average(d["credit_spread"], weights=d["w"]), include_groups=False),
        "credit_ret_cur": g.apply(lambda d: np.average(d["credit_ret"].fillna(0), weights=d["w"]), include_groups=False),
        "half_cost": g["half_spread_cost"].median(),
        "amt": g["w"].sum(),
        "n_bonds": g.size()}).reset_index()
    iss["m"] = pd.PeriodIndex(iss["month"], freq="M")
    iss = iss.sort_values(["cik", "m"])
    iss["credit_ret"] = iss.groupby("cik")["credit_ret_cur"].shift(-1)
    return iss.drop(columns=["month", "credit_ret_cur"])


def sic_codes(ciks):
    out = {}
    for c in ciks:
        p = RAW / f"submissions_{c}.json"
        if p.exists():
            s = json.loads(p.read_text(encoding="utf-8")).get("sic") or ""
            out[c] = int(s) if str(s).isdigit() else np.nan
    return pd.Series(out, name="sic")


def main():
    uni = pd.read_csv(INTERIM / "universe_sec.csv")
    panel, actions = ratings_monthly(uni)
    panel = panel[panel["notch"].between(8, 10)].copy()
    panel = labels(panel.reset_index(drop=True), actions)
    panel, dg = last_downgrade_gap(panel, actions)
    panel["notch_pos"] = panel["notch"] - 7
    panel["warn"] = (panel["outlook"].fillna("").str.contains("Negative")
                     | panel["watch"].fillna("").str.contains("Neg")).astype(int)

    iss = issuer_spreads(uni)
    panel = panel.merge(iss, on=["cik", "m"], how="left")
    med = panel.groupby("m")["spread"].transform("median")
    panel["rel_spread"] = panel["spread"] - med

    sic = sic_codes(panel["cik"].unique())
    panel = panel.merge(sic, left_on="cik", right_index=True, how="left")
    panel["sic2"] = (panel["sic"] // 100).astype("Int64")
    panel["financial"] = panel["sic"].between(6000, 6411).astype(int)

    dg = dg.merge(panel[["cik", "sic2"]].drop_duplicates("cik"), on="cik")
    cnt = panel.groupby(["sic2", "mi"])["cik"].nunique().rename("n_ind").reset_index()
    dgc = dg.groupby(["sic2", "dg_mi"])["cik"].nunique().rename("n_dg").reset_index().rename(columns={"dg_mi": "mi"})
    ind = cnt.merge(dgc, on=["sic2", "mi"], how="left").fillna({"n_dg": 0}).sort_values(["sic2", "mi"])
    ind["dg6"] = ind.groupby("sic2")["n_dg"].transform(lambda s: s.shift(1).rolling(6, min_periods=1).sum())
    ind["ind_dg_rate"] = ind["dg6"] / ind["n_ind"].clip(lower=1)
    panel = panel.merge(ind[["sic2", "mi", "ind_dg_rate"]], on=["sic2", "mi"], how="left")

    panel["month"] = panel["m"].astype(str)
    panel["year"] = panel["m"].apply(lambda p: p.year)
    panel = panel.drop(columns=["m"])
    panel.to_parquet(INTERIM / "panel_base.parquet", index=False)

    s = panel[(panel["year"] >= 2013) & (panel["financial"] == 0)]
    print(f"issuer-months {len(panel):,}; non-financial 2013+: {len(s):,}; issuers {s['cik'].nunique()}")
    print(f"with spread: {s['spread'].notna().mean():.0%}; warn share {s['warn'].mean():.1%}")
    ev = s[s["fa_month"].notna()].drop_duplicates(["cik", "fa_month"])
    print("fallen-angel events by year (non-financial):")
    print(ev.groupby(ev["fa_month"].str[:4]).size().to_string())


if __name__ == "__main__":
    main()
