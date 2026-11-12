import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
import build_panel_base as bpb

rng = np.random.default_rng(0)


def main():
    uni = pd.read_csv(ROOT / "data" / "interim" / "universe_sec.csv")
    iss = bpb.issuer_spreads(uni)
    iss["mi"] = iss["m"].apply(lambda p: p.ordinal)
    iss = iss.sort_values(["cik", "mi"])
    iss["ret_cur"] = iss.groupby("cik")["credit_ret"].shift(1).clip(-0.5, 0.5)
    p = pd.read_parquet(ROOT / "data" / "interim" / "panel_final.parquet")
    num = p.select_dtypes("number").columns
    p[num] = p[num].replace([np.inf, -np.inf], np.nan)
    p["llm_avg"] = p[["llm_bus", "llm_fin", "llm_liq", "llm_pol"]].mean(axis=1)
    q = p[p["credit_ret"].notna()].copy()
    q["w"] = q["amt"] / q.groupby("mi")["amt"].transform("sum")
    bbb = (q["w"] * q["credit_ret"].clip(-0.5, 0.5)).groupby(q["mi"]).sum()
    bbb.index = bbb.index + 1
    ev = p[p["fa_month"].notna()].copy()
    ev["fmi"] = ev["fa_month"].map(lambda x: pd.Period(x, freq="M").ordinal)
    last = ev[ev["mi"] == ev["fmi"] - 1].drop_duplicates(["cik", "fa_month"])
    rows = []
    for _, r in last.iterrows():
        g = iss[(iss["cik"] == r["cik"]) & iss["mi"].between(r["fmi"] + 1, r["fmi"] + 12)].set_index("mi")["ret_cur"]
        if g.notna().sum() < 6:
            continue
        rec = {"cik": r["cik"], "fa_month": r["fa_month"], "year": int(r["fa_month"][:4]), "llm_avg": r["llm_avg"], "llm_bus": r["llm_bus"],
               "llm_fin": r["llm_fin"], "rel_spread": r["rel_spread"], "known": r["known"]}
        for h in (6, 12):
            idx = range(int(r["fmi"]) + 1, int(r["fmi"]) + h + 1)
            rr = g.reindex(idx).fillna(0).values
            bb = bbb.reindex(idx).fillna(0).values
            rec[f"ret{h}"] = np.prod(1 + rr) - 1
            rec[f"xs{h}"] = rec[f"ret{h}"] - (np.prod(1 + bb) - 1)
        rows.append(rec)
    e = pd.DataFrame(rows)
    e.to_csv(ROOT / "results" / "full" / "ext_buy_fa_events.csv", index=False)
    print("events with post-downgrade bond data:", len(e), "| with LLM:", e["llm_avg"].notna().sum())
    print("all fallen angels: mean ret12 %.4f, xs12 %.4f, ret6 %.4f" % (e["ret12"].mean(), e["xs12"].mean(), e["ret6"].mean()))
    out = []
    for sig in ["llm_avg", "llm_bus", "llm_fin", "rel_spread"]:
        s = e[e[sig].notna()]
        med = s[sig].median()
        lo, hi = s[s[sig] <= med], s[s[sig] > med]
        for h in ("ret6", "ret12", "xs12"):
            diff = lo[h].mean() - hi[h].mean()
            t = stats.ttest_ind(lo[h], hi[h], equal_var=False)
            boots = []
            for _ in range(2000):
                b = s.sample(len(s), replace=True, random_state=int(rng.integers(1e9)))
                bm = b[sig].median()
                boots.append(b.loc[b[sig] <= bm, h].mean() - b.loc[b[sig] > bm, h].mean())
            boots = np.array(boots)
            rho = stats.spearmanr(s[sig], s[h])
            out.append({"signal": sig, "horizon": h, "n": len(s), "low_risk_mean": lo[h].mean(), "high_risk_mean": hi[h].mean(),
                        "diff_low_minus_high": diff, "p_welch": t.pvalue, "p_boot_one_sided": float((boots <= 0).mean()),
                        "spearman": rho.correlation, "p_spearman": rho.pvalue})
    o = pd.DataFrame(out)
    o.to_csv(ROOT / "results" / "full" / "ext_buy_fa.csv", index=False)
    print(o.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
