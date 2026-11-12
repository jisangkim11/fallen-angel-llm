import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
cik = int(sys.argv[1])
p = pd.read_parquet(ROOT / "data" / "interim" / "panel_final.parquet")
d = p[p["cik"] == cik].sort_values("mi").copy()
d["llm_avg"] = d[["llm_bus", "llm_fin", "llm_liq", "llm_pol"]].mean(axis=1)
cols = ["month", "rating", "outlook", "watch", "warn", "spread", "llm_bus", "llm_fin", "llm_liq", "llm_pol", "llm_avg", "fa_month", "credit_ret"]
d[cols].to_csv(ROOT / "results" / "full" / f"case_{cik}.csv", index=False)
print(d[cols].tail(30).round(3).to_string(index=False))
s = pd.concat([pd.read_csv(f) for f in glob.glob(str(ROOT / "results" / "scores_shard10-K_*.csv"))], ignore_index=True)
s = s[s["cik"] == cik].sort_values("filing_date")
keep = ["filing_date"] + [c for c in s.columns if c.endswith("_ev") or c.endswith("_evidence")]
s[keep].to_csv(ROOT / "results" / "full" / f"case_{cik}_evidence.csv", index=False)
pd.set_option("display.max_colwidth", 220)
for _, r in s.tail(3).iterrows():
    print("\n==", r["filing_date"])
    for ax in ["business", "financial", "liquidity", "policy"]:
        print(f"  {ax}: {r[ax + '_ev']:.2f} | {r[ax + '_evidence']}")
e = pd.concat([pd.read_csv(f) for f in glob.glob(str(ROOT / "results" / "scores_shard8-K_*.csv"))], ignore_index=True)
e = e[(e["cik"] == cik) & (e["filing_date"] >= "2018-06-01") & (e["filing_date"] <= "2020-06-30")].sort_values("filing_date")
print("\n8-K events:")
print(e[["filing_date", "pillar", "risk_change", "evidence"]].drop_duplicates().to_string(index=False))
