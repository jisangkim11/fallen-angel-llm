import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from make_results import load_names
from robust.models import LADDER, LogitModel

ROOT = Path(__file__).resolve().parents[2]
p = pd.read_parquet(ROOT / "data" / "interim" / "panel_final.parquet")
num = p.select_dtypes("number").columns
p[num] = p[num].replace([np.inf, -np.inf], np.nan)
last = p["month"].max()
trall = p[(p["censored"] == 0) & p["llm_bus"].notna() & (p["year"] >= 2014)]
cur = p[(p["month"] == last) & p["llm_bus"].notna() & p["rating"].astype(str).str.startswith("BBB")].copy()
cur["risk"] = LogitModel([c for c in LADDER["M4ns"] if c in p]).fit(trall, "y_fa12").predict(cur)
cur["name"] = cur["cik"].map(load_names())
cur = cur.sort_values("risk", ascending=False).reset_index(drop=True); cur["rank"] = cur.index + 1
print("month", last, "N BBB with LLM", len(cur), "BBB- count", (cur["rating"] == "BBB-").sum())
for k in ["PARAMOUNT", "COTY", "HOLOGIC", "CENTENE", "ORACLE", "WHIRLPOOL", "SKYDANCE"]:
    m = cur[cur["name"].astype(str).str.upper().str.contains(k)]
    print(k, m[["rank", "name", "rating", "outlook", "risk"]].to_string(index=False) if len(m) else "not in list")
allp = p[(p["month"] == last)]
nm = allp["cik"].map(load_names()).astype(str).str.upper()
for k in ["PARAMOUNT", "COTY", "CENTENE"]:
    m = allp[nm.str.contains(k)]
    print("panel", k, m[["cik", "rating", "llm_bus"]].to_string(index=False) if len(m) else "absent from panel")
cur[["rank", "cik", "name", "rating", "outlook", "risk"]].to_csv(ROOT / "results" / "full" / "live_full_rank.csv", index=False)
