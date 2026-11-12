import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
from pathlib import Path

import numpy as np
import pandas as pd

from robust import models as md

ROOT = Path(__file__).resolve().parents[2]
md.LADDER["M2L"] = md.LADDER["M2"] + md.LLM
p = pd.read_parquet(ROOT / "data" / "interim" / "panel_final.parquet")
num = p.select_dtypes("number").columns
p[num] = p[num].replace([np.inf, -np.inf], np.nan)
a = p[p["spread"].notna() & p["year"].between(2014, 2024)].reset_index(drop=True)
pred, _ = md.walk_forward(a, "y_fa12", models=("M1", "M2", "M2L"))
pred = pred[pred["year"].between(2016, 2024)]
pred[["cik", "month", "year", "warn", "known", "y_fa12", "M1", "M2", "M2L"]].to_csv(ROOT / "results" / "full" / "final_preds.csv", index=False)
print("ok", len(pred))
