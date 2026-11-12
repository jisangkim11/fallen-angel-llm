import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as PR, roc_auc_score as AUC

from robust import models as md
from extend_models import boot_p

ROOT = Path(__file__).resolve().parents[2]
md.LADDER["M2L"] = md.LADDER["M2"] + md.LLM


def main():
    p = pd.read_parquet(ROOT / "data" / "interim" / "panel_final.parquet")
    num = p.select_dtypes("number").columns
    p[num] = p[num].replace([np.inf, -np.inf], np.nan)
    a = p[p["spread"].notna() & p["year"].between(2014, 2024)].reset_index(drop=True)
    pred, _ = md.walk_forward(a, "y_fa12", models=("M2", "M2L", "M3", "M4"))
    nk = ~((pred["y_fa12"] == 1) & (pred["known"] == 1))
    rows = []
    for which, mask in [("경고 없는 구간 (주 분석)", nk & (pred["warn"] == 0)), ("S&P 경고 기업", nk & (pred["warn"] == 1)), ("전체", nk)]:
        s = pred[mask].reset_index(drop=True)
        d0, lo, hi, pv = boot_p(s, "y_fa12", "M2", "M2L", n=2000)
        rows.append({"sample": which, "n": len(s), "events": int(s["y_fa12"].sum()),
                     "PR_M2": PR(s["y_fa12"], s["M2"]), "PR_M2L": PR(s["y_fa12"], s["M2L"]),
                     "PR_M3": PR(s["y_fa12"], s["M3"]), "PR_M4": PR(s["y_fa12"], s["M4"]),
                     "AUC_M2": AUC(s["y_fa12"], s["M2"]), "AUC_M2L": AUC(s["y_fa12"], s["M2L"]),
                     "diff": d0, "ci_low": lo, "ci_high": hi, "p_one": pv})
    r = pd.DataFrame(rows)
    r.to_csv(ROOT / "results" / "full" / "ext_m2_llm.csv", index=False)
    pd.set_option("display.width", 240); print(r.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
