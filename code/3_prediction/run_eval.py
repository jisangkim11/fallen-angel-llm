import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import json
import sys
from pathlib import Path

import pandas as pd

from robust import stats_tests as st
from robust.evaluate import run_all
from robust.models import walk_forward

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
RESULTS = ROOT / "results"


def main():
    quick = "--quick" in sys.argv
    panel = pd.read_parquet(INTERIM / "panel_final.parquet")
    panel = panel[panel["spread"].notna() & panel["year"].between(2014, 2024)].reset_index(drop=True)
    if "llm_bus" not in panel:
        raise SystemExit("LLM scores missing: put scores_shard10-K_*.csv into results/ and rerun build_panel_final.py")
    if "--llm-only" in sys.argv:
        panel = panel[panel["llm_bus"].notna()].reset_index(drop=True)
        print("interim run on rows with LLM scores:", len(panel))

    pred, _ = walk_forward(panel, "y_fa12", models=("M3", "M4"))
    port = pred[["cik", "month", "credit_ret", "amt", "half_cost", "rel_spread", "M3", "M4"]].copy()
    port["is_bbb"], port["months_since_fa"] = 1, float("nan")

    rep = run_all(panel, port, n_boot=300 if quick else 2000, n_perm=100 if quick else 1000)
    RESULTS.mkdir(exist_ok=True)
    for k, v in rep.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(RESULTS / f"eval_{k}.csv", index=False)
    summary = {k: v for k, v in rep.items() if not isinstance(v, pd.DataFrame)}
    (RESULTS / "eval_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print("PRIMARY:", rep["primary"])
    print(rep["metrics_primary"].round(4).to_string(index=False))
    print(rep["holm"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
