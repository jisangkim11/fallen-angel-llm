import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import glob
import json
from pathlib import Path

import pandas as pd
import pysentiment2 as ps

ROOT = Path(__file__).resolve().parents[2]
TXT = ROOT / "data" / "interim" / "doc_text"


def main():
    lm = ps.LM()
    folder = TXT / ("10-K_v2" if (TXT / "10-K_v2").exists() else "10-K")
    rows = []
    for f in glob.glob(str(folder / "*.json")):
        d = json.load(open(f, encoding="utf-8"))
        m = d.pop("meta")
        tokens = lm.tokenize(" ".join(v for v in d.values() if v))
        s = lm.get_score(tokens)
        n = max(len(tokens), 1)
        rows.append({"key": f"{m['cik']}_{m['accession']}", "cik": m["cik"], "filing_date": m["filing_date"],
                     "lm_neg": 1000 * s["Negative"] / n, "lm_pol": float(s["Polarity"])})
    pd.DataFrame(rows).to_csv(ROOT / "data" / "interim" / "lm_dict.csv", index=False)
    print("10-K dictionary measures:", len(rows))


if __name__ == "__main__":
    main()
