import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from extract_text import INTERIM, process

CREDIT_ITEMS = ("2.01", "2.03", "2.05", "2.06")


def main():
    idx = pd.read_csv(INTERIM / "docs_index.csv")
    idx["items"] = idx["items"].fillna("")
    k = idx[(idx["form"] == "8-K") & idx["items"].apply(lambda s: any(i in s for i in CREDIT_ITEMS))]
    rows = list(k.itertuples(index=False))
    print("credit 8-Ks to extract:", len(rows), flush=True)
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, _ in enumerate(ex.map(process, rows)):
            if i % 500 == 0:
                print(i, "/", len(rows), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
