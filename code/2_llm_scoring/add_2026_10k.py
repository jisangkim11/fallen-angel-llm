import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
RAW = ROOT / "data" / "raw" / "sec"


def main():
    panel = pd.read_parquet(INTERIM / "panel_base.parquet", columns=["cik", "month", "notch", "financial"])
    live = panel[(panel["month"] == "2025-12") & panel["notch"].between(8, 10) & (panel["financial"] == 0)]
    rows = []
    for cik in live["cik"].unique():
        p = RAW / f"submissions_{cik}.json"
        if not p.exists():
            continue
        b = json.loads(p.read_text(encoding="utf-8"))["filings"]["recent"]
        for form, fd, acc, doc in zip(b["form"], b["filingDate"], b["accessionNumber"], b["primaryDocument"]):
            if form == "10-K" and fd >= "2026-01-01":
                rows.append({"cik": cik, "form": form, "filing_date": fd, "accession": acc, "document": doc,
                             "items": "", "notch": float(live.loc[live["cik"] == cik, "notch"].iloc[0])})
    new = pd.DataFrame(rows)
    idx = pd.read_csv(INTERIM / "docs_index.csv")
    idx = pd.concat([idx, new[~new["accession"].isin(idx["accession"])]], ignore_index=True)
    idx.to_csv(INTERIM / "docs_index.csv", index=False)
    print("2026 10-Ks added:", len(new), "for", new["cik"].nunique() if len(new) else 0, "issuers")


if __name__ == "__main__":
    main()
