import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
RAW = ROOT / "data" / "raw" / "sec"
EK_ITEMS = ("2.01", "2.03", "2.05", "2.06", "5.02")


def filings(cik):
    p = RAW / f"submissions_{cik}.json"
    sub = json.loads(p.read_text(encoding="utf-8"))
    blocks = [sub["filings"]["recent"]] + [
        json.loads((RAW / f["name"]).read_text(encoding="utf-8"))
        for f in sub["filings"].get("files", []) if (RAW / f["name"]).exists()]
    for b in blocks:
        n = len(b["form"])
        for form, fd, acc, doc, items in zip(b["form"], b["filingDate"], b["accessionNumber"],
                                             b["primaryDocument"], b.get("items", [""] * n)):
            yield form, fd, acc, doc, items or ""


def main():
    uni = pd.read_csv(INTERIM / "universe_sec.csv")
    rat = pd.read_csv(INTERIM / "sp_obligor_ratings.csv", parse_dates=["date"])
    rat = rat[rat["notch"].notna()].drop(columns=["cik"]).merge(uni[["obligor", "cik"]], on="obligor")
    rat = rat.sort_values("date")[["cik", "date", "notch"]]

    rows = []
    for cik in uni["cik"]:
        if not (RAW / f"submissions_{cik}.json").exists():
            continue
        for form, fd, acc, doc, items in filings(cik):
            if not ("2013-01-01" <= fd <= "2025-12-31"):
                continue
            if form in ("10-K", "10-Q", "424B5") or (form == "8-K" and any(i in items for i in EK_ITEMS)):
                rows.append({"cik": cik, "form": form, "filing_date": fd, "accession": acc,
                             "document": doc, "items": items})
    d = pd.DataFrame(rows).drop_duplicates("accession")
    d["filing_date"] = pd.to_datetime(d["filing_date"])
    d = pd.merge_asof(d.sort_values("filing_date"), rat, left_on="filing_date", right_on="date",
                      by="cik", direction="backward")
    d = d[d["notch"].between(7, 10)].drop(columns=["date"])
    d.to_csv(INTERIM / "docs_index.csv", index=False)
    print(d["form"].value_counts().to_string())
    print("firms:", d["cik"].nunique())


if __name__ == "__main__":
    main()
