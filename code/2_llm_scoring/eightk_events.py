import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
RAW = ROOT / "data" / "raw" / "sec"
ITEMS = {"1.01": "ek_agreement", "1.03": "ek_bankruptcy", "2.04": "ek_trigger",
         "2.06": "ek_impair", "4.02": "ek_restate", "5.02": "ek_departure"}


def eightks(cik):
    p = RAW / f"submissions_{cik}.json"
    if not p.exists():
        return pd.DataFrame()
    sub = json.loads(p.read_text(encoding="utf-8"))
    blocks = [sub["filings"]["recent"]] + [
        json.loads((RAW / f["name"]).read_text(encoding="utf-8"))
        for f in sub["filings"].get("files", []) if (RAW / f["name"]).exists()]
    rows = []
    for b in blocks:
        for form, fd, items in zip(b["form"], b["filingDate"], b.get("items", [""] * len(b["form"]))):
            if form == "8-K":
                rows.append({"date": fd, "items": items or ""})
    df = pd.DataFrame(rows)
    if len(df):
        df["date"] = pd.to_datetime(df["date"])
    return df


def main():
    s = pd.read_csv(INTERIM / "sample.csv", parse_dates=["filing_date"])
    out = []
    for cik, g in s.groupby("cik"):
        ek = eightks(int(cik))
        for r in g.itertuples(index=False):
            row = {"accession": r.accession}
            w = ek[(ek["date"] < r.filing_date) & (ek["date"] >= r.filing_date - pd.Timedelta(days=365))] if len(ek) else ek
            for code, name in ITEMS.items():
                row[name] = int(w["items"].str.contains(code, regex=False).sum()) if len(w) else 0
            row["ek_total"] = len(w)
            out.append(row)
    f = pd.DataFrame(out)
    f.to_csv(INTERIM / "eightk_features.csv", index=False)
    print(f.drop(columns="accession").mean().round(2).to_string())


if __name__ == "__main__":
    main()
