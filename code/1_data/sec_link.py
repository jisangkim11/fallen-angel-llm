import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
RAW = ROOT / "data" / "raw" / "sec"
HEADERS = {"User-Agent": os.environ["SEC_USER_AGENT"]}

SUFFIX = r"\b(THE|INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|LLC|L P|LP|PLC|N V|NV|S A|SA|AG|HOLDINGS?|GROUP)\b"


def norm(name):
    s = str(name).upper().replace("&", " AND ")
    s = re.sub(r"/[A-Z]{2,3}/?", " ", s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(SUFFIX, " ", s)
    return re.sub(r"\s+", " ", s).strip()


def get(url):
    for _ in range(3):
        r = requests.get(url, headers=HEADERS, timeout=60)
        if r.status_code == 200:
            return r
        time.sleep(2)
    r.raise_for_status()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    tick = get("https://www.sec.gov/files/company_tickers.json").json()
    sec = pd.DataFrame(tick.values()).rename(columns={"cik_str": "cik", "title": "sec_name"})
    lookup = RAW / "cik-lookup-data.txt"
    if not lookup.exists():
        lookup.write_bytes(get("https://www.sec.gov/Archives/edgar/cik-lookup-data.txt").content)
    allf = pd.read_csv(lookup, sep=r":(?=\d{10}:$)", engine="python", header=None,
                       names=["sec_name", "cik"], encoding="latin-1", on_bad_lines="skip")
    allf["cik"] = allf["cik"].str.rstrip(":").astype(int)
    allf["ticker"] = ""
    sec = pd.concat([sec, allf], ignore_index=True)
    sec["key"] = sec["sec_name"].map(norm)
    sec = sec.drop_duplicates("key")

    cand = pd.read_csv(INTERIM / "universe_candidates.csv")
    us = cand[(cand["country"] == "US") | (cand["hq_country"] == "US")].copy()
    us["key_sp"] = us["obligor"].map(norm)
    us["key_lei"] = us["legal_name"].map(norm)
    m1 = us.merge(sec, left_on="key_sp", right_on="key", how="left")
    miss = m1["cik"].isna()
    m2 = us[miss.values].merge(sec, left_on="key_lei", right_on="key", how="left")
    m1.loc[miss, ["cik", "ticker", "sec_name"]] = m2[["cik", "ticker", "sec_name"]].values
    linked = m1.dropna(subset=["cik"]).copy()
    linked["cik"] = linked["cik"].astype(int)
    linked = linked.sort_values("obligor", key=lambda s: s.str.len()).drop_duplicates("cik")
    linked[["obligor", "lei", "sector", "cik", "ticker", "sec_name"]].to_csv(INTERIM / "universe_sec.csv", index=False)
    print(f"US obligors {len(us):,} -> linked to listed SEC filers {len(linked):,}")

    rows = []
    for i, cik in enumerate(linked["cik"]):
        p = RAW / f"submissions_{cik}.json"
        if not p.exists():
            p.write_text(get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").text, encoding="utf-8")
            time.sleep(0.12)
        sub = json.loads(p.read_text(encoding="utf-8"))
        blocks = [sub["filings"]["recent"]]
        for f in sub["filings"].get("files", []):
            q = RAW / f["name"]
            if not q.exists():
                q.write_text(get("https://data.sec.gov/submissions/" + f["name"]).text, encoding="utf-8")
                time.sleep(0.12)
            blocks.append(json.loads(q.read_text(encoding="utf-8")))
        for b in blocks:
            for form, fd, rd, acc, doc in zip(b["form"], b["filingDate"], b["reportDate"],
                                              b["accessionNumber"], b["primaryDocument"]):
                if form in ("10-K", "10-K405") and "2013-01-01" <= fd <= "2026-12-31":
                    rows.append({"cik": cik, "filing_date": fd, "report_date": rd,
                                 "accession": acc, "document": doc})
        if i % 100 == 0:
            print(i, "companies processed")
    tk = pd.DataFrame(rows).drop_duplicates("accession")
    tk.to_csv(INTERIM / "tenk_filings.csv", index=False)
    print(f"10-K filings 2013-2026: {len(tk):,} for {tk['cik'].nunique():,} companies")


if __name__ == "__main__":
    main()
