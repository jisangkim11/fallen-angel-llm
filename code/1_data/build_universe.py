import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
GLEIF = "https://api.gleif.org/api/v1/lei-records"


def gleif_lookup(leis, batch=100):
    out = []
    for i in range(0, len(leis), batch):
        chunk = leis[i:i + batch]
        r = requests.get(GLEIF, params={"filter[lei]": ",".join(chunk), "page[size]": batch}, timeout=60)
        r.raise_for_status()
        for rec in r.json()["data"]:
            ent = rec["attributes"]["entity"]
            out.append({"lei": rec["attributes"]["lei"],
                        "legal_name": ent["legalName"]["name"],
                        "country": ent["legalAddress"]["country"],
                        "hq_country": ent["headquartersAddress"]["country"]})
        time.sleep(0.5)
    return pd.DataFrame(out)


def main():
    df = pd.read_csv(INTERIM / "sp_obligor_ratings.csv", parse_dates=["date"])
    rated = df[df["notch"].notna()]
    bbb = rated[(rated["notch"].between(8, 10)) & (rated["date"] >= "2014-01-01")]
    cand = (bbb.groupby(["obligor", "lei"], dropna=False)
               .agg(sector=("sector", "first"), cik_sp=("cik", "first"))
               .reset_index())
    leis = sorted(cand["lei"].dropna().unique())
    g = gleif_lookup(leis)
    cand = cand.merge(g, on="lei", how="left")
    cand.to_csv(INTERIM / "universe_candidates.csv", index=False)
    us = cand[(cand["country"] == "US") | (cand["hq_country"] == "US")]
    print(f"BBB-band obligors 2014+: {len(cand):,}  with LEI: {cand['lei'].notna().sum():,}  US: {len(us):,}")


if __name__ == "__main__":
    main()
