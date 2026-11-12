import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import re
import sys
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "ratings_sp" / "zip"
OUT = ROOT / "data" / "interim"
NS = {"r": "http://xbrl.sec.gov/ratings/2015-03-31"}

SCALE = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-",
         "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "SD", "D"]
NOTCH = {r: i + 1 for i, r in enumerate(SCALE)}


def text(el, tag):
    x = el.find(f"r:{tag}", NS)
    return x.text.strip() if x is not None and x.text else ""


def parse_obligor(xml_bytes, source):
    root = ET.fromstring(xml_bytes)
    od = root.find(".//r:OD", NS)
    if od is None:
        return []
    info = {
        "obligor": text(od, "OBNAME"),
        "lei": text(od, "LEI"),
        "cik": text(od, "CIK"),
        "sector": text(od, "OSC"),
        "source": source,
    }
    rows = []
    for ord_ in od.findall("r:ORD", NS):
        if text(ord_, "RT") != "Issuer Credit Rating":
            continue
        if text(ord_, "RST") not in ("Foreign Currency LT", "Local Currency LT"):
            continue
        rows.append({**info,
                     "date": text(ord_, "RAD"),
                     "rating": text(ord_, "R"),
                     "action": text(ord_, "RAC") or text(ord_, "OAN"),
                     "watch": text(ord_, "WST"),
                     "outlook": text(ord_, "ROL"),
                     "scale": text(ord_, "RST")})
    return rows


def main():
    rows = []
    for zp in sorted(RAW.glob("SP-*.zip")):
        source = re.sub(r"SP-(\w+)-.*", r"\1", zp.name)
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                if n.endswith("-OBLIGOR.xml"):
                    rows.extend(parse_obligor(z.read(n), source))
        print(zp.name, "rows so far:", len(rows), file=sys.stderr)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["notch"] = df["rating"].map(NOTCH)
    df = (df.sort_values(["obligor", "date", "scale"])
            .drop_duplicates(["obligor", "lei", "date"], keep="first")
            .reset_index(drop=True))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "sp_obligor_ratings.csv", index=False)
    print(f"saved {len(df):,} actions for {df['obligor'].nunique():,} obligors", file=sys.stderr)


if __name__ == "__main__":
    main()
