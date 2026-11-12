import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from extract_text import (CAPALLOC, ITEM1, ITEM1A, ITEM1A_END, ITEM7, ITEM7_END, ITEM9A, ITEM9A_END,
                          by_keyword, section, words)
from tenk_sections import LIQ_END, LIQ_START, pick_paragraphs, to_text

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
OUT = INTERIM / "doc_text" / "10-K_v2"
HEADERS = {"User-Agent": os.environ["SEC_USER_AGENT"]}

COMP_HEAD = re.compile(r"^\s*(competition|competitive\s+[a-z ,&]{0,40})\s*\.?\s*$", re.I | re.M)
COMP_WORD = re.compile(r"compet", re.I)
MDA_TITLE = re.compile(r"management.s\s+discussion\s+and\s+analysis\s+of\s+financial\s+condition", re.I)
MDA_END = re.compile(r"quantitative\s+and\s+qualitative\s+disclosures?\s+about\s+market\s+risk|item\s*8\b", re.I)
LIQ_WORD = re.compile(r"liquidity|credit\s+facilit|commercial\s+paper|revolv|refinanc|matur|covenant|borrowing", re.I)
MW_POS = re.compile(r"(identified|identify|existence of|there (was|were|is|are))\s+(a\s+)?material\s+weakness"
                    r"|material\s+weakness(es)?\s+(was|were|has been|have been)\s+identified"
                    r"|(were|was)\s+not\s+effective|did\s+not\s+maintain\s+effective", re.I)
MW_NEG = re.compile(r"(no|not\s+identif\w*\s+any)\s+material\s+weakness", re.I)


def competition(text, item1):
    src = item1 or text
    m = COMP_HEAD.search(src)
    if m:
        return words(src[m.end():], 600)
    return by_keyword(src, COMP_WORD, 600)


def mda(text):
    best = section(text, ITEM7, ITEM7_END)
    if len(best.split()) < 300:
        alt = section(text, MDA_TITLE, MDA_END)
        if len(alt.split()) > len(best.split()):
            best = alt
    return best


def liquidity(text, item7):
    liq = section(item7, LIQ_START, LIQ_END) if item7 else ""
    if len(liq.split()) < 150:
        liq = section(text, LIQ_START, LIQ_END)
    if len(liq.split()) < 150:
        liq = by_keyword(item7 or text, LIQ_WORD, 1200)
    return words(liq, 1500)


def controls(text):
    s = section(text, ITEM9A, ITEM9A_END, min_words=20)
    paras = [p for p in s.split("\n") if MW_POS.search(p) and not (MW_NEG.search(p) and not MW_POS.search(p))]
    return words("\n".join(paras), 400)


def tenk_v2(text):
    item1 = section(text, ITEM1, ITEM1A)
    item7 = mda(text)
    return {"competition": competition(text, item1),
            "risk": pick_paragraphs(section(text, ITEM1A, ITEM1A_END), budget=2000),
            "overview": words(item7, 800),
            "liquidity": liquidity(text, item7),
            "capital_allocation": by_keyword(item7 or text, CAPALLOC, 500),
            "controls": controls(text)}


def process(row):
    out = OUT / f"{row.cik}_{row.accession}.json"
    if out.exists():
        return
    url = f"https://www.sec.gov/Archives/edgar/data/{row.cik}/{row.accession.replace('-', '')}/{row.document}"
    try:
        time.sleep(0.6)
        r = requests.get(url, headers=HEADERS, timeout=120)
        r.raise_for_status()
        parts = tenk_v2(to_text(r.text))
        meta = {"cik": row.cik, "form": "10-K", "accession": row.accession,
                "filing_date": str(row.filing_date)[:10], "words": sum(len(v.split()) for v in parts.values())}
        out.write_text(json.dumps({"meta": meta, **parts}), encoding="utf-8")
    except Exception as e:
        print("ERR", row.accession, str(e)[:120], flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(INTERIM / "docs_index.csv")
    rows = list(idx[idx["form"] == "10-K"].itertuples(index=False))
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, _ in enumerate(ex.map(process, rows)):
            if i % 500 == 0:
                print(i, "/", len(rows), flush=True)
    print("done")


if __name__ == "__main__":
    main()
