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

from tenk_sections import KEYWORDS, LIQ_END, LIQ_START, longest_section, pick_paragraphs, to_text

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
OUT = INTERIM / "doc_text"
HEADERS = {"User-Agent": os.environ["SEC_USER_AGENT"]}

D = r"\.?\s*[\-–—:.]?\s*"
RX = lambda s: re.compile(s, re.I)
ITEM1 = RX(rf"item\s*1{D}business")
ITEM1A = RX(rf"item\s*1a{D}risk\s+factors")
ITEM1A_END = RX(rf"item\s*1b{D}unresolved|item\s*1c{D}cybersecurity|item\s*2{D}properties")
ITEM7 = RX(rf"item\s*7{D}management.s\s+discussion")
ITEM7_END = RX(rf"item\s*7a{D}quantitative|item\s*8{D}financial\s+statements")
ITEM9A = RX(rf"item\s*9a{D}controls")
ITEM9A_END = RX(rf"item\s*9b{D}other|item\s*10{D}directors")
Q_MDA = RX(rf"item\s*2{D}management.s\s+discussion")
Q_MDA_END = RX(rf"item\s*3{D}quantitative|item\s*4{D}controls")
Q_RISK = RX(rf"item\s*1a{D}risk\s+factors")
Q_RISK_END = RX(rf"item\s*2{D}unregistered|item\s*3{D}defaults|item\s*5{D}other|item\s*6{D}exhibits")
COMPETITION = RX(r"\n\s*competition\s*\n")
CAPALLOC = RX(r"repurchase|buyback|dividend|acquisition|leverage target|investment.grade|credit rating")
USE_START = RX(r"\n\s*use\s+of\s+proceeds\s*\n")
USE_END = RX(r"\n\s*(capitalization|description\s+of|dilution|ratio\s+of\s+earnings|underwriting|risk\s+factors)")


def words(s, n):
    return " ".join(s.split()[:n])


def section(text, start, end, min_words=80):
    return longest_section(text, start, end, min_words=min_words)


def by_keyword(sec, rx, budget):
    paras = [p.strip() for p in sec.split("\n") if len(p.split()) >= 20 and rx.search(p)]
    out, used = [], 0
    for p in paras:
        n = len(p.split())
        if used + n > budget:
            break
        out.append(p)
        used += n
    return "\n".join(out)


def tenk(text):
    item1 = section(text, ITEM1, ITEM1A)
    m = COMPETITION.search(item1)
    item7 = section(text, ITEM7, ITEM7_END)
    item9a = section(text, ITEM9A, ITEM9A_END, min_words=20)
    return {
        "competition": words(item1[m.end():], 600) if m else "",
        "risk": pick_paragraphs(section(text, ITEM1A, ITEM1A_END), budget=2000),
        "overview": words(item7, 800),
        "liquidity": words(section(item7 or text, LIQ_START, LIQ_END), 1500),
        "capital_allocation": by_keyword(item7, CAPALLOC, 500),
        "controls": words(item9a, 400) if re.search(r"material weakness", item9a, re.I) else "",
    }


def tenq(text):
    mda = section(text, Q_MDA, Q_MDA_END)
    return {
        "overview": words(mda, 600),
        "liquidity": words(section(mda or text, LIQ_START, LIQ_END), 1200),
        "risk_changes": words(section(text, Q_RISK, Q_RISK_END, min_words=10), 400),
    }


def eightk(text):
    body = re.split(r"\n\s*SIGNATURES?\s*\n", text, maxsplit=1, flags=re.I)[0]
    return {"event": words(body, 1500)}


def prospectus(text):
    return {"use_of_proceeds": words(section(text, USE_START, USE_END, min_words=20), 400)}


EXTRACT = {"10-K": tenk, "10-Q": tenq, "8-K": eightk, "424B5": prospectus}


def process(row):
    folder = OUT / row.form.replace("/", "_")
    out = folder / f"{row.cik}_{row.accession}.json"
    meta = {"cik": row.cik, "form": row.form, "accession": row.accession,
            "filing_date": str(row.filing_date)[:10], "items": row.items}
    if out.exists():
        return {**meta, "words": json.loads(out.read_text(encoding="utf-8"))["meta"]["words"], "error": ""}
    url = (f"https://www.sec.gov/Archives/edgar/data/{row.cik}/"
           f"{row.accession.replace('-', '')}/{row.document}")
    try:
        time.sleep(0.6)
        r = requests.get(url, headers=HEADERS, timeout=120)
        r.raise_for_status()
        parts = EXTRACT[row.form](to_text(r.text))
        meta["words"] = sum(len(v.split()) for v in parts.values())
        folder.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"meta": meta, **parts}), encoding="utf-8")
        return {**meta, "error": ""}
    except Exception as e:
        return {**meta, "words": 0, "error": str(e)[:200]}


def main():
    idx = pd.read_csv(INTERIM / "docs_index.csv")
    idx["items"] = idx["items"].fillna("")
    order = {"10-K": 0, "10-Q": 1, "8-K": 2, "424B5": 3}
    idx = idx.sort_values(by="form", key=lambda s: s.map(order))
    rows = list(idx.itertuples(index=False))
    metas = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, m in enumerate(ex.map(process, rows)):
            metas.append(m)
            if i % 500 == 0:
                print(i, "/", len(rows), flush=True)
    res = pd.DataFrame(metas)
    res.to_csv(INTERIM / "doc_text_index.csv", index=False)
    print(res.groupby("form").agg(n=("words", "size"), ok=("words", lambda w: (w > 0).sum()),
                                   median_words=("words", "median")).to_string())


if __name__ == "__main__":
    main()
