import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import html as htmllib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
OUT = INTERIM / "tenk_text"
HEADERS = {"User-Agent": os.environ["SEC_USER_AGENT"]}

KEYWORDS = re.compile(
    r"debt|indebted|leverage|liquidity|covenant|credit rating|downgrad|refinanc|borrow|"
    r"default|matur|impair|going concern|cash flow|capital market|interest expense|"
    r"collateral|credit facilit|secured|notes due|bankrupt|restructur|pension", re.I)
RISK_START = re.compile(r"item\s*1a\.?\s*[\-–—:.]?\s*risk\s+factors", re.I)
RISK_END = re.compile(r"item\s*1b\.?\s*[\-–—:.]?\s*unresolved|item\s*2\.?\s*[\-–—:.]?\s*properties", re.I)
LIQ_START = re.compile(r"liquidity\s+and\s+capital\s+resources", re.I)
LIQ_END = re.compile(r"critical\s+accounting|item\s*7a|quantitative\s+and\s+qualitative|"
                     r"recent(ly)?\s+(issued|adopted)\s+accounting|new\s+accounting\s+(standards|pronouncements)", re.I)


def to_text(raw):
    s = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    s = re.sub(r"(?i)</(p|div|tr|li|h\d|table)>|<br\s*/?>", "\n", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = htmllib.unescape(s).replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return re.sub(r"\n\s*\n+", "\n", s)


def longest_section(text, start, end, min_words=150):
    best = ""
    for m in start.finditer(text):
        e = end.search(text, m.end())
        chunk = text[m.end(): e.start() if e else m.end() + 200_000]
        if len(chunk.split()) > len(best.split()):
            best = chunk
    return best if len(best.split()) >= min_words else ""


def pick_paragraphs(section, budget=2500):
    paras = [p.strip() for p in section.split("\n") if len(p.split()) >= 25]
    scored = sorted(range(len(paras)),
                    key=lambda i: -len(KEYWORDS.findall(paras[i])) / max(len(paras[i].split()), 1) ** 0.5)
    keep, used = set(), 0
    for i in scored:
        n = len(paras[i].split())
        if used + n > budget:
            continue
        keep.add(i)
        used += n
    return "\n".join(paras[i] for i in sorted(keep))


def first_words(section, budget=2500):
    return " ".join(section.split()[:budget])


def process(row):
    out = OUT / f"{row.cik}_{row.accession}.json"
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))["meta"]
    url = (f"https://www.sec.gov/Archives/edgar/data/{row.cik}/"
           f"{row.accession.replace('-', '')}/{row.document}")
    meta = {"cik": row.cik, "accession": row.accession, "filing_date": row.filing_date}
    try:
        time.sleep(0.6)
        r = requests.get(url, headers=HEADERS, timeout=120)
        r.raise_for_status()
        text = to_text(r.text)
        risk = pick_paragraphs(longest_section(text, RISK_START, RISK_END))
        liq = first_words(longest_section(text, LIQ_START, LIQ_END))
        meta.update(risk_words=len(risk.split()), liq_words=len(liq.split()), error="")
        out.write_text(json.dumps({"meta": meta, "risk": risk, "liquidity": liq}), encoding="utf-8")
    except Exception as e:
        meta.update(risk_words=0, liq_words=0, error=str(e)[:200])
    return meta


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(INTERIM / "sample.csv")
    rows = list(s[["cik", "accession", "document", "filing_date"]].itertuples(index=False))
    with ThreadPoolExecutor(max_workers=6) as ex:
        metas = []
        for i, m in enumerate(ex.map(process, rows)):
            metas.append(m)
            if i % 200 == 0:
                print(i, "filings done", flush=True)
    idx = pd.DataFrame(metas)
    idx.to_csv(INTERIM / "tenk_text_index.csv", index=False)
    ok = (idx["risk_words"] > 0) | (idx["liq_words"] > 0)
    print(f"filings {len(idx):,}, with text {ok.sum():,}, "
          f"median words risk {idx['risk_words'].median():.0f} liq {idx['liq_words'].median():.0f}")


if __name__ == "__main__":
    main()
