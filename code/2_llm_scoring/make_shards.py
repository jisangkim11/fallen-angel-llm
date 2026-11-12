import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import glob
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TXT = ROOT / "data" / "interim" / "doc_text"
OUT = ROOT / "data" / "kaggle_upload"
N_SHARDS = {"10-K": 4, "10-Q": 4, "8-K": 4, "424B5": 1}
YEAR = re.compile(r"(?<![\d$.,])(19[5-9]\d|20[0-4]\d)(?![\d,]|\s*(million|billion|thousand))", re.I)
ORDER = {"10-K": 0, "10-Q": 1, "8-K": 2, "424B5": 3}


def mask_years(text, ref_year):
    def rep(m):
        k = int(m.group(1)) - ref_year
        return "[FY]" if k == 0 else f"[FY{k:+d}]"
    return YEAR.sub(rep, text)


def ref_year(form, filing_date):
    y = int(filing_date[:4])
    return y - 1 if form == "10-K" and int(filing_date[5:7]) <= 6 else y


def load(form):
    folder = TXT / ("10-K_v2" if form == "10-K" and (TXT / "10-K_v2").exists() else form)
    for f in glob.glob(str(folder / "*.json")):
        d = json.load(open(f, encoding="utf-8"))
        m = d.pop("meta")
        if sum(len(v.split()) for v in d.values()) < 30:
            continue
        if form == "8-K" and not any(i in (m.get("items") or "") for i in ("2.01", "2.03", "2.05", "2.06")):
            continue
        ry = ref_year(form, m["filing_date"])
        yield {"key": f"{m['cik']}_{m['accession']}", "form": form, "cik": m["cik"],
               "filing_date": m["filing_date"], "items": m.get("items", ""),
               "sections": {k: mask_years(v, ry) for k, v in d.items() if v}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for form in ORDER:
        docs = list(load(form))
        if not docs:
            continue
        docs.sort(key=lambda d: -sum(len(v.split()) for v in d["sections"].values()))
        n = N_SHARDS[form]
        load_w, shards = [0] * n, [[] for _ in range(n)]
        for d in docs:
            i = load_w.index(min(load_w))
            shards[i].append(d)
            load_w[i] += sum(len(v.split()) for v in d["sections"].values())
        for i, s in enumerate(shards):
            s.sort(key=lambda d: d["filing_date"])
            name = f"{form}_{i}"
            with open(OUT / f"shard_{name}.jsonl", "w", encoding="utf-8") as fh:
                for d in s:
                    fh.write(json.dumps(d) + "\n")
            rows += [{"shard": name, "key": d["key"], "form": form} for d in s]
        print(form, "docs", len(docs), "words per shard", load_w)
    pd.DataFrame(rows).to_csv(OUT / "manifest.csv", index=False)


if __name__ == "__main__":
    main()