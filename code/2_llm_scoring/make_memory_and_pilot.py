import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import glob
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from make_shards import mask_years, ref_year

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
OUT = ROOT / "data" / "kaggle_upload"
RNG = np.random.default_rng(7)


def memory_questions(panel, names):
    p = panel[(panel["financial"] == 0) & (panel["year"] >= 2013)]
    ev = p[p["fa_month"].notna()].drop_duplicates(["cik", "fa_month"])[["cik", "fa_month"]]
    ev["year"] = ev["fa_month"].str[:4].astype(int)
    rows = [{"cik": r.cik, "year": r.year, "type": "event", "true_month": r.fa_month} for r in ev.itertuples()]
    for r in ev.itertuples():
        pool = p[(p["year"] == r.year) & (p["cik"] != r.cik)]
        bad = set(p.loc[(p["year"] == r.year) & (p["y_fa12"] == 1), "cik"])
        pool = [c for c in pool["cik"].unique() if c not in bad]
        for c in RNG.choice(pool, size=min(2, len(pool)), replace=False):
            rows.append({"cik": int(c), "year": r.year, "type": "control", "true_month": ""})
    q = pd.DataFrame(rows).merge(names, on="cik", how="left")
    q.to_csv(OUT / "memory_questions.csv", index=False)
    print("memory questions:", q["type"].value_counts().to_dict())


def pilot(panel, names):
    folder = INTERIM / "doc_text" / ("10-K_v2" if (INTERIM / "doc_text" / "10-K_v2").exists() else "10-K")
    docs = {}
    for f in glob.glob(str(folder / "*.json")):
        d = json.load(open(f, encoding="utf-8"))
        docs[f"{d['meta']['cik']}_{d['meta']['accession']}"] = d
    p = panel[(panel["financial"] == 0)][["cik", "month", "y_fa12"]]
    keys = []
    meta = pd.DataFrame([{"key": k, "cik": d["meta"]["cik"], "month": d["meta"]["filing_date"][:7],
                          "words": d["meta"].get("words", 0)} for k, d in docs.items()])
    meta = meta[meta["words"] > 2500].merge(p, on=["cik", "month"], how="inner")
    pos = meta[meta["y_fa12"] == 1].drop_duplicates("cik")
    neg = meta[meta["y_fa12"] == 0].drop_duplicates("cik")
    pick = pd.concat([pos.sample(min(16, len(pos)), random_state=1), neg.sample(16, random_state=1)]).sample(frac=1, random_state=2)
    with open(OUT / "pilot_docs.jsonl", "w", encoding="utf-8") as fh, \
         open(OUT / "pilot_reading_pack.html", "w", encoding="utf-8") as hp:
        hp.write("<html><meta charset='utf-8'><body style='font-family:Malgun Gothic;max-width:900px;margin:auto'>")
        for i, r in enumerate(pick.itertuples(), 1):
            d = dict(docs[r.key])
            m = d.pop("meta")
            ry = ref_year("10-K", m["filing_date"])
            secs = {k: mask_years(v, ry) for k, v in d.items() if v}
            fh.write(json.dumps({"key": r.key, "form": "10-K", "cik": m["cik"], "filing_date": m["filing_date"],
                                 "items": "", "sections": secs}) + "\n")
            hp.write(f"<h2>Doc {i} · {html.escape(r.key)}</h2>")
            for k, v in secs.items():
                hp.write(f"<h4>{k}</h4><p style='white-space:pre-wrap'>{html.escape(' '.join(v.split()[:900]))}</p>")
            keys.append(r.key)
        hp.write("</body></html>")
    lab = pd.DataFrame({"doc_no": range(1, len(keys) + 1), "key": keys, "scorer": "",
                        "business": "", "financial": "", "liquidity": "", "policy": "", "note": ""})
    lab.to_csv(OUT / "human_labels.csv", index=False, encoding="utf-8-sig")
    print("pilot docs:", len(keys), "(outcome hidden from scorers)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(INTERIM / "panel_base.parquet")
    uni = pd.read_csv(INTERIM / "universe_sec.csv")
    names = uni[["cik", "obligor"]].rename(columns={"obligor": "company"}).drop_duplicates("cik")
    memory_questions(panel, names)
    pilot(panel, names)


if __name__ == "__main__":
    main()
