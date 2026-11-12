import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import glob
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "data" / "kaggle_upload"
CREDIT = ("2.01", "2.03", "2.05", "2.06")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    done = set()
    for f in glob.glob(str(ROOT / "results" / "scores_shard8-K*.csv")):
        done |= set(pd.read_csv(f, usecols=["key"])["key"])
    expect_1 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    with open(UP / "shard_8-K_1.jsonl", encoding="utf-8") as fh:
        done |= {json.loads(l)["key"] for _, l in zip(range(expect_1), fh)}
    docs, seen = [], set()
    for f in sorted(UP.glob("shard_8-K_[0-9].jsonl")):
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            if d["key"] in seen or d["key"] in done or not any(i in (d.get("items") or "") for i in CREDIT):
                continue
            seen.add(d["key"])
            docs.append(d)
    for old in UP.glob("shard_8-K_c*.jsonl"):
        old.unlink()
    docs.sort(key=lambda d: d["filing_date"])
    for i in range(n):
        part = docs[i::n]
        with open(UP / f"shard_8-K_c{i}.jsonl", "w", encoding="utf-8") as fh:
            for d in part:
                fh.write(json.dumps(d) + "\n")
        print(f"8-K_c{i}: {len(part)} docs (~{len(part) * 6.5 / 3600:.1f} h)")
    print(f"already scored: {len(done)}, remaining credit 8-Ks: {len(docs)}")


if __name__ == "__main__":
    main()

