import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
HORIZON = pd.Timedelta(days=365)


def main():
    uni = pd.read_csv(INTERIM / "universe_sec.csv")
    rat = pd.read_csv(INTERIM / "sp_obligor_ratings.csv", parse_dates=["date"])
    tk = pd.read_csv(INTERIM / "tenk_filings.csv", parse_dates=["filing_date", "report_date"])
    data_end = rat["date"].max()

    rat = (rat[rat["obligor"].isin(uni["obligor"])].drop(columns=["cik", "sector"])
              .merge(uni[["obligor", "cik"]], on="obligor"))
    rat = rat.sort_values("date")
    tk = tk.merge(uni[["cik", "obligor", "ticker", "sector"]], on="cik").sort_values("filing_date")

    cur = pd.merge_asof(tk, rat[["cik", "date", "rating", "notch", "outlook", "watch"]],
                        left_on="filing_date", right_on="date", by="cik", direction="backward")
    s = cur[cur["notch"].between(8, 10) & cur["filing_date"].between("2014-01-01", "2025-12-31")].copy()

    rated = rat[rat["notch"].notna()]
    labels, first_dates, downs = [], [], []
    for _, row in s.iterrows():
        w = rated[(rated["cik"] == row["cik"]) & (rated["date"] > row["filing_date"])
                  & (rated["date"] <= row["filing_date"] + HORIZON)]
        hit = w[w["notch"] >= 11]
        labels.append(int(len(hit) > 0))
        first_dates.append(hit["date"].min() if len(hit) else pd.NaT)
        downs.append(int((w["notch"] > row["notch"]).any()))
    s["fallen_angel_12m"] = labels
    s["fa_date"] = first_dates
    s["downgrade_12m"] = downs
    s["censored"] = (s["filing_date"] + HORIZON > data_end) & (s["fallen_angel_12m"] == 0)
    s["neg_outlook"] = s["outlook"].fillna("").str.contains("Negative").astype(int)
    s["watch_neg"] = s["watch"].fillna("").str.contains("Neg").astype(int)
    s = s.drop(columns=["date"]).rename(columns={"rating": "rating_at_filing", "notch": "notch_at_filing"})
    s.to_csv(INTERIM / "sample.csv", index=False)

    print(f"firm-years {len(s):,}, firms {s['cik'].nunique():,}, fallen angels {s['fallen_angel_12m'].sum()}")
    print(s.groupby(s["filing_date"].dt.year)[["fallen_angel_12m", "downgrade_12m"]].agg(["size", "sum"]).to_string())


if __name__ == "__main__":
    main()
