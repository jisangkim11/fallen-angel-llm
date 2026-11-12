import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from xbrl_financials import RAW, facts, pick

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"

TAGS = {
    "assets": ["Assets"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "lt_debt": ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"],
    "st_debt": ["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings"],
    "due_12m": ["LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths", "LongTermDebtCurrent", "DebtCurrent"],
    "op_income": ["OperatingIncomeLoss"],
    "da": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "DepreciationAmortizationAndAccretionNet"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "buyback": ["PaymentsForRepurchaseOfCommonStock"],
    "dividends": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
    "acquisitions": ["PaymentsToAcquireBusinessesNetOfCashAcquired", "PaymentsToAcquireBusinessesGross"],
}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(INTERIM / "docs_index.csv")
    tk = pd.read_csv(INTERIM / "tenk_filings.csv")[["cik", "accession", "report_date"]]
    k = idx[idx["form"] == "10-K"].merge(tk, on=["cik", "accession"], how="left")
    rows = []
    for cik, g in k.groupby("cik"):
        gaap = facts(int(cik))
        for r in g.itertuples(index=False):
            end = str(r.report_date)[:10] if isinstance(r.report_date, str) else None
            vals = {t: (pick(gaap, names, r.accession, end) if gaap and end else np.nan) for t, names in TAGS.items()}
            rows.append({"cik": cik, "accession": r.accession, "filing_date": r.filing_date, "report_date": end, **vals})
    f = pd.DataFrame(rows)
    debt = f[["lt_debt", "st_debt"]].fillna(0).sum(axis=1).where(f[["lt_debt", "st_debt"]].notna().any(axis=1))
    ebitda = f["op_income"] + f["da"].fillna(0)
    f["size_log"] = np.log(f["assets"])
    f["lev"] = (debt / ebitda).where(ebitda > 0, 20).clip(upper=20)
    f["mat_ratio"] = (f["due_12m"] / (f["cash"].fillna(0) + f["cfo"].clip(lower=0).fillna(0)).replace(0, np.nan)).clip(upper=10)
    f["payout_assets"] = f[["buyback", "dividends", "acquisitions"]].fillna(0).sum(axis=1) / f["assets"]
    f = f.sort_values(["cik", "filing_date"])
    f["lev_chg"] = f.groupby("cik")["lev"].diff()
    f["fy"] = f["report_date"].str[:4]
    sic = pd.read_parquet(INTERIM / "panel_base.parquet", columns=["cik", "sic2"]).drop_duplicates("cik")
    f = f.merge(sic, on="cik", how="left")
    f["lev_ind_adj"] = f["lev"] - f.groupby(["sic2", "fy"])["lev"].transform("median")
    f.to_csv(INTERIM / "fin_xbrl.csv", index=False)
    print(f"10-Ks {len(f):,}; non-missing:")
    print(f[["size_log", "lev", "lev_ind_adj", "lev_chg", "mat_ratio", "payout_assets"]].notna().mean().round(2).to_string())


if __name__ == "__main__":
    main()
