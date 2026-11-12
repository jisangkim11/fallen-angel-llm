import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
RAW = ROOT / "data" / "raw" / "sec" / "companyfacts"
HEADERS = {"User-Agent": os.environ["SEC_USER_AGENT"]}

TAGS = {
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "lt_debt": ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"],
    "st_debt": ["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings"],
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "op_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "interest": ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt"],
    "da": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "DepreciationAmortizationAndAccretionNet"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
}


def facts(cik):
    p = RAW / f"{cik}.json"
    if not p.exists():
        time.sleep(0.35)
        r = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", headers=HEADERS, timeout=120)
        if r.status_code != 200:
            return None
        p.write_text(r.text, encoding="utf-8")
    return json.loads(p.read_text(encoding="utf-8")).get("facts", {}).get("us-gaap", {})


def pick(gaap, names, accn, end):
    for n in names:
        for unit, vals in gaap.get(n, {}).get("units", {}).items():
            if unit != "USD":
                continue
            hits = [v for v in vals if v.get("accn") == accn and v.get("end") == end]
            full = [v for v in hits if "start" not in v or
                    300 <= (pd.Timestamp(v["end"]) - pd.Timestamp(v["start"])).days <= 380]
            if full:
                return float(full[-1]["val"])
    return np.nan


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(INTERIM / "sample.csv")
    rows = []
    for cik, g in s.groupby("cik"):
        gaap = facts(int(cik))
        for r in g.itertuples(index=False):
            vals = {k: (pick(gaap, v, r.accession, str(r.report_date)[:10]) if gaap else np.nan)
                    for k, v in TAGS.items()}
            rows.append({"cik": cik, "accession": r.accession, **vals})
    f = pd.DataFrame(rows)
    debt = f["lt_debt"].fillna(0) + f["st_debt"].fillna(0)
    debt = debt.where(f[["lt_debt", "st_debt"]].notna().any(axis=1))
    ebitda = f["op_income"] + f["da"].fillna(0)
    f["debt_assets"] = debt / f["assets"]
    f["debt_ebitda"] = (debt / ebitda).where(ebitda > 0, 20).clip(upper=20)
    f["int_cover"] = (f["op_income"] / f["interest"]).clip(-10, 50)
    f["op_margin"] = (f["op_income"] / f["revenue"]).clip(-1, 1)
    f["cash_assets"] = f["cash"] / f["assets"]
    f["cfo_debt"] = (f["cfo"] / debt).clip(-1, 2)
    f["roa"] = (f["net_income"] / f["assets"]).clip(-1, 1)
    f["log_assets"] = np.log(f["assets"])
    f.to_csv(INTERIM / "financials.csv", index=False)
    print(f"rows {len(f):,}; non-missing share:")
    print(f[["debt_assets", "debt_ebitda", "int_cover", "op_margin", "cash_assets", "roa", "log_assets"]].notna().mean().round(2).to_string())


if __name__ == "__main__":
    main()
