import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
import re
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
PANEL = ROOT / "data" / "raw" / "osbap" / "stage1_osbap_0k_volume_2025.parquet"
SPZIP = ROOT / "data" / "raw" / "ratings_sp" / "zip"
NS = {"r": "http://xbrl.sec.gov/ratings/2015-03-31"}
COLS = ["cusip_id", "trd_exctn_dt", "credit_spread", "mod_dur", "prc_bid", "prc_ask",
        "bond_amt_outstanding", "bond_maturity"]


def month_end_panel():
    pf = pq.ParquetFile(PANEL)
    parts = []
    for i in range(pf.num_row_groups):
        df = pf.read_row_group(i, columns=COLS).to_pandas()
        df["cusip_id"] = df["cusip_id"].astype(str)
        df = df[df["trd_exctn_dt"] >= "2012-01-01"]
        df["month"] = df["trd_exctn_dt"].dt.to_period("M")
        df = df.sort_values("trd_exctn_dt").groupby(["cusip_id", "month"]).tail(1)
        parts.append(df)
    m = pd.concat(parts).sort_values("trd_exctn_dt").groupby(["cusip_id", "month"]).tail(1)
    m = m.sort_values(["cusip_id", "month"]).reset_index(drop=True)
    g = m.groupby("cusip_id")
    m["prev_month"] = g["month"].shift()
    m["s_prev"], m["d_prev"] = g["credit_spread"].shift(), g["mod_dur"].shift()
    ok = (m["month"] - m["prev_month"]).apply(lambda x: getattr(x, "n", None)) == 1
    m["credit_ret"] = (m["s_prev"] / 12 - m["d_prev"] * (m["credit_spread"] - m["s_prev"])).where(ok)
    m["half_spread_cost"] = ((m["prc_ask"] - m["prc_bid"]) / 2 / ((m["prc_ask"] + m["prc_bid"]) / 2)).clip(lower=0)
    return m.drop(columns=["prev_month", "s_prev", "d_prev"])


def bond_issuer_link():
    rows = []
    for zp in sorted(SPZIP.glob("SP-*.zip")):
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                if not n.endswith("-ISSUER.xml"):
                    continue
                root = ET.fromstring(z.read(n))
                isd = root.find(".//r:ISD", NS)
                if isd is None:
                    continue
                name = (isd.findtext("r:ISSNAME", "", NS) or "").strip()
                lei = (isd.findtext("r:LEI", "", NS) or "").strip()
                for ind in isd.findall("r:IND", NS):
                    cusip = (ind.findtext("r:CUSIP", "", NS) or "").strip()
                    if re.fullmatch(r"[0-9A-Z]{9}", cusip):
                        rows.append({"cusip_id": cusip, "issuer": name, "issuer_lei": lei})
    return pd.DataFrame(rows).drop_duplicates("cusip_id")


def main():
    link = bond_issuer_link()
    link.to_csv(INTERIM / "bond_issuer_link.csv", index=False)
    print(f"S&P-rated bonds with CUSIP: {len(link):,}")
    m = month_end_panel()
    m["month"] = m["month"].astype(str)
    m.to_parquet(INTERIM / "bond_monthly.parquet", index=False)
    print(f"bond-months 2012+: {len(m):,}, bonds: {m['cusip_id'].nunique():,}, "
          f"linked to S&P issuers: {m['cusip_id'].isin(link['cusip_id']).mean():.0%}")


if __name__ == "__main__":
    main()
