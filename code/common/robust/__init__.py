"""Robust modeling and statistical testing for the fallen-angel early-warning study.

Panel schema (one row per issuer-month; built by the panel builder):
  cik, month ("YYYY-MM"), year
  y_fa12      1 if S&P rating falls to BB+ or below within 12 months
  y_dg12      1 if any downgrade within 12 months
  fa_month    month of the fallen-angel event this row points to (NaN if none)
  censored    1 if the 12-month window passes the end of rating data
  known       1 if the LLM memory test says the LLM already knew this event
  notch_pos   1 = BBB+, 2 = BBB, 3 = BBB-
  warn        1 if negative outlook or CreditWatch negative
  rel_spread  issuer spread minus same-month BBB median
  size_log, ind_dg_rate, lev_ind_adj, lev_chg, mat_ratio, payout_assets
  llm_bus, llm_fin, llm_liq, llm_pol   LLM pillar scores (within-month percentile)
  llm_age     months since the latest periodic filing
  lm_neg, lm_unc                         dictionary text measures (Loughran-McDonald)
"""
