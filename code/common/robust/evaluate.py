import numpy as np
import pandas as pd

from . import stats_tests as st
from .models import LLM, predict_with, walk_forward

PRIMARY = {"y": "y_fa12", "base": "M3", "new": "M4", "metric": "PR-AUC",
           "sample": "warn == 0, LLM-unknown events only, test years 2016-2024"}


def primary_sample(pred):
    s = pred[(pred["warn"] == 0) & ~((pred["y_fa12"] == 1) & (pred["known"] == 1))]
    return s.reset_index(drop=True)


def metric_table(df, y, models):
    rows = []
    for m in models:
        rows.append({"model": m, "events": int(df[y].sum()), "n": len(df),
                     "PR_AUC": st.pr_auc(df[y].values, df[m].values),
                     "AUC": st.roc_auc(df[y].values, df[m].values),
                     "AUC_within_month": st.within_month_auc(df, y, m),
                     "top10_capture": st.top_capture(df[y].values, df[m].values),
                     "Brier": float(np.mean((df[y] - df[m]) ** 2))})
    return pd.DataFrame(rows)


def event_table(pred, score="M4", q=0.10, window=24):
    p = pred.copy()
    p["alarm"] = p.groupby("month")[score].transform(lambda s: s >= s.quantile(1 - q)).astype(int)
    p["mi"] = pd.PeriodIndex(p["month"], freq="M").astype(int)
    ev = p.loc[p["fa_month"].notna(), ["cik", "fa_month", "known"]].drop_duplicates(["cik", "fa_month"])
    rows = []
    for r in ev.itertuples(index=False):
        fa = pd.Period(r.fa_month, freq="M").ordinal
        h = p[(p["cik"] == r.cik) & (p["mi"] < fa) & (p["mi"] >= fa - window)]
        first = lambda col: int(fa - h.loc[h[col] == 1, "mi"].min()) if (h[col] == 1).any() else 0
        within = lambda col: int(((h[col] == 1) & (h["mi"] >= fa - 12)).any())
        rows.append({"cik": r.cik, "fa_month": r.fa_month, "known": r.known,
                     "lead_model": first("alarm"), "lead_sp": first("warn"),
                     "hit_model_12m": within("alarm"), "hit_sp_12m": within("warn")})
    return pd.DataFrame(rows)


def run_all(panel, port=None, n_boot=2000, n_perm=1000):
    report = {"primary_spec": PRIMARY}

    pred, fitted = walk_forward(panel, "y_fa12")
    pred_dg, _ = walk_forward(panel, "y_dg12", models=("M3", "M4"))
    pred_cb, _ = walk_forward(panel, "y_fa12", models=("M3", "M4"), boost=True)
    prim = primary_sample(pred)

    boot = st.cluster_bootstrap_diff(prim, "y_fa12", "M3", "M4", st.pr_auc, n=n_boot)
    report["primary"] = {**boot, "decision": "LLM adds information" if boot["p_one_sided"] < 0.05 else "not shown"}
    report["metrics_primary"] = metric_table(prim, "y_fa12", ["M1", "M2", "M3", "M3b", "M4"])

    perm = st.permutation_test(prim, "y_fa12", predict_with(fitted, "M4"), [c for c in LLM if c in prim], "M3", n=n_perm)
    dl = st.delong_test(prim["y_fa12"].values, prim["M3"].values, prim["M4"].values)
    dm = st.diebold_mariano(pred, "y_fa12", "M3", "M4")
    m3b = st.cluster_bootstrap_diff(prim, "y_fa12", "M3b", "M4", st.pr_auc, n=n_boot)
    nest = st.nested_logit_tests(panel[panel["censored"] == 0], "y_fa12",
                                 ["notch_pos", "warn", "rel_spread", "size_log", "ind_dg_rate", "lev_ind_adj",
                                  "lev_chg", "mat_ratio", "payout_assets"], ["llm_bus", "llm_fin", "llm_liq", "llm_pol"])
    dg = st.cluster_bootstrap_diff(pred_dg[pred_dg["warn"] == 0].reset_index(drop=True), "y_dg12", "M3", "M4", st.pr_auc, n=n_boot)

    ev = event_table(pred)
    wil = st.wilcoxon_paired(ev["lead_model"], ev["lead_sp"])
    alert_rate = pred["warn"].mean()
    ev_eq = event_table(pred, q=max(alert_rate, 0.01))
    mcn = st.mcnemar_paired(ev_eq["hit_model_12m"], ev_eq["hit_sp_12m"])
    known_gap = (ev.loc[ev["known"] == 1, "hit_model_12m"].mean(), ev.loc[ev["known"] == 0, "hit_model_12m"].mean())

    report["secondary"] = {"permutation": perm, "delong": dl, "diebold_mariano": dm, "vs_dictionary": m3b,
                           "nested_logit": nest, "any_downgrade": dg, "lead_wilcoxon": wil,
                           "mcnemar_equal_alert_rate": mcn,
                           "hit_rate_known_vs_unknown": {"known": known_gap[0], "unknown": known_gap[1]}}
    report["holm"] = st.holm({"permutation": perm["p_value"], "delong": dl["p_two_sided"],
                              "diebold_mariano": dm["p_one_sided"], "vs_dictionary": m3b["p_one_sided"],
                              "wald_llm": nest["wald_p_twoway_cluster"], "any_downgrade": dg["p_one_sided"],
                              "lead_wilcoxon": wil["p_one_sided"], "mcnemar": mcn["p_two_sided"]})
    report["calibration_M4"] = st.calibration_tests(prim["y_fa12"].values, prim["M4"].values)
    report["events"] = ev

    rob = []
    calm = prim[~prim["year"].isin([2015, 2016, 2020])]
    rob.append(("exclude 2015-16, 2020", st.pr_auc(calm["y_fa12"], calm["M4"]) - st.pr_auc(calm["y_fa12"], calm["M3"])))
    cbp = primary_sample(pred_cb)
    rob.append(("CatBoost", st.pr_auc(cbp["y_fa12"], cbp["M4"]) - st.pr_auc(cbp["y_fa12"], cbp["M3"])))
    for C in (0.1, 2.0):
        pc, _ = walk_forward(panel, "y_fa12", models=("M3", "M4"), C=C)
        pc = primary_sample(pc)
        rob.append((f"logit C={C}", st.pr_auc(pc["y_fa12"], pc["M4"]) - st.pr_auc(pc["y_fa12"], pc["M3"])))
    for q in (0.05, 0.20):
        rob.append((f"top {int(q * 100)}% capture gain",
                    st.top_capture(prim["y_fa12"].values, prim["M4"].values, q) - st.top_capture(prim["y_fa12"].values, prim["M3"].values, q)))
    report["robustness"] = pd.DataFrame(rob, columns=["check", "M4_minus_M3"])

    if port is not None:
        from . import portfolio as pf
        res = pf.run(port)
        names = ["index", "hold_6m", "screen_rel_spread", "screen_M3", "screen_M4"]
        report["portfolio"] = pf.summary(res, names)
        report["portfolio_tests"] = {
            "M4_vs_index_mean": st.nw_paired_t(res["screen_M4"], res["index"]),
            "M4_vs_spread_mean": st.nw_paired_t(res["screen_M4"], res["screen_rel_spread"]),
            "M4_vs_index_cvar": st.stationary_bootstrap_diff(res["index"], res["screen_M4"]),
            "M4_vs_spread_cvar": st.stationary_bootstrap_diff(res["screen_rel_spread"], res["screen_M4"]),
        }
    return report
