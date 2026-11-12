import numpy as np
import pandas as pd

from .evaluate import primary_sample
from .models import walk_forward
from . import stats_tests as st


def make_panel(n_firms=400, signal=1.0, seed=0):
    rng = np.random.default_rng(seed)
    months = pd.period_range("2014-01", "2024-12", freq="M")
    rows = []
    for f in range(n_firms):
        latent = rng.normal(-0.5, 1)
        hidden = 0.0
        notch = rng.integers(1, 4)
        for m in months:
            hidden = 0.9 * hidden + rng.normal(0, 0.45)
            spread_info = 0.5 * latent + rng.normal(0, 1)
            z = -7.0 + 0.6 * notch + 0.8 * latent + signal * hidden
            p_month = 1 / (1 + np.exp(-z))
            rows.append({"cik": f, "month": str(m), "year": m.year, "notch_pos": notch,
                         "warn": int(rng.random() < 0.05 + 0.1 * (z > -5)),
                         "rel_spread": spread_info, "size_log": rng.normal(), "ind_dg_rate": rng.random(),
                         "lev_ind_adj": 0.5 * latent + rng.normal(), "lev_chg": rng.normal(),
                         "mat_ratio": rng.random(), "payout_assets": rng.random(),
                         "llm_bus": rng.normal(), "llm_fin": rng.normal(),
                         "llm_liq": hidden + rng.normal(0, 0.7), "llm_pol": rng.normal(),
                         "llm_age": rng.integers(0, 12), "lm_neg": rng.normal(), "lm_pol": rng.normal(),
                         "_p": p_month})
    df = pd.DataFrame(rows).sort_values(["cik", "month"]).reset_index(drop=True)
    df["_ev"] = (np.random.default_rng(seed + 1).random(len(df)) < df["_p"]).astype(int)
    df["_mi"] = pd.PeriodIndex(df["month"], freq="M").astype(int)
    fa = np.full(len(df), np.nan)
    for _, g in df.groupby("cik"):
        ev_mi = g.loc[g["_ev"] == 1, "_mi"].values
        for i, mi in zip(g.index, g["_mi"].values):
            nxt = ev_mi[(ev_mi > mi) & (ev_mi <= mi + 12)]
            if len(nxt):
                fa[i] = nxt[0]
    df["y_fa12"] = (~np.isnan(fa)).astype(int)
    df["fa_month"] = [str(pd.Period(ordinal=int(x), freq="M")) if not np.isnan(x) else np.nan for x in fa]
    df["y_dg12"] = df["y_fa12"]
    df["censored"] = 0
    known_firm = np.random.default_rng(seed + 2).random(df["cik"].nunique()) < 0.3
    df["known"] = known_firm[df["cik"].values].astype(int)
    return df.drop(columns=["_p", "_ev", "_mi"])


def make_port(panel, seed=0):
    rng = np.random.default_rng(seed)
    p = panel[["cik", "month", "rel_spread", "llm_liq"]].copy()
    p["is_bbb"] = 1
    p["amt"] = rng.uniform(0.5, 2, len(p))
    p["half_cost"] = 0.002
    p["months_since_fa"] = np.nan
    p["credit_ret"] = 0.002 - 0.02 * panel["y_fa12"].values * rng.random(len(p)) + rng.normal(0, 0.01, len(p))
    return p


def smoke_full():
    from .evaluate import run_all
    from .models import walk_forward as wf
    panel = make_panel(n_firms=200, signal=1.0)
    port = make_port(panel)
    pred, _ = wf(panel, "y_fa12", models=("M3", "M4"))
    port = port.merge(pred[["cik", "month", "M3", "M4"]], on=["cik", "month"], how="inner")
    rep = run_all(panel, port, n_boot=100, n_perm=20)
    print("primary:", {k: rep["primary"][k] for k in ("diff", "ci_low", "ci_high", "p_one_sided", "decision")})
    print(rep["holm"].round(4).to_string(index=False))
    print(rep["robustness"].round(4).to_string(index=False))
    print(rep["portfolio"].round(4).to_string(index=False))


def check(signal):
    panel = make_panel(signal=signal)
    pred, _ = walk_forward(panel, "y_fa12", models=("M3", "M4"))
    prim = primary_sample(pred)
    b = st.cluster_bootstrap_diff(prim, "y_fa12", "M3", "M4", st.pr_auc, n=300)
    dm = st.diebold_mariano(pred, "y_fa12", "M3", "M4")
    return {"signal": signal, "events": int(prim["y_fa12"].sum()), "PR_AUC_M3": st.pr_auc(prim["y_fa12"], prim["M3"]),
            "PR_AUC_M4": st.pr_auc(prim["y_fa12"], prim["M4"]), "boot_p": b["p_one_sided"],
            "ci": (round(b["ci_low"], 4), round(b["ci_high"], 4)), "DM_p": dm["p_one_sided"]}


if __name__ == "__main__":
    for s in (1.0, 0.0):
        print(check(s))
    smoke_full()
