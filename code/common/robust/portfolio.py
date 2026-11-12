import numpy as np
import pandas as pd


def _weights(df, keep):
    w = df["amt"].where(keep, 0.0)
    return w / w.sum() if w.sum() > 0 else w


def run(port, score_cols=("M3", "M4", "rel_spread"), q=0.10):
    months = sorted(port["month"].unique())
    rows, prev = [], {}
    for m in months:
        d = port[port["month"] == m].set_index("cik")
        base = d["is_bbb"] == 1
        books = {"index": base, "hold_6m": base | (d["months_since_fa"].between(0, 6))}
        for s in score_cols:
            ranked = d.loc[base, s]
            cut = ranked.quantile(1 - q)
            books[f"screen_{s}"] = base & ~(d[s] >= cut)
        rec = {"month": m}
        for name, keep in books.items():
            w = _weights(d, keep)
            gross = float((w * d["credit_ret"].fillna(0)).sum())
            w_prev = prev.get(name, pd.Series(dtype=float)).reindex(w.index).fillna(0)
            turnover = float((w - w_prev).abs().sum())
            cost = float(((w - w_prev).abs() * d["half_cost"].fillna(d["half_cost"].median())).sum())
            rec[name], rec[name + "_turnover"], rec[name + "_cost"] = gross - cost, turnover, cost
            rec[name + "_carry"] = float((w * d["rel_spread"].fillna(0)).sum())
            prev[name] = w
        rows.append(rec)
    return pd.DataFrame(rows)


def summary(res, names):
    from .stats_tests import cvar
    out = []
    for n in names:
        r = res[n].values
        wealth = np.cumprod(1 + r)
        mdd = float(np.max(1 - wealth / np.maximum.accumulate(wealth)))
        out.append({"strategy": n, "mean_m": r.mean(), "vol_m": r.std(ddof=1), "cvar95": cvar(r, 0.05),
                    "max_drawdown": mdd, "avg_turnover": res[n + "_turnover"].mean(),
                    "avg_cost": res[n + "_cost"].mean(), "avg_rel_carry": res[n + "_carry"].mean()})
    return pd.DataFrame(out)
