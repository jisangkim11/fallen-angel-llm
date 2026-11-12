import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

RNG = np.random.default_rng(20261102)


def pr_auc(y, p):
    return average_precision_score(y, p) if 0 < np.sum(y) < len(y) else np.nan


def roc_auc(y, p):
    return roc_auc_score(y, p) if 0 < np.sum(y) < len(y) else np.nan


def top_capture(y, p, q=0.10):
    if np.sum(y) == 0:
        return np.nan
    cut = np.quantile(p, 1 - q)
    return float(np.sum(y[p >= cut]) / np.sum(y))


def within_month_auc(df, y, p, month="month"):
    vals = [roc_auc(g[y].values, g[p].values) for _, g in df.groupby(month)]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


def cluster_bootstrap_diff(df, y, p_base, p_new, metric=pr_auc, cluster="cik", n=2000, rng=RNG):
    groups = [g.index.values for _, g in df.groupby(cluster)]
    yv, a, b = df[y].values, df[p_base].values, df[p_new].values
    pos = {k: i for i, k in enumerate(df.index)}
    groups = [np.array([pos[i] for i in g]) for g in groups]
    point = metric(yv, b) - metric(yv, a)
    diffs = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        d = metric(yv[idx], b[idx]) - metric(yv[idx], a[idx])
        if not np.isnan(d):
            diffs.append(d)
    diffs = np.array(diffs)
    return {"diff": point, "ci_low": np.percentile(diffs, 2.5), "ci_high": np.percentile(diffs, 97.5),
            "p_one_sided": float(np.mean(diffs <= 0)), "n_boot": len(diffs)}


def permutation_test(df, y, predict_fn, cols, p_base, metric=pr_auc, month="month", n=1000, rng=RNG):
    yv, base = df[y].values, df[p_base].values
    observed = metric(yv, predict_fn(df)) - metric(yv, base)
    month_idx = [g.index for _, g in df.groupby(month)]
    null = []
    for _ in range(n):
        sh = df.copy()
        for ix in month_idx:
            perm = rng.permutation(len(ix))
            sh.loc[ix, cols] = df.loc[ix, cols].values[perm]
        null.append(metric(yv, predict_fn(sh)) - metric(yv, base))
    null = np.array(null)
    return {"observed": observed, "null_mean": null.mean(), "p_value": float((1 + np.sum(null >= observed)) / (1 + n))}


def _midrank(x):
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong_test(y, p1, p2):
    y = np.asarray(y).astype(bool)
    preds = np.vstack([p1, p2])
    pos, neg = preds[:, y], preds[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([_midrank(r) for r in pos])
    ty = np.array([_midrank(r) for r in neg])
    tz = np.array([_midrank(r) for r in preds[:, np.r_[np.where(y)[0], np.where(~y)[0]]]])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    s = np.cov(v01) / m + np.cov(v10) / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    z = (aucs[1] - aucs[0]) / np.sqrt(var)
    return {"auc1": aucs[0], "auc2": aucs[1], "z": z, "p_two_sided": 2 * stats.norm.sf(abs(z))}


def nested_logit_tests(df, y, base_cols, add_cols, firm="cik", month="month"):
    d = df[[y, firm, month] + base_cols + add_cols].dropna()
    X0 = sm.add_constant(d[base_cols])
    X1 = sm.add_constant(d[base_cols + add_cols])
    groups = np.column_stack([pd.factorize(d[firm])[0], pd.factorize(d[month])[0]])
    r0 = sm.Logit(d[y], X0).fit(disp=0)
    r1 = sm.Logit(d[y], X1).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
    lr = 2 * (r1.llf - r0.llf)
    R = np.zeros((len(add_cols), X1.shape[1]))
    for i, c in enumerate(add_cols):
        R[i, list(X1.columns).index(c)] = 1
    w = r1.wald_test(R, scalar=True)
    return {"lr_stat": lr, "lr_p": stats.chi2.sf(lr, len(add_cols)),
            "wald_stat": float(w.statistic), "wald_p_twoway_cluster": float(w.pvalue),
            "coef": r1.params[add_cols].to_dict(), "se_twoway": r1.bse[add_cols].to_dict()}


def log_loss_vec(y, p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def diebold_mariano(df, y, p_base, p_new, month="month", loss="log", lags=12):
    f = (lambda yy, pp: log_loss_vec(yy, pp)) if loss == "log" else (lambda yy, pp: (yy - pp) ** 2)
    d = df.assign(_d=f(df[y].values, df[p_base].values) - f(df[y].values, df[p_new].values))
    series = d.groupby(month)["_d"].mean().sort_index().values
    res = sm.OLS(series, np.ones(len(series))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    t = res.params[0] / res.bse[0]
    return {"mean_diff": res.params[0], "t_nw": t, "p_one_sided": stats.norm.sf(t), "months": len(series)}


def wilcoxon_paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    if np.all(d == 0):
        return {"median_diff": 0.0, "p_one_sided": 1.0, "n": len(d)}
    s = stats.wilcoxon(d, alternative="greater", zero_method="zsplit")
    return {"median_diff": float(np.median(d)), "stat": float(s.statistic), "p_one_sided": float(s.pvalue), "n": len(d)}


def mcnemar_paired(hit_a, hit_b):
    a, b = np.asarray(hit_a).astype(bool), np.asarray(hit_b).astype(bool)
    table = [[np.sum(a & b), np.sum(a & ~b)], [np.sum(~a & b), np.sum(~a & ~b)]]
    r = mcnemar(table, exact=True)
    return {"only_a": int(table[0][1]), "only_b": int(table[1][0]), "p_two_sided": float(r.pvalue)}


def calibration_tests(y, p, eps=1e-6):
    p = np.clip(np.asarray(p), eps, 1 - eps)
    y = np.asarray(y)
    lp = np.log(p / (1 - p))
    slope_fit = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    int_fit = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
    num = np.sum((y - p) * (1 - 2 * p))
    den = np.sqrt(np.sum((1 - 2 * p) ** 2 * p * (1 - p)))
    z = num / den
    return {"slope": slope_fit.params[1], "slope_se": slope_fit.bse[1],
            "intercept_in_large": int_fit.params[0], "spiegelhalter_z": z,
            "spiegelhalter_p": 2 * stats.norm.sf(abs(z)), "brier": float(np.mean((y - p) ** 2))}


def nw_paired_t(r_a, r_b, lags=6):
    d = np.asarray(r_a) - np.asarray(r_b)
    res = sm.OLS(d, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    t = res.params[0] / res.bse[0]
    return {"mean_diff": res.params[0], "t_nw": t, "p_one_sided": stats.norm.sf(t)}


def cvar(r, alpha=0.05):
    r = np.sort(np.asarray(r))
    k = max(1, int(np.floor(alpha * len(r))))
    return float(-r[:k].mean())


def stationary_bootstrap_diff(r_a, r_b, fn=cvar, mean_block=6, n=5000, rng=RNG):
    r_a, r_b = np.asarray(r_a), np.asarray(r_b)
    T = len(r_a)
    point = fn(r_b) - fn(r_a)
    out = []
    for _ in range(n):
        idx, t = [], rng.integers(T)
        while len(idx) < T:
            idx.append(t)
            t = rng.integers(T) if rng.random() < 1 / mean_block else (t + 1) % T
        idx = np.array(idx)
        out.append(fn(r_b[idx]) - fn(r_a[idx]))
    out = np.array(out)
    return {"diff": point, "ci_low": np.percentile(out, 2.5), "ci_high": np.percentile(out, 97.5)}


def fisher_yes_rates(yes_events, yes_nonevents):
    a, b = int(np.sum(yes_events)), len(yes_events) - int(np.sum(yes_events))
    c, d = int(np.sum(yes_nonevents)), len(yes_nonevents) - int(np.sum(yes_nonevents))
    odds, p = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
    return {"yes_rate_events": a / max(a + b, 1), "yes_rate_nonevents": c / max(c + d, 1), "odds_ratio": odds, "p": p}


def icc_2_1(ratings):
    x = np.asarray(ratings, dtype=float)
    n, k = x.shape
    gm = x.mean()
    msr = k * np.sum((x.mean(axis=1) - gm) ** 2) / (n - 1)
    msc = n * np.sum((x.mean(axis=0) - gm) ** 2) / (k - 1)
    sse = np.sum((x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + gm) ** 2)
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def agreement_with_humans(llm, human):
    llm, human = np.asarray(llm), np.asarray(human)
    rho, p = stats.spearmanr(llm, human)
    kappa = cohen_kappa_score(np.round(llm).astype(int), np.round(human).astype(int), weights="quadratic")
    return {"spearman": rho, "spearman_p": p, "weighted_kappa": kappa}


def holm(pvals: dict, alpha=0.05):
    keys = list(pvals)
    rej, padj, _, _ = multipletests([pvals[k] for k in keys], alpha=alpha, method="holm")
    return pd.DataFrame({"test": keys, "p_raw": [pvals[k] for k in keys], "p_holm": padj, "reject": rej})


def power_auc_diff(n_pos, n_neg, auc_base, auc_new, corr=0.6, alpha=0.05):
    def var(a):
        q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
        return (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    v1, v2 = var(auc_base), var(auc_new)
    se = np.sqrt(v1 + v2 - 2 * corr * np.sqrt(v1 * v2))
    z = (auc_new - auc_base) / se
    return float(stats.norm.cdf(z - stats.norm.ppf(1 - alpha)))
