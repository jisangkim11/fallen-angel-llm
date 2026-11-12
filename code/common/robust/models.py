import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

AGENCY = ["notch_pos", "warn"]
MARKET = ["rel_spread"]
NUMERIC = ["size_log", "ind_dg_rate", "lev_ind_adj", "lev_chg", "mat_ratio", "payout_assets"]
DICT = ["lm_neg", "lm_pol"]
LLM = ["llm_bus", "llm_fin", "llm_liq", "llm_pol", "llm_age"]

LADDER = {
    "M1": AGENCY,
    "M2": AGENCY + MARKET,
    "M3": AGENCY + MARKET + NUMERIC,
    "M3b": AGENCY + MARKET + NUMERIC + DICT,
    "M4": AGENCY + MARKET + NUMERIC + LLM,
    "M3ns": AGENCY + NUMERIC,
    "M4ns": AGENCY + NUMERIC + LLM,
}


class Prep:

    def fit(self, X):
        self.med = X.median()
        Z = X.fillna(self.med)
        self.mu, self.sd = Z.mean(), Z.std().replace(0, 1)
        return self

    def transform(self, X):
        return ((X.fillna(self.med) - self.mu) / self.sd).values


class LogitModel:
    def __init__(self, cols, C=0.5):
        self.cols, self.C = cols, C

    def fit(self, df, y):
        self.prep = Prep().fit(df[self.cols])
        self.m = LogisticRegression(C=self.C, max_iter=5000).fit(self.prep.transform(df[self.cols]), df[y])
        return self

    def predict(self, df):
        return self.m.predict_proba(self.prep.transform(df[self.cols]))[:, 1]


class BoostModel:
    def __init__(self, cols):
        self.cols = cols

    def fit(self, df, y):
        self.m = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, l2_leaf_reg=5,
                                    verbose=0, random_seed=0, auto_class_weights=None)
        self.m.fit(df[self.cols], df[y])
        return self

    def predict(self, df):
        return self.m.predict_proba(df[self.cols])[:, 1]


def walk_forward(panel, y="y_fa12", models=("M1", "M2", "M3", "M3b", "M4"), test_years=range(2016, 2025),
                 C=0.5, boost=False, train_start=2014):
    out, fitted = [], {}
    for Y in test_years:
        train = panel[(panel["year"] >= train_start) & (panel["year"] <= Y - 2) & (panel["censored"] == 0)]
        test = panel[panel["year"] == Y].copy()
        if train[y].sum() < 5 or len(test) == 0:
            continue
        for name in models:
            cols = [c for c in LADDER[name] if c in panel.columns]
            m = (BoostModel(cols) if boost else LogitModel(cols, C)).fit(train, y)
            test[name] = m.predict(test)
            fitted[(name, Y)] = m
        out.append(test)
    return pd.concat(out).reset_index(drop=True), fitted


def predict_with(fitted, name):
    def fn(df):
        p = np.empty(len(df))
        for Y, idx in df.groupby("year").indices.items():
            p[idx] = fitted[(name, Y)].predict(df.iloc[idx])
        return p
    return fn
