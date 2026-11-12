import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / d) for d in ("common", "1_data", "2_llm_scoring", "3_prediction")]
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score as PR, roc_auc_score as AUC

from robust import models as md
from robust.models import BoostModel, LogitModel, Prep
from extend_models import boot_p

ROOT = Path(__file__).resolve().parents[2]


class HGB:
    def __init__(self, cols): self.cols = cols
    def fit(self, df, y):
        self.m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                                min_samples_leaf=50, random_state=0).fit(df[self.cols], df[y]); return self
    def predict(self, df): return self.m.predict_proba(df[self.cols])[:, 1]


class RF:
    def __init__(self, cols): self.cols = cols
    def fit(self, df, y):
        self.prep = Prep().fit(df[self.cols])
        self.m = RandomForestClassifier(n_estimators=300, min_samples_leaf=50, max_features="sqrt", n_jobs=-1,
                                        random_state=0).fit(self.prep.transform(df[self.cols]), df[y]); return self
    def predict(self, df): return self.m.predict_proba(self.prep.transform(df[self.cols]))[:, 1]


MODELS = {"로지스틱 (L2, 주 모형)": (LogitModel, "y_fa12"), "이산 생존 (해저드)": (LogitModel, "y_fa1"),
          "CatBoost": (BoostModel, "y_fa12"), "HistGB (LightGBM형)": (HGB, "y_fa12"), "랜덤포레스트": (RF, "y_fa12")}


def main():
    p = pd.read_parquet(ROOT / "data" / "interim" / "panel_final.parquet")
    num = p.select_dtypes("number").columns
    p[num] = p[num].replace([np.inf, -np.inf], np.nan)
    p["fa_mi"] = p["fa_month"].map(lambda x: pd.Period(x, freq="M").ordinal if isinstance(x, str) else np.nan)
    p["y_fa1"] = ((p["fa_mi"] - p["mi"]) == 1).astype(int)
    a = p[p["spread"].notna() & p["year"].between(2014, 2024)].reset_index(drop=True)
    preds = []
    for Y in range(2016, 2025):
        tr = a[(a["year"] >= 2014) & (a["year"] <= Y - 2) & (a["censored"] == 0)]
        te = a[a["year"] == Y].copy()
        for mname, (cls, y) in MODELS.items():
            for lad in ("M3", "M4"):
                cols = [c for c in md.LADDER[lad] if c in a.columns]
                te[f"{mname}|{lad}"] = cls(cols).fit(tr, y).predict(te)
        preds.append(te)
        print("done", Y, flush=True)
    pr = pd.concat(preds, ignore_index=True)
    nk = ~((pr["y_fa12"] == 1) & (pr["known"] == 1))
    rows = []
    for which, mask in [("경고 없는 구간 (주 분석)", nk & (pr["warn"] == 0)), ("S&P 경고 기업", nk & (pr["warn"] == 1)), ("전체", nk)]:
        s = pr[mask].reset_index(drop=True)
        for mname in MODELS:
            a3, a4 = f"{mname}|M3", f"{mname}|M4"
            d0, lo, hi, pv = boot_p(s, "y_fa12", a3, a4, n=500)
            rows.append({"sample": which, "model": mname, "PR_M3": PR(s["y_fa12"], s[a3]), "PR_M4": PR(s["y_fa12"], s[a4]),
                         "AUC_M3": AUC(s["y_fa12"], s[a3]), "AUC_M4": AUC(s["y_fa12"], s[a4]), "diff": d0, "p_one": pv,
                         "events": int(s["y_fa12"].sum())})
    r = pd.DataFrame(rows)
    r.to_csv(ROOT / "results" / "full" / "ext_model_compare.csv", index=False)
    pd.set_option("display.width", 220)
    print(r.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
