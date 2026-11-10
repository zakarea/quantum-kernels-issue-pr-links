"""Confirmatory analysis of the 30-seed run.

For each pre-registered comparison (A vs B, paired by seed):
  mean difference with 95% t-interval, Wilcoxon signed-rank p, Cliff's delta,
  Holm-corrected p across the whole family, and TOST equivalence
  (paired t, margins: AUC +-0.01, MRR +-0.02).
Reported on all 30 seeds and on fresh seeds 7-29 only.
"""
import sys
import numpy as np, pandas as pd
from scipy import stats

MARGIN = {"auc": 0.01, "mrr": 0.02}
COMPS = [("PQK-Z", "SVC-RBF"), ("PQK-Z", "LogReg"), ("PQK-Z", "XGB"), ("QSVC-Z", "SVC-RBF"),
         ("QSVC-ZZ1", "SVC-RBF"), ("PQK-Z", "XGB(full)"), ("PQK-Z", "XGB-Ranker(full)"),
         ("QSVC-Z", "QSVC-Z(untuned)")]


def cliffs(x, y):
    return ((x[:, None] > y).sum() - (x[:, None] < y).sum()) / (len(x) * len(y))


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, min(1, (m - k) * p[i])); adj[i] = run
    return adj


def analyze(df, label):
    rows = []
    for met in ("auc", "mrr"):
        for a, b in COMPS:
            x = df[df.model == a].sort_values("seed")[met].values; y = df[df.model == b].sort_values("seed")[met].values
            d = x - y; n = len(d); m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
            lo, hi = stats.t.interval(0.95, n - 1, loc=m, scale=se)
            eps = MARGIN[met]  # TOST: both one-sided tests must reject
            p_tost = max(stats.t.sf((m + eps) / se, n - 1), stats.t.cdf((m - eps) / se, n - 1))
            rows.append(dict(sample=label, metric=met, A=a, B=b, n=n, mean_A=round(x.mean(), 4), mean_B=round(y.mean(), 4),
                             diff=round(m, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                             p_wilcoxon=stats.wilcoxon(x, y).pvalue if np.any(d) else 1.0,
                             cliffs=round(cliffs(x, y), 3), margin=eps, p_tost=p_tost))
    r = pd.DataFrame(rows)
    r["p_holm"] = holm(r.p_wilcoxon); r["p_tost_holm"] = holm(r.p_tost)
    r["verdict"] = np.where(r.p_holm < 0.05, np.where(r["diff"] > 0, "A better", "B better"),
                            np.where(r.p_tost_holm < 0.05, "equivalent", "inconclusive"))
    for c in ("p_wilcoxon", "p_tost", "p_holm", "p_tost_holm"): r[c] = r[c].round(5)
    return r


if __name__ == "__main__":
    df = pd.concat([pd.read_csv(f) for f in sys.argv[1:]]).drop_duplicates(["seed", "model"])
    print("seeds:", df.seed.nunique(), "models:", sorted(df.model.unique()))
    summ = df.groupby("model")[["f1", "auc", "hit1", "hit5", "mrr", "fit_s"]].agg(["mean", "std"]).round(4)
    import os; OUT = os.environ.get("OUT", "results_final30"); summ.to_csv(f"{OUT}/summary.csv"); print(summ.sort_values(("mrr", "mean"), ascending=False).to_string())
    res = pd.concat([analyze(df, "all 30 seeds"), analyze(df[df.seed >= 7], "fresh seeds 7-29")])
    res.to_csv(f"{OUT}/confirmatory_tests.csv", index=False)
    pd.set_option("display.width", 260)
    print(res[["sample", "metric", "A", "B", "diff", "ci_lo", "ci_hi", "p_wilcoxon", "p_holm", "cliffs", "p_tost_holm", "verdict"]].to_string())
