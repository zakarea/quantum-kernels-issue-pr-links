"""Dependence-aware re-analysis of the confirmatory comparisons (reviewer comment on seed independence).

The 30 seeds re-split one dataset, so training and test sets overlap between seeds and seed-level
differences are not independent. Two alternatives to the plain paired Wilcoxon/t analysis:

  1. Nadeau-Bengio corrected resampled t-test (Nadeau & Bengio 2003): the variance of the mean
     difference is inflated from s^2/J to (1/J + n_test/n_train) s^2. Applied to AUC and MRR, with
     TOST against the pre-set margins (0.01 AUC, 0.02 MRR). n_test/n_train = 0.3/0.7 on both datasets.
  2. Cluster bootstrap over the pooled per-query reciprocal ranks of all 30 seeds (per_query.py):
     resample whole repositories (PI-Link) or whole issues and whole projects (GHPR, 13 projects),
     keeping every query of a cluster across all seeds. 90% interval for TOST, 95% for the difference.

  python dependence_stats.py            -> results_review/dependence_seed.csv, dependence_bootstrap.csv
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

RHO = 0.3 / 0.7
MARGIN = {"auc": 0.01, "mrr": 0.02}
REF = "SVC-RBF"; MODELS = ["PQK-Z", "QSVC-Z", "QSVC-ZZ1"]
SETTINGS = {"pilink": ["results_final30/a/raw_results.csv", "results_final30/b/raw_results.csv"],
            "ghpr": ["results_ghpr/raw_results.csv"], "pilink-e5": ["results_e5/pilink/raw_results.csv"],
            "ghpr-e5": ["results_e5/ghpr/raw_results.csv"]}
OUT = Path("results_review")


def nb(d, m):
    J = len(d); mean = d.mean(); se = np.sqrt((1 / J + RHO) * d.var(ddof=1)); df = J - 1
    p_diff = 2 * stats.t.sf(abs(mean) / se, df)
    p_tost = max(stats.t.sf((mean + m) / se, df), stats.t.sf((m - mean) / se, df))
    h = stats.t.ppf(0.975, df) * se
    return dict(mean_diff=mean, ci95_lo=mean - h, ci95_hi=mean + h, p_diff=p_diff, p_tost=p_tost,
                se_plain=d.std(ddof=1) / np.sqrt(J), se_corrected=se)


def seed_level():
    rows = []
    for s, files in SETTINGS.items():
        r = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates(["seed", "model"])
        w = {met: r.pivot(index="seed", columns="model", values=met) for met in MARGIN}
        for mdl in MODELS:
            for met, m in MARGIN.items():
                d = (w[met][mdl] - w[met][REF]).dropna().values
                rows.append(dict(setting=s, model=mdl, metric=met, seeds=len(d), **nb(d, m)))
    t = pd.DataFrame(rows)
    t["p_diff_holm"] = np.nan  # Holm within each setting over its 6 difference tests, as in the paper's families
    for s, g in t.groupby("setting"):
        p = g.p_diff.values; o = np.argsort(p); adj = np.maximum.accumulate(np.minimum(1, p[o] * (len(p) - np.arange(len(p)))))
        t.loc[g.index[o], "p_diff_holm"] = adj
    t["verdict"] = np.where(t.p_tost < 0.05, "equivalent", np.where(t.p_diff_holm < 0.05, np.where(t.mean_diff > 0, "better", "worse"), "inconclusive"))
    t.to_csv(OUT / "dependence_seed.csv", index=False)
    return t


def boot(df, cluster, mdl, B=5000, seed=0):
    d = (df[f"rr_{mdl}"] - df[f"rr_{REF}"]).values
    codes, uniq = pd.factorize(df[cluster]); n = len(uniq)
    sums = np.bincount(codes, weights=d, minlength=n); cnts = np.bincount(codes, minlength=n)
    rng = np.random.default_rng(seed); idx = rng.integers(0, n, size=(B, n))
    bs = sums[idx].sum(1) / cnts[idx].sum(1); mean = d.mean()
    lo90, hi90 = np.percentile(bs, [5, 95]); lo95, hi95 = np.percentile(bs, [2.5, 97.5])
    m = MARGIN["mrr"]
    return dict(clusters=n, queries=len(d), mean_diff=mean, ci95_lo=lo95, ci95_hi=hi95, ci90_lo=lo90, ci90_hi=hi90,
                p_diff=min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()) + 1 / B),
                equivalent=bool(lo90 > -m and hi90 < m))


def cluster_level():
    rows = []
    for s in SETTINGS:
        f = OUT / f"per_query_{s}.csv"
        if not f.exists():
            print("missing", f); continue
        df = pd.read_csv(f)
        for cl in (["repo"] if s.startswith("pilink") else ["issue", "repo"]):
            for mdl in MODELS:
                rows.append(dict(setting=s, cluster=cl if s.startswith("ghpr") or cl != "repo" else "repository", model=mdl, **boot(df, cl, mdl)))
    t = pd.DataFrame(rows)
    t.loc[t.setting.str.startswith("ghpr") & (t.cluster == "repo"), "cluster"] = "project"
    t.to_csv(OUT / "dependence_bootstrap.csv", index=False)
    return t


def budget():
    """Quantum kernels tuned with 9 configurations (budget_grid.py) vs the saved SVC-RBF: plain paired and corrected TOST."""
    rows = []
    for s in ("pilink", "ghpr"):
        f = OUT / f"budget_grid_{s}.csv"
        if not f.exists():
            continue
        b = pd.read_csv(f); ref = pd.concat([pd.read_csv(x) for x in SETTINGS[s]]).drop_duplicates(["seed", "model"])
        ref = ref[ref.model == REF].set_index("seed")
        for mdl, g in b.groupby("model"):
            g = g.set_index("seed")
            for met, m in MARGIN.items():
                d = (g[met] - ref.loc[g.index, met]).values; J = len(d); se = d.std(ddof=1) / np.sqrt(J)
                p_plain = max(stats.t.sf((d.mean() + m) / se, J - 1), stats.t.sf((m - d.mean()) / se, J - 1))
                rows.append(dict(setting=s, model=mdl, metric=met, seeds=J, mean_model=g[met].mean(), mean_ref=ref.loc[g.index, met].mean(),
                                 p_tost_plain=p_plain, **nb(d, m)))
    t = pd.DataFrame(rows)
    if len(t):
        t.to_csv(OUT / "budget_grid_tests.csv", index=False)
    return t


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    t = seed_level()
    print(t[["setting", "model", "metric", "mean_diff", "ci95_lo", "ci95_hi", "p_tost", "p_diff_holm", "verdict"]].round(4).to_string())
    print((t.se_corrected / t.se_plain).round(2).unique())
    b = cluster_level()
    if len(b):
        print(b.round(4).to_string())
    g = budget()
    if len(g):
        print(g.round(4).to_string())
