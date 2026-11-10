"""Stronger classical baselines on the best setting (all pair features + open-time metadata).

Same data, pairs, repo-level splits and seeds as qml_enhanced.py (PLS+meta).
Hyperparameters tuned by 3-fold GroupKFold (by repository) inside training, scored by AUC.

  BM25                unsupervised lexical retrieval score (issue as query, PR as document)
  XGB(full)           reference: same grid as qml_enhanced.py, no BM25 feature
  XGB(full+BM25)      wider grid, BM25 added as a feature
  LGBM, ExtraTrees, MLP   on full+BM25
  XGB-Ranker          LambdaMART-style pairwise ranker trained on 1+19 candidate sets from training repos
  Ensemble            mean of z-scored SVC-RBF(d=12, PLS+meta), LGBM and XGB(full+BM25) scores
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import xgboost as xgb, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).parent))
from qml_pilink import load_links, build_pairs, balanced_sample, metrics
from qml_enhanced import EnhancedFeaturizer, rank_eval
from bm25 import BM25


def fitted(est, grid, X, y, cv):
    gs = GridSearchCV(est, grid, cv=cv, scoring="roc_auc", n_jobs=1).fit(X, y)
    f = (lambda Z: gs.predict_proba(Z)[:, 1]) if hasattr(gs.best_estimator_, "predict_proba") and not isinstance(gs.best_estimator_, SVC) else gs.decision_function
    return f, gs.best_params_


def run(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    links = load_links(a.data); rows = []
    for seed in range(a.seeds):
        pairs, rank_sets = build_pairs(links, seed=seed)
        tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(pairs, groups=pairs.repo))
        tr, te = balanced_sample(pairs.iloc[tr_i], a.n_train, seed), balanced_sample(pairs.iloc[te_i], a.n_test, seed)
        te_repos, tr_repos = set(te.repo), set(pairs.iloc[tr_i].repo)
        rs = [r for r in rank_sets if r[2] in te_repos]
        rs = [rs[i] for i in np.random.default_rng(seed).permutation(len(rs))[:a.n_rank]]
        rtr = [r for r in rank_sets if r[2] in tr_repos]  # ranker training queries, training repos only
        rtr = [rtr[i] for i in np.random.default_rng(seed + 100).permutation(len(rtr))[:a.n_rank_train]]
        flat = lambda S: (np.concatenate([np.full(len(c), i) for i, c, _ in S]), np.concatenate([np.array(c) for _, c, _ in S]))
        iv, pv = flat(rs); ri, rp = flat(rtr); sizes = [len(c) for _, c, _ in rs]
        ytr, yte = tr.y.values, te.y.values
        cv = list(GroupKFold(3).split(np.zeros(len(tr)), ytr, tr.repo.values))

        feat = EnhancedFeaturizer(12, "pls", True, seed).fit(links, tr.iss.values, tr.pr.values, ytr)
        bm = BM25().fit(links.pr.values[np.unique(pairs.iloc[tr_i].pr.values)])
        F = lambda i, p: feat.full(links, i, p)
        FB = lambda i, p: np.hstack([F(i, p), np.log1p(bm.score(links.issue.values[i], links.pr.values[p]))[:, None]])
        Xtr, Xte, Xrk = F(tr.iss.values, tr.pr.values), F(te.iss.values, te.pr.values), F(iv, pv)
        Btr, Bte, Brk = FB(tr.iss.values, tr.pr.values), FB(te.iss.values, te.pr.values), FB(iv, pv)
        Rtr, Rte, Rrk = (feat.reduced(links, i, p) for i, p in ((tr.iss.values, tr.pr.values), (te.iss.values, te.pr.values), (iv, pv)))

        res = {}
        t0 = time.time(); res["BM25"] = (Bte[:, -1], Brk[:, -1], time.time() - t0, {})
        specs = [
            ("SVC-RBF(d=12)", SVC(class_weight="balanced"), {"C": [0.1, 1, 10], "gamma": [g / (Rtr.var() * 12) for g in (0.3, 1, 3)]}, "red"),
            ("XGB(full)", xgb.XGBClassifier(learning_rate=0.1, random_state=seed, eval_metric="logloss", n_jobs=2),
             {"max_depth": [2, 4], "n_estimators": [100, 300]}, "full"),
            ("XGB(full+BM25)", xgb.XGBClassifier(random_state=seed, eval_metric="logloss", n_jobs=2, subsample=0.8, colsample_bytree=0.8),
             {"max_depth": [2, 4, 6], "n_estimators": [200, 500], "learning_rate": [0.03, 0.1]}, "bm"),
            ("LGBM", lgb.LGBMClassifier(random_state=seed, n_jobs=2, verbose=-1, subsample=0.8, subsample_freq=1, colsample_bytree=0.8),
             {"num_leaves": [7, 15, 31], "n_estimators": [200, 500], "learning_rate": [0.03, 0.1]}, "bm"),
            ("ExtraTrees", ExtraTreesClassifier(500, random_state=seed, n_jobs=2, class_weight="balanced"),
             {"min_samples_leaf": [1, 5, 10], "max_features": ["sqrt", 0.3]}, "bm"),
            ("MLP", MLPClassifier((128, 64), max_iter=500, early_stopping=True, random_state=seed),
             {"alpha": [1e-4, 1e-3, 1e-2]}, "bm"),
        ]
        sets = {"red": (Rtr, Rte, Rrk), "full": (Xtr, Xte, Xrk), "bm": (Btr, Bte, Brk)}
        for name, est, grid, kind in specs:
            A, B, Cc = sets[kind]; t0 = time.time()
            f, bp = fitted(est, grid, A, ytr, cv)
            res[name] = (f(B), f(Cc), time.time() - t0, bp)

        # pairwise ranker on 1+19 candidate sets from training repos (more negatives per issue)
        t0 = time.time(); Xr = FB(ri, rp); yr = np.concatenate([[1] + [0] * (len(c) - 1) for _, c, _ in rtr])
        qid = np.concatenate([np.full(len(c), k) for k, (_, c, _) in enumerate(rtr)])
        rk = xgb.XGBRanker(objective="rank:pairwise", n_estimators=300, max_depth=4, learning_rate=0.1,
                           subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=2).fit(Xr, yr, qid=qid)
        res["XGB-Ranker"] = (rk.predict(Bte), rk.predict(Brk), time.time() - t0, {"train_queries": len(rtr)})

        z = lambda s, ref: (s - ref.mean()) / (ref.std() + 1e-9)
        ens = ["SVC-RBF(d=12)", "LGBM", "XGB(full+BM25)"]
        res["Ensemble"] = (np.mean([z(res[m][0], res[m][0]) for m in ens], 0),
                           np.mean([z(res[m][1], res[m][0]) for m in ens], 0), 0.0, {"members": ens})
        for name, (s, srk, ft, bp) in res.items():
            thr = np.median(s) if name in ("BM25", "XGB-Ranker", "Ensemble") else (0 if "SVC" in name else 0.5)
            row = dict(seed=seed, model=name, fit_s=round(ft, 2), params=json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in bp.items()}),
                       **metrics(yte, (s > thr).astype(int), s), **rank_eval(srk, sizes))
            rows.append(row); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        pd.DataFrame(rows).to_csv(out / "raw_results.csv", index=False)
    df = pd.DataFrame(rows)
    df.groupby("model")[["f1", "auc", "hit1", "hit5", "mrr", "fit_s"]].agg(["mean", "std"]).round(3).to_csv(out / "summary.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="pilink_pairs_meta.csv"); p.add_argument("--out", default="results_classical_plus")
    p.add_argument("--seeds", type=int, default=7); p.add_argument("--n_train", type=int, default=800)
    p.add_argument("--n_test", type=int, default=400); p.add_argument("--n_rank", type=int, default=200)
    p.add_argument("--n_rank_train", type=int, default=200)
    run(p.parse_args())
