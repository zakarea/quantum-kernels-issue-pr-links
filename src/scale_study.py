"""Scale study: how far each model family can go with more training data.

Final setting (PLS + open-time metadata + BM25), d = 12, fixed test pairs and ranking queries per seed.
Training sizes from 800 pairs up to all training pairs. Quantum kernels are run while their
n x n kernel stays tractable on this machine (QSVC-Z via the exact product-state kernel).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.svm import SVC
import xgboost as xgb, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).parent))
from qml_pilink import load_links, build_pairs, balanced_sample, metrics
from qml_enhanced import EnhancedFeaturizer, rank_eval, cv_auc, LAMS, CS
from quantum_extra import z_kernel, z_qubits
from bm25 import BM25

D = 12


def gs(est, grid, X, y, cv):
    g = GridSearchCV(est, grid, cv=cv, scoring="roc_auc", n_jobs=1).fit(X, y)
    f = g.decision_function if isinstance(g.best_estimator_, (SVC, LogisticRegression)) else (lambda Z: g.predict_proba(Z)[:, 1])
    return f, g.best_params_


def run(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    links = load_links(a.data); rows = []
    for seed in range(a.seeds):
        pairs, rank_sets = build_pairs(links, seed=seed)
        tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(pairs, groups=pairs.repo))
        tr_all, te = pairs.iloc[tr_i], balanced_sample(pairs.iloc[te_i], 400, seed)
        rs = [r for r in rank_sets if r[2] in set(te.repo)]
        rs = [rs[i] for i in np.random.default_rng(seed).permutation(len(rs))[:200]]
        rtr_all = [r for r in rank_sets if r[2] in set(tr_all.repo)]
        rtr_all = [rtr_all[i] for i in np.random.default_rng(seed + 100).permutation(len(rtr_all))]
        flat = lambda S: (np.concatenate([np.full(len(c), i) for i, c, _ in S]), np.concatenate([np.array(c) for _, c, _ in S]))
        iv, pv = flat(rs); sizes = [len(c) for _, c, _ in rs]; yte = te.y.values
        bm = BM25().fit(links.pr.values[np.unique(tr_all.pr.values)])
        sizes_n = [n for n in a.sizes if n < len(tr_all)] + [len(tr_all)]
        for n in sizes_n:
            tr = balanced_sample(tr_all, n, seed) if n < len(tr_all) else tr_all
            ytr = tr.y.values; cv = list(GroupKFold(3).split(np.zeros(len(tr)), ytr, tr.repo.values))
            t0 = time.time()
            feat = EnhancedFeaturizer(D, "pls", True, seed, bm).fit(links, tr.iss.values, tr.pr.values, ytr)
            R = lambda i, p: feat.reduced(links, i, p); F = lambda i, p: feat.full(links, i, p)
            Xtr, Xte, Xrk = R(tr.iss.values, tr.pr.values), R(te.iss.values, te.pr.values), R(iv, pv)
            Ftr, Fte, Frk = F(tr.iss.values, tr.pr.values), F(te.iss.values, te.pr.values), F(iv, pv)
            print(seed, n, "features", round(time.time() - t0, 1), flush=True)
            res = {}
            if n <= a.max_quantum:  # QSVC-Z, exact product-state kernel, lam and C tuned
                t0 = time.time()
                best = max(((lam, c) for lam in LAMS for c in CS),
                           key=lambda lc: cv_auc(z_kernel(lc[0] * Xtr, lc[0] * Xtr), ytr, cv, lc[1]))
                lam, C = best; q = z_qubits(lam * Xtr)
                m = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(z_kernel(lam * Xtr, lam * Xtr, q, q), ytr)
                res["QSVC-Z"] = (m.decision_function(z_kernel(lam * Xte, lam * Xtr, None, q)),
                                 m.decision_function(z_kernel(lam * Xrk, lam * Xtr, None, q)), time.time() - t0, dict(lam=lam, C=C))
            if n <= a.max_svc:
                t0 = time.time(); v = Xtr.var() * D
                f, bp = gs(SVC(class_weight="balanced"), {"C": CS, "gamma": [g / v for g in (0.3, 1, 3)]}, Xtr, ytr, cv)
                res["SVC-RBF"] = (f(Xte), f(Xrk), time.time() - t0, bp)
            t0 = time.time()
            f, bp = gs(LogisticRegression(max_iter=3000, class_weight="balanced"), {"C": [0.01, 0.1, 1, 10]}, Xtr, ytr, cv)
            res["LogReg"] = (f(Xte), f(Xrk), time.time() - t0, bp)
            big = n > 5000
            t0 = time.time()
            f, bp = gs(xgb.XGBClassifier(random_state=seed, eval_metric="logloss", n_jobs=2, subsample=0.8, colsample_bytree=0.8,
                                         learning_rate=0.1 if big else 0.03),
                       {"max_depth": [4, 6] if big else [2, 4, 6], "n_estimators": [300] if big else [200, 500]}, Ftr, ytr, cv)
            res["XGB(full)"] = (f(Fte), f(Frk), time.time() - t0, bp)
            t0 = time.time()
            f, bp = gs(lgb.LGBMClassifier(random_state=seed, n_jobs=2, verbose=-1, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                          learning_rate=0.1 if big else 0.03),
                       {"num_leaves": [15, 63] if big else [7, 15, 31], "n_estimators": [300] if big else [200, 500]}, Ftr, ytr, cv)
            res["LGBM(full)"] = (f(Fte), f(Frk), time.time() - t0, bp)
            t0 = time.time()  # ranker: training queries scale with n (n/4, as in the 800-pair runs)
            rtr = rtr_all[:max(50, min(len(rtr_all), n // 4))]; ri, rp = flat(rtr)
            yr = np.concatenate([[1] + [0] * (len(c) - 1) for _, c, _ in rtr])
            qid = np.concatenate([np.full(len(c), k) for k, (_, c, _) in enumerate(rtr)])
            rk = xgb.XGBRanker(objective="rank:pairwise", n_estimators=300, max_depth=4 if not big else 6, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=2).fit(F(ri, rp), yr, qid=qid)
            res["XGB-Ranker(full)"] = (rk.predict(Fte), rk.predict(Frk), time.time() - t0, dict(train_queries=len(rtr)))
            for name, (s, srk, ft, bp) in res.items():
                thr = np.median(s) if "Ranker" in name else (0 if name in ("QSVC-Z", "SVC-RBF", "LogReg") else 0.5)
                rows.append(dict(seed=seed, n_train=len(tr), model=name, fit_s=round(ft, 2),
                                 params=json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in bp.items()}),
                                 **metrics(yte, (s > thr).astype(int), s), **rank_eval(srk, sizes)))
            print(seed, len(tr), {k: round(r["mrr"], 3) for k, r in [(r["model"], r) for r in rows if r["seed"] == seed and r["n_train"] == len(tr)]}, flush=True)
            pd.DataFrame(rows).to_csv(out / "scale_results.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="pilink_pairs_meta.csv"); p.add_argument("--out", default="results_scale")
    p.add_argument("--seeds", type=int, default=5); p.add_argument("--sizes", type=int, nargs="+", default=[800, 3200, 12800])
    p.add_argument("--max_quantum", type=int, default=3200); p.add_argument("--max_svc", type=int, default=12800)
    run(p.parse_args())
