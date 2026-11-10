"""Final confirmatory run: 30 seeds, d = 12, PLS + open-time metadata + BM25.

Primary quantum model fixed in advance: PQK-Z (best on the 7-seed exploratory run).
Seeds 0-6 repeat the exploratory run; seeds 7-29 are fresh (reported separately as a check).

  python final30.py --seed_from 0 --seeds 15 --out results_final30/a
  python final30.py --seed_from 15 --seeds 30 --out results_final30/b
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.svm import SVC
import xgboost as xgb
sys.path.insert(0, str(Path(__file__).parent))
from qml_pilink import load_links, build_pairs, balanced_sample, metrics
from qml_enhanced import EnhancedFeaturizer, quantum_models, classical_models, states, rank_eval
from bm25 import BM25

D = 12


def run(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    links = load_links(a.data); rows = []
    e5 = None
    if a.e5:  # precomputed e5 cosines (e5_pairs.py), added as one more input feature for every model
        t = pd.read_csv(a.e5); e5 = dict(zip(zip(t.iss.astype(int), t.pr.astype(int)), t.cos.astype(float)))
        print("e5 pairs", len(e5), flush=True)
    if a.split == "temporal":  # negatives and ranking candidates come from the same repo AND period
        links_t = links.copy()
        r = links_t.groupby("repo").t_pr_open.rank(method="first", pct=True)
        links_t["repo"] = links_t.repo + np.where(r > 0.7, "|te", "|tr")
    for seed in range(a.seed_from, a.seeds):
        if a.split == "temporal":  # per repo: oldest 70% of links (by PR open time) train, newest 30% test
            pairs, rank_sets = build_pairs(links_t, seed=seed)
            tr_i = np.where(pairs.repo.str.endswith("|tr"))[0]; te_i = np.where(pairs.repo.str.endswith("|te"))[0]
        else:
            pairs, rank_sets = build_pairs(links, seed=seed)
            tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(pairs, groups=pairs.repo))
        tr, te = balanced_sample(pairs.iloc[tr_i], a.n_train, seed), balanced_sample(pairs.iloc[te_i], a.n_test, seed)
        te_repos, tr_repos = set(te.repo), set(pairs.iloc[tr_i].repo)
        rs = [r for r in rank_sets if r[2] in te_repos]
        rs = [rs[i] for i in np.random.default_rng(seed).permutation(len(rs))[:a.n_rank]]
        rtr = [r for r in rank_sets if r[2] in tr_repos]
        rtr = [rtr[i] for i in np.random.default_rng(seed + 100).permutation(len(rtr))[:a.n_rank]]
        flat = lambda S: (np.concatenate([np.full(len(c), i) for i, c, _ in S]), np.concatenate([np.array(c) for _, c, _ in S]))
        iv, pv = flat(rs); ri, rp = flat(rtr); sizes = [len(c) for _, c, _ in rs]
        ytr, yte = tr.y.values, te.y.values
        cv = list(GroupKFold(3).split(np.zeros(len(tr)), ytr, tr.repo.values))
        bm = BM25().fit(links.pr.values[np.unique(pairs.iloc[tr_i].pr.values)])
        feat = EnhancedFeaturizer(D, "pls", True, seed, bm, e5).fit(links, tr.iss.values, tr.pr.values, ytr)
        R = lambda i, p: feat.reduced(links, i, p); F = lambda i, p: feat.full(links, i, p)
        Xtr, Xte, Xrk = R(tr.iss.values, tr.pr.values), R(te.iss.values, te.pr.values), R(iv, pv)

        res = {}
        for fm in ("Z", "ZZ1"):
            res.update(quantum_models(fm, Xtr, ytr, Xte, Xrk, cv, D))
        t0 = time.time()
        Str, Ste, Srk = states("Z", Xtr), states("Z", Xte), states("Z", Xrk)
        m = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(np.abs(Str @ Str.conj().T) ** 2, ytr)
        res["QSVC-Z(untuned)"] = (m.decision_function(np.abs(Ste @ Str.conj().T) ** 2),
                                  m.decision_function(np.abs(Srk @ Str.conj().T) ** 2), time.time() - t0, {})
        res.update(classical_models(Xtr, ytr, Xte, Xrk, cv, seed, D))
        Ftr, Fte, Frk = F(tr.iss.values, tr.pr.values), F(te.iss.values, te.pr.values), F(iv, pv)
        res.update(classical_models(Ftr, ytr, Fte, Frk, cv, seed, D, suffix="(full)"))
        t0 = time.time()  # pairwise ranker on all features, 1+19 candidate sets from training repos
        Xr = F(ri, rp); yr = np.concatenate([[1] + [0] * (len(c) - 1) for _, c, _ in rtr])
        qid = np.concatenate([np.full(len(c), k) for k, (_, c, _) in enumerate(rtr)])
        rk = xgb.XGBRanker(objective="rank:pairwise", n_estimators=300, max_depth=4, learning_rate=0.1,
                           subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=1).fit(Xr, yr, qid=qid)
        res["XGB-Ranker(full)"] = (rk.predict(Fte), rk.predict(Frk), time.time() - t0, {})

        for name, (s, srk, ft, params) in res.items():
            thr = np.median(s) if "Ranker" in name else (0.5 if "XGB" in name else 0)
            row = dict(seed=seed, d=D, model=name, fit_s=round(ft, 2), params=json.dumps(params),
                       **metrics(yte, (s > thr).astype(int), s), **rank_eval(srk, sizes))
            rows.append(row)
        print(seed, {r["model"]: round(r["mrr"], 3) for r in rows if r["seed"] == seed}, flush=True)
        pd.DataFrame(rows).to_csv(out / "raw_results.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="pilink_pairs_meta.csv"); p.add_argument("--out", default="results_final30")
    p.add_argument("--seed_from", type=int, default=0); p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--n_train", type=int, default=800); p.add_argument("--n_test", type=int, default=400)
    p.add_argument("--n_rank", type=int, default=200)
    p.add_argument("--split", default="repo", choices=["repo", "temporal"])
    p.add_argument("--e5", default=None, help="csv(.gz) with iss, pr, cos from e5_pairs.py")
    run(p.parse_args())
