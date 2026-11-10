"""Item 2: realistic retrieval. Rank the true PR among ALL PRs of the same repository (and period,
for GHPR) opened from 7 days before to 365 days after the issue, instead of 19 sampled candidates.

Same test issues (the 200 ranking queries per seed), training data and tuned hyperparameters as the
30-seed runs without e5 (results_final30 for PI-Link, results_ghpr for GHPR); e5 is not used because
its similarities were only precomputed for the sampled candidates. Pools are capped at 500 negatives.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
sys.path.insert(0, str(Path(__file__).parent))
from qml_pilink import load_links
from qml_enhanced import EnhancedFeaturizer, sqd
from quantum_extra import z_kernel, z_qubits, z_bloch
from splits import get_split, flat, temporal_links
from bm25 import BM25

D, CAP, LO, HI = 12, 500, -7.0, 365.0
DATASETS = [("pilink", "pilink_pairs_meta.csv", "repo", ["results_final30/a/raw_results.csv", "results_final30/b/raw_results.csv"]),
            ("ghpr", "ghpr_pairs_meta.csv", "temporal", ["results_ghpr/raw_results.csv"])]


def pools(links, groups_, queries, seed):
    keys, itxt = links.pr_key.to_numpy(), links.issue.to_numpy()
    to, po = links.t_issue_open.to_numpy(), links.t_pr_open.to_numpy()
    by_group = {g: idx.to_numpy() for g, idx in pd.Series(links.index).groupby(groups_.to_numpy())}
    rng = np.random.default_rng(seed + 7); out = []
    for i in queries:
        idx = by_group[groups_.iloc[i]]
        ok = (keys[idx] != keys[i]) & (itxt[idx] != itxt[i]) & (po[idx] - to[i] >= LO) & (po[idx] - to[i] <= HI)
        cand = idx[ok]
        _, first = np.unique(keys[cand], return_index=True); cand = cand[np.sort(first)]  # one row per distinct PR
        if len(cand) > CAP: cand = rng.choice(cand, CAP, replace=False)
        out.append(np.r_[i, cand])
    return out


def chunked(fn, X, step=20000):
    return np.concatenate([fn(X[s:s + step]) for s in range(0, len(X), step)])


def ranks(S, sizes):
    pos = np.cumsum([0] + sizes); r = []
    for k in range(len(sizes)):
        s = S[pos[k]:pos[k + 1]]; r.append(int((s > s[0]).sum()) + 1)
    return np.array(r)


def run(seeds, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); rows = []
    for name, data, split, saved_files in DATASETS:
        links = load_links(data); lt = temporal_links(links) if split == "temporal" else None
        groups_ = (lt if lt is not None else links).repo
        saved = pd.concat([pd.read_csv(f) for f in saved_files])
        par = {(r.seed, r.model): json.loads(r.params) for r in saved.itertuples()}
        for seed in range(seeds):
            t0 = time.time()
            tr, te, rs, rtr, tr_all = get_split(links, seed, split, links_t=lt)
            rtr = rtr[:200]; ri, rp = flat(rtr); ytr = tr.y.values
            P = pools(links, groups_, [i for i, _, _ in rs], seed); sizes = [len(c) for c in P]
            iv = np.concatenate([np.full(len(c), c[0]) for c in P]); pv = np.concatenate(P)
            bm = BM25().fit(links.pr.values[np.unique(tr_all.pr.values)])
            feat = EnhancedFeaturizer(D, "pls", True, seed, bm).fit(links, tr.iss.values, tr.pr.values, ytr)
            Xtr = feat.reduced(links, tr.iss.values, tr.pr.values); Xrk = feat.reduced(links, iv, pv)
            Frk = feat.full(links, iv, pv); S = {}
            S["BM25"] = bm.score(links.issue.values[iv], links.pr.values[pv])
            p = par[(seed, "QSVC-Z")]; lam = p["lam"]; q = z_qubits(lam * Xtr)
            m = SVC(kernel="precomputed", C=p["C"], class_weight="balanced").fit(z_kernel(lam * Xtr, lam * Xtr, q, q), ytr)
            S["QSVC-Z"] = chunked(lambda X: m.decision_function(z_kernel(lam * X, lam * Xtr, None, q)), Xrk)
            p = par[(seed, "PQK-Z")]; lp = p["lam"]; Bt = z_bloch(lp * Xtr); D2 = sqd(Bt, Bt)
            g = p["g"] / (np.median(D2[~np.eye(len(D2), dtype=bool)]) + 1e-12)
            m2 = SVC(kernel="precomputed", C=p["C"], class_weight="balanced").fit(np.exp(-g * D2), ytr)
            S["PQK-Z"] = chunked(lambda X: m2.decision_function(np.exp(-g * sqd(z_bloch(lp * X), Bt))), Xrk)
            p = par[(seed, "SVC-RBF")]
            m3 = SVC(C=p["C"], gamma=p["gamma"], class_weight="balanced").fit(Xtr, ytr); S["SVC-RBF"] = chunked(m3.decision_function, Xrk)
            p = par[(seed, "LogReg")]
            S["LogReg"] = LogisticRegression(C=p["C"], max_iter=3000, class_weight="balanced").fit(Xtr, ytr).decision_function(Xrk)
            yr = np.concatenate([[1] + [0] * (len(c) - 1) for _, c, _ in rtr])
            qid = np.concatenate([np.full(len(c), k) for k, (_, c, _) in enumerate(rtr)])
            rk = xgb.XGBRanker(objective="rank:pairwise", n_estimators=300, max_depth=4, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=1).fit(feat.full(links, ri, rp), yr, qid=qid)
            S["XGB-Ranker(full)"] = rk.predict(Frk)
            gaps = links.t_pr_open.values[[c[0] for c in P]] - links.t_issue_open.values[[c[0] for c in P]]
            for mname, s in S.items():
                r = ranks(s, sizes)
                rows.append(dict(dataset=name, seed=seed, model=mname, queries=len(sizes), mean_pool=float(np.mean(sizes)),
                                 median_pool=float(np.median(sizes)), true_in_window=float(np.mean((gaps >= LO) & (gaps <= HI))),
                                 hit1=float(np.mean(r <= 1)), hit5=float(np.mean(r <= 5)), hit10=float(np.mean(r <= 10)), mrr=float(np.mean(1 / r))))
            print(name, seed, round(time.time() - t0), "s pool", round(np.mean(sizes)), {m: round(x["mrr"], 3) for m, x in [(r_["model"], r_) for r_ in rows[-len(S):]]}, flush=True)
            pd.DataFrame(rows).to_csv(out / "realistic_results.csv", index=False)


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 10, sys.argv[2] if len(sys.argv) > 2 else "results_realistic")
