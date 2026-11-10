"""Items 1, 3, 4: kernel similarity, what drives the links, and error analysis.

Rebuilds the final models (d = 12, PLS + metadata + BM25 + e5) for seeds 0-9 of each dataset from the
hyperparameters tuned in the 30-seed run (results_e5/*/raw_results.csv), so the models are the same
ones reported there (MRR is re-checked against the saved run).

  1. Centred kernel alignment (CKA) between training Gram matrices: quantum vs SVC-RBF vs linear,
     plus Spearman correlation of test-candidate scores between models.
  3. TreeSHAP (XGBoost pred_contribs) of the XGB-Ranker, aggregated by feature group.
  4. Per-query ranks of the main models with query characteristics, for error analysis.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
sys.path.insert(0, str(Path(__file__).parent))
from qml_pilink import load_links, metrics
from qml_enhanced import EnhancedFeaturizer, rank_eval, states, sqd
from quantum_extra import z_kernel, z_qubits, z_bloch
from splits import get_split, flat, temporal_links
from bm25 import BM25

D = 12
DATASETS = [("pilink", "pilink_pairs_meta.csv", "repo"), ("ghpr", "ghpr_pairs_meta.csv", "temporal")]


def cka(K, L):
    n = len(K); H = np.eye(n) - 1.0 / n
    Kc, Lc = H @ K @ H, H @ L @ H
    return float((Kc * Lc).sum() / (np.linalg.norm(Kc) * np.linalg.norm(Lc)))


def groups(n_feat):
    g = ["TF-IDF cosine", "Jaccard"] + ["TF-IDF a*b"] * 64 + ["TF-IDF |a-b|"] * 64 + ["BM25", "e5 cosine", "Time: PR vs issue open", "Time: PR vs issue open"]
    assert len(g) == n_feat, (len(g), n_feat)
    return np.array(g)


def run(seeds, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cka_rows, q_rows, shap_rows, check = [], [], [], []
    for name, data, split in DATASETS:
        links = load_links(data); lt = temporal_links(links) if split == "temporal" else None
        t = pd.read_csv(f"e5_pairs_{name}.csv.gz"); e5 = dict(zip(zip(t.iss.astype(int), t.pr.astype(int)), t.cos.astype(float)))
        saved = pd.read_csv(f"results_e5/{name}/raw_results.csv")
        par = {(r.seed, r.model): json.loads(r.params) for r in saved.itertuples()}
        for seed in range(seeds):
            t0 = time.time()
            tr, te, rs, rtr, tr_all = get_split(links, seed, split, links_t=lt)
            rtr = rtr[:200]; iv, pv = flat(rs); ri, rp = flat(rtr); sizes = [len(c) for _, c, _ in rs]
            ytr, yte = tr.y.values, te.y.values
            bm = BM25().fit(links.pr.values[np.unique(tr_all.pr.values)])
            feat = EnhancedFeaturizer(D, "pls", True, seed, bm, e5).fit(links, tr.iss.values, tr.pr.values, ytr)
            R = lambda i, p: feat.reduced(links, i, p); F = lambda i, p: feat.full(links, i, p)
            Xtr, Xte, Xrk = R(tr.iss.values, tr.pr.values), R(te.iss.values, te.pr.values), R(iv, pv)
            K, S = {}, {}  # training Gram matrices and test-candidate scores per model
            p = par[(seed, "QSVC-Z")]; lam = p["lam"]; q = z_qubits(lam * Xtr)
            K["QSVC-Z"] = z_kernel(lam * Xtr, lam * Xtr, q, q)
            m = SVC(kernel="precomputed", C=p["C"], class_weight="balanced").fit(K["QSVC-Z"], ytr)
            S["QSVC-Z"] = m.decision_function(z_kernel(lam * Xrk, lam * Xtr, None, q))
            K["QSVC-Z(untuned)"] = z_kernel(Xtr, Xtr)
            p = par[(seed, "PQK-Z")]; lam = p["lam"]; Bt = z_bloch(lam * Xtr); D2 = sqd(Bt, Bt)
            g = p["g"] / (np.median(D2[~np.eye(len(D2), dtype=bool)]) + 1e-12); K["PQK-Z"] = np.exp(-g * D2)
            m = SVC(kernel="precomputed", C=p["C"], class_weight="balanced").fit(K["PQK-Z"], ytr)
            S["PQK-Z"] = m.decision_function(np.exp(-g * sqd(z_bloch(lam * Xrk), Bt)))
            p = par[(seed, "QSVC-ZZ1")]; lam = p["lam"]; St = states("ZZ1", lam * Xtr)
            K["QSVC-ZZ1"] = np.abs(St @ St.conj().T) ** 2
            m = SVC(kernel="precomputed", C=p["C"], class_weight="balanced").fit(K["QSVC-ZZ1"], ytr)
            S["QSVC-ZZ1"] = m.decision_function(np.abs(states("ZZ1", lam * Xrk) @ St.conj().T) ** 2)
            p = par[(seed, "SVC-RBF")]; K["SVC-RBF"] = np.exp(-p["gamma"] * sqd(Xtr, Xtr))
            m = SVC(C=p["C"], gamma=p["gamma"], class_weight="balanced").fit(Xtr, ytr); S["SVC-RBF"] = m.decision_function(Xrk)
            K["Linear"] = Xtr @ Xtr.T
            p = par[(seed, "LogReg")]
            m = LogisticRegression(C=p["C"], max_iter=3000, class_weight="balanced").fit(Xtr, ytr); S["LogReg"] = m.decision_function(Xrk)
            yr = np.concatenate([[1] + [0] * (len(c) - 1) for _, c, _ in rtr])
            qid = np.concatenate([np.full(len(c), k) for k, (_, c, _) in enumerate(rtr)])
            rk = xgb.XGBRanker(objective="rank:pairwise", n_estimators=300, max_depth=4, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=1).fit(F(ri, rp), yr, qid=qid)
            Frk = F(iv, pv); S["XGB-Ranker(full)"] = rk.predict(Frk)
            for mname, s in S.items():  # re-check against the saved 30-seed run
                check.append(dict(dataset=name, seed=seed, model=mname, mrr_rebuilt=rank_eval(s, sizes)["mrr"],
                                  mrr_saved=float(saved[(saved.seed == seed) & (saved.model == mname)].mrr.iloc[0])))
            # 1. kernel similarity
            for a_, b_ in [("QSVC-Z", "SVC-RBF"), ("PQK-Z", "SVC-RBF"), ("QSVC-ZZ1", "SVC-RBF"), ("QSVC-Z", "PQK-Z"),
                           ("QSVC-Z(untuned)", "SVC-RBF"), ("SVC-RBF", "Linear"), ("QSVC-Z", "Linear")]:
                cka_rows.append(dict(dataset=name, seed=seed, A=a_, B=b_, cka=cka(K[a_], K[b_])))
            pos = np.cumsum([0] + sizes)
            for a_, b_ in [("QSVC-Z", "SVC-RBF"), ("PQK-Z", "SVC-RBF"), ("QSVC-ZZ1", "SVC-RBF"), ("PQK-Z", "XGB-Ranker(full)"), ("SVC-RBF", "XGB-Ranker(full)")]:
                rho = [spearmanr(S[a_][pos[k]:pos[k + 1]], S[b_][pos[k]:pos[k + 1]]).correlation for k in range(len(sizes))]
                top = [np.argmax(S[a_][pos[k]:pos[k + 1]]) == np.argmax(S[b_][pos[k]:pos[k + 1]]) for k in range(len(sizes))]
                cka_rows.append(dict(dataset=name, seed=seed, A=a_, B=b_, score_spearman=float(np.nanmean(rho)), same_top1=float(np.mean(top))))
            # 3. TreeSHAP of the ranker on test candidates, aggregated by feature group
            contrib = rk.get_booster().predict(xgb.DMatrix(Frk), pred_contribs=True)[:, :-1]
            gr = groups(Frk.shape[1]); imp = pd.Series(np.abs(contrib).mean(0)).groupby(gr).sum()
            for k, v in imp.items():
                shap_rows.append(dict(dataset=name, seed=seed, group=k, mean_abs_shap=float(v)))
            # 4. per-query error analysis
            for k, (i, cands, repo) in enumerate(rs):
                row = dict(dataset=name, seed=seed, repo=repo.split("|")[0], issue=int(i), pool=len(cands),
                           issue_words=len(links.issue.values[i].split()), pr_words=len(links.pr.values[i].split()),
                           gap_days=float(links.t_pr_open.values[i] - links.t_issue_open.values[i]),
                           bm25_true=float(Frk[pos[k], 130]), e5_true=float(e5[(int(i), int(i))]))
                for mname, s in S.items():
                    ss = s[pos[k]:pos[k + 1]]; row[f"rank_{mname}"] = int((ss > ss[0]).sum()) + 1
                q_rows.append(row)
            print(name, seed, round(time.time() - t0), "s", {m: round(rank_eval(s, sizes)["mrr"], 3) for m, s in S.items()}, flush=True)
            for fn, rows_ in [("cka.csv", cka_rows), ("queries.csv", q_rows), ("shap.csv", shap_rows), ("rebuild_check.csv", check)]:
                pd.DataFrame(rows_).to_csv(out / fn, index=False)


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 10, sys.argv[2] if len(sys.argv) > 2 else "results_analysis")
