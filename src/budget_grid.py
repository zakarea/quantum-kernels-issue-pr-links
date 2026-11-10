"""Tuning-budget sensitivity: quantum kernels tuned with a grid the size of the SVC-RBF grid (9 configurations).

  QSVC-Z(9): lambda in {0.05, 0.25, 1} x C in {0.1, 1, 10}
  PQK-Z(9):  lambda in {0.05, 0.25, 1} x C in {0.1, 1, 10}, g = 1 (gamma = 1 / median distance)

Same seeds, splits, features and grouped CV as final30.py; compared with the saved SVC-RBF of that run.

  python budget_grid.py pilink results_final30/a/raw_results.csv,results_final30/b/raw_results.csv
  python budget_grid.py ghpr   results_ghpr/raw_results.csv
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
sys.path.insert(0, str(Path(__file__).parent))
from qml_pilink import load_links, metrics
from qml_enhanced import EnhancedFeaturizer, rank_eval, sqd, cv_auc
from quantum_extra import z_kernel, z_qubits, z_bloch
from splits import get_split, flat, temporal_links
from bm25 import BM25

D, LAMS9, CS = 12, [0.05, 0.25, 1.0], [0.1, 1.0, 10.0]


def fit_eval(Ktr, Kte, Krk, ytr, yte, sizes, C):
    m = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(Ktr, ytr); s = m.decision_function(Kte)
    return dict(**metrics(yte, (s > 0).astype(int), s), **rank_eval(m.decision_function(Krk), sizes))


def run(name, saved_files, out="results_review"):
    data, split = {"pilink": ("pilink_pairs_meta.csv", "repo"), "ghpr": ("ghpr_pairs_meta.csv", "temporal")}[name]
    links = load_links(data); lt = temporal_links(links) if split == "temporal" else None
    seeds = sorted(pd.concat([pd.read_csv(f) for f in saved_files.split(",")]).seed.unique()); rows, done = [], set()
    f = Path(out) / f"budget_grid_{name}.csv"
    if f.exists():  # resume
        rows = pd.read_csv(f).to_dict("records"); done = {r["seed"] for r in rows}
    for seed in seeds:
        if seed in done:
            continue
        t0 = time.time()
        tr, te, rs, rtr, tr_all = get_split(links, seed, split, links_t=lt)
        iv, pv = flat(rs); sizes = [len(c) for _, c, _ in rs]; ytr, yte = tr.y.values, te.y.values
        cv = list(GroupKFold(3).split(np.zeros(len(tr)), ytr, tr.repo.values))
        bm = BM25().fit(links.pr.values[np.unique(tr_all.pr.values)])
        feat = EnhancedFeaturizer(D, "pls", True, seed, bm).fit(links, tr.iss.values, tr.pr.values, ytr)
        Xtr, Xte, Xrk = (feat.reduced(links, a, b) for a, b in [(tr.iss.values, tr.pr.values), (te.iss.values, te.pr.values), (iv, pv)])
        # QSVC-Z with 9 configurations
        best = max(((cv_auc(z_kernel(l * Xtr, l * Xtr), ytr, cv, c), l, c) for l in LAMS9 for c in CS))
        _, l, c = best; q = z_qubits(l * Xtr)
        r = fit_eval(z_kernel(l * Xtr, l * Xtr, q, q), z_kernel(l * Xte, l * Xtr, None, q), z_kernel(l * Xrk, l * Xtr, None, q), ytr, yte, sizes, c)
        rows.append(dict(dataset=name, seed=int(seed), model="QSVC-Z(9)", lam=l, C=c, **r))
        # PQK-Z with 9 configurations
        cand = []
        for l in LAMS9:
            Bt = z_bloch(l * Xtr); D2 = sqd(Bt, Bt); g = 1.0 / (np.median(D2[~np.eye(len(D2), dtype=bool)]) + 1e-12); K = np.exp(-g * D2)
            cand += [(cv_auc(K, ytr, cv, c), l, c, g) for c in CS]
        _, l, c, g = max(cand, key=lambda t: t[0]); Bt = z_bloch(l * Xtr)
        r = fit_eval(np.exp(-g * sqd(Bt, Bt)), np.exp(-g * sqd(z_bloch(l * Xte), Bt)), np.exp(-g * sqd(z_bloch(l * Xrk), Bt)), ytr, yte, sizes, c)
        rows.append(dict(dataset=name, seed=int(seed), model="PQK-Z(9)", lam=l, C=c, **r))
        print(name, seed, round(time.time() - t0), "s", [round(x["mrr"], 3) for x in rows[-2:]], flush=True)
        pd.DataFrame(rows).to_csv(Path(out) / f"budget_grid_{name}.csv", index=False)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
