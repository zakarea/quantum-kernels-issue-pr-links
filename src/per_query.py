"""Per-query reciprocal ranks for the confirmatory comparison, for dependence-aware statistics.

Rebuilds QSVC-Z, PQK-Z, QSVC-ZZ1 and SVC-RBF for all 30 seeds of a setting from the hyperparameters
saved by final30.py (so the models are exactly the ones reported) and records, for every ranking query,
the issue, its repository and the reciprocal rank of the true PR under each model. The per-seed MRR
is re-checked against the saved run.

  python per_query.py pilink    results_final30/a/raw_results.csv,results_final30/b/raw_results.csv
  python per_query.py ghpr      results_ghpr/raw_results.csv
  python per_query.py pilink-e5 results_e5/pilink/raw_results.csv
  python per_query.py ghpr-e5   results_e5/ghpr/raw_results.csv
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.svm import SVC
sys.path.insert(0, str(Path(__file__).parent))
from qml_pilink import load_links
from qml_enhanced import EnhancedFeaturizer, rank_eval, states, sqd
from quantum_extra import z_kernel, z_qubits, z_bloch
from splits import get_split, flat, temporal_links
from bm25 import BM25

D = 12
MODELS = ["QSVC-Z", "PQK-Z", "QSVC-ZZ1", "SVC-RBF"]


def run(setting, saved_files, out="results_review"):
    name = setting.split("-")[0]
    data, split = {"pilink": ("pilink_pairs_meta.csv", "repo"), "ghpr": ("ghpr_pairs_meta.csv", "temporal")}[name]
    links = load_links(data); lt = temporal_links(links) if split == "temporal" else None
    e5 = None
    if setting.endswith("e5"):
        t = pd.read_csv(f"e5_pairs_{name}.csv.gz"); e5 = dict(zip(zip(t.iss.astype(int), t.pr.astype(int)), t.cos.astype(float)))
    saved = pd.concat([pd.read_csv(f) for f in saved_files.split(",")])
    par = {(r.seed, r.model): json.loads(r.params) for r in saved.itertuples()}
    rows, check, done = [], [], set()
    fq, fc = Path(out) / f"per_query_{setting}.csv", Path(out) / f"per_query_check_{setting}.csv"
    if fq.exists() and fc.exists():  # resume: keep seeds already written
        rows, check = pd.read_csv(fq).to_dict("records"), pd.read_csv(fc).to_dict("records")
        done = set(pd.read_csv(fc).seed.unique())
    for seed in sorted(saved.seed.unique()):
        if seed in done:
            continue
        t0 = time.time()
        tr, te, rs, rtr, tr_all = get_split(links, seed, split, links_t=lt)
        iv, pv = flat(rs); sizes = [len(c) for _, c, _ in rs]; ytr = tr.y.values
        bm = BM25().fit(links.pr.values[np.unique(tr_all.pr.values)])
        feat = EnhancedFeaturizer(D, "pls", True, seed, bm, e5).fit(links, tr.iss.values, tr.pr.values, ytr)
        Xtr, Xrk = feat.reduced(links, tr.iss.values, tr.pr.values), feat.reduced(links, iv, pv)
        S = {}
        p = par[(seed, "QSVC-Z")]; lam = p["lam"]; q = z_qubits(lam * Xtr)
        m = SVC(kernel="precomputed", C=p["C"], class_weight="balanced").fit(z_kernel(lam * Xtr, lam * Xtr, q, q), ytr)
        S["QSVC-Z"] = m.decision_function(z_kernel(lam * Xrk, lam * Xtr, None, q))
        p = par[(seed, "PQK-Z")]; lam = p["lam"]; Bt = z_bloch(lam * Xtr); D2 = sqd(Bt, Bt)
        g = p["g"] / (np.median(D2[~np.eye(len(D2), dtype=bool)]) + 1e-12)
        m = SVC(kernel="precomputed", C=p["C"], class_weight="balanced").fit(np.exp(-g * D2), ytr)
        S["PQK-Z"] = m.decision_function(np.exp(-g * sqd(z_bloch(lam * Xrk), Bt)))
        p = par[(seed, "QSVC-ZZ1")]; lam = p["lam"]; St = states("ZZ1", lam * Xtr)
        m = SVC(kernel="precomputed", C=p["C"], class_weight="balanced").fit(np.abs(St @ St.conj().T) ** 2, ytr)
        S["QSVC-ZZ1"] = m.decision_function(np.abs(states("ZZ1", lam * Xrk) @ St.conj().T) ** 2)
        p = par[(seed, "SVC-RBF")]
        S["SVC-RBF"] = SVC(C=p["C"], gamma=p["gamma"], class_weight="balanced").fit(Xtr, ytr).decision_function(Xrk)
        pos = np.cumsum([0] + sizes)
        for k, (i, cands, repo) in enumerate(rs):
            row = dict(setting=setting, seed=int(seed), repo=repo.split("|")[0], issue=int(i))
            for mn in MODELS:
                ss = S[mn][pos[k]:pos[k + 1]]; row[f"rr_{mn}"] = 1.0 / ((ss > ss[0]).sum() + 1)
            rows.append(row)
        for mn in MODELS:
            check.append(dict(setting=setting, seed=int(seed), model=mn, mrr_rebuilt=rank_eval(S[mn], sizes)["mrr"],
                              mrr_saved=float(saved[(saved.seed == seed) & (saved.model == mn)].mrr.iloc[0])))
        print(setting, seed, round(time.time() - t0), "s", flush=True)
        pd.DataFrame(rows).to_csv(Path(out) / f"per_query_{setting}.csv", index=False)
        pd.DataFrame(check).to_csv(Path(out) / f"per_query_check_{setting}.csv", index=False)


if __name__ == "__main__":
    Path("results_review").mkdir(exist_ok=True)
    run(sys.argv[1], sys.argv[2])
