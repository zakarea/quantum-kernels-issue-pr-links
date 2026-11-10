"""Enhanced study: fair tuning, kernel bandwidth, projected quantum kernels,
supervised reduction (PLS) and open-time metadata features.

Same data, pairs, repo-level splits and seeds as qml_pilink.py, so results are
directly comparable with the main run. All hyperparameters are chosen by
3-fold GroupKFold (by repository) inside the training set, scored by AUC.

  --reducer pca|pls     unsupervised PCA (as before) or supervised PLS to d dims
  --meta                add open-time features (PR opened vs issue opened)

Quantum models (exact statevector simulation):
  QSVC-<map>      fidelity kernel |<phi(lx)|phi(ly)>|^2, bandwidth l and C tuned
  PQK-<map>       projected kernel exp(-g * sum_k ||rho_k(x) - rho_k(y)||^2) over
                  single-qubit reduced states (Huang et al. 2021), l, g, C tuned
  QSVC-Z(untuned) l = 1, C = 1: the configuration used in the main study
"""
import argparse, time, sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
sys.path.insert(0, str(Path(__file__).parent))
from qml_pilink import (load_links, build_pairs, balanced_sample, PairFeaturizer, metrics, feature_map)

LAMS, CS, GS = [0.025, 0.05, 0.1, 0.25, 0.5, 1.0], [0.1, 1.0, 10.0], [0.5, 1.0, 2.0]


# ---------------------------------------------------------------- features
class EnhancedFeaturizer(PairFeaturizer):
    def __init__(self, d, reducer="pca", meta=False, seed=0, bm25=None, e5=None):
        # e5: optional dict {(issue_row, pr_row): cosine} of precomputed e5-base-v2 similarities
        super().__init__(d, seed=seed); self.reducer, self.meta, self.bm25, self.e5 = reducer, meta, bm25, e5

    def raw(self, links, iss, pr):
        R = super().raw(links, iss, pr)
        if self.bm25 is not None:  # BM25(issue -> PR), corpus = PR texts of training repos only
            R = np.hstack([R, np.log1p(self.bm25.score(links.issue.values[iss], links.pr.values[pr]))[:, None]])
        if self.e5 is not None:
            R = np.hstack([R, np.array([self.e5[(int(i), int(p))] for i, p in zip(iss, pr)])[:, None]])
        if not self.meta:
            return R
        dt = links.t_pr_open.values[pr] - links.t_issue_open.values[iss]  # days, PR opened relative to issue
        dt = np.nan_to_num(dt, nan=0.0)
        return np.hstack([R, (np.sign(dt) * np.log1p(np.abs(dt)))[:, None], (dt > 0).astype(float)[:, None]])

    def fit(self, links, iss, pr, y):
        texts = np.concatenate([links.issue.values[iss], links.pr.values[pr]])
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self.tfidf = TfidfVectorizer(max_features=20000, sublinear_tf=True, token_pattern=r"(?u)\b\w\w+\b").fit(texts)
        self.svd = TruncatedSVD(self.svd_dim, random_state=self.seed).fit(self.tfidf.transform(texts))
        R = self.raw(links, iss, pr); self.std = StandardScaler().fit(R); Z = self.std.transform(R)
        self.red = (PCA(self.d, random_state=self.seed).fit(Z) if self.reducer == "pca"
                    else PLSRegression(self.d, scale=False).fit(Z, y))
        self.mm = MinMaxScaler((0, np.pi)).fit(self.red.transform(Z))
        return self

    def reduced(self, links, iss, pr):
        return np.clip(self.mm.transform(self.red.transform(self.full(links, iss, pr))), 0, np.pi)


# ---------------------------------------------------------------- fast exact statevectors
def _fwht(S):  # H on every qubit, batched: S is (N, 2^n)
    N, D = S.shape; h = 1
    S = S.copy()
    while h < D:
        S = S.reshape(N, -1, 2, h)
        a, b = S[:, :, 0, :].copy(), S[:, :, 1, :].copy()
        S[:, :, 0, :], S[:, :, 1, :] = a + b, a - b
        S = S.reshape(N, D); h *= 2
    return S / np.sqrt(D)


def _bits(n):
    z = np.arange(2 ** n)
    return ((z[:, None] >> np.arange(n)) & 1).astype(float)  # (2^n, n), qubit k = bit k (Qiskit order)


def states(fm, X):
    N, n = X.shape
    if fm == "Pauli":  # non-diagonal YY terms: use Qiskit's simulator
        from qiskit.quantum_info import Statevector
        c = feature_map("Pauli", n)
        return np.array([Statevector(c.assign_parameters(x)).data for x in X])
    B = _bits(n)
    phase = 2 * X @ B.T  # single-qubit P(2x) phases
    reps = 2 if fm == "Z" else 1
    if fm == "ZZ1":
        for i in range(n - 1):  # linear entanglement: phase 2(pi-xi)(pi-xj) when bits differ
            par = (B[:, i] != B[:, i + 1]).astype(float)
            phase = phase + 2 * np.outer((np.pi - X[:, i]) * (np.pi - X[:, i + 1]), par)
    S = np.full((N, 2 ** n), 2 ** (-n / 2), dtype=complex) * np.exp(1j * phase)
    for _ in range(reps - 1):
        S = _fwht(S) * np.exp(1j * phase)
    return S


def bloch(S, n):
    N = len(S); out = []
    for k in range(n):
        P = S.reshape(N, 2 ** (n - k - 1), 2, 2 ** k)
        rho = np.einsum("nasb,natb->nst", P, P.conj())
        out += [2 * rho[:, 0, 1].real, 2 * rho[:, 1, 0].imag, (rho[:, 0, 0] - rho[:, 1, 1]).real]
    return np.stack(out, 1)


def sqd(A, B):
    return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T, 0)


def cv_auc(K, y, cv, C):
    a = []
    for tr, va in cv:
        m = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(K[np.ix_(tr, tr)], y[tr])
        a.append(roc_auc_score(y[va], m.decision_function(K[np.ix_(va, tr)])))
    return float(np.mean(a))


# ---------------------------------------------------------------- evaluation helpers
def rank_eval(S, rank_sizes):
    h1, h5, rr, pos = [], [], [], 0
    for L in rank_sizes:
        s = S[pos:pos + L]; pos += L; r = int((s > s[0]).sum())
        h1.append(r < 1); h5.append(r < 5); rr.append(1 / (r + 1))
    return dict(hit1=np.mean(h1), hit5=np.mean(h5), mrr=np.mean(rr))


def quantum_models(fm, Xtr, ytr, Xte, Xrk, cv, n):
    """Tune bandwidth l (+ C) for QSVC, and l, g (+ C) for PQK, on training CV only."""
    best_q, best_p = (-1, None), (-1, None)
    for lam in LAMS:
        S = states(fm, lam * Xtr); K = np.abs(S @ S.conj().T) ** 2
        for C in CS:
            a = cv_auc(K, ytr, cv, C)
            if a > best_q[0]: best_q = (a, dict(lam=lam, C=C))
        Bt = bloch(S, n); D2 = sqd(Bt, Bt); med = np.median(D2[~np.eye(len(D2), dtype=bool)]) + 1e-12
        for g in GS:
            Kp = np.exp(-(g / med) * D2)
            for C in CS:
                a = cv_auc(Kp, ytr, cv, C)
                if a > best_p[0]: best_p = (a, dict(lam=lam, g=g, C=C, med=med))
    out = {}
    for kind, (_, p) in (("QSVC", best_q), ("PQK", best_p)):
        t0 = time.time()
        Str, Ste, Srk = (states(fm, p["lam"] * X) for X in (Xtr, Xte, Xrk))
        if kind == "QSVC":
            kern = lambda A, B: np.abs(A @ B.conj().T) ** 2
        else:
            Btr = bloch(Str, n); g = p["g"] / p["med"]
            kern = lambda A, B, Btr=Btr, g=g: np.exp(-g * sqd(bloch(A, n), Btr))
        Ktr = np.abs(Str @ Str.conj().T) ** 2 if kind == "QSVC" else np.exp(-(p["g"] / p["med"]) * sqd(Btr, Btr))
        m = SVC(kernel="precomputed", C=p["C"], class_weight="balanced").fit(Ktr, ytr)
        out[f"{kind}-{fm}"] = (m.decision_function(kern(Ste, Str)), m.decision_function(kern(Srk, Str)),
                               time.time() - t0, {k: round(v, 4) for k, v in p.items() if k != "med"})
    return out


def classical_models(Xtr, ytr, Xte, Xrk, cv, seed, d, suffix=""):
    import xgboost as xgb
    v = Xtr.var() * Xtr.shape[1]
    specs = {
        "SVC-RBF": (SVC(class_weight="balanced"), {"C": CS, "gamma": [g / v for g in (0.3, 1, 3)]}),
        "LogReg": (LogisticRegression(max_iter=3000, class_weight="balanced"), {"C": [0.01, 0.1, 1, 10]}),
        "XGB": (xgb.XGBClassifier(learning_rate=0.1, random_state=seed, eval_metric="logloss", n_jobs=1),
                {"max_depth": [2, 4], "n_estimators": [100, 300]}),
    }
    out = {}
    for name, (est, grid) in specs.items():
        if suffix and name == "LogReg": continue
        t0 = time.time()
        gs = GridSearchCV(est, grid, cv=cv, scoring="roc_auc", n_jobs=1).fit(Xtr, ytr)
        f = (lambda X: gs.decision_function(X)) if hasattr(gs.best_estimator_, "decision_function") else (lambda X: gs.predict_proba(X)[:, 1])
        out[name + suffix] = (f(Xte), f(Xrk), time.time() - t0,
                              {k: (round(v, 5) if isinstance(v, float) else v) for k, v in gs.best_params_.items()})
    return out


def run(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    setting = f"{a.reducer.upper()}{'+meta' if a.meta else ''}{'+BM25' if a.bm25 else ''}"
    links = load_links(a.data); rows = []
    print(setting, "links", len(links), flush=True)
    for seed in range(a.seed_from, a.seeds):
        pairs, rank_sets = build_pairs(links, seed=seed)
        tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(pairs, groups=pairs.repo))
        tr, te = balanced_sample(pairs.iloc[tr_i], a.n_train, seed), balanced_sample(pairs.iloc[te_i], a.n_test, seed)
        rs = [r for r in rank_sets if r[2] in set(te.repo)]
        rs = [rs[i] for i in np.random.default_rng(seed).permutation(len(rs))[:a.n_rank]]
        iv = np.concatenate([np.full(len(c), i) for i, c, _ in rs]); pv = np.concatenate([np.array(c) for _, c, _ in rs])
        sizes = [len(c) for _, c, _ in rs]
        ytr, yte = tr.y.values, te.y.values
        cv = list(GroupKFold(3).split(np.zeros(len(tr)), ytr, tr.repo.values))
        bm = None
        if a.bm25:
            from bm25 import BM25
            bm = BM25().fit(links.pr.values[np.unique(pairs.iloc[tr_i].pr.values)])
        for d in a.dims:
            feat = EnhancedFeaturizer(d, a.reducer, a.meta, seed, bm).fit(links, tr.iss.values, tr.pr.values, ytr)
            R = lambda i, p: feat.reduced(links, i, p)
            Xtr, Xte, Xrk = R(tr.iss.values, tr.pr.values), R(te.iss.values, te.pr.values), R(iv, pv)
            res = {}
            for fm in a.fmaps:
                res.update(quantum_models(fm, Xtr, ytr, Xte, Xrk, cv, d))
            t0 = time.time()  # untuned reference: the main-study configuration
            Str, Ste, Srk = states("Z", Xtr), states("Z", Xte), states("Z", Xrk)
            m = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(np.abs(Str @ Str.conj().T) ** 2, ytr)
            res["QSVC-Z(untuned)"] = (m.decision_function(np.abs(Ste @ Str.conj().T) ** 2),
                                      m.decision_function(np.abs(Srk @ Str.conj().T) ** 2), time.time() - t0, {})
            res.update(classical_models(Xtr, ytr, Xte, Xrk, cv, seed, d))
            if d == max(a.dims):
                F = lambda i, p: feat.full(links, i, p)
                res.update(classical_models(F(tr.iss.values, tr.pr.values), ytr, F(te.iss.values, te.pr.values),
                                            F(iv, pv), cv, seed, d, suffix="(full)"))
            for name, (s, srk, ft, params) in res.items():
                row = dict(setting=setting, seed=seed, d=d, model=name, fit_s=round(ft, 3), params=json.dumps(params),
                           **metrics(yte, (s > 0).astype(int) if "XGB" not in name else (s > 0.5).astype(int), s),
                           **rank_eval(srk, sizes))
                rows.append(row)
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            pd.DataFrame(rows).to_csv(out / "raw_results.csv", index=False)
    df = pd.DataFrame(rows)
    df.groupby(["d", "model"])[["f1", "auc", "hit1", "hit5", "mrr", "fit_s"]].agg(["mean", "std"]).round(3).to_csv(out / "summary.csv")
    json.dump(vars(a), open(out / "config.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="pilink_pairs_meta.csv"); p.add_argument("--out", default="results_enh")
    p.add_argument("--reducer", default="pca", choices=["pca", "pls"]); p.add_argument("--meta", action="store_true")
    p.add_argument("--bm25", action="store_true", help="add BM25(issue->PR) as an input feature")
    p.add_argument("--seeds", type=int, default=7); p.add_argument("--seed_from", type=int, default=0); p.add_argument("--dims", type=int, nargs="+", default=[4, 8, 12])
    p.add_argument("--fmaps", nargs="+", default=["Z", "ZZ1", "Pauli"])
    p.add_argument("--n_train", type=int, default=800); p.add_argument("--n_test", type=int, default=400)
    p.add_argument("--n_rank", type=int, default=200)
    run(p.parse_args())
