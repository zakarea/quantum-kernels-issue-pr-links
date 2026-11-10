"""
Quantum vs classical ML for issue-PR link recovery (PI-Link).

RQ1  QSVC / VQC vs classical models at the same reduced dimension
RQ2  effect of qubit count (d) and feature map
RQ3  ideal vs noisy simulation (small subset)
RQ4  training / inference cost
Task B  ranking: true PR vs K hard negatives -> Hit@1, Hit@5, MRR

Usage:
  python qml_pilink.py --data pilink.csv --out results/
  python qml_pilink.py --synthetic --out results_smoke/      (pipeline check)
"""
import argparse, json, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- data
COLS = {  # candidate column names, first match wins (edit after seeing real schema)
    "repo": ["repo", "repository", "repo_name", "project", "full_name"],
    "issue_title": ["issue_title", "title_issue", "issue title"],
    "issue_body": ["issue_body", "body_issue", "issue body"],
    "pr_title": ["pr_title", "title_pr", "pull_title", "pr title"],
    "pr_body": ["pr_body", "body_pr", "pull_body", "pr body"],
    "pr_id": ["pr_number", "pr_id", "pull_number", "pr_url"],
}


def pick(df, key, required=True):
    low = {c.lower(): c for c in df.columns}
    for c in COLS[key]:
        if c.lower() in low:
            return low[c.lower()]
    if required:
        raise KeyError(f"No column for '{key}'. Columns: {list(df.columns)}")
    return None


def load_links(path):
    df = pd.read_csv(path, low_memory=False)
    c = {k: pick(df, k, required=k in ("repo", "issue_title", "pr_title")) for k in COLS}
    out = pd.DataFrame({
        "repo": df[c["repo"]].astype(str),
        "issue": (df[c["issue_title"]].fillna("") + " " +
                  (df[c["issue_body"]].fillna("") if c["issue_body"] else "")).str.slice(0, 4000),
        "pr": (df[c["pr_title"]].fillna("") + " " +
               (df[c["pr_body"]].fillna("") if c["pr_body"] else "")).str.slice(0, 4000),
        "pr_key": df[c["repo"]].astype(str) + "#" + (df[c["pr_id"]].astype(str) if c["pr_id"] else df.index.astype(str)),
    })
    for col in [col for col in df.columns if col.startswith("t_")]:  # optional timestamps (extract_times.py)
        out[col] = df[col].values
    out = out[(out.issue.str.len() > 10) & (out.pr.str.len() > 5)].drop_duplicates(["repo", "issue", "pr"])
    return out.reset_index(drop=True)


def synthetic_links(n_repos=60, per_repo=25, seed=0):
    """Topic-structured fake data, only to smoke-test the pipeline."""
    rng = np.random.default_rng(seed)
    vocab = [f"w{i}" for i in range(800)]
    topics = [rng.choice(vocab, 25, replace=False) for _ in range(120)]
    rows = []
    for r in range(n_repos):
        for _ in range(per_repo):
            t = topics[rng.integers(len(topics))]
            noise = lambda k: list(rng.choice(vocab, k))
            iss = " ".join(list(rng.choice(t, 8)) + noise(12))
            pr = " ".join(list(rng.choice(t, 5)) + noise(10))
            rows.append((f"repo{r}", iss, pr, f"repo{r}#{len(rows)}"))
    return pd.DataFrame(rows, columns=["repo", "issue", "pr", "pr_key"])


def build_pairs(links, neg_per_pos=1, rank_k=19, seed=0):
    """Positives = true links. Hard negatives = other PRs from the SAME repo."""
    rng = np.random.default_rng(seed)
    pairs, rank_sets = [], []
    keys, itxt = links.pr_key.to_numpy(), links.issue.to_numpy()
    for repo, g in links.groupby("repo"):
        idx = g.index.to_numpy()
        if len(idx) < 3:
            continue
        for i in idx:
            # a negative must be a DIFFERENT PR (one PR can close several issues) for a different issue
            others = idx[(keys[idx] != keys[i]) & (itxt[idx] != itxt[i])]
            if len(others) == 0:
                continue
            pairs.append((i, i, 1, repo))
            for j in rng.choice(others, min(neg_per_pos, len(others)), replace=False):
                pairs.append((i, j, 0, repo))
            if len(others) >= rank_k:  # ranking only where a full candidate set exists
                rank_sets.append((i, [i] + list(rng.choice(others, rank_k, replace=False)), repo))
    pairs = pd.DataFrame(pairs, columns=["iss", "pr", "y", "repo"])
    return pairs, rank_sets


# ---------------------------------------------------------------- features
class PairFeaturizer:
    """TF-IDF -> SVD per side, pair features, PCA to d, scaled to [0, pi]."""

    def __init__(self, d, svd_dim=64, seed=0, emb=None):
        # emb: optional (E_issue, E_pr) dense embeddings aligned with links rows (e.g. CodeBERT);
        # if given they replace TF-IDF, and the same train-only SVD(64) is applied to them
        self.d, self.svd_dim, self.seed, self.emb = d, svd_dim, seed, emb

    def _side(self, links, side, idx):
        if self.emb is not None:
            z = self.svd.transform(self.emb[0 if side == "issue" else 1][idx])
        else:
            z = self.svd.transform(self.tfidf.transform(links[side].values[idx]))
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)

    def raw(self, links, iss, pr):
        a, b = self._side(links, "issue", iss), self._side(links, "pr", pr)
        cos = (a * b).sum(1, keepdims=True)
        tok = lambda s: set(s.lower().split())
        jac = np.array([[len(tok(x) & tok(y)) / (len(tok(x) | tok(y)) + 1e-9)]
                        for x, y in zip(links.issue.values[iss], links.pr.values[pr])])
        return np.hstack([cos, jac, a * b, np.abs(a - b)])

    def fit(self, links, iss, pr):
        if self.emb is not None:
            self.svd = TruncatedSVD(self.svd_dim, random_state=self.seed).fit(
                np.vstack([self.emb[0][iss], self.emb[1][pr]]).astype(np.float32))
        else:
            texts = np.concatenate([links.issue.values[iss], links.pr.values[pr]])
            self.tfidf = TfidfVectorizer(max_features=20000, sublinear_tf=True,
                                         token_pattern=r"(?u)\b\w\w+\b").fit(texts)
            self.svd = TruncatedSVD(self.svd_dim, random_state=self.seed).fit(self.tfidf.transform(texts))
        R = self.raw(links, iss, pr)
        self.std = StandardScaler().fit(R)
        self.pca = PCA(self.d, random_state=self.seed).fit(self.std.transform(R))
        self.mm = MinMaxScaler((0, np.pi)).fit(self.pca.transform(self.std.transform(R)))
        return self

    def full(self, links, iss, pr):
        return self.std.transform(self.raw(links, iss, pr))

    def reduced(self, links, iss, pr):
        X = self.mm.transform(self.pca.transform(self.full(links, iss, pr)))
        return np.clip(X, 0, np.pi)


# ---------------------------------------------------------------- models
def feature_map(name, d):
    from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap
    if name == "ZZ1":
        return ZZFeatureMap(d, reps=1, entanglement="linear")
    if name == "ZZ2":
        return ZZFeatureMap(d, reps=2, entanglement="linear")
    if name == "Z":
        return ZFeatureMap(d, reps=2)
    if name == "Pauli":
        return PauliFeatureMap(d, reps=1, paulis=["Z", "YY"], entanglement="linear")
    raise ValueError(name)


class StatevectorKernel:
    """Exact fidelity kernel k(x,y)=|<phi(x)|phi(y)>|^2 (same as Qiskit's
    FidelityStatevectorKernel) with training states cached between calls."""

    def __init__(self, fm):
        self.fm, self._cache = fm, {}

    def states(self, X):
        from qiskit.quantum_info import Statevector
        key = (X.shape, hash(X.tobytes()))
        if key not in self._cache:
            self._cache = {key: np.array([Statevector(self.fm.assign_parameters(x)).data for x in X])}
        return self._cache[key]

    def evaluate(self, X, Y=None):
        A = self.states(X) if Y is None else self._plain(X)
        B = A if Y is None else self.states(Y)
        return np.abs(A @ B.conj().T) ** 2

    def _plain(self, X):
        from qiskit.quantum_info import Statevector
        return np.array([Statevector(self.fm.assign_parameters(x)).data for x in X])


class QSVC_:
    """Fidelity quantum kernel + SVC (precomputed) == Qiskit QSVC, but reusable kernel."""

    def __init__(self, fmap="ZZ1", C=1.0, noisy=False, shots=1024):
        self.fmap, self.C, self.noisy, self.shots = fmap, C, noisy, shots

    def _kernel(self, d):
        fm = feature_map(self.fmap, d)
        if not self.noisy:
            return StatevectorKernel(fm)
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
        from qiskit_aer.primitives import SamplerV2
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        from qiskit_machine_learning.state_fidelities import ComputeUncompute
        nm = NoiseModel()  # IBM-Eagle-like error rates
        nm.add_all_qubit_quantum_error(depolarizing_error(3e-4, 1), ["sx", "x", "rz"])
        nm.add_all_qubit_quantum_error(depolarizing_error(8e-3, 2), ["cx", "ecr"])
        nm.add_all_qubit_readout_error(ReadoutError([[0.985, 0.015], [0.02, 0.98]]))
        sampler = SamplerV2(default_shots=self.shots,
                            options={"backend_options": {"noise_model": nm}})
        fm = fm.decompose(reps=4)  # keep parameter names; Aer handles basis gates
        return FidelityQuantumKernel(feature_map=fm, fidelity=ComputeUncompute(sampler=sampler))

    def fit(self, X, y):
        self.k = self._kernel(X.shape[1])
        self.Xtr = X
        self.svc = SVC(kernel="precomputed", C=self.C, class_weight="balanced").fit(
            self.k.evaluate(X), y)
        return self

    def decision_function(self, X):
        return self.svc.decision_function(self.k.evaluate(X, self.Xtr))

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


class VQC_:
    def __init__(self, maxiter=100, seed=0):
        self.maxiter, self.seed = maxiter, seed

    def fit(self, X, y):
        from qiskit.circuit.library import RealAmplitudes
        from qiskit_machine_learning.algorithms import VQC
        from qiskit_machine_learning.optimizers import COBYLA
        d = X.shape[1]
        self.m = VQC(feature_map=feature_map("ZZ1", d),
                     ansatz=RealAmplitudes(d, reps=2),
                     optimizer=COBYLA(maxiter=self.maxiter))
        self.m.fit(X, y)
        return self

    def decision_function(self, X):
        return self.m.predict_proba(X)[:, 1] - 0.5

    def predict(self, X):
        return self.m.predict(X).astype(int)


def classical(name, seed):
    return {
        "SVC-RBF": lambda: SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"),
        "LogReg": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RF": lambda: RandomForestClassifier(300, random_state=seed, class_weight="balanced", n_jobs=-1),
        "XGB": lambda: __import__("xgboost").XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1, random_state=seed,
            eval_metric="logloss", n_jobs=4),
    }[name]()


def balanced_sample(df, n, seed):
    parts = [g.sample(min(len(g), n // 2), random_state=seed) for _, g in df.groupby("y")]
    return pd.concat(parts)


def score(m, X):
    return m.decision_function(X) if hasattr(m, "decision_function") else m.predict_proba(X)[:, 1]


def metrics(y, pred, s):
    return dict(precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0),
                auc=roc_auc_score(y, s))


def rank_metrics(model, feat, links, rank_sets, X_kind):
    hits1, hits5, rr = [], [], []
    iv = np.concatenate([np.full(len(c), i) for i, c, _ in rank_sets])
    pv = np.concatenate([np.array(c) for _, c, _ in rank_sets])
    X = feat.reduced(links, iv, pv) if X_kind == "red" else feat.full(links, iv, pv)
    S = score(model, X)  # one batched call
    pos = 0
    for _, cands, _ in rank_sets:
        s = S[pos:pos + len(cands)]; pos += len(cands)
        r = int((s > s[0]).sum())  # rank of true PR (index 0)
        hits1.append(r < 1); hits5.append(r < 5); rr.append(1 / (r + 1))
    return dict(hit1=np.mean(hits1), hit5=np.mean(hits5), mrr=np.mean(rr))


# ---------------------------------------------------------------- experiment
def run(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    links = synthetic_links() if args.synthetic else load_links(args.data)
    print(f"links={len(links)} repos={links.repo.nunique()}")
    emb = None
    if args.emb:  # precomputed embeddings from embed_codebert.py, same row order as load_links
        z = np.load(args.emb)
        emb = (z["issue"].astype(np.float32), z["pr"].astype(np.float32))
        assert len(emb[0]) == len(links), "embeddings do not match the links file"
        print("using embeddings", args.emb, emb[0].shape)
    rows = []
    for seed in range(args.seeds):
        pairs, rank_sets = build_pairs(links, seed=seed)
        tr_idx, te_idx = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed)
                              .split(pairs, groups=pairs.repo))  # repo-level split, no leakage
        tr, te = pairs.iloc[tr_idx], pairs.iloc[te_idx]
        tr = balanced_sample(tr, args.n_train, seed)
        te = balanced_sample(te, args.n_test, seed)
        te_repos = set(te.repo)
        rs = [r for r in rank_sets if r[2] in te_repos]
        rs = [rs[i] for i in np.random.default_rng(seed).permutation(len(rs))[:args.n_rank]]

        for d in args.dims:
            feat = PairFeaturizer(d, seed=seed, emb=emb).fit(links, tr.iss.values, tr.pr.values)
            Xtr, Xte = feat.reduced(links, tr.iss.values, tr.pr.values), feat.reduced(links, te.iss.values, te.pr.values)
            ytr, yte = tr.y.values, te.y.values

            configs = [(f"QSVC-{fm}", lambda fm=fm: QSVC_(fm), "red") for fm in args.fmaps]
            configs += [(n, lambda n=n: classical(n, seed), "red") for n in args.classical]
            if args.vqc and d <= 8:
                configs.append(("VQC", lambda: VQC_(seed=seed), "red"))
            if d == max(args.dims):  # upper bound: classical on full (unreduced) features
                configs += [(f"{n}(full)", lambda n=n: classical(n, seed), "full")
                            for n in ("SVC-RBF", "XGB")]

            for name, mk, kind in configs:
                Xa, Xb = (Xtr, Xte) if kind == "red" else (
                    feat.full(links, tr.iss.values, tr.pr.values), feat.full(links, te.iss.values, te.pr.values))
                ntr = len(Xa) if name != "VQC" else min(len(Xa), args.vqc_n)
                sub = np.random.default_rng(seed).permutation(len(Xa))[:ntr]
                t0 = time.time(); m = mk().fit(Xa[sub], ytr[sub]); t_fit = time.time() - t0
                t0 = time.time(); s = score(m, Xb); pred = (s > 0).astype(int) if "QSVC" in name or "SVC" in name or name == "VQC" else m.predict(Xb)
                t_inf = time.time() - t0
                row = dict(seed=seed, d=d, model=name, n_train=ntr, n_test=len(Xb),
                           fit_s=t_fit, infer_s=t_inf, **metrics(yte, pred, s))
                if args.n_rank and name != "VQC":
                    row.update(rank_metrics(m, feat, links, rs, kind))
                rows.append(row)
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

        # RQ3: ideal vs noisy simulation, small subset, d=noise_d
        if args.noise_n:
            d = args.noise_d
            feat = PairFeaturizer(d, seed=seed, emb=emb).fit(links, tr.iss.values, tr.pr.values)
            sub = balanced_sample(tr, args.noise_n, seed)
            tsub = balanced_sample(te, args.noise_n, seed)
            Xa, Xb = feat.reduced(links, sub.iss.values, sub.pr.values), feat.reduced(links, tsub.iss.values, tsub.pr.values)
            K = {}
            for noisy in (False, True):
                t0 = time.time(); m = QSVC_("ZZ1", noisy=noisy).fit(Xa, sub.y.values)
                s = m.decision_function(Xb)
                K[noisy] = m.k.evaluate(Xa)
                row = dict(seed=seed, d=d, model=f"QSVC-ZZ1-{'noisy' if noisy else 'ideal'}(RQ3)",
                           n_train=len(Xa), n_test=len(Xb), fit_s=time.time() - t0, infer_s=0,
                           **metrics(tsub.y.values, (s > 0).astype(int), s))
                if noisy:  # how far the noisy kernel drifts from the exact one
                    off = ~np.eye(len(Xa), dtype=bool)
                    row["kernel_mae"] = float(np.abs(K[True] - K[False])[off].mean())
                    row["kernel_corr"] = float(np.corrcoef(K[True][off], K[False][off])[0, 1])
                rows.append(row)
                print(row, flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(out / "raw_results.csv", index=False)
    cols = [c for c in ["precision", "recall", "f1", "auc", "hit1", "hit5", "mrr", "kernel_mae", "kernel_corr", "fit_s", "infer_s"] if c in res]
    summ = res.groupby(["d", "model"])[cols].agg(["mean", "std"]).round(3)
    summ.to_csv(out / "summary.csv")

    # Wilcoxon: best QSVC vs SVC-RBF per d (paired over seeds)
    tests = []
    for d in args.dims:
        r = res[res.d == d]
        q = r[r.model.str.startswith("QSVC")].groupby("model").f1.mean()
        if q.empty or "SVC-RBF" not in set(r.model):
            continue
        best = q.idxmax()
        a = r[r.model == best].sort_values("seed").f1.values
        b = r[r.model == "SVC-RBF"].sort_values("seed").f1.values
        p = wilcoxon(a, b).pvalue if len(a) >= 5 and np.any(a != b) else float("nan")
        delta = np.mean([np.sign(x - y) for x in a for y in b])  # Cliff's delta
        tests.append(dict(d=d, quantum=best, q_f1=a.mean(), svc_f1=b.mean(), wilcoxon_p=p, cliffs_delta=delta))
    pd.DataFrame(tests).to_csv(out / "stat_tests.csv", index=False)
    json.dump(vars(args), open(out / "config.json", "w"), indent=2)
    print(summ.to_string())
    print(pd.DataFrame(tests).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data"); p.add_argument("--synthetic", action="store_true")
    p.add_argument("--out", default="results")
    p.add_argument("--emb", default=None, help="npz with 'issue' and 'pr' arrays (CodeBERT etc.)")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--dims", type=int, nargs="+", default=[4, 8, 12])
    p.add_argument("--fmaps", nargs="+", default=["ZZ1", "ZZ2", "Z", "Pauli"])
    p.add_argument("--classical", nargs="+", default=["SVC-RBF", "LogReg", "RF", "XGB"])
    p.add_argument("--n_train", type=int, default=800)
    p.add_argument("--n_test", type=int, default=400)
    p.add_argument("--n_rank", type=int, default=100)
    p.add_argument("--vqc", action="store_true"); p.add_argument("--vqc_n", type=int, default=200)
    p.add_argument("--noise_n", type=int, default=0); p.add_argument("--noise_d", type=int, default=4)
    run(p.parse_args())
