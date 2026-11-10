"""Extra quantum tests on the final setting (PLS + open-time metadata + BM25).

  shots   Test 1: tuned d=12 models under finite shots and ibm_torino device noise.
          Fidelity kernels: each entry K_ij is a Bernoulli frequency of the all-zeros outcome of
          U(x_i) U(x_j)^dagger. Device noise is modelled as K -> a K + b with (a, b) fitted on
          40 transpiled 12-qubit circuits (exact density-matrix simulation) simulated with FakeTorino's calibrated noise model.
          PQK: single-qubit <X>, <Y>, <Z> estimated from shots on U(x) (3 circuits per sample),
          device shrink factors fitted the same way. C re-selected by CV on the noisy kernel.
  conc    Test 2: kernel concentration (mean / variance of off-diagonal kernel values) vs
          qubits, feature map and bandwidth, with the CV AUC each configuration reaches.
  qubits  Test 3: Z-map models at d = 4..24 qubits (product-state kernel, exact), vs classical.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.svm import SVC
sys.path.insert(0, str(Path(__file__).parent))
from qml_pilink import load_links, build_pairs, balanced_sample, metrics
from qml_enhanced import (EnhancedFeaturizer, states, bloch, sqd, cv_auc, rank_eval, classical_models,
                          LAMS, CS, GS)
from bm25 import BM25

DATA = "pilink_pairs_meta.csv"


def prep(links, seed, d, n_train=800, n_test=400, n_rank=200):
    pairs, rank_sets = build_pairs(links, seed=seed)
    tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(pairs, groups=pairs.repo))
    tr, te = balanced_sample(pairs.iloc[tr_i], n_train, seed), balanced_sample(pairs.iloc[te_i], n_test, seed)
    rs = [r for r in rank_sets if r[2] in set(te.repo)]
    rs = [rs[i] for i in np.random.default_rng(seed).permutation(len(rs))[:n_rank]]
    iv = np.concatenate([np.full(len(c), i) for i, c, _ in rs]); pv = np.concatenate([np.array(c) for _, c, _ in rs])
    ytr, yte = tr.y.values, te.y.values
    cv = list(GroupKFold(3).split(np.zeros(len(tr)), ytr, tr.repo.values))
    bm = BM25().fit(links.pr.values[np.unique(pairs.iloc[tr_i].pr.values)])
    feat = EnhancedFeaturizer(d, "pls", True, seed, bm).fit(links, tr.iss.values, tr.pr.values, ytr)
    R = lambda i, p: feat.reduced(links, i, p)
    return dict(Xtr=R(tr.iss.values, tr.pr.values), Xte=R(te.iss.values, te.pr.values), Xrk=R(iv, pv),
                ytr=ytr, yte=yte, cv=cv, sizes=[len(c) for _, c, _ in rs])


def evaluate(Ktr, Kte, Krk, P, C=None):
    """Select C by CV on the (possibly noisy) training kernel, then score test + ranking."""
    if C is None:
        C = max(CS, key=lambda c: cv_auc(Ktr, P["ytr"], P["cv"], c))
    m = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(Ktr, P["ytr"])
    s = m.decision_function(Kte)
    return dict(C=C, **metrics(P["yte"], (s > 0).astype(int), s), **rank_eval(m.decision_function(Krk), P["sizes"]))


# ---------------------------------------------------------------- analytic Z-map (product state)
def z_qubits(X):
    e = np.exp(2j * X)
    return np.stack([(1 + e) / 2, e * (1 - e) / 2], -1)  # (N, n, 2): ZFeatureMap reps=2, per qubit


def z_kernel(A, B, qa=None, qb=None):
    qa = z_qubits(A) if qa is None else qa; qb = z_qubits(B) if qb is None else qb
    logK = np.zeros((len(A), len(B)))
    for k in range(A.shape[1]):
        ov = np.abs(qa[:, k].conj() @ qb[:, k].T) ** 2
        logK += np.log(np.maximum(ov, 1e-300))
    return np.exp(logK)


def z_bloch(X):
    q = z_qubits(X); r01 = q[..., 0] * q[..., 1].conj()
    B = np.stack([2 * r01.real, -2 * r01.imag, np.abs(q[..., 0]) ** 2 - np.abs(q[..., 1]) ** 2], -1)  # (N, n, 3)
    return B.reshape(len(X), -1)  # per-qubit order [x0, y0, z0, x1, ...], same as qml_enhanced.bloch


# ---------------------------------------------------------------- Test 1: shots + device noise
def device_calibration(fm, Xs, lam, seed=0, n_pairs=40, n_pqk=4, shots=4096):
    """Fit K_dev = a*K + b on transpiled 12-qubit fidelity circuits, and PQK shrink factors."""
    from qiskit import transpile, QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    from rq3_device import fmap
    be = FakeTorino(); sim = AerSimulator.from_backend(be, method="density_matrix", seed_simulator=seed); d = Xs.shape[1]
    rng = np.random.default_rng(seed)
    fx, fy = fmap(fm, d, "x"), fmap(fm, d, "y")
    qc = fx.compose(fy.inverse()); qc.measure_all()
    tqc = transpile(qc, be, optimization_level=3, seed_transpiler=seed)
    px = sorted([p for p in tqc.parameters if p.name.startswith("x")], key=lambda p: p.index)
    py = sorted([p for p in tqc.parameters if p.name.startswith("y")], key=lambda p: p.index)
    I, J = rng.integers(0, len(Xs), n_pairs), rng.integers(0, len(Xs), n_pairs)
    A, B = lam * Xs[I], lam * Xs[J]
    circs = [tqc.assign_parameters(dict(zip(px, a)) | dict(zip(py, b))) for a, b in zip(A, B)]
    res = sim.run(circs, shots=shots).result()
    meas = np.array([res.get_counts(k).get("0" * d, 0) / shots for k in range(n_pairs)])
    S = lambda Z: states(fm, Z)
    exact = np.abs((S(A).conj() * S(B)).sum(1)) ** 2
    a, b = np.polyfit(exact, meas, 1)
    r2 = 1 - ((meas - (a * exact + b)) ** 2).sum() / ((meas - meas.mean()) ** 2).sum()
    info = dict(fmap=fm, a=float(a), b=float(b), r2=float(r2), depth=tqc.depth(),
                twoq=int(sum(v for g, v in tqc.count_ops().items() if g in ("cz", "ecr", "cx"))),
                exact_range=[float(exact.min()), float(exact.max())])
    # PQK: <P_k> on U(x) under device noise; shrink factor per basis
    shrink = {}
    for basis in "XYZ":
        c = fx.copy()
        if basis == "X": c.h(range(d))
        if basis == "Y": c.sdg(range(d)); c.h(range(d))
        c.measure_all()
        tc = transpile(c, be, optimization_level=3, seed_transpiler=seed)
        pxx = sorted(tc.parameters, key=lambda p: p.index)
        idx = rng.integers(0, len(Xs), n_pqk)
        r = sim.run([tc.assign_parameters(dict(zip(pxx, lam * Xs[i]))) for i in idx], shots=shots).result()
        noisy, ideal = [], []
        Bfull = bloch(S(lam * Xs[idx]), d)  # columns: per qubit k -> [x, y, z] at 3k..3k+2
        col = "XYZ".index(basis)
        for k_, i in enumerate(idx):
            cnt = r.get_counts(k_); tot = sum(cnt.values())
            for q in range(d):
                p1 = sum(v for key, v in cnt.items() if key.replace(" ", "")[::-1][q] == "1") / tot
                noisy.append(1 - 2 * p1); ideal.append(Bfull[k_, 3 * q + col])
        noisy, ideal = np.array(noisy), np.array(ideal)
        den = (ideal ** 2).sum()  # <Z> can be identically 0 for some maps: no information, keep 1
        shrink[basis] = float((noisy * ideal).sum() / den) if den > 1e-6 else 1.0
    info["pqk_shrink"] = shrink
    return info


def sample_kernel(K, shots, rng, a=1.0, b=0.0, symmetric=False):
    P = np.clip(a * K + b, 0, 1)
    if shots is None:
        return P
    if symmetric:
        iu = np.triu_indices(len(K), 1); M = np.eye(len(K))
        M[iu] = rng.binomial(shots, P[iu]) / shots; M = M + np.triu(M, 1).T
        return M
    return rng.binomial(shots, P) / shots


def sample_bloch(Bv, shots, rng, shrink=None):
    if shrink is not None:
        Bv = Bv * np.tile([shrink["X"], shrink["Y"], shrink["Z"]], Bv.shape[1] // 3)
    if shots is None:
        return Bv
    return 2 * rng.binomial(shots, np.clip((1 + Bv) / 2, 0, 1)) / shots - 1


def test_shots(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    links = load_links(DATA); D = 12
    f30 = pd.concat([pd.read_csv(f"results_final30/{h}/raw_results.csv") for h in "ab"])
    par = {(r.seed, r.model): json.loads(r.params) for r in f30.itertuples()}
    rows, calib = [], {}
    conds = [("ideal", None, False)] + [(f"shots{s}", s, False) for s in a.shots] + [(f"torino+shots{s}", s, True) for s in a.dev_shots]
    for seed in range(a.seeds):
        P = prep(links, seed, D); rng = np.random.default_rng(1000 + seed)
        for fm in ("Z", "ZZ1"):
            p = par[(seed, f"QSVC-{fm}")]; lam = p["lam"]
            if fm not in calib:
                calib[fm] = device_calibration(fm, P["Xtr"], lam, seed=0); print("calib", calib[fm], flush=True)
                json.dump(calib, open(out / "device_calibration.json", "w"), indent=1)
            St, Se, Sr = states(fm, lam * P["Xtr"]), states(fm, lam * P["Xte"]), states(fm, lam * P["Xrk"])
            K = [np.abs(X @ St.conj().T) ** 2 for X in (St, Se, Sr)]
            offd = K[0][~np.eye(len(K[0]), dtype=bool)]
            for name, shots, dev in conds:
                ab = (calib[fm]["a"], calib[fm]["b"]) if dev else (1.0, 0.0)
                Kn = [sample_kernel(K[0], shots, rng, *ab, symmetric=True)] + [sample_kernel(k, shots, rng, *ab) for k in K[1:]]
                r = evaluate(*Kn, P)
                rows.append(dict(seed=seed, model=f"QSVC-{fm}", condition=name, shots=shots, device=dev, lam=lam,
                                 k_mean=float(offd.mean()), k_std=float(offd.std()), **r))
            if fm == "Z":  # PQK-Z, same states
                pp = par[(seed, "PQK-Z")]; lam = pp["lam"]
                St, Se, Sr = states("Z", lam * P["Xtr"]), states("Z", lam * P["Xte"]), states("Z", lam * P["Xrk"])
                Bt, Be, Br = bloch(St, D), bloch(Se, D), bloch(Sr, D)
                for name, shots, dev in conds:
                    sh = calib["Z"]["pqk_shrink"] if dev else None
                    bt, be, br = (sample_bloch(x, shots, rng, sh) for x in (Bt, Be, Br))
                    D2 = sqd(bt, bt); med = np.median(D2[~np.eye(len(D2), dtype=bool)]) + 1e-12; g = pp["g"] / med
                    r = evaluate(np.exp(-g * D2), np.exp(-g * sqd(be, bt)), np.exp(-g * sqd(br, bt)), P)
                    rows.append(dict(seed=seed, model="PQK-Z", condition=name, shots=shots, device=dev, lam=lam, **r))
        print(seed, pd.DataFrame(rows)[lambda x: x.seed == seed].pivot_table(index="condition", columns="model", values="mrr").round(3).to_dict(), flush=True)
        pd.DataFrame(rows).to_csv(out / "shots_results.csv", index=False)


# ---------------------------------------------------------------- Test 2: concentration
def test_conc(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    links = load_links(DATA); rows = []
    for seed in range(a.seeds):
        for d in (4, 8, 12, 16, 20, 24):
            P = prep(links, seed, d); X = P["Xtr"]
            for fm in ("Z", "ZZ1", "Pauli"):
                if d > 12 and fm != "Z": continue
                for lam in LAMS:
                    if d > 12:
                        K = z_kernel(lam * X, lam * X)
                    else:
                        S = states(fm, lam * X); K = np.abs(S @ S.conj().T) ** 2
                    off = K[~np.eye(len(K), dtype=bool)]
                    auc = max(cv_auc(K, P["ytr"], P["cv"], c) for c in CS)
                    rows.append(dict(seed=seed, d=d, fmap=fm, lam=lam, k_mean=off.mean(), k_var=off.var(),
                                     k_std=off.std(), cv_auc=auc))
            print(seed, d, flush=True)
            pd.DataFrame(rows).to_csv(out / "concentration.csv", index=False)


# ---------------------------------------------------------------- Test 3: more qubits (Z map)
def test_qubits(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    links = load_links(DATA); rows = []
    for seed in range(a.seeds):
        for d in a.dims:
            P = prep(links, seed, d); Xtr, Xte, Xrk = P["Xtr"], P["Xte"], P["Xrk"]
            res = {}
            t0 = time.time()  # QSVC-Z: tune lam, C
            best = max(((lam, c) for lam in LAMS for c in CS),
                       key=lambda lc: cv_auc(z_kernel(lc[0] * Xtr, lc[0] * Xtr), P["ytr"], P["cv"], lc[1]))
            lam, C = best; q = z_qubits(lam * Xtr)
            r = evaluate(z_kernel(lam * Xtr, lam * Xtr, q, q), z_kernel(lam * Xte, lam * Xtr, None, q),
                         z_kernel(lam * Xrk, lam * Xtr, None, q), P, C=C)
            res["QSVC-Z"] = dict(r, lam=lam, fit_s=time.time() - t0)
            t0 = time.time(); bestp = (-1, None)  # PQK-Z: tune lam, g, C
            for lam in LAMS:
                Bt = z_bloch(lam * Xtr); D2 = sqd(Bt, Bt); med = np.median(D2[~np.eye(len(D2), dtype=bool)]) + 1e-12
                for g in GS:
                    Kp = np.exp(-(g / med) * D2)
                    for c in CS:
                        s = cv_auc(Kp, P["ytr"], P["cv"], c)
                        if s > bestp[0]: bestp = (s, (lam, g / med, c))
            lam, g, C = bestp[1]; Bt = z_bloch(lam * Xtr)
            r = evaluate(np.exp(-g * sqd(Bt, Bt)), np.exp(-g * sqd(z_bloch(lam * Xte), Bt)),
                         np.exp(-g * sqd(z_bloch(lam * Xrk), Bt)), P, C=C)
            res["PQK-Z"] = dict(r, lam=lam, fit_s=time.time() - t0)
            for name, (s, srk, ft, bp) in classical_models(Xtr, P["ytr"], Xte, Xrk, P["cv"], seed, d).items():
                res[name] = dict(**metrics(P["yte"], (s > (0.5 if "XGB" in name else 0)).astype(int), s),
                                 **rank_eval(srk, P["sizes"]), fit_s=ft)
            for name, r in res.items():
                rows.append(dict(seed=seed, d=d, model=name, **{k: v for k, v in r.items() if k in
                            ("f1", "auc", "hit1", "hit5", "mrr", "lam", "fit_s")}))
            print(seed, d, {k: round(v["mrr"], 3) for k, v in res.items()}, flush=True)
            pd.DataFrame(rows).to_csv(out / "qubits_results.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("test", choices=["shots", "conc", "qubits"]); p.add_argument("--out", default="results_extra")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--shots", type=int, nargs="+", default=[256, 1024, 4096, 16384])
    p.add_argument("--dev_shots", type=int, nargs="+", default=[1024, 4096, 16384])
    p.add_argument("--dims", type=int, nargs="+", default=[4, 8, 12, 16, 20, 24])
    a = p.parse_args()
    {"shots": test_shots, "conc": test_conc, "qubits": test_qubits}[a.test](a)
