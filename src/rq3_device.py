"""RQ3b: QSVC with fidelity kernels estimated under device-calibrated IBM noise.

Each kernel entry |<phi(x)|phi(y)>|^2 is estimated as P(0...0) of U(x) U(y)^dagger,
transpiled onto the real coupling map and native gates of the IBM device
(snapshot calibrations from qiskit-ibm-runtime fake backends), 1024 shots.
Same data protocol as qml_pilink.py RQ3 (repo-level split, d=4, 100/100 pairs).
"""
import argparse, sys, time, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.svm import SVC
sys.path.insert(0, str(Path(__file__).parent))
from qml_pilink import load_links, build_pairs, balanced_sample, PairFeaturizer, metrics
from sklearn.model_selection import GroupShuffleSplit
from qiskit import transpile
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeTorino, FakeBrisbane

def fmap(name, d, prefix):
    if name == "ZZ1": return ZZFeatureMap(d, reps=1, entanglement="linear", parameter_prefix=prefix)
    if name == "Z": return ZFeatureMap(d, reps=2, parameter_prefix=prefix)

def exact_kernel(name, X, Y):
    fm = fmap(name, X.shape[1], "x")
    S = lambda Z: np.array([Statevector(fm.assign_parameters(z)).data for z in Z])
    A, B = S(X), S(Y)
    return np.abs(A @ B.conj().T) ** 2

def device_kernel(name, X, Y, backend, shots, symmetric, seed):
    d = X.shape[1]
    fx, fy = fmap(name, d, "x"), fmap(name, d, "y")
    qc = fx.compose(fy.inverse()); qc.measure_all()
    tqc = transpile(qc, backend, optimization_level=3, seed_transpiler=seed)
    px = sorted([p for p in tqc.parameters if p.name.startswith("x")], key=lambda p: p.index)
    py = sorted([p for p in tqc.parameters if p.name.startswith("y")], key=lambda p: p.index)
    pairs = [(i, j) for i in range(len(X)) for j in range(len(Y)) if not symmetric or j > i]
    circs = [tqc.assign_parameters(dict(zip(px, X[i])) | dict(zip(py, Y[j]))) for i, j in pairs]
    sim = AerSimulator.from_backend(backend, seed_simulator=seed)
    K = np.eye(len(X)) if symmetric else np.zeros((len(X), len(Y)))
    zero = "0" * d
    for s in range(0, len(circs), 500):
        res = sim.run(circs[s:s + 500], shots=shots).result()
        for k, (i, j) in enumerate(pairs[s:s + 500]):
            c = res.get_counts(k)
            p = sum(v for key, v in c.items() if key.replace(" ", "")[-d:] == zero) / shots
            K[i, j] = p
            if symmetric: K[j, i] = p
    info = dict(depth=tqc.depth(), twoq=sum(v for g, v in tqc.count_ops().items() if g in ("cz", "ecr", "cx")),
                layout=[tqc.layout.initial_layout[q] for q in qc.qubits[:d]] if tqc.layout else None)
    return K, info

def svc_eval(Ktr, Kte, ytr, yte):
    m = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(Ktr, ytr)
    s = m.decision_function(Kte)
    return metrics(yte, (s > 0).astype(int), s)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="pilink_pairs.csv"); ap.add_argument("--out", default="rq3_device")
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--d", type=int, default=4); ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--fmaps", nargs="+", default=["Z", "ZZ1"])
    a = ap.parse_args(); out = Path(a.out); out.mkdir(exist_ok=True)
    links = load_links(a.data); rows = []
    backends = {"ibm_torino(Heron)": FakeTorino(), "ibm_brisbane(Eagle)": FakeBrisbane()}
    for seed in range(a.seeds):
        pairs, _ = build_pairs(links, seed=seed)
        tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(pairs, groups=pairs.repo))
        tr, te = balanced_sample(pairs.iloc[tr_i], a.n, seed), balanced_sample(pairs.iloc[te_i], a.n, seed)
        feat = PairFeaturizer(a.d, seed=seed).fit(links, tr.iss.values, tr.pr.values)
        Xa, Xb = feat.reduced(links, tr.iss.values, tr.pr.values), feat.reduced(links, te.iss.values, te.pr.values)
        for fm in a.fmaps:
            Ka, Kb = exact_kernel(fm, Xa, Xa), exact_kernel(fm, Xb, Xa)
            r = dict(seed=seed, fmap=fm, backend="ideal", **svc_eval(Ka, Kb, tr.y.values, te.y.values)); rows.append(r); print(r, flush=True)
            for bn, be in backends.items():
                t0 = time.time()
                Ta, info = device_kernel(fm, Xa, Xa, be, a.shots, True, seed)
                Tb, _ = device_kernel(fm, Xb, Xa, be, a.shots, False, seed)
                off = ~np.eye(len(Xa), dtype=bool)
                r = dict(seed=seed, fmap=fm, backend=bn, **svc_eval(Ta, Tb, tr.y.values, te.y.values),
                         kernel_mae=float(np.abs(Ta - Ka)[off].mean()),
                         kernel_corr=float(np.corrcoef(Ta[off], Ka[off])[0, 1]),
                         test_kernel_corr=float(np.corrcoef(Tb.ravel(), Kb.ravel())[0, 1]),
                         sim_s=round(time.time() - t0, 1), **{k: v for k, v in info.items() if k != "layout"})
                rows.append(r); print(r, flush=True)
        pd.DataFrame(rows).to_csv(out / "raw.csv", index=False)
    df = pd.DataFrame(rows)
    df.groupby(["fmap", "backend"]).agg(["mean", "std"]).round(3).to_csv(out / "summary.csv")
    print(df.groupby(["fmap", "backend"]).mean(numeric_only=True).round(3).to_string())
