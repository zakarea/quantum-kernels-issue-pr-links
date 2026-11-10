"""RQ3c: QSVC fidelity kernels measured on a real IBM Quantum QPU.

Same protocol as rq3_device.py (repo-level split, d=4 qubits), sized for the
IBM Quantum open plan (10 QPU-minutes / month): by default 1 seed, 30 train +
30 test pairs, 512 shots, Z and ZZ1 maps = 4 jobs, roughly 4-7 QPU-minutes.
Each kernel is ONE job: a single transpiled circuit with a parameter array.

  pip install qiskit qiskit-ibm-runtime qiskit-aer scikit-learn pandas
  python ibm_hardware.py --token YOUR_IBM_QUANTUM_API_KEY              # real QPU
  python ibm_hardware.py --fake                                        # local dry run
"""
import argparse, sys, time, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
sys.path.insert(0, str(Path(__file__).parent))
from qml_pilink import load_links, build_pairs, balanced_sample, PairFeaturizer
from rq3_device import fmap, exact_kernel, svc_eval
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="pilink_pairs.csv"); ap.add_argument("--out", default="rq3_hardware")
ap.add_argument("--token"); ap.add_argument("--instance", default=None); ap.add_argument("--backend", default=None)
ap.add_argument("--fake", action="store_true", help="dry run on FakeTorino locally")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n", type=int, default=30)
ap.add_argument("--d", type=int, default=4); ap.add_argument("--shots", type=int, default=512)
ap.add_argument("--fmaps", nargs="+", default=["Z", "ZZ1"])
a = ap.parse_args(); out = Path(a.out); out.mkdir(exist_ok=True)

from qiskit_ibm_runtime import SamplerV2, QiskitRuntimeService
if a.fake:
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    backend = FakeTorino()
else:
    svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=a.token, instance=a.instance)
    backend = svc.backend(a.backend) if a.backend else svc.least_busy(operational=True, simulator=False, min_num_qubits=a.d)
print("backend:", backend.name, flush=True)

links = load_links(a.data); seed = a.seed
pairs, _ = build_pairs(links, seed=seed)
tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(pairs, groups=pairs.repo))
tr, te = balanced_sample(pairs.iloc[tr_i], a.n, seed), balanced_sample(pairs.iloc[te_i], a.n, seed)
feat = PairFeaturizer(a.d, seed=seed).fit(links, tr.iss.values, tr.pr.values)
Xa, Xb = feat.reduced(links, tr.iss.values, tr.pr.values), feat.reduced(links, te.iss.values, te.pr.values)

pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=seed)
sampler = SamplerV2(mode=backend)
sampler.options.default_shots = a.shots
sampler.options.dynamical_decoupling.enable = True  # free, no extra QPU time
rows, raw = [], {}

def hw_kernel(name, X, Y, symmetric):
    d = X.shape[1]; fx, fy = fmap(name, d, "x"), fmap(name, d, "y")
    qc = fx.compose(fy.inverse()); qc.measure_all()
    tqc = pm.run(qc)
    px = sorted([p for p in tqc.parameters if p.name.startswith("x")], key=lambda p: p.index)
    py = sorted([p for p in tqc.parameters if p.name.startswith("y")], key=lambda p: p.index)
    order = [p.name for p in tqc.parameters]  # bind values in the circuit's own parameter order
    idx = [(i, j) for i in range(len(X)) for j in range(len(Y)) if not symmetric or j > i]
    vals = np.array([[dict([(p.name, X[i][p.index]) for p in px] + [(p.name, Y[j][p.index]) for p in py])[n]
                      for n in order] for i, j in idx])
    t0 = time.time(); job = sampler.run([(tqc, vals)]); print(" job", job.job_id(), len(idx), "circuits", flush=True)
    res = job.result()[0]
    bits = res.data.meas  # BitArray, shape (len(idx),)
    p0 = np.array([bits[k].get_counts().get("0" * d, 0) for k in range(len(idx))]) / a.shots
    K = np.eye(len(X)) if symmetric else np.zeros((len(X), len(Y)))
    for (i, j), p in zip(idx, p0):
        K[i, j] = p
        if symmetric: K[j, i] = p
    usage = None
    try: usage = job.usage()  # QPU seconds (real backends only)
    except Exception: pass
    return K, dict(job_id=job.job_id(), wall_s=round(time.time() - t0, 1), qpu_s=usage,
                   depth=tqc.depth(), twoq=sum(v for g, v in tqc.count_ops().items() if g in ("cz", "ecr", "cx")))

for fm in a.fmaps:
    Ka, Kb = exact_kernel(fm, Xa, Xa), exact_kernel(fm, Xb, Xa)
    rows.append(dict(seed=seed, fmap=fm, backend="ideal", **svc_eval(Ka, Kb, tr.y.values, te.y.values)))
    Ta, ia = hw_kernel(fm, Xa, Xa, True); Tb, ib = hw_kernel(fm, Xb, Xa, False)
    off = ~np.eye(len(Xa), dtype=bool)
    rows.append(dict(seed=seed, fmap=fm, backend=backend.name, **svc_eval(Ta, Tb, tr.y.values, te.y.values),
                     kernel_mae=float(np.abs(Ta - Ka)[off].mean()), kernel_corr=float(np.corrcoef(Ta[off], Ka[off])[0, 1]),
                     test_kernel_corr=float(np.corrcoef(Tb.ravel(), Kb.ravel())[0, 1]),
                     train_job=ia["job_id"], test_job=ib["job_id"], qpu_s=(ia["qpu_s"] or 0) + (ib["qpu_s"] or 0),
                     depth=ia["depth"], twoq=ia["twoq"]))
    raw[fm] = dict(K_train=Ta.tolist(), K_test=Tb.tolist(), K_train_exact=Ka.tolist(), K_test_exact=Kb.tolist(),
                   X_train=Xa.tolist(), X_test=Xb.tolist(), y_train=tr.y.tolist(), y_test=te.y.tolist())
    print(rows[-2], "\n", rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out / "hardware_results.csv", index=False)
json.dump(raw, open(out / "hardware_kernels.json", "w"))
print(pd.DataFrame(rows).to_string())
