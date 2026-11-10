"""Real-hardware check of the FINAL tuned model: QSVC-Z, 12 qubits, PLS + metadata + BM25, bandwidth lam.

Budget-aware: 24 training + 24 test pairs, 1,024 shots -> 276 + 576 = 852 circuits (~0.9 M shots,
roughly 3-4 QPU-minutes on a Heron device). The Z map has no two-qubit gates.
Reports kernel fidelity vs exact simulation (main measure), and the agreement of SVM decisions.

  python hw_final.py --token KEY --instance CRN        # real QPU (least busy)
  python hw_final.py --fake                            # local dry run on FakeTorino
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
sys.path.insert(0, str(Path(__file__).parent))
from qml_pilink import load_links
from quantum_extra import prep
from qml_enhanced import states
from rq3_device import fmap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="pilink_pairs_meta.csv"); ap.add_argument("--out", default="hw_final")
ap.add_argument("--token"); ap.add_argument("--instance", default=None); ap.add_argument("--backend", default=None)
ap.add_argument("--fake", action="store_true"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n", type=int, default=24); ap.add_argument("--shots", type=int, default=1024)
ap.add_argument("--lam", type=float, default=0.25, help="tuned bandwidth of QSVC-Z on seed 0 (final30)")
a = ap.parse_args(); out = Path(a.out); out.mkdir(exist_ok=True); D = 12

from qiskit_ibm_runtime import SamplerV2, QiskitRuntimeService
if a.fake:
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    backend = FakeTorino()
else:
    svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=a.token, instance=a.instance or None)
    backend = svc.backend(a.backend) if a.backend else svc.least_busy(operational=True, simulator=False, min_num_qubits=D)
print("backend:", backend.name, flush=True)

links = load_links(a.data); P = prep(links, a.seed, D); rng = np.random.default_rng(a.seed)
pick = lambda y, n: np.concatenate([rng.choice(np.where(y == c)[0], n // 2, replace=False) for c in (0, 1)])
it, ie = pick(P["ytr"], a.n), pick(P["yte"], a.n)
Xa, ya, Xb, yb = a.lam * P["Xtr"][it], P["ytr"][it], a.lam * P["Xte"][ie], P["yte"][ie]
Sa, Sb = states("Z", Xa), states("Z", Xb)
Ka, Kb = np.abs(Sa @ Sa.conj().T) ** 2, np.abs(Sb @ Sa.conj().T) ** 2

pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=a.seed)
sampler = SamplerV2(mode=backend); sampler.options.default_shots = a.shots
try: sampler.options.dynamical_decoupling.enable = True
except Exception: pass


def hw_kernel(X, Y, symmetric):
    fx, fy = fmap("Z", D, "x"), fmap("Z", D, "y")
    qc = fx.compose(fy.inverse()); qc.measure_all(); tqc = pm.run(qc)
    px = {p.name: p.index for p in tqc.parameters if p.name.startswith("x")}
    order = [p.name for p in tqc.parameters]
    idx = [(i, j) for i in range(len(X)) for j in range(len(Y)) if not symmetric or j > i]
    vals = np.array([[X[i][p] if n in px else Y[j][p] for n, p in ((n, int(n.split("[")[1][:-1])) for n in order)] for i, j in idx])
    t0 = time.time(); job = sampler.run([(tqc, vals)]); print(" job", job.job_id(), len(idx), "circuits", flush=True)
    bits = job.result()[0].data.meas
    p0 = np.array([bits[k].get_counts().get("0" * D, 0) for k in range(len(idx))]) / a.shots
    K = np.eye(len(X)) if symmetric else np.zeros((len(X), len(Y)))
    for (i, j), p in zip(idx, p0):
        K[i, j] = p
        if symmetric: K[j, i] = p
    usage = None
    try: usage = job.usage()
    except Exception: pass
    return K, dict(job_id=job.job_id(), wall_s=round(time.time() - t0, 1), qpu_s=usage, depth=tqc.depth(),
                   twoq=int(sum(v for g, v in tqc.count_ops().items() if g in ("cz", "ecr", "cx"))))


Ta, ia = hw_kernel(Xa, Xa, True); Tb, ib = hw_kernel(Xb, Xa, False)
off = ~np.eye(len(Xa), dtype=bool); fit = np.polyfit(np.r_[Ka[off], Kb.ravel()], np.r_[Ta[off], Tb.ravel()], 1)
res = dict(backend=backend.name, lam=a.lam, n_train=len(Xa), n_test=len(Xb), shots=a.shots,
           kernel_corr_train=float(np.corrcoef(Ta[off], Ka[off])[0, 1]), kernel_corr_test=float(np.corrcoef(Tb.ravel(), Kb.ravel())[0, 1]),
           kernel_mae=float(np.abs(np.r_[Ta[off] - Ka[off], (Tb - Kb).ravel()]).mean()), affine_a=float(fit[0]), affine_b=float(fit[1]),
           qpu_s=(ia["qpu_s"] or 0) + (ib["qpu_s"] or 0), jobs=[ia["job_id"], ib["job_id"]], depth=ia["depth"], twoq=ia["twoq"])
for name, (A, B) in {"exact": (Ka, Kb), "hardware": (Ta, Tb)}.items():
    m = SVC(kernel="precomputed", C=1.0, class_weight="balanced").fit(A, ya); s = m.decision_function(B)
    res[f"auc_{name}"] = float(roc_auc_score(yb, s)); res[f"pred_{name}"] = (s > 0).astype(int).tolist()
res["decision_agreement"] = float(np.mean(np.array(res["pred_exact"]) == np.array(res["pred_hardware"])))
json.dump(res, open(out / "hw_final_results.json", "w"), indent=1)
json.dump(dict(K_train=Ta.tolist(), K_test=Tb.tolist(), K_train_exact=Ka.tolist(), K_test_exact=Kb.tolist(),
               y_train=ya.tolist(), y_test=yb.tolist()), open(out / "hw_final_kernels.json", "w"))
print({k: v for k, v in res.items() if not k.startswith("pred_")})
