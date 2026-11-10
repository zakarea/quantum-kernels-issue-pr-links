"""Hardware kernel-fidelity validation (reviewer request: more and deeper hardware evidence).

Measures kernel entries k(x, x') = |<phi(x)|phi(x')>|^2 on IBM hardware for random pairs of the paper's 12-dimensional
inputs (PLS + metadata + BM25, PI-Link) from several seeds, each at its tuned bandwidth, and compares them with exact
simulation and with the calibrated device model of the paper (K_device = a * K_exact + b).

  python hw_validation.py --tag z12   --map Z   --seeds 0 1 2 --pairs 135 --token KEY --instance CRN
  python hw_validation.py --tag zz12  --map ZZ1 --seeds 0 1 2 --pairs 100 --token KEY --instance CRN
  python hw_validation.py --tag repeat --map Z  --seeds 0 --pairs 120 --avoid ibm_marrakesh --token KEY --instance CRN
  python hw_validation.py --tag dry --map ZZ1 --seeds 0 --pairs 6 --fake      # local dry run on FakeTorino

One job per run; its ID is saved at submission, so an interrupted run resumes the same job when re-run
(or pass --job_id). Results are saved to --out/<tag>.json and <tag>_pairs.csv, and a finished tag is skipped on re-run.
"""
import argparse, datetime, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from qml_pilink import load_links
from quantum_extra import prep
from qml_enhanced import states
from rq3_device import fmap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--map", choices=["Z", "ZZ1"], required=True)
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2]); ap.add_argument("--pairs", type=int, default=100)
ap.add_argument("--shots", type=int, default=1024); ap.add_argument("--out", default="hw_validation")
ap.add_argument("--token"); ap.add_argument("--instance", default=None); ap.add_argument("--backend", default=None)
ap.add_argument("--avoid", default=None, help="pick the least busy device other than this one")
ap.add_argument("--fake", action="store_true"); ap.add_argument("--job_id", default=None, help="collect an already submitted job instead of submitting"); ap.add_argument("--data", default="pilink_pairs_meta.csv")
ap.add_argument("--params", default="results_final30/a/raw_results.csv")
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); D = 12
if (out / f"{a.tag}.json").exists():
    print(a.tag, "already done:", json.load(open(out / f"{a.tag}.json"))); sys.exit(0)

# ---- inputs: random pairs of the paper's inputs, each seed at its tuned bandwidth -----------------
model = {"Z": "QSVC-Z", "ZZ1": "QSVC-ZZ1"}[a.map]
saved = pd.read_csv(a.params); lam_of = {int(r.seed): json.loads(r.params)["lam"] for r in saved[saved.model == model].itertuples()}
links = load_links(a.data); X1, X2, exact, meta = [], [], [], []
for seed in a.seeds:
    P = prep(links, seed, D); lam = lam_of[seed]; X = lam * np.vstack([P["Xtr"], P["Xte"]])
    rng = np.random.default_rng(1000 + seed); I = rng.choice(len(X), size=(a.pairs, 2), replace=True)
    I = I[I[:, 0] != I[:, 1]]
    S = states(a.map, X[np.unique(I)]); pos = {k: n for n, k in enumerate(np.unique(I))}
    for i, j in I:
        X1.append(X[i]); X2.append(X[j]); exact.append(abs(np.vdot(S[pos[i]], S[pos[j]])) ** 2); meta.append((seed, lam, int(i), int(j)))
X1, X2, exact = np.array(X1), np.array(X2), np.array(exact)
print(f"{len(exact)} kernel entries, {a.map} map, seeds {a.seeds}, tuned lambda {[lam_of[s] for s in a.seeds]}", flush=True)

# ---- backend --------------------------------------------------------------------------------------
from qiskit_ibm_runtime import SamplerV2, QiskitRuntimeService
if a.fake:
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    backend = FakeTorino()
else:
    svc = QiskitRuntimeService(channel="ibm_quantum_platform", token=a.token, instance=a.instance or None)
jobfile = out / f"{a.tag}_job.txt"
prev = a.job_id or (jobfile.read_text().strip() if jobfile.exists() else None)
old_job = None
if prev and not a.fake:
    old_job = svc.job(prev); backend = old_job.backend()
    print("resuming job", prev, "on", backend.name, "- status:", old_job.status(), flush=True)
elif not a.fake:
    if a.backend:
        backend = svc.backend(a.backend)
    else:
        cands = [b for b in svc.backends(operational=True, simulator=False, min_num_qubits=D) if b.name != a.avoid]
        backend = min(cands, key=lambda b: b.status().pending_jobs)
print("backend:", backend.name, flush=True)

# ---- circuits: compute-uncompute U(x')^dagger U(x); entry = frequency of the all-zeros outcome ----------
pm = generate_preset_pass_manager(optimization_level=3, backend=backend, seed_transpiler=0)
fx, fy = fmap(a.map, D, "x"), fmap(a.map, D, "y")
qc = fx.compose(fy.inverse()); qc.measure_all(); tqc = pm.run(qc)
order = [(p.name[0], int(p.name.split("[")[1][:-1])) for p in tqc.parameters]
vals = np.array([[x[k] if s == "x" else y[k] for s, k in order] for x, y in zip(X1, X2)])
depth = tqc.depth(); twoq = int(sum(v for g, v in tqc.count_ops().items() if g in ("cz", "ecr", "cx")))
print(f"transpiled: depth {depth}, two-qubit gates {twoq}", flush=True)
sampler = SamplerV2(mode=backend); sampler.options.default_shots = a.shots
try: sampler.options.dynamical_decoupling.enable = True
except Exception: pass
t0 = time.time()
if old_job is not None:
    job = old_job; print("waiting for job", job.job_id(), "(no new job submitted)", flush=True)
else:
    job = sampler.run([(tqc, vals)]); jobfile.write_text(job.job_id())
    print("job", job.job_id(), "submitted and its ID saved to Drive; waiting in queue...", flush=True)
bits = job.result()[0].data.meas
hw = np.array([bits[k].get_counts().get("0" * D, 0) for k in range(len(exact))]) / a.shots
usage = None
try: usage = job.usage()
except Exception: pass

# ---- analysis ---------------------------------------------------------------------------------------
calib = json.load(open("results_extra/device_calibration.json"))[a.map]
A, B = np.polyfit(exact, hw, 1); resid = hw - (A * exact + B)
p = np.clip(A * exact + B, 1e-6, 1 - 1e-6); shot_sd = float(np.sqrt(np.mean(p * (1 - p) / a.shots)))
rng = np.random.default_rng(0); boots = []
for _ in range(2000):
    k = rng.integers(0, len(exact), len(exact)); boots.append(np.polyfit(exact[k], hw[k], 1))
boots = np.array(boots)
res = dict(tag=a.tag, map=a.map, backend=backend.name, date_utc=datetime.datetime.utcnow().isoformat(timespec="seconds"),
           job_id=job.job_id(), qpu_s=usage, wall_s=round(time.time() - t0, 1), seeds=a.seeds, lambdas=[lam_of[s] for s in a.seeds],
           n_entries=int(len(exact)), shots=a.shots, depth=depth, twoq=twoq,
           r=float(np.corrcoef(exact, hw)[0, 1]), affine_a=float(A), affine_b=float(B),
           affine_a_ci95=[float(np.percentile(boots[:, 0], 2.5)), float(np.percentile(boots[:, 0], 97.5))],
           predicted_a=calib["a"], predicted_b=calib["b"],
           mae_raw=float(np.mean(np.abs(hw - exact))), mae_after_affine=float(np.mean(np.abs(resid))),
           mae_vs_device_model=float(np.mean(np.abs(hw - (calib["a"] * exact + calib["b"])))),
           resid_sd=float(resid.std()), shot_noise_sd=shot_sd, exact_range=[float(exact.min()), float(exact.max())])
json.dump(res, open(out / f"{a.tag}.json", "w"), indent=1)
pd.DataFrame([dict(seed=m[0], lam=m[1], i=m[2], j=m[3], k_exact=e, k_hw=h) for m, e, h in zip(meta, exact, hw)]).to_csv(out / f"{a.tag}_pairs.csv", index=False)
print(json.dumps(res, indent=1), flush=True)
