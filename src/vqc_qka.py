"""Fair variational and trainable-kernel baselines (exact statevector, JAX gradients).

Final setting (PLS + open-time metadata + BM25), d = 8 qubits, same seeds/splits as qml_enhanced.py.

  VQC-RU    data re-uploading classifier: L layers of [RY(w*x+b), RZ(v)] per qubit + CZ ring,
            readout = linear combination of <Z_k>; logistic loss, Adam, early stopping.
            L in {2, 4} and learning rate chosen on a repository-grouped validation fold.
  QKA-ZZ1   trainable ZZ1 feature map: per-qubit input scales and per-edge entangling scales,
            trained to maximise kernel-target alignment, then QSVC (C by CV).
  QKA-Z     same for the non-entangling Z map (per-qubit scales).
  References: QSVC-Z / QSVC-ZZ1 with tuned bandwidth, SVC-RBF, LogReg, all on the same features.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import jax, jax.numpy as jnp, optax
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).parent))
from qml_pilink import load_links, metrics
from qml_enhanced import rank_eval, cv_auc, states, CS, LAMS
from quantum_extra import prep, evaluate

D = 8


# ---------------------------------------------------------------- statevector primitives (batched)
def apply_1q(S, U, k, n):
    """S: (N, 2^n); U: (N, 2, 2) per-sample gate on qubit k (Qiskit little-endian)."""
    N = S.shape[0]
    S = S.reshape(N, 2 ** (n - k - 1), 2, 2 ** k)
    S = jnp.einsum("nij,najb->naib", U, S)
    return S.reshape(N, 2 ** n)


def ry(t):
    c, s = jnp.cos(t / 2), jnp.sin(t / 2)
    return jnp.stack([jnp.stack([c, -s], -1), jnp.stack([s, c], -1)], -2).astype(jnp.complex128)


def rz(t):
    e = jnp.exp(-0.5j * t); z = jnp.zeros_like(e)
    return jnp.stack([jnp.stack([e, z], -1), jnp.stack([z, jnp.conj(e)], -1)], -2)


def bits(n):
    z = np.arange(2 ** n)
    return ((z[:, None] >> np.arange(n)) & 1).astype(np.float64)


B8 = jnp.asarray(bits(D))
CZ_RING = jnp.asarray(np.prod([np.where((bits(D)[:, k] == 1) & (bits(D)[:, (k + 1) % D] == 1), -1.0, 1.0) for k in range(D)], 0))
ZSIGN = jnp.asarray(1 - 2 * bits(D))  # (2^n, n): eigenvalue of Z_k on each basis state


def vqc_forward(params, X, L):
    N = X.shape[0]
    S = jnp.zeros((N, 2 ** D), jnp.complex128).at[:, 0].set(1.0)
    for l in range(L):
        for k in range(D):
            S = apply_1q(S, ry(params["w"][l, k] * X[:, k] + params["b"][l, k]), k, D)
            S = apply_1q(S, rz(jnp.broadcast_to(params["v"][l, k], (N,))), k, D)
        S = S * CZ_RING
    ez = (jnp.abs(S) ** 2) @ ZSIGN  # (N, n) <Z_k>
    return ez @ params["a"] + params["c"]


def train_vqc(Xtr, ytr, Xva, yva, L, lr, seed, steps=300):
    rng = np.random.default_rng(seed)
    params = dict(w=jnp.asarray(rng.normal(1.0, 0.1, (L, D))), b=jnp.asarray(rng.normal(0, 0.1, (L, D))),
                  v=jnp.asarray(rng.normal(0, 0.1, (L, D))), a=jnp.asarray(rng.normal(0, 0.1, D)), c=jnp.asarray(0.0))
    yt = jnp.asarray(2 * ytr - 1.0)
    loss = lambda p: jnp.mean(jnp.logaddexp(0, -yt * vqc_forward(p, jnp.asarray(Xtr), L)))
    vg = jax.jit(jax.value_and_grad(loss)); opt = optax.adam(lr); st = opt.init(params)
    best, best_p, bad = -1, params, 0
    for it in range(steps):
        _, g = vg(params); up, st = opt.update(g, st); params = optax.apply_updates(params, up)
        if it % 10 == 9 and Xva is not None:
            from sklearn.metrics import roc_auc_score
            a = roc_auc_score(yva, np.asarray(vqc_forward(params, jnp.asarray(Xva), L)))
            if a > best: best, best_p, bad = a, params, 0
            else:
                bad += 1
                if bad >= 5: break
    return (best_p if Xva is not None else params), best


# ---------------------------------------------------------------- trainable kernels
def qka_states(theta, X, fm):
    s = theta["s"]; Xs = X * s
    phase = 2 * Xs @ B8.T
    if fm == "ZZ1":
        for k in range(D - 1):
            par = (B8[:, k] != B8[:, k + 1]).astype(jnp.float64)
            phase = phase + 2 * theta["e"][k] * jnp.outer((jnp.pi - Xs[:, k]) * (jnp.pi - Xs[:, k + 1]), par)
        return jnp.exp(1j * phase) / 2 ** (D / 2)
    S = jnp.exp(1j * phase) / 2 ** (D / 2)  # Z map, reps = 2: H, P, H, P
    N = S.shape[0]; h = 1; T = S
    while h < 2 ** D:
        T = T.reshape(N, -1, 2, h); T = jnp.stack([T[:, :, 0] + T[:, :, 1], T[:, :, 0] - T[:, :, 1]], 2); h *= 2
    return T.reshape(N, 2 ** D) / 2 ** (D / 2) * jnp.exp(1j * phase)


def kernel(theta, A, B, fm):
    SA, SB = qka_states(theta, A, fm), qka_states(theta, B, fm)
    return jnp.abs(SA @ SB.conj().T) ** 2


def train_qka(Xtr, ytr, fm, lam0, seed, steps=150, n_sub=400):
    rng = np.random.default_rng(seed); idx = rng.permutation(len(Xtr))[:n_sub]
    Xs, ys = jnp.asarray(Xtr[idx]), jnp.asarray(2 * ytr[idx] - 1.0)
    Y = jnp.outer(ys, ys)
    def neg_kta(th):
        K = kernel(th, Xs, Xs, fm); Kc = K - K.mean(0) - K.mean(1)[:, None] + K.mean()  # centred alignment
        return -jnp.sum(Kc * Y) / (jnp.linalg.norm(Kc) * jnp.linalg.norm(Y))
    theta = dict(s=jnp.full(D, lam0), e=jnp.full(D - 1, 1.0))
    if fm == "Z": theta = dict(s=jnp.full(D, lam0))
    vg = jax.jit(jax.value_and_grad(neg_kta)); opt = optax.adam(0.02); st = opt.init(theta); hist = []
    for _ in range(steps):
        v, g = vg(theta); up, st = opt.update(g, st); theta = optax.apply_updates(theta, up); hist.append(float(v))
    return theta, -hist[0], -hist[-1]


def run(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    links = load_links(a.data); rows = []
    for seed in range(a.seeds):
        P = prep(links, seed, D); Xtr, Xte, Xrk, ytr = P["Xtr"], P["Xte"], P["Xrk"], P["ytr"]
        res = {}
        # VQC with data re-uploading; select L, lr on one grouped validation fold, then refit on all
        t0 = time.time(); tr_f, va_f = P["cv"][0]; cand = []
        for L in (2, 4):
            for lr in (0.03, 0.1):
                _, s = train_vqc(Xtr[tr_f], ytr[tr_f], Xtr[va_f], ytr[va_f], L, lr, seed); cand.append((s, L, lr))
        _, L, lr = max(cand)
        p, _ = train_vqc(Xtr[tr_f], ytr[tr_f], Xtr[va_f], ytr[va_f], L, lr, seed, steps=400)
        f = lambda X: np.asarray(vqc_forward(p, jnp.asarray(X), L))
        res["VQC-RU"] = (f(Xte), f(Xrk), time.time() - t0, dict(L=L, lr=lr, val_auc=round(max(cand)[0], 4)))
        # trainable kernels (alignment), init from the best fixed bandwidth for that map
        for fm in ("ZZ1", "Z"):
            t0 = time.time()
            lam0 = max(LAMS, key=lambda lam: max(cv_auc(np.abs(states(fm, lam * Xtr) @ states(fm, lam * Xtr).conj().T) ** 2, ytr, P["cv"], c) for c in CS))
            th, kta0, kta1 = train_qka(Xtr, ytr, fm, lam0, seed)
            K = lambda A, B: np.asarray(kernel(th, jnp.asarray(A), jnp.asarray(B), fm))
            Ktr = K(Xtr, Xtr); C = max(CS, key=lambda c: cv_auc(Ktr, ytr, P["cv"], c))
            m = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(Ktr, ytr)
            res[f"QKA-{fm}"] = (m.decision_function(K(Xte, Xtr)), m.decision_function(K(Xrk, Xtr)), time.time() - t0,
                                dict(lam0=lam0, C=C, kta_init=round(kta0, 4), kta_trained=round(kta1, 4)))
            # reference: same map with the fixed tuned bandwidth only
            t0 = time.time(); St = states(fm, lam0 * Xtr); Kf = np.abs(St @ St.conj().T) ** 2
            C = max(CS, key=lambda c: cv_auc(Kf, ytr, P["cv"], c))
            m = SVC(kernel="precomputed", C=C, class_weight="balanced").fit(Kf, ytr)
            res[f"QSVC-{fm}"] = (m.decision_function(np.abs(states(fm, lam0 * Xte) @ St.conj().T) ** 2),
                                 m.decision_function(np.abs(states(fm, lam0 * Xrk) @ St.conj().T) ** 2), time.time() - t0,
                                 dict(lam=lam0, C=C))
        t0 = time.time(); v = Xtr.var() * D
        g = GridSearchCV(SVC(class_weight="balanced"), {"C": CS, "gamma": [x / v for x in (0.3, 1, 3)]}, cv=P["cv"], scoring="roc_auc").fit(Xtr, ytr)
        res["SVC-RBF"] = (g.decision_function(Xte), g.decision_function(Xrk), time.time() - t0, g.best_params_)
        t0 = time.time()
        g = GridSearchCV(LogisticRegression(max_iter=3000, class_weight="balanced"), {"C": [0.01, 0.1, 1, 10]}, cv=P["cv"], scoring="roc_auc").fit(Xtr, ytr)
        res["LogReg"] = (g.decision_function(Xte), g.decision_function(Xrk), time.time() - t0, g.best_params_)
        for name, (s, srk, ft, bp) in res.items():
            rows.append(dict(seed=seed, d=D, model=name, fit_s=round(ft, 2), params=json.dumps({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in bp.items()}),
                             **metrics(P["yte"], (s > 0).astype(int), s), **rank_eval(srk, P["sizes"])))
        print(seed, {r["model"]: round(r["mrr"], 3) for r in rows if r["seed"] == seed}, flush=True)
        pd.DataFrame(rows).to_csv(out / "vqc_qka_results.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="pilink_pairs_meta.csv"); p.add_argument("--out", default="results_vqc_qka")
    p.add_argument("--seeds", type=int, default=7)
    run(p.parse_args())
