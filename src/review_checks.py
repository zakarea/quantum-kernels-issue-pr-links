"""Reviewer-requested checks.

  timeonly   Rank candidates by timing alone (no text): PRs opened at or after the issue first, closest first;
             PRs opened before the issue after them, closest first. Sampled 1+19 queries (30 seeds) and
             realistic full pools (10 seeds), same queries as the paper.
  shots      Shot + ibm_torino noise with the bandwidth RE-TUNED on the noisy kernels (QSVC-Z, PQK-Z),
             instead of reusing the bandwidth tuned on exact kernels. PI-Link, d = 12, 10 seeds.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from qml_pilink import load_links
from qml_enhanced import rank_eval, cv_auc, sqd, LAMS, CS, GS
from quantum_extra import prep, evaluate, z_kernel, z_qubits, z_bloch, sample_kernel, sample_bloch
from splits import get_split, flat, temporal_links
from realistic import pools


def time_score(links, iv, pv):
    g = links.t_pr_open.values[pv] - links.t_issue_open.values[iv]
    return np.where(g >= 0, -np.log1p(np.abs(g)), -1e3 - np.log1p(np.abs(g)))


def timeonly(out):
    rows = []
    for name, data, split in [("pilink", "pilink_pairs_meta.csv", "repo"), ("ghpr", "ghpr_pairs_meta.csv", "temporal")]:
        links = load_links(data); lt = temporal_links(links) if split == "temporal" else None
        groups_ = (lt if lt is not None else links).repo
        for seed in range(30):
            tr, te, rs, rtr, _ = get_split(links, seed, split, links_t=lt); iv, pv = flat(rs)
            r = rank_eval(time_score(links, iv, pv), [len(c) for _, c, _ in rs])
            rows.append(dict(dataset=name, seed=seed, setting="sampled 1+19", **r))
            if seed < 10:
                P = pools(links, groups_, [i for i, _, _ in rs], seed)
                iv2 = np.concatenate([np.full(len(c), c[0]) for c in P]); pv2 = np.concatenate(P)
                r = rank_eval(time_score(links, iv2, pv2), [len(c) for c in P])
                rows.append(dict(dataset=name, seed=seed, setting="full pool", **r))
        print(name, "done", flush=True)
    pd.DataFrame(rows).to_csv(Path(out) / "timeonly.csv", index=False)


def shots_retuned(out, seeds=10):
    links = load_links("pilink_pairs_meta.csv"); calib = json.load(open("results_extra/device_calibration.json"))
    a, b, sh = calib["Z"]["a"], calib["Z"]["b"], calib["Z"]["pqk_shrink"]; rows = []
    for seed in range(seeds):
        t0 = time.time(); P = prep(links, seed, 12); rng = np.random.default_rng(5000 + seed)
        Xtr, Xte, Xrk = P["Xtr"], P["Xte"], P["Xrk"]
        for shots in (1024, 4096, 16384):
            # QSVC-Z: noisy kernels for every lam; pick (lam, C) by CV on the NOISY training kernel
            cand = {}
            for lam in LAMS:
                q = z_qubits(lam * Xtr)
                Ktr = sample_kernel(z_kernel(lam * Xtr, lam * Xtr, q, q), shots, rng, a, b, symmetric=True)
                cand[lam] = (max((cv_auc(Ktr, P["ytr"], P["cv"], c), c) for c in CS), Ktr, q)
            lam = max(cand, key=lambda l: cand[l][0][0]); (_, C), Ktr, q = cand[lam]
            Kte = sample_kernel(z_kernel(lam * Xte, lam * Xtr, None, q), shots, rng, a, b)
            Krk = sample_kernel(z_kernel(lam * Xrk, lam * Xtr, None, q), shots, rng, a, b)
            rows.append(dict(seed=seed, model="QSVC-Z", shots=shots, lam=lam, **evaluate(Ktr, Kte, Krk, P, C=C)))
            # PQK-Z: noisy Bloch vectors for every lam; pick (lam, g, C) by CV on noisy kernels
            best = (-1, None)
            for lam in LAMS:
                bt = sample_bloch(z_bloch(lam * Xtr), shots, rng, sh); D2 = sqd(bt, bt)
                med = np.median(D2[~np.eye(len(D2), dtype=bool)]) + 1e-12
                for g in GS:
                    K = np.exp(-(g / med) * D2)
                    for c in CS:
                        s = cv_auc(K, P["ytr"], P["cv"], c)
                        if s > best[0]: best = (s, (lam, g / med, c, bt))
            lam, g, C, bt = best[1]
            be, br = sample_bloch(z_bloch(lam * Xte), shots, rng, sh), sample_bloch(z_bloch(lam * Xrk), shots, rng, sh)
            r = evaluate(np.exp(-g * sqd(bt, bt)), np.exp(-g * sqd(be, bt)), np.exp(-g * sqd(br, bt)), P, C=C)
            rows.append(dict(seed=seed, model="PQK-Z", shots=shots, lam=lam, **r))
        print(seed, round(time.time() - t0), "s", {(r["model"], r["shots"]): round(r["mrr"], 3) for r in rows if r["seed"] == seed}, flush=True)
        pd.DataFrame(rows).to_csv(Path(out) / "shots_retuned.csv", index=False)


if __name__ == "__main__":
    Path("results_review").mkdir(exist_ok=True)
    {"timeonly": timeonly, "shots": shots_retuned}[sys.argv[1]]("results_review")
