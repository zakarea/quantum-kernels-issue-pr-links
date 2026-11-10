"""Exact train / test / ranking splits used by final30.py (so Colab baselines see identical data)."""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from qml_pilink import build_pairs, balanced_sample


def temporal_links(links):
    lt = links.copy()
    r = lt.groupby("repo").t_pr_open.rank(method="first", pct=True)
    lt["repo"] = lt.repo + np.where(r > 0.7, "|te", "|tr")
    return lt


def get_split(links, seed, split="repo", n_train=800, n_test=400, n_rank=200, links_t=None):
    """Returns tr, te (pair frames: iss, pr, y, repo), rs (test ranking sets), rtr (training ranking sets), tr_all."""
    if split == "temporal":
        pairs, rank_sets = build_pairs(links_t if links_t is not None else temporal_links(links), seed=seed)
        tr_i = np.where(pairs.repo.str.endswith("|tr"))[0]; te_i = np.where(pairs.repo.str.endswith("|te"))[0]
    else:
        pairs, rank_sets = build_pairs(links, seed=seed)
        tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(pairs, groups=pairs.repo))
    tr_all = pairs.iloc[tr_i]
    tr, te = balanced_sample(tr_all, n_train, seed), balanced_sample(pairs.iloc[te_i], n_test, seed)
    rs = [r for r in rank_sets if r[2] in set(te.repo)]
    rs = [rs[i] for i in np.random.default_rng(seed).permutation(len(rs))[:n_rank]]
    rtr = [r for r in rank_sets if r[2] in set(tr_all.repo)]
    rtr = [rtr[i] for i in np.random.default_rng(seed + 100).permutation(len(rtr))]
    return tr, te, rs, rtr, tr_all


def flat(S):
    return (np.concatenate([np.full(len(c), i) for i, c, _ in S]), np.concatenate([np.array(c) for _, c, _ in S]))
