"""BM25 alone (unsupervised) on the sampled 1+19 ranking queries, seeds 0-6, both datasets.
BM25's IDF statistics are fitted on the training PRs of each seed, as in classical_plus.py."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from qml_pilink import load_links
from qml_enhanced import rank_eval
from splits import get_split, flat, temporal_links
from bm25 import BM25
rows = []
for name, data, split in [("pilink", "pilink_pairs_meta.csv", "repo"), ("ghpr", "ghpr_pairs_meta.csv", "temporal")]:
    links = load_links(data); lt = temporal_links(links) if split == "temporal" else None
    for seed in range(7):
        tr, te, rs, rtr, tr_all = get_split(links, seed, split, links_t=lt)
        iv, pv = flat(rs)
        bm = BM25().fit(links.pr.values[np.unique(tr_all.pr.values)])
        r = rank_eval(bm.score(links.issue.values[iv], links.pr.values[pv]), [len(c) for _, c, _ in rs])
        rows.append(dict(dataset=name, seed=seed, **r)); print(name, seed, round(r["mrr"], 3), flush=True)
t = pd.DataFrame(rows); Path("results_review").mkdir(exist_ok=True); t.to_csv("results_review/bm25_alone.csv", index=False)
print(t.groupby("dataset")[["hit1", "hit5", "mrr"]].agg(["mean", "std"]).round(3))
