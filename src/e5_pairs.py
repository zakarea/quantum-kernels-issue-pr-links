"""Compute e5-base-v2 cosine similarity for every (issue, PR) pair used by the 30-seed runs (Colab GPU).

Covers training pairs, test pairs, test ranking candidates and ranker training candidates,
for both datasets and all seeds, using the exact splits of final30.py. Output is small:
  e5_pairs_pilink.csv.gz, e5_pairs_ghpr.csv.gz  with columns iss, pr, cos
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).parent))
from qml_pilink import load_links
from splits import get_split, flat, temporal_links
from transformers import AutoTokenizer, AutoModel

DEV = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("intfloat/e5-base-v2")
model = AutoModel.from_pretrained("intfloat/e5-base-v2").to(DEV).eval()
if DEV == "cuda": model.half()


def embed(texts, prefix, bs=128):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = tok([prefix + t for t in texts[i:i + bs]], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEV)
            h = model(**b).last_hidden_state; m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = (h * m).sum(1) / m.sum(1); out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
            if (i // bs) % 50 == 0: print(f"  {prefix}{i}/{len(texts)}", flush=True)
    return np.vstack(out).astype(np.float32)


SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
for name, data, split in [("pilink", "pilink_pairs_meta.csv", "repo"), ("ghpr", "ghpr_pairs_meta.csv", "temporal")]:
    t0 = time.time(); links = load_links(data); lt = temporal_links(links) if split == "temporal" else None
    need = set()
    for seed in range(SEEDS):
        tr, te, rs, rtr, _ = get_split(links, seed, split, links_t=lt)
        iv, pv = flat(rs); ri, rp = flat(rtr[:200])
        for a, b in ((tr.iss.values, tr.pr.values), (te.iss.values, te.pr.values), (iv, pv), (ri, rp)):
            need.update(zip(a.tolist(), b.tolist()))
        print(name, "seed", seed, "pairs so far", len(need), flush=True)
    P = np.array(sorted(need)); ui, up = np.unique(P[:, 0]), np.unique(P[:, 1])
    print(name, len(P), "pairs,", len(ui), "issues,", len(up), "PRs", flush=True)
    Ei = dict(zip(ui, embed(list(links.issue.values[ui]), "query: ")))
    Ep = dict(zip(up, embed(list(links.pr.values[up]), "passage: ")))
    cos = np.array([Ei[i] @ Ep[p] for i, p in P])
    pd.DataFrame(dict(iss=P[:, 0], pr=P[:, 1], cos=cos.round(6))).to_csv(f"e5_pairs_{name}.csv.gz", index=False)
    print(name, "done in", round(time.time() - t0), "s", flush=True)
