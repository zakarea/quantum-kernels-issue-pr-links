"""Embed every issue and PR text with CodeBERT (mean-pooled last layer).

Rows follow qml_pilink.load_links(), so the output plugs straight into
  python qml_pilink.py --data pilink_pairs.csv --emb codebert_emb.npz ...
Run on a GPU (Colab T4: ~10 min for 27k links x 2 sides).
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
sys.path.insert(0, str(Path(__file__).parent))
from qml_pilink import load_links

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="pilink_pairs.csv"); ap.add_argument("--out", default="codebert_emb.npz")
ap.add_argument("--model", default="microsoft/codebert-base"); ap.add_argument("--max_len", type=int, default=256)
ap.add_argument("--bs", type=int, default=64)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(a.model); model = AutoModel.from_pretrained(a.model).to(dev).eval()
if dev == "cuda": model.half()
links = load_links(a.data)

@torch.no_grad()
def embed(texts):
    out = []
    for i in range(0, len(texts), a.bs):
        b = tok(list(texts[i:i + a.bs]), padding=True, truncation=True, max_length=a.max_len, return_tensors="pt").to(dev)
        h = model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        out.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
        if i // a.bs % 50 == 0: print(f"{i}/{len(texts)}", flush=True)
    return np.vstack(out).astype(np.float16)

np.savez_compressed(a.out, issue=embed(links.issue.values), pr=embed(links.pr.values), model=a.model)
print("saved", a.out, len(links))
