"""Re-run the Qwen2.5-1.5B zero-shot baseline with a memory-safe scorer (GHPR ran out of memory on a T4 GPU).

Same splits, prompt, 1,500-character truncation and score (logit Yes - logit No) as colab_eval.py. Only the memory use
changes: logits are computed for the last token only (not the full 151k-word vocabulary at every position), batches are
smaller and grouped by length. PI-Link is re-run as a check that the scores match the original run (MRR 0.484).

  python qwen_rerun.py --datasets ghpr pilink --seeds 0 1 2 --out DIR   -> DIR/{ghpr,pilink}.csv (resumable)
"""
import hashlib, json, os, sys, time
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).parent))
from qml_pilink import load_links, metrics
from qml_enhanced import rank_eval
from splits import get_split, flat, temporal_links

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT = ("You are given a GitHub issue and a pull request from the same repository.\n\n"
          "ISSUE:\n{iss}\n\nPULL REQUEST:\n{pr}\n\nDoes this pull request fix this issue? Answer Yes or No.")


class Scorer:
    def __init__(self, name="Qwen/Qwen2.5-1.5B-Instruct"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(name); self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(DEV).eval()
        self.yes = self.tok.encode("Yes", add_special_tokens=False)[0]; self.no = self.tok.encode("No", add_special_tokens=False)[0]

    def score(self, A, B, bs=8, chars=1500):
        txt = [self.tok.apply_chat_template([{"role": "user", "content": PROMPT.format(iss=a[:chars], pr=b[:chars])}],
                                            tokenize=False, add_generation_prompt=True) for a, b in zip(A, B)]
        order = np.argsort([len(t) for t in txt]); out = np.empty(len(txt), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(txt), bs):
                idx = order[i:i + bs]
                enc = self.tok([txt[k] for k in idx], return_tensors="pt", padding=True).to(DEV)
                h = self.model.model(**enc).last_hidden_state[:, -1, :]          # last position only
                lg = self.model.lm_head(h).float()
                out[idx] = (lg[:, self.yes] - lg[:, self.no]).cpu().numpy()
                del enc, h, lg
                if (i // bs) % 100 == 0: print(f"    {i}/{len(txt)}", flush=True)
        return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["ghpr", "pilink"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--out", default="results_qwen")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fp = json.load(open("split_fingerprints.json")); sc = None
    cfg = {"pilink": ("pilink_pairs_meta.csv", "repo"), "ghpr": ("ghpr_pairs_meta.csv", "temporal")}
    for name in a.datasets:
        data, split = cfg[name]; f = out / f"{name}.csv"
        rows = pd.read_csv(f).to_dict("records") if f.exists() else []   # resume: keep finished seeds
        done = {int(r["seed"]) for r in rows}
        todo = [s for s in a.seeds if s not in done]
        if not todo:
            print(name, "already done:", sorted(done), flush=True); continue
        links = load_links(data); lt = temporal_links(links) if split == "temporal" else None
        I, Pt = links.issue.values, links.pr.values
        for seed in todo:
            sc = sc or Scorer(); t0 = time.time()
            tr, te, rs, rtr, _ = get_split(links, seed, split, 800, 400, 200, lt)
            iv, pv = flat(rs); sizes = [len(c) for _, c, _ in rs]
            h = hashlib.md5(np.concatenate([te.iss.values, te.pr.values, iv, pv]).astype(np.int64).tobytes()).hexdigest()[:12]
            same = fp.get(name, {}).get(str(seed)) == h
            print(name, "seed", seed, "split ok" if same else "SPLIT MISMATCH", "- scoring", len(te) + len(iv), "pairs", flush=True)
            s = sc.score(I[te.iss.values], Pt[te.pr.values]); srk = sc.score(I[iv], Pt[pv])
            r = dict(dataset=name, seed=seed, split_hash=h, split_matches_paper=same, fit_s=round(time.time() - t0, 1),
                     **metrics(te.y.values, (s > 0).astype(int), s), **rank_eval(srk, sizes))
            rows.append(r); pd.DataFrame(rows).to_csv(f, index=False)   # saved after every seed
            print(name, seed, "AUC", round(r["auc"], 3), "MRR", round(r["mrr"], 3), f"{r['fit_s']}s  (saved)", flush=True)
            if DEV == "cuda": torch.cuda.empty_cache()
        print(name, "mean MRR over seeds", sorted(int(x["seed"]) for x in rows), round(pd.DataFrame(rows).mrr.mean(), 3), flush=True)


if __name__ == "__main__":
    main()
