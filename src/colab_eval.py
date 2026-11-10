"""Transformer and LLM baselines on the exact splits of final30.py (run on a Colab GPU).

  e5-zeroshot     cosine of intfloat/e5-base-v2 embeddings (no training)
  XGB-Ranker+e5   the best classical model (all features + BM25 + metadata) plus the e5 cosine feature
  CE-<base>-800   cross-encoder fine-tuned on the same 800 training pairs as every other model
  CE-MiniLM-all   cross-encoder fine-tuned on ALL training pairs (scale reference)
  LLM-zeroshot    Qwen2.5-1.5B-Instruct, score = logit(Yes) - logit(No)

  python colab_eval.py --data pilink_pairs_meta.csv --split repo --out results_tf/pilink
  python colab_eval.py --data ghpr_pairs_meta.csv --split temporal --out results_tf/ghpr
  add --smoke for a 1-seed, tiny-size check first.
"""
import argparse, hashlib, json, sys, time, traceback
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).parent))
from qml_pilink import load_links, metrics
from qml_enhanced import EnhancedFeaturizer, rank_eval
from splits import get_split, flat, temporal_links
from bm25 import BM25

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def e5_embed(texts, prefix, tok, model, bs=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = tok([prefix + t for t in texts[i:i + bs]], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEV)
            h = model(**b).last_hidden_state; m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = (h * m).sum(1) / m.sum(1); out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
    return np.vstack(out)


def train_cross_encoder(base, A, B, y, seed, epochs=3, bs=16, lr=2e-5, max_len=256):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    torch.manual_seed(seed); np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=1, ignore_mismatched_sizes=True).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(y) / bs)); sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=DEV == "cuda"); lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    for ep in range(epochs):
        perm = np.random.permutation(len(y))
        for i in range(0, len(y), bs):
            j = perm[i:i + bs]
            b = tok([A[k] for k in j], [B[k] for k in j], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(DEV)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEV == "cuda"):
                loss = lossf(model(**b).logits.squeeze(-1), torch.tensor(y[j], dtype=torch.float32, device=DEV))
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
    model.eval()

    def score(A2, B2, bs2=64):
        out = []
        with torch.no_grad():
            for i in range(0, len(A2), bs2):
                b = tok(list(A2[i:i + bs2]), list(B2[i:i + bs2]), padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(DEV)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEV == "cuda"):
                    out.append(model(**b).logits.squeeze(-1).float().cpu().numpy())
        return np.concatenate(out)
    return score, model


class LLMScorer:
    PROMPT = ("You are given a GitHub issue and a pull request from the same repository.\n\n"
              "ISSUE:\n{iss}\n\nPULL REQUEST:\n{pr}\n\nDoes this pull request fix this issue? Answer Yes or No.")

    def __init__(self, name="Qwen/Qwen2.5-1.5B-Instruct"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(name); self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(DEV).eval()
        self.yes = self.tok.encode("Yes", add_special_tokens=False)[0]; self.no = self.tok.encode("No", add_special_tokens=False)[0]

    def score(self, A, B, bs=16, chars=1500):
        out = []
        with torch.no_grad():
            for i in range(0, len(A), bs):
                msgs = [[{"role": "user", "content": self.PROMPT.format(iss=a[:chars], pr=b[:chars])}] for a, b in zip(A[i:i + bs], B[i:i + bs])]
                txt = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
                enc = self.tok(txt, return_tensors="pt", padding=True).to(DEV)
                lg = self.model(**enc).logits[:, -1, :].float()
                out.append((lg[:, self.yes] - lg[:, self.no]).cpu().numpy())
        return np.concatenate(out)


def run(a):
    import xgboost as xgb
    from transformers import AutoTokenizer, AutoModel
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    links = load_links(a.data); lt = temporal_links(links) if a.split == "temporal" else None
    rows = []; csv = out / "raw_results.csv"
    if csv.exists():
        rows = pd.read_csv(csv).to_dict("records")
    done = {(r["seed"], r["model"]) for r in rows}
    etok = AutoTokenizer.from_pretrained("intfloat/e5-base-v2"); emod = AutoModel.from_pretrained("intfloat/e5-base-v2").to(DEV).eval()
    if DEV == "cuda": emod.half()
    llm = None
    for seed in range(a.seeds):
        tr, te, rs, rtr, tr_all = get_split(links, seed, a.split, a.n_train, a.n_test, a.n_rank, lt)
        rtr = rtr[:a.n_rank]; iv, pv = flat(rs); ri, rp = flat(rtr); sizes = [len(c) for _, c, _ in rs]
        yte, ytr = te.y.values, tr.y.values
        I, Pt = links.issue.values, links.pr.values
        split_hash = hashlib.md5(np.concatenate([te.iss.values, te.pr.values, iv, pv]).astype(np.int64).tobytes()).hexdigest()[:12]
        print("seed", seed, "split", split_hash, flush=True)

        def add(name, s, srk, ft, params=None, thr=None):
            if (seed, name) in done: return
            thr = np.median(s) if thr is None else thr
            r = dict(seed=seed, model=name, split_hash=split_hash, fit_s=round(ft, 1), params=json.dumps(params or {}),
                     **metrics(yte, (s > thr).astype(int), s), **rank_eval(srk, sizes))
            rows.append(r); pd.DataFrame(rows).to_csv(csv, index=False)
            print(seed, name, "AUC", round(r["auc"], 3), "MRR", round(r["mrr"], 3), flush=True)

        try:  # e5 zero-shot + as a feature for the best classical ranker
            t0 = time.time()
            need_i = np.unique(np.concatenate([te.iss.values, iv, ri, tr.iss.values])); need_p = np.unique(np.concatenate([te.pr.values, pv, rp, tr.pr.values]))
            Ei = dict(zip(need_i, e5_embed(list(I[need_i]), "query: ", etok, emod)))
            Ep = dict(zip(need_p, e5_embed(list(Pt[need_p]), "passage: ", etok, emod)))
            cos = lambda ii, pp: np.array([Ei[x] @ Ep[y] for x, y in zip(ii, pp)])
            add("e5-zeroshot", cos(te.iss.values, te.pr.values), cos(iv, pv), time.time() - t0)
            t0 = time.time()
            bm = BM25().fit(Pt[np.unique(tr_all.pr.values)])
            feat = EnhancedFeaturizer(12, "pls", True, seed, bm).fit(links, tr.iss.values, tr.pr.values, ytr)
            F = lambda ii, pp: np.hstack([feat.full(links, ii, pp), cos(ii, pp)[:, None]])
            yr = np.concatenate([[1] + [0] * (len(c) - 1) for _, c, _ in rtr])
            qid = np.concatenate([np.full(len(c), k) for k, (_, c, _) in enumerate(rtr)])
            rk = xgb.XGBRanker(objective="rank:pairwise", n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8,
                               colsample_bytree=0.8, random_state=seed).fit(F(ri, rp), yr, qid=qid)
            add("XGB-Ranker+e5", rk.predict(F(te.iss.values, te.pr.values)), rk.predict(F(iv, pv)), time.time() - t0)
        except Exception:
            traceback.print_exc()

        for base, tag in [("microsoft/codebert-base", "CodeBERT"), ("cross-encoder/ms-marco-MiniLM-L-12-v2", "MiniLM")]:
            if (seed, f"CE-{tag}-800") in done: continue
            try:
                t0 = time.time()
                sc, m = train_cross_encoder(base, list(I[tr.iss.values]), list(Pt[tr.pr.values]), ytr.astype(float), seed, epochs=a.epochs)
                add(f"CE-{tag}-800", sc(I[te.iss.values], Pt[te.pr.values]), sc(I[iv], Pt[pv]), time.time() - t0, dict(base=base))
                del m; torch.cuda.empty_cache()
            except Exception:
                traceback.print_exc()

        if seed < a.full_seeds and (seed, "CE-MiniLM-all") not in done:
            try:
                t0 = time.time(); ta = tr_all if a.max_full is None else tr_all.sample(min(len(tr_all), a.max_full), random_state=seed)
                sc, m = train_cross_encoder("cross-encoder/ms-marco-MiniLM-L-12-v2", list(I[ta.iss.values]), list(Pt[ta.pr.values]),
                                            ta.y.values.astype(float), seed, epochs=2)
                add("CE-MiniLM-all", sc(I[te.iss.values], Pt[te.pr.values]), sc(I[iv], Pt[pv]), time.time() - t0, dict(n_train=len(ta)))
                del m; torch.cuda.empty_cache()
            except Exception:
                traceback.print_exc()

        if seed < a.llm_seeds and (seed, "LLM-zeroshot") not in done:
            try:
                t0 = time.time(); llm = llm or LLMScorer()
                add("LLM-zeroshot", llm.score(I[te.iss.values], Pt[te.pr.values]), llm.score(I[iv], Pt[pv]), time.time() - t0,
                    dict(model="Qwen2.5-1.5B-Instruct"), thr=0.0)
            except Exception:
                traceback.print_exc()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--split", default="repo", choices=["repo", "temporal"])
    p.add_argument("--out", required=True); p.add_argument("--seeds", type=int, default=7)
    p.add_argument("--n_train", type=int, default=800); p.add_argument("--n_test", type=int, default=400)
    p.add_argument("--n_rank", type=int, default=200); p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--full_seeds", type=int, default=3); p.add_argument("--max_full", type=int, default=None)
    p.add_argument("--llm_seeds", type=int, default=3); p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.seeds, a.n_train, a.n_test, a.n_rank, a.epochs, a.full_seeds, a.max_full, a.llm_seeds = 1, 64, 40, 10, 1, 1, 200, 1
        a.out = a.out + "_smoke"
    run(a)
