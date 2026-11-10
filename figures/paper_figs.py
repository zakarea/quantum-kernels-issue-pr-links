"""Publication figures for the TOSEM paper (vector PDF, ACM acmsmall text width ~ 5.5 in)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("paper_figures"); OUT.mkdir(exist_ok=True)
W = 5.5
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.4, "lines.markersize": 4,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
# fixed colour per entity (colour-blind-safe categorical order); quantum dashed, classical solid
COL = {"PQK-Z": "#2a78d6", "QSVC-Z": "#1baf7a", "QSVC-ZZ1": "#e87ba4", "SVC-RBF": "#eb6834", "LogReg": "#eda100",
       "XGB-Ranker(full)": "#3a3a36", "XGB(full)": "#8a8980", "LGBM(full)": "#b5b4ad", "BM25": "#c3c2b7",
       "Z": "#1baf7a", "ZZ1": "#e87ba4", "Pauli": "#2a78d6"}
LAB = {"PQK-Z": "PQK-Z", "QSVC-Z": "QSVC-Z", "QSVC-ZZ1": "QSVC-ZZ1", "SVC-RBF": "SVC-RBF", "LogReg": "LogReg",
       "XGB-Ranker(full)": "XGB-Ranker (all feat.)", "XGB(full)": "XGBoost (all feat.)", "LGBM(full)": "LightGBM (all feat.)",
       "BM25": "BM25"}
Q = {"PQK-Z", "QSVC-Z", "QSVC-ZZ1"}


def style(m):
    return dict(color=COL[m], ls="--" if m in Q else "-", marker="o" if m in Q else "s", label=LAB.get(m, m))


# ---------------------------------------------------------------- Fig 2: forest plot of confirmatory differences
def fig_forest():
    sets = [("PI-Link", "results_final30"), ("PI-Link + e5", "results_e5/pilink"), ("GHPR", "results_ghpr"), ("GHPR + e5", "results_e5/ghpr")]
    comps = [("PQK-Z", "SVC-RBF"), ("QSVC-Z", "SVC-RBF"), ("QSVC-ZZ1", "SVC-RBF")]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5), sharey=True)
    for ax, met, eps in zip(axes, ["mrr", "auc"], [0.02, 0.01]):
        ax.axvspan(-eps, eps, color="#eef3fb", zorder=0); ax.axvline(0, color="#8a8980", lw=0.6)
        y, ticks, labels = 0, [], []
        for sname, path in sets:
            t = pd.read_csv(f"{path}/confirmatory_tests.csv"); t = t[(t["sample"] == "all 30 seeds") & (t.metric == met)]
            for k, (a, b) in enumerate(comps):
                r = t[(t.A == a) & (t.B == b)].iloc[0]
                ax.errorbar(r["diff"], y, xerr=[[r["diff"] - r.ci_lo], [r.ci_hi - r["diff"]]], fmt="o" if a != "QSVC-ZZ1" else "D",
                            color=COL[a], ms=3.5, lw=1.1, capsize=1.5, label=f"{a} − SVC-RBF" if (sname == "PI-Link") else None)
                y -= 1
            ticks.append(y + 2); labels.append(sname); y -= 0.8
        ax.set_yticks(ticks); ax.set_yticklabels(labels); ax.grid(axis="y", visible=False)
        ax.set_xlabel(f"Paired difference in {met.upper()} (95% CI)")
        ax.set_title(f"{met.upper()}: equivalence margin ±{eps}", loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(OUT / "fig_forest.pdf")


# ---------------------------------------------------------------- Fig 3: concentration and qubits
def fig_qubits():
    c = pd.read_csv("results_extra/concentration.csv").groupby(["fmap", "d", "lam"]).k_std.mean().reset_index()
    q = pd.read_csv("results_extra/qubits_results.csv").groupby(["d", "model"]).mrr.mean().reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.4))
    for fm in ["Z", "ZZ1", "Pauli"]:
        for lam, ls, mk in [(1.0, "-", "o"), (0.1, ":", "^")]:
            g = c[(c.fmap == fm) & (c.lam == lam)]
            ax[0].plot(g.d, g.k_std, ls=ls, marker=mk, color=COL[fm], label=f"{fm}, λ = {lam:g}")
    ax[0].set_yscale("log"); ax[0].set_xlabel("Qubits"); ax[0].set_ylabel("SD of off-diagonal kernel values")
    ax[0].set_title("(a) Kernel concentration", loc="left"); ax[0].legend(frameon=False, ncol=3, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.30))
    for m in ["PQK-Z", "QSVC-Z", "SVC-RBF", "LogReg"]:
        g = q[q.model == m]; ax[1].plot(g.d, g.mrr, **style(m))
    ax[1].set_xticks([4, 8, 12, 16, 20, 24]); ax[1].set_xlabel("Qubits (= input dimension)"); ax[1].set_ylabel("MRR")
    ax[1].set_title("(b) Accuracy vs. qubits (Z map, exact)", loc="left"); ax[1].legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "fig_qubits.pdf")


# ---------------------------------------------------------------- Fig 4: shots
def fig_shots():
    old = pd.read_csv("results_extra/shots_results.csv"); old = old[old.device == True]
    new = pd.read_csv("results_review/shots_retuned.csv")
    f30 = pd.concat([pd.read_csv(f"results_final30/{h}/raw_results.csv") for h in "ab"])
    svc = f30[(f30.model == "SVC-RBF") & (f30.seed < 10)].mrr.mean()
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.3))
    ax.axhspan(svc - 0.02, svc + 0.02, color="#fbeee8", zorder=0); ax.axhline(svc, color=COL["SVC-RBF"], lw=1.2, label="SVC-RBF (exact) ± margin")
    for m, src, ls, lab in [("PQK-Z", old, "--", "PQK-Z, λ from exact"), ("PQK-Z", new, "-", "PQK-Z, λ re-tuned"),
                            ("QSVC-Z", old, "--", "QSVC-Z, λ from exact"), ("QSVC-Z", new, "-", "QSVC-Z, λ re-tuned"),
                            ("QSVC-ZZ1", old, "--", "QSVC-ZZ1, λ from exact")]:
        g = src[src.model == m].groupby("shots").mrr.mean()
        ax.plot(g.index, g.values, ls=ls, marker="o" if ls == "--" else "s", color=COL[m], label=lab)
    ax.set_xscale("log"); ax.set_xticks([1024, 4096, 16384]); ax.set_xticklabels(["1,024", "4,096", "16,384"]); ax.minorticks_off()
    ax.set_xlabel("Shots per circuit (with ibm_torino noise)"); ax.set_ylabel("MRR (12 qubits)")
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig_shots.pdf")


# ---------------------------------------------------------------- Fig 5: scale
def fig_scale():
    d = pd.read_csv("results_scale/scale_results.csv"); d["n"] = d.n_train.where(d.n_train < 20000, 31000)
    m = d.groupby(["n", "model"]).mrr.mean().reset_index()
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.3))
    for mo in ["XGB-Ranker(full)", "XGB(full)", "LGBM(full)", "SVC-RBF", "QSVC-Z", "LogReg"]:
        g = m[m.model == mo]; ax.plot(g.n, g.mrr, **style(mo))
    ax.set_xscale("log"); ax.set_xticks([800, 3200, 12800, 31000]); ax.set_xticklabels(["800", "3,200", "12,800", "all"]); ax.minorticks_off()
    ax.set_xlabel("Training pairs"); ax.set_ylabel("MRR"); ax.legend(frameon=False, fontsize=6, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout(); fig.savefig(OUT / "fig_scale.pdf")


# ---------------------------------------------------------------- Fig 6: sampled vs full-pool retrieval
def fig_realistic():
    r = pd.read_csv("results_realistic/realistic_results.csv")
    samp = {"pilink": pd.concat([pd.read_csv(f"results_final30/{h}/raw_results.csv") for h in "ab"]), "ghpr": pd.read_csv("results_ghpr/raw_results.csv")}
    models = ["XGB-Ranker(full)", "LogReg", "QSVC-Z", "PQK-Z", "SVC-RBF", "BM25"]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.1), sharey=True)
    for ax, ds, title in zip(axes, ["pilink", "ghpr"], ["(a) PI-Link (mean pool 63)", "(b) GHPR (mean pool 307)"]):
        s = samp[ds][samp[ds].seed < 10]
        for k, mo in enumerate(models):
            full = r[(r.dataset == ds) & (r.model == mo)].mrr.mean()
            sm = s[s.model == mo].mrr.mean() if mo != "BM25" else np.nan
            if not np.isnan(sm):
                ax.plot([sm, full], [k, k], color="#c3c2b7", lw=1.0, zorder=1)
                ax.plot(sm, k, "o", mfc="white", mec=COL[mo], ms=4.5, zorder=2)
            ax.plot(full, k, "o", color=COL[mo], ms=4.5, zorder=3)
        bm = r[(r.dataset == ds) & (r.model == "BM25")].mrr.mean(); ax.axvline(bm, color="#8a8980", lw=0.6, ls=":")
        ax.set_yticks(range(len(models))); ax.set_yticklabels([LAB[m] for m in models]); ax.grid(axis="y", visible=False)
        ax.set_xlabel("MRR"); ax.set_title(title, loc="left"); ax.invert_yaxis()
    axes[0].plot([], [], "o", mfc="white", mec="#3a3a36", label="1 + 19 sampled candidates")
    axes[0].plot([], [], "o", color="#3a3a36", label="All PRs in time window")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(); fig.savefig(OUT / "fig_realistic.pdf")


# ---------------------------------------------------------------- Fig 7: SHAP groups
def fig_shap():
    s = pd.read_csv("results_analysis/shap.csv"); t = s.groupby(["dataset", "group"]).mean_abs_shap.mean().unstack(0)
    t = t / t.sum() * 100; t = t.sort_values("pilink")
    t.index = [i.replace("a*b", "a \u2299 b").replace("|a-b|", "|a \u2212 b|") for i in t.index]
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.0)); y = np.arange(len(t))
    ax.barh(y + 0.2, t.pilink, 0.38, color="#2a78d6", label="PI-Link"); ax.barh(y - 0.2, t.ghpr, 0.38, color="#eb6834", label="GHPR")
    ax.set_yticks(y); ax.set_yticklabels(t.index); ax.set_xlabel("Share of mean |SHAP| (%)"); ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig_shap.pdf")


if __name__ == "__main__":
    for f in (fig_forest, fig_qubits, fig_shots, fig_scale, fig_realistic, fig_shap):
        f(); print("ok", f.__name__)
