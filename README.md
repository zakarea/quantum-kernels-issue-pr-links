# Quantum Kernels for Issue–Pull Request Link Recovery: Replication Package

Code, data, raw results and figures for the paper
**"Quantum Kernels for Issue–Pull Request Link Recovery: A Rigorous Benchmark"**, submitted to *IEEE Transactions on Software Engineering* (TSE).

**Author:** Zakarea Alshara (corresponding author)

1. Department of Information Technology, Ajman University, Ajman, United Arab Emirates (z.alshara@ajman.ac.ae)
2. Department of Software Engineering, Jordan University of Science and Technology, Irbid 22110, Jordan (zmalshara@just.edu.jo)

The package reproduces every table and figure in the paper, starting from the processed datasets. Experiments that need a GPU or an IBM Quantum account come with Google Colab notebooks.

## Main findings

- **Tuning decides the comparison.** Untuned quantum kernels lag far behind classical learners (MRR 0.50 vs 0.77 on PI-Link). Tuned the same way as the classical models, non-entangling quantum kernels are statistically equivalent to an RBF SVM on both datasets (TOST, 30 seeds). For example, with e5 features PQK-Z scores MRR 0.860 vs 0.859 on PI-Link and 0.916 vs 0.917 on GHPR.
- **The tuned quantum kernel behaves like an RBF kernel.** Kernel alignment with the RBF kernel (CKA) is 0.98–0.99, and the two pick the same top PR for 95–97% of issues. Entangling feature maps never perform significantly better and are often worse.
- **Results under noise and on hardware.** Under shot noise and calibrated device noise, the classically simulable Z-map kernels keep parity with RBF when the bandwidth is re-tuned for the shot budget. On IBM hardware, 12-qubit kernels from ibm_marrakesh match simulation (r = 0.99).
- **Best model and other baselines.** The best model is classical: a pairwise XGBoost ranker with e5 similarity (MRR 0.896 PI-Link, 0.931 GHPR). Kernel models tie BM25 on full candidate pools for PI-Link and fall below it on GHPR. A timing-only rule with no learning comes within 0.013 MRR of the kernel models on PI-Link without e5.

## Contents

| Folder | Contents |
| --- | --- |
| `data/` | Processed link files for both datasets, precomputed e5 similarities, split fingerprints |
| `src/` | All experiment and analysis code (Python) |
| `colab/` | Notebooks for the GPU experiments (CodeBERT, transformer and LLM baselines, e5) and the IBM Quantum hardware runs |
| `results/` | Raw per-seed outputs, summaries and statistical tests for every experiment (see the table below) |
| `figures/paper/` | The paper's vector figures (PDF); `figures/paper_figs.py` redraws them |
| `figures/exploratory/` | Figures from the exploratory stage; `figures/figs*.py` redraw them |
| `logs/` | Console logs of the original runs |
| `run_all.sh` | Every local (CPU) experiment, in paper order |
| `SHA256SUMS.txt` | Checksums of every file |

## Data

The three link files are stored compressed (`.csv.gz`) to stay within GitHub's file-size limits. `run_all.sh` unpacks them; to do it by hand: `gunzip -k data/*_pairs*.csv.gz`.

| File | Source | Rows | Notes |
| --- | --- | --- | --- |
| `data/pilink_pairs.csv.gz`, `data/pilink_pairs_meta.csv.gz` | PI-Link (Alshara et al.), Kaggle `zakareaalshara/android-closed-issues-20110101-20210101-clean` | 34,530 (27,081 usable) | Built by `src/flatten.py` and `src/extract_times.py` from the released JSON |
| `data/ghpr_pairs_meta.csv.gz` | GHPR (Soroush J.), github.com/soroushj/ghpr-dataset and ghpr-dataset-raw, CC BY 4.0 | 13,247 (13,243 usable) | Built by `src/build_ghpr.py` from the raw JSON |
| `data/e5_pairs_*.csv.gz` | Computed with `colab/round3_colab.ipynb` | about 250,000 pairs each | e5-base-v2 cosine similarity for every pair used in the 30-seed runs |
| `data/split_fingerprints.json` | `src/fingerprints.py` | – | MD5 of the test pairs and ranking sets per seed; confirms that the GPU runs used the same splits |

**Leakage control.** All texts have `#N` references, URLs and closing keywords ("fixes #12") removed. PR and issue close times are never used as features, because 64% (PI-Link) and 89% (GHPR) of linked PRs close within one hour of their issue.

To rebuild the data from the original sources:

```bash
python -I src/flatten.py <PI-Link JSON> data/pilink_pairs.csv
python -I src/extract_times.py <PI-Link JSON> data/pilink_pairs.csv data/pilink_pairs_meta.csv
python -I src/build_ghpr.py <ghpr-dataset-raw>/repos data/ghpr_pairs_meta.csv
```

## Environment

```bash
python3 -m pip install -r requirements.txt
```

Python 3.13; numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, scikit-learn 1.9.1, xgboost 3.4.1, lightgbm 4.7.0, qiskit 2.6.1, qiskit-aer 0.18.2, qiskit-ibm-runtime 0.51.0, qiskit-machine-learning 0.9.1, jax 0.12.0.1, optax 0.2.8, matplotlib 3.11.2.

- **CPU experiments.** These ran on 2 CPU cores with 7 GB RAM. Quantum kernels use exact statevector simulation. A custom simulator is checked to machine precision against Qiskit `Statevector` (`src/qml_enhanced.py`, `src/quantum_extra.py`).
- **GPU experiments.** These ran on a Colab T4.
- **Hardware runs.** These used IBM Quantum open-plan devices: ibm_fez (4 qubits) and ibm_marrakesh (12 qubits).

## Running

```bash
bash run_all.sh            # all CPU experiments, about 29 h on 2 cores
```

`run_all.sh` creates a `work/` folder that links code, data and results together, and runs from there. Result folders keep the names the scripts write (for example `results_final30/`), because later analyses read earlier results.

**GPU and hardware parts.** Upload the notebook from `colab/` to Google Colab, along with the `src/` and `data/` files it asks for, then run all cells. The hardware notebooks ask for your IBM Quantum API key through a hidden prompt; never paste a key into a file. Replace `PASTE_YOUR_INSTANCE_CRN_HERE` with your own instance CRN from quantum.cloud.ibm.com.

## Experiments and where their results are

Seeds index repository-level splits (PI-Link) or sampling under a fixed time split (GHPR). Splits are deterministic per seed and identical across all scripts (`src/splits.py`). Runtimes are for 2 CPU cores unless marked GPU or QPU.

| Paper section | Script | Results in `results/` | Runtime |
| --- | --- | --- | --- |
| Exploratory RQ1–RQ2: quantum vs classical, feature maps, VQC (7 seeds) | `qml_pilink.py` | `results/` | about 3 h |
| RQ3: generic noise and device-calibrated noise | `qml_pilink.py`, `rq3_device.py` | `results/`, `rq3_device/` | about 1 h |
| RQ3: real hardware, 4 qubits (ibm_fez) | `colab/pilink_qml_colab.ipynb` → `ibm_hardware.py` | `results_hardware/` | 5 QPU-minutes |
| RQ4: cost and scaling | `qml_pilink.py --n_train {200..3200}`, `scale_study.py` | `scaling/`, `results_scale/` | about 3 h |
| Robustness: CodeBERT features | `colab/pilink_qml_colab.ipynb` → `embed_codebert.py` | `results_codebert/` | about 1 h (GPU) |
| Tuning, bandwidth, PQK, PLS, metadata, BM25 inputs | `qml_enhanced.py` | `results_enh/` | about 2.5 h |
| Stronger classical baselines | `classical_plus.py` | `results_classical_plus/` | about 15 min |
| Confirmatory 30 seeds, PI-Link (Tables 3–4) | `final30.py`, `analyze_final30.py` | `results_final30/` | about 1 h |
| Concentration, more qubits, shots, device noise | `quantum_extra.py {conc,qubits,shots}` | `results_extra/` | about 2.5 h |
| Second dataset, GHPR (30 seeds) | `final30.py --split temporal` | `results_ghpr/` | about 1 h |
| Fair VQC and trainable kernels (QKA) | `vqc_qka.py` | `results_vqc_qka/` | about 1 h |
| Transformer and LLM baselines | `colab/round2_colab.ipynb` → `colab_eval.py` | `results_transformers/` | about 1.5 h (GPU) |
| Real hardware, 12 qubits, final model (ibm_marrakesh) | `colab/round2_colab.ipynb` → `hw_final.py` | `results_hardware_final/` | 162 QPU-seconds |
| e5 similarity as an input to all models | `colab/round3_colab.ipynb` → `e5_pairs.py`; `final30.py --e5` | `results_e5/` | 20 min GPU + 1.5 h |
| Kernel similarity (CKA), SHAP, error analysis | `analysis.py` | `results_analysis/` | about 40 min |
| Realistic retrieval against full candidate pools | `realistic.py` | `results_realistic/` | about 40 min |
| Timing-only baseline; shot noise with bandwidth re-tuned | `review_checks.py {timeonly,shots}` | `results_review/` | about 3 h |
| Seed-dependence re-analysis: per-query ranks rebuilt from saved hyperparameters, corrected resampled t-test, cluster bootstrap (Table 15) | `per_query.py`, then `dependence_stats.py` | `results_review/per_query_*.csv`, `dependence_*.csv` | about 4 h |
| Tuning-budget check: quantum grids cut to 9 configurations | `budget_grid.py`, then `dependence_stats.py` | `results_review/budget_grid_*.csv`, `budget_grid_tests.csv` | about 1.5 h |
| BM25 alone on the sampled queries, both datasets (Table 10) | `bm25_alone.py` | `results_review/bm25_alone.csv` | about 15 min |
| Qwen2.5-1.5B zero-shot, memory-safe re-run on both datasets (Table 10) | `colab/qwen_colab.ipynb` → `qwen_rerun.py` | `results_qwen/` | about 1.5 h (GPU) |
| Hardware kernel-fidelity validation: Z and entangling ZZ1 circuits at 12 qubits, repeat on a second device (**pending**, see below) | `colab/hw_validation_colab.ipynb` → `hw_validation.py` | `hw_validation/` (after the jobs run) | ~4–5 QPU-minutes |

**Statistics.** Comparisons use paired Wilcoxon signed-rank tests over seeds, Cliff's δ and 95% t-intervals of the paired differences, with Holm correction within each family of comparisons. Equivalence uses TOST with margins of ±0.01 AUC and ±0.02 MRR (`analyze_final30.py`). Because seeds re-split the same data, `dependence_stats.py` repeats the main comparisons with the Nadeau–Bengio corrected resampled t-test and a repository-, issue- and project-level cluster bootstrap.

## Pending hardware runs

Three IBM Quantum jobs for the extended hardware validation were submitted on 10 November 2026 and are queued until the
open-plan allowance renews (3 December 2026): `z12` (job `dbpk6r0qtr0c738eip8g`, ibm_kingston), `zz12` and `repeat`
(ibm_marrakesh). `hw_validation.py` saves each job ID when it is submitted and collects the same job when re-run, so no
job is submitted twice. Their results are not part of this version of the paper.

## Reproducibility notes

- Seeds 0–6 of the 30-seed runs reproduce the exploratory 7-seed runs exactly.
- `analysis.py` rebuilds the final models from the saved hyperparameters and checks their MRR against the saved run (`results/results_analysis/rebuild_check.csv`).
- Colab and local runs used identical splits; `data/split_fingerprints.json` verifies this per seed.
- IBM job IDs are recorded in `results/results_hardware*/`. Hardware results depend on the device calibration at run time.
- Fit times were measured while several experiments shared 2 CPU cores, so treat them as indicative.

## Licence and citation

- **Code** (`src/`, `figures/*.py`, `colab/`, `run_all.sh`): MIT licence, see `LICENSE`.
- **Results and figures:** CC BY 4.0.
- **Data:** GHPR-derived files keep GHPR's CC BY 4.0 licence; please cite the GHPR dataset. PI-Link-derived files follow the licence of the PI-Link Kaggle release; please cite the PI-Link paper (IEEE Access, 2023).

If you use this package, please cite the paper (see `CITATION.cff`).

Contact: Zakarea Alshara (z.alshara@ajman.ac.ae; zmalshara@just.edu.jo)
Department of Information Technology, Ajman University, United Arab Emirates, and
Department of Software Engineering, Jordan University of Science and Technology, Jordan
