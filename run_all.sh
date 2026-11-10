#!/bin/bash
# Re-runs every local (CPU) experiment in paper order, writing into results/.
# GPU and IBM hardware parts: see colab/. Total about 29 h on 2 CPU cores.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
# unpack the compressed data files (kept compressed in git)
for f in "$ROOT"/data/*_pairs*.csv.gz; do [ -f "${f%.gz}" ] || gunzip -k "$f"; done
mkdir -p "$ROOT/work"; cd "$ROOT/work"
ln -sf "$ROOT"/src/*.py . ; ln -sf "$ROOT"/data/* . ; for d in "$ROOT"/results/*/; do ln -sfn "$d" "$(basename "$d")"; done
export OMP_NUM_THREADS=1
python3 qml_pilink.py --data pilink_pairs.csv --out results --seeds 10 --dims 4 8 12 --n_train 800 --n_test 400 --n_rank 200 --vqc --vqc_n 200 --noise_n 100 --noise_d 4
for n in 200 400 800 1600 3200; do python3 qml_pilink.py --data pilink_pairs.csv --out scaling/n$n --seeds 3 --dims 8 --fmaps Z ZZ1 --classical SVC-RBF XGB --n_train $n; done
python3 rq3_device.py --data pilink_pairs.csv --seeds 5 --n 100 --out rq3_device
python3 qml_enhanced.py --reducer pca --out results_enh/pca
python3 qml_enhanced.py --reducer pls --out results_enh/pls
python3 qml_enhanced.py --reducer pls --meta --out results_enh/pls_meta
python3 qml_enhanced.py --reducer pls --meta --bm25 --out results_enh/pls_meta_bm25
python3 classical_plus.py --out results_classical_plus
python3 final30.py --seeds 15 --out results_final30/a && python3 final30.py --seed_from 15 --seeds 30 --out results_final30/b
OUT=results_final30 python3 analyze_final30.py results_final30/a/raw_results.csv results_final30/b/raw_results.csv
python3 quantum_extra.py conc --seeds 3 --out results_extra
python3 quantum_extra.py qubits --seeds 10 --out results_extra
python3 quantum_extra.py shots --seeds 10 --out results_extra
python3 final30.py --data ghpr_pairs_meta.csv --split temporal --seeds 30 --out results_ghpr && OUT=results_ghpr python3 analyze_final30.py results_ghpr/raw_results.csv
python3 scale_study.py --max_svc 3200 --out results_scale
python3 vqc_qka.py --out results_vqc_qka
python3 final30.py --e5 e5_pairs_pilink.csv.gz --seeds 30 --out results_e5/pilink && OUT=results_e5/pilink python3 analyze_final30.py results_e5/pilink/raw_results.csv
python3 final30.py --data ghpr_pairs_meta.csv --split temporal --e5 e5_pairs_ghpr.csv.gz --seeds 30 --out results_e5/ghpr && OUT=results_e5/ghpr python3 analyze_final30.py results_e5/ghpr/raw_results.csv
python3 analysis.py 10 results_analysis
python3 realistic.py 10 results_realistic
python3 review_checks.py timeonly   # results_review/timeonly.csv
python3 review_checks.py shots      # results_review/shots_retuned.csv
python3 per_query.py pilink    results_final30/a/raw_results.csv,results_final30/b/raw_results.csv
python3 per_query.py pilink-e5 results_e5/pilink/raw_results.csv
python3 per_query.py ghpr      results_ghpr/raw_results.csv
python3 per_query.py ghpr-e5   results_e5/ghpr/raw_results.csv
python3 budget_grid.py pilink results_final30/a/raw_results.csv,results_final30/b/raw_results.csv
python3 budget_grid.py ghpr   results_ghpr/raw_results.csv
python3 bm25_alone.py              # results_review/bm25_alone.csv
python3 dependence_stats.py        # results_review/dependence_*.csv, budget_grid_tests.csv
# Figures: exploratory (work/figs/) and paper (work/paper_figures/)
mkdir -p figs; for f in "$ROOT"/figures/figs*.py; do python3 "$f"; done
python3 "$ROOT"/figures/paper_figs.py
