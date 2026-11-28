#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# TAMNet audit — full reproduction pipeline (POSIX order of operations).
#
# Every stage is *skip-if-done*: training keys on the presence of
# results/phase3/runs/<run>/metrics.json, evaluation on eval_*/metrics.json,
# and the figure/table/paper stages simply overwrite their outputs. Re-running
# this script therefore never re-trains a finished run and never touches the
# TEST set more than once per configuration (the audit protocol).
#
#   ./run_all.sh              # stages from SPLIT onward
#   ./run_all.sh split train  # only the named stages
#   ./run_all.sh check        # QA only (numbers, figures, tests)
#
# Windows users: run_all.ps1 is the same pipeline for PowerShell.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python}"
A=("$PY" -X utf8)

stages_default=(split train evaluate aggregate bootstrap external uncertainty
                robustness seg brats3d attention efficiency qual figures
                tables paper check)
stages=("$@")
[ ${#stages[@]} -eq 0 ] && stages=("${stages_default[@]}")

run() { echo; echo "=== $* ==="; "$@"; }

for s in "${stages[@]}"; do
  case "$s" in
    split)      run "${A[@]}" scripts/phase3_split.py ;;
    train)      run "${A[@]}" scripts/phase3_train.py ;;
    evaluate)    run "${A[@]}" scripts/phase3_evaluate.py ;;
    aggregate)   run "${A[@]}" scripts/phase3_aggregate.py ;;
    bootstrap)   run "${A[@]}" scripts/phase3_bootstrap.py ;;
    external)    run "${A[@]}" scripts/phase3_external_eval.py ;;
    uncertainty) run "${A[@]}" scripts/phase3_uncertainty.py ;;
    robustness)  run "${A[@]}" scripts/phase3_robustness.py ;;
    seg)         run "${A[@]}" scripts/phase3_seg_eval.py ;;
    # standalone 3-D U-Net on BraTS — source of the WT/TC/ET numbers (D3)
    brats3d)     run "${A[@]}" -m scripts.evaluate_segmentation --split test
                 run "${A[@]}" scripts/phase5_ingest_unet3d.py ;;
    attention)   run "${A[@]}" scripts/phase3_attention_alignment.py ;;
    efficiency)  run "${A[@]}" scripts/phase3_efficiency.py ;;
    # qualitative fig7 + per-sample predictions + supplementary S1-S10
    qual)        run "${A[@]}" scripts/phase4_seg_qualitative.py
                 run "${A[@]}" scripts/phase4_save_predictions.py
                 run "${A[@]}" scripts/make_supplementary.py ;;
    figures)     run "${A[@]}" scripts/make_figures.py
                 run "${A[@]}" scripts/make_figures_static.py ;;
    tables)      run "${A[@]}" scripts/make_tables.py ;;
    paper)       run "${A[@]}" paper/rewrite_paper.py ;;
    check)       run "${A[@]}" scripts/check_paper_numbers.py
                 run "${A[@]}" scripts/check_figures.py
                 run "${A[@]}" -m pytest tests -q ;;
    test)        run "${A[@]}" -m pytest tests -q ;;
    install)     run "$PY" -m pip install -r requirements.txt ;;
    *) echo "unknown stage: $s" >&2; exit 2 ;;
  esac
done

echo
echo "run_all.sh: done (${stages[*]})"
