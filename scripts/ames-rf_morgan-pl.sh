#!/usr/bin/env bash
# Passive learning on ames with RandomForest/Morgan.
#
# Reproduces results-initial/passive_learning/pl-ames-*
# (100 molalkit_run invocations: 5 error rates x 20 seeds).
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-./results-initial/passive_learning}"
mkdir -p "${OUT_ROOT}"

for ERR in 0 10 20 30 40; do
  case "${ERR}" in
    0)  ERATE=0   ;;
    10) ERATE=0.1 ;;
    20) ERATE=0.2 ;;
    30) ERATE=0.3 ;;
    40) ERATE=0.4 ;;
  esac
  for SEED in $(seq 0 19); do
    SAVE_DIR="${OUT_ROOT}/pl-ames-${ERR}-${SEED}"
    rm -rf "${SAVE_DIR}"
    molalkit_run \
      --data_public ames \
      --metrics roc_auc accuracy balanced_accuracy precision recall f1_score mcc \
      --model_configs RandomForest_Morgan_Config \
      --split_type scaffold_random --split_sizes 0.5 0.5 \
      --evaluate_stride 1 \
      --n_jobs 2 \
      --seed "${SEED}" \
      --error_rate "${ERATE}" \
      --select_method random \
      --save_dir "${SAVE_DIR}"
  done
done
