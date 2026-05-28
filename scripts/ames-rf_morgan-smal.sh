#!/usr/bin/env bash
# Short-term Memory Active Learning (SMAL) on ames with RandomForest/Morgan.
#
# Reproduces results-initial/smal/smal-ames-*
# (600 molalkit_run invocations: 5 error rates x 20 seeds x 6 forget methods).
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-./results-initial/smal}"
mkdir -p "${OUT_ROOT}"

ERRS=(0 10 20 30 40)
ERATES=(0 0.1 0.2 0.3 0.4)
F_MINS=(2938 2971 2753 2391 2881)  # f_min_train_size per error rate
MAX_ITER=3636

FORGET_METHODS=(random first max_oob_error min_oob_error max_oob_uncertainty min_oob_uncertainty)

for i in "${!ERRS[@]}"; do
  ERR="${ERRS[$i]}"
  ERATE="${ERATES[$i]}"
  FMIN="${F_MINS[$i]}"
  for METHOD in "${FORGET_METHODS[@]}"; do
    for SEED in $(seq 0 19); do
      SAVE_DIR="${OUT_ROOT}/smal-ames-${ERR}-${METHOD}-${SEED}"
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
        --select_method explorative \
        --forget_method "${METHOD}" \
        --n_forget 1 \
        --f_min_train_size "${FMIN}" \
        --max_iter "${MAX_ITER}" \
        --save_dir "${SAVE_DIR}"
    done
  done
done
