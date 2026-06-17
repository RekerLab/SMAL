#!/usr/bin/env bash
# Active learning on pgp_broccatelli with DMPNN.
#
# Reproduces results-dmpnn/active_learning/al-pgp_broccatelli-0-*
# (20 molalkit_run invocations: clean labels x 20 seeds).
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-./results-dmpnn/active_learning}"
mkdir -p "${OUT_ROOT}"

DATASET="pgp_broccatelli"
ERROR="0"
MODEL_CONFIG="DMPNN_BinaryClassification_Config"
MAX_ITER="${MAX_ITER:-606}"

for SEED in $(seq 0 19); do
  SAVE_DIR="${OUT_ROOT}/al-${DATASET}-${ERROR}-${SEED}"
  rm -rf "${SAVE_DIR}"
  molalkit_run \
    --data_public "${DATASET}" \
    --metrics roc_auc accuracy balanced_accuracy precision recall f1_score mcc \
    --model_configs "${MODEL_CONFIG}" \
    --split_type scaffold_random \
    --split_sizes 0.5 0.5 \
    --evaluate_stride 1 \
    --seed "${SEED}" \
    --n_jobs 8 \
    --error_rate "${ERROR}" \
    --select_method explorative \
    --s_batch_size 1 \
    --max_iter "${MAX_ITER}" \
    --save_cpt_stride 20 \
    --load_checkpoint \
    --save_dir "${SAVE_DIR}"
done
