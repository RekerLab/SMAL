#!/usr/bin/env bash
# Active learning on pgp_broccatelli with MolFormer.
#
# Reproduces results-molformer/active_learning/al-pgp_broccatelli-0-*
# (20 molalkit_run invocations: clean labels x 20 seeds).
#
# MolFormer requires the IBM pretrained checkpoint file. Download it from:
#   https://github.com/IBM/molformer
# Then set MOLFORMER_CKPT to the downloaded N-Step-Checkpoint_3_30000.ckpt path.
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-./results-molformer/active_learning}"
MOLFORMER_CKPT="${MOLFORMER_CKPT:-N-Step-Checkpoint_3_30000.ckpt}"
MAX_ITER="${MAX_ITER:-606}"

if [[ ! -f "${MOLFORMER_CKPT}" ]]; then
  cat >&2 <<EOF
ERROR: MolFormer checkpoint not found: ${MOLFORMER_CKPT}

Download the pretrained checkpoint from https://github.com/IBM/molformer
and rerun with:
  MOLFORMER_CKPT=/path/to/N-Step-Checkpoint_3_30000.ckpt bash $0
EOF
  exit 2
fi

mkdir -p "${OUT_ROOT}"

DATASET="pgp_broccatelli"
ERROR="0"
TMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/MolFormer_Config.XXXXXX")"
trap 'rm -f "${TMP_CONFIG}"' EXIT

cat > "${TMP_CONFIG}" <<EOF
{
    "data_format": "mgktools",
    "model": "MolFormer",
    "pretrained_path": "${MOLFORMER_CKPT}",
    "n_head": 12,
    "n_layer": 12,
    "n_embd": 768,
    "d_dropout": 0.1,
    "dropout": 0.1,
    "learning_rate": 3e-5,
    "num_feats": 32,
    "batch_size": 32,
    "epochs": 50,
    "ensemble_size": 1
}
EOF

for SEED in $(seq 0 19); do
  SAVE_DIR="${OUT_ROOT}/al-${DATASET}-${ERROR}-${SEED}"
  rm -rf "${SAVE_DIR}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  molalkit_run \
    --data_public "${DATASET}" \
    --metrics roc_auc accuracy balanced_accuracy precision recall f1_score mcc \
    --model_configs "${TMP_CONFIG}" \
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
