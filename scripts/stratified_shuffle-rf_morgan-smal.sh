#!/usr/bin/env bash
# SMAL with stratified-shuffle label errors on the six manuscript datasets.
#
# Reproduces results-stratified_shuffle/smal/smal-*
# (2880 molalkit_run invocations: 6 datasets x 4 error rates x 6 forget methods x 20 seeds).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DATA="$(python -c "import molalkit, os; print(os.path.join(os.path.dirname(molalkit.__file__), 'data', 'datasets'))")"

OUT_ROOT="${OUT_ROOT:-./results-stratified_shuffle/smal}"
FMIN_TABLE="${FMIN_TABLE:-${SCRIPT_DIR}/stratified_shuffle-smal-params.csv}"
FMIN_FLOOR=51

if [ ! -f "${FMIN_TABLE}" ]; then
  echo "SMAL parameter table not found: ${FMIN_TABLE}" >&2
  exit 1
fi
mkdir -p "${OUT_ROOT}"

DATASETS=(
  pgp_broccatelli
  PAMPA_NCATS
  MDR1_MDCK_classification2
  ames
  CYP3A4_Veith
  CYP2D6_Veith
)
FORGET_METHODS=(random first max_oob_error min_oob_error max_oob_uncertainty min_oob_uncertainty)

declare -A FMIN_MAP
declare -A MAX_ITER_MAP

while IFS=, read -r DATASET ERR FMIN MAX_ITER; do
  [ "${DATASET}" = "dataset" ] && continue
  if [ "${FMIN}" -lt "${FMIN_FLOOR}" ]; then
    FMIN="${FMIN_FLOOR}"
  fi
  FMIN_MAP["${DATASET},${ERR}"]="${FMIN}"
  MAX_ITER_MAP["${DATASET},${ERR}"]="${MAX_ITER}"
done < "${FMIN_TABLE}"

dataset_flags() {
  local dataset="$1"
  DATA_FLAGS=()
  case "${dataset}" in
    PAMPA_NCATS|MDR1_MDCK_classification2)
      DATA_FLAGS=(--data_path "${PKG_DATA}/${dataset}.csv" --smiles_columns SMILES --targets_columns Y --task_type binary)
      ;;
    *)
      DATA_FLAGS=(--data_public "${dataset}")
      ;;
  esac
}

error_rate() {
  case "$1" in
    10) printf '0.1' ;;
    20) printf '0.2' ;;
    30) printf '0.3' ;;
    40) printf '0.4' ;;
    *)
      echo "Unsupported error code: $1" >&2
      exit 1
      ;;
  esac
}

for DATASET in "${DATASETS[@]}"; do
  dataset_flags "${DATASET}"
  for ERR in 10 20 30 40; do
    ERATE="$(error_rate "${ERR}")"
    FMIN="${FMIN_MAP["${DATASET},${ERR}"]:-}"
    MAX_ITER="${MAX_ITER_MAP["${DATASET},${ERR}"]:-}"
    if [ -z "${FMIN}" ] || [ -z "${MAX_ITER}" ]; then
      echo "Missing SMAL parameters for ${DATASET}, error ${ERR}" >&2
      exit 1
    fi

    for METHOD in "${FORGET_METHODS[@]}"; do
      for SEED in $(seq 0 19); do
        SAVE_DIR="${OUT_ROOT}/smal-${DATASET}-${ERR}-${METHOD}-${SEED}"
        rm -rf "${SAVE_DIR}"
        molalkit_run \
          "${DATA_FLAGS[@]}" \
          --metrics roc_auc accuracy balanced_accuracy precision recall f1_score mcc \
          --model_configs RandomForest_Morgan_Config \
          --split_type scaffold_random --split_sizes 0.5 0.5 \
          --evaluate_stride 1 \
          --n_jobs 2 \
          --seed "${SEED}" \
          --error_rate "${ERATE}" \
          --error_algorithm stratified_shuffle \
          --select_method explorative \
          --forget_method "${METHOD}" \
          --n_forget 1 \
          --f_min_train_size "${FMIN}" \
          --max_iter "${MAX_ITER}" \
          --save_dir "${SAVE_DIR}"
      done
    done
  done
done
