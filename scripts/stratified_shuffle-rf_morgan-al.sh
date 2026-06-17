#!/usr/bin/env bash
# Active learning with stratified-shuffle label errors on the six manuscript datasets.
#
# Reproduces results-stratified_shuffle/active_learning/al-*
# (480 molalkit_run invocations: 6 datasets x 4 error rates x 20 seeds).
set -euo pipefail

PKG_DATA="$(python -c "import molalkit, os; print(os.path.join(os.path.dirname(molalkit.__file__), 'data', 'datasets'))")"

OUT_ROOT="${OUT_ROOT:-./results-stratified_shuffle/active_learning}"
mkdir -p "${OUT_ROOT}"

DATASETS=(
  pgp_broccatelli
  PAMPA_NCATS
  MDR1_MDCK_classification2
  ames
  CYP3A4_Veith
  CYP2D6_Veith
)

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
    for SEED in $(seq 0 19); do
      SAVE_DIR="${OUT_ROOT}/al-${DATASET}-${ERR}-${SEED}"
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
        --save_dir "${SAVE_DIR}"
    done
  done
done
