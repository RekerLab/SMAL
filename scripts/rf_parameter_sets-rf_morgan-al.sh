#!/usr/bin/env bash
# Active-learning RF parameter sweep on pgp_broccatelli, PAMPA_NCATS, and MDR1_MDCK_classification2.
#
# Reproduces results-rf-parameter-sets/active_learning/al-*
# (480 molalkit_run invocations: 3 datasets x 8 RF parameter sets x 20 seeds).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DATA="$(python -c "import molalkit, os; print(os.path.join(os.path.dirname(molalkit.__file__), 'data', 'datasets'))")"

OUT_ROOT="${OUT_ROOT:-./results-rf-parameter-sets}"
CONFIG_ROOT="${CONFIG_ROOT:-${SCRIPT_DIR}/rf_parameter_configs}"
mkdir -p "${OUT_ROOT}/active_learning"
OUT_ROOT="$(cd "${OUT_ROOT}" && pwd)"
CONFIG_ROOT="$(cd "${CONFIG_ROOT}" && pwd)"

MANIFEST="${OUT_ROOT}/manifest.csv"
printf 'task_id,job_name,dataset,rf_set,n_estimators,max_depth,config,error_rate,seed,save_dir\n' > "${MANIFEST}"

DATASETS=(pgp_broccatelli PAMPA_NCATS MDR1_MDCK_classification2)
RF_PARAMETER_SETS=(
  'rf-n50-d10|50|10|RandomForest_Morgan_rf_n50_d10_Config'
  'rf-n50-d20|50|20|RandomForest_Morgan_rf_n50_d20_Config'
  'rf-n50-none|50|None|RandomForest_Morgan_rf_n50_none_Config'
  'rf-n100-d10|100|10|RandomForest_Morgan_rf_n100_d10_Config'
  'rf-n100-d20|100|20|RandomForest_Morgan_rf_n100_d20_Config'
  'rf-n500-d10|500|10|RandomForest_Morgan_rf_n500_d10_Config'
  'rf-n500-d20|500|20|RandomForest_Morgan_rf_n500_d20_Config'
  'rf-n500-none|500|None|RandomForest_Morgan_rf_n500_none_Config'
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

TASK_ID=1
for DATASET in "${DATASETS[@]}"; do
  dataset_flags "${DATASET}"
  for RF_ROW in "${RF_PARAMETER_SETS[@]}"; do
    IFS='|' read -r RF_SET N_ESTIMATORS MAX_DEPTH CONFIG_NAME <<< "${RF_ROW}"
    CONFIG="${CONFIG_ROOT}/${CONFIG_NAME}"
    if [ ! -f "${CONFIG}" ]; then
      echo "RF config not found: ${CONFIG}" >&2
      exit 1
    fi

    for SEED in $(seq 0 19); do
      JOB_NAME="al-${DATASET}-${RF_SET}-0-${SEED}"
      SAVE_DIR="${OUT_ROOT}/active_learning/${JOB_NAME}"
      printf '%s,%s,%s,%s,%s,%s,%s,0,%s,%s\n' \
        "${TASK_ID}" "${JOB_NAME}" "${DATASET}" "${RF_SET}" "${N_ESTIMATORS}" "${MAX_DEPTH}" \
        "${CONFIG}" "${SEED}" "${SAVE_DIR}" >> "${MANIFEST}"

      rm -rf "${SAVE_DIR}"
      molalkit_run \
        "${DATA_FLAGS[@]}" \
        --metrics roc_auc accuracy balanced_accuracy precision recall f1_score mcc \
        --model_configs "${CONFIG}" \
        --split_type scaffold_random --split_sizes 0.5 0.5 \
        --evaluate_stride 1 \
        --n_jobs 2 \
        --seed "${SEED}" \
        --error_rate 0 \
        --select_method explorative \
        --save_dir "${SAVE_DIR}"

      TASK_ID=$((TASK_ID + 1))
    done
  done
done
