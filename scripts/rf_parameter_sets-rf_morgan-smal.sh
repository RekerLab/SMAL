#!/usr/bin/env bash
# SMAL RF parameter sweep on pgp_broccatelli, PAMPA_NCATS, and MDR1_MDCK_classification2.
#
# Reproduces results-rf-parameter-sets/smal/smal-*
# (2880 molalkit_run invocations: 3 datasets x 8 RF parameter sets x 6 forget methods x 20 seeds).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DATA="$(python -c "import molalkit, os; print(os.path.join(os.path.dirname(molalkit.__file__), 'data', 'datasets'))")"

OUT_ROOT="${OUT_ROOT:-./results-rf-parameter-sets}"
CONFIG_ROOT="${CONFIG_ROOT:-${SCRIPT_DIR}/rf_parameter_configs}"
FMIN_TABLE="${FMIN_TABLE:-${SCRIPT_DIR}/rf_parameter_sets-smal-params.csv}"

if [ ! -f "${FMIN_TABLE}" ]; then
  echo "SMAL parameter table not found: ${FMIN_TABLE}" >&2
  exit 1
fi
mkdir -p "${OUT_ROOT}/smal"
OUT_ROOT="$(cd "${OUT_ROOT}" && pwd)"
CONFIG_ROOT="$(cd "${CONFIG_ROOT}" && pwd)"

MANIFEST="${OUT_ROOT}/smal_manifest.csv"
printf 'task_id,job_name,dataset,rf_set,n_estimators,max_depth,config,error_rate,seed,forget_method,f_min_train_size,max_iter,save_dir\n' > "${MANIFEST}"

FORGET_METHODS=(random first max_oob_error min_oob_error max_oob_uncertainty min_oob_uncertainty)

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

config_for_rf_set() {
  case "$1" in
    rf-n50-d10) printf '%s/RandomForest_Morgan_rf_n50_d10_Config' "${CONFIG_ROOT}" ;;
    rf-n50-d20) printf '%s/RandomForest_Morgan_rf_n50_d20_Config' "${CONFIG_ROOT}" ;;
    rf-n50-none) printf '%s/RandomForest_Morgan_rf_n50_none_Config' "${CONFIG_ROOT}" ;;
    rf-n100-d10) printf '%s/RandomForest_Morgan_rf_n100_d10_Config' "${CONFIG_ROOT}" ;;
    rf-n100-d20) printf '%s/RandomForest_Morgan_rf_n100_d20_Config' "${CONFIG_ROOT}" ;;
    rf-n500-d10) printf '%s/RandomForest_Morgan_rf_n500_d10_Config' "${CONFIG_ROOT}" ;;
    rf-n500-d20) printf '%s/RandomForest_Morgan_rf_n500_d20_Config' "${CONFIG_ROOT}" ;;
    rf-n500-none) printf '%s/RandomForest_Morgan_rf_n500_none_Config' "${CONFIG_ROOT}" ;;
    *)
      echo "Unsupported RF set: $1" >&2
      exit 1
      ;;
  esac
}

TASK_ID=1
mapfile -t PARAM_ROWS < <(tail -n +2 "${FMIN_TABLE}")
for PARAM_ROW in "${PARAM_ROWS[@]}"; do
  IFS=, read -r DATASET RF_SET N_ESTIMATORS MAX_DEPTH ERROR_RATE FMIN MAX_ITER _REST <<< "${PARAM_ROW}"
  [ -z "${DATASET}" ] && continue
  dataset_flags "${DATASET}"
  CONFIG="$(config_for_rf_set "${RF_SET}")"
  if [ ! -f "${CONFIG}" ]; then
    echo "RF config not found: ${CONFIG}" >&2
    exit 1
  fi
  if [ -z "${MAX_DEPTH}" ]; then
    MAX_DEPTH_LABEL=None
  else
    MAX_DEPTH_LABEL="${MAX_DEPTH%.0}"
  fi

  for METHOD in "${FORGET_METHODS[@]}"; do
    for SEED in $(seq 0 19); do
      JOB_NAME="smal-${DATASET}-${RF_SET}-${ERROR_RATE}-${METHOD}-${SEED}"
      SAVE_DIR="${OUT_ROOT}/smal/${JOB_NAME}"
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${TASK_ID}" "${JOB_NAME}" "${DATASET}" "${RF_SET}" "${N_ESTIMATORS}" "${MAX_DEPTH_LABEL}" \
        "${CONFIG}" "${ERROR_RATE}" "${SEED}" "${METHOD}" "${FMIN}" "${MAX_ITER}" "${SAVE_DIR}" >> "${MANIFEST}"

      rm -rf "${SAVE_DIR}"
      molalkit_run \
        "${DATA_FLAGS[@]}" \
        --metrics roc_auc accuracy balanced_accuracy precision recall f1_score mcc \
        --model_configs "${CONFIG}" \
        --split_type scaffold_random --split_sizes 0.5 0.5 \
        --evaluate_stride 1 \
        --n_jobs 2 \
        --seed "${SEED}" \
        --error_rate "${ERROR_RATE}" \
        --select_method explorative \
        --forget_method "${METHOD}" \
        --n_forget 1 \
        --f_min_train_size "${FMIN}" \
        --max_iter "${MAX_ITER}" \
        --save_dir "${SAVE_DIR}"

      TASK_ID=$((TASK_ID + 1))
    done
  done
done
