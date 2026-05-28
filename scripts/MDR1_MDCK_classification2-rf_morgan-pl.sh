#!/usr/bin/env bash
# Passive learning on MDR1_MDCK_classification2 with RandomForest/Morgan.
#
# Reproduces results-initial/passive_learning/pl-MDR1_MDCK_classification2-*
# (100 molalkit_run invocations: 5 error rates x 20 seeds).
set -euo pipefail

PKG_DATA="$(python -c "import molalkit, os; print(os.path.join(os.path.dirname(molalkit.__file__), 'data', 'datasets'))")"

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
    SAVE_DIR="${OUT_ROOT}/pl-MDR1_MDCK_classification2-${ERR}-${SEED}"
    rm -rf "${SAVE_DIR}"
    molalkit_run \
      --data_path "${PKG_DATA}/MDR1_MDCK_classification2.csv" --smiles_columns SMILES --targets_columns Y --task_type binary \
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
