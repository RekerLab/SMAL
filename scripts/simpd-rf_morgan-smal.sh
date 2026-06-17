#!/usr/bin/env bash
# Short-term Memory Active Learning (SMAL) on all SIMPD ChEMBL datasets with RandomForest/Morgan.
#
# Reproduces results-simpd/smal/smal-CHEMBL*-*
# (59400 molalkit_run invocations: 99 datasets x 5 error rates x 20 seeds x 6 forget methods).
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-./results-simpd/smal}"
DATA_ROOT="${DATA_ROOT:-./datasets}"
FMIN_TABLE="${FMIN_TABLE:-./scripts/simpd-smal-params.csv}"
if [ -z "${TMP_DATA_ROOT:-}" ]; then
  TMP_DATA_ROOT="$(mktemp -d)"
  CLEAN_TMP_DATA_ROOT=1
else
  CLEAN_TMP_DATA_ROOT=0
fi
mkdir -p "${OUT_ROOT}" "${TMP_DATA_ROOT}"
trap 'if [ "${CLEAN_TMP_DATA_ROOT}" -eq 1 ]; then rm -rf "${TMP_DATA_ROOT}"; fi' EXIT

prepare_simpd_dataset() {
  local src="$1"
  local dataset="$2"
  python - "$src" "${TMP_DATA_ROOT}/${dataset}" <<'PY'
import csv
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
out_prefix = pathlib.Path(sys.argv[2])
paths = {
    "train": out_prefix.with_name(out_prefix.name + "-train.csv"),
    "test": out_prefix.with_name(out_prefix.name + "-test.csv"),
}
writers = {}
files = {}
try:
    for split in ("train", "test"):
        files[split] = paths[split].open("w", newline="")
        writers[split] = csv.DictWriter(files[split], fieldnames=["SMILES", "Y"])
        writers[split].writeheader()

    counts = {"train": 0, "test": 0}
    with src.open(newline="") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            if split not in counts:
                raise SystemExit(f"Unsupported split value {split!r} in {src}")
            out_row = {"SMILES": row["canonical_smiles"], "Y": row["active"]}
            writers[split].writerow(out_row)
            counts[split] += 1
    if not counts["train"] or not counts["test"]:
        raise SystemExit(f"{src} must contain both train and test rows")
finally:
    for file in files.values():
        file.close()

print(paths["train"])
print(paths["test"])
PY
}

lookup_smal_params() {
  local dataset="$1"
  local err="$2"
  python - "$FMIN_TABLE" "$dataset" "$err" <<'PY'
import csv
import sys

table, dataset, err = sys.argv[1:]
with open(table, newline="") as handle:
    for row in csv.DictReader(handle):
        if row["dataset"] == dataset and row["error"] == err:
            print(row["f_min_train_size"])
            print(row["end_iter"])
            raise SystemExit
raise SystemExit(f"No f_min_train_size/end_iter row for {dataset} error {err} in {table}")
PY
}

if [ ! -f "${FMIN_TABLE}" ]; then
  echo "SMAL parameter table not found: ${FMIN_TABLE}" >&2
  echo "Set FMIN_TABLE=/path/to/f_min_train_size.csv to override." >&2
  exit 1
fi

FORGET_METHODS=(random first max_oob_error min_oob_error max_oob_uncertainty min_oob_uncertainty)
shopt -s nullglob
DATASETS=("${DATA_ROOT}"/CHEMBL*.csv)
if [ "${#DATASETS[@]}" -eq 0 ]; then
  echo "No SIMPD datasets found at ${DATA_ROOT}/CHEMBL*.csv" >&2
  exit 1
fi

for DATA_PATH in "${DATASETS[@]}"; do
  DATASET="$(basename "${DATA_PATH}" .csv)"
  mapfile -t SPLIT_PATHS < <(prepare_simpd_dataset "${DATA_PATH}" "${DATASET}")
  DATA_FLAGS=(
    --data_path_training "${SPLIT_PATHS[0]}"
    --data_path_val "${SPLIT_PATHS[1]}"
    --smiles_columns SMILES
    --targets_columns Y
    --task_type binary
  )

  for ERR in 0 10 20 30 40; do
    case "${ERR}" in
      0)  ERATE=0   ;;
      10) ERATE=0.1 ;;
      20) ERATE=0.2 ;;
      30) ERATE=0.3 ;;
      40) ERATE=0.4 ;;
    esac
    mapfile -t SMAL_PARAMS < <(lookup_smal_params "${DATASET}" "${ERR}")
    FMIN="${SMAL_PARAMS[0]}"
    MAX_ITER="${SMAL_PARAMS[1]}"
    for METHOD in "${FORGET_METHODS[@]}"; do
      for SEED in $(seq 0 19); do
        SAVE_DIR="${OUT_ROOT}/smal-${DATASET}-${ERR}-${METHOD}-${SEED}"
        rm -rf "${SAVE_DIR}"
        molalkit_run \
          "${DATA_FLAGS[@]}" \
          --metrics roc_auc accuracy balanced_accuracy precision recall f1_score mcc \
          --model_configs RandomForest_Morgan_Config \
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
done
