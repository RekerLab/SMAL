#!/usr/bin/env bash
# Active learning on all SIMPD ChEMBL datasets with RandomForest/Morgan.
#
# Reproduces results-simpd/active_learning/al-CHEMBL*-*
# (9900 molalkit_run invocations: 99 datasets x 5 error rates x 20 seeds).
set -euo pipefail

OUT_ROOT="${OUT_ROOT:-./results-simpd/active_learning}"
DATA_ROOT="${DATA_ROOT:-./datasets}"
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
    for SEED in $(seq 0 19); do
      SAVE_DIR="${OUT_ROOT}/al-${DATASET}-${ERR}-${SEED}"
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
        --save_dir "${SAVE_DIR}"
    done
  done
done
