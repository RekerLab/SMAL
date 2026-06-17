# Reproduction Scripts

One bash script per **(dataset, model, learning_type)** combination, named:

```
<dataset>-<model>-<learning_type>.sh
```

Where `learning_type` is one of:

| code | meaning |
|------|---------|
| `al`   | Active learning (informative selection: `--select_method explorative`) |
| `pl`   | Passive learning (random selection: `--select_method random`) |
| `smal` | Short-term Memory Active Learning (AL with `--forget_method` and `--n_forget`) |

Each run script loops over the seeds, error rates, and (for SMAL) forget methods used in the manuscript, calling `molalkit_run` once per combination. The grouped SIMPD scripts use one script per learning type and loop over all committed `datasets/CHEMBL*.csv` files.

## Prerequisites

Install MolALKit v1.2.0 following the instructions in the [main README](../README.md). The `molalkit_run` CLI must be on your `PATH`, and `python -c "import molalkit"` must succeed (used by some scripts to locate packaged dataset CSVs).

DMPNN scripts additionally require a MolALKit environment with Chemprop support and a CUDA-capable GPU for the manuscript-scale runs.

MolFormer scripts additionally require the IBM MolFormer pretrained checkpoint file. Download the checkpoint from <https://github.com/IBM/molformer> and pass its path with `MOLFORMER_CKPT`:

```bash
MOLFORMER_CKPT=/path/to/N-Step-Checkpoint_3_30000.ckpt bash scripts/pgp_broccatelli-molformer-al.sh
```

## Running

From the repository root:

```bash
bash scripts/ames-rf_morgan-al.sh
```

Outputs are written under `./results-initial/<active_learning|passive_learning|smal>/...`. Override the destination with the `OUT_ROOT` environment variable:

```bash
OUT_ROOT=/path/to/output bash scripts/ames-rf_morgan-smal.sh
```

DMPNN and MolFormer pgp active-learning scripts default to model-specific output roots, `./results-dmpnn/active_learning` and `./results-molformer/active_learning`, respectively. They also accept `OUT_ROOT`.

SIMPD scripts default to `./results-simpd/<active_learning|passive_learning|smal>`. Each SIMPD dataset CSV contains a `split` column; the scripts preserve this split by creating temporary MolALKit input files with `SMILES` and `Y` columns, using the `train` rows for active learning and the `test` rows for validation. Override the dataset location with `DATA_ROOT` if needed:

```bash
DATA_ROOT=/path/to/simpd_csvs OUT_ROOT=/path/to/output bash scripts/simpd-rf_morgan-al.sh
```

The SIMPD SMAL script reads per-dataset `f_min_train_size` and `max_iter` values from `scripts/simpd-smal-params.csv`. Override with `FMIN_TABLE` only if you have regenerated those manuscript parameters.

Stratified-shuffle label-error scripts default to `./results-stratified_shuffle/<active_learning|smal>`:

```bash
bash scripts/stratified_shuffle-rf_morgan-al.sh
bash scripts/stratified_shuffle-rf_morgan-smal.sh
```

The stratified-shuffle SMAL script reads `f_min_train_size` and `max_iter` from `scripts/stratified_shuffle-smal-params.csv`.

RF parameter-set scripts default to `./results-rf-parameter-sets`. They use the custom RandomForest/Morgan configs in `scripts/rf_parameter_configs/`; the active-learning script writes `manifest.csv`, and the SMAL script writes `smal_manifest.csv` for `scripts/build_processed_results.py`.

```bash
bash scripts/rf_parameter_sets-rf_morgan-al.sh
bash scripts/rf_parameter_sets-rf_morgan-smal.sh
```

The RF parameter-set SMAL script reads `f_min_train_size` and `max_iter` from `scripts/rf_parameter_sets-smal-params.csv`.

The scripts are intentionally **simple sequential loops** — they do not submit SLURM jobs, parallelize, or skip already-completed runs. For HPC use, wrap each `molalkit_run` invocation in your own job scheduler.

## Currently included experiments

| Model | Datasets | Learning types |
|-------|----------|----------------|
| RandomForest/Morgan | ames, CYP2D6_Veith, CYP3A4_Veith, MDR1_MDCK_classification2, PAMPA_NCATS, pgp_broccatelli | al, pl, smal |
| RandomForest/Morgan | SIMPD ChEMBL datasets (`CHEMBL*.csv`) | al, pl, smal |
| RandomForest/Morgan | ames, CYP2D6_Veith, CYP3A4_Veith, MDR1_MDCK_classification2, PAMPA_NCATS, pgp_broccatelli with stratified-shuffle label errors | al, smal |
| RandomForest/Morgan RF parameter sweep | MDR1_MDCK_classification2, PAMPA_NCATS, pgp_broccatelli | al, smal |
| DMPNN | pgp_broccatelli | al |
| MolFormer | pgp_broccatelli | al |
