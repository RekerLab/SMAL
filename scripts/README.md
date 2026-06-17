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

Each script is self-contained: it loops over the seeds, error rates, and (for SMAL) forget methods used in the manuscript, calling `molalkit_run` once per combination.

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

The scripts are intentionally **simple sequential loops** — they do not submit SLURM jobs, parallelize, or skip already-completed runs. For HPC use, wrap each `molalkit_run` invocation in your own job scheduler.

## Currently included experiments

| Model | Datasets | Learning types |
|-------|----------|----------------|
| RandomForest/Morgan | ames, CYP2D6_Veith, CYP3A4_Veith, MDR1_MDCK_classification2, PAMPA_NCATS, pgp_broccatelli | al, pl, smal |
| DMPNN | pgp_broccatelli | al |
| MolFormer | pgp_broccatelli | al |

Additional experiments (imbalanced datasets, SIMPD split, stratified shuffle split, RF hyperparameter sweep) will be added in the same `<dataset>-<model>-<learning_type>.sh` format.
