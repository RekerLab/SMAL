# SMAL

**Short-term Memory Active Learning** for molecular property prediction.

This repository contains the code, configurations, and analysis materials needed to reproduce the experiments and figures from the SMAL manuscript.

## Installation

SMAL is built on top of [MolALKit](https://github.com/RekerLab/MolALKit). All experiments in this repository were run with **MolALKit v1.2.0** — please install that version following the instructions in the MolALKit README to reproduce the published results.

## Reproducing the experiments

The [`scripts/`](scripts/) folder contains one bash script per *(dataset, model, learning_type)* combination. See [`scripts/README.md`](scripts/README.md) for the file-naming convention and usage.

## Processed results and figures

The [`results_processed/`](results_processed/) folder contains processed figure inputs. The [`figures/`](figures/) folder contains one notebook per manuscript figure (`figure2` through `figure6` and `figureS1` through `figureS14`); running a notebook regenerates the corresponding SVG from the processed inputs.

## Datasets

The [`datasets/`](datasets/) folder contains the CSV input datasets used in this study:

- 6 ADMET benchmark datasets: `ames`, `CYP2D6_Veith`, `CYP3A4_Veith`, `MDR1_MDCK_classification2`, `PAMPA_NCATS`, and `pgp_broccatelli`.
- 99 SIMPD ChEMBL Ki assay datasets, named `CHEMBL*.csv`.

These files are copied from the local MolALKit packaged datasets and the SIMPD source-data directory so the study inputs are available directly from this repository.

## Colab demo

For a single-run, browser-only walkthrough, upload [`colab_demo.ipynb`](colab_demo.ipynb) to [Google Colab](https://colab.research.google.com/). The notebook installs MolALKit v1.2.0, exposes the full parameter space used in `scripts/` (dataset, model, learning strategy, seed, error rate, forget method), and runs an SMAL example on the `pgp_broccatelli` dataset by default.
