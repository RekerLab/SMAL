#!/usr/bin/env python3
"""Build processed CSV inputs used by the figure notebooks.

The raw result folders are expected next to this repository, e.g.
```
project-SMAL/
  SMAL/
  results-initial/
  results-stratified_shuffle/
  results-rf-parameter-sets/
  results-yoked/
```
The generated notebooks read only the CSV files emitted here.
"""

from __future__ import annotations

import csv
import io
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT.parent
OUT_DIR = REPO_ROOT / "results_processed"

MCC_COL = "mcc-model_0"
START_PERCENT = 10
SEEDS = set(range(20))
ERROR_RATES = [0, 10, 20, 30, 40]
NONZERO_ERROR_RATES = [10, 20, 30, 40]
MAX_CURVE_POINTS = 121

DATASETS = [
    "pgp_broccatelli",
    "PAMPA_NCATS",
    "MDR1_MDCK_classification2",
    "ames",
    "CYP3A4_Veith",
    "CYP2D6_Veith",
]
DATASET_LABELS = {
    "pgp_broccatelli": "Pgp",
    "PAMPA_NCATS": "PAMPA",
    "MDR1_MDCK_classification2": "MDR1-MDCK",
    "ames": "Ames",
    "CYP3A4_Veith": "CYP3A4",
    "CYP2D6_Veith": "CYP2D6",
}
STRATEGIES = [
    "random",
    "first",
    "min_oob_error",
    "max_oob_error",
    "min_oob_uncertainty",
    "max_oob_uncertainty",
]
STRATEGY_LABELS = {
    "random": "Random",
    "first": "First",
    "min_oob_error": "MinE",
    "max_oob_error": "MaxE",
    "min_oob_uncertainty": "MinU",
    "max_oob_uncertainty": "MaxU",
}

RF_DATASETS = ["pgp_broccatelli", "PAMPA_NCATS", "MDR1_MDCK_classification2"]
RF_DATASET_LABELS = {
    "pgp_broccatelli": "Pgp",
    "PAMPA_NCATS": "PAMPA",
    "MDR1_MDCK_classification2": "MDR1-MDCK",
}
RF_LABELS = {
    "rf-n50-d10": "50/10",
    "rf-n50-d20": "50/20",
    "rf-n50-none": "50/None",
    "rf-n100-d10": "100/10",
    "rf-n100-d20": "100/20",
    "rf-n500-d10": "500/10",
    "rf-n500-d20": "500/20",
    "rf-n500-none": "500/None",
}
RF_FIGURE_SPECS = [
    ("figureS1", ["rf-n50-d10", "rf-n50-d20"]),
    ("figureS2", ["rf-n100-d10", "rf-n100-d20"]),
    ("figureS3", ["rf-n500-d10", "rf-n500-d20"]),
    ("figureS4", ["rf-n50-none", "rf-n500-none"]),
]

MODEL_ORDER = ["Random Forest", "D-MPNN", "MolFormer"]
DEEP_MODEL_DATASET = "pgp_broccatelli"
DEEP_MODEL_LABEL = "Pgp"
YOKED_MODELS = [("dmpnn", "D-MPNN"), ("molformer", "MolFormer")]
YOKED_STRATEGY = "max_oob_error"


def trapz(values: np.ndarray, x: np.ndarray, axis: int = 1) -> np.ndarray:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, x, axis=axis)
    return np.trapz(values, x, axis=axis)


def plot_x(common_iter: np.ndarray) -> tuple[np.ndarray, int]:
    start_idx = int(len(common_iter) * START_PERCENT / 100)
    if len(common_iter) > 1:
        start_idx = min(start_idx, len(common_iter) - 2)
    else:
        start_idx = 0
    x = np.linspace(100 * start_idx / len(common_iter), 100, len(common_iter) - start_idx)
    return x, start_idx


def sampled_indices(length: int, max_points: int = MAX_CURVE_POINTS) -> np.ndarray:
    if length <= max_points:
        return np.arange(length)
    return np.unique(np.rint(np.linspace(0, length - 1, max_points)).astype(int))


@lru_cache(maxsize=None)
def read_trajectory(path_text: str) -> pd.DataFrame:
    path = Path(path_text)
    with path.open() as handle:
        header = handle.readline().rstrip("\n").split(",")
    try:
        n_iter_field = header.index("n_iter") + 1
        mcc_field = header.index(MCC_COL) + 1
    except ValueError as exc:
        raise ValueError(f"{path} is missing n_iter or {MCC_COL}") from exc
    output = subprocess.check_output(
        ["cut", "-d,", f"-f{n_iter_field},{mcc_field}", str(path)],
        text=True,
    )
    return pd.read_csv(io.StringIO(output)).rename(columns={MCC_COL: "mcc"})


def curve_series(curve: pd.DataFrame, common_iter: np.ndarray) -> pd.Series:
    series = (
        curve.loc[curve["n_iter"] >= 0, ["n_iter", "mcc"]]
        .drop_duplicates("n_iter")
        .set_index("n_iter")["mcc"]
        .reindex(common_iter)
    )
    return series.interpolate(limit_direction="both")


def aligned_series(al_curve: pd.DataFrame, smal_curve: pd.DataFrame | None, common_iter: np.ndarray) -> pd.Series:
    al_series = curve_series(al_curve, common_iter)
    if smal_curve is None:
        return al_series

    return curve_series(smal_curve, common_iter).interpolate(limit_direction="both")


def aggregate_matrix(
    al_curves: dict[int, pd.DataFrame],
    smal_curves: dict[int, pd.DataFrame] | None,
    common_iter: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    if smal_curves is None:
        seeds = sorted(al_curves)
        matrix = np.vstack([aligned_series(al_curves[seed], None, common_iter).to_numpy(dtype=float) for seed in seeds])
    else:
        seeds = sorted(set(al_curves).intersection(smal_curves))
        matrix = np.vstack(
            [
                aligned_series(al_curves[seed], smal_curves[seed], common_iter).to_numpy(dtype=float)
                for seed in seeds
            ]
        )
    return matrix, seeds


def append_curve_rows(
    rows: list[dict[str, object]],
    *,
    source: str,
    error_rate: int,
    dataset: str,
    strategy: str,
    method: str,
    matrix: np.ndarray,
    common_iter: np.ndarray,
    extra: dict[str, object] | None = None,
) -> None:
    x, start_idx = plot_x(common_iter)
    mean = np.nanmean(matrix, axis=0)[start_idx:]
    std = np.nanstd(matrix, axis=0, ddof=1)[start_idx:] if matrix.shape[0] > 1 else np.zeros_like(mean)
    keep = sampled_indices(len(x))
    base = {
        "source": source,
        "error_rate": error_rate,
        "dataset": dataset,
        "dataset_label": DATASET_LABELS.get(dataset, RF_DATASET_LABELS.get(dataset, dataset)),
        "strategy": strategy,
        "strategy_label": STRATEGY_LABELS.get(strategy, strategy),
        "method": method,
        "n_seeds": int(matrix.shape[0]),
    }
    if extra:
        base.update(extra)
    for idx in keep:
        rows.append(
            {
                **base,
                "learning_progress": float(x[idx]),
                "mcc_mean": float(mean[idx]),
                "mcc_std": float(std[idx]),
            }
        )


def paired_auc_stats(al_matrix: np.ndarray, smal_matrix: np.ndarray, common_iter: np.ndarray) -> dict[str, object]:
    x, start_idx = plot_x(common_iter)
    al_auc = trapz(al_matrix[:, start_idx:], x, axis=1)
    smal_auc = trapz(smal_matrix[:, start_idx:], x, axis=1)
    delta = float(np.mean(smal_auc - al_auc))
    p_value = 1.0 if len(al_auc) < 2 or np.allclose(smal_auc, al_auc) else float(wilcoxon(smal_auc, al_auc).pvalue)
    if p_value < 0.05 and delta > 0:
        sign_class = 1
    elif p_value < 0.05 and delta < 0:
        sign_class = -1
    else:
        sign_class = 0
    return {
        "delta_auc": delta,
        "p_value": p_value,
        "n": int(len(al_auc)),
        "sign_class": sign_class,
    }


def load_al_curves(results_root: Path, dataset: str, error_rate: int) -> dict[int, pd.DataFrame]:
    curves = {}
    for path in sorted((results_root / "active_learning").glob(f"al-{dataset}-{error_rate}-*/al_traj.csv")):
        seed = int(path.parent.name.rsplit("-", 1)[-1])
        if seed in SEEDS:
            curves[seed] = read_trajectory(str(path))
    return curves


def load_smal_curves(results_root: Path, dataset: str, error_rate: int, strategy: str) -> dict[int, pd.DataFrame]:
    curves = {}
    for path in sorted((results_root / "smal").glob(f"smal-{dataset}-{error_rate}-{strategy}-*/al_traj.csv")):
        seed = int(path.parent.name.rsplit("-", 1)[-1])
        if seed in SEEDS:
            curves[seed] = read_trajectory(str(path))
    return curves


def compute_learning_results() -> None:
    curve_rows: list[dict[str, object]] = []
    stats_rows: list[dict[str, object]] = []
    sources = {
        "initial": (RAW_ROOT / "results-initial", ERROR_RATES),
        "stratified_shuffle": (RAW_ROOT / "results-stratified_shuffle", NONZERO_ERROR_RATES),
    }

    for source, (results_root, errors) in sources.items():
        print(f"Processing learning curves: {source}", flush=True)
        for error_rate in errors:
            print(f"  error_rate={error_rate}", flush=True)
            for dataset in DATASETS:
                print(f"    dataset={dataset}", flush=True)
                al_curves = load_al_curves(results_root, dataset, error_rate)
                if not al_curves:
                    raise FileNotFoundError(f"No AL curves for {source} {dataset} error={error_rate}")
                common_iter = np.arange(
                    0,
                    min(int(curve.loc[curve["n_iter"] >= 0, "n_iter"].max()) for curve in al_curves.values()) + 1,
                )
                al_matrix_all, _al_seeds = aggregate_matrix(al_curves, None, common_iter)
                append_curve_rows(
                    curve_rows,
                    source=source,
                    error_rate=error_rate,
                    dataset=dataset,
                    strategy="baseline",
                    method="al",
                    matrix=al_matrix_all,
                    common_iter=common_iter,
                )
                for strategy in STRATEGIES:
                    smal_curves = load_smal_curves(results_root, dataset, error_rate, strategy)
                    if not smal_curves:
                        raise FileNotFoundError(
                            f"No SMAL curves for {source} {dataset} error={error_rate} strategy={strategy}"
                        )
                    smal_matrix, smal_seeds = aggregate_matrix(al_curves, smal_curves, common_iter)
                    al_matrix, al_seed_order = aggregate_matrix(
                        {seed: al_curves[seed] for seed in smal_seeds},
                        None,
                        common_iter,
                    )
                    if al_seed_order != smal_seeds:
                        raise RuntimeError("Seed alignment failed")
                    stats = paired_auc_stats(al_matrix, smal_matrix, common_iter)
                    stats_rows.append(
                        {
                            "source": source,
                            "error_rate": error_rate,
                            "dataset": dataset,
                            "dataset_label": DATASET_LABELS[dataset],
                            "strategy": strategy,
                            "strategy_label": STRATEGY_LABELS[strategy],
                            **stats,
                        }
                    )
                    append_curve_rows(
                        curve_rows,
                        source=source,
                        error_rate=error_rate,
                        dataset=dataset,
                        strategy=strategy,
                        method="smal",
                        matrix=smal_matrix,
                        common_iter=common_iter,
                    )
        pd.DataFrame(curve_rows).to_csv(OUT_DIR / "learning_curves.csv", index=False)
        pd.DataFrame(stats_rows).to_csv(OUT_DIR / "learning_auc_stats.csv", index=False)
        print(
            f"Checkpointed learning rows after {source}: "
            f"{len(curve_rows)} curve rows, {len(stats_rows)} AULC rows",
            flush=True,
        )

    pd.DataFrame(curve_rows).to_csv(OUT_DIR / "learning_curves.csv", index=False)
    pd.DataFrame(stats_rows).to_csv(OUT_DIR / "learning_auc_stats.csv", index=False)
    print(f"Saved {len(curve_rows)} learning curve rows and {len(stats_rows)} AULC rows", flush=True)


def compute_label_error_reduction() -> None:
    rows: list[dict[str, object]] = []
    sources = {
        "initial": (RAW_ROOT / "results-initial", NONZERO_ERROR_RATES, "target_error_rate"),
        "stratified_shuffle": (RAW_ROOT / "results-stratified_shuffle", NONZERO_ERROR_RATES, "shuffle_fraction"),
    }
    for source, (results_root, errors, error_column) in sources.items():
        print(f"Processing label-error reductions: {source}", flush=True)
        for dataset in DATASETS:
            for strategy in STRATEGIES:
                for error_rate in errors:
                    for seed in sorted(SEEDS):
                        smal_dir = results_root / "smal" / f"smal-{dataset}-{error_rate}-{strategy}-{seed}"
                        if not smal_dir.exists():
                            continue
                        try:
                            full = pd.read_csv(smal_dir / "full.csv", usecols=["uidx", "Y"])
                            train_init = pd.read_csv(smal_dir / "train_init.csv", usecols=["uidx", "Y"])
                            pool_init = pd.read_csv(smal_dir / "pool_init.csv", usecols=["uidx", "Y"])
                            train_end = pd.read_csv(smal_dir / "train_end.csv", usecols=["uidx"])
                        except FileNotFoundError:
                            continue

                        true_labels = full.rename(columns={"Y": "Y_true"})
                        initial_dataset = pd.concat([train_init, pool_init], ignore_index=True).drop_duplicates("uidx")
                        introduced_dataset = initial_dataset.merge(true_labels, on="uidx", how="inner")
                        final_training = train_end.drop_duplicates("uidx").merge(
                            introduced_dataset[["uidx", "Y", "Y_true"]],
                            on="uidx",
                            how="inner",
                        )
                        if introduced_dataset.empty or final_training.empty:
                            continue

                        introduced_errors = (introduced_dataset["Y"] != introduced_dataset["Y_true"]).sum()
                        final_errors = (final_training["Y"] != final_training["Y_true"]).sum()
                        introduced_error_pct = introduced_errors / len(introduced_dataset) * 100
                        final_error_pct = final_errors / len(final_training) * 100
                        record = {
                            "source": source,
                            "dataset": dataset,
                            "dataset_label": DATASET_LABELS[dataset],
                            "strategy": strategy,
                            "strategy_label": STRATEGY_LABELS[strategy],
                            "error_rate": int(error_rate),
                            error_column: int(error_rate),
                            "seed": int(seed),
                            "introduced_error_pct": float(introduced_error_pct),
                            "final_error_pct": float(final_error_pct),
                            "error_reduction": float(introduced_error_pct - final_error_pct),
                        }
                        rows.append(record)

    error_df = pd.DataFrame(rows)
    error_df.to_csv(OUT_DIR / "label_error_reduction.csv", index=False)
    realized = (
        error_df.groupby(["source", "dataset", "dataset_label", "error_rate"], as_index=False)
        .agg(
            mean_realized_error_pct=("introduced_error_pct", "mean"),
            std_realized_error_pct=("introduced_error_pct", "std"),
            n=("introduced_error_pct", "count"),
        )
        .sort_values(["source", "dataset", "error_rate"])
    )
    realized.to_csv(OUT_DIR / "realized_error_rates.csv", index=False)
    print(f"Saved {len(error_df)} label-error rows", flush=True)


def compute_rf_parameter_results() -> None:
    results_root = RAW_ROOT / "results-rf-parameter-sets"
    al_manifest = pd.read_csv(results_root / "manifest.csv")
    smal_manifest = pd.read_csv(results_root / "smal_manifest.csv")
    al_paths: dict[tuple[str, str], dict[int, Path]] = {}
    smal_paths: dict[tuple[str, str, str], dict[int, Path]] = {}

    for _idx, row in al_manifest.iterrows():
        dataset = row["dataset"]
        rf_set = row["rf_set"]
        seed = int(row["seed"])
        if dataset in RF_DATASETS and rf_set in RF_LABELS and seed in SEEDS:
            al_paths.setdefault((rf_set, dataset), {})[seed] = Path(row["save_dir"]) / "al_traj.csv"

    for _idx, row in smal_manifest.iterrows():
        dataset = row["dataset"]
        rf_set = row["rf_set"]
        strategy = row["forget_method"]
        seed = int(row["seed"])
        if dataset in RF_DATASETS and rf_set in RF_LABELS and strategy in STRATEGIES and seed in SEEDS:
            smal_paths.setdefault((rf_set, dataset, strategy), {})[seed] = Path(row["save_dir"]) / "al_traj.csv"

    curve_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    print("Processing RF-parameter supplemental curves", flush=True)
    for figure_id, rf_sets in RF_FIGURE_SPECS:
        for rf_set in rf_sets:
            for dataset in RF_DATASETS:
                al_curves = {seed: read_trajectory(str(path)) for seed, path in al_paths[(rf_set, dataset)].items()}
                common_iter = np.arange(
                    0,
                    min(int(curve.loc[curve["n_iter"] >= 0, "n_iter"].max()) for curve in al_curves.values()) + 1,
                )
                al_matrix_all, _al_seeds = aggregate_matrix(al_curves, None, common_iter)
                extra = {
                    "figure_id": figure_id,
                    "rf_set": rf_set,
                    "rf_label": RF_LABELS[rf_set],
                }
                append_curve_rows(
                    curve_rows,
                    source="rf_parameter_sets",
                    error_rate=0,
                    dataset=dataset,
                    strategy="baseline",
                    method="al",
                    matrix=al_matrix_all,
                    common_iter=common_iter,
                    extra=extra,
                )
                for strategy in STRATEGIES:
                    smal_curves = {
                        seed: read_trajectory(str(path))
                        for seed, path in smal_paths[(rf_set, dataset, strategy)].items()
                    }
                    smal_matrix, smal_seeds = aggregate_matrix(al_curves, smal_curves, common_iter)
                    al_matrix, al_seed_order = aggregate_matrix(
                        {seed: al_curves[seed] for seed in smal_seeds},
                        None,
                        common_iter,
                    )
                    if al_seed_order != smal_seeds:
                        raise RuntimeError("Seed alignment failed")
                    stats = paired_auc_stats(al_matrix, smal_matrix, common_iter)
                    source_rows.append(
                        {
                            "figure": figure_id,
                            "rf_set": rf_set,
                            "rf_label": RF_LABELS[rf_set],
                            "dataset": dataset,
                            "dataset_label": RF_DATASET_LABELS[dataset],
                            "strategy": strategy,
                            "strategy_label": STRATEGY_LABELS[strategy],
                            "start_percent": START_PERCENT,
                            "significant": bool(stats["p_value"] < 0.05),
                            "smal_better": bool(stats["delta_auc"] > 0),
                            **stats,
                        }
                    )
                    append_curve_rows(
                        curve_rows,
                        source="rf_parameter_sets",
                        error_rate=0,
                        dataset=dataset,
                        strategy=strategy,
                        method="smal",
                        matrix=smal_matrix,
                        common_iter=common_iter,
                        extra=extra,
                    )

    source = pd.DataFrame(source_rows)
    source.to_csv(OUT_DIR / "figureS1-S4_source_data.csv", index=False)
    summary = (
        source.groupby(["figure", "rf_set", "rf_label"], observed=True)
        .agg(
            n_cells=("delta_auc", "size"),
            positive_cells=("smal_better", "sum"),
            significant_positive_cells=("sign_class", lambda values: int((values == 1).sum())),
            significant_negative_cells=("sign_class", lambda values: int((values == -1).sum())),
            mean_delta_auc=("delta_auc", "mean"),
            median_delta_auc=("delta_auc", "median"),
        )
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "figureS1-S4_summary.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(OUT_DIR / "rf_parameter_curves.csv", index=False)
    print(f"Saved {len(curve_rows)} RF-parameter curve rows", flush=True)


def parse_runtime_hours(path: Path) -> float:
    from dateutil import parser

    tzinfos = {"EDT": -4 * 3600, "EST": -5 * 3600}
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("SLURM Node:")]
    date_lines = [line for line in lines if "EDT" in line or "EST" in line]
    if len(date_lines) < 2:
        return float("nan")
    start = parser.parse(date_lines[0], tzinfos=tzinfos)
    end = parser.parse(date_lines[-1], tzinfos=tzinfos)
    return (end - start).total_seconds() / 3600


def deep_model_al_dir(model_name: str) -> Path:
    if model_name == "Random Forest":
        return RAW_ROOT / "results-initial" / "active_learning"
    if model_name == "D-MPNN":
        return RAW_ROOT / "results-dmpnn" / "active_learning"
    if model_name == "MolFormer":
        return RAW_ROOT / "results-molformer" / "active_learning"
    raise KeyError(model_name)


def compute_deep_model_curves() -> None:
    print("Processing figure6 deep-model curves", flush=True)
    model_curves: dict[str, dict[int, pd.DataFrame]] = {}
    for model_name in MODEL_ORDER:
        curves = {}
        for seed in range(20):
            if model_name == "MolFormer" and seed == 11:
                continue
            path = deep_model_al_dir(model_name) / f"al-{DEEP_MODEL_DATASET}-0-{seed}" / "al_traj.csv"
            if path.exists():
                curves[seed] = pd.read_csv(path, usecols=["n_iter", MCC_COL]).rename(columns={MCC_COL: "mcc"})
        if curves:
            model_curves[model_name] = curves
    common_max = min(max(df["n_iter"].max() for df in curves.values()) for curves in model_curves.values())
    rows = []
    for model_name, curves in model_curves.items():
        finished = {seed: df for seed, df in curves.items() if df["n_iter"].max() >= common_max}
        common_iter = np.arange(0, int(common_max) + 1)
        matrix = []
        for seed, df in sorted(finished.items()):
            series = (
                df.loc[df["n_iter"] >= 0, ["n_iter", "mcc"]]
                .drop_duplicates("n_iter")
                .set_index("n_iter")["mcc"]
                .reindex(common_iter)
                .interpolate(limit_direction="both")
            )
            matrix.append(series.to_numpy(dtype=float))
        matrix_arr = np.vstack(matrix)
        for start in range(0, len(common_iter), 10):
            stop = min(start + 10, len(common_iter))
            window_iters = common_iter[start:stop]
            window_matrix = matrix_arr[:, start:stop]
            window_values = np.nanmean(window_matrix, axis=1)
            rows.append(
                {
                    "dataset": DEEP_MODEL_DATASET,
                    "dataset_label": DEEP_MODEL_LABEL,
                    "model": model_name,
                    "learning_progress": float(100 * window_iters.mean() / common_max),
                    "mcc_mean": float(np.nanmean(window_values)),
                    "mcc_std": float(np.nanstd(window_values, ddof=1)),
                    "n_seeds": int(matrix_arr.shape[0]),
                    "available_trajectories": int(len(curves)),
                }
            )
    pd.DataFrame(rows).to_csv(OUT_DIR / "figure6_model_curves.csv", index=False)


def compute_yoked_results() -> None:
    print("Processing figure6 yoked model summaries", flush=True)
    pattern = re.compile(
        r"yk-(?P<model>\w+?)-(?P<ds>.+)-(?P<err>\d+)-"
        r"(?P<strat>random|first|max_oob_error|min_oob_error|max_oob_uncertainty|min_oob_uncertainty)-"
        r"(?P<seed>s\d+|full)$"
    )
    records = []
    for model, _label in YOKED_MODELS:
        for result_dir in (RAW_ROOT / "results-yoked" / model).glob(f"yk-{model}-*"):
            match = pattern.match(result_dir.name)
            if not match or match["ds"] not in DATASET_LABELS:
                continue
            metrics_path = result_dir / "test_ext_metrics.csv"
            if not metrics_path.exists():
                continue
            with metrics_path.open(newline="") as handle:
                vals = {row["metric"]: float(row["value"]) for row in csv.DictReader(handle)}
            records.append(
                {
                    "model": match["model"],
                    "model_label": dict(YOKED_MODELS)[match["model"]],
                    "dataset": match["ds"],
                    "dataset_label": DATASET_LABELS[match["ds"]],
                    "error_rate": int(match["err"]),
                    "strategy": match["strat"],
                    "strategy_label": STRATEGY_LABELS[match["strat"]],
                    "seed": match["seed"],
                    "mcc": vals.get("mcc"),
                }
            )

    ydf = pd.DataFrame(records)
    full_val = ydf[ydf["seed"] == "full"].groupby(["model", "dataset", "error_rate"])["mcc"].mean()
    ysub = ydf[ydf["seed"] != "full"].copy()
    agg = (
        ysub.groupby(["model", "model_label", "dataset", "dataset_label", "error_rate", "strategy", "strategy_label"])["mcc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .merge(full_val.rename("full_mcc").reset_index(), on=["model", "dataset", "error_rate"], how="left")
    )
    cell_ok = (agg["count"] == 20) & agg["full_mcc"].notna()
    complete = {
        key
        for key, value in agg[cell_ok].groupby(["model", "dataset"]).size().items()
        if value == len(STRATEGIES) * len(ERROR_RATES)
    }
    agg["complete_panel"] = [bool((row.model, row.dataset) in complete) for row in agg.itertuples()]
    agg.to_csv(OUT_DIR / "figure6_yoked_curve_stats.csv", index=False)

    drows = []
    for model, model_label in YOKED_MODELS:
        for dataset in DATASETS:
            for strategy in STRATEGIES:
                for error_rate in ERROR_RATES:
                    x = ysub[
                        (ysub["model"] == model)
                        & (ysub["dataset"] == dataset)
                        & (ysub["strategy"] == strategy)
                        & (ysub["error_rate"] == error_rate)
                    ]["mcc"].dropna().to_numpy(dtype=float)
                    fv = full_val.get((model, dataset, error_rate), np.nan)
                    ok = len(x) == 20 and np.isfinite(fv)
                    cls = np.nan
                    p_value = np.nan
                    delta = np.nan
                    if ok:
                        delta = float(x.mean() - fv)
                        diff = x - fv
                        try:
                            p_value = 1.0 if np.allclose(diff, 0) else float(wilcoxon(diff).pvalue)
                        except ValueError:
                            p_value = 1.0
                        cls = 0 if (p_value >= 0.05 or delta == 0) else (1 if delta > 0 else -1)
                    drows.append(
                        {
                            "model": model,
                            "model_label": model_label,
                            "dataset": dataset,
                            "dataset_label": DATASET_LABELS[dataset],
                            "strategy": strategy,
                            "strategy_label": STRATEGY_LABELS[strategy],
                            "error_rate": error_rate,
                            "sign_class": cls,
                            "ok": ok,
                            "delta_mcc": delta,
                            "p_value": p_value,
                        }
                    )
    pd.DataFrame(drows).to_csv(OUT_DIR / "figure6_yoked_significance.csv", index=False)


def copy_existing_processed_tables() -> None:
    figure_dir = RAW_ROOT / "figures"
    names = [
        "figure3_summary.csv",
        "figure5_source_data.csv",
        "figure5_summary.csv",
        "figure6_compute_time.csv",
        "figure6_summary.csv",
        "figureS5_source_data.csv",
        "figureS5_summary.csv",
    ]
    for name in names:
        shutil.copy2(figure_dir / name, OUT_DIR / name)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    compute_learning_results()
    compute_label_error_reduction()
    compute_rf_parameter_results()
    compute_deep_model_curves()
    compute_yoked_results()
    copy_existing_processed_tables()
    print(f"Processed results written to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
