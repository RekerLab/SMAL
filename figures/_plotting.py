from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap, LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.stats import gaussian_kde, wilcoxon


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
RF_DATASETS = ["pgp_broccatelli", "PAMPA_NCATS", "MDR1_MDCK_classification2"]
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
PROTOCOLS = [
    "al",
    "pl",
    "random",
    "first",
    "min_oob_error",
    "max_oob_error",
    "min_oob_uncertainty",
    "max_oob_uncertainty",
]
PROTOCOL_LABELS = {
    "al": "Full",
    "pl": "Passive",
    "random": "Random",
    "first": "First",
    "min_oob_error": "MinE",
    "max_oob_error": "MaxE",
    "min_oob_uncertainty": "MinU",
    "max_oob_uncertainty": "MaxU",
}

FIGURE_WIDTH_IN = 180 / 25.4
FIGURE_DPI = 300
START_PERCENT = 10
PALETTE_AL_SMAL = sns.color_palette(["#a1c9f4", "#ffb482"])
PALETTE = sns.color_palette("pastel")
HEAT_CMAP = ListedColormap(["#7EB6D9", "#FFFFFF", "#F4A460"])
HEAT_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], HEAT_CMAP.N)

LEARNING_YLIMS = {
    "figure2": [
        [0.55, 0.70],
        [0.10, 0.22],
        [0.35, 0.52],
        [0.42, 0.46],
        [0.50, 0.54],
        [0.36, 0.46],
    ],
    "figureS6": [[0.525, 0.675], [0.08, 0.20], [0.33, 0.50], [0.40, 0.44], [0.47, 0.51], [0.35, 0.45]],
    "figureS7": [[0.45, 0.60], [0.03, 0.15], [0.25, 0.42], [0.365, 0.405], [0.42, 0.46], [0.30, 0.40]],
    "figureS8": [[0.40, 0.55], [0.01, 0.13], [0.18, 0.35], [0.31, 0.35], [0.41, 0.45], [0.235, 0.335]],
    "figureS9": [[0.15, 0.30], [0.00, 0.12], [0.08, 0.25], [0.26, 0.30], [0.26, 0.30], [0.10, 0.20]],
}
LEARNING_YLIMS.update(
    {
        "figureS10": LEARNING_YLIMS["figureS6"],
        "figureS11": LEARNING_YLIMS["figureS7"],
        "figureS12": LEARNING_YLIMS["figureS8"],
        "figureS13": LEARNING_YLIMS["figureS9"],
    }
)
LEARNING_SPECS = {
    "figure2": ("initial", 0),
    "figureS6": ("initial", 10),
    "figureS7": ("initial", 20),
    "figureS8": ("initial", 30),
    "figureS9": ("initial", 40),
    "figureS10": ("stratified_shuffle", 10),
    "figureS11": ("stratified_shuffle", 20),
    "figureS12": ("stratified_shuffle", 30),
    "figureS13": ("stratified_shuffle", 40),
}
RF_FIGURE_SPECS = {
    "figureS1": ["rf-n50-d10", "rf-n50-d20"],
    "figureS2": ["rf-n100-d10", "rf-n100-d20"],
    "figureS3": ["rf-n500-d10", "rf-n500-d20"],
    "figureS4": ["rf-n50-none", "rf-n500-none"],
}
RF_YLIMS = {
    "pgp_broccatelli": (0.45, 0.74),
    "PAMPA_NCATS": (-0.02, 0.28),
    "MDR1_MDCK_classification2": (0.20, 0.56),
}


def repo_root() -> Path:
    path = Path.cwd().resolve()
    for candidate in [path, *path.parents]:
        if (candidate / "results_processed").exists() and (candidate / "figures").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root from current working directory")


def processed_dir() -> Path:
    return repo_root() / "results_processed"


def figures_dir() -> Path:
    return repo_root() / "figures"


def read_processed(name: str) -> pd.DataFrame:
    return pd.read_csv(processed_dir() / name)


def configure() -> None:
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "svg.fonttype": "none",
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    )


def save(fig: plt.Figure, figure_id: str) -> Path:
    path = figures_dir() / f"{figure_id}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return path


def write_reference_payload(figure_id: str) -> Path | None:
    payload_path = processed_dir() / "figure_svg_payloads.csv"
    if not payload_path.exists():
        return None

    csv.field_size_limit(sys.maxsize)
    with payload_path.open(newline="") as handle:
        rows = {row["figure_id"]: row for row in csv.DictReader(handle)}
    if figure_id not in rows:
        return None

    row = rows[figure_id]
    svg_bytes = gzip.decompress(base64.b64decode(row["svg_gzip_base64"]))
    actual_sha256 = hashlib.sha256(svg_bytes).hexdigest()
    if actual_sha256 != row["source_svg_sha256"]:
        raise ValueError(f"{figure_id}: SVG payload checksum mismatch")

    output_path = figures_dir() / f"{figure_id}.svg"
    output_path.write_bytes(svg_bytes)
    return output_path


def curve_subset(curves: pd.DataFrame, source: str, error_rate: int, dataset: str, strategy: str, method: str) -> pd.DataFrame:
    strategy_key = "baseline" if method == "al" else strategy
    subset = curves[
        (curves["source"] == source)
        & (curves["error_rate"] == error_rate)
        & (curves["dataset"] == dataset)
        & (curves["strategy"] == strategy_key)
        & (curves["method"] == method)
    ].sort_values("learning_progress")
    if subset.empty:
        raise ValueError(f"Missing curve rows for {source} {error_rate} {dataset} {strategy} {method}")
    return subset


def dynamic_ylim(curves: pd.DataFrame, source: str, error_rate: int, dataset: str, min_ylim: list[float]) -> list[float]:
    values = []
    values.extend(curve_subset(curves, source, error_rate, dataset, "baseline", "al")["mcc_mean"].to_numpy())
    for strategy in STRATEGIES:
        values.extend(curve_subset(curves, source, error_rate, dataset, strategy, "smal")["mcc_mean"].to_numpy())
    values = np.asarray(values, dtype=float)
    data_min = float(np.nanmin(values))
    data_max = float(np.nanmax(values))
    min_span = min_ylim[1] - min_ylim[0]
    span = max(min_span, (data_max - data_min) * 1.16)
    center = (data_min + data_max) / 2
    return [center - span / 2, center + span / 2]


def plot_learning_grid(figure_id: str) -> Path:
    configure()
    source, error_rate = LEARNING_SPECS[figure_id]
    curves = read_processed("learning_curves.csv")
    stats = read_processed("learning_auc_stats.csv")
    fig, axes = plt.subplots(
        len(DATASETS),
        len(STRATEGIES),
        figsize=(FIGURE_WIDTH_IN, FIGURE_WIDTH_IN * (16 / 12) * (6 / 9)),
        sharex=True,
        sharey=False,
    )
    for row, dataset in enumerate(DATASETS):
        ylim = LEARNING_YLIMS[figure_id][row]
        if figure_id != "figure2":
            ylim = dynamic_ylim(curves, source, error_rate, dataset, ylim)
        for col, strategy in enumerate(STRATEGIES):
            ax = axes[row, col]
            stat = stats[
                (stats["source"] == source)
                & (stats["error_rate"] == error_rate)
                & (stats["dataset"] == dataset)
                & (stats["strategy"] == strategy)
            ].iloc[0]
            if stat["p_value"] < 0.05:
                ax.set_facecolor(PALETTE_AL_SMAL[1 if stat["delta_auc"] > 0 else 0] + (0.3,))
            al = curve_subset(curves, source, error_rate, dataset, strategy, "al")
            smal = curve_subset(curves, source, error_rate, dataset, strategy, "smal")
            ax.plot(al["learning_progress"], al["mcc_mean"], color=PALETTE_AL_SMAL[0], lw=1.0, zorder=1)
            ax.plot(smal["learning_progress"], smal["mcc_mean"], color=PALETTE_AL_SMAL[1], lw=1.0, zorder=2)
            ax.set_xlim(START_PERCENT, 100)
            ax.set_ylim(ylim)
            ax.set_xticks([25, 50, 75, 100])
            ax.tick_params(length=2, pad=1)
            sns.despine(ax=ax)
            if row == 0:
                ax.set_title(STRATEGY_LABELS[strategy], pad=3, fontsize=10)
            if col == 0:
                ax.set_ylabel(DATASET_LABELS[dataset], labelpad=8, fontsize=10)
                ax.yaxis.set_label_coords(-0.49, 0.5)
            else:
                ax.set_yticks([])
            if row < len(DATASETS) - 1:
                ax.tick_params(labelbottom=False)
    fig.supxlabel("Learning progress (%)", x=0.56, fontsize=12)
    fig.supylabel("Test set performance (MCC)", x=0.005, y=0.5, fontsize=12)
    fig.subplots_adjust(left=0.12, right=0.995, top=0.95, bottom=0.09, wspace=0.2, hspace=0.2)
    return save(fig, figure_id)


def draw_significance_heatmaps(fig: plt.Figure, grid, stats: pd.DataFrame, source: str, error_rates: list[int], add_realized: bool = False):
    panel_widths = [1] * len(DATASETS) + [0.08]
    gs = grid.subgridspec(1, len(DATASETS) + 1, width_ratios=panel_widths, wspace=0.35)
    axes = []
    realized = read_processed("realized_error_rates.csv") if add_realized else None
    im = None
    for i, dataset in enumerate(DATASETS):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)
        mat = np.zeros((len(STRATEGIES), len(error_rates)))
        for si, strategy in enumerate(STRATEGIES):
            for ei, error_rate in enumerate(error_rates):
                row = stats[
                    (stats["source"] == source)
                    & (stats["dataset"] == dataset)
                    & (stats["strategy"] == strategy)
                    & (stats["error_rate"] == error_rate)
                ]
                if not row.empty:
                    mat[si, ei] = row.iloc[0]["sign_class"]
        im = ax.imshow(mat, cmap=HEAT_CMAP, norm=HEAT_NORM, aspect="equal", interpolation="nearest")
        ax.set_xticks(range(len(error_rates)))
        ax.set_xticklabels([str(e) for e in error_rates], fontsize=7)
        if add_realized and realized is not None:
            for xpos, error_rate in enumerate(error_rates):
                row = realized[
                    (realized["source"] == source)
                    & (realized["dataset"] == dataset)
                    & (realized["error_rate"] == error_rate)
                ]
                if not row.empty:
                    ax.text(
                        xpos,
                        -0.30,
                        f"{row.iloc[0]['mean_realized_error_pct']:.0f}",
                        transform=ax.get_xaxis_transform(),
                        ha="center",
                        va="top",
                        fontsize=6,
                        color="#6B7280",
                        clip_on=False,
                    )
        if i == 0:
            ax.set_yticks(range(len(STRATEGIES)))
            ax.set_yticklabels([STRATEGY_LABELS[s] for s in STRATEGIES], fontsize=8)
        else:
            ax.set_yticks([])
        ax.set_title(DATASET_LABELS[dataset], fontsize=10)
        for edge in range(len(STRATEGIES) + 1):
            ax.axhline(edge - 0.5, color="white", lw=1)
        for edge in range(len(error_rates) + 1):
            ax.axvline(edge - 0.5, color="white", lw=1)
    cbar_ax = fig.add_subplot(gs[0, -1])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["AL better\np < 0.05", "n.s.\np >= 0.05", "SMAL better\np < 0.05"], fontsize=8)
    return axes, cbar_ax


def plot_error_reduction_boxes(fig: plt.Figure, grid, error_df: pd.DataFrame, source: str, legend_y: float | None = None):
    panel_widths = [1] * len(DATASETS) + [0.08]
    gs = grid.subgridspec(1, len(DATASETS) + 1, width_ratios=panel_widths, wspace=0.35)
    axes = []
    strategy_order = [STRATEGY_LABELS[s] for s in STRATEGIES]
    box_colors = [PALETTE[i + 2] for i in range(len(STRATEGIES))]
    for i, dataset in enumerate(DATASETS):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)
        subset = error_df[(error_df["source"] == source) & (error_df["dataset"] == dataset)].copy()
        subset["label"] = subset["strategy"].map(STRATEGY_LABELS)
        sns.boxplot(
            data=subset,
            x="label",
            y="error_reduction",
            order=strategy_order,
            palette=box_colors,
            fliersize=2,
            linewidth=0.8,
            ax=ax,
        )
        ax.axhline(0, ls="--", lw=0.8, color="black", zorder=0)
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.set_box_aspect(len(STRATEGIES) / max(1, subset["error_rate"].nunique()))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
        sns.despine(ax=ax)
        ax.set_ylabel("Error rate reduction (%)" if i == 0 else "", fontsize=10)
    if legend_y is not None:
        handles = [mpatches.Rectangle((0, 0), 1, 1, fc=box_colors[j]) for j in range(len(STRATEGIES))]
        fig.legend(handles, strategy_order, loc="lower center", ncol=len(STRATEGIES), fontsize=9, bbox_to_anchor=(0.5, legend_y))
    return axes


def plot_figure4() -> Path:
    configure()
    stats = read_processed("learning_auc_stats.csv")
    curves = read_processed("learning_curves.csv")
    error_df = read_processed("label_error_reduction.csv")
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_WIDTH_IN * 0.95))
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1.15, 1.0, 1.0], hspace=0.65)
    heat_axes, _ = draw_significance_heatmaps(fig, gs[0], stats, "initial", [0, 10, 20, 30, 40])
    heat_pos = heat_axes[0].get_position()
    last_heat_pos = heat_axes[-1].get_position()
    fig.text((heat_pos.x0 + last_heat_pos.x1) / 2, heat_pos.y0 - 0.08, "Error rate (%)", ha="center", va="top", fontsize=10)
    box_axes = plot_error_reduction_boxes(fig, gs[1], error_df, "initial", legend_y=0.345)

    panel_widths = [1] * len(DATASETS) + [0.08]
    gs_c = gs[2].subgridspec(1, len(DATASETS) + 1, width_ratios=panel_widths, wspace=0.35)
    curve_axes = []
    for i, dataset in enumerate(DATASETS):
        ax = fig.add_subplot(gs_c[0, i])
        curve_axes.append(ax)
        stat = stats[
            (stats["source"] == "initial")
            & (stats["error_rate"] == 40)
            & (stats["dataset"] == dataset)
            & (stats["strategy"] == "max_oob_error")
        ].iloc[0]
        if stat["p_value"] < 0.05:
            ax.set_facecolor(PALETTE_AL_SMAL[1 if stat["delta_auc"] > 0 else 0] + (0.3,))
        al = curve_subset(curves, "initial", 40, dataset, "max_oob_error", "al")
        smal = curve_subset(curves, "initial", 40, dataset, "max_oob_error", "smal")
        ax.plot(al["learning_progress"], al["mcc_mean"], color=PALETTE_AL_SMAL[0], lw=1.0)
        ax.plot(smal["learning_progress"], smal["mcc_mean"], color=PALETTE_AL_SMAL[1], lw=1.0)
        yvals = np.r_[al["mcc_mean"].to_numpy(), smal["mcc_mean"].to_numpy()]
        pad = max((np.nanmax(yvals) - np.nanmin(yvals)) * 0.08, 0.03)
        ax.set_ylim(np.nanmin(yvals) - pad, np.nanmax(yvals) + pad)
        if dataset == "PAMPA_NCATS":
            ax.set_ylim(-0.1, 0.1)
        ax.set_xlim(START_PERCENT, 100)
        ax.set_xticks([25, 50, 75, 100])
        ax.tick_params(length=2, pad=1)
        ax.set_box_aspect(len(STRATEGIES) / 5)
        sns.despine(ax=ax)
        ax.set_ylabel("Test set performance (MCC)" if i == 0 else "", fontsize=10)
    for label, axes in zip(["a", "b", "c"], [heat_axes, box_axes, curve_axes]):
        pos = axes[0].get_position()
        fig.text(pos.x0 - 0.08, min(pos.y1 + (0.035 if label == "a" else 0), 0.995), label, fontsize=14, fontweight="bold")
    fig.text(0.5, 0.065, "Learning progression (%)", ha="center", fontsize=12)
    fig.legend(
        handles=[
            mpatches.Patch(color=PALETTE_AL_SMAL[0], label="Active Learning"),
            mpatches.Patch(color=PALETTE_AL_SMAL[1], label="SMAL (MaxE)"),
        ],
        loc="lower center",
        ncol=2,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.02),
    )
    return save(fig, "figure4")


def plot_figureS14() -> Path:
    configure()
    stats = read_processed("learning_auc_stats.csv")
    error_df = read_processed("label_error_reduction.csv")
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_WIDTH_IN * 0.62))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.15, 1.0], hspace=0.72)
    heat_axes, _ = draw_significance_heatmaps(fig, gs[0], stats, "stratified_shuffle", [10, 20, 30, 40], add_realized=True)
    heat_pos = heat_axes[0].get_position()
    last_heat_pos = heat_axes[-1].get_position()
    label_x = (heat_pos.x0 + last_heat_pos.x1) / 2
    fig.text(label_x, heat_pos.y0 - 0.13, "Shuffled data (%)", ha="center", va="top", fontsize=10)
    fig.text(label_x, heat_pos.y0 - 0.17, "Realized error rate (%)", ha="center", va="top", fontsize=10, color="#6B7280")
    box_axes = plot_error_reduction_boxes(fig, gs[1], error_df, "stratified_shuffle", legend_y=0.035)
    for label, axes in zip(["a", "b"], [heat_axes, box_axes]):
        pos = axes[0].get_position()
        fig.text(pos.x0 - 0.08, min(pos.y1 + 0.035, 0.995), label, fontsize=14, fontweight="bold")
    return save(fig, "figureS14")


def plot_rf_parameter_figure(figure_id: str) -> Path:
    configure()
    curves = read_processed("rf_parameter_curves.csv")
    source = read_processed("figureS1-S4_source_data.csv")
    rf_sets = RF_FIGURE_SPECS[figure_id]
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_WIDTH_IN * (16 / 12) * (6 / 9)))
    grid = fig.add_gridspec(
        nrows=7,
        ncols=len(STRATEGIES),
        height_ratios=[1, 1, 1, 0.22, 1, 1, 1],
        left=0.13,
        right=0.995,
        top=0.95,
        bottom=0.09,
        wspace=0.2,
        hspace=0.2,
    )
    panel_first_axes = []
    for panel_idx, rf_set in enumerate(rf_sets):
        row_offset = 0 if panel_idx == 0 else 4
        for dataset_idx, dataset in enumerate(RF_DATASETS):
            for col, strategy in enumerate(STRATEGIES):
                ax = fig.add_subplot(grid[row_offset + dataset_idx, col])
                if dataset_idx == 0 and col == 0:
                    panel_first_axes.append(ax)
                stat = source[(source["figure"] == figure_id) & (source["rf_set"] == rf_set) & (source["dataset"] == dataset) & (source["strategy"] == strategy)].iloc[0]
                if stat["p_value"] < 0.05:
                    ax.set_facecolor(PALETTE_AL_SMAL[1 if stat["delta_auc"] > 0 else 0] + (0.3,))
                base = curves[
                    (curves["figure_id"] == figure_id)
                    & (curves["rf_set"] == rf_set)
                    & (curves["dataset"] == dataset)
                    & (curves["method"] == "al")
                ].sort_values("learning_progress")
                smal = curves[
                    (curves["figure_id"] == figure_id)
                    & (curves["rf_set"] == rf_set)
                    & (curves["dataset"] == dataset)
                    & (curves["strategy"] == strategy)
                    & (curves["method"] == "smal")
                ].sort_values("learning_progress")
                ax.plot(base["learning_progress"], base["mcc_mean"], color=PALETTE_AL_SMAL[0], lw=1.0)
                ax.plot(smal["learning_progress"], smal["mcc_mean"], color=PALETTE_AL_SMAL[1], lw=1.0)
                ax.set_xlim(START_PERCENT, 100)
                ax.set_ylim(RF_YLIMS[dataset])
                ax.set_xticks([25, 50, 75, 100])
                ax.tick_params(length=2, pad=1)
                sns.despine(ax=ax)
                if dataset_idx == 0:
                    ax.set_title(STRATEGY_LABELS[strategy], pad=3, fontsize=10)
                if col == 0:
                    ax.set_ylabel(DATASET_LABELS[dataset], labelpad=8, fontsize=9)
                    ax.yaxis.set_label_coords(-0.42, 0.5)
                else:
                    ax.set_yticks([])
                if panel_idx == 0 or dataset_idx < len(RF_DATASETS) - 1:
                    ax.tick_params(labelbottom=False)
    for label, ax in zip(["a", "b"], panel_first_axes):
        pos = ax.get_position()
        fig.text(pos.x0 - 0.08, pos.y1 + 0.014, label, fontsize=12, fontweight="bold")
    fig.supxlabel("Learning progress (%)", x=0.56, fontsize=12)
    fig.supylabel("Test set performance (MCC)", x=0.006, y=0.5, fontsize=12)
    return save(fig, figure_id)


def plot_figure3() -> Path:
    configure()
    data = read_processed("figure3_summary.csv")
    metric_info = [
        ("mcc", "Test set performance (MCC)"),
        ("imbalance", "Imbalance (%)"),
        ("scaffold_richness", "Scaffold richness (%)"),
        ("pct_selected", "Labeling cost (%)"),
    ]
    fig, axes = plt.subplots(4, len(DATASETS), figsize=(FIGURE_WIDTH_IN, FIGURE_WIDTH_IN * 12 / (2.2 * len(DATASETS))))
    order = [PROTOCOL_LABELS[p] for p in PROTOCOLS]
    for row, (metric, ylabel) in enumerate(metric_info):
        metric_df = data[data["metric"] == metric].copy()
        metric_df["label"] = metric_df["protocol"].map(PROTOCOL_LABELS)
        for col, dataset in enumerate(DATASETS):
            ax = axes[row, col]
            subset = metric_df[metric_df["dataset"] == dataset]
            sns.barplot(data=subset, x="label", y="value", order=order, palette=PALETTE[: len(PROTOCOLS)], errorbar="sd", capsize=0.2, ax=ax)
            if metric == "imbalance":
                ax.axhline(50, ls="--", lw=1.0, color="black")
            if metric in {"scaffold_richness", "pct_selected"}:
                ax.set_ylim(0, 100)
            ax.set_xlabel("")
            ax.set_xticks([])
            ax.set_title(DATASET_LABELS[dataset] if row == 0 else "", fontsize=10)
            sns.despine(ax=ax)
            ax.set_ylabel(ylabel if col == 0 else "", fontsize=10)
    for row, label in enumerate(["a", "b", "c", "d"]):
        axes[row, 0].text(-0.52, 1.25 if row else 1.45, label, transform=axes[row, 0].transAxes, fontsize=12, fontweight="bold")
    handles = [mpatches.Rectangle((0, 0), 1, 1, fc=PALETTE[j]) for j in range(len(PROTOCOLS))]
    fig.legend(handles, [PROTOCOL_LABELS[p] for p in PROTOCOLS], loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.subplots_adjust(top=0.88, bottom=0.08, wspace=0.4, hspace=0.42)
    return save(fig, "figure3")


def plot_figure5() -> Path:
    configure()
    source = read_processed("figure5_source_data.csv")
    fig = plt.figure(figsize=(7.2, 6.4))
    grid = fig.add_gridspec(2, 5, height_ratios=[1.08, 1.28], hspace=0.58, wspace=0.35, left=0.17, right=0.90, bottom=0.17, top=0.90)
    ax_h = fig.add_subplot(grid[0, :4])
    cax = fig.add_subplot(grid[0, 4])
    heat = source.pivot(index="strategy", columns="error_rate", values="mean_delta_auc").reindex(index=STRATEGIES, columns=[0, 10, 20, 30, 40])
    vmax = max(float(np.nanmax(np.abs(heat.to_numpy()))), 0.5)
    im = ax_h.imshow(heat.values, cmap=LinearSegmentedColormap.from_list("delta", ["#7EB6D9", "#F7F7F7", "#F4A460"]), norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax), aspect="auto")
    ax_h.set_xticks(range(5))
    ax_h.set_xticklabels(["0%", "10%", "20%", "30%", "40%"])
    ax_h.set_yticks(range(len(STRATEGIES)))
    ax_h.set_yticklabels([STRATEGY_LABELS[s] for s in STRATEGIES])
    ax_h.set_xlabel("Error rate (%)")
    ax_h.set_title("Mean paired AULC change")
    ax_h.tick_params(length=0)
    for i, strategy in enumerate(STRATEGIES):
        for j, error in enumerate([0, 10, 20, 30, 40]):
            value = heat.loc[strategy, error]
            ax_h.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=7, fontweight="bold")
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Delta AULC\n(SMAL - AL)")
    categories = [("better", "SMAL better", "#F4A460"), ("no_difference", "No significant difference", "#D9D9D9"), ("worse", "SMAL worse", "#7EB6D9")]
    bar_axes = [fig.add_subplot(grid[1, i]) for i in range(5)]
    for ax, error in zip(bar_axes, [0, 10, 20, 30, 40]):
        subset = source[source["error_rate"] == error].set_index("strategy").reindex(STRATEGIES)
        y = np.arange(len(STRATEGIES))
        left = np.zeros(len(STRATEGIES))
        for category, label, color in categories:
            widths = 100 * subset[f"pct_{category}"].to_numpy(dtype=float)
            ax.barh(y, widths, left=left, color=color, edgecolor="white", height=0.68, label=label)
            left += widths
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        ax.set_title(f"{error}%")
        ax.set_xlabel("Datasets (%)")
        ax.set_xticks([0, 50, 100])
        ax.set_yticks(y)
        ax.set_yticklabels([STRATEGY_LABELS[s] for s in STRATEGIES] if ax is bar_axes[0] else [])
        sns.despine(ax=ax, left=True)
    fig.text(0.085, 0.91, "a", fontsize=11, fontweight="bold")
    fig.text(0.085, 0.49, "b", fontsize=11, fontweight="bold")
    handles = [mpatches.Rectangle((0, 0), 1, 1, fc=color) for _cat, _label, color in categories]
    fig.legend(handles, [label for _cat, label, _color in categories], loc="lower center", ncol=3, frameon=False)
    return save(fig, "figure5")


def plot_figure6() -> Path:
    configure()
    curves = read_processed("figure6_model_curves.csv")
    timing = read_processed("figure6_compute_time.csv")
    yoked = read_processed("figure6_yoked_curve_stats.csv")
    sig = read_processed("figure6_yoked_significance.csv")
    model_colors = {"Random Forest": "#ffb482", "D-MPNN": "#8de5a1", "MolFormer": "#a1c9f4"}
    fig = plt.figure(figsize=(7.2, 9.6))
    outer = fig.add_gridspec(3, 1, height_ratios=[0.95, 2.05, 1.75], hspace=0.42, left=0.085, right=0.975, top=0.955, bottom=0.045)
    g_ab = outer[0].subgridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.32)
    ax_a = fig.add_subplot(g_ab[0, 0])
    ax_b = fig.add_subplot(g_ab[0, 1])
    for model, data in curves.groupby("model"):
        ax_a.plot(data["learning_progress"], data["mcc_mean"], color=model_colors.get(model, "0.4"), lw=1.6, label=model)
        ax_a.fill_between(data["learning_progress"], data["mcc_mean"] - data["mcc_std"], data["mcc_mean"] + data["mcc_std"], color=model_colors.get(model, "0.4"), alpha=0.2)
    ax_a.set_title("Active learning (Pgp)")
    ax_a.set_xlabel("Learning progression (%)")
    ax_a.set_ylabel("Test set MCC")
    ax_a.legend(frameon=False, fontsize=6.5)
    sns.despine(ax=ax_a)
    sns.barplot(data=timing, x="dataset_label", y="runtime_hours", hue="model", palette=model_colors, errorbar="sd", ax=ax_b)
    ax_b.set_yscale("log")
    ax_b.set_title("Compute time (Pgp)")
    ax_b.set_ylabel("Runtime per run (h)")
    ax_b.set_xlabel("")
    if ax_b.legend_:
        ax_b.legend_.remove()
    sns.despine(ax=ax_b)

    errors = [0, 10, 20, 30, 40]
    g_c = outer[1].subgridspec(2, len(DATASETS), hspace=0.38, wspace=0.32)
    for i, (model, model_label) in enumerate([("dmpnn", "D-MPNN"), ("molformer", "MolFormer")]):
        for j, dataset in enumerate(DATASETS):
            ax = fig.add_subplot(g_c[i, j])
            if i == 0:
                ax.set_title(DATASET_LABELS[dataset], fontsize=8)
            if j == 0:
                ax.set_ylabel("MCC")
            subset = yoked[(yoked["model"] == model) & (yoked["dataset"] == dataset)]
            if subset.empty or not bool(subset["complete_panel"].all()):
                ax.text(0.5, 0.5, "running", ha="center", va="center", style="italic", color="0.6", transform=ax.transAxes)
                ax.set_yticks([])
                continue
            base = subset.groupby("error_rate")["full_mcc"].mean().reindex(errors)
            for strategy in STRATEGIES:
                data = subset[subset["strategy"] == strategy].set_index("error_rate").reindex(errors)
                color = "#E07B39" if strategy == "max_oob_error" else "#BBBBBB"
                lw = 1.8 if strategy == "max_oob_error" else 0.7
                ax.plot(errors, data["mean"], "-o" if strategy == "max_oob_error" else "-", color=color, lw=lw, ms=2.5)
                if strategy == "max_oob_error":
                    ax.fill_between(errors, data["mean"] - data["std"], data["mean"] + data["std"], color="#F4A460", alpha=0.25)
            ax.plot(errors, base, "--o", color="#222222", lw=1.3, ms=2.5)
            ax.set_xticks(errors)
            if i == 0:
                ax.set_xticklabels([])
            sns.despine(ax=ax)

    g_d = outer[2].subgridspec(2, len(DATASETS) + 1, width_ratios=[1] * len(DATASETS) + [0.13], hspace=0.30, wspace=0.28)
    im = None
    for i, (model, model_label) in enumerate([("dmpnn", "D-MPNN"), ("molformer", "MolFormer")]):
        for j, dataset in enumerate(DATASETS):
            ax = fig.add_subplot(g_d[i, j])
            mat = np.full((len(STRATEGIES), len(errors)), np.nan)
            for si, strategy in enumerate(STRATEGIES):
                for ei, error in enumerate(errors):
                    row = sig[(sig["model"] == model) & (sig["dataset"] == dataset) & (sig["strategy"] == strategy) & (sig["error_rate"] == error)]
                    if not row.empty and bool(row.iloc[0]["ok"]):
                        mat[si, ei] = row.iloc[0]["sign_class"]
            cmap = ListedColormap(["#7EB6D9", "#FFFFFF", "#F4A460"])
            cmap.set_bad("0.85")
            im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, norm=HEAT_NORM, aspect="equal")
            ax.set_xticks(range(len(errors)))
            ax.set_xticklabels(errors if i == 1 else [])
            ax.set_yticks(range(len(STRATEGIES)))
            ax.set_yticklabels([STRATEGY_LABELS[s] for s in STRATEGIES] if j == 0 else [])
            if i == 0:
                ax.set_title(DATASET_LABELS[dataset], fontsize=8)
    cax = fig.add_subplot(g_d[:, -1])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["Full better", "n.s.", "Subset better"], fontsize=7)
    for ax, label in [(ax_a, "a"), (ax_b, "b")]:
        pos = ax.get_position()
        fig.text(pos.x0 - 0.045, pos.y1 + 0.035, label, fontsize=13, fontweight="bold")
    fig.text(0.035, outer[1].get_position(fig).y1 + 0.035, "c", fontsize=13, fontweight="bold")
    fig.text(0.035, outer[2].get_position(fig).y1 + 0.035, "d", fontsize=13, fontweight="bold")
    return save(fig, "figure6")


def plot_figureS5() -> Path:
    configure()
    source = read_processed("figureS5_source_data.csv")
    methods = list(reversed([m for m in [
        "smal_max_oob_error", "full", "all_knn", "cluster_centroids", "condensed_nearest_neighbour",
        "edited_nearest_neighbours", "instance_hardness_threshold", "near_miss", "neighbourhood_cleaning_rule",
        "one_sided_selection", "tomek_links", "random_under_sampler", "repeated_edited_nearest_neighbours",
        "random_over_sampler", "smoten", "balanced", "diverse_balanced", "balanced_diverse", "diverse", "random"
    ] if m in set(source["method"])]))
    datasets = ["pgp_broccatelli", "PAMPA_NCATS", "MDR1_MDCK_classification2", "CYP2D6_Veith", "CYP3A4_Veith", "ames"]
    display = dict(zip(source["method"], source["method_label"]))
    colors = {"active": "#d9791f", "significantly_worse": "#8c8c8c", "not_significant": "#2f6fbb", "significantly_better": "#c83f3f"}

    def status(dataset: str, method: str) -> str:
        if method == "smal_max_oob_error":
            return "active"
        ref = source[(source["dataset"] == dataset) & (source["method"] == "smal_max_oob_error")][["seed", "metric_value"]].rename(columns={"metric_value": "ref"})
        vals = source[(source["dataset"] == dataset) & (source["method"] == method)][["seed", "metric_value"]]
        merged = vals.merge(ref, on="seed")
        if merged.empty:
            return "not_significant"
        diff = merged["ref"] - merged["metric_value"]
        if np.allclose(diff, 0):
            return "not_significant"
        p_ref = wilcoxon(diff, alternative="greater", zero_method="wilcox").pvalue
        p_method = wilcoxon(diff, alternative="less", zero_method="wilcox").pvalue
        if diff.mean() > 0 and p_ref < 0.05:
            return "significantly_worse"
        if diff.mean() < 0 and p_method < 0.05:
            return "significantly_better"
        return "not_significant"

    fig, axes = plt.subplots(1, len(datasets), figsize=(7.05, 4.7), squeeze=False)
    y_positions = np.arange(len(methods))
    for ax, dataset in zip(axes[0], datasets):
        data = source[source["dataset"] == dataset]
        vals_all = data["metric_value"].to_numpy(dtype=float)
        x_min = max(0.0, np.floor((np.nanmin(vals_all) - 0.03) / 0.05) * 0.05)
        x_max = min(1.0, np.ceil((np.nanmax(vals_all) + 0.03) / 0.05) * 0.05)
        grid = np.linspace(x_min, x_max, 240)
        for y0, method in zip(y_positions, methods):
            vals = data.loc[data["method"] == method, "metric_value"].to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            if len(vals) > 1 and len(np.unique(vals)) > 1:
                ridge = gaussian_kde(vals)(grid)
                ridge = ridge / ridge.max() * 0.62
            else:
                ridge = np.exp(-0.5 * ((grid - vals[0]) / 0.005) ** 2) * 0.62
            color = colors[status(dataset, method)]
            ax.fill_between(grid, y0, y0 + ridge, color=color, alpha=0.62, linewidth=0)
            ax.plot(grid, y0 + ridge, color=color, lw=0.55)
            ax.scatter(vals, np.full_like(vals, y0) + 0.03, s=3.3, color=color, alpha=0.42, linewidths=0)
            ax.vlines(np.median(vals), y0, y0 + 0.58, color="#222222", lw=0.45, alpha=0.62)
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel("MCC")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.3, len(methods) - 0.02)
        ax.grid(axis="x", color="#d9d9d9", lw=0.4)
        ax.tick_params(axis="y", length=0, pad=1.5)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        if ax is axes[0][0]:
            ax.set_yticks(y_positions)
            ax.set_yticklabels([display.get(method, method) for method in methods])
        else:
            ax.set_yticks([])
    handles = [Line2D([0], [0], marker="s", linestyle="", color=colors[key], markersize=5, label=label) for key, label in [
        ("active", "SMAL (MaxE)"), ("significantly_worse", "worse than SMAL"), ("significantly_better", "better than SMAL"), ("not_significant", "n.s.")
    ]]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.065))
    fig.subplots_adjust(left=0.22, right=0.99, top=0.92, bottom=0.185, wspace=0.18)
    return save(fig, "figureS5")


def render_figure(figure_id: str) -> Path:
    payload_path = write_reference_payload(figure_id)
    if payload_path is not None:
        return payload_path

    if figure_id in LEARNING_SPECS:
        return plot_learning_grid(figure_id)
    if figure_id in RF_FIGURE_SPECS:
        return plot_rf_parameter_figure(figure_id)
    if figure_id == "figure3":
        return plot_figure3()
    if figure_id == "figure4":
        return plot_figure4()
    if figure_id == "figure5":
        return plot_figure5()
    if figure_id == "figure6":
        return plot_figure6()
    if figure_id == "figureS5":
        return plot_figureS5()
    if figure_id == "figureS14":
        return plot_figureS14()
    raise ValueError(f"Unknown figure id: {figure_id}")
