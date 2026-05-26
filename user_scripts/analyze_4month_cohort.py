"""Run the IntelliCage place-learning workflow for the BioMedX 4-month cohort.

This user-facing script is the main entry point for the current poster
workflow. It reads the four IntelliCage runs, merges metadata and behavior
tables, computes activity and place-learning metrics, and renders poster-ready
plots into a results directory that always lives inside the selected dataset
directory.
"""
# %% IMPORTS
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from intellicage_place_learning.loader import load_cohort_data
from intellicage_place_learning.loader import attach_analysis_time_columns
from intellicage_place_learning.metrics import (
    build_phase_time_limit_table,
    build_analysis_phase_window_table,
    compute_awake_day_rate_tables,
    compute_awake_day_ratio_tables,
    compute_binomial_glm_group_statistics,
    compute_clustered_binomial_gee_group_statistics,
    compute_experiment_drinking_visit_bins,
    compute_experiment_lick_count_bins,
    compute_experiment_nosepoke_count_bins,
    compute_experiment_visit_bins,
    compute_first_hours_rate_table,
    compute_group_day_violin_statistics,
    compute_onset_group_statistics,
    compute_phase4_reversal_rate_bins,
    compute_phase4_reversal_count_bins,
    compute_phase2_adaptation_bins,
    compute_phase_activity_medians,
    compute_phase_activity_statistics,
    compute_phase_segment_rate_tables,
    compute_phase_visit_count_bins,
    compute_place_learning_count_bins,
    compute_place_learning_rate_bins,
    compute_responder_group_statistics,
    compute_role_cumulative_curves,
    compute_threshold_responder_table,
    compute_time_window_learning_curves,
    compute_visit_window_learning_curves,
    filter_visits_by_phase_limits,
    flag_iqr_outliers,
    suggest_common_phase_limits,
)
from intellicage_place_learning.plotting import (
    configure_plot_style,
    plot_experiment_dual_metric_bars,
    plot_experiment_overview,
    plot_experiment_overview_groups,
    plot_cumulative_role_curves,
    plot_group_day_violin,
    plot_onset_violin,
    plot_phase2_adaptation,
    plot_phase_activity_boxplot,
    plot_phase_learning_counts,
    plot_phase_learning_counts_groups,
    plot_phase_learning_rate,
    plot_phase_learning_rate_groups,
    plot_phase_segment_rate_groups,
    plot_phase4_reversal_components,
    plot_visit_learning_curve_groups,
    sanitize_filename_part,
    set_group_colors,
    set_figure_size_presets,
)
# %% PARAMETERS AND DEFAULTS
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "Data IntelliCage" / "BioMedX_4MonthCohort_2019"
DEFAULT_PHASE_DISPLAY_NAMES = {
    1: "Free Hab",
    2: "NPA",
    3: "PL",
    4: "PR",
}
DEFAULT_PHASE_MAX_HOURS = {
    3: 72.0,
    4: 72.0,
}
DEFAULT_EXCLUDED_GROUPS = ["WT", "Tau 1-421", "Tau 66-421"]
DEFAULT_GROUP_RENAMES = {
    "WT": "WT",
    "tdTomato": "Control",
    "Tau 1-441": "Tau",
    "Tau 1-421": "Tau 1-421",
    "Tau 66-421": "Tau 66-421",
}
DEFAULT_GROUP_COLORS = {
    "WT": "#264653",
    "tdTomato": "#6c757d",
    "Tau 66-421": "#2a9d8f",
    "Tau 1-421": "#e9a820",
    "Tau 1-441": "#4ade80",
}
DEFAULT_FIGSIZE_CM = {# always a pair of (width, height)
    "LONG_FIGSIZE_CM":          (24, 10.0), # (18.2, 7.4),
    "LONG_FIGSIZE_2_CM":        (24, 10.0), #(15.2, 7.4),
    "PHASE2_FIGSIZE_CM":        (15.0, 10.0), #(10.4, 7.0),
    "MEDIUM_FIGSIZE_CM":        (11.8, 7.6),
    "MEDIUM_WIDE_FIGSIZE_CM":   (12.8, 8.0),
    "SEGMENT_FIGSIZE_CM":       (18.0, 11.0), #(12.6, 7.9),
    "VIOLIN_FIGSIZE_CM":        (5.8, 10), #(5.8, 7.2),
    "ONSET_FIGSIZE_CM":         (5.8, 10), #(5.8, 7.0),
    "ACTIVITY_FIGSIZE_CM":      (8.8, 8.1),
    "WIDE_GROUP_FIGSIZE_CM":    (27, 10.0),#(18.2, 7.4),
}
DEFAULT_PHASE2_PLOT_STYLE = "line"
DEFAULT_MOUSE_DAY_START_HOUR = 6.0
DEFAULT_AWAKE_DURATION_HOURS = 12.0
DEFAULT_SCHEDULED_PHASE_START_HOURS = {
    1: 0.0,
    2: 74.0,
    3: 122.0,
    4: 194.0,
    5: 266.0,
}
USER_DATASET_ROOT = DEFAULT_DATASET_ROOT
USER_RESULTS_SUBDIR = Path("results")
USER_BIN_HOURS = [1, 2]
USER_PHASE2_SECONDARY_METRIC = "lick_positive_visits"
USER_SPREAD_METRIC = "sem"
USER_PLOT_STYLE = "line"
USER_PHASE2_PLOT_STYLE = "line"
USER_PHASE_MAX_HOURS = DEFAULT_PHASE_MAX_HOURS.copy()
USER_EXCLUDED_GROUPS = DEFAULT_EXCLUDED_GROUPS.copy()
USER_GROUP_RENAMES = DEFAULT_GROUP_RENAMES.copy()
USER_GROUP_COLORS = DEFAULT_GROUP_COLORS.copy()
USER_FIGSIZE_CM = DEFAULT_FIGSIZE_CM.copy()
USER_MOUSE_DAY_START_HOUR = DEFAULT_MOUSE_DAY_START_HOUR
USER_AWAKE_DURATION_HOURS = DEFAULT_AWAKE_DURATION_HOURS
USER_SCHEDULED_PHASE_START_HOURS = DEFAULT_SCHEDULED_PHASE_START_HOURS.copy()
USER_BASE_FONT_SIZE = 15 #10.0
USER_EXCLUDE_VIOLIN_OUTLIERS = True
USER_RATE_THRESHOLD_PCTS = [50.0, 60.0, 70.0, 80.0]
USER_THRESHOLD_ONSET_BIN_HOURS = 1
USER_RESPONDER_HORIZONS_HOURS = [24.0, 48.0, 72.0]
USER_BINOMIAL_MODEL_FIRST_HOURS = 24.0
USER_SUMMARY_RESPONDER_HORIZON_HOURS = 24.0
USER_SUMMARY_FIGSIZE_CM = (8.2, 7.1)
CM_TO_INCH = 2.54
# %% FUNCTIONS
def parse_numeric_mapping(raw_items: list[str]) -> dict[int, float]:
    """Parse `key=value` CLI strings into a numeric dictionary."""

    limits: dict[int, float] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid mapping '{item}'. Use the form KEY=VALUE.")
        phase_text, hour_text = item.split("=", 1)
        limits[int(phase_text)] = float(hour_text)
    return limits

def parse_group_rename_mapping(raw_items: list[str] | None) -> dict[str, str]:
    """Parse `old=new` group-renaming entries into a string dictionary."""

    if not raw_items:
        return {}
    mapping: dict[str, str] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid group rename '{item}'. Use the form OLD=NEW.")
        old_name, new_name = item.split("=", 1)
        mapping[old_name] = new_name
    return mapping

def resolve_output_root(dataset_root: Path, results_subdir: Path) -> Path:
    """Resolve a results directory that is always relative to the dataset root."""

    if results_subdir.is_absolute():
        raise ValueError("`--results-subdir` must be a relative path below the dataset root.")
    return dataset_root / results_subdir

def ordered_group_names(visits) -> list[str]:
    """Extract pathology-group names in their categorical display order."""

    categories = getattr(visits["Group"].dtype, "categories", None)
    if categories is not None:
        return [str(category) for category in categories if str(category) != "nan"]
    return sorted(visits["Group"].dropna().astype(str).unique())

def active_period_bounds(mouse_day_start_hour: float, awake_duration_hours: float) -> tuple[float, float]:
    """Return the absolute clock bounds of the configured awake period."""

    awake_end_clock_hour = mouse_day_start_hour + awake_duration_hours
    if awake_end_clock_hour <= mouse_day_start_hour or awake_end_clock_hour > 24.0:
        raise ValueError(
            "The current plotting helpers support a same-day awake window. "
            "Please choose `mouse_day_start_hour + awake_duration_hours <= 24`."
        )
    return mouse_day_start_hour, awake_end_clock_hour

def resolved_group_colors(
    *,
    group_renames: dict[str, str],
    group_colors: dict[str, str] | None,
) -> dict[str, str]:
    """Resolve group colors against the active rename mapping for plotting."""

    base_colors = DEFAULT_GROUP_COLORS.copy()
    if group_colors:
        base_colors.update(group_colors)
    resolved: dict[str, str] = {}
    for original_name, color in base_colors.items():
        resolved[group_renames.get(original_name, original_name)] = color
    if group_colors:
        for key, value in group_colors.items():
            resolved.setdefault(str(key), str(value))
    return resolved

def render_mouse_age_at_phase1_start_plot(
    metadata: pd.DataFrame,
    phase_manifest: pd.DataFrame,
    output_root: Path,
    *,
    group_renames: dict[str, str],
) -> None:
    """Plot mouse age at the observed start of phase 1 for all loaded mice.

    This plot is intentionally derived from the full metadata table before any
    later analysis exclusions are applied, so it always reflects every mouse
    listed in `Mice.txt`.
    """

    if metadata.empty or phase_manifest.empty:
        return

    phase1_starts = (
        phase_manifest.loc[phase_manifest["PhaseNumber"].eq(1), ["RunGroup", "PhaseStart"]]
        .rename(columns={"PhaseStart": "Phase1Start"})
        .copy()
    )
    if phase1_starts.empty:
        return

    age_table = metadata.merge(phase1_starts, on="RunGroup", how="left", validate="many_to_one").copy()
    age_table["GroupOriginal"] = age_table["Group"].astype(str)
    age_table["Group"] = age_table["GroupOriginal"].map(group_renames).fillna(age_table["GroupOriginal"])
    age_table["age_days_at_phase1_start"] = (
        age_table["Phase1Start"] - age_table["DOB"]
    ).dt.total_seconds() / 86400.0
    age_table["age_months_at_phase1_start"] = age_table["age_days_at_phase1_start"] / 30.4375
    age_table = age_table.loc[age_table["age_months_at_phase1_start"].notna()].copy()
    if age_table.empty:
        return

    preferred_groups = [str(name) for name in group_renames.values()]
    present_groups = set(age_table["Group"].astype(str))
    ordered_groups = [group_name for group_name in preferred_groups if group_name in present_groups]
    ordered_groups.extend(
        group_name for group_name in age_table["Group"].astype(str) if group_name not in ordered_groups
    )
    age_table["Group"] = pd.Categorical(age_table["Group"], categories=ordered_groups, ordered=True)
    age_table["PhaseNumber"] = 1

    save_table(age_table, output_root / "mouse_age_at_phase1_start_months_mouse.tsv")
    flagged = flag_iqr_outliers(
        age_table,
        value_col="age_months_at_phase1_start",
        group_cols=["Group"],
    )
    save_table(flagged, output_root / "mouse_age_at_phase1_start_months_mouse_with_outliers.tsv")
    omnibus, pairwise = compute_onset_group_statistics(
        flagged.rename(columns={"age_months_at_phase1_start": "onset_hours"}),
        onset_col="onset_hours",
        phase_number=1,
        metric_name="mouse_age_at_phase1_start_months",
        exclude_outliers=False,
    )
    save_table(omnibus, output_root / "mouse_age_at_phase1_start_months_omnibus_stats.tsv")
    save_table(pairwise, output_root / "mouse_age_at_phase1_start_months_pairwise_stats.tsv")
    plot_onset_violin(
        flagged.rename(columns={"age_months_at_phase1_start": "onset_hours"}),
        onset_col="onset_hours",
        phase_display_name="Phase 1 start",
        title_label="Mouse age at phase 1 start",
        ylabel="Age [months]",
        output_path=output_root / "mouse_age_at_phase1_start_months_violin.png",
        pairwise_stats=pairwise,
        outlier_col="is_outlier",
        reference_line=None,
    )

def phase_origin_clock_hour(mouse_day_start_hour: float, scheduled_phase_start_hour: float) -> float:
    """Return the wall-clock hour that corresponds to phase-relative time zero."""

    return float((mouse_day_start_hour + (scheduled_phase_start_hour % 24.0)) % 24.0)


def experiment_day_from_scheduled_start(scheduled_phase_start_hour: float) -> int:
    """Convert one scheduled phase-start hour to its global experiment-day number."""

    return int(scheduled_phase_start_hour // 24.0)

def save_table(dataframe, output_path: Path) -> None:
    """Save a DataFrame as a tab-separated text file with parent creation."""

    target_path = output_path.parent / "csv" / output_path.name
    target_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(target_path, sep="\t", index=False)

def csv_output_path(output_path: Path) -> Path:
    """Return the standardized CSV destination below a local `csv` subfolder."""

    return output_path.parent / "csv" / output_path.name

def _summary_output_paths(output_root: Path, stem: str) -> tuple[Path, Path]:
    """Return the PNG and PDF destinations for one summary figure stem."""

    png_path = output_root / f"{stem}.png"
    pdf_path = output_root / "pdf" / f"{stem}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    return png_path, pdf_path

def _load_result_table(output_root: Path, filename: str) -> pd.DataFrame:
    """Load one tab-separated result table from the standardized CSV folder."""

    table_path = output_root / "csv" / filename
    if not table_path.exists():
        raise FileNotFoundError(f"Expected result table not found: {table_path}")
    return pd.read_csv(table_path, sep="\t")

def _load_binned_result_table(output_root: Path, *, bin_hours: int, filename: str) -> pd.DataFrame:
    """Load one result table from a bin-specific CSV subfolder."""

    table_path = output_root / f"{int(bin_hours)}h_bins" / "csv" / filename
    if not table_path.exists():
        raise FileNotFoundError(f"Expected binned result table not found: {table_path}")
    return pd.read_csv(table_path, sep="\t")

def _ordered_available_groups(frame: pd.DataFrame, preferred_order: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Return preferred groups that are actually present in one result table."""

    present = set(frame["Group"].astype(str))
    return tuple(group_name for group_name in preferred_order if group_name in present)

def apply_group_preferences(
    visits: pd.DataFrame,
    metadata: pd.DataFrame,
    nosepokes: pd.DataFrame,
    *,
    excluded_groups: list[str] | None,
    group_renames: dict[str, str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply optional group exclusion and renaming to all analysis tables."""

    visits = visits.copy()
    metadata = metadata.copy()
    nosepokes = nosepokes.copy()

    visits["GroupOriginal"] = visits["Group"].astype(str)
    metadata["GroupOriginal"] = metadata["Group"].astype(str)

    if excluded_groups:
        exclude_set = {str(group_name) for group_name in excluded_groups}
        visits = visits.loc[~visits["GroupOriginal"].isin(exclude_set)].copy()
        metadata = metadata.loc[~metadata["GroupOriginal"].isin(exclude_set)].copy()

    if group_renames:
        visits["Group"] = visits["GroupOriginal"].map(group_renames).fillna(visits["GroupOriginal"])
        metadata["Group"] = metadata["GroupOriginal"].map(group_renames).fillna(metadata["GroupOriginal"])
    else:
        visits["Group"] = visits["GroupOriginal"]
        metadata["Group"] = metadata["GroupOriginal"]

    if visits.empty:
        raise ValueError("No visits remain after applying the group-selection settings.")

    seen: list[str] = []
    preferred_order = [str(name) for name in (group_renames or {}).values()]
    available_groups = set(visits["Group"].astype(str))
    for group_name in preferred_order:
        if group_name in available_groups and group_name not in seen:
            seen.append(group_name)
    for group_name in visits["Group"].astype(str):
        if group_name not in seen:
            seen.append(group_name)
    visits["Group"] = pd.Categorical(visits["Group"], categories=seen, ordered=True)
    metadata["Group"] = pd.Categorical(metadata["Group"], categories=seen, ordered=True)

    kept_visit_keys = visits.loc[:, ["RunGroup", "Phase", "PhaseNumber", "VisitID"]].drop_duplicates()
    nosepokes = nosepokes.merge(
        kept_visit_keys,
        on=["RunGroup", "Phase", "PhaseNumber", "VisitID"],
        how="inner",
        validate="many_to_one",
    )
    return visits, metadata, nosepokes

def _draw_distribution_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    value_col: str,
    group_order: tuple[str, ...] | list[str],
    group_colors: dict[str, str],
    title: str,
    ylabel: str,
    pairwise_stats: pd.DataFrame | None = None,
    pairwise_filter_column: str | None = None,
    pairwise_filter_value: object | None = None,
    as_percent: bool = True,
    reference_line: float | None = None,
) -> None:
    """Draw one compact distribution panel with points and violin summaries."""

    panel = data.loc[data["Group"].isin(group_order)].copy()
    if panel.empty:
        ax.set_axis_off()
        return

    value_scale = 100.0 if as_percent else 1.0
    positions = np.arange(1, len(group_order) + 1)
    violin_data = [
        panel.loc[panel["Group"].astype(str).eq(group_name), value_col].dropna().to_numpy(dtype=float) * value_scale
        for group_name in group_order
    ]
    violins = ax.violinplot(violin_data, positions=positions, widths=0.72, showmeans=False, showmedians=True)
    for body, group_name in zip(violins["bodies"], group_order):
        body.set_facecolor(group_colors[group_name])
        body.set_edgecolor("none")
        body.set_linewidth(0.0)
        body.set_alpha(0.25)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in violins:
            violins[key].set_color("#555555")
            violins[key].set_linewidth(1.0)

    for position, group_name, values in zip(positions, group_order, violin_data):
        jitter = np.linspace(-0.10, 0.10, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=24,
            color=group_colors[group_name],
            edgecolor="none",
            zorder=3,
            alpha=0.9,
        )

    if reference_line is not None:
        ref_value = float(reference_line) * value_scale if as_percent else float(reference_line)
        ax.axhline(ref_value, color="#4f4f4f", linestyle="--", linewidth=1.0, zorder=1)

    ax.set_xticks(positions)
    ax.set_xticklabels(group_order, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=plt.rcParams["axes.titlesize"])
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    numeric_values = (
        np.concatenate([values for values in violin_data if len(values) > 0])
        if any(len(values) > 0 for values in violin_data)
        else np.array([0.0])
    )
    y_max = float(np.nanmax(numeric_values)) if numeric_values.size else 1.0
    default_min = 100.0 if as_percent else 1.0
    y_data_max = float(max(default_min, y_max))
    y_base = max(
        (104.0 if as_percent else y_data_max * 1.05),
        y_data_max + (5.0 if as_percent else max(0.1, y_data_max * 0.08)),
    )
    y_step = 10.0 if as_percent else max(0.12, y_data_max * 0.10)
    y_limit = 124.0 if as_percent else max(1.25, y_base + 0.4)

    significant_pairs = pd.DataFrame()
    if pairwise_stats is not None and not pairwise_stats.empty and {"group1", "group2", "p_value"}.issubset(pairwise_stats.columns):
        significant_pairs = pairwise_stats.copy()
        if pairwise_filter_column is not None and pairwise_filter_column in significant_pairs.columns:
            significant_pairs = significant_pairs.loc[
                significant_pairs[pairwise_filter_column].eq(pairwise_filter_value)
            ].copy()
        significant_pairs = significant_pairs.loc[
            significant_pairs["group1"].astype(str).isin(group_order)
            & significant_pairs["group2"].astype(str).isin(group_order)
            & significant_pairs["p_value"].lt(0.05)
        ].copy()
        significant_pairs["left_pos"] = significant_pairs["group1"].astype(str).map(
            {group: idx + 1 for idx, group in enumerate(group_order)}
        )
        significant_pairs["right_pos"] = significant_pairs["group2"].astype(str).map(
            {group: idx + 1 for idx, group in enumerate(group_order)}
        )
        significant_pairs["span"] = (significant_pairs["right_pos"] - significant_pairs["left_pos"]).abs()
        significant_pairs = significant_pairs.sort_values(["span", "left_pos", "right_pos"]).reset_index(drop=True)
        for pair_index, (_, row) in enumerate(significant_pairs.iterrows()):
            left = int(min(row["left_pos"], row["right_pos"]))
            right = int(max(row["left_pos"], row["right_pos"]))
            line_y = y_base + pair_index * y_step
            y_limit = max(y_limit, line_y + 7.0)
            ax.plot([left, left, right, right], [line_y - 1.1, line_y, line_y, line_y - 1.1], color="#444444", linewidth=1.0)
            ax.text(
                (left + right) / 2.0,
                line_y + (1.8 if as_percent else max(0.04, y_step * 0.18)),
                f"p={float(row['p_value']):.3g}",
                ha="center",
                va="bottom",
                fontsize=max(6.0, float(plt.rcParams["font.size"]) - 2.0),
                color="#444444",
            )

    if as_percent:
        ax.set_ylim(0, max(y_limit, 105.0))
    else:
        ax.set_ylim(0, max(y_limit, 1.05, y_max * 1.18))

def _draw_responder_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    group_order: tuple[str, ...] | list[str],
    threshold_pcts: list[float] | tuple[float, ...],
    group_colors: dict[str, str],
    title: str,
    ylabel: str,
) -> None:
    """Draw one responder-rate summary panel across thresholds."""

    panel = data.loc[data["Group"].isin(group_order)].copy()
    if panel.empty:
        ax.set_axis_off()
        return

    for group_name in group_order:
        group_frame = panel.loc[panel["Group"].astype(str).eq(group_name)].copy()
        if group_frame.empty:
            continue
        group_frame = group_frame.sort_values("threshold_pct")
        ax.plot(
            group_frame["threshold_pct"].to_numpy(dtype=float),
            group_frame["responder_rate"].to_numpy(dtype=float) * 100.0,
            marker="o",
            linewidth=1.2,
            markersize=4.5,
            color=group_colors[group_name],
            label=group_name,
        )
    ax.set_title(title)
    ax.set_xlabel("Threshold [%]")
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(threshold_pcts))
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")

def _save_panel_figure(
    output_root: Path,
    stem: str,
    draw_fn,
) -> None:
    """Create and save one single-panel summary figure."""

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(USER_SUMMARY_FIGSIZE_CM[0] / CM_TO_INCH, USER_SUMMARY_FIGSIZE_CM[1] / CM_TO_INCH),
    )
    draw_fn(ax)
    png_path, pdf_path = _summary_output_paths(output_root, stem)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

def _build_pl_rewarded_gee_stats(
    output_root: Path,
    *,
    awake_duration_hours: float,
    bin_hours: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute and save mouse-clustered GEE stats for early PL rewarded-correct rates."""

    hourly = _load_binned_result_table(
        output_root,
        bin_hours=bin_hours,
        filename="phase3_rewarded_correct_corner_visit_rate_mouse_bins_1h.tsv",
    )
    hourly = hourly.loc[hourly["all_visits"].fillna(0).gt(0)].copy()

    day1_awake = hourly.loc[
        hourly["bin_start_hours"].ge(0.0)
        & hourly["bin_start_hours"].lt(float(awake_duration_hours))
    ].copy()
    first24h = hourly.loc[
        hourly["bin_start_hours"].ge(0.0)
        & hourly["bin_start_hours"].lt(24.0)
    ].copy()

    day1_omnibus, day1_pairwise = compute_clustered_binomial_gee_group_statistics(
        day1_awake,
        phase_number=3,
        metric_name=f"rewarded_correct_corner_day1awake_gee_{int(bin_hours)}h",
        success_col="correct_visits",
        total_col="all_visits",
    )
    first24_omnibus, first24_pairwise = compute_clustered_binomial_gee_group_statistics(
        first24h,
        phase_number=3,
        metric_name=f"rewarded_correct_corner_first24h_gee_{int(bin_hours)}h",
        success_col="correct_visits",
        total_col="all_visits",
    )

    save_table(day1_awake, output_root / f"phase3_rewarded_correct_corner_day1awake_gee_{int(bin_hours)}h_mouse_bins.tsv")
    save_table(day1_omnibus, output_root / f"phase3_rewarded_correct_corner_day1awake_gee_{int(bin_hours)}h_omnibus_stats.tsv")
    save_table(day1_pairwise, output_root / f"phase3_rewarded_correct_corner_day1awake_gee_{int(bin_hours)}h_pairwise_stats.tsv")
    save_table(first24h, output_root / f"phase3_rewarded_correct_corner_first24h_gee_{int(bin_hours)}h_mouse_bins.tsv")
    save_table(first24_omnibus, output_root / f"phase3_rewarded_correct_corner_first24h_gee_{int(bin_hours)}h_omnibus_stats.tsv")
    save_table(first24_pairwise, output_root / f"phase3_rewarded_correct_corner_first24h_gee_{int(bin_hours)}h_pairwise_stats.tsv")
    return day1_omnibus, day1_pairwise, first24_omnibus, first24_pairwise

def render_target_group_summary_panels(
    *,
    output_root: Path,
    all_groups_order: tuple[str, ...],
    threshold_pcts: list[float] | tuple[float, ...],
    responder_horizon_hours: float,
    group_colors: dict[str, str],
    awake_duration_hours: float,
) -> None:
    """Render single-panel summary figures for targeted and all-group views."""
    rewarded_day1 = _load_result_table(output_root, "phase3_rewarded_correct_corner_awake_day_rate_mouse.tsv")
    rewarded_day1 = rewarded_day1.loc[rewarded_day1["phase_day"].eq(1)].copy()
    rewarded_day1_omnibus, rewarded_day1_stats, first24h_omnibus, first24h_stats = _build_pl_rewarded_gee_stats(
        output_root,
        awake_duration_hours=awake_duration_hours,
        bin_hours=1,
    )

    first24h = _load_result_table(output_root, "phase3_rewarded_correct_corner_first24h_count_model_mouse.tsv")

    reversal_pref = _load_result_table(output_root, "phase4_reversal_preference_index_awake_day_rate_mouse.tsv")
    reversal_pref = reversal_pref.loc[reversal_pref["phase_day"].eq(3)].copy()
    reversal_pref_stats = _load_result_table(output_root, "phase4_reversal_preference_index_awake_day_rate_pairwise_stats.tsv")

    responder_frames: list[pd.DataFrame] = []
    for threshold_pct in threshold_pcts:
        frame = _load_result_table(
            output_root,
            f"phase3_rewarded_correct_corner_1h_threshold_responder_{int(threshold_pct)}pct_summary.tsv",
        )
        frame = frame.loc[frame["horizon_hours"].eq(float(responder_horizon_hours))].copy()
        responder_frames.append(frame)
    responder_summary = pd.concat(responder_frames, ignore_index=True) if responder_frames else pd.DataFrame()

    all_rewarded = _ordered_available_groups(rewarded_day1, all_groups_order)
    all_first24h = _ordered_available_groups(first24h, all_groups_order)
    all_reversal = _ordered_available_groups(reversal_pref, all_groups_order)
    all_responder = _ordered_available_groups(responder_summary, all_groups_order)
    _save_panel_figure(
        output_root,
        "all_groups_pl_day1_awake_rewarded_correct_corner",
        lambda ax: _draw_distribution_panel(
            ax,
            rewarded_day1,
            value_col="value",
            group_order=all_rewarded,
            group_colors=group_colors,
            title="PL day 1 awake\nRewarded correct-corner rate",
            ylabel="Rewarded correct-corner rate [%]",
            pairwise_stats=rewarded_day1_stats,
            as_percent=True,
            reference_line=0.25,
        ),
    )
    _save_panel_figure(
        output_root,
        "all_groups_pl_first24h_rewarded_correct_corner",
        lambda ax: _draw_distribution_panel(
            ax,
            first24h,
            value_col="value",
            group_order=all_first24h,
            group_colors=group_colors,
            title="PL first 24 h\nRewarded correct-corner rate",
            ylabel="Rewarded correct-corner rate [%]",
            pairwise_stats=first24h_stats,
            as_percent=True,
            reference_line=0.25,
        ),
    )
    _save_panel_figure(
        output_root,
        f"all_groups_pl_responders_{int(responder_horizon_hours)}h",
        lambda ax: _draw_responder_panel(
            ax,
            responder_summary,
            group_order=all_responder,
            threshold_pcts=threshold_pcts,
            group_colors=group_colors,
            title=f"PL responders by {int(responder_horizon_hours)} h",
            ylabel="Responder rate [%]",
        ),
    )
    _save_panel_figure(
        output_root,
        "all_groups_pr_day3_reversal_preference_index",
        lambda ax: _draw_distribution_panel(
            ax,
            reversal_pref,
            value_col="value",
            group_order=all_reversal,
            group_colors=group_colors,
            title="PR day 3 awake\nReversal preference index",
            ylabel="New / (new + previous)\n(higher = better)",
            pairwise_stats=reversal_pref_stats,
            pairwise_filter_column="phase_day",
            pairwise_filter_value=3,
            as_percent=False,
            reference_line=0.5,
        ),
    )

def render_overview_plots(
    visits,
    output_dir: Path,
    *,
    bin_hours: int,
    group_names: list[str],
    phase_window_table,
    phase_display_names: dict[int, str],
    spread_metric: str,
    plot_style: str,
    mouse_day_start_hour: float,
    awake_end_clock_hour: float,
) -> None:
    """Create full-experiment visit-activity plots for every pathology group."""

    mouse_bins, summary_bins = compute_experiment_visit_bins(visits, bin_hours=bin_hours)
    save_table(mouse_bins, output_dir / f"overview_visits_mouse_bins_{bin_hours}h.tsv")
    save_table(summary_bins, output_dir / f"overview_visits_group_summary_{bin_hours}h.tsv")
    group_end_hours = (
        visits.groupby("Group", observed=True)["analysis_experiment_elapsed_hours"].max() + float(bin_hours)
    ).astype(float).to_dict()

    for group_name in group_names:
        plot_experiment_overview(
            mouse_bins,
            summary_bins,
            group_name=group_name,
            bin_hours=bin_hours,
            output_path=output_dir / f"overview_all_phases_visits_{sanitize_filename_part(group_name)}_{bin_hours}h.png",
            phase_window_table=phase_window_table,
            phase_display_names=phase_display_names,
            spread_metric=spread_metric,
            x_end_hours=group_end_hours.get(group_name),
            plot_style=plot_style,
            origin_clock_hour=mouse_day_start_hour,
            awake_start_clock_hour=mouse_day_start_hour,
            awake_end_clock_hour=awake_end_clock_hour,
        )

    plot_experiment_overview_groups(
        summary_bins,
        output_path=output_dir / f"overview_all_phases_visits_all_groups_{bin_hours}h.png",
        phase_window_table=phase_window_table,
        phase_display_names=phase_display_names,
        spread_metric=spread_metric,
        plot_style=plot_style,
        origin_clock_hour=mouse_day_start_hour,
        awake_start_clock_hour=mouse_day_start_hour,
        awake_end_clock_hour=awake_end_clock_hour,
    )

def render_experiment_metric_plots(
    mouse_bins,
    summary_bins,
    output_dir: Path,
    *,
    file_stub: str,
    title_label: str,
    group_title_label: str,
    ylabel: str,
    bin_hours: int,
    group_names: list[str],
    phase_window_table,
    phase_display_names: dict[int, str],
    spread_metric: str,
    plot_style: str,
    mouse_day_start_hour: float,
    awake_end_clock_hour: float,
) -> None:
    """Render one full-experiment metric in the same visual style as the overview."""

    save_table(mouse_bins, output_dir / f"{file_stub}_mouse_bins_{bin_hours}h.tsv")
    save_table(summary_bins, output_dir / f"{file_stub}_group_summary_{bin_hours}h.tsv")
    group_end_hours = (
        summary_bins.groupby("Group", observed=True)["bin_end_hours"].max().astype(float).to_dict()
    )

    for group_name in group_names:
        plot_experiment_overview(
            mouse_bins,
            summary_bins,
            group_name=group_name,
            bin_hours=bin_hours,
            output_path=output_dir / f"{file_stub}_{sanitize_filename_part(group_name)}_{bin_hours}h.png",
            phase_window_table=phase_window_table,
            phase_display_names=phase_display_names,
            spread_metric=spread_metric,
            x_end_hours=group_end_hours.get(group_name),
            plot_style=plot_style,
            show_individual_labels=False,
            title_label=title_label,
            ylabel=ylabel,
            origin_clock_hour=mouse_day_start_hour,
            awake_start_clock_hour=mouse_day_start_hour,
            awake_end_clock_hour=awake_end_clock_hour,
        )

    plot_experiment_overview_groups(
        summary_bins,
        output_path=output_dir / f"{file_stub}_all_groups_{bin_hours}h.png",
        phase_window_table=phase_window_table,
        phase_display_names=phase_display_names,
        spread_metric=spread_metric,
        plot_style=plot_style,
        title_label=group_title_label,
        ylabel=ylabel,
        origin_clock_hour=mouse_day_start_hour,
        awake_start_clock_hour=mouse_day_start_hour,
        awake_end_clock_hour=awake_end_clock_hour,
    )

def render_phase2_plots(
    visits,
    output_dir: Path,
    *,
    bin_hours: int,
    group_names: list[str],
    secondary_metric: str,
    phase_display_names: dict[int, str],
    plot_style: str,
    spread_metric: str,
    phase_origin_hour: float,
    phase_start_day: int,
    mouse_day_start_hour: float,
    awake_end_clock_hour: float,
) -> None:
    """Create phase-2 adaptation summaries and paired-bar plots."""

    metric_tables = compute_phase2_adaptation_bins(
        visits,
        bin_hours=bin_hours,
        secondary_metric=secondary_metric,
    )
    primary_mouse, primary_summary = metric_tables["visits"]
    secondary_mouse, secondary_summary = metric_tables["drinking_metric"]
    lick_mouse, lick_summary = metric_tables["lick_count"]
    lick_positive_mouse, lick_positive_summary = metric_tables["lick_positive_visits"]

    save_table(primary_mouse, output_dir / f"phase2_visits_mouse_bins_{bin_hours}h.tsv")
    save_table(primary_summary, output_dir / f"phase2_visits_group_summary_{bin_hours}h.tsv")
    save_table(secondary_mouse, output_dir / f"phase2_{secondary_metric}_mouse_bins_{bin_hours}h.tsv")
    save_table(secondary_summary, output_dir / f"phase2_{secondary_metric}_group_summary_{bin_hours}h.tsv")
    save_table(lick_positive_mouse, output_dir / f"phase2_lick_positive_visits_mouse_bins_{bin_hours}h.tsv")
    save_table(lick_positive_summary, output_dir / f"phase2_lick_positive_visits_group_summary_{bin_hours}h.tsv")
    save_table(lick_mouse, output_dir / f"phase2_lick_count_mouse_bins_{bin_hours}h.tsv")
    save_table(lick_summary, output_dir / f"phase2_lick_count_group_summary_{bin_hours}h.tsv")

    secondary_label = "Drinking visits" if secondary_metric == "lick_positive_visits" else "Lick count"
    phase2_end_hours = (
        visits.loc[visits["AnalysisPhaseNumber"].eq(2)]
        .groupby("Group", observed=True)["analysis_phase_elapsed_hours"]
        .max()
        .add(float(bin_hours))
        .astype(float)
        .to_dict()
    )
    for group_name in group_names:
        plot_phase2_adaptation(
            primary_summary,
            secondary_summary,
            group_name=group_name,
            bin_hours=bin_hours,
            output_path=output_dir / f"phase2_visits_vs_{sanitize_filename_part(secondary_metric)}_{sanitize_filename_part(group_name)}_{bin_hours}h.png",
            secondary_label=secondary_label,
            phase_display_name=phase_display_names[2],
            plot_style=plot_style,
            spread_metric=spread_metric,
            x_end_hours=phase2_end_hours.get(group_name),
            origin_clock_hour=phase_origin_hour,
            awake_start_clock_hour=mouse_day_start_hour,
            awake_end_clock_hour=awake_end_clock_hour,
            starting_day=phase_start_day,
        )

def render_phase2_control_plots(
    visits,
    output_dir: Path,
    *,
    bin_hours: int,
    group_names: list[str],
    phase_window_table,
    phase_display_names: dict[int, str],
    plot_style: str,
    spread_metric: str,
    mouse_day_start_hour: float,
    awake_end_clock_hour: float,
) -> None:
    """Create full-experiment control plots for visits versus drinking visits."""

    primary_mouse, primary_summary = compute_experiment_visit_bins(visits, bin_hours=bin_hours)
    drinking_mouse, drinking_summary = compute_experiment_drinking_visit_bins(visits, bin_hours=bin_hours)
    nosepoke_mouse, nosepoke_summary = compute_experiment_nosepoke_count_bins(visits, bin_hours=bin_hours)
    lick_mouse, lick_summary = compute_experiment_lick_count_bins(visits, bin_hours=bin_hours)
    save_table(primary_mouse, output_dir / f"phase2_control_all_phases_visits_mouse_bins_{bin_hours}h.tsv")
    save_table(primary_summary, output_dir / f"phase2_control_all_phases_visits_group_summary_{bin_hours}h.tsv")
    save_table(drinking_mouse, output_dir / f"phase2_control_all_phases_drinking_visits_mouse_bins_{bin_hours}h.tsv")
    save_table(drinking_summary, output_dir / f"phase2_control_all_phases_drinking_visits_group_summary_{bin_hours}h.tsv")

    for group_name in group_names:
        plot_experiment_dual_metric_bars(
            primary_summary,
            drinking_summary,
            group_name=group_name,
            bin_hours=bin_hours,
            output_path=output_dir / f"phase2_control_all_phases_visits_vs_drinking_visits_{sanitize_filename_part(group_name)}_{bin_hours}h.png",
            secondary_label="Drinking visits",
            phase_window_table=phase_window_table,
            phase_display_names=phase_display_names,
            plot_style=plot_style,
            spread_metric=spread_metric,
            origin_clock_hour=mouse_day_start_hour,
            awake_start_clock_hour=mouse_day_start_hour,
            awake_end_clock_hour=awake_end_clock_hour,
        )

    render_experiment_metric_plots(
        nosepoke_mouse,
        nosepoke_summary,
        output_dir,
        file_stub="phase2_control_all_phases_nosepoke_counts",
        title_label="nose-poke counts across all phases",
        group_title_label="Nose-poke counts across all phases by group",
        ylabel="Nose pokes per mouse and bin",
        bin_hours=bin_hours,
        group_names=group_names,
        phase_window_table=phase_window_table,
        phase_display_names=phase_display_names,
        spread_metric=spread_metric,
        plot_style="line",
        mouse_day_start_hour=mouse_day_start_hour,
        awake_end_clock_hour=awake_end_clock_hour,
    )
    render_experiment_metric_plots(
        lick_mouse,
        lick_summary,
        output_dir,
        file_stub="phase2_control_all_phases_lick_counts",
        title_label="lick counts across all phases",
        group_title_label="Lick counts across all phases by group",
        ylabel="Licks per mouse and bin",
        bin_hours=bin_hours,
        group_names=group_names,
        phase_window_table=phase_window_table,
        phase_display_names=phase_display_names,
        spread_metric=spread_metric,
        plot_style="line",
        mouse_day_start_hour=mouse_day_start_hour,
        awake_end_clock_hour=awake_end_clock_hour,
    )

def render_phase_learning_plots(
    visits,
    output_dir: Path,
    *,
    bin_hours: int,
    group_names: list[str],
    phase_display_names: dict[int, str],
    spread_metric: str,
    plot_style: str,
    scheduled_phase_start_hours: dict[int, float],
    mouse_day_start_hour: float,
    awake_end_clock_hour: float,
) -> None:
    """Create phase-3 and phase-4 count and rate plots."""

    phase_group_end_hours = (
        visits.groupby(["Group", "AnalysisPhaseNumber"], observed=True)["analysis_phase_elapsed_hours"].max()
        + float(bin_hours)
    ).astype(float).to_dict()
    phase_end_hours = (
        visits.groupby("AnalysisPhaseNumber", observed=True)["analysis_phase_elapsed_hours"].max() + float(bin_hours)
    ).astype(float).to_dict()

    metric_specs = [
        {
            "file_stub": "correct_corner_visit_rate",
            "success_col": "correct_corner_visit",
            "title_label": "correct-corner visit rate",
            "ylabel": "Correct-corner visit rate [%]",
            "chance_level": 25.0,
            "count_file_stub": "correct_corner_visits_absolute",
            "count_title_label": "correct-corner visits",
            "count_ylabel": "Correct-corner visits per mouse and bin",
        },
        {
            "file_stub": "correct_np_visit_rate",
            "success_col": "correct_np_visit",
            "title_label": "correct NP visit rate",
            "ylabel": "Correct NP visit rate [%]",
            "chance_level": 25.0,
            "count_file_stub": "correct_np_visits_absolute",
            "count_title_label": "correct NP visits",
            "count_ylabel": "Correct NP visits per mouse and bin",
        },
        {
            "file_stub": "rewarded_correct_corner_visit_rate",
            "success_col": "rewarded_correct_corner_visit",
            "title_label": "rewarded correct-corner visit rate",
            "ylabel": "Rewarded correct-corner visit rate [%]",
            "chance_level": 25.0,
            "count_file_stub": "rewarded_correct_corner_visits_absolute",
            "count_title_label": "rewarded correct-corner visits",
            "count_ylabel": "Rewarded correct-corner visits per mouse and bin",
        },
    ]

    for phase_number in (3, 4):
        phase_origin_hour = phase_origin_clock_hour(
            mouse_day_start_hour,
            scheduled_phase_start_hours[phase_number],
        )
        phase_start_day = experiment_day_from_scheduled_start(scheduled_phase_start_hours[phase_number])
        phase_visit_mouse, phase_visit_summary = compute_phase_visit_count_bins(
            visits,
            phase_number=phase_number,
            bin_hours=bin_hours,
        )
        rewarded_count_mouse, rewarded_count_summary = compute_place_learning_count_bins(
            visits,
            phase_number=phase_number,
            bin_hours=bin_hours,
            success_col="rewarded_correct_corner_visit",
        )
        matlab_rate_mouse, matlab_rate_summary = compute_place_learning_rate_bins(
            visits,
            phase_number=phase_number,
            bin_hours=bin_hours,
            success_col="correct_place_visit",
        )

        save_table(
            phase_visit_mouse,
            output_dir / f"phase{phase_number}_all_visit_counts_mouse_bins_{bin_hours}h.tsv",
        )
        save_table(
            phase_visit_summary,
            output_dir / f"phase{phase_number}_all_visit_counts_group_summary_{bin_hours}h.tsv",
        )
        save_table(
            rewarded_count_mouse,
            output_dir / f"phase{phase_number}_rewarded_correct_corner_visits_absolute_mouse_bins_{bin_hours}h.tsv",
        )
        save_table(
            rewarded_count_summary,
            output_dir / f"phase{phase_number}_rewarded_correct_corner_visits_absolute_group_summary_{bin_hours}h.tsv",
        )
        save_table(
            matlab_rate_mouse,
            output_dir / f"phase{phase_number}_matlab_placeerror_only_rate_mouse_bins_{bin_hours}h.tsv",
        )
        save_table(
            matlab_rate_summary,
            output_dir / f"phase{phase_number}_matlab_placeerror_only_rate_group_summary_{bin_hours}h.tsv",
        )

        for metric_spec in metric_specs:
            metric_count_mouse, metric_count_summary = compute_place_learning_count_bins(
                visits,
                phase_number=phase_number,
                bin_hours=bin_hours,
                success_col=metric_spec["success_col"],
            )
            metric_mouse, metric_summary = compute_place_learning_rate_bins(
                visits,
                phase_number=phase_number,
                bin_hours=bin_hours,
                success_col=metric_spec["success_col"],
            )
            save_table(
                metric_mouse,
                output_dir / f"phase{phase_number}_{metric_spec['file_stub']}_mouse_bins_{bin_hours}h.tsv",
            )
            save_table(
                metric_summary,
                output_dir / f"phase{phase_number}_{metric_spec['file_stub']}_group_summary_{bin_hours}h.tsv",
            )
            save_table(
                metric_count_mouse,
                output_dir / f"phase{phase_number}_{metric_spec['count_file_stub']}_mouse_bins_{bin_hours}h.tsv",
            )
            save_table(
                metric_count_summary,
                output_dir / f"phase{phase_number}_{metric_spec['count_file_stub']}_group_summary_{bin_hours}h.tsv",
            )

            for group_name in group_names:
                plot_phase_learning_counts(
                    metric_count_mouse,
                    metric_count_summary,
                    group_name=group_name,
                    phase_number=phase_number,
                    phase_display_name=phase_display_names[phase_number],
                    bin_hours=bin_hours,
                    output_path=output_dir / f"phase{phase_number}_{metric_spec['count_file_stub']}_{sanitize_filename_part(group_name)}_{bin_hours}h.png",
                    spread_metric=spread_metric,
                    title_label=metric_spec["count_title_label"],
                    ylabel=metric_spec["count_ylabel"],
                    x_end_hours=phase_group_end_hours.get((group_name, phase_number)),
                    plot_style=plot_style,
                    origin_clock_hour=phase_origin_hour,
                    awake_start_clock_hour=mouse_day_start_hour,
                    awake_end_clock_hour=awake_end_clock_hour,
                    starting_day=phase_start_day,
                )
                plot_phase_learning_rate(
                    metric_mouse,
                    metric_summary,
                    group_name=group_name,
                    phase_number=phase_number,
                    phase_display_name=phase_display_names[phase_number],
                    bin_hours=bin_hours,
                    output_path=output_dir / f"phase{phase_number}_{metric_spec['file_stub']}_{sanitize_filename_part(group_name)}_{bin_hours}h.png",
                    spread_metric=spread_metric,
                    title_label=metric_spec["title_label"],
                    ylabel=metric_spec["ylabel"],
                    chance_level=metric_spec["chance_level"],
                    x_end_hours=phase_group_end_hours.get((group_name, phase_number)),
                    plot_style=plot_style,
                    origin_clock_hour=phase_origin_hour,
                    awake_start_clock_hour=mouse_day_start_hour,
                    awake_end_clock_hour=awake_end_clock_hour,
                    starting_day=phase_start_day,
                )

            plot_phase_learning_counts_groups(
                metric_count_summary,
                phase_display_name=phase_display_names[phase_number],
                bin_hours=bin_hours,
                output_path=output_dir / f"phase{phase_number}_{metric_spec['count_file_stub']}_all_groups_{bin_hours}h.png",
                spread_metric=spread_metric,
                x_end_hours=phase_end_hours.get(phase_number),
                plot_style=plot_style,
                title_prefix=f"{metric_spec['count_title_label'].capitalize()} across groups",
                ylabel=metric_spec["count_ylabel"],
                origin_clock_hour=phase_origin_hour,
                awake_start_clock_hour=mouse_day_start_hour,
                awake_end_clock_hour=awake_end_clock_hour,
                starting_day=phase_start_day,
            )
            plot_phase_learning_rate_groups(
                metric_summary,
                phase_number=phase_number,
                phase_display_name=phase_display_names[phase_number],
                bin_hours=bin_hours,
                output_path=output_dir / f"phase{phase_number}_{metric_spec['file_stub']}_all_groups_{bin_hours}h.png",
                spread_metric=spread_metric,
                title_label=metric_spec["title_label"],
                ylabel=metric_spec["ylabel"],
                chance_level=metric_spec["chance_level"],
                x_end_hours=phase_end_hours.get(phase_number),
                plot_style=plot_style,
                origin_clock_hour=phase_origin_hour,
                awake_start_clock_hour=mouse_day_start_hour,
                awake_end_clock_hour=awake_end_clock_hour,
                starting_day=phase_start_day,
            )

        plot_phase_learning_counts_groups(
            phase_visit_summary,
            phase_display_name=phase_display_names[phase_number],
            bin_hours=bin_hours,
            output_path=output_dir / f"phase{phase_number}_all_visit_counts_all_groups_{bin_hours}h.png",
            spread_metric=spread_metric,
            x_end_hours=phase_end_hours.get(phase_number),
            plot_style=plot_style,
            title_prefix="All visit counts across groups",
            ylabel="All visits per mouse and bin",
            origin_clock_hour=phase_origin_hour,
            awake_start_clock_hour=mouse_day_start_hour,
            awake_end_clock_hour=awake_end_clock_hour,
            starting_day=phase_start_day,
        )

        if phase_number == 4:
            reversal_rate_tables = compute_phase4_reversal_rate_bins(visits, bin_hours=bin_hours)
            reversal_count_tables = compute_phase4_reversal_count_bins(visits, bin_hours=bin_hours)
            reversal_group_summaries: dict[str, pd.DataFrame] = {}
            for component_name, (component_mouse, component_summary) in reversal_rate_tables.items():
                save_table(
                    component_mouse,
                    output_dir / f"phase4_{component_name}_visit_rate_mouse_bins_{bin_hours}h.tsv",
                )
                save_table(
                    component_summary,
                    output_dir / f"phase4_{component_name}_visit_rate_group_summary_{bin_hours}h.tsv",
                )
                reversal_group_summaries[component_name] = component_summary
            for component_name, (component_mouse, component_summary) in reversal_count_tables.items():
                save_table(
                    component_mouse,
                    output_dir / f"phase4_{component_name}_visits_absolute_mouse_bins_{bin_hours}h.tsv",
                )
                save_table(
                    component_summary,
                    output_dir / f"phase4_{component_name}_visits_absolute_group_summary_{bin_hours}h.tsv",
                )
                chance_level = 25.0
                title_map = {
                    "new_correct_corner": "new correct-corner visit rate",
                    "previous_correct_corner": "previous correct-corner visit rate",
                    "neutral_incorrect_corner_1": "neutral incorrect-corner 1 visit rate",
                    "neutral_incorrect_corner_2": "neutral incorrect-corner 2 visit rate",
                }
                ylabel_map = {
                    "new_correct_corner": "New correct-corner visit rate [%]",
                    "previous_correct_corner": "Previous correct-corner visit rate [%]",
                    "neutral_incorrect_corner_1": "Neutral incorrect-corner 1 visit rate [%]",
                    "neutral_incorrect_corner_2": "Neutral incorrect-corner 2 visit rate [%]",
                }
                count_title_map = {
                    "new_correct_corner": "new correct-corner visits",
                    "previous_correct_corner": "previous correct-corner visits",
                    "neutral_incorrect_corner_1": "neutral incorrect-corner 1 visits",
                    "neutral_incorrect_corner_2": "neutral incorrect-corner 2 visits",
                }
                count_ylabel_map = {
                    "new_correct_corner": "New correct-corner visits per mouse and bin",
                    "previous_correct_corner": "Previous correct-corner visits per mouse and bin",
                    "neutral_incorrect_corner_1": "Neutral incorrect-corner 1 visits per mouse and bin",
                    "neutral_incorrect_corner_2": "Neutral incorrect-corner 2 visits per mouse and bin",
                }
                for group_name in group_names:
                    plot_phase_learning_rate(
                        reversal_rate_tables[component_name][0],
                        reversal_rate_tables[component_name][1],
                        group_name=group_name,
                        phase_number=4,
                        phase_display_name=phase_display_names[4],
                        bin_hours=bin_hours,
                        output_path=output_dir / f"phase4_{component_name}_visit_rate_{sanitize_filename_part(group_name)}_{bin_hours}h.png",
                        spread_metric=spread_metric,
                        title_label=title_map[component_name],
                        ylabel=ylabel_map[component_name],
                        chance_level=chance_level,
                        x_end_hours=phase_group_end_hours.get((group_name, 4)),
                        plot_style=plot_style,
                        origin_clock_hour=phase_origin_hour,
                        awake_start_clock_hour=mouse_day_start_hour,
                        awake_end_clock_hour=awake_end_clock_hour,
                        starting_day=phase_start_day,
                    )
                    plot_phase_learning_counts(
                        component_mouse,
                        component_summary,
                        group_name=group_name,
                        phase_number=4,
                        phase_display_name=phase_display_names[4],
                        bin_hours=bin_hours,
                        output_path=output_dir / f"phase4_{component_name}_visits_absolute_{sanitize_filename_part(group_name)}_{bin_hours}h.png",
                        spread_metric=spread_metric,
                        title_label=count_title_map[component_name],
                        ylabel=count_ylabel_map[component_name],
                        x_end_hours=phase_group_end_hours.get((group_name, 4)),
                        plot_style=plot_style,
                        origin_clock_hour=phase_origin_hour,
                        awake_start_clock_hour=mouse_day_start_hour,
                        awake_end_clock_hour=awake_end_clock_hour,
                        starting_day=phase_start_day,
                    )
                plot_phase_learning_rate_groups(
                    reversal_rate_tables[component_name][1],
                    phase_number=4,
                    phase_display_name=phase_display_names[4],
                    bin_hours=bin_hours,
                    output_path=output_dir / f"phase4_{component_name}_visit_rate_all_groups_{bin_hours}h.png",
                    spread_metric=spread_metric,
                    title_label=title_map[component_name],
                    ylabel=ylabel_map[component_name],
                    chance_level=chance_level,
                    x_end_hours=phase_end_hours.get(4),
                    plot_style=plot_style,
                    origin_clock_hour=phase_origin_hour,
                    awake_start_clock_hour=mouse_day_start_hour,
                    awake_end_clock_hour=awake_end_clock_hour,
                    starting_day=phase_start_day,
                )
                plot_phase_learning_counts_groups(
                    component_summary,
                    phase_display_name=phase_display_names[4],
                    bin_hours=bin_hours,
                    output_path=output_dir / f"phase4_{component_name}_visits_absolute_all_groups_{bin_hours}h.png",
                    spread_metric=spread_metric,
                    x_end_hours=phase_end_hours.get(4),
                    plot_style=plot_style,
                    title_prefix=f"{count_title_map[component_name].capitalize()} across groups",
                    ylabel=count_ylabel_map[component_name],
                    origin_clock_hour=phase_origin_hour,
                    awake_start_clock_hour=mouse_day_start_hour,
                    awake_end_clock_hour=awake_end_clock_hour,
                    starting_day=phase_start_day,
                )
            for group_name in group_names:
                plot_phase4_reversal_components(
                    reversal_group_summaries,
                    group_name=group_name,
                    phase_display_name=phase_display_names[4],
                    bin_hours=bin_hours,
                    output_path=output_dir / f"phase4_reversal_corner_components_{sanitize_filename_part(group_name)}_{bin_hours}h.png",
                    spread_metric=spread_metric,
                    x_end_hours=phase_group_end_hours.get((group_name, 4)),
                    plot_style=plot_style,
                    origin_clock_hour=phase_origin_hour,
                    awake_start_clock_hour=mouse_day_start_hour,
                    awake_end_clock_hour=awake_end_clock_hour,
                    starting_day=phase_start_day,
                )

def render_phase_activity_plot(
    visits,
    output_dir: Path,
    *,
    phase_display_names: dict[int, str],
) -> None:
    """Render the median hourly activity boxplot across phases and groups."""

    mouse_phase_activity = compute_phase_activity_medians(visits)
    activity_stats = compute_phase_activity_statistics(mouse_phase_activity)
    save_table(mouse_phase_activity, output_dir / "phase_activity_median_visits_per_hour_mouse.tsv")
    save_table(activity_stats, output_dir / "phase_activity_median_visits_per_hour_stats.tsv")

    plot_phase_activity_boxplot(
        mouse_phase_activity,
        activity_stats,
        phase_display_names=phase_display_names,
        output_path=output_dir / "phase_activity_median_visits_per_hour_boxplot.png",
    )

def render_phase_segment_plots(
    visits,
    output_dir: Path,
    *,
    phase_display_names: dict[int, str],
    spread_metric: str,
    scheduled_phase_start_hours: dict[int, float],
    mouse_day_start_hour: float,
    awake_end_clock_hour: float,
) -> None:
    """Render awake/sleep-segment learning trajectories for PL and PR across groups."""

    segment_metrics = [
        ("correct_corner_visit", "correct-corner visit rate", "Correct-corner visit rate [%]"),
        ("correct_np_visit", "correct NP visit rate", "Correct NP visit rate [%]"),
        ("rewarded_correct_corner_visit", "rewarded correct-corner visit rate", "Rewarded correct-corner visit rate [%]"),
    ]
    for phase_number in (3, 4):
        phase_origin_hour = phase_origin_clock_hour(mouse_day_start_hour, scheduled_phase_start_hours[phase_number])
        phase_start_day = experiment_day_from_scheduled_start(scheduled_phase_start_hours[phase_number])
        for success_col, title_label, ylabel in segment_metrics:
            mouse_table, summary = compute_phase_segment_rate_tables(
                visits,
                phase_number=phase_number,
                success_col=success_col,
                origin_clock_hour=phase_origin_hour,
                awake_start_clock_hour=mouse_day_start_hour,
                awake_end_clock_hour=awake_end_clock_hour,
                max_days=3,
            )
            metric_stub = success_col.replace("_visit", "")
            save_table(
                mouse_table,
                output_dir / f"phase{phase_number}_{metric_stub}_awake_sleep_segment_rate_mouse.tsv",
            )
            save_table(
                summary,
                output_dir / f"phase{phase_number}_{metric_stub}_awake_sleep_segment_rate_group_summary.tsv",
            )
            plot_phase_segment_rate_groups(
                summary,
                phase_number=phase_number,
                phase_display_name=phase_display_names[phase_number],
                title_label=title_label,
                ylabel=ylabel,
                output_path=output_dir / f"phase{phase_number}_{metric_stub}_awake_sleep_segment_rate_all_groups.png",
                spread_metric=spread_metric,
                starting_day=phase_start_day,
                chance_level=25.0,
            )

def render_awake_day_violin_plots(
    visits,
    output_dir: Path,
    *,
    phase_display_names: dict[int, str],
    scheduled_phase_start_hours: dict[int, float],
    mouse_day_start_hour: float,
    awake_end_clock_hour: float,
    exclude_outliers: bool,
) -> None:
    """Render awake-only daily violin plots plus omnibus, pairwise, and chance statistics."""

    violin_metrics = [
        ("correct_corner_visit", "correct_corner", "correct-corner visit rate", "Correct-corner visit rate [%]", 25.0),
        ("correct_np_visit", "correct_np", "correct NP visit rate", "Correct NP visit rate [%]", 25.0),
        (
            "rewarded_correct_corner_visit",
            "rewarded_correct_corner",
            "rewarded correct-corner visit rate",
            "Rewarded correct-corner visit rate [%]",
            25.0,
        ),
    ]
    for phase_number in (3, 4):
        phase_origin_hour = phase_origin_clock_hour(mouse_day_start_hour, scheduled_phase_start_hours[phase_number])
        for success_col, metric_stub, title_label, ylabel, chance_level in violin_metrics:
            mouse_table, _ = compute_awake_day_rate_tables(
                visits,
                phase_number=phase_number,
                success_col=success_col,
                origin_clock_hour=phase_origin_hour,
                awake_start_clock_hour=mouse_day_start_hour,
                awake_end_clock_hour=awake_end_clock_hour,
                max_days=3,
            )
            mouse_table["PhaseNumber"] = phase_number
            mouse_table = flag_iqr_outliers(
                mouse_table,
                value_col="value",
                group_cols=["phase_day", "Group"],
            )
            omnibus, pairwise, chance = compute_group_day_violin_statistics(
                mouse_table,
                phase_number=phase_number,
                metric_name=metric_stub,
                chance_level=chance_level / 100.0,
                exclude_outliers=exclude_outliers,
            )
            save_table(
                mouse_table,
                output_dir / f"phase{phase_number}_{metric_stub}_awake_day_rate_mouse.tsv",
            )
            save_table(
                omnibus,
                output_dir / f"phase{phase_number}_{metric_stub}_awake_day_rate_omnibus_stats.tsv",
            )
            save_table(
                pairwise,
                output_dir / f"phase{phase_number}_{metric_stub}_awake_day_rate_pairwise_stats.tsv",
            )
            save_table(
                chance,
                output_dir / f"phase{phase_number}_{metric_stub}_awake_day_rate_chance_stats.tsv",
            )
            for phase_day in (1, 2, 3):
                plot_group_day_violin(
                    mouse_table,
                    phase_number=phase_number,
                    phase_display_name=phase_display_names[phase_number],
                    phase_day=phase_day,
                    metric_title=title_label,
                    ylabel=ylabel,
                    pairwise_stats=pairwise,
                    chance_stats=chance,
                    output_path=output_dir / f"phase{phase_number}_{metric_stub}_awake_day{phase_day}_violin.png",
                    outlier_col="is_outlier",
                )

def render_cumulative_role_plots(
    visits,
    output_dir: Path,
    *,
    group_names: list[str],
    plot_style: str,
    phase_display_names: dict[int, str],
    spread_metric: str,
    scheduled_phase_start_hours: dict[int, float],
    mouse_day_start_hour: float,
    awake_end_clock_hour: float,
) -> None:
    """Render event-based cumulative and relative cumulative PL-to-PR corner-role plots."""

    pre_phase_hours = 24.0
    mouse_counts, summary = compute_role_cumulative_curves(visits, pre_phase_hours=pre_phase_hours)
    if mouse_counts.empty or summary.empty:
        return

    save_table(mouse_counts, output_dir / "pl_pr_cumulative_corner_roles_mouse_events.tsv")
    save_table(summary, output_dir / "pl_pr_cumulative_corner_roles_group_summary.tsv")

    phase_window_table = pd.DataFrame(
        [
            {"PhaseNumber": 2, "start_hours": -pre_phase_hours, "end_hours": 0.0},
            {"PhaseNumber": 3, "start_hours": 0.0, "end_hours": 72.0},
            {"PhaseNumber": 4, "start_hours": 72.0, "end_hours": 144.0},
        ]
    )
    origin_clock_hour = phase_origin_clock_hour(mouse_day_start_hour, scheduled_phase_start_hours[3])
    aligned_x_start = -pre_phase_hours + (mouse_day_start_hour - origin_clock_hour)
    cumulative_start_day = max(0, experiment_day_from_scheduled_start(scheduled_phase_start_hours[3]) - 1)
    pl_curve_mouse, _, pl_onset = compute_time_window_learning_curves(
        visits,
        phase_number=3,
        success_col="correct_corner_visit",
    )
    pr_curve_mouse, _, pr_onset = compute_time_window_learning_curves(
        visits,
        phase_number=4,
        success_col="correct_corner_visit",
    )
    absolute_summary = summary.rename(
        columns={
            "mean_value_absolute": "mean_value",
            "sem_value_absolute": "sem_value",
            "std_value_absolute": "std_value",
        }
    )
    relative_summary = summary.rename(
        columns={
            "mean_value_relative": "mean_value",
            "sem_value_relative": "sem_value",
            "std_value_relative": "std_value",
        }
    )
    relative_summary["mean_value_rate"] = relative_summary["mean_value"] * 100.0
    relative_summary["sem_value_rate"] = relative_summary["sem_value"] * 100.0
    relative_summary["std_value_rate"] = relative_summary["std_value"] * 100.0

    for group_name in group_names:
        onset_points: list[dict[str, float | str]] = []
        pl_onset_group = pl_onset.loc[pl_onset["Group"].astype(str).eq(group_name), "onset_hours"].dropna()
        pr_onset_group = pr_onset.loc[pr_onset["Group"].astype(str).eq(group_name), "onset_hours"].dropna()
        if not pl_onset_group.empty:
            onset_points.append(
                {
                    "corner_role": "PL target corner",
                    "x_hours": float(pl_onset_group.median()),
                }
            )
        if not pr_onset_group.empty:
            onset_points.append(
                {
                    "corner_role": "PR target corner",
                    "x_hours": 72.0 + float(pr_onset_group.median()),
                }
            )
        plot_cumulative_role_curves(
            absolute_summary,
            group_name=group_name,
            output_path=output_dir / f"pl_pr_cumulative_corner_roles_absolute_{sanitize_filename_part(group_name)}.png",
            title_label="cumulative corner-role visits from late NPA to PR",
            ylabel="Cumulative visits per mouse",
            value_col="mean_value",
            spread_col=f"{spread_metric}_value",
            plot_style=plot_style,
            phase_window_table=phase_window_table,
            phase_display_names={2: phase_display_names[2], 3: phase_display_names[3], 4: phase_display_names[4]},
            origin_clock_hour=origin_clock_hour,
            awake_start_clock_hour=mouse_day_start_hour,
            awake_end_clock_hour=awake_end_clock_hour,
            x_start_hours=aligned_x_start,
            onset_points=onset_points,
            starting_day=cumulative_start_day,
        )
        plot_cumulative_role_curves(
            relative_summary,
            group_name=group_name,
            output_path=output_dir / f"pl_pr_cumulative_corner_roles_relative_{sanitize_filename_part(group_name)}.png",
            title_label="relative cumulative corner-role visits from late NPA to PR",
            ylabel="Relative cumulative visits [%]",
            value_col="mean_value_rate",
            spread_col="sem_value_rate" if spread_metric == "sem" else "std_value_rate",
            plot_style=plot_style,
            phase_window_table=phase_window_table,
            phase_display_names={2: phase_display_names[2], 3: phase_display_names[3], 4: phase_display_names[4]},
            origin_clock_hour=origin_clock_hour,
            awake_start_clock_hour=mouse_day_start_hour,
            awake_end_clock_hour=awake_end_clock_hour,
            x_start_hours=aligned_x_start,
            onset_points=onset_points,
            starting_day=cumulative_start_day,
        )

def render_experience_learning_plots(
    visits,
    output_dir: Path,
    *,
    phase_display_names: dict[int, str],
    spread_metric: str,
    exclude_outliers: bool,
) -> None:
    """Render experience-dependent learning curves and onset violins for PL and PR."""

    learning_metrics = [
        ("correct_corner_visit", "correct_corner", "correct-corner learning by visit number"),
        ("correct_np_visit", "correct_np", "correct NP learning by visit number"),
        ("rewarded_correct_corner_visit", "rewarded_correct_corner", "rewarded correct-corner learning by visit number"),
    ]
    for phase_number in (3, 4):
        for success_col, metric_stub, title_label in learning_metrics:
            curve_mouse, curve_summary, onset_visits = compute_visit_window_learning_curves(
                visits,
                phase_number=phase_number,
                success_col=success_col,
            )
            save_table(
                curve_mouse,
                output_dir / f"phase{phase_number}_{metric_stub}_experience_learning_mouse.tsv",
            )
            save_table(
                curve_summary,
                output_dir / f"phase{phase_number}_{metric_stub}_experience_learning_group_summary.tsv",
            )
            save_table(
                onset_visits,
                output_dir / f"phase{phase_number}_{metric_stub}_experience_learning_onset.tsv",
            )
            onset_visits = flag_iqr_outliers(
                onset_visits,
                value_col="onset_visit",
                group_cols=["Group"],
            )
            save_table(
                onset_visits,
                output_dir / f"phase{phase_number}_{metric_stub}_experience_learning_onset_with_outliers.tsv",
            )
            onset_omnibus, onset_pairwise = compute_onset_group_statistics(
                onset_visits,
                onset_col="onset_visit",
                phase_number=phase_number,
                metric_name=metric_stub,
                exclude_outliers=exclude_outliers,
            )
            save_table(
                onset_omnibus,
                output_dir / f"phase{phase_number}_{metric_stub}_experience_learning_onset_omnibus_stats.tsv",
            )
            save_table(
                onset_pairwise,
                output_dir / f"phase{phase_number}_{metric_stub}_experience_learning_onset_pairwise_stats.tsv",
            )
            plot_visit_learning_curve_groups(
                curve_summary,
                phase_display_name=phase_display_names[phase_number],
                title_label=title_label,
                ylabel="Success probability [%]",
                output_path=output_dir / f"phase{phase_number}_{metric_stub}_experience_learning_all_groups.png",
                spread_metric=spread_metric,
            )
            plot_onset_violin(
                onset_visits,
                onset_col="onset_visit",
                phase_display_name=phase_display_names[phase_number],
                title_label=f"{title_label} onset",
                ylabel="Learning onset [visit number]",
                output_path=output_dir / f"phase{phase_number}_{metric_stub}_experience_learning_onset_violin.png",
                pairwise_stats=onset_pairwise,
                outlier_col="is_outlier",
            )

            time_curve_mouse, time_curve_summary, onset_hours = compute_time_window_learning_curves(
                visits,
                phase_number=phase_number,
                success_col=success_col,
            )
            save_table(
                time_curve_mouse,
                output_dir / f"phase{phase_number}_{metric_stub}_clocktime_learning_mouse.tsv",
            )
            save_table(
                time_curve_summary,
                output_dir / f"phase{phase_number}_{metric_stub}_clocktime_learning_group_summary.tsv",
            )
            save_table(
                onset_hours,
                output_dir / f"phase{phase_number}_{metric_stub}_clocktime_learning_onset.tsv",
            )
            onset_hours = flag_iqr_outliers(
                onset_hours,
                value_col="onset_hours",
                group_cols=["Group"],
            )
            save_table(
                onset_hours,
                output_dir / f"phase{phase_number}_{metric_stub}_clocktime_learning_onset_with_outliers.tsv",
            )
            onset_hour_omnibus, onset_hour_pairwise = compute_onset_group_statistics(
                onset_hours,
                onset_col="onset_hours",
                phase_number=phase_number,
                metric_name=metric_stub,
                exclude_outliers=exclude_outliers,
            )
            save_table(
                onset_hour_omnibus,
                output_dir / f"phase{phase_number}_{metric_stub}_clocktime_learning_onset_omnibus_stats.tsv",
            )
            save_table(
                onset_hour_pairwise,
                output_dir / f"phase{phase_number}_{metric_stub}_clocktime_learning_onset_pairwise_stats.tsv",
            )
            plot_onset_violin(
                onset_hours,
                onset_col="onset_hours",
                phase_display_name=phase_display_names[phase_number],
                title_label=f"{title_label} onset",
                ylabel="Learning onset [hours]",
                output_path=output_dir / f"phase{phase_number}_{metric_stub}_clocktime_learning_onset_violin.png",
                pairwise_stats=onset_hour_pairwise,
                outlier_col="is_outlier",
            )

def compute_rate_threshold_onset_table(
    visits: pd.DataFrame,
    *,
    phase_number: int,
    success_col: str,
    bin_hours: int,
    threshold_pct: float,
) -> pd.DataFrame:
    """Return the first binned hour at which each mouse exceeds a rate threshold."""

    mouse_bins, _ = compute_place_learning_rate_bins(
        visits,
        phase_number=phase_number,
        bin_hours=bin_hours,
        success_col=success_col,
    )
    if mouse_bins.empty:
        return pd.DataFrame()

    threshold = float(threshold_pct) / 100.0
    onset_rows: list[dict[str, object]] = []
    for (group_name, et, et_label, sex), mouse_data in mouse_bins.groupby(
        ["Group", "ET", "ETLabel", "SEX"], observed=True
    ):
        valid = mouse_data.loc[mouse_data["all_visits"].gt(0) & mouse_data["value"].gt(threshold)].copy()
        onset_hour = float(valid["bin_start_hours"].min()) if not valid.empty else pd.NA
        onset_rows.append(
            {
                "Group": str(group_name),
                "ET": et,
                "ETLabel": str(et_label),
                "SEX": sex,
                "onset_hours": onset_hour,
            }
        )
    return pd.DataFrame(onset_rows)

def render_rate_threshold_onset_plots(
    visits: pd.DataFrame,
    output_dir: Path,
    *,
    phase_display_names: dict[int, str],
    threshold_pcts: list[float] | tuple[float, ...],
    bin_hours: int,
    exclude_outliers: bool,
) -> None:
    """Render violin plots for first threshold crossing of binned learning rates."""

    def _shared_onset_ylim(
        flagged_tables: list[pd.DataFrame],
        pairwise_tables: list[pd.DataFrame],
    ) -> tuple[float, float]:
        """Compute one shared onset y-limit across matching PL/PR panels."""

        non_empty = [frame for frame in flagged_tables if not frame.empty and frame["onset_hours"].notna().any()]
        if not non_empty:
            return (0.0, 1.0)
        combined = pd.concat(non_empty, ignore_index=True)
        y_max = float(combined["onset_hours"].max())
        y_min = float(min(0.0, combined["onset_hours"].min()))
        data_span = max(1.0, y_max - y_min)
        significant_pair_count = 0
        for pairwise in pairwise_tables:
            if pairwise is None or pairwise.empty or "p_value" not in pairwise.columns:
                continue
            significant_pair_count = max(significant_pair_count, int(pairwise["p_value"].lt(0.05).sum()))
        if significant_pair_count > 0:
            base_y = y_max + data_span * 0.12
            step_y = data_span * 0.10
            top = base_y + max(1, significant_pair_count) * step_y + data_span * 0.08
        else:
            top = max(y_max * 1.05, y_max + data_span * 0.08, 1.0)
        return (0.0, float(top))

    metric_specs = [
        ("correct_corner_visit", "correct_corner", "correct-corner visit rate"),
        ("correct_np_visit", "correct_np", "correct NP visit rate"),
        ("rewarded_correct_corner_visit", "rewarded_correct_corner", "rewarded correct-corner visit rate"),
    ]
    for success_col, metric_stub, title_label in metric_specs:
        for threshold_pct in threshold_pcts:
            phase_payloads: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
            flagged_tables: list[pd.DataFrame] = []
            pairwise_tables: list[pd.DataFrame] = []
            for phase_number in (3, 4):
                threshold_tag = f"{int(threshold_pct)}pct"
                onset_table = compute_rate_threshold_onset_table(
                    visits,
                    phase_number=phase_number,
                    success_col=success_col,
                    bin_hours=bin_hours,
                    threshold_pct=threshold_pct,
                )
                if onset_table.empty:
                    continue
                save_table(
                    onset_table,
                    output_dir / f"phase{phase_number}_{metric_stub}_{bin_hours}h_threshold_onset_{threshold_tag}_mouse.tsv",
                )
                flagged = flag_iqr_outliers(
                    onset_table,
                    value_col="onset_hours",
                    group_cols=["Group"],
                )
                save_table(
                    flagged,
                    output_dir / f"phase{phase_number}_{metric_stub}_{bin_hours}h_threshold_onset_{threshold_tag}_mouse_with_outliers.tsv",
                )
                omnibus, pairwise = compute_onset_group_statistics(
                    flagged,
                    onset_col="onset_hours",
                    phase_number=phase_number,
                    metric_name=f"{metric_stub}_{bin_hours}h_threshold_onset_gt_{threshold_tag}",
                    exclude_outliers=exclude_outliers,
                )
                save_table(
                    omnibus,
                    output_dir / f"phase{phase_number}_{metric_stub}_{bin_hours}h_threshold_onset_{threshold_tag}_omnibus_stats.tsv",
                )
                save_table(
                    pairwise,
                    output_dir / f"phase{phase_number}_{metric_stub}_{bin_hours}h_threshold_onset_{threshold_tag}_pairwise_stats.tsv",
                )
                phase_payloads[phase_number] = (flagged, pairwise)
                flagged_tables.append(flagged)
                pairwise_tables.append(pairwise)
            shared_ylim = _shared_onset_ylim(flagged_tables, pairwise_tables)
            for phase_number in (3, 4):
                if phase_number not in phase_payloads:
                    continue
                flagged, pairwise = phase_payloads[phase_number]
                plot_onset_violin(
                    flagged,
                    onset_col="onset_hours",
                    phase_display_name=phase_display_names[phase_number],
                    title_label=f"{title_label} first exceeds {int(threshold_pct)}%",
                    ylabel="Threshold onset [hours]",
                    output_path=output_dir / f"phase{phase_number}_{metric_stub}_{bin_hours}h_threshold_onset_{threshold_tag}_violin.png",
                    pairwise_stats=pairwise,
                    outlier_col="is_outlier",
                    y_limits=shared_ylim,
                )

def render_count_model_and_responder_analyses(
    visits: pd.DataFrame,
    output_dir: Path,
    *,
    phase_display_names: dict[int, str],
    scheduled_phase_start_hours: dict[int, float],
    mouse_day_start_hour: float,
    awake_end_clock_hour: float,
    threshold_pcts: list[float] | tuple[float, ...],
    threshold_bin_hours: int,
    responder_horizons_hours: list[float] | tuple[float, ...],
    glm_first_hours: float,
) -> None:
    """Render count-based model summaries and criterion-reached responder tables.

    These analyses complement the existing percent-based plots with models that
    operate directly on success/failure counts and with mouse-level responder
    labels derived from threshold-onset times.
    """

    metric_specs = [
        ("correct_corner_visit", "correct_corner", "correct-corner visit rate"),
        ("correct_np_visit", "correct_np", "correct NP visit rate"),
        ("rewarded_correct_corner_visit", "rewarded_correct_corner", "rewarded correct-corner visit rate"),
    ]

    for phase_number in (3, 4):
        phase_origin_hour = phase_origin_clock_hour(mouse_day_start_hour, scheduled_phase_start_hours[phase_number])
        for success_col, metric_stub, title_label in metric_specs:
            awake_mouse, _ = compute_awake_day_rate_tables(
                visits,
                phase_number=phase_number,
                success_col=success_col,
                origin_clock_hour=phase_origin_hour,
                awake_start_clock_hour=mouse_day_start_hour,
                awake_end_clock_hour=awake_end_clock_hour,
                max_days=3,
            )
            if not awake_mouse.empty:
                awake_mouse["PhaseNumber"] = phase_number
                save_table(
                    awake_mouse,
                    output_dir / f"phase{phase_number}_{metric_stub}_awake_day_count_model_mouse.tsv",
                )
                omnibus, pairwise = compute_binomial_glm_group_statistics(
                    awake_mouse,
                    phase_number=phase_number,
                    metric_name=f"{metric_stub}_awake_day_count_model",
                    success_col="correct_visits",
                    total_col="all_visits",
                    subset_col="phase_day",
                    subset_label="phase_day",
                )
                save_table(
                    omnibus,
                    output_dir / f"phase{phase_number}_{metric_stub}_awake_day_count_model_omnibus_stats.tsv",
                )
                save_table(
                    pairwise,
                    output_dir / f"phase{phase_number}_{metric_stub}_awake_day_count_model_pairwise_stats.tsv",
                )

            first_hours_table = compute_first_hours_rate_table(
                visits,
                phase_number=phase_number,
                success_col=success_col,
                first_hours=glm_first_hours,
            )
            if not first_hours_table.empty:
                save_table(
                    first_hours_table,
                    output_dir / f"phase{phase_number}_{metric_stub}_first{int(glm_first_hours)}h_count_model_mouse.tsv",
                )
                first_omnibus, first_pairwise = compute_binomial_glm_group_statistics(
                    first_hours_table,
                    phase_number=phase_number,
                    metric_name=f"{metric_stub}_first{int(glm_first_hours)}h_count_model",
                    success_col="correct_visits",
                    total_col="all_visits",
                )
                save_table(
                    first_omnibus,
                    output_dir / f"phase{phase_number}_{metric_stub}_first{int(glm_first_hours)}h_count_model_omnibus_stats.tsv",
                )
                save_table(
                    first_pairwise,
                    output_dir / f"phase{phase_number}_{metric_stub}_first{int(glm_first_hours)}h_count_model_pairwise_stats.tsv",
                )

            for threshold_pct in threshold_pcts:
                threshold_tag = f"{int(threshold_pct)}pct"
                onset_table = compute_rate_threshold_onset_table(
                    visits,
                    phase_number=phase_number,
                    success_col=success_col,
                    bin_hours=threshold_bin_hours,
                    threshold_pct=threshold_pct,
                )
                if onset_table.empty:
                    continue
                responder_table = compute_threshold_responder_table(
                    onset_table,
                    phase_number=phase_number,
                    threshold_pct=threshold_pct,
                    horizons_hours=tuple(float(value) for value in responder_horizons_hours),
                )
                if responder_table.empty:
                    continue
                save_table(
                    responder_table,
                    output_dir / f"phase{phase_number}_{metric_stub}_{threshold_bin_hours}h_threshold_responder_{threshold_tag}_mouse.tsv",
                )
                responder_summary, responder_omnibus, responder_pairwise = compute_responder_group_statistics(
                    responder_table,
                    phase_number=phase_number,
                    metric_name=f"{metric_stub}_{threshold_bin_hours}h_threshold_responder_{threshold_tag}",
                )
                save_table(
                    responder_summary,
                    output_dir / f"phase{phase_number}_{metric_stub}_{threshold_bin_hours}h_threshold_responder_{threshold_tag}_summary.tsv",
                )
                save_table(
                    responder_omnibus,
                    output_dir / f"phase{phase_number}_{metric_stub}_{threshold_bin_hours}h_threshold_responder_{threshold_tag}_omnibus_stats.tsv",
                )
                save_table(
                    responder_pairwise,
                    output_dir / f"phase{phase_number}_{metric_stub}_{threshold_bin_hours}h_threshold_responder_{threshold_tag}_pairwise_stats.tsv",
                )

def compute_auc_above_chance_table(
    visits: pd.DataFrame,
    *,
    phase_number: int,
    success_col: str,
    bin_hours: int,
    chance_level_pct: float = 25.0,
    first_hours: float = 24.0,
) -> pd.DataFrame:
    """Compute per-mouse AUC above chance from binned phase rates in the early phase."""

    mouse_bins, _ = compute_place_learning_rate_bins(
        visits,
        phase_number=phase_number,
        bin_hours=bin_hours,
        success_col=success_col,
    )
    if mouse_bins.empty:
        return pd.DataFrame()

    early = mouse_bins.loc[mouse_bins["bin_start_hours"].lt(float(first_hours))].copy()
    if early.empty:
        return pd.DataFrame()
    chance = float(chance_level_pct) / 100.0
    early["auc_component"] = np.clip(early["value"].fillna(0.0) - chance, a_min=0.0, a_max=None) * float(bin_hours)
    summary = (
        early.groupby(["Group", "ET", "ETLabel", "SEX"], observed=True)
        .agg(
            auc_above_chance=("auc_component", "sum"),
            contributing_bins=("all_visits", lambda values: int(np.sum(pd.Series(values).gt(0)))),
        )
        .reset_index()
    )
    return summary

def render_derived_metric_plots(
    visits: pd.DataFrame,
    output_dir: Path,
    *,
    phase_display_names: dict[int, str],
    scheduled_phase_start_hours: dict[int, float],
    mouse_day_start_hour: float,
    awake_end_clock_hour: float,
    exclude_outliers: bool,
) -> None:
    """Render completion-efficiency, perseveration-index, and early-AUC summary plots."""

    ratio_visits = visits.copy()
    ratio_visits["new_or_previous_correct_corner_visit"] = (
        ratio_visits["correct_corner_visit"].fillna(False).astype(bool)
        | ratio_visits["previous_correct_corner_visit"].fillna(False).astype(bool)
    )

    derived_specs = [
        {
            "phase_number": 3,
            "metric_stub": "completion_efficiency",
            "title_label": "completion efficiency",
            "ylabel": "Rewarded correct / correct NP [%]",
            "numerator_col": "rewarded_correct_corner_visit",
            "denominator_col": "correct_np_visit",
            "reference_line": None,
            "pseudocount": 0.0,
        },
        {
            "phase_number": 4,
            "metric_stub": "completion_efficiency",
            "title_label": "completion efficiency",
            "ylabel": "Rewarded correct / correct NP [%]",
            "numerator_col": "rewarded_correct_corner_visit",
            "denominator_col": "correct_np_visit",
            "reference_line": None,
            "pseudocount": 0.0,
        },
        {
            "phase_number": 4,
            "metric_stub": "perseveration_index",
            "title_label": "perseveration index (new / previous; higher = better)",
            "ylabel": "Perseveration index\n(new / previous; higher = better)",
            "numerator_col": "correct_corner_visit",
            "denominator_col": "previous_correct_corner_visit",
            "reference_line": 1.0,
            "pseudocount": 0.5,
            "value_scale": 1.0,
            "format_as_percent": False,
        },
        {
            "phase_number": 4,
            "metric_stub": "reversal_preference_index",
            "title_label": "reversal preference index (new / (new + previous); higher = better)",
            "ylabel": "Reversal preference index\n(new / (new + previous); higher = better)",
            "numerator_col": "correct_corner_visit",
            "denominator_col": "new_or_previous_correct_corner_visit",
            "reference_line": 0.5,
            "pseudocount": 0.0,
            "value_scale": 1.0,
            "format_as_percent": False,
        },
    ]
    for spec in derived_specs:
        phase_number = int(spec["phase_number"])
        phase_origin_hour = phase_origin_clock_hour(mouse_day_start_hour, scheduled_phase_start_hours[phase_number])
        mouse_table, summary = compute_awake_day_ratio_tables(
            ratio_visits,
            phase_number=phase_number,
            numerator_col=str(spec["numerator_col"]),
            denominator_col=str(spec["denominator_col"]),
            origin_clock_hour=phase_origin_hour,
            awake_start_clock_hour=mouse_day_start_hour,
            awake_end_clock_hour=awake_end_clock_hour,
            max_days=3,
            pseudocount=float(spec["pseudocount"]),
        )
        if mouse_table.empty:
            continue
        mouse_table["PhaseNumber"] = phase_number
        save_table(
            mouse_table,
            output_dir / f"phase{phase_number}_{spec['metric_stub']}_awake_day_rate_mouse.tsv",
        )
        flagged = flag_iqr_outliers(
            mouse_table,
            value_col="value",
            group_cols=["Group", "phase_day"],
        )
        save_table(
            flagged,
            output_dir / f"phase{phase_number}_{spec['metric_stub']}_awake_day_rate_mouse_with_outliers.tsv",
        )
        omnibus, pairwise, _ = compute_group_day_violin_statistics(
            flagged,
            phase_number=phase_number,
            metric_name=str(spec["metric_stub"]),
            chance_level=0.25,
            exclude_outliers=exclude_outliers,
        )
        save_table(
            omnibus,
            output_dir / f"phase{phase_number}_{spec['metric_stub']}_awake_day_rate_omnibus_stats.tsv",
        )
        save_table(
            pairwise,
            output_dir / f"phase{phase_number}_{spec['metric_stub']}_awake_day_rate_pairwise_stats.tsv",
        )
        for phase_day in (1, 2, 3):
            plot_group_day_violin(
                flagged,
                phase_number=phase_number,
                phase_display_name=phase_display_names[phase_number],
                phase_day=phase_day,
                metric_title=str(spec["title_label"]),
                ylabel=str(spec["ylabel"]),
                pairwise_stats=pairwise,
                chance_stats=None,
                output_path=output_dir / f"phase{phase_number}_{spec['metric_stub']}_awake_day{phase_day}_violin.png",
                reference_line=spec["reference_line"],
                value_scale=float(spec.get("value_scale", 100.0)),
                format_as_percent=bool(spec.get("format_as_percent", True)),
            )

    auc_specs = [
        ("correct_corner_visit", "correct_corner", "correct-corner"),
        ("correct_np_visit", "correct_np", "correct NP"),
        ("rewarded_correct_corner_visit", "rewarded_correct_corner", "rewarded correct-corner"),
    ]
    for phase_number in (3, 4):
        for success_col, metric_stub, title_label in auc_specs:
            auc_table = compute_auc_above_chance_table(
                visits,
                phase_number=phase_number,
                success_col=success_col,
                bin_hours=1,
                chance_level_pct=25.0,
                first_hours=24.0,
            )
            if auc_table.empty:
                continue
            save_table(
                auc_table,
                output_dir / f"phase{phase_number}_{metric_stub}_auc_above_chance_first24h_mouse.tsv",
            )
            flagged = flag_iqr_outliers(
                auc_table,
                value_col="auc_above_chance",
                group_cols=["Group"],
            )
            save_table(
                flagged,
                output_dir / f"phase{phase_number}_{metric_stub}_auc_above_chance_first24h_mouse_with_outliers.tsv",
            )
            omnibus, pairwise = compute_onset_group_statistics(
                flagged.rename(columns={"auc_above_chance": "onset_hours"}),
                onset_col="onset_hours",
                phase_number=phase_number,
                metric_name=f"{metric_stub}_auc_above_chance_first24h",
                exclude_outliers=exclude_outliers,
            )
            save_table(
                omnibus,
                output_dir / f"phase{phase_number}_{metric_stub}_auc_above_chance_first24h_omnibus_stats.tsv",
            )
            save_table(
                pairwise,
                output_dir / f"phase{phase_number}_{metric_stub}_auc_above_chance_first24h_pairwise_stats.tsv",
            )
            plot_onset_violin(
                flagged.rename(columns={"auc_above_chance": "onset_hours"}),
                onset_col="onset_hours",
                phase_display_name=phase_display_names[phase_number],
                title_label=f"{title_label} AUC above chance in first 24 h",
                ylabel="AUC above chance [%*h]",
                output_path=output_dir / f"phase{phase_number}_{metric_stub}_auc_above_chance_first24h_violin.png",
                pairwise_stats=pairwise,
                outlier_col="is_outlier",
                reference_line=0.0,
            )

def run_analysis(
    *,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    results_subdir: Path = Path("results"),
    bin_hours: list[int] | tuple[int, ...] = (1, 2),
    phase2_secondary_metric: str = "lick_positive_visits",
    spread_metric: str = "sem",
    plot_style: str = "line",
    phase2_plot_style: str = DEFAULT_PHASE2_PLOT_STYLE,
    phase_max_hours: dict[int, float] | None = None,
    phase_name_map: dict[str, int] | None = None,
    optional_phase_names: list[str] | tuple[str, ...] | set[str] | None = None,
    drop_unmatched_visits: bool = False,
    excluded_groups: list[str] | None = None,
    group_renames: dict[str, str] | None = None,
    group_colors: dict[str, str] | None = None,
    figure_size_cm: dict[str, tuple[float, float]] | None = None,
    mouse_day_start_hour: float = DEFAULT_MOUSE_DAY_START_HOUR,
    awake_duration_hours: float = DEFAULT_AWAKE_DURATION_HOURS,
    experiment_day0_start_hour: float | None = None,
    schedule_anchor_phase_number: int | None = None,
    scheduled_phase_start_hours: dict[int, float] | None = None,
    base_font_size: float = 10.0,
    exclude_violin_outliers: bool = True,
    rate_threshold_pcts: list[float] | tuple[float, ...] = (50.0, 60.0, 70.0, 80.0),
    threshold_onset_bin_hours: int = 1,
    responder_horizons_hours: list[float] | tuple[float, ...] = (24.0, 48.0, 72.0),
    binomial_model_first_hours: float = 24.0,
) -> Path:
    """Run the 4-month cohort pipeline from a normal Python function call.

    This function mirrors the CLI configuration but can be imported and called
    directly from any user script without argument parsing.
    """

    merged_phase_limits = DEFAULT_PHASE_MAX_HOURS.copy()
    if phase_max_hours:
        merged_phase_limits.update(phase_max_hours)
    merged_scheduled_phase_starts = DEFAULT_SCHEDULED_PHASE_START_HOURS.copy()
    if scheduled_phase_start_hours:
        merged_scheduled_phase_starts.update(scheduled_phase_start_hours)
    selected_excluded_groups = DEFAULT_EXCLUDED_GROUPS if excluded_groups is None else list(excluded_groups)
    selected_group_renames = DEFAULT_GROUP_RENAMES.copy()
    if group_renames:
        selected_group_renames.update(group_renames)
    selected_group_colors = resolved_group_colors(
        group_renames=selected_group_renames,
        group_colors=group_colors,
    )
    configure_plot_style(font_size=base_font_size, font_family="Arial")
    set_group_colors(selected_group_colors)
    set_figure_size_presets(figure_size_cm)
    awake_start_clock_hour, awake_end_clock_hour = active_period_bounds(
        mouse_day_start_hour,
        awake_duration_hours,
    )
    output_root = resolve_output_root(dataset_root, results_subdir)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading cohort data from {dataset_root}...")
    cohort = load_cohort_data(
        dataset_root,
        phase_name_map=phase_name_map,
        optional_phase_names=optional_phase_names,
        drop_unmatched_visits=drop_unmatched_visits,
    )
    render_mouse_age_at_phase1_start_plot(
        cohort.metadata,
        cohort.phase_manifest,
        output_root,
        group_renames=selected_group_renames,
    )
    aligned_visits = attach_analysis_time_columns(
        cohort.visits,
        cohort.phase_manifest,
        scheduled_phase_start_hours=merged_scheduled_phase_starts,
        mouse_day_start_hour=mouse_day_start_hour,
        experiment_day0_start_hour=experiment_day0_start_hour,
        schedule_anchor_phase_number=schedule_anchor_phase_number,
    )
    selected_visits, selected_metadata, selected_nosepokes = apply_group_preferences(
        aligned_visits,
        cohort.metadata,
        cohort.nosepokes,
        excluded_groups=selected_excluded_groups,
        group_renames=selected_group_renames,
    )
    limit_table = build_phase_time_limit_table(cohort.phase_manifest)
    save_table(selected_metadata, output_root / "mouse_metadata.tsv")
    save_table(cohort.phase_manifest, output_root / "phase_manifest.tsv")
    save_table(limit_table, output_root / "phase_time_limit_recommendations.tsv")
    save_table(
        pd.DataFrame(
            {
                "Setting": [
                    "mouse_day_start_hour",
                    "awake_duration_hours",
                    "experiment_day0_start_hour",
                    "schedule_anchor_phase_number",
                    "phase2_plot_style",
                        "exclude_groups",
                        "phase_name_map",
                        "optional_phase_names",
                        "drop_unmatched_visits",
                    "group_rename_mapping",
                    "group_color_mapping",
                    "figure_size_cm",
                    "base_font_size",
                    "exclude_violin_outliers",
                    "rate_threshold_pcts",
                    "threshold_onset_bin_hours",
                    "responder_horizons_hours",
                    "binomial_model_first_hours",
                ],
                "Value": [
                    mouse_day_start_hour,
                    awake_duration_hours,
                        "" if experiment_day0_start_hour is None else experiment_day0_start_hour,
                        "" if schedule_anchor_phase_number is None else schedule_anchor_phase_number,
                        phase2_plot_style,
                        ",".join(selected_excluded_groups) if selected_excluded_groups else "",
                        ";".join(f"{key}={value}" for key, value in sorted((phase_name_map or {}).items())),
                        ",".join(sorted(str(name) for name in (optional_phase_names or []))),
                        drop_unmatched_visits,
                        ";".join(f"{key}={value}" for key, value in selected_group_renames.items()),
                        ";".join(f"{key}={value}" for key, value in selected_group_colors.items()),
                        ";".join(f"{key}={value}" for key, value in (figure_size_cm or {}).items()),
                        base_font_size,
                        exclude_violin_outliers,
                        ",".join(str(value) for value in rate_threshold_pcts),
                        threshold_onset_bin_hours,
                        ",".join(str(value) for value in responder_horizons_hours),
                        binomial_model_first_hours,
                    ],
            }
        ),
        output_root / "analysis_settings.tsv",
    )
    save_table(
        pd.DataFrame(
            {
                "PhaseNumber": list(sorted(merged_scheduled_phase_starts)),
                "ScheduledStartHours": [
                    merged_scheduled_phase_starts[key] for key in sorted(merged_scheduled_phase_starts)
                ],
            }
        ),
        output_root / "scheduled_phase_start_hours.tsv",
    )
    print(f"done. Output root directory: {output_root}")
    
    print("Filtering visits by phase time limits...")
    filtered_visits = filter_visits_by_phase_limits(selected_visits, merged_phase_limits)
    filtered_visits.to_csv(csv_output_path(output_root / "merged_visits.tsv.gz"), sep="\t", index=False, compression="gzip")
    selected_nosepokes.to_csv(csv_output_path(output_root / "merged_nosepokes.tsv.gz"), sep="\t", index=False, compression="gzip")
    print("done.")

    print("Rendering analysis plots...")
    group_names = ordered_group_names(filtered_visits)
    phase_window_table = build_analysis_phase_window_table(filtered_visits, merged_scheduled_phase_starts)
    save_table(phase_window_table, output_root / "analysis_phase_windows.tsv")
    suggested_limits = suggest_common_phase_limits(cohort.phase_manifest)
    save_table(
        pd.DataFrame(
            {
                "PhaseNumber": list(sorted(suggested_limits)),
                "SuggestedCommonLimitHours": [suggested_limits[key] for key in sorted(suggested_limits)],
            }
        ),
        output_root / "suggested_common_phase_limits.tsv",
    )
    print(f"done. Rendered plots will be saved to {output_root} in subdirectories by binning parameter.")
    
    print("Rendering phase-aligned learning curves and visit counts across groups and binning parameters...")
    for current_bin_hours in sorted(set(bin_hours)):
        bin_dir = output_root / f"{current_bin_hours}h_bins"
        render_overview_plots(
            filtered_visits,
            bin_dir,
            bin_hours=current_bin_hours,
            group_names=group_names,
            phase_window_table=phase_window_table,
            phase_display_names=DEFAULT_PHASE_DISPLAY_NAMES,
            spread_metric=spread_metric,
            plot_style=plot_style,
            mouse_day_start_hour=mouse_day_start_hour,
            awake_end_clock_hour=awake_end_clock_hour,
        )
        render_phase2_plots(
            filtered_visits,
            bin_dir,
            bin_hours=current_bin_hours,
            group_names=group_names,
            secondary_metric=phase2_secondary_metric,
            phase_display_names=DEFAULT_PHASE_DISPLAY_NAMES,
            plot_style=phase2_plot_style,
            spread_metric=spread_metric,
            phase_origin_hour=phase_origin_clock_hour(
                mouse_day_start_hour,
                merged_scheduled_phase_starts[2],
            ),
            phase_start_day=experiment_day_from_scheduled_start(merged_scheduled_phase_starts[2]),
            mouse_day_start_hour=awake_start_clock_hour,
            awake_end_clock_hour=awake_end_clock_hour,
        )
        render_phase2_control_plots(
            filtered_visits,
            bin_dir,
            bin_hours=current_bin_hours,
            group_names=group_names,
            phase_window_table=phase_window_table,
            phase_display_names=DEFAULT_PHASE_DISPLAY_NAMES,
            plot_style=phase2_plot_style,
            spread_metric=spread_metric,
            mouse_day_start_hour=mouse_day_start_hour,
            awake_end_clock_hour=awake_end_clock_hour,
        )
        render_phase_learning_plots(
            filtered_visits,
            bin_dir,
            bin_hours=current_bin_hours,
            group_names=group_names,
            phase_display_names=DEFAULT_PHASE_DISPLAY_NAMES,
            spread_metric=spread_metric,
            plot_style=plot_style,
            scheduled_phase_start_hours=merged_scheduled_phase_starts,
            mouse_day_start_hour=mouse_day_start_hour,
            awake_end_clock_hour=awake_end_clock_hour,
        )
    print("Rendering phase activity, segment, awake-day, experience-learning, and cumulative role plots...")
    render_phase_activity_plot(
        filtered_visits,
        output_root,
        phase_display_names=DEFAULT_PHASE_DISPLAY_NAMES,
    )
    print("done. Rendering awake/sleep segment plots...")
    render_phase_segment_plots(
        filtered_visits,
        output_root,
        phase_display_names=DEFAULT_PHASE_DISPLAY_NAMES,
        spread_metric=spread_metric,
        scheduled_phase_start_hours=merged_scheduled_phase_starts,
        mouse_day_start_hour=mouse_day_start_hour,
        awake_end_clock_hour=awake_end_clock_hour,
    )
    print("done. Rendering awake-day violin plots...")
    render_awake_day_violin_plots(
        filtered_visits,
        output_root,
        phase_display_names=DEFAULT_PHASE_DISPLAY_NAMES,
        scheduled_phase_start_hours=merged_scheduled_phase_starts,
        mouse_day_start_hour=mouse_day_start_hour,
        awake_end_clock_hour=awake_end_clock_hour,
        exclude_outliers=exclude_violin_outliers,
    )
    print("done. Rendering experience-learning plots...")
    render_experience_learning_plots(
        filtered_visits,
        output_root,
        phase_display_names=DEFAULT_PHASE_DISPLAY_NAMES,
        spread_metric=spread_metric,
        exclude_outliers=exclude_violin_outliers,
    )
    print("done. Rendering threshold-onset violin plots...")
    render_rate_threshold_onset_plots(
        filtered_visits,
        output_root,
        phase_display_names=DEFAULT_PHASE_DISPLAY_NAMES,
        threshold_pcts=rate_threshold_pcts,
        bin_hours=threshold_onset_bin_hours,
        exclude_outliers=exclude_violin_outliers,
    )
    print("done. Rendering derived ratio and AUC plots...")
    render_derived_metric_plots(
        filtered_visits,
        output_root,
        phase_display_names=DEFAULT_PHASE_DISPLAY_NAMES,
        scheduled_phase_start_hours=merged_scheduled_phase_starts,
        mouse_day_start_hour=mouse_day_start_hour,
        awake_end_clock_hour=awake_end_clock_hour,
        exclude_outliers=exclude_violin_outliers,
    )
    print("done. Rendering responder and count-model analyses...")
    render_count_model_and_responder_analyses(
        filtered_visits,
        output_root,
        phase_display_names=DEFAULT_PHASE_DISPLAY_NAMES,
        scheduled_phase_start_hours=merged_scheduled_phase_starts,
        mouse_day_start_hour=mouse_day_start_hour,
        awake_end_clock_hour=awake_end_clock_hour,
        threshold_pcts=rate_threshold_pcts,
        threshold_bin_hours=threshold_onset_bin_hours,
        responder_horizons_hours=responder_horizons_hours,
        glm_first_hours=binomial_model_first_hours,
    )
    print("done. Rendering cumulative role plots...")
    render_cumulative_role_plots(
        filtered_visits,
        output_root,
        group_names=group_names,
        plot_style=plot_style,
        phase_display_names=DEFAULT_PHASE_DISPLAY_NAMES,
        spread_metric=spread_metric,
        scheduled_phase_start_hours=merged_scheduled_phase_starts,
        mouse_day_start_hour=mouse_day_start_hour,
        awake_end_clock_hour=awake_end_clock_hour,
    )
    print("done. All analysis plots rendered.")
    return output_root
# %% MAIN FUNCTION
def main() -> None:
    """Run the full 4-month place-learning workflow from in-script settings."""

    output_root = run_analysis(
        dataset_root=USER_DATASET_ROOT,
        results_subdir=USER_RESULTS_SUBDIR,
        bin_hours=USER_BIN_HOURS,
        phase2_secondary_metric=USER_PHASE2_SECONDARY_METRIC,
        spread_metric=USER_SPREAD_METRIC,
        plot_style=USER_PLOT_STYLE,
        phase2_plot_style=USER_PHASE2_PLOT_STYLE,
        phase_max_hours=USER_PHASE_MAX_HOURS,
        excluded_groups=USER_EXCLUDED_GROUPS,
        group_renames=USER_GROUP_RENAMES,
        group_colors=USER_GROUP_COLORS,
        figure_size_cm=USER_FIGSIZE_CM,
        mouse_day_start_hour=USER_MOUSE_DAY_START_HOUR,
        awake_duration_hours=USER_AWAKE_DURATION_HOURS,
        scheduled_phase_start_hours=USER_SCHEDULED_PHASE_START_HOURS,
        base_font_size=USER_BASE_FONT_SIZE,
        exclude_violin_outliers=USER_EXCLUDE_VIOLIN_OUTLIERS,
        rate_threshold_pcts=USER_RATE_THRESHOLD_PCTS,
        threshold_onset_bin_hours=USER_THRESHOLD_ONSET_BIN_HOURS,
        responder_horizons_hours=USER_RESPONDER_HORIZONS_HOURS,
        binomial_model_first_hours=USER_BINOMIAL_MODEL_FIRST_HOURS,
    )
    all_groups_order = tuple(
        group_name
        for group_name in USER_GROUP_RENAMES.values()
        if group_name not in set(USER_EXCLUDED_GROUPS)
    )
    render_target_group_summary_panels(
        output_root=output_root,
        all_groups_order=all_groups_order,
        threshold_pcts=USER_RATE_THRESHOLD_PCTS,
        responder_horizon_hours=USER_SUMMARY_RESPONDER_HORIZON_HOURS,
        group_colors=resolved_group_colors(
            group_renames=USER_GROUP_RENAMES,
            group_colors=USER_GROUP_COLORS,
        ),
        awake_duration_hours=USER_AWAKE_DURATION_HOURS,
    )
    
    print(f"All summary panels rendered. Final output directory: {output_root}")
    print("done.")

# %% ENTRY POINT
if __name__ == "__main__":
    main()
# %% END
