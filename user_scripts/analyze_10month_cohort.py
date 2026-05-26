"""Run the IntelliCage workflow for the BioMedX 10-month cohort.

This user-facing script mirrors the 4-month analysis pipeline but adapts the
defaults to the 10-month cohort:

- seven IntelliCage run folders, including WT sex-specific runs
- mouse day starts at 07:00 and the awake period ends at 19:00
- protocol phase timing follows the protocol-aligned 10-month schedule
- experiment day counting follows the mouse-day definition, so day 0 begins
  at 07:00 and captures any pre-day-1 placement interval
- an additional sugar-preference analysis is performed on SP day 2 only

The phase 1-4 place-learning analysis is intentionally kept separate from the
SP analysis so the additional sugar-preference folders do not alter the
overview and learning plots of the main experiment.
"""
# %% IMPORTS
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "matplotlib"
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib.pyplot as plt

from intellicage_place_learning.loader import load_cohort_data
from intellicage_place_learning.metrics import (
    compute_clustered_binomial_gee_group_statistics,
    compute_onset_group_statistics,
    flag_iqr_outliers,
)
from intellicage_place_learning.plotting import (
    configure_plot_style,
    plot_onset_violin,
    set_group_colors,
    set_figure_size_presets,
)
from user_scripts.analyze_4month_cohort import (
    DEFAULT_GROUP_COLORS,
    apply_group_preferences,
    resolved_group_colors,
    run_analysis,
    save_table,
)

# %% PARAMETERS AND DEFAULTS
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "Data IntelliCage" / "BioMedX_10MonthCohort_2019"
DEFAULT_RESULTS_SUBDIR = Path("results")
DEFAULT_PHASE_DISPLAY_NAMES = {
    1: "Free Hab",
    2: "NPA",
    3: "PL",
    4: "PR",
}
DEFAULT_PHASE_NAME_MAP = {
    "Phase1": 1,
    "Phase2": 2,
    "Phase3": 3,
    "Phase4": 4,
}
DEFAULT_PHASE_NAME_MAP_WITH_SP = {
    "Phase1": 1,
    "Phase2": 2,
    "Phase3": 3,
    "Phase4": 4,
    "SP1": 5,
    "SP2": 6,
}
DEFAULT_OPTIONAL_PHASE_NAMES_WITH_SP = {"SP2"}
DEFAULT_PHASE_MAX_HOURS = {
    3: 71.0,
    4: 71.0,
}
DEFAULT_FIGSIZE_CM = { # always a pair of (width, height)
    "LONG_FIGSIZE_CM": (18.2, 7.4),
    "LONG_FIGSIZE_2_CM": (15.2, 7.4),
    "PHASE2_FIGSIZE_CM": (10.4, 7.0),
    "MEDIUM_FIGSIZE_CM": (11.8, 7.6),
    "MEDIUM_WIDE_FIGSIZE_CM": (12.8, 8.0),
    "SEGMENT_FIGSIZE_CM": (12.6, 7.9),
    "VIOLIN_FIGSIZE_CM": (5.8, 7.2),
    "ONSET_FIGSIZE_CM": (5.8, 7.0),
    "ACTIVITY_FIGSIZE_CM": (8.8, 8.1),
    "WIDE_GROUP_FIGSIZE_CM": (18.2, 7.4),
}
DEFAULT_SCHEDULED_PHASE_START_HOURS = {
    1: 0.0,
    2: 74.0,
    3: 122.0,
    4: 194.0,
    5: 266.0,
    6: 290.0,
}
DEFAULT_GROUP_RENAMES = {
    "WT":           "WT",
    "WT_female":    "WT female",
    "WT_male":      "WT male",
    "tdTomato":     "tdTomato",
    "Tau 1-441":    "Tau 1-441",
    "Tau 1-421":    "Tau 1-421",
    "Tau 66-421":   "Tau 66-421",
}

USER_DATASET_ROOT = DEFAULT_DATASET_ROOT
USER_RESULTS_SUBDIR = DEFAULT_RESULTS_SUBDIR
USER_BIN_HOURS = [1, 2]
USER_PHASE2_SECONDARY_METRIC = "lick_positive_visits"
USER_SPREAD_METRIC = "sem"
USER_PLOT_STYLE = "line"
USER_PHASE2_PLOT_STYLE = "line"
USER_PHASE_MAX_HOURS = DEFAULT_PHASE_MAX_HOURS.copy()
USER_PHASE_NAME_MAP = DEFAULT_PHASE_NAME_MAP.copy()
USER_PHASE_NAME_MAP_WITH_SP = DEFAULT_PHASE_NAME_MAP_WITH_SP.copy()
USER_OPTIONAL_PHASE_NAMES_WITH_SP = DEFAULT_OPTIONAL_PHASE_NAMES_WITH_SP.copy()
USER_DROP_UNMATCHED_VISITS = True
USER_EXCLUDED_GROUPS: list[str] = ["WT"]
USER_GROUP_RENAMES = DEFAULT_GROUP_RENAMES.copy()
USER_GROUP_COLORS = {
    "WT":        "#264653",
    "tdTomato":  "#6c757d",
    "Tau 1-441": "#4ade80",
    "Tau 1-421": "#e9a820",
    "Tau 66-421":"#2a9d8f",
    "WT female": "#264653",
    "WT male":   "#5194AE",
}
USER_FIGSIZE_CM = DEFAULT_FIGSIZE_CM.copy()
USER_MOUSE_DAY_START_HOUR = 7.0
USER_AWAKE_DURATION_HOURS = 12.0
USER_EXPERIMENT_DAY0_START_HOUR = None
USER_SCHEDULE_ANCHOR_PHASE_NUMBER = 2
USER_SCHEDULED_PHASE_START_HOURS = DEFAULT_SCHEDULED_PHASE_START_HOURS.copy()
USER_RATE_THRESHOLD_PCTS = [50.0, 60.0, 70.0, 80.0]
USER_THRESHOLD_ONSET_BIN_HOURS = 1
USER_RESPONDER_HORIZONS_HOURS = [24.0, 48.0, 72.0]
USER_BINOMIAL_MODEL_FIRST_HOURS = 24.0
USER_BASE_FONT_SIZE = 10.0
USER_EXCLUDE_VIOLIN_OUTLIERS = True
USER_SUMMARY_RESPONDER_HORIZON_HOURS = 24.0
USER_SUMMARY_FIGSIZE_CM = (16.5, 12.5)
CM_TO_INCH = 2.54
# %% HELPER FUNCTIONS
def compute_sp2_preference_table(
    visits: pd.DataFrame,
    nosepokes: pd.DataFrame,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize sugar-side versus water-side licking for SP day 2.

    The legacy MATLAB script derived sugar preference from `Nosepokes.txt`
    rather than from visit-level totals because the bottle side matters during
    the sugar-preference test. This function follows the same logic:

    - restrict to raw phase number 6 (`SP2`)
    - keep only nose-pokes with a lick response
    - classify the licks by side correctness / side condition
    - compute per-mouse sugar preference ratio as
      `sugar_licks / (sugar_licks + water_licks) * 100`
    """

    sp2_visits = visits.loc[visits["PhaseNumber"].eq(6)].copy()
    if sp2_visits.empty:
        return pd.DataFrame(), pd.DataFrame()

    visit_keys = (
        sp2_visits.loc[:, ["RunGroup", "Phase", "PhaseNumber", "VisitID", "AnimalTag", "Group", "ET", "ETLabel", "SEX"]]
        .drop_duplicates()
        .rename(columns={"AnimalTag": "RFID"})
    )
    sp2_nosepokes = nosepokes.loc[nosepokes["PhaseNumber"].eq(6)].copy()
    if sp2_nosepokes.empty:
        return pd.DataFrame(), pd.DataFrame()

    sp2_nosepokes = sp2_nosepokes.merge(
        visit_keys,
        on=["RunGroup", "Phase", "PhaseNumber", "VisitID"],
        how="inner",
        validate="many_to_one",
    )
    lick_positive = sp2_nosepokes["LickNumber"].fillna(0).gt(0) | sp2_nosepokes["LickDuration"].fillna(0).gt(0)
    sp2_nosepokes = sp2_nosepokes.loc[lick_positive].copy()
    if sp2_nosepokes.empty:
        return pd.DataFrame(), pd.DataFrame()

    if "SideCondition" in sp2_nosepokes.columns:
        sugar_mask = sp2_nosepokes["SideCondition"].eq(1)
        water_mask = sp2_nosepokes["SideCondition"].eq(-1)
    else:
        sugar_mask = sp2_nosepokes["SideError"].eq(0)
        water_mask = sp2_nosepokes["SideError"].eq(1)

    sp2_nosepokes["sugar_licks"] = np.where(sugar_mask, sp2_nosepokes["LickNumber"].fillna(0), 0)
    sp2_nosepokes["water_licks"] = np.where(water_mask, sp2_nosepokes["LickNumber"].fillna(0), 0)
    sp2_nosepokes["sugar_side_lick_event"] = sugar_mask.astype(int)
    sp2_nosepokes["water_side_lick_event"] = water_mask.astype(int)

    mouse_summary = (
        sp2_nosepokes.groupby(["Group", "ET", "ETLabel", "SEX"], observed=True)
        .agg(
            sugar_licks=("sugar_licks", "sum"),
            water_licks=("water_licks", "sum"),
            sugar_side_lick_events=("sugar_side_lick_event", "sum"),
            water_side_lick_events=("water_side_lick_event", "sum"),
            total_nosepoke_rows=("VisitID", "size"),
        )
        .reset_index()
    )
    mouse_summary["total_licks"] = mouse_summary["sugar_licks"] + mouse_summary["water_licks"]
    mouse_summary["sugar_preference_ratio_pct"] = np.where(
        mouse_summary["total_licks"].gt(0),
        mouse_summary["sugar_licks"] / mouse_summary["total_licks"] * 100.0,
        np.nan,
    )

    mouse_index = metadata.loc[:, ["Group", "ET", "ETLabel", "SEX"]].drop_duplicates().reset_index(drop=True)
    mouse_summary = mouse_index.merge(
        mouse_summary,
        on=["Group", "ET", "ETLabel", "SEX"],
        how="left",
        validate="one_to_one",
    )
    fill_zero_cols = [
        "sugar_licks",
        "water_licks",
        "sugar_side_lick_events",
        "water_side_lick_events",
        "total_nosepoke_rows",
        "total_licks",
    ]
    for column in fill_zero_cols:
        mouse_summary[column] = mouse_summary[column].fillna(0.0)
    mouse_summary["has_sp2_data"] = mouse_summary["total_nosepoke_rows"].gt(0)

    group_summary = (
        mouse_summary.groupby("Group", observed=True)
        .agg(
            mean_preference_ratio_pct=("sugar_preference_ratio_pct", "mean"),
            median_preference_ratio_pct=("sugar_preference_ratio_pct", "median"),
            std_preference_ratio_pct=("sugar_preference_ratio_pct", "std"),
            mouse_n=("ET", "nunique"),
            contributing_mouse_n=("sugar_preference_ratio_pct", lambda values: int(values.notna().sum())),
            mean_sugar_licks=("sugar_licks", "mean"),
            mean_water_licks=("water_licks", "mean"),
        )
        .reset_index()
    )
    group_summary["std_preference_ratio_pct"] = group_summary["std_preference_ratio_pct"].fillna(0.0)
    group_summary["sem_preference_ratio_pct"] = (
        group_summary["std_preference_ratio_pct"] / np.sqrt(group_summary["mouse_n"].clip(lower=1))
    )
    return mouse_summary, group_summary

def render_sp2_sugar_preference_analysis(
    *,
    dataset_root: Path,
    output_root: Path,
    excluded_groups: list[str] | None,
    group_renames: dict[str, str] | None,
    group_colors: dict[str, str] | None,
    base_font_size: float,
    exclude_outliers: bool,
) -> None:
    """Load SP1/SP2 data and render the SP day-2 sugar-preference summary."""

    cohort = load_cohort_data(
        dataset_root,
        phase_name_map=USER_PHASE_NAME_MAP_WITH_SP,
        optional_phase_names=USER_OPTIONAL_PHASE_NAMES_WITH_SP,
        drop_unmatched_visits=USER_DROP_UNMATCHED_VISITS,
    )
    selected_visits, selected_metadata, selected_nosepokes = apply_group_preferences(
        cohort.visits,
        cohort.metadata,
        cohort.nosepokes,
        excluded_groups=excluded_groups,
        group_renames=group_renames,
    )

    configure_plot_style(font_size=base_font_size, font_family="Arial")
    set_group_colors(
        resolved_group_colors(
            group_renames=group_renames or {},
            group_colors=group_colors,
        )
    )
    set_figure_size_presets(USER_FIGSIZE_CM)

    mouse_summary, group_summary = compute_sp2_preference_table(
        selected_visits,
        selected_nosepokes,
        selected_metadata,
    )
    if mouse_summary.empty:
        return

    save_table(mouse_summary, output_root / "sp2_sugar_preference_mouse.tsv")
    save_table(group_summary, output_root / "sp2_sugar_preference_group_summary.tsv")

    mouse_with_outliers = flag_iqr_outliers(
        mouse_summary,
        value_col="sugar_preference_ratio_pct",
        group_cols=["Group"],
    )
    save_table(mouse_with_outliers, output_root / "sp2_sugar_preference_mouse_with_outliers.tsv")

    omnibus_stats, pairwise_stats = compute_onset_group_statistics(
        mouse_with_outliers,
        onset_col="sugar_preference_ratio_pct",
        phase_number=6,
        metric_name="sugar_preference_ratio",
        exclude_outliers=exclude_outliers,
    )
    save_table(omnibus_stats, output_root / "sp2_sugar_preference_omnibus_stats.tsv")
    save_table(pairwise_stats, output_root / "sp2_sugar_preference_pairwise_stats.tsv")

    plot_onset_violin(
        mouse_with_outliers,
        onset_col="sugar_preference_ratio_pct",
        phase_display_name="SP day 2",
        title_label="Sugar preference ratio",
        ylabel="Sugar preference ratio [%]",
        output_path=output_root / "sp2_sugar_preference_ratio_violin.png",
        pairwise_stats=pairwise_stats,
        outlier_col="is_outlier",
        reference_line=50.0,
    )

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

def _pairwise_p_value_for_groups(
    pairwise_stats: pd.DataFrame,
    *,
    group_a: str,
    group_b: str,
    filter_column: str | None = None,
    filter_value: object | None = None,
) -> float | None:
    """Return one pairwise p-value for two named groups from one stats table."""

    if pairwise_stats.empty or {"group1", "group2", "p_value"} - set(pairwise_stats.columns):
        return None
    subset = pairwise_stats.copy()
    if filter_column is not None:
        subset = subset.loc[subset[filter_column].eq(filter_value)].copy()
    subset = subset.loc[
        (
            subset["group1"].astype(str).eq(group_a)
            & subset["group2"].astype(str).eq(group_b)
        )
        | (
            subset["group1"].astype(str).eq(group_b)
            & subset["group2"].astype(str).eq(group_a)
        )
    ].copy()
    if subset.empty:
        return None
    return float(subset["p_value"].iloc[0])

def _omnibus_p_value(
    omnibus_stats: pd.DataFrame,
    *,
    filter_column: str | None = None,
    filter_value: object | None = None,
) -> float | None:
    """Return one omnibus p-value from a stats table."""

    if omnibus_stats.empty or "p_value" not in omnibus_stats.columns:
        return None
    subset = omnibus_stats.copy()
    if filter_column is not None:
        subset = subset.loc[subset[filter_column].eq(filter_value)].copy()
    if subset.empty:
        return None
    return float(subset["p_value"].iloc[0])

def _ordered_available_groups(frame: pd.DataFrame, preferred_order: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Return preferred groups that are actually present in one result table."""

    present = set(frame["Group"].astype(str))
    return tuple(group_name for group_name in preferred_order if group_name in present)

def _draw_distribution_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    value_col: str,
    group_order: tuple[str, ...] | list[str],
    group_colors: dict[str, str],
    title: str,
    ylabel: str,
    p_value: float | None = None,
    p_label: str = "p",
    pairwise_stats: pd.DataFrame | None = None,
    pairwise_filter_column: str | None = None,
    pairwise_filter_value: object | None = None,
    as_percent: bool = True,
    reference_line: float | None = None,
) -> None:
    """Draw one compact distribution panel with points and violins."""

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

    numeric_values = np.concatenate([values for values in violin_data if len(values) > 0]) if any(len(values) > 0 for values in violin_data) else np.array([0.0])
    y_max = float(np.nanmax(numeric_values)) if numeric_values.size else 1.0
    default_min = 100.0 if as_percent else 1.0
    y_data_max = float(max(default_min, y_max))
    y_base = max((104.0 if as_percent else y_data_max * 1.05), y_data_max + (5.0 if as_percent else max(0.1, y_data_max * 0.08)))
    y_step = 10.0 if as_percent else max(0.12, y_data_max * 0.10)
    y_limit = 124.0 if as_percent else max(1.25, y_base + 0.4)

    significant_pairs = pd.DataFrame()
    if pairwise_stats is not None and not pairwise_stats.empty and {"group1", "group2", "p_value"}.issubset(pairwise_stats.columns):
        significant_pairs = pairwise_stats.copy()
        if pairwise_filter_column is not None:
            significant_pairs = significant_pairs.loc[significant_pairs[pairwise_filter_column].eq(pairwise_filter_value)].copy()
        significant_pairs = significant_pairs.loc[
            significant_pairs["group1"].astype(str).isin(group_order)
            & significant_pairs["group2"].astype(str).isin(group_order)
            & significant_pairs["p_value"].lt(0.05)
        ].copy()
        significant_pairs["left_pos"] = significant_pairs["group1"].astype(str).map({group: idx + 1 for idx, group in enumerate(group_order)})
        significant_pairs["right_pos"] = significant_pairs["group2"].astype(str).map({group: idx + 1 for idx, group in enumerate(group_order)})
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
    if p_value is not None and not np.isnan(p_value):
        ax.text(
            0.98,
            0.98,
            f"{p_label}={p_value:.3g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=plt.rcParams["legend.fontsize"],
            color="#444444",
        )

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

def _save_panel_figure(
    output_root: Path,
    stem: str,
    draw_fn,
) -> None:
    """Create and save one single-panel summary figure."""

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(8.2 / CM_TO_INCH, 7.1 / CM_TO_INCH),
    )
    draw_fn(ax)
    png_path, pdf_path = _summary_output_paths(output_root, stem)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

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
    reversal_pref_stats = _load_result_table(output_root, "phase4_reversal_preference_index_awake_day_rate_pairwise_stats.tsv")
    reversal_pref_omnibus = _load_result_table(output_root, "phase4_reversal_preference_index_awake_day_rate_omnibus_stats.tsv")

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
        "all_groups_pr_day1_reversal_preference_index",
        lambda ax: _draw_distribution_panel(
            ax,
            reversal_pref.loc[reversal_pref["phase_day"].eq(1)].copy(),
            value_col="value",
            group_order=all_reversal,
            group_colors=group_colors,
            title="PR day 1 awake\nReversal preference index",
            ylabel="New / (new + previous)\n(higher = better)",
            pairwise_stats=reversal_pref_stats,
            pairwise_filter_column="phase_day",
            pairwise_filter_value=1,
            as_percent=False,
            reference_line=0.5,
        ),
    )
    _save_panel_figure(
        output_root,
        "all_groups_pr_day3_reversal_preference_index",
        lambda ax: _draw_distribution_panel(
            ax,
            reversal_pref.loc[reversal_pref["phase_day"].eq(3)].copy(),
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

# %% MAIN WORKFLOW
def main() -> None:
    """Run the 10-month place-learning and SP day-2 analyses."""

    output_root = run_analysis(
        dataset_root=USER_DATASET_ROOT,
        results_subdir=USER_RESULTS_SUBDIR,
        bin_hours=USER_BIN_HOURS,
        phase2_secondary_metric=USER_PHASE2_SECONDARY_METRIC,
        spread_metric=USER_SPREAD_METRIC,
        plot_style=USER_PLOT_STYLE,
        phase2_plot_style=USER_PHASE2_PLOT_STYLE,
        phase_max_hours=USER_PHASE_MAX_HOURS,
        phase_name_map=USER_PHASE_NAME_MAP,
        drop_unmatched_visits=USER_DROP_UNMATCHED_VISITS,
        excluded_groups=USER_EXCLUDED_GROUPS,
        group_renames=USER_GROUP_RENAMES,
        group_colors=USER_GROUP_COLORS,
        figure_size_cm=USER_FIGSIZE_CM,
        mouse_day_start_hour=USER_MOUSE_DAY_START_HOUR,
        awake_duration_hours=USER_AWAKE_DURATION_HOURS,
        experiment_day0_start_hour=USER_EXPERIMENT_DAY0_START_HOUR,
        schedule_anchor_phase_number=USER_SCHEDULE_ANCHOR_PHASE_NUMBER,
        scheduled_phase_start_hours=USER_SCHEDULED_PHASE_START_HOURS,
        rate_threshold_pcts=USER_RATE_THRESHOLD_PCTS,
        threshold_onset_bin_hours=USER_THRESHOLD_ONSET_BIN_HOURS,
        responder_horizons_hours=USER_RESPONDER_HORIZONS_HOURS,
        binomial_model_first_hours=USER_BINOMIAL_MODEL_FIRST_HOURS,
        base_font_size=USER_BASE_FONT_SIZE,
        exclude_violin_outliers=USER_EXCLUDE_VIOLIN_OUTLIERS,
    )

    render_sp2_sugar_preference_analysis(
        dataset_root=USER_DATASET_ROOT,
        output_root=output_root,
        excluded_groups=USER_EXCLUDED_GROUPS,
        group_renames=USER_GROUP_RENAMES,
        group_colors=USER_GROUP_COLORS,
        base_font_size=USER_BASE_FONT_SIZE,
        exclude_outliers=USER_EXCLUDE_VIOLIN_OUTLIERS,
    )

    resolved_colors = resolved_group_colors(
        group_renames=USER_GROUP_RENAMES,
        group_colors=USER_GROUP_COLORS,
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
        group_colors=resolved_colors,
        awake_duration_hours=USER_AWAKE_DURATION_HOURS,
    )
    print("Analysis complete. Summary figures saved to:", output_root)
    print("done.")

# %% ENTRY POINT
if __name__ == "__main__":
    main()
# %% END
