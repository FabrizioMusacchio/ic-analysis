"""Object-oriented experiment containers for IntelliCage analyses."""
# %% IMPORTS
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from . import metrics as mt
from .loader import CohortData, attach_analysis_time_columns, load_cohort_data
from .metadata import ExperimentMetadata, SubjectRegistry
from . import plotting as plotting_module
from .plotting import (
    configure_plot_style,
    plot_bottle_preference_groups,
    set_group_colors)
from .workflows import place_learning_reversal as plr

# %% TYPES
SpreadMetric = Literal["sem", "std"]
PhaseSelection = Literal["all"] | int | tuple[int, ...] | list[int]
DayPhase = Literal["day", "night", "all"]
# %% HELPERS
def _yaml_scalar(value: Any) -> str:
    """Return a small YAML-safe scalar representation."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    text = str(value).replace('"', '\\"')
    return f'"{text}"'

def _write_yaml_lines(value: Any, indent: int = 0) -> list[str]:
    """Serialize simple dict/list/scalar metadata to readable YAML."""

    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, dict | list | tuple):
                lines.append(f"{prefix}{key}:")
                lines.extend(_write_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(item)}")
        return lines
    if isinstance(value, list | tuple):
        lines = []
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{prefix}-")
                lines.extend(_write_yaml_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return lines
    return [f"{prefix}{_yaml_scalar(value)}"]

def _write_yaml(path: Path, value: Any) -> None:
    """Write metadata as a small YAML file without adding a runtime dependency."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_write_yaml_lines(value)) + "\n", encoding="utf-8")

def _phase_window_to_strings(window: tuple[Any, Any] | None) -> list[str] | None:
    """Return phase window timestamps as strings for audit metadata."""

    if window is None:
        return None
    return [str(pd.to_datetime(window[0])), str(pd.to_datetime(window[1]))]

def _normalize_phase_selection(phases: PhaseSelection, available_phases: list[int]) -> list[int]:
    """Normalize ``all``, one phase, or several phases into phase numbers."""

    if phases == "all":
        return list(available_phases)
    requested = [phases] if isinstance(phases, int) else list(phases)
    normalized: list[int] = []
    for phase in requested:
        phase_number = int(phase)
        if phase_number in available_phases:
            normalized.append(phase_number)
        elif 0 <= phase_number < len(available_phases):
            normalized.append(available_phases[phase_number])
        else:
            raise ValueError(f"Unknown phase selection {phase!r}. Available phases are {available_phases}.")
    return normalized

def _phase_file_tag(phases: list[int], available_phases: list[int] | None = None) -> str:
    """Return a compact file-name tag for a phase selection."""

    if available_phases is not None and phases == available_phases:
        return "all_phases"
    if len(phases) == 1:
        return f"phase{phases[0]}"
    return "phases_" + "_".join(str(phase) for phase in phases)

def _bin_file_tag(bin_h: int | float) -> str:
    """Return a compact file-name tag for one bin width."""

    return f"{float(bin_h):g}h".replace(".", "p")

def _binned_output_dir(results_data_path: Path, bin_h: int | float, dayphase: DayPhase) -> Path:
    """Return the default binned output directory for one dayphase selection."""

    bin_tag = _bin_file_tag(bin_h)
    suffix = "" if dayphase == "all" else f"_{dayphase}"
    return results_data_path / f"{bin_tag}{suffix}_bins"

def _raw_export_file_counts(root_data_path: Path) -> dict[str, int]:
    """Count common IntelliCage export tables below one dataset root."""

    counts: dict[str, int] = {}
    for filename in ("Visits.txt", "Nosepokes.txt"):
        counts[filename] = sum(1 for _ in root_data_path.rglob(filename))
    return counts

@contextmanager
def _temporary_figsize(keys: tuple[str, ...], figsize_cm: tuple[float, float] | None):
    """Temporarily map one user-facing figure-size tuple to internal plot types."""

    if figsize_cm is None:
        yield
        return
    saved = {key: getattr(plotting_module, key) for key in keys}
    try:
        for key in keys:
            setattr(plotting_module, key, (float(figsize_cm[0]), float(figsize_cm[1])))
        yield
    finally:
        for key, value in saved.items():
            setattr(plotting_module, key, value)

# %% DATA CLASSES
@dataclass
class PlotLayout:
    """Optional plot-level styling overrides for one figure family.

    ``PlotLayout`` is a small convenience container used by plotting methods
    that accept ``plot_layout``. Users can pass either a ``PlotLayout`` instance
    or a plain dictionary with the same keys.

    :param figsize_cm: Optional figure size as ``(width_cm, height_cm)``.
        Default is ``None``, which uses the method's ``figsize_cm`` argument or
        the toolkit default for that plot type.
    :param title: Optional custom plot title. Default is ``None``.
    :param xlabel: Optional custom x-axis label. Default is ``None``.
    :param ylabel: Optional custom y-axis label. Default is ``None``.
    :param xlim: Optional x-axis limits as ``(min, max)`` in the plotted x-axis
        unit. Default is ``None``.
    :param ylim: Optional y-axis limits as ``(min, max)``. Default is ``None``.
    :param xticks: Optional explicit x tick locations. Default is ``None``.
    :param yticks: Optional explicit y tick locations. Default is ``None``.
    :param legend: Optional legend switch. Use ``True`` to show, ``False`` to
        hide, or ``None`` to use the plot default.
    :param legend_loc: Optional Matplotlib legend location such as
        ``"upper right"``, ``"upper left"``, ``"lower right"``, ``"best"``, or
        an integer location code. Default is ``None``, which keeps the
        individual plot's default legend placement.
    :param legend_font_size: Optional legend font size in points. Default is
        ``None``, which uses the current ``base_font_size``.
    :param extra: Optional dictionary for plot-specific controls. Current
        high-level methods use ``show_N`` and ``xtick_rotation`` where relevant.
    """

    figsize_cm: tuple[float, float] | None = None
    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    xticks: list[float] | None = None
    yticks: list[float] | None = None
    legend: bool | None = None
    legend_loc: str | int | None = None
    legend_font_size: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: "PlotLayout | dict[str, Any] | None") -> "PlotLayout":
        """Normalize a layout mapping passed by a user script."""

        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        known = {
            key: value[key]
            for key in cls.__dataclass_fields__
            if key in value and key != "extra"}
        extra = {
            key: item
            for key, item in value.items()
            if key not in known}
        known["extra"] = extra
        return cls(**known)

@dataclass
class IntelliCageExperiment:
    """Generic user-facing IntelliCage experiment object."""

    experiment: ExperimentMetadata
    subjects: SubjectRegistry
    cohort: CohortData | None = None
    visits: pd.DataFrame | None = None
    phase_windows: pd.DataFrame | None = None
    analysis_visits: pd.DataFrame | None = None
    analysis_metadata: pd.DataFrame | None = None
    analysis_nosepokes: pd.DataFrame | None = None
    analysis_phase_windows: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        """Create the configured result directory when the object is built."""

        self.results_data_path.mkdir(parents=True, exist_ok=True)

    @property
    def root_data_path(self) -> Path:
        """Return the raw dataset root."""

        return Path(self.experiment.root_data_path)

    @property
    def results_data_path(self) -> Path:
        """Return the configured result directory."""

        return Path(self.experiment.results_data_path)

    @property
    def scheduled_phase_start_hours(self) -> dict[int, float]:
        """Return the active protocol schedule."""

        return self.subjects.scheduled_phase_start_hours(self.experiment)

    @property
    def phase_display_names(self) -> dict[int, str]:
        """Return short phase names for plotting."""

        return self.experiment.phase_display_names

    @property
    def phases(self) -> dict[int, Any]:
        """Return the experiment phase metadata."""

        return self.experiment.phases

    @property
    def group_names(self) -> list[str]:
        """Return the configured group order."""

        return list(self.experiment.group_names)

    @property
    def awake_start_clock_hour(self) -> float:
        """Return the configured active-period start."""

        return plr.active_period_bounds(
            self.experiment.mouse_day_start_hour,
            self.experiment.awake_duration_hours)[0]

    @property
    def awake_end_clock_hour(self) -> float:
        """Return the configured active-period end."""

        return plr.active_period_bounds(
            self.experiment.mouse_day_start_hour,
            self.experiment.awake_duration_hours)[1]

    def load(self) -> None:
        """Load raw IntelliCage exports and write reproducibility audit files.

        ``load`` reads all configured ``Visits.txt`` and ``Nosepokes.txt``
        exports below ``experiment.root_data_path``, keeps only animals declared
        in ``subjects``, joins subject metadata, summarizes nose-poke events per
        visit, assigns phase labels, and creates aligned experiment-time
        columns. It also writes the loaded visit/nose-poke tables plus
        ``experiment.yaml``, ``phases.yaml``, and ``subjects.yaml`` to the
        results folder.

        :returns: ``None``. The loaded tables are stored on the experiment
            object as ``visits``, ``cohort``, and ``phase_windows``.
        """

        subject_frame = self.subjects.to_loader_frame(self.experiment)
        raw_counts = _raw_export_file_counts(self.root_data_path)
        self.cohort = load_cohort_data(
            self.root_data_path,
            phase_name_map=self.experiment.phase_name_map,
            optional_phase_names=self.experiment.optional_phase_names,
            group_names=self.experiment.group_names,
            subject_metadata=subject_frame,
            drop_unregistered_subjects=True)
        self.visits = attach_analysis_time_columns(
            self.cohort.visits,
            self.cohort.phase_manifest,
            scheduled_phase_start_hours=self.scheduled_phase_start_hours,
            mouse_day_start_hour=self.experiment.mouse_day_start_hour,
            experiment_day0_start_hour=self.experiment.experiment_day0_start_hour,
            schedule_anchor_phase_number=self.experiment.schedule_anchor_phase_number)
        self.phase_windows = mt.build_analysis_phase_window_table(
            self.visits,
            self.scheduled_phase_start_hours)
        self._write_load_outputs(subject_frame)
        self._print_load_summary(raw_counts=raw_counts, registered_subject_count=len(subject_frame))

    def require_loaded(self) -> tuple[CohortData, pd.DataFrame]:
        """Return loaded cohort data and visits.

        :returns: ``(cohort, visits)`` after :meth:`load` has been called.
        :raises RuntimeError: If the experiment has not been loaded yet.
        """

        if self.cohort is None or self.visits is None:
            raise RuntimeError("Call `.load()` before computing metrics or plotting.")
        return self.cohort, self.visits

    def filtered_visits(
        self,
        phase_max_hours: dict[int, float] | None = None,
        dayphase: DayPhase = "all") -> pd.DataFrame:
        """Return loaded visits filtered by phase duration and dayphase.

        :param phase_max_hours: Optional dictionary mapping phase number to the
            maximum number of phase-elapsed hours to keep, for example
            ``{3: 72.0, 4: 72.0}``. Default is ``None`` and keeps all available
            phase time.
        :param dayphase: ``"day"`` keeps the configured active mouse-day,
            ``"night"`` keeps the inactive interval, and ``"all"`` keeps both.
            Default is ``"all"`` for this low-level accessor.
        :returns: A filtered visit-level ``DataFrame``.
        """

        _, visits = self.require_loaded()
        filtered = mt.filter_visits_by_phase_limits(visits, phase_max_hours)
        return self._filter_dayphase(filtered, dayphase=dayphase)

    def prepare_analysis(
        self,
        *,
        phase_max_hours: dict[int, float] | None = None,
        excluded_groups: list[str] | None = None,
        verbose: bool = True) -> None:
        """Prepare reusable analysis tables from loaded raw data.

        This method applies optional phase-duration limits, applies group
        inclusion/exclusion, stores analysis-ready visits, metadata, nosepokes,
        and phase windows on the object, and writes audit tables to the results
        folder. Most plotting methods call this automatically when needed, but
        calling it explicitly is recommended in interactive scripts so the
        printed summary and audit files are created before plotting.

        :param phase_max_hours: Optional phase limits in hours, e.g.
            ``{3: 72.0, 4: 72.0}``. Default is ``None``. Use this when learning
            phases have unequal recording lengths and you want comparable
            analysis windows.
        :param excluded_groups: Optional list of group labels to remove.
            Default is ``None``. Use it for temporary QC reruns without editing
            the subject metadata.
        :param verbose: Whether to print the analysis-preparation summary.
            Default is ``True`` for explicit interactive calls. Internal plot
            calls use ``False`` to avoid repeated terminal output.
        :returns: ``None``. Prepared tables are stored as ``analysis_visits``,
            ``analysis_metadata``, ``analysis_nosepokes``, and
            ``analysis_phase_windows``.
        """

        cohort, aligned_visits = self.require_loaded()
        raw_visit_count = len(aligned_visits)
        raw_nosepoke_count = len(cohort.nosepokes)
        group_renames = {group: group for group in self.group_names}
        selected_visits, selected_metadata, selected_nosepokes = plr.apply_group_preferences(
            aligned_visits,
            cohort.metadata,
            cohort.nosepokes,
            excluded_groups=excluded_groups or [],
            group_renames=group_renames)
        self.analysis_visits = mt.filter_visits_by_phase_limits(selected_visits, phase_max_hours)
        self.analysis_metadata = selected_metadata
        self.analysis_nosepokes = selected_nosepokes
        self.analysis_phase_windows = mt.build_analysis_phase_window_table(
            self.analysis_visits,
            self.scheduled_phase_start_hours)
        self._write_analysis_audit_outputs(phase_max_hours=phase_max_hours)
        if verbose:
            self._print_analysis_summary(
                raw_visit_count=raw_visit_count,
                raw_nosepoke_count=raw_nosepoke_count,
                phase_max_hours=phase_max_hours,
                excluded_groups=excluded_groups or [])

    def plot_ages(
        self,
        *,
        output_path: Path | None = None,
        time_unit: Literal["months", "days", "years"] = "months",
        show_N: bool = True,
        plot_layout: PlotLayout | dict | None = None,
        base_font_size: float = 10.0,
        font_family: str = "Arial",
        figsize_cm: tuple[float, float] | None = None) -> None:
        """Plot mouse age at the first experiment phase start.

        Age is computed from ``date_of_birth`` to the first phase start for
        each subject. If no date of birth is available, the subject-level
        ``age_months`` value is used when possible. The plotted table is saved
        alongside the figure for downstream use in GraphPad Prism or similar
        tools.

        :param output_path: Optional explicit output file path. Default is
            ``None``, which writes ``mouse_age_at_phase1_start_<unit>_violin`` to
            the experiment results folder.
        :param time_unit: Unit for age values. Default is ``"months"``.
            Accepted values are ``"days"``, ``"months"``, and ``"years"``.
        :param show_N: Whether x tick labels include per-group sample size.
            Default is ``True`` and recommended for public/reporting figures.
        :param plot_layout: Optional ``dict`` or :class:`PlotLayout`. Supported
            keys are ``title``, ``ylabel``, ``ylim``, and ``figsize_cm`` for
            this plot. Other common keys such as ``xlabel``, ``xlim``,
            ``xticks``, ``yticks``, and ``legend`` are ignored when they do not
            apply to the figure.
        :param base_font_size: Base font size in points. Default is ``10.0``.
        :param font_family: Matplotlib font family. Default is ``"Arial"``.
        :param figsize_cm: Optional figure size as ``(width_cm, height_cm)``.
            Default is ``None`` and uses the plot preset unless overridden by
            ``plot_layout["figsize_cm"]``.
        :returns: ``None``.
        """

        cohort, _ = self.require_loaded()
        self._apply_plot_settings(base_font_size=base_font_size, font_family=font_family)
        layout = PlotLayout.from_mapping(plot_layout)
        plr.render_mouse_age_at_phase1_start_plot(
            cohort.metadata,
            cohort.phase_manifest,
            Path(output_path).parent if output_path is not None else self.results_data_path,
            group_renames={group: group for group in self.group_names},
            figure_output_path=Path(output_path) if output_path is not None else None,
            time_unit=time_unit,
            show_n=show_N,
            figsize_cm=layout.figsize_cm or figsize_cm,
            title=layout.title,
            ylabel=layout.ylabel,
            y_limits=layout.ylim)

    def plot_mice_activity(
        self,
        *,
        bin_hours: int = 1,
        phases: PhaseSelection = "all",
        dayphase: DayPhase = "all",
        phase_max_hours: dict[int, float] | None = None,
        spread_metric: SpreadMetric = "sem",
        plot_style: str = "line",
        day_night_indicator: tuple[str, str] | None = ("awake", "sleep"),
        output_dir: Path | None = None,
        plot_layout: PlotLayout | dict | None = None,
        base_font_size: float = 10.0,
        font_family: str = "Arial",
        figsize_cm: tuple[float, float] | None = None) -> Path:
        """Plot visit activity across all phases or a selected phase subset.

        The plot shows visit counts per mouse and bin, plus group mean
        ``μ ± SEM`` or ``μ ± SD`` depending on ``spread_metric``. Group-specific
        figures include individual mouse traces; the all-groups figure shows
        group summaries only. This is a quality-control plot, not a learning
        metric.

        :param bin_hours: Width of time bins in hours. Default is ``1``. Use
            ``1`` for detailed QC and larger values such as ``2`` or ``24`` for
            smoother overview plots.
        :param phases: Phase selection. Default is ``"all"``. Use an integer
            such as ``2`` for one phase or a tuple such as ``(2, 4)`` for
            several phases.
        :param dayphase: ``"all"`` plots full 24 h behavior, ``"day"`` plots the
            configured active mouse-day, and ``"night"`` plots the inactive
            interval. Default is ``"all"`` for activity QC.
        :param phase_max_hours: Optional phase limits in hours. Default is
            ``None``.
        :param spread_metric: Group spread around the mean. Default is
            ``"sem"``; use ``"std"`` to show standard deviation.
        :param plot_style: ``"line"`` or ``"step"``. Default is ``"line"``.
        :param output_dir: Optional output folder. Default is a bin/dayphase
            folder below the experiment results folder.
        :param plot_layout: Optional ``dict`` or :class:`PlotLayout`. Common
            keys are ``title``, ``xlabel``, ``ylabel``, ``xlim``, ``ylim``,
            ``xticks``, ``yticks``, ``legend``, and ``figsize_cm``. Use
            ``{"legend": False}`` to hide legends or ``{"xlim": (0, 48)}`` to
            crop the displayed time range.
        :param base_font_size: Base font size in points. Default is ``10.0``.
        :param font_family: Matplotlib font family. Default is ``"Arial"``.
        :param figsize_cm: Optional figure size as ``(width_cm, height_cm)``.
        :param day_night_indicator: Labels for active/inactive background
            bands. Default is ``("awake", "sleep")``. Use ``("aw", "sl")`` for
            compact labels or ``None`` to keep shading without text.
        :returns: The output directory as ``Path``.
        """

        visits = self._prepared_visits(phase_max_hours=phase_max_hours, dayphase=dayphase)
        available_phases = list(self.experiment.phases)
        selected_phases = _normalize_phase_selection(phases, available_phases)
        phase_visits = visits.loc[visits["AnalysisPhaseNumber"].notna()].copy()
        selected_visits = phase_visits.loc[phase_visits["AnalysisPhaseNumber"].astype(int).isin(selected_phases)].copy()
        if selected_visits.empty:
            raise ValueError(f"No visits were available for phases {selected_phases}.")
        self._apply_plot_settings(base_font_size=base_font_size, font_family=font_family)
        layout = PlotLayout.from_mapping(plot_layout)
        destination = Path(output_dir) if output_dir is not None else _binned_output_dir(self.results_data_path, bin_hours, dayphase)
        destination.mkdir(parents=True, exist_ok=True)
        phase_window_table = mt.build_analysis_phase_window_table(selected_visits, self.scheduled_phase_start_hours)
        mouse_bins, summary_bins = mt.compute_experiment_visit_bins(selected_visits, bin_hours=int(bin_hours))
        phase_tag = _phase_file_tag(selected_phases, available_phases)
        prefix = "overview_all_phases_visits" if selected_phases == available_phases else f"mice_activity_{phase_tag}"
        plr.save_table(mouse_bins, destination / f"{prefix}_mouse_bins_{int(bin_hours)}h.tsv")
        plr.save_table(summary_bins, destination / f"{prefix}_group_summary_{int(bin_hours)}h.tsv")
        group_end_hours = (
            selected_visits.groupby("Group", observed=True)["analysis_experiment_elapsed_hours"].max()
            + float(bin_hours)
        ).astype(float).to_dict()
        with _temporary_figsize(("LONG_FIGSIZE_CM", "WIDE_GROUP_FIGSIZE_CM"), layout.figsize_cm or figsize_cm):
            for group_name in plr.ordered_group_names(selected_visits):
                plr.plot_experiment_overview(
                    mouse_bins,
                    summary_bins,
                    group_name=group_name,
                    bin_hours=int(bin_hours),
                    output_path=destination / f"{prefix}_{plr.sanitize_filename_part(group_name)}_{int(bin_hours)}h.png",
                    phase_window_table=phase_window_table,
                    phase_display_names=self.phase_display_names,
                    spread_metric=spread_metric,
                    x_end_hours=layout.xlim[1] if layout.xlim else group_end_hours.get(group_name),
                    plot_style=plot_style,
                    title_label=layout.title or "visits across selected phases",
                    ylabel=layout.ylabel or "Visits per mouse and bin",
                    origin_clock_hour=self.experiment.mouse_day_start_hour,
                    awake_start_clock_hour=self.awake_start_clock_hour,
                    awake_end_clock_hour=self.awake_end_clock_hour,
                    show_legend=True if layout.legend is None else bool(layout.legend),
                    legend_loc=layout.legend_loc,
                    legend_font_size=layout.legend_font_size,
                    xlim=layout.xlim,
                    ylim=layout.ylim,
                    xticks=layout.xticks,
                    yticks=layout.yticks,
                    xlabel=layout.xlabel,
                    day_night_indicator=day_night_indicator)
            plr.plot_experiment_overview_groups(
                summary_bins,
                output_path=destination / f"{prefix}_all_groups_{int(bin_hours)}h.png",
                phase_window_table=phase_window_table,
                phase_display_names=self.phase_display_names,
                spread_metric=spread_metric,
                x_end_hours=layout.xlim[1] if layout.xlim else None,
                plot_style=plot_style,
                title_label=layout.title or f"Group-average visits across selected phases, μ ± {spread_metric.upper()}",
                ylabel=layout.ylabel or "Visits per mouse and bin",
                origin_clock_hour=self.experiment.mouse_day_start_hour,
                awake_start_clock_hour=self.awake_start_clock_hour,
                awake_end_clock_hour=self.awake_end_clock_hour,
                show_legend=True if layout.legend is None else bool(layout.legend),
                legend_loc=layout.legend_loc,
                legend_font_size=layout.legend_font_size,
                xlim=layout.xlim,
                ylim=layout.ylim,
                xticks=layout.xticks,
                yticks=layout.yticks,
                xlabel=layout.xlabel,
                day_night_indicator=day_night_indicator)
        return destination

    def plot_NP_adaptation(
        self,
        *,
        phases: PhaseSelection = 2,
        phase_number: int | None = None,
        bin_hours: int = 1,
        dayphase: DayPhase = "day",
        secondary_metric: str = "lick_positive_visits",
        phase_max_hours: dict[int, float] | None = None,
        spread_metric: SpreadMetric = "sem",
        plot_style: str = "line",
        day_night_indicator: tuple[str, str] | None = ("awake", "sleep"),
        output_dir: Path | None = None,
        plot_layout: PlotLayout | dict | None = None,
        base_font_size: float = 10.0,
        font_family: str = "Arial",
        figsize_cm: tuple[float, float] | None = None) -> Path:
        """Plot visits against drinking visits for selected phases.

        This diagnostic plot compares all visits with visits that contain
        licking. In formula form, the two plotted binned values are
        ``visits_per_bin`` and ``lick_positive_visits_per_bin`` per mouse,
        summarized as group mean plus the selected spread. For selected phases,
        the x-axis is compacted so only requested phases are shown; for example
        ``phases=(2, 4)`` places phase 4 directly after phase 2.

        :param phases: Phase selection. Default is ``2`` for the usual
            nose-poke adaptation phase. Use ``"all"``, an integer, or a tuple of
            integers.
        :param phase_number: Backward-compatible alias for one phase. Default
            is ``None``; prefer ``phases`` in new scripts.
        :param bin_hours: Width of time bins in hours. Default is ``1``.
        :param dayphase: ``"day"``, ``"night"``, or ``"all"``. Default is
            ``"day"``. Use ``"all"`` when the aim is protocol QC.
        :param secondary_metric: Reserved label selector. Default is
            ``"lick_positive_visits"`` and currently plots drinking visits.
        :param phase_max_hours: Optional phase limits in hours. Default is
            ``None``.
        :param spread_metric: ``"sem"`` or ``"std"``. Default is ``"sem"``.
        :param plot_style: ``"line"`` or ``"bar"``. Default is ``"line"``.
        :param output_dir: Optional output folder. Default is a bin/dayphase
            folder below the experiment results folder.
        :param plot_layout: Optional ``dict`` or :class:`PlotLayout`. Supported
            keys are ``title``, ``xlabel``, ``ylabel``, ``xlim``, ``ylim``,
            ``xticks``, ``yticks``, ``legend``, and ``figsize_cm``.
        :param base_font_size: Base font size in points. Default is ``10.0``.
        :param font_family: Matplotlib font family. Default is ``"Arial"``.
        :param figsize_cm: Optional figure size as ``(width_cm, height_cm)``.
        :param day_night_indicator: Background labels. Default is
            ``("awake", "sleep")``; use ``None`` to hide text labels.
        :returns: The output directory as ``Path``.
        """

        selected = phase_number if phase_number is not None else phases
        visits, selected_phases, available_phases = self._selected_phase_visits(
            phases=selected,
            phase_max_hours=phase_max_hours,
            dayphase=dayphase)
        visits, phase_window_table, origin_clock_hour = self._compact_phase_timeline(
            visits,
            selected_phases=selected_phases,
            bin_hours=int(bin_hours))
        destination = Path(output_dir) if output_dir is not None else _binned_output_dir(self.results_data_path, bin_hours, dayphase)
        destination.mkdir(parents=True, exist_ok=True)
        self._apply_plot_settings(base_font_size=base_font_size, font_family=font_family)
        layout = PlotLayout.from_mapping(plot_layout)
        primary_mouse, primary_summary = mt.compute_experiment_visit_bins(visits, bin_hours=int(bin_hours))
        secondary_mouse, secondary_summary = mt.compute_experiment_drinking_visit_bins(visits, bin_hours=int(bin_hours))
        phase_tag = _phase_file_tag(selected_phases, available_phases)
        secondary_label = "Drinking visits" if secondary_metric == "lick_positive_visits" else "Drinking visits"
        file_stub = f"np_adaptation_{phase_tag}_visits_vs_drinking_visits"
        plr.save_table(primary_mouse, destination / f"{file_stub}_visits_mouse_bins_{int(bin_hours)}h.tsv")
        plr.save_table(primary_summary, destination / f"{file_stub}_visits_group_summary_{int(bin_hours)}h.tsv")
        plr.save_table(secondary_mouse, destination / f"{file_stub}_drinking_visits_mouse_bins_{int(bin_hours)}h.tsv")
        plr.save_table(secondary_summary, destination / f"{file_stub}_drinking_visits_group_summary_{int(bin_hours)}h.tsv")
        with _temporary_figsize(("LONG_FIGSIZE_2_CM",), layout.figsize_cm or figsize_cm):
            self._plot_dual_experiment_metric(
                primary_summary,
                secondary_summary,
                destination=destination,
                file_stub=file_stub,
                bin_hours=int(bin_hours),
                secondary_label=secondary_label,
                phase_window_table=phase_window_table,
                spread_metric=spread_metric,
                plot_style=plot_style,
                layout=layout,
                day_night_indicator=day_night_indicator,
                origin_clock_hour=origin_clock_hour)
        return destination

    def plot_NP_counts(
        self,
        *,
        phases: PhaseSelection = "all",
        bin_hours: int = 1,
        dayphase: DayPhase = "day",
        phase_max_hours: dict[int, float] | None = None,
        spread_metric: SpreadMetric = "sem",
        plot_style: str = "line",
        day_night_indicator: tuple[str, str] | None = ("awake", "sleep"),
        output_dir: Path | None = None,
        plot_layout: PlotLayout | dict | None = None,
        base_font_size: float = 10.0,
        font_family: str = "Arial",
        figsize_cm: tuple[float, float] | None = None) -> Path:
        """Plot nose-poke event counts for selected phases.

        The plotted value is the sum of nose-poke events per mouse and time
        bin, summarized by group. Use this alongside learning-rate plots to
        check whether group differences reflect operant engagement rather than
        spatial learning. Selected phases are displayed on a compact x-axis
        without empty gaps for unrequested phases.

        :param phases: Phase selection. Default is ``"all"``. Use one integer
            or a tuple of integers to restrict the plot.
        :param bin_hours: Width of time bins in hours. Default is ``1``.
        :param dayphase: ``"day"``, ``"night"``, or ``"all"``. Default is
            ``"day"``.
        :param phase_max_hours: Optional phase limits in hours. Default is
            ``None``.
        :param spread_metric: ``"sem"`` or ``"std"``. Default is ``"sem"``.
        :param plot_style: ``"line"`` or ``"step"``. Default is ``"line"``.
        :param output_dir: Optional output folder. Default is a bin/dayphase
            folder below the experiment results folder.
        :param plot_layout: Optional ``dict`` or :class:`PlotLayout` with
            ``title``, ``xlabel``, ``ylabel``, ``xlim``, ``ylim``, ``xticks``,
            ``yticks``, ``legend``, and ``figsize_cm``.
        :param base_font_size: Base font size in points. Default is ``10.0``.
        :param font_family: Matplotlib font family. Default is ``"Arial"``.
        :param figsize_cm: Optional figure size as ``(width_cm, height_cm)``.
        :param day_night_indicator: Background labels, default
            ``("awake", "sleep")``. Use ``None`` to hide text labels.
        :returns: The output directory as ``Path``.
        """

        visits, selected_phases, available_phases = self._selected_phase_visits(
            phases=phases,
            phase_max_hours=phase_max_hours,
            dayphase=dayphase)
        visits, phase_window_table, origin_clock_hour = self._compact_phase_timeline(
            visits,
            selected_phases=selected_phases,
            bin_hours=int(bin_hours))
        destination = Path(output_dir) if output_dir is not None else _binned_output_dir(self.results_data_path, bin_hours, dayphase)
        destination.mkdir(parents=True, exist_ok=True)
        self._apply_plot_settings(base_font_size=base_font_size, font_family=font_family)
        layout = PlotLayout.from_mapping(plot_layout)
        mouse_bins, summary_bins = mt.compute_experiment_nosepoke_count_bins(visits, bin_hours=int(bin_hours))
        phase_tag = _phase_file_tag(selected_phases, available_phases)
        file_stub = f"np_counts_{phase_tag}"
        with _temporary_figsize(("LONG_FIGSIZE_CM", "WIDE_GROUP_FIGSIZE_CM"), layout.figsize_cm or figsize_cm):
            self._plot_single_experiment_metric(
                mouse_bins,
                summary_bins,
                destination=destination,
                file_stub=file_stub,
                bin_hours=int(bin_hours),
                phase_window_table=phase_window_table,
                spread_metric=spread_metric,
                plot_style=plot_style,
                title_label=layout.title or "nose-poke counts across selected phases",
                ylabel=layout.ylabel or "Nose pokes per mouse and bin",
                legend_spread_label=False,
                layout=layout,
                day_night_indicator=day_night_indicator,
                origin_clock_hour=origin_clock_hour)
        return destination

    def plot_licking_counts(
        self,
        *,
        phases: PhaseSelection = "all",
        bin_hours: int = 1,
        dayphase: DayPhase = "day",
        phase_max_hours: dict[int, float] | None = None,
        spread_metric: SpreadMetric = "sem",
        plot_style: str = "line",
        output_dir: Path | None = None,
        plot_layout: PlotLayout | dict | None = None,
        base_font_size: float = 10.0,
        font_family: str = "Arial",
        figsize_cm: tuple[float, float] | None = None,
        day_night_indicator: tuple[str, str] | None = ("awake", "sleep")) -> Path:
        """Plot lick counts for selected phases.

        The plotted value is the total number of licks per mouse and time bin,
        summarized by group. This is useful for separating learning effects from
        changes in drinking output, motivation, or bottle interaction. Selected
        phases are displayed on a compact x-axis without empty gaps.

        :param phases: Phase selection. Default is ``"all"``. Use one integer
            or a tuple of integers to restrict the plot.
        :param bin_hours: Width of time bins in hours. Default is ``1``.
        :param dayphase: ``"day"``, ``"night"``, or ``"all"``. Default is
            ``"day"``.
        :param phase_max_hours: Optional phase limits in hours. Default is
            ``None``.
        :param spread_metric: ``"sem"`` or ``"std"``. Default is ``"sem"``.
        :param plot_style: ``"line"`` or ``"step"``. Default is ``"line"``.
        :param output_dir: Optional output folder. Default is a bin/dayphase
            folder below the experiment results folder.
        :param plot_layout: Optional ``dict`` or :class:`PlotLayout` with
            ``title``, ``xlabel``, ``ylabel``, ``xlim``, ``ylim``, ``xticks``,
            ``yticks``, ``legend``, and ``figsize_cm``.
        :param base_font_size: Base font size in points. Default is ``10.0``.
        :param font_family: Matplotlib font family. Default is ``"Arial"``.
        :param figsize_cm: Optional figure size as ``(width_cm, height_cm)``.
        :param day_night_indicator: Background labels, default
            ``("awake", "sleep")``. Use ``None`` to hide text labels.
        :returns: The output directory as ``Path``.
        """

        visits, selected_phases, available_phases = self._selected_phase_visits(
            phases=phases,
            phase_max_hours=phase_max_hours,
            dayphase=dayphase)
        visits, phase_window_table, origin_clock_hour = self._compact_phase_timeline(
            visits,
            selected_phases=selected_phases,
            bin_hours=int(bin_hours))
        destination = Path(output_dir) if output_dir is not None else _binned_output_dir(self.results_data_path, bin_hours, dayphase)
        destination.mkdir(parents=True, exist_ok=True)
        self._apply_plot_settings(base_font_size=base_font_size, font_family=font_family)
        layout = PlotLayout.from_mapping(plot_layout)
        mouse_bins, summary_bins = mt.compute_experiment_lick_count_bins(visits, bin_hours=int(bin_hours))
        phase_tag = _phase_file_tag(selected_phases, available_phases)
        file_stub = f"licking_counts_{phase_tag}"
        with _temporary_figsize(("LONG_FIGSIZE_CM", "WIDE_GROUP_FIGSIZE_CM"), layout.figsize_cm or figsize_cm):
            self._plot_single_experiment_metric(
                mouse_bins,
                summary_bins,
                destination=destination,
                file_stub=file_stub,
                bin_hours=int(bin_hours),
                phase_window_table=phase_window_table,
                spread_metric=spread_metric,
                plot_style=plot_style,
                title_label=layout.title or "lick counts across selected phases",
                ylabel=layout.ylabel or "Licks per mouse and bin",
                legend_spread_label=False,
                layout=layout,
                day_night_indicator=day_night_indicator,
                origin_clock_hour=origin_clock_hour)
        return destination

    def plot_bottle_preference(
        self,
        *,
        phases: PhaseSelection = "all",
        dayphase: DayPhase = "day",
        left_bottle: str = "plain water",
        right_bottle: str = "saccharin",
        calc: str = "left_bottle",
        bin_h: int | float = 24,
        phase_max_hours: dict[int, float] | None = None,
        consumption_col: str = "LickNumber",
        left_sides: tuple[int, ...] | list[int] = (1, 3, 5, 7),
        right_sides: tuple[int, ...] | list[int] = (2, 4, 6, 8),
        spread_metric: SpreadMetric = "sem",
        plot_style: str = "line",
        x_unit: Literal["hours", "days", "weeks"] = "hours",
        indicate_dots: bool = False,
        output_dir: Path | None = None,
        plot_layout: PlotLayout | dict | None = None,
        base_font_size: float = 10.0,
        font_family: str = "Arial",
        figsize_cm: tuple[float, float] | None = None,
        day_night_indicator: tuple[str, str] | None = ("awake", "sleep")) -> Path:
        """Plot left/right bottle consumption or relative bottle preference.

        Bottle use is computed from nose-poke side events. Raw modes plot
        summed consumption for one side class. Relative modes compute
        ``left / (left + right)`` or ``right / (left + right)`` per mouse and
        bin, then plot group mean preference in percent. Binned x positions are
        centered in their bins.

        :param phases: Phase selection. Default is ``"all"``.
        :param dayphase: ``"day"``, ``"night"``, or ``"all"``. Default is
            ``"day"`` and is recommended for preference analyses when inactive
            periods contain sparse visits.
        :param left_bottle: Label for left-side bottle positions. Default is
            ``"plain water"``.
        :param right_bottle: Label for right-side bottle positions. Default is
            ``"saccharin"``.
        :param calc: Calculation mode. Default is ``"left_bottle"``. Accepted
            values are ``"left_bottle"``, ``"right_bottle"``,
            ``"left_bottle/right_bottle"``, and
            ``"right_bottle/left_bottle"``.
        :param bin_h: Bin width in hours. Default is ``24``. Use ``1`` for
            hourly, ``24`` for daily, and ``7 * 24`` for weekly summaries.
        :param phase_max_hours: Optional phase limits in hours. Default is
            ``None``.
        :param consumption_col: Nose-poke column to sum. Default is
            ``"LickNumber"``.
        :param left_sides: Side numbers treated as left bottle positions.
            Default is ``(1, 3, 5, 7)``.
        :param right_sides: Side numbers treated as right bottle positions.
            Default is ``(2, 4, 6, 8)``.
        :param spread_metric: ``"sem"`` or ``"std"``. Default is ``"sem"``.
        :param plot_style: ``"line"`` or ``"step"``. Default is ``"line"``.
        :param x_unit: X-axis unit. Default is ``"hours"``. Accepted values are
            ``"hours"``, ``"days"``, and ``"weeks"``.
        :param indicate_dots: If ``True``, draws a point at each binned mean.
            Default is ``False``. Useful for daily or weekly bins.
        :param output_dir: Optional output folder. Default is a bin/dayphase
            folder below the experiment results folder.
        :param plot_layout: Optional ``dict`` or :class:`PlotLayout` with
            ``title``, ``xlabel``, ``ylabel``, ``xlim``, ``ylim``, ``xticks``,
            ``yticks``, ``legend``, and ``figsize_cm``.
        :param base_font_size: Base font size in points. Default is ``10.0``.
        :param font_family: Matplotlib font family. Default is ``"Arial"``.
        :param figsize_cm: Optional figure size as ``(width_cm, height_cm)``.
        :param day_night_indicator: Background labels, default
            ``("awake", "sleep")``. Labels are shown only when ``x_unit`` is
            ``"hours"``.
        :returns: The output directory as ``Path``.
        """

        visits = self._prepared_visits(phase_max_hours=phase_max_hours, dayphase=dayphase)
        available_phases = list(self.experiment.phases)
        selected_phases = _normalize_phase_selection(phases, available_phases)
        nosepokes = self._analysis_nosepokes_with_timing(phase_max_hours=phase_max_hours, dayphase=dayphase)
        self._apply_plot_settings(base_font_size=base_font_size, font_family=font_family)
        layout = PlotLayout.from_mapping(plot_layout)
        destination = Path(output_dir) if output_dir is not None else _binned_output_dir(self.results_data_path, bin_h, dayphase)
        destination.mkdir(parents=True, exist_ok=True)
        mouse_bins, summary_bins = mt.compute_bottle_preference_bins(
            nosepokes,
            phases=selected_phases,
            bin_h=bin_h,
            left_sides=left_sides,
            right_sides=right_sides,
            consumption_col=consumption_col,
            calc=calc)
        phase_tag = _phase_file_tag(selected_phases, available_phases)
        calc_tag = plr.sanitize_filename_part(calc.replace("/", "_over_"))
        file_stem = f"bottle_preference_{calc_tag}_{phase_tag}_{_bin_file_tag(bin_h)}"
        plr.save_table(mouse_bins, destination / f"{file_stem}_mouse_bins.tsv")
        plr.save_table(summary_bins, destination / f"{file_stem}_group_summary.tsv")
        selected_phase_visits = visits.loc[visits["AnalysisPhaseNumber"].notna()].copy()
        selected_phase_visits = selected_phase_visits.loc[
            selected_phase_visits["AnalysisPhaseNumber"].astype(int).isin(selected_phases)].copy()
        phase_window_table = mt.build_analysis_phase_window_table(
            selected_phase_visits,
            self.scheduled_phase_start_hours)
        plot_bottle_preference_groups(
            summary_bins,
            output_path=destination / f"{file_stem}_all_groups.png",
            left_bottle=left_bottle,
            right_bottle=right_bottle,
            calc=calc,
            bin_h=bin_h,
            spread_metric=spread_metric,
            plot_style=plot_style,
            x_unit=x_unit,
            indicate_dots=indicate_dots,
            figsize_cm=layout.figsize_cm or figsize_cm,
            show_legend=True if layout.legend is None else bool(layout.legend),
            legend_loc=layout.legend_loc,
            legend_font_size=layout.legend_font_size,
            xlim=layout.xlim,
            ylim=layout.ylim,
            xticks=layout.xticks,
            yticks=layout.yticks,
            xlabel=layout.xlabel,
            phase_window_table=phase_window_table,
            phase_display_names=self.phase_display_names,
            title_label=layout.title,
            ylabel=layout.ylabel,
            origin_clock_hour=self.experiment.mouse_day_start_hour,
            awake_start_clock_hour=self.awake_start_clock_hour,
            awake_end_clock_hour=self.awake_end_clock_hour,
            day_night_indicator=day_night_indicator)
        return destination

    def plot_plr_learning_rate(
        self,
        *,
        phase_number: int,
        metric: str = "rewarded_correct_corner_visit",
        bin_hours: int = 1,
        dayphase: DayPhase = "day",
        phase_max_hours: dict[int, float] | None = None,
        spread_metric: SpreadMetric = "sem",
        plot_style: str = "line",
        output_dir: Path | None = None,
        plot_layout: PlotLayout | dict | None = None,
        base_font_size: float = 10.0,
        font_family: str = "Arial",
        figsize_cm: tuple[float, float] | None = None) -> Path:
        """Plot one PLR success-rate metric for one selected phase.

        For each mouse and bin, the rate is
        ``success_visits / all_visits``. The plotted group trace is the mean
        rate in percent with ``SEM`` or ``SD`` shading. Use this for the main
        time-resolved place-learning or reversal-learning readout.

        :param phase_number: Phase number to analyze, for example ``3`` for PL
            or ``4`` for PR in the synthetic example. Required.
        :param metric: Success definition. Default is
            ``"rewarded_correct_corner_visit"``. Accepted values are
            ``"correct_corner_visit"``, ``"correct_np_visit"``, and
            ``"rewarded_correct_corner_visit"``.
        :param bin_hours: Bin width in hours. Default is ``1``.
        :param dayphase: ``"day"``, ``"night"``, or ``"all"``. Default is
            ``"day"``.
        :param phase_max_hours: Optional phase limits in hours. Default is
            ``None``.
        :param spread_metric: ``"sem"`` or ``"std"``. Default is ``"sem"``.
        :param plot_style: ``"line"`` or ``"step"``. Default is ``"line"``.
        :param output_dir: Optional output folder. Default is a bin/dayphase
            folder below the experiment results folder.
        :param plot_layout: Optional ``dict`` or :class:`PlotLayout` with
            ``title``, ``xlabel``, ``ylabel``, ``xlim``, ``ylim``, ``xticks``,
            ``yticks``, ``legend``, and ``figsize_cm``. Use
            ``{"legend": False}`` for manuscript panels with external labels.
        :param base_font_size: Base font size in points. Default is ``10.0``.
        :param font_family: Matplotlib font family. Default is ``"Arial"``.
        :param figsize_cm: Optional figure size as ``(width_cm, height_cm)``.
        :returns: The output directory as ``Path``.
        """

        spec = self._plr_metric_spec(metric)
        visits = self._prepared_visits(phase_max_hours=phase_max_hours, dayphase=dayphase)
        destination = Path(output_dir) if output_dir is not None else _binned_output_dir(self.results_data_path, bin_hours, dayphase)
        destination.mkdir(parents=True, exist_ok=True)
        self._apply_plot_settings(base_font_size=base_font_size, font_family=font_family)
        layout = PlotLayout.from_mapping(plot_layout)
        phase_number = int(phase_number)
        mouse_bins, summary_bins = plr.compute_place_learning_rate_bins(
            visits,
            phase_number=phase_number,
            bin_hours=int(bin_hours),
            success_col=str(spec["success_col"]))
        plr.save_table(mouse_bins, destination / f"phase{phase_number}_{spec['file_stub']}_mouse_bins_{int(bin_hours)}h.tsv")
        plr.save_table(summary_bins, destination / f"phase{phase_number}_{spec['file_stub']}_group_summary_{int(bin_hours)}h.tsv")
        phase_origin_hour, phase_start_day = self._phase_plot_origin(phase_number)
        phase_group_end_hours, phase_end_hours = self._phase_end_hours(visits, bin_hours=int(bin_hours))
        with _temporary_figsize(("MEDIUM_FIGSIZE_CM", "WIDE_GROUP_FIGSIZE_CM"), layout.figsize_cm or figsize_cm):
            for group_name in plr.ordered_group_names(visits):
                plr.plot_phase_learning_rate(
                    mouse_bins,
                    summary_bins,
                    group_name=group_name,
                    phase_number=phase_number,
                    phase_display_name=self.phase_display_names[phase_number],
                    bin_hours=int(bin_hours),
                    output_path=destination / f"phase{phase_number}_{spec['file_stub']}_{plr.sanitize_filename_part(group_name)}_{int(bin_hours)}h.png",
                    spread_metric=spread_metric,
                    title_label=layout.title or str(spec["title_label"]),
                    ylabel=layout.ylabel or str(spec["ylabel"]),
                    chance_level=float(spec["chance_level"]),
                    x_end_hours=layout.xlim[1] if layout.xlim else phase_group_end_hours.get((group_name, phase_number)),
                    plot_style=plot_style,
                    origin_clock_hour=phase_origin_hour,
                    awake_start_clock_hour=self.awake_start_clock_hour,
                    awake_end_clock_hour=self.awake_end_clock_hour,
                    starting_day=phase_start_day,
                    show_legend=True if layout.legend is None else bool(layout.legend),
                    legend_loc=layout.legend_loc,
                    legend_font_size=layout.legend_font_size,
                    xlim=layout.xlim,
                    ylim=layout.ylim,
                    xticks=layout.xticks,
                    yticks=layout.yticks,
                    xlabel=layout.xlabel)
            plr.plot_phase_learning_rate_groups(
                summary_bins,
                phase_number=phase_number,
                phase_display_name=self.phase_display_names[phase_number],
                bin_hours=int(bin_hours),
                output_path=destination / f"phase{phase_number}_{spec['file_stub']}_all_groups_{int(bin_hours)}h.png",
                spread_metric=spread_metric,
                title_label=layout.title or str(spec["title_label"]),
                ylabel=layout.ylabel or str(spec["ylabel"]),
                chance_level=float(spec["chance_level"]),
                x_end_hours=layout.xlim[1] if layout.xlim else phase_end_hours.get(phase_number),
                plot_style=plot_style,
                origin_clock_hour=phase_origin_hour,
                awake_start_clock_hour=self.awake_start_clock_hour,
                awake_end_clock_hour=self.awake_end_clock_hour,
                starting_day=phase_start_day,
                show_legend=True if layout.legend is None else bool(layout.legend),
                legend_loc=layout.legend_loc,
                legend_font_size=layout.legend_font_size,
                xlim=layout.xlim,
                ylim=layout.ylim,
                xticks=layout.xticks,
                yticks=layout.yticks,
                xlabel=layout.xlabel)
        return destination

    def plot_plr_learning_counts(
        self,
        *,
        phase_number: int,
        metric: str = "rewarded_correct_corner_visit",
        bin_hours: int = 1,
        dayphase: DayPhase = "day",
        phase_max_hours: dict[int, float] | None = None,
        spread_metric: SpreadMetric = "sem",
        plot_style: str = "line",
        output_dir: Path | None = None,
        plot_layout: PlotLayout | dict | None = None,
        base_font_size: float = 10.0,
        font_family: str = "Arial",
        figsize_cm: tuple[float, float] | None = None) -> Path:
        """Plot one PLR absolute-count metric for one selected phase.

        For each mouse and bin, this plots the number of successful visits
        rather than a denominator-normalized rate. Use count plots whenever
        visit activity differs between groups and rate plots might hide sparse
        sampling.

        :param phase_number: Phase number to analyze. Required.
        :param metric: Success definition. Default is
            ``"rewarded_correct_corner_visit"``. Accepted values are
            ``"correct_corner_visit"``, ``"correct_np_visit"``, and
            ``"rewarded_correct_corner_visit"``.
        :param bin_hours: Bin width in hours. Default is ``1``.
        :param dayphase: ``"day"``, ``"night"``, or ``"all"``. Default is
            ``"day"``.
        :param phase_max_hours: Optional phase limits in hours. Default is
            ``None``.
        :param spread_metric: ``"sem"`` or ``"std"``. Default is ``"sem"``.
        :param plot_style: ``"line"`` or ``"step"``. Default is ``"line"``.
        :param output_dir: Optional output folder. Default is a bin/dayphase
            folder below the experiment results folder.
        :param plot_layout: Optional ``dict`` or :class:`PlotLayout` with
            ``title``, ``xlabel``, ``ylabel``, ``xlim``, ``ylim``, ``xticks``,
            ``yticks``, ``legend``, and ``figsize_cm``.
        :param base_font_size: Base font size in points. Default is ``10.0``.
        :param font_family: Matplotlib font family. Default is ``"Arial"``.
        :param figsize_cm: Optional figure size as ``(width_cm, height_cm)``.
        :returns: The output directory as ``Path``.
        """

        spec = self._plr_metric_spec(metric)
        visits = self._prepared_visits(phase_max_hours=phase_max_hours, dayphase=dayphase)
        destination = Path(output_dir) if output_dir is not None else _binned_output_dir(self.results_data_path, bin_hours, dayphase)
        destination.mkdir(parents=True, exist_ok=True)
        self._apply_plot_settings(base_font_size=base_font_size, font_family=font_family)
        layout = PlotLayout.from_mapping(plot_layout)
        phase_number = int(phase_number)
        mouse_bins, summary_bins = plr.compute_place_learning_count_bins(
            visits,
            phase_number=phase_number,
            bin_hours=int(bin_hours),
            success_col=str(spec["success_col"]))
        plr.save_table(mouse_bins, destination / f"phase{phase_number}_{spec['count_file_stub']}_mouse_bins_{int(bin_hours)}h.tsv")
        plr.save_table(summary_bins, destination / f"phase{phase_number}_{spec['count_file_stub']}_group_summary_{int(bin_hours)}h.tsv")
        phase_origin_hour, phase_start_day = self._phase_plot_origin(phase_number)
        phase_group_end_hours, phase_end_hours = self._phase_end_hours(visits, bin_hours=int(bin_hours))
        with _temporary_figsize(("MEDIUM_FIGSIZE_CM", "WIDE_GROUP_FIGSIZE_CM"), layout.figsize_cm or figsize_cm):
            for group_name in plr.ordered_group_names(visits):
                plr.plot_phase_learning_counts(
                    mouse_bins,
                    summary_bins,
                    group_name=group_name,
                    phase_number=phase_number,
                    phase_display_name=self.phase_display_names[phase_number],
                    bin_hours=int(bin_hours),
                    output_path=destination / f"phase{phase_number}_{spec['count_file_stub']}_{plr.sanitize_filename_part(group_name)}_{int(bin_hours)}h.png",
                    spread_metric=spread_metric,
                    title_label=layout.title or str(spec["count_title_label"]),
                    ylabel=layout.ylabel or str(spec["count_ylabel"]),
                    x_end_hours=layout.xlim[1] if layout.xlim else phase_group_end_hours.get((group_name, phase_number)),
                    plot_style=plot_style,
                    origin_clock_hour=phase_origin_hour,
                    awake_start_clock_hour=self.awake_start_clock_hour,
                    awake_end_clock_hour=self.awake_end_clock_hour,
                    starting_day=phase_start_day,
                    show_legend=True if layout.legend is None else bool(layout.legend),
                    legend_loc=layout.legend_loc,
                    legend_font_size=layout.legend_font_size,
                    xlim=layout.xlim,
                    ylim=layout.ylim,
                    xticks=layout.xticks,
                    yticks=layout.yticks,
                    xlabel=layout.xlabel)
            plr.plot_phase_learning_counts_groups(
                summary_bins,
                phase_display_name=self.phase_display_names[phase_number],
                bin_hours=int(bin_hours),
                output_path=destination / f"phase{phase_number}_{spec['count_file_stub']}_all_groups_{int(bin_hours)}h.png",
                spread_metric=spread_metric,
                x_end_hours=layout.xlim[1] if layout.xlim else phase_end_hours.get(phase_number),
                plot_style=plot_style,
                title_prefix=layout.title or f"{spec['count_title_label'].capitalize()} across groups",
                ylabel=layout.ylabel or str(spec["count_ylabel"]),
                origin_clock_hour=phase_origin_hour,
                awake_start_clock_hour=self.awake_start_clock_hour,
                awake_end_clock_hour=self.awake_end_clock_hour,
                starting_day=phase_start_day,
                phase_number=phase_number,
                show_legend=True if layout.legend is None else bool(layout.legend),
                legend_loc=layout.legend_loc,
                legend_font_size=layout.legend_font_size,
                xlim=layout.xlim,
                ylim=layout.ylim,
                xticks=layout.xticks,
                yticks=layout.yticks,
                xlabel=layout.xlabel)
        return destination

    def plot_plr_reversal_components(
        self,
        *,
        phase_number: int,
        bin_hours: int = 1,
        dayphase: DayPhase = "day",
        phase_max_hours: dict[int, float] | None = None,
        spread_metric: SpreadMetric = "sem",
        plot_style: str = "line",
        output_dir: Path | None = None,
        plot_layout: PlotLayout | dict | None = None,
        base_font_size: float = 10.0,
        font_family: str = "Arial",
        figsize_cm: tuple[float, float] | None = None) -> Path:
        """Plot new, previous, and neutral corner components for reversal.

        This decomposes reversal-phase corner visits into four mutually
        interpretable rates: new correct corner, previous correct corner, and
        two neutral incorrect corners. Rates are computed as
        ``component_visits / all_visits`` per bin and mouse, then summarized by
        group. It is most useful when phase-level reversal performance is low
        and you need to distinguish weak new learning from perseveration.

        :param phase_number: Reversal phase number. Default-free and required;
            use ``4`` for the synthetic PL/PR example.
        :param bin_hours: Bin width in hours. Default is ``1``.
        :param dayphase: ``"day"``, ``"night"``, or ``"all"``. Default is
            ``"day"``.
        :param phase_max_hours: Optional phase limits in hours. Default is
            ``None``.
        :param spread_metric: ``"sem"`` or ``"std"``. Default is ``"sem"``.
        :param plot_style: ``"line"`` or ``"step"``. Default is ``"line"``.
        :param output_dir: Optional output folder. Default is a bin/dayphase
            folder below the experiment results folder.
        :param plot_layout: Optional ``dict`` or :class:`PlotLayout` with
            ``title``, ``xlabel``, ``ylabel``, ``xlim``, ``ylim``, ``xticks``,
            ``yticks``, ``legend``, and ``figsize_cm``.
        :param base_font_size: Base font size in points. Default is ``10.0``.
        :param font_family: Matplotlib font family. Default is ``"Arial"``.
        :param figsize_cm: Optional figure size as ``(width_cm, height_cm)``.
        :returns: The output directory as ``Path``.
        """

        visits = self._prepared_visits(phase_max_hours=phase_max_hours, dayphase=dayphase)
        destination = Path(output_dir) if output_dir is not None else _binned_output_dir(self.results_data_path, bin_hours, dayphase)
        destination.mkdir(parents=True, exist_ok=True)
        self._apply_plot_settings(base_font_size=base_font_size, font_family=font_family)
        layout = PlotLayout.from_mapping(plot_layout)
        phase_number = int(phase_number)
        rate_tables = plr.compute_phase4_reversal_rate_bins(visits, bin_hours=int(bin_hours), phase_number=phase_number)
        summaries: dict[str, pd.DataFrame] = {}
        for component_name, (mouse_bins, summary_bins) in rate_tables.items():
            plr.save_table(mouse_bins, destination / f"phase{phase_number}_{component_name}_visit_rate_mouse_bins_{int(bin_hours)}h.tsv")
            plr.save_table(summary_bins, destination / f"phase{phase_number}_{component_name}_visit_rate_group_summary_{int(bin_hours)}h.tsv")
            summaries[component_name] = summary_bins
        phase_origin_hour, phase_start_day = self._phase_plot_origin(phase_number)
        phase_group_end_hours, _ = self._phase_end_hours(visits, bin_hours=int(bin_hours))
        with _temporary_figsize(("MEDIUM_WIDE_FIGSIZE_CM",), layout.figsize_cm or figsize_cm):
            for group_name in plr.ordered_group_names(visits):
                plr.plot_phase4_reversal_components(
                    summaries,
                    group_name=group_name,
                    phase_display_name=self.phase_display_names[phase_number],
                    bin_hours=int(bin_hours),
                    output_path=destination / f"phase{phase_number}_reversal_corner_components_{plr.sanitize_filename_part(group_name)}_{int(bin_hours)}h.png",
                    spread_metric=spread_metric,
                    x_end_hours=phase_group_end_hours.get((group_name, phase_number)),
                    plot_style=plot_style,
                    origin_clock_hour=phase_origin_hour,
                    awake_start_clock_hour=self.awake_start_clock_hour,
                    awake_end_clock_hour=self.awake_end_clock_hour,
                    starting_day=phase_start_day,
                    show_legend=True if layout.legend is None else bool(layout.legend),
                    legend_loc=layout.legend_loc,
                    legend_font_size=layout.legend_font_size,
                    xlim=layout.xlim,
                    ylim=layout.ylim,
                    xticks=layout.xticks,
                    yticks=layout.yticks,
                    xlabel=layout.xlabel,
                    ylabel=layout.ylabel,
                    title_label=layout.title)
        return destination

    def plot_phase_activity_summary(
        self,
        *,
        dayphase: DayPhase = "day",
        summary_metric: Literal["mean", "median"] = "mean",
        phase_max_hours: dict[int, float] | None = None,
        output_dir: Path | None = None,
        plot_layout: PlotLayout | dict | None = None,
        show_N: bool = True,
        xtick_rotation: float = 45.0,
        base_font_size: float = 10.0,
        font_family: str = "Arial",
        figsize_cm: tuple[float, float] | None = None) -> Path:
        """Plot phase-wise activity summary boxplots.

        The method summarizes visit activity per mouse and phase as either the
        mean or median number of visits per hour. The mean is usually more
        informative for sparse IntelliCage data because median hourly bins can
        be zero even when mice are active in bursts.

        :param dayphase: ``"day"``, ``"night"``, or ``"all"``. Default is
            ``"day"``.
        :param summary_metric: ``"mean"`` or ``"median"`` visits per hour.
            Default is ``"mean"``.
        :param phase_max_hours: Optional phase limits in hours. Default is
            ``None``.
        :param output_dir: Optional output folder. Default is
            ``results/plr_activity``.
        :param plot_layout: Optional ``dict`` or :class:`PlotLayout`. Supported
            keys are ``title``, ``ylabel``, ``ylim``, ``yticks``, ``legend``,
            ``figsize_cm``, plus ``show_N`` and ``xtick_rotation`` in
            ``extra`` or directly in the dictionary.
        :param show_N: Whether group x tick labels include sample size.
            Default is ``True``.
        :param xtick_rotation: Rotation angle for group labels. Default is
            ``45.0``.
        :param base_font_size: Base font size in points. Default is ``10.0``.
        :param font_family: Matplotlib font family. Default is ``"Arial"``.
        :param figsize_cm: Optional figure size as ``(width_cm, height_cm)``.
        :returns: The output directory as ``Path``.
        """

        visits = self._prepared_visits(phase_max_hours=phase_max_hours, dayphase=dayphase)
        destination = Path(output_dir) if output_dir is not None else self.results_data_path / "plr_activity"
        destination.mkdir(parents=True, exist_ok=True)
        self._apply_plot_settings(base_font_size=base_font_size, font_family=font_family)
        layout = PlotLayout.from_mapping(plot_layout)
        plr.render_phase_activity_plot(
            visits,
            destination,
            phase_display_names=self.phase_display_names,
            summary_metric=summary_metric,
            figsize_cm=layout.figsize_cm or figsize_cm,
            show_legend=True if layout.legend is None else bool(layout.legend),
            legend_loc=layout.legend_loc,
            legend_font_size=layout.legend_font_size,
            show_n=show_N if "show_N" not in layout.extra else bool(layout.extra["show_N"]),
            xtick_rotation=float(layout.extra.get("xtick_rotation", xtick_rotation)),
            ylim=layout.ylim,
            yticks=layout.yticks,
            title=layout.title,
            ylabel=layout.ylabel)
        return destination

    def plot_plr_phase_segment_rate(
        self,
        *,
        phase_number: int,
        metric: str = "rewarded_correct_corner_visit",
        dayphase: DayPhase = "day",
        phase_max_hours: dict[int, float] | None = None,
        spread_metric: SpreadMetric = "sem",
        output_dir: Path | None = None,
        plot_layout: PlotLayout | dict | None = None,
        base_font_size: float = 10.0,
        font_family: str = "Arial",
        figsize_cm: tuple[float, float] | None = None) -> Path:
        """Plot PLR rate across awake/sleep mouse-day segments.

        Each selected phase is split into alternating awake and sleep segments.
        For each mouse and segment, the rate is
        ``success_visits / all_visits``. The figure shows group mean rate over
        sequential mouse-day segments, which helps reveal whether learning is
        concentrated in active periods or continues across the full clock day.

        :param phase_number: Phase number to analyze. Required.
        :param metric: Success definition. Default is
            ``"rewarded_correct_corner_visit"``. Accepted values are
            ``"correct_corner_visit"``, ``"correct_np_visit"``, and
            ``"rewarded_correct_corner_visit"``.
        :param dayphase: ``"day"``, ``"night"``, or ``"all"`` applied before
            segment aggregation. Default is ``"day"``; use ``"all"`` when you
            want the explicit awake/sleep segment comparison.
        :param phase_max_hours: Optional phase limits in hours. Default is
            ``None``.
        :param spread_metric: ``"sem"`` or ``"std"``. Default is ``"sem"``.
        :param output_dir: Optional output folder. Default is
            ``results/plr_segments``.
        :param plot_layout: Optional ``dict`` or :class:`PlotLayout` with
            ``title``, ``ylabel``, ``legend``, and ``figsize_cm``. Axis-limit
            keys are reserved for future support in this plot family.
        :param base_font_size: Base font size in points. Default is ``10.0``.
        :param font_family: Matplotlib font family. Default is ``"Arial"``.
        :param figsize_cm: Optional figure size as ``(width_cm, height_cm)``.
        :returns: The output directory as ``Path``.
        """

        spec = self._plr_metric_spec(metric)
        visits = self._prepared_visits(phase_max_hours=phase_max_hours, dayphase=dayphase)
        destination = Path(output_dir) if output_dir is not None else self.results_data_path / "plr_segments"
        destination.mkdir(parents=True, exist_ok=True)
        self._apply_plot_settings(base_font_size=base_font_size, font_family=font_family)
        layout = PlotLayout.from_mapping(plot_layout)
        phase_number = int(phase_number)
        phase_origin_hour, phase_start_day = self._phase_plot_origin(phase_number)
        mouse_table, summary = plr.compute_phase_segment_rate_tables(
            visits,
            phase_number=phase_number,
            success_col=str(spec["success_col"]),
            origin_clock_hour=phase_origin_hour,
            awake_start_clock_hour=self.awake_start_clock_hour,
            awake_end_clock_hour=self.awake_end_clock_hour,
            max_days=3)
        metric_stub = str(spec["success_col"]).replace("_visit", "")
        plr.save_table(mouse_table, destination / f"phase{phase_number}_{metric_stub}_awake_sleep_segment_rate_mouse.tsv")
        plr.save_table(summary, destination / f"phase{phase_number}_{metric_stub}_awake_sleep_segment_rate_group_summary.tsv")
        with _temporary_figsize(("SEGMENT_FIGSIZE_CM",), layout.figsize_cm or figsize_cm):
            plr.plot_phase_segment_rate_groups(
                summary,
                phase_number=phase_number,
                phase_display_name=self.phase_display_names[phase_number],
                title_label=layout.title or str(spec["title_label"]),
                ylabel=layout.ylabel or str(spec["ylabel"]),
                output_path=destination / f"phase{phase_number}_{metric_stub}_awake_sleep_segment_rate_all_groups.png",
                spread_metric=spread_metric,
                starting_day=phase_start_day,
                chance_level=float(spec["chance_level"]),
                show_legend=True if layout.legend is None else bool(layout.legend),
                legend_loc=layout.legend_loc,
                legend_font_size=layout.legend_font_size)
        return destination

    def plot_plr_phase_segment_error_rate(
        self,
        *,
        phase_number: int,
        metric: str = "rewarded_correct_corner_visit",
        error_against: Literal["selected_success", "spatial_correct"] = "selected_success",
        dayphase: DayPhase = "day",
        phase_max_hours: dict[int, float] | None = None,
        spread_metric: SpreadMetric = "sem",
        output_dir: Path | None = None,
        plot_layout: PlotLayout | dict | None = None,
        base_font_size: float = 10.0,
        font_family: str = "Arial",
        figsize_cm: tuple[float, float] | None = None) -> Path:
        """Plot PLR error rate across awake/sleep mouse-day segments.

        The error rate is the complement of the selected success definition:
        ``(all_visits - success_visits) / all_visits``. By default,
        ``success_visits`` is determined by ``metric``. For rewarded metrics,
        ``error_against="spatial_correct"`` instead defines errors as visits
        to non-correct corners, independent of whether reward delivery occurred.

        :param phase_number: Phase number to analyze. Required.
        :param metric: Success definition used to derive errors. Default is
            ``"rewarded_correct_corner_visit"``.
        :param error_against: ``"selected_success"`` or ``"spatial_correct"``.
            Default is ``"selected_success"``. Use ``"spatial_correct"`` when
            reward delivery should not be counted as part of the error concept.
        :param dayphase: ``"day"``, ``"night"``, or ``"all"`` applied before
            segment aggregation. Default is ``"day"``; use ``"all"`` for the
            explicit awake/sleep segment comparison.
        :param phase_max_hours: Optional phase limits in hours. Default is
            ``None``.
        :param spread_metric: ``"sem"`` or ``"std"``. Default is ``"sem"``.
        :param output_dir: Optional output folder. Default is
            ``results/plr_segments``.
        :param plot_layout: Optional ``dict`` or :class:`PlotLayout` with
            ``title``, ``ylabel``, ``legend``, ``legend_loc``,
            ``legend_font_size``, and ``figsize_cm``.
        :param base_font_size: Base font size in points. Default is ``10.0``.
        :param font_family: Matplotlib font family. Default is ``"Arial"``.
        :param figsize_cm: Optional figure size as ``(width_cm, height_cm)``.
        :returns: The output directory as ``Path``.
        """

        spec = self._plr_error_metric_spec(metric, error_against=error_against)
        visits = self._prepared_visits(phase_max_hours=phase_max_hours, dayphase=dayphase)
        destination = Path(output_dir) if output_dir is not None else self.results_data_path / "plr_segments"
        destination.mkdir(parents=True, exist_ok=True)
        self._apply_plot_settings(base_font_size=base_font_size, font_family=font_family)
        layout = PlotLayout.from_mapping(plot_layout)
        phase_number = int(phase_number)
        phase_origin_hour, phase_start_day = self._phase_plot_origin(phase_number)
        mouse_table, summary = plr.compute_phase_segment_error_rate_tables(
            visits,
            phase_number=phase_number,
            success_col=str(spec["success_col"]),
            origin_clock_hour=phase_origin_hour,
            awake_start_clock_hour=self.awake_start_clock_hour,
            awake_end_clock_hour=self.awake_end_clock_hour,
            max_days=3)
        metric_stub = str(spec["metric_stub"])
        plr.save_table(mouse_table, destination / f"phase{phase_number}_{metric_stub}_awake_sleep_segment_error_rate_mouse.tsv")
        plr.save_table(summary, destination / f"phase{phase_number}_{metric_stub}_awake_sleep_segment_error_rate_group_summary.tsv")
        with _temporary_figsize(("SEGMENT_FIGSIZE_CM",), layout.figsize_cm or figsize_cm):
            plr.plot_phase_segment_rate_groups(
                summary,
                phase_number=phase_number,
                phase_display_name=self.phase_display_names[phase_number],
                title_label=layout.title or str(spec["title_label"]),
                ylabel=layout.ylabel or str(spec["ylabel"]),
                output_path=destination / f"phase{phase_number}_{metric_stub}_awake_sleep_segment_error_rate_all_groups.png",
                spread_metric=spread_metric,
                starting_day=phase_start_day,
                chance_level=float(spec["chance_level"]),
                show_legend=True if layout.legend is None else bool(layout.legend),
                legend_loc=layout.legend_loc,
                legend_font_size=layout.legend_font_size)
        return destination

    def plot_plr_awake_day_rate(
        self,
        *,
        phase_number: int,
        metric: str = "rewarded_correct_corner_visit",
        phase_day: int | tuple[int, ...] = (1, 2, 3),
        dayphase: DayPhase = "day",
        phase_max_hours: dict[int, float] | None = None,
        exclude_outliers: bool = True,
        output_dir: Path | None = None,
        plot_layout: PlotLayout | dict | None = None,
        base_font_size: float = 10.0,
        font_family: str = "Arial",
        figsize_cm: tuple[float, float] | None = None) -> Path:
        """Plot mouse-level PLR rates per phase day and dayphase.

        For each mouse and selected phase day, the endpoint is
        ``success_visits / all_visits`` during the selected dayphase.
        The method writes mouse-level tables, omnibus/pairwise statistics, and
        one violin plot per requested phase day.

        :param phase_number: Phase number to analyze. Required.
        :param metric: Success definition. Default is
            ``"rewarded_correct_corner_visit"``.
        :param phase_day: Phase day or days to plot. Default is ``(1, 2, 3)``.
            Use an integer for one day or a tuple for several days.
        :param dayphase: ``"day"``, ``"night"``, or ``"all"``. Default is
            ``"day"``. ``"day"`` analyzes the awake segment, ``"night"``
            analyzes the sleep segment, and ``"all"`` combines both into a
            full phase-day rate.
        :param phase_max_hours: Optional phase limits in hours. Default is
            ``None``.
        :param exclude_outliers: Whether IQR-flagged outliers are excluded from
            statistics and hidden from violin bodies. Default is ``True``.
        :param output_dir: Optional output folder. Default is
            ``results/plr_endpoints``.
        :param plot_layout: Optional ``dict`` or :class:`PlotLayout`. Supported
            keys are ``title``, ``ylabel``, ``ylim``, and ``figsize_cm``.
        :param base_font_size: Base font size in points. Default is ``10.0``.
        :param font_family: Matplotlib font family. Default is ``"Arial"``.
        :param figsize_cm: Optional figure size as ``(width_cm, height_cm)``.
        :returns: The output directory as ``Path``.
        """

        spec = self._plr_metric_spec(metric)
        visits = self._prepared_visits(phase_max_hours=phase_max_hours, dayphase=dayphase)
        destination = Path(output_dir) if output_dir is not None else self.results_data_path / "plr_endpoints"
        destination.mkdir(parents=True, exist_ok=True)
        self._apply_plot_settings(base_font_size=base_font_size, font_family=font_family)
        layout = PlotLayout.from_mapping(plot_layout)
        phase_number = int(phase_number)
        days = (phase_day,) if isinstance(phase_day, int) else tuple(int(day) for day in phase_day)
        phase_origin_hour, _ = self._phase_plot_origin(phase_number)
        segment_by_dayphase = {"day": "awake", "night": "sleep", "all": "all"}
        segment_label_by_dayphase = {"day": "awake", "night": "sleep", "all": "full day"}
        segment_tag_by_dayphase = {"day": "awake_day", "night": "sleep_day", "all": "full_day"}
        segment_name = segment_by_dayphase[dayphase]
        segment_label = segment_label_by_dayphase[dayphase]
        segment_tag = segment_tag_by_dayphase[dayphase]
        mouse_table, _ = plr.compute_awake_day_rate_tables(
            visits,
            phase_number=phase_number,
            success_col=str(spec["success_col"]),
            origin_clock_hour=phase_origin_hour,
            awake_start_clock_hour=self.awake_start_clock_hour,
            awake_end_clock_hour=self.awake_end_clock_hour,
            max_days=max(days),
            dayphase_segment=segment_name)
        mouse_table["PhaseNumber"] = phase_number
        flagged = plr.flag_iqr_outliers(mouse_table, value_col="value", group_cols=["phase_day", "Group"])
        omnibus, pairwise, chance = plr.compute_group_day_violin_statistics(
            flagged,
            phase_number=phase_number,
            metric_name=str(spec["metric_stub"]),
            chance_level=float(spec["chance_level"]) / 100.0,
            exclude_outliers=exclude_outliers)
        plr.save_table(flagged, destination / f"phase{phase_number}_{spec['metric_stub']}_{segment_tag}_rate_mouse.tsv")
        plr.save_table(omnibus, destination / f"phase{phase_number}_{spec['metric_stub']}_{segment_tag}_rate_omnibus_stats.tsv")
        plr.save_table(pairwise, destination / f"phase{phase_number}_{spec['metric_stub']}_{segment_tag}_rate_pairwise_stats.tsv")
        plr.save_table(chance, destination / f"phase{phase_number}_{spec['metric_stub']}_{segment_tag}_rate_chance_stats.tsv")
        with _temporary_figsize(("VIOLIN_FIGSIZE_CM",), layout.figsize_cm or figsize_cm):
            for current_day in days:
                plr.plot_group_day_violin(
                    flagged,
                    phase_number=phase_number,
                    phase_display_name=self.phase_display_names[phase_number],
                    phase_day=current_day,
                    metric_title=layout.title or str(spec["title_label"]),
                    ylabel=layout.ylabel or str(spec["ylabel"]),
                    pairwise_stats=pairwise,
                    chance_stats=chance,
                    output_path=destination / f"phase{phase_number}_{spec['metric_stub']}_{segment_tag}{current_day}_violin.png",
                    dayphase_label=segment_label,
                    outlier_col="is_outlier")
        return destination

    def plot_plr_awake_day_error_rate(
        self,
        *,
        phase_number: int,
        metric: str = "rewarded_correct_corner_visit",
        error_against: Literal["selected_success", "spatial_correct"] = "selected_success",
        phase_day: int | tuple[int, ...] = (1, 2, 3),
        dayphase: DayPhase = "day",
        phase_max_hours: dict[int, float] | None = None,
        exclude_outliers: bool = True,
        output_dir: Path | None = None,
        plot_layout: PlotLayout | dict | None = None,
        base_font_size: float = 10.0,
        font_family: str = "Arial",
        figsize_cm: tuple[float, float] | None = None) -> Path:
        """Plot mouse-level PLR error rates per phase day and dayphase.

        The endpoint is ``(all_visits - success_visits) / all_visits`` for
        each mouse and selected phase day. ``success_visits`` is usually the
        success column selected by ``metric``. For rewarded metrics,
        ``error_against="spatial_correct"`` switches the complement to
        ``correct_corner_visit`` and therefore asks whether mice chose a wrong
        corner, independent of reward delivery.

        :param phase_number: Phase number to analyze. Required.
        :param metric: Success definition used to derive errors. Default is
            ``"rewarded_correct_corner_visit"``.
        :param error_against: ``"selected_success"`` or ``"spatial_correct"``.
            Default is ``"selected_success"``.
        :param phase_day: Phase day or days to plot. Default is ``(1, 2, 3)``.
        :param dayphase: ``"day"``, ``"night"``, or ``"all"``. Default is
            ``"day"``.
        :param phase_max_hours: Optional phase limits in hours. Default is
            ``None``.
        :param exclude_outliers: Whether IQR-flagged outliers are excluded from
            statistics and hidden from violin bodies. Default is ``True``.
        :param output_dir: Optional output folder. Default is
            ``results/plr_endpoints``.
        :param plot_layout: Optional ``dict`` or :class:`PlotLayout` with
            ``title``, ``ylabel``, ``ylim``, ``legend``, ``legend_loc``,
            ``legend_font_size``, and ``figsize_cm``.
        :param base_font_size: Base font size in points. Default is ``10.0``.
        :param font_family: Matplotlib font family. Default is ``"Arial"``.
        :param figsize_cm: Optional figure size as ``(width_cm, height_cm)``.
        :returns: The output directory as ``Path``.
        """

        spec = self._plr_error_metric_spec(metric, error_against=error_against)
        visits = self._prepared_visits(phase_max_hours=phase_max_hours, dayphase=dayphase)
        destination = Path(output_dir) if output_dir is not None else self.results_data_path / "plr_endpoints"
        destination.mkdir(parents=True, exist_ok=True)
        self._apply_plot_settings(base_font_size=base_font_size, font_family=font_family)
        layout = PlotLayout.from_mapping(plot_layout)
        phase_number = int(phase_number)
        days = (phase_day,) if isinstance(phase_day, int) else tuple(int(day) for day in phase_day)
        phase_origin_hour, _ = self._phase_plot_origin(phase_number)
        segment_by_dayphase = {"day": "awake", "night": "sleep", "all": "all"}
        segment_label_by_dayphase = {"day": "awake", "night": "sleep", "all": "full day"}
        segment_tag_by_dayphase = {"day": "awake_day", "night": "sleep_day", "all": "full_day"}
        segment_name = segment_by_dayphase[dayphase]
        segment_label = segment_label_by_dayphase[dayphase]
        segment_tag = segment_tag_by_dayphase[dayphase]
        mouse_table, _ = plr.compute_awake_day_error_rate_tables(
            visits,
            phase_number=phase_number,
            success_col=str(spec["success_col"]),
            origin_clock_hour=phase_origin_hour,
            awake_start_clock_hour=self.awake_start_clock_hour,
            awake_end_clock_hour=self.awake_end_clock_hour,
            max_days=max(days),
            dayphase_segment=segment_name)
        mouse_table["PhaseNumber"] = phase_number
        flagged = plr.flag_iqr_outliers(mouse_table, value_col="value", group_cols=["phase_day", "Group"])
        omnibus, pairwise, chance = plr.compute_group_day_violin_statistics(
            flagged,
            phase_number=phase_number,
            metric_name=str(spec["metric_stub"]),
            chance_level=float(spec["chance_level"]) / 100.0,
            exclude_outliers=exclude_outliers)
        plr.save_table(flagged, destination / f"phase{phase_number}_{spec['metric_stub']}_{segment_tag}_error_rate_mouse.tsv")
        plr.save_table(omnibus, destination / f"phase{phase_number}_{spec['metric_stub']}_{segment_tag}_error_rate_omnibus_stats.tsv")
        plr.save_table(pairwise, destination / f"phase{phase_number}_{spec['metric_stub']}_{segment_tag}_error_rate_pairwise_stats.tsv")
        plr.save_table(chance, destination / f"phase{phase_number}_{spec['metric_stub']}_{segment_tag}_error_rate_chance_stats.tsv")
        with _temporary_figsize(("VIOLIN_FIGSIZE_CM",), layout.figsize_cm or figsize_cm):
            for current_day in days:
                plr.plot_group_day_violin(
                    flagged,
                    phase_number=phase_number,
                    phase_display_name=self.phase_display_names[phase_number],
                    phase_day=current_day,
                    metric_title=layout.title or str(spec["title_label"]),
                    ylabel=layout.ylabel or str(spec["ylabel"]),
                    pairwise_stats=pairwise,
                    chance_stats=chance,
                    output_path=destination / f"phase{phase_number}_{spec['metric_stub']}_{segment_tag}{current_day}_error_violin.png",
                    dayphase_label=segment_label,
                    outlier_col="is_outlier",
                    reference_line=float(spec["chance_level"]))
        return destination

    def plot_plr_experience_learning_curve(
        self,
        *,
        phase_number: int,
        metric: str = "rewarded_correct_corner_visit",
        dayphase: DayPhase = "day",
        phase_max_hours: dict[int, float] | None = None,
        spread_metric: SpreadMetric = "sem",
        exclude_outliers: bool = True,
        output_dir: Path | None = None,
        plot_layout: PlotLayout | dict | None = None,
        base_font_size: float = 10.0,
        font_family: str = "Arial",
        figsize_cm: tuple[float, float] | None = None) -> Path:
        """Plot experience-normalized PLR learning curves.

        Instead of clock-time bins, this method bins each mouse by visit number
        and computes ``success_visits / visits`` within sliding visit windows.
        It is useful when groups differ in gross activity and you want to ask
        how performance changes as a function of accumulated experience.

        :param phase_number: Phase number to analyze. Required.
        :param metric: Success definition. Default is
            ``"rewarded_correct_corner_visit"``.
        :param dayphase: ``"day"``, ``"night"``, or ``"all"``. Default is
            ``"day"``.
        :param phase_max_hours: Optional phase limits in hours. Default is
            ``None``.
        :param spread_metric: ``"sem"`` or ``"std"``. Default is ``"sem"``.
        :param exclude_outliers: Reserved for API symmetry with onset plots.
            Default is ``True``.
        :param output_dir: Optional output folder. Default is
            ``results/plr_experience``.
        :param plot_layout: Optional ``dict`` or :class:`PlotLayout` with
            ``title``, ``xlabel``, ``ylabel``, ``xlim``, ``ylim``, ``xticks``,
            ``yticks``, ``legend``, and ``figsize_cm``.
        :param base_font_size: Base font size in points. Default is ``10.0``.
        :param font_family: Matplotlib font family. Default is ``"Arial"``.
        :param figsize_cm: Optional figure size as ``(width_cm, height_cm)``.
        :returns: The output directory as ``Path``.
        """

        spec = self._plr_metric_spec(metric)
        visits = self._prepared_visits(phase_max_hours=phase_max_hours, dayphase=dayphase)
        destination = Path(output_dir) if output_dir is not None else self.results_data_path / "plr_experience"
        destination.mkdir(parents=True, exist_ok=True)
        self._apply_plot_settings(base_font_size=base_font_size, font_family=font_family)
        layout = PlotLayout.from_mapping(plot_layout)
        phase_number = int(phase_number)
        curve_mouse, curve_summary, onset_visits = plr.compute_visit_window_learning_curves(
            visits,
            phase_number=phase_number,
            success_col=str(spec["success_col"]))
        plr.save_table(curve_mouse, destination / f"phase{phase_number}_{spec['metric_stub']}_experience_learning_mouse.tsv")
        plr.save_table(curve_summary, destination / f"phase{phase_number}_{spec['metric_stub']}_experience_learning_group_summary.tsv")
        plr.save_table(onset_visits, destination / f"phase{phase_number}_{spec['metric_stub']}_experience_learning_onset_mouse.tsv")
        plr.plot_visit_learning_curve_groups(
            curve_summary,
            phase_display_name=self.phase_display_names[phase_number],
            title_label=layout.title or f"{spec['title_label']} by visit number",
            ylabel=layout.ylabel or "Success probability [%]",
            output_path=destination / f"phase{phase_number}_{spec['metric_stub']}_experience_learning_all_groups.png",
            spread_metric=spread_metric,
            chance_level=float(spec["chance_level"]),
            show_legend=True if layout.legend is None else bool(layout.legend),
            legend_loc=layout.legend_loc,
            legend_font_size=layout.legend_font_size,
            xlim=layout.xlim,
            ylim=layout.ylim,
            xticks=layout.xticks,
            yticks=layout.yticks,
            xlabel=layout.xlabel,
            figsize_cm=layout.figsize_cm or figsize_cm)
        return destination

    def plot_plr_experience_learning_onset(
        self,
        *,
        phase_number: int,
        metric: str = "rewarded_correct_corner_visit",
        dayphase: DayPhase = "day",
        phase_max_hours: dict[int, float] | None = None,
        exclude_outliers: bool = True,
        output_dir: Path | None = None,
        plot_layout: PlotLayout | dict | None = None,
        show_N: bool = True,
        base_font_size: float = 10.0,
        font_family: str = "Arial",
        figsize_cm: tuple[float, float] | None = None) -> Path:
        """Plot experience-normalized PLR learning-onset distributions.

        Learning onset is the first visit-window position at which a mouse
        reaches the internal success criterion for the selected metric. The
        result is a mouse-level onset value in visit number, plotted as group
        violins with optional outlier handling and pairwise statistics.

        :param phase_number: Phase number to analyze. Required.
        :param metric: Success definition. Default is
            ``"rewarded_correct_corner_visit"``.
        :param dayphase: ``"day"``, ``"night"``, or ``"all"``. Default is
            ``"day"``.
        :param phase_max_hours: Optional phase limits in hours. Default is
            ``None``.
        :param exclude_outliers: Whether IQR-flagged outliers are excluded from
            statistics and hidden from violin bodies. Default is ``True``.
        :param output_dir: Optional output folder. Default is
            ``results/plr_experience``.
        :param plot_layout: Optional ``dict`` or :class:`PlotLayout`. Supported
            keys are ``title``, ``ylabel``, ``ylim``, ``figsize_cm``, and
            ``show_N`` via ``extra`` or directly in the dictionary.
        :param show_N: Whether group x tick labels include sample size.
            Default is ``True``.
        :param base_font_size: Base font size in points. Default is ``10.0``.
        :param font_family: Matplotlib font family. Default is ``"Arial"``.
        :param figsize_cm: Optional figure size as ``(width_cm, height_cm)``.
        :returns: The output directory as ``Path``.
        """

        spec = self._plr_metric_spec(metric)
        visits = self._prepared_visits(phase_max_hours=phase_max_hours, dayphase=dayphase)
        destination = Path(output_dir) if output_dir is not None else self.results_data_path / "plr_experience"
        destination.mkdir(parents=True, exist_ok=True)
        self._apply_plot_settings(base_font_size=base_font_size, font_family=font_family)
        layout = PlotLayout.from_mapping(plot_layout)
        phase_number = int(phase_number)
        _, _, onset_visits = plr.compute_visit_window_learning_curves(
            visits,
            phase_number=phase_number,
            success_col=str(spec["success_col"]))
        flagged = plr.flag_iqr_outliers(onset_visits, value_col="onset_visit", group_cols=["Group"])
        omnibus, pairwise = plr.compute_onset_group_statistics(
            flagged,
            onset_col="onset_visit",
            phase_number=phase_number,
            metric_name=str(spec["metric_stub"]),
            exclude_outliers=exclude_outliers)
        plr.save_table(flagged, destination / f"phase{phase_number}_{spec['metric_stub']}_experience_learning_onset_with_outliers.tsv")
        plr.save_table(omnibus, destination / f"phase{phase_number}_{spec['metric_stub']}_experience_learning_onset_omnibus_stats.tsv")
        plr.save_table(pairwise, destination / f"phase{phase_number}_{spec['metric_stub']}_experience_learning_onset_pairwise_stats.tsv")
        plr.plot_onset_violin(
            flagged,
            onset_col="onset_visit",
            phase_display_name=self.phase_display_names[phase_number],
            title_label=layout.title or f"{spec['title_label']} onset",
            ylabel=layout.ylabel or "Learning onset [visit number]",
            output_path=destination / f"phase{phase_number}_{spec['metric_stub']}_experience_learning_onset_violin.png",
            pairwise_stats=pairwise,
            outlier_col="is_outlier",
            y_limits=layout.ylim,
            show_n=show_N if "show_N" not in layout.extra else bool(layout.extra["show_N"]),
            figsize_cm=layout.figsize_cm or figsize_cm)
        return destination

    def plot_plr_threshold_onset(
        self,
        *,
        phase_number: int,
        metric: str = "rewarded_correct_corner_visit",
        threshold_pct: float = 70.0,
        bin_hours: int = 1,
        dayphase: DayPhase = "day",
        phase_max_hours: dict[int, float] | None = None,
        exclude_outliers: bool = True,
        output_dir: Path | None = None,
        plot_layout: PlotLayout | dict | None = None,
        base_font_size: float = 10.0,
        font_family: str = "Arial",
        figsize_cm: tuple[float, float] | None = None) -> Path:
        """Plot first threshold-crossing time for a binned PLR rate.

        For each mouse, the selected success rate is binned in clock time. The
        onset value is the first bin start at which the rate reaches or exceeds
        ``threshold_pct``. This is useful for comparing learning speed rather
        than final performance.

        :param phase_number: Phase number to analyze. Required.
        :param metric: Success definition. Default is
            ``"rewarded_correct_corner_visit"``.
        :param threshold_pct: Threshold in percent. Default is ``70.0``. Use
            lower values such as ``50`` for permissive responder screens and
            higher values such as ``80`` for robust learning.
        :param bin_hours: Bin width in hours. Default is ``1``.
        :param dayphase: ``"day"``, ``"night"``, or ``"all"``. Default is
            ``"day"``.
        :param phase_max_hours: Optional phase limits in hours. Default is
            ``None``.
        :param exclude_outliers: Whether IQR-flagged outliers are excluded from
            statistics and hidden from violin bodies. Default is ``True``.
        :param output_dir: Optional output folder. Default is
            ``results/plr_thresholds``.
        :param plot_layout: Optional ``dict`` or :class:`PlotLayout`. Supported
            keys are ``title``, ``ylabel``, ``ylim``, and ``figsize_cm``.
        :param base_font_size: Base font size in points. Default is ``10.0``.
        :param font_family: Matplotlib font family. Default is ``"Arial"``.
        :param figsize_cm: Optional figure size as ``(width_cm, height_cm)``.
        :returns: The output directory as ``Path``.
        """

        spec = self._plr_metric_spec(metric)
        visits = self._prepared_visits(phase_max_hours=phase_max_hours, dayphase=dayphase)
        destination = Path(output_dir) if output_dir is not None else self.results_data_path / "plr_thresholds"
        destination.mkdir(parents=True, exist_ok=True)
        self._apply_plot_settings(base_font_size=base_font_size, font_family=font_family)
        layout = PlotLayout.from_mapping(plot_layout)
        phase_number = int(phase_number)
        threshold_tag = f"{int(threshold_pct)}pct"
        onset_table = plr.compute_rate_threshold_onset_table(
            visits,
            phase_number=phase_number,
            success_col=str(spec["success_col"]),
            bin_hours=int(bin_hours),
            threshold_pct=float(threshold_pct))
        flagged = plr.flag_iqr_outliers(onset_table, value_col="onset_hours", group_cols=["Group"])
        omnibus, pairwise = plr.compute_onset_group_statistics(
            flagged,
            onset_col="onset_hours",
            phase_number=phase_number,
            metric_name=f"{spec['metric_stub']}_{int(bin_hours)}h_threshold_onset_gt_{threshold_tag}",
            exclude_outliers=exclude_outliers)
        plr.save_table(flagged, destination / f"phase{phase_number}_{spec['metric_stub']}_{int(bin_hours)}h_threshold_onset_{threshold_tag}_mouse_with_outliers.tsv")
        plr.save_table(omnibus, destination / f"phase{phase_number}_{spec['metric_stub']}_{int(bin_hours)}h_threshold_onset_{threshold_tag}_omnibus_stats.tsv")
        plr.save_table(pairwise, destination / f"phase{phase_number}_{spec['metric_stub']}_{int(bin_hours)}h_threshold_onset_{threshold_tag}_pairwise_stats.tsv")
        with _temporary_figsize(("ONSET_FIGSIZE_CM",), layout.figsize_cm or figsize_cm):
            plr.plot_onset_violin(
                flagged,
                onset_col="onset_hours",
                phase_display_name=self.phase_display_names[phase_number],
                title_label=layout.title or f"{spec['title_label']} first exceeds {int(threshold_pct)}%",
                ylabel=layout.ylabel or "Threshold onset [hours]",
                output_path=destination / f"phase{phase_number}_{spec['metric_stub']}_{int(bin_hours)}h_threshold_onset_{threshold_tag}_violin.png",
                pairwise_stats=pairwise,
                outlier_col="is_outlier")
        return destination

    def plot_plr_derived_ratio(
        self,
        *,
        phase_number: int,
        numerator_col: str = "rewarded_correct_corner_visit",
        denominator_col: str = "correct_np_visit",
        metric_name: str = "completion_efficiency",
        title: str = "completion efficiency",
        ylabel: str = "Rewarded correct / correct NP [%]",
        phase_day: int | tuple[int, ...] = (1, 2, 3),
        dayphase: DayPhase = "day",
        phase_max_hours: dict[int, float] | None = None,
        pseudocount: float = 0.0,
        value_scale: float = 100.0,
        format_as_percent: bool = True,
        reference_line: float | None = None,
        exclude_outliers: bool = True,
        output_dir: Path | None = None,
        plot_layout: PlotLayout | dict | None = None,
        base_font_size: float = 10.0,
        font_family: str = "Arial",
        figsize_cm: tuple[float, float] | None = None) -> Path:
        """Plot a phase/day-wise derived ratio endpoint.

        The endpoint is computed per mouse as
        ``numerator / denominator * value_scale`` within each selected phase
        day. It supports custom ratios such as completion efficiency
        (rewarded correct visits divided by correct nose-poke visits) or
        reversal preference (new correct visits divided by new plus previous
        correct visits).

        :param phase_number: Phase number to analyze. Required.
        :param numerator_col: Visit-level numerator column. Default is
            ``"rewarded_correct_corner_visit"``.
        :param denominator_col: Visit-level denominator column. Default is
            ``"correct_np_visit"``. The special value
            ``"new_or_previous_correct_corner_visit"`` is built internally for
            reversal preference.
        :param metric_name: Filename-safe metric name. Default is
            ``"completion_efficiency"``.
        :param title: Figure title stem. Default is ``"completion efficiency"``.
        :param ylabel: Y-axis label. Default is
            ``"Rewarded correct / correct NP [%]"``.
        :param phase_day: Phase day or days to plot. Default is ``(1, 2, 3)``.
        :param dayphase: ``"day"``, ``"night"``, or ``"all"``. Default is
            ``"day"``.
        :param phase_max_hours: Optional phase limits in hours. Default is
            ``None``.
        :param pseudocount: Value added to numerator and denominator before
            division. Default is ``0.0``. Use only when zero-denominator mice
            should remain in exploratory plots.
        :param value_scale: Multiplicative scale. Default is ``100.0`` for
            percent-like outputs; use ``1.0`` for bounded ratios.
        :param format_as_percent: Whether to format the y-axis as percent.
            Default is ``True``.
        :param reference_line: Optional horizontal reference line. Default is
            ``None``. Use ``0.5`` for reversal preference when
            ``value_scale=1.0``.
        :param exclude_outliers: Whether IQR-flagged outliers are excluded from
            statistics. Default is ``True``.
        :param output_dir: Optional output folder. Default is
            ``results/plr_derived``.
        :param plot_layout: Optional ``dict`` or :class:`PlotLayout`. Supported
            keys are ``title``, ``ylabel``, ``ylim``, ``legend``,
            ``legend_loc``, ``legend_font_size``, and ``figsize_cm``.
        :param base_font_size: Base font size in points. Default is ``10.0``.
        :param font_family: Matplotlib font family. Default is ``"Arial"``.
        :param figsize_cm: Optional figure size as ``(width_cm, height_cm)``.
        :returns: The output directory as ``Path``.
        """

        visits = self._prepared_visits(phase_max_hours=phase_max_hours, dayphase=dayphase)
        destination = Path(output_dir) if output_dir is not None else self.results_data_path / "plr_derived"
        destination.mkdir(parents=True, exist_ok=True)
        self._apply_plot_settings(base_font_size=base_font_size, font_family=font_family)
        layout = PlotLayout.from_mapping(plot_layout)
        phase_number = int(phase_number)
        days = (phase_day,) if isinstance(phase_day, int) else tuple(int(day) for day in phase_day)
        segment_by_dayphase = {"day": "awake", "night": "sleep", "all": "all"}
        segment_label_by_dayphase = {"day": "awake", "night": "sleep", "all": "full day"}
        segment_tag_by_dayphase = {"day": "awake_day", "night": "sleep_day", "all": "full_day"}
        segment_name = segment_by_dayphase[dayphase]
        segment_label = segment_label_by_dayphase[dayphase]
        segment_tag = segment_tag_by_dayphase[dayphase]
        ratio_visits = visits.copy()
        if denominator_col == "new_or_previous_correct_corner_visit" and denominator_col not in ratio_visits:
            ratio_visits[denominator_col] = (
                ratio_visits["correct_corner_visit"].fillna(False).astype(bool)
                | ratio_visits["previous_correct_corner_visit"].fillna(False).astype(bool))
        phase_origin_hour, _ = self._phase_plot_origin(phase_number)
        mouse_table, _ = plr.compute_awake_day_ratio_tables(
            ratio_visits,
            phase_number=phase_number,
            numerator_col=numerator_col,
            denominator_col=denominator_col,
            origin_clock_hour=phase_origin_hour,
            awake_start_clock_hour=self.awake_start_clock_hour,
            awake_end_clock_hour=self.awake_end_clock_hour,
            max_days=max(days),
            pseudocount=float(pseudocount),
            dayphase_segment=segment_name)
        if mouse_table.empty:
            return destination
        mouse_table["PhaseNumber"] = phase_number
        flagged = plr.flag_iqr_outliers(mouse_table, value_col="value", group_cols=["Group", "phase_day"])
        omnibus, pairwise, _ = plr.compute_group_day_violin_statistics(
            flagged,
            phase_number=phase_number,
            metric_name=metric_name,
            chance_level=0.25,
            exclude_outliers=exclude_outliers)
        plr.save_table(flagged, destination / f"phase{phase_number}_{metric_name}_{segment_tag}_rate_mouse_with_outliers.tsv")
        plr.save_table(omnibus, destination / f"phase{phase_number}_{metric_name}_{segment_tag}_rate_omnibus_stats.tsv")
        plr.save_table(pairwise, destination / f"phase{phase_number}_{metric_name}_{segment_tag}_rate_pairwise_stats.tsv")
        with _temporary_figsize(("VIOLIN_FIGSIZE_CM",), layout.figsize_cm or figsize_cm):
            for current_day in days:
                plr.plot_group_day_violin(
                    flagged,
                    phase_number=phase_number,
                    phase_display_name=self.phase_display_names[phase_number],
                    phase_day=current_day,
                    metric_title=layout.title or title,
                    ylabel=layout.ylabel or ylabel,
                    pairwise_stats=pairwise,
                    chance_stats=None,
                    output_path=destination / f"phase{phase_number}_{metric_name}_{segment_tag}{current_day}_violin.png",
                    dayphase_label=segment_label,
                    reference_line=reference_line,
                    value_scale=float(value_scale),
                    format_as_percent=format_as_percent,
                    y_limits=layout.ylim)
        return destination

    def plot_plr_cumulative_preferences(
        self,
        *,
        phases: PhaseSelection = (2, 3, 4),
        dayphase: DayPhase = "day",
        phase_max_hours: dict[int, float] | None = None,
        spread_metric: SpreadMetric = "sem",
        plot_style: str = "line",
        day_night_indicator: tuple[str, str] | None = ("awake", "sleep"),
        output_dir: Path | None = None,
        plot_layout: PlotLayout | dict | None = None,
        base_font_size: float = 10.0,
        font_family: str = "Arial",
        figsize_cm: tuple[float, float] | None = None) -> Path:
        """Plot cumulative corner-role preference trajectories.

        The method tracks cumulative visits to role-defined corners across the
        selected phases and plots relative preference trajectories by group.
        Roles include current correct, previous correct, and neutral incorrect
        corners where those roles are defined. This is useful for visualizing
        how place-learning and reversal histories unfold across phases.

        :param phases: Phase selection. Default is ``(2, 3, 4)`` for NPA, PL,
            and PR in the synthetic example. Use ``"all"``, one integer, or a
            tuple of integers.
        :param dayphase: ``"day"``, ``"night"``, or ``"all"``. Default is
            ``"day"``.
        :param phase_max_hours: Optional phase limits in hours. Default is
            ``None``.
        :param spread_metric: ``"sem"`` or ``"std"``. Default is ``"sem"``.
        :param plot_style: ``"line"`` or ``"step"``. Default is ``"line"``.
        :param day_night_indicator: Labels for the active/inactive background
            bands. Default is ``("awake", "sleep")``. Use shorter labels such
            as ``("aw", "sl")`` for compact figures, or ``None`` to hide these
            labels while keeping the shaded bands.
        :param output_dir: Optional output folder. Default is
            ``results/plr_cumulative``.
        :param plot_layout: Optional ``dict`` or :class:`PlotLayout` with
            ``legend``, ``legend_loc``, ``legend_font_size``, and
            ``figsize_cm``.
        :param base_font_size: Base font size in points. Default is ``10.0``.
        :param font_family: Matplotlib font family. Default is ``"Arial"``.
        :param figsize_cm: Optional figure size as ``(width_cm, height_cm)``.
        :returns: The output directory as ``Path``.
        """

        visits = self._prepared_visits(phase_max_hours=phase_max_hours, dayphase=dayphase)
        available_phases = list(self.experiment.phases)
        selected_phases = _normalize_phase_selection(phases, available_phases)
        phase_visits = visits.loc[visits["AnalysisPhaseNumber"].notna()].copy()
        phase_visits = phase_visits.loc[phase_visits["AnalysisPhaseNumber"].astype(int).isin(selected_phases)].copy()
        destination = Path(output_dir) if output_dir is not None else self.results_data_path / "plr_cumulative"
        destination.mkdir(parents=True, exist_ok=True)
        self._apply_plot_settings(base_font_size=base_font_size, font_family=font_family)
        layout = PlotLayout.from_mapping(plot_layout)
        with _temporary_figsize(("MEDIUM_WIDE_FIGSIZE_CM",), layout.figsize_cm or figsize_cm):
            plr.render_cumulative_role_plots(
                phase_visits,
                destination,
                group_names=plr.ordered_group_names(phase_visits),
                plot_style=plot_style,
                phase_display_names=self.phase_display_names,
                spread_metric=spread_metric,
                scheduled_phase_start_hours=self.scheduled_phase_start_hours,
                mouse_day_start_hour=self.experiment.mouse_day_start_hour,
                awake_end_clock_hour=self.awake_end_clock_hour,
                selected_phase_numbers=selected_phases,
                day_night_indicator=day_night_indicator,
                show_legend=True if layout.legend is None else bool(layout.legend),
                legend_loc=layout.legend_loc,
                legend_font_size=layout.legend_font_size)
        return destination

    def _selected_phase_visits(
        self,
        *,
        phases: PhaseSelection,
        phase_max_hours: dict[int, float] | None = None,
        dayphase: DayPhase = "all") -> tuple[pd.DataFrame, list[int], list[int]]:
        """Return prepared visits restricted to one normalized phase selection."""

        visits = self._prepared_visits(phase_max_hours=phase_max_hours, dayphase=dayphase)
        available_phases = list(self.experiment.phases)
        selected_phases = _normalize_phase_selection(phases, available_phases)
        phase_visits = visits.loc[visits["AnalysisPhaseNumber"].notna()].copy()
        selected_visits = phase_visits.loc[
            phase_visits["AnalysisPhaseNumber"].astype(int).isin(selected_phases)].copy()
        if selected_visits.empty:
            raise ValueError(f"No visits were available for phases {selected_phases}.")
        return selected_visits, selected_phases, available_phases

    def _compact_phase_timeline(
        self,
        visits: pd.DataFrame,
        *,
        selected_phases: list[int],
        bin_hours: int) -> tuple[pd.DataFrame, pd.DataFrame, float]:
        """Rewrite experiment elapsed time so selected phases appear without gaps."""

        compact = visits.copy()
        phase_rows: list[dict[str, float | int]] = []
        offset_hours = 0.0
        for phase_number in selected_phases:
            phase_mask = compact["AnalysisPhaseNumber"].astype(int).eq(int(phase_number))
            phase_data = compact.loc[phase_mask]
            if phase_data.empty:
                continue
            phase_elapsed = phase_data["analysis_phase_elapsed_hours"].astype(float)
            compact.loc[phase_mask, "analysis_experiment_elapsed_hours"] = phase_elapsed + offset_hours
            duration_hours = float(phase_elapsed.max()) + float(bin_hours)
            duration_hours = max(float(bin_hours), duration_hours)
            phase_rows.append({
                "PhaseNumber": int(phase_number),
                "start_hours": offset_hours,
                "end_hours": offset_hours + duration_hours})
            offset_hours += duration_hours
        if not phase_rows:
            raise ValueError(f"No visits were available for phases {selected_phases}.")
        first_phase_start = self.scheduled_phase_start_hours[int(selected_phases[0])]
        return (
            compact,
            pd.DataFrame(phase_rows),
            plr.phase_origin_clock_hour(self.experiment.mouse_day_start_hour, first_phase_start))

    def _plot_single_experiment_metric(
        self,
        mouse_bins: pd.DataFrame,
        summary_bins: pd.DataFrame,
        *,
        destination: Path,
        file_stub: str,
        bin_hours: int,
        phase_window_table: pd.DataFrame,
        spread_metric: SpreadMetric,
        plot_style: str,
        title_label: str,
        ylabel: str,
        layout: PlotLayout,
        day_night_indicator: tuple[str, str] | None,
        origin_clock_hour: float,
        legend_spread_label: bool = True) -> None:
        """Save and plot one full-timeline single metric."""

        plr.save_table(mouse_bins, destination / f"{file_stub}_mouse_bins_{bin_hours}h.tsv")
        plr.save_table(summary_bins, destination / f"{file_stub}_group_summary_{bin_hours}h.tsv")
        group_end_hours = summary_bins.groupby("Group", observed=True)["bin_end_hours"].max().astype(float).to_dict()
        all_group_end = float(summary_bins["bin_end_hours"].max())
        for group_name in plr.ordered_group_names(summary_bins):
            default_xlim = None if group_name not in group_end_hours else (0.0, group_end_hours[group_name])
            plr.plot_experiment_overview(
                mouse_bins,
                summary_bins,
                group_name=group_name,
                bin_hours=bin_hours,
                output_path=destination / f"{file_stub}_{plr.sanitize_filename_part(group_name)}_{bin_hours}h.png",
                phase_window_table=phase_window_table,
                phase_display_names=self.phase_display_names,
                spread_metric=spread_metric,
                legend_spread_label=legend_spread_label,
                x_end_hours=layout.xlim[1] if layout.xlim else group_end_hours.get(group_name),
                plot_style=plot_style,
                show_individual_labels=False,
                title_label=title_label,
                ylabel=ylabel,
                origin_clock_hour=origin_clock_hour,
                awake_start_clock_hour=self.awake_start_clock_hour,
                awake_end_clock_hour=self.awake_end_clock_hour,
                show_legend=True if layout.legend is None else bool(layout.legend),
                legend_inside=not legend_spread_label,
                legend_loc=layout.legend_loc,
                legend_font_size=layout.legend_font_size,
                xlim=layout.xlim or default_xlim,
                ylim=layout.ylim,
                xticks=layout.xticks,
                yticks=layout.yticks,
                xlabel=layout.xlabel,
                day_night_indicator=day_night_indicator)
        plr.plot_experiment_overview_groups(
            summary_bins,
            output_path=destination / f"{file_stub}_all_groups_{bin_hours}h.png",
            phase_window_table=phase_window_table,
            phase_display_names=self.phase_display_names,
            spread_metric=spread_metric,
            legend_spread_label=legend_spread_label,
            x_end_hours=layout.xlim[1] if layout.xlim else all_group_end,
            plot_style=plot_style,
            title_label=layout.title or f"{title_label} by group, μ ± {spread_metric.upper()}",
            ylabel=ylabel,
            origin_clock_hour=origin_clock_hour,
            awake_start_clock_hour=self.awake_start_clock_hour,
            awake_end_clock_hour=self.awake_end_clock_hour,
            show_legend=True if layout.legend is None else bool(layout.legend),
            legend_inside=not legend_spread_label,
            legend_loc=layout.legend_loc,
            legend_font_size=layout.legend_font_size,
            xlim=layout.xlim or (0.0, all_group_end),
            ylim=layout.ylim,
            xticks=layout.xticks,
            yticks=layout.yticks,
            xlabel=layout.xlabel,
            day_night_indicator=day_night_indicator)

    def _plot_dual_experiment_metric(
        self,
        primary_summary: pd.DataFrame,
        secondary_summary: pd.DataFrame,
        *,
        destination: Path,
        file_stub: str,
        bin_hours: int,
        secondary_label: str,
        phase_window_table: pd.DataFrame,
        spread_metric: SpreadMetric,
        plot_style: str,
        layout: PlotLayout,
        day_night_indicator: tuple[str, str] | None,
        origin_clock_hour: float) -> None:
        """Plot one visits-versus-secondary full-timeline metric."""

        group_end_hours = primary_summary.groupby("Group", observed=True)["bin_end_hours"].max().astype(float).to_dict()
        for group_name in plr.ordered_group_names(primary_summary):
            plr.plot_experiment_dual_metric_bars(
                primary_summary,
                secondary_summary,
                group_name=group_name,
                bin_hours=bin_hours,
                output_path=destination / f"{file_stub}_{plr.sanitize_filename_part(group_name)}_{bin_hours}h.png",
                secondary_label=secondary_label,
                phase_window_table=phase_window_table,
                phase_display_names=self.phase_display_names,
                plot_style=plot_style,
                spread_metric=spread_metric,
                origin_clock_hour=origin_clock_hour,
                awake_start_clock_hour=self.awake_start_clock_hour,
                awake_end_clock_hour=self.awake_end_clock_hour,
                show_legend=True if layout.legend is None else bool(layout.legend),
                legend_loc=layout.legend_loc,
                legend_font_size=layout.legend_font_size,
                xlim=layout.xlim if layout.xlim else (None if group_name not in group_end_hours else (0.0, group_end_hours[group_name])),
                ylim=layout.ylim,
                xticks=layout.xticks,
                yticks=layout.yticks,
                xlabel=layout.xlabel,
                ylabel=layout.ylabel,
                title_label=layout.title,
                day_night_indicator=day_night_indicator)
        plr.plot_experiment_dual_metric_groups(
            primary_summary,
            secondary_summary,
            group_names=plr.ordered_group_names(primary_summary),
            bin_hours=bin_hours,
            output_path=destination / f"{file_stub}_all_groups_{bin_hours}h.png",
            secondary_label=secondary_label,
            phase_window_table=phase_window_table,
            phase_display_names=self.phase_display_names,
                origin_clock_hour=origin_clock_hour,
            awake_start_clock_hour=self.awake_start_clock_hour,
            awake_end_clock_hour=self.awake_end_clock_hour,
            show_legend=True if layout.legend is None else bool(layout.legend),
            legend_loc=layout.legend_loc,
            legend_font_size=layout.legend_font_size,
            xlim=layout.xlim,
            ylim=layout.ylim,
            xticks=layout.xticks,
            yticks=layout.yticks,
            xlabel=layout.xlabel,
            ylabel=layout.ylabel,
            title_label=layout.title,
            day_night_indicator=day_night_indicator)

    def _plr_error_metric_spec(
        self,
        metric: str,
        *,
        error_against: Literal["selected_success", "spatial_correct"] = "selected_success") -> dict[str, str | float]:
        """Return labels and filenames for one PLR error-rate metric."""

        if error_against not in {"selected_success", "spatial_correct"}:
            raise ValueError("`error_against` must be 'selected_success' or 'spatial_correct'.")
        spec = dict(self._plr_metric_spec(metric))
        if error_against == "spatial_correct":
            spec["success_col"] = "correct_corner_visit"
            spec["metric_stub"] = "spatial_correct_corner_error"
            spec["title_label"] = "spatial correct-corner error rate"
            spec["ylabel"] = "Spatial correct-corner error rate [%]"
            spec["chance_level"] = 75.0
            return spec
        spec["metric_stub"] = f"{spec['metric_stub']}_error"
        spec["title_label"] = str(spec["title_label"]).replace(" rate", " error rate")
        spec["ylabel"] = str(spec["ylabel"]).replace(" rate", " error rate")
        spec["chance_level"] = 100.0 - float(spec["chance_level"])
        return spec

    def _plr_metric_spec(self, metric: str) -> dict[str, str | float]:
        """Return labels and filenames for one public PLR metric name."""

        specs: dict[str, dict[str, str | float]] = {
            "correct_corner_visit": {
                "success_col": "correct_corner_visit",
                "metric_stub": "correct_corner",
                "file_stub": "correct_corner_visit_rate",
                "title_label": "correct-corner visit rate",
                "ylabel": "Correct-corner visit rate [%]",
                "chance_level": 25.0,
                "count_file_stub": "correct_corner_visits_absolute",
                "count_title_label": "correct-corner visits",
                "count_ylabel": "Correct-corner visits per mouse and bin"},
            "correct_np_visit": {
                "success_col": "correct_np_visit",
                "metric_stub": "correct_np",
                "file_stub": "correct_np_visit_rate",
                "title_label": "correct NP visit rate",
                "ylabel": "Correct NP visit rate [%]",
                "chance_level": 25.0,
                "count_file_stub": "correct_np_visits_absolute",
                "count_title_label": "correct NP visits",
                "count_ylabel": "Correct NP visits per mouse and bin"},
            "rewarded_correct_corner_visit": {
                "success_col": "rewarded_correct_corner_visit",
                "metric_stub": "rewarded_correct_corner",
                "file_stub": "rewarded_correct_corner_visit_rate",
                "title_label": "rewarded correct-corner visit rate",
                "ylabel": "Rewarded correct-corner visit rate [%]",
                "chance_level": 25.0,
                "count_file_stub": "rewarded_correct_corner_visits_absolute",
                "count_title_label": "rewarded correct-corner visits",
                "count_ylabel": "Rewarded correct-corner visits per mouse and bin"}}
        if metric not in specs:
            available = ", ".join(sorted(specs))
            raise ValueError(f"Unknown PLR metric {metric!r}. Choose one of: {available}.")
        return specs[metric]

    def _phase_plot_origin(self, phase_number: int) -> tuple[float, int]:
        """Return clock origin and experiment day for one selected phase."""

        phase_start = self.scheduled_phase_start_hours[int(phase_number)]
        return (
            plr.phase_origin_clock_hour(self.experiment.mouse_day_start_hour, phase_start),
            plr.experiment_day_from_scheduled_start(phase_start))

    def _phase_end_hours(
        self,
        visits: pd.DataFrame,
        *,
        bin_hours: int) -> tuple[dict[tuple[str, int], float], dict[int, float]]:
        """Return binned phase end hours by group and phase."""

        phase_visits = visits.loc[visits["AnalysisPhaseNumber"].notna()].copy()
        if phase_visits.empty:
            return {}, {}
        phase_visits["AnalysisPhaseNumber"] = phase_visits["AnalysisPhaseNumber"].astype(int)
        group_end = (
            phase_visits.groupby(["Group", "AnalysisPhaseNumber"], observed=True)["analysis_phase_elapsed_hours"].max()
            + float(bin_hours)
        ).astype(float).to_dict()
        phase_end = (
            phase_visits.groupby("AnalysisPhaseNumber", observed=True)["analysis_phase_elapsed_hours"].max()
            + float(bin_hours)
        ).astype(float).to_dict()
        return group_end, phase_end

    def _prepared_visits(
        self,
        *,
        phase_max_hours: dict[int, float] | None = None,
        dayphase: DayPhase = "all") -> pd.DataFrame:
        """Return prepared analysis visits, preparing them on demand."""

        if self.analysis_visits is None or phase_max_hours is not None:
            self.prepare_analysis(phase_max_hours=phase_max_hours, verbose=False)
        if self.analysis_visits is None:
            raise RuntimeError("Analysis visit data could not be prepared.")
        return self._filter_dayphase(self.analysis_visits, dayphase=dayphase)

    def _analysis_nosepokes_with_timing(
        self,
        *,
        phase_max_hours: dict[int, float] | None = None,
        dayphase: DayPhase = "all") -> pd.DataFrame:
        """Return nose-poke events with mouse, group, phase, and analysis-time columns."""

        visits = self._prepared_visits(phase_max_hours=phase_max_hours, dayphase=dayphase)
        if self.analysis_nosepokes is None:
            raise RuntimeError("Analysis nose-poke data could not be prepared.")
        keys = ["RunGroup", "Phase", "PhaseNumber", "VisitID"]
        timing_columns = keys + [
            "AnimalID",
            "Group",
            "ET",
            "ETLabel",
            "SEX",
            "AnalysisPhaseNumber",
            "AnalysisPhase",
            "analysis_experiment_elapsed_hours",
            "analysis_phase_elapsed_hours",
            "analysis_experiment_day",
            "analysis_phase_day"]
        visit_timing = visits.loc[:, timing_columns].drop_duplicates(subset=keys)
        merged = self.analysis_nosepokes.merge(
            visit_timing,
            on=keys,
            how="inner",
            validate="many_to_one")
        return merged

    def _filter_dayphase(self, data: pd.DataFrame, *, dayphase: DayPhase) -> pd.DataFrame:
        """Apply the experiment-level mouse-day day/night filter."""

        return mt.filter_by_dayphase(
            data,
            dayphase=dayphase,
            awake_start_clock_hour=self.awake_start_clock_hour,
            awake_end_clock_hour=self.awake_end_clock_hour,
            origin_clock_hour=self.experiment.mouse_day_start_hour)

    def _apply_plot_settings(
        self,
        *,
        base_font_size: float,
        font_family: str,
        figure_size_cm: dict[str, tuple[float, float]] | None = None) -> None:
        """Apply plot settings for one user-facing plot call."""

        configure_plot_style(font_size=base_font_size, font_family=font_family)
        set_group_colors(self.experiment.group_colors)

    def _write_load_outputs(self, subject_frame: pd.DataFrame) -> None:
        """Save loaded data and metadata definitions after ``load()``."""

        if self.cohort is None or self.visits is None:
            return
        csv_dir = self.results_data_path / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        self.visits.to_csv(csv_dir / "loaded_visits.tsv.gz", sep="\t", index=False, compression="gzip")
        self.cohort.nosepokes.to_csv(csv_dir / "loaded_nosepokes.tsv.gz", sep="\t", index=False, compression="gzip")
        self.cohort.metadata.to_csv(csv_dir / "loaded_mouse_metadata.tsv", sep="\t", index=False)
        self.cohort.phase_manifest.to_csv(csv_dir / "loaded_phase_manifest.tsv", sep="\t", index=False)
        subject_frame.to_csv(csv_dir / "script_subject_metadata.tsv", sep="\t", index=False)
        _write_yaml(self.results_data_path / "experiment.yaml", self._experiment_to_dict())
        _write_yaml(self.results_data_path / "phases.yaml", self._phases_to_dict())
        _write_yaml(self.results_data_path / "subjects.yaml", self._subjects_to_dict())

    def _print_load_summary(self, *, raw_counts: dict[str, int], registered_subject_count: int) -> None:
        """Print a compact load summary for scripts and interactive notebooks."""

        if self.cohort is None or self.visits is None:
            return
        run_group_count = int(self.cohort.phase_manifest["RunGroup"].nunique()) if "RunGroup" in self.cohort.phase_manifest else 0
        phase_count = int(self.visits["AnalysisPhaseNumber"].nunique()) if "AnalysisPhaseNumber" in self.visits else 0
        group_count = int(self.visits["Group"].astype(str).nunique()) if "Group" in self.visits else 0
        print("IntelliCage data loaded")
        print(f"  Experiment: {self.experiment.name}")
        print(f"  Raw data root: {self.root_data_path}")
        print(f"  Results root: {self.results_data_path}")
        print(f"  Run groups / cage folders: {run_group_count}")
        print(
            "  Export files: "
            f"{raw_counts.get('Visits.txt', 0)} Visits.txt, "
            f"{raw_counts.get('Nosepokes.txt', 0)} Nosepokes.txt")
        print(f"  Registered subjects: {registered_subject_count}")
        print(f"  Loaded subjects: {self.cohort.metadata['ET'].nunique()}")
        print(f"  Groups: {group_count}")
        print(f"  Phases with visits: {phase_count}")
        print(f"  Visits: {len(self.visits):,}")
        print(f"  Nosepokes: {len(self.cohort.nosepokes):,}")
        print(f"  Audit tables written to: {self.results_data_path / 'csv'}")

    def _print_analysis_summary(
        self,
        *,
        raw_visit_count: int,
        raw_nosepoke_count: int,
        phase_max_hours: dict[int, float] | None,
        excluded_groups: list[str]) -> None:
        """Print a compact analysis-preparation summary."""

        if self.analysis_visits is None or self.analysis_metadata is None or self.analysis_nosepokes is None:
            return
        groups = ", ".join(str(group) for group in self.analysis_metadata["Group"].dropna().astype(str).unique())
        limits = "none" if not phase_max_hours else ", ".join(f"phase {key}: {value:g} h" for key, value in sorted(phase_max_hours.items()))
        exclusions = "none" if not excluded_groups else ", ".join(excluded_groups)
        print("Analysis tables prepared")
        print(f"  Included subjects: {self.analysis_metadata['ET'].nunique()}")
        print(f"  Included groups: {groups}")
        print(f"  Excluded groups: {exclusions}")
        print(f"  Phase time limits: {limits}")
        print(f"  Visits kept: {len(self.analysis_visits):,} of {raw_visit_count:,}")
        print(f"  Nosepokes kept: {len(self.analysis_nosepokes):,} of {raw_nosepoke_count:,}")
        print(f"  Analysis tables written to: {self.results_data_path / 'csv'}")

    def _write_analysis_audit_outputs(self, *, phase_max_hours: dict[int, float] | None) -> None:
        """Save generic analysis audit tables."""

        if self.cohort is None or self.analysis_visits is None:
            return
        phase_window_table = self.analysis_phase_windows
        suggested_limits = mt.suggest_common_phase_limits(self.cohort.phase_manifest)
        plr.save_table(self.analysis_metadata, self.results_data_path / "mouse_metadata.tsv")
        plr.save_table(self.cohort.phase_manifest, self.results_data_path / "phase_manifest.tsv")
        plr.save_table(mt.build_phase_time_limit_table(self.cohort.phase_manifest), self.results_data_path / "phase_time_limit_recommendations.tsv")
        plr.save_table(phase_window_table, self.results_data_path / "analysis_phase_windows.tsv")
        plr.save_table(
            pd.DataFrame({
                "PhaseNumber": list(sorted(suggested_limits)),
                "SuggestedCommonLimitHours": [suggested_limits[key] for key in sorted(suggested_limits)]}),
            self.results_data_path / "suggested_common_phase_limits.tsv")
        plr.save_table(
            pd.DataFrame({
                "PhaseNumber": list(sorted(self.scheduled_phase_start_hours)),
                "ScheduledStartHours": [self.scheduled_phase_start_hours[key] for key in sorted(self.scheduled_phase_start_hours)]}),
            self.results_data_path / "scheduled_phase_start_hours.tsv")
        plr.save_table(
            pd.DataFrame({
                "Setting": ["phase_max_hours", "mouse_day_start_hour", "awake_duration_hours"],
                "Value": [
                    "" if phase_max_hours is None else ";".join(f"{key}={value}" for key, value in phase_max_hours.items()),
                    self.experiment.mouse_day_start_hour,
                    self.experiment.awake_duration_hours]}),
            self.results_data_path / "analysis_settings.tsv")
        self.analysis_visits.to_csv(self.results_data_path / "csv" / "merged_visits.tsv.gz", sep="\t", index=False, compression="gzip")
        if self.analysis_nosepokes is not None:
            self.analysis_nosepokes.to_csv(self.results_data_path / "csv" / "merged_nosepokes.tsv.gz", sep="\t", index=False, compression="gzip")

    def _experiment_to_dict(self) -> dict[str, Any]:
        """Return experiment metadata as serializable audit content."""

        return {
            "name": self.experiment.name,
            "root_data_path": self.experiment.root_data_path,
            "results_data_path": self.experiment.results_data_path,
            "group_names": self.experiment.group_names,
            "group_colors": self.experiment.group_colors,
            "mouse_day_start_hour": self.experiment.mouse_day_start_hour,
            "awake_duration_hours": self.experiment.awake_duration_hours,
            "experiment_day0_start_hour": self.experiment.experiment_day0_start_hour,
            "schedule_anchor_phase_number": self.experiment.schedule_anchor_phase_number}

    def _phases_to_dict(self) -> dict[int, dict[str, Any]]:
        """Return phase metadata as serializable audit content."""

        return {
            number: {
                "short_name": phase.short_name,
                "long_name": phase.long_name,
                "folder_name": phase.raw_folder_name,
                "color": phase.color,
                "scheduled_start_hour": phase.scheduled_start_hour}
            for number, phase in self.experiment.phases.items()}

    def _subjects_to_dict(self) -> dict[str, dict[str, Any]]:
        """Return subject metadata as serializable audit content."""

        return {
            animal_id: {
                "group": subject.group,
                "sex": subject.sex,
                "true_id": subject.true_id,
                "age_months": subject.age_months,
                "date_of_birth": subject.date_of_birth,
                "phase_windows": {
                    phase_number: _phase_window_to_strings(window)
                    for phase_number, window in subject.phase_windows.items()},
                "corner_assignments": subject.corner_assignments}
            for animal_id, subject in self.subjects.subjects.items()}

    def _plot_selected_plr_learning_phases(
        self,
        visits: pd.DataFrame,
        *,
        selected_phases: list[int],
        output_dir: Path,
        bin_hours: int,
        spread_metric: SpreadMetric,
        plot_style: str) -> None:
        """Plot selected PLR phases when the caller requests a subset."""

        metric_specs = [
            ("correct_corner_visit", "correct_corner_visit_rate", "correct-corner visit rate", "Correct-corner visit rate [%]"),
            ("correct_np_visit", "correct_np_visit_rate", "correct NP visit rate", "Correct NP visit rate [%]"),
            ("rewarded_correct_corner_visit", "rewarded_correct_corner_visit_rate", "rewarded correct-corner visit rate", "Rewarded correct-corner visit rate [%]")]
        for phase_number in selected_phases:
            phase_start = self.scheduled_phase_start_hours[phase_number]
            phase_origin_hour = plr.phase_origin_clock_hour(self.experiment.mouse_day_start_hour, phase_start)
            phase_start_day = plr.experiment_day_from_scheduled_start(phase_start)
            for success_col, file_stub, title_label, ylabel in metric_specs:
                mouse_bins, summary_bins = mt.compute_place_learning_rate_bins(
                    visits,
                    phase_number=phase_number,
                    bin_hours=bin_hours,
                    success_col=success_col)
                count_bins, count_summary = mt.compute_place_learning_count_bins(
                    visits,
                    phase_number=phase_number,
                    bin_hours=bin_hours,
                    success_col=success_col)
                plr.save_table(mouse_bins, output_dir / f"phase{phase_number}_{file_stub}_mouse_bins_{bin_hours}h.tsv")
                plr.save_table(summary_bins, output_dir / f"phase{phase_number}_{file_stub}_group_summary_{bin_hours}h.tsv")
                plr.save_table(count_bins, output_dir / f"phase{phase_number}_{file_stub}_counts_mouse_bins_{bin_hours}h.tsv")
                plr.save_table(count_summary, output_dir / f"phase{phase_number}_{file_stub}_counts_group_summary_{bin_hours}h.tsv")
                plr.plot_phase_learning_rate_groups(
                    summary_bins,
                    phase_number=phase_number,
                    phase_display_name=self.phase_display_names[phase_number],
                    bin_hours=bin_hours,
                    output_path=output_dir / f"phase{phase_number}_{file_stub}_all_groups_{bin_hours}h.png",
                    spread_metric=spread_metric,
                    title_label=title_label,
                    ylabel=ylabel,
                    chance_level=25.0,
                    plot_style=plot_style,
                    origin_clock_hour=phase_origin_hour,
                    awake_start_clock_hour=self.awake_start_clock_hour,
                    awake_end_clock_hour=self.awake_end_clock_hour,
                    starting_day=phase_start_day)
# %% END
