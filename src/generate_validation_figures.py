"""
================================================================================
ADR1D Validation: Scientific Figure Generation
================================================================================

Generate publication-ready validation figures exclusively from the persisted
ADR1D-ML, ADR1D-NN, traditional numerical-reference, and computational-cost
evidence. The script never loads or executes either predictive model.

Main Operations
---------------
1. Verify the structure and status of the persisted validation artifacts.
2. Aggregate the five ADR1D-ML noise replicates by locked base scenario.
3. Visualize parameter recovery, field error, numerical convergence, and cost.
4. Save every figure in raster and vector formats with stable styling.
5. Record source and output SHA-256 digests in a machine-readable manifest.

Authors
-------
Gerardo Tinoco-Guerrero
Francisco J. Domínguez-Mota
J. Alberto Guzmán-Torres

Universidad Michoacana de San Nicolás de Hidalgo, Morelia, Mexico.
Contact: gerardo.tinoco@umich.mx

Funding & Institutional Support
-------------------------------
This work received institutional and financial support from:
- Secretariat of Science, Humanities, Technology and Innovation (SECIHTI),
  Mexico.
- Coordination of Scientific Research, Universidad Michoacana de San Nicolás
  de Hidalgo (CIC-UMSNH), Mexico.
- SIIIA MATH: Soluciones en Ingeniería.
- International Centre for Numerical Methods in Engineering (CIMNE).
- Aula CIMNE Morelia.

Revision History
----------------
- Initial release: July 2026.
- Last update: July 2026.
================================================================================
"""

# Standard library
import argparse
import hashlib
import json
import os
from pathlib import Path

# Stabilize PDF metadata before importing Matplotlib.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

# Third-party libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter


ROOT                       = Path(__file__).resolve().parents[1]
RESULTS                    = ROOT / "results"
DEFAULT_OUTPUT             = RESULTS / "figures"
DEFAULT_MANIFEST           = RESULTS / "figure_manifest.json"
ML_PREDICTIONS_PATH        = RESULTS / "adr1d_ml_challenge_predictions.csv"
ML_METRICS_PATH            = RESULTS / "adr1d_ml_challenge_metrics.json"
NN_PREDICTIONS_PATH        = RESULTS / "adr1d_nn_challenge_predictions.csv"
NN_SCENARIOS_PATH          = RESULTS / "adr1d_nn_challenge_scenarios.csv"
NN_METRICS_PATH            = RESULTS / "adr1d_nn_challenge_metrics.json"
NUMERICAL_METRICS_PATH     = RESULTS / "numerical_reference_metrics.json"
COMPUTATIONAL_COST_PATH    = RESULTS / "computational_cost.json"
PROTOCOL_PATH              = ROOT / "configs/validation_protocol.json"
REGIME_ORDER               = ["conservative", "decay_only", "retardation_only", "retardation_and_decay"]
REGIME_LABELS              = {
    "conservative": "Conservative",
    "decay_only": "Decay only",
    "retardation_only": "Retardation only",
    "retardation_and_decay": "Retardation and decay",
}
REGIME_COLORS              = {
    "conservative": "#0072B2",
    "decay_only": "#D55E00",
    "retardation_only": "#CC79A7",
    "retardation_and_decay": "#009E73",
}
FIGURE_NAMES               = [
    "figure_01_parameter_model_validation",
    "figure_02_neural_field_validation",
    "figure_03_numerical_reference_convergence",
    "figure_04_computational_cost",
]


class FigureGenerationError(RuntimeError):
    """Represent invalid persisted evidence or an unsafe output operation."""


def sha256(path):
    """
    Compute the SHA-256 digest of one file in bounded memory.

    Parameters
    ----------
    path : pathlib.Path
        Existing file whose digest is required.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.

    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    """
    Load a UTF-8 JSON object from disk.

    Parameters
    ----------
    path : pathlib.Path
        Existing JSON artifact.

    Returns
    -------
    dict
        Parsed top-level JSON object.

    Raises
    ------
    FigureGenerationError
        If the top-level JSON value is not an object.

    """
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise FigureGenerationError(f"Expected a JSON object in {path}")
    return content


def require(condition, message):
    """
    Enforce one evidence or output invariant.

    Parameters
    ----------
    condition : bool
        Condition that must be true.
    message : str
        Actionable failure description.

    Returns
    -------
    None
        The function returns only when the condition is satisfied.

    Raises
    ------
    FigureGenerationError
        If the condition is false.

    """
    if not bool(condition):
        raise FigureGenerationError(message)


def parse_arguments():
    """
    Parse output destinations and overwrite policy.

    Returns
    -------
    argparse.Namespace
        Validated command-line arguments.

    """
    parser = argparse.ArgumentParser(description="Generate ADR1D validation figures from persisted evidence.")
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT, help="Directory for PNG and PDF figures.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Destination for the figure manifest JSON.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing figures and manifest.")
    return parser.parse_args()


def configure_matplotlib():
    """
    Apply a compact and publication-oriented Matplotlib style.

    Returns
    -------
    None
        Matplotlib global settings are updated in place.

    """
    matplotlib.rcParams.update({
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlelocation": "left",
        "axes.titlesize": 9.5,
        "axes.titleweight": "semibold",
        "figure.dpi": 120,
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "text.color": "#222222",
        "xtick.color": "#333333",
        "xtick.labelsize": 7.5,
        "ytick.color": "#333333",
        "ytick.labelsize": 7.5,
    })


def panel_label(axis, label):
    """
    Add one stable panel identifier to an axis.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axis receiving the identifier.
    label : str
        Uppercase panel identifier.

    Returns
    -------
    None
        The artist is added directly to the axis.

    """
    axis.text(-0.12, 1.06, label, transform=axis.transAxes, fontsize=10, fontweight="bold", va="top")


def geometric_mean(values):
    """
    Compute the geometric mean of strictly positive values.

    Parameters
    ----------
    values : pandas.Series or array-like
        Positive observations.

    Returns
    -------
    float
        Geometric mean in the original scale.

    Raises
    ------
    FigureGenerationError
        If one or more observations are nonpositive or nonfinite.

    """
    array = np.asarray(values, dtype=float)
    require(np.all(np.isfinite(array)) and np.all(array > 0.0), "Geometric-mean inputs must be finite and positive.")
    return float(np.exp(np.mean(np.log(array))))


def aggregate_ml_predictions(predictions, metrics):
    """
    Aggregate locked sensor-noise replicates to base-scenario predictions.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Persisted ADR1D-ML replicate-level predictions.
    metrics : dict
        Persisted ADR1D-ML challenge report.

    Returns
    -------
    pandas.DataFrame
        One row per locked base scenario.

    """
    required = {
        "base_scenario_id", "replicate_id", "regime", "actual_effective_velocity_m_s",
        "predicted_effective_velocity_m_s", "actual_effective_dispersion_m2_s",
        "predicted_effective_dispersion_m2_s", "actual_decay_resolvable",
        "predicted_decay_resolvable_probability", "actual_decay_rate_s_1",
        "predicted_decay_rate_if_resolvable_s_1",
    }
    require(required.issubset(predictions.columns), "ADR1D-ML predictions are missing required columns.")
    require(len(predictions) == 600, "ADR1D-ML predictions must contain 600 noise realizations.")
    counts = predictions.groupby("base_scenario_id", sort=True).size()
    require(len(counts) == 120 and bool((counts == 5).all()), "Each ADR1D-ML base scenario must contain five replicates.")

    rows = []
    threshold = float(metrics["evidence"]["decay_resolvability"]["decision_threshold"])
    for scenario_id, group in predictions.groupby("base_scenario_id", sort=True):
        constant_columns = [
            "regime", "actual_effective_velocity_m_s", "actual_effective_dispersion_m2_s",
            "actual_decay_resolvable", "actual_decay_rate_s_1",
        ]
        require(all(group[column].nunique(dropna=False) == 1 for column in constant_columns), f"Truth values vary across replicates for {scenario_id}.")
        probability = float(group["predicted_decay_resolvable_probability"].mean())
        rows.append({
            "scenario_id": scenario_id,
            "regime": str(group["regime"].iloc[0]),
            "actual_velocity": float(group["actual_effective_velocity_m_s"].iloc[0]),
            "predicted_velocity": geometric_mean(group["predicted_effective_velocity_m_s"]),
            "actual_dispersion": float(group["actual_effective_dispersion_m2_s"].iloc[0]),
            "predicted_dispersion": geometric_mean(group["predicted_effective_dispersion_m2_s"]),
            "actual_decay_resolvable": int(group["actual_decay_resolvable"].iloc[0]),
            "predicted_decay_probability": probability,
            "predicted_decay_resolvable": int(probability >= threshold),
            "actual_decay_rate": float(group["actual_decay_rate_s_1"].iloc[0]),
            "predicted_conditional_decay": geometric_mean(group["predicted_decay_rate_if_resolvable_s_1"]),
        })
    aggregated = pd.DataFrame(rows)
    require(set(aggregated["regime"]) == set(REGIME_ORDER), "ADR1D-ML regime coverage is incomplete.")
    return aggregated


def draw_parity_panel(axis, frame, actual_column, predicted_column, title, xlabel, ylabel):
    """
    Draw one logarithmic parity panel colored by physical regime.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Target axis.
    frame : pandas.DataFrame
        Scenario-level values and regime labels.
    actual_column : str
        Column containing reference values.
    predicted_column : str
        Column containing model values.
    title : str
        Panel title.
    xlabel, ylabel : str
        Axis labels with units.

    Returns
    -------
    None
        Artists are added directly to the axis.

    """
    values = np.concatenate([frame[actual_column].to_numpy(dtype=float), frame[predicted_column].to_numpy(dtype=float)])
    require(np.all(np.isfinite(values)) and np.all(values > 0.0), f"Parity values for {title} must be finite and positive.")
    lower = 10.0 ** np.floor(np.log10(values.min()))
    upper = 10.0 ** np.ceil(np.log10(values.max()))
    for regime in REGIME_ORDER:
        subset = frame.loc[frame["regime"] == regime]
        if not subset.empty:
            axis.scatter(subset[actual_column], subset[predicted_column], s=20, color=REGIME_COLORS[regime], alpha=0.78, edgecolor="white", linewidth=0.35, label=REGIME_LABELS[regime])
    axis.plot([lower, upper], [lower, upper], color="#222222", linewidth=1.0, linestyle="--", zorder=0)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, which="major", color="#D9D9D9", linewidth=0.5)


def deterministic_offsets(count):
    """
    Create bounded deterministic horizontal offsets for strip plots.

    Parameters
    ----------
    count : int
        Number of offsets required.

    Returns
    -------
    numpy.ndarray
        Values in approximately `[-0.17, 0.17]`.

    """
    index = np.arange(int(count), dtype=float)
    return 0.17 * np.sin(index * np.pi * (3.0 - np.sqrt(5.0)))


def create_parameter_figure(aggregated, metrics):
    """
    Create the four-panel ADR1D-ML challenge figure.

    Parameters
    ----------
    aggregated : pandas.DataFrame
        One row per locked base scenario.
    metrics : dict
        Persisted ADR1D-ML challenge report.

    Returns
    -------
    matplotlib.figure.Figure
        Completed publication figure.

    """
    evidence = metrics["evidence"]
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), constrained_layout=True)
    draw_parity_panel(axes[0, 0], aggregated, "actual_velocity", "predicted_velocity", "Effective velocity", r"Reference $v_{\mathrm{eff}}$ (m s$^{-1}$)", r"Predicted $v_{\mathrm{eff}}$ (m s$^{-1}$)")
    axes[0, 0].text(0.04, 0.95, rf"$R^2_{{\log_{{10}}}}={evidence['effective_velocity']['r2_log10']:.3f}$" + "\n" + rf"Median APE = {100.0 * evidence['effective_velocity']['median_absolute_percentage_error']:.1f}%", transform=axes[0, 0].transAxes, va="top", fontsize=7.5)

    draw_parity_panel(axes[0, 1], aggregated, "actual_dispersion", "predicted_dispersion", "Effective dispersion", r"Reference $D_{\mathrm{eff}}$ (m$^2$ s$^{-1}$)", r"Predicted $D_{\mathrm{eff}}$ (m$^2$ s$^{-1}$)")
    axes[0, 1].text(0.04, 0.95, rf"$R^2_{{\log_{{10}}}}={evidence['effective_dispersion']['r2_log10']:.3f}$" + "\n" + rf"Median APE = {100.0 * evidence['effective_dispersion']['median_absolute_percentage_error']:.1f}%", transform=axes[0, 1].transAxes, va="top", fontsize=7.5)

    decay = aggregated.loc[aggregated["actual_decay_resolvable"] == 1].copy()
    draw_parity_panel(axes[1, 0], decay, "actual_decay_rate", "predicted_conditional_decay", "Conditional decay rate", r"Reference $\lambda$ (s$^{-1}$)", r"Predicted $\lambda$ (s$^{-1}$)")
    axes[1, 0].text(0.04, 0.95, rf"$R^2_{{\log_{{10}}}}={evidence['conditional_decay']['r2_log10']:.3f}$" + "\n" + rf"Median APE = {100.0 * evidence['conditional_decay']['median_absolute_percentage_error']:.1f}%", transform=axes[1, 0].transAxes, va="top", fontsize=7.5)

    axis = axes[1, 1]
    threshold = float(evidence["decay_resolvability"]["decision_threshold"])
    for truth, color in [(0, "#6C757D"), (1, "#E69F00")]:
        subset = aggregated.loc[aggregated["actual_decay_resolvable"] == truth].sort_values("scenario_id")
        offsets = deterministic_offsets(len(subset))
        axis.scatter(truth + offsets, subset["predicted_decay_probability"], s=18, color=color, alpha=0.7, edgecolor="white", linewidth=0.3)
    axis.axhline(threshold, color="#222222", linestyle="--", linewidth=1.0, label=f"Frozen threshold = {threshold:.2f}")
    axis.set_xlim(-0.45, 1.45)
    axis.set_ylim(-0.03, 1.03)
    axis.set_xticks([0, 1], ["Not resolvable\n(n = 97)", "Resolvable\n(n = 23)"])
    axis.set_title("Decay resolvability")
    axis.set_ylabel("Predicted probability")
    axis.grid(True, axis="y", color="#D9D9D9", linewidth=0.5)
    axis.legend(loc="upper left")
    classification = evidence["decay_resolvability"]
    axis.text(0.96, 0.05, f"Balanced accuracy = {classification['balanced_accuracy']:.3f}\nROC AUC = {classification['roc_auc']:.3f}", transform=axis.transAxes, ha="right", va="bottom", fontsize=7.5)

    for label, axis in zip("ABCD", axes.flat):
        panel_label(axis, label)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=4, bbox_to_anchor=(0.5, 1.035))
    return figure


def select_worst_nn_scenario(scenarios):
    """
    Select the persisted scenario with the largest neural-field RMSE.

    Parameters
    ----------
    scenarios : pandas.DataFrame
        Persisted scenario-level ADR1D-NN metrics.

    Returns
    -------
    pandas.Series
        Deterministically selected scenario row.

    """
    require(len(scenarios) == 120, "ADR1D-NN scenario evidence must contain 120 rows.")
    required = {"scenario_id", "regime", "design_component", "peclet_number", "damkohler_number", "rmse", "integrated_mass_relative_error"}
    require(required.issubset(scenarios.columns), "ADR1D-NN scenario evidence is missing required columns.")
    ordered = scenarios.sort_values(["rmse", "scenario_id"], ascending=[False, True], kind="mergesort")
    return ordered.iloc[0]


def create_field_figure(scenarios, predictions, metrics):
    """
    Create scenario-level and field-resolved ADR1D-NN diagnostics.

    Parameters
    ----------
    scenarios : pandas.DataFrame
        Persisted scenario-level ADR1D-NN metrics.
    predictions : pandas.DataFrame
        Persisted pointwise ADR1D-NN predictions.
    metrics : dict
        Persisted ADR1D-NN challenge report.

    Returns
    -------
    tuple
        Completed figure and selected worst-scenario identifier.

    """
    require(len(predictions) == 299880, "ADR1D-NN pointwise evidence must contain 299,880 rows.")
    require(int(metrics["evidence"]["field"]["rows"]) == len(predictions), "ADR1D-NN metric and prediction row counts differ.")
    worst = select_worst_nn_scenario(scenarios)
    scenario_id = str(worst["scenario_id"])
    field = predictions.loc[predictions["scenario_id"] == scenario_id].copy()
    require(len(field) == 2499, f"Worst-scenario field {scenario_id} must contain 2,499 rows.")
    require(not field.duplicated(["time_s", "x_m"]).any(), f"Duplicate field coordinates found for {scenario_id}.")

    times = np.sort(field["time_s"].unique())
    positions = np.sort(field["x_m"].unique())
    require(len(times) == 49 and len(positions) == 51, f"Unexpected field dimensions for {scenario_id}.")
    reference = field.pivot(index="x_m", columns="time_s", values="reference_normalized_concentration").loc[positions, times].to_numpy(dtype=float)
    prediction = field.pivot(index="x_m", columns="time_s", values="predicted_normalized_concentration").loc[positions, times].to_numpy(dtype=float)
    error = np.abs(prediction - reference)

    figure, axes = plt.subplots(2, 2, figsize=(7.5, 5.8), constrained_layout=True)
    axis = axes[0, 0]
    for regime in REGIME_ORDER:
        subset = scenarios.loc[scenarios["regime"] == regime]
        for component, marker in [("latin_hypercube", "o"), ("boundary_profile", "s")]:
            selected = subset.loc[subset["design_component"] == component]
            axis.scatter(selected["peclet_number"], selected["rmse"], s=22 if component == "latin_hypercube" else 30, marker=marker, color=REGIME_COLORS[regime], alpha=0.78, edgecolor="white", linewidth=0.35)
    axis.scatter([worst["peclet_number"]], [worst["rmse"]], s=72, facecolor="none", edgecolor="#111111", linewidth=1.2, zorder=5)
    axis.axhline(0.05, color="#222222", linestyle="--", linewidth=1.0)
    axis.set_xscale("log")
    axis.set_ylim(0.0075, 0.0615)
    axis.set_xlabel("Peclet number")
    axis.set_ylabel("Scenario RMSE")
    axis.set_title("Scenario-level neural error")
    axis.grid(True, which="major", color="#D9D9D9", linewidth=0.5)
    axis.text(0.98, 0.76, "Pre-specified P90\ncriterion = 0.05", transform=axis.transAxes, ha="right", va="top", fontsize=7.3)
    axis.annotate(f"Worst: {scenario_id}\nRMSE = {worst['rmse']:.4f}", xy=(worst["peclet_number"], worst["rmse"]), xytext=(5.2, 0.0585), fontsize=7.3, ha="left", va="center", arrowprops={"arrowstyle": "-", "color": "#333333", "linewidth": 0.7})

    time_hours = times / 3600.0
    heatmap_options = {"shading": "auto", "rasterized": True}
    reference_mesh = axes[0, 1].pcolormesh(time_hours, positions, reference, cmap="viridis", vmin=0.0, vmax=1.0, **heatmap_options)
    axes[0, 1].set_title(f"Analytical reference: {scenario_id}")
    axes[0, 1].set_xlabel("Time (h)")
    axes[0, 1].set_ylabel("Position (m)")
    colorbar = figure.colorbar(reference_mesh, ax=axes[0, 1], fraction=0.046, pad=0.03)
    colorbar.set_label(r"$C/C_0$")

    prediction_mesh = axes[1, 0].pcolormesh(time_hours, positions, prediction, cmap="viridis", vmin=0.0, vmax=1.0, **heatmap_options)
    axes[1, 0].set_title("Frozen neural prediction")
    axes[1, 0].set_xlabel("Time (h)")
    axes[1, 0].set_ylabel("Position (m)")
    colorbar = figure.colorbar(prediction_mesh, ax=axes[1, 0], fraction=0.046, pad=0.03)
    colorbar.set_label(r"$\widehat{C}/C_0$")

    error_mesh = axes[1, 1].pcolormesh(time_hours, positions, error, cmap="magma", vmin=0.0, vmax=float(error.max()), **heatmap_options)
    axes[1, 1].set_title("Pointwise absolute error")
    axes[1, 1].set_xlabel("Time (h)")
    axes[1, 1].set_ylabel("Position (m)")
    colorbar = figure.colorbar(error_mesh, ax=axes[1, 1], fraction=0.046, pad=0.03)
    colorbar.set_label(r"$|\widehat{C}-C|/C_0$")

    for label, axis in zip("ABCD", axes.flat):
        panel_label(axis, label)
    regime_handles = [Line2D([0], [0], color=REGIME_COLORS[name], marker="o", linewidth=0.0, markersize=5, label=REGIME_LABELS[name]) for name in REGIME_ORDER]
    design_handles = [
        Line2D([0], [0], color="#444444", marker="o", linewidth=0.0, markersize=5, label="Latin hypercube"),
        Line2D([0], [0], color="#444444", marker="s", linewidth=0.0, markersize=5, label="Boundary profile"),
    ]
    figure.legend(handles=[*regime_handles, *design_handles], loc="outside lower center", ncol=6, bbox_to_anchor=(0.5, -0.04))
    return figure, scenario_id


def numerical_evidence_frame(metrics):
    """
    Convert nested numerical-reference evidence to one tidy table.

    Parameters
    ----------
    metrics : dict
        Persisted traditional numerical-reference report.

    Returns
    -------
    pandas.DataFrame
        One row per selected scenario with convergence diagnostics.

    """
    evidence = metrics.get("scenario_evidence", [])
    require(len(evidence) == 16, "Traditional numerical evidence must contain 16 scenarios.")
    rows = []
    for item in evidence:
        rows.append({
            "scenario_id": str(item["scenario_id"]),
            "regime": str(item["regime"]),
            "selection_order": int(item["selection_order_within_regime"]),
            "peclet_number": float(item["truth"]["peclet_number"]),
            "coarse_rmse": float(item["rmse_by_level"]["coarse"]),
            "medium_rmse": float(item["rmse_by_level"]["medium"]),
            "fine_rmse": float(item["rmse_by_level"]["fine"]),
            "medium_fine_rmse": float(item["intergrid_rmse"]["medium_vs_restricted_fine"]),
            "passed": bool(item["checks"]["medium_vs_fine_rmse"]),
        })
    frame = pd.DataFrame(rows)
    frame["regime_order"] = frame["regime"].map({name: index for index, name in enumerate(REGIME_ORDER)})
    return frame.sort_values(["regime_order", "selection_order"], kind="mergesort").reset_index(drop=True)


def create_numerical_figure(metrics):
    """
    Create the traditional numerical-reference convergence figure.

    Parameters
    ----------
    metrics : dict
        Persisted traditional numerical-reference report.

    Returns
    -------
    matplotlib.figure.Figure
        Completed publication figure.

    """
    frame = numerical_evidence_frame(metrics)
    figure, axes = plt.subplots(1, 3, figsize=(10.0, 3.45), constrained_layout=True)
    steps = np.array([10.0, 5.0, 2.5])

    axis = axes[0]
    for _, row in frame.iterrows():
        values = [row["coarse_rmse"], row["medium_rmse"], row["fine_rmse"]]
        axis.plot(steps, values, color=REGIME_COLORS[row["regime"]], marker="X" if not row["passed"] else "o", markersize=5.0 if not row["passed"] else 3.4, linewidth=1.7 if not row["passed"] else 0.9, alpha=1.0 if not row["passed"] else 0.52)
    axis.axhline(0.03, color="#222222", linestyle="--", linewidth=1.0)
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlim(11.0, 2.2)
    axis.set_xticks(steps)
    axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.set_xlabel(r"Spatial step $\Delta x$ (m)")
    axis.set_ylabel("RMSE against analytical solution")
    axis.set_title("Grid-refinement behavior")
    axis.grid(True, which="major", color="#D9D9D9", linewidth=0.5)

    x = np.arange(len(frame))
    labels = frame["scenario_id"].str.replace("A5-CH-", "", regex=False)
    colors = frame["regime"].map(REGIME_COLORS).tolist()
    failure_edges = ["#B2182B" if not passed else "white" for passed in frame["passed"]]
    failure_widths = [1.4 if not passed else 0.4 for passed in frame["passed"]]

    axis = axes[1]
    axis.bar(x, frame["fine_rmse"], color=colors, edgecolor="white", linewidth=0.4)
    axis.axhline(0.03, color="#222222", linestyle="--", linewidth=1.0, label="Criterion = 0.03")
    axis.set_xticks(x, labels, rotation=55, ha="right")
    axis.set_ylabel("Fine-grid RMSE")
    axis.set_title("Fine grid against analytical solution")
    axis.grid(True, axis="y", color="#D9D9D9", linewidth=0.5)
    axis.legend(loc="upper right")

    axis = axes[2]
    axis.bar(x, frame["medium_fine_rmse"], color=colors, edgecolor=failure_edges, linewidth=failure_widths)
    axis.axhline(0.01, color="#222222", linestyle="--", linewidth=1.0, label="Criterion = 0.01")
    axis.set_xticks(x, labels, rotation=55, ha="right")
    axis.set_ylabel("Restricted medium-fine RMSE")
    axis.set_title("Inter-grid agreement")
    axis.grid(True, axis="y", color="#D9D9D9", linewidth=0.5)
    axis.legend(loc="upper right")

    for label, axis in zip("ABC", axes):
        panel_label(axis, label)
    handles = [Line2D([0], [0], color=REGIME_COLORS[name], marker="o", linewidth=1.2, label=REGIME_LABELS[name]) for name in REGIME_ORDER]
    handles.append(Line2D([0], [0], color="#B2182B", marker="X", linewidth=0.0, markersize=6, label="Inter-grid criterion not met"))
    figure.legend(handles=handles, loc="outside upper center", ncol=5, bbox_to_anchor=(0.5, 1.06))
    return figure


def timing_panel(axis, measurements, identifiers, labels, title, units_label):
    """
    Draw one horizontal median-and-IQR timing panel.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Target axis.
    measurements : dict
        Persisted workload timing reports.
    identifiers : list of str
        Workload identifiers in display order.
    labels : list of str
        Human-readable operation labels.
    title : str
        Panel title.
    units_label : str
        Description of the timed workload size.

    Returns
    -------
    None
        Artists are added directly to the axis.

    """
    medians = np.array([float(measurements[name]["summary"]["median_seconds"]) for name in identifiers])
    quartile_25 = np.array([float(measurements[name]["summary"]["quartile_25_seconds"]) for name in identifiers])
    quartile_75 = np.array([float(measurements[name]["summary"]["quartile_75_seconds"]) for name in identifiers])
    errors = np.vstack([medians - quartile_25, quartile_75 - medians])
    colors = ["#4C78A8", "#F2CF5B", "#59A14F", "#E15759"][:len(identifiers)]
    positions = np.arange(len(identifiers))
    axis.barh(positions, medians, xerr=errors, color=colors, edgecolor="white", linewidth=0.5, error_kw={"ecolor": "#222222", "elinewidth": 0.8, "capsize": 2})
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xscale("log")
    axis.set_xlabel("Wall time (s, logarithmic scale)")
    axis.set_title(title, y=1.06)
    axis.grid(True, axis="x", which="major", color="#D9D9D9", linewidth=0.5)
    axis.text(0.0, 1.005, units_label, transform=axis.transAxes, fontsize=7.0, color="#555555", va="bottom")
    for position, median in zip(positions, medians):
        axis.text(median * 1.08, position, f"{1000.0 * median:.1f} ms", va="center", fontsize=6.8)
    lower = max(float(np.min(quartile_25)) / 2.0, 1.0e-4)
    upper = float(np.max(quartile_75)) * 2.6
    axis.set_xlim(lower, upper)


def create_cost_figure(cost):
    """
    Create component-wise computational-cost panels.

    Parameters
    ----------
    cost : dict
        Persisted seven-repetition computational-cost report.

    Returns
    -------
    matplotlib.figure.Figure
        Completed publication figure.

    """
    require(cost.get("status") == "complete", "Computational-cost evidence is incomplete.")
    require(cost["scope"].get("challenge_model_inference_runs") == 0, "Cost report includes prohibited challenge inference.")
    require(cost["integrity"].get("protected_artifacts_unchanged") is True, "Protected artifacts changed during cost measurement.")
    measurements = cost["measurements"]
    require(len(measurements) == 11, "Computational-cost evidence must contain 11 workloads.")

    figure, axes = plt.subplots(1, 3, figsize=(10.0, 3.65), constrained_layout=True)
    timing_panel(axes[0], measurements,
                 ["adr1d_ml_artifact_verification_and_loading", "adr1d_ml_feature_construction", "adr1d_ml_model_inference", "adr1d_ml_end_to_end"],
                 ["Verify + load", "Features", "Inference", "End to end"],
                 "ADR1D-ML", "Canonical test split: 45 scenarios")
    timing_panel(axes[1], measurements,
                 ["adr1d_nn_artifact_verification_and_loading", "adr1d_nn_feature_construction", "adr1d_nn_model_inference", "adr1d_nn_end_to_end"],
                 ["Verify + load", "Features", "Inference", "End to end"],
                 "ADR1D-NN", "Canonical test split: 112,455 points")
    timing_panel(axes[2], measurements,
                 ["analytical_evaluation", "traditional_numerical_solution", "numerical_reference_end_to_end"],
                 ["Analytical", "Numerical solve", "End to end"],
                 "Reference methods", "16 cases: 376,320 retained values")
    for label, axis in zip("ABC", axes):
        panel_label(axis, label)
    figure.text(0.5, -0.07, "CPU, one computational thread; bars show medians and interquartile ranges across seven timed repetitions. Timings are machine-specific.", ha="center", fontsize=7.5)
    return figure


def output_paths(output_directory):
    """
    Build all expected PNG and PDF destinations.

    Parameters
    ----------
    output_directory : pathlib.Path
        Figure output directory.

    Returns
    -------
    list of pathlib.Path
        Eight deterministic figure paths.

    """
    return [output_directory / f"{name}.{extension}" for name in FIGURE_NAMES for extension in ["png", "pdf"]]


def validate_destinations(paths, manifest_path, overwrite):
    """
    Refuse accidental replacement of figure artifacts.

    Parameters
    ----------
    paths : list of pathlib.Path
        Expected figure destinations.
    manifest_path : pathlib.Path
        Expected manifest destination.
    overwrite : bool
        Whether replacement is explicitly permitted.

    Returns
    -------
    None
        The function returns after validating all destinations.

    """
    existing = [path for path in [*paths, manifest_path] if path.exists()]
    require(overwrite or not existing, "Outputs already exist; use --overwrite to replace them: " + ", ".join(str(path) for path in existing))


def save_figure(figure, base_path):
    """
    Save one figure in deterministic raster and vector formats.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Completed figure.
    base_path : pathlib.Path
        Destination without a filename extension.

    Returns
    -------
    list of pathlib.Path
        PNG and PDF paths written to disk.

    """
    png_path = base_path.with_suffix(".png")
    pdf_path = base_path.with_suffix(".pdf")
    figure.savefig(png_path, metadata={"Software": "ADR1D validation figure generator"})
    figure.savefig(pdf_path, metadata={"Creator": "ADR1D validation figure generator", "CreationDate": None, "ModDate": None})
    plt.close(figure)
    return [png_path, pdf_path]


def portable_path(path):
    """
    Represent one artifact relative to the validation root when possible.

    Parameters
    ----------
    path : pathlib.Path
        Artifact path to serialize in a portable manifest.

    Returns
    -------
    str
        POSIX-style relative path, or an absolute path outside the project.

    """
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def build_manifest(source_paths, figure_paths, selected_scenario):
    """
    Build the figure-generation traceability manifest.

    Parameters
    ----------
    source_paths : list of pathlib.Path
        Persisted input artifacts used by the figures.
    figure_paths : list of pathlib.Path
        Generated PNG and PDF outputs.
    selected_scenario : str
        Worst persisted ADR1D-NN scenario shown in the field figure.

    Returns
    -------
    dict
        JSON-serializable traceability record.

    """
    return {
        "status": "complete",
        "generator": {
            "path": portable_path(Path(__file__)),
            "sha256": sha256(Path(__file__).resolve()),
            "matplotlib_version": matplotlib.__version__,
            "model_loading_or_inference": False,
        },
        "selection": {
            "neural_field_case": selected_scenario,
            "criterion": "Maximum persisted ADR1D-NN scenario RMSE; ties resolved by ascending scenario identifier.",
        },
        "sources": {portable_path(path): sha256(path) for path in source_paths},
        "figures": {
            portable_path(path): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in figure_paths
        },
    }


def main():
    """
    Generate all validation figures and their traceability manifest.

    Returns
    -------
    None
        Figure files, hashes, and a concise JSON summary are written.

    """
    arguments = parse_arguments()
    source_paths = [
        PROTOCOL_PATH, ML_PREDICTIONS_PATH, ML_METRICS_PATH, NN_PREDICTIONS_PATH,
        NN_SCENARIOS_PATH, NN_METRICS_PATH, NUMERICAL_METRICS_PATH, COMPUTATIONAL_COST_PATH,
    ]
    require(all(path.is_file() for path in source_paths), "One or more required validation artifacts are missing.")
    figure_paths = output_paths(arguments.output_directory)
    validate_destinations(figure_paths, arguments.manifest, arguments.overwrite)

    ml_predictions = pd.read_csv(ML_PREDICTIONS_PATH)
    ml_metrics = load_json(ML_METRICS_PATH)
    nn_predictions = pd.read_csv(NN_PREDICTIONS_PATH)
    nn_scenarios = pd.read_csv(NN_SCENARIOS_PATH)
    nn_metrics = load_json(NN_METRICS_PATH)
    numerical_metrics = load_json(NUMERICAL_METRICS_PATH)
    cost = load_json(COMPUTATIONAL_COST_PATH)
    require(ml_metrics.get("status") == "challenge_evaluation_complete", "ADR1D-ML challenge evidence is incomplete.")
    require(nn_metrics.get("status") == "challenge_evaluation_complete", "ADR1D-NN challenge evidence is incomplete.")
    require(numerical_metrics.get("status") == "numerical_reference_complete", "Traditional numerical-reference evidence is incomplete.")

    configure_matplotlib()
    aggregated = aggregate_ml_predictions(ml_predictions, ml_metrics)
    parameter_figure = create_parameter_figure(aggregated, ml_metrics)
    field_figure, selected_scenario = create_field_figure(nn_scenarios, nn_predictions, nn_metrics)
    numerical_figure = create_numerical_figure(numerical_metrics)
    cost_figure = create_cost_figure(cost)

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    arguments.manifest.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for name, figure in zip(FIGURE_NAMES, [parameter_figure, field_figure, numerical_figure, cost_figure]):
        written.extend(save_figure(figure, arguments.output_directory / name))
    manifest = build_manifest(source_paths, written, selected_scenario)
    arguments.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "figures": len(written),
        "manifest": str(arguments.manifest.resolve()),
        "model_loading_or_inference": False,
        "output_directory": str(arguments.output_directory.resolve()),
        "selected_neural_field_case": selected_scenario,
        "status": "complete",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
