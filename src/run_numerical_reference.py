"""
================================================================================
ADR1D Validation: Traditional Numerical Reference
================================================================================

Solve the locked ADR1D challenge cases with a cell-centered finite-volume
method and compare three refinement levels with the analytical pulse response.

Main Operations
---------------
1. Select 16 reference scenarios using only locked physical truth parameters.
2. Assemble Scharfetter-Gummel advection-dispersion fluxes on three grids.
3. Advance the reactive transport equation with fully implicit backward Euler.
4. Evaluate analytical, inter-grid, arrival, balance, and positivity metrics.
5. Persist scenario-level evidence and an aggregate machine-readable report.

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
- Initial release: May 2025.
- Last update: July 2026.
================================================================================
"""

# Standard library
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

# Third-party libraries
import numpy as np
import pandas as pd
import scipy
from scipy.special import erfc
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu


ROOT                = Path(__file__).resolve().parents[1]
PROTOCOL_PATH       = ROOT / "configs/validation_protocol.json"
CHALLENGE_LOCK_PATH = ROOT / "results/challenge_lock.json"
SCENARIO_PATH       = ROOT / "data/challenge_scenarios.csv"
FIELD_PATH          = ROOT / "data/challenge_analytical_field.csv"
DEFAULT_OUTPUT      = ROOT / "results"
METRICS_NAME        = "numerical_reference_metrics.json"
CASES_NAME          = "numerical_reference_cases.csv"


class NumericalReferenceError(RuntimeError):
    """Represent an invalid numerical-reference input or solver state."""


def sha256(path):
    """
    Compute the SHA-256 digest of a file in bounded memory.

    Parameters
    ----------
    path : pathlib.Path
        Existing artifact whose digest is required.

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
    Load a UTF-8 JSON object.

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
    NumericalReferenceError
        If the top-level JSON value is not an object.

    """
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise NumericalReferenceError(f"Expected a JSON object in {path}")
    return content


def require(condition, message):
    """
    Enforce one solver or protocol condition.

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
    NumericalReferenceError
        If the condition is false.

    """
    if not bool(condition):
        raise NumericalReferenceError(message)


def parse_arguments():
    """
    Parse command-line destinations and overwrite policy.

    Returns
    -------
    argparse.Namespace
        Validated output directory and overwrite flag.

    """
    parser = argparse.ArgumentParser(description="Run the locked ADR1D traditional numerical reference.")
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT, help="Directory for numerical-reference CSV and JSON artifacts.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing numerical-reference outputs in the selected directory.")
    return parser.parse_args()


def bernoulli(value):
    """
    Evaluate the Scharfetter-Gummel Bernoulli function stably.

    Parameters
    ----------
    value : float or numpy.ndarray
        Dimensionless local cell Peclet number.

    Returns
    -------
    float or numpy.ndarray
        Value of `B(z) = z / (exp(z) - 1)` with its analytic limit at zero.

    """
    values = np.asarray(value, dtype=float)
    small  = np.abs(values) < 1.0e-6
    result = np.empty_like(values)
    z      = values[small]
    result[small] = 1.0 - z / 2.0 + z**2 / 12.0 - z**4 / 720.0
    result[~small] = values[~small] / np.expm1(values[~small])
    return float(result) if result.ndim == 0 else result


def normalized_parameter_matrix(frame, protocol):
    """
    Map one physical regime to the locked normalized parameter coordinates.

    Parameters
    ----------
    frame : pandas.DataFrame
        Scenario rows from one physical regime.
    protocol : dict
        Locked challenge design and parameter ranges.

    Returns
    -------
    numpy.ndarray
        Matrix with one row per scenario and seven normalized coordinates.

    Notes
    -----
    Dispersion, source concentration, and active decay use logarithmic
    coordinates. Inactive retardation or decay dimensions are set to zero.
    """
    ranges = protocol["challenge_design"]["parameter_ranges"]
    columns = []
    definitions = (
        ("velocity_m_s", "velocity_m_s", False),
        ("dispersion_m2_s", "dispersion_m2_s", True),
        ("source_concentration_mg_L", "source_concentration_mg_L", True),
        ("source_start_s", "source_start_s", False),
        ("source_duration_s", "source_duration_s", False),
        ("retardation_factor", "retardation_factor_when_active", False),
        ("decay_rate_s_1", "decay_rate_s_1_when_active", True),
    )
    for column, range_key, logarithmic in definitions:
        values = frame[column].to_numpy(dtype=float)
        lower, upper = (float(value) for value in ranges[range_key])
        if column == "retardation_factor" and np.all(values == 1.0):
            normalized = np.zeros_like(values)
        elif column == "decay_rate_s_1" and np.all(values == 0.0):
            normalized = np.zeros_like(values)
        elif logarithmic:
            normalized = (np.log10(values) - math.log10(lower)) / (math.log10(upper) - math.log10(lower))
        else:
            normalized = (values - lower) / (upper - lower)
        columns.append(normalized)
    matrix = np.column_stack(columns)
    require(np.isfinite(matrix).all(), "Normalized selection coordinates contain a non-finite value")
    return matrix


def first_available(frame, ordered_indexes, selected_ids):
    """
    Return the first ordered scenario not selected by an earlier rule.

    Parameters
    ----------
    frame : pandas.DataFrame
        Candidate scenarios indexed by their source table rows.
    ordered_indexes : iterable of int
        Candidate indexes in deterministic preference order.
    selected_ids : set of str
        Scenario identifiers already chosen for the regime.

    Returns
    -------
    pandas.Series
        First unique candidate in the requested order.

    Raises
    ------
    NumericalReferenceError
        If no unused candidate remains.

    """
    for index in ordered_indexes:
        row = frame.loc[index]
        if str(row["scenario_id"]) not in selected_ids:
            return row
    raise NumericalReferenceError("No unused scenario remains for the selection rule")


def select_reference_scenarios(scenarios, protocol):
    """
    Select four truth-parameter reference cases per physical regime.

    Parameters
    ----------
    scenarios : pandas.DataFrame
        Locked 120-row challenge scenario table.
    protocol : dict
        Pre-specified selection order, tie break, and parameter ranges.

    Returns
    -------
    pandas.DataFrame
        Sixteen unique scenario rows with selection rule and medoid score.

    """
    reference = protocol["traditional_numerical_reference"]
    require(reference["selection_uses_only_truth_parameters"], "Numerical selection is not restricted to truth parameters")
    require(reference["selection_rule_order"] == ["minimum_peclet_number", "maximum_peclet_number", "maximum_damkohler_number_or_advective_travel_time_when_decay_is_zero", "normalized_parameter_space_medoid"], "Unexpected numerical-reference selection order")
    selected_rows = []
    for regime, frame in scenarios.groupby("regime", sort=True):
        frame        = frame.sort_values("scenario_id").copy()
        selected_ids = set()
        candidates   = []
        candidates.append(("minimum_peclet_number", first_available(frame, frame.sort_values(["peclet_number", "scenario_id"], ascending=[True, True], kind="mergesort").index, selected_ids), math.nan))
        selected_ids.add(str(candidates[-1][1]["scenario_id"]))
        candidates.append(("maximum_peclet_number", first_available(frame, frame.sort_values(["peclet_number", "scenario_id"], ascending=[False, True], kind="mergesort").index, selected_ids), math.nan))
        selected_ids.add(str(candidates[-1][1]["scenario_id"]))
        third_metric = "damkohler_number" if (frame["decay_rate_s_1"] > 0.0).any() else "advective_travel_time_s"
        candidates.append(("maximum_damkohler_number" if third_metric == "damkohler_number" else "maximum_advective_travel_time", first_available(frame, frame.sort_values([third_metric, "scenario_id"], ascending=[False, True], kind="mergesort").index, selected_ids), math.nan))
        selected_ids.add(str(candidates[-1][1]["scenario_id"]))
        coordinates = normalized_parameter_matrix(frame, protocol)
        medoid_score = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2).sum(axis=1)
        medoid_order = frame.assign(medoid_score=medoid_score).sort_values(["medoid_score", "scenario_id"], ascending=[True, True], kind="mergesort").index
        medoid       = first_available(frame, medoid_order, selected_ids)
        medoid_index = frame.index.get_loc(medoid.name)
        candidates.append(("normalized_parameter_space_medoid", medoid, float(medoid_score[medoid_index])))
        selected_ids.add(str(medoid["scenario_id"]))
        require(len(selected_ids) == int(reference["selection_per_regime"]), f"Selection for {regime} is not unique")
        for order, (criterion, row, score) in enumerate(candidates, start=1):
            selected = row.copy()
            selected["selection_order_within_regime"] = order
            selected["selection_criterion"]           = criterion
            selected["medoid_total_distance"]         = score
            selected_rows.append(selected)
    result = pd.DataFrame(selected_rows).reset_index(drop=True)
    require(len(result) == int(reference["selected_scenarios"]), "Numerical-reference scenario count differs from the protocol")
    require(result["scenario_id"].is_unique, "Numerical-reference selection contains duplicate scenarios")
    require(result.groupby("regime").size().eq(int(reference["selection_per_regime"])).all(), "Numerical-reference selection is not balanced by regime")
    return result


def build_time_grid(final_time_s, maximum_time_step_s, report_times_s, source_start_s, source_end_s):
    """
    Build a time grid containing report and source-switch times exactly.

    Parameters
    ----------
    final_time_s : float
        Simulation end time in seconds.
    maximum_time_step_s : float
        Largest permitted backward-Euler step in seconds.
    report_times_s : array-like of float
        Times at which numerical fields must be retained.
    source_start_s : float
        Pulse activation time in seconds.
    source_end_s : float
        Pulse deactivation time in seconds.

    Returns
    -------
    numpy.ndarray
        Strictly increasing time coordinates including zero and final time.

    """
    anchors = {0.0, float(final_time_s)}
    anchors.update(float(value) for value in report_times_s if 0.0 <= float(value) <= final_time_s)
    anchors.update(value for value in (float(source_start_s), float(source_end_s)) if 0.0 <= value <= final_time_s)
    anchors = sorted(anchors)
    times   = [anchors[0]]
    for left, right in zip(anchors[:-1], anchors[1:]):
        interval = right - left
        steps    = max(1, int(math.ceil(interval / maximum_time_step_s - 1.0e-14)))
        times.extend(np.linspace(left, right, steps + 1, dtype=float)[1:].tolist())
    result = np.asarray(times, dtype=float)
    require(np.all(np.diff(result) > 0.0), "Numerical time grid is not strictly increasing")
    require(float(np.max(np.diff(result))) <= maximum_time_step_s * (1.0 + 1.0e-12), "Numerical time grid exceeds its maximum step")
    for value in (source_start_s, source_end_s):
        if 0.0 <= value <= final_time_s:
            require(np.any(result == float(value)), "A source-switch time is absent from the numerical grid")
    return result


def inlet_value(time_s, source_start_s, source_end_s):
    """
    Evaluate the normalized finite-pulse inlet away from switch ambiguity.

    Parameters
    ----------
    time_s : float
        Interior time of one integration interval in seconds.
    source_start_s : float
        Pulse activation time in seconds.
    source_end_s : float
        Pulse deactivation time in seconds.

    Returns
    -------
    float
        One while the normalized source is active and zero otherwise.

    """
    return 1.0 if source_start_s <= time_s < source_end_s else 0.0


def analytical_step_response(positions_m, elapsed_s, effective_velocity_m_s, effective_dispersion_m2_s, decay_rate_s_1):
    """
    Evaluate the reactive Ogata-Banks unit-step response in scaled variables.

    Parameters
    ----------
    positions_m : numpy.ndarray
        Positive spatial coordinates in meters.
    elapsed_s : float
        Elapsed time after step activation in seconds.
    effective_velocity_m_s : float
        Retardation-scaled advective velocity in meters per second.
    effective_dispersion_m2_s : float
        Retardation-scaled dispersion in square meters per second.
    decay_rate_s_1 : float
        First-order decay rate in inverse seconds.

    Returns
    -------
    numpy.ndarray
        Dimensionless step response at every position.

    """
    positions = np.asarray(positions_m, dtype=float)
    if elapsed_s <= 0.0:
        return np.zeros_like(positions)
    speed           = math.sqrt(effective_velocity_m_s**2 + 4.0 * effective_dispersion_m2_s * decay_rate_s_1)
    denominator     = 2.0 * math.sqrt(effective_dispersion_m2_s * elapsed_s)
    first_exponent  = (effective_velocity_m_s - speed) * positions / (2.0 * effective_dispersion_m2_s)
    second_exponent = (effective_velocity_m_s + speed) * positions / (2.0 * effective_dispersion_m2_s)
    first_argument  = (positions - speed * elapsed_s) / denominator
    second_argument = (positions + speed * elapsed_s) / denominator
    return 0.5 * (np.exp(first_exponent) * erfc(first_argument) + np.exp(second_exponent) * erfc(second_argument))


def analytical_pulse(positions_m, times_s, scenario):
    """
    Evaluate the locked normalized finite-pulse analytical solution.

    Parameters
    ----------
    positions_m : numpy.ndarray
        Positive comparison coordinates in meters.
    times_s : numpy.ndarray
        Requested report times in seconds.
    scenario : mapping
        Physical velocity, dispersion, retardation, decay, and pulse times.

    Returns
    -------
    numpy.ndarray
        Array shaped `(times, positions)` of normalized concentrations.

    """
    velocity   = float(scenario["velocity_m_s"]) / float(scenario["retardation_factor"])
    dispersion = float(scenario["dispersion_m2_s"]) / float(scenario["retardation_factor"])
    decay      = float(scenario["decay_rate_s_1"])
    start      = float(scenario["source_start_s"])
    duration   = float(scenario["source_duration_s"])
    rows       = []
    for time_s in np.asarray(times_s, dtype=float):
        response_on  = analytical_step_response(positions_m, time_s - start, velocity, dispersion, decay)
        response_off = analytical_step_response(positions_m, time_s - start - duration, velocity, dispersion, decay)
        values       = response_on - response_off
        values[(values < 0.0) & (values >= -1.0e-12)] = 0.0
        require(np.isfinite(values).all() and float(np.min(values)) >= 0.0, "Analytical reference produced an invalid concentration")
        rows.append(values)
    return np.vstack(rows)


def validate_analytical_reference(selected, field):
    """
    Compare the local analytical implementation with the locked challenge field.

    Parameters
    ----------
    selected : pandas.DataFrame
        Sixteen truth-selected numerical-reference scenarios.
    field : pandas.DataFrame
        Locked analytical challenge field on the published node grid.

    Returns
    -------
    dict
        Compared rows and maximum absolute normalized-concentration difference.

    Notes
    -----
    The inlet node is excluded because its exact discontinuous Dirichlet value
    is imposed directly by the finite-pulse boundary condition.
    """
    compared = 0
    maximum  = 0.0
    for _, scenario in selected.iterrows():
        scenario_field = field.loc[field["scenario_id"].eq(scenario["scenario_id"]) & field["x_m"].gt(0.0)].sort_values(["time_s", "x_m"])
        times           = np.sort(scenario_field["time_s"].unique().astype(float))
        positions       = np.sort(scenario_field["x_m"].unique().astype(float))
        calculated      = analytical_pulse(positions, times, scenario).ravel()
        reported        = scenario_field["normalized_concentration"].to_numpy(dtype=float)
        require(len(calculated) == len(reported), "Analytical validation row count differs from the locked field")
        maximum = max(maximum, float(np.max(np.abs(calculated - reported))))
        compared += len(reported)
    require(maximum <= 1.0e-10, "Local analytical response differs from the locked challenge field")
    return {"selected_scenarios": int(len(selected)), "compared_rows": int(compared), "maximum_absolute_difference": maximum}


def build_transport_operator(cell_count, spatial_step_m, effective_velocity_m_s, effective_dispersion_m2_s, decay_rate_s_1):
    """
    Assemble the cell-centered Scharfetter-Gummel transport operator.

    Parameters
    ----------
    cell_count : int
        Number of finite-volume cells on the 1,200 m computational domain.
    spatial_step_m : float
        Uniform cell width in meters.
    effective_velocity_m_s : float
        Retardation-scaled velocity in meters per second.
    effective_dispersion_m2_s : float
        Retardation-scaled dispersion in square meters per second.
    decay_rate_s_1 : float
        First-order reaction rate in inverse seconds.

    Returns
    -------
    tuple
        Sparse semidiscrete operator, inlet source coefficient, and face-flux
        coefficients required by the discrete mass-balance diagnostic.

    """
    cell_peclet = effective_velocity_m_s * spatial_step_m / effective_dispersion_m2_s
    alpha       = effective_dispersion_m2_s / spatial_step_m * bernoulli(-cell_peclet)
    beta        = effective_dispersion_m2_s / spatial_step_m * bernoulli(cell_peclet)
    half_step   = spatial_step_m / 2.0
    half_peclet = effective_velocity_m_s * half_step / effective_dispersion_m2_s
    inlet_alpha = effective_dispersion_m2_s / half_step * bernoulli(-half_peclet)
    inlet_beta  = effective_dispersion_m2_s / half_step * bernoulli(half_peclet)
    lower       = np.full(cell_count - 1, alpha / spatial_step_m)
    upper       = np.full(cell_count - 1, beta / spatial_step_m)
    diagonal    = np.full(cell_count, -(alpha + beta) / spatial_step_m - decay_rate_s_1)
    diagonal[0]  = -(alpha + inlet_beta) / spatial_step_m - decay_rate_s_1
    diagonal[-1] = -(effective_velocity_m_s + beta) / spatial_step_m - decay_rate_s_1
    operator      = diags((lower, diagonal, upper), offsets=(-1, 0, 1), shape=(cell_count, cell_count), format="csc")
    inlet_source  = inlet_alpha / spatial_step_m
    constant_residual = operator @ np.ones(cell_count) + decay_rate_s_1 * np.ones(cell_count)
    constant_residual[0] += inlet_source
    require(float(np.max(np.abs(constant_residual))) <= 5.0e-13, "Scharfetter-Gummel operator fails the constant-state transport check")
    require(math.isclose(alpha - beta, effective_velocity_m_s, rel_tol=1.0e-12, abs_tol=1.0e-14), "Interior fitted flux does not recover the advective flux")
    require(math.isclose(inlet_alpha - inlet_beta, effective_velocity_m_s, rel_tol=1.0e-12, abs_tol=1.0e-14), "Inlet fitted flux does not recover the advective flux")
    return operator, inlet_source, inlet_alpha, inlet_beta, cell_peclet


def solve_level(scenario, level, reference):
    """
    Solve one selected scenario on one locked refinement level.

    Parameters
    ----------
    scenario : mapping
        Selected physical scenario and pulse definition.
    level : mapping
        Grid name, cell width, and maximum time step from the protocol.
    reference : dict
        Locked computational domain and comparison interval.

    Returns
    -------
    dict
        Retained fields, grid metadata, runtime, positivity, and balance data.

    """
    domain_length = float(reference["computational_domain_length_m"])
    spatial_step  = float(level["spatial_step_m"])
    maximum_step  = float(level["maximum_time_step_s"])
    cell_count    = int(round(domain_length / spatial_step))
    require(math.isclose(cell_count * spatial_step, domain_length, rel_tol=0.0, abs_tol=1.0e-12), "Spatial grid does not partition the computational domain")
    positions     = (np.arange(cell_count, dtype=float) + 0.5) * spatial_step
    final_time    = float(scenario["final_time_s"])
    report_times  = np.arange(int(scenario["time_nodes"]), dtype=float) * float(scenario["time_step_s"])
    source_start  = float(scenario["source_start_s"])
    source_end    = float(scenario["source_end_s"])
    times         = build_time_grid(final_time, maximum_step, report_times, source_start, source_end)
    velocity      = float(scenario["velocity_m_s"]) / float(scenario["retardation_factor"])
    dispersion    = float(scenario["dispersion_m2_s"]) / float(scenario["retardation_factor"])
    decay         = float(scenario["decay_rate_s_1"])
    operator, inlet_source, inlet_alpha, inlet_beta, cell_peclet = build_transport_operator(cell_count, spatial_step, velocity, dispersion, decay)
    report_indexes = {int(np.flatnonzero(times == value)[0]): index for index, value in enumerate(report_times)}
    require(len(report_indexes) == len(report_times), "A report time is absent from the integration grid")
    snapshots       = np.empty((len(report_times), cell_count), dtype=float)
    snapshots[0]    = 0.0
    current          = np.zeros(cell_count, dtype=float)
    minimum          = 0.0
    maximum          = 0.0
    maximum_balance  = 0.0
    maximum_relative = 0.0
    factorization_cache = {}
    identity         = eye(cell_count, format="csc")
    started          = time.perf_counter()
    for time_index in range(1, len(times)):
        left_time  = float(times[time_index - 1])
        right_time = float(times[time_index])
        time_step  = right_time - left_time
        cache_key  = float(round(time_step, 12))
        if cache_key not in factorization_cache:
            factorization_cache[cache_key] = splu(identity - time_step * operator)
        boundary = inlet_value(0.5 * (left_time + right_time), source_start, source_end)
        previous = current
        right_hand_side = previous.copy()
        right_hand_side[0] += time_step * inlet_source * boundary
        current = factorization_cache[cache_key].solve(right_hand_side)
        mass_previous = float(np.sum(previous) * spatial_step)
        mass_current  = float(np.sum(current) * spatial_step)
        inlet_flux    = inlet_alpha * boundary - inlet_beta * current[0]
        outlet_flux   = velocity * current[-1]
        reaction_loss = decay * mass_current
        balance       = (mass_current - mass_previous) / time_step + outlet_flux - inlet_flux + reaction_loss
        scale         = max(abs(inlet_flux), abs(outlet_flux), abs(reaction_loss), 1.0e-14)
        maximum_balance  = max(maximum_balance, abs(balance))
        maximum_relative = max(maximum_relative, abs(balance) / scale)
        minimum = min(minimum, float(np.min(current)))
        maximum = max(maximum, float(np.max(current)))
        if time_index in report_indexes:
            snapshots[report_indexes[time_index]] = current
    runtime = time.perf_counter() - started
    require(np.isfinite(snapshots).all(), "Numerical solution contains a non-finite value")
    require(np.allclose(snapshots[0], 0.0, rtol=0.0, atol=0.0), "Numerical initial condition is not exact")
    return {"level": str(level["name"]), "spatial_step_m": spatial_step, "maximum_time_step_s": maximum_step, "cell_count": cell_count, "positions_m": positions, "report_times_s": report_times, "snapshots": snapshots, "time_steps": int(len(times) - 1), "unique_time_steps": int(len(factorization_cache)), "cell_peclet_number": float(cell_peclet), "minimum_normalized_concentration": minimum, "maximum_normalized_concentration": maximum, "mass_balance_maximum_absolute_residual": maximum_balance, "mass_balance_maximum_relative_residual": maximum_relative, "runtime_seconds": float(runtime), "source_switch_times_inserted_exactly": True}


def field_metrics(reference, prediction, active_threshold):
    """
    Calculate normalized-field errors for one numerical grid.

    Parameters
    ----------
    reference : numpy.ndarray
        Analytical concentration field on numerical cell centers.
    prediction : numpy.ndarray
        Matching finite-volume concentration field.
    active_threshold : float
        Analytical concentration threshold defining active rows.

    Returns
    -------
    dict
        MAE, RMSE, R2, maximum error, and active-region RMSE.

    """
    reference  = np.asarray(reference, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    residual   = prediction - reference
    active     = reference > active_threshold
    denominator = np.sum(np.square(reference - reference.mean()))
    return {"rows": int(reference.size), "active_rows": int(active.sum()), "mae": float(np.mean(np.abs(residual))), "rmse": float(np.sqrt(np.mean(np.square(residual)))), "r2": float(1.0 - np.sum(np.square(residual)) / denominator) if denominator > 0.0 else None, "maximum_absolute_error": float(np.max(np.abs(residual))), "active_rmse": float(np.sqrt(np.mean(np.square(residual[active])))) if active.any() else None}


def arrival_error(reference, prediction, report_times_s, active_threshold):
    """
    Calculate median first-threshold-crossing error over active positions.

    Parameters
    ----------
    reference : numpy.ndarray
        Analytical field shaped `(times, positions)`.
    prediction : numpy.ndarray
        Numerical field on the same report grid.
    report_times_s : numpy.ndarray
        Uniform report times in seconds.
    active_threshold : float
        Threshold defining first arrival.

    Returns
    -------
    dict
        Median absolute error, compared positions, and missed arrivals.

    """
    errors = []
    missed = 0
    penalty = float(report_times_s[-1] + report_times_s[1] - report_times_s[0])
    for position_index in range(reference.shape[1]):
        actual_indexes = np.flatnonzero(reference[:, position_index] > active_threshold)
        if actual_indexes.size == 0:
            continue
        predicted_indexes = np.flatnonzero(prediction[:, position_index] > active_threshold)
        actual_time = float(report_times_s[actual_indexes[0]])
        if predicted_indexes.size:
            predicted_time = float(report_times_s[predicted_indexes[0]])
        else:
            predicted_time = penalty
            missed += 1
        errors.append(abs(predicted_time - actual_time))
    return {"median_absolute_error_s": float(np.median(errors)) if errors else None, "positions": len(errors), "missed_positions": missed}


def restrict_cell_averages(fine_values, ratio):
    """
    Restrict a fine cell-average field to a nested coarser finite-volume grid.

    Parameters
    ----------
    fine_values : numpy.ndarray
        Field shaped `(times, fine cells)`.
    ratio : int
        Integer number of fine cells contained in each coarse cell.

    Returns
    -------
    numpy.ndarray
        Conservative arithmetic cell averages on the coarser grid.

    """
    require(fine_values.shape[1] % ratio == 0, "Fine grid cannot be restricted by the requested integer ratio")
    return fine_values.reshape(fine_values.shape[0], fine_values.shape[1] // ratio, ratio).mean(axis=2)


def evaluate_scenario(scenario, protocol):
    """
    Solve and compare all three refinement levels for one selected scenario.

    Parameters
    ----------
    scenario : mapping
        Selected challenge scenario with truth-only selection metadata.
    protocol : dict
        Locked grids, comparison interval, metrics, and acceptance criteria.

    Returns
    -------
    tuple
        Level records for the CSV and a scenario-level acceptance summary.

    """
    reference        = protocol["traditional_numerical_reference"]
    active_threshold = float(protocol["metrics"]["active_normalized_concentration_threshold"])
    comparison_end   = float(reference["reported_comparison_interval_m"][1])
    solutions        = {}
    level_records    = []
    for level in reference["grid_levels"]:
        solution   = solve_level(scenario, level, reference)
        mask       = solution["positions_m"] <= comparison_end
        analytical = analytical_pulse(solution["positions_m"][mask], solution["report_times_s"], scenario)
        numerical  = solution["snapshots"][:, mask]
        metrics    = field_metrics(analytical, numerical, active_threshold)
        arrival    = arrival_error(analytical, numerical, solution["report_times_s"], active_threshold)
        record = {"scenario_id": str(scenario["scenario_id"]), "regime": str(scenario["regime"]), "selection_order_within_regime": int(scenario["selection_order_within_regime"]), "selection_criterion": str(scenario["selection_criterion"]), "level": solution["level"], "spatial_step_m": solution["spatial_step_m"], "maximum_time_step_s": solution["maximum_time_step_s"], "cell_count": solution["cell_count"], "time_steps": solution["time_steps"], "unique_time_steps": solution["unique_time_steps"], "cell_peclet_number": solution["cell_peclet_number"], "comparison_rows": metrics["rows"], "active_rows": metrics["active_rows"], "mae": metrics["mae"], "rmse": metrics["rmse"], "r2": metrics["r2"], "maximum_absolute_error": metrics["maximum_absolute_error"], "active_rmse": metrics["active_rmse"], "arrival_time_median_absolute_error_s": arrival["median_absolute_error_s"], "arrival_positions": arrival["positions"], "missed_arrival_positions": arrival["missed_positions"], "minimum_normalized_concentration": solution["minimum_normalized_concentration"], "maximum_normalized_concentration": solution["maximum_normalized_concentration"], "mass_balance_maximum_absolute_residual": solution["mass_balance_maximum_absolute_residual"], "mass_balance_maximum_relative_residual": solution["mass_balance_maximum_relative_residual"], "runtime_seconds": solution["runtime_seconds"]}
        level_records.append(record)
        solutions[solution["level"]] = {"solution": solution, "metrics": metrics}
    coarse = solutions["coarse"]
    medium = solutions["medium"]
    fine   = solutions["fine"]
    medium_from_fine = restrict_cell_averages(fine["solution"]["snapshots"], 2)
    coarse_from_medium = restrict_cell_averages(medium["solution"]["snapshots"], 2)
    medium_mask = medium["solution"]["positions_m"] <= comparison_end
    coarse_mask = coarse["solution"]["positions_m"] <= comparison_end
    medium_fine_rmse = float(np.sqrt(np.mean(np.square(medium["solution"]["snapshots"][:, medium_mask] - medium_from_fine[:, medium_mask]))))
    coarse_medium_rmse = float(np.sqrt(np.mean(np.square(coarse["solution"]["snapshots"][:, coarse_mask] - coarse_from_medium[:, coarse_mask]))))
    coarse_error = coarse["metrics"]["rmse"]
    medium_error = medium["metrics"]["rmse"]
    fine_error   = fine["metrics"]["rmse"]
    order_coarse_medium = float(math.log(coarse_error / medium_error, 2.0)) if coarse_error > 0.0 and medium_error > 0.0 else None
    order_medium_fine   = float(math.log(medium_error / fine_error, 2.0)) if medium_error > 0.0 and fine_error > 0.0 else None
    acceptance = reference["acceptance"]
    minimum = min(record["minimum_normalized_concentration"] for record in level_records)
    checks = {"fine_vs_analytical_rmse": fine_error <= float(acceptance["fine_vs_analytical_rmse_max"]), "medium_vs_fine_rmse": medium_fine_rmse <= float(acceptance["medium_vs_fine_rmse_max"]), "minimum_normalized_concentration": minimum >= float(acceptance["minimum_allowed_normalized_concentration"])}
    outcome = "meets_all_pre_specified_criteria" if all(checks.values()) else "mixed_evidence" if any(checks.values()) else "does_not_meet_pre_specified_criteria"
    summary = {"scenario_id": str(scenario["scenario_id"]), "regime": str(scenario["regime"]), "selection_order_within_regime": int(scenario["selection_order_within_regime"]), "selection_criterion": str(scenario["selection_criterion"]), "truth": {"peclet_number": float(scenario["peclet_number"]), "damkohler_number": float(scenario["damkohler_number"]), "advective_travel_time_s": float(scenario["advective_travel_time_s"])}, "rmse_by_level": {"coarse": coarse_error, "medium": medium_error, "fine": fine_error}, "intergrid_rmse": {"coarse_vs_restricted_medium": coarse_medium_rmse, "medium_vs_restricted_fine": medium_fine_rmse}, "observed_error_order": {"coarse_to_medium": order_coarse_medium, "medium_to_fine": order_medium_fine}, "minimum_normalized_concentration": minimum, "checks": checks, "outcome": outcome}
    return level_records, summary


def write_csv(table, path):
    """
    Write a deterministic numerical-reference table.

    Parameters
    ----------
    table : pandas.DataFrame
        Ordered level-by-scenario evidence.
    path : pathlib.Path
        Destination CSV path.

    Returns
    -------
    dict
        Path, dimensions, byte count, and SHA-256 digest.

    """
    table.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "rows": int(len(table)), "columns": int(len(table.columns)), "bytes": path.stat().st_size, "sha256": sha256(path)}


def execute_numerical_reference(output_directory, overwrite=False):
    """
    Execute the locked 16-case, three-level traditional numerical reference.

    Parameters
    ----------
    output_directory : pathlib.Path
        Existing or creatable destination for CSV and JSON artifacts.
    overwrite : bool, optional
        Whether existing numerical-reference outputs may be replaced.

    Returns
    -------
    dict
        Aggregate numerical-reference report persisted as JSON.

    """
    protocol       = load_json(PROTOCOL_PATH)
    challenge_lock = load_json(CHALLENGE_LOCK_PATH)
    require(protocol["status"] == "locked_before_challenge_generation_and_model_inference", "Validation protocol is not locked")
    require(sha256(PROTOCOL_PATH) == challenge_lock["protocol"]["sha256"], "Protocol digest differs from the challenge lock")
    require(sha256(SCENARIO_PATH) == challenge_lock["challenge_files"]["challenge_scenarios"]["sha256"], "Challenge scenarios differ from their lock")
    require(sha256(FIELD_PATH) == challenge_lock["challenge_files"]["challenge_analytical_field"]["sha256"], "Analytical challenge field differs from its lock")
    scenarios = pd.read_csv(SCENARIO_PATH)
    field      = pd.read_csv(FIELD_PATH)
    selected  = select_reference_scenarios(scenarios, protocol)
    analytical_validation = validate_analytical_reference(selected, field)
    output_directory.mkdir(parents=True, exist_ok=True)
    cases_path   = output_directory / CASES_NAME
    metrics_path = output_directory / METRICS_NAME
    if not overwrite:
        require(not cases_path.exists() and not metrics_path.exists(), "Numerical-reference output already exists; use --overwrite only for an intentional replacement")
    started            = time.perf_counter()
    case_records       = []
    scenario_summaries = []
    for _, scenario in selected.iterrows():
        records, summary = evaluate_scenario(scenario, protocol)
        case_records.extend(records)
        scenario_summaries.append(summary)
    total_runtime = time.perf_counter() - started
    cases         = pd.DataFrame(case_records)
    require(len(cases) == 48, "Numerical-reference table must contain 16 scenarios on three levels")
    require(cases.groupby("scenario_id").size().eq(3).all(), "A selected scenario does not contain three refinement levels")
    case_artifact = write_csv(cases, cases_path)
    fine_rows     = cases.loc[cases["level"].eq("fine")]
    intergrid     = np.asarray([item["intergrid_rmse"]["medium_vs_restricted_fine"] for item in scenario_summaries], dtype=float)
    outcomes      = [item["outcome"] for item in scenario_summaries]
    failed_ids    = [item["scenario_id"] for item in scenario_summaries if item["outcome"] != "meets_all_pre_specified_criteria"]
    monotonic     = [item["rmse_by_level"]["coarse"] > item["rmse_by_level"]["medium"] > item["rmse_by_level"]["fine"] for item in scenario_summaries]
    coarse_medium_orders = np.asarray([item["observed_error_order"]["coarse_to_medium"] for item in scenario_summaries], dtype=float)
    medium_fine_orders   = np.asarray([item["observed_error_order"]["medium_to_fine"] for item in scenario_summaries], dtype=float)
    checks        = {"selected_scenarios": len(scenario_summaries) == 16, "three_grid_levels_per_scenario": len(cases) == 48, "all_fine_vs_analytical_rmse": all(item["checks"]["fine_vs_analytical_rmse"] for item in scenario_summaries), "all_medium_vs_fine_rmse": all(item["checks"]["medium_vs_fine_rmse"] for item in scenario_summaries), "all_minimum_concentrations": all(item["checks"]["minimum_normalized_concentration"] for item in scenario_summaries)}
    overall_outcome = "meets_all_pre_specified_criteria" if all(checks.values()) else "mixed_evidence" if any(checks.values()) else "does_not_meet_pre_specified_criteria"
    report = {
        "status": "numerical_reference_complete",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "challenge_lock_sha256": sha256(CHALLENGE_LOCK_PATH),
        "challenge_scenarios_sha256": sha256(SCENARIO_PATH),
        "challenge_analytical_field_sha256": sha256(FIELD_PATH),
        "analytical_reference_validation": analytical_validation,
        "implementation": {"method": protocol["traditional_numerical_reference"]["method"], "equation": protocol["traditional_numerical_reference"]["equation_after_retardation_scaling"], "inlet_boundary": protocol["traditional_numerical_reference"]["inlet_boundary"], "outlet_boundary": protocol["traditional_numerical_reference"]["outlet_boundary"], "script_path": str(Path(__file__).resolve().relative_to(ROOT)), "script_sha256": sha256(Path(__file__).resolve())},
        "selection": {"uses_only_truth_parameters": True, "tie_break": protocol["traditional_numerical_reference"]["selection_tie_break"], "selected_scenarios": [{"scenario_id": str(row["scenario_id"]), "regime": str(row["regime"]), "selection_order_within_regime": int(row["selection_order_within_regime"]), "selection_criterion": str(row["selection_criterion"]), "peclet_number": float(row["peclet_number"]), "damkohler_number": float(row["damkohler_number"]), "advective_travel_time_s": float(row["advective_travel_time_s"]), "medoid_total_distance": None if pd.isna(row["medoid_total_distance"]) else float(row["medoid_total_distance"])} for _, row in selected.iterrows()]},
        "execution": {"selected_scenarios": 16, "grid_levels_per_scenario": 3, "traditional_solver_runs": 48, "source_switch_times_inserted_exactly": True, "total_runtime_seconds": float(total_runtime), "thread_scope": "Numerical reference executed under caller-provided process thread settings."},
        "aggregate": {"fine_rmse": {"median": float(fine_rows["rmse"].median()), "percentile_90": float(fine_rows["rmse"].quantile(0.90)), "maximum": float(fine_rows["rmse"].max())}, "medium_vs_fine_rmse": {"median": float(np.median(intergrid)), "percentile_90": float(np.quantile(intergrid, 0.90)), "maximum": float(np.max(intergrid))}, "observed_error_order": {"coarse_to_medium_median": float(np.median(coarse_medium_orders)), "medium_to_fine_median": float(np.median(medium_fine_orders)), "all_analytical_rmse_monotonically_decreasing": bool(all(monotonic))}, "minimum_normalized_concentration": float(cases["minimum_normalized_concentration"].min()), "maximum_mass_balance_absolute_residual": float(cases["mass_balance_maximum_absolute_residual"].max()), "maximum_mass_balance_relative_residual": float(cases["mass_balance_maximum_relative_residual"].max())},
        "acceptance": {"criteria": protocol["traditional_numerical_reference"]["acceptance"], "checks": checks, "scenarios_meeting_all_criteria": int(sum(value == "meets_all_pre_specified_criteria" for value in outcomes)), "scenarios_total": len(outcomes), "scenarios_not_meeting_all_criteria": failed_ids, "overall_outcome": overall_outcome},
        "scenario_evidence": scenario_summaries,
        "artifacts": {"case_metrics": case_artifact},
        "software": {"implementation": platform.python_implementation(), "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "platform": platform.platform()},
        "reporting_scope": "The timing is descriptive for this execution and is not the seven-repetition computational-cost benchmark.",
    }
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main():
    """
    Run the numerical reference and print its compact completion record.

    Returns
    -------
    None
        Numerical case metrics and an aggregate JSON report are persisted.

    """
    arguments = parse_arguments()
    report    = execute_numerical_reference(arguments.output_directory.resolve(), arguments.overwrite)
    summary   = {"status": report["status"], "overall_outcome": report["acceptance"]["overall_outcome"], "selected_scenarios": report["execution"]["selected_scenarios"], "traditional_solver_runs": report["execution"]["traditional_solver_runs"], "fine_rmse_maximum": report["aggregate"]["fine_rmse"]["maximum"], "medium_vs_fine_rmse_maximum": report["aggregate"]["medium_vs_fine_rmse"]["maximum"]}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
