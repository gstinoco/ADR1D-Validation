"""
================================================================================
ADR1D Validation: Independent Challenge-Set Verification
================================================================================

Independently reconstruct and verify the locked ADR1D challenge scenarios,
analytical fields, and noisy sensor realizations before model inference.

Main Operations
---------------
1. Verify the frozen protocol, source artifacts, and generated manifest.
2. Reconstruct the design, analytical solution, and noise streams independently.
3. Reject malformed, duplicated, collided, or numerically inconsistent records.
4. Persist a validation report and the pre-inference challenge lock.

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
import csv
import hashlib
import json
import math
import platform
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT                   = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL       = ROOT / "configs/validation_protocol.json"
PROTOCOL_LOCK          = ROOT / "results/validation_protocol_lock.json"
DEFAULT_DATA_DIR       = ROOT / "data"
DEFAULT_RESULTS_DIR    = ROOT / "results"
GENERATOR              = ROOT / "src/generate_challenge_set.py"
SCENARIO_FILE_NAME     = "challenge_scenarios.csv"
FIELD_FILE_NAME        = "challenge_analytical_field.csv"
SENSOR_FILE_NAME       = "challenge_sensor_observations.csv"
MANIFEST_FILE_NAME     = "challenge_manifest.json"
VALIDATION_FILE_NAME   = "challenge_validation.json"
CHALLENGE_LOCK_NAME    = "challenge_lock.json"
NUMERIC_ABSOLUTE_LIMIT = 2.0e-10
COLLISION_TOLERANCE    = 1.0e-12

SCENARIO_FIELDS = [
    "scenario_id",
    "split",
    "regime",
    "design_component",
    "design_index",
    "domain_length_m",
    "spatial_nodes",
    "spatial_step_m",
    "final_time_s",
    "time_nodes",
    "time_step_s",
    "velocity_m_s",
    "dispersion_m2_s",
    "retardation_factor",
    "decay_rate_s_1",
    "source_concentration_mg_L",
    "source_start_s",
    "source_duration_s",
    "source_end_s",
    "dispersivity_m",
    "peclet_number",
    "damkohler_number",
    "advective_travel_time_s",
]
FIELD_FIELDS = [
    "scenario_id",
    "split",
    "regime",
    "design_component",
    "time_s",
    "x_m",
    "concentration_mg_L",
    "normalized_concentration",
]
SENSOR_FIELDS = [
    "observation_id",
    "base_scenario_id",
    "scenario_id",
    "replicate_id",
    "split",
    "regime",
    "sensor_id",
    "x_m",
    "time_s",
    "concentration_true_mg_L",
    "noise_std_mg_L",
    "concentration_observed_mg_L",
    "detection_limit_mg_L",
    "is_below_detection_limit",
]


class ValidationError(RuntimeError):
    """Represent a violation of the locked challenge-set contract."""


def sha256(path):
    """
    Compute the SHA-256 digest of a file without loading it entirely into memory.

    Parameters
    ----------
    path : pathlib.Path
        File whose digest is required.

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


def format_float(value):
    """
    Format a numeric value according to the frozen ADR1D CSV contract.

    Parameters
    ----------
    value : float
        Finite numeric value to serialize.

    Returns
    -------
    str
        Stable decimal representation with twelve significant digits.

    """
    return format(value, ".12g")


def parse_args():
    """
    Parse challenge-validation paths.

    Returns
    -------
    argparse.Namespace
        Protocol, data-directory, and results-directory arguments.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def project_path(relative_path):
    """
    Resolve a protocol path relative to the activity root.

    Parameters
    ----------
    relative_path : str
        Relative or absolute path stored in the validation protocol.

    Returns
    -------
    pathlib.Path
        Absolute normalized filesystem path.

    """
    path = Path(relative_path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_json(path):
    """
    Load a UTF-8 JSON object from disk.

    Parameters
    ----------
    path : pathlib.Path
        Existing JSON file.

    Returns
    -------
    dict
        Parsed top-level JSON object.

    Raises
    ------
    FileNotFoundError
        If the requested file does not exist.
    ValidationError
        If the top-level JSON value is not an object.

    """
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValidationError(f"Expected a JSON object in {path}")
    return value


def require(condition, message):
    """
    Raise a validation error when a required condition is false.

    Parameters
    ----------
    condition : bool
        Contract condition to test.
    message : str
        Actionable failure description.

    Returns
    -------
    None
        The function returns only when the condition is true.

    Raises
    ------
    ValidationError
        If `condition` is false.

    """
    if not condition:
        raise ValidationError(message)


def require_absent(paths):
    """
    Refuse to overwrite a validation report or challenge lock.

    Parameters
    ----------
    paths : iterable of pathlib.Path
        Output paths that must not exist.

    Returns
    -------
    None
        The function validates the output state.

    Raises
    ------
    FileExistsError
        If any target path already exists.

    """
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite validation artifacts: " + ", ".join(str(path) for path in existing))


def load_locked_protocol(protocol_path):
    """
    Load the protocol after verifying its immutable pre-generation lock.

    Parameters
    ----------
    protocol_path : pathlib.Path
        Machine-readable validation protocol.

    Returns
    -------
    tuple of dict
        Verified protocol and protocol-lock objects.

    Raises
    ------
    ValidationError
        If identity, digest, or zero-query state differs from the lock.

    """
    protocol = load_json(protocol_path)
    lock     = load_json(PROTOCOL_LOCK)
    require(sha256(protocol_path) == lock["protocol"]["sha256"], "Validation protocol hash differs from its lock")
    require(protocol["protocol_id"] == lock["protocol"]["protocol_id"], "Validation protocol identifier differs from its lock")
    require(protocol["protocol_version"] == lock["protocol"]["protocol_version"], "Validation protocol version differs from its lock")
    require(protocol["status"] == "locked_before_challenge_generation_and_model_inference", "Validation protocol is not in its locked pre-generation state")
    zero_keys = ("challenge_scenarios_generated", "adr1d_ml_challenge_inference_runs", "adr1d_nn_challenge_inference_runs", "traditional_numerical_solver_runs")
    require(all(int(lock["state_at_lock"][key]) == 0 for key in zero_keys), "Protocol lock does not record a zero-query challenge state")
    return protocol, lock


def collect_frozen_files(value):
    """
    Collect path-and-digest records recursively from frozen protocol inputs.

    Parameters
    ----------
    value : object
        Nested protocol value to inspect.

    Returns
    -------
    list of dict
        Frozen records containing both `path` and `sha256` fields.

    """
    records = []
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            records.append(value)
        else:
            for nested in value.values():
                records.extend(collect_frozen_files(nested))
    elif isinstance(value, list):
        for nested in value:
            records.extend(collect_frozen_files(nested))
    return records


def verify_frozen_inputs(protocol, protocol_lock):
    """
    Verify every frozen file referenced by the protocol without deserializing models.

    Parameters
    ----------
    protocol : dict
        Locked validation protocol.
    protocol_lock : dict
        Immutable pre-generation lock.

    Returns
    -------
    dict
        Number of verified files and mismatch count.

    Raises
    ------
    FileNotFoundError
        If a frozen file is absent.
    ValidationError
        If a digest differs from the frozen value.

    """
    records = collect_frozen_files(protocol["frozen_inputs"])
    expected_count = int(protocol_lock["integrity_verification"]["frozen_inputs_verified"])
    require(len(records) == expected_count, f"Expected {expected_count} frozen file records, found {len(records)}")
    for record in records:
        path = project_path(record["path"])
        if not path.exists():
            raise FileNotFoundError(path)
        require(sha256(path) == record["sha256"], f"Frozen input hash mismatch: {record['path']}")
    return {"frozen_files_verified": len(records), "frozen_file_mismatches": 0, "serialized_models_loaded": 0}


def linear_scale(unit_value, bounds):
    """
    Map a unit-interval value to a bounded linear interval independently.

    Parameters
    ----------
    unit_value : float
        Unit-interval coordinate.
    bounds : sequence of float
        Inclusive lower and upper physical bounds.

    Returns
    -------
    float
        Linearly scaled physical value.

    """
    low, high = (float(value) for value in bounds)
    return low + unit_value * (high - low)


def log_scale(unit_value, bounds):
    """
    Map a unit-interval value to a positive logarithmic interval independently.

    Parameters
    ----------
    unit_value : float
        Unit-interval coordinate.
    bounds : sequence of float
        Inclusive positive lower and upper physical bounds.

    Returns
    -------
    float
        Logarithmically scaled physical value.

    """
    low, high = (float(value) for value in bounds)
    return 10.0 ** linear_scale(unit_value, (math.log10(low), math.log10(high)))


def inverse_linear(value, bounds):
    """
    Normalize a physical value from a linear interval.

    Parameters
    ----------
    value : float
        Physical value to normalize.
    bounds : sequence of float
        Inclusive lower and upper physical bounds.

    Returns
    -------
    float
        Corresponding unit-interval coordinate.

    """
    low, high = (float(item) for item in bounds)
    return (float(value) - low) / (high - low)


def inverse_log(value, bounds):
    """
    Normalize a positive physical value from a logarithmic interval.

    Parameters
    ----------
    value : float
        Positive physical value to normalize.
    bounds : sequence of float
        Inclusive positive lower and upper physical bounds.

    Returns
    -------
    float
        Corresponding logarithmic unit coordinate.

    """
    low, high = (math.log10(float(item)) for item in bounds)
    return (math.log10(float(value)) - low) / (high - low)


def latin_hypercube(sample_count, dimension_count, rng):
    """
    Reconstruct the protocol-defined open-interval Latin hypercube independently.

    Parameters
    ----------
    sample_count : int
        Number of rows in the design.
    dimension_count : int
        Number of independently stratified dimensions.
    rng : random.Random
        Seeded standard-library random stream.

    Returns
    -------
    list of list of float
        Unit-coordinate matrix with one row per scenario.

    """
    columns = []
    for _ in range(dimension_count):
        column = [(index + rng.random()) / sample_count for index in range(sample_count)]
        rng.shuffle(column)
        columns.append(column)
    return [[columns[dimension][sample] for dimension in range(dimension_count)] for sample in range(sample_count)]


def build_expected_scenario(scenario_id, regime, design_component, design_index, unit_coordinates, protocol):
    """
    Reconstruct one expected challenge scenario from the locked physical design.

    Parameters
    ----------
    scenario_id : str
        Stable base-scenario identifier.
    regime : str
        Protocol-defined physical regime.
    design_component : str
        Interior Latin hypercube or boundary profile.
    design_index : int
        One-based row index within the component and regime.
    unit_coordinates : sequence of float
        Seven coordinates in the locked parameter order.
    protocol : dict
        Locked validation protocol.

    Returns
    -------
    dict
        Full-precision scenario metadata reconstructed without generator imports.

    """
    design               = protocol["challenge_design"]
    ranges               = design["parameter_ranges"]
    domain               = design["domain"]
    velocity             = linear_scale(unit_coordinates[0], ranges["velocity_m_s"])
    dispersion           = log_scale(unit_coordinates[1], ranges["dispersion_m2_s"])
    source_concentration = log_scale(unit_coordinates[2], ranges["source_concentration_mg_L"])
    source_start         = linear_scale(unit_coordinates[3], ranges["source_start_s"])
    source_duration      = linear_scale(unit_coordinates[4], ranges["source_duration_s"])
    retardation_active   = regime in {"retardation_only", "retardation_and_decay"}
    decay_active         = regime in {"decay_only", "retardation_and_decay"}
    retardation          = linear_scale(unit_coordinates[5], ranges["retardation_factor_when_active"]) if retardation_active else float(design["inactive_parameter_values"]["retardation_factor"])
    decay                = log_scale(unit_coordinates[6], ranges["decay_rate_s_1_when_active"]) if decay_active else float(design["inactive_parameter_values"]["decay_rate_s_1"])
    length               = float(domain["length_m"])
    source_end           = source_start + source_duration
    return {
        "scenario_id": scenario_id,
        "split": design["split_label"],
        "regime": regime,
        "design_component": design_component,
        "design_index": design_index,
        "domain_length_m": length,
        "spatial_nodes": int(domain["spatial_nodes"]),
        "spatial_step_m": float(domain["spatial_step_m"]),
        "final_time_s": float(domain["final_time_s"]),
        "time_nodes": int(domain["time_nodes"]),
        "time_step_s": float(domain["time_step_s"]),
        "velocity_m_s": velocity,
        "dispersion_m2_s": dispersion,
        "retardation_factor": retardation,
        "decay_rate_s_1": decay,
        "source_concentration_mg_L": source_concentration,
        "source_start_s": source_start,
        "source_duration_s": source_duration,
        "source_end_s": source_end,
        "dispersivity_m": dispersion / velocity,
        "peclet_number": velocity * length / dispersion,
        "damkohler_number": decay * retardation * length / velocity,
        "advective_travel_time_s": retardation * length / velocity,
        "unit_coordinates": list(unit_coordinates),
    }


def build_expected_scenarios(protocol):
    """
    Reconstruct every challenge scenario from the frozen seed and boundary rows.

    Parameters
    ----------
    protocol : dict
        Locked validation protocol.

    Returns
    -------
    list of dict
        Ordered full-precision scenarios with their latent unit coordinates.

    """
    design          = protocol["challenge_design"]
    rng             = random.Random(int(design["parameter_seed"]))
    scenarios       = []
    scenario_number = 1
    interior_count  = int(design["interior_latin_hypercube_per_regime"])
    dimension_count = len(design["parameter_order"])

    for regime in design["regimes"]:
        interior_rows = latin_hypercube(interior_count, dimension_count, rng)
        for index, coordinates in enumerate(interior_rows, start=1):
            scenario_id = f"A5-CH-{scenario_number:04d}"
            scenarios.append(build_expected_scenario(scenario_id, regime, "latin_hypercube", index, coordinates, protocol))
            scenario_number += 1
        for index, coordinates in enumerate(design["boundary_profile_unit_coordinates"], start=1):
            scenario_id = f"A5-CH-{scenario_number:04d}"
            scenarios.append(build_expected_scenario(scenario_id, regime, "boundary_profile", index, coordinates, protocol))
            scenario_number += 1

    require(len(scenarios) == int(design["base_scenarios"]), "Reconstructed scenario count differs from the protocol")
    return scenarios


def serialized_scenario_value(value):
    """
    Convert one expected scenario value to its canonical CSV representation.

    Parameters
    ----------
    value : object
        Scenario value represented as text, integer, or floating point.

    Returns
    -------
    str
        Canonical serialized value.

    """
    return format_float(value) if isinstance(value, float) else str(value)


def active_unit_dimensions(regime):
    """
    Return the sampled unit-coordinate indexes that remain active in a regime.

    Parameters
    ----------
    regime : str
        Protocol-defined physical regime.

    Returns
    -------
    list of int
        Active coordinate indexes in locked parameter order.

    """
    dimensions = [0, 1, 2, 3, 4]
    if regime in {"retardation_only", "retardation_and_decay"}:
        dimensions.append(5)
    if regime in {"decay_only", "retardation_and_decay"}:
        dimensions.append(6)
    return dimensions


def validate_latin_stratification(scenarios, protocol):
    """
    Verify one occupied Latin-hypercube stratum per active dimension and regime.

    Parameters
    ----------
    scenarios : list of dict
        Reconstructed full-precision challenge scenarios.
    protocol : dict
        Locked validation protocol.

    Returns
    -------
    dict
        Verified active dimensions for each physical regime.

    Raises
    ------
    ValidationError
        If a unit coordinate is outside the open interval or repeats a stratum.

    """
    design          = protocol["challenge_design"]
    parameter_names = design["parameter_order"]
    sample_count    = int(design["interior_latin_hypercube_per_regime"])
    verified        = {}
    for regime in design["regimes"]:
        interior = [row for row in scenarios if row["regime"] == regime and row["design_component"] == "latin_hypercube"]
        require(len(interior) == sample_count, f"Latin-hypercube count differs for {regime}")
        names = []
        for dimension in active_unit_dimensions(regime):
            coordinates = [float(row["unit_coordinates"][dimension]) for row in interior]
            require(all(0.0 < value < 1.0 for value in coordinates), f"Latin-hypercube coordinate is not open-interval in {regime}")
            strata = sorted(math.floor(value * sample_count) for value in coordinates)
            require(strata == list(range(sample_count)), f"Latin-hypercube strata are incomplete in {regime}, dimension {parameter_names[dimension]}")
            names.append(parameter_names[dimension])
        verified[regime] = names
    return verified


def normalized_coordinates(row, regime, ranges):
    """
    Normalize active physical parameters for canonical collision checks.

    Parameters
    ----------
    row : mapping
        Scenario record containing physical parameter values.
    regime : str
        Physical regime controlling active retardation and decay dimensions.
    ranges : dict
        Locked physical parameter ranges.

    Returns
    -------
    list of float
        Active normalized coordinates in a common unit space.

    """
    coordinates = [
        inverse_linear(row["velocity_m_s"], ranges["velocity_m_s"]),
        inverse_log(row["dispersion_m2_s"], ranges["dispersion_m2_s"]),
        inverse_log(row["source_concentration_mg_L"], ranges["source_concentration_mg_L"]),
        inverse_linear(row["source_start_s"], ranges["source_start_s"]),
        inverse_linear(row["source_duration_s"], ranges["source_duration_s"]),
    ]
    if regime in {"retardation_only", "retardation_and_decay"}:
        coordinates.append(inverse_linear(row["retardation_factor"], ranges["retardation_factor_when_active"]))
    if regime in {"decay_only", "retardation_and_decay"}:
        coordinates.append(inverse_log(row["decay_rate_s_1"], ranges["decay_rate_s_1_when_active"]))
    return coordinates


def validate_canonical_separation(scenarios, protocol):
    """
    Reject parameter duplicates and measure separation from canonical ADR1D cases.

    Parameters
    ----------
    scenarios : list of dict
        Reconstructed challenge scenarios.
    protocol : dict
        Locked protocol containing canonical metadata and collision tolerance.

    Returns
    -------
    dict
        Pair count, minimum normalized Euclidean distance, and closest identifiers.

    Raises
    ------
    ValidationError
        If identifiers overlap or all normalized coordinates collide within tolerance.

    """
    metadata       = protocol["frozen_inputs"]["benchmark"]["scenarios"]
    canonical_path = project_path(metadata["path"])
    ranges         = protocol["challenge_design"]["parameter_ranges"]
    with canonical_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        canonical_rows = list(reader)

    challenge_ids = {row["scenario_id"] for row in scenarios}
    canonical_ids = {row["scenario_id"] for row in canonical_rows}
    require(not challenge_ids.intersection(canonical_ids), "Challenge identifiers overlap canonical ADR1D identifiers")

    minimum_distance = math.inf
    closest_pair     = None
    compared_pairs   = 0
    for challenge in scenarios:
        regime                = challenge["regime"]
        challenge_coordinates = normalized_coordinates(challenge, regime, ranges)
        for canonical in canonical_rows:
            if canonical["regime"] != regime:
                continue
            canonical_coordinates = normalized_coordinates(canonical, regime, ranges)
            differences           = [abs(left - right) for left, right in zip(challenge_coordinates, canonical_coordinates)]
            require(max(differences) > COLLISION_TOLERANCE, f"Parameter collision between {challenge['scenario_id']} and {canonical['scenario_id']}")
            distance = math.sqrt(sum(value**2 for value in differences))
            compared_pairs += 1
            if distance < minimum_distance:
                minimum_distance = distance
                closest_pair = [challenge["scenario_id"], canonical["scenario_id"]]
    return {
        "canonical_scenarios": len(canonical_rows),
        "compared_same_regime_pairs": compared_pairs,
        "collision_tolerance_linf": COLLISION_TOLERANCE,
        "collisions": 0,
        "distance_metric": "Euclidean distance over active normalized parameters",
        "minimum_normalized_distance": minimum_distance,
        "closest_pair": closest_pair,
        "identifier_overlaps": 0,
    }


def validate_scenarios(path, protocol):
    """
    Validate scenario serialization, balance, stratification, and canonical novelty.

    Parameters
    ----------
    path : pathlib.Path
        Generated challenge-scenario CSV.
    protocol : dict
        Locked validation protocol.

    Returns
    -------
    tuple
        Full-precision scenarios, manifest summary, and validation diagnostics.

    Raises
    ------
    ValidationError
        If the table differs from any locked design requirement.

    """
    expected = build_expected_scenarios(protocol)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == SCENARIO_FIELDS, "Challenge-scenario header differs from the contract")
        rows = list(reader)
    require(len(rows) == len(expected), f"Expected {len(expected)} challenge scenarios, found {len(rows)}")

    for row_number, (actual, reference) in enumerate(zip(rows, expected), start=2):
        for field in SCENARIO_FIELDS:
            required = serialized_scenario_value(reference[field])
            require(actual[field] == required, f"Scenario mismatch at CSV row {row_number}, field {field}: {actual[field]} != {required}")

    identifiers = [row["scenario_id"] for row in rows]
    require(len(identifiers) == len(set(identifiers)), "Challenge-scenario identifiers are not unique")
    regime_counts = dict(sorted(Counter(row["regime"] for row in rows).items()))
    component_counts = dict(sorted(Counter(row["design_component"] for row in rows).items()))
    expected_regime_count = int(protocol["challenge_design"]["base_scenarios_per_regime"])
    require(all(regime_counts.get(regime) == expected_regime_count for regime in protocol["challenge_design"]["regimes"]), "Challenge scenarios are not balanced by regime")

    manifest_summary = {
        "rows": len(rows),
        "columns": len(SCENARIO_FIELDS),
        "regime_counts": regime_counts,
        "design_component_counts": component_counts,
    }
    validation = {
        "rows": len(rows),
        "columns": len(SCENARIO_FIELDS),
        "unique_identifiers": len(set(identifiers)),
        "regime_counts": regime_counts,
        "design_component_counts": component_counts,
        "latin_hypercube_active_dimensions": validate_latin_stratification(expected, protocol),
        "boundary_profiles_verified": int(protocol["challenge_design"]["boundary_profiles_per_regime"]) * len(protocol["challenge_design"]["regimes"]),
        "canonical_separation": validate_canonical_separation(expected, protocol),
        "serialization_mismatches": 0,
    }
    return expected, manifest_summary, validation


def step_response(x_m, elapsed_s, velocity_m_s, dispersion_m2_s, retardation_factor, decay_rate_s_1):
    """
    Evaluate the reactive semi-infinite step solution without generator imports.

    Parameters
    ----------
    x_m : float
        Spatial coordinate in meters.
    elapsed_s : float
        Time elapsed since activation of the inlet step, in seconds.
    velocity_m_s : float
        Pore-water velocity in meters per second.
    dispersion_m2_s : float
        Dispersion coefficient in square meters per second.
    retardation_factor : float
        Dimensionless linear retardation factor.
    decay_rate_s_1 : float
        First-order decay rate in inverse seconds.

    Returns
    -------
    float
        Dimensionless Ogata--Banks step response.

    Notes
    -----
    This implementation is intentionally local and does not import either the
    challenge generator or the ADR1D benchmark-generation module.

    """
    if elapsed_s <= 0.0:
        return 0.0
    reactive_speed = math.sqrt(velocity_m_s * velocity_m_s + 4.0 * dispersion_m2_s * retardation_factor * decay_rate_s_1)
    root_term      = 2.0 * math.sqrt(dispersion_m2_s * retardation_factor * elapsed_s)
    minus_weight   = math.exp((velocity_m_s - reactive_speed) * x_m / (2.0 * dispersion_m2_s))
    plus_weight    = math.exp((velocity_m_s + reactive_speed) * x_m / (2.0 * dispersion_m2_s))
    minus_argument = (retardation_factor * x_m - reactive_speed * elapsed_s) / root_term
    plus_argument  = (retardation_factor * x_m + reactive_speed * elapsed_s) / root_term
    return 0.5 * (minus_weight * math.erfc(minus_argument) + plus_weight * math.erfc(plus_argument))


def pulse_concentration(x_m, time_s, scenario):
    """
    Evaluate the independently reconstructed finite-pulse concentration.

    Parameters
    ----------
    x_m : float
        Spatial coordinate in meters.
    time_s : float
        Simulation time in seconds.
    scenario : mapping
        Full-precision physical parameters and source definition.

    Returns
    -------
    float
        Concentration in milligrams per liter.

    Raises
    ------
    ValidationError
        If the independent analytical result is negative or non-finite.

    """
    source_start = float(scenario["source_start_s"])
    source_end   = float(scenario["source_end_s"])
    source       = float(scenario["source_concentration_mg_L"])
    if x_m == 0.0:
        return source if source_start <= time_s < source_end else 0.0

    common        = (float(scenario["velocity_m_s"]), float(scenario["dispersion_m2_s"]), float(scenario["retardation_factor"]), float(scenario["decay_rate_s_1"]))
    response_on   = step_response(x_m, time_s - source_start, *common)
    response_off  = step_response(x_m, time_s - source_end, *common)
    concentration = source * (response_on - response_off)
    tolerance     = 1.0e-12 * max(1.0, source)
    if -tolerance <= concentration < 0.0:
        concentration = 0.0
    require(concentration >= 0.0 and math.isfinite(concentration), f"Independent analytical solution failed for {scenario['scenario_id']}, x={x_m}, t={time_s}")
    return concentration


def numeric_tolerance(expected):
    """
    Compute the absolute comparison tolerance for a serialized numeric value.

    Parameters
    ----------
    expected : float
        Full-precision reference value.

    Returns
    -------
    float
        Scale-aware absolute tolerance.

    """
    return NUMERIC_ABSOLUTE_LIMIT * max(1.0, abs(expected))


def require_numeric(actual_text, expected, label):
    """
    Validate a serialized finite number against an independent reference.

    Parameters
    ----------
    actual_text : str
        Numeric text read from a generated CSV.
    expected : float
        Full-precision independently reconstructed value.
    label : str
        Row and field context for a possible error.

    Returns
    -------
    float
        Parsed finite actual value.

    Raises
    ------
    ValidationError
        If parsing fails, the value is non-finite, or the tolerance is exceeded.

    """
    try:
        actual = float(actual_text)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"Non-numeric value at {label}: {actual_text}") from error
    require(math.isfinite(actual), f"Non-finite value at {label}: {actual_text}")
    difference = abs(actual - expected)
    require(difference <= numeric_tolerance(expected), f"Numeric mismatch at {label}: absolute error {difference:.6g}")
    return actual


def validate_field(path, scenarios):
    """
    Stream and validate every analytical field value and ordering contract.

    Parameters
    ----------
    path : pathlib.Path
        Generated analytical-field CSV.
    scenarios : list of dict
        Full-precision independently reconstructed scenarios.

    Returns
    -------
    tuple of dict
        Manifest-compatible summary and independent validation diagnostics.

    Raises
    ------
    ValidationError
        If dimensions, identifiers, coordinates, or concentrations differ.

    """
    expected_rows              = sum(int(row["time_nodes"]) * int(row["spatial_nodes"]) for row in scenarios)
    row_count                  = 0
    nonzero_rows               = 0
    minimum                    = math.inf
    maximum                    = -math.inf
    expected_minimum           = math.inf
    expected_maximum           = -math.inf
    maximum_error              = 0.0
    normalized_error           = 0.0
    initial_time_maximum       = 0.0
    expected_initial_time      = 0.0
    initial_interior_maximum   = 0.0
    expected_initial_interior  = 0.0
    boundary_max_error         = 0.0

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == FIELD_FIELDS, "Analytical-field header differs from the contract")
        for scenario in scenarios:
            source = float(scenario["source_concentration_mg_L"])
            for time_index in range(int(scenario["time_nodes"])):
                time_s = time_index * float(scenario["time_step_s"])
                for space_index in range(int(scenario["spatial_nodes"])):
                    row = next(reader, None)
                    require(row is not None, f"Analytical field ended before row {expected_rows}")
                    x_m = space_index * float(scenario["spatial_step_m"])
                    expected = pulse_concentration(x_m, time_s, scenario)
                    expected_normalized = expected / source
                    require(row["scenario_id"] == scenario["scenario_id"], f"Field scenario identifier mismatch at data row {row_count + 1}")
                    require(row["split"] == scenario["split"] and row["regime"] == scenario["regime"], f"Field stratum mismatch at data row {row_count + 1}")
                    require(row["design_component"] == scenario["design_component"], f"Field design component mismatch at data row {row_count + 1}")
                    require(row["time_s"] == format_float(time_s) and row["x_m"] == format_float(x_m), f"Field coordinate mismatch at data row {row_count + 1}")
                    actual = require_numeric(row["concentration_mg_L"], expected, f"field row {row_count + 2}, concentration_mg_L")
                    actual_normalized = require_numeric(row["normalized_concentration"], expected_normalized, f"field row {row_count + 2}, normalized_concentration")
                    row_count += 1
                    nonzero_rows += actual > 0.0
                    minimum = min(minimum, actual)
                    maximum = max(maximum, actual)
                    expected_minimum = min(expected_minimum, expected)
                    expected_maximum = max(expected_maximum, expected)
                    maximum_error = max(maximum_error, abs(actual - expected))
                    normalized_error = max(normalized_error, abs(actual_normalized - expected_normalized))
                    if time_s == 0.0:
                        initial_time_maximum = max(initial_time_maximum, abs(actual))
                        expected_initial_time = max(expected_initial_time, abs(expected))
                        if x_m > 0.0:
                            initial_interior_maximum = max(initial_interior_maximum, abs(actual))
                            expected_initial_interior = max(expected_initial_interior, abs(expected))
                    if x_m == 0.0:
                        boundary_expected = source if float(scenario["source_start_s"]) <= time_s < float(scenario["source_end_s"]) else 0.0
                        boundary_max_error = max(boundary_max_error, abs(actual - boundary_expected))
        require(next(reader, None) is None, "Analytical field contains extra rows")

    require(row_count == expected_rows, f"Expected {expected_rows} analytical rows, validated {row_count}")
    manifest_summary = {
        "rows": row_count,
        "columns": len(FIELD_FIELDS),
        "nonzero_rows": nonzero_rows,
        "minimum_concentration_mg_L": expected_minimum,
        "maximum_concentration_mg_L": expected_maximum,
        "initial_time_maximum_abs_including_inlet_mg_L": expected_initial_time,
        "initial_interior_maximum_abs_mg_L": expected_initial_interior,
        "inlet_boundary_maximum_abs_error_mg_L": 0.0,
    }
    validation = {
        "rows": row_count,
        "columns": len(FIELD_FIELDS),
        "minimum_serialized_concentration_mg_L": minimum,
        "maximum_serialized_concentration_mg_L": maximum,
        "maximum_absolute_concentration_error_mg_L": maximum_error,
        "maximum_absolute_normalized_error": normalized_error,
        "initial_time_maximum_abs_including_inlet_mg_L": initial_time_maximum,
        "initial_interior_maximum_abs_mg_L": initial_interior_maximum,
        "inlet_boundary_maximum_abs_error_mg_L": boundary_max_error,
        "analytical_values_reconstructed": row_count,
        "generator_module_imported": False,
    }
    return manifest_summary, validation


def replicate_random_stream(sensor_noise_seed, scenario_id, replicate_id):
    """
    Independently initialize the random stream for one sensor realization.

    Parameters
    ----------
    sensor_noise_seed : int
        Locked protocol-level noise seed.
    scenario_id : str
        Base-scenario identifier.
    replicate_id : str
        Replicate label such as `R01`.

    Returns
    -------
    random.Random
        Deterministic independent standard-library random stream.

    """
    material = f"{sensor_noise_seed}|{scenario_id}|{replicate_id}".encode("utf-8")
    return random.Random(int(hashlib.sha256(material).hexdigest(), 16))


def validate_sensors(path, scenarios, protocol):
    """
    Stream and reconstruct every noisy sensor record and censoring decision.

    Parameters
    ----------
    path : pathlib.Path
        Generated sensor-observation CSV.
    scenarios : list of dict
        Full-precision independently reconstructed scenarios.
    protocol : dict
        Locked protocol containing the sensor model and noise seed.

    Returns
    -------
    tuple of dict
        Manifest-compatible summary and independent validation diagnostics.

    Raises
    ------
    ValidationError
        If any row differs from the locked stochastic contract.

    """
    sensor_model          = protocol["challenge_design"]["sensor_model"]
    sensor_noise_seed     = int(protocol["challenge_design"]["sensor_noise_seed"])
    replicates            = int(sensor_model["noise_replicates_per_base_scenario"])
    relative_noise        = float(sensor_model["relative_gaussian_noise"])
    absolute_noise_floor  = float(sensor_model["absolute_noise_floor_mg_L"])
    detection_limit       = float(sensor_model["detection_limit_mg_L"])
    below_limit_value     = float(sensor_model["below_limit_substitution_mg_L"])
    sensor_positions      = [float(value) for value in sensor_model["positions_m"]]
    expected_rows         = len(scenarios) * replicates * len(sensor_positions) * int(scenarios[0]["time_nodes"])
    row_count             = 0
    below_detection_rows  = 0
    maximum_truth_error   = 0.0
    maximum_observed_error = 0.0
    standard_noise_sum    = 0.0
    standard_noise_square = 0.0

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == SENSOR_FIELDS, "Sensor-observation header differs from the contract")
        for scenario in scenarios:
            for replicate_number in range(1, replicates + 1):
                replicate_id = f"R{replicate_number:02d}"
                instance_id  = f"{scenario['scenario_id']}-{replicate_id}"
                rng          = replicate_random_stream(sensor_noise_seed, scenario["scenario_id"], replicate_id)
                for sensor_number, x_m in enumerate(sensor_positions, start=1):
                    sensor_id = f"S{sensor_number:02d}"
                    for time_index in range(int(scenario["time_nodes"])):
                        row = next(reader, None)
                        require(row is not None, f"Sensor table ended before row {expected_rows}")
                        time_s                 = time_index * float(scenario["time_step_s"])
                        truth                  = pulse_concentration(x_m, time_s, scenario)
                        noise_std              = max(absolute_noise_floor, relative_noise * truth)
                        standard_noise         = rng.gauss(0.0, 1.0)
                        uncensored_observation = max(0.0, truth + standard_noise * noise_std)
                        below_detection        = uncensored_observation < detection_limit
                        observed               = below_limit_value if below_detection else uncensored_observation
                        observation_id         = f"{instance_id}-{sensor_id}-T{time_index:03d}"

                        require(row["observation_id"] == observation_id, f"Observation identifier mismatch at data row {row_count + 1}")
                        require(row["base_scenario_id"] == scenario["scenario_id"] and row["scenario_id"] == instance_id, f"Sensor scenario identifier mismatch at data row {row_count + 1}")
                        require(row["replicate_id"] == replicate_id and row["sensor_id"] == sensor_id, f"Sensor replicate or station mismatch at data row {row_count + 1}")
                        require(row["split"] == scenario["split"] and row["regime"] == scenario["regime"], f"Sensor stratum mismatch at data row {row_count + 1}")
                        require(row["x_m"] == format_float(x_m) and row["time_s"] == format_float(time_s), f"Sensor coordinate mismatch at data row {row_count + 1}")
                        actual_truth = require_numeric(row["concentration_true_mg_L"], truth, f"sensor row {row_count + 2}, concentration_true_mg_L")
                        require_numeric(row["noise_std_mg_L"], noise_std, f"sensor row {row_count + 2}, noise_std_mg_L")
                        actual_observed = require_numeric(row["concentration_observed_mg_L"], observed, f"sensor row {row_count + 2}, concentration_observed_mg_L")
                        require_numeric(row["detection_limit_mg_L"], detection_limit, f"sensor row {row_count + 2}, detection_limit_mg_L")
                        require(row["is_below_detection_limit"] == str(below_detection).lower(), f"Censoring mismatch at sensor row {row_count + 2}")

                        row_count += 1
                        below_detection_rows += below_detection
                        maximum_truth_error = max(maximum_truth_error, abs(actual_truth - truth))
                        maximum_observed_error = max(maximum_observed_error, abs(actual_observed - observed))
                        standard_noise_sum += standard_noise
                        standard_noise_square += standard_noise**2
        require(next(reader, None) is None, "Sensor-observation table contains extra rows")

    require(row_count == expected_rows, f"Expected {expected_rows} sensor rows, validated {row_count}")
    noise_mean     = standard_noise_sum / row_count
    noise_variance = max(0.0, standard_noise_square / row_count - noise_mean**2)
    noise_std      = math.sqrt(noise_variance)
    manifest_summary = {
        "rows": row_count,
        "columns": len(SENSOR_FIELDS),
        "base_scenarios": len(scenarios),
        "scenario_replicates": len(scenarios) * replicates,
        "replicates_per_base_scenario": replicates,
        "sensor_positions": sensor_positions,
        "times_per_sensor": int(scenarios[0]["time_nodes"]),
        "below_detection_rows": below_detection_rows,
        "below_detection_fraction": below_detection_rows / row_count,
        "generated_standard_noise_mean": noise_mean,
        "generated_standard_noise_std": noise_std,
    }
    validation = dict(manifest_summary)
    validation.update(
        {
            "maximum_absolute_truth_error_mg_L": maximum_truth_error,
            "maximum_absolute_observed_error_mg_L": maximum_observed_error,
            "noise_streams_reconstructed": len(scenarios) * replicates,
            "censoring_mismatches": 0,
        }
    )
    return manifest_summary, validation


def compare_structure(actual, expected, label):
    """
    Compare nested manifest values while allowing roundoff in floating numbers.

    Parameters
    ----------
    actual : object
        Value read from the generated manifest.
    expected : object
        Independently reconstructed value.
    label : str
        Hierarchical context for a possible mismatch.

    Returns
    -------
    None
        The function returns after a successful recursive comparison.

    Raises
    ------
    ValidationError
        If keys, sequence lengths, scalar values, or numeric tolerances differ.

    """
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"Expected an object at {label}")
        require(set(actual) == set(expected), f"Manifest keys differ at {label}")
        for key in expected:
            compare_structure(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f"Manifest sequence differs at {label}")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            compare_structure(actual_item, expected_item, f"{label}[{index}]")
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and math.isclose(float(actual), expected, rel_tol=1.0e-12, abs_tol=1.0e-14), f"Manifest numeric value differs at {label}")
    else:
        require(actual == expected, f"Manifest value differs at {label}: {actual} != {expected}")


def file_metadata(path, logical_path, rows, columns):
    """
    Build expected manifest metadata for one generated CSV.

    Parameters
    ----------
    path : pathlib.Path
        Existing generated CSV.
    logical_path : str
        Protocol-defined project path.
    rows : int
        Number of data rows excluding the header.
    columns : int
        Number of CSV columns.

    Returns
    -------
    dict
        Expected path, dimensions, byte size, and SHA-256 digest.

    """
    return {"path": logical_path, "rows": rows, "columns": columns, "bytes": path.stat().st_size, "sha256": sha256(path)}


def validate_manifest(path, protocol_path, protocol, scenario_path, field_path, sensor_path, scenario_summary, field_summary, sensor_summary):
    """
    Verify challenge provenance, summaries, file digests, and zero-query state.

    Parameters
    ----------
    path : pathlib.Path
        Generated challenge manifest.
    protocol_path : pathlib.Path
        Locked protocol used for generation.
    protocol : dict
        Verified validation protocol.
    scenario_path : pathlib.Path
        Generated scenario CSV.
    field_path : pathlib.Path
        Generated analytical-field CSV.
    sensor_path : pathlib.Path
        Generated sensor-observation CSV.
    scenario_summary : dict
        Independently reconstructed scenario summary.
    field_summary : dict
        Independently reconstructed analytical-field summary.
    sensor_summary : dict
        Independently reconstructed sensor summary.

    Returns
    -------
    tuple of dict
        Parsed manifest and concise integrity summary.

    Raises
    ------
    ValidationError
        If provenance, summaries, digests, or model-access flags differ.

    """
    manifest          = load_json(path)
    planned           = protocol["planned_artifacts"]
    generation_record = protocol["frozen_inputs"]["benchmark"]["generation_module"]
    expected_files = {
        "challenge_scenarios": file_metadata(scenario_path, planned["challenge_scenarios"], scenario_summary["rows"], scenario_summary["columns"]),
        "challenge_analytical_field": file_metadata(field_path, planned["challenge_analytical_field"], field_summary["rows"], field_summary["columns"]),
        "challenge_sensor_observations": file_metadata(sensor_path, planned["challenge_sensor_observations"], sensor_summary["rows"], sensor_summary["columns"]),
    }

    require(manifest["manifest_version"] == "1.0.0", "Unsupported challenge-manifest version")
    require(manifest["status"] == "generated_before_model_inference", "Challenge manifest is not in its pre-inference state")
    require(manifest["protocol_id"] == protocol["protocol_id"] and manifest["protocol_version"] == protocol["protocol_version"], "Challenge manifest references a different protocol")
    require(manifest["protocol_sha256"] == sha256(protocol_path), "Challenge manifest contains an incorrect protocol digest")
    require(manifest["protocol_lock_sha256"] == sha256(PROTOCOL_LOCK), "Challenge manifest contains an incorrect protocol-lock digest")
    require(manifest["generator"] == {"path": "src/generate_challenge_set.py", "sha256": sha256(GENERATOR)}, "Challenge manifest contains an incorrect generator record")
    require(manifest["analytical_generation_module"] == {"path": generation_record["path"], "sha256": generation_record["sha256"]}, "Challenge manifest contains an incorrect analytical-module record")
    require(manifest["seeds"] == {"parameter_seed": int(protocol["challenge_design"]["parameter_seed"]), "sensor_noise_seed": int(protocol["challenge_design"]["sensor_noise_seed"])}, "Challenge manifest seed record differs from the protocol")
    compare_structure(manifest["scenario_summary"], scenario_summary, "scenario_summary")
    compare_structure(manifest["field_summary"], field_summary, "field_summary")
    compare_structure(manifest["sensor_summary"], sensor_summary, "sensor_summary")
    compare_structure(manifest["files"], expected_files, "files")
    require(manifest["model_access"] == {"adr1d_ml_loaded": False, "adr1d_nn_loaded": False, "challenge_inference_runs": 0}, "Challenge manifest does not record a zero-query model state")
    require(manifest["software"] == {"python": platform.python_version(), "implementation": platform.python_implementation()}, "Challenge manifest software record differs from the validation environment")
    return manifest, {"path": planned["challenge_manifest"], "bytes": path.stat().st_size, "sha256": sha256(path), "file_hashes_verified": len(expected_files), "model_access_verified_zero": True}


def write_validation_report(path, protocol_path, protocol, frozen_summary, scenario_validation, field_validation, sensor_validation, manifest_validation):
    """
    Persist the deterministic independent challenge-validation report.

    Parameters
    ----------
    path : pathlib.Path
        Destination JSON report.
    protocol_path : pathlib.Path
        Locked protocol used for validation.
    protocol : dict
        Verified validation protocol.
    frozen_summary : dict
        Frozen-input integrity result.
    scenario_validation : dict
        Scenario-design validation result.
    field_validation : dict
        Independent analytical-field validation result.
    sensor_validation : dict
        Independent stochastic reconstruction result.
    manifest_validation : dict
        Challenge-manifest integrity result.

    Returns
    -------
    dict
        Persisted report content.

    """
    report = {
        "report_version": "1.0.0",
        "status": "passed_before_model_inference",
        "validated_date": datetime.now(timezone.utc).date().isoformat(),
        "protocol": {"protocol_id": protocol["protocol_id"], "protocol_version": protocol["protocol_version"], "sha256": sha256(protocol_path)},
        "integrity": frozen_summary,
        "scenario_validation": scenario_validation,
        "analytical_field_validation": field_validation,
        "sensor_validation": sensor_validation,
        "manifest_validation": manifest_validation,
        "independence": {
            "challenge_generator_imported": False,
            "adr1d_generation_module_imported": False,
            "serialized_models_loaded": False,
            "analytical_values_reconstructed": field_validation["analytical_values_reconstructed"] + sensor_validation["rows"],
            "noise_streams_reconstructed": sensor_validation["noise_streams_reconstructed"],
        },
        "model_access": {"adr1d_ml_loaded": False, "adr1d_nn_loaded": False, "adr1d_ml_challenge_inference_runs": 0, "adr1d_nn_challenge_inference_runs": 0},
        "validator": {"path": "src/validate_challenge_set.py", "sha256": sha256(Path(__file__))},
        "software": {"python": platform.python_version(), "implementation": platform.python_implementation()},
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def write_challenge_lock(path, protocol_path, protocol, manifest_path, validation_path, scenario_path, field_path, sensor_path, scenario_summary, field_summary, sensor_summary):
    """
    Lock validated challenge artifacts before any model inference is permitted.

    Parameters
    ----------
    path : pathlib.Path
        Destination challenge-lock JSON.
    protocol_path : pathlib.Path
        Locked validation protocol.
    protocol : dict
        Verified protocol content.
    manifest_path : pathlib.Path
        Verified challenge manifest.
    validation_path : pathlib.Path
        Independent validation report written immediately before this lock.
    scenario_path : pathlib.Path
        Validated scenario table.
    field_path : pathlib.Path
        Validated analytical-field table.
    sensor_path : pathlib.Path
        Validated sensor-observation table.
    scenario_summary : dict
        Validated scenario dimensions.
    field_summary : dict
        Validated analytical-field dimensions.
    sensor_summary : dict
        Validated sensor dimensions.

    Returns
    -------
    dict
        Persisted immutable challenge-lock content.

    """
    planned = protocol["planned_artifacts"]
    lock = {
        "lock_version": "1.0.0",
        "status": "locked_before_model_inference",
        "locked_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "protocol": {"path": "configs/validation_protocol.json", "sha256": sha256(protocol_path), "protocol_id": protocol["protocol_id"], "protocol_version": protocol["protocol_version"]},
        "challenge_manifest": {"path": planned["challenge_manifest"], "sha256": sha256(manifest_path), "bytes": manifest_path.stat().st_size},
        "independent_validation": {"path": f"results/{VALIDATION_FILE_NAME}", "sha256": sha256(validation_path), "bytes": validation_path.stat().st_size, "status": "passed"},
        "implementations": {
            "generator": {"path": "src/generate_challenge_set.py", "sha256": sha256(GENERATOR)},
            "independent_validator": {"path": "src/validate_challenge_set.py", "sha256": sha256(Path(__file__))},
        },
        "challenge_files": {
            "challenge_scenarios": file_metadata(scenario_path, planned["challenge_scenarios"], scenario_summary["rows"], scenario_summary["columns"]),
            "challenge_analytical_field": file_metadata(field_path, planned["challenge_analytical_field"], field_summary["rows"], field_summary["columns"]),
            "challenge_sensor_observations": file_metadata(sensor_path, planned["challenge_sensor_observations"], sensor_summary["rows"], sensor_summary["columns"]),
        },
        "state_at_lock": {
            "challenge_base_scenarios": scenario_summary["rows"],
            "challenge_sensor_realizations": sensor_summary["scenario_replicates"],
            "adr1d_ml_challenge_inference_runs": 0,
            "adr1d_nn_challenge_inference_runs": 0,
            "traditional_numerical_solver_runs": 0,
            "acceptance_criteria_inspected_against_challenge_results": False,
        },
        "frozen_models": {
            "adr1d_ml_sha256": protocol["frozen_inputs"]["adr1d_ml"]["model"]["sha256"],
            "adr1d_nn_sha256": protocol["frozen_inputs"]["adr1d_nn"]["model"]["sha256"],
        },
        "change_policy": {
            "silent_edits_allowed": False,
            "regeneration_requires_new_protocol_version": True,
            "model_changes_or_threshold_tuning_allowed": False,
        },
    }
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return lock


def main():
    """
    Validate and lock the complete challenge set before model inference.

    Returns
    -------
    None
        The function writes a validation report and immutable challenge lock.

    """
    args            = parse_args()
    protocol_path   = args.protocol.resolve()
    data_dir        = args.data_dir.resolve()
    results_dir     = args.results_dir.resolve()
    scenario_path   = data_dir / SCENARIO_FILE_NAME
    field_path      = data_dir / FIELD_FILE_NAME
    sensor_path     = data_dir / SENSOR_FILE_NAME
    manifest_path   = results_dir / MANIFEST_FILE_NAME
    validation_path = results_dir / VALIDATION_FILE_NAME
    lock_path       = results_dir / CHALLENGE_LOCK_NAME
    require_absent((validation_path, lock_path))

    for required_path in (scenario_path, field_path, sensor_path, manifest_path, GENERATOR):
        if not required_path.exists():
            raise FileNotFoundError(required_path)
    protocol, protocol_lock = load_locked_protocol(protocol_path)
    frozen_summary = verify_frozen_inputs(protocol, protocol_lock)
    scenarios, scenario_summary, scenario_validation = validate_scenarios(scenario_path, protocol)
    field_summary, field_validation = validate_field(field_path, scenarios)
    sensor_summary, sensor_validation = validate_sensors(sensor_path, scenarios, protocol)
    _, manifest_validation = validate_manifest(manifest_path, protocol_path, protocol, scenario_path, field_path, sensor_path, scenario_summary, field_summary, sensor_summary)

    results_dir.mkdir(parents=True, exist_ok=True)
    report = write_validation_report(validation_path, protocol_path, protocol, frozen_summary, scenario_validation, field_validation, sensor_validation, manifest_validation)
    lock   = write_challenge_lock(lock_path, protocol_path, protocol, manifest_path, validation_path, scenario_path, field_path, sensor_path, scenario_summary, field_summary, sensor_summary)
    print(
        json.dumps(
            {
                "status": lock["status"],
                "validation_status": report["status"],
                "base_scenarios": scenario_summary["rows"],
                "field_rows": field_summary["rows"],
                "sensor_rows": sensor_summary["rows"],
                "challenge_inference_runs": 0,
                "challenge_lock": str(lock_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
