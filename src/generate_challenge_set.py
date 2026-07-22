"""
================================================================================
ADR1D Validation: Locked Challenge-Set Generation
================================================================================

Generate the pre-specified ADR1D validation scenarios, analytical fields, and
independent noisy sensor realizations without loading either evaluated model.

Main Operations
---------------
1. Verify the locked validation protocol and analytical generation module.
2. Generate balanced interior and boundary-profile challenge scenarios.
3. Evaluate analytical fields and independent noisy sensor realizations.
4. Write a manifest containing dimensions, provenance, and SHA-256 digests.

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
import csv
import hashlib
import importlib.util
import json
import math
import platform
import random
from collections import Counter
from pathlib import Path


ROOT                  = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL      = ROOT / "configs/validation_protocol.json"
PROTOCOL_LOCK         = ROOT / "results/validation_protocol_lock.json"
DEFAULT_DATA_DIR      = ROOT / "data"
DEFAULT_RESULTS_DIR   = ROOT / "results"
SCENARIO_FILE_NAME    = "challenge_scenarios.csv"
FIELD_FILE_NAME       = "challenge_analytical_field.csv"
SENSOR_FILE_NAME      = "challenge_sensor_observations.csv"
MANIFEST_FILE_NAME    = "challenge_manifest.json"
CHALLENGE_LOCK_NAME   = "challenge_lock.json"

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
    Format a numeric value with the stable precision used by ADR1D tables.

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
    Parse challenge-generation paths.

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


def display_path(path):
    """
    Return a stable project-relative path when possible.

    Parameters
    ----------
    path : pathlib.Path
        Artifact path to represent in a manifest.

    Returns
    -------
    str
        Project-relative path or absolute fallback.

    """
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def load_locked_protocol(protocol_path):
    """
    Load the validation protocol after checking its immutable lock.

    Parameters
    ----------
    protocol_path : pathlib.Path
        Machine-readable validation protocol.

    Returns
    -------
    dict
        Verified protocol content.

    Raises
    ------
    FileNotFoundError
        If the protocol or its lock is absent.
    RuntimeError
        If the protocol digest, version, or pre-inference state differs from the lock.

    """
    if not protocol_path.exists():
        raise FileNotFoundError(protocol_path)
    if not PROTOCOL_LOCK.exists():
        raise FileNotFoundError(PROTOCOL_LOCK)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    lock     = json.loads(PROTOCOL_LOCK.read_text(encoding="utf-8"))
    if sha256(protocol_path) != lock["protocol"]["sha256"]:
        raise RuntimeError("Validation protocol hash differs from its lock")
    if protocol["protocol_id"] != lock["protocol"]["protocol_id"] or protocol["protocol_version"] != lock["protocol"]["protocol_version"]:
        raise RuntimeError("Validation protocol identity differs from its lock")
    if protocol["status"] != "locked_before_challenge_generation_and_model_inference":
        raise RuntimeError("Validation protocol is not in its pre-generation state")
    if any(int(lock["state_at_lock"][key]) != 0 for key in ("challenge_scenarios_generated", "adr1d_ml_challenge_inference_runs", "adr1d_nn_challenge_inference_runs", "traditional_numerical_solver_runs")):
        raise RuntimeError("Protocol lock does not record a zero-query challenge state")
    return protocol


def load_generation_module(protocol):
    """
    Load the trusted ADR1D analytical generator after digest verification.

    Parameters
    ----------
    protocol : dict
        Locked validation protocol containing the upstream module path and digest.

    Returns
    -------
    module
        Imported ADR1D generation module.

    Raises
    ------
    FileNotFoundError
        If the upstream module is absent.
    RuntimeError
        If its digest or import contract is invalid.

    """
    metadata = protocol["frozen_inputs"]["benchmark"]["generation_module"]
    path     = project_path(metadata["path"])
    if not path.exists():
        raise FileNotFoundError(path)
    if sha256(path) != metadata["sha256"]:
        raise RuntimeError("ADR1D generation module hash differs from the protocol")
    specification = importlib.util.spec_from_file_location("locked_adr1d_generator", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("Unable to create the ADR1D generation-module specification")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    for name in ("latin_hypercube", "linear_scale", "log_scale", "pulse_concentration"):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"ADR1D generation module lacks callable {name}")
    return module


def build_scenario(scenario_id, regime, design_component, design_index, unit_coordinates, protocol, generation_module):
    """
    Transform one seven-dimensional unit sample into a physical challenge scenario.

    Parameters
    ----------
    scenario_id : str
        Stable base-scenario identifier.
    regime : str
        One of the four pre-specified physical regimes.
    design_component : str
        Either `latin_hypercube` or `boundary_profile`.
    design_index : int
        One-based index within the regime and design component.
    unit_coordinates : sequence of float
        Seven unit-interval coordinates in protocol order.
    protocol : dict
        Locked validation protocol.
    generation_module : module
        Verified ADR1D analytical generation module.

    Returns
    -------
    dict
        Complete scenario metadata and derived dimensionless quantities.

    """
    ranges               = protocol["challenge_design"]["parameter_ranges"]
    domain               = protocol["challenge_design"]["domain"]
    velocity             = generation_module.linear_scale(unit_coordinates[0], ranges["velocity_m_s"])
    dispersion           = generation_module.log_scale(unit_coordinates[1], ranges["dispersion_m2_s"])
    source_concentration = generation_module.log_scale(unit_coordinates[2], ranges["source_concentration_mg_L"])
    source_start         = generation_module.linear_scale(unit_coordinates[3], ranges["source_start_s"])
    source_duration      = generation_module.linear_scale(unit_coordinates[4], ranges["source_duration_s"])
    retardation_active   = regime in {"retardation_only", "retardation_and_decay"}
    decay_active         = regime in {"decay_only", "retardation_and_decay"}
    retardation          = generation_module.linear_scale(unit_coordinates[5], ranges["retardation_factor_when_active"]) if retardation_active else float(protocol["challenge_design"]["inactive_parameter_values"]["retardation_factor"])
    decay                = generation_module.log_scale(unit_coordinates[6], ranges["decay_rate_s_1_when_active"]) if decay_active else float(protocol["challenge_design"]["inactive_parameter_values"]["decay_rate_s_1"])
    length               = float(domain["length_m"])
    source_end           = source_start + source_duration
    return {
        "scenario_id": scenario_id,
        "split": protocol["challenge_design"]["split_label"],
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
    }


def build_scenarios(protocol, generation_module):
    """
    Generate all balanced interior and boundary-profile challenge scenarios.

    Parameters
    ----------
    protocol : dict
        Locked validation protocol.
    generation_module : module
        Verified ADR1D analytical generation module.

    Returns
    -------
    list of dict
        Ordered challenge scenarios.

    Raises
    ------
    ValueError
        If the protocol counts do not produce the prescribed total.

    """
    design          = protocol["challenge_design"]
    rng             = random.Random(int(design["parameter_seed"]))
    scenarios       = []
    scenario_number = 1
    interior_count  = int(design["interior_latin_hypercube_per_regime"])
    boundary_rows   = design["boundary_profile_unit_coordinates"]

    for regime in design["regimes"]:
        interior_rows = generation_module.latin_hypercube(interior_count, len(design["parameter_order"]), rng)
        for index, coordinates in enumerate(interior_rows, start=1):
            scenario_id = f"A5-CH-{scenario_number:04d}"
            scenarios.append(build_scenario(scenario_id, regime, "latin_hypercube", index, coordinates, protocol, generation_module))
            scenario_number += 1
        for index, coordinates in enumerate(boundary_rows, start=1):
            scenario_id = f"A5-CH-{scenario_number:04d}"
            scenarios.append(build_scenario(scenario_id, regime, "boundary_profile", index, coordinates, protocol, generation_module))
            scenario_number += 1

    if len(scenarios) != int(design["base_scenarios"]):
        raise ValueError(f"Expected {design['base_scenarios']} scenarios, generated {len(scenarios)}")
    return scenarios


def write_scenarios(scenarios, path):
    """
    Write challenge scenario metadata in deterministic order.

    Parameters
    ----------
    scenarios : list of dict
        Ordered challenge scenarios.
    path : pathlib.Path
        Destination CSV path.

    Returns
    -------
    dict
        Scenario dimensions and stratification counts.

    """
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SCENARIO_FIELDS)
        writer.writeheader()
        for scenario in scenarios:
            writer.writerow({field: format_float(value) if isinstance(value, float) else value for field, value in scenario.items()})
    return {
        "rows": len(scenarios),
        "columns": len(SCENARIO_FIELDS),
        "regime_counts": dict(sorted(Counter(row["regime"] for row in scenarios).items())),
        "design_component_counts": dict(sorted(Counter(row["design_component"] for row in scenarios).items())),
    }


def write_analytical_field(scenarios, path, generation_module):
    """
    Evaluate and write the complete normalized analytical challenge fields.

    Parameters
    ----------
    scenarios : list of dict
        Ordered challenge scenarios.
    path : pathlib.Path
        Destination CSV path.
    generation_module : module
        Verified ADR1D analytical generation module.

    Returns
    -------
    dict
        Field dimensions, value range, and boundary checks.

    """
    row_count          = 0
    nonzero_rows       = 0
    minimum            = math.inf
    maximum            = -math.inf
    boundary_max_error       = 0.0
    initial_time_maximum     = 0.0
    initial_interior_maximum = 0.0
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELD_FIELDS)
        writer.writeheader()
        for scenario in scenarios:
            source = float(scenario["source_concentration_mg_L"])
            for time_index in range(int(scenario["time_nodes"])):
                time_s = time_index * float(scenario["time_step_s"])
                for space_index in range(int(scenario["spatial_nodes"])):
                    x_m           = space_index * float(scenario["spatial_step_m"])
                    concentration = generation_module.pulse_concentration(x_m, time_s, scenario)
                    normalized    = concentration / source
                    writer.writerow(
                        {
                            "scenario_id": scenario["scenario_id"],
                            "split": scenario["split"],
                            "regime": scenario["regime"],
                            "design_component": scenario["design_component"],
                            "time_s": format_float(time_s),
                            "x_m": format_float(x_m),
                            "concentration_mg_L": format_float(concentration),
                            "normalized_concentration": format_float(normalized),
                        }
                    )
                    row_count += 1
                    nonzero_rows += concentration > 0.0
                    minimum = min(minimum, concentration)
                    maximum = max(maximum, concentration)
                    if time_s == 0.0:
                        initial_time_maximum = max(initial_time_maximum, abs(concentration))
                        if x_m > 0.0:
                            initial_interior_maximum = max(initial_interior_maximum, abs(concentration))
                    if x_m == 0.0:
                        expected           = source if float(scenario["source_start_s"]) <= time_s < float(scenario["source_end_s"]) else 0.0
                        boundary_max_error = max(boundary_max_error, abs(concentration - expected))
    return {
        "rows": row_count,
        "columns": len(FIELD_FIELDS),
        "nonzero_rows": nonzero_rows,
        "minimum_concentration_mg_L": minimum,
        "maximum_concentration_mg_L": maximum,
        "initial_time_maximum_abs_including_inlet_mg_L": initial_time_maximum,
        "initial_interior_maximum_abs_mg_L": initial_interior_maximum,
        "inlet_boundary_maximum_abs_error_mg_L": boundary_max_error,
    }


def replicate_random_stream(sensor_noise_seed, scenario_id, replicate_id):
    """
    Construct an order-independent random stream for one sensor realization.

    Parameters
    ----------
    sensor_noise_seed : int
        Protocol-level noise seed.
    scenario_id : str
        Stable base-scenario identifier.
    replicate_id : str
        Stable replicate identifier such as `R01`.

    Returns
    -------
    random.Random
        Independent deterministic standard-library random stream.

    """
    material = f"{sensor_noise_seed}|{scenario_id}|{replicate_id}".encode("utf-8")
    return random.Random(int(hashlib.sha256(material).hexdigest(), 16))


def write_sensor_observations(scenarios, path, protocol, generation_module):
    """
    Generate and write independent noisy sensor histories for every base scenario.

    Parameters
    ----------
    scenarios : list of dict
        Ordered challenge scenarios.
    path : pathlib.Path
        Destination CSV path.
    protocol : dict
        Locked validation protocol containing the sensor model.
    generation_module : module
        Verified ADR1D analytical generation module.

    Returns
    -------
    dict
        Sensor dimensions, censoring counts, and generated-noise diagnostics.

    """
    sensor_model          = protocol["challenge_design"]["sensor_model"]
    sensor_noise_seed     = int(protocol["challenge_design"]["sensor_noise_seed"])
    replicates            = int(sensor_model["noise_replicates_per_base_scenario"])
    relative_noise        = float(sensor_model["relative_gaussian_noise"])
    absolute_noise_floor  = float(sensor_model["absolute_noise_floor_mg_L"])
    detection_limit       = float(sensor_model["detection_limit_mg_L"])
    below_limit_value     = float(sensor_model["below_limit_substitution_mg_L"])
    sensor_positions      = [float(value) for value in sensor_model["positions_m"]]
    row_count             = 0
    below_detection_rows  = 0
    standard_noise_sum    = 0.0
    standard_noise_square = 0.0

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SENSOR_FIELDS)
        writer.writeheader()
        for scenario in scenarios:
            for replicate_number in range(1, replicates + 1):
                replicate_id = f"R{replicate_number:02d}"
                instance_id  = f"{scenario['scenario_id']}-{replicate_id}"
                rng          = replicate_random_stream(sensor_noise_seed, scenario["scenario_id"], replicate_id)
                for sensor_number, x_m in enumerate(sensor_positions, start=1):
                    sensor_id = f"S{sensor_number:02d}"
                    for time_index in range(int(scenario["time_nodes"])):
                        time_s                 = time_index * float(scenario["time_step_s"])
                        truth                  = generation_module.pulse_concentration(x_m, time_s, scenario)
                        noise_std              = max(absolute_noise_floor, relative_noise * truth)
                        standard_noise         = rng.gauss(0.0, 1.0)
                        uncensored_observation = max(0.0, truth + standard_noise * noise_std)
                        below_detection        = uncensored_observation < detection_limit
                        observed               = below_limit_value if below_detection else uncensored_observation
                        observation_id         = f"{instance_id}-{sensor_id}-T{time_index:03d}"
                        writer.writerow(
                            {
                                "observation_id": observation_id,
                                "base_scenario_id": scenario["scenario_id"],
                                "scenario_id": instance_id,
                                "replicate_id": replicate_id,
                                "split": scenario["split"],
                                "regime": scenario["regime"],
                                "sensor_id": sensor_id,
                                "x_m": format_float(x_m),
                                "time_s": format_float(time_s),
                                "concentration_true_mg_L": format_float(truth),
                                "noise_std_mg_L": format_float(noise_std),
                                "concentration_observed_mg_L": format_float(observed),
                                "detection_limit_mg_L": format_float(detection_limit),
                                "is_below_detection_limit": str(below_detection).lower(),
                            }
                        )
                        row_count += 1
                        below_detection_rows += below_detection
                        standard_noise_sum += standard_noise
                        standard_noise_square += standard_noise**2

    noise_mean     = standard_noise_sum / row_count
    noise_variance = max(0.0, standard_noise_square / row_count - noise_mean**2)
    return {
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
        "generated_standard_noise_std": math.sqrt(noise_variance),
    }


def file_metadata(path, logical_path, rows, columns):
    """
    Describe one generated CSV for the challenge manifest.

    Parameters
    ----------
    path : pathlib.Path
        Generated CSV path.
    logical_path : str
        Protocol-defined project path, independent of the output location.
    rows : int
        Number of data rows excluding the header.
    columns : int
        Number of CSV columns.

    Returns
    -------
    dict
        Path, dimensions, byte size, and SHA-256 digest.

    """
    return {"path": logical_path, "rows": int(rows), "columns": int(columns), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_manifest(path, protocol_path, protocol, scenario_path, field_path, sensor_path, scenario_summary, field_summary, sensor_summary):
    """
    Write challenge provenance, dimensions, and artifact digests.

    Parameters
    ----------
    path : pathlib.Path
        Destination manifest path.
    protocol_path : pathlib.Path
        Locked protocol used for generation.
    protocol : dict
        Verified protocol content.
    scenario_path : pathlib.Path
        Generated scenario table.
    field_path : pathlib.Path
        Generated analytical field table.
    sensor_path : pathlib.Path
        Generated sensor-observation table.
    scenario_summary : dict
        Scenario dimensions and strata.
    field_summary : dict
        Analytical-field dimensions and checks.
    sensor_summary : dict
        Sensor dimensions and noise checks.

    Returns
    -------
    dict
        Persisted manifest content.

    """
    generation_metadata = protocol["frozen_inputs"]["benchmark"]["generation_module"]
    planned_artifacts   = protocol["planned_artifacts"]
    manifest = {
        "manifest_version": "1.0.0",
        "status": "generated_before_model_inference",
        "protocol_id": protocol["protocol_id"],
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": sha256(protocol_path),
        "protocol_lock_sha256": sha256(PROTOCOL_LOCK),
        "generator": {"path": display_path(Path(__file__)), "sha256": sha256(Path(__file__))},
        "analytical_generation_module": {"path": generation_metadata["path"], "sha256": generation_metadata["sha256"]},
        "seeds": {
            "parameter_seed": int(protocol["challenge_design"]["parameter_seed"]),
            "sensor_noise_seed": int(protocol["challenge_design"]["sensor_noise_seed"]),
        },
        "scenario_summary": scenario_summary,
        "field_summary": field_summary,
        "sensor_summary": sensor_summary,
        "files": {
            "challenge_scenarios": file_metadata(scenario_path, planned_artifacts["challenge_scenarios"], scenario_summary["rows"], scenario_summary["columns"]),
            "challenge_analytical_field": file_metadata(field_path, planned_artifacts["challenge_analytical_field"], field_summary["rows"], field_summary["columns"]),
            "challenge_sensor_observations": file_metadata(sensor_path, planned_artifacts["challenge_sensor_observations"], sensor_summary["rows"], sensor_summary["columns"]),
        },
        "model_access": {
            "adr1d_ml_loaded": False,
            "adr1d_nn_loaded": False,
            "challenge_inference_runs": 0,
        },
        "software": {"python": platform.python_version(), "implementation": platform.python_implementation()},
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def require_absent(paths):
    """
    Refuse to overwrite challenge artifacts or a completed challenge lock.

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
        raise FileExistsError("Refusing to overwrite challenge artifacts: " + ", ".join(str(path) for path in existing))


def main():
    """
    Generate the complete challenge set and its pre-validation manifest.

    Returns
    -------
    None
        The function writes CSV and JSON artifacts and prints a summary.

    """
    args           = parse_args()
    protocol_path  = args.protocol.resolve()
    data_dir       = args.data_dir.resolve()
    results_dir    = args.results_dir.resolve()
    scenario_path  = data_dir / SCENARIO_FILE_NAME
    field_path     = data_dir / FIELD_FILE_NAME
    sensor_path    = data_dir / SENSOR_FILE_NAME
    manifest_path  = results_dir / MANIFEST_FILE_NAME
    challenge_lock = results_dir / CHALLENGE_LOCK_NAME
    require_absent((scenario_path, field_path, sensor_path, manifest_path, challenge_lock))

    protocol          = load_locked_protocol(protocol_path)
    generation_module = load_generation_module(protocol)
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    scenarios       = build_scenarios(protocol, generation_module)
    scenario_summary = write_scenarios(scenarios, scenario_path)
    field_summary    = write_analytical_field(scenarios, field_path, generation_module)
    sensor_summary   = write_sensor_observations(scenarios, sensor_path, protocol, generation_module)
    manifest         = write_manifest(manifest_path, protocol_path, protocol, scenario_path, field_path, sensor_path, scenario_summary, field_summary, sensor_summary)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "base_scenarios": scenario_summary["rows"],
                "field_rows": field_summary["rows"],
                "sensor_rows": sensor_summary["rows"],
                "challenge_inference_runs": 0,
                "manifest": str(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
