"""
================================================================================
ADR1D Validation: Computational-Cost Benchmark
================================================================================

Measure the machine-specific computational cost of the frozen ADR1D workflows
without repeating model inference on the locked Activity 05 challenge set.

Main Operations
---------------
1. Verify protected models, canonical inputs, and challenge-result artifacts.
2. Build canonical ADR1D-ML and ADR1D-NN test workloads in memory.
3. Time two warm-up runs and seven retained repetitions per component.
4. Check deterministic outputs and reproduce stored canonical predictions.
5. Persist timings, allocation diagnostics, environment, and scope limitations.

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
import functools
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import time
import tracemalloc
import warnings
from pathlib import Path

# Third-party libraries
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from threadpoolctl import threadpool_info, threadpool_limits


ROOT                  = Path(__file__).resolve().parents[1]
PROTOCOL_PATH         = ROOT / "configs/validation_protocol.json"
CHALLENGE_LOCK_PATH   = ROOT / "results/challenge_lock.json"
CHALLENGE_SCENARIOS   = ROOT / "data/challenge_scenarios.csv"
CHALLENGE_FIELD       = ROOT / "data/challenge_analytical_field.csv"
CHALLENGE_SENSORS     = ROOT / "data/challenge_sensor_observations.csv"
CHALLENGE_OUTPUTS     = (
    ROOT / "results/adr1d_ml_challenge_predictions.csv",
    ROOT / "results/adr1d_ml_challenge_metrics.json",
    ROOT / "results/adr1d_nn_challenge_predictions.csv",
    ROOT / "results/adr1d_nn_challenge_scenarios.csv",
    ROOT / "results/adr1d_nn_challenge_metrics.json",
)
NUMERICAL_CASES_PATH  = ROOT / "results/numerical_reference_cases.csv"
OUTPUT_PATH           = ROOT / "results/computational_cost.json"

DATASET_ROOT          = ROOT.parent / "02_datasets"
CANONICAL_SCENARIOS   = DATASET_ROOT / "metadata/synthetic_adr1d_scenarios.csv"
CANONICAL_FIELD       = DATASET_ROOT / "data_processed/synthetic_adr1d_field.csv"
CANONICAL_SENSORS     = DATASET_ROOT / "data_processed/synthetic_adr1d_sensor_observations.csv"

ML_ROOT               = ROOT.parent / "03_modelos_ml_parametros"
ML_MODEL_PATH         = ML_ROOT / "models/adr1d_parameter_models.joblib"
ML_MANIFEST_PATH      = ML_ROOT / "models/model_manifest.json"
ML_PROTOCOL_PATH      = ML_ROOT / "results/final_model_protocol.json"
ML_FEATURE_TABLE      = ML_ROOT / "results/adr1d_modeling_table_physics.csv"
ML_TEST_PREDICTIONS   = ML_ROOT / "results/final_test_predictions.csv"
ML_BASE_MODULE_PATH   = ML_ROOT / "scripts/build_adr1d_modeling_table.py"
ML_PHYSICS_MODULE_PATH = ML_ROOT / "scripts/build_adr1d_physics_features.py"

NN_ROOT               = ROOT.parent / "04_redes_neuronales_transporte"
NN_MODEL_PATH         = NN_ROOT / "models/adr1d_nn.pt"
NN_MANIFEST_PATH      = NN_ROOT / "models/model_manifest.json"
NN_PROTOCOL_PATH      = NN_ROOT / "configs/final_model_protocol.json"
NN_METRICS_PATH       = NN_ROOT / "results/final_test_metrics.json"
NN_INFERENCE_PATH     = NN_ROOT / "src/predict_concentration.py"

NUMERICAL_MODULE_PATH = ROOT / "src/run_numerical_reference.py"


class BenchmarkError(RuntimeError):
    """Represent a computational-cost contract or reproducibility failure."""


def sha256(path):
    """
    Compute the SHA-256 digest of a local file in bounded memory.

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
    BenchmarkError
        If the top-level JSON value is not an object.

    """
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise BenchmarkError(f"Expected a JSON object in {path}")
    return content


def require(condition, message):
    """
    Enforce one benchmark condition.

    Parameters
    ----------
    condition : bool
        Condition that must evaluate to true.
    message : str
        Actionable failure description.

    Returns
    -------
    None
        The function returns only after a successful check.

    Raises
    ------
    BenchmarkError
        If the condition is false.

    """
    if not bool(condition):
        raise BenchmarkError(message)


def parse_arguments():
    """
    Parse the computational-cost output and overwrite policy.

    Returns
    -------
    argparse.Namespace
        Parsed command-line options.

    """
    parser = argparse.ArgumentParser(description="Benchmark the frozen ADR1D computational workflows on canonical data.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH, help="Destination JSON report.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing cost report intentionally.")
    return parser.parse_args()


def load_module(name, path):
    """
    Load one trusted local Python module from an explicit path.

    Parameters
    ----------
    name : str
        Unique in-process module name.
    path : pathlib.Path
        Existing Python source path.

    Returns
    -------
    module
        Executed local module object.

    Raises
    ------
    ImportError
        If Python cannot construct a module specification.

    """
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def protected_hashes():
    """
    Record hashes of every artifact that the benchmark must not alter.

    Returns
    -------
    dict
        Relative or absolute paths mapped to SHA-256 digests.

    """
    paths = (PROTOCOL_PATH, CHALLENGE_LOCK_PATH, CHALLENGE_SCENARIOS, CHALLENGE_FIELD, CHALLENGE_SENSORS, ML_MODEL_PATH, ML_MANIFEST_PATH, NN_MODEL_PATH, NN_MANIFEST_PATH, *CHALLENGE_OUTPUTS)
    return {str(path): sha256(path) for path in paths}


def verify_protocol_and_threads(protocol):
    """
    Verify the locked repetition count, device, and single-thread environment.

    Parameters
    ----------
    protocol : dict
        Locked Activity 05 validation protocol.

    Returns
    -------
    dict
        Normalized benchmark settings used by all workloads.

    """
    settings = protocol["computational_cost"]
    require(settings["device"] == "CPU", "Computational-cost protocol does not request CPU")
    require(int(settings["warmup_runs"]) == 2, "Warm-up count differs from the locked protocol")
    require(int(settings["timed_repetitions"]) == 7, "Timed repetition count differs from the locked protocol")
    required_components = {"artifact_verification_and_loading", "feature_construction", "model_inference", "analytical_evaluation", "traditional_numerical_solution", "end_to_end"}
    require(set(settings["separate_components"]) == required_components, "Benchmark components differ from the locked protocol")
    thread_variables = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
    values = {name: os.environ.get(name) for name in thread_variables}
    require(all(value == "1" for value in values.values()), "All numerical thread environment variables must equal 1")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        require(torch.get_num_interop_threads() == 1, "PyTorch interoperation threads are not fixed at one")
    return {"warmup_runs": 2, "timed_repetitions": 7, "thread_environment": values}


def array_digest(values):
    """
    Compute a deterministic digest for one numerical result array.

    Parameters
    ----------
    values : array-like
        Numerical output whose values, shape, and dtype identify a run.

    Returns
    -------
    str
        SHA-256 digest of metadata and contiguous array bytes.

    """
    array  = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def table_digest(table):
    """
    Compute a deterministic digest for a model-ready feature table.

    Parameters
    ----------
    table : pandas.DataFrame
        Ordered table containing scenario identifiers and feature columns.

    Returns
    -------
    str
        SHA-256 digest of identifiers, columns, and float64 feature bytes.

    """
    features = [column for column in table if column.startswith("feature_")]
    digest   = hashlib.sha256()
    digest.update("\n".join(table["scenario_id"].astype(str)).encode("utf-8"))
    digest.update("\n".join(features).encode("utf-8"))
    digest.update(np.ascontiguousarray(table[features].to_numpy(dtype=float)).tobytes())
    return digest.hexdigest()


def ml_bundle_digest(bundle):
    """
    Summarize a loaded ADR1D-ML bundle without serializing estimators again.

    Parameters
    ----------
    bundle : dict
        Trusted Joblib model bundle.

    Returns
    -------
    str
        Digest of stable bundle contract values.

    """
    summary = {"bundle_version": bundle["bundle_version"], "protocol_sha256": bundle["protocol_sha256"], "model_names": sorted(bundle["models"]), "feature_counts": {key: len(value) for key, value in sorted(bundle["feature_columns"].items())}, "decision_threshold": float(bundle["decision_threshold"])}
    return hashlib.sha256(json.dumps(summary, sort_keys=True).encode("utf-8")).hexdigest()


def nn_surrogate_digest(surrogate):
    """
    Summarize a loaded ADR1D-NN surrogate contract.

    Parameters
    ----------
    surrogate : ADR1DNeuralSurrogate
        Verified in-memory neural surrogate.

    Returns
    -------
    str
        Digest of model hash, protocol version, features, and batch size.

    """
    summary = {"model_sha256": surrogate.manifest["model_sha256"], "protocol_version": surrogate.protocol["protocol_version"], "feature_names": list(surrogate.checkpoint["feature_names"]), "batch_size": int(surrogate.batch_size), "device": str(surrogate.device)}
    return hashlib.sha256(json.dumps(summary, sort_keys=True).encode("utf-8")).hexdigest()


def summarize_times(values):
    """
    Summarize seven retained elapsed-time measurements.

    Parameters
    ----------
    values : array-like of float
        Positive elapsed times in seconds.

    Returns
    -------
    dict
        Median, quartiles, interquartile range, minimum, and maximum.

    """
    values = np.asarray(values, dtype=float)
    require(values.size == 7 and np.isfinite(values).all() and (values > 0.0).all(), "Timed measurements are not seven finite positive values")
    lower = float(np.quantile(values, 0.25))
    upper = float(np.quantile(values, 0.75))
    return {"median_seconds": float(np.median(values)), "quartile_25_seconds": lower, "quartile_75_seconds": upper, "interquartile_range_seconds": upper - lower, "minimum_seconds": float(np.min(values)), "maximum_seconds": float(np.max(values))}


def measure_allocations(operation, signature):
    """
    Measure one un-timed Python allocation probe for a workload.

    Parameters
    ----------
    operation : callable
        Zero-argument workload operation.
    signature : callable
        Function returning a deterministic output digest.

    Returns
    -------
    dict
        Current and peak traced allocation bytes with output digest.

    Notes
    -----
    `tracemalloc` does not guarantee coverage of allocations retained solely by
    native numerical libraries, so this is not a resident-memory measurement.
    """
    gc.collect()
    tracemalloc.start()
    result = operation()
    current, peak = tracemalloc.get_traced_memory()
    digest = signature(result)
    tracemalloc.stop()
    return {"current_traced_bytes": int(current), "peak_traced_bytes": int(peak), "output_sha256": digest, "scope": "Python allocations observed by tracemalloc; native-library allocations may be undercounted."}


def measure_workload(label, operation, signature, work_units, work_unit_name, settings):
    """
    Execute warm-ups, timed repetitions, output checks, and an allocation probe.

    Parameters
    ----------
    label : str
        Stable workload identifier.
    operation : callable
        Zero-argument computational operation.
    signature : callable
        Deterministic result-signature function executed outside timing.
    work_units : int
        Number of scenarios, points, or artifacts processed in one operation.
    work_unit_name : str
        Human-readable unit associated with throughput.
    settings : dict
        Locked warm-up and timed repetition counts.

    Returns
    -------
    tuple
        Machine-readable measurement record and the last timed result.

    """
    signatures = []
    result     = None
    for _ in range(int(settings["warmup_runs"])):
        result = None
        gc.collect()
        result = operation()
        signatures.append(signature(result))
    elapsed = []
    for _ in range(int(settings["timed_repetitions"])):
        result = None
        gc.collect()
        started = time.perf_counter_ns()
        result  = operation()
        stopped = time.perf_counter_ns()
        elapsed.append((stopped - started) / 1.0e9)
        signatures.append(signature(result))
    require(len(set(signatures)) == 1, f"{label} produced inconsistent outputs across repetitions")
    summary    = summarize_times(elapsed)
    allocations = measure_allocations(operation, signature)
    require(allocations["output_sha256"] == signatures[0], f"{label} allocation probe produced a different output")
    record = {"label": label, "warmup_runs": int(settings["warmup_runs"]), "timed_repetitions": int(settings["timed_repetitions"]), "timed_seconds": elapsed, "summary": summary, "work_units": int(work_units), "work_unit_name": work_unit_name, "median_throughput_per_second": float(work_units / summary["median_seconds"]), "output_sha256": signatures[0], "deterministic_repetitions": True, "allocation_probe": allocations}
    return record, result


def load_ml_bundle():
    """
    Verify and deserialize the frozen ADR1D-ML bundle.

    Returns
    -------
    dict
        Trusted model bundle with estimators and feature contracts.

    """
    manifest = load_json(ML_MANIFEST_PATH)
    require(manifest["model_sha256"] == sha256(ML_MODEL_PATH), "ADR1D-ML model differs from its manifest")
    require(manifest["protocol_sha256"] == sha256(ML_PROTOCOL_PATH), "ADR1D-ML protocol differs from its manifest")
    bundle = joblib.load(ML_MODEL_PATH)
    require(bundle["protocol_sha256"] == manifest["protocol_sha256"], "ADR1D-ML bundle protocol differs from its manifest")
    return bundle


def construct_ml_features(test_scenarios, test_observations, base_module, physics_module):
    """
    Build all 86 ADR1D-ML predictors for the 45 canonical test scenarios.

    Parameters
    ----------
    test_scenarios : pandas.DataFrame
        Canonical metadata for exactly 45 test scenarios.
    test_observations : pandas.DataFrame
        Canonical sensor histories for those scenarios.
    base_module : module
        Trusted base feature-construction module.
    physics_module : module
        Trusted physics feature-construction module.

    Returns
    -------
    pandas.DataFrame
        Ordered model-ready canonical test table.

    """
    base         = base_module.build_table(test_scenarios, test_observations)
    physics_rows = [physics_module.build_physics_features(row) for _, row in base.iterrows()]
    output       = pd.concat([base.reset_index(drop=True), pd.DataFrame(physics_rows)], axis=1)
    require(len(output) == 45 and len([column for column in output if column.startswith("feature_")]) == 86, "ADR1D-ML feature workload has invalid dimensions")
    return output


def predict_ml(bundle, table):
    """
    Evaluate all four frozen ADR1D-ML estimators on a model-ready table.

    Parameters
    ----------
    bundle : dict
        Loaded frozen parameter-model bundle.
    table : pandas.DataFrame
        Canonical test table containing all required features.

    Returns
    -------
    numpy.ndarray
        Velocity, dispersion, resolvability probability, and conditional decay.

    """
    columns = bundle["feature_columns"]
    models  = bundle["models"]
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        velocity      = np.power(10.0, models["effective_velocity"].predict(table[columns["effective_parameters"]]))
        dispersion    = np.power(10.0, models["effective_dispersion"].predict(table[columns["effective_parameters"]]))
        probabilities = models["decay_resolvability"].predict_proba(table[columns["decay_resolvability"]])
        decay          = np.power(10.0, models["decay_rate_resolvable"].predict(table[columns["decay_rate_resolvable"]]))
    positive_index = int(np.where(models["decay_resolvability"].classes_ == 1)[0][0])
    output = np.column_stack((velocity, dispersion, probabilities[:, positive_index], decay))
    require(output.shape == (45, 4) and np.isfinite(output).all(), "ADR1D-ML inference workload produced invalid output")
    return output


def run_ml_end_to_end(test_scenarios, test_observations, base_module, physics_module):
    """
    Load ADR1D-ML, construct canonical features, and infer all four outputs.

    Parameters
    ----------
    test_scenarios : pandas.DataFrame
        Canonical metadata for 45 test scenarios.
    test_observations : pandas.DataFrame
        Canonical sensor histories for those scenarios.
    base_module : module
        Trusted base feature module.
    physics_module : module
        Trusted physics feature module.

    Returns
    -------
    numpy.ndarray
        Complete four-column canonical prediction matrix.

    """
    bundle = load_ml_bundle()
    table  = construct_ml_features(test_scenarios, test_observations, base_module, physics_module)
    return predict_ml(bundle, table)


def build_nn_input(canonical_scenarios, canonical_field):
    """
    Construct the physical ADR1D-NN input table for the canonical test split.

    Parameters
    ----------
    canonical_scenarios : pandas.DataFrame
        Canonical metadata for all 300 scenarios.
    canonical_field : pandas.DataFrame
        Canonical normalized concentration field.

    Returns
    -------
    tuple of pandas.DataFrame
        Ordered nine-column model input and matching analytical reference rows.

    """
    scenarios = canonical_scenarios.loc[canonical_scenarios["split"].eq("test")].copy()
    scenarios["effective_velocity_m_s"]   = scenarios["velocity_m_s"] / scenarios["retardation_factor"]
    scenarios["effective_dispersion_m2_s"] = scenarios["dispersion_m2_s"] / scenarios["retardation_factor"]
    metadata = scenarios.loc[:, ["scenario_id", "domain_length_m", "final_time_s", "effective_velocity_m_s", "effective_dispersion_m2_s", "decay_rate_s_1", "source_start_s", "source_duration_s"]]
    field    = canonical_field.loc[canonical_field["split"].eq("test"), ["scenario_id", "time_s", "x_m", "normalized_concentration"]]
    joined   = field.merge(metadata, on="scenario_id", how="inner", validate="many_to_one").sort_values(["scenario_id", "time_s", "x_m"]).reset_index(drop=True)
    columns  = ["scenario_id", "x_m", "time_s", "domain_length_m", "final_time_s", "effective_velocity_m_s", "effective_dispersion_m2_s", "decay_rate_s_1", "source_start_s", "source_duration_s"]
    require(len(joined) == 112455 and joined["scenario_id"].nunique() == 45, "ADR1D-NN canonical workload has invalid dimensions")
    return joined.loc[:, columns], joined.loc[:, ["scenario_id", "normalized_concentration"]]


def construct_nn_features(surrogate, table):
    """
    Validate canonical physical rows and construct standardized neural features.

    Parameters
    ----------
    surrogate : ADR1DNeuralSurrogate
        Loaded verified neural surrogate.
    table : pandas.DataFrame
        Canonical physical point inputs.

    Returns
    -------
    numpy.ndarray
        Float32 standardized features with seven columns.

    """
    numeric  = surrogate._validated_numeric_frame(table, True)
    features = surrogate._standardized_features(numeric)
    require(features.shape == (112455, 7), "ADR1D-NN feature workload has invalid dimensions")
    return features


def run_nn_end_to_end(nn_module, table):
    """
    Load ADR1D-NN and predict the complete canonical test field.

    Parameters
    ----------
    nn_module : module
        Trusted verified neural inference module.
    table : pandas.DataFrame
        Canonical physical point inputs.

    Returns
    -------
    numpy.ndarray
        Constrained normalized predictions in canonical row order.

    """
    surrogate = nn_module.load_verified_surrogate(NN_MODEL_PATH, NN_MANIFEST_PATH, device="cpu")
    output    = surrogate.predict_table(table, strict_domain=True)
    return output["predicted_normalized_concentration"].to_numpy(dtype=float)


def analytical_workload(selected, fine_level, numerical_reference):
    """
    Evaluate analytical fields for all 16 selected reference scenarios.

    Parameters
    ----------
    selected : pandas.DataFrame
        Truth-selected challenge scenarios from the locked numerical reference.
    fine_level : dict
        Fine grid dimensions from the validation protocol.
    numerical_reference : module
        Trusted analytical and finite-volume implementation module.

    Returns
    -------
    numpy.ndarray
        Concatenated analytical fields for 376,320 cell-center values.

    """
    domain_length = 1200.0
    spatial_step  = float(fine_level["spatial_step_m"])
    positions     = (np.arange(int(round(domain_length / spatial_step)), dtype=float) + 0.5) * spatial_step
    values        = []
    for _, scenario in selected.iterrows():
        times = np.arange(int(scenario["time_nodes"]), dtype=float) * float(scenario["time_step_s"])
        values.append(numerical_reference.analytical_pulse(positions, times, scenario).ravel())
    output = np.concatenate(values)
    require(output.size == 376320 and np.isfinite(output).all(), "Analytical cost workload produced invalid output")
    return output


def numerical_workload(selected, fine_level, reference, numerical_reference):
    """
    Solve all 16 selected scenarios on the locked fine numerical grid.

    Parameters
    ----------
    selected : pandas.DataFrame
        Truth-selected challenge scenarios.
    fine_level : dict
        Fine spatial and maximum temporal steps.
    reference : dict
        Locked traditional numerical-reference contract.
    numerical_reference : module
        Trusted finite-volume implementation module.

    Returns
    -------
    numpy.ndarray
        Concatenated numerical fields for 376,320 cell-center values.

    """
    values = [numerical_reference.solve_level(scenario, fine_level, reference)["snapshots"].ravel() for _, scenario in selected.iterrows()]
    output = np.concatenate(values)
    require(output.size == 376320 and np.isfinite(output).all(), "Traditional numerical cost workload produced invalid output")
    return output


def numerical_end_to_end(selected, fine_level, reference, numerical_reference):
    """
    Evaluate analytical and numerical fields for the 16 selected scenarios.

    Parameters
    ----------
    selected : pandas.DataFrame
        Truth-selected challenge scenarios.
    fine_level : dict
        Fine spatial and maximum temporal steps.
    reference : dict
        Locked traditional numerical-reference contract.
    numerical_reference : module
        Trusted analytical and finite-volume implementation module.

    Returns
    -------
    numpy.ndarray
        Two-column analytical and numerical values in matching order.

    """
    analytical = analytical_workload(selected, fine_level, numerical_reference)
    numerical  = numerical_workload(selected, fine_level, reference, numerical_reference)
    return np.column_stack((analytical, numerical))


def validate_ml_results(feature_table, predictions, stored_features, stored_predictions):
    """
    Compare benchmark ADR1D-ML features and predictions with canonical evidence.

    Parameters
    ----------
    feature_table : pandas.DataFrame
        Last independently built canonical test features.
    predictions : numpy.ndarray
        Last four-column parameter prediction matrix.
    stored_features : pandas.DataFrame
        Published physics-enhanced canonical test table.
    stored_predictions : pandas.DataFrame
        Published final canonical test predictions.

    Returns
    -------
    dict
        Maximum feature and prediction differences with verified row counts.

    """
    stored_features   = stored_features.sort_values("scenario_id").reset_index(drop=True)
    stored_predictions = stored_predictions.sort_values("scenario_id").reset_index(drop=True)
    require(feature_table["scenario_id"].tolist() == stored_features["scenario_id"].tolist() == stored_predictions["scenario_id"].tolist(), "ADR1D-ML canonical row order differs")
    feature_columns = [column for column in feature_table if column.startswith("feature_")]
    feature_difference = np.abs(feature_table[feature_columns].to_numpy(dtype=float) - stored_features[feature_columns].to_numpy(dtype=float))
    prediction_columns = ["predicted_effective_velocity_m_s", "predicted_effective_dispersion_m2_s", "predicted_decay_resolvable_probability", "predicted_decay_rate_if_resolvable_s_1"]
    prediction_difference = np.abs(predictions - stored_predictions[prediction_columns].to_numpy(dtype=float))
    maximum_feature    = float(np.nanmax(feature_difference))
    maximum_prediction = float(np.nanmax(prediction_difference))
    require(maximum_feature <= 1.0e-7, "ADR1D-ML benchmark features differ from canonical evidence")
    require(maximum_prediction <= 2.0e-9, "ADR1D-ML benchmark predictions differ from canonical evidence")
    return {"canonical_scenarios": 45, "feature_columns": 86, "maximum_absolute_feature_difference": maximum_feature, "maximum_absolute_prediction_difference": maximum_prediction}


def validate_nn_results(predictions, reference, stored_metrics):
    """
    Reproduce canonical ADR1D-NN point metrics from benchmark predictions.

    Parameters
    ----------
    predictions : numpy.ndarray
        Last constrained canonical neural predictions.
    reference : pandas.DataFrame
        Matching canonical normalized concentrations.
    stored_metrics : dict
        Published final-test metrics.

    Returns
    -------
    dict
        Reproduced MAE, RMSE, R2, and maximum discrepancy from stored values.

    """
    actual   = reference["normalized_concentration"].to_numpy(dtype=float)
    residual = predictions - actual
    metrics  = {"mae": float(np.mean(np.abs(residual))), "rmse": float(np.sqrt(np.mean(np.square(residual)))), "r2": float(1.0 - np.sum(np.square(residual)) / np.sum(np.square(actual - actual.mean()))), "maximum_absolute_error": float(np.max(np.abs(residual)))}
    differences = {key: abs(value - float(stored_metrics["final_model"][key])) for key, value in metrics.items()}
    maximum = max(differences.values())
    require(maximum <= 2.0e-12, "ADR1D-NN benchmark predictions do not reproduce canonical metrics")
    return {"canonical_scenarios": 45, "point_predictions": int(len(predictions)), "metrics": metrics, "maximum_absolute_metric_difference": maximum}


def validate_numerical_results(analytical, numerical, selected, stored_cases):
    """
    Reproduce fine-grid RMSE values from the cost-workload fields.

    Parameters
    ----------
    analytical : numpy.ndarray
        Concatenated fine analytical fields.
    numerical : numpy.ndarray
        Concatenated fine numerical fields.
    selected : pandas.DataFrame
        Selected scenarios in benchmark order.
    stored_cases : pandas.DataFrame
        Persisted 48-row numerical-reference metrics table.

    Returns
    -------
    dict
        Scenario count and maximum RMSE difference from persisted evidence.

    """
    rows_per_scenario = 49 * 480
    comparison_cells  = int(np.sum(((np.arange(480, dtype=float) + 0.5) * 2.5) <= 900.0))
    observed = []
    for index in range(len(selected)):
        start = index * rows_per_scenario
        stop  = start + rows_per_scenario
        actual = analytical[start:stop].reshape(49, 480)[:, :comparison_cells]
        predicted = numerical[start:stop].reshape(49, 480)[:, :comparison_cells]
        observed.append(float(np.sqrt(np.mean(np.square(predicted - actual)))))
    stored = stored_cases.loc[stored_cases["level"].eq("fine")].set_index("scenario_id").loc[selected["scenario_id"], "rmse"].to_numpy(dtype=float)
    maximum = float(np.max(np.abs(np.asarray(observed) - stored)))
    require(maximum <= 2.0e-12, "Numerical cost workload differs from persisted fine-grid evidence")
    return {"selected_scenarios": int(len(selected)), "retained_field_values": int(len(numerical)), "maximum_absolute_rmse_difference": maximum}


def physical_memory_bytes():
    """
    Determine installed physical memory using portable operating-system data.

    Returns
    -------
    int or None
        Installed physical memory in bytes when available.

    """
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def processor_name():
    """
    Resolve a descriptive processor name without network access.

    Returns
    -------
    str
        Processor brand when available, otherwise machine architecture.

    """
    name = platform.processor().strip()
    if name:
        return name
    if platform.system() == "Darwin":
        process = subprocess.run(("sysctl", "-n", "machdep.cpu.brand_string"), capture_output=True, text=True, check=False)
        if process.returncode == 0 and process.stdout.strip():
            return process.stdout.strip()
    return platform.machine()


def environment_record(settings):
    """
    Record the locked CPU environment and relevant library versions.

    Parameters
    ----------
    settings : dict
        Verified thread environment and repetition counts.

    Returns
    -------
    dict
        Operating system, processor, memory, Python, libraries, and threads.

    """
    pools = []
    for item in threadpool_info():
        pools.append({"user_api": item.get("user_api"), "internal_api": item.get("internal_api"), "prefix": item.get("prefix"), "num_threads": item.get("num_threads"), "version": item.get("version")})
    return {"operating_system": platform.platform(), "processor": processor_name(), "machine": platform.machine(), "logical_cpu_count": os.cpu_count(), "physical_memory_bytes": physical_memory_bytes(), "python_version": platform.python_version(), "python_implementation": platform.python_implementation(), "library_versions": {"joblib": joblib.__version__, "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "torch": torch.__version__}, "thread_policy": {"environment": settings["thread_environment"], "torch_intraop_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "detected_native_pools": pools}}


def execute_benchmark(output_path, overwrite=False):
    """
    Execute all locked computational-cost workloads and persist their report.

    Parameters
    ----------
    output_path : pathlib.Path
        Destination JSON report path.
    overwrite : bool, optional
        Whether an existing report may be intentionally replaced.

    Returns
    -------
    dict
        Complete benchmark report.

    """
    require(overwrite or not output_path.exists(), "Computational-cost report already exists; use --overwrite only for an intentional replacement")
    protocol       = load_json(PROTOCOL_PATH)
    challenge_lock = load_json(CHALLENGE_LOCK_PATH)
    require(sha256(PROTOCOL_PATH) == challenge_lock["protocol"]["sha256"], "Activity 05 protocol differs from the challenge lock")
    settings       = verify_protocol_and_threads(protocol)
    before_hashes  = protected_hashes()
    base_module    = load_module("adr1d_cost_base_features", ML_BASE_MODULE_PATH)
    physics_module = load_module("adr1d_cost_physics_features", ML_PHYSICS_MODULE_PATH)
    nn_module      = load_module("adr1d_cost_nn_inference", NN_INFERENCE_PATH)
    numerical_reference = load_module("adr1d_cost_numerical_reference", NUMERICAL_MODULE_PATH)

    canonical_scenarios = pd.read_csv(CANONICAL_SCENARIOS)
    canonical_field     = pd.read_csv(CANONICAL_FIELD)
    canonical_sensors   = pd.read_csv(CANONICAL_SENSORS)
    test_ids            = set(canonical_scenarios.loc[canonical_scenarios["split"].eq("test"), "scenario_id"])
    challenge_ids       = set(pd.read_csv(CHALLENGE_SCENARIOS, usecols=("scenario_id",))["scenario_id"])
    require(len(test_ids) == 45 and not test_ids & challenge_ids, "Canonical timing identifiers overlap the locked challenge")
    test_scenarios      = canonical_scenarios.loc[canonical_scenarios["scenario_id"].isin(test_ids)].copy()
    test_observations   = canonical_sensors.loc[canonical_sensors["scenario_id"].isin(test_ids)].copy()
    nn_input, nn_reference = build_nn_input(canonical_scenarios, canonical_field)
    require(set(nn_input["scenario_id"]) == test_ids, "ADR1D-NN timing input is not the canonical test split")

    stored_ml_features    = pd.read_csv(ML_FEATURE_TABLE).loc[lambda frame: frame["split"].eq("test")].copy()
    stored_ml_predictions = pd.read_csv(ML_TEST_PREDICTIONS)
    stored_nn_metrics     = load_json(NN_METRICS_PATH)
    stored_numerical      = pd.read_csv(NUMERICAL_CASES_PATH)
    reference_contract    = protocol["traditional_numerical_reference"]
    selected              = numerical_reference.select_reference_scenarios(pd.read_csv(CHALLENGE_SCENARIOS), protocol)
    fine_level            = next(level for level in reference_contract["grid_levels"] if level["name"] == "fine")

    ml_bundle   = load_ml_bundle()
    ml_features = construct_ml_features(test_scenarios, test_observations, base_module, physics_module)
    nn_surrogate = nn_module.load_verified_surrogate(NN_MODEL_PATH, NN_MANIFEST_PATH, device="cpu")
    nn_features  = construct_nn_features(nn_surrogate, nn_input)
    measurements = {}
    last_results = {}

    workloads = (
        ("adr1d_ml_artifact_verification_and_loading", load_ml_bundle, ml_bundle_digest, 1, "verified_model_bundle"),
        ("adr1d_ml_feature_construction", functools.partial(construct_ml_features, test_scenarios, test_observations, base_module, physics_module), table_digest, 45, "base_scenario"),
        ("adr1d_ml_model_inference", functools.partial(predict_ml, ml_bundle, ml_features), array_digest, 45, "base_scenario"),
        ("adr1d_ml_end_to_end", functools.partial(run_ml_end_to_end, test_scenarios, test_observations, base_module, physics_module), array_digest, 45, "base_scenario"),
        ("adr1d_nn_artifact_verification_and_loading", functools.partial(nn_module.load_verified_surrogate, NN_MODEL_PATH, NN_MANIFEST_PATH, device="cpu"), nn_surrogate_digest, 1, "verified_neural_surrogate"),
        ("adr1d_nn_feature_construction", functools.partial(construct_nn_features, nn_surrogate, nn_input), array_digest, 112455, "point_feature_row"),
        ("adr1d_nn_model_inference", functools.partial(nn_surrogate._raw_prediction, nn_features), array_digest, 112455, "point_prediction"),
        ("adr1d_nn_end_to_end", functools.partial(run_nn_end_to_end, nn_module, nn_input), array_digest, 112455, "constrained_point_prediction"),
        ("analytical_evaluation", functools.partial(analytical_workload, selected, fine_level, numerical_reference), array_digest, 376320, "normalized_field_value"),
        ("traditional_numerical_solution", functools.partial(numerical_workload, selected, fine_level, reference_contract, numerical_reference), array_digest, 376320, "retained_field_value"),
        ("numerical_reference_end_to_end", functools.partial(numerical_end_to_end, selected, fine_level, reference_contract, numerical_reference), array_digest, 376320, "paired_analytical_numerical_value"),
    )
    with threadpool_limits(limits=1):
        for label, operation, signature, work_units, unit_name in workloads:
            record, result = measure_workload(label, operation, signature, work_units, unit_name, settings)
            measurements[label] = record
            last_results[label] = result

    validations = {
        "adr1d_ml": validate_ml_results(last_results["adr1d_ml_feature_construction"], last_results["adr1d_ml_model_inference"], stored_ml_features, stored_ml_predictions),
        "adr1d_nn": validate_nn_results(last_results["adr1d_nn_end_to_end"], nn_reference, stored_nn_metrics),
        "numerical_reference": validate_numerical_results(last_results["analytical_evaluation"], last_results["traditional_numerical_solution"], selected, stored_numerical),
    }
    after_hashes = protected_hashes()
    require(before_hashes == after_hashes, "A protected model, input, or challenge result changed during benchmarking")
    report = {
        "status": "complete",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "benchmark_script": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": sha256(Path(__file__).resolve())},
        "scope": {"device": "CPU", "timing_claim": protocol["computational_cost"]["timing_claim_scope"], "data_residency": "Input DataFrames were loaded before timing; CSV input and result serialization are excluded.", "model_timing_data": "Canonical ADR1D 1.0.0 test split only.", "model_timing_base_scenarios": 45, "adr1d_ml_timing_rows": 45, "adr1d_nn_timing_rows": 112455, "numerical_timing_data": "Sixteen truth-selected locked numerical-reference scenarios on the fine grid.", "numerical_timing_rows": 376320, "challenge_model_inference_runs": 0, "persisted_model_predictions_created": 0, "post_challenge_tuning_performed": False, "speedup_claim_made": False},
        "settings": {"warmup_runs": settings["warmup_runs"], "timed_repetitions": settings["timed_repetitions"], "summary_statistics": protocol["computational_cost"]["summary"], "garbage_collection": "A full collection was requested before each warm-up and timed repetition; collection time was excluded.", "clock": "time.perf_counter_ns", "allocation_probe_runs": 1},
        "measurements": measurements,
        "canonical_and_numerical_reproduction": validations,
        "environment": environment_record(settings),
        "artifact_sizes_bytes": {"adr1d_ml_model": ML_MODEL_PATH.stat().st_size, "adr1d_nn_model": NN_MODEL_PATH.stat().st_size, "canonical_ml_raw_input_memory": int(test_scenarios.memory_usage(index=True, deep=True).sum() + test_observations.memory_usage(index=True, deep=True).sum()), "canonical_ml_feature_table_memory": int(ml_features.memory_usage(index=True, deep=True).sum()), "canonical_nn_input_memory": int(nn_input.memory_usage(index=True, deep=True).sum())},
        "integrity": {"protected_artifacts_unchanged": True, "protected_artifact_count": len(before_hashes), "before_sha256": before_hashes, "after_sha256": after_hashes},
        "limitations": ["Timings describe this machine, software environment, workload size, warm-cache state, and single-thread policy.", "Model timings use only the published canonical test split and do not repeat inference on the locked Activity 05 challenge set.", "Input CSV parsing and output serialization are excluded from timed computational cores.", "Tracemalloc allocation probes may undercount memory managed only by native numerical libraries.", "Different workflows solve different scientific tasks; their raw times are not presented as interchangeable universal speed-ups."],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main():
    """
    Run the locked computational-cost benchmark and print compact results.

    Returns
    -------
    None
        A complete machine-readable benchmark report is written to `results/`.

    """
    arguments = parse_arguments()
    report    = execute_benchmark(arguments.output.resolve(), arguments.overwrite)
    medians   = {label: values["summary"]["median_seconds"] for label, values in report["measurements"].items()}
    print(json.dumps({"status": report["status"], "output": str(arguments.output), "challenge_model_inference_runs": report["scope"]["challenge_model_inference_runs"], "median_seconds": medians}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
