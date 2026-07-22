"""
================================================================================
ADR1D Validation: Locked Challenge Evaluation
================================================================================

Evaluate the frozen ADR1D-ML and ADR1D-NN models exactly once on the locked
challenge set and retain predictions, metrics, uncertainty, and failure cases.

Main Operations
---------------
1. Verify the evaluation lock and rebuild byte-identical model inputs.
2. Invoke each frozen public inference interface once on the challenge data.
3. Aggregate nested sensor replicates and calculate pre-specified metrics.
4. Apply scenario-cluster bootstrap intervals and adequacy criteria.
5. Persist complete predictions and machine-readable evaluation evidence.

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
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

# Third-party libraries
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, r2_score, recall_score, roc_auc_score

# Local modules
import prepare_challenge_evaluation as preparation


ROOT                = Path(__file__).resolve().parents[1]
PROTOCOL_PATH       = ROOT / "configs/validation_protocol.json"
EVALUATION_LOCK     = ROOT / "results/challenge_evaluation_lock.json"
SCENARIO_PATH       = ROOT / "data/challenge_scenarios.csv"
FIELD_PATH          = ROOT / "data/challenge_analytical_field.csv"
SENSOR_PATH         = ROOT / "data/challenge_sensor_observations.csv"
STAGING_DIRECTORY   = ROOT / "results/.challenge_inference_staging"
INCIDENT_PATH       = ROOT / "results/challenge_inference_incident.json"

ML_ROOT             = ROOT.parent / "03_modelos_ml_parametros"
ML_API              = ML_ROOT / "scripts/predict_parameters.py"
ML_MODEL            = ML_ROOT / "models/adr1d_parameter_models.joblib"
ML_MANIFEST         = ML_ROOT / "models/model_manifest.json"
ML_PROTOCOL         = ML_ROOT / "results/final_model_protocol.json"

NN_ROOT             = ROOT.parent / "04_redes_neuronales_transporte"
NN_API              = NN_ROOT / "src/predict_concentration.py"
NN_MODEL            = NN_ROOT / "models/adr1d_nn.pt"
NN_MANIFEST         = NN_ROOT / "models/model_manifest.json"

ML_RAW_NAME         = "adr1d_ml_api_output.csv"
NN_RAW_NAME         = "adr1d_nn_api_output.csv"
ML_FINAL_NAME       = "adr1d_ml_challenge_predictions.csv"
ML_METRICS_NAME     = "adr1d_ml_challenge_metrics.json"
NN_FINAL_NAME       = "adr1d_nn_challenge_predictions.csv"
NN_SCENARIOS_NAME   = "adr1d_nn_challenge_scenarios.csv"
NN_METRICS_NAME     = "adr1d_nn_challenge_metrics.json"


class EvaluationError(RuntimeError):
    """Represent a violation of the locked challenge-evaluation contract."""


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
        Existing JSON file.

    Returns
    -------
    dict
        Parsed top-level JSON object.

    Raises
    ------
    FileNotFoundError
        If the file is absent.
    EvaluationError
        If its top-level value is not an object.

    """
    if not path.exists():
        raise FileNotFoundError(path)
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise EvaluationError(f"Expected a JSON object in {path}")
    return content


def require(condition, message):
    """
    Enforce one locked evaluation condition.

    Parameters
    ----------
    condition : bool
        Condition that must be true.
    message : str
        Failure description.

    Returns
    -------
    None
        The function returns only when the condition is satisfied.

    Raises
    ------
    EvaluationError
        If the condition is false.

    """
    if not condition:
        raise EvaluationError(message)


def verify_evaluation_lock(protocol):
    """
    Verify the self-referential code lock and all protected pre-inference state.

    Parameters
    ----------
    protocol : dict
        Locked Activity 05 validation protocol.

    Returns
    -------
    dict
        Verified challenge-evaluation lock.

    Raises
    ------
    EvaluationError
        If code, input, model, API, or zero-query state differs from the lock.

    """
    lock = load_json(EVALUATION_LOCK)
    require(lock["status"] == "locked_before_challenge_inference", "Evaluation is not locked before challenge inference")
    require(lock["protocol"]["sha256"] == sha256(PROTOCOL_PATH), "Evaluation lock references a different protocol")
    require(lock["locked_code"]["evaluation"]["sha256"] == sha256(Path(__file__)), "Locked evaluator code has changed")
    require(lock["locked_code"]["preparation"]["sha256"] == sha256(Path(preparation.__file__)), "Locked preparation code has changed")
    state = lock["state_at_lock"]
    require(not state["adr1d_ml_model_loaded"] and not state["adr1d_nn_model_loaded"], "Evaluation lock reports a model loaded before lock")
    require(int(state["adr1d_ml_challenge_inference_runs"]) == 0 and int(state["adr1d_nn_challenge_inference_runs"]) == 0, "Evaluation lock does not record zero challenge inference")
    require(math.isclose(float(lock["execution_contract"]["adr1d_ml_decision_threshold"]), ml_decision_threshold(), rel_tol=0.0, abs_tol=0.0), "Frozen ADR1D-ML decision threshold differs from the evaluation lock")

    guard = preparation.verify_preparation_state(protocol)
    require(guard == lock["challenge_guard"], "Protected challenge state differs from the evaluation lock")
    frozen_ml = protocol["frozen_inputs"]["adr1d_ml"]
    frozen_nn = protocol["frozen_inputs"]["adr1d_nn"]
    require(sha256(ML_MODEL) == frozen_ml["model"]["sha256"] and sha256(ML_API) == frozen_ml["inference_module"]["sha256"], "Frozen ADR1D-ML artifacts differ")
    require(sha256(NN_MODEL) == frozen_nn["model"]["sha256"] and sha256(NN_API) == frozen_nn["inference_module"]["sha256"], "Frozen ADR1D-NN artifacts differ")
    return lock


def inference_environment():
    """
    Construct the single-threaded environment for persisted challenge inference.

    Returns
    -------
    dict
        Process environment with bytecode disabled and numerical threads fixed.

    """
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return environment


def run_api(command, label):
    """
    Execute one frozen public inference command and retain its console evidence.

    Parameters
    ----------
    command : list of str
        Complete subprocess argument vector.
    label : str
        Model label used in errors.

    Returns
    -------
    dict
        Command, standard output, and standard error.

    Raises
    ------
    EvaluationError
        If the public inference command exits unsuccessfully.

    """
    process = subprocess.run(command, cwd=ROOT.parent.parent, env=inference_environment(), capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise EvaluationError(f"{label} public inference failed with exit code {process.returncode}: {process.stderr.strip()}")
    return {"command": command, "stdout": process.stdout.strip(), "stderr": process.stderr.strip(), "return_code": process.returncode}


def regression_metrics(actual, predicted):
    """
    Compute locked logarithmic and relative regression metrics.

    Parameters
    ----------
    actual : array-like of float
        Positive reference values.
    predicted : array-like of float
        Positive predictions in matching order.

    Returns
    -------
    dict
        Logarithmic MAE, RMSE, R2, and relative-error summaries.

    """
    actual       = np.asarray(actual, dtype=float)
    predicted    = np.asarray(predicted, dtype=float)
    actual_log   = np.log10(actual)
    predicted_log = np.log10(predicted)
    residual     = predicted_log - actual_log
    percentage   = np.abs(predicted - actual) / actual
    return {
        "rows": int(actual.size),
        "mae_log10": float(np.mean(np.abs(residual))),
        "rmse_log10": float(np.sqrt(np.mean(np.square(residual)))),
        "r2_log10": float(r2_score(actual_log, predicted_log)),
        "median_absolute_percentage_error": float(np.median(percentage)),
        "percentile_90_absolute_percentage_error": float(np.quantile(percentage, 0.90)),
    }


def classification_metrics(actual, probability, threshold):
    """
    Compute locked binary decay-resolvability metrics at the frozen threshold.

    Parameters
    ----------
    actual : array-like of int
        Binary truth labels at base-scenario level.
    probability : array-like of float
        Mean resolvability probabilities at base-scenario level.
    threshold : float
        Frozen probability decision threshold.

    Returns
    -------
    dict
        Classification scores and confusion matrix.

    """
    actual      = np.asarray(actual, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predicted   = (probability >= threshold).astype(int)
    matrix      = confusion_matrix(actual, predicted, labels=[0, 1]).ravel().astype(int)
    return {
        "rows": int(actual.size),
        "positive_truth_rows": int(actual.sum()),
        "decision_threshold": float(threshold),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, probability)),
        "log_loss": float(log_loss(actual, np.column_stack((1.0 - probability, probability)), labels=[0, 1])),
        "confusion_matrix_tn_fp_fn_tp": matrix.tolist(),
    }


def ml_decision_threshold():
    """
    Read the frozen ADR1D-ML decay-resolvability threshold.

    Returns
    -------
    float
        Probability threshold fixed before the original test evaluation.

    """
    protocol = load_json(ML_PROTOCOL)
    return float(protocol["models"]["decay_resolvability"]["decision_threshold"])


def conditional_decay_metrics(actual, predicted, resolvable):
    """
    Evaluate conditional decay regression on truth-labeled resolvable scenarios.

    Parameters
    ----------
    actual : array-like of float
        True first-order decay rates.
    predicted : array-like of float
        Geometric-mean conditional predictions.
    resolvable : array-like of bool
        Truth labels fixed by the Damkohler threshold.

    Returns
    -------
    dict
        Conditional logarithmic errors and relative-error median.

    """
    mask      = np.asarray(resolvable, dtype=bool)
    actual    = np.asarray(actual, dtype=float)[mask]
    predicted = np.asarray(predicted, dtype=float)[mask]
    actual_log = np.log10(actual)
    predicted_log = np.log10(predicted)
    residual = predicted_log - actual_log
    return {
        "rows": int(actual.size),
        "mae_log10": float(np.mean(np.abs(residual))),
        "rmse_log10": float(np.sqrt(np.mean(np.square(residual)))),
        "r2_log10": float(r2_score(actual_log, predicted_log)) if actual.size >= 2 else None,
        "median_absolute_percentage_error": float(np.median(np.abs(predicted - actual) / actual)),
    }


def enrich_ml_predictions(raw_path, protocol):
    """
    Join public ADR1D-ML outputs with replicate metadata and hidden truth values.

    Parameters
    ----------
    raw_path : pathlib.Path
        Temporary public-API output with 600 realization predictions.
    protocol : dict
        Locked validation protocol.

    Returns
    -------
    pandas.DataFrame
        Complete replicate-level prediction evidence.

    Raises
    ------
    EvaluationError
        If identifiers, rows, probabilities, or physical predictions differ.

    """
    raw          = pd.read_csv(raw_path)
    scenarios    = pd.read_csv(SCENARIO_PATH)
    observations = pd.read_csv(SENSOR_PATH, usecols=("scenario_id", "base_scenario_id", "replicate_id"))
    identifiers  = observations.drop_duplicates().sort_values("scenario_id").reset_index(drop=True)
    require(len(raw) == 600 and raw["scenario_id"].tolist() == identifiers["scenario_id"].tolist(), "ADR1D-ML public output identifiers or ordering differ")
    metadata = identifiers.merge(scenarios, left_on="base_scenario_id", right_on="scenario_id", how="left", validate="many_to_one", suffixes=("", "_base"))
    threshold = float(protocol["challenge_design"]["decay_resolvability"]["damkohler_threshold"])
    result = metadata.loc[:, ["base_scenario_id", "scenario_id", "replicate_id", "split", "regime", "design_component", "design_index"]].copy()
    result["actual_effective_velocity_m_s"] = metadata["velocity_m_s"] / metadata["retardation_factor"]
    result["predicted_effective_velocity_m_s"] = raw["effective_velocity_m_s"].to_numpy(dtype=float)
    result["actual_effective_dispersion_m2_s"] = metadata["dispersion_m2_s"] / metadata["retardation_factor"]
    result["predicted_effective_dispersion_m2_s"] = raw["effective_dispersion_m2_s"].to_numpy(dtype=float)
    result["actual_decay_resolvable"] = metadata["damkohler_number"].ge(threshold).astype(int)
    result["predicted_decay_resolvable_probability"] = raw["decay_resolvable_probability"].to_numpy(dtype=float)
    result["predicted_decay_resolvable"] = raw["decay_resolvable"].to_numpy(dtype=int)
    result["actual_decay_rate_s_1"] = metadata["decay_rate_s_1"].to_numpy(dtype=float)
    result["predicted_decay_rate_if_resolvable_s_1"] = raw["decay_rate_if_resolvable_s_1"].to_numpy(dtype=float)
    result["reported_decay_rate_s_1"] = raw["reported_decay_rate_s_1"].to_numpy(dtype=float)
    require((result["predicted_effective_velocity_m_s"] > 0.0).all(), "ADR1D-ML produced a non-positive velocity")
    require((result["predicted_effective_dispersion_m2_s"] > 0.0).all(), "ADR1D-ML produced a non-positive dispersion")
    require(result["predicted_decay_resolvable_probability"].between(0.0, 1.0).all(), "ADR1D-ML produced a probability outside [0, 1]")
    require((result["predicted_decay_rate_if_resolvable_s_1"] > 0.0).all(), "ADR1D-ML produced a non-positive conditional decay rate")
    return result


def geometric_mean(values):
    """
    Compute the geometric mean of strictly positive predictions.

    Parameters
    ----------
    values : array-like of float
        Strictly positive values.

    Returns
    -------
    float
        Geometric mean.

    """
    values = np.asarray(values, dtype=float)
    return float(np.power(10.0, np.mean(np.log10(values))))


def aggregate_ml_predictions(predictions, threshold):
    """
    Aggregate five nested noise realizations to each independent base scenario.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Complete 600-row replicate-level evidence.
    threshold : float
        Frozen resolvability decision threshold.

    Returns
    -------
    pandas.DataFrame
        One row per base scenario using geometric means and mean probability.

    """
    rows = []
    for base_scenario_id, frame in predictions.groupby("base_scenario_id", sort=True):
        probability = float(frame["predicted_decay_resolvable_probability"].mean())
        rows.append(
            {
                "base_scenario_id": base_scenario_id,
                "regime": str(frame["regime"].iloc[0]),
                "design_component": str(frame["design_component"].iloc[0]),
                "actual_effective_velocity_m_s": float(frame["actual_effective_velocity_m_s"].iloc[0]),
                "predicted_effective_velocity_m_s": geometric_mean(frame["predicted_effective_velocity_m_s"]),
                "actual_effective_dispersion_m2_s": float(frame["actual_effective_dispersion_m2_s"].iloc[0]),
                "predicted_effective_dispersion_m2_s": geometric_mean(frame["predicted_effective_dispersion_m2_s"]),
                "actual_decay_resolvable": int(frame["actual_decay_resolvable"].iloc[0]),
                "predicted_decay_resolvable_probability": probability,
                "predicted_decay_resolvable": int(probability >= threshold),
                "actual_decay_rate_s_1": float(frame["actual_decay_rate_s_1"].iloc[0]),
                "predicted_decay_rate_if_resolvable_s_1": geometric_mean(frame["predicted_decay_rate_if_resolvable_s_1"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_distribution(values):
    """
    Summarize a finite one-dimensional diagnostic distribution.

    Parameters
    ----------
    values : array-like of float
        Finite values to summarize.

    Returns
    -------
    dict
        Count, median, 90th percentile, and maximum.

    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "rows": int(values.size),
        "median": float(np.median(values)),
        "percentile_90": float(np.quantile(values, 0.90)),
        "maximum": float(np.max(values)),
    }


def ml_repeatability(predictions, aggregated):
    """
    Quantify variation among the five sensor-noise realizations per scenario.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Replicate-level ADR1D-ML predictions.
    aggregated : pandas.DataFrame
        Base-scenario predictions and final classifications.

    Returns
    -------
    dict
        Distribution summaries for standard deviation, coefficient of
        variation, probability variation, and classification agreement.

    """
    aggregate_labels = aggregated.set_index("base_scenario_id")["predicted_decay_resolvable"]
    diagnostics = {
        "velocity_log10_standard_deviation": [],
        "velocity_coefficient_of_variation": [],
        "dispersion_log10_standard_deviation": [],
        "dispersion_coefficient_of_variation": [],
        "conditional_decay_log10_standard_deviation": [],
        "conditional_decay_coefficient_of_variation": [],
        "probability_standard_deviation": [],
        "classification_agreement_fraction": [],
    }
    for base_scenario_id, frame in predictions.groupby("base_scenario_id", sort=True):
        velocity   = frame["predicted_effective_velocity_m_s"].to_numpy(dtype=float)
        dispersion = frame["predicted_effective_dispersion_m2_s"].to_numpy(dtype=float)
        decay      = frame["predicted_decay_rate_if_resolvable_s_1"].to_numpy(dtype=float)
        probability = frame["predicted_decay_resolvable_probability"].to_numpy(dtype=float)
        diagnostics["velocity_log10_standard_deviation"].append(float(np.std(np.log10(velocity), ddof=1)))
        diagnostics["velocity_coefficient_of_variation"].append(float(np.std(velocity, ddof=1) / np.mean(velocity)))
        diagnostics["dispersion_log10_standard_deviation"].append(float(np.std(np.log10(dispersion), ddof=1)))
        diagnostics["dispersion_coefficient_of_variation"].append(float(np.std(dispersion, ddof=1) / np.mean(dispersion)))
        diagnostics["conditional_decay_log10_standard_deviation"].append(float(np.std(np.log10(decay), ddof=1)))
        diagnostics["conditional_decay_coefficient_of_variation"].append(float(np.std(decay, ddof=1) / np.mean(decay)))
        diagnostics["probability_standard_deviation"].append(float(np.std(probability, ddof=1)))
        diagnostics["classification_agreement_fraction"].append(float(np.mean(frame["predicted_decay_resolvable"].to_numpy(dtype=int) == int(aggregate_labels.loc[base_scenario_id]))))
    return {key: summarize_distribution(value) for key, value in diagnostics.items()}


def bootstrap_interval(values, confidence_level):
    """
    Calculate a percentile confidence interval from finite bootstrap values.

    Parameters
    ----------
    values : array-like of float
        Bootstrap estimates, possibly containing NaN for non-estimable draws.
    confidence_level : float
        Requested central confidence level.

    Returns
    -------
    dict
        Lower and upper percentiles with the valid-resample count.

    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    alpha  = (1.0 - confidence_level) / 2.0
    return {"lower": float(np.quantile(values, alpha)), "upper": float(np.quantile(values, 1.0 - alpha)), "valid_resamples": int(values.size)}


def ml_bootstrap(aggregated, protocol):
    """
    Calculate cluster-bootstrap intervals over independent base scenarios.

    Parameters
    ----------
    aggregated : pandas.DataFrame
        One ADR1D-ML row per base scenario.
    protocol : dict
        Locked bootstrap count, seed, and confidence level.

    Returns
    -------
    dict
        Percentile intervals for primary regression and classification metrics.

    """
    analysis    = protocol["statistical_analysis"]
    resamples   = int(analysis["bootstrap_resamples"])
    confidence  = float(analysis["confidence_level"])
    rng         = np.random.default_rng(int(analysis["bootstrap_seed"]))
    sample_size = len(aggregated)
    threshold   = ml_decision_threshold()
    collected = {key: [] for key in ("velocity_rmse_log10", "velocity_r2_log10", "velocity_median_ape", "dispersion_rmse_log10", "dispersion_r2_log10", "dispersion_median_ape", "decay_balanced_accuracy", "decay_roc_auc", "conditional_decay_rmse_log10", "conditional_decay_median_ape")}
    for _ in range(resamples):
        frame = aggregated.iloc[rng.integers(0, sample_size, size=sample_size)]
        velocity = regression_metrics(frame["actual_effective_velocity_m_s"], frame["predicted_effective_velocity_m_s"])
        dispersion = regression_metrics(frame["actual_effective_dispersion_m2_s"], frame["predicted_effective_dispersion_m2_s"])
        actual_class = frame["actual_decay_resolvable"].to_numpy(dtype=int)
        probability = frame["predicted_decay_resolvable_probability"].to_numpy(dtype=float)
        predicted_class = (probability >= threshold).astype(int)
        collected["velocity_rmse_log10"].append(velocity["rmse_log10"])
        collected["velocity_r2_log10"].append(velocity["r2_log10"])
        collected["velocity_median_ape"].append(velocity["median_absolute_percentage_error"])
        collected["dispersion_rmse_log10"].append(dispersion["rmse_log10"])
        collected["dispersion_r2_log10"].append(dispersion["r2_log10"])
        collected["dispersion_median_ape"].append(dispersion["median_absolute_percentage_error"])
        collected["decay_balanced_accuracy"].append(float(balanced_accuracy_score(actual_class, predicted_class)) if np.unique(actual_class).size == 2 else math.nan)
        collected["decay_roc_auc"].append(float(roc_auc_score(actual_class, probability)) if np.unique(actual_class).size == 2 else math.nan)
        mask = actual_class == 1
        if int(mask.sum()) >= 2:
            conditional = conditional_decay_metrics(frame["actual_decay_rate_s_1"], frame["predicted_decay_rate_if_resolvable_s_1"], mask)
            collected["conditional_decay_rmse_log10"].append(conditional["rmse_log10"])
            collected["conditional_decay_median_ape"].append(conditional["median_absolute_percentage_error"])
        else:
            collected["conditional_decay_rmse_log10"].append(math.nan)
            collected["conditional_decay_median_ape"].append(math.nan)
    return {key: bootstrap_interval(values, confidence) for key, values in collected.items()}


def ml_subgroups(aggregated, threshold):
    """
    Compute ADR1D-ML metrics within every pre-specified physical regime.

    Parameters
    ----------
    aggregated : pandas.DataFrame
        One prediction row per base scenario.
    threshold : float
        Frozen decay-resolvability probability threshold.

    Returns
    -------
    dict
        Regression and classification metrics by regime.

    """
    result = {}
    for regime, frame in aggregated.groupby("regime", sort=True):
        result[str(regime)] = {
            "base_scenarios": int(len(frame)),
            "effective_velocity": regression_metrics(frame["actual_effective_velocity_m_s"], frame["predicted_effective_velocity_m_s"]),
            "effective_dispersion": regression_metrics(frame["actual_effective_dispersion_m2_s"], frame["predicted_effective_dispersion_m2_s"]),
            "decay_resolvability": classification_metrics(frame["actual_decay_resolvable"], frame["predicted_decay_resolvable_probability"], threshold) if frame["actual_decay_resolvable"].nunique() == 2 else {"status": "not_estimable_single_truth_class", "rows": int(len(frame))},
        }
    return result


def evaluate_ml_metrics(predictions, protocol):
    """
    Calculate all locked ADR1D-ML metrics, intervals, and adequacy checks.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Complete replicate-level challenge predictions.
    protocol : dict
        Locked validation protocol.

    Returns
    -------
    dict
        Base-scenario metrics, repeatability, uncertainty, and criteria.

    """
    threshold  = ml_decision_threshold()
    aggregated = aggregate_ml_predictions(predictions, threshold)
    velocity   = regression_metrics(aggregated["actual_effective_velocity_m_s"], aggregated["predicted_effective_velocity_m_s"])
    dispersion = regression_metrics(aggregated["actual_effective_dispersion_m2_s"], aggregated["predicted_effective_dispersion_m2_s"])
    classification = classification_metrics(aggregated["actual_decay_resolvable"], aggregated["predicted_decay_resolvable_probability"], threshold)
    conditional = conditional_decay_metrics(aggregated["actual_decay_rate_s_1"], aggregated["predicted_decay_rate_if_resolvable_s_1"], aggregated["actual_decay_resolvable"].eq(1))
    physical = {
        "positive_velocity_fraction": float(np.mean(predictions["predicted_effective_velocity_m_s"] > 0.0)),
        "positive_dispersion_fraction": float(np.mean(predictions["predicted_effective_dispersion_m2_s"] > 0.0)),
        "probability_in_unit_interval_fraction": float(np.mean(predictions["predicted_decay_resolvable_probability"].between(0.0, 1.0))),
        "positive_conditional_decay_fraction": float(np.mean(predictions["predicted_decay_rate_if_resolvable_s_1"] > 0.0)),
    }
    criteria = protocol["minimum_adequacy_criteria"]["adr1d_ml"]
    checks = {
        "effective_velocity_median_absolute_percentage_error": velocity["median_absolute_percentage_error"] <= criteria["effective_velocity_median_absolute_percentage_error_max"],
        "effective_velocity_percentile_90_absolute_percentage_error": velocity["percentile_90_absolute_percentage_error"] <= criteria["effective_velocity_percentile_90_absolute_percentage_error_max"],
        "effective_velocity_r2_log10": velocity["r2_log10"] >= criteria["effective_velocity_r2_log10_min"],
        "effective_dispersion_median_absolute_percentage_error": dispersion["median_absolute_percentage_error"] <= criteria["effective_dispersion_median_absolute_percentage_error_max"],
        "effective_dispersion_percentile_90_absolute_percentage_error": dispersion["percentile_90_absolute_percentage_error"] <= criteria["effective_dispersion_percentile_90_absolute_percentage_error_max"],
        "effective_dispersion_r2_log10": dispersion["r2_log10"] >= criteria["effective_dispersion_r2_log10_min"],
        "decay_balanced_accuracy": classification["balanced_accuracy"] >= criteria["decay_balanced_accuracy_min"],
        "decay_roc_auc": classification["roc_auc"] >= criteria["decay_roc_auc_min"],
        "physical_validity": min(physical.values()) >= criteria["physical_validity_fraction_required"],
    }
    conditional_available = conditional["rows"] >= criteria["conditional_decay_minimum_resolvable_base_scenarios_for_gate"]
    conditional_checks = {
        "status": "evaluated" if conditional_available else "not_estimable_insufficient_resolvable_scenarios",
        "minimum_required_rows": int(criteria["conditional_decay_minimum_resolvable_base_scenarios_for_gate"]),
        "observed_rows": int(conditional["rows"]),
        "median_absolute_percentage_error": conditional["median_absolute_percentage_error"] <= criteria["conditional_decay_median_absolute_percentage_error_max"] if conditional_available else None,
        "r2_log10": conditional["r2_log10"] >= criteria["conditional_decay_r2_log10_min"] if conditional_available else None,
    }
    evaluated_checks = list(checks.values()) + [value for key, value in conditional_checks.items() if key in ("median_absolute_percentage_error", "r2_log10") and value is not None]
    outcome = "meets_all_pre_specified_criteria" if all(evaluated_checks) and conditional_available else "mixed_evidence"
    if evaluated_checks and not any(evaluated_checks):
        outcome = "does_not_meet_pre_specified_criteria"
    return {
        "aggregation_policy": "Geometric mean for positive regressions and arithmetic mean probability over five sensor-noise replicates; classification then uses the frozen 0.19 threshold.",
        "base_scenarios": int(len(aggregated)),
        "replicate_predictions": int(len(predictions)),
        "effective_velocity": velocity,
        "effective_dispersion": dispersion,
        "decay_resolvability": classification,
        "conditional_decay": conditional,
        "noise_repeatability": ml_repeatability(predictions, aggregated),
        "physical_checks": physical,
        "by_regime": ml_subgroups(aggregated, threshold),
        "confidence_intervals_95": ml_bootstrap(aggregated, protocol),
        "adequacy": {"overall_outcome": outcome, "checks": checks, "conditional_decay_checks": conditional_checks},
    }


def point_metrics(reference, prediction, active_threshold):
    """
    Compute pooled normalized-field errors for one set of point rows.

    Parameters
    ----------
    reference : array-like of float
        Analytical normalized concentrations.
    prediction : array-like of float
        Neural normalized concentrations.
    active_threshold : float
        Reference threshold defining active transport rows.

    Returns
    -------
    dict
        Pooled error, R2, active error, and boundedness metrics.

    """
    reference  = np.asarray(reference, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    residual   = prediction - reference
    active     = reference > active_threshold
    denominator = np.sum(np.square(reference - reference.mean()))
    return {
        "rows": int(reference.size),
        "active_rows": int(active.sum()),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "r2": float(1.0 - np.sum(np.square(residual)) / denominator) if denominator > 0.0 else None,
        "active_mae": float(np.mean(np.abs(residual[active]))) if active.any() else None,
        "active_rmse": float(np.sqrt(np.mean(np.square(residual[active])))) if active.any() else None,
        "maximum_absolute_error": float(np.max(np.abs(residual))),
        "negative_prediction_fraction": float(np.mean(prediction < 0.0)),
        "above_one_prediction_fraction": float(np.mean(prediction > 1.0)),
    }


def enrich_nn_predictions(raw_path):
    """
    Join public ADR1D-NN outputs with analytical truth and scenario metadata.

    Parameters
    ----------
    raw_path : pathlib.Path
        Temporary public-API output with 299,880 predictions.

    Returns
    -------
    pandas.DataFrame
        Complete point-level prediction evidence in locked field order.

    Raises
    ------
    EvaluationError
        If rows, identifiers, coordinates, or bounds differ from the input.

    """
    raw       = pd.read_csv(raw_path)
    field     = pd.read_csv(FIELD_PATH)
    scenarios = pd.read_csv(SCENARIO_PATH)
    require(len(raw) == len(field) == 299880, "ADR1D-NN output row count differs from the locked field")
    require(raw["scenario_id"].equals(field["scenario_id"]), "ADR1D-NN output scenario order differs")
    require(np.array_equal(raw["time_s"].to_numpy(dtype=float), field["time_s"].to_numpy(dtype=float)), "ADR1D-NN output time coordinates differ")
    require(np.array_equal(raw["x_m"].to_numpy(dtype=float), field["x_m"].to_numpy(dtype=float)), "ADR1D-NN output spatial coordinates differ")
    metadata = scenarios.loc[:, ["scenario_id", "peclet_number", "damkohler_number"]]
    result = field.loc[:, ["scenario_id", "split", "regime", "design_component", "time_s", "x_m", "normalized_concentration"]].rename(columns={"normalized_concentration": "reference_normalized_concentration"})
    result = result.merge(metadata, on="scenario_id", how="left", validate="many_to_one", sort=False)
    result["predicted_normalized_concentration"] = raw["predicted_normalized_concentration"].to_numpy(dtype=float)
    result["absolute_error"] = np.abs(result["predicted_normalized_concentration"] - result["reference_normalized_concentration"])
    result["constraint_applied"] = raw["constraint_applied"].astype(str).to_numpy()
    require(result["predicted_normalized_concentration"].between(0.0, 1.0).all(), "ADR1D-NN prediction lies outside [0, 1]")
    return result


def trapezoidal(values, coordinates, axis=-1):
    """
    Integrate sampled values with NumPy's current trapezoidal implementation.

    Parameters
    ----------
    values : numpy.ndarray
        Values sampled along one axis.
    coordinates : numpy.ndarray
        Strictly increasing coordinates for the integration axis.
    axis : int, optional
        Array axis corresponding to `coordinates`.

    Returns
    -------
    numpy.ndarray or float
        Composite trapezoidal integral.

    """
    return np.trapezoid(values, x=coordinates, axis=axis)


def nn_scenario_metrics(predictions, scenarios, active_threshold):
    """
    Calculate point, mass, arrival, and PDE diagnostics for every base scenario.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Complete point-level challenge evidence.
    scenarios : pandas.DataFrame
        Locked physical scenario metadata.
    active_threshold : float
        Reference threshold defining active transport and arrival.

    Returns
    -------
    tuple of pandas.DataFrame
        Public scenario table and internal sufficient-statistics table.

    """
    scenario_lookup = scenarios.set_index("scenario_id")
    public_rows = []
    internal_rows = []
    for scenario_id, frame in predictions.groupby("scenario_id", sort=True):
        frame = frame.sort_values(["time_s", "x_m"])
        scenario = scenario_lookup.loc[scenario_id]
        times = np.sort(frame["time_s"].unique().astype(float))
        positions = np.sort(frame["x_m"].unique().astype(float))
        reference = frame["reference_normalized_concentration"].to_numpy(dtype=float).reshape(len(times), len(positions))
        prediction = frame["predicted_normalized_concentration"].to_numpy(dtype=float).reshape(len(times), len(positions))
        metrics = point_metrics(reference.ravel(), prediction.ravel(), active_threshold)

        reference_mass = trapezoidal(reference, positions, axis=1)
        predicted_mass = trapezoidal(prediction, positions, axis=1)
        reference_exposure = float(trapezoidal(reference_mass, times))
        predicted_exposure = float(trapezoidal(predicted_mass, times))
        mass_relative_error = abs(predicted_exposure - reference_exposure) / reference_exposure if reference_exposure > 0.0 else math.nan

        arrival_errors = []
        missed_arrivals = 0
        for position_index in range(1, len(positions)):
            reference_active = np.flatnonzero(reference[:, position_index] > active_threshold)
            if reference_active.size == 0:
                continue
            predicted_active = np.flatnonzero(prediction[:, position_index] > active_threshold)
            reference_arrival = float(times[reference_active[0]])
            if predicted_active.size:
                predicted_arrival = float(times[predicted_active[0]])
            else:
                predicted_arrival = float(times[-1] + (times[1] - times[0]))
                missed_arrivals += 1
            arrival_errors.append(abs(predicted_arrival - reference_arrival))
        arrival_error = float(np.median(arrival_errors)) if arrival_errors else math.nan

        time_step = float(times[1] - times[0])
        space_step = float(positions[1] - positions[0])
        effective_velocity = float(scenario["velocity_m_s"] / scenario["retardation_factor"])
        effective_dispersion = float(scenario["dispersion_m2_s"] / scenario["retardation_factor"])
        decay_rate = float(scenario["decay_rate_s_1"])
        temporal_derivative = (prediction[2:, 1:-1] - prediction[:-2, 1:-1]) / (2.0 * time_step)
        spatial_derivative = (prediction[1:-1, 2:] - prediction[1:-1, :-2]) / (2.0 * space_step)
        second_derivative = (prediction[1:-1, 2:] - 2.0 * prediction[1:-1, 1:-1] + prediction[1:-1, :-2]) / space_step**2
        residual = temporal_derivative - effective_dispersion * second_derivative + effective_velocity * spatial_derivative + decay_rate * prediction[1:-1, 1:-1]
        pde_rmse = float(np.sqrt(np.mean(np.square(residual))))
        initial_error = float(np.max(np.abs(prediction[0, 1:] - reference[0, 1:])))
        inlet_error = float(np.max(np.abs(prediction[:, 0] - reference[:, 0])))

        public_rows.append(
            {
                "scenario_id": scenario_id,
                "regime": str(scenario["regime"]),
                "design_component": str(scenario["design_component"]),
                "peclet_number": float(scenario["peclet_number"]),
                "damkohler_number": float(scenario["damkohler_number"]),
                "rows": metrics["rows"],
                "active_rows": metrics["active_rows"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "active_mae": metrics["active_mae"],
                "active_rmse": metrics["active_rmse"],
                "maximum_absolute_error": metrics["maximum_absolute_error"],
                "integrated_mass_relative_error": mass_relative_error,
                "arrival_time_absolute_error_s": arrival_error,
                "arrival_positions": len(arrival_errors),
                "missed_arrival_positions": missed_arrivals,
                "pde_residual_rmse_s_1": pde_rmse,
                "initial_condition_maximum_absolute_error": initial_error,
                "inlet_boundary_maximum_absolute_error": inlet_error,
            }
        )
        flat_reference = reference.ravel()
        flat_prediction = prediction.ravel()
        active = flat_reference > active_threshold
        internal_rows.append(
            {
                "scenario_id": scenario_id,
                "rows": flat_reference.size,
                "sum_absolute_error": float(np.sum(np.abs(flat_prediction - flat_reference))),
                "sum_squared_error": float(np.sum(np.square(flat_prediction - flat_reference))),
                "sum_reference": float(np.sum(flat_reference)),
                "sum_squared_reference": float(np.sum(np.square(flat_reference))),
                "active_rows": int(active.sum()),
                "active_sum_absolute_error": float(np.sum(np.abs(flat_prediction[active] - flat_reference[active]))),
                "active_sum_squared_error": float(np.sum(np.square(flat_prediction[active] - flat_reference[active]))),
                "scenario_rmse": metrics["rmse"],
                "integrated_mass_relative_error": mass_relative_error,
                "arrival_time_absolute_error_s": arrival_error,
                "pde_residual_sum_squares": float(np.sum(np.square(residual))),
                "pde_residual_rows": int(residual.size),
            }
        )
    return pd.DataFrame(public_rows), pd.DataFrame(internal_rows)


def nn_group_metrics(predictions, scenario_groups, active_threshold):
    """
    Calculate pooled field metrics for named scenario subgroups.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Complete point-level predictions.
    scenario_groups : mapping of str to iterable of str
        Scenario identifiers assigned to each subgroup.
    active_threshold : float
        Reference threshold defining active rows.

    Returns
    -------
    dict
        Pooled point metrics and base-scenario counts by group.

    """
    result = {}
    for label, identifiers in scenario_groups.items():
        identifiers = set(identifiers)
        frame = predictions.loc[predictions["scenario_id"].isin(identifiers)]
        result[str(label)] = {
            "base_scenarios": len(identifiers),
            **point_metrics(frame["reference_normalized_concentration"], frame["predicted_normalized_concentration"], active_threshold),
        }
    return result


def nn_subgroups(predictions, scenarios, active_threshold, damkohler_threshold):
    """
    Compute all pre-specified ADR1D-NN subgroup metrics.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Complete point-level predictions.
    scenarios : pandas.DataFrame
        Locked physical scenario metadata.
    active_threshold : float
        Reference concentration threshold.
    damkohler_threshold : float
        Locked decay-resolvability threshold.

    Returns
    -------
    dict
        Metrics by regime, design component, Peclet quartile, decay state, and
        active or inactive field region.

    """
    regime_groups = {str(label): frame["scenario_id"].tolist() for label, frame in scenarios.groupby("regime", sort=True)}
    design_groups = {str(label): frame["scenario_id"].tolist() for label, frame in scenarios.groupby("design_component", sort=True)}
    quartile_labels = pd.qcut(scenarios["peclet_number"].rank(method="first"), 4, labels=("Q1", "Q2", "Q3", "Q4"))
    quartile_groups = {str(label): scenarios.loc[quartile_labels.eq(label), "scenario_id"].tolist() for label in quartile_labels.cat.categories}
    decay_state = np.select([scenarios["decay_rate_s_1"].eq(0.0), scenarios["damkohler_number"].ge(damkohler_threshold)], ["zero", "resolvable"], default="below_resolution")
    decay_groups = {label: scenarios.loc[decay_state == label, "scenario_id"].tolist() for label in ("zero", "below_resolution", "resolvable")}
    active = predictions["reference_normalized_concentration"] > active_threshold
    regions = {}
    for label, mask in (("active", active), ("inactive", ~active)):
        frame = predictions.loc[mask]
        reference = frame["reference_normalized_concentration"].to_numpy(dtype=float)
        predicted = frame["predicted_normalized_concentration"].to_numpy(dtype=float)
        residual = predicted - reference
        regions[label] = {"rows": int(len(frame)), "mae": float(np.mean(np.abs(residual))), "rmse": float(np.sqrt(np.mean(np.square(residual)))), "maximum_absolute_error": float(np.max(np.abs(residual)))}
    return {
        "regime": nn_group_metrics(predictions, regime_groups, active_threshold),
        "design_component": nn_group_metrics(predictions, design_groups, active_threshold),
        "peclet_quartile": nn_group_metrics(predictions, quartile_groups, active_threshold),
        "damkohler_state": nn_group_metrics(predictions, decay_groups, active_threshold),
        "field_region": regions,
    }


def nn_bootstrap(internal, protocol):
    """
    Calculate vectorized cluster-bootstrap intervals over base scenarios.

    Parameters
    ----------
    internal : pandas.DataFrame
        One row of sufficient statistics per scenario.
    protocol : dict
        Locked bootstrap count, seed, and confidence level.

    Returns
    -------
    dict
        Percentile intervals for pooled and scenario-level neural metrics.

    """
    analysis   = protocol["statistical_analysis"]
    resamples  = int(analysis["bootstrap_resamples"])
    confidence = float(analysis["confidence_level"])
    rng        = np.random.default_rng(int(analysis["bootstrap_seed"]))
    indexes    = rng.integers(0, len(internal), size=(resamples, len(internal)))

    def sampled_sum(column):
        """Sum one sufficient-statistic column over every bootstrap sample."""
        values = internal[column].to_numpy(dtype=float)
        return values[indexes].sum(axis=1)

    rows       = sampled_sum("rows")
    absolute   = sampled_sum("sum_absolute_error")
    squared    = sampled_sum("sum_squared_error")
    reference  = sampled_sum("sum_reference")
    reference2 = sampled_sum("sum_squared_reference")
    active_rows = sampled_sum("active_rows")
    active_squared = sampled_sum("active_sum_squared_error")
    sst = reference2 - np.square(reference) / rows
    values = {
        "pooled_mae": absolute / rows,
        "pooled_rmse": np.sqrt(squared / rows),
        "pooled_r2": 1.0 - squared / sst,
        "active_rmse": np.sqrt(active_squared / active_rows),
        "percentile_90_scenario_rmse": np.quantile(internal["scenario_rmse"].to_numpy(dtype=float)[indexes], 0.90, axis=1),
        "median_integrated_mass_relative_error": np.nanmedian(internal["integrated_mass_relative_error"].to_numpy(dtype=float)[indexes], axis=1),
        "percentile_90_integrated_mass_relative_error": np.nanquantile(internal["integrated_mass_relative_error"].to_numpy(dtype=float)[indexes], 0.90, axis=1),
        "median_arrival_time_absolute_error_s": np.nanmedian(internal["arrival_time_absolute_error_s"].to_numpy(dtype=float)[indexes], axis=1),
        "percentile_90_arrival_time_absolute_error_s": np.nanquantile(internal["arrival_time_absolute_error_s"].to_numpy(dtype=float)[indexes], 0.90, axis=1),
    }
    return {key: bootstrap_interval(value, confidence) for key, value in values.items()}


def evaluate_nn_metrics(predictions, scenarios, scenario_metrics, internal, protocol):
    """
    Calculate all locked ADR1D-NN metrics, intervals, and adequacy checks.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Complete point-level challenge predictions.
    scenarios : pandas.DataFrame
        Locked physical scenario metadata.
    scenario_metrics : pandas.DataFrame
        Public scenario diagnostics.
    internal : pandas.DataFrame
        Sufficient statistics for cluster bootstrap.
    protocol : dict
        Locked validation protocol.

    Returns
    -------
    dict
        Pooled, scenario, physical, subgroup, uncertainty, and adequacy evidence.

    """
    active_threshold = float(protocol["metrics"]["active_normalized_concentration_threshold"])
    damkohler_threshold = float(protocol["challenge_design"]["decay_resolvability"]["damkohler_threshold"])
    field = point_metrics(predictions["reference_normalized_concentration"], predictions["predicted_normalized_concentration"], active_threshold)
    mass = scenario_metrics["integrated_mass_relative_error"].to_numpy(dtype=float)
    arrival = scenario_metrics["arrival_time_absolute_error_s"].to_numpy(dtype=float)
    pde_sum = float(internal["pde_residual_sum_squares"].sum())
    pde_rows = int(internal["pde_residual_rows"].sum())
    physical = {
        "initial_condition_maximum_absolute_error": float(scenario_metrics["initial_condition_maximum_absolute_error"].max()),
        "inlet_boundary_maximum_absolute_error": float(scenario_metrics["inlet_boundary_maximum_absolute_error"].max()),
        "negative_prediction_fraction": field["negative_prediction_fraction"],
        "above_one_prediction_fraction": field["above_one_prediction_fraction"],
        "median_integrated_mass_relative_error": float(np.nanmedian(mass)),
        "percentile_90_integrated_mass_relative_error": float(np.nanquantile(mass, 0.90)),
        "median_arrival_time_absolute_error_s": float(np.nanmedian(arrival)),
        "percentile_90_arrival_time_absolute_error_s": float(np.nanquantile(arrival, 0.90)),
        "arrival_estimable_scenarios": int(np.isfinite(arrival).sum()),
        "pde_residual_rmse_s_1": float(math.sqrt(pde_sum / pde_rows)),
        "pde_residual_rows": pde_rows,
    }
    distribution = {
        "median_scenario_rmse": float(scenario_metrics["rmse"].median()),
        "percentile_90_scenario_rmse": float(scenario_metrics["rmse"].quantile(0.90)),
        "maximum_scenario_rmse": float(scenario_metrics["rmse"].max()),
    }
    criteria = protocol["minimum_adequacy_criteria"]["adr1d_nn"]
    checks = {
        "pooled_rmse": field["rmse"] <= criteria["pooled_rmse_max"],
        "active_rmse": field["active_rmse"] <= criteria["active_rmse_max"],
        "r2": field["r2"] >= criteria["r2_min"],
        "percentile_90_scenario_rmse": distribution["percentile_90_scenario_rmse"] <= criteria["percentile_90_scenario_rmse_max"],
        "initial_condition_maximum_absolute_error": physical["initial_condition_maximum_absolute_error"] <= criteria["initial_condition_maximum_absolute_error_max"],
        "inlet_boundary_maximum_absolute_error": physical["inlet_boundary_maximum_absolute_error"] <= criteria["inlet_boundary_maximum_absolute_error_max"],
        "negative_prediction_fraction": physical["negative_prediction_fraction"] <= criteria["negative_prediction_fraction_max"],
        "above_one_prediction_fraction": physical["above_one_prediction_fraction"] <= criteria["above_one_prediction_fraction_max"],
        "median_integrated_mass_relative_error": physical["median_integrated_mass_relative_error"] <= criteria["median_integrated_mass_relative_error_max"],
        "percentile_90_integrated_mass_relative_error": physical["percentile_90_integrated_mass_relative_error"] <= criteria["percentile_90_integrated_mass_relative_error_max"],
        "median_arrival_time_absolute_error_s": physical["median_arrival_time_absolute_error_s"] <= criteria["median_arrival_time_absolute_error_s_max"],
        "percentile_90_arrival_time_absolute_error_s": physical["percentile_90_arrival_time_absolute_error_s"] <= criteria["percentile_90_arrival_time_absolute_error_s_max"],
    }
    if all(checks.values()):
        outcome = "meets_all_pre_specified_criteria"
    elif any(checks.values()):
        outcome = "mixed_evidence"
    else:
        outcome = "does_not_meet_pre_specified_criteria"
    return {
        "field": field,
        "scenario_distribution": distribution,
        "physical": physical,
        "metric_definitions": {
            "integrated_mass_relative_error": "Absolute relative error in the time integral of the spatially integrated normalized concentration for each scenario.",
            "arrival_time_absolute_error_s": "Median absolute first-threshold-crossing error over interior grid positions where the analytical field exceeds 1e-4; a missed prediction is assigned final_time + time_step.",
            "pde_residual_diagnostic": "Centered finite-difference RMSE of dC/dt - D_eff*d2C/dx2 + v_eff*dC/dx + lambda*C over interior space-time nodes.",
        },
        "subgroups": nn_subgroups(predictions, scenarios, active_threshold, damkohler_threshold),
        "confidence_intervals_95": nn_bootstrap(internal, protocol),
        "adequacy": {"overall_outcome": outcome, "checks": checks},
    }


def write_csv(table, path):
    """
    Write a deterministic challenge result CSV.

    Parameters
    ----------
    table : pandas.DataFrame
        Ordered result table.
    path : pathlib.Path
        Destination path inside the staging directory.

    Returns
    -------
    dict
        Rows, columns, bytes, and SHA-256 digest.

    """
    table.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")
    return {"rows": int(len(table)), "columns": int(len(table.columns)), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(content, path):
    """
    Write a deterministic machine-readable challenge result.

    Parameters
    ----------
    content : dict
        JSON-compatible result object.
    path : pathlib.Path
        Destination path inside the staging directory.

    Returns
    -------
    dict
        Bytes and SHA-256 digest after writing.

    """
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def promote_outputs(staging_paths, protocol):
    """
    Atomically move fully validated staging outputs to protocol-defined paths.

    Parameters
    ----------
    staging_paths : mapping of str to pathlib.Path
        Completed staging artifacts keyed by protocol output name.
    protocol : dict
        Locked protocol containing destination paths.

    Returns
    -------
    dict
        Final path and digest for every promoted artifact.

    """
    promoted = {}
    for key, staging_path in staging_paths.items():
        destination = preparation.project_path(protocol["planned_artifacts"][key])
        require(not destination.exists(), f"Refusing to overwrite challenge output: {destination}")
        staging_path.replace(destination)
        promoted[key] = {"path": protocol["planned_artifacts"][key], "bytes": destination.stat().st_size, "sha256": sha256(destination)}
    return promoted


def write_incident(error, run_state):
    """
    Persist failure evidence after any challenge inference has begun.

    Parameters
    ----------
    error : Exception
        Exception that interrupted evaluation.
    run_state : dict
        Attempt and completion flags for both public APIs.

    Returns
    -------
    None
        An incident JSON is written while staging files are retained.

    """
    staging_files = {}
    if STAGING_DIRECTORY.exists():
        for path in sorted(STAGING_DIRECTORY.iterdir()):
            if path.is_file():
                staging_files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    incident = {
        "status": "challenge_inference_interrupted_no_automatic_rerun",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "run_state": run_state,
        "staging_files": staging_files,
        "traceback": traceback.format_exc(),
    }
    INCIDENT_PATH.write_text(json.dumps(incident, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def execute_evaluation():
    """
    Execute and persist the one permitted challenge inference for each model.

    Returns
    -------
    dict
        Final status, output records, and adequacy outcomes.

    Raises
    ------
    EvaluationError
        If any locked input, public API, result, metric, or output contract fails.

    """
    protocol = load_json(PROTOCOL_PATH)
    lock     = verify_evaluation_lock(protocol)
    STAGING_DIRECTORY.mkdir(parents=False, exist_ok=False)
    prepared = preparation.prepare_input_files(protocol, STAGING_DIRECTORY)
    prepared_metadata = preparation.comparable_metadata(prepared)
    require(prepared_metadata == lock["prepared_inputs"], "Rebuilt model inputs differ from the evaluation lock")

    ml_raw = STAGING_DIRECTORY / ML_RAW_NAME
    nn_raw = STAGING_DIRECTORY / NN_RAW_NAME
    run_state = {"adr1d_ml_attempted": False, "adr1d_ml_completed": False, "adr1d_nn_attempted": False, "adr1d_nn_completed": False}
    try:
        run_state["adr1d_ml_attempted"] = True
        ml_console = run_api([sys.executable, "-B", str(ML_API), "--input-csv", str(prepared["adr1d_ml"]["path"]), "--output-csv", str(ml_raw), "--model", str(ML_MODEL), "--manifest", str(ML_MANIFEST)], "ADR1D-ML")
        run_state["adr1d_ml_completed"] = True
        run_state["adr1d_nn_attempted"] = True
        nn_console = run_api([sys.executable, "-B", str(NN_API), "--input-csv", str(prepared["adr1d_nn"]["path"]), "--output-csv", str(nn_raw), "--model", str(NN_MODEL), "--manifest", str(NN_MANIFEST), "--device", "cpu"], "ADR1D-NN")
        run_state["adr1d_nn_completed"] = True

        ml_predictions = enrich_ml_predictions(ml_raw, protocol)
        nn_predictions = enrich_nn_predictions(nn_raw)
        scenarios      = pd.read_csv(SCENARIO_PATH)
        active_threshold = float(protocol["metrics"]["active_normalized_concentration_threshold"])
        nn_scenarios, nn_internal = nn_scenario_metrics(nn_predictions, scenarios, active_threshold)
        ml_evidence = evaluate_ml_metrics(ml_predictions, protocol)
        nn_evidence = evaluate_nn_metrics(nn_predictions, scenarios, nn_scenarios, nn_internal, protocol)

        ml_final_path = STAGING_DIRECTORY / ML_FINAL_NAME
        nn_final_path = STAGING_DIRECTORY / NN_FINAL_NAME
        nn_scenario_path = STAGING_DIRECTORY / NN_SCENARIOS_NAME
        ml_prediction_record = write_csv(ml_predictions, ml_final_path)
        nn_prediction_record = write_csv(nn_predictions, nn_final_path)
        nn_scenario_record = write_csv(nn_scenarios, nn_scenario_path)

        ml_metrics = {
            "status": "challenge_evaluation_complete",
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "evaluation_lock_sha256": sha256(EVALUATION_LOCK),
            "model_sha256": sha256(ML_MODEL),
            "inference_module_sha256": sha256(ML_API),
            "prepared_input": prepared_metadata["adr1d_ml"],
            "api_raw_output_sha256": sha256(ml_raw),
            "predictions": ml_prediction_record,
            "inference": {"persisted_challenge_runs": 1, "public_api_return_code": ml_console["return_code"], "post_challenge_tuning_performed": False},
            "evidence": ml_evidence,
        }
        nn_metrics = {
            "status": "challenge_evaluation_complete",
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "evaluation_lock_sha256": sha256(EVALUATION_LOCK),
            "model_sha256": sha256(NN_MODEL),
            "inference_module_sha256": sha256(NN_API),
            "prepared_input": prepared_metadata["adr1d_nn"],
            "api_raw_output_sha256": sha256(nn_raw),
            "predictions": nn_prediction_record,
            "scenario_metrics": nn_scenario_record,
            "inference": {"persisted_challenge_runs": 1, "device": "cpu", "public_api_return_code": nn_console["return_code"], "post_challenge_tuning_performed": False},
            "evidence": nn_evidence,
        }
        ml_metrics_path = STAGING_DIRECTORY / ML_METRICS_NAME
        nn_metrics_path = STAGING_DIRECTORY / NN_METRICS_NAME
        write_json(ml_metrics, ml_metrics_path)
        write_json(nn_metrics, nn_metrics_path)

        staging_paths = {
            "adr1d_ml_predictions": ml_final_path,
            "adr1d_ml_metrics": ml_metrics_path,
            "adr1d_nn_predictions": nn_final_path,
            "adr1d_nn_scenario_metrics": nn_scenario_path,
            "adr1d_nn_metrics": nn_metrics_path,
        }
        promoted = promote_outputs(staging_paths, protocol)
        shutil.rmtree(STAGING_DIRECTORY)
        return {
            "status": "challenge_evaluation_complete",
            "outputs": promoted,
            "adr1d_ml_outcome": ml_evidence["adequacy"]["overall_outcome"],
            "adr1d_nn_outcome": nn_evidence["adequacy"]["overall_outcome"],
            "adr1d_ml_challenge_inference_runs": 1,
            "adr1d_nn_challenge_inference_runs": 1,
            "post_challenge_tuning_performed": False,
        }
    except Exception as error:
        if run_state["adr1d_ml_attempted"] or run_state["adr1d_nn_attempted"]:
            write_incident(error, run_state)
        elif STAGING_DIRECTORY.exists():
            shutil.rmtree(STAGING_DIRECTORY)
        raise


def main():
    """
    Run the locked challenge evaluation and print a compact completion record.

    Returns
    -------
    None
        Five protocol-defined evaluation artifacts are written to `results/`.

    """
    result = execute_evaluation()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
