"""
================================================================================
ADR1D Validation: Independent Challenge-Result Audit
================================================================================

Independently audit the persisted ADR1D-ML and ADR1D-NN challenge results
without loading either model or invoking either public inference interface.

Main Operations
---------------
1. Verify the complete protocol, challenge, code, model, and result hash chain.
2. Recalculate ADR1D-ML metrics from the persisted replicate predictions.
3. Recalculate ADR1D-NN point, scenario, physical, and subgroup diagnostics.
4. Reproduce both scenario-cluster bootstrap analyses from the locked seed.
5. Persist a deterministic machine-readable audit report.

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
import hashlib
import json
import math
from pathlib import Path

# Third-party libraries
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, r2_score, recall_score, roc_auc_score


ROOT                    = Path(__file__).resolve().parents[1]
PROTOCOL_PATH           = ROOT / "configs/validation_protocol.json"
CHALLENGE_LOCK_PATH     = ROOT / "results/challenge_lock.json"
EVALUATION_LOCK_PATH    = ROOT / "results/challenge_evaluation_lock.json"
CANONICAL_AUDIT_PATH    = ROOT / "results/canonical_reproduction.json"
SCENARIO_PATH           = ROOT / "data/challenge_scenarios.csv"
FIELD_PATH              = ROOT / "data/challenge_analytical_field.csv"
SENSOR_PATH             = ROOT / "data/challenge_sensor_observations.csv"
ML_PREDICTIONS_PATH     = ROOT / "results/adr1d_ml_challenge_predictions.csv"
ML_METRICS_PATH         = ROOT / "results/adr1d_ml_challenge_metrics.json"
NN_PREDICTIONS_PATH     = ROOT / "results/adr1d_nn_challenge_predictions.csv"
NN_SCENARIOS_PATH       = ROOT / "results/adr1d_nn_challenge_scenarios.csv"
NN_METRICS_PATH         = ROOT / "results/adr1d_nn_challenge_metrics.json"
VALIDATION_PATH         = ROOT / "results/challenge_result_validation.json"
PREPARATION_PATH        = ROOT / "src/prepare_challenge_evaluation.py"
EVALUATOR_PATH          = ROOT / "src/evaluate_locked_challenge.py"
ML_PROTOCOL_PATH        = ROOT.parent / "03_modelos_ml_parametros/results/final_model_protocol.json"
ABSOLUTE_TOLERANCE      = 2.0e-9
RELATIVE_TOLERANCE      = 2.0e-8


class ValidationError(RuntimeError):
    """Represent a failed challenge-result integrity or numerical check."""


class Audit:
    """
    Record independent validation checks and numerical comparison precision.

    Attributes
    ----------
    checks : int
        Number of successful logical, structural, or numerical checks.
    numerical_comparisons : int
        Number of floating-point or integer values compared.
    maximum_absolute_difference : float
        Largest absolute difference observed in a finite numerical comparison.

    """

    def __init__(self):
        """Initialize empty validation counters."""
        self.checks                      = 0
        self.numerical_comparisons       = 0
        self.maximum_absolute_difference = 0.0

    def require(self, condition, label):
        """
        Require one logical condition and increment the audit counter.

        Parameters
        ----------
        condition : bool
            Condition that must evaluate to true.
        label : str
            Actionable description included in a failure message.

        Returns
        -------
        None
            The method returns after recording a successful check.

        Raises
        ------
        ValidationError
            If the condition is false.

        """
        if not bool(condition):
            raise ValidationError(label)
        self.checks += 1

    def equal(self, actual, expected, label):
        """
        Require exact equality for a non-floating validation value.

        Parameters
        ----------
        actual : object
            Independently observed value.
        expected : object
            Locked or reported value.
        label : str
            Comparison path included in a failure message.

        Returns
        -------
        None
            The method returns after recording a successful check.

        """
        self.require(actual == expected, f"{label}: observed {actual!r}, expected {expected!r}")

    def close(self, actual, expected, label):
        """
        Compare two finite numerical values under documented audit tolerances.

        Parameters
        ----------
        actual : float
            Value recalculated from persisted predictions.
        expected : float
            Value stored in the published metric artifact.
        label : str
            Hierarchical metric name included in a failure message.

        Returns
        -------
        None
            The method records the comparison and its absolute difference.

        """
        actual   = float(actual)
        expected = float(expected)
        difference = abs(actual - expected)
        self.maximum_absolute_difference = max(self.maximum_absolute_difference, difference)
        self.numerical_comparisons += 1
        self.require(math.isclose(actual, expected, rel_tol=RELATIVE_TOLERANCE, abs_tol=ABSOLUTE_TOLERANCE), f"{label}: observed {actual:.17g}, expected {expected:.17g}")

    def compare_tree(self, actual, expected, label):
        """
        Recursively compare a recalculated evidence tree with reported values.

        Parameters
        ----------
        actual : object
            Independently reconstructed mapping, sequence, scalar, or null.
        expected : object
            Corresponding value from a persisted metrics JSON.
        label : str
            Root path used to identify any discrepancy.

        Returns
        -------
        None
            Every leaf and container is checked recursively.

        """
        if isinstance(expected, dict):
            self.equal(set(actual), set(expected), f"{label} keys")
            for key in expected:
                self.compare_tree(actual[key], expected[key], f"{label}.{key}")
            return
        if isinstance(expected, list):
            self.equal(len(actual), len(expected), f"{label} length")
            for index, expected_value in enumerate(expected):
                self.compare_tree(actual[index], expected_value, f"{label}[{index}]")
            return
        if isinstance(expected, bool) or expected is None or isinstance(expected, str):
            self.equal(actual, expected, label)
            return
        if isinstance(expected, (int, float)):
            self.close(actual, expected, label)
            return
        self.equal(actual, expected, label)


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
    ValidationError
        If the top-level value is not an object.

    """
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValidationError(f"Expected a JSON object in {path}")
    return content


def file_record(path, table=None):
    """
    Describe one validated artifact with optional tabular dimensions.

    Parameters
    ----------
    path : pathlib.Path
        Existing artifact inside the Activity 05 directory.
    table : pandas.DataFrame, optional
        Loaded table used to add row and column counts.

    Returns
    -------
    dict
        Relative path, byte count, digest, and optional table dimensions.

    """
    record = {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
    if table is not None:
        record.update({"rows": int(len(table)), "columns": int(len(table.columns))})
    return record


def verify_frozen_artifacts(protocol, audit):
    """
    Verify every path-and-digest pair frozen by the validation protocol.

    Parameters
    ----------
    protocol : dict
        Locked Activity 05 validation protocol.
    audit : Audit
        Mutable independent audit recorder.

    Returns
    -------
    int
        Number of frozen path-and-digest records verified.

    """
    verified = 0

    def visit(node, label):
        """Recursively locate and verify frozen artifact records."""
        nonlocal verified
        if isinstance(node, dict) and "path" in node and "sha256" in node:
            path = ROOT / node["path"]
            audit.require(path.is_file(), f"Frozen artifact is absent: {label}")
            audit.equal(sha256(path), node["sha256"], f"Frozen artifact digest {label}")
            verified += 1
            return
        if isinstance(node, dict):
            for key, value in node.items():
                visit(value, f"{label}.{key}")

    visit(protocol["frozen_inputs"], "frozen_inputs")
    return verified


def verify_hash_chain(protocol, challenge_lock, evaluation_lock, ml_metrics, nn_metrics, audit):
    """
    Verify locks, protected code, models, data, and persisted output metadata.

    Parameters
    ----------
    protocol : dict
        Locked Activity 05 protocol.
    challenge_lock : dict
        Lock created before any challenge inference.
    evaluation_lock : dict
        Lock created after exact input preparation and before inference.
    ml_metrics : dict
        Persisted ADR1D-ML challenge metrics.
    nn_metrics : dict
        Persisted ADR1D-NN challenge metrics.
    audit : Audit
        Mutable independent audit recorder.

    Returns
    -------
    dict
        Counts and digests describing the verified chain.

    """
    protocol_hash   = sha256(PROTOCOL_PATH)
    challenge_hash  = sha256(CHALLENGE_LOCK_PATH)
    evaluation_hash = sha256(EVALUATION_LOCK_PATH)
    canonical_hash  = sha256(CANONICAL_AUDIT_PATH)
    audit.equal(protocol_hash, challenge_lock["protocol"]["sha256"], "Challenge-lock protocol digest")
    audit.equal(protocol_hash, evaluation_lock["protocol"]["sha256"], "Evaluation-lock protocol digest")
    audit.equal(challenge_hash, evaluation_lock["challenge_guard"]["challenge_lock_sha256"], "Evaluation-lock challenge digest")
    audit.equal(canonical_hash, evaluation_lock["challenge_guard"]["canonical_reproduction_sha256"], "Evaluation-lock canonical-audit digest")
    audit.equal(sha256(PREPARATION_PATH), evaluation_lock["locked_code"]["preparation"]["sha256"], "Locked preparation-code digest")
    audit.equal(sha256(EVALUATOR_PATH), evaluation_lock["locked_code"]["evaluation"]["sha256"], "Locked evaluator-code digest")
    audit.equal(challenge_lock["status"], "locked_before_model_inference", "Challenge-lock status")
    audit.equal(evaluation_lock["status"], "locked_before_challenge_inference", "Evaluation-lock status")
    audit.equal(evaluation_lock["state_at_lock"]["adr1d_ml_challenge_inference_runs"], 0, "ADR1D-ML runs at evaluation lock")
    audit.equal(evaluation_lock["state_at_lock"]["adr1d_nn_challenge_inference_runs"], 0, "ADR1D-NN runs at evaluation lock")
    audit.equal(evaluation_lock["state_at_lock"]["acceptance_criteria_inspected_against_challenge_results"], False, "Criteria-inspection state at evaluation lock")
    frozen_count = verify_frozen_artifacts(protocol, audit)

    for key, record in challenge_lock["challenge_files"].items():
        path = ROOT / record["path"]
        audit.equal(sha256(path), record["sha256"], f"Challenge data digest {key}")
        audit.equal(path.stat().st_size, record["bytes"], f"Challenge data bytes {key}")
    audit.equal(sha256(ROOT / challenge_lock["challenge_manifest"]["path"]), challenge_lock["challenge_manifest"]["sha256"], "Challenge-manifest digest")
    audit.equal(sha256(ROOT / challenge_lock["independent_validation"]["path"]), challenge_lock["independent_validation"]["sha256"], "Challenge-set validation digest")

    for label, metrics, predictions_path in (("adr1d_ml", ml_metrics, ML_PREDICTIONS_PATH), ("adr1d_nn", nn_metrics, NN_PREDICTIONS_PATH)):
        audit.equal(metrics["status"], "challenge_evaluation_complete", f"{label} result status")
        audit.equal(metrics["protocol_sha256"], protocol_hash, f"{label} protocol digest")
        audit.equal(metrics["evaluation_lock_sha256"], evaluation_hash, f"{label} evaluation-lock digest")
        audit.equal(metrics["predictions"]["sha256"], sha256(predictions_path), f"{label} prediction digest")
        audit.equal(metrics["predictions"]["bytes"], predictions_path.stat().st_size, f"{label} prediction bytes")
        audit.equal(metrics["prepared_input"], evaluation_lock["prepared_inputs"][label], f"{label} prepared-input record")
        audit.equal(metrics["inference"]["persisted_challenge_runs"], 1, f"{label} persisted inference runs")
        audit.equal(metrics["inference"]["public_api_return_code"], 0, f"{label} API return code")
        audit.equal(metrics["inference"]["post_challenge_tuning_performed"], False, f"{label} post-challenge tuning state")
        audit.equal(metrics["model_sha256"], evaluation_lock["frozen_models"][label]["model_sha256"], f"{label} model digest")
        audit.equal(metrics["inference_module_sha256"], evaluation_lock["frozen_models"][label]["inference_module_sha256"], f"{label} inference-module digest")

    audit.equal(nn_metrics["scenario_metrics"]["sha256"], sha256(NN_SCENARIOS_PATH), "ADR1D-NN scenario-table digest")
    audit.equal(nn_metrics["scenario_metrics"]["bytes"], NN_SCENARIOS_PATH.stat().st_size, "ADR1D-NN scenario-table bytes")
    audit.require(not (ROOT / "results/challenge_inference_incident.json").exists(), "Unexpected challenge-inference incident record")
    audit.require(not (ROOT / "results/.challenge_inference_staging").exists(), "Unexpected challenge-inference staging directory")
    return {"frozen_artifacts_verified": frozen_count, "protocol_sha256": protocol_hash, "challenge_lock_sha256": challenge_hash, "evaluation_lock_sha256": evaluation_hash, "canonical_reproduction_sha256": canonical_hash}


def regression_metrics(actual, predicted):
    """
    Recalculate logarithmic and relative regression metrics.

    Parameters
    ----------
    actual : array-like of float
        Positive reference values.
    predicted : array-like of float
        Positive predictions in matching order.

    Returns
    -------
    dict
        Row count, log-space errors, R2, and relative-error summaries.

    """
    actual        = np.asarray(actual, dtype=float)
    predicted     = np.asarray(predicted, dtype=float)
    actual_log    = np.log10(actual)
    predicted_log = np.log10(predicted)
    residual      = predicted_log - actual_log
    relative      = np.abs(predicted - actual) / actual
    return {"rows": int(actual.size), "mae_log10": float(np.mean(np.abs(residual))), "rmse_log10": float(np.sqrt(np.mean(np.square(residual)))), "r2_log10": float(r2_score(actual_log, predicted_log)), "median_absolute_percentage_error": float(np.median(relative)), "percentile_90_absolute_percentage_error": float(np.quantile(relative, 0.90))}


def classification_metrics(actual, probability, threshold):
    """
    Recalculate binary decay-resolvability metrics at the frozen threshold.

    Parameters
    ----------
    actual : array-like of int
        Binary truth labels for independent base scenarios.
    probability : array-like of float
        Mean predicted probabilities for the same scenarios.
    threshold : float
        Frozen classification threshold.

    Returns
    -------
    dict
        Classification scores and ordered confusion-matrix entries.

    """
    actual      = np.asarray(actual, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predicted   = (probability >= threshold).astype(int)
    matrix      = confusion_matrix(actual, predicted, labels=[0, 1]).ravel().astype(int)
    return {"rows": int(actual.size), "positive_truth_rows": int(actual.sum()), "decision_threshold": float(threshold), "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)), "precision": float(precision_score(actual, predicted, zero_division=0)), "recall": float(recall_score(actual, predicted, zero_division=0)), "f1": float(f1_score(actual, predicted, zero_division=0)), "roc_auc": float(roc_auc_score(actual, probability)), "log_loss": float(log_loss(actual, np.column_stack((1.0 - probability, probability)), labels=[0, 1])), "confusion_matrix_tn_fp_fn_tp": matrix.tolist()}


def conditional_decay_metrics(actual, predicted, resolvable):
    """
    Recalculate conditional decay regression for truth-resolvable scenarios.

    Parameters
    ----------
    actual : array-like of float
        True first-order decay rates in inverse seconds.
    predicted : array-like of float
        Positive conditional predictions in inverse seconds.
    resolvable : array-like of bool
        Truth labels selecting the estimable subset.

    Returns
    -------
    dict
        Conditional log-space errors, R2, and median relative error.

    """
    mask          = np.asarray(resolvable, dtype=bool)
    actual        = np.asarray(actual, dtype=float)[mask]
    predicted     = np.asarray(predicted, dtype=float)[mask]
    actual_log    = np.log10(actual)
    predicted_log = np.log10(predicted)
    residual      = predicted_log - actual_log
    return {"rows": int(actual.size), "mae_log10": float(np.mean(np.abs(residual))), "rmse_log10": float(np.sqrt(np.mean(np.square(residual)))), "r2_log10": float(r2_score(actual_log, predicted_log)) if actual.size >= 2 else None, "median_absolute_percentage_error": float(np.median(np.abs(predicted - actual) / actual))}


def summarize_distribution(values):
    """
    Summarize a finite diagnostic distribution.

    Parameters
    ----------
    values : array-like of float
        Values from independent base scenarios.

    Returns
    -------
    dict
        Finite row count, median, 90th percentile, and maximum.

    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {"rows": int(values.size), "median": float(np.median(values)), "percentile_90": float(np.quantile(values, 0.90)), "maximum": float(np.max(values))}


def bootstrap_interval(values, confidence_level):
    """
    Calculate a central percentile interval from finite bootstrap estimates.

    Parameters
    ----------
    values : array-like of float
        Bootstrap estimates, possibly including non-finite draws.
    confidence_level : float
        Requested central interval probability.

    Returns
    -------
    dict
        Lower bound, upper bound, and finite-resample count.

    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    alpha  = (1.0 - confidence_level) / 2.0
    return {"lower": float(np.quantile(values, alpha)), "upper": float(np.quantile(values, 1.0 - alpha)), "valid_resamples": int(values.size)}


def aggregate_ml_predictions(predictions, threshold):
    """
    Aggregate five nested sensor realizations for each independent scenario.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Persisted 600-row ADR1D-ML challenge output.
    threshold : float
        Frozen probability threshold for decay resolvability.

    Returns
    -------
    pandas.DataFrame
        One row per base scenario with geometric regression means.

    """
    rows = []
    for scenario_id, frame in predictions.groupby("base_scenario_id", sort=True):
        probability = float(frame["predicted_decay_resolvable_probability"].mean())
        rows.append({"base_scenario_id": scenario_id, "regime": str(frame["regime"].iloc[0]), "design_component": str(frame["design_component"].iloc[0]), "actual_effective_velocity_m_s": float(frame["actual_effective_velocity_m_s"].iloc[0]), "predicted_effective_velocity_m_s": float(10.0 ** np.mean(np.log10(frame["predicted_effective_velocity_m_s"].to_numpy(dtype=float)))), "actual_effective_dispersion_m2_s": float(frame["actual_effective_dispersion_m2_s"].iloc[0]), "predicted_effective_dispersion_m2_s": float(10.0 ** np.mean(np.log10(frame["predicted_effective_dispersion_m2_s"].to_numpy(dtype=float)))), "actual_decay_resolvable": int(frame["actual_decay_resolvable"].iloc[0]), "predicted_decay_resolvable_probability": probability, "predicted_decay_resolvable": int(probability >= threshold), "actual_decay_rate_s_1": float(frame["actual_decay_rate_s_1"].iloc[0]), "predicted_decay_rate_if_resolvable_s_1": float(10.0 ** np.mean(np.log10(frame["predicted_decay_rate_if_resolvable_s_1"].to_numpy(dtype=float))))})
    return pd.DataFrame(rows)


def ml_repeatability(predictions, aggregated):
    """
    Recalculate variation among nested sensor-noise realizations.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Replicate-level ADR1D-ML predictions.
    aggregated : pandas.DataFrame
        Base-scenario predictions and final classifications.

    Returns
    -------
    dict
        Distribution summaries for variation and classification agreement.

    """
    labels = aggregated.set_index("base_scenario_id")["predicted_decay_resolvable"]
    diagnostics = {key: [] for key in ("velocity_log10_standard_deviation", "velocity_coefficient_of_variation", "dispersion_log10_standard_deviation", "dispersion_coefficient_of_variation", "conditional_decay_log10_standard_deviation", "conditional_decay_coefficient_of_variation", "probability_standard_deviation", "classification_agreement_fraction")}
    for scenario_id, frame in predictions.groupby("base_scenario_id", sort=True):
        velocity    = frame["predicted_effective_velocity_m_s"].to_numpy(dtype=float)
        dispersion  = frame["predicted_effective_dispersion_m2_s"].to_numpy(dtype=float)
        decay       = frame["predicted_decay_rate_if_resolvable_s_1"].to_numpy(dtype=float)
        probability = frame["predicted_decay_resolvable_probability"].to_numpy(dtype=float)
        diagnostics["velocity_log10_standard_deviation"].append(float(np.std(np.log10(velocity), ddof=1)))
        diagnostics["velocity_coefficient_of_variation"].append(float(np.std(velocity, ddof=1) / np.mean(velocity)))
        diagnostics["dispersion_log10_standard_deviation"].append(float(np.std(np.log10(dispersion), ddof=1)))
        diagnostics["dispersion_coefficient_of_variation"].append(float(np.std(dispersion, ddof=1) / np.mean(dispersion)))
        diagnostics["conditional_decay_log10_standard_deviation"].append(float(np.std(np.log10(decay), ddof=1)))
        diagnostics["conditional_decay_coefficient_of_variation"].append(float(np.std(decay, ddof=1) / np.mean(decay)))
        diagnostics["probability_standard_deviation"].append(float(np.std(probability, ddof=1)))
        diagnostics["classification_agreement_fraction"].append(float(np.mean(frame["predicted_decay_resolvable"].to_numpy(dtype=int) == int(labels.loc[scenario_id]))))
    return {key: summarize_distribution(values) for key, values in diagnostics.items()}


def ml_subgroups(aggregated, threshold):
    """
    Recalculate ADR1D-ML metrics for each pre-specified physical regime.

    Parameters
    ----------
    aggregated : pandas.DataFrame
        One independently aggregated row per base scenario.
    threshold : float
        Frozen probability threshold.

    Returns
    -------
    dict
        Regression and classification evidence by physical regime.

    """
    result = {}
    for regime, frame in aggregated.groupby("regime", sort=True):
        classification = classification_metrics(frame["actual_decay_resolvable"], frame["predicted_decay_resolvable_probability"], threshold) if frame["actual_decay_resolvable"].nunique() == 2 else {"status": "not_estimable_single_truth_class", "rows": int(len(frame))}
        result[str(regime)] = {"base_scenarios": int(len(frame)), "effective_velocity": regression_metrics(frame["actual_effective_velocity_m_s"], frame["predicted_effective_velocity_m_s"]), "effective_dispersion": regression_metrics(frame["actual_effective_dispersion_m2_s"], frame["predicted_effective_dispersion_m2_s"]), "decay_resolvability": classification}
    return result


def ml_bootstrap(aggregated, protocol, threshold):
    """
    Reproduce the locked cluster bootstrap for ADR1D-ML.

    Parameters
    ----------
    aggregated : pandas.DataFrame
        One row per independent base scenario.
    protocol : dict
        Locked resample count, confidence level, and random seed.
    threshold : float
        Frozen decay-resolvability probability threshold.

    Returns
    -------
    dict
        Percentile intervals for all pre-specified primary metrics.

    """
    analysis    = protocol["statistical_analysis"]
    rng         = np.random.default_rng(int(analysis["bootstrap_seed"]))
    sample_size = len(aggregated)
    collected   = {key: [] for key in ("velocity_rmse_log10", "velocity_r2_log10", "velocity_median_ape", "dispersion_rmse_log10", "dispersion_r2_log10", "dispersion_median_ape", "decay_balanced_accuracy", "decay_roc_auc", "conditional_decay_rmse_log10", "conditional_decay_median_ape")}
    for _ in range(int(analysis["bootstrap_resamples"])):
        frame          = aggregated.iloc[rng.integers(0, sample_size, size=sample_size)]
        velocity       = regression_metrics(frame["actual_effective_velocity_m_s"], frame["predicted_effective_velocity_m_s"])
        dispersion     = regression_metrics(frame["actual_effective_dispersion_m2_s"], frame["predicted_effective_dispersion_m2_s"])
        actual_class   = frame["actual_decay_resolvable"].to_numpy(dtype=int)
        probability    = frame["predicted_decay_resolvable_probability"].to_numpy(dtype=float)
        predicted      = (probability >= threshold).astype(int)
        collected["velocity_rmse_log10"].append(velocity["rmse_log10"])
        collected["velocity_r2_log10"].append(velocity["r2_log10"])
        collected["velocity_median_ape"].append(velocity["median_absolute_percentage_error"])
        collected["dispersion_rmse_log10"].append(dispersion["rmse_log10"])
        collected["dispersion_r2_log10"].append(dispersion["r2_log10"])
        collected["dispersion_median_ape"].append(dispersion["median_absolute_percentage_error"])
        collected["decay_balanced_accuracy"].append(float(balanced_accuracy_score(actual_class, predicted)) if np.unique(actual_class).size == 2 else math.nan)
        collected["decay_roc_auc"].append(float(roc_auc_score(actual_class, probability)) if np.unique(actual_class).size == 2 else math.nan)
        mask = actual_class == 1
        if int(mask.sum()) >= 2:
            conditional = conditional_decay_metrics(frame["actual_decay_rate_s_1"], frame["predicted_decay_rate_if_resolvable_s_1"], mask)
            collected["conditional_decay_rmse_log10"].append(conditional["rmse_log10"])
            collected["conditional_decay_median_ape"].append(conditional["median_absolute_percentage_error"])
        else:
            collected["conditional_decay_rmse_log10"].append(math.nan)
            collected["conditional_decay_median_ape"].append(math.nan)
    confidence = float(analysis["confidence_level"])
    return {key: bootstrap_interval(values, confidence) for key, values in collected.items()}


def validate_ml_inputs(predictions, scenarios, protocol, threshold, audit):
    """
    Verify ADR1D-ML identifiers, truth columns, replicates, and physical output.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Persisted replicate-level challenge predictions.
    scenarios : pandas.DataFrame
        Locked base-scenario metadata.
    protocol : dict
        Locked challenge design and thresholds.
    threshold : float
        Frozen model decision threshold.
    audit : Audit
        Mutable independent audit recorder.

    Returns
    -------
    None
        Structural and truth checks are recorded in `audit`.

    """
    audit.equal(len(predictions), 600, "ADR1D-ML prediction rows")
    audit.equal(predictions["base_scenario_id"].nunique(), 120, "ADR1D-ML base scenarios")
    counts = predictions.groupby("base_scenario_id").size()
    audit.require(counts.eq(5).all(), "ADR1D-ML does not contain five replicates per base scenario")
    audit.equal(set(predictions["replicate_id"]), {"R01", "R02", "R03", "R04", "R05"}, "ADR1D-ML replicate identifiers")
    audit.require(predictions["scenario_id"].is_unique, "ADR1D-ML replicate scenario identifiers are not unique")
    truth = scenarios.set_index("scenario_id").loc[predictions["base_scenario_id"]].reset_index(drop=True)
    audit.require(np.allclose(predictions["actual_effective_velocity_m_s"], truth["velocity_m_s"] / truth["retardation_factor"], rtol=0.0, atol=5.0e-12), "ADR1D-ML effective-velocity truth differs from locked scenarios")
    audit.require(np.allclose(predictions["actual_effective_dispersion_m2_s"], truth["dispersion_m2_s"] / truth["retardation_factor"], rtol=0.0, atol=5.0e-11), "ADR1D-ML effective-dispersion truth differs from locked scenarios")
    audit.require(np.allclose(predictions["actual_decay_rate_s_1"], truth["decay_rate_s_1"], rtol=0.0, atol=5.0e-16), "ADR1D-ML decay truth differs from locked scenarios")
    damkohler_threshold = float(protocol["challenge_design"]["decay_resolvability"]["damkohler_threshold"])
    audit.require(np.array_equal(predictions["actual_decay_resolvable"].to_numpy(dtype=int), truth["damkohler_number"].ge(damkohler_threshold).to_numpy(dtype=int)), "ADR1D-ML decay-resolvability truth differs")
    audit.require(np.array_equal(predictions["predicted_decay_resolvable"].to_numpy(dtype=int), predictions["predicted_decay_resolvable_probability"].ge(threshold).to_numpy(dtype=int)), "ADR1D-ML replicate classes differ from the frozen threshold")
    audit.require((predictions["predicted_effective_velocity_m_s"] > 0.0).all(), "ADR1D-ML includes a non-positive velocity")
    audit.require((predictions["predicted_effective_dispersion_m2_s"] > 0.0).all(), "ADR1D-ML includes a non-positive dispersion")
    audit.require(predictions["predicted_decay_resolvable_probability"].between(0.0, 1.0).all(), "ADR1D-ML includes an invalid probability")
    audit.require((predictions["predicted_decay_rate_if_resolvable_s_1"] > 0.0).all(), "ADR1D-ML includes a non-positive conditional decay rate")


def calculate_ml_evidence(predictions, protocol, threshold):
    """
    Independently reconstruct all reported ADR1D-ML evidence.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Persisted replicate-level challenge output.
    protocol : dict
        Locked metrics, criteria, and statistical-analysis settings.
    threshold : float
        Frozen model probability threshold.

    Returns
    -------
    dict
        Complete recalculated evidence tree matching the metrics artifact.

    """
    aggregated     = aggregate_ml_predictions(predictions, threshold)
    velocity       = regression_metrics(aggregated["actual_effective_velocity_m_s"], aggregated["predicted_effective_velocity_m_s"])
    dispersion     = regression_metrics(aggregated["actual_effective_dispersion_m2_s"], aggregated["predicted_effective_dispersion_m2_s"])
    classification = classification_metrics(aggregated["actual_decay_resolvable"], aggregated["predicted_decay_resolvable_probability"], threshold)
    conditional    = conditional_decay_metrics(aggregated["actual_decay_rate_s_1"], aggregated["predicted_decay_rate_if_resolvable_s_1"], aggregated["actual_decay_resolvable"].eq(1))
    physical = {"positive_velocity_fraction": float(np.mean(predictions["predicted_effective_velocity_m_s"] > 0.0)), "positive_dispersion_fraction": float(np.mean(predictions["predicted_effective_dispersion_m2_s"] > 0.0)), "probability_in_unit_interval_fraction": float(np.mean(predictions["predicted_decay_resolvable_probability"].between(0.0, 1.0))), "positive_conditional_decay_fraction": float(np.mean(predictions["predicted_decay_rate_if_resolvable_s_1"] > 0.0))}
    criteria = protocol["minimum_adequacy_criteria"]["adr1d_ml"]
    checks = {"effective_velocity_median_absolute_percentage_error": velocity["median_absolute_percentage_error"] <= criteria["effective_velocity_median_absolute_percentage_error_max"], "effective_velocity_percentile_90_absolute_percentage_error": velocity["percentile_90_absolute_percentage_error"] <= criteria["effective_velocity_percentile_90_absolute_percentage_error_max"], "effective_velocity_r2_log10": velocity["r2_log10"] >= criteria["effective_velocity_r2_log10_min"], "effective_dispersion_median_absolute_percentage_error": dispersion["median_absolute_percentage_error"] <= criteria["effective_dispersion_median_absolute_percentage_error_max"], "effective_dispersion_percentile_90_absolute_percentage_error": dispersion["percentile_90_absolute_percentage_error"] <= criteria["effective_dispersion_percentile_90_absolute_percentage_error_max"], "effective_dispersion_r2_log10": dispersion["r2_log10"] >= criteria["effective_dispersion_r2_log10_min"], "decay_balanced_accuracy": classification["balanced_accuracy"] >= criteria["decay_balanced_accuracy_min"], "decay_roc_auc": classification["roc_auc"] >= criteria["decay_roc_auc_min"], "physical_validity": min(physical.values()) >= criteria["physical_validity_fraction_required"]}
    available = conditional["rows"] >= criteria["conditional_decay_minimum_resolvable_base_scenarios_for_gate"]
    conditional_checks = {"status": "evaluated" if available else "not_estimable_insufficient_resolvable_scenarios", "minimum_required_rows": int(criteria["conditional_decay_minimum_resolvable_base_scenarios_for_gate"]), "observed_rows": int(conditional["rows"]), "median_absolute_percentage_error": conditional["median_absolute_percentage_error"] <= criteria["conditional_decay_median_absolute_percentage_error_max"] if available else None, "r2_log10": conditional["r2_log10"] >= criteria["conditional_decay_r2_log10_min"] if available else None}
    evaluated = list(checks.values()) + [conditional_checks[key] for key in ("median_absolute_percentage_error", "r2_log10") if conditional_checks[key] is not None]
    outcome = "meets_all_pre_specified_criteria" if all(evaluated) and available else "mixed_evidence"
    if evaluated and not any(evaluated):
        outcome = "does_not_meet_pre_specified_criteria"
    evidence = {"aggregation_policy": "Geometric mean for positive regressions and arithmetic mean probability over five sensor-noise replicates; classification then uses the frozen 0.19 threshold.", "base_scenarios": int(len(aggregated)), "replicate_predictions": int(len(predictions)), "effective_velocity": velocity, "effective_dispersion": dispersion, "decay_resolvability": classification, "conditional_decay": conditional, "noise_repeatability": ml_repeatability(predictions, aggregated), "physical_checks": physical, "by_regime": ml_subgroups(aggregated, threshold), "confidence_intervals_95": ml_bootstrap(aggregated, protocol, threshold), "adequacy": {"overall_outcome": outcome, "checks": checks, "conditional_decay_checks": conditional_checks}}
    return evidence


def point_metrics(reference, prediction, active_threshold):
    """
    Recalculate pooled normalized-field errors for point rows.

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
        Pooled errors, R2, active errors, and boundedness fractions.

    """
    reference   = np.asarray(reference, dtype=float)
    prediction  = np.asarray(prediction, dtype=float)
    residual    = prediction - reference
    active      = reference > active_threshold
    denominator = np.sum(np.square(reference - reference.mean()))
    return {"rows": int(reference.size), "active_rows": int(active.sum()), "mae": float(np.mean(np.abs(residual))), "rmse": float(np.sqrt(np.mean(np.square(residual)))), "r2": float(1.0 - np.sum(np.square(residual)) / denominator) if denominator > 0.0 else None, "active_mae": float(np.mean(np.abs(residual[active]))) if active.any() else None, "active_rmse": float(np.sqrt(np.mean(np.square(residual[active])))) if active.any() else None, "maximum_absolute_error": float(np.max(np.abs(residual))), "negative_prediction_fraction": float(np.mean(prediction < 0.0)), "above_one_prediction_fraction": float(np.mean(prediction > 1.0))}


def validate_nn_inputs(predictions, field, scenarios, audit):
    """
    Verify ADR1D-NN ordering, analytical truth, metadata, grids, and constraints.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Persisted point-level ADR1D-NN challenge output.
    field : pandas.DataFrame
        Locked analytical challenge field.
    scenarios : pandas.DataFrame
        Locked base-scenario metadata.
    audit : Audit
        Mutable independent audit recorder.

    Returns
    -------
    None
        Structural and truth checks are recorded in `audit`.

    """
    audit.equal(len(predictions), 299880, "ADR1D-NN prediction rows")
    audit.equal(predictions["scenario_id"].nunique(), 120, "ADR1D-NN base scenarios")
    audit.require(predictions.groupby("scenario_id").size().eq(2499).all(), "ADR1D-NN does not contain 2,499 points per scenario")
    for column in ("scenario_id", "split", "regime", "design_component"):
        audit.require(predictions[column].equals(field[column]), f"ADR1D-NN {column} differs from the locked field")
    for column in ("time_s", "x_m"):
        audit.require(np.array_equal(predictions[column].to_numpy(dtype=float), field[column].to_numpy(dtype=float)), f"ADR1D-NN {column} coordinates differ from the locked field")
    audit.require(np.allclose(predictions["reference_normalized_concentration"], field["normalized_concentration"], rtol=0.0, atol=5.0e-12), "ADR1D-NN analytical truth differs from the locked field")
    lookup = scenarios.set_index("scenario_id").loc[predictions["scenario_id"]]
    audit.require(np.allclose(predictions["peclet_number"], lookup["peclet_number"], rtol=0.0, atol=5.0e-10), "ADR1D-NN Peclet metadata differs")
    audit.require(np.allclose(predictions["damkohler_number"], lookup["damkohler_number"], rtol=0.0, atol=5.0e-10), "ADR1D-NN Damkohler metadata differs")
    recalculated_error = np.abs(predictions["predicted_normalized_concentration"] - predictions["reference_normalized_concentration"])
    audit.require(np.allclose(predictions["absolute_error"], recalculated_error, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE), "ADR1D-NN stored absolute errors differ")
    audit.require(predictions["predicted_normalized_concentration"].between(0.0, 1.0).all(), "ADR1D-NN includes a prediction outside [0, 1]")
    audit.require(predictions["constraint_applied"].isin(("neural_interior", "initial_interior", "inlet_active", "inlet_inactive")).all(), "ADR1D-NN includes an unknown constraint label")
    constrained = predictions["constraint_applied"].ne("neural_interior")
    audit.require(np.allclose(predictions.loc[constrained, "predicted_normalized_concentration"], predictions.loc[constrained, "reference_normalized_concentration"], rtol=0.0, atol=ABSOLUTE_TOLERANCE), "ADR1D-NN exact constraints differ from analytical truth")


def calculate_nn_scenarios(predictions, scenarios, active_threshold):
    """
    Independently reconstruct every ADR1D-NN scenario diagnostic.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Persisted point-level neural predictions and analytical truth.
    scenarios : pandas.DataFrame
        Locked physical metadata for 120 base scenarios.
    active_threshold : float
        Normalized concentration defining active transport and arrival.

    Returns
    -------
    tuple of pandas.DataFrame
        Public scenario metrics and internal bootstrap sufficient statistics.

    """
    lookup        = scenarios.set_index("scenario_id")
    public_rows   = []
    internal_rows = []
    for scenario_id, frame in predictions.groupby("scenario_id", sort=True):
        frame      = frame.sort_values(["time_s", "x_m"])
        scenario   = lookup.loc[scenario_id]
        times      = np.sort(frame["time_s"].unique().astype(float))
        positions  = np.sort(frame["x_m"].unique().astype(float))
        reference  = frame["reference_normalized_concentration"].to_numpy(dtype=float).reshape(len(times), len(positions))
        prediction = frame["predicted_normalized_concentration"].to_numpy(dtype=float).reshape(len(times), len(positions))
        metrics    = point_metrics(reference.ravel(), prediction.ravel(), active_threshold)
        reference_mass     = np.trapezoid(reference, x=positions, axis=1)
        predicted_mass     = np.trapezoid(prediction, x=positions, axis=1)
        reference_exposure = float(np.trapezoid(reference_mass, x=times))
        predicted_exposure = float(np.trapezoid(predicted_mass, x=times))
        mass_error         = abs(predicted_exposure - reference_exposure) / reference_exposure if reference_exposure > 0.0 else math.nan
        arrival_errors     = []
        missed_arrivals    = 0
        for position_index in range(1, len(positions)):
            reference_active = np.flatnonzero(reference[:, position_index] > active_threshold)
            if reference_active.size == 0:
                continue
            predicted_active = np.flatnonzero(prediction[:, position_index] > active_threshold)
            reference_arrival = float(times[reference_active[0]])
            if predicted_active.size:
                predicted_arrival = float(times[predicted_active[0]])
            else:
                predicted_arrival = float(times[-1] + times[1] - times[0])
                missed_arrivals += 1
            arrival_errors.append(abs(predicted_arrival - reference_arrival))
        arrival_error       = float(np.median(arrival_errors)) if arrival_errors else math.nan
        time_step           = float(times[1] - times[0])
        space_step          = float(positions[1] - positions[0])
        effective_velocity  = float(scenario["velocity_m_s"] / scenario["retardation_factor"])
        effective_dispersion = float(scenario["dispersion_m2_s"] / scenario["retardation_factor"])
        decay_rate          = float(scenario["decay_rate_s_1"])
        temporal_derivative = (prediction[2:, 1:-1] - prediction[:-2, 1:-1]) / (2.0 * time_step)
        spatial_derivative  = (prediction[1:-1, 2:] - prediction[1:-1, :-2]) / (2.0 * space_step)
        second_derivative   = (prediction[1:-1, 2:] - 2.0 * prediction[1:-1, 1:-1] + prediction[1:-1, :-2]) / space_step**2
        residual            = temporal_derivative - effective_dispersion * second_derivative + effective_velocity * spatial_derivative + decay_rate * prediction[1:-1, 1:-1]
        pde_rmse            = float(np.sqrt(np.mean(np.square(residual))))
        initial_error       = float(np.max(np.abs(prediction[0, 1:] - reference[0, 1:])))
        inlet_error         = float(np.max(np.abs(prediction[:, 0] - reference[:, 0])))
        public_rows.append({"scenario_id": scenario_id, "regime": str(scenario["regime"]), "design_component": str(scenario["design_component"]), "peclet_number": float(scenario["peclet_number"]), "damkohler_number": float(scenario["damkohler_number"]), "rows": metrics["rows"], "active_rows": metrics["active_rows"], "mae": metrics["mae"], "rmse": metrics["rmse"], "r2": metrics["r2"], "active_mae": metrics["active_mae"], "active_rmse": metrics["active_rmse"], "maximum_absolute_error": metrics["maximum_absolute_error"], "integrated_mass_relative_error": mass_error, "arrival_time_absolute_error_s": arrival_error, "arrival_positions": len(arrival_errors), "missed_arrival_positions": missed_arrivals, "pde_residual_rmse_s_1": pde_rmse, "initial_condition_maximum_absolute_error": initial_error, "inlet_boundary_maximum_absolute_error": inlet_error})
        flat_reference = reference.ravel()
        flat_prediction = prediction.ravel()
        active = flat_reference > active_threshold
        internal_rows.append({"scenario_id": scenario_id, "rows": flat_reference.size, "sum_absolute_error": float(np.sum(np.abs(flat_prediction - flat_reference))), "sum_squared_error": float(np.sum(np.square(flat_prediction - flat_reference))), "sum_reference": float(np.sum(flat_reference)), "sum_squared_reference": float(np.sum(np.square(flat_reference))), "active_rows": int(active.sum()), "active_sum_absolute_error": float(np.sum(np.abs(flat_prediction[active] - flat_reference[active]))), "active_sum_squared_error": float(np.sum(np.square(flat_prediction[active] - flat_reference[active]))), "scenario_rmse": metrics["rmse"], "integrated_mass_relative_error": mass_error, "arrival_time_absolute_error_s": arrival_error, "pde_residual_sum_squares": float(np.sum(np.square(residual))), "pde_residual_rows": int(residual.size)})
    return pd.DataFrame(public_rows), pd.DataFrame(internal_rows)


def compare_scenario_table(actual, expected, audit):
    """
    Compare all 120 independently recalculated scenario rows and columns.

    Parameters
    ----------
    actual : pandas.DataFrame
        Independently reconstructed scenario diagnostics.
    expected : pandas.DataFrame
        Persisted ADR1D-NN scenario table.
    audit : Audit
        Mutable independent audit recorder.

    Returns
    -------
    None
        Every table cell is compared and recorded.

    """
    audit.equal(list(actual.columns), list(expected.columns), "ADR1D-NN scenario-table columns")
    audit.equal(len(actual), len(expected), "ADR1D-NN scenario-table rows")
    for column in actual.columns:
        if pd.api.types.is_numeric_dtype(expected[column]):
            for index, (actual_value, expected_value) in enumerate(zip(actual[column], expected[column])):
                audit.close(actual_value, expected_value, f"ADR1D-NN scenario table {column}[{index}]")
        else:
            audit.require(actual[column].equals(expected[column]), f"ADR1D-NN scenario-table column differs: {column}")


def nn_group_metrics(predictions, groups, active_threshold):
    """
    Recalculate pooled neural field metrics for named scenario groups.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Complete point-level prediction evidence.
    groups : mapping of str to iterable of str
        Scenario identifiers assigned to each subgroup.
    active_threshold : float
        Reference threshold defining active transport rows.

    Returns
    -------
    dict
        Pooled point metrics and base-scenario counts by subgroup.

    """
    result = {}
    for label, identifiers in groups.items():
        identifiers = set(identifiers)
        frame = predictions.loc[predictions["scenario_id"].isin(identifiers)]
        result[str(label)] = {"base_scenarios": len(identifiers), **point_metrics(frame["reference_normalized_concentration"], frame["predicted_normalized_concentration"], active_threshold)}
    return result


def nn_subgroups(predictions, scenarios, active_threshold, damkohler_threshold):
    """
    Recalculate all pre-specified ADR1D-NN subgroup metrics.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Complete point-level prediction evidence.
    scenarios : pandas.DataFrame
        Locked scenario metadata.
    active_threshold : float
        Reference concentration threshold.
    damkohler_threshold : float
        Locked decay-resolvability threshold.

    Returns
    -------
    dict
        Metrics by regime, design, Peclet quartile, decay state, and field zone.

    """
    regime_groups   = {str(label): frame["scenario_id"].tolist() for label, frame in scenarios.groupby("regime", sort=True)}
    design_groups   = {str(label): frame["scenario_id"].tolist() for label, frame in scenarios.groupby("design_component", sort=True)}
    quartile_labels = pd.qcut(scenarios["peclet_number"].rank(method="first"), 4, labels=("Q1", "Q2", "Q3", "Q4"))
    quartile_groups = {str(label): scenarios.loc[quartile_labels.eq(label), "scenario_id"].tolist() for label in quartile_labels.cat.categories}
    decay_state     = np.select([scenarios["decay_rate_s_1"].eq(0.0), scenarios["damkohler_number"].ge(damkohler_threshold)], ["zero", "resolvable"], default="below_resolution")
    decay_groups    = {label: scenarios.loc[decay_state == label, "scenario_id"].tolist() for label in ("zero", "below_resolution", "resolvable")}
    active          = predictions["reference_normalized_concentration"] > active_threshold
    regions         = {}
    for label, mask in (("active", active), ("inactive", ~active)):
        frame      = predictions.loc[mask]
        reference  = frame["reference_normalized_concentration"].to_numpy(dtype=float)
        predicted  = frame["predicted_normalized_concentration"].to_numpy(dtype=float)
        residual   = predicted - reference
        regions[label] = {"rows": int(len(frame)), "mae": float(np.mean(np.abs(residual))), "rmse": float(np.sqrt(np.mean(np.square(residual)))), "maximum_absolute_error": float(np.max(np.abs(residual)))}
    return {"regime": nn_group_metrics(predictions, regime_groups, active_threshold), "design_component": nn_group_metrics(predictions, design_groups, active_threshold), "peclet_quartile": nn_group_metrics(predictions, quartile_groups, active_threshold), "damkohler_state": nn_group_metrics(predictions, decay_groups, active_threshold), "field_region": regions}


def nn_bootstrap(internal, protocol):
    """
    Reproduce the locked vectorized cluster bootstrap for ADR1D-NN.

    Parameters
    ----------
    internal : pandas.DataFrame
        One row of sufficient statistics per independent scenario.
    protocol : dict
        Locked resample count, confidence level, and random seed.

    Returns
    -------
    dict
        Percentile intervals for pooled and scenario-level neural metrics.

    """
    analysis   = protocol["statistical_analysis"]
    rng        = np.random.default_rng(int(analysis["bootstrap_seed"]))
    indexes    = rng.integers(0, len(internal), size=(int(analysis["bootstrap_resamples"]), len(internal)))

    def sampled_sum(column):
        """Sum one sufficient-statistic column over every bootstrap sample."""
        values = internal[column].to_numpy(dtype=float)
        return values[indexes].sum(axis=1)

    rows           = sampled_sum("rows")
    absolute       = sampled_sum("sum_absolute_error")
    squared        = sampled_sum("sum_squared_error")
    reference      = sampled_sum("sum_reference")
    reference2     = sampled_sum("sum_squared_reference")
    active_rows    = sampled_sum("active_rows")
    active_squared = sampled_sum("active_sum_squared_error")
    sst            = reference2 - np.square(reference) / rows
    values = {"pooled_mae": absolute / rows, "pooled_rmse": np.sqrt(squared / rows), "pooled_r2": 1.0 - squared / sst, "active_rmse": np.sqrt(active_squared / active_rows), "percentile_90_scenario_rmse": np.quantile(internal["scenario_rmse"].to_numpy(dtype=float)[indexes], 0.90, axis=1), "median_integrated_mass_relative_error": np.nanmedian(internal["integrated_mass_relative_error"].to_numpy(dtype=float)[indexes], axis=1), "percentile_90_integrated_mass_relative_error": np.nanquantile(internal["integrated_mass_relative_error"].to_numpy(dtype=float)[indexes], 0.90, axis=1), "median_arrival_time_absolute_error_s": np.nanmedian(internal["arrival_time_absolute_error_s"].to_numpy(dtype=float)[indexes], axis=1), "percentile_90_arrival_time_absolute_error_s": np.nanquantile(internal["arrival_time_absolute_error_s"].to_numpy(dtype=float)[indexes], 0.90, axis=1)}
    confidence = float(analysis["confidence_level"])
    return {key: bootstrap_interval(value, confidence) for key, value in values.items()}


def calculate_nn_evidence(predictions, scenarios, scenario_metrics, internal, protocol):
    """
    Independently reconstruct all reported ADR1D-NN evidence.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Persisted point-level challenge output.
    scenarios : pandas.DataFrame
        Locked physical scenario metadata.
    scenario_metrics : pandas.DataFrame
        Independently recalculated per-scenario diagnostics.
    internal : pandas.DataFrame
        Scenario-level sufficient statistics for bootstrap calculations.
    protocol : dict
        Locked metrics, criteria, and statistical-analysis settings.

    Returns
    -------
    dict
        Complete recalculated evidence tree matching the metrics artifact.

    """
    active_threshold     = float(protocol["metrics"]["active_normalized_concentration_threshold"])
    damkohler_threshold  = float(protocol["challenge_design"]["decay_resolvability"]["damkohler_threshold"])
    field                = point_metrics(predictions["reference_normalized_concentration"], predictions["predicted_normalized_concentration"], active_threshold)
    mass                 = scenario_metrics["integrated_mass_relative_error"].to_numpy(dtype=float)
    arrival              = scenario_metrics["arrival_time_absolute_error_s"].to_numpy(dtype=float)
    pde_sum              = float(internal["pde_residual_sum_squares"].sum())
    pde_rows             = int(internal["pde_residual_rows"].sum())
    physical = {"initial_condition_maximum_absolute_error": float(scenario_metrics["initial_condition_maximum_absolute_error"].max()), "inlet_boundary_maximum_absolute_error": float(scenario_metrics["inlet_boundary_maximum_absolute_error"].max()), "negative_prediction_fraction": field["negative_prediction_fraction"], "above_one_prediction_fraction": field["above_one_prediction_fraction"], "median_integrated_mass_relative_error": float(np.nanmedian(mass)), "percentile_90_integrated_mass_relative_error": float(np.nanquantile(mass, 0.90)), "median_arrival_time_absolute_error_s": float(np.nanmedian(arrival)), "percentile_90_arrival_time_absolute_error_s": float(np.nanquantile(arrival, 0.90)), "arrival_estimable_scenarios": int(np.isfinite(arrival).sum()), "pde_residual_rmse_s_1": float(math.sqrt(pde_sum / pde_rows)), "pde_residual_rows": pde_rows}
    distribution = {"median_scenario_rmse": float(scenario_metrics["rmse"].median()), "percentile_90_scenario_rmse": float(scenario_metrics["rmse"].quantile(0.90)), "maximum_scenario_rmse": float(scenario_metrics["rmse"].max())}
    criteria = protocol["minimum_adequacy_criteria"]["adr1d_nn"]
    checks = {"pooled_rmse": field["rmse"] <= criteria["pooled_rmse_max"], "active_rmse": field["active_rmse"] <= criteria["active_rmse_max"], "r2": field["r2"] >= criteria["r2_min"], "percentile_90_scenario_rmse": distribution["percentile_90_scenario_rmse"] <= criteria["percentile_90_scenario_rmse_max"], "initial_condition_maximum_absolute_error": physical["initial_condition_maximum_absolute_error"] <= criteria["initial_condition_maximum_absolute_error_max"], "inlet_boundary_maximum_absolute_error": physical["inlet_boundary_maximum_absolute_error"] <= criteria["inlet_boundary_maximum_absolute_error_max"], "negative_prediction_fraction": physical["negative_prediction_fraction"] <= criteria["negative_prediction_fraction_max"], "above_one_prediction_fraction": physical["above_one_prediction_fraction"] <= criteria["above_one_prediction_fraction_max"], "median_integrated_mass_relative_error": physical["median_integrated_mass_relative_error"] <= criteria["median_integrated_mass_relative_error_max"], "percentile_90_integrated_mass_relative_error": physical["percentile_90_integrated_mass_relative_error"] <= criteria["percentile_90_integrated_mass_relative_error_max"], "median_arrival_time_absolute_error_s": physical["median_arrival_time_absolute_error_s"] <= criteria["median_arrival_time_absolute_error_s_max"], "percentile_90_arrival_time_absolute_error_s": physical["percentile_90_arrival_time_absolute_error_s"] <= criteria["percentile_90_arrival_time_absolute_error_s_max"]}
    if all(checks.values()):
        outcome = "meets_all_pre_specified_criteria"
    elif any(checks.values()):
        outcome = "mixed_evidence"
    else:
        outcome = "does_not_meet_pre_specified_criteria"
    definitions = {"integrated_mass_relative_error": "Absolute relative error in the time integral of the spatially integrated normalized concentration for each scenario.", "arrival_time_absolute_error_s": "Median absolute first-threshold-crossing error over interior grid positions where the analytical field exceeds 1e-4; a missed prediction is assigned final_time + time_step.", "pde_residual_diagnostic": "Centered finite-difference RMSE of dC/dt - D_eff*d2C/dx2 + v_eff*dC/dx + lambda*C over interior space-time nodes."}
    return {"field": field, "scenario_distribution": distribution, "physical": physical, "metric_definitions": definitions, "subgroups": nn_subgroups(predictions, scenarios, active_threshold, damkohler_threshold), "confidence_intervals_95": nn_bootstrap(internal, protocol), "adequacy": {"overall_outcome": outcome, "checks": checks}}


def validate_results():
    """
    Execute the read-only independent audit and write its deterministic report.

    Returns
    -------
    dict
        Complete validation report written to `results/`.

    Raises
    ------
    ValidationError
        If any integrity, structural, physical, or numerical check fails.

    Notes
    -----
    This function does not import model frameworks, deserialize model files,
    call public inference interfaces, or alter any challenge prediction.
    """
    audit           = Audit()
    protocol        = load_json(PROTOCOL_PATH)
    challenge_lock  = load_json(CHALLENGE_LOCK_PATH)
    evaluation_lock = load_json(EVALUATION_LOCK_PATH)
    ml_metrics      = load_json(ML_METRICS_PATH)
    nn_metrics      = load_json(NN_METRICS_PATH)
    scenarios       = pd.read_csv(SCENARIO_PATH)
    field           = pd.read_csv(FIELD_PATH)
    ml_predictions  = pd.read_csv(ML_PREDICTIONS_PATH)
    nn_predictions  = pd.read_csv(NN_PREDICTIONS_PATH)
    nn_scenarios    = pd.read_csv(NN_SCENARIOS_PATH)
    chain           = verify_hash_chain(protocol, challenge_lock, evaluation_lock, ml_metrics, nn_metrics, audit)

    ml_protocol = load_json(ML_PROTOCOL_PATH)
    threshold   = float(ml_protocol["models"]["decay_resolvability"]["decision_threshold"])
    audit.close(threshold, evaluation_lock["execution_contract"]["adr1d_ml_decision_threshold"], "Frozen ADR1D-ML threshold")
    validate_ml_inputs(ml_predictions, scenarios, protocol, threshold, audit)
    ml_evidence = calculate_ml_evidence(ml_predictions, protocol, threshold)
    audit.compare_tree(ml_evidence, ml_metrics["evidence"], "adr1d_ml.evidence")

    validate_nn_inputs(nn_predictions, field, scenarios, audit)
    active_threshold = float(protocol["metrics"]["active_normalized_concentration_threshold"])
    calculated_scenarios, internal = calculate_nn_scenarios(nn_predictions, scenarios, active_threshold)
    compare_scenario_table(calculated_scenarios, nn_scenarios, audit)
    nn_evidence = calculate_nn_evidence(nn_predictions, scenarios, calculated_scenarios, internal, protocol)
    audit.compare_tree(nn_evidence, nn_metrics["evidence"], "adr1d_nn.evidence")

    audit.equal(ml_metrics["predictions"]["rows"], len(ml_predictions), "ADR1D-ML reported prediction rows")
    audit.equal(ml_metrics["predictions"]["columns"], len(ml_predictions.columns), "ADR1D-ML reported prediction columns")
    audit.equal(nn_metrics["predictions"]["rows"], len(nn_predictions), "ADR1D-NN reported prediction rows")
    audit.equal(nn_metrics["predictions"]["columns"], len(nn_predictions.columns), "ADR1D-NN reported prediction columns")
    audit.equal(nn_metrics["scenario_metrics"]["rows"], len(nn_scenarios), "ADR1D-NN reported scenario rows")
    audit.equal(nn_metrics["scenario_metrics"]["columns"], len(nn_scenarios.columns), "ADR1D-NN reported scenario columns")

    artifacts = {"adr1d_ml_predictions": file_record(ML_PREDICTIONS_PATH, ml_predictions), "adr1d_ml_metrics": file_record(ML_METRICS_PATH), "adr1d_nn_predictions": file_record(NN_PREDICTIONS_PATH, nn_predictions), "adr1d_nn_scenario_metrics": file_record(NN_SCENARIOS_PATH, nn_scenarios), "adr1d_nn_metrics": file_record(NN_METRICS_PATH)}
    report = {
        "status": "passed",
        "validation_version": "1.0.0",
        "scope": "Independent read-only recalculation from persisted challenge predictions; no model deserialization or inference.",
        "execution": {"adr1d_ml_model_loads": 0, "adr1d_nn_model_loads": 0, "adr1d_ml_inference_runs": 0, "adr1d_nn_inference_runs": 0, "post_challenge_tuning_performed": False},
        "tolerances": {"absolute": ABSOLUTE_TOLERANCE, "relative": RELATIVE_TOLERANCE},
        "hash_chain": chain,
        "artifacts": artifacts,
        "recalculation": {
            "adr1d_ml": {"replicate_predictions": int(len(ml_predictions)), "base_scenarios": int(ml_predictions["base_scenario_id"].nunique()), "bootstrap_resamples": int(protocol["statistical_analysis"]["bootstrap_resamples"]), "adequacy_outcome": ml_evidence["adequacy"]["overall_outcome"], "reported_evidence_tree_reproduced": True},
            "adr1d_nn": {"point_predictions": int(len(nn_predictions)), "base_scenarios": int(nn_predictions["scenario_id"].nunique()), "scenario_table_rows_reproduced": int(len(calculated_scenarios)), "bootstrap_resamples": int(protocol["statistical_analysis"]["bootstrap_resamples"]), "adequacy_outcome": nn_evidence["adequacy"]["overall_outcome"], "reported_evidence_tree_reproduced": True},
        },
        "checks": {"successful_checks": audit.checks, "numerical_values_compared": audit.numerical_comparisons, "maximum_absolute_difference": audit.maximum_absolute_difference},
        "limitations": {"raw_api_outputs_retained": False, "note": "Raw public-API staging files were intentionally not retained; the enriched immutable prediction tables and their hashes are the auditable inference evidence."},
        "validator": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": sha256(Path(__file__).resolve())},
    }
    VALIDATION_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main():
    """
    Run the independent challenge-result audit and print a compact summary.

    Returns
    -------
    None
        A deterministic JSON validation report is written to `results/`.

    """
    report = validate_results()
    summary = {"status": report["status"], "output": str(VALIDATION_PATH.relative_to(ROOT)), "successful_checks": report["checks"]["successful_checks"], "numerical_values_compared": report["checks"]["numerical_values_compared"], "maximum_absolute_difference": report["checks"]["maximum_absolute_difference"]}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
