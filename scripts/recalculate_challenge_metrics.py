"""
================================================================================
ADR1D-Validation: Persisted Challenge Recalculation
================================================================================

Recalculate ADR1D-ML and ADR1D-NN challenge metrics from the distributed CSV
predictions. This public audit does not deserialize either model, perform new
inference, or overwrite the historical evidence.

Main Operations
---------------
1. Verify the local protocol, locks, prediction tables, and evaluation code.
2. Recalculate scenario aggregation, metrics, physical checks, and bootstrap
   confidence intervals from persisted predictions.
3. Compare the recalculated evidence trees with the distributed JSON reports.
4. Write a separate audit record under the ignored `reproduction/` directory.

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
import importlib.util
import json
from pathlib import Path


ROOT           = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reproduction/challenge_metric_recalculation.json"


class RecalculationError(RuntimeError):
    """Represent an invalid destination or unavailable audit implementation."""


def parse_arguments():
    """
    Parse the portable result-recalculation command line.

    Returns
    -------
    argparse.Namespace
        Output path and overwrite policy.
    """
    parser = argparse.ArgumentParser(description="Recalculate challenge metrics from persisted ADR1D predictions.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination for the independent recalculation report.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing recalculation report.")
    return parser.parse_args()


def require(condition, message):
    """
    Enforce one portable-recalculation invariant.

    Parameters
    ----------
    condition : bool
        Condition that must evaluate to true.
    message : str
        Actionable failure description.

    Raises
    ------
    RecalculationError
        If the condition is false.
    """
    if not bool(condition):
        raise RecalculationError(message)


def load_audit_module():
    """
    Load the original numerical audit as a local implementation module.

    Returns
    -------
    module
        Imported `src/validate_challenge_results.py` module.
    """
    path = ROOT / "src/validate_challenge_results.py"
    spec = importlib.util.spec_from_file_location("adr1d_persisted_challenge_audit", path)
    require(spec is not None and spec.loader is not None, f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recalculate(module):
    """
    Recalculate and compare both complete persisted evidence trees.

    Parameters
    ----------
    module : module
        Loaded original challenge-result audit implementation.

    Returns
    -------
    dict
        Portable audit record without model access or upstream path checks.
    """
    audit           = module.Audit()
    protocol        = module.load_json(module.PROTOCOL_PATH)
    challenge_lock  = module.load_json(module.CHALLENGE_LOCK_PATH)
    evaluation_lock = module.load_json(module.EVALUATION_LOCK_PATH)
    ml_metrics      = module.load_json(module.ML_METRICS_PATH)
    nn_metrics      = module.load_json(module.NN_METRICS_PATH)
    scenarios       = module.pd.read_csv(module.SCENARIO_PATH)
    field           = module.pd.read_csv(module.FIELD_PATH)
    ml_predictions  = module.pd.read_csv(module.ML_PREDICTIONS_PATH)
    nn_predictions  = module.pd.read_csv(module.NN_PREDICTIONS_PATH)
    nn_scenarios    = module.pd.read_csv(module.NN_SCENARIOS_PATH)

    protocol_hash = module.sha256(module.PROTOCOL_PATH)
    audit.equal(protocol_hash, challenge_lock["protocol"]["sha256"], "Challenge-lock protocol digest")
    audit.equal(protocol_hash, evaluation_lock["protocol"]["sha256"], "Evaluation-lock protocol digest")
    audit.equal(module.sha256(module.PREPARATION_PATH), evaluation_lock["locked_code"]["preparation"]["sha256"], "Locked preparation-code digest")
    audit.equal(module.sha256(module.EVALUATOR_PATH), evaluation_lock["locked_code"]["evaluation"]["sha256"], "Locked evaluation-code digest")
    audit.equal(module.sha256(module.ML_PREDICTIONS_PATH), ml_metrics["predictions"]["sha256"], "ADR1D-ML prediction digest")
    audit.equal(module.sha256(module.NN_PREDICTIONS_PATH), nn_metrics["predictions"]["sha256"], "ADR1D-NN prediction digest")
    audit.equal(module.sha256(module.NN_SCENARIOS_PATH), nn_metrics["scenario_metrics"]["sha256"], "ADR1D-NN scenario-metric digest")

    threshold = float(evaluation_lock["execution_contract"]["adr1d_ml_decision_threshold"])
    module.validate_ml_inputs(ml_predictions, scenarios, protocol, threshold, audit)
    ml_evidence = module.calculate_ml_evidence(ml_predictions, protocol, threshold)
    audit.compare_tree(ml_evidence, ml_metrics["evidence"], "adr1d_ml.evidence")

    module.validate_nn_inputs(nn_predictions, field, scenarios, audit)
    active_threshold = float(protocol["metrics"]["active_normalized_concentration_threshold"])
    calculated_scenarios, internal = module.calculate_nn_scenarios(nn_predictions, scenarios, active_threshold)
    module.compare_scenario_table(calculated_scenarios, nn_scenarios, audit)
    nn_evidence = module.calculate_nn_evidence(nn_predictions, scenarios, calculated_scenarios, internal, protocol)
    audit.compare_tree(nn_evidence, nn_metrics["evidence"], "adr1d_nn.evidence")

    return {
        "checks": {
            "maximum_absolute_difference": audit.maximum_absolute_difference,
            "numerical_values_compared": audit.numerical_comparisons,
            "successful_checks": audit.checks,
        },
        "execution": {
            "adr1d_ml_inference_runs": 0,
            "adr1d_ml_model_loads": 0,
            "adr1d_nn_inference_runs": 0,
            "adr1d_nn_model_loads": 0,
            "bootstrap_resamples_per_model": int(protocol["statistical_analysis"]["bootstrap_resamples"]),
        },
        "scope": "Independent recalculation from persisted challenge predictions; upstream model artifacts are identified by the locked records but are not loaded.",
        "status": "passed",
        "validation_version": "1.0.0-public",
    }


def main():
    """
    Run the public recalculation and persist its separate audit record.

    Returns
    -------
    None
        The JSON report and a compact completion summary are written.
    """
    arguments = parse_arguments()
    output    = arguments.output.resolve()
    require(arguments.overwrite or not output.exists(), f"Output already exists; add --overwrite to replace {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = recalculate(load_audit_module())
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"maximum_absolute_difference": report["checks"]["maximum_absolute_difference"], "output": str(output), "status": report["status"], "successful_checks": report["checks"]["successful_checks"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

