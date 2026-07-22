"""
================================================================================
ADR1D Validation: Challenge-Evaluation Preparation
================================================================================

Build and lock the exact ADR1D-ML and ADR1D-NN challenge input tables before
either serialized model is loaded or queried.

Main Operations
---------------
1. Verify the protocol, challenge lock, and frozen feature-building modules.
2. Reconstruct model-ready tables without importing model-inference libraries.
3. Confirm byte-identical table construction in two temporary directories.
4. Lock input digests, evaluation code, APIs, models, and zero-query state.

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
import importlib.util
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Third-party libraries
import pandas as pd


ROOT                = Path(__file__).resolve().parents[1]
PROTOCOL_PATH       = ROOT / "configs/validation_protocol.json"
CHALLENGE_LOCK_PATH = ROOT / "results/challenge_lock.json"
CANONICAL_AUDIT     = ROOT / "results/canonical_reproduction.json"
EVALUATION_LOCK     = ROOT / "results/challenge_evaluation_lock.json"
EVALUATOR_PATH      = ROOT / "src/evaluate_locked_challenge.py"
SCENARIO_PATH       = ROOT / "data/challenge_scenarios.csv"
FIELD_PATH          = ROOT / "data/challenge_analytical_field.csv"
SENSOR_PATH         = ROOT / "data/challenge_sensor_observations.csv"
ML_INPUT_NAME       = "adr1d_ml_challenge_input.csv"
NN_INPUT_NAME       = "adr1d_nn_challenge_input.csv"

EXPECTED_OUTPUT_KEYS = (
    "adr1d_ml_predictions",
    "adr1d_ml_metrics",
    "adr1d_nn_predictions",
    "adr1d_nn_scenario_metrics",
    "adr1d_nn_metrics",
)


class PreparationError(RuntimeError):
    """Represent a failure of the pre-inference preparation contract."""


def sha256(path):
    """
    Compute the SHA-256 digest of a file in bounded memory.

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
        If the requested file is absent.
    PreparationError
        If the top-level value is not an object.

    """
    if not path.exists():
        raise FileNotFoundError(path)
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise PreparationError(f"Expected a JSON object in {path}")
    return content


def require(condition, message):
    """
    Enforce one preparation condition.

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
    PreparationError
        If the condition is false.

    """
    if not condition:
        raise PreparationError(message)


def project_path(path_text):
    """
    Resolve a protocol path relative to the Activity 05 root.

    Parameters
    ----------
    path_text : str
        Relative or absolute path recorded in the protocol.

    Returns
    -------
    pathlib.Path
        Absolute normalized path.

    """
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def verify_record(metadata, label):
    """
    Verify one path-and-digest metadata record.

    Parameters
    ----------
    metadata : mapping
        Record containing `path` and `sha256`.
    label : str
        Human-readable artifact description.

    Returns
    -------
    pathlib.Path
        Verified absolute artifact path.

    Raises
    ------
    FileNotFoundError
        If the artifact is absent.
    PreparationError
        If its digest differs.

    """
    path = project_path(metadata["path"])
    if not path.exists():
        raise FileNotFoundError(path)
    require(sha256(path) == metadata["sha256"], f"Digest mismatch for {label}: {metadata['path']}")
    return path


def verify_preparation_state(protocol):
    """
    Verify the frozen challenge, canonical audit, APIs, and absent outputs.

    Parameters
    ----------
    protocol : dict
        Locked Activity 05 validation protocol.

    Returns
    -------
    dict
        Verified challenge and canonical-audit digests.

    Raises
    ------
    PreparationError
        If any protected state differs or an evaluation output already exists.

    """
    challenge_lock = load_json(CHALLENGE_LOCK_PATH)
    require(challenge_lock["status"] == "locked_before_model_inference", "Challenge set is not locked before inference")
    verified = 0
    for label, metadata in challenge_lock["challenge_files"].items():
        verify_record(metadata, label)
        verified += 1
    for label in ("challenge_manifest", "independent_validation"):
        verify_record(challenge_lock[label], label)
        verified += 1
    for label, metadata in challenge_lock["implementations"].items():
        verify_record(metadata, label)
        verified += 1
    state = challenge_lock["state_at_lock"]
    require(int(state["adr1d_ml_challenge_inference_runs"]) == 0 and int(state["adr1d_nn_challenge_inference_runs"]) == 0, "Challenge lock does not record zero model queries")

    canonical = load_json(CANONICAL_AUDIT)
    require(canonical["status"] == "canonical_results_reproduced", "Canonical audit is incomplete")
    require(int(canonical["model_access"]["adr1d_ml_challenge_inference_runs"]) == 0 and int(canonical["model_access"]["adr1d_nn_challenge_inference_runs"]) == 0, "Canonical audit reports challenge inference")

    for model_name in ("adr1d_ml", "adr1d_nn"):
        frozen = protocol["frozen_inputs"][model_name]
        for label in ("model", "manifest", "protocol", "inference_module"):
            verify_record(frozen[label], f"{model_name}.{label}")
    for label in ("base_feature_module", "physics_feature_module"):
        verify_record(protocol["frozen_inputs"]["adr1d_ml"][label], f"adr1d_ml.{label}")

    planned = protocol["planned_artifacts"]
    existing_outputs = [project_path(planned[key]) for key in EXPECTED_OUTPUT_KEYS if project_path(planned[key]).exists()]
    require(not existing_outputs, "Challenge-evaluation outputs already exist: " + ", ".join(str(path) for path in existing_outputs))
    require(not (ROOT / "results/.challenge_inference_staging").exists(), "Challenge-inference staging directory already exists")
    require(not (ROOT / "results/challenge_inference_incident.json").exists(), "A prior challenge-inference incident is recorded")
    return {"challenge_lock_sha256": sha256(CHALLENGE_LOCK_PATH), "canonical_reproduction_sha256": sha256(CANONICAL_AUDIT), "protected_challenge_components_verified": verified}


def load_module(module_name, metadata):
    """
    Import a frozen feature-building module after digest verification.

    Parameters
    ----------
    module_name : str
        Isolated import name.
    metadata : mapping
        Protocol record containing module path and digest.

    Returns
    -------
    module
        Imported trusted feature-building module.

    Raises
    ------
    PreparationError
        If an import specification cannot be created.

    """
    path = verify_record(metadata, module_name)
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise PreparationError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(specification)
    sys.path.insert(0, str(path.parent))
    try:
        specification.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def build_ml_input(protocol):
    """
    Construct one model-ready ADR1D-ML row per noisy sensor realization.

    Parameters
    ----------
    protocol : dict
        Locked Activity 05 validation protocol.

    Returns
    -------
    pandas.DataFrame
        Ordered table with 600 identifiers and 86 predictor columns.

    Raises
    ------
    PreparationError
        If realization counts, feature counts, or identifiers differ.

    """
    frozen       = protocol["frozen_inputs"]["adr1d_ml"]
    base_module  = load_module("locked_adr1d_base_features", frozen["base_feature_module"])
    physics      = load_module("locked_adr1d_physics_features", frozen["physics_feature_module"])
    scenarios    = pd.read_csv(SCENARIO_PATH)
    observations = pd.read_csv(SENSOR_PATH)
    identifiers  = observations.loc[:, ["scenario_id", "base_scenario_id", "replicate_id"]].drop_duplicates().reset_index(drop=True)
    require(len(identifiers) == 600 and identifiers["scenario_id"].nunique() == 600, "Expected 600 unique ADR1D-ML realization identifiers")

    base = scenarios.rename(columns={"scenario_id": "base_scenario_id"})
    instances = identifiers.merge(base, on="base_scenario_id", how="left", validate="many_to_one")
    require(not instances.isna().any().any(), "ADR1D-ML instance metadata contain missing values")
    model_table = base_module.build_table(instances, observations)
    physics_rows = pd.DataFrame([physics.build_physics_features(row) for _, row in model_table.iterrows()])
    model_table = pd.concat([model_table.reset_index(drop=True), physics_rows], axis=1)
    feature_columns = [name for name in model_table if name.startswith("feature_")]
    require(len(model_table) == 600 and model_table["scenario_id"].nunique() == 600, "ADR1D-ML feature table has invalid dimensions")
    require(len(feature_columns) == 86, f"Expected 86 ADR1D-ML predictors, found {len(feature_columns)}")
    return model_table.loc[:, ["scenario_id"] + feature_columns]


def build_nn_input(protocol):
    """
    Construct the ordered physical point table required by ADR1D-NN.

    Parameters
    ----------
    protocol : dict
        Locked Activity 05 validation protocol.

    Returns
    -------
    pandas.DataFrame
        Ordered table with 299,880 point rows and the nine public API inputs.

    Raises
    ------
    PreparationError
        If joins, dimensions, coordinates, or columns differ from the contract.

    """
    scenarios = pd.read_csv(SCENARIO_PATH)
    field     = pd.read_csv(FIELD_PATH, usecols=("scenario_id", "time_s", "x_m"))
    scenarios["effective_velocity_m_s"] = scenarios["velocity_m_s"] / scenarios["retardation_factor"]
    scenarios["effective_dispersion_m2_s"] = scenarios["dispersion_m2_s"] / scenarios["retardation_factor"]
    metadata_columns = [
        "scenario_id",
        "domain_length_m",
        "final_time_s",
        "effective_velocity_m_s",
        "effective_dispersion_m2_s",
        "decay_rate_s_1",
        "source_start_s",
        "source_duration_s",
    ]
    field["_order"] = range(len(field))
    table = field.merge(scenarios.loc[:, metadata_columns], on="scenario_id", how="left", validate="many_to_one", sort=False)
    table = table.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    input_columns = list(load_json(project_path(protocol["frozen_inputs"]["adr1d_nn"]["protocol"]["path"]))["feature_contract"]["input_columns"])
    expected_columns = ["scenario_id"] + input_columns
    require(len(table) == 299880 and table["scenario_id"].nunique() == 120, "ADR1D-NN input table has invalid dimensions")
    require(not table[expected_columns].isna().any().any(), "ADR1D-NN input table contains missing values")
    return table.loc[:, expected_columns]


def write_input_table(table, path):
    """
    Serialize one prepared model input under the frozen decimal contract.

    Parameters
    ----------
    table : pandas.DataFrame
        Ordered model-ready input table.
    path : pathlib.Path
        Temporary CSV destination.

    Returns
    -------
    dict
        Row count, column count, byte size, and SHA-256 digest.

    """
    table.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")
    return {"rows": int(len(table)), "columns": int(len(table.columns)), "bytes": path.stat().st_size, "sha256": sha256(path)}


def prepare_input_files(protocol, directory):
    """
    Build and serialize both locked challenge input tables.

    Parameters
    ----------
    protocol : dict
        Locked Activity 05 validation protocol.
    directory : pathlib.Path
        Existing temporary output directory.

    Returns
    -------
    dict
        Input paths, metadata, and in-memory tables.

    """
    ml_table = build_ml_input(protocol)
    nn_table = build_nn_input(protocol)
    ml_path  = directory / ML_INPUT_NAME
    nn_path  = directory / NN_INPUT_NAME
    ml_metadata = write_input_table(ml_table, ml_path)
    nn_metadata = write_input_table(nn_table, nn_path)
    return {
        "adr1d_ml": {"path": ml_path, "metadata": ml_metadata, "table": ml_table},
        "adr1d_nn": {"path": nn_path, "metadata": nn_metadata, "table": nn_table},
    }


def comparable_metadata(prepared):
    """
    Extract location-independent metadata from prepared input records.

    Parameters
    ----------
    prepared : dict
        Result returned by :func:`prepare_input_files`.

    Returns
    -------
    dict
        Stable metadata keyed by evaluated model.

    """
    return {key: value["metadata"] for key, value in prepared.items()}


def main():
    """
    Reproduce exact challenge inputs twice and write the evaluation lock.

    Returns
    -------
    None
        The function writes `results/challenge_evaluation_lock.json`.

    Raises
    ------
    FileExistsError
        If the evaluation lock already exists.
    PreparationError
        If preparation is not deterministic or protected state differs.

    """
    if EVALUATION_LOCK.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation lock: {EVALUATION_LOCK}")
    if not EVALUATOR_PATH.exists():
        raise FileNotFoundError(EVALUATOR_PATH)
    protocol = load_json(PROTOCOL_PATH)
    guard    = verify_preparation_state(protocol)

    with tempfile.TemporaryDirectory(prefix="adr1d-challenge-input-a-") as first_directory, tempfile.TemporaryDirectory(prefix="adr1d-challenge-input-b-") as second_directory:
        first  = prepare_input_files(protocol, Path(first_directory))
        second = prepare_input_files(protocol, Path(second_directory))
        first_metadata  = comparable_metadata(first)
        second_metadata = comparable_metadata(second)
        require(first_metadata == second_metadata, "Repeated challenge input construction is not byte-identical")

    frozen_ml = protocol["frozen_inputs"]["adr1d_ml"]
    frozen_nn = protocol["frozen_inputs"]["adr1d_nn"]
    ml_protocol = load_json(project_path(frozen_ml["protocol"]["path"]))
    ml_decision_threshold = float(ml_protocol["models"]["decay_resolvability"]["decision_threshold"])
    lock = {
        "lock_version": "1.0.0",
        "status": "locked_before_challenge_inference",
        "locked_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "protocol": {"path": "configs/validation_protocol.json", "sha256": sha256(PROTOCOL_PATH), "protocol_id": protocol["protocol_id"], "protocol_version": protocol["protocol_version"]},
        "challenge_guard": guard,
        "prepared_inputs": first_metadata,
        "deterministic_rebuilds_verified": 2,
        "locked_code": {
            "preparation": {"path": "src/prepare_challenge_evaluation.py", "sha256": sha256(Path(__file__))},
            "evaluation": {"path": "src/evaluate_locked_challenge.py", "sha256": sha256(EVALUATOR_PATH)},
        },
        "frozen_models": {
            "adr1d_ml": {"model_sha256": frozen_ml["model"]["sha256"], "inference_module_sha256": frozen_ml["inference_module"]["sha256"]},
            "adr1d_nn": {"model_sha256": frozen_nn["model"]["sha256"], "inference_module_sha256": frozen_nn["inference_module"]["sha256"]},
        },
        "planned_outputs": {key: protocol["planned_artifacts"][key] for key in EXPECTED_OUTPUT_KEYS},
        "state_at_lock": {
            "adr1d_ml_model_loaded": False,
            "adr1d_nn_model_loaded": False,
            "adr1d_ml_challenge_inference_runs": 0,
            "adr1d_nn_challenge_inference_runs": 0,
            "acceptance_criteria_inspected_against_challenge_results": False,
        },
        "execution_contract": {
            "permitted_persisted_inference_runs_per_model": 1,
            "sensor_replicates_nested_within_base_scenario": True,
            "base_scenario_is_independent_unit": True,
            "bootstrap_resamples": int(protocol["statistical_analysis"]["bootstrap_resamples"]),
            "bootstrap_seed": int(protocol["statistical_analysis"]["bootstrap_seed"]),
            "adr1d_ml_decision_threshold": ml_decision_threshold,
            "model_changes_allowed": False,
            "threshold_tuning_allowed": False,
        },
        "failure_policy": {
            "retain_staging_outputs_after_started_inference": True,
            "write_incident_record_after_started_inference": True,
            "automatic_rerun_after_failure_allowed": False,
        },
    }
    EVALUATION_LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": lock["status"], "adr1d_ml_input": first_metadata["adr1d_ml"], "adr1d_nn_input": first_metadata["adr1d_nn"], "challenge_inference_runs": 0, "evaluation_lock": str(EVALUATION_LOCK)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
