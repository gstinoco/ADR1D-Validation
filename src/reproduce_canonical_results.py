"""
================================================================================
ADR1D Validation: Canonical Result Reproduction
================================================================================

Reproduce the published ADR1D-ML and ADR1D-NN test validations while preserving
their source artifacts and keeping the locked challenge set unopened.

Main Operations
---------------
1. Verify the challenge lock and frozen canonical models.
2. Run each source project's independent validator with temporary output paths.
3. Compare reproduced reports byte for byte with the published evidence.
4. Persist one Activity 05 record describing the canonical audit.

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
import contextlib
import hashlib
import importlib.util
import io
import json
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT                = Path(__file__).resolve().parents[1]
ACTIVITIES_ROOT     = ROOT.parent
PROTOCOL_PATH       = ROOT / "configs/validation_protocol.json"
CHALLENGE_LOCK_PATH = ROOT / "results/challenge_lock.json"
OUTPUT_PATH         = ROOT / "results/canonical_reproduction.json"

ML_ROOT             = ACTIVITIES_ROOT / "03_modelos_ml_parametros"
ML_VALIDATOR        = ML_ROOT / "scripts/validate_final_models.py"
ML_SOURCE_REPORT    = ML_ROOT / "results/final_model_validation.json"
ML_MODEL            = ML_ROOT / "models/adr1d_parameter_models.joblib"
ML_MODEL_MANIFEST   = ML_ROOT / "models/model_manifest.json"
ML_PROTOCOL         = ML_ROOT / "results/final_model_protocol.json"
ML_METRICS          = ML_ROOT / "results/final_test_metrics.json"
ML_PREDICTIONS      = ML_ROOT / "results/final_test_predictions.csv"
ML_BASE_TABLE       = ML_ROOT / "results/adr1d_modeling_table.csv"
ML_DECAY_TABLE      = ML_ROOT / "results/adr1d_decay_detectability_table.csv"
ML_TRAINING_SCRIPT  = ML_ROOT / "scripts/train_and_evaluate_final_models.py"

NN_ROOT             = ACTIVITIES_ROOT / "04_redes_neuronales_transporte"
NN_VALIDATOR        = NN_ROOT / "src/validate_final_results.py"
NN_SOURCE_REPORT    = NN_ROOT / "results/final_model_validation.json"
NN_MODEL            = NN_ROOT / "models/adr1d_nn.pt"
NN_MODEL_MANIFEST   = NN_ROOT / "models/model_manifest.json"
NN_PROTOCOL         = NN_ROOT / "configs/final_model_protocol.json"
NN_EVALUATION_LOCK  = NN_ROOT / "configs/final_evaluation_lock.json"
NN_METRICS          = NN_ROOT / "results/final_test_metrics.json"
NN_SCENARIOS        = NN_ROOT / "results/final_test_scenarios.csv"
NN_RESULT_MANIFEST  = NN_ROOT / "results/final_result_manifest.json"
NN_FIELD_FIGURE     = NN_ROOT / "results/final_test_fields.png"
NN_PROFILE_FIGURE   = NN_ROOT / "results/final_test_profiles.png"
NN_EVALUATION_SCRIPT = NN_ROOT / "src/evaluate_final_test.py"

DATASET_ROOT        = ACTIVITIES_ROOT / "02_datasets"
CANONICAL_SCENARIOS = DATASET_ROOT / "metadata/synthetic_adr1d_scenarios.csv"
CANONICAL_FIELD     = DATASET_ROOT / "data_processed/synthetic_adr1d_field.csv"


class ReproductionError(RuntimeError):
    """Represent a failure of the canonical reproduction contract."""


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
        Parsed top-level object.

    Raises
    ------
    FileNotFoundError
        If the file is absent.
    ReproductionError
        If the top-level JSON value is not an object.

    """
    if not path.exists():
        raise FileNotFoundError(path)
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ReproductionError(f"Expected a JSON object in {path}")
    return content


def require(condition, message):
    """
    Enforce one canonical-reproduction condition.

    Parameters
    ----------
    condition : bool
        Condition that must evaluate to true.
    message : str
        Failure description.

    Returns
    -------
    None
        The function returns only when the condition is satisfied.

    Raises
    ------
    ReproductionError
        If the required condition is false.

    """
    if not condition:
        raise ReproductionError(message)


def activity_path(logical_path):
    """
    Resolve an Activity 05 logical path.

    Parameters
    ----------
    logical_path : str
        Path stored in a challenge record relative to the activity root.

    Returns
    -------
    pathlib.Path
        Absolute filesystem path.

    """
    return (ROOT / logical_path).resolve()


def verify_challenge_lock():
    """
    Verify that the challenge remains locked and internally unchanged.

    Returns
    -------
    dict
        Lock digest, artifact count, and zero-query state.

    Raises
    ------
    ReproductionError
        If status, hashes, or pre-inference counters differ from the lock.

    """
    lock = load_json(CHALLENGE_LOCK_PATH)
    require(lock["status"] == "locked_before_model_inference", "Challenge set is not in its locked state")
    verified = 0
    for metadata in lock["challenge_files"].values():
        path = activity_path(metadata["path"])
        require(path.exists() and sha256(path) == metadata["sha256"], f"Locked challenge artifact differs: {metadata['path']}")
        verified += 1
    for key in ("challenge_manifest", "independent_validation"):
        metadata = lock[key]
        path = activity_path(metadata["path"])
        require(path.exists() and sha256(path) == metadata["sha256"], f"Locked challenge record differs: {metadata['path']}")
        verified += 1
    for metadata in lock["implementations"].values():
        path = activity_path(metadata["path"])
        require(path.exists() and sha256(path) == metadata["sha256"], f"Locked challenge implementation differs: {metadata['path']}")
        verified += 1
    state = lock["state_at_lock"]
    require(int(state["adr1d_ml_challenge_inference_runs"]) == 0 and int(state["adr1d_nn_challenge_inference_runs"]) == 0, "Challenge lock does not record zero model queries")
    require(int(state["traditional_numerical_solver_runs"]) == 0, "Challenge lock does not record zero numerical-solver runs")
    return {"path": "results/challenge_lock.json", "sha256": sha256(CHALLENGE_LOCK_PATH), "locked_artifacts_verified": verified, "challenge_inference_runs_before_canonical_audit": 0}


def load_validator(module_name, path, search_directory):
    """
    Load a trusted source-project validator under an isolated module name.

    Parameters
    ----------
    module_name : str
        Unique import name used only during this process.
    path : pathlib.Path
        Validator module to execute.
    search_directory : pathlib.Path
        Directory containing its local imports.

    Returns
    -------
    module
        Imported validator module.

    Raises
    ------
    FileNotFoundError
        If the validator is absent.
    ReproductionError
        If an import specification cannot be created.

    """
    if not path.exists():
        raise FileNotFoundError(path)
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ReproductionError(f"Unable to create an import specification for {path}")
    module = importlib.util.module_from_spec(specification)
    sys.path.insert(0, str(search_directory))
    try:
        specification.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def run_validator(module, report_path, cache_path=None):
    """
    Execute a source validator while redirecting all writable paths.

    Parameters
    ----------
    module : module
        Imported source-project validator exposing `main` and `REPORT_PATH`.
    report_path : pathlib.Path
        Temporary report destination.
    cache_path : pathlib.Path, optional
        Temporary cache location for validators that remove their import cache.

    Returns
    -------
    tuple of dict and str
        Parsed reproduced report and captured standard output.

    Raises
    ------
    ReproductionError
        If the module contract or generated report is invalid.

    """
    require(callable(getattr(module, "main", None)), "Source validator does not expose main()")
    require(hasattr(module, "REPORT_PATH"), "Source validator does not expose REPORT_PATH")
    module.REPORT_PATH = report_path
    if cache_path is not None and hasattr(module, "CACHE_PATH"):
        module.CACHE_PATH = cache_path
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        module.main()
    require(report_path.exists(), "Source validator did not write its redirected report")
    report = load_json(report_path)
    require(report.get("status") == "ok", "Source validator did not report status ok")
    return report, captured.getvalue()


def snapshot(paths):
    """
    Capture immutable content digests for source artifacts.

    Parameters
    ----------
    paths : iterable of pathlib.Path
        Existing artifacts that must remain unchanged.

    Returns
    -------
    dict
        Absolute path strings mapped to SHA-256 digests.

    """
    values = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        values[str(path.resolve())] = sha256(path)
    return values


def component_record(name, validator_path, source_report_path, reproduced_path, report, source_paths):
    """
    Build one audited canonical-reproduction component record.

    Parameters
    ----------
    name : str
        Stable component identifier.
    validator_path : pathlib.Path
        Source validator used for independent reproduction.
    source_report_path : pathlib.Path
        Published validation report being reproduced.
    reproduced_path : pathlib.Path
        Temporary report generated during this audit.
    report : dict
        Parsed reproduced validation content.
    source_paths : iterable of pathlib.Path
        Canonical artifacts read by the component.

    Returns
    -------
    dict
        Digests, byte comparison, inference count, and reproduced evidence.

    """
    source_report = load_json(source_report_path)
    require(report == source_report, f"{name} reproduced report content differs from published evidence")
    return {
        "status": "reproduced",
        "validator": {"path": str(validator_path.relative_to(ACTIVITIES_ROOT)), "sha256": sha256(validator_path)},
        "source_validation_report": {"path": str(source_report_path.relative_to(ACTIVITIES_ROOT)), "sha256": sha256(source_report_path)},
        "reproduced_report_sha256": sha256(reproduced_path),
        "byte_identical_to_source_report": reproduced_path.read_bytes() == source_report_path.read_bytes(),
        "canonical_artifacts": {str(path.relative_to(ACTIVITIES_ROOT)): sha256(path) for path in source_paths},
        "canonical_inference_runs": 1,
        "challenge_inference_runs": 0,
        "reproduced_evidence": report,
    }


def main():
    """
    Reproduce both canonical validations and write one immutable audit record.

    Returns
    -------
    None
        The function writes `results/canonical_reproduction.json` and prints it.

    Raises
    ------
    FileExistsError
        If the canonical audit record already exists.
    ReproductionError
        If source evidence, hashes, or reproduced reports differ.

    """
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite canonical audit: {OUTPUT_PATH}")
    protocol       = load_json(PROTOCOL_PATH)
    challenge_lock = verify_challenge_lock()
    ml_artifacts   = (ML_MODEL, ML_MODEL_MANIFEST, ML_PROTOCOL, ML_METRICS, ML_PREDICTIONS, ML_BASE_TABLE, ML_DECAY_TABLE, ML_TRAINING_SCRIPT)
    nn_artifacts   = (NN_MODEL, NN_MODEL_MANIFEST, NN_PROTOCOL, NN_EVALUATION_LOCK, NN_METRICS, NN_SCENARIOS, NN_RESULT_MANIFEST, NN_FIELD_FIGURE, NN_PROFILE_FIGURE, NN_EVALUATION_SCRIPT, CANONICAL_SCENARIOS, CANONICAL_FIELD)
    ml_paths       = (ML_VALIDATOR, ML_SOURCE_REPORT) + ml_artifacts
    nn_paths       = (NN_VALIDATOR, NN_SOURCE_REPORT) + nn_artifacts
    before         = snapshot(ml_paths + nn_paths)
    require(sha256(ML_MODEL) == protocol["frozen_inputs"]["adr1d_ml"]["model"]["sha256"], "ADR1D-ML model differs from the frozen protocol")
    require(sha256(NN_MODEL) == protocol["frozen_inputs"]["adr1d_nn"]["model"]["sha256"], "ADR1D-NN model differs from the frozen protocol")

    with tempfile.TemporaryDirectory(prefix="adr1d-canonical-audit-") as temporary_directory:
        temporary = Path(temporary_directory)
        ml_output = temporary / "adr1d_ml_validation.json"
        nn_output = temporary / "adr1d_nn_validation.json"
        ml_module = load_validator("adr1d_ml_canonical_validator", ML_VALIDATOR, ML_VALIDATOR.parent)
        ml_report, _ = run_validator(ml_module, ml_output)
        nn_module = load_validator("adr1d_nn_canonical_validator", NN_VALIDATOR, NN_VALIDATOR.parent)
        nn_report, _ = run_validator(nn_module, nn_output, temporary / "nn_cache")
        ml_record = component_record("ADR1D-ML", ML_VALIDATOR, ML_SOURCE_REPORT, ml_output, ml_report, ml_artifacts)
        nn_record = component_record("ADR1D-NN", NN_VALIDATOR, NN_SOURCE_REPORT, nn_output, nn_report, nn_artifacts)

    after = snapshot(ml_paths + nn_paths)
    require(before == after, "A source artifact changed during canonical reproduction")
    challenge_after = verify_challenge_lock()
    require(challenge_lock == challenge_after, "Challenge artifacts changed during canonical reproduction")

    result = {
        "report_version": "1.0.0",
        "status": "canonical_results_reproduced",
        "executed_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "protocol": {"protocol_id": protocol["protocol_id"], "protocol_version": protocol["protocol_version"], "sha256": sha256(PROTOCOL_PATH)},
        "challenge_guard": challenge_lock,
        "adr1d_ml": ml_record,
        "adr1d_nn": nn_record,
        "source_artifacts_unchanged": True,
        "challenge_artifacts_unchanged": True,
        "model_access": {
            "adr1d_ml_canonical_model_loads": 1,
            "adr1d_nn_canonical_model_loads": 1,
            "adr1d_ml_canonical_inference_runs": 1,
            "adr1d_nn_canonical_inference_runs": 1,
            "adr1d_ml_challenge_inference_runs": 0,
            "adr1d_nn_challenge_inference_runs": 0,
        },
        "software": {"python": platform.python_version(), "implementation": platform.python_implementation()},
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
