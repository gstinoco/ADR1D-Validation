"""
================================================================================
ADR1D Validation: Final Deliverable Audit
================================================================================

Audit the complete ADR1D numerical-validation deliverable without loading or
executing either predictive model. The audit verifies evidence status, hashes,
figure traceability, report metadata, and the absence of LaTeX build debris.

Main Operations
---------------
1. Verify the locked protocol, challenge, model, and numerical evidence.
2. Recompute every figure hash recorded in the portable figure manifest.
3. Inspect the final PDF metadata and expected page count with Poppler.
4. Confirm that the results directory contains no temporary LaTeX artifacts.
5. Write a portable manifest marking the product ready for investigator review.

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
import shutil
import subprocess
from pathlib import Path


ROOT                    = Path(__file__).resolve().parents[1]
RESULTS                 = ROOT / "results"
PROTOCOL_PATH           = ROOT / "configs/validation_protocol.json"
PROTOCOL_LOCK           = RESULTS / "validation_protocol_lock.json"
CHALLENGE_MANIFEST      = RESULTS / "challenge_manifest.json"
CHALLENGE_VALIDATION    = RESULTS / "challenge_validation.json"
CHALLENGE_LOCK          = RESULTS / "challenge_lock.json"
EVALUATION_LOCK         = RESULTS / "challenge_evaluation_lock.json"
CHALLENGE_SCENARIOS     = ROOT / "data/challenge_scenarios.csv"
CHALLENGE_FIELD         = ROOT / "data/challenge_analytical_field.csv"
CHALLENGE_SENSORS       = ROOT / "data/challenge_sensor_observations.csv"
CANONICAL_REPRODUCTION  = RESULTS / "canonical_reproduction.json"
RESULT_AUDIT            = RESULTS / "challenge_result_validation.json"
ML_METRICS              = RESULTS / "adr1d_ml_challenge_metrics.json"
ML_PREDICTIONS          = RESULTS / "adr1d_ml_challenge_predictions.csv"
NN_METRICS              = RESULTS / "adr1d_nn_challenge_metrics.json"
NN_PREDICTIONS          = RESULTS / "adr1d_nn_challenge_predictions.csv"
NN_SCENARIOS            = RESULTS / "adr1d_nn_challenge_scenarios.csv"
NUMERICAL_CASES         = RESULTS / "numerical_reference_cases.csv"
NUMERICAL_METRICS       = RESULTS / "numerical_reference_metrics.json"
COMPUTATIONAL_COST      = RESULTS / "computational_cost.json"
FIGURE_MANIFEST         = RESULTS / "figure_manifest.json"
REPORT_TEX              = RESULTS / "adr1d_validacion_numerica.tex"
REPORT_PDF              = RESULTS / "adr1d_validacion_numerica.pdf"
DEFAULT_OUTPUT          = RESULTS / "final_validation_manifest.json"
PROTOCOL_SHA256         = "340ed49f3b3054a4b343b99e2863a6a157634bd716274949c6631563495231d9"
MODEL_SHA256            = {
    "adr1d_ml": "6890df1b30f5572611e5fcdc0d80a4f923e3877f93594b642c0d084c3f361cea",
    "adr1d_nn": "9cc762b5e5f45d368f52a2da76695f4a762f4d70226db65357a1791b59762918",
}
LATEX_TEMPORARY_SUFFIXES = {".aux", ".fdb_latexmk", ".fls", ".log", ".out", ".toc"}


class FinalDeliverableError(RuntimeError):
    """Represent an incomplete or internally inconsistent final deliverable."""


def sha256(path):
    """
    Compute the SHA-256 digest of one file in bounded memory.

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
        Parsed top-level object.

    Raises
    ------
    FinalDeliverableError
        If the top-level JSON value is not an object.

    """
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise FinalDeliverableError(f"Expected a JSON object in {path}")
    return content


def require(condition, message):
    """
    Enforce one final-deliverable invariant.

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
    FinalDeliverableError
        If the condition is false.

    """
    if not bool(condition):
        raise FinalDeliverableError(message)


def parse_arguments():
    """
    Parse the manifest destination and overwrite policy.

    Returns
    -------
    argparse.Namespace
        Validated command-line arguments.

    """
    parser = argparse.ArgumentParser(description="Audit the final ADR1D numerical-validation deliverable.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination for the portable final manifest.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing final manifest.")
    return parser.parse_args()


def portable_path(path):
    """
    Represent an artifact relative to the validation root when possible.

    Parameters
    ----------
    path : pathlib.Path
        Artifact path to serialize.

    Returns
    -------
    str
        POSIX-style project-relative path.

    """
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def artifact_record(path):
    """
    Describe one immutable deliverable artifact.

    Parameters
    ----------
    path : pathlib.Path
        Existing file to describe.

    Returns
    -------
    dict
        Portable path, byte size, and SHA-256 digest.

    """
    require(path.is_file(), f"Required artifact is missing: {path}")
    return {"path": portable_path(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def parse_pdfinfo(path):
    """
    Read PDF metadata with the Poppler `pdfinfo` command.

    Parameters
    ----------
    path : pathlib.Path
        Final report PDF.

    Returns
    -------
    dict
        Metadata fields parsed from `pdfinfo` output.

    Raises
    ------
    FinalDeliverableError
        If Poppler is unavailable or the command fails.

    """
    executable = shutil.which("pdfinfo")
    require(executable is not None, "Poppler pdfinfo is required to audit the final report")
    process = subprocess.run([executable, str(path)], capture_output=True, text=True, check=False)
    require(process.returncode == 0, f"pdfinfo failed for {path}: {process.stderr.strip()}")
    metadata = {}
    for line in process.stdout.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
    return metadata


def validate_protocol_reports(reports):
    """
    Verify that all confirmatory reports reference the locked protocol.

    Parameters
    ----------
    reports : dict
        Named parsed JSON reports.

    Returns
    -------
    None
        The function returns after all available protocol hashes agree.

    """
    require(sha256(PROTOCOL_PATH) == PROTOCOL_SHA256, "The locked protocol hash changed")
    for name in ["ml", "nn", "numerical", "cost"]:
        require(reports[name].get("protocol_sha256") == PROTOCOL_SHA256, f"{name} report references a different protocol")


def validate_evidence(reports):
    """
    Validate final status, adequacy, and immutable-model assertions.

    Parameters
    ----------
    reports : dict
        Named parsed JSON reports.

    Returns
    -------
    dict
        Concise evidence summary for the final manifest.

    """
    require(reports["challenge_validation"].get("status") == "passed_before_model_inference", "Challenge-set validation did not pass before inference")
    require(reports["challenge_lock"].get("status") == "locked_before_model_inference", "Challenge lock is invalid")
    require(reports["evaluation_lock"].get("status") == "locked_before_challenge_inference", "Evaluation lock is invalid")
    require(reports["canonical"].get("status") == "canonical_results_reproduced", "Canonical results were not reproduced")
    require(reports["audit"].get("status") == "passed", "Independent result audit did not pass")
    require(reports["ml"].get("status") == "challenge_evaluation_complete", "ADR1D-ML challenge evidence is incomplete")
    require(reports["nn"].get("status") == "challenge_evaluation_complete", "ADR1D-NN challenge evidence is incomplete")
    require(reports["ml"]["inference"].get("persisted_challenge_runs") == 1, "ADR1D-ML challenge inference count differs from one")
    require(reports["nn"]["inference"].get("persisted_challenge_runs") == 1, "ADR1D-NN challenge inference count differs from one")
    require(reports["ml"]["inference"].get("post_challenge_tuning_performed") is False, "ADR1D-ML reports post-challenge tuning")
    require(reports["nn"]["inference"].get("post_challenge_tuning_performed") is False, "ADR1D-NN reports post-challenge tuning")
    require(reports["ml"]["evidence"]["adequacy"].get("overall_outcome") == "meets_all_pre_specified_criteria", "ADR1D-ML adequacy outcome changed")
    require(reports["nn"]["evidence"]["adequacy"].get("overall_outcome") == "meets_all_pre_specified_criteria", "ADR1D-NN adequacy outcome changed")

    numerical = reports["numerical"]
    require(numerical.get("status") == "numerical_reference_complete", "Numerical-reference report is incomplete")
    require(numerical["acceptance"].get("overall_outcome") == "mixed_evidence", "Numerical-reference outcome must retain mixed evidence")
    failed = numerical["acceptance"].get("scenarios_not_meeting_all_criteria")
    require(failed == ["A5-CH-0117", "A5-CH-0087"], "Numerical-reference failure cases changed")

    cost = reports["cost"]
    require(cost.get("status") == "complete", "Computational-cost report is incomplete")
    require(cost["scope"].get("challenge_model_inference_runs") == 0, "Cost benchmark repeated challenge-model inference")
    require(cost["scope"].get("persisted_model_predictions_created") == 0, "Cost benchmark persisted new model predictions")
    require(cost["integrity"].get("protected_artifacts_unchanged") is True, "Protected artifacts changed during cost measurement")
    require(len(cost.get("measurements", {})) == 11, "Computational-cost workload count differs from 11")
    require(all(item.get("deterministic_repetitions") is True for item in cost["measurements"].values()), "One or more cost workloads were nondeterministic")

    audit_checks = reports["audit"]["checks"]
    return {
        "adr1d_ml": {
            "adequacy": "meets_all_pre_specified_criteria",
            "challenge_runs": 1,
            "model_sha256": MODEL_SHA256["adr1d_ml"],
        },
        "adr1d_nn": {
            "adequacy": "meets_all_pre_specified_criteria",
            "challenge_runs": 1,
            "model_sha256": MODEL_SHA256["adr1d_nn"],
        },
        "independent_result_audit": {
            "maximum_absolute_difference": audit_checks["maximum_absolute_difference"],
            "numerical_values_compared": audit_checks["numerical_values_compared"],
            "successful_checks": audit_checks["successful_checks"],
            "status": "passed",
        },
        "numerical_reference": {
            "outcome": "mixed_evidence",
            "scenarios_meeting_all_criteria": numerical["acceptance"]["scenarios_meeting_all_criteria"],
            "scenarios_not_meeting_all_criteria": failed,
            "scenarios_total": numerical["acceptance"]["scenarios_total"],
        },
        "computational_cost": {
            "challenge_model_inference_runs": 0,
            "protected_artifacts_unchanged": True,
            "timed_repetitions_per_workload": 7,
            "workloads": 11,
        },
    }


def validate_figures(manifest):
    """
    Verify every portable source and figure digest in the figure manifest.

    Parameters
    ----------
    manifest : dict
        Parsed figure manifest.

    Returns
    -------
    list of pathlib.Path
        Verified PNG and PDF paths.

    """
    require(manifest.get("status") == "complete", "Figure manifest is incomplete")
    require(manifest["generator"].get("model_loading_or_inference") is False, "Figure generation reports model access")
    require(len(manifest.get("figures", {})) == 8, "Figure manifest must contain eight artifacts")
    for relative, expected in manifest.get("sources", {}).items():
        path = ROOT / relative
        require(path.is_file() and sha256(path) == expected, f"Figure source hash mismatch: {relative}")
    verified = []
    for relative, record in manifest["figures"].items():
        path = ROOT / relative
        require(path.is_file(), f"Figure artifact is missing: {relative}")
        require(path.stat().st_size == int(record["bytes"]), f"Figure size mismatch: {relative}")
        require(sha256(path) == record["sha256"], f"Figure hash mismatch: {relative}")
        verified.append(path)
    require(sum(path.suffix == ".pdf" for path in verified) == 4, "Expected four vector figures")
    require(sum(path.suffix == ".png" for path in verified) == 4, "Expected four raster figures")
    return sorted(verified)


def validate_report():
    """
    Verify final report structure, metadata, and build cleanliness.

    Returns
    -------
    dict
        Final report metadata and artifact records.

    """
    require(REPORT_TEX.is_file() and REPORT_PDF.is_file(), "Final report source or PDF is missing")
    source = REPORT_TEX.read_text(encoding="utf-8")
    require("Actividad 05" not in source and "Actividad 5" not in source, "Standalone report contains internal activity numbering")
    require(source.count("\\includegraphics") == 4, "Final report must include four scientific figures")
    metadata = parse_pdfinfo(REPORT_PDF)
    page_count = int(metadata.get("Pages", "0"))
    require(page_count >= 15, "Final report must contain at least 15 pages")
    require("A4" in metadata.get("Page size", ""), "Final report page size is not A4")
    require(metadata.get("Title") == "Validación numérica independiente de ADR1D-ML y ADR1D-NN", "Final report title metadata changed")
    require(metadata.get("CreationDate") == "Tue Jul 21 00:00:00 2026 CST", "Final report does not use the fixed build date")
    temporary = [path for path in RESULTS.iterdir() if path.suffix in LATEX_TEMPORARY_SUFFIXES]
    require(not temporary, "Temporary LaTeX files remain in results: " + ", ".join(path.name for path in temporary))
    return {
        "page_count": page_count,
        "page_size": "A4",
        "pdf": artifact_record(REPORT_PDF),
        "source": artifact_record(REPORT_TEX),
        "title": metadata["Title"],
    }


def main():
    """
    Audit the final product and write its portable manifest.

    Returns
    -------
    None
        The final manifest and a concise JSON summary are written.

    """
    arguments = parse_arguments()
    require(arguments.overwrite or not arguments.output.exists(), f"Output already exists; use --overwrite to replace {arguments.output}")
    report_paths = {
        "challenge_validation": CHALLENGE_VALIDATION,
        "challenge_lock": CHALLENGE_LOCK,
        "evaluation_lock": EVALUATION_LOCK,
        "canonical": CANONICAL_REPRODUCTION,
        "audit": RESULT_AUDIT,
        "ml": ML_METRICS,
        "nn": NN_METRICS,
        "numerical": NUMERICAL_METRICS,
        "cost": COMPUTATIONAL_COST,
    }
    reports = {name: load_json(path) for name, path in report_paths.items()}
    validate_protocol_reports(reports)
    evidence = validate_evidence(reports)
    figures = validate_figures(load_json(FIGURE_MANIFEST))
    report = validate_report()

    core_paths = [
        PROTOCOL_PATH, PROTOCOL_LOCK, CHALLENGE_MANIFEST, CHALLENGE_VALIDATION,
        CHALLENGE_LOCK, EVALUATION_LOCK, CHALLENGE_SCENARIOS, CHALLENGE_FIELD,
        CHALLENGE_SENSORS, CANONICAL_REPRODUCTION, RESULT_AUDIT, ML_METRICS,
        ML_PREDICTIONS, NN_METRICS, NN_PREDICTIONS, NN_SCENARIOS, NUMERICAL_CASES,
        NUMERICAL_METRICS, COMPUTATIONAL_COST, FIGURE_MANIFEST,
    ]
    source_paths = sorted((ROOT / "src").glob("*.py"))
    manifest = {
        "artifact_groups": {
            "core_evidence": [artifact_record(path) for path in core_paths],
            "figures": [artifact_record(path) for path in figures],
            "source_code": [artifact_record(path) for path in source_paths],
        },
        "closure": {
            "activity_remains_active": True,
            "explicit_investigator_approval_required": True,
            "next_activity_activated": False,
        },
        "evidence": evidence,
        "protocol": {
            "id": "ADR1D-NUMERICAL-VALIDATION",
            "sha256": PROTOCOL_SHA256,
            "version": "1.0.0",
        },
        "report": report,
        "status": "ready_for_investigator_review",
        "validator": artifact_record(Path(__file__).resolve()),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "figures_verified": len(figures),
        "next_activity_activated": False,
        "output": str(arguments.output.resolve()),
        "report_pages": report["page_count"],
        "source_modules_verified": len(source_paths),
        "status": manifest["status"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
