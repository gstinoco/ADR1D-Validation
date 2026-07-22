"""
================================================================================
ADR1D-Validation: Public Release Audit
================================================================================

Audit the public technical-report package without loading or executing either
predictive model. The script checks the scientific evidence, report, figures,
licenses, citation metadata, portable paths, and the machine-readable release
manifest.

Main Operations
---------------
1. Create or verify the scientific-artifact integrity manifest.
2. Run the original final-deliverable audit in a temporary directory.
3. Check public metadata, licenses, row counts, and selected reported values.
4. Reject local absolute paths and incomplete publication placeholders.

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
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT             = Path(__file__).resolve().parents[1]
RESULTS          = ROOT / "results"
RELEASE_MANIFEST = RESULTS / "release_manifest.json"
REPORT_TEX_ES    = RESULTS / "adr1d_validacion_numerica.tex"
REPORT_PDF_ES    = RESULTS / "adr1d_validacion_numerica.pdf"
REPORT_TEX_EN    = RESULTS / "adr1d_numerical_validation_en.tex"
REPORT_PDF_EN    = RESULTS / "adr1d_numerical_validation_en.pdf"
REPOSITORY_URL   = "https://github.com/gstinoco/ADR1D-Validation"
TEXT_SUFFIXES    = {".cff", ".json", ".md", ".py", ".tex", ".txt"}

CORE_EVIDENCE = [
    ROOT / "configs/validation_protocol.json",
    RESULTS / "validation_protocol_lock.json",
    RESULTS / "challenge_manifest.json",
    RESULTS / "challenge_validation.json",
    RESULTS / "challenge_lock.json",
    RESULTS / "challenge_evaluation_lock.json",
    ROOT / "data/challenge_scenarios.csv",
    ROOT / "data/challenge_analytical_field.csv",
    ROOT / "data/challenge_sensor_observations.csv",
    RESULTS / "canonical_reproduction.json",
    RESULTS / "challenge_result_validation.json",
    RESULTS / "adr1d_ml_challenge_metrics.json",
    RESULTS / "adr1d_ml_challenge_predictions.csv",
    RESULTS / "adr1d_nn_challenge_metrics.json",
    RESULTS / "adr1d_nn_challenge_predictions.csv",
    RESULTS / "adr1d_nn_challenge_scenarios.csv",
    RESULTS / "numerical_reference_cases.csv",
    RESULTS / "numerical_reference_metrics.json",
    RESULTS / "computational_cost.json",
    RESULTS / "figure_manifest.json",
]


class ReleaseValidationError(RuntimeError):
    """Represent an incomplete or internally inconsistent public release."""


def parse_arguments():
    """
    Parse the release-audit command line.

    Returns
    -------
    argparse.Namespace
        Manifest-generation and overwrite options.
    """
    parser = argparse.ArgumentParser(description="Audit the public ADR1D numerical-validation release.")
    parser.add_argument("--write-manifest", action="store_true", help="Create the scientific-artifact release manifest before validation.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing release manifest when used with --write-manifest.")
    return parser.parse_args()


def require(condition, message):
    """
    Enforce one release invariant.

    Parameters
    ----------
    condition : bool
        Condition that must evaluate to true.
    message : str
        Actionable failure description.

    Raises
    ------
    ReleaseValidationError
        If the condition is false.
    """
    if not bool(condition):
        raise ReleaseValidationError(message)


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
        Lowercase hexadecimal digest.
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
    """
    content = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(content, dict), f"Expected a JSON object in {path}")
    return content


def portable_path(path):
    """
    Convert a repository artifact to a POSIX-style relative path.

    Parameters
    ----------
    path : pathlib.Path
        Artifact located inside the repository root.

    Returns
    -------
    str
        Repository-relative path.
    """
    resolved = path.resolve()
    require(resolved.is_relative_to(ROOT.resolve()), f"Artifact is outside the repository: {resolved}")
    return resolved.relative_to(ROOT.resolve()).as_posix()


def artifact_record(path):
    """
    Build one machine-readable artifact record.

    Parameters
    ----------
    path : pathlib.Path
        Existing repository file.

    Returns
    -------
    dict
        Relative path, byte count, and SHA-256 digest.
    """
    require(path.is_file(), f"Required artifact is missing: {path}")
    return {"path": portable_path(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def build_manifest():
    """
    Describe the immutable scientific content of the public release.

    Returns
    -------
    dict
        Release metadata and grouped artifact records.

    Notes
    -----
    README and citation metadata are intentionally excluded because the DOI is
    added after reservation. Scientific evidence remains independently locked.
    """
    figure_paths = sorted((RESULTS / "figures").glob("*.pdf")) + sorted((RESULTS / "figures").glob("*.png"))
    source_paths = sorted((ROOT / "src").glob("*.py"))
    return {
        "artifact_groups": {
            "core_evidence": [artifact_record(path) for path in CORE_EVIDENCE],
            "figures": [artifact_record(path) for path in figure_paths],
            "report": [artifact_record(REPORT_TEX_ES), artifact_record(REPORT_PDF_ES), artifact_record(REPORT_TEX_EN), artifact_record(REPORT_PDF_EN)],
            "source_code": [artifact_record(path) for path in source_paths],
        },
        "release": {
            "name": "ADR1D-Validation",
            "package_version": "1.0.0",
            "report_version": "1.1",
            "repository": REPOSITORY_URL,
        },
        "status": "complete",
    }


def write_manifest(overwrite):
    """
    Persist the release manifest after explicit authorization.

    Parameters
    ----------
    overwrite : bool
        Whether an existing manifest may be replaced.
    """
    require(overwrite or not RELEASE_MANIFEST.exists(), "Release manifest already exists; add --overwrite to replace it")
    RELEASE_MANIFEST.write_text(json.dumps(build_manifest(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_manifest():
    """
    Verify every scientific artifact against the release manifest.

    Returns
    -------
    dict
        Number of verified artifacts by group.
    """
    manifest = load_json(RELEASE_MANIFEST)
    require(manifest.get("status") == "complete", "Release manifest is incomplete")
    require(manifest.get("release", {}).get("repository") == REPOSITORY_URL, "Release repository URL changed")
    expected_counts = {"core_evidence": 20, "figures": 8, "report": 4, "source_code": 10}
    counts = {}
    for group, expected_count in expected_counts.items():
        records = manifest.get("artifact_groups", {}).get(group, [])
        require(len(records) == expected_count, f"Unexpected artifact count in {group}")
        for record in records:
            relative = Path(record["path"])
            require(not relative.is_absolute() and ".." not in relative.parts, f"Non-portable manifest path: {relative}")
            path = ROOT / relative
            require(path.is_file(), f"Manifest artifact is missing: {relative}")
            require(path.stat().st_size == int(record["bytes"]), f"Artifact size changed: {relative}")
            require(sha256(path) == record["sha256"], f"Artifact digest changed: {relative}")
        counts[group] = len(records)
    return counts


def count_csv_rows(path):
    """
    Count data rows in a CSV file without loading the full table into memory.

    Parameters
    ----------
    path : pathlib.Path
        CSV artifact with one header row.

    Returns
    -------
    int
        Number of records after the header.
    """
    with path.open("r", encoding="utf-8", newline="") as stream:
        return sum(1 for _ in csv.reader(stream)) - 1


def close(actual, expected, label, tolerance=1.0e-12):
    """
    Compare one finite scalar against its documented value.

    Parameters
    ----------
    actual : float
        Value read from persisted evidence.
    expected : float
        Documented reference value.
    label : str
        Quantity named in a failure message.
    tolerance : float, optional
        Absolute and relative comparison tolerance.
    """
    require(math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance), f"Reported value changed: {label}")


def validate_scientific_summary():
    """
    Check row counts, statuses, and selected report-level results.

    Returns
    -------
    dict
        Concise scientific release summary.
    """
    expected_rows = {
        ROOT / "data/challenge_scenarios.csv": 120,
        ROOT / "data/challenge_analytical_field.csv": 299880,
        ROOT / "data/challenge_sensor_observations.csv": 176400,
        RESULTS / "adr1d_ml_challenge_predictions.csv": 600,
        RESULTS / "adr1d_nn_challenge_predictions.csv": 299880,
        RESULTS / "adr1d_nn_challenge_scenarios.csv": 120,
        RESULTS / "numerical_reference_cases.csv": 48,
    }
    for path, expected in expected_rows.items():
        require(count_csv_rows(path) == expected, f"Unexpected row count: {portable_path(path)}")

    ml        = load_json(RESULTS / "adr1d_ml_challenge_metrics.json")
    nn        = load_json(RESULTS / "adr1d_nn_challenge_metrics.json")
    numerical = load_json(RESULTS / "numerical_reference_metrics.json")
    audit     = load_json(RESULTS / "challenge_result_validation.json")
    cost      = load_json(RESULTS / "computational_cost.json")
    require(ml["evidence"]["adequacy"]["overall_outcome"] == "meets_all_pre_specified_criteria", "ADR1D-ML adequacy outcome changed")
    require(nn["evidence"]["adequacy"]["overall_outcome"] == "meets_all_pre_specified_criteria", "ADR1D-NN adequacy outcome changed")
    require(numerical["acceptance"]["overall_outcome"] == "mixed_evidence", "Numerical-reference outcome changed")
    require(numerical["acceptance"]["scenarios_not_meeting_all_criteria"] == ["A5-CH-0117", "A5-CH-0087"], "Numerical-reference exception cases changed")
    require(audit.get("status") == "passed", "Persisted independent result audit did not pass")
    require(cost.get("status") == "complete" and cost["integrity"]["protected_artifacts_unchanged"] is True, "Computational-cost evidence is incomplete")
    require(cost["integrity"]["before_sha256"] == cost["integrity"]["after_sha256"], "Protected artifacts changed during cost measurement")

    close(ml["evidence"]["effective_velocity"]["median_absolute_percentage_error"], 0.03945134872834183, "ADR1D-ML velocity median relative error")
    close(ml["evidence"]["effective_dispersion"]["median_absolute_percentage_error"], 0.24942934909083544, "ADR1D-ML dispersion median relative error")
    close(ml["evidence"]["decay_resolvability"]["balanced_accuracy"], 0.8357238906320036, "ADR1D-ML balanced accuracy")
    close(ml["evidence"]["decay_resolvability"]["roc_auc"], 0.9112505602868669, "ADR1D-ML ROC AUC")
    close(nn["evidence"]["field"]["rmse"], 0.02257771389037237, "ADR1D-NN field RMSE")
    close(nn["evidence"]["field"]["r2"], 0.9893676936226676, "ADR1D-NN field R2")
    close(numerical["aggregate"]["fine_rmse"]["maximum"], 0.01866985040764215, "fine numerical-reference maximum RMSE")

    return {
        "adr1d_ml_base_scenarios": ml["evidence"]["effective_velocity"]["rows"],
        "adr1d_nn_point_predictions": nn["predictions"]["rows"],
        "independent_audit_checks": audit["checks"]["successful_checks"],
        "numerical_reference_cases": numerical["execution"]["selected_scenarios"],
    }


def validate_english_report():
    """
    Verify the English report edition and its equivalence in structure.

    Returns
    -------
    dict
        Page count and structural elements shared by both language editions.
    """
    require(shutil.which("pdfinfo") is not None, "Poppler pdfinfo is required; install Poppler and retry")
    process = subprocess.run(["pdfinfo", str(REPORT_PDF_EN)], cwd=ROOT, capture_output=True, text=True, check=False)
    require(process.returncode == 0, "The English report PDF could not be inspected")
    require("Title:           Independent Numerical Validation of ADR1D-ML and ADR1D-NN" in process.stdout, "The English report title metadata changed")
    require(re.search(r"^Pages:\s+27$", process.stdout, flags=re.MULTILINE), "The English report must contain 27 pages")
    require(re.search(r"^Page size:\s+595\.276 x 841\.89 pts \(A4\)$", process.stdout, flags=re.MULTILINE), "The English report is not A4")

    spanish = REPORT_TEX_ES.read_text(encoding="utf-8")
    english = REPORT_TEX_EN.read_text(encoding="utf-8")
    require("\\usepackage[english]{babel}" in english, "The English report does not select English typography")
    patterns = {
        "labels": r"\\label\{([^}]+)\}",
        "references": r"\\(?:ref|eqref)\{([^}]+)\}",
        "citations": r"\\cite\{([^}]+)\}",
        "figures": r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",
    }
    for label, pattern in patterns.items():
        require(re.findall(pattern, spanish) == re.findall(pattern, english), f"The two report editions differ in {label}")
    require(english.count("\\begin{table}") == 11, "The English report must contain 11 tables")
    require(english.count("\\begin{equation}") + english.count("\\begin{align}") == 19, "The English report must contain 19 equation blocks")
    return {"figures": 4, "language": "English", "pages": 27, "tables": 11}


def validate_public_metadata():
    """
    Verify README, citation, licenses, and path portability.

    Returns
    -------
    dict
        Metadata checks completed by the audit.
    """
    required = [ROOT / "README.md", ROOT / "CITATION.cff", ROOT / "LICENSE", ROOT / "LICENSE-CONTENT", ROOT / "requirements.txt"]
    require(all(path.is_file() for path in required), "One or more public metadata files are missing")
    readme   = (ROOT / "README.md").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    require(REPOSITORY_URL in readme and REPOSITORY_URL in citation, "Repository URL is inconsistent")
    require("results/adr1d_validacion_numerica.pdf" in readme, "README does not link the Spanish technical report")
    require("results/adr1d_numerical_validation_en.pdf" in readme, "README does not link the English technical report")
    require("LICENSE-CONTENT" in readme and "LICENSE" in readme, "README does not explain both licenses")
    require("preferred-citation:" in citation and "type: report" in citation, "CITATION.cff does not identify the preferred technical report")
    author_tokens = ["Tinoco-Guerrero", "Gerardo", "Domínguez-Mota", "Francisco J.", "Guzmán-Torres", "J. Alberto"]
    require(all(token in citation for token in author_tokens), "CITATION.cff author list is incomplete")
    placeholder_patterns = [r"10\.5281/zenodo\.(?:TODO|XXXX|0000)", r"DOI\s*[:=]\s*(?:TBD|pending|to be assigned)", r"<DOI>"]
    require(not any(re.search(pattern, readme + citation, flags=re.IGNORECASE) for pattern in placeholder_patterns), "Publication metadata contains a DOI placeholder")

    scanned = 0
    local_patterns = ["/" + "Users" + "/", "/" + "home" + "/", "C:" + "\\Users\\"]
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES and ".git" not in path.parts:
            content = path.read_text(encoding="utf-8")
            require(not any(pattern in content for pattern in local_patterns), f"Local absolute path found in {portable_path(path)}")
            scanned += 1
    return {"metadata_files": len(required), "portable_text_files": scanned}


def run_original_deliverable_audit():
    """
    Execute the original no-inference audit with a temporary output.

    Returns
    -------
    dict
        Page, figure, and source-module counts from the original audit.

    Raises
    ------
    ReleaseValidationError
        If Poppler is unavailable or the original audit fails.
    """
    require(shutil.which("pdfinfo") is not None, "Poppler pdfinfo is required; install Poppler and retry")
    validator = ROOT / "src/validate_final_deliverable.py"
    with tempfile.TemporaryDirectory(prefix="adr1d_validation_audit_") as directory:
        output  = Path(directory) / "final_validation_manifest.json"
        process = subprocess.run([sys.executable, str(validator), "--output", str(output)], cwd=ROOT, capture_output=True, text=True, check=False)
        require(process.returncode == 0, "Original deliverable audit failed: " + (process.stderr.strip() or process.stdout.strip()))
        manifest = load_json(output)
    return {
        "figures": len(manifest["artifact_groups"]["figures"]),
        "report_pages": manifest["report"]["page_count"],
        "source_modules": len(manifest["artifact_groups"]["source_code"]),
    }


def main():
    """
    Create the optional manifest and audit the complete public package.

    Returns
    -------
    None
        A concise JSON summary is printed after every check passes.
    """
    arguments = parse_arguments()
    if arguments.write_manifest:
        write_manifest(arguments.overwrite)
    require(RELEASE_MANIFEST.is_file(), "Release manifest is missing; run with --write-manifest once")
    summary = {
        "artifact_groups": validate_manifest(),
        "deliverable": run_original_deliverable_audit(),
        "english_report": validate_english_report(),
        "metadata": validate_public_metadata(),
        "scientific_evidence": validate_scientific_summary(),
        "status": "ok",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
