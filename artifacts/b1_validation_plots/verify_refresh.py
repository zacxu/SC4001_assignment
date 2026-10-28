"""Record the validation-only B1 presentation refresh and current package hashes."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

import pandas as pd
import pymupdf

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


baseline = read("artifacts/b1_validation_plots/baseline.json")
protected = baseline["protected_numeric_and_model_artifacts"]
for relative, expected in protected.items():
    assert sha(ROOT / relative) == expected, relative

manifest = read("submission/manifest.json")
latest = read("artifacts/submission_checks/latest.json")
report_manifest = read("reports/manifest.json")
report_verification = read("reports/verification.json")
assert latest["status"] == report_verification["status"] == "passed"
assert latest["parts"]["part_a"]["tests_passed"] == 60
assert latest["parts"]["part_b"]["tests_passed"] == 64
assert latest["full_experiment_reruns"] == 0
for relative, expected in report_manifest["inputs"].items():
    assert sha(ROOT / relative) == expected, relative
checksums = {}
for line in (ROOT / "submission/SHA256SUMS.txt").read_text().splitlines():
    digest, filename = line.split(maxsplit=1)
    checksums[filename.lstrip("*")] = digest
pages, archives = {}, {}
for filename, record in manifest["files"].items():
    path = ROOT / "submission" / filename
    digest = sha(path)
    assert digest == record["sha256"] == latest["files"][filename]["sha256"] == checksums[filename]
    assert path.stat().st_size == record["bytes"] == latest["files"][filename]["bytes"]
    if path.suffix == ".pdf":
        assert path.read_bytes() == (ROOT / "reports" / filename).read_bytes()
        with pymupdf.open(path) as pdf:
            assert len(pdf) == 5
            pages[filename] = len(pdf)
    else:
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            embedded = json.loads(archive.read("PACKAGE_MANIFEST.json"))
            for relative, expected in embedded["files"].items():
                assert hashlib.sha256(archive.read(relative)).hexdigest() == expected
                if relative.startswith(("part_a/outputs/", "part_b/outputs/")):
                    assert archive.read(relative) == (ROOT / relative).read_bytes(), relative
            archives[filename] = len(archive.namelist())
            if filename.startswith("PartB_"):
                for relative in ["part_b/src/b1_exports.py", "part_b/outputs/b1/analysis.md"]:
                    assert archive.read(relative) == (ROOT / relative).read_bytes()
                assert archive.read("README.md") == (ROOT / "submission_support/README_B.md").read_bytes()
                assert b"validation-only" in archive.read("README.md")

old_archive = OUT / "previous_files/submission/PartB_codes_YourName.zip"
with zipfile.ZipFile(old_archive) as archive:
    changed_outputs = sorted(relative for relative in archive.namelist()
        if relative.startswith("part_b/outputs/")
        and archive.read(relative) != (ROOT / relative).read_bytes())
assert changed_outputs == ["part_b/outputs/b1/analysis.md",
    "part_b/outputs/b1/figures/validation_accuracy.pdf",
    "part_b/outputs/b1/figures/validation_accuracy.png",
    "part_b/outputs/b1/figures/validation_loss.pdf",
    "part_b/outputs/b1/figures/validation_loss.png"]

coverage = []
for row in read("part_b/outputs/b1/learning_rate_search.json"):
    history = pd.read_csv(ROOT / "part_b/outputs/b1" / row["run_directory"] / "history.csv")
    assert history.epoch.tolist() == list(range(1, row["epochs_completed"] + 1))
    assert row["best_epoch"] in history.epoch.values
    selected = history.loc[history.epoch == row["best_epoch"]].iloc[0]
    for metric in ("loss", "accuracy"):
        assert abs(float(selected[f"validation_{metric}"]) - row[f"validation_{metric}"]) < 1e-12
    coverage.append({"learning_rate": row["learning_rate"],
        "epochs_completed": row["epochs_completed"], "checkpoint_epoch": row["best_epoch"],
        "minimum_validation_accuracy": float(history.validation_accuracy.min()),
        "maximum_validation_accuracy": float(history.validation_accuracy.max()),
        "maximum_validation_loss": float(history.validation_loss.max())})
assert len(coverage) == 5

figure_paths = [f"part_b/outputs/b1/figures/validation_{metric}.{extension}"
    for metric in ("loss", "accuracy") for extension in ("png", "pdf")]
figure_paths += [f"reports/assets/b1_{metric}.{extension}"
    for metric in ("loss", "accuracy") for extension in ("png", "pdf")]
for relative in figure_paths:
    if relative.endswith(".pdf"):
        with pymupdf.open(ROOT / relative) as pdf:
            labels = pdf[0].get_text().lower()
            assert "validation" in labels or "val." in labels
            assert "train" not in labels and "test" not in labels

timestamp = datetime.now(timezone.utc).isoformat()
audit = {"status": "passed_current_report_and_submission", "timestamp": timestamp,
    "b1_validation_only_loss_and_accuracy": True,
    "learning_rates": coverage, "all_five_series_and_checkpoint_stars_visually_reviewed": True,
    "actual_stopping_lengths_retained": True,
    "changed_part_b_output_files": changed_outputs,
    "b2_b3_experiment_outputs_unchanged": True,
    "protected_numeric_and_model_artifacts_unchanged": len(protected),
    "new_assignment_training_runs": 0, "new_test_set_model_inference": 0,
    "figure_hashes": {relative: sha(ROOT / relative) for relative in figure_paths},
    "report_inputs_checked": len(report_manifest["inputs"]),
    "protected_pre_FAQ_files_unchanged": report_verification["protected_pre_FAQ_output_hashes_unchanged"],
    "reports_byte_identical_to_submission": True, "report_pages": pages,
    "package_entries": archives,
    "fresh_extracted_package_tests": {part: record["tests_passed"] for part, record in latest["parts"].items()},
    "package_audit": latest["audit_directory"],
    "visual_review": "Independent review of four B1 PNGs and parent review of final Part B page 2 passed; readable labels, legends and checkpoints, no clipping",
    "submission_files": manifest["files"]}
(OUT / "verification.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

closure = read("artifacts/faq_corrections/closure.json")
closure["timestamp"] = timestamp
closure["date"] = timestamp[:10]
closure["latest_package_verification"] = (Path(latest["audit_directory"]) / "verification.json").relative_to(ROOT).as_posix()
closure["latest_b1_validation_plots"] = "artifacts/b1_validation_plots/verification.json"
closure["report_inputs_verified"] = len(report_manifest["inputs"])
for letter in ("A", "B"):
    pdf_name = report_manifest["reports"][letter]
    zip_name = pdf_name.replace("_report_", "_codes_").replace(".pdf", ".zip")
    closure["reports"][letter].update(sha256=manifest["files"][pdf_name]["sha256"], pages=pages[pdf_name], submission_copy_identical=True)
    closure["code_archives"][letter].update(sha256=manifest["files"][zip_name]["sha256"], entries=archives[zip_name])
(ROOT / "artifacts/faq_corrections/closure.json").write_text(json.dumps(closure, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": audit["status"], "unchanged_numeric_model_files": len(protected),
    "package_tests": audit["fresh_extracted_package_tests"], "report_pages": pages,
    "changed_experiment_outputs": changed_outputs}, indent=2))
