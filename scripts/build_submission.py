"""Build four submission files from verified reports and explicit source allowlists."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def package_files(part):
    letter = "A" if part == "part_a" else "B"
    result = {}

    def add(source, destination=None):
        path = ROOT / source
        relative = destination or path.relative_to(ROOT).as_posix()
        if path.is_symlink():
            raise ValueError(f"Unexpected symlink: {path}")
        result[relative] = path.read_bytes()

    for relative in ["requirements.txt", "docs/GENAI_USAGE.md", "assignment.pdf", "assignment_slides.pdf", "clarifications.md", "Q&A.md"]:
        add(relative, "GENAI_USAGE.md" if relative.startswith("docs/") else None)
    add(f"submission_support/README_{letter}.md", "README.md")
    for name in ["run_all.py", "verify_package.py"]:
        add("submission_support/" + name, name)
    for folder in ["assignment_common", part + "/src", part + "/scripts", part + "/configs"]:
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json"} and "__pycache__" not in path.parts:
                add(path.relative_to(ROOT))
    add(part + "/__init__.py")
    add(part + "/requirements.txt")
    result[part + "/README.md"] = (f"# Part {letter}\n\nSee the [standalone package README](../README.md) for environment, seeds, data, exact commands and reproduction. Run commands from the extraction root.\n").encode()
    tests = ["__init__.py", "test_metrics.py", "test_training.py"]
    tests += (["test_housing_data.py", "test_a1.py", "test_a2.py", "test_a3.py", "test_a4.py"] if letter == "A"
              else ["test_xray_data.py", "test_xray_models.py", "test_b1.py", "test_b2.py", "test_b3.py", "test_b4.py"])
    for optional in (["test_a2_diagnostics.py"] if letter == "A" else ["test_b_refit.py"]):
        if (ROOT / "tests" / optional).is_file():
            tests.append(optional)
    for name in tests:
        add("tests/" + name)
    if letter == "A":
        reference = ROOT / "housing_baseline_data/hdb-price-prediction-assignment-main"
        for path in sorted(reference.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".csv", ".md", ".toml", ".lock"} and "__pycache__" not in path.parts:
                add(path.relative_to(ROOT))
    else:
        add("chestxray_binary_64.npz")
    outputs = ROOT / part / "outputs"
    folders = ["data"] + [f"{letter.lower()}{i}" for i in range(1, 5)]
    omitted = []
    for folder in folders:
        for path in sorted((outputs / folder).rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(outputs / folder)
            obsolete_selection_test = (letter == "B" and folder in {"b1", "b2", "b3"}
                and path.name == "test_metrics.json" and rel.parts[0] in {"runs", "run"})
            if obsolete_selection_test:
                omitted.append(path.relative_to(ROOT).as_posix())
                continue
            # Ray keeps redundant checkpoint copies and opaque state. Retain
            # each trial's canonical training folder; new runs recreate Tune.
            ray_extra = folder == "a1" and rel.parts[0] == "grid" and (len(rel.parts) < 4 or rel.parts[2] != "training")
            if ray_extra or path.suffix not in {".py", ".json", ".jsonl", ".csv", ".pt", ".png", ".pdf", ".md"}:
                omitted.append(path.relative_to(ROOT).as_posix())
                continue
            add(path.relative_to(ROOT))
    for name, value in result.items():
        assert not any(p in {".venv", "__pycache__", ".git", ".codex", "node_modules"} for p in Path(name).parts)
        assert not name.startswith("part_b/" if letter == "A" else "part_a/")
        if name.endswith(".py"):
            ast.parse(value.decode("utf-8"), filename=name)
    result["PACKAGE_MANIFEST.json"] = (json.dumps({"part": part, "files": {key: digest_bytes(value) for key, value in sorted(result.items())},
        "raw_data_included": True, "historical_provenance_preserved": True,
        "controlling_sources": ["assignment.pdf", "assignment_slides.pdf", "clarifications.md", "Q&A.md"],
        "faq_protocol": "fresh_combined_part_b_refits_and_train_validation_part_a_epoch_curves",
        "omitted_noncanonical_or_ray_files": omitted, "omitted_historical_folders": (["part_b/outputs/b4_attempt_001"] if letter == "B" else []), "venv_and_caches_excluded": True}, indent=2) + "\n").encode()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "submission")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Submission destination must stay within the assignment workspace")
    reports = read(ROOT / "reports/manifest.json")
    verified = read(ROOT / "reports/verification.json")
    assert verified["status"] == "passed"
    output.mkdir(parents=True, exist_ok=True)
    names = []
    records = {}
    for part, letter in [("part_a", "A"), ("part_b", "B")]:
        report_name = reports["reports"][letter]
        assert digest_bytes((ROOT / "reports" / report_name).read_bytes()) == verified["reports"][letter]["pdf_sha256"], "Run verify_reports.py after report changes"
        shutil.copy2(ROOT / "reports" / report_name, output / report_name)
        code_name = report_name.replace("_report_", "_codes_").replace(".pdf", ".zip")
        entries = package_files(part)
        with zipfile.ZipFile(output / code_name, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for relative, value in sorted(entries.items()):
                info = zipfile.ZipInfo(relative, date_time=(2026, 10, 3, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, value)
        with zipfile.ZipFile(output / code_name) as archive:
            assert archive.testzip() is None and len(archive.namelist()) == len(entries)
        for filename in [report_name, code_name]:
            path = output / filename
            records[filename] = {"sha256": digest_bytes(path.read_bytes()), "bytes": path.stat().st_size}
            names.append(filename)
        print(f"Part {letter}: {len(entries)} ZIP files, {(output / code_name).stat().st_size / 1048576:.2f} MiB", flush=True)
    manifest = {"step": 15, "status": "packaged_pending_final_audit", "name": reports["name"],
                "name_placeholder": reports["name_placeholder"], "submission_files": names, "files": records,
                "requirements_revision": "current_brief_slides_clarifications_and_QA",
                "controlling_sources": {name: digest_bytes((ROOT / name).read_bytes())
                    for name in ("assignment.pdf", "assignment_slides.pdf", "clarifications.md", "Q&A.md")},
                "part_b_final_protocol": "fresh_combined_train_validation_fixed_selected_epochs",
                "a1_refit_train_test_curve": True, "a2_a3_train_validation_epoch_curves": True,
                "ntulearn_submitted": False, "submission_receipt": None}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
