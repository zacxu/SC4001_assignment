"""Safely extract both ZIPs and exercise their standalone documented entry points."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import pymupdf

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=ROOT / "submission")
    parser.add_argument("--reuse-extraction", type=Path, help="Re-audit final files using an earlier identical-code extracted-folder validation")
    args = parser.parse_args()
    directory = args.submission.resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    output = ROOT / "artifacts/submission_checks" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "passed", "name_placeholder": manifest["name_placeholder"], "files": {}, "parts": {},
              "ntulearn_submitted": False, "submission_receipt": None, "fresh_dependency_installation": False,
              "verification_venv": str(Path(sys.prefix).resolve()), "full_experiment_reruns": 0}
    assert Path(sys.prefix).resolve() == (ROOT / ".venv").resolve()
    assert len(manifest["submission_files"]) == 4
    if "controlling_sources" in manifest:
        for name, expected in manifest["controlling_sources"].items():
            assert digest(ROOT / name) == expected, "Controlling source changed; rebuild submission"
        result["controlling_sources"] = manifest["controlling_sources"]
    for filename in manifest["submission_files"]:
        path = directory / filename
        assert digest(path) == manifest["files"][filename]["sha256"]
        result["files"][filename] = {"sha256": digest(path), "bytes": path.stat().st_size}
        if path.suffix == ".pdf":
            with pymupdf.open(path) as document:
                assert len(document) == 5 and document.metadata["author"] == manifest["name"]
                assert "Generative AI" in document[0].get_text()
                assert all(page.get_text().strip() for page in document)
                result["files"][filename]["pages"] = len(document)
        else:
            part = "part_a" if filename.startswith("PartA_") else "part_b"
            destination = output / "extracted" / part
            with zipfile.ZipFile(path) as archive:
                assert archive.testzip() is None
                names = archive.namelist()
                assert len(names) == len(set(names))
                for name in names:
                    assert "\\" not in name and not name.startswith("/")
                    target = (destination / name).resolve()
                    assert target.is_relative_to(destination.resolve()), name
                    assert not re.match(r"^[A-Za-z]:", name)
                    assert not any(piece in {".venv", "__pycache__", ".git", ".codex", "node_modules"} for piece in Path(name).parts)
                assert all(name in names for name in ["README.md", "requirements.txt", "verify_package.py", "run_all.py", "PACKAGE_MANIFEST.json", "assignment.pdf", "assignment_slides.pdf", "clarifications.md", "Q&A.md"])
                embedded = json.loads(archive.read("PACKAGE_MANIFEST.json"))
                for name, expected in manifest.get("controlling_sources", {}).items():
                    assert hashlib.sha256(archive.read(name)).hexdigest() == expected, "Package contains an obsolete controlling source"
                for name, expected in embedded["files"].items():
                    assert hashlib.sha256(archive.read(name)).hexdigest() == expected
                result["files"][filename]["zip_entries"] = len(names)
                if args.reuse_extraction:
                    previous = json.loads((args.reuse_extraction / "verification.json").read_text())
                    prior = previous["parts"][part]
                    assert prior["package_manifest_sha256"] == hashlib.sha256(archive.read("PACKAGE_MANIFEST.json")).hexdigest(), "Package changed; fresh extraction required"
                    result["parts"][part] = prior
                    continue
                destination.mkdir(parents=True, exist_ok=False)
                archive.extractall(destination)

            # The venv is reused, but all assignment imports/data/outputs belong
            # to this extraction. -I removes inherited Python search paths.
            bootstrap = "import os,runpy,sys; sys.argv=[sys.argv[1]]+sys.argv[2:]; runpy.run_path(sys.argv[0],run_name='__main__')"
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            environment["MPLCONFIGDIR"] = str(destination / "artifacts/cache/matplotlib")
            environment["RAY_USAGE_STATS_ENABLED"] = "0"
            command = [sys.executable, "-I", "-B", "-c", bootstrap, str(destination / "verify_package.py"), "--prepare", "--export"]
            if part == "part_a":
                command.append("--ray-smoke")
            log = output / f"{part}_verification.log"
            with log.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(command, cwd=destination, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            if completed.returncode:
                print(log.read_text(encoding="utf-8", errors="replace")[-9000:])
                raise RuntimeError(f"{part} failed; see {log}")
            checks = sorted((destination / "artifacts/package_checks").glob("*/verification.json"))
            assert len(checks) == 1
            checked = json.loads(checks[0].read_text())
            assert checked["status"] == "passed" and checked["test_set_model_inference"] == 0
            result["parts"][part] = {"extracted_root": str(destination), "verification": str(checks[0]),
                "package_manifest_sha256": digest(destination / "PACKAGE_MANIFEST.json"), "tests_passed": checked["tests_passed"],
                "exports": checked["exports_regenerated"], "prepared_tensors": checked["prepared_tensors_compared"],
                "ray_worker": checked["ray_worker"], "synthetic_forward_checkpoints": checked["synthetic_forward_checkpoints"]}
            print(f"{part}: extracted-folder tests, preprocessing, local checkpoint loads and all four exports passed", flush=True)
    # The current report audit ties reported values to canonical evidence,
    # controlling FAQ hashes, and the protected historical-output snapshot.
    subprocess.run([sys.executable, "-B", str(ROOT / "scripts/verify_reports.py")], cwd=ROOT, check=True)
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (ROOT / "artifacts/submission_checks/latest.json").write_text(json.dumps({**result, "audit_directory": str(output)}, indent=2) + "\n", encoding="utf-8")
    manifest["status"] = "verified_pending_author_and_portal" if manifest["name_placeholder"] else "verified_pending_portal"
    manifest["verification"] = str(output.relative_to(ROOT) / "verification.json")
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (directory / "SHA256SUMS.txt").write_text("".join(f"{value['sha256']}  {filename}\n" for filename, value in manifest["files"].items()))
    print(f"All four packaged files passed. Evidence: {output}", flush=True)


if __name__ == "__main__":
    main()
