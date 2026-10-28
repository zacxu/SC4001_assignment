"""Set the author, rebuild/audit both reports, and produce verified submission files."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Actual report author name, quoted if it contains spaces")
    args = parser.parse_args()
    name = args.name.strip()
    if not name or name == "YourName" or not re.sub(r"[^\w-]+", "_", name).strip("_"):
        raise ValueError("Supply your actual name")
    old = json.loads((ROOT / "reports/manifest.json").read_text())
    slug = re.sub(r"[^\w-]+", "_", name).strip("_")
    replacements = {old["reports"][part]: f"Part{part}_report_{slug}.pdf" for part in ("A", "B")}
    replacements.update({key.replace(".pdf", ".md"): value.replace(".pdf", ".md") for key, value in list(replacements.items())})
    replacements.update({key.replace("_report_", "_codes_").replace(".pdf", ".zip"): value.replace("_report_", "_codes_").replace(".pdf", ".zip")
                         for key, value in list(replacements.items()) if key.endswith(".pdf")})
    # Keep project links synchronized; leave historical author-placeholder notes
    # in the GenAI log untouched because they accurately record that earlier step.
    docs = [ROOT / "README.md", ROOT / "ASSIGNMENT_PLAN.md", ROOT / "reports/README.md",
            ROOT / "part_a/README.md", ROOT / "part_b/README.md", ROOT / "docs/REPORTS_COMPLETED.md",
            ROOT / "docs/SUBMISSION_VERIFICATION.md", ROOT / "submission/README.md", ROOT / "docs/GENAI_USAGE.md"]
    for path in docs:
        text = path.read_text(encoding="utf-8")
        for before, after in replacements.items():
            text = text.replace(before, after)
        text = text.replace("Student-name replacement and NTULearn upload/receipt remain pending.", "NTULearn upload/receipt remains pending.")
        text = text.replace("Actual-name replacement and NTULearn upload/receipt remain pending.", "NTULearn upload/receipt remains pending.")
        text = text.replace("Actual student-name replacement and NTULearn upload/receipt remain pending.", "NTULearn upload/receipt remains pending.")
        text = text.replace("Student-name replacement and NTULearn submission/receipt remain pending.", "NTULearn submission/receipt remains pending.")
        text = text.replace("author and portal actions pending", "portal actions pending")
        text = text.replace("`YourName` remains a placeholder.", f"The report author is **{name}**.")
        text = text.replace("`YourName` is still an explicit placeholder because the student's name has not been provided.", f"The student name is now **{name}**.")
        text = text.replace("- [ ] Replace the author placeholder consistently with the student's actual name.", "- [x] Replace the author placeholder consistently with the student's actual name.")
        text = text.replace("Author-name replacement and NTULearn upload/receipt remain.", "NTULearn upload/receipt remains.")
        if path == ROOT / "submission/README.md":
            text = text.replace("**The name currently remains `YourName`.** The actual student name has not been supplied. Replace it in the visible report authors, PDF metadata and all four filenames together by running this from the assignment workspace root:",
                                f"**The report author is {name}.** All four current filenames and author metadata are listed in the manifest. To change the author consistently, run this from the assignment workspace root:")
            text = text.replace("After replacing the placeholder, upload exactly", "Upload exactly")
        if path == ROOT / "reports/README.md":
            text = text.replace("**YourName is a placeholder** because the report author name has not been supplied.", f"**The report author is {name}.**")
            text = text.replace("actual-name replacement and NTULearn upload remain pending", "NTULearn upload remains pending")
        if path == ROOT / "docs/SUBMISSION_VERIFICATION.md":
            text = text.replace("Student-name replacement and NTULearn upload/receipt remain pending;", "The author is finalized; NTULearn upload/receipt remains pending;")
            text = text.replace("Final author replacement requires the student's actual name, which has not been supplied.", f"The author is finalized as **{name}** in all four submission filenames and report metadata.")
        path.write_text(text, encoding="utf-8")

    def run(script, *options):
        subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / script), *map(str, options)], cwd=ROOT, check=True)

    run("build_reports.py", "--name", name)
    run("verify_reports.py")
    submission = ROOT / "submission"
    previous = submission / "manifest.json"
    if previous.exists():
        prior = json.loads(previous.read_text())
        archive = ROOT / "artifacts/submission_checks/previous_packages" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        for filename in prior["submission_files"]:
            source = (submission / filename).resolve()
            if source.parent != submission.resolve():
                raise ValueError("Invalid prior submission filename")
            # Archive only the tool's four prior files, within the workspace.
            if source.exists():
                archive.mkdir(parents=True, exist_ok=True)
                destination = (archive / source.name).resolve()
                assert destination.is_relative_to((ROOT / "artifacts/submission_checks").resolve())
                source.rename(destination)
    run("build_submission.py")
    latest = ROOT / "artifacts/submission_checks/latest.json"
    options = []
    if latest.exists():
        prior = json.loads(latest.read_text())
        current = json.loads((submission / "manifest.json").read_text())
        unchanged = True
        for filename in current["submission_files"]:
            if filename.endswith(".zip"):
                part = "part_a" if filename.startswith("PartA_") else "part_b"
                with zipfile.ZipFile(submission / filename) as archive:
                    unchanged &= hashlib.sha256(archive.read("PACKAGE_MANIFEST.json")).hexdigest() == prior["parts"][part]["package_manifest_sha256"]
        if unchanged:
            options = ["--reuse-extraction", prior["audit_directory"]]
    run("verify_submission.py", *options)
    manifest = json.loads((submission / "manifest.json").read_text())
    (submission / "SHA256SUMS.txt").write_text("".join(f"{value['sha256']}  {filename}\n" for filename, value in manifest["files"].items()))
    print("Author and filenames updated; four files verified. NTULearn upload/receipt still required.")


if __name__ == "__main__":
    main()
