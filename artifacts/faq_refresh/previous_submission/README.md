# Verified submission files

**October 10 FAQ review: the current four files require corrections before upload.** The previous checks below verify the earlier artifacts. They do not establish compliance with [Q&A.md](../Q&A.md): all Part B final models need fresh combined-data refits, A2/A3 need the required epoch plots in the report, and B4 must use the corrected final checkpoints. See the [complete FAQ review](../docs/FAQ_COMPLIANCE_REVIEW.md). Naming/finalization alone is insufficient; regenerate and verify the corrected reports and code ZIPs first.

**October 4 recheck:** the current reports correct the housing sample-input/population-price SD wording and explicitly disclose the final-output ReLU interpretation. All questions and saved numerical evidence were checked again; see the [full requirements review](../docs/FINAL_REQUIREMENTS_RECHECK.md).

The four upload files are listed in [manifest.json](manifest.json), with sizes and SHA-256 hashes. Both PDFs are **five pages**; the code ZIPs include independently runnable code, full standalone READMEs, exact dependency pins, supplied datasets, prepared data and saved evidence. Extract each ZIP into its own folder. Do not upload this README, manifests, checksum list or verification logs as extra coursework files.

**The name currently remains `YourName`.** The actual student name has not been supplied. Replace it in the visible report authors, PDF metadata and all four filenames together by running this from the assignment workspace root:

```powershell
.\.venv\Scripts\python.exe -B scripts/finalize_submission.py --name "Your Full Name"
```

The helper rebuilds/audits reports, archives the four earlier package files within `artifacts/submission_checks/previous_packages/`, rebuilds both ZIPs, verifies the four current files and refreshes checksums. It reuses extracted-folder checks only if every packaged code/data/document byte is identical; if a package changed, run `scripts/verify_submission.py` without `--reuse-extraction` for fresh validation. Historical reports in `reports/` remain as provenance. All Python packages remain in the project venv.

Validation passed **57 Part A tests** and **62 Part B tests** (23 shared tests appear in both; **96 distinct behavioral tests**). Fresh extracted folders regenerated preprocessing exactly, loaded selected checkpoints, checked synthetic CUDA forwards, regenerated all eight answers' assets with unchanged numeric tables, and verified a local Ray CUDA worker. Both ZIP integrity checks passed. The report audit matched saved evidence and preserved **938 experiment output hashes and 12 supplied-file hashes**. Full experiments and test-set model inference were not repeated. Validation used the existing pinned venv with inherited Python paths removed; it did not perform a new dependency installation.

**NTULearn upload and receipt remain pending.** There is no authenticated NTULearn connection available in this session. After replacing the placeholder, upload exactly the two report PDFs and two code ZIPs listed in the manifest to the appropriate Part A/Part B areas. Verify each filename/file size, complete the portal's final submission action, reopen the submission record and retain its receipt/confirmation. The written brief gives **October 12, 2026**; use the portal's actual cutoff time because the brief does not specify one. No successful submission is claimed here.

Detailed completion/limits: [submission verification record](../docs/SUBMISSION_VERIFICATION.md). Full validation evidence: [latest audit](../artifacts/submission_checks/latest.json). Checksums: [SHA256SUMS.txt](SHA256SUMS.txt).
