# Current verified submission files

**Updated October 11, 2026:** the four current files follow assignment.pdf, assignment_slides.pdf, clarifications.md and Q&A.md. The [manifest](manifest.json) lists exact filenames, sizes and hashes; [SHA256SUMS.txt](SHA256SUMS.txt) contains matching checksums. Both PDFs are five pages. Both code ZIPs include standalone venv/dependency/seed/run instructions, relevant code/tests/data/checkpoints/evidence and controlling source documents.

FAQ corrections are complete: A2/A3 train/validation epoch evidence is included; A1 retains its separately required authorized train/test refit curve; B1/B2/B3 final results use fresh combined-data 9/13/9-epoch refits; B4 uses those new final checkpoints. Labels and B1/B2 dropout/output wording are instructor-confirmed. Earlier Part B final scores, failed B4 attempt and obsolete selection-run test records are archived outside the current packages.

Fresh isolated extractions passed **60 Part A / 64 Part B tests** (23 shared; 101 distinct), exact preprocessing, local final-checkpoint loading/synthetic CUDA forwards, all eight asset exporters plus A2 diagnostics and a local Ray worker. They reused the existing pinned venv without fresh dependency installation, full experiment retraining or test-set model inference. Final PDFs are byte-identical to reports/; their last clarity changes were reverified while reusing exactly identical freshly checked code ZIPs. [Latest verification](../artifacts/submission_checks/latest.json) links the fresh extraction and final file audits. [Detailed verification record](../docs/SUBMISSION_VERIFICATION.md) explains scope and limits.

**The name currently remains `YourName`.** The actual student name has not been supplied. Replace it in the visible report authors, PDF metadata and all four filenames together by running this from the assignment workspace root:

```powershell
.\.venv\Scripts\python.exe -B scripts/finalize_submission.py --name "Your Full Name"
```

The helper rebuilds/audits reports, archives earlier four payloads, rebuilds ZIPs, verifies current files and refreshes checksums. It reuses extraction checks only when packaged code/data/document bytes are exactly unchanged; changed code requires fresh extraction validation. All dependencies remain in a venv.

**NTULearn upload and receipt remain pending.** After replacing the placeholder, upload exactly the four manifest files to the proper Part A/Part B areas, verify names/sizes, complete final submission and retain confirmation. Do not upload this README, manifests, checksums or logs as extra coursework files. The brief deadline is October 12, 2026; check the portal's exact cutoff time. No portal submission is claimed.
