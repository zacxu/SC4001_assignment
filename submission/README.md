# Current verified submission files

**Updated October 12, 2026:** the four current files follow assignment.pdf, assignment_slides.pdf, clarifications.md and Q&A.md. The [manifest](manifest.json) lists exact filenames, sizes and hashes; [SHA256SUMS.txt](SHA256SUMS.txt) contains matching checksums. Both PDFs are five pages. Both code ZIPs include standalone venv/dependency/seed/run instructions, relevant code/tests/data/checkpoints/evidence and controlling source documents.

FAQ corrections are complete: A2/A3 train/validation epoch evidence is included; A1 retains its separately required authorized train/test refit curve; B1/B2/B3 final results use fresh combined-data 9/13/9-epoch refits; B4 uses those new final checkpoints. Labels and B1/B2 dropout/output wording are instructor-confirmed. Earlier Part B final scores, failed B4 attempt and obsolete selection-run test records are archived outside the current packages.

A1's explanation now also incorporates the [interpretation review](../docs/A1_INTERPRETATION_REVIEW.md): our measured winner remains H=256,d=4; both learning histories support late overfitting, while a high test plateau or width relative to sample count does not establish underfitting. Settings, checkpoint weights and numeric results are unchanged. The revised explanation and its asset generator are included in the current Part A code ZIP.

A2 plots now use shared zoomed y axes across its two ablations. The final five-seed point/error-bar comparisons cover RMSE SGD 90,000–120,000 and R² 0.500–0.725; epoch plots use padded ranges that retain every train/validation point. The current report, PNG/PDF figures and packaged generators all contain this update.

A3's baseline-versus-wide-and-deep comparison now uses zoomed paired points and mean ± SD, covering RMSE SGD 90,000–115,000 and R² 0.575–0.725. Every observation/error-bar endpoint remains visible. Its PNG/PDF figures, report asset/description and packaged plotting code are updated; numerical results and model weights remain unchanged.

B1 and B2's loss and accuracy figures now show validation curves only against epoch for all five learning rates (B1) and five batch sizes (B2), retaining checkpoint stars and actual stopping lengths. The standalone PNG/PDF figures, report assets and caption, documentation and packaged plotting code all include this update. Saved histories, numerical results and model weights are unchanged.

Fresh isolated extractions passed **60 Part A / 64 Part B tests** (23 shared; 101 distinct), exact preprocessing, local final-checkpoint loading/synthetic CUDA forwards, all eight asset exporters plus A2 diagnostics and a local Ray worker. They reused the existing pinned venv without fresh dependency installation, full experiment retraining or test-set model inference. Final PDFs are byte-identical to reports/. The latest A1 explanation, A2/A3 axis changes and B1/B2 validation-only figures were checked again from fresh extractions of both current ZIPs. [Latest verification](../artifacts/submission_checks/latest.json) links the fresh extraction and final file audits. [Detailed verification record](../docs/SUBMISSION_VERIFICATION.md) explains scope and limits.

**The name currently remains `YourName`.** The actual student name has not been supplied. Replace it in the visible report authors, PDF metadata and all four filenames together by running this from the assignment workspace root:

```powershell
.\.venv\Scripts\python.exe -B scripts/finalize_submission.py --name "Your Full Name"
```

The helper rebuilds/audits reports, archives earlier four payloads, rebuilds ZIPs, verifies current files and refreshes checksums. It reuses extraction checks only when packaged code/data/document bytes are exactly unchanged; changed code requires fresh extraction validation. All dependencies remain in a venv.

**NTULearn upload and receipt remain pending.** After replacing the placeholder, upload exactly the four manifest files to the proper Part A/Part B areas, verify names/sizes, complete final submission and retain confirmation. Do not upload this README, manifests, checksums or logs as extra coursework files. The brief deadline is October 12, 2026; check the portal's exact cutoff time. No portal submission is claimed.
