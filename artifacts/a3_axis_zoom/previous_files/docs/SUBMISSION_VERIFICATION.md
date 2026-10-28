# Current submission verification and packaging

Completed local FAQ corrections and verification **October 11, 2026**. Both report PDFs and both code ZIPs now follow assignment.pdf, assignment_slides.pdf, clarifications.md and Q&A.md. The [current manifest](../submission/manifest.json) records the four filenames, exact hashes/sizes, controlling-source fingerprints and `verified_pending_author_and_portal` status. **Student-name replacement and NTULearn upload/receipt remain pending.**

## Corrected answer evidence

A1 retains its explicit brief-required train/test refit RMSE curve from already saved professor-authorized monitoring; its test-assisted stopping remains disclosed. The FAQ train/validation-only epoch-plot answer is specific to A2/A3. A2 has separate categorical/activation plots from three genuine original-split seed-42 diagnostics; A3 includes already saved train/validation RMSE/R2 curves. Required Part A five-seed final results and full A4 evidence are unchanged.

The subsequent [A1 interpretation review](A1_INTERPRETATION_REVIEW.md) replaces the weak width-versus-sample-count underfitting argument with measured grid comparisons and evidence of late overfitting in both learning histories. H=256,d=4 remains the observed winner. Updated prose and its generator are included in the Part A ZIP, and the reports/submission PDFs match exactly.

A2's final five-seed plots now show paired points and mean ± SD on common zoomed y axes: RMSE SGD 90,000–120,000 and R² 0.500–0.725. Both report and standalone epoch plots also share padded zoomed limits, retaining every training/validation point. All four standalone PNGs and the revised report page were visually checked. The plot revision preserves all 871 snapshotted numeric/model artifacts, excluding verification metadata that necessarily refreshes figure hashes. Numerical results and model weights are unchanged.

B1/B2/B3 each use new seed-42 model/Adam state trained on combined 5,232 supplied training/validation rows for their frozen 9/13/9 epochs. Original training-only pixel statistics remain fixed under the brief, with no validation/test monitoring during final refitting. Each fresh final model has one current final test pass; final checkpoint lineage differs from validation selection. Corrected performance tables and B4 use these final models. Q&A confirms labels, B1 dropout, B2 no dropout and B1/B2 raw outputs. B3 raw-logit handling retains the shared binary-loss interpretation explicitly.

Current final accuracy/sensitivity/specificity is 90.7051%/97.6923%/79.0598% for B1, 85.2564%/100%/60.6838% for B2, and 81.0897%/100%/49.5726% for B3. B4 retains the same five previously sampled pneumonia rows, regenerated maps and probabilities tied to those new final checkpoint hashes.

## Final payloads and historical separation

Each PDF is five pages including declaration/references. Part A ZIP has **529 entries, 20.31 MiB**; Part B has **228 entries, 77.72 MiB**. Exact bytes/hash values in the manifest take precedence. Both packages have independently runnable code, common modules, relevant tests, supplied raw/prepared data, current checkpoints/predictions/figures and root README/pinned requirements/seeds/commands. Both include byte-identical controlling PDFs, clarifications and Q&A.md plus embedded file manifests.

Virtual environments, caches, opposite-part code and opaque duplicate Ray state are excluded. The obsolete selection-run test records and historical failed B4 attempt are also excluded from current packages, so they do not conflict with fresh final answers. Original selection runs/data/checkpoints, previous no-refit results and prior documentation/payloads remain preserved in workspace archives. Root `checkpoint` frozen fields retain selection provenance; `final_checkpoint` identifies current final models.

## Observed checks

All **101 distinct root behavioral tests passed**. The saved-artifact review checked **68 run histories** and independently recomputed preprocessing, grid/stopping/seed statistics, all A4 aggregations, classifier confusion/BCE metrics and B4 stored gradient/map algebra without new model inference. The current report audit verifies **877 unchanged original files**, explicitly allowing the required final-result replacements and updated A1/A2 plot assets. Current Grad-CAM algebra verification is scoped to stored tensors/final lineage, not a claim of a new independent autograd execution for corrected models.

Both code ZIPs were freshly extracted and run with isolated Python search paths using the existing pinned project venv. **Dependencies were not freshly installed.**

| Check | Part A | Part B |
| --- | --- | --- |
| Packaged behavioral tests | 60 passed | 64 passed |
| Preparation from bundled raw data | Both selection/refit tensor-statistic protocols exactly reproduced | Original train/validation/test tensors/statistics exactly reproduced |
| Local checkpoint/synthetic CUDA forward | A1 and A3 final seed-42 models | All three fresh final-refit checkpoints |
| Saved asset regeneration | A1–A4 plus A2 diagnostics; numeric tables preserved | B1–B4; numeric tables preserved |
| Ray worker | Local Part A imports and synthetic CUDA forward passed | Not required |
| Opposite part or original personal paths needed | No | No |

Shared 23 tests appear in both packages, yielding 101 distinct tests. These checks perform zero full experiment reruns and zero test-set model inference. They exercise tests, synthetic inputs, preprocessing and saved asset exporters; full reproduction commands are included but the entire workload was not repeated from extracted folders.

The [latest audit](../artifacts/submission_checks/latest.json) links current fresh extraction evidence for both packages after the A1 explanation and A2 plot revisions. Both submission PDFs are byte-identical to reports/. Checksums match the manifest. The [report audit](../reports/verification.json) verifies current report inputs and values; the [requirements recheck](../artifacts/faq_corrections/recheck_verification.json) records the unchanged experiment protocols/results. Earlier October 11 audits remain historical evidence of previous presentation versions.

## Rebuild and remaining actions

```powershell
.\.venv\Scripts\python.exe -B scripts/build_submission.py
.\.venv\Scripts\python.exe -B scripts/verify_submission.py
# Requires the student's actual name.
.\.venv\Scripts\python.exe -B scripts/finalize_submission.py --name "Your Full Name"
```

The actual student name has not been supplied. After replacing YourName consistently in both report authors/metadata and all four filenames, upload exactly the manifest's two PDFs and two ZIPs to NTULearn, verify names/sizes, complete the portal's final submission action and retain its receipt. The brief deadline is October 12, 2026; the portal determines the precise cutoff. No authenticated NTULearn submission or receipt is available here, and none is claimed.
