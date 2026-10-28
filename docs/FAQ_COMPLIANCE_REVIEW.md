# FAQ compliance review and completed corrections

Updated **October 11, 2026** using all six questions/answers in [Q&A.md](../Q&A.md), all seven brief pages and six slides, source code, saved histories, final-model lineage and concise reports. The original review found two material gaps: no fresh Part B final refits, and omitted A2/A3 epoch evidence. That earlier review and packages are preserved in [historical documentation](../artifacts/faq_refresh/previous_docs/docs/FAQ_COMPLIANCE_REVIEW.md) and [prior submission payloads](../artifacts/faq_refresh/previous_submission/manifest.json).

| FAQ | Current answer and evidence | Status |
| --- | --- | --- |
| Refit A1 scaler/encoder on combined data | Separate selection transforms fit2017–2020 and refit transforms fit2017–2021; A1/A2/A3 final models use refit bundle. | Correct; existing A1 result unchanged. |
| B3 stride-2 SAME padding | ConvBNReLU uses kernel_size//2: 3x3 stride2 padding1, 1x1 projection stride2 padding0. Both paths halve even-sized maps. | Correct and instructor-confirmed. |
| B3 combined-data final refit | Original train/validation run freezes epoch 9; new seed 42 model/Adam train on5,232 combined rows for9 epochs; only the new final checkpoint receives the current final test. | Corrected. |
| A2/A3 train/validation RMSE/R2 versus epochs; separate A2 ablations | Three A2 original-split seed 42 diagnostics supply separate categorical/activation RMSE/R2 plots; A3 uses saved original-split lambda-selection histories. Both are included in the concise report. A1 separately retains its brief-required train/test RMSE curve from the authorized test-assisted refit. The required five-seed combined-data final scores remain unchanged. | Corrected; no A2/A3 test metric epoch plots. |
| Fresh B1/B2/B3 final models, selected hyperparameters and epoch counts, one final test | Fresh seed 42 model/optimizer state, train+validation5,232 rows, LR 0.001, batch 128/16/16, exactly9/13/9 epochs. Final checkpoint hashes differ from validation selection. Each final test has624 rows and one evaluation. B4 uses these final models. | Corrected for all three. |
| Labels, dropout and raw-logit B1/B2 outputs | Instructor confirms0 = healthy/1 = pneumonia, B1 dropout0.5 after both hidden-FC ReLUs, B2 no dropout, B1/B2 FC1 raw logit directly to BCEWithLogitsLoss. | Existing code correct; obsolete assumption wording replaced. FAQ does not separately name B3 FC1. |

Current final accuracy/sensitivity/specificity is **90.7051%/97.6923%/79.0598%** for B1, **85.2564%/100%/60.6838%** for B2, and **81.0897%/100%/49.5726%** for B3. The earlier Part B test scores are historical results of a different protocol and are not the current answers. The new B4 uses the original fixed row IDs486,279,413,53,276; examples were not redrawn after seeing corrected results.

## Deliberate interpretation boundaries

The detailed brief still says normalize X-rays with original training-pixel statistics. Q&A explicitly requires refitting **A1** statistics, but does not change that Part B normalization rule. Current Part B refits retain original training-only mean/SD consistently; no test pixels fit statistics. Fresh final epoch budgets supersede blanket early stopping because the FAQ prescribes fixed selected epochs with no held-out partition. B3 uses the same binary-loss raw-logit interpretation as B1/B2, documented as an interpretation rather than a separately explicit FAQ confirmation. A3 retains A1 E*=4 for a controlled five-seed final comparison; its test-assisted origin remains disclosed.

A2 diagnostics use2017–2020 training and2021 validation with selection-fitted transforms and four completed epochs. They never access test samples, change frozen settings or replace the required combined-data final results. The five-seed final fits do not mislabel their in-sample2021rows as held-out validation. Existing valid Ray/lambda/Part B search evidence and all Part A final scores are preserved.

## Verification and delivery

[Current saved-evidence recheck](../artifacts/faq_corrections/recheck_verification.json) records current protocols, confusion/BCE recomputation, 68 histories, full A4 aggregation, B4 stored-map algebra and protected-output hashes. [Report verification](../reports/verification.json) ties both five-page PDFs to current evidence and controlling Q&A hashes. [Submission manifest](../submission/manifest.json) and [latest fresh extraction checks](../artifacts/submission_checks/latest.json) are the authority for current four payload hashes, test counts and completion status. Package audits use the existing pinned venv, synthetic checkpoints and saved outputs; they do not claim a fresh dependency installation or repeat the full training workload/test inference.

The corrected experiment execution genuinely includes three fresh classifier refits and their three final test passes, plus three A2 diagnostic runs and regenerated selected-image Grad-CAM. Subsequent numerical/report/package audits are verification only. Actual student-name replacement and NTULearn upload/receipt remain pending; the current filenames explicitly retain YourName.
