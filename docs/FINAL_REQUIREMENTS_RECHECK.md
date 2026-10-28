# Final requirements recheck: current brief, slides and FAQ

Updated **October 11, 2026**. The current controlling sources are [assignment.pdf](../assignment.pdf), [assignment_slides.pdf](../assignment_slides.pdf), [clarifications.md](../clarifications.md) and [Q&A.md](../Q&A.md). Latest instructor answers override earlier assumptions; use the detailed brief for slide typos (lambda1,2 instead of1.2; batch256 instead of258; five pages to satisfy less than six).

| Required answer | Concise report location | Current evidence and check |
| --- | --- | --- |
| A1 baseline diagram/trainable parameters | A p1 | Supplied defaultH64,d8:3,505parameters; selectedH256,d4:7,065. Baseline retains LayerNorm and standardized target. |
| A1 Ray full25-grid, best validation RMSE/configuration | A p1–2 | Required widths16/32/64/128/256 and dimensions2/4/8/12/16; selected validation RMSE58,627.10, E_val6. |
| A1 selected train/validation curves and fresh combined refit | A p2 | Refit preprocessing fitted2017–2021, E*=4, final RMSE96,901.64/R20.676082. Internal test stopping follows explicit clarification; the requested A1 train/test RMSE plot reuses those saved records. The FAQ no-test-epoch context applies to A2/A3. |
| A1 under/overfitting and justification | A p2 | Validation advantage/curves support selection; temporal shift and test-assisted final stopping disclosed. |
| A2 baseline and two ablations, five distinct seeds, matched controls | A p3 |15fresh combined-data final fits, seeds42–46, AdamLR0.01/batch512/four epochs. Remove all four embeddings or replace hidden ReLU with Sigmoid, preserving LayerNorm. |
| A2 mean/sampleSD RMSE/R2, separate epoch curves and conclusions | A p3 | Final means/sampleSD independently computed; separate original-split seed 42 train/validation RMSE/R2 curves, no test scoring or retuning. |
| A3 shared-input linear+MLP branches and five-lambda search | A p4 | MLP+lambda*linear; lambda2selected on validation, selected width/embedding fixed. |
| A3 five-seed final comparison, epoch plots, mechanism and limitation | A p4 | Four-epoch final baseline comparison unchanged; saved original-split RMSE/R2 curves included. Extra parameters, branch rescaling, seed/update budget and inherited test-assisted limit discussed. |
| A4 full test ten original features, same training seed, SGD importance | A p5 | Both26,702x10signed matrices, matched seed 42/epoch4, mean absolute aggregation, descending ten-feature bars, top three and reference/correlation limitations. |
| B1 exact architecture, five LRs, batch 128 | B p1–2 |99,569parameters; no convbias, BN/ReLU, hiddenFC0.5dropout, raw final logit. All five validation candidates preserved. |
| B1 separate loss/accuracy curves, complete six-column table, selection/discussion | B p1–2 | LR 0.001by validationerror/loss/epoch; table rows coherent; class imbalance, sensitivity/specificity and convergence discussed. |
| B1 fresh combined-data final and one final test | B p2 |9 epochs,5,232 rows;90.7051%accuracy/97.6923%sensitivity/79.0598%specificity. Final checkpoint separate from selection. |
| B2 exact architecture, frozenB1LR, five batches, curves/table/selection | B p2–3 |18,185parameters, no dropout, batches16/32/64/128/256; selected16/epoch 13. Separate curves; update-budget confound. |
| B2 fresh combined-data final and one final test | B p3 |13 epochs,5,232 rows;85.2564%accuracy/100%sensitivity/60.6838%specificity. |
| B3 residual architecture, inheritedLR/batch and final refit | B p4 |77,169parameters, aligned stride2projections/BN/ReLU and sumReLU; fresh9 epochs,5,232 rows;81.0897%accuracy/100%sensitivity/49.5726%specificity. |
| B3 all required comparison columns, FLOPs definition, architecture pros/cons | B p4 | Corrected final performance, error and parameter counts;39,063,616/19,464,256/25,325,696FLOPs,image with twoFLOPs/MAC+residualadds; exclusions/causal limits stated. |
| B4 random five pneumonia images, three-model overlay comparison | B p5 | Same seededIDs486,279,413,53,276; five originals/15maps from fresh final models. Relative maps and absent lesion annotations limit claims. |
| General training and data rules | Both methods | Adam/max20/patience4/delta1e-4during selection; prescribed fixed final budgets; housing chronology; X-ray supplied partitions/training-only stats. |
| Declarations, concise reports, runnable code, separate PDFs/ZIPs | Both p1 and package READMEs | GenAI declaration first, each reportfive pages inclreferences, pinnedvenv/seeds/commands, included data/code/evidence, independent PartA/B imports. |

## Verification boundaries and remaining actions

The [saved-evidence audit](../artifacts/faq_corrections/recheck_verification.json) validates current final lineage, all grids/stopping/68 histories, seed summaries, exact preprocessing, A4 scores and B4 stored arithmetic without model inference. [Report audit](../reports/verification.json) checks source hashes, every reported numeric answer and layout; [package audit](../artifacts/submission_checks/latest.json) checks fresh isolated extractions with the existing pinned venv, local prepared tensors/checkpoints and all exporters. Verification does not repeat full experiments or test-model performance. Earlier October4audit and outputs remain archived history; deliberately corrected finalBrecords are not falsely claimed unchanged.

Q&A explicitly confirms labels, strided padding and B1/B2 raw outputs/dropout. It does not separately name B3 final output; the same binary-loss interpretation remains documented. PartB retains original training-only pixel statistics as the brief requires. A3 final four-epoch budget follows the controlled-comparison interpretation. A1 test-assisted stopping and inherited PartA budget remain explicit limitations.

Author identity still usesYourName. NTULearn final submission and receipt are not available or claimed. The deadline isOctober12,2026; the portal controls the exact cutoff time. Current deliverable filenames/hashes/status are in [submission manifest](../submission/manifest.json).
