# SC4001 assignment: step-by-step completion plan

Updated October 11, 2026 after reviewing both PDFs, clarifications and all six questions in [Q&A.md](Q&A.md). Steps 1–15's implementation, experiments, report writing and local packaging follow the current instructor answers. Identity replacement and actual NTULearn submission/receipt remain pending. See [current requirements matrix](docs/FINAL_REQUIREMENTS_RECHECK.md) and [FAQ review](docs/FAQ_COMPLIANCE_REVIEW.md).

## 1. Establish requirements and working interpretations

Read the complete brief and slides, inspect architecture diagrams, and apply latest instructor Q&A/clarifications first, detailed brief second, slides third. Both parts have 45 marks plus 10 presentation marks. Deadline October 12, 2026; verify the portal cutoff. Use two separate reports of at most five pages including GenAI declaration first and references, plus two standalone code ZIPs. Brief lambda grid {0,0.25,0.5,1,2} and batch grid {16,32,64,128,256} override slide typos. Record any remaining interpretations explicitly.

**Complete:** [decisions](docs/ASSIGNMENT_DECISIONS.md); labels/padding/B1/B2 output and dropout are now instructor-confirmed. B3 output and A3 final four-epoch budget remain documented interpretations.

## 2. Set up Python venv and dependencies

Create `.venv`, install all required packages inside it, verify CUDA, save exact dependency pins/hardware/seed configuration. Test actual training, Ray and Captum on small probes, keeping probes separate from final experiment results.

**Complete:** [setup report](docs/SETUP_REPORT.md), root requirements and `.venv`; local RTX 3060 Ti CUDA, Python 3.12, Torch 2.11.0+cu128.

## 3. Implement shared training and evaluation rules

Use Adam, selection maximum 20 epochs/patience 4/minimum improvement1e-4. Housing selects minimum validation RMSE; classification stops by validation BCE and selects lowest validation classification error, then lower loss/earlier epoch. Weight aggregate metrics by sample count, retain short batches, restore checkpoints, save histories/model/preprocessing lineage and single-pass final predictions. Fixed final refits follow the prescribed selected epoch budget without a holdout monitor. Keep test-assisted A1 stopping private and disclose it.

**Complete:** [training rules](docs/TRAINING_RULES.md), assignment_common, behavioral tests.

## 4. Prepare Part A housing data

Use 2017–2020 train/2021 validation/2022 test; exclude2023 and excluded input fields. Four categorical and six continuous inputs. Fit selection transforms on training only and separately refit encoders/scalers on combined2017–2021 as Q&A requires. Preserve baseline sample input SD/population target SD, unknown category ID0, row membership and source hash.

**Complete:** [housing data](docs/HOUSING_DATA.md); 87,370/29,057/26,702 rows and116,427 combined refit rows.

## 5. Complete A1: baseline and Ray grid search — 13 marks

Draw baseline/count parameters; run Ray's full25 width/embedding grid; select minimum best validation RMSE, show same-axis train/validation RMSE. Fresh combined refit uses explicit professor-authorized test RMSE stopping, retain best epoch E*, report final test RMSE/R2 and the brief's requested train/test refit curve, using saved internal monitoring data. Latest FAQ's no-test-epoch rule applies specifically to A2/A3. Discuss fitting/search evidence and test-assisted independence limit.

**Complete:** [A1](docs/A1_EXPERIMENT.md): default3,505 parameters; selectedH256/d4,7,065, validationRMSE58,627.10/E_val6; refitE*=4, testRMSE96,901.64/R20.676082.

## 6. Complete A2: controlled ablations across five seeds — 13 marks

Train fresh baseline, continuous-only with embeddings removed, and Sigmoid replacing hidden ReLU on combined2017–2021 for seeds42–46. Preserve selected preprocessing, H/d/LR/batch and E*=4. Test each final model once; report mean/sampleSD RMSE/R2 and conclusions. Separately show the two ablations' genuine train/validation RMSE/R2 versus epochs. Use frozen seed42 original-split diagnostics with train-only transforms, exactly4epochs and no test pass; never treat combined-training rows as held-out validation or retune final settings from diagnostics.

**Complete:** [A2](docs/A2_EXPERIMENT.md):15 final fits plus3 diagnostic fits; separate categorical/activation epoch figures; unchanged five-seed final results.

## 7. Complete A3: wide-and-deep comparison — 13 marks

Implement shared encoded input to MLP and single-output linear branches, outputMLP+lambda*linear. Search all five lambdas on original split holding A1 settings fixed; show saved train/validation RMSE/R2 curves. Fresh selected-lambda combined-data fits for five matching seeds at E*=4; compare mean/sampleSD final test metrics against A2 baseline. Explain plausible optimization/generalization mechanism and at least one confound.

**Complete:** [A3](docs/A3_EXPERIMENT.md):lambda2, validationRMSE57,905.65; finalRMSE101,413.74±6,407.58/R20.644081±0.044905. Existing epoch evidence included; no new A3 test pass.

## 8. Complete A4: FeatureAblation in Singapore dollars — 6 marks

Use same-seed42 final A1/A3 models and common valid references. Captum ablates ten original features for every test row, inverse scaling once. Aggregate mean absolute SGD effects, plot ten descending labeled scores per model, list top three, compare and disclose reference/correlation/noncausal limits.

**Complete:** [A4](docs/A4_EXPERIMENT.md), full26,702x10 matrices for both models.

## 9. Prepare Part B data and verify all three architectures

Preserve supplied4,708/524/624 partitions; uint8/255 then original-training pixel mean/populationSD, reused consistently. Confirm labels0healthy/1pneumonia from Q&A. Implement all FigureB layers with no convbias, FCbias, Conv/BN/ReLU, kernel//2 SAME padding, stride2 alignment, pools and post-residual ReLU. B1 hiddenFC dropout0.5; B2 no dropout; B1/B2 raw FC1 directly BCEWithLogitsLoss; B3 same-loss output interpretation explicit. Audit duplicates, shapes, counts and gradients.

**Complete:** [Part B preparation](docs/PART_B_PREPARATION.md);99,569/18,185/77,169parameters.

## 10. Complete B1: AlexNet-like learning-rate selection — 13 marks

Run five learning rates1e-5…1e-1 atbatch128; save separate validation loss/accuracy figures and coherent required table. Select lowest classification error; discuss convergence and sensitivity/specificity. FreezeLR/selectedepoch; **initialize fresh model/Adam**, train5232combinedrows for selected9epochs, then test final model once. Preserve separate selection/final hashes and update final evidence.

**Complete:** [B1](docs/B1_EXPERIMENT.md):LR0.001/epoch9; finalaccuracy90.7051%, sensitivity97.6923%, specificity79.0598%.

## 11. Complete B2: VGG-like batch-size selection — 13 marks

ReuseB1LR; run batches16/32/64/128/256. Separate loss/accuracy figures, required table, lowest-error selection/reasoning including update-budget confound. Freezebatch/epoch; fresh combined-data13epoch final refit and one test.

**Complete:** [B2](docs/B2_EXPERIMENT.md):batch16/epoch13; finalaccuracy85.2564%, sensitivity100%, specificity60.6838%.

## 12. Complete B3: ResNet-like model and comparison — 13 marks

ReuseB1LR/B2batch. Original-split validation run freezesepoch9; fresh combined-data final refit for9epochs, one final test. Report accuracy/sensitivity/specificity and compare all fresh finalmodels in required table with error/parameters/FLOPs. Define counting convention/exclusions and pros/cons; avoid attributing recipe differences solely to architecture.

**Complete:** [B3](docs/B3_EXPERIMENT.md):finalaccuracy81.0897%, sensitivity100%, specificity49.5726%; complete corrected comparison and unchanged architecture counts.

## 13. Complete B4: Grad-CAM comparison — 6 marks

Keep uniform seed42 pneumonia drawIDs486,279,413,53,276. Use exact fresh final-refit models and last spatial conv positive-logit Captum maps. Show five raw originals/15overlays with common display protocol, saved probabilities and quantitative/visual context. Confirm final-model hashes, stored gradients/maps and unchanged model state; disclose coarse/relative maps and missing lesion truth. Selected-image attribution is separate from full-test performance.

**Complete:** [B4](docs/B4_EXPERIMENT.md), regenerated from current final models without redrawing examples.

## 14. Write two concise reports from saved evidence

Place accurate GenAI declaration first; include all eight question answers, required tables/figures/interpretations and limited primary references withinfivepages each. Include A2/A3 epoch evidence and correctedBfinals/B4. Rebuild PDF/Markdown and verify every reported value/current sourcehash/pagegeometry, then visually reviewalltenpages. Source manifest includesQ&A.md; archived reports retain historical status.

**Complete:** [report completion record](docs/REPORTS_COMPLETED.md); currentaudit in reports/verification.json.

## 15. Verify and package submission

Run meaningful behavioral/saved-evidence checks; explicit allowlists create separate independently runnableZIPs with raw/prepareddata, checkpoints, source PDFs/Q&A, pinnedrequirements, seeds and commands. Archive previous fourfiles, rebuildreports and packages, freshly extract both ZIPs with isolated paths; test localpreprocessing/finalcheckpointloading/syntheticforwards/all exporters/Rayworker. No verification claims fullrerun or freshdependencyinstallation. Currentmanifests/checksums are authoritative.

**Local work complete; identity/portal pending:** [submission verification](docs/SUBMISSION_VERIFICATION.md), submission/manifest.json and latestextractionaudit. Runfinalize_submission.py withactualname, upload exactlyfourfiles toNTULearn, verify filenames/sizes, completeportal submission and retainreceipt. No name or authenticatedportal receipt has been supplied.

## Final completion checklist

- [x] A1–A4 and B1–B4 answer evidence follows brief/slides/latestFAQ.
- [x] Genuine A2/A3 train/validation epoch plots; separate A2 ablations.
- [x] Fresh B1/B2/B3 combined-data final models;9/13/9selectedepochs; one currentfinaltest per model.
- [x] B4 uses currentfinalcheckpoint hashes and samefixed fiveimages.
- [x] Currentlabels/dropout/outputwording and declaredremaininginterpretations.
- [x] Twofive-pagePDFs and two standalone codeZIPs; updatedREADME/provenance/verification/checksums.
- [ ] ReplaceYourName everywhere usingactualstudentname.
- [ ] NTULearn final upload/submission andreceipt; verifyportal cutoff.
