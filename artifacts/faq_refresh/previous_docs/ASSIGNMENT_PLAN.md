# SC4001 assignment: step-by-step completion plan

**October 10, 2026 update:** the new [Q&A.md](Q&A.md) supersedes the no-refit Part B instructions and final-only A2/A3 plot interpretation below. Completed-run statuses below describe the earlier protocol. Current submission completion requires the [FAQ corrections](docs/FAQ_COMPLIANCE_REVIEW.md):

- [ ] Add separate A2a/A2b training/validation RMSE and R² epoch plots with genuine held-out validation evidence.
- [ ] Include A3's already saved train/validation RMSE and R² epoch curves in the Part A report.
- [ ] Refit fresh B1/B2/B3 models on combined train+validation for selected 9/13/9 epochs and test each once.
- [ ] Refresh Part B final metrics/comparison and regenerate B4 from the corrected final checkpoints.
- [ ] Update confirmed label/B1/B2 output wording, protocol documentation and checks; rebuild and verify both reports and packages with FAQ provenance.

Prepared on **October 1, 2026**, after reading all seven pages of [assignment.pdf](assignment.pdf), all six pages of [assignment_slides.pdf](assignment_slides.pdf), including the architecture diagrams, and the professor's guidance supplied in [clarifications.md](clarifications.md).

**Deadline: October 12, 2026, via NTULearn.** The documents do not specify a submission time; check the NTULearn submission portal. Aim to submit on October 11.

**Progress: steps 1–5 completed on October 1, 2026; steps 6–14 and step 15's local verification/packaging completed on October 3, 2026. Student-name replacement and NTULearn submission/receipt remain pending.** See [README.md](README.md), [submission verification](docs/SUBMISSION_VERIFICATION.md), [setup](docs/SETUP_REPORT.md), [training rules](docs/TRAINING_RULES.md), [housing data](docs/HOUSING_DATA.md), [A1](docs/A1_EXPERIMENT.md), [A2](docs/A2_EXPERIMENT.md), [A3](docs/A3_EXPERIMENT.md), [A4](docs/A4_EXPERIMENT.md), [B1](docs/B1_EXPERIMENT.md), [B2](docs/B2_EXPERIMENT.md), [B3](docs/B3_EXPERIMENT.md), [B4](docs/B4_EXPERIMENT.md), [Part B preparation](docs/PART_B_PREPARATION.md), and [completed five-page reports](docs/REPORTS_COMPLETED.md).

Both assignment PDFs, professor clarifications, supplied source and data have been inspected. Environment probes, A1–A4, Part B preparation, B1–B4 and both five-page final reports are complete. Submission-package verification, author-name replacement and packaging remain.

### Confirmed starting materials and environment

| Item | Verified finding |
| --- | --- |
| Housing project | `housing_baseline_data/hdb-price-prediction-assignment-main/`, containing README, `pyproject.toml`, `uv.lock`, CSV, and `src/{baseline,config,data,model}.py` |
| Housing splits | 87,370 rows in 2017–2020; 29,057 in 2021; 26,702 in 2022. Combined refit data: 116,427 rows. The CSV also has 16,424 rows from 2023, which must be excluded. |
| Housing quality checks | No missing source values, empty required categories, exact duplicate rows, or nonfinite required numeric values. Step 4 saves strict validation and unknown-category checks. |
| Baseline architecture | Four embeddings → concatenate six continuous inputs → `Linear(38,64)` → `LayerNorm(64)` → ReLU → `Linear(64,1)`; output represents standardized price. |
| Baseline defaults | Embedding dimensions `(8,8,8,8)`, hidden width 64, Adam learning rate `1e-2`, batch size 512, seed 42; validation RMSE early stopping with the required 20/4/`1e-4` settings. |
| Embedding table sizes | Including the unknown-category entry: month 13, town 27, flat model type 44, storey range 18. These sizes are also unchanged on 2017–2021. |
| Baseline parameters | **3,505 trainable parameters**, including LayerNorm's scale and bias; calculated from source/cardinalities and verified on the instantiated model during step 2. |
| Completed A1 | Full 25-trial CUDA Ray grid selected **H=256,d=4**, 7,065 parameters, validation RMSE SGD 58,627.10, `E_val=6`. Fresh combined-data refit selected **E*=4**; final test RMSE SGD 96,901.64 and R² 0.676082 under the clarified test-assisted stopping protocol. |
| X-ray archive | `chestxray_binary_64.npz`: `train_images/train_labels`, `val_images/val_labels`, `test_images/test_labels`. Images are `uint8`, 64×64, in `[0,255]`; divide by 255 once. |
| X-ray partitions | Train: 4,708 images, label counts 0:1,214 / 1:3,494. Validation: 524, counts 0:135 / 1:389. Test: 624, counts 0:234 / 1:390. Working class mapping: 0 = healthy, 1 = pneumonia; this is an assumption because the archive has no class-name metadata. |
| Hardware | User selected local NVIDIA GPU/CUDA. Read-only hardware check identified **RTX 3060 Ti, 8,192 MiB**. |
| Environment setup | Project-local `.venv`, Python 3.12.0; PyTorch 2.11.0+cu128/CUDA 12.8; NumPy 2.5.2; pandas 3.0.5; Matplotlib 3.11.1; Ray 2.58.0; Captum 0.9.0; ipykernel 7.3.0. CUDA, dependencies, and Ray GPU workers verified. |
| Instructor clarifications | Incorporated `clarifications.md`: plot training/validation metrics against epochs rather than test metrics; present the two ablations separately; report final test metrics in tables and/or bar plots; use the test partition for early stopping in A1's combined-data refit. |

## 1. Establish the requirements and document working assumptions

Use the supplied professor's clarifications first, then the detailed assignment brief, then the slides. Use reasonable, explicitly recorded assumptions only where these materials leave a gap.

| Item | Requirement / working interpretation | Source |
| --- | --- | --- |
| Scope and marks | Complete A1–A4 and B1–B4. Each part is worth 45 marks; presentation is worth 10 marks. | Brief pp. 3–7; slides p. 6 |
| Training | Adam; maximum 20 epochs per model; early-stopping patience 4; minimum improvement `1e-4`. A2 also specifies an identical A1-selected epoch budget for all runs. | Brief pp. 1, 3 |
| Evaluation | Select hyperparameters using validation data. The professor explicitly permits test-based early stopping for A1's final combined-data refit. | Brief pp. 1–2, 5; clarifications |
| Epoch plots | Plot training/validation metrics against epochs, **not test metrics**. This supersedes A1's original request for a training/testing epoch plot. | Clarifications |
| Ablation presentation | Present the two A2 ablation experiments separately. Final test RMSE/R² may be reported in tables and/or bar plots. | Clarifications |
| Reports | Two separate PDF reports, each **less than six pages**. Target at most five pages each, including the declaration, to satisfy the stricter wording. | Brief p. 1; slides p. 6 says six pages |
| Code | Separate ZIPs containing clean, labelled, runnable `.py` files or `.ipynb` notebooks, with a README. Python scripts alone satisfy the brief. | Brief p. 1 |
| README | Describe environments, dependencies, seeds, data setup, and commands. | Brief p. 1 |
| GenAI declaration | Put a declaration at the beginning of each report: tools used, purposes, and how their outputs were verified. Planning assistance already counts as use. | Brief p. 1 |
| Individual work | The submission must be the student's own unique work, understood and verified by the student. | Brief p. 1 |
| A3 lambda grid | Use `{0, 0.25, 0.5, 1, 2}`. The slides instead print `{0, 0.25, 0.5, 1.2}`. | Brief p. 4; slides p. 3 |
| B2 batch sizes | Use `{16, 32, 64, 128, 256}`. The slides print `258`. | Brief p. 6; slides p. 5 |
| Late penalty | The brief states 5% per day, up to three days. Submit before the deadline. | Brief p. 1 |

The user confirmed that all available materials have been provided and authorized using the best reasonable assumptions. Proceed under these choices, recording them in the methods and README; revise them if later official guidance becomes available:

1. **Materials located:** use the extracted housing project and X-ray archive listed above; the original housing ZIP is not needed to proceed.
2. **Compute confirmed:** use the local RTX 3060 Ti. Verify the installed PyTorch build actually supports CUDA, set Ray concurrency from measured memory usage, and estimate runtime after a short benchmark.
3. **Combined-data refits and early stopping:** Follow the professor's explicit answer: monitor test RMSE for early stopping in **A1's final 2017–2021 refit**. Save its best checkpoint and call that checkpoint's epoch `E*`; distinguish it from the best original validation epoch `E_val`. Use `E*` as the fixed epoch budget for all A2 runs, as A2 specifically requests equal epochs, and for A3 final refits as a controlled-comparison assumption. Keep A1's test-monitoring history internal; show final test scores in tables/bar plots rather than epoch plots. This test-assisted checkpoint selection means A1's final test score is not an independent held-out estimate; the inherited epoch budget also limits independence of the later comparisons. Describe this limitation plainly.
4. **Final CNN activation:** Interpret Figure B's “all the layers” ReLU wording as applying to convolutional and hidden fully connected layers and residual additions, excluding the final `fc, 1`. Use an unrestricted output logit with binary cross-entropy. ReLU on a logit would constrain sigmoid probabilities to at least 0.5. Document this interpretation because the PDF does not explicitly state the output-layer exception.
5. **X-ray class mapping:** Assume label 0 = healthy and label 1 = pneumonia. Treat this as an explicit dataset-semantic assumption rather than claiming it was encoded in archive metadata. Use it consistently for sensitivity, specificity, positive-class probabilities, and Grad-CAM.

The supplied professor guidance resolves the epoch-plot and A1 refit-monitoring questions. No further clarification is a prerequisite under the user's instruction to make reasonable assumptions. If official updates arrive, record them and update the affected protocol. For reference, the brief directs general queries to the NTULearn Discussion Board and personal queries to `cx4042@e.ntu.edu.sg`.

**Completion evidence:** required files are located, compute is identified, and implementation assumptions are recorded.

**Status: complete.** Decisions are recorded in [docs/ASSIGNMENT_DECISIONS.md](docs/ASSIGNMENT_DECISIONS.md) and [configs/project.json](configs/project.json), including the supplied clarifications and user-authorized assumptions.

## 2. Set up a reproducible project

1. Preserve the supplied housing project as the reference. Its README instructs running `uv sync` and then `uv run python src/baseline.py` from that project's root. The supplied baseline trains standardized targets with MSE; its reported RMSE is converted back to SGD.
2. Use standard Python `venv` plus pip, as the user requested. The root `.venv` is installed and verified. Keep the supplied lock as reference: its PyTorch 2.13.0 pin was unavailable on the chosen official CUDA 12.8 index, so use the verified 2.11.0+cu128 build. Other direct dependencies match the supplied Python-3.12 lock. Exact installed versions and the CUDA wheel index are saved in the root and both parts' `requirements.txt` files. The smoke script explicitly moves models and batches onto CUDA; apply the same device handling in production code at step 3.
3. Use PyTorch for models, `ray[tune]` for the required A1 search, and Captum for A4 and B4. Use NumPy/pandas for data, a plotting library for figures, and consistent metric utilities.
4. Separate Part A and Part B code and outputs so each submission can run independently. A practical proposed layout is:

```text
SC4001 assignment/
  ASSIGNMENT_PLAN.md
  README.md
  .venv/                        # all assignment packages; exclude from submissions
  requirements.txt              # full installed pins and CUDA wheel index
  assignment_common/            # shared trainer, metrics, artifacts, reproducibility
  tests/                        # behavior checks for the training/evaluation rules
  configs/project.json          # seeds, resource settings, and decision defaults
  docs/                         # decisions, setup evidence, GenAI log
  scripts/                      # venv setup and CUDA/Ray smoke checks
  artifacts/setup/              # verification reports and provenance
  data/                         # supplied files; preserve the originals
  part_a/
    README.md
    requirements.txt
    src/                        # preprocessing, models, training, metrics, XAI
    scripts/                    # run A1, A2, A3, A4 and regenerate figures
    configs/
    outputs/                    # histories, checkpoints, tables, figures
  part_b/
    README.md
    requirements.txt
    src/
    scripts/                    # run B1, B2, B3, B4 and regenerate figures
    configs/
    outputs/
  reports/
  submission/
```

5. Fix the selection seed as the supplied default `42`, and use five distinct Part A evaluation seeds, proposed as `[42, 43, 44, 45, 46]`. Use the same five seeds for the compared models. Seed Python, NumPy, PyTorch, and data-loader shuffling; record deterministic settings and any remaining nondeterminism.
6. Save each run's configuration, category maps, continuous/target statistics, epoch history, checkpoint, seed, and metrics. The supplied checkpoint saves only `state_dict`; its category maps and scalar target statistics are not all recoverable from that file, so export them explicitly. Export tables and plots from saved results rather than transcribing them manually.
7. Measure one epoch on a small, valid run before scheduling the full experiments. Start with one full-GPU Ray trial at a time on the 8 GB device; increase concurrency only if measured resource usage supports it. Keep all required grid values even if trials run sequentially. Use import-safe entry points for Windows multiprocessing.

**Completion evidence:** both parts have a reproducible environment, a documented run structure, and a successful loading/forward/backward smoke run.

**Status: complete.** Dependency consistency, actual CUDA execution, the supplied baseline's parameter count, housing/X-ray loading and optimizer updates, Captum methods, plotting, and two Ray Tune GPU-worker trials passed. Setup-only subset timing: 4,096 housing rows in approximately 0.215 s and 512 X-rays in approximately 0.127 s for one probe epoch. These are not full-experiment runtime estimates. See [docs/SETUP_REPORT.md](docs/SETUP_REPORT.md).

## 3. Implement shared training and evaluation rules

1. Implement Adam and the maximum-20-epoch, patience-4, minimum-improvement-`1e-4` rule. Preserve the supplied A baseline's validation-RMSE monitor and SGD units for selection runs. For A1's combined-data refit, use test RMSE as explicitly clarified by the professor. Use validation loss as the proposed B stopping monitor and document it.
2. Keep early-stopping monitoring separate from the question's selection criterion. For A1/A3, save the minimum-validation-RMSE checkpoint. For B1/B2, save the minimum-validation-classification-error checkpoint; break equal-error ties by lower validation loss, then earlier epoch. B's loss monitor can determine when training ends while the error checkpoint determines the final selected model.
3. Calculate every row of a results table from the same selected checkpoint. Record both its best epoch and the total epochs completed.
4. Evaluate with dropout disabled and BatchNorm in evaluation mode. Compute dataset-level metrics correctly, weighting unequal batch sizes by sample count. The baseline's printed training RMSE aggregates predictions made during updates, whereas validation RMSE uses the completed epoch model; for directly comparable learning curves, also evaluate the full training set at the epoch checkpoint and clearly label the chosen convention. Grad-CAM additionally needs gradients enabled.
5. Follow the professor's clarified plotting rule: show training/validation epoch curves and report final test metrics in tables and/or bar plots. A1's final refit is the explicit exception to test-only final evaluation: inspect test RMSE internally for early stopping, then report the restored checkpoint's final RMSE/R². Do not extend that exception to hyperparameter selection or other test-based checkpoint searches. A2 requires one final test evaluation per trained run; B1/B2 require one final test evaluation each.
6. Use these metric definitions consistently:

| Metric | Definition |
| --- | --- |
| RMSE | `sqrt(mean((prediction - target)^2))`, in Singapore dollars |
| R² | `1 - sum((target - prediction)^2) / sum((target - mean(target))^2)` |
| Accuracy | `(TP + TN) / (TP + TN + FP + FN)` |
| Classification error | `1 - accuracy` |
| Sensitivity | `TP / (TP + FN)`, with pneumonia as the positive class |
| Specificity | `TN / (TN + FP)` |
| Five-seed sample standard deviation | `sqrt(sum((metric - mean_metric)^2) / (5 - 1))` |

Store individual seed results as well as their means and sample standard deviations. Verify metric calculations on a small known example and check that restoring a checkpoint reproduces its recorded validation metrics.

**Completion evidence:** checkpoint selection, early stopping, metrics, and logging work before the full searches begin.

**Status: complete.** Implemented common utilities and Part A/B entry points. Passed 23 behavior tests, then verified CUDA training on small supplied housing/X-ray subsets and reproduced selected validation metrics after loading fresh models. Run artifacts preserve metadata, preprocessing, coherent selected metrics, and public/internal histories. See [docs/TRAINING_RULES.md](docs/TRAINING_RULES.md) and [artifacts/training_checks/latest.json](artifacts/training_checks/latest.json). Full data preparation and assignment experiments remain for later steps.

## 4. Prepare Part A's housing data

Source: brief p. 2.

1. Use `housing_baseline_data/hdb-price-prediction-assignment-main/hdb_price_prediction.csv`. Verify the schema, missing/empty values, exact duplicates, finite numeric values, category counts, and year coverage. Follow the baseline's documented processing and record any necessary additional handling.
2. Make the required chronological split: **2017–2020 training (87,370), 2021 validation (29,057), 2022 final test (26,702)**. Exclude the 2023 rows completely. Do not replace this with a random split.
3. Use exactly these ten original model inputs:

| Categorical inputs: four separate embedding tables | Continuous inputs: six values |
| --- | --- |
| `month` | `dist_to_nearest_stn` |
| `town` | `dist_to_dhoby` |
| `flat_model_type` | `degree_centrality` |
| `storey_range` | `eigenvector_centrality` |
| | `remaining_lease_years` |
| | `floor_area_sqm` |

4. Predict `resale_price`. Use `year` only for splitting. Exclude `full_address` and `nearest_stn` as instructed.
5. Preserve the supplied training-only mappings: sorted category values are assigned IDs starting at 1 and unknown categories use ID 0. Standardize continuous inputs using training means and sample standard deviations, clamped to at least `1e-8`; standardize targets using training mean and population standard deviation. Do not learn these quantities from the test set.
6. The supplied baseline has no combined-data refit implementation. Under the proposed protocol, fit preprocessing on 2017–2021 (116,427 rows) once and reuse exactly that preprocessing across baseline, ablations, and wide-and-deep comparisons. Keep the selection and refit transformations separately identified. The categorical cardinalities remain the same, but scaling statistics can change.
7. Compute reported RMSE and explanations in original SGD units, reversing target transformations when needed.

**Completion evidence:** saved split counts, category cardinalities, ten-feature schema, and preprocessing artifacts, with no test-data fitting.

**Status: complete.** Exported chronological selection and combined-refit datasets, typed category maps, scaling statistics, source checksum, quality/schema report, and original-row split membership under `part_a/outputs/data/`. Both protocols match every supplied-baseline tensor and statistic exactly; all 33 behavior tests passed, including leakage and artifact round-trip checks. Five test rows have unseen `flat_model_type="5 ROOM, 3Gen"` and map to ID 0. No cleaning-based row removal/imputation or prediction-performance evaluation was needed. See [docs/HOUSING_DATA.md](docs/HOUSING_DATA.md) and [artifacts/housing_checks/latest.json](artifacts/housing_checks/latest.json).

## 5. Complete A1: baseline and Ray grid search — 13 marks

Source: brief p. 3; slides p. 2.

1. Draw the supplied baseline's block diagram and verify its parameter count. The inspected defaults are embedding dimension 8 and hidden width 64; category table sizes are 13, 27, 44, and 18, including unknown entries.
2. Show its exact flow: four embeddings concatenate with six continuous inputs → hidden linear layer → **LayerNorm** → ReLU → one linear output in standardized price units → inverse target scaling for SGD predictions. Preserve LayerNorm throughout the tuned MLP and subsequent controlled comparisons.
3. With shared embedding **dimension** `d`, hidden width `H`, and embedding-table sizes `C1` through `C4`, the input width is `D = 4d + 6`. Including LayerNorm's trainable scale/bias, the count is:

```text
P = d(C1 + C2 + C3 + C4) + (D × H + H) + 2H + (H + 1)
  = d(C1 + C2 + C3 + C4) + (4d + 10)H + 1

Default P = 8(13 + 27 + 44 + 18) + (4×8 + 10)64 + 1 = 3,505
```

The four tables have independent weights; only their dimensions match. Cross-check the formula against the instantiated model after environment setup. Report the supplied baseline count and clearly distinguish the tuned model if its count changes.

4. Run the full **25-combination Ray Tune grid**:
   - Hidden width: `{16, 32, 64, 128, 256}`.
   - One embedding dimension applied to all four tables: `{2, 4, 8, 12, 16}`.
5. Keep the other baseline settings fixed: Adam learning rate `1e-2`, batch size 512, MSE on standardized targets, LayerNorm, and selection seed 42. Log training and validation RMSE each epoch, and each trial's minimum validation RMSE, best epoch, and checkpoint. Ray's grid search enumerates the Cartesian product of grid variables. [Ray grid-search documentation](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.grid_search.html).
6. Select the configuration with the **lowest best validation RMSE**. For exact ties, use fewer trainable parameters, then an explicitly documented deterministic ordering. Freeze `H*` and `d*`; record the original selected checkpoint epoch as `E_val`.
7. Plot training and validation RMSE together for the selected trial. Export the complete grid table and a compact heatmap to support the model-selection discussion.
8. Initialize a fresh selected model and retrain on **2017–2021 combined data**. Under the professor's explicit clarification, monitor 2022 test RMSE with maximum 20 epochs, patience 4, and minimum improvement `1e-4`; restore the best checkpoint. Record its best epoch as `E*` for A2's common epoch budget, separately from the later stopping epoch.
9. Follow the professor's updated plotting instruction: **omit the superseded training/testing epoch plot**. Show the selected trial's training/validation plot; an additional combined-data training-only curve is optional. Report the final restored refit checkpoint's test RMSE and R² in a table and/or bar plot, and disclose the test-assisted stopping protocol.
10. Explain why the selected width and embedding size are supported by the grid and curves. Discuss underfitting or overfitting using actual errors and gaps. Recognize that chronological market changes can also affect the validation/test gap.

**Completion evidence:** baseline diagram and parameter count, 25 trial results, selected settings, recorded `E_val` and refit `E*`, training/validation curves, final RMSE/R², and evidence-based discussion including the test-assisted stopping limitation.

**Status: complete.** Ran the full grid and fresh final refit on the RTX 3060 Ti in approximately 12.3 minutes. Selected **H=256,d=4**, 7,065 parameters, validation RMSE **SGD 58,627.10** (5.61% lower than the supplied default), `E_val=6`. Combined-data refit restored **E*=4** after 8 completed epochs; final test RMSE **SGD 96,901.64**, R² **0.676082**. Saved every trial/checkpoint/history, complete tables, baseline diagram/count, selected learning curves, heatmap, final checkpoint, frozen A2/A3 settings, and discussion under `part_a/outputs/a1/`. Passed 38 tests and reproduced all 25 restored validation scores on CUDA; audited stopping, metadata, final metric coherence, and unchanged supplied files. No test epoch plot was produced. See [docs/A1_EXPERIMENT.md](docs/A1_EXPERIMENT.md) and [artifacts/a1_checks/latest.json](artifacts/a1_checks/latest.json).

## 6. Complete A2: controlled ablations across five seeds — 13 marks

Source: brief p. 3; slides p. 2.

1. Freeze the selected A1 preprocessing, learning rate, batch size, hidden width, embedding dimension where applicable, and fixed epoch budget `E*`.
2. Build the three comparison models:
   - Selected A1 baseline with categorical and continuous inputs and hidden ReLU.
   - Continuous-only model: remove all four embedding tables; change the first linear input width to 6; preserve hidden width and LayerNorm.
   - Sigmoid model: preserve all inputs, embedding tables, and LayerNorm; replace only the hidden ReLU with Sigmoid.
3. Train each model on combined 2017–2021 data for each of the five seeds: **15 fresh fixed-epoch runs**. Include a fresh seed-42 baseline run so all A2 runs actually complete the same `E*` epochs and receive one final test evaluation, independently of A1's test-monitored stopping procedure.
4. Keep epoch counts the same across these runs, as A2 requires, under the documented refit interpretation. Evaluate each trained model once on the 2022 test set.
5. Report the mean and sample standard deviation of test RMSE and R² for all three variants in a table. Follow the professor's separate-ablation instruction: make one figure comparing baseline versus continuous-only and a separate figure comparing baseline versus Sigmoid. Each figure can contain RMSE and R² panels with labelled error bars and/or seed points. Test scores belong in these final-result plots, not epoch curves.
6. Discuss what the evidence says about categorical information and activation choice. Explain plausible effects, including Sigmoid saturation, without assuming either ablation must perform worse. Note that removing embeddings also changes parameter count.

**Completion evidence:** per-seed results, three-model mean/standard-deviation table, separately presented ablation figures covering RMSE and R², and controlled-comparison discussion.

**Status: complete.** All 15 fresh CUDA runs completed on October 3 using the exact saved refit preprocessing, frozen H=256/d=4 where applicable, learning rate 0.01, batch size 512, and exactly four epochs. Test RMSE mean ± sample SD: baseline **SGD 102,427.90 ± 5,289.17**, continuous-only **108,252.68 ± 9,605.81**, Sigmoid **101,821.96 ± 4,082.21**. R² mean ± sample SD: **0.637311 ± 0.037428**, **0.593204 ± 0.068794**, and **0.641892 ± 0.028889**, respectively. Categorical removal worsened RMSE in four of five seeds; Sigmoid's small mean gain occurred in three of five. Saved all checkpoints, per-seed/summary/paired tables, separate PNG/vector-PDF figures, and discussion in `part_a/outputs/a2/`. Passed 44 tests; restored all 15 checkpoints and reproduced their training metrics on CUDA, audited controls/statistics, and verified unchanged supplied/A1 files without another test evaluation. See [docs/A2_EXPERIMENT.md](docs/A2_EXPERIMENT.md) and [artifacts/a2_checks/latest.json](artifacts/a2_checks/latest.json). A3 should reuse this five-seed baseline and the frozen four-epoch recipe.

## 7. Complete A3: wide-and-deep comparison — 13 marks

Source: brief pp. 3–4 and Figure A; slides p. 3.

1. Implement one shared concatenated input of the four embeddings and all six continuous features. Feed that same input into the selected A1 MLP and a single-output linear branch.
2. Combine branch outputs in the baseline's standardized target space, then apply the target inverse transform once: `price = target_mean + target_std × (MLP(input) + lambda × Linear(input))`. Lambda is a fixed searched scalar, not an extra learned parameter. Keep A1 hyperparameters fixed, including LayerNorm; do not add the target mean independently to both branches.
3. Search `{0, 0.25, 0.5, 1, 2}` using the original training/validation split and a documented minimum-validation-RMSE selection rule. Do not select lambda from test scores.
4. Verify that lambda 0 reproduces baseline behavior when the MLP parameters, inputs, and initialization are matched. Creating an extra branch must not unintentionally change the baseline MLP's random initialization in this check.
5. Refit the selected lambda configuration on combined 2017–2021 data for the same five seeds, using the A1 final refit's epoch budget `E*` for the controlled comparison. Do not select a new stopping epoch from A3 test results. Change this only if later official guidance specifies another protocol.
6. Report mean and **sample** standard deviation of test RMSE and R²; plot both metrics. Compare against the matching five-seed baseline from A2, alongside A1's individual final result as appropriate.
7. Explain the possible optimization/generalization mechanism and whether the observations support it. For an unconstrained linear branch, positive lambda values can be absorbed into its weights, so different positive lambdas need not create different function classes; training dynamics and any regularization may instead explain differences. Confirm the baseline's regularization before using this argument.
8. Discuss at least one confound or limitation: extra parameters, initialization, selection on a single seed, the fixed short training budget, or its inheritance from A1's test-assisted stopping. Do not claim statistical significance from five means alone.

**Completion evidence:** lambda search results, selected lambda, five-seed metrics and plots, baseline comparison, mechanism explanation, and at least one limitation.

**Status: complete.** All five validation-only lambda candidates and five fresh four-epoch CUDA refits completed on October 3. Validation selected **λ=2**, RMSE **SGD 57,905.65**, R² **0.873239**, saved epoch 3 (7 completed). Final five-seed mean ± sample SD: test RMSE **SGD 101,413.74 ± 6,407.58**, R² **0.644081 ± 0.044905**. Against the matching saved A2 baseline, mean RMSE improved **0.99%** in four of five seeds; the gain is modest relative to paired variation. The model has **7,088 parameters**, including a 23-parameter linear branch. Lambda 0 reproduced A1's baseline weights and metrics exactly. Saved all ten checkpoints, tables, four PNG/vector-PDF figures, discussion, and A4 settings under `part_a/outputs/a3/`. Passed 51 tests; restored all ten checkpoints on CUDA, reproduced all training and five validation metric sets, and verified controls/statistics plus unchanged supplied/A1/A2 files without another test evaluation. See [docs/A3_EXPERIMENT.md](docs/A3_EXPERIMENT.md) and [artifacts/a3_checks/latest.json](artifacts/a3_checks/latest.json). The discussion addresses mechanisms, positive-lambda rescaling, extra parameters, single-seed selection, and the inherited test-assisted budget.

## 8. Complete A4: FeatureAblation in Singapore dollars — 6 marks

Source: brief p. 4; slides p. 3.

1. Use final combined-data baseline and selected wide-and-deep checkpoints trained with the **same seed**, proposed as 42. Reuse matching A1/A3 checkpoints.
2. Apply Captum `FeatureAblation` to **every test sample**. Explain the ten original inputs, with each categorical variable treated as one feature. Ablate category indices before their embedding lookup rather than reporting separate embedding coordinates.
3. Provide valid reference values, document them, and use the same references for both models. Proposed references are training/refit-set category modes and continuous means, represented in the model's preprocessing space. Captum supports explicit baselines and feature masks. [Captum FeatureAblation documentation](https://captum.ai/api/feature_ablation.html).
4. Make the explained forward function return predicted prices in SGD, including inverse target transformation if necessary. Check one manual prediction difference against its ablation attribution.
5. Calculate each global score as `mean(abs(attribution_for_feature))` over the full test set, producing exactly ten scores per model. Process batches if memory is limited; aggregate over all samples.
6. Produce one labelled descending bar plot per model, with an SGD axis. Report each model's top three features and compare their rankings and magnitudes.
7. Explain one limitation, such as reference dependence or correlated features producing unrealistic ablated inputs. Treat these scores as model explanations rather than causal effects on housing prices.

**Completion evidence:** full-test attributions or reproducible aggregates, two ten-feature bar plots in SGD, top-three lists, comparison, and a limitation.

**Status: complete.** Captum FeatureAblation explained all **26,702 test samples × ten original features** for both final seed-42, epoch-4 A1/A3 checkpoints on CUDA. Shared references use 2017–2021 training categorical modes and saved continuous means; categorical IDs are ablated before embedding lookup and the target inverse transform is applied once. Baseline top three: **floor area SGD 71,537.72**, **remaining lease 46,187.61**, **distance to Dhoby Ghaut 41,163.95**. Wide-and-deep top three: **floor area 73,119.34**, **remaining lease 48,364.21**, **flat model type 42,280.28**. These are full-test mean absolute SGD attributions; two top-three features are shared, and only third/fourth place swap. Saved full signed matrices/row IDs, exact references, global/ranked/comparison tables, two descending PNG/vector-PDF plots with a shared scale, manual checks, and discussion under `part_a/outputs/a4/`. Passed 57 tests; the audit independently recomputed all global means and verified first/last-batch manual SGD differences (zero error), full coverage, model state, and unchanged supplied/A1/A2/A3 files. The discussion records reference dependence, correlated-input replacements, nonlinear effects, one-seed scope, and the models' test-assisted origin. See [docs/A4_EXPERIMENT.md](docs/A4_EXPERIMENT.md) and [artifacts/a4_checks/latest.json](artifacts/a4_checks/latest.json).

## 9. Prepare Part B data and verify all three architectures

Source: brief p. 5 and Figure B; slides p. 4.

1. Use the inspected NPZ keys `train_images/train_labels`, `val_images/val_labels`, and `test_images/test_labels`. Preserve its 4,708/524/624 split sizes. Use the documented assumption that label 0 is healthy and label 1 is pneumonia for clinical metrics and B4 selection.
2. Convert the verified `uint8` image arrays to floating point and divide by 255 once. Calculate mean and standard deviation over **training-set pixels only**, then apply that same normalization to every partition.
3. Add the channel dimension to the stored `N × 64 × 64` arrays to obtain `N × 1 × 64 × 64`. Preserve unnormalized images for overlays.
4. Follow Figure B exactly: BatchNorm then ReLU after every convolution, convolution bias disabled, fully connected bias enabled, stride 1 unless marked `/2`, ReLU after residual sums, and dropout 0.5 only at B1's two hidden fully connected layers. This includes the figure's projection convolutions; do not substitute a library's default ResNet block.
5. Implement and check these layer sequences. `Conv(k, c, s)` means kernel size, output channels, and stride; each convolution includes the prescribed BatchNorm/ReLU.

| Architecture | Required sequence |
| --- | --- |
| B1: AlexNet-like | `Conv(5,16,1) → MaxPool(2,2) → Conv(5,32,1) → MaxPool(2,2) → Conv(3,64,1) → MaxPool(2,2) → AveragePool(output 4×4) → Flatten → FC64/ReLU/Dropout0.5 → FC32/ReLU/Dropout0.5 → FC1` |
| B2: VGG-like | `Conv(3,8,1) → Conv(3,8,1) → MaxPool(2,2) → Conv(3,16,1) → Conv(3,16,1) → MaxPool(2,2) → Conv(3,32,1) → Conv(3,32,1) → MaxPool(2,2) → AveragePool(output 1×1) → Flatten → FC1` |
| B3: ResNet-like | `Conv(3,16,1) → MaxPool(2,2) → residual block16 → residual block32/downsample → residual block64/downsample → AveragePool(output 1×1) → Flatten → FC1` |

6. Implement B3's blocks as drawn:
   - Block16: two `3×3, 16` convolutions; identity shortcut; add and apply ReLU.
   - Block32: `3×3, 32, stride2` then `3×3, 32, stride1`; shortcut `1×1, 32, stride2`; add and apply ReLU.
   - Block64: `3×3, 64, stride2` then `3×3, 64, stride1`; shortcut `1×1, 64, stride2`; add and apply ReLU.
7. Check spatial sizes. B1/B2 pool from 64 to 32 to 16 to 8. B1 then averages to 4×4, yielding 1,024 flattened values. B3 pools to 32, keeps 32 in Block16, and downsamples to 16 then 8.
8. Implement SAME padding so stride-one convolutions preserve dimensions and stride-two layers produce the required halved sizes with compatible shortcut alignment. PyTorch's string `padding='same'` supports only stride 1; use explicit padding for strided layers, checking shape and alignment rather than assuming the string works. [PyTorch Conv2d documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html).
9. Following Step 1's final-output interpretation, train a single unrestricted logit with `BCEWithLogitsLoss`; use sigmoid probabilities and a fixed 0.5 decision threshold for metrics. This loss includes sigmoid internally. [PyTorch BCEWithLogitsLoss documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html).
10. Check forward shapes, one backward pass, and all residual-addition shapes before starting the learning-rate sweep. Record training class imbalance without adding unrequested architecture changes, augmentation, or weighting to the comparison.

**Completion evidence:** saved training normalization statistics, class counts, verified model shapes, and three implementations matching the figure under recorded assumptions.

**Status: complete.** Saved the supplied **4,708/524/624** partitions, labels, original row IDs, raw uint8 overlay images, and normalized float32 NCHW tensors under `part_b/outputs/data/`. Training pixel mean **0.5716994212632378** and population std **0.17409753982645407** are fitted after one division by 255 and reused throughout. Exact hashing found 18 extra training copies, three extra test copies, and eight distinct train/validation shared images; provided membership is preserved and this limitation documented. Implemented Figure B B1/B2/B3 with **99,569 / 18,185 / 77,169** trainable parameters, projection BN/ReLU, center-aligned explicit padding, fixed average-pooling windows, and the recorded unrestricted final logit. Passed **70 tests**, independent full pixel/label/row export checks, residual shape/alignment checks, and CUDA forward/backward/Adam/checkpoint-restoration probes. Largest planned batches 128/256/256 passed; supplied and 724 Part A output-file hashes were unchanged. Smoke runs did not evaluate model test performance or select B hyperparameters. See [docs/PART_B_PREPARATION.md](docs/PART_B_PREPARATION.md) and [artifacts/part_b_checks/latest.json](artifacts/part_b_checks/latest.json). Next: B1's five-rate experiment in step 10.

## 10. Complete B1: AlexNet-like learning-rate selection — 13 marks

Source: brief p. 6; slides p. 5.

1. Use B1 with **batch size 128**. Run the five learning rates `{1e-5, 1e-4, 1e-3, 1e-2, 1e-1}` with the shared training rules and fixed seed.
2. Save validation loss and accuracy per epoch, plus the selected checkpoint and confusion-matrix metrics. Preserve actual early-stopped curve lengths.
3. Produce **two separate figures**: validation loss versus epoch for all five rates, and validation accuracy versus epoch for all five rates.
4. Produce the required five-row table: learning rate, best epoch, validation loss, validation classification error, sensitivity, and specificity. Use metrics from the same selected epoch in each row.
5. Choose the learning rate with the **lowest validation classification error**, following the documented tie rule. Freeze it as `lr*` for B2 and B3.
6. Explain convergence and stability from the curves. Explain why accuracy alone can hide unequal performance on pneumonia and healthy cases, using sensitivity/specificity and the actual class distribution.
7. Evaluate the selected final B1 checkpoint **once** on the test set. Report accuracy, sensitivity, and specificity. Keep the original partitioning; B does not request combined train/validation refits.

**Completion evidence:** five-rate table, two validation-curve figures, justified selected learning rate, and final test metrics.

**Status: complete.** Ran all five rates on CUDA with batch 128, seed 42, identical initialization/shuffling, saved training-only normalization, and Adam with the 20/4/1e-4 stopping rules. Validation selected **LR 0.001, epoch 9**, error **2.0992%**, accuracy **97.9008%**, loss **0.064911**. The run stopped at epoch 12; its minimum loss was at epoch 8, illustrating the separate loss-stopping/error-selection rules. Evaluated the selected checkpoint **once** on all 624 test images: accuracy **85.4167%**, sensitivity **99.7436%**, specificity **61.5385%**, TP/TN/FP/FN **389/144/90/1**. Saved the required five-rate table, two separate PNG/vector-PDF loss/accuracy figures with train/validation panels, histories/checkpoints, same-pass final test predictions, discussion, and `frozen_b1_settings.json` for B2/B3. Passed **76 tests**; all five restored checkpoints reproduced full training/validation metrics exactly on CUDA. Audited stopping/selection histories and final metrics from captured predictions, with no additional test inference; all supplied and 782 previous output-file hashes were unchanged. The report explains slow/unstable learning, minority-class performance, the validation/test specificity gap, and the supplied duplicate-image limitation. See [docs/B1_EXPERIMENT.md](docs/B1_EXPERIMENT.md) and [artifacts/b1_checks/latest.json](artifacts/b1_checks/latest.json). Next: B2 batch-size selection using **LR 0.001**.

## 11. Complete B2: VGG-like batch-size selection — 13 marks

Source: brief p. 6; slides p. 5.

1. Use B2 with the **B1-selected `lr*`**. Run batch sizes `{16, 32, 64, 128, 256}` with fixed remaining settings and the same selection seed.
2. Produce **two separate figures**: validation loss versus epoch and validation accuracy versus epoch, each showing all five batch sizes.
3. Produce the required table: batch size, best epoch, validation loss, validation classification error, sensitivity, and specificity, all evaluated at each run's selected checkpoint.
4. Select by **lowest validation classification error**, using the documented tie rule. Freeze the selected size as `batch*` for B3.
5. Discuss the results using the curves and table. Consider gradient noise, BatchNorm statistics, and the different number of optimizer steps per epoch; do not assume a particular batch size must win.
6. Evaluate the selected final B2 checkpoint **once** on the test set and report accuracy, sensitivity, and specificity.

**Completion evidence:** five-size table, two validation-curve figures, selected batch size with justification, and final test metrics.

**Status: complete.** Ran batch sizes **16/32/64/128/256** on CUDA with the frozen B1 learning rate **0.001**, seed 42, identical VGG initialization/underlying shuffled order, saved training-only normalization, and Adam with 20/4/1e-4 stopping. Validation selected **batch 16, checkpoint epoch 13**, error **2.0992%**, accuracy **97.9008%**, loss **0.069390**. Minimum loss occurred at epoch 9; loss patience expired at 13, while classification error selected epoch 13. Evaluated the exact selected checkpoint **once** on 624 test images: accuracy **85.5769%**, sensitivity **99.2308%**, specificity **62.8205%**, TP/TN/FP/FN **387/147/87/3**. Saved the required five-size table, two separate PNG/vector-PDF loss/accuracy figures with train/validation panels, all 65 epoch records, checkpoints, captured same-pass test predictions, discussion, and `frozen_b2_settings.json`. Passed **82 tests**; all five checkpoints reproduced full train/validation metrics exactly on CUDA. The audit checked histories, Adam/BN update counters, B1/data lineage, final metrics from saved predictions with no additional test inference, and unchanged supplied/833 prior output-file hashes. The report discusses update-budget differences, curve fluctuations, class performance, and validation/test/data limitations. See [docs/B2_EXPERIMENT.md](docs/B2_EXPERIMENT.md) and [artifacts/b2_checks/latest.json](artifacts/b2_checks/latest.json). Next: B3 with **LR 0.001 and batch 16**.

## 12. Complete B3: ResNet-like model and architecture comparison — 13 marks

Source: brief p. 6; slides p. 5.

1. Train B3 using **`lr*` from B1 and `batch*` from B2**, with Adam and the shared early-stopping rules.
2. Select its checkpoint using the same documented B protocol, save its curves, and evaluate the final model on the test set. Report accuracy, sensitivity, and specificity.
3. Count trainable parameters for all three selected models, including BatchNorm affine parameters and fully connected biases, excluding nontrainable running-statistic buffers.
4. Measure inference FLOPs consistently for one `1×1×64×64` image in evaluation mode. State the convention, for example whether one multiply-add counts as two FLOPs, and the profiler's included/unsupported operations. Check convolution/linear estimates against their tensor shapes. Do not label MACs as FLOPs without stating a conversion.
5. Create a comparison table with **test accuracy, sensitivity, specificity, classification error, learnable parameters, and FLOPs** for B1, B2, and B3. Reuse B1/B2's saved test results rather than reevaluating to choose a better result.
6. Explain pros and cons using the observed results, parameter counts, computation, and learning curves. Discuss receptive fields, B1's larger fully connected head, B2's stacked small kernels, and B3's shortcut paths where supported.
7. State that this compares the assignment's selected pipelines: B1 uses batch 128, B2/B3 use `batch*`, and the learning rate was selected on B1. The outcome does not isolate architecture alone; B is not required to be evaluated over five seeds.

**Completion evidence:** B3 metrics/curves, complete three-model comparison table, documented FLOPs convention, and strengths/limitations discussion.

**Status: complete.** Trained Figure B3 on CUDA using the exact B1/B2-frozen **LR 0.001 and batch 16**, seed 42, saved data/normalization, and Adam with 20/4/1e-4 stopping. Validation selected **epoch 9**: accuracy **97.5191%**, error **2.4809%**, BCE **0.061988**; loss patience stopped training at epoch 13. One final evaluation on all 624 test rows gave **accuracy 82.8526%, sensitivity 99.7436%, specificity 54.7009%**, TP/TN/FP/FN **389/128/106/1**. Saved separate train/validation loss and accuracy figures, histories/checkpoint, captured same-pass predictions, B4 settings, and the required B1–B3 final-test/parameter/FLOPs table. Reused existing B1/B2 test results without inference. Learnable parameters are **99,569 / 18,185 / 77,169**; profiled per-image FLOPs **39,063,616 / 19,464,256 / 25,325,696** count two per convolution/FC multiply-add plus residual additions. Recorded included/uncounted operators and verified all estimates against executed shapes. Passed **88 tests**; B3 full train/validation metrics reproduced exactly on CUDA, all three CPU complexity records reproduced without accessing data, and saved test predictions reproduced final metrics/BCE without new test inference. Supplied files and all 884 prior output hashes were unchanged. The report discusses residual/FC/global-pooling tradeoffs, theoretical receptive fields, false positives, different selected training recipes/update budgets, one-seed scope, and supplied-data limitations. See [docs/B3_EXPERIMENT.md](docs/B3_EXPERIMENT.md) and [artifacts/b3_checks/latest.json](artifacts/b3_checks/latest.json). The step 13 Grad-CAM comparison is now complete; see [B4](docs/B4_EXPERIMENT.md).

## 13. Complete B4: Grad-CAM comparison — 6 marks

Source: brief pp. 6–7; slides p. 5.

1. Use a fixed sampling seed to randomly choose **five pneumonia test images**. Save their indices before inspecting their heatmaps and use the same five images for all three models. Document what “representative” means; do not select only attractive explanations.
2. Load the selected B1–B3 checkpoints. Use Captum `LayerGradCam` for the pneumonia output and each model's final main-path convolution before pooling. Record the exact layer for each model.
3. Keep models in evaluation mode while enabling gradients. Apply positive-attribution ReLU to match conventional Grad-CAM, then upsample the maps to 64×64. Captum exposes this through `relu_attributions=True` and interpolation. [Captum LayerGradCam documentation](https://captum.ai/api/layer.html#captum.attr.LayerGradCam).
4. Overlay the maps on the original grayscale images, using the same display protocol, color map, and transparency. If maps are normalized separately for display, state that colors do not establish comparable absolute attribution magnitudes across models.
5. Produce a five-row figure with the original image and B1/B2/B3 overlays: **15 overlays total**. Include sample IDs and predicted pneumonia probabilities where readable.
6. Compare whether the highlighted areas overlap visible white/cloudy regions or instead emphasize borders and background. Explain that coarse heatmaps and the absence of lesion annotations limit conclusions about correct localization.

**Completion evidence:** reproducible five-image selection, 15 labelled overlays, layer/target description, and cross-model comparison.

**Status: complete.** On October 3, sampled positive-class test IDs **486, 279, 413, 53, 276** uniformly without replacement using seed **42**, saved the selection before explanations, and loaded the exact B1/B2/B3 checkpoints. Captum positive-logit Grad-CAM targets the final main-path convolutions in evaluation mode with gradients enabled; native grids are **16×16/16×16/8×8**. Saved all 15 nonempty maps, original images, activations/gradients/channel weights, logits/probabilities, three CSVs, and the five-row original/overlay PNG/PDF with bilinear 64×64 interpolation, per-map maximum normalization, inferno, and shared opacity. All logits/probabilities match the original final pass exactly after repeating only selected inputs to match source batch sizes and discarding padding maps; the initial failed five-row attempt is retained with unchanged selection and no tolerance relaxation. Passed **96 tests**. Independent PyTorch autograd reproduced every tensor exactly on CUDA; source models/data and all supplied/922 prior output hashes remained unchanged. No retraining, retuning, or full-test performance evaluation occurred. The image-by-image report describes central B1, more upper/peripheral B2, and broader/coarser B3 evidence, while distinguishing visible brightness from validated lesion localization. See [docs/B4_EXPERIMENT.md](docs/B4_EXPERIMENT.md), [comparison figure](part_b/outputs/b4/figures/gradcam_comparison.png), and [artifacts/b4_checks/latest.json](artifacts/b4_checks/latest.json). Both step 14 reports are now complete; next is step 15, package verification.

## 14. Write the two concise reports from saved evidence

1. Start both reports with the required GenAI declaration. Name the actual tools used, their purposes, and concrete verification; update it as work progresses. Record this planning assistance now rather than leaving disclosure to the end.
2. Answer A1–A4 and B1–B4 in order. Put each result beside its requested discussion. Keep background brief and avoid unrelated experiments.
3. Target this five-page allocation, adjusting layouts while preserving all required answers:

| Page | Part A | Part B |
| --- | --- | --- |
| 1 | Declaration, concise protocol, A1 baseline diagram/count and search summary | Declaration, normalization/training protocol, B1 table and selection |
| 2 | A1 selected training/validation curves, final refit metrics, fitting discussion and stopping limitation | B1 required curves/test metrics and B2 table |
| 3 | A2 five-seed table, separate ablation figures with both metrics, interpretation | B2 required curves/test metrics and selection discussion |
| 4 | A3 lambda selection, five-seed comparison plots/table, mechanism and limitation | B3 results, architecture/parameter/FLOPs table, pros and cons |
| 5 | A4 ranked plots, top-three features, comparison and limitation | B4 originals/overlays and comparison |

4. Include enough protocol detail to interpret the evidence: splits, seeds, stopping/checkpoint rules, refit budget, units, and uncertainty definitions. Label every figure axis and table column clearly.
5. Check that every claim follows from the recorded results. Report weaker or mixed outcomes honestly. Keep readable font sizes and captions; the report is the primary assessed artifact.
6. Export PDFs and verify the total page count is at most five each under the strict interpretation. Keep necessary references within the limit unless the instructor explicitly exempts them.

**Completion evidence:** two readable PDFs covering every requested answer and beginning with accurate GenAI declarations.

**Status: complete.** On October 3, assembled [Part A](reports/PartA_report_YourName.pdf) and [Part B](reports/PartB_report_YourName.pdf), **five A4 pages each** including GenAI declarations, all requested answers, figures, interpretation and primary references. Matching editable Markdown and sixteen report-sized vector/PNG assets are in `reports/`. Read saved CSV/JSON/tensors directly; performed no new training or model inference. Preserved the clarified plotting rules, fixed-budget comparisons, test-assisted A1 limitation, scoped FLOPs and attribution/data caveats. Rendered and visually inspected all ten pages; corrected title/legend overlaps, missing glyphs and clipped Grad-CAM labels. Independent verification checked page counts, declaration/section order, bounds/font sizes/Unicode/links, all 25 grid/five lambda rows, 20 recomputed seed-statistic values, 40 B1/B2 table values, complete B3 comparison, 20 feature scores and 15 Grad-CAM titles/IDs. All 56 source inputs, 938 prior output hashes and 12 supplied files match. Smallest text is 8 pt; main text is principally 10–10.5 pt. `YourName` remains the identity placeholder pending the user's name and must be replaced consistently before packaging. See [docs/REPORTS_COMPLETED.md](docs/REPORTS_COMPLETED.md), [reports/verification.json](reports/verification.json), and [reports/README.md](reports/README.md). Next: step 15, verify and package the submission.

## 15. Verify and package the submission

**Status: local verification and packaging complete; author and portal actions pending.** Two standalone ZIPs include full READMEs, venv dependency pins, raw/prepared data, this part's code/tests, checkpoints and result assets. Fresh extraction checks passed **57 Part A / 62 Part B tests** (96 distinct), exact preprocessing, selected-checkpoint synthetic CUDA forwards, all eight saved asset exporters and a local Ray CUDA worker. Fixed A1/B3 exporter path portability without changing original experiment evidence. Both reports remain five pages; all 938 earlier output and 12 supplied-file hashes match. The [submission folder](submission/README.md) holds four inspected files and SHA-256 manifests. `YourName` is still an explicit placeholder because the student's name has not been provided. No authenticated NTULearn connection exists here, and no successful upload/receipt is claimed. See [verification details](docs/SUBMISSION_VERIFICATION.md) and [latest audit](artifacts/submission_checks/latest.json).

1. Finish a standalone README for each part: environment/hardware, dependencies and versions, data locations/setup, seeds, exact commands, output locations, and how to regenerate tables/figures. Document the data files required even if they are not bundled in the ZIP.
2. Run each part from a fresh extracted submission folder using only its documented dependencies and data setup. Check for missing modules, stale notebook state, absolute personal paths, or undocumented cross-part dependencies.
3. Check all required experiment results exist. Reuse previously validated results and rerun only when a code/protocol change or discrepancy requires it. Verify reported metrics and plots match the saved artifacts.
4. Package clean code and the README; exclude environments, caches, and unrelated files. Include supporting artifacts only as needed for reproducibility or as required by the supplied README.
5. Replace `YourName` consistently and create exactly:
   - `PartA_report_YourName.pdf`
   - `PartA_codes_YourName.zip`
   - `PartB_report_YourName.pdf`
   - `PartB_codes_YourName.zip`
6. Check the professor's updated presentation requirements one last time: no test-metric epoch plots; separate ablation presentations; final test RMSE/R² in tables and/or bar plots. Open both PDFs and both ZIPs, verify their contents and page counts, then submit the four files via NTULearn before its stated cutoff. Check that the uploads succeeded and retain the submission receipt.

**Completion evidence:** four correctly named, inspected files and a successful NTULearn submission.

## Experiment workload and dependencies

| Work item | Planned training runs | Depends on |
| --- | --- | --- |
| A1 grid | 25 | Housing data, supplied baseline, verified training loop |
| A1 final combined-data refit | 1 | Selected A1 settings; professor's test-RMSE stopping instruction |
| A2 final baseline and two ablations | 15 fresh runs | A1-selected settings and final refit epoch `E*` |
| A3 lambda search | 5 configurations | A1-selected architecture |
| A3 final selected model | 5 seeds | Selected lambda and documented fixed-epoch refit protocol |
| A4 | No new training if paired checkpoints exist | A1/A3 final checkpoints |
| B1 learning-rate search | 5 | Verified B1/data and documented output interpretation |
| B2 batch-size search | 5 | Selected B1 learning rate |
| B3 | 1 | Selected B1 learning rate and B2 batch size |
| B4 | No new training | Three selected CNN checkpoints |

Budget approximately **62 training runs**, including a separate A1 final refit and 15 fixed-epoch A2 runs. Smoke runs or later instructor-required protocol changes add to this estimate. Reusing the lambda-0 selection result may reduce work only when the protocols and initialization truly match. Every early-stopped run has a maximum of 20 epochs, and `E*` cannot exceed 20; estimate actual runtime from the first benchmark rather than assuming a duration.

Selection work for Part A and Part B can progress independently on available hardware. Within Part B, keep the required order **B1 → B2 → B3 → B4**. Within Part A, A1 determines A2/A3 settings, and A4 uses A1/A3 final models.

## Suggested schedule: October 1–12, 2026

| Date | Milestone |
| --- | --- |
| October 1 | Finish document/code/data initialization and plan; files and GPU are located; incorporate professor clarifications and document remaining assumptions. |
| October 2 | Inspect baseline/data; implement shared logging and metrics; verify preprocessing and all model shapes; benchmark. |
| October 3–4 | Complete A1 grid and selection; complete B1 learning-rate search. |
| October 5–6 | Complete A2 refits and A3 search/refits; complete B2 batch-size selection. |
| October 7 | Complete B3, parameter/FLOPs comparison, and all remaining final metrics. |
| October 8 | Complete A4 and B4 explanations and figures. |
| October 9–10 | Write reports; regenerate figures/tables from saved results; verify the submission code. |
| October 11 | Final page-count, README, ZIP, and integrity checks; submit and verify receipt. |
| October 12 | Deadline buffer for upload or verification issues, subject to the portal's actual cutoff. |

Adjust this schedule after the first timing benchmark and based on file availability. Reserve report and verification time because assessment is based primarily on the reports.

## Final completion checklist

**Rechecked October 4:** [every question and subrequirement](docs/FINAL_REQUIREMENTS_RECHECK.md) was compared with the brief/slides, reports, implementation and saved evidence. Corrected the housing SD wording; the CNN final-output caption interpretation remains a disclosed assumption. All 96 root tests and the newhistory and evidence audit passed. Name replacement and portal receipt remain unchecked below.

- [x] Read both PDFs fully and inspect the model diagrams.
- [x] Map all eight questions, marks, submission rules, and document discrepancies into this plan.
- [x] Locate and inspect supplied data, baseline source, README, and GPU hardware.
- [x] Incorporate professor guidance on plots, separate ablations, and test-assisted A1 refit stopping; record remaining user-authorized assumptions.
- [x] Complete step 1: decisions and machine-readable defaults recorded.
- [x] Complete step 2: venv dependencies, project folders/READMEs, seeds/pins, CUDA/data/Captum/plotting smoke checks, and Ray GPU-worker verification.
- [x] Complete step 3: shared training/evaluation, stopping/selection, metrics, checkpoint/preprocessing persistence, logging, and behavioral/CUDA checks.
- [x] Complete step 4: housing selection/refit preprocessing, chronological datasets, schema/quality/provenance artifacts, exact baseline parity, and behavior tests.
- [x] Finish and validate X-ray preprocessing modules and artifacts in step 9 (Part B).
- [x] Complete A1's full Ray grid, baseline diagram/count, training/validation curves, clarified final refit, final test metrics, frozen settings, report assets/discussion, and checkpoint audit.
- [x] Complete A2's five-seed ablations with RMSE/R² means, sample standard deviations, separate ablation figures, discussion, and checkpoint audit.
- [x] Complete A3's lambda selection, five-seed comparison, figures, mechanism/limitations, exact baseline control, and checkpoint audit.
- [x] Complete A4's full-test ten-feature SGD attributions, ranked plots, top-three comparison, limitations, manual checks, and audit.
- [x] Complete B1's five learning rates, required table/two plots, and final test metrics.
- [x] Complete B2's five batch sizes, required table/two plots, and final test metrics.
- [x] Complete B3 and the full performance/parameter/FLOPs comparison.
- [x] Complete B4's five shared pneumonia examples and 15 Grad-CAM overlays.
- [x] Produce two concise reports, each beginning with the GenAI declaration and satisfying the page limit.
- [x] Verify independently runnable code packages and complete READMEs.
- [x] Inspect both five-page PDFs and both clean standalone ZIPs; retain integrity/extraction evidence.
- [ ] Replace the author placeholder consistently with the student's actual name.
- [ ] Submit the four named files through NTULearn; retain confirmation/receipt.
