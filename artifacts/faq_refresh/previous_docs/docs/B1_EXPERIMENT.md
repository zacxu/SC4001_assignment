# B1 — AlexNet-like learning-rate selection

Completed October 3, 2026. Sources: [assignment brief pp. 5–6 / Figure B1](../assignment.pdf), [slides pp. 4–5](../assignment_slides.pdf), and [professor clarifications](../clarifications.md). **All five CUDA trials, validation-only selection, the single final test evaluation, required report assets, and independent audit are complete.**

## Protocol and architecture

Use the [verified Figure B1 model](PART_B_PREPARATION.md): three Conv/BN/ReLU stages with 16/32/64 channels, three stride-two max pools, average pooling to 4×4, flatten to 1,024, FC64/ReLU/dropout 0.5, FC32/ReLU/dropout 0.5, and one unrestricted output logit. All convolutions have no bias; FC layers have bias. There are **99,569 trainable parameters**. The previously recorded output and padding interpretations are unchanged.

Train on all **4,708** provided training images; use all **524** validation images for stopping and selection. Preserve the saved training-only normalization (mean 0.5716994212632378, population std 0.17409753982645407 after dividing by 255 once), supplied row membership/order, and the assumed class mapping 0=healthy, 1=pneumonia. No augmentation, resampling, loss weighting, or train/validation refit is used.

Each learning rate `{1e-5,1e-4,1e-3,1e-2,1e-1}` has a fresh model and loader with **seed 42**, identical initial model-state fingerprint, and identical first-epoch shuffle fingerprint. Controls: **Adam, batch 128, maximum 20 epochs, patience 4, minimum significant loss improvement 1e-4**. Every epoch retains all 4,708 samples: 36 full batches and one final batch of 100. BCEWithLogitsLoss trains raw logits; metrics threshold sigmoid probabilities at **0.5**.

Early stopping monitors validation loss. Within a run, checkpoint selection minimizes validation classification error, then loss, then earlier epoch. Across the five candidate checkpoints, use the same ordering and smaller learning rate as a final deterministic tie-break. **Every table row's loss, error, sensitivity, and specificity comes from one selected checkpoint**, even when its epoch differs from the minimum-loss epoch. The selected learning rate was frozen before accessing the final test loader.

## Required five-rate table

CSV values are proportions; this table displays error/sensitivity/specificity as percentages. "Best epoch" is the saved error-selected checkpoint.

| Learning rate | Best epoch | Validation loss | Classification error | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1e-5 | 18 | 0.283130 | 9.5420% | 97.1722% | 71.1111% |
| 1e-4 | 19 | 0.076439 | 3.0534% | 97.4293% | 95.5556% |
| **1e-3** | **9** | **0.064911** | **2.0992%** | **98.2005%** | **97.0370%** |
| 1e-2 | 5 | 0.199377 | 5.9160% | 98.7147% | 80.7407% |
| 1e-1 | 17 | 0.103817 | 4.5802% | 96.1440% | 93.3333% |

**Select `lr*=0.001`.** It has the lowest validation error: **11/524** misclassifications, compared with **16/524** for the next-best rate 0.0001. Its selected validation accuracy is **97.9008%**, sensitivity **98.2005%**, specificity **97.0370%**, and loss **0.064911**. It also has the lowest selected-checkpoint loss, so no exact error tie-break was needed. [frozen_b1_settings.json](../part_b/outputs/b1/frozen_b1_settings.json) records this choice for B2/B3 and the checkpoint for later B4.

| Rate | Saved epoch | Completed epochs | Selected train accuracy | Validation accuracy | Stop reason |
| --- | ---: | ---: | ---: | ---: | --- |
| 1e-5 | 18 | 20 | 91.5888% | 90.4580% | 20-epoch cap |
| 1e-4 | 19 | 20 | 97.8972% | 96.9466% | 20-epoch cap |
| 1e-3 | 9 | 12 | 98.5981% | 97.9008% | Loss patience exhausted |
| 1e-2 | 5 | 5 | 95.0085% | 94.0840% | Loss patience exhausted |
| 1e-1 | 17 | 20 | 95.7944% | 95.4198% | 20-epoch cap |

## Learning curves and interpretation

The two required figures are separate [loss](../part_b/outputs/b1/figures/validation_loss.png) and [accuracy](../part_b/outputs/b1/figures/validation_accuracy.png) plots, each available as PNG and vector PDF. Their upper panels compare all five validation curves; lower panels show all five training curves, following the professor's clarification. Training metrics evaluate the completed-epoch model with BN/dropout in evaluation mode; online training loss is logged separately. Stars mark saved checkpoints, and each curve ends at its actual stopping epoch. Accuracy axes begin at 70%, visibly labelled, so differences near 95–98% remain legible. No test epoch curves are produced.

- **1e-5:** loss declines gradually from **0.6346 to 0.2688** across 20 epochs. Best accuracy is only **90.46%**, and loss is still falling at the cap. This is consistent with learning too slowly for the permitted budget, rather than evidence that the architecture cannot fit the problem.
- **1e-4:** improves more gradually from validation loss **0.4338** to approximately **0.0785**, achieving **96.95%** best accuracy at epoch 19. It uses the full budget with occasional reversals and approaches, but does not match, the selected rate's error.
- **1e-3:** starts with a high validation loss **0.9189**, falls to **0.1015** at epoch 2, and reaches its minimum loss **0.062908 at epoch 8**. **Epoch 9** has slightly higher loss **0.064911** but fewer classification errors, so it is the saved checkpoint. Epochs 9–12 fail to significantly improve the epoch-8 loss, exhausting patience. The selected train/validation loss is **0.035620/0.064911**, and the accuracy gap is **0.6973 percentage points**. Later fluctuations justify restoring the selected checkpoint; they do not establish a unique cause or global convergence.
- **1e-2:** gives a strong first epoch but validation loss worsens from **0.1598 to 0.1994**, with training loss also worsening. It stops at epoch 5 after four unsuccessful loss improvements. Its best classification accuracy still occurs at epoch 5. The behavior indicates an unfavorable optimization trajectory at this rate; it is not a simple monotonic training-loss/validation-loss overfitting pattern.
- **1e-1:** both curves show repeated reversals. Best accuracy is **95.42%**, with selected loss **0.103817 at epoch 17**; training reaches epoch 20. It learns, but is less stable and gives worse selected error than 1e-3 and 1e-4.

These observations come from the saved histories and [curve diagnostics](../part_b/outputs/b1/curve_diagnostics.json). Early stopping is a practical finite-budget rule, not proof of convergence to a global optimum. BN and dropout are part of every run; their individual contributions to the fluctuations were not isolated.

## Why accuracy needs sensitivity and specificity

Training and validation contain approximately **74.2% pneumonia-labelled images**. An all-positive classifier would score **74.24% validation accuracy**, with **100% sensitivity** but **0% specificity**. Accuracy alone could therefore hide failure on the minority healthy class.

Sensitivity `TP/(TP+FN)` measures how many pneumonia-labelled images are detected. Specificity `TN/(TN+FP)` measures how many healthy-labelled images are correctly rejected as pneumonia. The 0.01 run illustrates the trade-off: sensitivity **98.71%** exceeds the selected model's **98.20%**, but specificity is only **80.74%**, versus **97.04%**. The selected model's validation confusion counts are **TP=382, TN=131, FP=4, FN=7**; reporting both rates exposes this balance.

## One final test evaluation

The exact validation-selected **epoch-9** model was restored and evaluated **once on all 624 test images**, with the same normalization and 0.5 threshold. There was no fresh refit or test-assisted selection.

| Accuracy | Sensitivity | Specificity | Classification error | BCE loss |
| ---: | ---: | ---: | ---: | ---: |
| **85.4167%** | **99.7436%** | **61.5385%** | 14.5833% | 0.748729 |

Confusion counts: **TP=389, TN=144, FP=90, FN=1**. There are 390 positive and 234 negative test labels. Only one positive is missed, but **90/234 healthy-labelled images are false positives**. Accuracy is **12.48 percentage points below validation**, chiefly accompanied by a specificity decrease from **97.04% to 61.54%**. The test positive fraction is **62.5%**, but prevalence change alone cannot explain a specificity decrease because specificity is measured within the negative class. Dataset differences and optimistic validation from supplied overlap are plausible explanations; this experiment does not establish their individual effects. The chosen rate and threshold were preserved after observing test results.

Logits, probabilities, labels, and original row IDs were captured in the **same single test forward pass** and saved in [final_test_predictions.pt](../part_b/outputs/b1/final_test_predictions.pt). Later audits recompute metrics from these saved values, without another model test evaluation. The candidate directories have exactly one `test_metrics.json`, belonging to the selected run.

## Verification, limitations, and files

**76 behavioral tests passed**, including six B1 checks for error-first selection, exact tie-breaking, invalid/incomplete grids, initialization and real loader control, selected-checkpoint metric coherence, and single-pass prediction capture with repeat-evaluation protection.

The [CUDA audit](../part_b/outputs/b1/verification.json) restored **all five** checkpoints and exactly reproduced their full train/validation metrics (**maximum differences 0.0**). It independently verified every epoch's metric arithmetic, patience history and checkpoint choice, model/normalization metadata, BN batch counts, optimizer learning rates, initial-weight/shuffle hashes, tables and figure files, and choice frozen before test access. It recomputed all final classification metrics from saved predictions and BCE from an independent stable formula. Supplied files and **782 previous output files** (Part A, prepared X-rays, and architecture checks) were unchanged. Both final figures were visually inspected.

The supplied archive has **eight distinct pixel-identical train/validation shared images**, occupying 10 training rows and 8 validation rows, so validation results may be optimistic. Source membership was preserved. Three extra test copies reduce distinct-image coverage. No exact train/test or validation/test pixel overlaps were found, but absent patient IDs prevent checking patient overlap. Class names remain a documented assumption. B1 is a single-seed experiment and does not estimate seed variability.

Saved under `part_b/outputs/b1/`:

- `runs/lr_*/`: five histories, selected checkpoints, normalization, configs, and coherent metrics.
- `learning_rate_search.json`, `tables/learning_rate_selection.csv`: complete results and the required six-column table.
- `figures/validation_loss.{png,pdf}`, `figures/validation_accuracy.{png,pdf}`: two separate curve figures.
- `final_test.json`, `tables/final_test.csv`, `final_test_predictions.pt`: one final test record and captured predictions.
- `frozen_b1_settings.json`: selected **LR 0.001**, checkpoint hash, input/preprocessing hashes, and B2/B3 handoff.
- `analysis.md`, `curve_diagnostics.json`, `verification.json`: discussion, numerical curve summaries, and audit evidence.

## Reproduce

From the assignment root, use the existing `.venv`:

```powershell
# Existing experiments are protected; choose a new destination to reproduce.
.\.venv\Scripts\python.exe -B part_b\scripts\run_b1.py --output part_b\outputs\b1_reproduction

# Reuse saved results to rebuild assets, with no training or test inference.
.\.venv\Scripts\python.exe -B part_b\scripts\run_b1.py --export-only
.\.venv\Scripts\python.exe -B scripts\verify_b1.py
```

The default experiment output is `part_b/outputs/b1`; `--data` selects another prepared-data location. All libraries remain in `.venv`; no new packages were needed. Full implementation: [b1.py](../part_b/src/b1.py), [exports](../part_b/src/b1_exports.py), [runner](../part_b/scripts/run_b1.py), [auditor](../scripts/verify_b1.py), and [behavior tests](../tests/test_b1.py).

B1's frozen learning rate **0.001** supplies B2 and B3. [Step 11's B2 experiment](B2_EXPERIMENT.md) selected **batch 16**, and [step 12's B3 comparison](B3_EXPERIMENT.md) is complete using both settings and the saved B1/B2 test records. [Step 13's B4 comparison](B4_EXPERIMENT.md) has now explained the three selected models on the same five randomly selected images. Both five-page final reports are complete; submission packaging remains step 15. See [report completion](REPORTS_COMPLETED.md).
