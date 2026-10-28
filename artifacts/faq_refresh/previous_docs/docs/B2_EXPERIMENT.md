# B2: VGG-like batch-size selection — step 11

Completed **October 3, 2026** using the project `.venv` and local **NVIDIA GeForce RTX 3060 Ti / CUDA**. All five batch-size trials are complete. Validation selected **batch 16, checkpoint epoch 13**, using the **B1-frozen learning rate 0.001**. The selected model achieved **97.9008% validation accuracy**. Its single final test evaluation achieved **85.5769% accuracy, 99.2308% sensitivity, and 62.8205% specificity**.

Sources: [assignment brief pp. 5–6 / Figure B2](../assignment.pdf), [slides pp. 4–5](../assignment_slides.pdf), [professor clarifications](../clarifications.md), [recorded assumptions](ASSIGNMENT_DECISIONS.md), implemented [models](../part_b/src/models.py), and saved experiment artifacts. The required five-row table, two separate figures, justified choice, and final test metrics are included below. [B3 and the architecture/FLOPs comparison](B3_EXPERIMENT.md) and [B4 Grad-CAM](B4_EXPERIMENT.md) are now complete. The five-page final report is now complete; package verification remains step 15. See [report completion](REPORTS_COMPLETED.md).

## Controlled protocol

The prepared data retains all **4,708 training, 524 validation, and 624 test** images in their supplied partitions. Inputs are float32 `N×1×64×64`, scaled from uint8 by 255 once and normalized with saved training-pixel mean **0.5716994212632378** and population standard deviation **0.17409753982645407**. Normalization is reused without fitting any new statistics. See [Part B preparation](PART_B_PREPARATION.md).

Figure B2's VGG-like model stacks two 3×3 Conv/BN/ReLU stages at each width **8, 16, 32**, with a 2×2 max pool after each pair. An 8×8 average-pooling window reduces the final spatial grid to 1×1, followed by a biased FC1. All six convolutions have no bias; BN retains its affine scale/bias and running statistics. There is no dropout. The model has **18,185 trainable parameters**: **17,928 convolution weights + 224 BN affine parameters + 33 FC parameters**. The recorded output assumption uses an unrestricted binary logit, BCEWithLogitsLoss, and `sigmoid(logit) >= 0.5`.

| Setting | Executed value |
| --- | --- |
| Batch-size grid | **16, 32, 64, 128, 256**, following the brief over the slide's 258 |
| Learning rate | **0.001**, loaded from [B1's frozen selection](../part_b/outputs/b1/frozen_b1_settings.json) and verified against its checkpoint/table |
| Optimizer and seed | Adam with its existing defaults; **42** |
| Epoch limit | **20** per candidate |
| Early stopping | Validation BCE loss; patience **4**, minimum significant improvement **0.0001** |
| Checkpoint selection | Lowest validation classification error, then lower validation loss, then earlier epoch |
| Across-candidate ties | Same error/loss/epoch ordering, then smaller batch size |
| Data changes | No augmentation, weighting, resampling, or combined-data refit |
| Loader | Fresh per candidate; workers 0; keep final partial batches |
| Metric computation | Sample-weighted BCE; full training/validation metrics in evaluation mode after each completed epoch |

Each candidate starts from identical VGG weights and BN buffers. Training sees the same underlying shuffled row sequence, partitioned into different batch groups. Fingerprints record this control; changing batch size still changes the optimization trajectory and BN estimates. Each row below uses one selected checkpoint for every reported metric. Stopping and checkpoint selection are separate decisions.

## Required table and selection

The [required CSV](../part_b/outputs/b2/tables/batch_size_selection.csv) stores proportions; this table displays percentages.

| Batch size | Best epoch | Validation loss | Classification error | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| **16** | **13** | **0.069390** | **2.0992%** | **98.9717%** | **94.8148%** |
| 32 | 13 | 0.090230 | 2.8626% | 96.9152% | 97.7778% |
| 64 | 6 | 0.097501 | 2.8626% | 98.7147% | 92.5926% |
| 128 | 5 | 0.135308 | 4.7710% | 98.7147% | 85.1852% |
| 256 | 16 | 0.085114 | 2.8626% | 98.2005% | 94.0741% |

**Batch 16 wins uniquely by validation classification error:** 11 errors in 524 images, compared with 15 for batches 32/64/256 and 25 for batch 128. Its confusion counts are TP/TN/FP/FN **385/128/7/4**. The three runners-up tie on classification error; their losses rank batch 256 before 32 before 64. These tie rules do not change the winner. Batch 32 has higher specificity than batch 16, but the assignment's aggregate error criterion selects batch 16.

The frozen choice was written before accessing test. [frozen_b2_settings.json](../part_b/outputs/b2/frozen_b2_settings.json) records **LR 0.001, batch 16**, epoch 13, the selected checkpoint hash, and B1/source/preprocessing lineage. B3 must reuse these hyperparameters.

## Two separate curve figures

Each figure shows all five validation curves above their corresponding training curves. Stars mark selected checkpoints, and curves end at the actual completed epoch. The training panels evaluate the completed model with BN running statistics, consistently with validation. No test metrics are plotted against epochs.

![All five B2 loss curves](../part_b/outputs/b2/figures/validation_loss.png)

[Loss vector PDF](../part_b/outputs/b2/figures/validation_loss.pdf)

![All five B2 accuracy curves](../part_b/outputs/b2/figures/validation_accuracy.png)

[Accuracy vector PDF](../part_b/outputs/b2/figures/validation_accuracy.pdf)

**Batch 16** improves from validation loss 0.2002 to 0.0694, with temporary regressions at epochs 3 and 7. Its minimum validation loss is **0.066596 at epoch 9**, whereas its lowest classification error occurs at **epoch 13**. No sufficiently lower loss appears during epochs 10–13, so loss patience expires at 13 while that epoch still becomes the error-selected checkpoint. At selection, training/validation accuracies are **99.1079% / 97.9008%** and losses **0.025170 / 0.069390**. The 1.21 percentage-point accuracy gap shows stronger performance on training; it does not establish reliable generalization to the different test partition.

**Batch 32** initially improves but has a large epoch-6 loss spike to approximately **0.7611** and validation accuracy **73.6641%**, then recovers. Its minimum loss occurs at epoch 9, and its selected accuracy **97.1374%** occurs at epoch 13. It stops after four further epochs without significant loss improvement. The simultaneous training/validation spike indicates a change in the completed model's evaluation behavior, so attributing it solely to validation overfitting would be unsupported.

**Batch 64** reaches its best checkpoint at **epoch 6**, with loss **0.097501** and accuracy **97.1374%**. The following four epochs fail the loss-improvement rule; the last epoch's loss rises to **0.4954** and accuracy falls to **82.4427%**. Training evaluation also deteriorates. Restoring epoch 6 is necessary; the last epoch does not represent this candidate's selected result.

**Batch 128** begins at **74.2366%** validation accuracy, matching the positive-class majority rate. It reaches its lowest loss and best accuracy at **epoch 5**: loss **0.135308**, accuracy **95.2290%**. Later metrics fluctuate, and training stops at epoch 9 with loss **0.1486**, accuracy **94.0840%**. This candidate has the most validation errors at its best checkpoint, despite having a larger batch than three other candidates.

**Batch 256** remains at the **74.2366%** majority accuracy for the first two epochs, then improves over more epochs. Its selected checkpoint and minimum loss are at **epoch 16**, loss **0.085114**, accuracy **97.1374%**. Epoch 17 regresses before partial recovery. Four subsequent epochs fail significant loss improvement exactly when the **20-epoch limit** is reached; the saved status therefore records `stopped_early=False`, while the history also shows patience exhaustion at that limit. Its last-epoch accuracy **96.7557%** is below the selected checkpoint's accuracy.

## Interpreting batch size and the update budget

| Batch | Selected/completed epoch | Updates per epoch | Updates to selected checkpoint | Final training batch | Selected training accuracy | Selected validation accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 13/13 | 295 | 3,835 | 4 | 99.1079% | 97.9008% |
| 32 | 13/13 | 148 | 1,924 | 4 | 98.3857% | 97.1374% |
| 64 | 6/10 | 74 | 444 | 36 | 97.9609% | 97.1374% |
| 128 | 5/9 | 37 | 185 | 100 | 96.5378% | 95.2290% |
| 256 | 16/20 | 19 | 304 | 100 | 98.7681% | 97.1374% |

Updates per epoch are `ceil(4708 / batch_size)`. Batch 16 receives approximately **12.6 times as many optimizer updates** by its selected checkpoint as batch 256 receives by its selected checkpoint. Batch 256's slower initial progress is consistent with its smaller number of updates per epoch, but the experiment does not measure this as a causal explanation. The common epoch cap and learning rate define the required training recipes; they do not provide equal optimizer-update budgets.

Smaller batches use fewer images to estimate each gradient and the BN batch/spatial statistics. Their estimates can vary more with the composition of each group. Larger batches average over more images, but perform fewer weight updates and fewer BN running-statistic updates per epoch. Here all six BN layers use momentum 0.1; the retained final partial batches also update those statistics, including the four-image tails for batches 16 and 32. These mechanisms can contribute to fluctuations, but gradient variance and BN effects were not measured separately. Both training and validation evaluation curves fluctuate, including for large batches, so the data does not support a claim that increasing batch size guarantees smoother or better convergence. A controlled BN ablation or equal-update experiment would be needed to isolate the mechanisms; neither is part of this completed search.

This is one seed, with the learning rate selected on B1 and then prescribed for B2. Batch 16's four-error advantage over the tied runners-up supports the required selection on this validation set; it is not evidence of a universal batch-size ranking or a statistically established gain over repeated runs.

## One final test evaluation

The exact validation-selected checkpoint was restored and evaluated **once** on all **624 test images**, using batch 16 and the frozen probability threshold. [final_test.json](../part_b/outputs/b2/final_test.json) and the [CSV](../part_b/outputs/b2/tables/final_test.csv) retain the result. Logits, probabilities, labels, and original row IDs were captured during that same pass for independent auditing.

| Accuracy | Sensitivity | Specificity | Classification error | BCE loss |
| ---: | ---: | ---: | ---: | ---: |
| **85.5769%** | **99.2308%** | **62.8205%** | **14.4231%** | **0.510661** |

| Actual label | Predicted healthy | Predicted pneumonia |
| --- | ---: | ---: |
| Healthy, 234 images | TN **147** | FP **87** |
| Pneumonia, 390 images | FN **3** | TP **387** |

Test accuracy is **12.32 percentage points** below validation accuracy. Sensitivity remains high, but specificity decreases from **94.8148% to 62.8205%**: **87 of 234 negative-class images** are incorrectly classified as pneumonia, compared with only three missed positive-class images. The positive fraction changes from **74.24% in validation to 62.5% in test**; this affects aggregate accuracy, but prevalence alone cannot explain a decrease in specificity, which is conditioned on negative-class images. The model's low specificity is the main observed limitation of this final result. Neither the threshold nor selected hyperparameters were changed after seeing it.

The archive does not identify patients or encode semantic class names, so **0=healthy / 1=pneumonia remains an assumption**. Eight distinct pixel-identical images occur in both training and validation, potentially making validation optimistic; they occupy ten training and eight validation rows. Three extra copies occur within test. No exact cross-split train/test or validation/test pixel matches were found, but patient independence cannot be established. These are supplied-data limitations, and the provided partitions remain intact. The experiment does not establish the cause of the validation/test gap.

## Verification and reproducibility

The full suite passed **82 tests**, including six B2 tests covering selection/ties and invalid grids, B1-frozen controls and lineage, identical initialization/sample order with different batch groups, coherent selected metrics/update counts, and actual VGG training/restoration with one-pass prediction capture and repeat guards.

The [CUDA audit](../scripts/verify_b2.py) restored **all five trained checkpoints** and independently reproduced full training and validation metrics with **maximum difference 0.0** for every candidate. It checked every history's metric arithmetic, loss stopping and error selection, six BN update counters and Adam step counters, normalization/data hashes, B1 frozen settings, the required table, all 65 recorded epochs, and figure files. Final test metrics were recomputed from saved predictions with **zero additional test model evaluations**; stable float64 BCE differed by approximately **4.02×10⁻⁹**. Only the selected run has a final test record, and selection was frozen before it was written.

All supplied-file hashes and **833 previous output-file hashes** were unchanged. The [verification record](../part_b/outputs/b2/verification.json) is also copied to [artifacts/b2_checks/latest.json](../artifacts/b2_checks/latest.json). Both generated figures were visually inspected. Libraries remain in the existing isolated `.venv`; no additional package installation was needed for B2.

Run from the assignment root:

```powershell
# Check completed checkpoints and saved predictions without another test inference.
.\.venv\Scripts\python.exe -B scripts\verify_b2.py

# Reproduce the entire experiment in a new directory; existing results are protected.
.\.venv\Scripts\python.exe -B part_b\scripts\run_b2.py --output part_b\outputs\b2_reproduction

# Rebuild tables/figures/discussion only, with no training or test inference.
.\.venv\Scripts\python.exe -B part_b\scripts\run_b2.py --export-only
```

The runner defaults to the saved data directory and verified B1 experiment. `--data` and `--b1` can point to alternate matching artifacts; B1's frozen selection and audit are checked before training. The [implementation](../part_b/src/b2.py), [asset exporter](../part_b/src/b2_exports.py), [final-test capture helper](../part_b/src/evaluation.py), and [behavioral tests](../tests/test_b2.py) are saved with per-candidate histories/checkpoints under `part_b/outputs/b2/`. The [generated discussion draft](../part_b/outputs/b2/analysis.md) and CSVs support the eventual Part B report. [Step 12](B3_EXPERIMENT.md) reused **LR 0.001 and batch 16** for B3 and the saved B1/B2 final test records for comparison. [B4](B4_EXPERIMENT.md) has now used all three saved selected checkpoints for the fifteen Grad-CAM overlays.
