# B3: compact ResNet and architecture comparison — step 12

Completed **October 3, 2026**, in the existing isolated `.venv` on the local **RTX 3060 Ti / CUDA**. B3 used the frozen **LR 0.001 from B1** and **batch 16 from B2**. Validation selected **checkpoint epoch 9**, after 13 completed epochs. Its single final test evaluation achieved **82.8526% accuracy, 99.7436% sensitivity, and 54.7009% specificity**.

Sources: [assignment brief pp. 5–6 / Figure B3](../assignment.pdf), [slides pp. 4–5](../assignment_slides.pdf), [professor clarifications](../clarifications.md), [recorded assumptions](ASSIGNMENT_DECISIONS.md), verified [CNN implementation](../part_b/src/models.py), and saved B1/B2/B3 artifacts. This report includes B3's training curves, final metrics, the required three-model performance/parameter/FLOPs table, and architecture strengths and limitations. [B4 explanations](B4_EXPERIMENT.md) are now complete; the five-page final report is now complete; package verification remains step 15. See [report completion](REPORTS_COMPLETED.md).

## Executed architecture and training

B3 follows the compact Figure B3 network: a **3×3, 16-channel Conv/BN/ReLU stem**, 2×2 max pool, and three residual blocks with widths **16, 32, 64**. Each main branch contains two 3×3 Conv/BN/ReLU stages. The latter two blocks downsample with stride 2 in their first main convolution and use a **1×1 stride-2 Conv/BN/ReLU projection** on the shortcut. The first block has an identity shortcut. ReLU follows each residual sum. An 8×8 average pool produces a 1×1 spatial output, followed by one biased FC1. All **nine convolutions are bias-free**, all nine BN layers have affine scale/bias and running statistics, and there is no dropout. This follows the assignment's recorded ReLU placement, including the projection branches, rather than using a library ResNet preset.

Symmetric odd-kernel padding preserves stride-one spatial size and aligns both stride-two branches at centers 0,2,4,… . The main spatial sizes are **64×64 → 32×32 → 32×32 → 16×16 → 8×8 → 1×1**. The final output is an unrestricted logit, with BCEWithLogitsLoss and `sigmoid(logit) >= 0.5`, following the documented binary-output assumption.

| Setting | Executed value |
| --- | --- |
| Hyperparameters | **LR 0.001, batch 16**, verified against B1/B2 frozen settings, validation tables, and checkpoints |
| Training data | All **4,708** supplied training images |
| Validation data | All **524** supplied validation images |
| Normalization | Saved training-pixel mean **0.5716994212632378**, population std **0.17409753982645407**, after dividing uint8 by 255 once |
| Optimizer / seed | Adam / **42** |
| Epoch cap / loss stopping | **20** / patience **4**, significant improvement **0.0001** |
| Checkpoint rule | Lowest validation classification error; ties use lower loss, then earlier epoch |
| Data handling | Provided partitions retained; no augmentation, class weighting, resampling, or combined-data refit |
| Loader | Workers 0; retain the final four-image training batch |
| Updates | **295 per epoch**, **2,655 through selected epoch 9**, **3,835 through completed epoch 13** |

The source/data/preprocessing hashes match B1/B2. No new normalization statistics or hyperparameters were fitted. The [frozen B3 settings](../part_b/outputs/b3/frozen_b3_settings.json) were written before test access and record the exact epoch/checkpoint, inherited settings, class-name assumption, and final main-path convolution for B4.

## B3 curves and validation selection

![B3 training and validation BCE loss](../part_b/outputs/b3/figures/training_validation_loss.png)

[Loss vector PDF](../part_b/outputs/b3/figures/training_validation_loss.pdf)

![B3 training and validation accuracy](../part_b/outputs/b3/figures/training_validation_accuracy.png)

[Accuracy vector PDF](../part_b/outputs/b3/figures/training_validation_accuracy.pdf)

Training metrics evaluate the completed model over the full training partition in evaluation mode, so BN uses its stored running statistics, consistently with validation. Stars mark the selected checkpoint; no test metrics are plotted against epochs. Full histories and the [curves CSV](../part_b/outputs/b3/tables/learning_curves.csv) retain all 13 epochs.

Validation starts at **93.8931% accuracy / 0.162906 loss**, regresses at epoch 2, and improves through epoch 6. A temporary epoch-7 regression affects both training and validation evaluation, then the model recovers. At **epoch 9**, both minimum validation loss and lowest classification error are achieved: **0.061988 loss**, **97.5191% accuracy**, **2.4809% error**, **99.2288% sensitivity**, **92.5926% specificity**, with TP/TN/FP/FN **386/125/10/3**.

Epochs 10–13 do not improve validation loss by the required amount, so patience expires at 13. Epoch 13 ties epoch 9's **97.5191%** accuracy but has higher validation loss **0.072681**, so epoch 9 remains selected. Its classification-error/loss tie rule is therefore material: the final epoch is not the chosen model.

At the selected checkpoint, training accuracy is **99.4265%** and training loss is **0.021046**, compared with **97.5191% / 0.061988** on validation. By epoch 13, training loss falls further to **0.007252** and accuracy reaches **99.8726%**, while validation does not improve. The growing training/validation separation after epoch 9 is consistent with overfitting. Earlier simultaneous curve fluctuations also show that a single validation spike should not automatically be attributed solely to overfitting. This experiment does not isolate effects of BN statistics, batch composition, or the shortcut routes.

B3's selected validation loss is slightly lower than B1/B2's selected losses, yet its classification error is higher: **13 validation errors**, compared with **11** for each of B1/B2. Loss and thresholded accuracy measure different behavior; lower BCE alone does not imply fewer classification errors. B1/B2's checkpoints and hyperparameters remain frozen.

## Single final test result

The exact selected checkpoint was restored once and evaluated on all **624 test images**, at batch 16 with threshold 0.5. The same pass captured logits/probabilities/labels/row IDs in `final_test_predictions.pt`. The [JSON record](../part_b/outputs/b3/final_test.json) and [CSV](../part_b/outputs/b3/tables/final_test.csv) preserve the result.

| Accuracy | Sensitivity | Specificity | Classification error | BCE loss |
| ---: | ---: | ---: | ---: | ---: |
| **82.8526%** | **99.7436%** | **54.7009%** | **17.1474%** | **0.690535** |

| Actual label | Predicted healthy | Predicted pneumonia |
| --- | ---: | ---: |
| Healthy, 234 images | TN **128** | FP **106** |
| Pneumonia, 390 images | FN **1** | TP **389** |

Test accuracy is approximately **14.67 percentage points** below validation. The model detects almost every positive-class image, but misclassifies **106 of 234 negative-class images**. Specificity decreases from **92.5926% validation to 54.7009% test**. Class prevalence changes from **74.24% positive in validation to 62.5% in test**, influencing aggregate accuracy; it cannot alone explain the drop in conditional negative-class specificity. No threshold, checkpoint, or hyperparameter was changed in response to test performance.

## Required B1–B3 comparison

All performance columns are **final test results**, with percentages displayed here and proportions stored in the [comparison CSV](../part_b/outputs/b3/tables/architecture_comparison.csv). B1/B2's saved final test records were reused **without another model test inference**. Operation counts refer to one **float32 `1×1×64×64` image**, evaluation mode, within the explicitly defined profiler scope below.

| Model | Test accuracy | Sensitivity | Specificity | Classification error | Learnable parameters | FLOPs/image, profiled scope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 AlexNet-like | **85.4167%** | **99.7436%** | **61.5385%** | **14.5833%** | **99,569** | **39,063,616** |
| B2 VGG-like | **85.5769%** | **99.2308%** | **62.8205%** | **14.4231%** | **18,185** | **19,464,256** |
| B3 ResNet-like | **82.8526%** | **99.7436%** | **54.7009%** | **17.1474%** | **77,169** | **25,325,696** |

| Selected pipeline | LR | Batch | Checkpoint epoch | Optimizer updates through checkpoint | TP / TN / FP / FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 | 0.001 | 128 | 9 | 333 | 389 / 144 / 90 / 1 |
| B2 | 0.001 | 16 | 13 | 3,835 | 387 / 147 / 87 / 3 |
| B3 | 0.001 | 16 | 9 | 2,655 | 389 / 128 / 106 / 1 |

B2 has the highest observed accuracy and specificity, and the fewest parameters/estimated operations. Its accuracy advantage over B1 is only **one additional correct image out of 624**, so a strong claim of superior predictive performance is unsupported by this single-seed comparison. B1 and B3 tie at **389 detected positive cases and one missed case**. B3 makes **16 more false positives than B1**, fully accounting for its 16 fewer correct predictions and **2.56 percentage-point** accuracy decrease. Relative to B2, B3 detects two additional positives but makes 19 additional false positives, giving 17 fewer correct predictions. High sensitivity alone would conceal this weaker negative-class performance.

## Learnable-parameter counts

Counts include all trainable weights and biases, including BN affine scale/bias and FC biases. Running means/variances and batch counters are buffers, so they are excluded. Convolution biases are disabled by the assignment rule. The [breakdown CSV](../part_b/outputs/b3/tables/parameter_breakdown.csv) and per-model complexity JSONs record the calculations.

| Model | Convolution weights | BN affine parameters | FC weights + biases | Total |
| --- | ---: | ---: | ---: | ---: |
| B1 | 31,632 | 224 | 67,713 | **99,569** |
| B2 | 17,928 | 224 | 33 | **18,185** |
| B3 | 76,432 | 672 | 65 | **77,169** |

For a bias-free convolution, parameters are `C_out × (C_in/groups) × k_h × k_w`; affine BN contributes `2 × channels`; a biased linear layer contributes `out_features × in_features + out_features`. These formulas match the instantiated models and selected checkpoint metadata.

## FLOPs convention and independent checks

The installed **PyTorch 2.11 profiler**, `torch.profiler.profile(with_flops=True, record_shapes=True, acc_events=True)`, profiles a **CPU evaluation copy of each selected checkpoint** using one synthetic zero image. Shapes are fixed, so weights and image contents do not change the counted operations. This uses no dataset rows and leaves original checkpoint states/BN buffers unchanged. PyTorch documents formula-based FLOP estimates for convolution and matrix multiplication; the actual recorded events here additionally count B3's three residual `aten::add` operations. [PyTorch profiler documentation](https://docs.pytorch.org/docs/2.11/profiler.html).

One convolution/linear **multiply-add counts as two FLOPs**. Each residual elementwise addition counts as one FLOP. The actual positive-FLOP events are `aten::conv2d`, `aten::addmm`, and, for B3, `aten::add`. The [shape-checking implementation](../part_b/src/complexity.py) independently calculates:

- Conv MACs = `N × H_out × W_out × C_out × (C_in/groups) × k_h × k_w`.
- Linear MACs = `number_of_output_elements × in_features`.
- Residual additions = one per output element in each of the three block sums.
- Reported FLOPs = `2 × (Conv MACs + Linear MACs) + residual additions`.

| Model | Conv/linear MACs | Conv/linear FLOPs | Residual-addition FLOPs | Reported FLOPs |
| --- | ---: | ---: | ---: | ---: |
| B1 | 19,531,808 | 39,063,616 | 0 | **39,063,616** |
| B2 | 9,732,128 | 19,464,256 | 0 | **19,464,256** |
| B3 | 12,648,512 | 25,297,024 | **28,672** | **25,325,696** |

B3's additions are **16×32×32 + 32×16×16 + 64×8×8 = 28,672**. All three profiler totals exactly match the independent shape calculations. Layer-by-layer input/output shapes, channels, kernels, strides, parameter breakdowns, nonzero and zero-FLOP profiler operators, and residual additions are retained under [complexity/](../part_b/outputs/b3/complexity/) and in [layer CSV](../part_b/outputs/b3/tables/flops_layers.csv), [operator CSV](../part_b/outputs/b3/tables/flops_operator_counts.csv), and [residual CSV](../part_b/outputs/b3/tables/residual_addition_flops.csv).

**Excluded operations:** BN, ReLU, max/average pooling, FC bias addition, flatten/views, data normalization/transfers, sigmoid/threshold, and backward/optimizer work. Dropout is inactive during inference. Uncounted executed module/operator names are recorded, and their omission does not imply zero cost. The table therefore presents a consistent **profiled FLOPs scope**, rather than a complete scalar-operation inventory. These are formula-based operation counts, not FLOPs per second, measured latency, or memory usage; no throughput or device-speed ranking is claimed.

## Architecture strengths and limitations

**B1** uses only three convolutional stages, but its first two 5×5 kernels cover a wider local region per layer than a single 3×3 kernel. Its 4×4 average output retains spatial structure for the FC64/FC32 head, with hidden-layer dropout. That head provides flexible spatial feature combinations, but contributes **67,713 of 99,569 parameters**. B1 also has the highest profiled FLOPs here, driven mainly by its convolutional feature extraction rather than the FC head. It retains the same high test sensitivity as B3 and higher specificity in this experiment, while remaining slightly below B2 in observed specificity.

**B2** stacks six 3×3 convolution/BN/ReLU stages with narrower channels, combining multiple nonlinear transformations before global averaging. Its output head has only **33 parameters**, giving the lowest parameter and profiled operation counts. The tradeoff is spatial compression before classification and no residual shortcut routes. B2 has the strongest observed negative-class performance here, but its one-image accuracy advantage over B1 and one-seed scope do not establish a robust predictive advantage. Its selected small-batch training also receives many more optimizer/BN updates per epoch than B1.

**B3** has seven main-path convolutions plus two projection convolutions, and shortcut routes that carry identity or learned downsampled features into later block outputs. Residual learning is intended to ease the optimization of deeper networks. [He et al., *Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385). This run does not separately measure shortcut benefits. Wider channels and projections increase convolutional parameters relative to B2; global averaging keeps its head at **65 parameters**. Compared with B1, B3 uses approximately **22.50% fewer parameters and 35.17% fewer profiled FLOPs**, but its observed test specificity is lower. Training successfully fitting the data does not imply better test performance from the residual architecture.

The maximum theoretical local receptive-field sides at the final **8×8 grid before average pooling** are **28 pixels for B1, 36 for B2, and 48 for B3**. These follow `r_next = r + (kernel−1)×jump` and `jump_next = jump×stride`, taking the larger main/shortcut extent at residual sums. B2's stacked small kernels and B3's longer main path thus enlarge the theoretical local footprint. Subsequent averaging and B1's dense head aggregate across spatial locations. These geometric calculations concern possible dependencies, not effective learned attention or proof of lesion localization; [B4](B4_EXPERIMENT.md) inspects the selected models' coarse Grad-CAM maps separately.

The comparison concerns **the assignment's selected pipelines**: LR was selected on B1, batch size on B2; B1 uses batch 128, while B2/B3 use batch 16. Different architectures, checkpoint epochs, update budgets, and B1-only dropout prevent isolating architecture alone. It uses one seed per final model, so it does not estimate seed variability. The test table is a reporting comparison; it does not trigger any new model selection or retuning.

## Supplied-data limitations

The NPZ lacks semantic class-name metadata, so **0=healthy / 1=pneumonia is an assumption**. Eight distinct identical images are shared by training and validation, occupying ten training and eight validation rows, potentially making validation optimistic. Three extra copies occur within test. Exact pixel hashing found no train/test or validation/test cross-split matches, but patient IDs are absent, so patient independence cannot be established. The supplied partitions are retained. Neither duplicated validation images nor class prevalence alone is established as the cause of the test specificity gap.

## Verification, artifacts, and B4 handoff

All **88 tests passed**, including six new B3 tests covering inherited hyperparameters/data lineage, grouped convolution and linear MAC formulas, all three parameter/FLOPs counts and state preservation, explicit failure on unspecified profiling scope, saved-result comparison integrity, and actual ResNet training/checkpoint restoration with one-pass test capture and repeat guards.

The [independent CUDA audit](../scripts/verify_b3.py) restored B3 and reproduced its full training and validation metrics with **maximum difference 0.0**. It checked every epoch's metric arithmetic, loss stopping and error selection, all nine BN counters, Adam steps, seed/sample order, data/preprocessing metadata, and inherited frozen settings. Final metrics/BCE were recomputed from saved same-pass predictions with **no new test model inference**. It separately restored all three checkpoints on CPU and reproduced complexity records, checked shape formulas and parameter counts, and verified every comparison row against the original saved final test result. All supplied-file hashes and **884 previous output-file hashes** were unchanged. Both curve figures were visually inspected.

Evidence: [verification.json](../part_b/outputs/b3/verification.json), [artifacts/b3_checks/latest.json](../artifacts/b3_checks/latest.json), [generated discussion](../part_b/outputs/b3/analysis.md), and [comparison inputs/lineage](../part_b/outputs/b3/comparison_inputs.json). All packages remain in `.venv`; no additional profiling library was installed.

Run from the assignment root:

```powershell
# Audit checkpoints, captured predictions, and complexity without new test inference.
.\.venv\Scripts\python.exe -B scripts\verify_b3.py

# Reproduce to a new directory; the completed default output is protected.
.\.venv\Scripts\python.exe -B part_b\scripts\run_b3.py --output part_b\outputs\b3_reproduction

# Rebuild report assets only, with no training, profiling, or test inference.
.\.venv\Scripts\python.exe -B part_b\scripts\run_b3.py --export-only
```

The runner checks the data export and verified B1/B2 frozen settings before training. `--data`, `--b1`, and `--b2` can point to matching alternate artifacts. [B4](B4_EXPERIMENT.md) has now loaded the three saved selected checkpoints, using B3's final convolution **`features.block64.conv2.conv`** as recorded in the frozen settings. Step 13 sampled five shared positive-class images and produced fifteen independently verified Grad-CAM overlays without changing these B3 results.
