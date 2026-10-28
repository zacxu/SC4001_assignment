# B3: ResNet-like - current FAQ-corrected evidence

Updated October 11, 2026 against [assignment.pdf](../assignment.pdf), [assignment_slides.pdf](../assignment_slides.pdf), [clarifications.md](../clarifications.md), and [Q&A.md](../Q&A.md). Original selection runs are preserved; current final performance comes from a separate fresh combined-data refit.

## Validation selection and controls

B3 retains B1 LR 0.001 and B2 batch 16, selects epoch 9 by validation error/loss, and preserves the diagram's Conv/BN/ReLU projections and post-addition ReLU. No new hyperparameter search is performed.

Original supplied training/validation sizes are4,708/524. All selection runs use seed 42, Adam, maximum20epochs, validation-loss patience4, significant improvement1e-4; choose the lowest validation classification error, lower loss, then earlier epoch. Within grid ties use smaller LR/batch. Every row's metrics belong to one checkpoint. Train/validation learning curves are saved separately for loss and accuracy; no test metric epoch plots appear. Train metrics evaluate completed-epoch models with frozen BN and disabled dropout.

| Selected epoch | Best epoch | Validation loss | Error | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9 | 9 | 0.061988 | 2.4809% | 99.2288% | 92.5926% |

All convolution biases are disabled, FC biases enabled; SAME padding is `kernel_size // 2`, confirmed by the FAQ. B1 alone uses dropout0.5 after hidden FC64/FC32 ReLU; B2/B3 have no dropout. The instructor confirms B1/B2 raw FC1 logits directly into BCEWithLogitsLoss. B3 uses the same binary-loss interpretation, which that answer does not separately name. Labels0=healthy/1=pneumonia are instructor-confirmed. Trainable parameters:77,169.

## Fresh final training and one final test

A new seed 42 model and Adam optimizer train on the combined5,232 original training/validation rows for exactly **9 epochs**, LR **0.001**, batch **16**. Initialization does not load selection weights. No validation/test monitor is supplied; the prescribed final epoch is retained. Original training-only mean0.5716994212632378/SD0.17409753982645407 after uint8/255 normalization stays fixed, following the brief. The FAQ explicitly changes housing preprocessing but does not require refitting Part B pixel statistics.

The new checkpoint `refit/best_model.pt` receives one final test pass on624 images. Accuracy **81.0897%**, sensitivity **100.0000%**, specificity **49.5726%**, classification error **18.9103%**, BCE **0.823993**. TP/TN/FP/FN **390/116/118/0**. The fixed sigmoid>=0.5 threshold is unchanged. Captured logits/probabilities/labels/IDs enable metric recomputation without another test pass. Validation and final test belong to separately trained models; their difference is not a within-model generalization-gap estimate.

[Final record](../part_b/outputs/b3/final_test.json), [refit protocol](../part_b/outputs/b3/refit_protocol.json), [current verification](../part_b/outputs/b3/verification.json), [generated detailed interpretation](../part_b/outputs/b3/analysis.md), and [frozen settings](../part_b/outputs/b3/frozen_b3_settings.json) provide full-precision evidence. Frozen `checkpoint` identifies validation selection; `final_checkpoint` identifies the true final model for B4 and packaging. Historical no-refit scores and audits are archived separately and are not the current answer.

## Limits

Eight distinct images overlap supplied train/validation, making selection potentially optimistic. No exact training/test or validation/test matches exist, but absent patient IDs prevent patient-level independence checks; three extra test copies reduce distinctness. Each final architecture uses one seed; hyperparameters are inherited across architectures and batches/epoch/update budgets differ. Specificity, false positives and sensitivity must all inform interpretation. Test results never retune LR, batch, epochs or threshold.

## Current architecture comparison

| Model | Final epoch | Batch | Accuracy | Sensitivity | Specificity | Error | TP/TN/FP/FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B1 | 9 | 128 | 90.7051% | 97.6923% | 79.0598% | 9.2949% | 381/185/49/9 |
| B2 | 13 | 16 | 85.2564% | 100.0000% | 60.6838% | 14.7436% | 390/142/92/0 |
| B3 | 9 | 16 | 81.0897% | 100.0000% | 49.5726% | 18.9103% | 390/116/118/0 |

B1/B2/B3 parameter counts are99,569/18,185/77,169 and FLOPs/image39,063,616/19,464,256/25,325,696. Count two FLOPs per convolution/linear MAC plus28,672 B3 residual additions; exclude BN, ReLU, pooling, FC bias addition, preprocessing, sigmoid, transfers and training. Existing synthetic executed-shape complexity counts remain applicable to identically shaped refit architectures; they are operation estimates, not latency. B1 has the largest spatial FC head/count; B2 has the smallest global-average head/count; B3 adds identity/projection paths intended to aid optimization at additional convolution cost. In these final runs B1 has the best accuracy/specificity; B2/B3 detect all 390 positive rows at 92/118 false positives. One seed/differing recipes do not establish architecture causality. [Required complete table](../part_b/outputs/b3/tables/architecture_comparison.csv).
