# B2: VGG-like - current FAQ-corrected evidence

Updated October 12, 2026 against [assignment.pdf](../assignment.pdf), [assignment_slides.pdf](../assignment_slides.pdf), [clarifications.md](../clarifications.md), and [Q&A.md](../Q&A.md). Original selection runs are preserved; current final performance comes from a separate fresh combined-data refit.

## Validation selection and controls

B2 retains B1 LR 0.001 and selects batch 16 by validation error then loss. The required five sizes are16/32/64/128/256, overriding slide258. The selected error epoch 13 differs from minimum-loss epoch 9. Smaller batches receive more Adam/BN updates per epoch, so equal epochs do not isolate batch size at equal updates.

Original supplied training/validation sizes are4,708/524. All selection runs use seed 42, Adam, maximum20epochs, validation-loss patience4, significant improvement1e-4; choose the lowest validation classification error, lower loss, then earlier epoch. Within grid ties use smaller LR/batch. Every row's metrics belong to one checkpoint. Separate loss and accuracy PNG/PDF figures now show validation curves only against epoch for all five batch sizes. Checkpoint stars and actual stopping lengths are retained in both standalone figures and report assets. Training metrics remain in the saved histories for numerical analysis; no test metric epoch plots appear.

| Batch size | Best epoch | Validation loss | Error | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 13 | 0.069390 | 2.0992% | 98.9717% | 94.8148% |
| 32 | 13 | 0.090230 | 2.8626% | 96.9152% | 97.7778% |
| 64 | 6 | 0.097501 | 2.8626% | 98.7147% | 92.5926% |
| 128 | 5 | 0.135308 | 4.7710% | 98.7147% | 85.1852% |
| 256 | 16 | 0.085114 | 2.8626% | 98.2005% | 94.0741% |

All convolution biases are disabled, FC biases enabled; SAME padding is `kernel_size // 2`, confirmed by the FAQ. B1 alone uses dropout0.5 after hidden FC64/FC32 ReLU; B2/B3 have no dropout. The instructor confirms B1/B2 raw FC1 logits directly into BCEWithLogitsLoss. B3 uses the same binary-loss interpretation, which that answer does not separately name. Labels0=healthy/1=pneumonia are instructor-confirmed. Trainable parameters:18,185.

## Fresh final training and one final test

A new seed 42 model and Adam optimizer train on the combined5,232 original training/validation rows for exactly **13 epochs**, LR **0.001**, batch **16**. Initialization does not load selection weights. No validation/test monitor is supplied; the prescribed final epoch is retained. Original training-only mean0.5716994212632378/SD0.17409753982645407 after uint8/255 normalization stays fixed, following the brief. The FAQ explicitly changes housing preprocessing but does not require refitting Part B pixel statistics.

The new checkpoint `refit/best_model.pt` receives one final test pass on624 images. Accuracy **85.2564%**, sensitivity **100.0000%**, specificity **60.6838%**, classification error **14.7436%**, BCE **0.540155**. TP/TN/FP/FN **390/142/92/0**. The fixed sigmoid>=0.5 threshold is unchanged. Captured logits/probabilities/labels/IDs enable metric recomputation without another test pass. Validation and final test belong to separately trained models; their difference is not a within-model generalization-gap estimate.

[Final record](../part_b/outputs/b2/final_test.json), [refit protocol](../part_b/outputs/b2/refit_protocol.json), [current verification](../part_b/outputs/b2/verification.json), [generated detailed interpretation](../part_b/outputs/b2/analysis.md), and [frozen settings](../part_b/outputs/b2/frozen_b2_settings.json) provide full-precision evidence. Frozen `checkpoint` identifies validation selection; `final_checkpoint` identifies the true final model for B4 and packaging. Historical no-refit scores and audits are archived separately and are not the current answer.

## Limits

Eight distinct images overlap supplied train/validation, making selection potentially optimistic. No exact training/test or validation/test matches exist, but absent patient IDs prevent patient-level independence checks; three extra test copies reduce distinctness. Each final architecture uses one seed; hyperparameters are inherited across architectures and batches/epoch/update budgets differ. Specificity, false positives and sensitivity must all inform interpretation. Test results never retune LR, batch, epochs or threshold.
