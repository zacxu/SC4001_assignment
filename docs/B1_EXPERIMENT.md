# B1: AlexNet-like - current FAQ-corrected evidence

Updated October 12, 2026 against [assignment.pdf](../assignment.pdf), [assignment_slides.pdf](../assignment_slides.pdf), [clarifications.md](../clarifications.md), and [Q&A.md](../Q&A.md), with validation-only B1 figure presentation. Original selection runs are preserved; current final performance comes from a separate fresh combined-data refit.

## Validation selection and controls

B1 selects LR 0.001 at batch 128 from all five required learning rates. The lowest learning rate still improves slowly at its 20-epoch cap; high rates have unstable trajectories. Loss stopping and classification-error selection yield different best epochs. Sensitivity/specificity reveal class-conditional errors hidden by prevalence-weighted accuracy.

Original supplied training/validation sizes are4,708/524. All selection runs use seed 42, Adam, maximum20epochs, validation-loss patience4, significant improvement1e-4; choose the lowest validation classification error, lower loss, then earlier epoch. Within grid ties use smaller LR/batch. Every row's metrics belong to one checkpoint. Separate figures show validation loss and validation accuracy against epochs for all five learning rates, with selected-checkpoint stars and each run's actual stopping length. The standalone PNG/PDF figures and concise report use this validation-only presentation. Saved histories still contain training metrics evaluated with frozen BN and disabled dropout; results, settings and model weights are unchanged.

| Learning rate | Best epoch | Validation loss | Error | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1e-05 | 18 | 0.283130 | 9.5420% | 97.1722% | 71.1111% |
| 0.0001 | 19 | 0.076439 | 3.0534% | 97.4293% | 95.5556% |
| 0.001 | 9 | 0.064911 | 2.0992% | 98.2005% | 97.0370% |
| 0.01 | 5 | 0.199377 | 5.9160% | 98.7147% | 80.7407% |
| 0.1 | 17 | 0.103817 | 4.5802% | 96.1440% | 93.3333% |

All convolution biases are disabled, FC biases enabled; SAME padding is `kernel_size // 2`, confirmed by the FAQ. B1 alone uses dropout0.5 after hidden FC64/FC32 ReLU; B2/B3 have no dropout. The instructor confirms B1/B2 raw FC1 logits directly into BCEWithLogitsLoss. B3 uses the same binary-loss interpretation, which that answer does not separately name. Labels0=healthy/1=pneumonia are instructor-confirmed. Trainable parameters:99,569.

## Fresh final training and one final test

A new seed 42 model and Adam optimizer train on the combined5,232 original training/validation rows for exactly **9 epochs**, LR **0.001**, batch **128**. Initialization does not load selection weights. No validation/test monitor is supplied; the prescribed final epoch is retained. Original training-only mean0.5716994212632378/SD0.17409753982645407 after uint8/255 normalization stays fixed, following the brief. The FAQ explicitly changes housing preprocessing but does not require refitting Part B pixel statistics.

The new checkpoint `refit/best_model.pt` receives one final test pass on624 images. Accuracy **90.7051%**, sensitivity **97.6923%**, specificity **79.0598%**, classification error **9.2949%**, BCE **0.355638**. TP/TN/FP/FN **381/185/49/9**. The fixed sigmoid>=0.5 threshold is unchanged. Captured logits/probabilities/labels/IDs enable metric recomputation without another test pass. Validation and final test belong to separately trained models; their difference is not a within-model generalization-gap estimate.

[Final record](../part_b/outputs/b1/final_test.json), [refit protocol](../part_b/outputs/b1/refit_protocol.json), [current verification](../part_b/outputs/b1/verification.json), [generated detailed interpretation](../part_b/outputs/b1/analysis.md), and [frozen settings](../part_b/outputs/b1/frozen_b1_settings.json) provide full-precision evidence. Frozen `checkpoint` identifies validation selection; `final_checkpoint` identifies the true final model for B4 and packaging. Historical no-refit scores and audits are archived separately and are not the current answer.

## Limits

Eight distinct images overlap supplied train/validation, making selection potentially optimistic. No exact training/test or validation/test matches exist, but absent patient IDs prevent patient-level independence checks; three extra test copies reduce distinctness. Each final architecture uses one seed; hyperparameters are inherited across architectures and batches/epoch/update budgets differ. Specificity, false positives and sensitivity must all inform interpretation. Test results never retune LR, batch, epochs or threshold.
