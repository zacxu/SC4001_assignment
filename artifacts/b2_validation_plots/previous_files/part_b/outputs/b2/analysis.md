# B2 results and discussion draft

## Controlled protocol

Five fresh Figure B2 VGG-like networks (18,185 parameters) use the B1-frozen learning rate **1e-03**, seed 42, identical initial weights and underlying shuffled sample order, batch sizes 16/32/64/128/256, the supplied 4,708 training/524 validation images, and saved training-only normalization. Adam, maximum 20 epochs, patience 4 and significant loss improvement 1e-4 are fixed. Models have six bias-free Conv/BN/ReLU stages, three max pools, global averaging, and a biased raw-logit FC1; there is no dropout. The brief's batch 256 overrides the slide's 258 typo. No augmentation, weighting or resampling is used. After selection a fresh combined-data refit supplies the final model.

Stopping monitors validation BCE loss; checkpoint selection minimizes validation classification error, then loss, then earlier epoch. Across candidates, smaller batch size is a final exact-tie break. Metrics use sigmoid >=0.5 and the instructor-confirmed mapping 0=healthy, 1=pneumonia. The selected batch was frozen before accessing test.

## Required table and batch selection

Each row reports metrics from one restored selected checkpoint. Percentages are used here; CSV exports store proportions.

| Batch size | Best epoch | Validation loss | Classification error | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 13 | 0.069390 | 2.0992% | 98.9717% | 94.8148% |
| 32 | 13 | 0.090230 | 2.8626% | 96.9152% | 97.7778% |
| 64 | 6 | 0.097501 | 2.8626% | 98.7147% | 92.5926% |
| 128 | 5 | 0.135308 | 4.7710% | 98.7147% | 85.1852% |
| 256 | 16 | 0.085114 | 2.8626% | 98.2005% | 94.0741% |

**Select batch 16**, epoch **13**: validation accuracy **97.9008%**, error **2.0992%**, loss **0.069390**. There is one lowest-error candidate. The frozen B1 rate and B2 batch are the prescribed inputs to B3.

| Batch | Saved/completed epoch | Updates/epoch | Updates to saved checkpoint | Train accuracy | Validation accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 13/13 | 295 | 3835 | 99.1079% | 97.9008% |
| 32 | 13/13 | 148 | 1924 | 98.3857% | 97.1374% |
| 64 | 6/10 | 74 | 444 | 97.9609% | 97.1374% |
| 128 | 5/9 | 37 | 185 | 96.5378% | 95.2290% |
| 256 | 16/20 | 19 | 304 | 98.7681% | 97.1374% |

## Curves and batch-size interpretation

Separate `figures/validation_loss` and `figures/validation_accuracy` PNG/vector-PDF figures show five validation curves above five training curves. Stars mark saved checkpoints and actual stopping lengths are preserved. Training metrics use evaluation mode at each completed epoch. No test epoch curves are plotted.

- **Batch 16:** first/final validation loss 0.2002/0.0694; minimum loss 0.066596 at epoch 9; selected accuracy 97.9008% at epoch 13; 13 completed epochs (loss patience exhausted).
- **Batch 32:** first/final validation loss 0.1966/0.0902; minimum loss 0.073823 at epoch 9; selected accuracy 97.1374% at epoch 13; 13 completed epochs (loss patience exhausted).
- **Batch 64:** first/final validation loss 0.2221/0.4954; minimum loss 0.097501 at epoch 6; selected accuracy 97.1374% at epoch 6; 10 completed epochs (loss patience exhausted).
- **Batch 128:** first/final validation loss 0.6666/0.1486; minimum loss 0.135308 at epoch 5; selected accuracy 95.2290% at epoch 5; 9 completed epochs (loss patience exhausted).
- **Batch 256:** first/final validation loss 0.6213/0.0959; minimum loss 0.085114 at epoch 16; selected accuracy 97.1374% at epoch 16; 20 completed epochs (20-epoch cap reached).

The selected model has minimum validation loss at epoch **9**, compared with error-selected epoch **13**; loss stopping and classification-error selection are separate rules. Its train/validation accuracies are **99.1079%/97.9008%**, with losses **0.025170/0.069390**.

Batch size changes the gradient's sample group and BN's batch/spatial statistics. Smaller groups can give noisier gradient/statistic estimates, while larger groups average more samples; no separate gradient-variance or BN-ablation measurement was performed, so these mechanisms are interpretations rather than demonstrated causes of the ranking. It also changes optimizer updates per epoch: **295/148/74/37/19** for batches **16/32/64/128/256**. Final partial batches are retained: **4/4/36/100/100** images respectively. The running BN estimates are updated once per training batch, with the configured momentum; batch sizes therefore also change the number of running-statistic updates. Equal epoch limits and learning rate do not imply equal update budgets. This assignment compares those complete training recipes; it does not isolate batch size at a fixed number of optimizer steps. There is no rule that the smallest or largest batch must win.

## Final test, once

Following Q&A.md, a fresh model and Adam state were initialized with seed 42 and trained on the combined 5,232 training/validation images for exactly 13 epochs, using the chosen LR and batch. Training-only pixel normalization is retained consistently, as required by the brief. This new `refit/best_model.pt` checkpoint was evaluated once on all 624 test images at the chosen inference batch size. The same pass captured logits/probabilities/labels/row IDs in `final_test_predictions.pt` for auditing without further model test inference.

| Accuracy | Sensitivity | Specificity | Classification error | BCE loss |
| ---: | ---: | ---: | ---: | ---: |
| 85.2564% | 100.0000% | 60.6838% | 14.7436% | 0.540155 |

Confusion: TP=390, TN=142, FP=92, FN=0; test contains 390 positive and 234 negative labels. Validation's positive fraction is 74.24%, compared with 62.5% in test, so sensitivity/specificity complement prevalence-weighted accuracy. Test specificity is **60.6838%**, versus **94.8148%** at validation; conditional-class rates clarify which class accounts for errors. Neither the B1 learning rate nor B2 selection/threshold was revised using test results.

## Limitations and handoff

The supplied splits share eight distinct pixel-identical images between train and validation (10 training rows, 8 validation rows), so validation may be optimistic. Three extra copies occur within test; no exact train/test or validation/test pixel overlaps were found, but patient IDs are absent. Membership is preserved. Class names are instructor-confirmed by Q&A.md, while the NPZ lacks semantic metadata. B2 uses one seed and does not estimate seed variability. Its learning rate was selected on B1, not optimized independently for VGG. Exact causes of validation/test differences and curve instability were not established.

`frozen_b2_settings.json` records **LR 1e-03, batch 16**, selected epoch, separate validation/final-refit checkpoint hashes and B1/data/preprocessing lineage for B3/B4. Further architecture comparison reuses saved B1/B2 test records; B3/B4 use fresh final-refit checkpoints and the same fixed image draw.
