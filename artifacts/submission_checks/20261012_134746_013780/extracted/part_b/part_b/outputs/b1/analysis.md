# B1 results and discussion draft

## Controlled experiment

Five fresh Figure B1 AlexNet-like models were trained on the 4,708 supplied training images, with the 524 supplied validation images used for stopping and selection. All runs use the same seed 42, initial weights, first-epoch shuffle fingerprint, saved training-only pixel normalization, Adam, batch size 128, maximum 20 epochs, patience 4, and minimum loss improvement 1e-4. The model has 99,569 trainable parameters, BN/ReLU after all bias-free convolutions, FC bias, two hidden-FC dropout layers at 0.5, and an unrestricted final binary logit. Metrics use sigmoid probability >=0.5 with instructor-confirmed labels 0=healthy, 1=pneumonia. Q&A.md confirms hidden-FC dropout and the raw-logit output for B1/B2. No augmentation, class weighting or resampling is used.

Early stopping monitors validation BCE loss. Each saved checkpoint minimizes validation classification error, then loss, then earlier epoch. Across the five runs, the same ordering is used, with smaller learning rate as a final deterministic tie-break. Loss minimum and saved epoch can therefore differ. Hyperparameters were frozen before the single final test evaluation.

## Required learning-rate table

All metrics in a row belong to that run's one restored selected checkpoint. Error, sensitivity, and specificity are displayed as percentages; CSV exports contain proportions.

| Learning rate | Best epoch | Validation loss | Classification error | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1e-05 | 18 | 0.283130 | 9.5420% | 97.1722% | 71.1111% |
| 1e-04 | 19 | 0.076439 | 3.0534% | 97.4293% | 95.5556% |
| 1e-03 | 9 | 0.064911 | 2.0992% | 98.2005% | 97.0370% |
| 1e-02 | 5 | 0.199377 | 5.9160% | 98.7147% | 80.7407% |
| 1e-01 | 17 | 0.103817 | 4.5802% | 96.1440% | 93.3333% |

**Selected learning rate: 1e-03**, epoch **9**, validation classification error **2.0992%**, accuracy **97.9008%**, loss **0.064911**. This is the validation-only selection used for B2 and B3.

| Learning rate | Saved epoch | Completed epochs | Selected train accuracy | Selected validation accuracy | Early stopped |
| --- | ---: | ---: | ---: | ---: | --- |
| 1e-05 | 18 | 20 | 91.5888% | 90.4580% | False |
| 1e-04 | 19 | 20 | 97.8972% | 96.9466% | False |
| 1e-03 | 9 | 12 | 98.5981% | 97.9008% | True |
| 1e-02 | 5 | 5 | 95.0085% | 94.0840% | True |
| 1e-01 | 17 | 20 | 95.7944% | 95.4198% | False |

## Learning curves

`figures/validation_loss` and `figures/validation_accuracy` are two separate PNG/vector-PDF figures showing only validation metrics against epochs for all five learning rates. Stars identify the error-selected checkpoint; lines preserve actual stopping lengths. Validation metrics evaluate each completed-epoch model with dropout disabled and BN statistics frozen. Training and online-loss values remain available in the saved histories for numerical analysis.

The lowest rate, 1e-5, improved slowly: validation loss moved from 0.6346 to 0.2688 over 20 epochs, with best accuracy 90.46%. Its loss was still falling at the cap, consistent with slow learning within the allowed budget. Rate 1e-4 made more gradual, substantial progress, moving from loss 0.4338 to 0.0785, with best accuracy 96.95%; it also reached the cap.

Rate 1e-3 started at loss 0.9189 and rapidly improved thereafter. Its minimum loss was 0.062908 at epoch 8, while its error-selected checkpoint was epoch 9. It stopped at epoch 12 after four epochs without a significant loss improvement. This illustrates why minimum loss and the reported best epoch differ.

Rate 1e-2 began with validation loss 0.1598, which ended at 0.1994. Training loss also deteriorated over this run; loss did not improve enough to reset patience, so it ended at epoch 5. Nevertheless its accuracy-selected checkpoint was epoch 5, with accuracy 94.08%. This pattern indicates an unfavorable optimization trajectory at that rate; it does not by itself establish overfitting. Rate 1e-1 was more erratic, with repeated loss/accuracy reversals; its saved accuracy was 95.42% at epoch 17, below the selected configuration.

The selected checkpoint has training accuracy 98.5981% versus validation accuracy 97.9008%, a gap of 0.6974 percentage points, and train/validation losses 0.035620/0.064911. The later loss fluctuations do not identify a unique cause. Early stopping ends this finite schedule; it is not proof of global convergence. All raw values and minima are in `curve_diagnostics.json`.

## Why sensitivity and specificity matter

The training and validation partitions contain 74.21% and 74.24% pneumonia images. An all-positive classifier would achieve 74.24% validation accuracy while having sensitivity 100% and specificity 0%. Sensitivity is TP/(TP+FN), the fraction of pneumonia-labelled images detected; specificity is TN/(TN+FP), the fraction of healthy-labelled images rejected as pneumonia. They reveal false negatives and false positives that a single prevalence-weighted accuracy hides.

## One final test evaluation

Following Q&A.md, a new model and Adam state were initialized with seed 42 and trained on all 5,232 combined train/validation images for exactly **9 epochs**, batch 128, selected LR. The original training-only pixel normalization is retained consistently, as required by the brief. The fresh final checkpoint `refit/best_model.pt` was evaluated once on all **624** test images. No test metrics determined settings or stopping. Test results are a final table, with no test epoch curves.

| Accuracy | Sensitivity | Specificity | Classification error | BCE loss |
| ---: | ---: | ---: | ---: | ---: |
| 90.7051% | 97.6923% | 79.0598% | 9.2949% | 0.355638 |

Confusion counts: TP=381, TN=185, FP=49, FN=9. Test label counts are 390 positive and 234 negative; its positive fraction is 62.5%, different from training/validation. Captured logits, probabilities, labels, and original test-row IDs are saved in `final_test_predictions.pt` from the same forward pass, so audits can recompute metrics without running test inference again.

Test accuracy is 7.20 percentage points below validation. Sensitivity remains high, but specificity falls from 97.04% to 79.06%: 49 of 234 healthy-labelled images are false positives, compared with only 9 false negative among 390 pneumonia-labelled images. Specificity is computed within the negative class, so prevalence change alone cannot explain that drop. Distribution differences or optimistic validation caused by supplied overlap are possible explanations; the exact cause was not established. The learning rate and 0.5 threshold remain the choices frozen before test access.

## Limitations and handoff

The supplied data contain eight distinct pixel-identical images shared by training and validation (10 training rows, 8 validation rows), so validation results may be optimistic. The provided partitions and repeated images were preserved. There are three extra pixel-identical copies within test and no exact train/test or validation/test pixel matches, but patient IDs are absent and patient overlap cannot be excluded. The instructor confirms class names in Q&A.md; the archive itself lacks semantic metadata.

This is one seed's learning-rate comparison, as requested for B1; it does not establish variability over seeds. Final output/padding/pooling interpretations are recorded in the preparation report. `frozen_b1_settings.json` holds the selected learning rate, separate validation-selection and fresh final-refit checkpoint hashes, preprocessing/data hashes, threshold, and fixed training rules for B2/B3/B4. Future steps reuse the saved B1 test record rather than reevaluating it to choose a better result.
