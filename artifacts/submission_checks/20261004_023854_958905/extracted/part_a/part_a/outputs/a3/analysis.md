# A3 results and discussion draft

All results come from five completed validation-search runs and five fresh combined-data refits.

## Architecture and controlled protocol

One shared concatenation of four embeddings (dimension 4 each) and six continuous variables feeds both branches. The deep branch is the selected A1 Linear(22,256) → LayerNorm → ReLU → Linear(256,1) network; the wide branch is a single Linear(22,1). Their standardized outputs are added as `deep + lambda * wide`; the target mean and standard deviation are applied once afterward. Lambda is a fixed scalar, not a learned parameter.

The baseline has 7,065 parameters. The linear branch adds 23, giving 7,088 allocated trainable parameters (0.33% more); at lambda 0 the added branch is inactive and only 7,065 participate in optimization. Constructing the MLP before the wide branch preserves same-seed baseline initialization. Lambda 0 exactly reproduced A1's selected architecture's CUDA training/validation metrics and checkpoint epoch.

Search uses only 2017–2020 training and 2021 validation, seed 42, the original selection preprocessing, Adam learning rate 0.01, batch size 512, maximum 20 epochs, patience 4, and minimum monitored improvement 1e-4. All five candidates start from identical weights and use same-seed shuffling. Select the lowest best validation RMSE, breaking exact ties by smaller lambda. The detailed brief's grid {0, 0.25, 0.5, 1, 2} takes precedence over the slide typo.

## Validation lambda selection

| Lambda | Saved / completed epoch | Train RMSE (SGD) | Validation RMSE (SGD) | Validation R² |
| --- | ---: | ---: | ---: | ---: |
| 0 | 6 / 10 | 34,372.58 | 58,627.10 | 0.870061 |
| 0.25 | 6 / 10 | 34,371.57 | 59,279.07 | 0.867155 |
| 0.5 | 6 / 10 | 34,465.69 | 59,294.97 | 0.867084 |
| 1 | 6 / 10 | 34,460.04 | 58,315.02 | 0.871441 |
| 2 | 3 / 7 | 36,881.17 | 57,905.65 | 0.873239 |

Validation selected **lambda=2**, RMSE **SGD 57,905.65**, R² **0.873239**, at epoch **3**. Its validation RMSE change from lambda 0 was **SGD -721.46**. The selected train/validation RMSE gap was **SGD 21,024.48**; this chronological gap alone does not establish overfitting. `figures/lambda_search` shows both validation metrics, and `figures/selected_learning_curves` shows training/validation progression and saved epochs.

## Five-seed final test comparison

The selected configuration was initialized fresh for seeds 42–46, using the exact A1 refit preprocessing, 116,427 combined 2017–2021 rows, and fixed **E*=4** epochs. Each fourth-epoch model received one final evaluation on all 26,702 test rows from 2022. No validation/test monitoring loader was supplied during these final refits. The saved A2 five-seed baseline was reused without retraining or another test evaluation. Each seed's MLP initialization and training-row order matched its A2 baseline.

Values are mean ± **sample** SD (`ddof=1`) across five seeds, calculated from complete-partition metrics for each checkpoint.

| Model | Parameters | Test RMSE (SGD) | Test R² |
| --- | ---: | ---: | ---: |
| A2 baseline (ReLU) | 7,065 | 102,427.90 ± 5,289.17 | 0.637311 ± 0.037428 |
| Selected wide-and-deep | 7,088 | 101,413.74 ± 6,407.58 | 0.644081 ± 0.044905 |

The selected wide-and-deep model improved mean RMSE by **SGD 1,014.15 (0.99%)**. RMSE improved in **4/5** paired seeds. Paired wide-and-deep-minus-baseline changes were RMSE **-1,014.15 ± 2,811.40 SGD** and R² **+0.006770 ± 0.020248**. These standard deviations describe variation across seeds, rather than confidence intervals or statistical significance. `figures/baseline_comparison` presents both final metrics with matched-seed lines and mean/sample-SD markers.

A1's individual seed-42 final score remains **RMSE SGD 96,901.64, R² 0.676082**. That result used test-assisted stopping; it is separately identified rather than treated as a five-seed mean.

## Mechanism, evidence, and limitations

The linear branch can directly fit additive associations in the shared encoded features while the MLP models nonlinear interactions. It also creates a shorter gradient path to the shared embeddings. These changes can affect optimization, but neither gradient magnitudes nor branch-specific causal effects were measured. The observed validation/test directions provide evidence about this training recipe, rather than proof of a particular mechanism.

The test mean moves in the direction predicted by a beneficial wide contribution, with substantial seed variability still visible.

The selected checkpoint occurred at epoch 3, compared with epoch 6 for lambda 0. Its training RMSE was SGD +2,508.59 different while validation RMSE changed by SGD -721.46. These observations are consistent with a change in optimization/generalization under the fixed recipe. They do not establish reliably faster convergence from a single selection seed. The absolute mean paired test-RMSE change was SGD 1,014.15, compared with paired sample SD SGD 2,811.40; the average gain is modest relative to the observed variation.

For any positive lambda, the multiplier can be absorbed into the unconstrained linear weights and bias: `W_effective=lambda*W`, `b_effective=lambda*b`. All positive lambda candidates therefore have the same function class, although their initial effective branch scales and finite-time optimization differ. There is no weight decay in the existing Adam trainer, and the baseline has no dropout. This rescaling equivalence is also verified with matched predictions in a behavioral test. The zero-weight case removes the active linear contribution.

Limitations include the 23 additional parameters when the branch is active, shared learned embeddings rather than fixed raw features, lambda selection on one seed and one validation year, and a short fixed final budget selected for A1. That budget inherits A1's professor-authorized test-assisted stopping, so the 2022 comparison is not fully independent of prior test use. Chronological price/feature-distribution changes may affect generalization. Test outcomes were not used to change lambda, training settings, or stopping epochs; a validation advantage can fail to transfer to 2022. Five seeds quantify descriptive variability and support no significance claim by themselves.

## Saved evidence and A4 handoff

- `lambda_search.json`, `selection.json`, and `tables/lambda_search.csv`: all five candidates and the validation-only choice.
- `search/lambda_*/`: configurations, original-selection preprocessing, epoch histories, and validation-selected checkpoints.
- `refit/seed_*/`: five fresh configurations, exact refit preprocessing, training-only histories, fourth-epoch checkpoints, and single final-test records.
- `tables/per_seed_comparison.csv`, `tables/summary_metrics.csv`, and `tables/paired_differences.csv`: ten comparison records, required sample-SD summaries, and paired differences.
- `figures/`: architecture, lambda search, selected learning curves, and five-seed baseline comparison, each as PNG and vector PDF.
- `baseline_parity.json`, `configuration.json`, `summary.json`, and `frozen_a3_settings.json`: baseline control, source/settings provenance, completed results, and A4 settings.

For A4, pair `refit/seed_42/best_model.pt` with A1's seed-42 `refit/best_model.pt`, using the same saved refit preprocessing. Both checkpoints correspond to the four-epoch budget. This step does not execute A4 attributions.
