# A2 results and discussion draft

This material is for the later Part A report. All results come from 15 completed saved runs.

## Controlled protocol

The A1-frozen hidden width 256, embedding dimension 4 where applicable, Adam learning rate 0.01, batch size 512, and **E*=4** were used unchanged. Each model was initialized fresh for each of seeds 42,43,44,45,46 and trained on the same 116,427 rows from 2017–2021. The exact A1 combined-data encoders/scalers were reused without fitting again. Model loss was standardized-target MSE, while all RMSE values below are in SGD.

Every run completed exactly 4 epochs, saved the final epoch, and received one final evaluation on the same 26,702 test rows from 2022. No validation/test monitoring loader was supplied during A2 training. The fresh seed-42 baseline was trained independently of A1's longer test-monitored run. The common epoch budget retains its origin in A1's professor-authorized test-assisted stopping.

All variants kept LayerNorm. The baseline and Sigmoid model use all ten inputs; only the hidden ReLU becomes Sigmoid, with a linear standardized-price output retained. Their initial parameter tensors are identical for every matched seed. Continuous-only has no embedding tables and a six-input first linear layer. Shuffling uses a separate, same-seed loader generator, so each matched seed sees the same training-row order across variants. Different continuous-only matrix shapes and parameter counts also change its random initialization; its tensors cannot be matched one-for-one to the all-input models.

## Mean and sample standard deviation

Each row summarizes five independent initialization/training seeds. Standard deviations use denominator 5−1 (`ddof=1`). These are seed-variability summaries, not confidence intervals or standard errors; each seed's R² is computed over the complete test partition before averaging.

| Model | Parameters | Test RMSE, SGD: mean ± sample SD | Test R²: mean ± sample SD |
| --- | ---: | ---: | ---: |
| Baseline (ReLU) | 7,065 | 102,427.90 ± 5,289.17 | 0.637311 ± 0.037428 |
| Continuous only | 2,561 | 108,252.68 ± 9,605.81 | 0.593204 ± 0.068794 |
| All inputs (Sigmoid) | 7,065 | 101,821.96 ± 4,082.21 | 0.641892 ± 0.028889 |

## Categorical-input ablation

Continuous only had mean test RMSE SGD 5,824.79 higher than baseline (5.69%). Its mean paired R² change was -0.044107; RMSE improved in 1/5 paired seeds. The sample SD of the paired RMSE difference was SGD 8,760.40.

This comparison measures the observed contribution of categorical inputs under this fixed training recipe. Removing them also reduces the parameter count from 7,065 to 2,561 (63.75% fewer), so information loss and capacity changes are confounded. Town, flat model type, month, and storey range can carry predictive associations beyond the six continuous variables, but this experiment neither isolates their individual importance nor identifies causal price effects. A continuous-only model retuned independently could behave differently; retuning was deliberately outside the controlled comparison.

The separate `figures/categorical_ablation` figure shows baseline versus continuous-only, with RMSE and R² panels, mean ± sample SD, and connected matched-seed observations. Both final comparison figures use identical zoomed y-axis limits: RMSE SGD 90,000–120,000 and R² 0.500–0.725. All observations and SD error bars remain visible; these figures display paired points and summary error bars.

## Hidden-activation ablation

All inputs (Sigmoid) had mean test RMSE SGD 605.94 lower than baseline (0.59%). Its mean paired R² change was +0.004582; RMSE improved in 3/5 paired seeds. The sample SD of the paired RMSE difference was SGD 2,024.47.

This comparison isolates the hidden activation while preserving all inputs, parameter count, initial weights, preprocessing, batch order, optimizer settings, and epoch budget. Sigmoid is bounded, with derivative σ(x)(1−σ(x)) ≤ 1/4; gradients approach zero at extreme inputs. Saturation can slow optimization, while ReLU retains a unit derivative for positive inputs. LayerNorm affects the preactivation distribution. Saturation and gradient statistics were not measured here, so they are plausible mechanisms rather than demonstrated causes of the observed result. Four epochs and hyperparameters selected for the ReLU model may also affect relative optimization speed; either activation could win under another recipe or seed.

The separate `figures/activation_ablation` figure shows baseline versus Sigmoid, covering both final test metrics. No test-metric epoch plot is produced.

## Interpretation limits and next step

Five matched seeds quantify variability, and paired differences provide descriptive direction. No significance claim is made from these five averages. Chronological property/price-distribution changes can also affect generalization. Test results describe the requested comparisons; they were not used to retune architectures, learning rate, or stopping epochs.

Use this A2 **five-seed baseline** for the later A3 comparison. A1's individual final result remains separately identified. Preserve H=256, d=4, learning rate 0.01, batch size 512, the same refit preprocessing, and fixed E*=4 for A3's controlled final refits.

## Saved evidence

- `tables/per_seed_results.csv`: all 15 results, parameter/epoch counts, initial-state fingerprints, loader-generator fingerprints, and checkpoint paths.
- `tables/summary_metrics.csv`: the required three-model mean/sample-SD table.
- `tables/paired_differences.csv`: same-seed ablation-minus-baseline differences.
- `figures/categorical_ablation.{png,pdf}` and `figures/activation_ablation.{png,pdf}`: separately presented final comparisons.
- `runs/<variant>/seed_<seed>/`: configuration, preprocessing, training history, final checkpoint, and single final-test record.
- `configuration.json`: frozen A1 source/settings/preprocessing provenance and this run's environment.

Activation definition: [PyTorch 2.11 Sigmoid documentation](https://docs.pytorch.org/docs/2.11/generated/torch.nn.Sigmoid.html). The derivative statement follows directly from that definition.
