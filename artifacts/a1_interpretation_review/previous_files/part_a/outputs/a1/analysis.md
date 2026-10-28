# A1 results and discussion draft

This is material for the Part A report, not the complete final report. All values below come from saved run artifacts.

## Baseline architecture and parameter count

Four independent category embeddings concatenate with six continuous inputs. The MLP is `Linear(4d+6,H) → LayerNorm(H) → ReLU → Linear(H,1)`. Its output is a standardized price; inverse target scaling gives SGD predictions. Category table sizes including unknown ID 0 are 13, 27, 44, and 18.

`P = d(13+27+44+18) + (4d+10)H + 1` includes all embedding weights, linear weights/biases, and LayerNorm scale/bias. At supplied defaults H=64,d=8, P=3,505. The selected model has **7,065 parameters**. The architecture diagram describes the supplied default and labels each parameter contribution.

## Search and selection

Ray Tune completed all 25 combinations of H∈{16,32,64,128,256} and d∈{2,4,8,12,16} with fresh seed-42 initialization. Adam learning rate 0.01, batch size 512, standardized-target MSE, and LayerNorm stayed fixed. Maximum 20 epochs, patience 4, and minimum significant improvement 1e-4 used validation RMSE in SGD. Encoders/scalers were fitted only on 2017–2020; 2021 selected the checkpoint/configuration. No trial used test predictions or test scores.

The supplied-default trial achieved validation RMSE **SGD 62,109.01** at epoch 4. The grid selected **H=256, d=4**, with validation RMSE **SGD 58,627.10**, validation R² **0.870061**, and **E_val=6**. Its validation RMSE is 5.61% lower than the supplied default. Selection uses the lowest actual checkpoint RMSE, then fewer parameters, smaller H, and smaller d for exact ties.

| Rank | H | d | Parameters | Best epoch | Validation RMSE, SGD |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 256 | 4 | 7,065 | 6 | 58,627.10 |
| 2 | 128 | 12 | 8,649 | 4 | 59,688.99 |
| 3 | 256 | 16 | 20,577 | 6 | 60,835.40 |
| 4 | 64 | 16 | 6,369 | 4 | 60,859.85 |
| 5 | 128 | 16 | 11,105 | 4 | 60,972.93 |

The full table and heatmap retain every combination. The chosen width/dimension are supported by their joint measured score within this grid and seed, rather than a presumption that the largest model is best. Results do not establish a global optimum or stability across seeds.

Holding d=4 fixed, H=16 had training/validation RMSE SGD 34,554.54/64,672.61, compared with SGD 34,372.58/58,627.10 for the selected width. At H=256, the best alternative dimension was d=16 with validation RMSE SGD 60,835.40. Thus embedding size and hidden width interact; increasing embedding dimension does not uniformly improve validation error. The narrow model's training error is only SGD 181.96 higher. This gives weak evidence of relative underfitting within the budget; the validation improvement is much larger, so optimization and generalization differences also matter.

## Learning behavior

At E_val, training RMSE was **SGD 34,372.58** and the validation–training gap was **SGD 24,254.53**. The trial completed 10 epochs. Its last completed epoch had training RMSE SGD 31,678.96 and validation RMSE SGD 70,715.09. At the last epoch, training RMSE was lower but validation RMSE was higher than at the selected epoch. This is consistent with overfitting after the selected checkpoint, though one noisy trajectory cannot establish its cause.

Both curves evaluate the completed epoch's model in evaluation mode across the full partition, making their convention comparable. Chronological market changes between fitting and validation years may also contribute to a gap; these curves alone do not separate temporal shift from overfitting. Small configurations with higher errors on both partitions are compatible with limited model capacity under this training budget, but grid scores alone cannot prove underfitting without their learning curves.

## Final combined-data refit

A fresh seed-42 selected model trained on all 116,427 rows from 2017–2021, using the separately saved refit preprocessing. Following the professor's explicit answer, 2022 test RMSE monitored stopping under the same 20/4/1e-4 rule. The restored checkpoint is **E*=4**; training completed 8 epochs. E_val and E* refer to different runs and must not be interchanged.

| H | d | Parameters | E_val | E* | Final test RMSE, SGD | Final test R² |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 4 | 7,065 | 6 | 4 | 96,901.64 | 0.676082 |

The final metrics cover all 26,702 test rows and come from one restored checkpoint. A1's requested refit plot shows combined-training and test RMSE against all completed epochs on the same axes, using the saved `refit/internal_test_monitor.jsonl` values; no new inference is performed. The minimum test-monitor point is E*. The no-test-epoch Q&A answer explicitly addresses A2/A3; those sections plot train/validation diagnostics and present test scores only as final tables. A1's test-based stopping is separately and explicitly professor-authorized, and its holdout limitation remains disclosed.

**Limitation:** The 2022 test partition selects the A1 stopping checkpoint; its score is not an untouched holdout estimate. The A2/A3 epoch budget inherits this test-assisted selection. Validation and test scores also use different fitting periods and scaling statistics; their difference is not a controlled estimate of the benefit of retraining.

Observed mean prices in the saved data are SGD 442,495.91 for 2017–2020, SGD 511,417.39 for 2021, and SGD 549,701.07 for 2022. The refit training mean is SGD 459,696.83. These post-fit diagnostics support considering temporal distribution shift when interpreting the higher test RMSE. They are unadjusted transaction averages and also reflect property composition, so they do not identify a causal market effect. Year is deliberately excluded from the ten model inputs. No configuration, transform, or checkpoint was chosen using these diagnostics.

## Next-step frozen settings

A2/A3 refits inherit H=256, d=4, learning rate 0.01, batch size 512, and E*=4. Reuse the saved refit preprocessing unchanged. A2 trains fresh baseline/ablation models with seeds 42–46 for exactly E* epochs each, with one final test evaluation per run.

## Artifacts and implementation references

- `tables/grid_results.csv`: every trial's selected metrics, epoch counts, parameter count, runtime, and checkpoint location.
- `figures/selected_learning_curve.{png,pdf}`: required training/validation curves.
- `figures/refit_training_curve.{png,pdf}`: required A1 combined-training/test RMSE curves from saved epoch histories.
- `figures/validation_grid_heatmap.{png,pdf}`: full grid support for selection.
- `figures/baseline_architecture.{png,pdf}`: supplied-default diagram and count.
- `tables/final_test_metrics.csv`: coherent restored refit result.
- `tables/partition_price_summary.csv`: post-fit target-distribution diagnostics; no additional predictions or test-based tuning.
- `refit/best_model.pt`: final A1 model plus optimizer/task/model/preprocessing metadata.
- `frozen_a1_settings.json`: settings and E* for later comparisons.

Ray's [grid_search documentation](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.grid_search.html) describes the Cartesian product. Its [report documentation](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.report.html) describes metric/checkpoint registration. One final Ray report records the restored checkpoint; explicit epoch fields are used because report-call counts are not epoch counts.
