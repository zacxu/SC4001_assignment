# SC4001 — Part A

Author: YourName | October 11, 2026


<!-- PDF page 1 -->
## Regression: A1–A4

### Generative AI declaration

I used OpenAI Codex / ChatGPT for planning, code generation/debugging, experiment execution, analysis, figures, report drafting and submission packaging. Outputs were checked against the brief/slides, Q&A and library source, behavioral tests, CUDA checkpoint restoration and artifact/hash audits. FeatureAblation scores were independently aggregated and checked by manual replacements. All results below come from completed saved experiments. I remain responsible for reviewing and understanding the submitted work.

### Data and common training protocol

Chronological partitions: 2017–2020 train (87,370), 2021 validation (29,057), 2022 test (26,702); 2023 is excluded. Inputs are month, town, flat model type, storey range and six continuous variables; year, address and station name are excluded. Category encoders and standardization statistics are fitted on training only (sample SD for continuous inputs; population SD for price), then refitted on 2017–2021 for combined-data runs. Adam uses LR 0.01, batch 512, standardized-target MSE. Selection uses seed 42, at most 20 epochs, patience 4 and RMSE improvement 0.0001 SGD.

### A1. Baseline diagram, parameters and Ray grid search

![Figure A1a. Supplied baseline (H=64, d=8); the output before inverse scaling is linear.](assets/a1_baseline_diagram.png)

Figure A1a. Supplied baseline (H=64, d=8); the output before inverse scaling is linear.

P = d(13+27+44+18) + (4d+10)H + 1. Default: 816 embedding + 2,496 first-linear + 128 LayerNorm + 65 output-linear = 3,505 trainable parameters. Unknown category ID 0 is an ordinary learned embedding.

![Figure A1b. Minimum actual validation RMSE per fresh Ray Tune trial [1]; outlined cell is selected.](assets/a1_grid.png)

Figure A1b. Minimum actual validation RMSE per fresh Ray Tune trial [1]; outlined cell is selected.

All H in {16,32,64,128,256} × d in {2,4,8,12,16} combinations completed. Select H=256, d=4: 7,065 parameters, validation RMSE SGD 58,627.10, R² 0.870061, epoch 6. This is 5.61% lower RMSE than the default (SGD 62,109.01). Exact ties favor fewer parameters, smaller H, then smaller d.


<!-- PDF page 2 -->
## A1 · learning curves, final refit and interpretation

![Figure A1c–d. Selected trial: train/validation RMSE on the same axes. A1 refit: train/test RMSE on the same axes, as the brief requests. Dashed lines mark epochs 6 and 4; the test-stopping history is disclosed.](assets/a1_selected_curves.png)

![Figure A1c–d. Selected trial: train/validation RMSE on the same axes. A1 refit: train/test RMSE on the same axes, as the brief requests. Dashed lines mark epochs 6 and 4; the test-stopping history is disclosed.](assets/a1_refit_training.png)

Figure A1c–d. Selected trial: train/validation RMSE on the same axes. A1 refit: train/test RMSE on the same axes, as the brief requests. Dashed lines mark epochs 6 and 4; the test-stopping history is disclosed.

### Evidence for the selected configuration

Mean best validation RMSE across embedding sizes falls from SGD 65,100.41 at H=16 to 60,991.97 at H=256. The measured winner is H=256,d=4: 58,627.10 SGD, versus 63,213.60 for H=256,d=2. Increasing d is not uniformly beneficial at each width; d=4 also beats every larger d at H=256. This supports the chosen 7,065-parameter configuration within this grid, rather than a universal optimum.

The selected trial's epoch-6 training RMSE is SGD 34,372.58, versus validation 58,627.10. At epoch 10, training falls to 31,678.96 while validation rises to 70,715.09; every later validation checkpoint is worse than epoch 6. This is consistent with late overfitting and supports restoring epoch 6. The refit similarly improves training RMSE from 41,239.37 to 39,074.75 after epoch 4 while test RMSE worsens from 96,901.64 to 99,668.21 SGD.

At fixed d=4, H=16 has training RMSE 34,554.54 SGD, only 181.96 above H=256. These similar training errors do not establish substantial underfitting. High test error or model width relative to sample count cannot alone diagnose underfitting or rule out overfitting. Optimization fluctuations and temporal shift may contribute; one seed and one validation year do not establish ranking stability.

### Fresh combined-data refit and final test result

| H / d | Parameters | E_val / E* | 2022 test RMSE (SGD) | Test R² |
| --- | --- | --- | --- | --- |
| 256 / 4 | 7,065 | 6 / 4 | 96,901.64 | 0.676082 |

A fresh seed-42 model was trained on all 116,427 combined 2017–2021 rows using the saved refit preprocessing. As explicitly authorized by the professor, test RMSE monitored this refit's stopping under the same 20/4/0.0001 rule. Eight epochs completed; the restored best checkpoint is E*=4. E_val=6 belongs to the separate original-split selection run. Final scores above cover every test row.

### Test-assisted stopping limitation

The 2022 test set selected A1's stopping checkpoint, so its final score is not an untouched-holdout estimate. A2/A3 inherit the selected four-epoch budget. The higher test RMSE can reflect temporal distribution changes and model error; it is not proof of overfitting alone. Validation and test scores use different fitting periods and transforms, so their difference does not isolate the benefit of refitting. Q&A's no-test-epoch plotting answer addresses A2/A3; their test scores appear only in final tables. A1 retains its expressly requested train/test curve and separately authorized test-assisted stopping.

Sources: assignment brief pp. 2–4, slides pp. 2–3; supplied professor clarifications on plots and refit stopping.

[[1] Ray Tune 2.58.0 documentation: grid_search (Cartesian-product search).](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.grid_search.html)


<!-- PDF page 3 -->
## A2 · controlled ablations across five seeds

All three variants were initialized fresh for seeds 42–46 and trained for exactly four epochs on 2017–2021. They reuse A1's refit preprocessing, H=256, LR 0.01, batch 512 and the same seeded row order. All-input variants use d=4 and matched initial weights. Each run receives one final test evaluation; there is no A2 validation/test stopping. SD is sample SD (ddof=1), not a confidence interval. Separately, the FAQ epoch diagnostics use fresh seed-42 original-split runs for four fixed epochs: 2017–2020 training, 2021 validation and training-only transforms. They do not select settings or use test data.

| Model | Parameters | Test RMSE (SGD): mean ± SD | Test R²: mean ± SD |
| --- | --- | --- | --- |
| Baseline (ReLU) | 7,065 | 102,427.90 ± 5,289.17 | 0.637311 ± 0.037428 |
| Continuous only | 2,561 | 108,252.68 ± 9,605.81 | 0.593204 ± 0.068794 |
| All inputs (Sigmoid) | 7,065 | 101,821.96 ± 4,082.21 | 0.641892 ± 0.028889 |

### A2a. Remove all four categorical embeddings

![Figure A2a. Separate categorical ablation: train/validation RMSE and R² against epochs in original-split seed-42 diagnostics. Final test results are in the five-seed table.](assets/a2_categorical.png)

Figure A2a. Separate categorical ablation: train/validation RMSE and R² against epochs in original-split seed-42 diagnostics. Final test results are in the five-seed table.

Only six continuous inputs enter Linear(6,256) → LayerNorm → ReLU → Linear(256,1), giving 2,561 parameters. Mean test RMSE rises by 5,824.79 SGD (5.69%), worsening in four of five seeds; mean R² falls by 0.044107. The diagnostic also has higher train/validation error, consistent with reduced fit. Removal reduces capacity by 63.75%, confounding information loss with fewer parameters.

### A2b. Replace hidden ReLU with Sigmoid

![Figure A2b. Separate activation ablation: matched initialization, train/validation RMSE and R² against epochs. Neither diagnostic uses test metrics or changes the final four-epoch budget.](assets/a2_activation.png)

Figure A2b. Separate activation ablation: matched initialization, train/validation RMSE and R² against epochs. Neither diagnostic uses test metrics or changes the final four-epoch budget.

Sigmoid lowers mean RMSE by only 605.94 SGD (0.59%) and improves three of five seeds; paired RMSE difference SD is 2,024.47 SGD. Diagnostic validation RMSE rebounds from 60,460.29 to 68,030.75 SGD between epochs 3–4, showing instability under this budget. Sigmoid's bounded output and small derivatives can slow optimization relative to ReLU; saturation was not measured. The small, sign-changing final gain does not establish an advantage. The short budget was chosen for ReLU and inherited from test-assisted A1 stopping; no settings were retuned.


<!-- PDF page 4 -->
## A3 · wide-and-deep comparison

Both branches receive the same 22-vector of shared embeddings and continuous inputs: ẑ = MLP(x) + λ·Linear(22,1)(x); apply the target inverse transform once after the sum. The MLP is A1's Linear(22,256) → LayerNorm → ReLU → Linear(256,1). The linear branch adds 23 parameters, for 7,088 total. Keep all A1 settings fixed and select λ using original train/validation partitions, seed 42 and validation-RMSE stopping. Exact ties favor smaller λ.

| λ | Saved / completed epoch | Train RMSE (SGD) | Validation RMSE (SGD) | Validation R² |
| --- | --- | --- | --- | --- |
| 0 | 6 / 10 | 34,372.58 | 58,627.10 | 0.870061 |
| 0.25 | 6 / 10 | 34,371.57 | 59,279.07 | 0.867155 |
| 0.5 | 6 / 10 | 34,465.69 | 59,294.97 | 0.867084 |
| 1 | 6 / 10 | 34,460.04 | 58,315.02 | 0.871441 |
| 2 | 3 / 7 | 36,881.17 | 57,905.65 | 0.873239 |

Select λ=2: validation RMSE improves by 721.46 SGD (1.23%) over λ=0, with saved epoch 3. λ=0 reproduced A1's trained baseline tensors and scores exactly. For the controlled final comparison, fresh seeds 42–46 use the saved combined-data transforms and A1's four-epoch budget, matching baseline MLP initialization and row order; the five A2 baseline results are reused.

| Model | Parameters | Test RMSE (SGD): mean ± SD | Test R²: mean ± SD |
| --- | --- | --- | --- |
| Baseline (ReLU) | 7,065 | 102,427.90 ± 5,289.17 | 0.637311 ± 0.037428 |
| Wide + deep, λ=2 | 7,088 | 101,413.74 ± 6,407.58 | 0.644081 ± 0.044905 |

![Figure A3. Original-split, seed-42 baseline (λ=0) and selected wide-and-deep (λ=2): train/validation RMSE and R² against epochs. Five-seed final test scores appear only in the table.](assets/a3_learning_curves.png)

Figure A3. Original-split, seed-42 baseline (λ=0) and selected wide-and-deep (λ=2): train/validation RMSE and R² against epochs. Five-seed final test scores appear only in the table.

Mean RMSE improves 1,014.15 SGD (0.99%), in four of five seeds; paired difference SD is 2,811.40 SGD. Mean R² improves 0.006770. Seed 45 worsens by 3,246.47 SGD, so the gain is modest relative to seed variability and is not a statistical-significance claim.

A direct linear route can fit additive associations and shorten the gradient path to embeddings. The earlier/lower-error validation checkpoint is consistent with altered optimization or generalization, but branch-specific effects were not measured. Any positive λ can be absorbed into unrestricted linear weights/bias, so positive candidates share a function class but differ in effective initialization and finite-time optimization. Confounds include 23 extra parameters, jointly learned embeddings, one-seed λ selection and the short test-assisted epoch budget. The evidence does not isolate one mechanism.


<!-- PDF page 5 -->
## A4 · FeatureAblation in Singapore dollars

Captum FeatureAblation [2] explains the matching seed-42, epoch-4 A1/A3 checkpoints on all 26,702 test rows and all ten original inputs. For feature j, aᵢⱼ = f_SGD(xᵢ) − f_SGD(xᵢ with j replaced); importance is meanᵢ|aᵢⱼ|. Categorical IDs are replaced before embedding lookup, with ten distinct mask IDs. Shared combined-training (2017–2021) references are month 7, SENGKANG, flat type ‘4 ROOM, Model A’, storey ‘04 TO 06’, and each continuous variable's combined-training mean (standardized zero).

![Figure A4a. All ten baseline scores, ranked descending. Values are mean absolute prediction changes in SGD.](assets/a4_baseline.png)

Figure A4a. All ten baseline scores, ranked descending. Values are mean absolute prediction changes in SGD.

![Figure A4b. All ten wide-and-deep scores under the same replacement references and saved preprocessing.](assets/a4_wide_deep.png)

Figure A4b. All ten wide-and-deep scores under the same replacement references and saved preprocessing.

Baseline top three: floor area (71,537.72 SGD), remaining lease (46,187.61), distance to Dhoby Ghaut (41,163.95). Wide-and-deep: floor area (73,119.34), remaining lease (48,364.21), flat model type (42,280.28). Flat type and Dhoby distance swap third/fourth; eight other ranks match. The similar rankings suggest broadly similar reliance under these references, with jointly trained branch/embedding differences.

These are reference-dependent model sensitivities, not causal price effects. Replacing one correlated input can create unrealistic feature combinations; mean absolute scores hide signs and sample variability. Nonlinear effects need not add to the total prediction change, and larger scores do not imply better accuracy. Only one matched seed was explained. Independent global aggregation and manual replacements matched, with maximum manual difference 0 SGD and unchanged checkpoints.

[[2] Captum 0.9.0 documentation: FeatureAblation (replacement references and feature masks).](https://captum.ai/api/feature_ablation.html)

