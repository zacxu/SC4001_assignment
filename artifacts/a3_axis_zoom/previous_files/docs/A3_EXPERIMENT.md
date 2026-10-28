# A3 wide-and-deep comparison — completed October 3, 2026

All **five lambda-search runs and five fresh final refits** completed on the local RTX 3060 Ti using `.venv`. The experiment loop took approximately **177 seconds**. Step 7 is complete; next is step 8, A4 FeatureAblation.

## Architecture and selection

The implementation follows Figure A in the brief. One shared 22-dimensional concatenation of four dimension-4 embeddings and six continuous inputs feeds the selected A1 MLP and a single `Linear(22,1)` branch:

```text
features = concatenate(shared embeddings, standardized continuous variables)
standardized_prediction = MLP(features) + lambda * Linear(features)
price_SGD = target_mean + target_std * standardized_prediction
```

The MLP remains `Linear(22,256) → LayerNorm → ReLU → Linear(256,1)`. Lambda is fixed, not learned. The inverse target transform is applied once after the sum. The linear branch adds **23 parameters** to the baseline's 7,065, giving **7,088**. At lambda 0 those extra parameters are allocated but inactive; only 7,065 receive gradients.

The detailed brief's grid **{0, 0.25, 0.5, 1, 2}** overrides the slide typo. Search used 87,370 training rows from 2017–2020 and 29,057 validation rows from 2021, the saved original-selection preprocessing, seed 42, Adam learning rate 0.01, batch size 512, maximum 20 epochs, patience 4, and minimum monitored improvement 1e-4. The minimum actual validation RMSE selected each checkpoint; exact lambda ties favor the smaller lambda. No test performance was evaluated in this search.

All candidates began with identical parameter tensors and the same seeded shuffle stream. The baseline MLP was constructed before the extra branch, preserving its original initialization. **Lambda 0 reproduced A1's selected baseline's training/validation scores, saved epoch, and trained baseline weights exactly on CUDA.**

| Lambda | Saved / completed epoch | Train RMSE (SGD) | Validation RMSE (SGD) | Validation R² |
| --- | ---: | ---: | ---: | ---: |
| 0 | 6 / 10 | 34,372.58 | 58,627.10 | 0.870061 |
| 0.25 | 6 / 10 | 34,371.57 | 59,279.07 | 0.867155 |
| 0.5 | 6 / 10 | 34,465.69 | 59,294.97 | 0.867084 |
| 1 | 6 / 10 | 34,460.04 | 58,315.02 | 0.871441 |
| **2** | **3 / 7** | **36,881.17** | **57,905.65** | **0.873239** |

Validation selected **lambda=2**, improving RMSE by **SGD 721.46 (1.23%)** from lambda 0. Its saved search epoch is 3; the controlled final refit budget remains A1's **E*=4**.

## Five-seed final comparison

The selected configuration was initialized fresh for seeds **42, 43, 44, 45, 46** and trained on 116,427 combined 2017–2021 rows for exactly four epochs. It reused A1's exact refit encoders/scalers and received one final evaluation per run on all 26,702 test rows from 2022. Final refits used no validation/test stopping loader. Each seed's initial MLP weights and shuffled training-row order matched its A2 baseline.

The saved A2 baseline results were reused as the matching five-seed comparator. They were not retrained or evaluated again. A1's individual seed-42 score is retained separately in the discussion.

| Model | Parameters | Test RMSE, SGD: mean ± sample SD | Test R²: mean ± sample SD |
| --- | ---: | ---: | ---: |
| A2 baseline (ReLU) | 7,065 | **102,427.90 ± 5,289.17** | **0.637311 ± 0.037428** |
| Selected wide-and-deep, λ=2 | 7,088 | **101,413.74 ± 6,407.58** | **0.644081 ± 0.044905** |

Standard deviations use `ddof=1`. The mean RMSE improvement is **SGD 1,014.15 (0.99%)**, with improvement in **four of five** matched seeds. Paired wide-and-deep-minus-baseline changes are **RMSE −1,014.15 ± 2,811.40 SGD** and **R² +0.006770 ± 0.020248**. The gain is modest relative to seed variation; seed 45 worsened by SGD 3,246.47. These are descriptive results rather than a statistical-significance claim.

## Mechanism and limitations

The linear branch can fit additive associations directly in the shared feature representation and provides a shorter gradient path to the embeddings. Its best validation checkpoint occurred earlier than lambda 0 and had lower validation error despite higher training error at the saved checkpoints. The validation and mean test improvements are consistent with a change in optimization/generalization, but gradients and branch-specific effects were not measured, so these observations do not establish a particular cause or reliably faster convergence.

Every positive lambda can be absorbed into the unconstrained linear weights and bias. Thus the positive candidates have the same function class, while their initial effective branch scales and finite-time optimization differ. This equivalence was checked with matched predictions. The shared Adam trainer uses no weight decay, and the baseline has no dropout.

Confounds include 23 extra active parameters, shared learned embeddings, lambda selection on one seed/year, chronological distribution changes, and the fixed short final budget. That budget inherits A1's professor-authorized **test-assisted stopping**, limiting independence of the final comparison. Test outcomes did not change lambda, training settings, or stopping epochs. See the [discussion draft](../part_a/outputs/a3/analysis.md) for the complete report-ready analysis.

## Saved deliverables

All artifacts are in [part_a/outputs/a3](../part_a/outputs/a3/):

- [analysis.md](../part_a/outputs/a3/analysis.md): protocol, lambda-search table, final comparison, mechanism, evidence, and limitations.
- [tables/lambda_search.csv](../part_a/outputs/a3/tables/lambda_search.csv): all five validation-selected candidate results.
- [tables/per_seed_comparison.csv](../part_a/outputs/a3/tables/per_seed_comparison.csv), [summary_metrics.csv](../part_a/outputs/a3/tables/summary_metrics.csv), and [paired_differences.csv](../part_a/outputs/a3/tables/paired_differences.csv): ten comparison records, mean/sample-SD table, and five paired differences.
- [figures/architecture.png](../part_a/outputs/a3/figures/architecture.png): shared-input architecture and single SGD inverse transform.
- [figures/lambda_search.png](../part_a/outputs/a3/figures/lambda_search.png): both validation metrics across lambda.
- [figures/selected_learning_curves.png](../part_a/outputs/a3/figures/selected_learning_curves.png): selected and lambda-0 training/validation RMSE/R² histories.
- [figures/baseline_comparison.png](../part_a/outputs/a3/figures/baseline_comparison.png): final test RMSE/R², matched-seed lines, and mean/sample-SD markers.
- Every figure also has a vector PDF version; all four figures were visually inspected.
- `search/lambda_*/` and `refit/seed_*/`: all ten configurations, preprocessing records, epoch histories, summaries, and checkpoints. Only the final refit directories contain test metrics.
- `selection.json`, `baseline_parity.json`, `summary.json`, `configuration.json`, `frozen_a3_settings.json`, and `seed_results/`: selection, exact baseline parity, completed results/provenance, A4 settings, and reusable seed summaries.
- [verification.json](../part_a/outputs/a3/verification.json) and [artifacts/a3_checks/latest.json](../artifacts/a3_checks/latest.json): successful production audit and timestamped evidence.

## Reproduction and verification

The model is in [wide_deep.py](../part_a/src/wide_deep.py), experiment in [a3.py](../part_a/src/a3.py), and exports in [a3_exports.py](../part_a/src/a3_exports.py). Existing outputs are protected. Run from the assignment root:

```powershell
.\.venv\Scripts\python.exe -B -u part_a\scripts\run_a3.py --output part_a\outputs\a3_reproduction
.\.venv\Scripts\python.exe -B part_a\scripts\run_a3.py --export-only
.\.venv\Scripts\python.exe -B -u scripts\verify_a3.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

The complete suite passed **51 tests**, including seven new A3 tests: zero-lambda initialization/prediction/training parity, shared branch inputs and gradients, single target inverse transform, positive-lambda rescaling equivalence, checkpoint restoration, validation selection/ties, and five-seed/budget guards.

The production audit restored **all ten checkpoints on CUDA**, reproduced all ten training metric sets plus all five validation metric sets, and verified validation-only selection, early stopping, final four-epoch records, exact preprocessing, matched initial MLP weights and shuffle state, sample statistics, report assets, and unchanged supplied files plus all A1/A2 checkpoints. It performed no additional test-performance evaluation.

PyMuPDF **1.28.2** was installed inside `.venv` to recheck the brief/slides and render Figure A. It is pinned in `requirements.in` and all three full requirements files; `pip check` passed. The PDF extracts/render are saved under `artifacts/a3_checks/requirements/`.

## A4 handoff

Use [A3's seed-42 checkpoint](../part_a/outputs/a3/refit/seed_42/best_model.pt) with [A1's seed-42 final checkpoint](../part_a/outputs/a1/refit/best_model.pt), and the exact [refit preprocessing](../part_a/outputs/a3/refit_preprocessing.pt). Their SGD predictions are available through `predict_price`. Construct the A3 architecture with `WideDeepHousingModel.from_metadata`, then use shared `restore_checkpoint` to restore weights and target statistics. A4 attributions have since been completed; see [A4_EXPERIMENT.md](A4_EXPERIMENT.md).

## October 10 FAQ epoch-plot inclusion

[Q&A.md](../Q&A.md) requires train/validation RMSE and R2 versus epochs, keeping test values in final tables/barplots. The existing [selected-learning-curves figure](../part_a/outputs/a3/figures/selected_learning_curves.png) supplies both metrics for the baseline control and selected lambda-2 model on the original chronological training/validation split. These are selection histories, distinct from the combined-data fixed-four-epoch final fits. Include them in the concise Part A report alongside the unchanged five-seed final table. No new A3 training or test pass was required.
