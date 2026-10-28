# A2 controlled ablations — completed October 3, 2026

All **15 fresh CUDA runs** are complete: three model variants × seeds **42, 43, 44, 45, 46**. The training/evaluation loop took approximately 168 seconds on the local RTX 3060 Ti using the root `.venv` interpreter. The required final five-seed experiments retain their original results. October 10 adds the FAQ-required original-split learning-curve diagnostics.

## Fixed comparison protocol

Every run used the same 116,427 combined training rows from 2017–2021, the exact saved A1 refit encoders/scalers, standardized-target MSE, Adam learning rate **0.01**, batch size **512**, and hidden width **256**. All-input models kept embedding dimension **4**. No preprocessing was fitted again and 2023 remained excluded.

Each model was initialized fresh and trained for exactly **E*=4 epochs**. Its fourth-epoch checkpoint received one final evaluation on the 26,702 test rows from 2022. No validation/test monitoring loader was used during A2 fitting, and no early stopping selected an A2 checkpoint. The fresh seed-42 baseline reproduces A1's final metrics, but was trained separately and stopped at the fixed budget.

Matched seeds use separate identically seeded data-loader generators, ensuring the same shuffled row order across the three variants. Baseline and Sigmoid initial parameter tensors match exactly for each seed. The continuous-only architecture has different tensor shapes and fewer parameters, so its initialization cannot match those tensors one-for-one.

| Variant | Architecture after feature preparation | Trainable parameters |
| --- | --- | ---: |
| Baseline | Four embeddings + six continuous inputs → Linear(22,256) → LayerNorm → ReLU → Linear(256,1) | 7,065 |
| Continuous only | Six continuous inputs → Linear(6,256) → LayerNorm → ReLU → Linear(256,1); no embedding tables | 2,561 |
| Sigmoid | Baseline architecture with only the hidden ReLU replaced by Sigmoid | 7,065 |

The output remains linear in standardized-price space for every variant; predictions are inverse-transformed once for SGD metrics.

## Final test results

Each value is the mean ± **sample standard deviation** across five seeds (`ddof=1`). R² is computed separately over the complete test partition for each run before averaging. The standard deviations describe seed variability, rather than confidence intervals.

| Model | Test RMSE, SGD | Test R² |
| --- | ---: | ---: |
| Baseline (ReLU) | **102,427.90 ± 5,289.17** | **0.637311 ± 0.037428** |
| Continuous only | **108,252.68 ± 9,605.81** | **0.593204 ± 0.068794** |
| All inputs (Sigmoid) | **101,821.96 ± 4,082.21** | **0.641892 ± 0.028889** |

Removing categorical inputs increased mean RMSE by **SGD 5,824.79 (5.69%)**, and worsened it in four of five matched seeds. The mean paired R² change was −0.044107. The paired RMSE differences have sample SD SGD 8,760.40. This supports retaining categorical information under the fixed recipe, while removal also reduces capacity by 63.75%, confounding information and parameter-count effects.

Sigmoid reduced mean RMSE by **SGD 605.94 (0.59%)**, and improved it in three of five matched seeds. The mean paired R² change was +0.004582. Its small mean advantage is accompanied by paired RMSE sample SD SGD 2,024.47 and changes sign across seeds. These results do not establish a reliable activation advantage. The [discussion draft](../part_a/outputs/a2/analysis.md) explains saturation as a possible optimization mechanism; gradients and saturation were not measured.

The fixed budget was originally chosen through A1's professor-authorized **test-assisted stopping**. A2 does not perform its own test-based selection, but inherits this budget's influence. Four epochs and settings selected for the ReLU model also limit generalization of the activation comparison. No architecture, learning rate, or epoch budget was retuned using these results, and no statistical-significance claim is made.

## Saved deliverables

Everything is under [part_a/outputs/a2](../part_a/outputs/a2/):

- [analysis.md](../part_a/outputs/a2/analysis.md): report-ready protocol, required results table, separate ablation discussions, mechanisms, and limitations.
- [tables/per_seed_results.csv](../part_a/outputs/a2/tables/per_seed_results.csv): all 15 results with counts, epochs, initial-state/loader fingerprints, and checkpoint paths.
- [tables/summary_metrics.csv](../part_a/outputs/a2/tables/summary_metrics.csv): full-precision means and sample standard deviations.
- [tables/paired_differences.csv](../part_a/outputs/a2/tables/paired_differences.csv): ten same-seed ablation-minus-baseline comparisons.
- [figures/categorical_ablation.png](../part_a/outputs/a2/figures/categorical_ablation.png) and [figures/activation_ablation.png](../part_a/outputs/a2/figures/activation_ablation.png): separate two-panel RMSE/R² figures, each with seed points, paired lines, and mean ± sample-SD markers; vector PDF versions are also saved.
- `runs/<variant>/seed_<seed>/`: configuration, preprocessing, four-epoch training history, final checkpoint, single final-test metrics, and run summary.
- `seed_results/<variant>/`: reusable five-seed CSV/JSON summaries; use `baseline` for the A3 comparison.
- `configuration.json`, `preprocessing.pt`, `summary.json`, and `progress.json`: frozen A1 provenance, environment, completed-run statistics, and completion status.
- [verification.json](../part_a/outputs/a2/verification.json): completed production audit.

Both figures were visually inspected. They follow the professor's instruction to present the two ablations separately; test scores appear only as final-result comparisons.

## Implementation and reproduction

The models are in [ablations.py](../part_a/src/ablations.py), the experiment in [a2.py](../part_a/src/a2.py), and saved-result exports in [a2_exports.py](../part_a/src/a2_exports.py). The experiment uses the existing shared fixed-refit trainer and final-evaluation guard. `model_from_a2_metadata` constructs the matching architecture for `restore_checkpoint`, which restores target statistics as well as weights.

Run from the assignment root. Existing run directories are protected; reproduction writes a new directory.

```powershell
.\.venv\Scripts\python.exe -B -u part_a\scripts\run_a2.py --output part_a\outputs\a2_reproduction

# Refresh tables, figures, and discussion from saved results.
.\.venv\Scripts\python.exe -B part_a\scripts\run_a2.py --export-only

# Audit existing artifacts; reproduce training metrics only.
.\.venv\Scripts\python.exe -B scripts\verify_a2.py

.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

All assignment packages remain inside `.venv`; no additional dependencies were needed for A2.

## Verification

The complete suite passed **44 tests**, including six A2 tests covering matched activation initialization, removal of embeddings, categorical-input invariance, fixed-epoch checkpoint restoration, test-access guards, frozen settings, and five-seed/paired statistics.

The production audit restored **all 15 checkpoints on CUDA** and reproduced their full combined-training metrics. It checked four completed epochs, architecture/parameter counts, exact saved preprocessing, same-seed initial tensors and loader-generator states, single final-test record coherence, independent sample-SD calculations, separate figures, and unchanged supplied files plus all A1 checkpoints. The audit did **not** perform another test-performance evaluation.

Evidence: [verification.json](../part_a/outputs/a2/verification.json) and [artifacts/a2_checks/latest.json](../artifacts/a2_checks/latest.json), pointing to the timestamped October 3 audit.

For A3, retain H=256, d=4, learning rate 0.01, batch size 512, the exact refit preprocessing, fixed four-epoch final refits, and seeds 42–46. Compare its five-seed results against this A2 baseline; keep A1's individual final result separately identified.

## October 10 FAQ epoch-plot correction

[Q&A.md](../Q&A.md) requires training/validation RMSE and R2 against epochs and separate plots for the two ablations. Three supplementary seed-42 diagnostics train on 2017-2020 (87,370 rows), validate on untouched 2021 (29,057 rows), and use preprocessing fitted on 2017-2020 only. All use the frozen H=256/d=4 where relevant, Adam LR0.01/batch512, and exactly four completed epochs. These runs receive **no test evaluation**, do not retune any settings, and do not replace the combined-data five-seed final table above. Diagnostic selected checkpoints are not final-model checkpoints.

- [A2a categorical-removal epoch curves](../part_a/outputs/a2/diagnostics/figures/categorical_epoch_curves.png): baseline versus continuous-only, each train and validation, RMSE and R2.
- [A2b activation epoch curves](../part_a/outputs/a2/diagnostics/figures/activation_epoch_curves.png): baseline versus Sigmoid, each train and validation, RMSE and R2.
- [Saved diagnostic summary](../part_a/outputs/a2/diagnostics/summary.json) and [verification](../part_a/outputs/a2/diagnostics/verification.json) record the split, four-epoch coverage and zero test passes.

At epoch four, train/validation RMSE is SGD33,348.72/63,246.90 for baseline, 50,464.84/72,073.21 for continuous-only, and 34,780.02/68,030.75 for Sigmoid. Categorical removal has higher error on both partitions. Sigmoid has no diagnostic validation gain at epoch four. This original-split trajectory and the small five-seed final Sigmoid gain concern different fitting protocols and do not establish a contradiction or robust superiority.

```powershell
.\.venv\Scripts\python.exe -B part_a/scripts/run_a2_diagnostics.py --export-only
# Train new diagnostic evidence in a new destination; no test scoring.
.\.venv\Scripts\python.exe -B part_a/scripts/run_a2_diagnostics.py --output part_a/outputs/new_a2_diagnostics
```
