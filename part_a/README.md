# Part A — housing regression

Part A experiments A1–A4, the [five-page report](../reports/PartA_report_YourName.pdf), and local submission packaging are complete and verified. [Editable Markdown](../reports/PartA_report_YourName.md) and the [report completion record](../docs/REPORTS_COMPLETED.md) accompany it; `YourName` remains a placeholder. The [standalone ZIP README](../submission_support/README_A.md) supplies independent environment/data/training/export commands. Shared training is in `src/training.py` using `../assignment_common/`, data in `src/data.py`, A1 in `src/model.py`/`src/a1.py`, A2 in `src/ablations.py`/`src/a2.py`, A3 in `src/wide_deep.py`/`src/a3.py`, and A4 in `src/a4.py`. [Submission details](../submission/README.md) record author/upload actions still pending.

`train_regressor` supports validation-selected training, the professor-authorized A1 test-monitored refit, and fixed-epoch A2/A3 refits. It uses standardized-target MSE and SGD metrics, persists preprocessing/configuration/selected checkpoints, and restores the selected model before returning. See [training rules](../docs/TRAINING_RULES.md) for the API and verification commands. Include `assignment_common/` at the root of this part's final ZIP.

Use the assignment root's `.venv` interpreter. A copy of its pinned requirements is kept here so this part can later be packaged independently. For setup commands, see `../README.md`.

The supplied baseline remains at `../housing_baseline_data/hdb-price-prediction-assignment-main/`. It has four embeddings, a 38→64 hidden linear layer, LayerNorm, ReLU, and a 64→1 output: 3,505 trainable parameters at its default dimensions. Targets are standardized for MSE training and inverse-transformed for SGD predictions.

Required partitions: train 2017–2020, validation 2021, test 2022; exclude 2023. Selection seed: 42. Evaluation seeds: 42, 43, 44, 45, 46. Follow `../docs/ASSIGNMENT_DECISIONS.md`, including the professor's A1 test-assisted stopping clarification.

Prepared datasets and encoders/scalers are in `outputs/data/selection/` and `outputs/data/refit/`. `outputs/data/report.json` records the schema, source checksum, quality checks, mappings, fitted statistics, and per-partition unknown-category counts. `split_membership.csv` preserves original row IDs. Both protocols have cardinalities `[13, 27, 44, 18]`; five test rows have an unseen flat model type and correctly use ID 0.

Load a bundle with `load_data_bundle(path, expected_protocol="selection")` or `expected_protocol="refit"` from `part_a.src.data`. Pass its preprocessing scalars and `metadata()` to the shared trainer. Dataset targets are raw SGD values; the trainer standardizes them for loss. Preserve one saved refit transformation across A1/A2/A3. See [housing preparation](../docs/HOUSING_DATA.md) for the full API and artifact format.

From the assignment root:

```powershell
# Reproduce into a new directory; the existing export is protected.
.\.venv\Scripts\python.exe -B part_a\scripts\prepare_data.py --output part_a\outputs\data_reproduction
.\.venv\Scripts\python.exe -B scripts\verify_housing_data.py
```

All saved tensors and statistics match the supplied baseline exactly, and the expanded suite passes 57 tests. Future experiment outputs also belong under `outputs/`: configurations, epoch histories, checkpoints, individual seed metrics, summary tables, and figures. Export preprocessing explicitly because the supplied model's `state_dict` omits category maps and scalar target statistics.

## Completed A1 experiment

All 25 Ray Tune grid configurations completed on CUDA. Validation selected **H=256, d=4**, giving 7,065 parameters, RMSE SGD 58,627.10, and `E_val=6` (10 epochs completed). The supplied-default trial has 3,505 parameters and validation RMSE SGD 62,109.01.

The fresh 2017–2021 refit restored **E*=4** after 8 completed epochs, using the professor-authorized test-RMSE stopping rule. Final test RMSE is **SGD 96,901.64**, R² **0.676082**. This score reflects test-assisted checkpoint selection. All 25 validation checkpoints were restored and their metrics reproduced on CUDA; all supplied files remain unchanged.

Results are in `outputs/a1/`: `analysis.md`, complete grid/final-test CSV tables, PNG/vector-PDF diagrams and learning curves, every trial checkpoint, the final `refit/best_model.pt`, and audit evidence. [docs/A1_EXPERIMENT.md](../docs/A1_EXPERIMENT.md) describes the artifacts, reproduction, and interpretation.

```powershell
# From the assignment root; reproduce into a new output directory.
.\.venv\Scripts\python.exe -B -u part_a\scripts\run_a1.py --output part_a\outputs\a1_reproduction
.\.venv\Scripts\python.exe -B part_a\scripts\run_a1.py --export-only
.\.venv\Scripts\python.exe -B scripts\verify_a1.py
```

For A2/A3, use `outputs/a1/frozen_a1_settings.json`: H=256, d=4, Adam learning rate 0.01, batch size 512, fixed 4 epochs, seeds 42–46, and the saved refit preprocessing. `configs/default.json` retains the supplied baseline defaults and assignment grids; frozen selected settings are stored separately.

## Completed A2 experiment

All 15 fresh runs completed on CUDA using the same 2017–2021 data, exact A1 refit preprocessing, and fixed four-epoch budget. Each received one final test evaluation. LayerNorm was retained in every model; continuous-only removed all embeddings, and Sigmoid changed only the hidden activation. Baseline and Sigmoid initialization matched exactly for each seed, and every matched seed used the same shuffled row order.

| Variant | Parameters | Test RMSE, SGD: mean ± sample SD | Test R²: mean ± sample SD |
| --- | ---: | ---: | ---: |
| Baseline (ReLU) | 7,065 | 102,427.90 ± 5,289.17 | 0.637311 ± 0.037428 |
| Continuous only | 2,561 | 108,252.68 ± 9,605.81 | 0.593204 ± 0.068794 |
| All inputs (Sigmoid) | 7,065 | 101,821.96 ± 4,082.21 | 0.641892 ± 0.028889 |

Removing categorical inputs worsened RMSE in four of five seeds; Sigmoid's small mean gain occurred in three of five. These are descriptive comparisons across seeds 42–46. A2 performed no test-based stopping, while its fixed budget inherits A1's test-assisted selection. See [A2 experiment report](../docs/A2_EXPERIMENT.md) for controls and limitations.

`outputs/a2/` contains all per-seed checkpoints/histories, full-precision tables, mean/sample-SD summaries, paired differences, two separate PNG/vector-PDF ablation figures, [discussion](outputs/a2/analysis.md), and audit evidence. All 15 checkpoints were restored and their training metrics reproduced on CUDA; supplied files and A1 checkpoints remain unchanged.

```powershell
# From the assignment root; use a new directory for reproduction.
.\.venv\Scripts\python.exe -B -u part_a\scripts\run_a2.py --output part_a\outputs\a2_reproduction
.\.venv\Scripts\python.exe -B part_a\scripts\run_a2.py --export-only
.\.venv\Scripts\python.exe -B scripts\verify_a2.py
```

Use `outputs/a2/seed_results/baseline/` as A3's matching five-seed baseline.

## Completed A3 experiment

The wide-and-deep model shares the four embeddings and all six continuous inputs between the selected MLP and a single-output linear branch. It adds outputs in standardized space and applies the price inverse transform once. It has **7,088 parameters**, including 23 in the additional branch. Lambda 0 reproduced A1's trained baseline weights and metrics exactly on CUDA.

Five validation-only candidates `{0, 0.25, 0.5, 1, 2}` selected **λ=2**, best validation RMSE **SGD 57,905.65**, R² **0.873239**, at epoch 3 (7 completed). All other A1 settings remained fixed. Five fresh combined-data refits used the exact saved preprocessing and **four epochs**, with one final test evaluation each.

| Model | Test RMSE, SGD: mean ± sample SD | Test R²: mean ± sample SD |
| --- | ---: | ---: |
| Matching A2 baseline | 102,427.90 ± 5,289.17 | 0.637311 ± 0.037428 |
| Wide-and-deep, λ=2 | 101,413.74 ± 6,407.58 | 0.644081 ± 0.044905 |

The mean RMSE improvement was **0.99%**, occurring in four of five paired seeds. The gain is modest relative to seed variability, and the four-epoch budget inherits A1's test-assisted selection. Positive lambdas have equivalent function classes through linear-weight rescaling; initialization and optimization can still differ. See [A3 experiment report](../docs/A3_EXPERIMENT.md) for the mechanism discussion and limitations.

`outputs/a3/` contains every search/refit checkpoint, full tables, architecture/search/learning-curve/final-comparison PNG and vector-PDF figures, [discussion](outputs/a3/analysis.md), and production audit. All ten checkpoints passed CUDA restoration checks for training metrics, and all five search validation metric sets were reproduced. The A2 baseline was reused without another test evaluation.

```powershell
# From the assignment root; reproduce into a new directory.
.\.venv\Scripts\python.exe -B -u part_a\scripts\run_a3.py --output part_a\outputs\a3_reproduction
.\.venv\Scripts\python.exe -B part_a\scripts\run_a3.py --export-only
.\.venv\Scripts\python.exe -B scripts\verify_a3.py
```

For A4 use `outputs/a3/frozen_a3_settings.json`, A3's `refit/seed_42/best_model.pt`, A1's `refit/best_model.pt`, and shared refit preprocessing. Construct the A3 model using `WideDeepHousingModel.from_metadata`; `restore_checkpoint` also restores target statistics.

## Completed A4 experiment

Captum FeatureAblation explained **all 26,702 test rows and ten original inputs** for both seed-42, epoch-4 final models. Shared references are refit training categorical modes and saved continuous training means. Category IDs are replaced before embedding lookup; SGD target scaling is applied once after the model output. Global scores use **mean(abs(per-row attribution))** over the full test partition.

| Rank | Baseline feature | Score (SGD) | Wide-and-deep feature | Score (SGD) |
| --- | --- | ---: | --- | ---: |
| 1 | Floor area | 71,537.72 | Floor area | 73,119.34 |
| 2 | Remaining lease years | 46,187.61 | Remaining lease years | 48,364.21 |
| 3 | Distance to Dhoby Ghaut | 41,163.95 | Flat model type | 42,280.28 |

The models share two top-three features; flat model type and distance to Dhoby Ghaut swap third/fourth place. Rankings are otherwise unchanged. References and correlated-input replacements limit the interpretation to model sensitivity. See [A4 experiment report](../docs/A4_EXPERIMENT.md) and [discussion](outputs/a4/analysis.md) for the full comparison and limitations.

`outputs/a4/` contains two full 26,702×10 signed SGD matrices, original row IDs/predictions, exact references, full ranked/comparison tables, separate descending PNG/vector-PDF plots with the same axis scale, and manual/audit evidence. The production audit recomputed every global score and checked first/last-batch manual differences on CUDA; errors were zero for both models. Supplied files and A1–A3 checkpoints remain unchanged.

```powershell
# From the assignment root; reproduce into a new directory.
.\.venv\Scripts\python.exe -B -u part_a\scripts\run_a4.py --output part_a\outputs\a4_reproduction
.\.venv\Scripts\python.exe -B part_a\scripts\run_a4.py --export-only
.\.venv\Scripts\python.exe -B scripts\verify_a4.py
```

Load per-row artifacts with `torch.load(..., map_location="cpu", weights_only=True)`. All execution uses a Python venv; no additional A4 dependencies were installed. The earlier standalone Part A ZIP passed 57 tests. Current FAQ-corrected packaging adds three diagnostic behavior tests and validates all exporters, exact preprocessing, local checkpoints and Ray; the latest extraction audit records the current count. See [submission verification](../docs/SUBMISSION_VERIFICATION.md).

October 11 FAQ completion adds three original-split seed-42 fixed-four-epoch A2 diagnostics and separate genuine training/validation RMSE/R2 curves, without retuning or test scoring. A3 saved train/validation epoch curves are included in the concise report. A1 retains its separate explicit brief-required training/testing RMSE refit plot from the professor-authorized monitoring history. Run `part_a/scripts/run_a2_diagnostics.py --export-only` from the workspace root to regenerate only the diagnostic figures.
