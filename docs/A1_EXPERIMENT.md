# A1 baseline and Ray grid search — step 5

**Completed October 1, 2026.** The full 25-trial grid, fresh final refit, report assets, and CUDA checkpoint audit passed. Total experiment time was approximately 12.3 minutes on the RTX 3060 Ti, including Ray worker startup and the refit.

| Result | Value |
| --- | --- |
| Supplied default | H=64, d=8; 3,505 parameters; validation RMSE SGD 62,109.01 |
| Selected configuration | **H=256, d=4; 7,065 parameters** |
| Selected validation RMSE / R² | **SGD 58,627.10 / 0.870061**; 5.61% lower RMSE than the default |
| Selected validation checkpoint | **E_val=6**, 10 epochs completed |
| Fresh combined-data refit checkpoint | **E*=4**, 8 epochs completed |
| Final 2022 test RMSE / R² | **SGD 96,901.64 / 0.676082**, on 26,702 rows |
| Later fixed refit budget | **4 epochs**, seeds 42–46, preserving the saved refit transforms |

The higher test error is reported directly. [analysis.md](../part_a/outputs/a1/analysis.md) discusses the curves, parameter/embedding tradeoff, single-seed selection, and possible temporal shift. Post-fit raw-price summaries in `tables/partition_price_summary.csv` show observed means rising across the chronological partitions; these summaries did not influence any fit, configuration, or checkpoint. Unadjusted price averages also reflect the composition of properties sold.

## October 11 interpretation review

The selected configuration remains H=256,d=4 because its best validation RMSE, SGD 58,627.10, is the minimum of our measured grid; H=256,d=2 scores SGD 63,213.60. Width-averaged validation RMSE decreases from SGD 65,100.41 at H=16 to SGD 60,991.97 at H=256. Embedding effects are not uniformly beneficial at each fixed width, even though their marginal averages improve as d increases. These are empirical trends within one seed and one validation year, not evidence of a universal optimum.

The selected curve shows stronger evidence of **late overfitting** than persistent underfitting: epoch 6 train/validation RMSE is SGD 34,372.58/58,627.10; epoch 10 is SGD 31,678.96/70,715.09. All four later validation checkpoints are worse than epoch 6. The combined-data refit similarly improves training RMSE after epoch 4 while its test RMSE worsens. Early stopping restores the better checkpoint instead of retaining the last one.

The controlled width comparison holds d=4 fixed: H=16 train/validation RMSE is SGD 34,554.54/64,672.61. Its training error is only SGD 181.96 higher than H=256; these similar training errors do not establish substantial underfitting. These scores belong to each trial's best validation checkpoint, not an equal-epoch comparison. A model can overfit without memorizing every example; neither width relative to sample count nor a high/plateauing test error alone diagnoses underfitting. Temporal distribution differences and optimization fluctuations remain possible contributors. See the [detailed reasoning review](A1_INTERPRETATION_REVIEW.md). This revision changes explanation only; all histories, checkpoint weights, configurations and numeric results are preserved.

Implementation: [model.py](../part_a/src/model.py), [a1.py](../part_a/src/a1.py), [a1_exports.py](../part_a/src/a1_exports.py), and [run_a1.py](../part_a/scripts/run_a1.py). These use the existing project `.venv`, saved housing bundles, and shared trainer.

## Protocol and architecture

The independent `HousingMLP` reproduces the supplied baseline's layer ordering, parameter names, initialization, standardized-price output, and SGD prediction transform. Four independent embeddings concatenate with six continuous inputs, then pass through `Linear(4d+6,H) → LayerNorm(H) → ReLU → Linear(H,1)`. ID 0 retains the supplied ordinary embedding behavior; it is not forced to a zero vector.

With embedding cardinalities `[13,27,44,18]`, the trainable parameter count is:

```text
P = d × sum(cardinalities) + (4d + 10) × H + 1
Default H=64,d=8: 816 + 2,496 + 128 + 65 = 3,505
```

The four embedding tables share a dimension, not their weights. LayerNorm contributes H scale and H bias parameters. The diagram labels the supplied default; tuned model counts are reported separately.

Ray Tune runs all 25 combinations of hidden widths `{16,32,64,128,256}` and shared embedding dimensions `{2,4,8,12,16}`. Every model starts fresh with seed 42. Adam learning rate 0.01, batch size 512, LayerNorm, all ten features, and standardized-target MSE stay fixed. The original training/validation split and its saved training-only transforms are reused. One trial uses the GPU at a time; each worker reserves four CPUs, uses four PyTorch CPU threads, and keeps data-loader workers at 0.

The supplied default runs first and is one of the 25 required trials. No grid combination is pruned by a search scheduler. Each uses maximum 20 epochs, patience 4, and minimum significant improvement `1e-4`, monitored in validation RMSE in SGD. Checkpoint selection uses the actual minimum, independently of the significant-improvement threshold. A configuration tie is resolved by fewer parameters, then smaller H, then smaller d. `E_val` is the selected trial's checkpoint epoch, not its completed epoch count.

Each epoch evaluates the completed model on the whole training/validation partition. Ray receives epoch metrics and the running minimum; its final report registers the restored checkpoint and coherent selected metrics. The final report does not represent an extra training epoch. Use explicit epoch fields and the saved histories, rather than Ray's `training_iteration`, for epoch plots.

A fresh selected model then refits on 2017–2021, using the saved combined-data preprocessing. As explicitly clarified by the professor, 2022 test RMSE controls early stopping for this run only. The best restored checkpoint's epoch is `E*`; A2/A3 inherit that fixed epoch budget. Final test metrics come from one restored checkpoint. The A1-specific train/test RMSE refit plot reuses the saved internal stopping history, with no new model inference; the FAQ no-test-epoch rule applies to A2/A3.

**Limitation:** A1's test score follows selection of a stopping checkpoint on that same test partition, so it is not an untouched holdout estimate. The later common epoch budget inherits this influence. Hyperparameter selection itself uses validation only.

## Commands

Run from the assignment root using its `.venv` interpreter:

```powershell
# First run in a fresh workspace; requires the prepared data from step 4.
.\.venv\Scripts\python.exe -B -u part_a\scripts\run_a1.py

# Reproduce without overwriting the completed default run.
.\.venv\Scripts\python.exe -B -u part_a\scripts\run_a1.py --output part_a\outputs\a1_reproduction

# Regenerate report assets from saved results, with no training/test reevaluation.
.\.venv\Scripts\python.exe -B part_a\scripts\run_a1.py --export-only

# Audit checkpoints, logs, tables, and figures.
.\.venv\Scripts\python.exe -B scripts\verify_a1.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

The run command protects existing directories. `--prepared PATH` supports another saved housing-data export. The audit command supports `--output PATH` and `--prepared PATH`; it reproduces validation metrics for all 25 checkpoints and compares existing refit metrics rather than performing another test evaluation. Ray shuts down its local services before refitting and on grid errors.

The project libraries remain in `.venv`; no additional packages are needed for A1. Local paths and worker `PYTHONPATH` make the Windows Ray workers independent of their trial working directory. Ray uses its standard local temporary directory for runtime services and keeps persistent results under the selected output path.

## Saved results and report assets

The completed run is organized under `part_a/outputs/a1/`:

```text
a1/
  configuration.json               # protocol, source hash, environment, fixed/grid settings
  status.json / progress.json
  grid/
    h064_d08/                       # supplied default; all 25 h/d directories are retained
      training/
        config.json / preprocessing.pt
        history.csv / history.jsonl
        best_model.pt / summary.json
      checkpoint_000000/            # Ray-registered checkpoint copy
      ...                           # Ray result/log/experiment files
  grid_results.json
  selection.json                    # winning configuration, tie rule, E_val
  refit/
    best_model.pt / preprocessing.pt
    config.json / summary.json
    history.csv / history.jsonl      # combined training only
    internal_test_monitor.jsonl     # professor-authorized stopping evidence
    test_metrics.json               # final restored-checkpoint metrics
  summary.json
  frozen_a1_settings.json            # H*, d*, LR, batch size, E_val/E*, later seeds/paths
  analysis.md                       # A1 report material grounded in saved results
  tables/
    grid_results.csv                # every trial, parameter count, metrics, epochs, timing/path
    baseline_selection_comparison.csv
    final_test_metrics.csv
    partition_price_summary.csv     # post-fit distribution context, no prediction/tuning
  figures/                          # every figure as PNG and vector PDF
    baseline_architecture.*
    selected_learning_curve.*
    default_baseline_learning_curve.*
    validation_grid_heatmap.*
    refit_training_curve.*
  verification.json                 # added by the audit script
```

The required selected-model plot shows training/validation RMSE together. The default-model curve supports comparison, and the heatmap covers the entire grid with the selected cell outlined. The required refit curve shows combined training and the authorized test-monitor RMSE together. Final restored-checkpoint RMSE/R² also appear in a table. A2/A3 epoch curves follow the FAQ training/validation-only instruction.

The report should use the baseline diagram/count, selected learning curve, compact grid evidence, final test table, and a concise discussion from `analysis.md`. The [five-page final Part A report](../reports/PartA_report_YourName.pdf) is now complete; final code packaging remains step 15.

For later steps, read `frozen_a1_settings.json`, construct the matching model, and restore `refit/best_model.pt` with the shared `restore_checkpoint`. Checkpoints include the architecture, target transform, encoders/scalers, fitting years, source hash, and selected metrics. Reuse saved metrics for later tables and saved preprocessing for all controlled refits.

## Verification and references

The expanded suite passed **38 tests**, including exact supplied-model initialization/forward/SGD-prediction parity, parameter counts for all 25 configurations, model metadata reconstruction, complete-grid validation, and deterministic selection/tie handling. Earlier training/data tests remain in the suite.

The full-run audit **passed**: all 25 restored checkpoint validation scores were reproduced on CUDA. Final refit metrics match the restored checkpoint's selected monitoring record, and the epoch rules, metadata, tables, figures, and supplied-file hashes were verified. The baseline diagram and learning-curve/heatmap PNGs were also visually inspected. No additional test predictions were evaluated during the audit.

The full-run audit checks every restored CUDA validation score, checkpoint/history/row coherence, the stopping rule, training-only selection transforms, combined-refit transforms, E_val/E*, single final-test records, public histories, report tables/figures, and unchanged supplied-file hashes. Its evidence is saved in the run folder and indexed by [artifacts/a1_checks/latest.json](../artifacts/a1_checks/latest.json).

Primary API references: Ray's [grid_search](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.grid_search.html) and [report](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.report.html) documentation, checked against installed Ray 2.58.0. Assignment rules and the refit exception come from the supplied PDFs and [clarifications.md](../clarifications.md).

## October 11 plot-scope reconciliation

The latest Q&A asks about epoch plots specifically in A2/A3, so its train/validation-only answer does not override A1's separate explicit brief requirement to plot training/testing RMSE during refitting. A1's eight already saved internal test-monitor observations are now shown with refit training RMSE; test-assisted stopping was independently authorized in clarifications. No new test inference or training is needed. Report that same independence limitation. Earlier descriptions omitting the A1 test curve are superseded by this context-specific reading.
