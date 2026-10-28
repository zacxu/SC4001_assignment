# Shared training and evaluation — step 3

Completed October 1, 2026. Implementation: [assignment_common/training.py](../assignment_common/training.py), [metrics.py](../assignment_common/metrics.py), [artifacts.py](../assignment_common/artifacts.py), and [reproducibility.py](../assignment_common/reproducibility.py). Part-specific entry points are [train_regressor](../part_a/src/training.py) and [train_classifier](../part_b/src/training.py).

## Training protocols

`TrainingConfig` defaults to Adam learning rate `1e-3`, a maximum of 20 epochs, patience 4, minimum improvement `1e-4`, seed 42, and CUDA. Part A must explicitly set learning rate `1e-2`. Epoch budgets above 20 are rejected. Call `seed_everything(seed)` **before constructing the model**, then build loaders with `make_loader(..., seed=seed)`. `fit` also seeds training randomness; loader generators are independent.

| Mode | Permitted monitoring data | Stopping | Selected checkpoint |
| --- | --- | --- | --- |
| `selection`, regression | Validation only | Validation RMSE in SGD | Lowest validation RMSE; earlier epoch for an exact tie |
| `selection`, classification | Validation only | Validation BCE loss | Lowest validation classification error; then lower loss; then earlier epoch |
| `a1_refit` | Explicit `test_monitor_loader`, regression only | Test RMSE in SGD, following professor guidance | Lowest test RMSE; earlier epoch for an exact tie |
| `fixed_refit` | No monitoring loaders | Complete the specified `E*` epochs | Final epoch |

A monitored decrease must be strictly greater than `min_delta` to reset patience, matching the supplied baseline comparison. Checkpoint selection uses the actual minimum independently: a smaller improvement can still save the lowest-RMSE checkpoint without resetting patience. Training stops after four consecutive epochs without significant monitored improvement, or at the configured cap. A2/A3 fixed refits use `max_epochs=E*` with `mode="fixed_refit"`.

The trainer returns `FitResult` with the checkpoint path, selected epoch, total completed epochs, early-stopping flag, selected-epoch metrics, and public history. It restores the selected weights before returning, so the returned model and reported row refer to the same epoch. The model is left in evaluation mode.

## Loss and metric semantics

- Regression model outputs are standardized prices. Loss is MSE against `(SGD_target - target_mean) / target_std`; RMSE/R² use inverse-transformed SGD predictions. Target scaling is stored in the checkpoint and restored into the supplied model's scalar attributes.
- Classification uses one raw logit and `BCEWithLogitsLoss`. Metrics use sigmoid probability with `probability >= 0.5` as the positive class, under the recorded label-1 pneumonia assumption.
- Evaluation runs under `no_grad` with dropout disabled and BatchNorm frozen, and restores the caller's prior train/eval mode. Grad-CAM runs separately with gradients enabled.
- Losses are weighted by sample count, including short final batches. RMSE comes from global squared error, not the average of batch RMSEs. R² uses a stable combined target-variance calculation. Confusion-matrix counts are accumulated over the full loader.
- Empty inputs, nonfinite outputs/losses, wrong output shapes, invalid labels, and evaluation loaders that drop their last batch fail explicitly. Failed runs are recorded as failed and cannot receive a final test evaluation.
- Undefined R² (constant targets or fewer than two samples), sensitivity (no positive labels), or specificity (no negative labels) is returned as `None` and saved as JSON `null`. It is not replaced by a misleading zero. Five-seed summary creation rejects undefined metric values.

## Histories, checkpoints, and final test evaluation

Every fit uses a **new run directory**. An existing directory raises an error so previous results remain intact.

```text
run_directory/
  config.json                   # training/task/model settings and stopping/selection rules
  preprocessing.pt             # CPU tensors/basic values; original category key types preserved
  history.jsonl
  history.csv                   # completed-epoch train metrics and validation metrics when present
  best_model.pt                 # selected weights, Adam state, metadata, preprocessing, selected metrics
  summary.json                  # selected epoch, completed epochs, completion status
  internal_test_monitor.jsonl   # only the explicitly permitted A1 refit
  test_metrics.json             # written only by final test evaluation
```

Public `train_loss`, `train_rmse`, and other training metrics evaluate the full training dataset using the completed epoch's weights, alongside validation metrics at that epoch. `train_loss_online` separately records the sample-weighted optimization loss while the model is being updated. A separate sequential evaluation loader prevents metric calculation from advancing the training loader's shuffle generator.

For A1's combined-data refit, test monitoring is saved only in the internal file. Public histories and epoch callbacks contain no test metrics and therefore support the professor's train/validation-only plotting instruction. The best test-assisted score and protocol are recorded in the summary/checkpoint; the report must retain the stated independence limitation.

Use `restore_checkpoint(path, model, device=...)` with a matching newly constructed architecture. It loads with `weights_only=True`, restores model state and target transforms, and returns saved preprocessing/model/training metadata plus the task object. An optimizer can optionally receive its saved state; this is not a complete exact-resume mechanism because RNG/loader positions are not exported.

Use `evaluate_final_checkpoint(run_directory, model, test_loader, device=...)` for final test scoring. It requires a completed run, restores its selected checkpoint, computes all final metrics together, and saves `test_metrics.json`. A second call fails before reading test samples; use the existing file for later tables instead. A1's earlier internal stopping evaluations are the explicit professor-authorized exception.

A4's required perturbed test-input forward passes compute explanations with fixed restored models. The original A1/A3 performance records are reused. Full signed SGD attributions, references, row IDs, mean-absolute aggregates, and independent manual differences are saved separately under `part_a/outputs/a4/`; see [A4 experiment report](A4_EXPERIMENT.md).

Pass all actual encoders/scaling statistics through `preprocessing`, and architecture details through `model_config`, when the data/model modules become available in later steps. Checkpoints preserve those supplied values rather than trying to reconstruct them from weights.

## Five-seed results and packaging

`summarize_seed_results(records)` requires five distinct integer seeds and reports each metric's mean and **sample** standard deviation (`ddof=1`), retaining individual records. `save_seed_results(directory, records)` writes the summary JSON and per-seed CSV to a new directory. Default regression metric names are `rmse` and `r2`.

Both parts use the same `assignment_common` implementation. Include that package at the root of **each** final code ZIP alongside its part's code so both submissions can run independently. No common module imports the sibling part, workspace config files, or supplied reference project.

## Verification

Run from the assignment root:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B scripts\verify_training.py
```

**23 tests passed**, covering golden regression/confusion-matrix values, unequal final batches, sample standard deviation, early-stopping boundaries, separate loss/error selection, selected-checkpoint restoration, target transforms/preprocessing, mode/BatchNorm/dropout behavior, seed reproducibility, fixed refits, internal A1 monitoring, single final test evaluation, failed runs, and existing artifact protection.

The CUDA integration check passed with the supplied housing baseline and a small X-ray probe, each for two epochs. It restored each checkpoint into a fresh model and reproduced every selected validation metric. The housing probe used 257 training/65 validation samples; the X-ray probe used 129/65, with short final batches. The X-ray probe is not a B1–B3 architecture. No real test-set prediction performance or full assignment experiment was evaluated.

Latest integration evidence: [artifacts/training_checks/latest.json](../artifacts/training_checks/latest.json). Each timestamped check folder retains its two run directories and [verification script](../scripts/verify_training.py) output. Synthetic tests use temporary directories; no new packages were needed beyond the existing `.venv`.
