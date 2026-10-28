# Part B — chest-X-ray classification

**Steps 9–14 and local submission packaging are complete.** The X-ray data, three Figure B CNNs, B1–B4 experiments, architecture comparison and [five-page Part B report](../reports/PartB_report_YourName.pdf) are verified. [Editable Markdown](../reports/PartB_report_YourName.md) accompanies it; `YourName` remains a placeholder. The [standalone code ZIP](../submission/PartB_codes_YourName.zip) passed fresh extracted-folder checks. Actual-name replacement and NTULearn upload/receipt remain pending. Detailed evidence: [preparation](../docs/PART_B_PREPARATION.md), [B1](../docs/B1_EXPERIMENT.md), [B2](../docs/B2_EXPERIMENT.md), [B3](../docs/B3_EXPERIMENT.md), [B4](../docs/B4_EXPERIMENT.md), [report completion](../docs/REPORTS_COMPLETED.md), and [submission verification](../docs/SUBMISSION_VERIFICATION.md).

## Data and models

Source: `../chestxray_binary_64.npz`, keys `train_images/train_labels`, `val_images/val_labels`, `test_images/test_labels`. Preserved sizes: **4,708/524/624**. Inputs are normalized float32 `N×1×64×64`; targets are float32 binary labels. Raw uint8 images and original within-split row IDs are retained for later overlays.

Scale uint8 pixels by 255 once. Training-only pixel mean **0.5716994212632378**, population std **0.17409753982645407**; all splits reuse these values. `outputs/data/` contains `datasets.pt`, `preprocessing.pt`, `normalization.json`, `data_report.json`, and a SHA256 manifest. Safe loading verifies prepared-file hashes and metadata.

Class interpretation **0 = healthy, 1 = pneumonia is assumed**, because the NPZ has no class-name metadata. Exact pixel duplicates include **eight distinct images shared by train and validation**, 18 extra copies within training, and three within test. The assignment's supplied partitions are unchanged; later reporting must disclose this limitation. No augmentation, weighting, or resampling is applied.

| Model | Implementation | Trainable parameters |
| --- | --- | ---: |
| B1 | `AlexNetLike`: three convolutions, three max pools, 4×4 average output, FC64/FC32 with dropout 0.5, FC1 | **99,569** |
| B2 | `VGGLike`: six convolutions, three max pools, 1×1 average output, FC1 | **18,185** |
| B3 | `ResNetLike`: stem, max pool, three residual blocks including two stride-two projections, 1×1 average output, FC1 | **77,169** |

`src/models.py` follows Figure B, including BN/ReLU after every convolution and projection, ReLU after residual sums, convolution bias disabled, FC bias enabled, and an unrestricted final logit. Symmetric explicit padding preserves aligned residual centers. Fixed average-pooling windows produce the specified output sizes for 64×64 inputs and passed CUDA backward checks. `last_convolution` exposes the final main-path convolution for B4.

## Reproduce and use

Run from the **assignment root**, using its existing `.venv` interpreter. All packages remain inside that venv; pinned requirements are copied here for later independent packaging.

```powershell
# Existing exports are protected; use a new destination for reproduction.
.\.venv\Scripts\python.exe -B part_b\scripts\prepare_data.py --output part_b\outputs\data_reproduction
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
.\.venv\Scripts\python.exe -B scripts\verify_part_b.py
.\.venv\Scripts\python.exe -B scripts\verify_b1.py
.\.venv\Scripts\python.exe -B scripts\verify_b2.py
.\.venv\Scripts\python.exe -B scripts\verify_b3.py
```

Preparation defaults to `part_b/outputs/data` if `--output` is omitted. Verification reads that export and writes fresh timestamped smoke probes under `outputs/architecture_checks/`, plus `../artifacts/part_b_checks/latest.json`. The current fixed-pooling audit is `outputs/architecture_checks/20261003_140719/`; the earlier adaptive-pooling probe is marked superseded.

```python
from assignment_common import make_loader, seed_everything
from part_b.src.data import load_data_bundle
from part_b.src.models import build_model, model_from_metadata

bundle = load_data_bundle("part_b/outputs/data")
seed_everything(42)
model = build_model("b1")  # "b2" and "b3" construct the other figures.
train_loader = make_loader(bundle.datasets["train"], 128, training=True, seed=42, device="cuda")
validation_loader = make_loader(bundle.datasets["validation"], 128, device="cuda")
```

`src/training.py` delegates to `../assignment_common/`: Adam, maximum 20 epochs, patience 4, minimum improvement `1e-4`; BCEWithLogitsLoss with sigmoid threshold 0.5. Stop on validation loss and select by validation classification error, then loss, then earlier epoch. Pass `bundle.preprocessing.metadata()` and `model.metadata()` to `train_classifier` so checkpoints record normalization and architecture. See [training rules](../docs/TRAINING_RULES.md). Final test evaluation is a separate recorded operation after selection.

All **96 tests** passed in the current suite. The preparation audit checked every exported pixel, actual shapes/counts, residual branches, gradients/Adam updates, evaluation behavior, and restoration. Each model's one-epoch smoke run used 129 training/65 validation samples; backward probes passed at batches 128/256/256. **These architecture probes are distinct from the completed full B1/B2/B3 experiments below.**

## B1 completed results

All five learning rates were trained at **batch 128, seed 42** with identical initial weights and loader seeds, full supplied training/validation partitions, Adam, and the 20/4/1e-4 rules. Validation selected **LR 0.001, checkpoint epoch 9**: accuracy **97.9008%**, error **2.0992%**, loss **0.064911**. Loss stopping ended that run at epoch 12; minimum loss was at epoch 8, because loss stopping and error selection use different criteria.

One final test evaluation of that exact checkpoint gave **accuracy 85.4167%, sensitivity 99.7436%, specificity 61.5385%**, with TP/TN/FP/FN **389/144/90/1**. The [report](../docs/B1_EXPERIMENT.md) discusses the substantial specificity decrease and supplied-data limitations. The threshold and learning rate were frozen before test access.

`outputs/b1/` stores the required five-rate table, two separate loss/accuracy figures with training/validation panels, histories and checkpoints, test records, captured same-pass logits/probabilities, discussion, and `frozen_b1_settings.json`. Every candidate's full train/validation metrics reproduced exactly on CUDA; final test metrics were recomputed from saved predictions without another model test pass. Previous output files and supplied files were unchanged.

```powershell
# Reproduce to a new directory; the completed default output is protected.
.\.venv\Scripts\python.exe -B part_b\scripts\run_b1.py --output part_b\outputs\b1_reproduction

# Rebuild only report assets from the saved experiment.
.\.venv\Scripts\python.exe -B part_b\scripts\run_b1.py --export-only
```

## B2 completed results

All five batch sizes **16/32/64/128/256** were trained on CUDA with B1's frozen **LR 0.001**, seed 42, identical initial VGG weights and underlying sample order, full saved training/validation data, and Adam with 20/4/1e-4 stopping. Validation selected **batch 16, checkpoint epoch 13**: accuracy **97.9008%**, error **2.0992%**, loss **0.069390**. Minimum validation loss was at epoch 9; loss stopping ended at epoch 13. The [B2 report](../docs/B2_EXPERIMENT.md) includes the required table, both figures, candidate-specific curve interpretation, and the different optimizer/BN update budgets.

The selected checkpoint's single final test evaluation gave **accuracy 85.5769%, sensitivity 99.2308%, specificity 62.8205%**, TP/TN/FP/FN **387/147/87/3**. `outputs/b2/` contains histories/checkpoints, required and detailed CSVs, separate PNG/vector-PDF loss/accuracy figures, same-pass final test predictions, discussion, and `frozen_b2_settings.json` for B3. All five checkpoints reproduced full training/validation metrics exactly on CUDA. The audit checked epoch rules, Adam/BN counters, B1/data hashes, and final metrics from captured predictions with no new model test pass. Supplied files and all 833 previous output files were preserved.

```powershell
# Reproduce in a new directory; the completed default output is protected.
.\.venv\Scripts\python.exe -B part_b\scripts\run_b2.py --output part_b\outputs\b2_reproduction

# Rebuild only report assets from the saved experiment.
.\.venv\Scripts\python.exe -B part_b\scripts\run_b2.py --export-only

# Audit saved checkpoints and predictions without another test inference.
.\.venv\Scripts\python.exe -B scripts\verify_b2.py
```

## B3 and architecture comparison completed

B3 used the exact frozen **LR 0.001 and batch 16**, seed 42, saved full training/validation data, and shared Adam/stopping rules. Validation selected **epoch 9**, accuracy **97.5191%**, error **2.4809%**, loss **0.061988**; training ended at epoch 13. The single final test gave **accuracy 82.8526%, sensitivity 99.7436%, specificity 54.7009%**, TP/TN/FP/FN **389/128/106/1**. B3 detected the same 389 positives as B1 but made 16 more false positives. The [report](../docs/B3_EXPERIMENT.md) discusses this specificity limitation and the architecture/training-recipe tradeoffs.

`outputs/b3/` stores the required three-model comparison table using saved B1/B2 test results, B3's separate loss/accuracy figures, histories/checkpoint, captured predictions, B4 frozen settings, and detailed parameter/FLOPs records. B1/B2/B3 have **99,569 / 18,185 / 77,169 parameters** and **39,063,616 / 19,464,256 / 25,325,696 profiled FLOPs/image**. Profiling uses a synthetic float32 `1×1×64×64` CPU evaluation copy, counting two FLOPs per convolution/linear MAC plus residual additions. BN/activation/pooling/bias and other uncounted operations are documented. These estimates are not latency or a complete scalar-operation count. All three profiler totals matched independent shape calculations.

All 88 tests passed. B3 full train/validation metrics reproduced exactly on CUDA; its captured final predictions reproduced metrics/BCE without another test inference. The audit checked B1/B2 inherited settings, all nine BN counters and optimizer steps, comparison-file lineage, synthetic-only complexity/state preservation, and unchanged supplied/884 previous output hashes.

```powershell
# Reproduce in a new directory, preserving completed results.
.\.venv\Scripts\python.exe -B part_b\scripts\run_b3.py --output part_b\outputs\b3_reproduction

# Rebuild only report assets, without training/profiling/test inference.
.\.venv\Scripts\python.exe -B part_b\scripts\run_b3.py --export-only

# Audit checkpoints, predictions, and shape-checked FLOPs.
.\.venv\Scripts\python.exe -B scripts\verify_b3.py
```

## B4 completed Grad-CAM comparison

Seed **42** sampled positive-class test row IDs **486, 279, 413, 53, 276** once before explanations. The exact selected models target `features.conv3.conv`, `features.conv6.conv`, and `features.block64.conv2.conv` with Captum `LayerGradCam`, scalar positive-class logits, evaluation mode, gradients enabled, and positive-attribution ReLU. Native grids are **16×16/16×16/8×8**. The [figure](outputs/b4/figures/gradcam_comparison.png) and [PDF](outputs/b4/figures/gradcam_comparison.pdf) show five originals and **15 overlays**, using shared inferno/opacity rules and per-map maximum normalization after bilinear interpolation to 64×64. All maps are nonempty. Colors indicate relative importance within each map.

The [report](../docs/B4_EXPERIMENT.md) compares every image and visible brightness: compact central B1 hotspots, more upper/peripheral B2 patches, and broader/coarser B3 regions in this fixed sample. Lung/lesion annotations are absent, so these patterns do not validate disease localization. Numeric labels retain the recorded semantic assumption. The original CUDA batch sizes are matched with duplicates of selected inputs only; padding maps are discarded. All 15 logits/probabilities match saved final-test values exactly. `outputs/b4/gradcam.pt` retains raw pixels, normalized inputs, activations, gradients, weights, signed/positive maps, display maps, probabilities, and row IDs; three CSVs preserve selection and numeric diagnostics.

All **96 tests passed**. The independent CUDA [audit](../scripts/verify_b4.py) uses PyTorch autograd without Captum and reproduced every map tensor exactly. It checked sampling, pixels, layers, interpolation/normalization, tables, probabilities, model/BN state, source hashes, and all **922 unchanged prior output files**. Attribution uses only the fixed selected images, with duplicate padding; no full-test performance evaluation, model update, or retuning occurs. The initial small-batch consistency failure is archived at `outputs/b4_attempt_001/` and explained in the report.

```powershell
# Run once into a NEW output directory, using existing data and selected models.
..\.venv\Scripts\python.exe -B -u scripts\run_b4.py --output outputs\b4_reproduction
..\.venv\Scripts\python.exe -B -u ..\scripts\verify_b4.py --output outputs\b4_reproduction

# Regenerate figures/CSVs from saved tensors without inference.
..\.venv\Scripts\python.exe -B scripts\run_b4.py --export-only
```

The final reports and standalone code ZIP are complete and verified. The [standalone README](../submission_support/README_B.md) documents extraction-root commands and venv setup. The ZIP contains `assignment_common/`, this part's modules/tests/data/evidence, and passed 62 tests, exact preparation, all four exporters and local selected-checkpoint loading. See [submission verification](../docs/SUBMISSION_VERIFICATION.md); author-name replacement and NTULearn upload/receipt remain pending.
