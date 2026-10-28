# Part B data and architecture verification — step 9

Completed October 3, 2026. Sources: [assignment brief p. 5 / Figure B](../assignment.pdf), [slides p. 4](../assignment_slides.pdf), [recorded decisions](ASSIGNMENT_DECISIONS.md), and the supplied NPZ. Figure B was rechecked visually before implementation. **Data preparation and the three CNN implementations are complete.** Subsequent results are recorded in the [B1 report](B1_EXPERIMENT.md), [B2 report](B2_EXPERIMENT.md), [B3 architecture-comparison report](B3_EXPERIMENT.md), and completed [B4 explanation report](B4_EXPERIMENT.md).

## Supplied data and normalization

The archive has exactly six numeric arrays, loaded with `allow_pickle=False`. Every image is uint8, 64×64, with values in 0–255. Original membership, ordering, repeated images, and labels are retained. Each prepared split stores normalized float32 inputs `N×1×64×64`, float32 binary targets `N`, raw uint8 images `N×64×64`, and original within-split row IDs. Raw images are available for B4 overlays without reversing normalization.

| Provided partition | Images | Label 0 | Label 1 | Label-1 fraction |
| --- | ---: | ---: | ---: | ---: |
| Training | 4,708 | 1,214 | 3,494 | 74.21% |
| Validation | 524 | 135 | 389 | 74.24% |
| Test | 624 | 234 | 390 | 62.50% |

**Class-name assumption:** 0 = healthy, 1 = pneumonia. These names are not present in the archive metadata; they remain an explicitly recorded interpretation. Label 1 will be the positive class for sensitivity and Grad-CAM. Training is imbalanced, so later results must include sensitivity and specificity alongside accuracy. No resampling, augmentation, or class/loss weighting has been added.

Scale each image to float32 `x / 255.0` once, then normalize using `(x_scaled − mean_train) / std_train`. Fit statistics over all **19,283,968 training pixels**, using float64 reductions on the scaled float32 values and population standard deviation (`ddof=0`). Apply the saved statistics with float32 arithmetic to every partition:

- Training mean: **0.5716994212632378**.
- Training population standard deviation: **0.17409753982645407**.
- Normalized training-pixel mean: approximately **−1.15×10⁻⁷**; standard deviation: **1.0000000183**. The tiny residual reflects float32 transformation arithmetic.

The fitting API receives only training images; validation/test values do not fit any preprocessing. Already-scaled float arrays are rejected to prevent double scaling. Invalid shapes, missing/extra keys, noninteger/nonbinary labels, constant training images, and unexpected partition sizes are rejected. Rows are never silently removed.

### Exact repeated-image audit

SHA256 hashes of 64×64 uint8 pixel bytes identified **18 extra copies within training**, **0 within validation**, and **3 within test**. **Eight distinct pixel-identical images appear in both training and validation**, occupying 10 training rows and 8 validation rows. There are no exact pixel matches between training/test or validation/test, and no conflicting labels among repeated images.

The provided splits are preserved as required. Train/validation overlap means later validation results are not entirely independent of training images and may be optimistic. Repeated test images reduce the number of distinct test images. Exact pixel hashing does not detect similar images or patient overlap; patient identifiers are absent. State these limitations in the Part B report. Group hashes and within-split row IDs are saved in [data_report.json](../part_b/outputs/data/data_report.json).

Source NPZ SHA256: `55ab6e2c7167c451c3bbaa47727ecf810fc2d6285e0d8090a1078c67cbfa3cd8`.

## Architecture implementation

[models.py](../part_b/src/models.py) implements the compact networks drawn in Figure B. They use the verified PyTorch 2.11.0+cu128 environment and default PyTorch parameter initialization. Every convolution has **no bias** and is immediately followed by **BatchNorm2d then ReLU**. BatchNorm uses affine scale/bias, running statistics, epsilon `1e-5`, and momentum `0.1`. FC layers include bias. B1 alone has dropout 0.5 after each hidden FC/ReLU stage.

| Model | Convolution parameters | BatchNorm parameters | FC parameters | Total trainable |
| --- | ---: | ---: | ---: | ---: |
| B1 AlexNet-like | 31,632 | 224 | 67,713 | **99,569** |
| B2 VGG-like | 17,928 | 224 | 33 | **18,185** |
| B3 ResNet-like | 76,432 | 672 | 65 | **77,169** |

Counts include BatchNorm gamma/beta and FC biases, exclude running-statistic buffers, and match independent formulas: convolution `C_in × C_out × k²`, BatchNorm `2 × C_out`, FC `n_in × n_out + n_out`. Actual forward traces, including leaf modules and residual blocks, are saved in each model's `layers.csv` under the [final architecture audit](../part_b/outputs/architecture_checks/20261003_140719/).

Spatial dimensions below omit the batch axis:

| Model | Verified feature sizes and head |
| --- | --- |
| B1 | `1×64×64 → Conv5/16:16×64×64 → MaxPool:16×32×32 → Conv5/32:32×32×32 → MaxPool:32×16×16 → Conv3/64:64×16×16 → MaxPool:64×8×8 → AvgPool:64×4×4 → Flatten1024 → FC64/ReLU/Dropout → FC32/ReLU/Dropout → FC1` |
| B2 | `1×64×64 → Conv3/8 → Conv3/8 → MaxPool:8×32×32 → Conv3/16 → Conv3/16 → MaxPool:16×16×16 → Conv3/32 → Conv3/32 → MaxPool:32×8×8 → AvgPool:32×1×1 → Flatten32 → FC1` |
| B3 | `1×64×64 → Conv3/16:16×64×64 → MaxPool:16×32×32 → Block16:16×32×32 → Block32:32×16×16 → Block64:64×8×8 → AvgPool:64×1×1 → Flatten64 → FC1` |

B3's Block16 has two 3×3/16 convolutions and an identity shortcut. Block32 uses a stride-two 3×3/32, a stride-one 3×3/32, and a stride-two 1×1/32 shortcut. Block64 repeats this pattern at 64 channels. **Both main convolutions and each projection include BN/ReLU; every addition is followed by ReLU.** This follows the figure rather than substituting a standard library ResNet block. All three residual additions had exactly matching main/shortcut/sum shapes.

### Recorded padding, pooling, and output choices

The brief specifies SAME padding without defining which side receives padding at stride two. Use symmetric zero padding `kernel_size // 2`: 2 for 5×5, 1 for 3×3, and 0 for 1×1. This preserves stride-one sizes and produces `ceil(size/2)` at stride two. Main-path 3×3 and projection 1×1 both sample kernel centers at input positions **0,2,4,…**; coordinate-grid tests verified alignment for even and odd sizes. This explicit, center-aligned SAME interpretation is used throughout. PyTorch's `padding='same'` string supports stride one only. [PyTorch 2.11 Conv2d documentation](https://docs.pytorch.org/docs/2.11/generated/torch.nn.Conv2d.html).

Average pooling uses fixed windows: B1 `AvgPool2d(kernel=2,stride=2)` maps 8×8 to 4×4; B2/B3 `AvgPool2d(kernel=8,stride=8)` map 8×8 to 1×1. They produce the same required averages as adaptive pooling for these fixed inputs. An initial adaptive-pooling CUDA probe warned its backward pass lacked a deterministic implementation, so final models use fixed windows. Known-grid average/gradient tests passed, and the final CUDA audit completed without that warning. Inputs are restricted to the supplied `N×1×64×64` shape.

Each network returns one **unrestricted logit per image**, including shape `(1,)` for a single image. Following the prior recorded interpretation, final `FC1` has no ReLU or sigmoid. `BCEWithLogitsLoss` applies sigmoid internally; probabilities are formed only for metrics and thresholded at 0.5. Negative-logit tests verified that predicting the healthy class is possible. [PyTorch 2.11 BCEWithLogitsLoss documentation](https://docs.pytorch.org/docs/2.11/generated/torch.nn.BCEWithLogitsLoss.html).

Models expose their final main-path spatial convolution as `last_convolution`, plus strict metadata and reconstruction for checkpoint reuse. [B4](B4_EXPERIMENT.md) has now used those convolutions in the exact selected B1/B2/B3 checkpoints for fifteen Grad-CAM overlays. The architecture probes reported here remain separate from that completed explanation experiment.

## Executed verification

**All 70 behavioral tests passed**, including 13 new X-ray tests covering training-only fitting, scaling arithmetic, raw-image/row preservation, strict validation, exact duplicates, safe loading/alteration guards, Figure B layers/counts/bias/dropout, spatial sizes, projection alignment, residual gradients, fixed averaging, negative logits, and an actual CNN shared-trainer/checkpoint roundtrip.

The final CUDA audit ran on the **NVIDIA GeForce RTX 3060 Ti** and:

1. Independently recomputed every normalized pixel from the NPZ and matched every exported raw pixel, label, and row ID. Maximum inverse-scaling difference was below `2e-7`.
2. Traced all three networks, checked parameter formulas and residual additions, and verified aligned strided sampling on a coordinate grid.
3. Ran one Adam smoke epoch per model on the same seeded 129 training and 65 validation samples, batch 128, LR 0.001. Verified two BN training updates, finite nonzero gradients/updates in every convolution including projections, and output-weight changes.
4. Restored each safe CPU-tensor checkpoint into a fresh model, reproduced saved validation probe metrics exactly, and checked repeatable evaluation outputs with frozen BN/dropout and single-image batches.
5. Ran another forward/backward pass at B1 batch 128 and B2/B3's largest possible planned batch 256, with finite gradients for every parameter. Peak allocated memory including the fresh restored model was approximately **304.2 / 345.6 / 426.2 MiB** for B1/B2/B3; these are smoke checks, not full experiment profiling.
6. Verified original supplied-file hashes and **all 724 existing Part A output-file hashes** were unchanged. No model test performance was evaluated.

These short runs verify execution and artifact compatibility, not convergence, hyperparameter selection, or classification performance. The smoke LR is not a B1 selection result. The [final verification JSON](../part_b/outputs/architecture_checks/20261003_140719/verification.json) and [latest audit](../artifacts/part_b_checks/latest.json) hold the evidence.

## Commands and handoff

Run from the assignment root with the existing `.venv`:

```powershell
# Already exported. Use a new destination to reproduce without overwriting.
.\.venv\Scripts\python.exe -B part_b\scripts\prepare_data.py --output part_b\outputs\data_reproduction
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
.\.venv\Scripts\python.exe -B scripts\verify_part_b.py
```

Preparation defaults to `part_b/outputs/data` and protects existing exports. Verification reads that saved export by default and creates a fresh timestamped smoke directory; `--data` selects another prepared directory. All dependencies remain inside `.venv`; no new packages were needed.

```python
from assignment_common import make_loader, seed_everything
from part_b.src.data import load_data_bundle
from part_b.src.models import build_model

bundle = load_data_bundle("part_b/outputs/data")
seed_everything(42)  # Before constructing each fresh search model.
model = build_model("b1")
train_loader = make_loader(bundle.datasets["train"], 128, training=True, seed=42, device="cuda")
validation_loader = make_loader(bundle.datasets["validation"], 128, device="cuda")
```

For step 10, pass `bundle.preprocessing.metadata()` and `model.metadata()` to `train_classifier`, run the five configured learning rates with established validation stopping/selection rules, and evaluate only the chosen final model on test. Step 9 has not created B1/B2/B3 hyperparameter-selection or final-test results.
