# B4: Grad-CAM comparison — completed

**Step 13 is complete.** The exact selected B1/B2/B3 checkpoints were explained on the same five randomly selected positive-class test images. All **15 Grad-CAM overlays** are saved in a labelled comparison figure, with original grayscale images, saved prediction probabilities, the shared display scale, and an image-by-image comparison below. All maps are nonempty. Independent CUDA autograd reproduced every saved activation, gradient, channel weight, logit, probability, and map **exactly**. All **96 tests passed**, and all **922 prior output files** and supplied files retain their recorded hashes.

Sources: [assignment brief, B4 pp. 6–7](../assignment.pdf), [slides, B4 p. 5](../assignment_slides.pdf), [recorded assumptions](ASSIGNMENT_DECISIONS.md), [verified CNNs](../part_b/src/models.py), and the completed [B1](B1_EXPERIMENT.md), [B2](B2_EXPERIMENT.md), and [B3](B3_EXPERIMENT.md) experiments. Both five-page final reports are complete; package verification remains step 15. See [report completion](REPORTS_COMPLETED.md).

## Reproducible selection and retained checkpoints

NumPy `default_rng(42)` / PCG64 drew five indices uniformly **without replacement** from all **390 label-1 test rows**, in random draw order: **486, 279, 413, 53, 276**. The IDs and selection rule were saved before computing explanations. These images were retained without filtering by appearance, probability, correctness, or heatmap quality. “Representative” is interpreted as random examples from the supplied positive-class partition; five draws do not establish population representativeness. [Selection record](../part_b/outputs/b4/selection.json).

The archive supplies numeric labels without class-name metadata. As in B1–B3, **label 1 is assumed to mean pneumonia**, and label 0 healthy. Original row IDs, uint8 pixels, orientation, and the training-only normalization are preserved. Attribution inputs are the same saved float32 `1×64×64` tensors used in the classification experiments, normalized with training-pixel mean **0.5716994212632378** and standard deviation **0.17409753982645407** after division by 255.

| Model | Selected epoch | LR / training batch | Selected checkpoint | Grad-CAM layer | Native activation grid |
| --- | ---: | --- | --- | --- | --- |
| B1, AlexNet-like | 9 | 0.001 / 128 | `b1/runs/lr_1e-03/best_model.pt` | `features.conv3.conv` | 64 channels, 16×16 |
| B2, VGG-like | 13 | 0.001 / 16 | `b2/runs/batch_016/best_model.pt` | `features.conv6.conv` | 32 channels, 16×16 |
| B3, ResNet-like | 9 | 0.001 / 16 | `b3/run/best_model.pt` | `features.block64.conv2.conv` | 64 channels, 8×8 |

The paths are relative to `part_b/outputs/`. [Configuration](../part_b/outputs/b4/configuration.json) records checkpoint, frozen-settings, data, preprocessing, saved-prediction, and final-test-result hashes. No model is retrained, no hyperparameter is selected, and no full-test performance evaluation is performed in B4. The attribution forwards and independent gradient checks operate on the fixed selected images.

## Attribution and display method

Models remain in **evaluation mode**, so dropout is disabled and BN uses its saved running statistics; gradients are explicitly enabled. Captum `LayerGradCam.attribute` targets the unrestricted positive-class **logit**, with `target=None` because each row has one scalar output. Gradients do not target the sigmoid probability, avoiding its saturation in highly confident examples. The selected convolution's **output before its following BN/ReLU** is explained, with `attribute_to_layer_input=False` and `relu_attributions=True`. [Captum LayerGradCam documentation](https://captum.ai/api/layer.html#captum.attr.LayerGradCam).

For activation channel (A^k) and positive-class logit (z), the implementation computes

\[
\alpha_k=\frac{1}{HW}\sum_{i,j}\frac{\partial z}{\partial A^k_{ij}},\qquad
L=\operatorname{ReLU}\left(\sum_k\alpha_k A^k\right).
\]

Spatially averaged gradients therefore weight the activation channels, followed by a channel sum and positive-evidence ReLU. Signed pre-ReLU maps are also retained. This follows the Grad-CAM construction described by [Selvaraju et al. (2017)](https://arxiv.org/abs/1610.02391). B3 targets its final main-path convolution; shortcut activations are not separately attributed. These models and selected layers differ, so the maps compare the selected classifiers rather than isolate the causal effect of one architectural change.

Each nonnegative map is **bilinearly upsampled to 64×64**, with `align_corners=False`, and divided by **its own upsampled maximum**. Empty maps would remain zero, be transparent, and be labelled empty; none occurred in this selection. Every overlay uses **inferno**, range **0–1**, and per-pixel opacity **0.55 × normalized heat**. Zero heat leaves the grayscale unchanged. Raw grayscale uses the common **0–255** range, archive orientation, and no flips. A shared colorbar states the relative normalization: equal colors across models do **not** represent equal absolute attribution magnitude.

### Matching saved CUDA predictions

An initial five-row attempt was halted by its strict consistency check: B2's maximum absolute logit difference from the original batch-16 test pass was **0.00237560**, with a probability difference of **0.0000177622**. Predictions and checkpoint hashes were unchanged. The failed attempt is preserved in `part_b/outputs/b4_attempt_001/`; its selection is identical to the completed run. No images were redrawn, and comparison tolerances were not relaxed.

Matching the original inference batch sizes resolved the numerical difference. Attribution batches repeat **only the five selected images** to obtain 128 rows for B1 and 16 for B2/B3; the 123/11/11 duplicate padding maps are discarded. No other test images are fetched for padding. Evaluation-mode BN/dropout make the rows independent. The runner also handles a selected image from an original partial final batch by matching that batch's size. With this protocol, all **15 retained logits and probabilities match the original saved test pass exactly**. PyTorch documents that batched calculations can differ numerically from calculations with another batching arrangement; the precise CUDA kernel cause here was not isolated. [PyTorch 2.11 numerical-accuracy notes](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html).

## Original images and fifteen overlays

![Five original X-rays and B1/B2/B3 Grad-CAM overlays](../part_b/outputs/b4/figures/gradcam_comparison.png)

[Publication figure PDF](../part_b/outputs/b4/figures/gradcam_comparison.pdf). Each row is the same image across columns. Probability titles reuse the saved final-test results; four-decimal display rounding can show `1.0000`. Exact probabilities and all map diagnostics are in the [15-row CSV](../part_b/outputs/b4/tables/sample_predictions_and_maps.csv).

All models predict the assumed pneumonia class at the unchanged threshold **0.5** for all five selected images. Row **413** is least confident for every model: B1 **0.968064**, B2 **0.917330**, B3 **0.993003**. Agreement on a label does not imply agreement on the supporting image regions.

## Image-by-image visual comparison

Directions refer to the **displayed image**, using the original upper-left coordinate origin, rather than anatomical left/right. Observations are descriptive comparisons with visible grayscale structures; no lung masks, lesion outlines, or radiologist annotations are available.

| Test row | Visible grayscale context | B1, AlexNet-like | B2, VGG-like | B3, ResNet-like |
| --- | --- | --- | --- | --- |
| **486** | Broad bright central and lower image-right region, with darker ribbed regions toward the sides. | A narrow central vertical hotspot overlaps part of the bright region; its peak is **(34,30)**. It covers only a small portion of the visible brightness. | Strongest patches lie along the upper edge, peak **(14,6)**, with weak scattered central/lower activity. This emphasizes upper structures more than the broad central brightness. | Strong image-left lower boundary/bottom activity, peak **(12,60)**, with a secondary upper image-right patch. Much of the strongest activity is peripheral rather than centrally bright. |
| **279** | Bright central/lower region, dark ribbed sides, a small peripheral bright marking at upper image-left, and dark margins. | A compact lower central/image-right hotspot, peak **(38,49)**, overlaps the bright lower central region. | Peak **(18,6)** lies near the upper boundary; weaker lower central/image-right patches provide only partial overlap with the broad bright region. | Peak **(12,60)** lies near the lower image-left edge; a broader warm region extends toward the center. Peripheral and central contributions differ from B1's compact hotspot. |
| **413** | Clearly visible ribs with a bright central column and lighter lower areas. | A small central/image-right hotspot, peak **(37,38)**, lies within the bright central area. | The strongest patch is lower image-left, peak **(22,46)**, near the boundary between a darker ribbed region and lighter inferior tissue; weaker upper and image-right patches remain. | A broader central/image-left patch, peak **(27,35)**, overlaps part of the inner thoracic brightness. Its spatial extent is larger than B1's. |
| **53** | Diffuse light grayscale across much of the central/lower chest, with ribs still visible. | Concentrated lower central activity, peak **(33,45)**, overlaps a lighter area. | Several separated hotspots emphasize upper and image-right structures, peak **(49,26)**, including peripheral patches. The map does not uniformly cover the diffuse light region. | A central/image-left hotspot, peak **(27,28)**, spreads over a larger middle region, with weaker lower image-left activity. |
| **276** | Bright central/lower column and darker side regions containing visible ribs. | A narrow central/image-right hotspot, peak **(38,37)**, overlaps the bright central column. | Upper-edge patches dominate, peak **(45,5)**, with additional lower image-left and weak lower image-right patches; much of the central bright column is weakly highlighted. | A broad central/image-left hotspot, peak **(27,35)**, overlaps part of the bright central area, with weak peripheral lower activity. |

**B1 consistently concentrates positive evidence in small central or lower central regions** across these five draws. These hotspots overlap visible brightness, but central brightness can include the heart, bones, and other normal structures. The compact maps therefore do not prove that B1 found pneumonia lesions or captured all cloudy areas.

**B2 places more activity near upper boundaries and separates evidence into multiple patches.** Row 413 provides a clearer lower interior hotspot; the other four have conspicuous upper-edge activity. This pattern raises a question about peripheral structures or acquisition cues, but the overlays alone cannot identify a causal shortcut or show that the model ignores disease-relevant tissue.

**B3 produces broader, smoother regions, with image-left lower-boundary activity in rows 486/279 and more central activity in rows 413/53/276.** Its **8×8** native grid has fewer spatial cells than B1/B2's **16×16** grids; interpolation helps explain the coarser display. It cannot restore missing spatial resolution. B3's residual routes are not separately visualized, and broader coverage is not evidence of superior localization or predictive performance.

### Descriptive geometric check

As an auditable supplement, measure the fraction of each positive interpolated map's mass in the **outer eight-pixel frame**. This fixed frame occupies **43.75%** of the image pixels. The five-image mean is:

| Model | Mean CAM mass in outer eight-pixel frame | Empty positive maps |
| --- | ---: | ---: |
| B1 | **0.00%** | 0/5 |
| B2 | **36.62%** | 0/5 |
| B3 | **18.89%** | 0/5 |

[Geometric CSV](../part_b/outputs/b4/tables/geometric_diagnostics.csv). These values support the observed differences in the maps' spatial distributions. The frame includes anatomy as well as margins, is neither a lung nor lesion mask, and cannot measure background reliance or localization accuracy. It is not a model-selection criterion.

The five images contain only positive labels and all are predicted positive. They provide no evidence about Grad-CAM behavior on healthy images or false positives, which matter given the previously measured low test specificity. High probabilities and visually plausible overlap cannot establish anatomical faithfulness, diagnostic correctness, or a general architecture ranking. The supplied resolution is low, the semantic label mapping is assumed, patient identities are absent, and the supplied train/validation overlap remains documented in the [preparation report](PART_B_PREPARATION.md). No test-based retuning follows from these explanations.

## Verification and reproducible artifacts

The full suite passed **96 tests**, including eight B4 tests covering fixed positive-only sampling, analytic known-value gradient maps and ReLU, interpolation/max normalization/empty maps, image coordinates, model/BN/dropout state preservation, data/checkpoint guards, full/tail batch padding with row-order restoration, and overwrite prevention.

The independent [B4 auditor](../scripts/verify_b4.py) reconstructs the sample directly with NumPy, checks every selected raw and normalized pixel, loads the exact saved checkpoints on CUDA, and differentiates summed scalar logits using **`torch.autograd.grad` without Captum**. It then independently computes channel weights, signed and positive maps, bilinear interpolation, per-map normalization, probabilities, and CSV geometry. All compared tensors have **maximum absolute difference 0**. No parameter gradients accumulate, no BN statistics change, and source checkpoints remain unchanged. Saved test logits/probabilities match exactly, and all supplied/922 prior output hashes pass before and after the audit. It makes three selected-image gradient forwards with duplicate padding; it does **not** repeat full-test performance evaluation. [Verification record](../part_b/outputs/b4/verification.json), [audit mirror](../artifacts/b4_checks/latest.json).

The saved [tensor artifact](../part_b/outputs/b4/gradcam.pt) contains row IDs, labels, original pixels, normalized inputs, activations, gradients, channel weights, signed/positive native maps, upsampled/normalized maps, peaks, logits/probabilities, and state/protocol evidence. [Summary](../part_b/outputs/b4/summary.json), [selected-image CSV](../part_b/outputs/b4/tables/selected_images.csv), and [generated method/numerical draft](../part_b/outputs/b4/analysis.md) support report assembly.

From the workspace root, use the project virtual environment:

```powershell
# Recompute explanations into a NEW directory; source experiments are preserved.
.\.venv\Scripts\python.exe -B -u part_b\scripts\run_b4.py --output part_b\outputs\b4_reproduction
.\.venv\Scripts\python.exe -B -u scripts\verify_b4.py --output part_b\outputs\b4_reproduction

# Regenerate the existing figures/CSVs directly from saved maps, with no inference.
.\.venv\Scripts\python.exe -B part_b\scripts\run_b4.py --export-only
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Existing output directories are protected against recomputing/overwriting explanations. Optional `--data`, `--b1`, and `--b2`/`--b3` arguments identify matching saved source artifacts. All packages, including Captum 0.9.0 and PyTorch 2.11.0+cu128, remain in the root `.venv`; no additional packages were needed. Both concise five-page reports are now complete. Next: **step 15, verify and package the submission**.
