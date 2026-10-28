"""Render saved Grad-CAM tensors with a shared display protocol and no inference."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "artifacts/cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import pandas as pd
import torch

from part_b.src.b4 import DISPLAY, TARGET_LAYERS


def map_diagnostics(full_map, normalized_map, *, border_width=8):
    """Describe geometry only: the outer frame is not an anatomical mask."""
    full = full_map.squeeze().double()
    normalized = normalized_map.squeeze()
    if full.shape != (64, 64) or not torch.isfinite(full).all() or (full < 0).any():
        raise ValueError("Diagnostics require a finite nonnegative 64x64 map")
    total = float(full.sum())
    border = torch.ones_like(full, dtype=torch.bool)
    border[border_width:-border_width, border_width:-border_width] = False
    if total == 0:
        return {"empty_positive_map": True, "peak_x": None, "peak_y": None,
                "weighted_center_x": None, "weighted_center_y": None, "outer_8_pixel_frame_mass_fraction": None}
    peak = int(normalized.flatten().argmax())
    ys, xs = torch.meshgrid(torch.arange(64, dtype=torch.float64), torch.arange(64, dtype=torch.float64), indexing="ij")
    return {"empty_positive_map": False, "peak_x": peak % 64, "peak_y": peak // 64,
        "weighted_center_x": float((xs * full).sum()) / total, "weighted_center_y": float((ys * full).sum()) / total,
        "outer_8_pixel_frame_mass_fraction": float(full[border].sum()) / total}


def export_b4(directory):
    output = Path(directory)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "completed":
        raise ValueError("B4 exports require completed attribution results")
    payload = torch.load(output / "gradcam.pt", map_location="cpu", weights_only=True)
    if payload["display"] != DISPLAY or payload["row_ids"].tolist() != summary["sample_row_ids"]:
        raise ValueError("Saved IDs and display protocol must match")
    figures, tables = output / "figures", output / "tables"
    figures.mkdir(exist_ok=True); tables.mkdir(exist_ok=True)
    rows = []
    for index, row_id in enumerate(payload["row_ids"].tolist()):
        for architecture in TARGET_LAYERS:
            result = payload["models"][architecture]
            probability = float(result["saved_final_test_probabilities"][index])
            rows.append({"row_id": row_id, "true_label": 1, "architecture": architecture,
                "target_layer": result["target_layer"], "native_grid_size": result["native_grid_size"],
                "saved_final_test_probability": probability, "saved_predicted_label": int(probability >= .5),
                "gradcam_forward_probability": float(result["probabilities"][index]),
                "upsampled_positive_max": float(result["upsampled_peaks"][index]),
                **map_diagnostics(result["upsampled_positive_maps"][index], result["normalized_maps"][index])})
    frame = pd.DataFrame(rows)
    frame.to_csv(tables / "sample_predictions_and_maps.csv", index=False)
    pd.DataFrame([{"row_id": int(row), "label": int(label), "sampling_seed": summary["sample_seed"]}
                  for row, label in zip(payload["row_ids"], payload["labels"])]).to_csv(tables / "selected_images.csv", index=False)
    model_diagnostics = frame.groupby("architecture", sort=False).agg(
        sampled_images=("row_id", "count"), empty_positive_maps=("empty_positive_map", "sum"),
        mean_outer_8_pixel_frame_mass_fraction=("outer_8_pixel_frame_mass_fraction", "mean"))
    model_diagnostics.to_csv(tables / "geometric_diagnostics.csv")
    figure, axes = plt.subplots(5, 4, figsize=(10.5, 13.4))
    for index, row_id in enumerate(payload["row_ids"].tolist()):
        raw = payload["raw_images"][index].numpy()
        axes[index, 0].imshow(raw, cmap="gray", vmin=0, vmax=255, interpolation="nearest", origin="upper")
        axes[index, 0].set_title(f"Test row {row_id}\nLabel 1 (pneumonia)", fontsize=8)
        for column, architecture in enumerate(TARGET_LAYERS, 1):
            result = payload["models"][architecture]
            heat = result["normalized_maps"][index, 0].numpy()
            probability = float(result["saved_final_test_probabilities"][index])
            axes[index, column].imshow(raw, cmap="gray", vmin=0, vmax=255, interpolation="nearest", origin="upper")
            axes[index, column].imshow(heat, cmap=DISPLAY["colormap"], vmin=0, vmax=1,
                alpha=DISPLAY["max_opacity"] * heat, interpolation="nearest", origin="upper")
            title = f"{architecture.upper()} | p={probability:.4f}\n" + ("pred. pneumonia" if probability >= .5 else "pred. healthy")
            if float(result["upsampled_peaks"][index]) == 0:
                title += " | empty CAM"
            axes[index, column].set_title(title, fontsize=8)
        for axis in axes[index]:
            axis.axis("off")
    figure.suptitle("B4: same five positive-class test images, three fresh final-refit checkpoints\n"
                   f"Seed {summary['sample_seed']} | pneumonia-logit Grad-CAM | original grayscale at left", fontsize=11, y=.995)
    figure.subplots_adjust(left=.015, right=.91, top=.91, bottom=.075, wspace=.055, hspace=.22)
    color_axis = figure.add_axes([.932, .25, .018, .45])
    figure.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=DISPLAY["colormap"]), cax=color_axis,
                   label="Relative positive CAM (each map / its own maximum)")
    figure.text(.465, .025, "B1/B2 native maps: 16×16; B3: 8×8. Bilinear upsampling to 64×64.\n"
                "Shared color/opacity rule; intensities are not comparable absolute attribution across models.\n"
                "p = saved final-test probability. No lesion annotations; maps do not establish correct localization.",
                ha="center", va="bottom", fontsize=8)
    figure.savefig(figures / "gradcam_comparison.png", dpi=240, bbox_inches="tight")
    figure.savefig(figures / "gradcam_comparison.pdf", bbox_inches="tight")
    plt.close(figure)
    table = "\n".join(f"| {row['row_id']} | {row['architecture'].upper()} | {row['saved_final_test_probability']:.6f} | "
        f"{row['native_grid_size']}×{row['native_grid_size']} | {row['peak_x']}, {row['peak_y']} | "
        f"{row['outer_8_pixel_frame_mass_fraction']:.2%} |" if row['outer_8_pixel_frame_mass_fraction'] is not None
        else f"| {row['row_id']} | {row['architecture'].upper()} | {row['saved_final_test_probability']:.6f} | "
             f"{row['native_grid_size']}×{row['native_grid_size']} | empty | n/a |" for row in rows)
    text = f"""# B4 Grad-CAM method and numerical comparison draft

Selected test row IDs **{summary['sample_row_ids']}**, seed **{summary['sample_seed']}**, uniformly without replacement among all 390 label-1 test rows. Indices were saved before explanations, and all three models use these same five images in the same order. Q&A.md confirms class 1 is pneumonia; the archive itself has no semantic class-name metadata. “Representative” means random examples of the supplied positive-class partition, without choosing by appearance, predictions, or map quality.

The exact fresh combined-data final B1/B2/B3 checkpoints use final main-path convolutions `{TARGET_LAYERS['b1']}`, `{TARGET_LAYERS['b2']}`, and `{TARGET_LAYERS['b3']}`. Models remain in evaluation mode with gradients enabled. Captum `LayerGradCam.attribute` targets the unrestricted positive-class logit (`target=None` for scalar-per-row outputs), layer output before its BN/ReLU, with `relu_attributions=True`. Averaged spatial gradients weight each activation channel; their weighted sum is followed by ReLU. Native grids are 16×16, 16×16, and 8×8. This is coarse localization via the selected layer, including only B3's selected main convolution rather than a separate map of its shortcut route. [Captum documentation](https://captum.ai/api/layer.html#captum.attr.LayerGradCam), [Selvaraju et al. Grad-CAM paper](https://arxiv.org/abs/1610.02391).

Positive maps are bilinearly interpolated to 64×64 and divided by their own maximum. Zero maps remain zero and have transparent overlays. Originals retain uint8 grayscale range 0–255 and archive orientation. All overlays use inferno, heat range 0–1, and alpha=0.55×normalized heat. Equal colors across panels describe relative within-map importance, not equal absolute attribution. Titles reuse saved final-test probabilities. No full-test performance is recomputed or used for tuning.

To match the original CUDA inference shapes, each attribution batch is padded by repeating only the selected images: B1 uses 128 rows and B2/B3 use 16 rows. The five retained rows' logits and probabilities match their saved final-test values exactly; all padding maps are discarded. Evaluation-mode BN/dropout keep rows independent. The runner also handles a selected image originating from a partial final batch. The historical pre-FAQ five-row attempt, archived separately and excluded from the current code package, records a batch-shape consistency failure for the earlier model protocol. The current explanation uses fresh final-refit models, the same preselected rows and source inference batch sizes. Tolerances were not relaxed. PyTorch documents that floating-point results can vary between batch shapes; the exact kernel cause was not isolated. [PyTorch numerical-accuracy notes](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html).

![Five originals and fifteen Grad-CAM overlays](figures/gradcam_comparison.png)

| Test row | Model | Saved pneumonia probability | Native grid | Peak (x,y), displayed image | CAM mass in outer 8-pixel frame |
| --- | --- | ---: | --- | --- | ---: |
{table}

The geometric statistic counts positive interpolated CAM mass within the image's outer eight-pixel frame (43.75% of pixels). It is not a lung or lesion mask, and it cannot measure disease localization or establish a causal reliance on borders. Peak coordinates use the archive's displayed upper-left origin and are not anatomical laterality labels. Raw signed/positive maps, activations, gradients, weights, original pixels, row IDs, and saved/attribution probabilities are retained in `gradcam.pt`; CSVs preserve all fifteen map records.

Compare highlighted regions with visible grayscale opacity in the original panels, while recognizing that white regions may include bones and other structures. Visual overlap with bright/cloudy areas is qualitative, not a diagnosis or lesion-localization validation. Smaller native grids and interpolation can broaden or smooth hotspots. Saturated probability does not guarantee anatomical faithfulness. No lesion/lung annotations or patient identities are provided, and these five examples do not establish performance or a general architecture ranking. See the separately written B4 experiment report for image-by-image visual observations.
"""
    (output / "analysis.md").write_text(text, encoding="utf-8")
