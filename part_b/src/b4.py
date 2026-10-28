"""Fixed five-image Grad-CAM comparison of the saved B1/B2/B3 checkpoints."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import platform
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "artifacts/cache/matplotlib"))

import captum
from captum.attr import LayerAttribution, LayerGradCam
import numpy as np
import torch

from assignment_common import restore_checkpoint, seed_everything
from assignment_common.artifacts import write_checkpoint, write_json
from part_b.src.b1 import state_fingerprint
from part_b.src.data import CLASS_MAPPING, load_data_bundle, sha256_file
from part_b.src.models import EXPECTED_PARAMETERS, build_model


SAMPLE_SEED = 42
SAMPLE_COUNT = 5
TARGET_LAYERS = {"b1": "features.conv3.conv", "b2": "features.conv6.conv", "b3": "features.block64.conv2.conv"}
GRID_SIZES = {"b1": 16, "b2": 16, "b3": 8}
DISPLAY = {"colormap": "inferno", "max_opacity": .55,
    "opacity_rule": "0.55 * per-map normalized positive Grad-CAM",
    "normalization": "divide each nonnegative upsampled map by its own maximum; zero maps remain zero",
    "grayscale_range": [0, 255], "heatmap_range": [0, 1], "orientation": "original archive orientation; no flips",
    "interpolation": "bilinear to 64x64; align_corners=False (PyTorch default)",
    "probability_annotations": "saved final-test-pass probabilities; gradients target the unrestricted logit"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sample_positive_rows(labels, row_ids, *, seed=SAMPLE_SEED, count=SAMPLE_COUNT):
    labels = torch.as_tensor(labels).cpu()
    row_ids = torch.as_tensor(row_ids).cpu()
    if (labels.ndim != 1 or row_ids.shape != labels.shape or not len(labels)
            or row_ids.dtype != torch.long or len(torch.unique(row_ids)) != len(row_ids)
            or not torch.isfinite(labels).all() or not ((labels == 0) | (labels == 1)).all()
            or type(count) is not int or count < 1 or type(seed) is not int or seed < 0):
        raise ValueError("Sampling requires binary labels and unique original row IDs")
    eligible = torch.where(labels == 1)[0].numpy()
    if len(eligible) < count:
        raise ValueError("Too few positive-class images for sampling without replacement")
    positions = np.random.default_rng(seed).choice(eligible, size=count, replace=False)
    return {"seed": seed, "rng": "numpy.random.default_rng / PCG64", "sample_count": count,
        "eligible_positive_rows": len(eligible), "sampling": "uniform without replacement among label-1 test rows",
        "positions": positions.tolist(), "row_ids": row_ids[positions].tolist(), "labels": labels[positions].long().tolist(),
        "display_order": "random draw order, shared by all models",
        "representative_interpretation": "random examples of the supplied positive-class partition; no visual, prediction, or map filtering"}


def resize_and_normalize(positive_maps, *, size=(64, 64)):
    if (positive_maps.ndim != 4 or positive_maps.shape[1] != 1 or not len(positive_maps)
            or not torch.isfinite(positive_maps).all() or (positive_maps < 0).any()):
        raise ValueError("Grad-CAM display requires finite nonnegative N x 1 x H x W maps")
    upsampled = LayerAttribution.interpolate(positive_maps, size, interpolate_mode="bilinear")
    peaks = upsampled.flatten(1).amax(dim=1).reshape(-1, 1, 1, 1)
    normalized = upsampled / torch.where(peaks > 0, peaks, torch.ones_like(peaks))
    return upsampled, normalized, peaks.flatten()


def explain_batch(model, inputs, layer, *, inference_batch_size=None):
    """Capture one Captum forward/backward, plus same-pass manual CAM evidence.

    Models return one logit per row, so target=None is the positive-class logit.
    All module train/eval flags are restored; no optimizer step or parameter-grad
    accumulation is performed. Activations are taken before the following BN.
    """
    if inputs.ndim != 4 or not len(inputs) or not torch.isfinite(inputs).all():
        raise ValueError("Expected finite nonempty image batches")
    count = len(inputs)
    forward_size = count if inference_batch_size is None else inference_batch_size
    if type(forward_size) is not int or forward_size < count:
        raise ValueError("Inference padding batch must hold every selected image")
    modes = {module: module.training for module in model.modules()}
    activations, gradients, logits, handles, gradient_handles = [], [], [], [], []
    def layer_hook(module, values, output):
        activations.append(output.detach().clone())
        gradient_handles.append(output.register_hook(lambda gradient: gradients.append(gradient.detach().clone())))
    def output_hook(module, values, output):
        logits.append(output.detach().clone())
    handles.append(layer.register_forward_hook(layer_hook))
    handles.append(model.register_forward_hook(output_hook))
    model.eval()
    try:
        with torch.enable_grad():
            images = inputs.detach().clone().requires_grad_(True)
            if forward_size > count:
                images = images.repeat((forward_size + count - 1) // count, 1, 1, 1)[:forward_size]
            positive = LayerGradCam(model, layer).attribute(images, target=None,
                attribute_to_layer_input=False, relu_attributions=True)
        if len(activations) != 1 or len(gradients) != 1 or len(logits) != 1 or logits[0].shape != (forward_size,):
            raise RuntimeError("Expected one scalar-logit Captum forward/backward per image batch")
        activations[0], gradients[0], logits[0], positive = (value[:count] for value in
            (activations[0], gradients[0], logits[0], positive))
        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        signed = (weights * activations[0]).sum(dim=1, keepdim=True)
        manual_positive = torch.relu(signed)
        torch.testing.assert_close(positive, manual_positive, rtol=1e-6, atol=1e-7)
        positive = positive.detach()
        full, normalized, peaks = resize_and_normalize(positive)
        result = {"activations": activations[0], "gradients": gradients[0], "channel_weights": weights,
            "signed_coarse_maps": signed, "positive_coarse_maps": positive, "upsampled_positive_maps": full,
            "normalized_maps": normalized, "upsampled_peaks": peaks,
            "logits": logits[0], "probabilities": torch.sigmoid(logits[0]),
            "same_pass_manual_positive_max_difference": float((positive - manual_positive).abs().max()),
            "captum_forward_calls": 1, "forward_batch_size": forward_size, "padding_rows": forward_size - count,
            "target": "positive-class unrestricted binary logit", "captum_target": None,
            "relu_attributions": True, "attribute_to_layer_input": False}
        return {key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value for key, value in result.items()}
    finally:
        for handle in handles + gradient_handles:
            handle.remove()
        for module, training in modes.items():
            module.training = training


def explain_selected(model, inputs, layer, positions, *, test_count, inference_batch_size):
    """Match saved inference batch shapes using duplicates of selected images only.

    Eval-mode BN/dropout make rows independent. Matching the original full/tail
    batch shape avoids cuDNN numerical differences without reevaluating test rows
    outside the fixed selection. Duplicate padding maps are discarded.
    """
    if (not len(inputs) or len(positions) != len(inputs) or type(inference_batch_size) is not int or inference_batch_size < 1
            or any(type(position) is not int or not 0 <= position < test_count for position in positions)):
        raise ValueError("Original positions and source inference batch size must be valid")
    groups = {}
    for index, position in enumerate(positions):
        size = min(inference_batch_size, test_count - (position // inference_batch_size) * inference_batch_size)
        groups.setdefault(size, []).append(index)
    results, group_order = [], []
    for size, indices in groups.items():
        results.append(explain_batch(model, inputs[indices], layer, inference_batch_size=size))
        group_order.extend(indices)
    restore_order = torch.tensor(group_order).argsort()
    merged = {key: torch.cat([result[key] for result in results])[restore_order]
              if isinstance(value, torch.Tensor) else value for key, value in results[0].items()}
    merged.update(captum_forward_calls=len(results), forward_batch_sizes=[result["forward_batch_size"] for result in results],
        padding_rows=sum(result["padding_rows"] for result in results),
        same_pass_manual_positive_max_difference=max(result["same_pass_manual_positive_max_difference"] for result in results))
    merged.pop("forward_batch_size")
    return merged


def validate_frozen(frozen, architecture, *, source_sha256, dataset_sha256, preprocessing_sha256):
    expected = {"format_version": 1, "architecture": architecture, "trainable_parameters": EXPECTED_PARAMETERS[architecture],
        "source_sha256": source_sha256, "prepared_dataset_sha256": dataset_sha256,
        "prepared_preprocessing_sha256": preprocessing_sha256, "probability_threshold": .5, "refit": True,
        "class_mapping": {str(key): value for key, value in CLASS_MAPPING.items()},
        "class_mapping_status": "confirmed_by_instructor_QA"}
    if any(frozen.get(key) != value for key, value in expected.items()):
        raise ValueError("B4 requires the recorded architecture, class assumption, checkpoint, and prepared data lineage")
    if (not frozen.get("final_checkpoint") or not frozen.get("final_checkpoint_sha256")
            or frozen.get("final_epoch") != frozen.get("selected_epoch")
            or frozen.get("final_checkpoint_sha256") == frozen.get("checkpoint_sha256")):
        raise ValueError("B4 requires the fresh final refit checkpoint, separate from selection")


def run_b4(output_directory, prepared_directory, model_directories, *, seed=SAMPLE_SEED):
    output, prepared = Path(output_directory).resolve(), Path(prepared_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new B4 output directory: {output}")
    if set(model_directories) != set(TARGET_LAYERS):
        raise ValueError("B4 requires all three saved models")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this B4 experiment")
    torch.set_num_threads(4)
    bundle = load_data_bundle(prepared)
    dataset_hash, preprocessing_hash = sha256_file(prepared / "datasets.pt"), sha256_file(prepared / "preprocessing.pt")
    sources = {}
    for architecture, path in model_directories.items():
        directory = Path(path).resolve()
        frozen_path = directory / f"frozen_{architecture}_settings.json"
        frozen = read_json(frozen_path)
        validate_frozen(frozen, architecture, source_sha256=bundle.source_sha256,
            dataset_sha256=dataset_hash, preprocessing_sha256=preprocessing_hash)
        if (read_json(directory / "status.json")["status"] != "completed"
                or read_json(directory / "verification.json")["status"] != "passed"
                or sha256_file(directory / frozen["final_checkpoint"]) != frozen["final_checkpoint_sha256"]):
            raise ValueError("B4 requires completed, verified source checkpoints")
        sources[architecture] = {"directory": str(directory), "frozen_settings": frozen,
            "frozen_settings_sha256": sha256_file(frozen_path), "checkpoint": frozen["final_checkpoint"],
            "checkpoint_sha256": frozen["final_checkpoint_sha256"], "target_layer": TARGET_LAYERS[architecture],
            "final_test_predictions_sha256": sha256_file(directory / "final_test_predictions.pt"),
            "final_test_sha256": sha256_file(directory / "final_test.json")}
    output.mkdir(parents=True, exist_ok=False)
    selection = sample_positive_rows(bundle.datasets["test"].tensors[1], bundle.row_ids["test"], seed=seed)
    selection.update(source_sha256=bundle.source_sha256, partition="test", class_mapping=dict(CLASS_MAPPING),
        class_mapping_status="confirmed_by_instructor_QA", selection_frozen_before_explanations=True)
    write_json(output / "selection.json", selection)
    configuration = {"created_at_singapore": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "seed": seed, "prepared_directory": str(prepared), "source_sha256": bundle.source_sha256,
        "prepared_dataset_sha256": dataset_hash, "prepared_preprocessing_sha256": preprocessing_hash,
        "models": sources, "display": DISPLAY, "method": "captum.attr.LayerGradCam",
        "target": "positive-class unrestricted binary logit", "captum_target": None,
        "relu_attributions": True, "attribute_to_layer_input": False, "models_in_eval_mode": True,
        "gradients_enabled": True, "sample_count": 5, "performance_results_reused": True,
        "numerical_batch_protocol": "match each selected row's original final-inference full/tail batch size using duplicates of selected inputs only; discard padding maps",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "captum": captum.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()}}
    write_json(output / "configuration.json", configuration)
    write_json(output / "status.json", {"status": "running", "stage": "explanations"})
    positions = torch.tensor(selection["positions"], dtype=torch.long)
    images, labels = (tensor[positions] for tensor in bundle.datasets["test"].tensors)
    payload = {"format_version": 1, "source_sha256": bundle.source_sha256,
        "row_ids": bundle.row_ids["test"][positions], "labels": labels,
        "raw_images": bundle.raw_images["test"][positions], "normalized_inputs": images,
        "selection": selection, "display": DISPLAY, "models": {}}
    try:
        for architecture, source in sources.items():
            seed_everything(seed)
            model = build_model(architecture)
            checkpoint = Path(source["directory"]) / source["checkpoint"]
            restored = restore_checkpoint(checkpoint, model, device="cuda")
            if restored["model_config"] != model.metadata() or restored["preprocessing"] != bundle.preprocessing.metadata():
                raise ValueError("Selected checkpoint architecture/preprocessing metadata must match")
            actual_layer = next(name for name, layer in model.named_modules() if layer is model.last_convolution)
            if actual_layer != TARGET_LAYERS[architecture]:
                raise ValueError("Grad-CAM must target the final main-path convolution")
            before = state_fingerprint(model)
            source_batch_size = source["frozen_settings"].get("b1_batch_size", source["frozen_settings"].get("selected_batch_size"))
            result = explain_selected(model, images.to("cuda"), model.last_convolution, selection["positions"],
                test_count=len(bundle.datasets["test"]), inference_batch_size=source_batch_size)
            expected_shape = (5, 1, GRID_SIZES[architecture], GRID_SIZES[architecture])
            if result["positive_coarse_maps"].shape != expected_shape:
                raise ValueError("Grad-CAM native spatial resolution disagrees with Figure B")
            saved_predictions = torch.load(Path(source["directory"]) / "final_test_predictions.pt", map_location="cpu", weights_only=True)
            if saved_predictions["checkpoint_sha256"] != source["checkpoint_sha256"] or saved_predictions["source_sha256"] != bundle.source_sha256:
                raise ValueError("Source final predictions must belong to the selected checkpoint/data")
            torch.testing.assert_close(saved_predictions["row_ids"][positions], payload["row_ids"], rtol=0, atol=0)
            torch.testing.assert_close(saved_predictions["labels"][positions], labels, rtol=0, atol=0)
            expected_logits, expected_probabilities = saved_predictions["logits"][positions], saved_predictions["probabilities"][positions]
            torch.testing.assert_close(result["logits"], expected_logits, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(result["probabilities"], expected_probabilities, rtol=1e-6, atol=2e-7)
            if not torch.equal(result["probabilities"] >= .5, expected_probabilities >= .5):
                raise ValueError("XAI forward must preserve the recorded predictions")
            if before != state_fingerprint(model) or any(parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError("Explanation must preserve model state and avoid accumulated parameter gradients")
            if sha256_file(checkpoint) != source["checkpoint_sha256"]:
                raise RuntimeError("Source checkpoint changed")
            result.update(target_layer=actual_layer, architecture=architecture, checkpoint_sha256=source["checkpoint_sha256"],
                native_grid_size=GRID_SIZES[architecture], saved_final_test_logits=expected_logits,
                saved_final_test_probabilities=expected_probabilities,
                maximum_probability_difference_from_saved_test=float((result["probabilities"]-expected_probabilities).abs().max()),
                model_state_sha256_before=before, model_state_sha256_after=state_fingerprint(model), model_state_unchanged=True)
            payload["models"][architecture] = result
            print(f"{architecture.upper()}: five {GRID_SIZES[architecture]}x{GRID_SIZES[architecture]} positive Grad-CAM maps; "
                  f"manual difference {result['same_pass_manual_positive_max_difference']:.3g}; "
                  f"empty maps {int((result['upsampled_peaks'] == 0).sum())}", flush=True)
            del model
        write_checkpoint(output / "gradcam.pt", payload)
        write_json(output / "summary.json", {"status": "completed", "sample_seed": seed,
            "sample_row_ids": selection["row_ids"], "sample_count": 5, "models": 3, "overlay_count": 15,
            "native_grid_sizes": GRID_SIZES, "target_layers": TARGET_LAYERS, "display": DISPLAY,
            "attribution_artifact": "gradcam.pt", "attribution_sha256": sha256_file(output / "gradcam.pt"),
            "source_sha256": bundle.source_sha256, "selection_frozen_before_explanations": True,
            "models_and_source_checkpoints_unchanged": True, "performance_results_reused": True,
            "full_test_performance_evaluations": 0,
            "zero_map_counts": {key: int((value["upsampled_peaks"] == 0).sum()) for key, value in payload["models"].items()}})
        from part_b.src.b4_exports import export_b4
        export_b4(output)
        write_json(output / "status.json", {"status": "completed", "stage": "completed"})
        print(f"B4 complete: rows {selection['row_ids']}; 15 overlays from the three frozen checkpoints", flush=True)
        return read_json(output / "summary.json")
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise
