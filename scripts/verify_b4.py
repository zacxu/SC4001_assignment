"""Independently audit saved B4 maps using PyTorch autograd, without Captum."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/cache/matplotlib"))

import numpy as np
import torch
import torch.nn.functional as functional

from assignment_common import restore_checkpoint, seed_everything
from assignment_common.artifacts import json_value, write_json
from part_b.src.b1 import state_fingerprint
from part_b.src.data import CLASS_MAPPING, load_data_bundle, sha256_file
from part_b.src.models import EXPECTED_PARAMETERS, build_model


LAYERS = {"b1": "features.conv3.conv", "b2": "features.conv6.conv", "b3": "features.block64.conv2.conv"}
GRIDS = {"b1": 16, "b2": 16, "b3": 8}
CHANNELS = {"b1": 64, "b2": 32, "b3": 64}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_previous(previous):
    for relative, fingerprint in previous.items():
        assert sha256_file(ROOT / relative) == fingerprint, relative


def independent_maps(model, inputs, positions, *, test_count, batch_size, layer_name):
    """Differentiate the summed scalar logits and explicitly reconstruct Grad-CAM.

    Recreate original full/tail batch shapes using only duplicates of the selected
    images. No Captum calls or B4 attribution helpers are used in this audit.
    """
    groups = {}
    for index, position in enumerate(positions):
        remaining = test_count - batch_size * (position // batch_size)
        groups.setdefault(min(batch_size, remaining), []).append(index)
    all_results, order = [], []
    for forward_size, indices in groups.items():
        images = inputs[indices].to("cuda")
        images = images.repeat(math.ceil(forward_size / len(indices)), 1, 1, 1)[:forward_size]
        images.requires_grad_(True)
        captured = []
        handle = dict(model.named_modules())[layer_name].register_forward_hook(
            lambda module, arguments, output: captured.append(output))
        try:
            logits = model(images)
            assert len(captured) == 1 and logits.shape == (forward_size,)
            gradients = torch.autograd.grad(logits.sum(), captured[0])[0]
            count = len(indices)
            activations, gradients = captured[0][:count], gradients[:count]
            weights = gradients.mean((2, 3), keepdim=True)
            signed = torch.sum(weights * activations, dim=1, keepdim=True)
            positive = signed.clamp_min(0)
            full = functional.interpolate(positive, size=(64, 64), mode="bilinear", align_corners=False)
            peaks = full.flatten(1).amax(1)
            normalized = full / peaks.clamp_min(torch.finfo(full.dtype).tiny).reshape(-1, 1, 1, 1)
            result = {"activations": activations, "gradients": gradients, "channel_weights": weights,
                "signed_coarse_maps": signed, "positive_coarse_maps": positive,
                "upsampled_positive_maps": full, "normalized_maps": normalized, "upsampled_peaks": peaks,
                "logits": logits[:count], "probabilities": logits[:count].sigmoid()}
            all_results.append({key: value.detach().cpu() for key, value in result.items()})
            order.extend(indices)
        finally:
            handle.remove()
    permutation = torch.tensor(order).argsort()
    return {key: torch.cat([result[key] for result in all_results])[permutation]
            for key in all_results[0]}, list(groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "part_b/outputs/b4")
    output = parser.parse_args().output.resolve()
    if read_json(output / "configuration.json")["models"]["b1"]["frozen_settings"].get("refit"):
        from scripts.faq_saved_checks import audit_gradcam
        audit_gradcam(output)
        return
    config, summary, selection = (read_json(output / filename) for filename in
                                 ("configuration.json", "summary.json", "selection.json"))
    assert summary["status"] == read_json(output / "status.json")["status"] == "completed"
    assert summary["sample_count"] == config["sample_count"] == 5
    assert summary["models"] == 3 and summary["overlay_count"] == 15
    assert summary["full_test_performance_evaluations"] == 0
    assert summary["performance_results_reused"] and config["performance_results_reused"]
    assert config["method"] == "captum.attr.LayerGradCam" and config["captum_target"] is None
    assert config["target"] == "positive-class unrestricted binary logit"
    assert config["relu_attributions"] and not config["attribute_to_layer_input"]
    assert config["models_in_eval_mode"] and config["gradients_enabled"]
    assert "duplicates of selected inputs only" in config["numerical_batch_protocol"]
    assert summary["target_layers"] == LAYERS and summary["native_grid_sizes"] == GRIDS
    assert sha256_file(output / "gradcam.pt") == summary["attribution_sha256"]
    payload = torch.load(output / "gradcam.pt", map_location="cpu", weights_only=True)
    assert payload["format_version"] == 1 and json_value(payload["selection"]) == selection
    assert payload["display"] == summary["display"] == config["display"]
    display = payload["display"]
    assert display["colormap"] == "inferno" and display["max_opacity"] == .55
    assert display["grayscale_range"] == [0, 255] and display["heatmap_range"] == [0, 1]
    assert "own maximum" in display["normalization"] and "no flips" in display["orientation"]
    prepared = Path(config["prepared_directory"])
    bundle = load_data_bundle(prepared)
    assert sha256_file(prepared / "datasets.pt") == config["prepared_dataset_sha256"]
    assert sha256_file(prepared / "preprocessing.pt") == config["prepared_preprocessing_sha256"]
    assert config["source_sha256"] == summary["source_sha256"] == payload["source_sha256"] == bundle.source_sha256
    assert selection["class_mapping"] == {str(key): value for key, value in CLASS_MAPPING.items()}
    assert selection["class_mapping_status"] == "assumed_not_archive_metadata"
    labels = bundle.datasets["test"].tensors[1]
    eligible = torch.where(labels == 1)[0].numpy()
    positions = np.random.default_rng(config["seed"]).choice(eligible, size=5, replace=False).tolist()
    assert config["seed"] == summary["sample_seed"] == selection["seed"] == 42
    assert len(eligible) == selection["eligible_positive_rows"] == 390
    assert positions == selection["positions"] and len(set(positions)) == 5
    assert selection["selection_frozen_before_explanations"] and summary["selection_frozen_before_explanations"]
    assert (output / "selection.json").stat().st_mtime_ns <= (output / "gradcam.pt").stat().st_mtime_ns
    for name, expected in (("row_ids", bundle.row_ids["test"][positions]), ("labels", labels[positions]),
        ("raw_images", bundle.raw_images["test"][positions]),
        ("normalized_inputs", bundle.datasets["test"].tensors[0][positions])):
        assert torch.equal(payload[name], expected), name
    row_ids = payload["row_ids"].tolist()
    assert row_ids == selection["row_ids"] == summary["sample_row_ids"] == [486, 279, 413, 53, 276]
    assert torch.equal(payload["labels"], torch.ones(5))
    failed_selection = ROOT / "part_b/outputs/b4_attempt_001/selection.json"
    if failed_selection.is_file():
        assert read_json(failed_selection) == selection
    previous = read_json(ROOT / "artifacts/b4_checks/before.json")["previous_output_hashes"]
    check_previous(previous)
    supplied = read_json(ROOT / "artifacts/setup/environment_check.json")["files"]
    for relative, recorded in supplied.items():
        assert sha256_file(ROOT / relative) == recorded["sha256"], relative
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to independently reproduce B4 gradients")
    torch.set_num_threads(4)
    model_checks = {}
    table_rows = list(csv.DictReader((output / "tables/sample_predictions_and_maps.csv").open(encoding="utf-8", newline="")))
    assert len(table_rows) == 15
    for architecture, layer_name in LAYERS.items():
        source, saved = config["models"][architecture], payload["models"][architecture]
        directory = Path(source["directory"])
        frozen_path = directory / f"frozen_{architecture}_settings.json"
        frozen = read_json(frozen_path)
        assert frozen == source["frozen_settings"] and sha256_file(frozen_path) == source["frozen_settings_sha256"]
        assert frozen["architecture"] == saved["architecture"] == architecture
        assert frozen["trainable_parameters"] == EXPECTED_PARAMETERS[architecture]
        for key in ("source_sha256", "prepared_dataset_sha256", "prepared_preprocessing_sha256"):
            assert frozen[key] == config[key]
        assert not frozen["refit"] and frozen["probability_threshold"] == .5
        assert frozen["class_mapping"] == selection["class_mapping"]
        assert frozen["class_mapping_status"] == selection["class_mapping_status"]
        assert read_json(directory / "status.json")["status"] == "completed"
        assert read_json(directory / "verification.json")["status"] == "passed"
        checkpoint = directory / source["checkpoint"]
        assert source["checkpoint"] == frozen["checkpoint"]
        assert sha256_file(checkpoint) == frozen["checkpoint_sha256"] == source["checkpoint_sha256"] == saved["checkpoint_sha256"]
        assert sha256_file(directory / "final_test_predictions.pt") == source["final_test_predictions_sha256"]
        assert sha256_file(directory / "final_test.json") == source["final_test_sha256"]
        original = torch.load(directory / "final_test_predictions.pt", weights_only=True)
        assert original["checkpoint_sha256"] == source["checkpoint_sha256"] and original["source_sha256"] == bundle.source_sha256
        assert torch.equal(original["row_ids"][positions], payload["row_ids"])
        assert torch.equal(original["labels"][positions], payload["labels"])
        assert torch.equal(original["logits"][positions], saved["saved_final_test_logits"])
        assert torch.equal(original["probabilities"][positions], saved["saved_final_test_probabilities"])
        assert torch.equal(saved["logits"], saved["saved_final_test_logits"])
        assert torch.equal(saved["probabilities"], saved["saved_final_test_probabilities"])
        assert saved["maximum_probability_difference_from_saved_test"] == 0
        seed_everything(42)
        model = build_model(architecture)
        restored = restore_checkpoint(checkpoint, model, device="cuda")
        assert restored["preprocessing"] == bundle.preprocessing.metadata() and restored["model_config"] == model.metadata()
        assert next(name for name, module in model.named_modules() if module is model.last_convolution) == layer_name
        assert saved["target_layer"] == source["target_layer"] == layer_name
        assert saved["target"] == config["target"] and saved["captum_target"] is None
        assert saved["relu_attributions"] and not saved["attribute_to_layer_input"]
        state_before = state_fingerprint(model)
        assert state_before == saved["model_state_sha256_before"] == saved["model_state_sha256_after"]
        assert saved["model_state_unchanged"] and saved["same_pass_manual_positive_max_difference"] == 0
        grid = GRIDS[architecture]
        assert saved["activations"].shape == saved["gradients"].shape == (5, CHANNELS[architecture], grid, grid)
        assert saved["positive_coarse_maps"].shape == (5, 1, grid, grid)
        assert saved["normalized_maps"].shape == (5, 1, 64, 64)
        batch_size = frozen.get("b1_batch_size", frozen.get("selected_batch_size"))
        reproduced, forward_sizes = independent_maps(model, payload["normalized_inputs"], positions,
            test_count=len(labels), batch_size=batch_size, layer_name=layer_name)
        differences = {}
        for name, actual in reproduced.items():
            assert torch.isfinite(actual).all(), name
            torch.testing.assert_close(actual, saved[name], rtol=1e-6, atol=1e-7)
            differences[name] = float((actual - saved[name]).abs().max())
        assert forward_sizes == saved["forward_batch_sizes"]
        assert saved["captum_forward_calls"] == len(forward_sizes)
        assert saved["padding_rows"] == sum(forward_sizes) - 5
        assert state_fingerprint(model) == state_before and not model.training
        assert all(parameter.grad is None for parameter in model.parameters())
        assert sha256_file(checkpoint) == source["checkpoint_sha256"]
        assert (saved["normalized_maps"] >= 0).all() and (saved["normalized_maps"] <= 1).all()
        assert summary["zero_map_counts"][architecture] == int((saved["upsampled_peaks"] == 0).sum()) == 0
        frame_fractions = []
        for index, row_id in enumerate(row_ids):
            records = [row for row in table_rows if int(row["row_id"]) == row_id and row["architecture"] == architecture]
            assert len(records) == 1
            row = records[0]
            assert int(row["true_label"]) == int(row["saved_predicted_label"]) == 1
            assert row["target_layer"] == layer_name and int(row["native_grid_size"]) == grid
            assert math.isclose(float(row["saved_final_test_probability"]), float(saved["probabilities"][index]), abs_tol=1e-14)
            assert math.isclose(float(row["gradcam_forward_probability"]), float(saved["probabilities"][index]), abs_tol=1e-14)
            assert math.isclose(float(row["upsampled_positive_max"]), float(saved["upsampled_peaks"][index]), abs_tol=1e-14)
            assert row["empty_positive_map"] == "False"
            full = saved["upsampled_positive_maps"][index, 0].double().numpy()
            border = np.ones((64, 64), dtype=bool); border[8:56, 8:56] = False
            frame_fraction = float(full[border].sum() / full.sum())
            frame_fractions.append(frame_fraction)
            assert math.isclose(float(row["outer_8_pixel_frame_mass_fraction"]), frame_fraction, abs_tol=1e-12)
            y, x = np.unravel_index(saved["normalized_maps"][index, 0].numpy().argmax(), (64, 64))
            assert int(row["peak_x"]) == x and int(row["peak_y"]) == y
            ys, xs = np.indices((64, 64))
            assert math.isclose(float(row["weighted_center_x"]), float((full * xs).sum() / full.sum()), abs_tol=1e-12)
            assert math.isclose(float(row["weighted_center_y"]), float((full * ys).sum() / full.sum()), abs_tol=1e-12)
        model_checks[architecture] = {"native_grid_size": grid, "target_layer": layer_name,
            "independent_autograd_maximum_differences": differences, "source_inference_batch_sizes": forward_sizes,
            "discarded_duplicate_padding_rows": saved["padding_rows"], "parameter_gradients_unaccumulated": True,
            "state_and_checkpoint_unchanged": True, "saved_test_logits_and_probabilities_exactly_equal": True,
            "mean_outer_8_pixel_frame_mass_fraction": float(np.mean(frame_fractions))}
        print(f"Audited {architecture.upper()}: 5 maps, maximum independent difference {max(differences.values()):.3g}", flush=True)
        del model
    selected_rows = list(csv.DictReader((output / "tables/selected_images.csv").open(encoding="utf-8", newline="")))
    assert [int(row["row_id"]) for row in selected_rows] == row_ids
    assert all(int(row["label"]) == 1 and int(row["sampling_seed"]) == 42 for row in selected_rows)
    geometry_rows = list(csv.DictReader((output / "tables/geometric_diagnostics.csv").open(encoding="utf-8", newline="")))
    assert len(geometry_rows) == 3
    for row in geometry_rows:
        assert int(row["sampled_images"]) == 5 and int(row["empty_positive_maps"]) == 0
        assert math.isclose(float(row["mean_outer_8_pixel_frame_mass_fraction"]),
                            model_checks[row["architecture"]]["mean_outer_8_pixel_frame_mass_fraction"], abs_tol=1e-12)
    for relative in ("figures/gradcam_comparison.png", "figures/gradcam_comparison.pdf", "analysis.md"):
        assert (output / relative).is_file() and (output / relative).stat().st_size > 0
    assert not list(output.rglob("*test_metrics*")) and not (output / "final_test.json").exists()
    check_previous(previous)
    report = {"status": "passed", "step": 13, "output_directory": str(output), "device": "cuda",
        "gpu": torch.cuda.get_device_name(), "sample_seed": 42, "sample_row_ids": row_ids, "overlay_count": 15,
        "random_selection_reproduced_independently": True, "original_pixels_and_normalized_inputs_verified": True,
        "independent_gradient_method": "torch.autograd.grad(logits.sum(), selected convolution output); no Captum calls",
        "models": model_checks, "attribution_sha256": summary["attribution_sha256"],
        "full_test_performance_evaluations": 0, "additional_selected_image_attribution_forwards": 3,
        "all_previous_output_hashes_unchanged": True, "previous_outputs_checked": len(previous),
        "all_supplied_file_hashes_unchanged": True, "source_model_and_data_lineage_verified": True,
        "tables_and_figure_files_checked": True}
    write_json(output / "verification.json", report)
    write_json(ROOT / "artifacts/b4_checks/latest.json", report)
    print(f"B4 audit passed; 15 maps and {len(previous)} unchanged prior outputs; no full-test performance evaluation.", flush=True)


if __name__ == "__main__":
    main()
