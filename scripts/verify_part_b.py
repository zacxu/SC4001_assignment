"""Audit full X-ray exports and Figure B CNNs with CUDA training/checkpoint probes."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from assignment_common import (
    TrainingConfig, evaluate, make_loader, restore_checkpoint, seed_everything,
)
from assignment_common.artifacts import write_json
from part_b.src.data import EXPECTED_COUNTS, SPLITS, load_data_bundle, sha256_file
from part_b.src.models import ConvBNReLU, MODEL_NAMES, ResidualBlock, build_model, model_from_metadata
from part_b.src.training import train_classifier


# Independent arithmetic from Figure B, including BN scale/bias and FC bias.
EXPECTED_COUNTS_BY_MODEL = {
    "b1": {"conv": 400 + 12800 + 18432, "batchnorm": 2 * (16 + 32 + 64),
           "fc": (1024 * 64 + 64) + (64 * 32 + 32) + (32 + 1)},
    "b2": {"conv": 72 + 576 + 1152 + 2304 + 4608 + 9216,
           "batchnorm": 2 * (8 + 8 + 16 + 16 + 32 + 32), "fc": 32 + 1},
    "b3": {"conv": 144 + 2304 + 2304 + 4608 + 9216 + 512 + 18432 + 36864 + 2048,
           "batchnorm": 2 * (16 + 16 + 16 + 32 + 32 + 32 + 64 + 64 + 64), "fc": 64 + 1},
}
EXPECTED_FEATURE_SHAPES = {
    "b1": [(16, 64, 64), (16, 32, 32), (32, 32, 32), (32, 16, 16),
           (64, 16, 16), (64, 8, 8), (64, 4, 4)],
    "b2": [(8, 64, 64), (8, 64, 64), (8, 32, 32), (16, 32, 32), (16, 32, 32),
           (16, 16, 16), (32, 16, 16), (32, 16, 16), (32, 8, 8), (32, 1, 1)],
    "b3": [(16, 64, 64), (16, 32, 32), (16, 32, 32), (32, 16, 16), (64, 8, 8), (64, 1, 1)],
}


def _part_a_snapshot():
    return {str(path.relative_to(ROOT)): sha256_file(path)
            for path in sorted((ROOT / "part_a/outputs").rglob("*")) if path.is_file()}


def _audit_data(bundle, source):
    assert sha256_file(source) == bundle.source_sha256
    checks = {}
    with np.load(source, allow_pickle=False) as archive:
        scaled_train = archive["train_images"].astype(np.float32) / np.float32(255)
        mean = float(scaled_train.mean(dtype=np.float64))
        std = float(scaled_train.std(dtype=np.float64, ddof=0))
        assert mean == bundle.preprocessing.mean and std == bundle.preprocessing.std
        assert bundle.preprocessing.fit_row_count == EXPECTED_COUNTS["train"]
        for split, prefix in SPLITS.items():
            raw = archive[prefix + "_images"]
            labels = archive[prefix + "_labels"]
            images, targets = bundle.datasets[split].tensors
            np.testing.assert_array_equal(bundle.raw_images[split].numpy(), raw)
            np.testing.assert_array_equal(targets.numpy(), labels.astype(np.float32))
            np.testing.assert_array_equal(bundle.row_ids[split].numpy(), np.arange(len(raw)))
            # Recompute directly from the source, without calling the production transform.
            expected = ((raw.astype(np.float32) / np.float32(255)) - np.float32(mean)) / np.float32(std)
            np.testing.assert_array_equal(images.numpy(), expected[:, None])
            reconstructed = images[:, 0].numpy() * np.float32(std) + np.float32(mean)
            maximum_error = float(np.abs(reconstructed - raw.astype(np.float32) / np.float32(255)).max())
            assert maximum_error < 2e-7
            counts = {str(label): int((labels == label).sum()) for label in (0, 1)}
            assert counts == bundle.report["partitions"][split]["class_counts"]
            checks[split] = {"rows": len(raw), "all_pixels_and_labels_match_source": True,
                             "all_normalized_pixels_match_independent_recomputation": True,
                             "original_row_order_preserved": True, "class_counts": counts,
                             "maximum_inverse_scaling_error": maximum_error}
    return checks


def _trace(model, images):
    rows, handles = [], []
    def record(name):
        def hook(module, inputs, output):
            rows.append({"layer": name, "type": type(module).__name__,
                         "input_shape": list(inputs[0].shape), "output_shape": list(output.shape),
                         "own_trainable_parameters": sum(p.numel() for p in module.parameters(recurse=False)
                                                         if p.requires_grad)})
        return hook
    for name, module in model.named_modules():
        if name and (not list(module.children()) or isinstance(module, (ConvBNReLU, ResidualBlock))):
            handles.append(module.register_forward_hook(record(name)))
    model.eval()
    try:
        with torch.no_grad():
            output = model(images)
    finally:
        for handle in handles:
            handle.remove()
    assert output.shape == (len(images),)
    return rows


def _aligned_projection_probe(device):
    block = ResidualBlock(16, 32, stride=2).to(device)
    with torch.no_grad():
        block.conv1.conv.weight.zero_()
        block.shortcut.conv.weight.zero_()
        block.conv1.conv.weight[0, 0, 1, 1] = 1
        block.shortcut.conv.weight[0, 0, 0, 0] = 1
        grid = torch.arange(32 * 32, dtype=torch.float32, device=device).reshape(1, 1, 32, 32).repeat(1, 16, 1, 1)
        expected = grid[:, :1, ::2, ::2]
        torch.testing.assert_close(block.conv1.conv(grid)[:, :1], expected, rtol=0, atol=0)
        torch.testing.assert_close(block.shortcut.conv(grid)[:, :1], expected, rtol=0, atol=0)
    return {"main_kernel": 3, "projection_kernel": 1, "stride": 2,
            "both_paths_sample_centers": "0,2,4,...,30 on each input axis",
            "coordinate_probe_matches_exactly": True}


def _audit_model(name, bundle, directory, device):
    seed_everything(42)
    model = build_model(name).to(device)
    for layer in model.modules():
        if isinstance(layer, ConvBNReLU):
            assert [type(child) for child in layer.children()] == [nn.Conv2d, nn.BatchNorm2d, nn.ReLU]
            assert layer.conv.bias is None
        elif isinstance(layer, nn.Linear):
            assert layer.bias is not None
    counts = {}
    for kind, layer_type in (("conv", nn.Conv2d), ("batchnorm", nn.BatchNorm2d), ("fc", nn.Linear)):
        counts[kind] = sum(p.numel() for module in model.modules() if isinstance(module, layer_type)
                           for p in module.parameters(recurse=False) if p.requires_grad)
    assert counts == EXPECTED_COUNTS_BY_MODEL[name]
    assert sum(counts.values()) == sum(p.numel() for p in model.parameters() if p.requires_grad)
    inputs = bundle.datasets["train"].tensors[0][:2].to(device)
    traces = _trace(model, inputs)
    feature_stages = [row for row in traces if row["layer"].startswith("features.") and row["layer"].count(".") == 1]
    assert [tuple(row["output_shape"][1:]) for row in feature_stages] == EXPECTED_FEATURE_SHAPES[name]
    residuals = []
    for block_name, block in model.named_modules():
        if isinstance(block, ResidualBlock):
            captured = {row["layer"]: row["output_shape"] for row in traces}
            main_shape, shortcut_shape = captured[block_name + ".conv2"], captured[block_name + ".shortcut"]
            assert main_shape == shortcut_shape == captured[block_name + ".add_relu"]
            residuals.append({"block": block_name, "main_shape": main_shape,
                              "shortcut_shape": shortcut_shape, "sum_shape": captured[block_name + ".add_relu"]})
    rows_path = directory / "layers.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(traces[0]))
        writer.writeheader()
        writer.writerows(traces)
    initial_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    generator = torch.Generator().manual_seed(42)
    train_ids = torch.randperm(len(bundle.datasets["train"]), generator=generator)[:129]
    validation_ids = torch.randperm(len(bundle.datasets["validation"]), generator=generator)[:65]
    subsets = {}
    for split, ids in (("train", train_ids), ("validation", validation_ids)):
        tensors = bundle.datasets[split].tensors
        subsets[split] = TensorDataset(*(tensor[ids] for tensor in tensors))
    torch.cuda.reset_peak_memory_stats()
    result = train_classifier(model, make_loader(subsets["train"], 128, training=True, device=device),
        validation_loader=make_loader(subsets["validation"], 32, device=device),
        config=TrainingConfig(learning_rate=.001, max_epochs=1, device=device),
        run_directory=directory / "smoke_training", preprocessing=bundle.preprocessing.metadata(),
        model_config=model.metadata())
    assert result.epochs_completed == result.best_epoch == 1
    assert not any((directory / "smoke_training").glob("*test*"))
    batchnorms = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
    assert all(module.num_batches_tracked.item() == 2 for module in batchnorms)
    assert all(not torch.equal(initial_state[key], value.cpu()) for key, value in model.state_dict().items()
               if key.endswith("conv.weight") or key == "classifier.output.weight")
    gradients = {key: float(parameter.grad.norm().item()) for key, parameter in model.named_parameters()
                 if key.endswith("conv.weight")}
    assert all(math.isfinite(value) and value > 0 for value in gradients.values())
    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
    restored = model_from_metadata(checkpoint["model_config"])
    state = restore_checkpoint(result.checkpoint_path, restored, device=device)
    assert state["preprocessing"] == bundle.preprocessing.metadata()
    restored_metrics = evaluate(restored, make_loader(subsets["validation"], 32, device=device),
                                state["task_object"], device)
    assert restored_metrics == result.selected_metrics["validation"]
    with torch.no_grad():
        outputs = restored(inputs)
        torch.testing.assert_close(restored(inputs), outputs, rtol=0, atol=0)
        torch.testing.assert_close(model(inputs), outputs, rtol=0, atol=0)
        assert restored(inputs[:1]).shape == (1,)
    # A second backward probe covers the largest possible future batch for B2/B3.
    probe_batch = 128 if name == "b1" else 256
    restored.train()
    restored.zero_grad(set_to_none=True)
    probe_inputs, probe_labels = (tensor[:probe_batch].to(device) for tensor in bundle.datasets["train"].tensors)
    probe_logits = restored(probe_inputs)
    probe_loss = nn.BCEWithLogitsLoss()(probe_logits, probe_labels)
    probe_loss.backward()
    assert torch.isfinite(probe_loss) and all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                             for parameter in restored.parameters())
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    # Revert the exploratory BatchNorm buffers before finishing; smoke checkpoint remains unchanged.
    restore_checkpoint(result.checkpoint_path, restored, device=device)
    assert not any(key.startswith("test_") for row in result.history for key in row)
    record = {"architecture": name, "model_config": model.metadata(), "parameters": counts,
              "trainable_parameters": sum(counts.values()), "layers": traces,
              "residual_additions": residuals, "all_convolutions_have_bn_relu_and_no_bias": True,
              "smoke_train_samples": 129, "smoke_validation_samples": 65, "smoke_epochs": 1,
              "smoke_batch_size": 128, "smoke_learning_rate": .001,
              "smoke_only_not_a_b1_b2_b3_experiment": True,
              "all_convolution_weights_and_output_weights_updated": True,
              "convolution_gradient_norms": gradients,
              "batchnorm_training_batches_per_layer": 2,
              "evaluation_repeat_predictions_exact": True, "single_image_output_shape": [1],
              "restored_validation_probe_metrics_match": True,
              "validation_probe_loss": restored_metrics["loss"],
              "largest_planned_batch_backward_pass": probe_batch,
              "all_parameters_receive_finite_gradients": True,
              "average_pooling": "Fixed AvgPool2d windows equivalent to required output on 64x64 inputs",
              "peak_allocated_gpu_mib_including_fresh_restored_model": peak_memory,
              "checkpoint_sha256": sha256_file(result.checkpoint_path), "test_performance_evaluated": False}
    write_json(directory / "architecture.json", record)
    print(f"{name.upper()}: {record['trainable_parameters']:,} parameters; CUDA forward/backward/Adam and "
          f"restoration passed; largest batch={probe_batch}; peak memory={peak_memory:.1f} MiB", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "part_b/outputs/data")
    parser.add_argument("--npz", type=Path, default=ROOT / "chestxray_binary_64.npz")
    parser.add_argument("--output", type=Path, help="New directory for smoke probes and shape/count tables")
    arguments = parser.parse_args()
    timestamp = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d_%H%M%S")
    output = arguments.output or ROOT / "part_b/outputs/architecture_checks" / timestamp
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("The user's CUDA environment is required for this architecture audit")
    torch.set_num_threads(4)
    supplied = json.loads((ROOT / "artifacts/setup/environment_check.json").read_text(encoding="utf-8"))["files"]
    for relative, recorded in supplied.items():
        assert sha256_file(ROOT / relative) == recorded["sha256"]
    prior_part_a = _part_a_snapshot()
    bundle = load_data_bundle(arguments.data)
    data_checks = _audit_data(bundle, arguments.npz)
    output.mkdir(parents=True, exist_ok=False)
    architecture_checks = {}
    for name in MODEL_NAMES:
        destination = output / name
        destination.mkdir()
        architecture_checks[name] = _audit_model(name, bundle, destination, "cuda")
    alignment = _aligned_projection_probe("cuda")
    assert prior_part_a == _part_a_snapshot()
    for relative, recorded in supplied.items():
        assert sha256_file(ROOT / relative) == recorded["sha256"]
    parameter_rows = [{"architecture": name, **record["parameters"],
                       "total_trainable_parameters": record["trainable_parameters"]}
                      for name, record in architecture_checks.items()]
    with (output / "parameter_counts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(parameter_rows[0]))
        writer.writeheader()
        writer.writerows(parameter_rows)
    report = {"status": "passed", "step": 9, "output_directory": str(output.resolve()),
              "prepared_directory": str(arguments.data.resolve()), "source_sha256": bundle.source_sha256,
              "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(), "device": "cuda",
              "data": data_checks, "preprocessing": bundle.preprocessing.metadata(),
              "duplicates": bundle.report["duplicates"], "projection_alignment": alignment,
              "models": architecture_checks, "supplied_file_hashes_unchanged": True,
              "all_part_a_output_hashes_unchanged": True, "part_a_outputs_checked": len(prior_part_a),
              "test_performance_evaluated": False,
              "scope": "Full data preparation and architecture verification; no full B1-B3 experiments"}
    write_json(output / "verification.json", report)
    write_json(ROOT / "artifacts/part_b_checks/latest.json", report)
    print(f"Step 9 audit passed. Evidence: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
