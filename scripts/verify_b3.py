"""Audit B3 checkpoint, saved final predictions, and all three complexity records."""

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

import numpy as np
import torch

from assignment_common import binary_metrics, evaluate, make_loader, restore_checkpoint, seed_everything
from assignment_common.artifacts import write_json
from part_b.src.b1 import shuffle_fingerprint, state_fingerprint
from part_b.src.b3 import SELECTION_RULE, _upstream, comparison_rows, validate_b2_frozen
from part_b.src.complexity import FLOPS_CONVENTION, profile_model
from part_b.src.data import CLASS_MAPPING, load_data_bundle, sha256_file
from part_b.src.models import build_model
from scripts.verify_b1 import audit_history, compare_metrics


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_previous(recorded):
    for relative, fingerprint in recorded.items():
        assert sha256_file(ROOT / relative) == fingerprint, relative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "part_b/outputs/b3")
    output = parser.parse_args().output.resolve()
    if read_json(output / "summary.json").get("refit"):
        from scripts.faq_saved_checks import audit_classifier
        audit_classifier(output)
        return
    config, summary = read_json(output / "configuration.json"), read_json(output / "summary.json")
    frozen, selected = read_json(output / "frozen_b3_settings.json"), read_json(output / "selected.json")
    assert summary["status"] == read_json(output / "status.json")["status"] == "completed"
    assert config["architecture"] == summary["architecture"] == frozen["architecture"] == "b3"
    assert config["seed"] == summary["seed"] == frozen["selection_seed"] == selected["seed"] == 42
    assert config["selection_rule"] == summary["selection_rule"] == frozen["selection_rule"] == SELECTION_RULE
    assert selected == summary["selected"]
    assert config["optimizer"] == frozen["optimizer"] == "Adam"
    assert config["max_epochs"] == frozen["max_epochs"] == 20
    assert config["patience"] == frozen["patience"] == 4 and config["min_delta"] == frozen["min_delta"] == 1e-4
    assert config["probability_threshold"] == frozen["probability_threshold"] == .5
    assert config["stopping_monitor"] == "validation_loss"
    assert not config["refit"] and not summary["refit"] and not frozen["refit"]
    assert config["augmentation"] is None and config["class_weighting"] is None
    assert frozen["trainable_parameters"] == selected["trainable_parameters"] == summary["trainable_parameters"] == 77169
    assert frozen["last_convolution"] == "features.block64.conv2.conv"
    for key in ("checkpoint", "checkpoint_sha256", "validation_loss", "validation_classification_error"):
        assert selected[key] == frozen[key]
    assert selected["best_epoch"] == frozen["selected_epoch"]
    prepared = Path(config["prepared_directory"])
    bundle = load_data_bundle(prepared)
    for key, actual in (("source_sha256", bundle.source_sha256),
                        ("prepared_dataset_sha256", sha256_file(prepared / "datasets.pt")),
                        ("prepared_preprocessing_sha256", sha256_file(prepared / "preprocessing.pt"))):
        assert actual == config[key] == frozen[key]
    assert torch.load(output / "preprocessing.pt", weights_only=True) == bundle.preprocessing.metadata()
    assert config["provided_partitions"] == {key: len(value) for key, value in bundle.datasets.items()}
    assert config["class_mapping"] == frozen["class_mapping"] == {str(key): value for key, value in CLASS_MAPPING.items()}
    assert config["class_mapping_status"] == frozen["class_mapping_status"] == "assumed_not_archive_metadata"
    assert config["training_validation_overlap_unique_images"] == frozen["training_validation_overlap_unique_images"] == 8
    b1, b2 = Path(config["b1_directory"]), Path(config["b2_directory"])
    inherited = {}
    for name, directory in (("b1", b1), ("b2", b2)):
        inherited[name] = read_json(directory / f"frozen_{name}_settings.json")
        assert inherited[name] == config[f"{name}_frozen_settings"]
        assert sha256_file(directory / f"frozen_{name}_settings.json") == config[f"{name}_frozen_settings_sha256"] == frozen[f"{name}_frozen_settings_sha256"]
        _upstream(directory, name, inherited[name])
    assert inherited["b2"]["b1_frozen_settings_sha256"] == config["b1_frozen_settings_sha256"]
    learning_rate, batch_size = validate_b2_frozen(inherited["b2"], inherited["b1"], source_sha256=bundle.source_sha256,
        dataset_sha256=config["prepared_dataset_sha256"], preprocessing_sha256=config["prepared_preprocessing_sha256"])
    assert learning_rate == config["learning_rate"] == summary["learning_rate"] == frozen["selected_learning_rate"] == selected["learning_rate"]
    assert batch_size == config["batch_size"] == summary["batch_size"] == frozen["selected_batch_size"] == selected["batch_size"]
    previous = read_json(ROOT / "artifacts/b3_checks/before.json")["previous_output_hashes"]
    check_previous(previous)
    for relative, recorded in read_json(ROOT / "artifacts/setup/environment_check.json")["files"].items():
        assert sha256_file(ROOT / relative) == recorded["sha256"], relative

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce full B3 train/validation metrics")
    torch.set_num_threads(4)
    run = output / "run"
    run_config, run_summary = read_json(run / "config.json"), read_json(run / "summary.json")
    assert run_summary["status"] == "completed" and not run_summary["test_assisted_stopping"]
    assert run_config["batch_size"] == batch_size and run_config["trainable_parameters"] == 77169
    assert run_config["task"] == {"threshold": .5, "kind": "binary_classification"}
    assert run_config["checkpoint_selection"] == "lowest_error_then_loss_then_earlier_epoch"
    assert run_config["stopping_metric"] == "validation_loss" and not run_config["test_assisted_stopping"]
    assert run_config["training"] == {"learning_rate": learning_rate, "max_epochs": 20,
        "patience": 4, "min_delta": 1e-4, "seed": 42, "device": "cuda", "mode": "selection"}
    history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
    history_check = audit_history(history, run_summary, {"train": (4708, 3494, 1214), "validation": (524, 389, 135)})
    assert selected["epochs_completed"] == run_summary["epochs_completed"]
    assert selected["stopped_early"] == run_summary["stopped_early"]
    steps = math.ceil(4708 / batch_size)
    assert selected["optimizer_steps_per_epoch"] == steps
    assert selected["optimizer_steps_to_selected_epoch"] == steps * selected["best_epoch"]
    assert selected["optimizer_steps_completed"] == steps * selected["epochs_completed"]
    assert selected["last_training_batch_size"] == (4708 % batch_size or batch_size)
    seed_everything(42)
    assert selected["initial_state_sha256"] == state_fingerprint(build_model("b3"))
    assert selected["first_epoch_shuffle_sha256"] == shuffle_fingerprint(4708, batch_size=batch_size)
    checkpoint_path = output / selected["checkpoint"]
    assert sha256_file(checkpoint_path) == selected["checkpoint_sha256"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == selected["best_epoch"] == run_summary["best_epoch"]
    assert checkpoint["selected_metrics"] == run_summary["selected_metrics"]
    assert checkpoint["training_config"] == run_config["training"]
    assert checkpoint["preprocessing"] == bundle.preprocessing.metadata()
    assert checkpoint["model_config"] == run_config["model_config"] == build_model("b3").metadata()
    assert checkpoint["optimizer_state"]["param_groups"][0]["lr"] == learning_rate
    bn_counts = [value.item() for key, value in checkpoint["model_state"].items() if key.endswith("num_batches_tracked")]
    assert len(bn_counts) == 9 and all(value == steps * selected["best_epoch"] for value in bn_counts)
    assert all(value["step"].item() == steps * selected["best_epoch"] for value in checkpoint["optimizer_state"]["state"].values())
    model = build_model("b3")
    restored = restore_checkpoint(checkpoint_path, model, device="cuda")
    differences = {}
    for split in ("train", "validation"):
        metrics = evaluate(model, make_loader(bundle.datasets[split], batch_size, seed=42, device="cuda"),
                           restored["task_object"], "cuda")
        differences[split] = compare_metrics(metrics, run_summary["selected_metrics"][split])
        assert all(selected[f"{split}_{key}"] == value for key, value in run_summary["selected_metrics"][split].items())
    print(f"B3 full CUDA train/validation reproduction: {differences}", flush=True)
    del model

    final = read_json(output / "final_test.json")
    test_files = list(output.rglob("test_metrics.json"))
    assert test_files == [run / "test_metrics.json"]
    test_record = read_json(test_files[0])
    assert test_record["checkpoint_epoch"] == final["checkpoint_epoch"] == selected["best_epoch"]
    assert test_record["seed"] == final["seed"] == 42
    assert test_record["metrics"] == final["metrics"] == summary["final_test"]
    assert final["test_evaluations"] == summary["test_evaluations"] == 1
    assert not test_record["test_assisted_stopping"] and not final["test_assisted_stopping"]
    assert final["checkpoint_sha256"] == selected["checkpoint_sha256"]
    assert final["learning_rate"] == learning_rate and final["batch_size"] == batch_size
    assert (output / "frozen_b3_settings.json").stat().st_mtime_ns <= test_files[0].stat().st_mtime_ns
    predictions = torch.load(output / "final_test_predictions.pt", map_location="cpu", weights_only=True)
    assert predictions["architecture"] == "b3" and predictions["evaluation_batch_size"] == batch_size
    assert predictions["threshold"] == .5 and predictions["seed"] == 42
    assert predictions["checkpoint_sha256"] == selected["checkpoint_sha256"] and predictions["source_sha256"] == bundle.source_sha256
    assert predictions["class_mapping"] == CLASS_MAPPING and predictions["metrics"] == final["metrics"]
    torch.testing.assert_close(predictions["row_ids"], bundle.row_ids["test"], atol=0, rtol=0)
    torch.testing.assert_close(predictions["labels"], bundle.datasets["test"].tensors[1], atol=0, rtol=0)
    assert predictions["logits"].shape == predictions["probabilities"].shape == (624,)
    computed = binary_metrics(predictions["labels"], predictions["probabilities"], threshold=.5)
    assert {key: final["metrics"][key] for key in computed} == computed
    torch.testing.assert_close(predictions["probabilities"], torch.sigmoid(predictions["logits"]), rtol=1e-6, atol=2e-7)
    logits, labels = (predictions[key].numpy().astype(np.float64) for key in ("logits", "labels"))
    independent_loss = float((np.maximum(logits, 0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))).mean())
    assert math.isclose(independent_loss, final["metrics"]["loss"], rel_tol=1e-6, abs_tol=1e-7)

    inputs = read_json(output / "comparison_inputs.json")
    assert inputs["b1_b2_test_results_reused_without_inference"] and summary["b1_b2_test_results_reused_without_inference"]
    complexity_checks, saved_complexities = {}, {}
    expected_counts = {"b1": (99569, 39063616, 0), "b2": (18185, 19464256, 0), "b3": (77169, 25325696, 28672)}
    for name, directory, settings in (("b1", b1, inherited["b1"]), ("b2", b2, inherited["b2"]), ("b3", output, frozen)):
        source_final = read_json(directory / "final_test.json")
        record, lineage = inputs["records"][name], inputs["lineage"][name]
        assert source_final["metrics"] == record["metrics"]
        assert record["learning_rate"] == settings["selected_learning_rate"]
        assert record["batch_size"] == (128 if name == "b1" else settings["selected_batch_size"])
        assert record["checkpoint_epoch"] == source_final["checkpoint_epoch"] == settings["selected_epoch"]
        assert record["seed"] == source_final["seed"] == 42 and record["test_evaluations"] == source_final["test_evaluations"] == 1
        assert record["checkpoint_sha256"] == settings["checkpoint_sha256"] == lineage["checkpoint_sha256"]
        assert sha256_file(directory / lineage["checkpoint"]) == lineage["checkpoint_sha256"]
        assert sha256_file(directory / "final_test.json") == lineage["final_test_sha256"]
        assert sha256_file(directory / f"frozen_{name}_settings.json") == lineage["frozen_settings_sha256"]
        saved = read_json(output / "complexity" / f"{name}.json")
        profile_copy = build_model(name)
        restore_checkpoint(directory / settings["checkpoint"], profile_copy, device="cpu")
        fingerprint = state_fingerprint(profile_copy)
        reproduced = profile_model(profile_copy)
        assert saved == reproduced
        assert state_fingerprint(profile_copy) == fingerprint
        assert saved["convention"] == FLOPS_CONVENTION and saved["input_shape"] == [1, 1, 64, 64]
        total, flops, residual = expected_counts[name]
        assert saved["trainable_parameters"] == total and saved["inference_flops"] == saved["profiler_flops"] == flops
        assert saved["residual_addition_flops"] == residual
        shape_macs = 0
        for layer in saved["layers"]:
            if layer["type"] == "Conv2d":
                macs = math.prod(layer["output_shape"]) * layer["in_channels"] // layer["groups"] * math.prod(layer["kernel_size"])
                assert not layer["bias"]
            elif layer["type"] == "Linear":
                macs = math.prod(layer["output_shape"]) * layer["in_features"]
                assert layer["bias"]
            else:
                macs = 0
            assert layer["macs"] == macs and layer["flops"] == 2 * macs
            shape_macs += macs
        assert shape_macs == saved["convolution_and_linear_macs"]
        assert 2 * shape_macs + sum(math.prod(row["output_shape"]) for row in saved["residual_additions"]) == flops
        saved_complexities[name] = saved
        complexity_checks[name] = {"parameters": total, "inference_flops": flops, "residual_addition_flops": residual,
            "profiler_and_independent_shape_formulas_agree": True, "checkpoint_state_unchanged": True,
            "synthetic_input_only": True}
        print(f"Audited {name.upper()}: {total:,} parameters; {flops:,} FLOPs/image (documented scope)", flush=True)
    rows = comparison_rows(inputs["records"], saved_complexities)
    assert rows == summary["architecture_comparison"] == read_json(output / "architecture_comparison.json")
    with (output / "tables/architecture_comparison.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 3
    for csv_row, row in zip(csv_rows, rows):
        for key, value in row.items():
            if isinstance(value, (int, float)):
                assert math.isclose(float(csv_row[key]), value, rel_tol=1e-12, abs_tol=1e-12)
            else:
                assert csv_row[key] == value
    for relative in ("figures/training_validation_loss.png", "figures/training_validation_loss.pdf",
                     "figures/training_validation_accuracy.png", "figures/training_validation_accuracy.pdf",
                     "tables/final_test.csv", "tables/validation_selection.csv", "tables/learning_curves.csv",
                     "tables/selected_pipeline_learning_curves.csv", "tables/parameter_breakdown.csv",
                     "tables/flops_layers.csv", "tables/flops_operator_counts.csv", "tables/residual_addition_flops.csv",
                     "curve_diagnostics.json", "analysis.md"):
        assert (output / relative).is_file() and (output / relative).stat().st_size > 0
    assert sha256_file(checkpoint_path) == selected["checkpoint_sha256"]
    check_previous(previous)
    report = {"status": "passed", "step": 12, "output_directory": str(output), "device": "cuda",
        "gpu": torch.cuda.get_device_name(), "learning_rate": learning_rate, "batch_size": batch_size,
        "selected_epoch": selected["best_epoch"], "history": history_check,
        "maximum_restored_metric_differences": differences, "full_training_validation_restoration": True,
        "b1_b2_frozen_hyperparameters_and_lineage_verified": True,
        "optimizer_and_all_nine_batchnorm_counters_verified": True,
        "test_metrics_recomputed_from_same_pass_saved_predictions": True,
        "independent_test_bce_loss": independent_loss,
        "independent_test_bce_absolute_difference": abs(independent_loss - final["metrics"]["loss"]),
        "test_prediction_sha256": sha256_file(output / "final_test_predictions.pt"),
        "additional_test_model_evaluations": 0, "b1_b2_additional_test_model_evaluations": 0,
        "selected_run_test_records": 1, "validation_selection_frozen_before_test": True,
        "complexity": complexity_checks, "flops_convention": FLOPS_CONVENTION,
        "complexity_profiles_use_synthetic_images_only": True, "all_comparison_rows_match_saved_test_results": True,
        "all_previous_output_hashes_unchanged": True, "previous_outputs_checked": len(previous),
        "all_supplied_file_hashes_unchanged": True, "data_and_preprocessing_hashes_match": True,
        "tables_and_figure_files_checked": True}
    write_json(output / "verification.json", report)
    write_json(ROOT / "artifacts/b3_checks/latest.json", report)
    print("B3 audit passed; no additional model test inference for any architecture.", flush=True)


if __name__ == "__main__":
    main()
