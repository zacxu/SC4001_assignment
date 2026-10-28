"""Audit B2 CUDA checkpoints and saved test predictions without another test inference."""

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
from part_b.src.b1 import choose_learning_rate, shuffle_fingerprint, state_fingerprint
from part_b.src.b2 import BATCH_SIZES, SELECTION_RULE, choose_batch_size, validate_b1_frozen
from part_b.src.data import CLASS_MAPPING, load_data_bundle, sha256_file
from part_b.src.models import build_model, model_from_metadata
from scripts.verify_b1 import audit_history, compare_metrics


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_previous_outputs(recorded):
    for relative, fingerprint in recorded.items():
        assert sha256_file(ROOT / relative) == fingerprint, relative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "part_b/outputs/b2")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if read_json(output / "summary.json").get("refit"):
        from scripts.faq_saved_checks import audit_classifier
        audit_classifier(output)
        return
    config, summary = read_json(output / "configuration.json"), read_json(output / "summary.json")
    frozen = read_json(output / "frozen_b2_settings.json")
    rows = read_json(output / "batch_size_search.json")
    assert summary["status"] == read_json(output / "status.json")["status"] == "completed"
    assert config["architecture"] == summary["architecture"] == frozen["architecture"] == "b2"
    assert config["selection_rule"] == summary["selection_rule"] == frozen["selection_rule"] == SELECTION_RULE
    assert config["batch_size_grid"] == summary["batch_size_grid"] == list(BATCH_SIZES)
    assert config["seed"] == summary["seed"] == frozen["selection_seed"] == 42
    assert config["max_epochs"] == frozen["max_epochs"] == 20
    assert config["patience"] == frozen["patience"] == 4
    assert config["min_delta"] == frozen["min_delta"] == 1e-4
    assert config["optimizer"] == frozen["optimizer"] == "Adam"
    assert config["probability_threshold"] == frozen["probability_threshold"] == .5
    assert config["stopping_monitor"] == "validation_loss"
    assert not config["refit"] and not summary["refit"] and not frozen["refit"]
    assert config["augmentation"] is None and config["class_weighting"] is None
    assert config["fresh_model_and_loader_per_trial"] and config["same_sample_order_different_batch_grouping"]
    assert summary["trials"] == 5 and summary["trainable_parameters"] == frozen["trainable_parameters"] == 18185
    selected = choose_batch_size(rows)
    assert selected == summary["selected"]
    assert selected["batch_size"] == frozen["selected_batch_size"]
    assert selected["best_epoch"] == frozen["selected_epoch"]
    for key in ("checkpoint", "checkpoint_sha256", "validation_loss", "validation_classification_error"):
        assert selected[key] == frozen[key]

    prepared, b1 = Path(config["prepared_directory"]), Path(config["b1_directory"])
    bundle = load_data_bundle(prepared)
    for key, actual in (("source_sha256", bundle.source_sha256),
                        ("prepared_dataset_sha256", sha256_file(prepared / "datasets.pt")),
                        ("prepared_preprocessing_sha256", sha256_file(prepared / "preprocessing.pt"))):
        assert actual == config[key] == frozen[key]
    assert torch.load(output / "preprocessing.pt", weights_only=True) == bundle.preprocessing.metadata()
    assert config["provided_partitions"] == {key: len(value) for key, value in bundle.datasets.items()}
    assert config["training_validation_overlap_unique_images"] == frozen["training_validation_overlap_unique_images"] == 8
    assert config["class_mapping"] == frozen["class_mapping"] == {str(key): value for key, value in CLASS_MAPPING.items()}
    assert config["class_mapping_status"] == frozen["class_mapping_status"] == "assumed_not_archive_metadata"
    b1_frozen = read_json(b1 / "frozen_b1_settings.json")
    assert b1_frozen == config["b1_frozen_settings"]
    assert sha256_file(b1 / "frozen_b1_settings.json") == config["b1_frozen_settings_sha256"] == frozen["b1_frozen_settings_sha256"]
    learning_rate = validate_b1_frozen(b1_frozen, source_sha256=bundle.source_sha256,
        dataset_sha256=config["prepared_dataset_sha256"], preprocessing_sha256=config["prepared_preprocessing_sha256"])
    assert learning_rate == config["learning_rate"] == summary["learning_rate"] == frozen["selected_learning_rate"] == frozen["b1_selected_learning_rate"]
    assert read_json(b1 / "verification.json")["status"] == "passed"
    assert read_json(b1 / "status.json")["status"] == read_json(b1 / "summary.json")["status"] == "completed"
    b1_selected = choose_learning_rate(read_json(b1 / "learning_rate_search.json"))
    assert b1_selected == read_json(b1 / "summary.json")["selected"]
    for key, frozen_key in (("learning_rate", "selected_learning_rate"), ("best_epoch", "selected_epoch"),
                            ("validation_loss", "validation_loss"), ("validation_classification_error", "validation_classification_error"),
                            ("checkpoint", "checkpoint"), ("checkpoint_sha256", "checkpoint_sha256")):
        assert b1_selected[key] == b1_frozen[frozen_key]
    assert sha256_file(b1 / b1_frozen["checkpoint"]) == b1_frozen["checkpoint_sha256"] == frozen["b1_checkpoint_sha256"]
    source_files = read_json(ROOT / "artifacts/setup/environment_check.json")["files"]
    for relative, recorded in source_files.items():
        assert sha256_file(ROOT / relative) == recorded["sha256"], relative
    previous = read_json(ROOT / "artifacts/b2_checks/before.json")["previous_output_hashes"]
    check_previous_outputs(previous)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce saved B2 training/validation metrics")
    torch.set_num_threads(4)
    seed_everything(42)
    initial_fingerprint = state_fingerprint(build_model("b2"))
    shuffle = shuffle_fingerprint(4708)
    checks = []
    counts = {"train": (4708, 3494, 1214), "validation": (524, 389, 135)}
    for row in rows:
        batch_size = row["batch_size"]
        steps = math.ceil(4708 / batch_size)
        run = output / row["run_directory"]
        run_config, run_summary = read_json(run / "config.json"), read_json(run / "summary.json")
        assert run_summary["status"] == "completed" and not run_summary["test_assisted_stopping"]
        assert run_config["batch_size"] == batch_size and run_config["trainable_parameters"] == row["trainable_parameters"] == 18185
        assert row["learning_rate"] == learning_rate and row["seed"] == 42
        assert row["epochs_completed"] == run_summary["epochs_completed"] and row["stopped_early"] == run_summary["stopped_early"]
        assert row["optimizer_steps_per_epoch"] == steps
        assert row["optimizer_steps_to_selected_epoch"] == steps * row["best_epoch"]
        assert row["optimizer_steps_completed"] == steps * row["epochs_completed"]
        assert row["last_training_batch_size"] == (4708 % batch_size or batch_size)
        assert run_config["checkpoint_selection"] == "lowest_error_then_loss_then_earlier_epoch"
        assert run_config["stopping_metric"] == "validation_loss" and not run_config["test_assisted_stopping"]
        assert run_config["task"] == {"threshold": .5, "kind": "binary_classification"}
        assert run_config["training"] == {"learning_rate": learning_rate, "max_epochs": 20,
            "patience": 4, "min_delta": 1e-4, "seed": 42, "device": "cuda", "mode": "selection"}
        assert row["initial_state_sha256"] == initial_fingerprint
        assert row["first_epoch_shuffle_sha256"] == shuffle == shuffle_fingerprint(4708, batch_size=batch_size)
        checkpoint_path = output / row["checkpoint"]
        assert sha256_file(checkpoint_path) == row["checkpoint_sha256"]
        history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
        history_check = audit_history(history, run_summary, counts)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert checkpoint["epoch"] == row["best_epoch"] == run_summary["best_epoch"]
        assert checkpoint["selected_metrics"] == run_summary["selected_metrics"]
        assert checkpoint["training_config"] == run_config["training"]
        assert checkpoint["preprocessing"] == bundle.preprocessing.metadata()
        assert checkpoint["optimizer_state"]["param_groups"][0]["lr"] == learning_rate
        assert checkpoint["model_config"] == run_config["model_config"] == build_model("b2").metadata()
        batchnorm_counts = [value.item() for key, value in checkpoint["model_state"].items() if key.endswith("num_batches_tracked")]
        assert len(batchnorm_counts) == 6 and all(value == steps * row["best_epoch"] for value in batchnorm_counts)
        assert all(value["step"].item() == steps * row["best_epoch"] for value in checkpoint["optimizer_state"]["state"].values())
        seed_everything(42)
        model = model_from_metadata(checkpoint["model_config"])
        restored = restore_checkpoint(checkpoint_path, model, device="cuda")
        differences = {}
        for split in ("train", "validation"):
            metrics = evaluate(model, make_loader(bundle.datasets[split], batch_size, seed=42, device="cuda"),
                               restored["task_object"], "cuda")
            differences[split] = compare_metrics(metrics, run_summary["selected_metrics"][split])
            assert all(row[f"{split}_{key}"] == value for key, value in run_summary["selected_metrics"][split].items())
        assert sha256_file(checkpoint_path) == row["checkpoint_sha256"]
        checks.append({"batch_size": batch_size, "checkpoint": row["checkpoint"],
            "checkpoint_sha256": row["checkpoint_sha256"], "history": history_check,
            "maximum_restored_metric_differences": differences, "full_train_and_validation_reproduced": True,
            "initialization_and_first_epoch_order_match": True, "optimizer_steps_per_epoch": steps,
            "selected_optimizer_and_batchnorm_updates": steps * row["best_epoch"]})
        print(f"Audited batch {batch_size}: checkpoint epoch {row['best_epoch']}, "
              f"full train/validation metric differences {differences}", flush=True)
        del model

    test_files = list((output / "runs").rglob("test_metrics.json"))
    assert test_files == [output / selected["run_directory"] / "test_metrics.json"]
    test_record = read_json(test_files[0])
    final = read_json(output / "final_test.json")
    assert test_record["checkpoint_epoch"] == final["checkpoint_epoch"] == selected["best_epoch"]
    assert test_record["seed"] == final["seed"] == 42
    assert final["selected_batch_size"] == selected["batch_size"] and final["learning_rate"] == learning_rate
    assert not test_record["test_assisted_stopping"] and not final["test_assisted_stopping"]
    assert test_record["metrics"] == final["metrics"] == summary["final_test"]
    assert final["test_evaluations"] == summary["test_evaluations"] == 1
    predictions = torch.load(output / "final_test_predictions.pt", map_location="cpu", weights_only=True)
    assert predictions["architecture"] == "b2" and predictions["evaluation_batch_size"] == selected["batch_size"]
    assert predictions["seed"] == 42 and predictions["checkpoint_sha256"] == selected["checkpoint_sha256"]
    assert predictions["source_sha256"] == bundle.source_sha256
    assert predictions["threshold"] == .5 and predictions["metrics"] == final["metrics"]
    assert predictions["class_mapping"] == CLASS_MAPPING and predictions["class_mapping_status"] == "assumed_not_archive_metadata"
    torch.testing.assert_close(predictions["row_ids"], bundle.row_ids["test"], rtol=0, atol=0)
    torch.testing.assert_close(predictions["labels"], bundle.datasets["test"].tensors[1], rtol=0, atol=0)
    assert predictions["logits"].shape == predictions["probabilities"].shape == (624,)
    expected_metrics = binary_metrics(predictions["labels"], predictions["probabilities"], threshold=.5)
    assert {key: final["metrics"][key] for key in expected_metrics} == expected_metrics
    assert expected_metrics["tp"] + expected_metrics["fn"] == 390
    assert expected_metrics["tn"] + expected_metrics["fp"] == 234
    torch.testing.assert_close(predictions["probabilities"], torch.sigmoid(predictions["logits"]), rtol=1e-6, atol=2e-7)
    logits, labels = predictions["logits"].numpy().astype(np.float64), predictions["labels"].numpy().astype(np.float64)
    independent_loss = float((np.maximum(logits, 0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))).mean())
    assert math.isclose(independent_loss, final["metrics"]["loss"], rel_tol=1e-6, abs_tol=1e-7)
    assert (output / "frozen_b2_settings.json").stat().st_mtime_ns <= test_files[0].stat().st_mtime_ns
    with (output / "tables/batch_size_selection.csv").open(newline="", encoding="utf-8") as stream:
        table_rows = list(csv.DictReader(stream))
    assert len(table_rows) == 5
    for saved, row in zip(table_rows, rows):
        for key in saved:
            assert math.isclose(float(saved[key]), row[key], rel_tol=1e-12, abs_tol=1e-12)
    with (output / "tables/learning_curves.csv").open(newline="", encoding="utf-8") as stream:
        curves = list(csv.DictReader(stream))
    assert len(curves) == sum(row["epochs_completed"] for row in rows)
    for relative in ("figures/validation_loss.png", "figures/validation_loss.pdf",
                     "figures/validation_accuracy.png", "figures/validation_accuracy.pdf",
                     "tables/final_test.csv", "tables/batch_size_search_details.csv", "curve_diagnostics.json", "analysis.md"):
        assert (output / relative).is_file() and (output / relative).stat().st_size > 0
    check_previous_outputs(previous)
    report = {"status": "passed", "step": 11, "output_directory": str(output),
        "device": "cuda", "gpu": torch.cuda.get_device_name(), "trials_audited": 5,
        "learning_rate": learning_rate, "selected_batch_size": selected["batch_size"], "selected_epoch": selected["best_epoch"],
        "checkpoints": checks, "full_training_validation_restoration": True,
        "b1_frozen_learning_rate_and_lineage_verified": True,
        "test_metrics_recomputed_from_same_pass_saved_predictions": True,
        "independent_test_bce_loss": independent_loss,
        "independent_test_bce_absolute_difference": abs(independent_loss - final["metrics"]["loss"]),
        "test_prediction_sha256": sha256_file(output / "final_test_predictions.pt"),
        "additional_test_model_evaluations": 0, "selected_run_test_records": 1,
        "validation_selection_frozen_before_test": True,
        "all_previous_output_hashes_unchanged": True, "previous_outputs_checked": len(previous),
        "all_supplied_file_hashes_unchanged": True, "data_and_preprocessing_hashes_match": True,
        "optimizer_and_batchnorm_update_counts_verified": True, "tables_and_figure_files_checked": True}
    write_json(output / "verification.json", report)
    write_json(ROOT / "artifacts/b2_checks/latest.json", report)
    print("B2 audit passed. Final test metrics were recomputed from saved predictions; no new test inference.", flush=True)


if __name__ == "__main__":
    main()
