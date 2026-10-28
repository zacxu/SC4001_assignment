"""Audit all B1 checkpoints on CUDA; recompute final metrics from saved test predictions."""

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
from part_b.src.b1 import LEARNING_RATES, SELECTION_RULE, choose_learning_rate, shuffle_fingerprint, state_fingerprint
from part_b.src.data import load_data_bundle, sha256_file
from part_b.src.models import build_model, model_from_metadata


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_metrics(actual, expected):
    assert actual.keys() == expected.keys()
    differences = []
    for key, value in expected.items():
        if isinstance(value, float):
            difference = abs(actual[key] - value)
            assert difference <= 1e-7, (key, actual[key], value)
            differences.append(difference)
        else:
            assert actual[key] == value, (key, actual[key], value)
    return max(differences, default=0.)


def audit_history(history, summary, counts):
    assert [row["epoch"] for row in history] == list(range(1, len(history) + 1))
    assert len(history) == summary["epochs_completed"] <= 20
    loss_reference, bad_epochs, selection_key = math.inf, 0, None
    for index, row in enumerate(history):
        assert not any(key.startswith("test_") for key in row)
        for partition in ("train", "validation"):
            metrics = {key[len(partition)+1:]: value for key, value in row.items() if key.startswith(partition + "_")}
            total = metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"]
            assert total == metrics["count"] == counts[partition][0]
            assert metrics["tp"] + metrics["fn"] == counts[partition][1]
            assert metrics["tn"] + metrics["fp"] == counts[partition][2]
            assert math.isclose(metrics["accuracy"], (metrics["tp"] + metrics["tn"]) / total, abs_tol=1e-12)
            assert math.isclose(metrics["classification_error"], 1 - metrics["accuracy"], abs_tol=1e-12)
            assert math.isclose(metrics["sensitivity"], metrics["tp"] / counts[partition][1], abs_tol=1e-12)
            assert math.isclose(metrics["specificity"], metrics["tn"] / counts[partition][2], abs_tol=1e-12)
            assert math.isfinite(metrics["loss"]) and metrics["loss"] >= 0
        if row["validation_loss"] < loss_reference - 1e-4:
            loss_reference, bad_epochs = row["validation_loss"], 0
        else:
            bad_epochs += 1
        if bad_epochs >= 4:
            assert index == len(history) - 1, "Training must end when patience is exhausted"
        key = (row["validation_classification_error"], row["validation_loss"], row["epoch"])
        selected = selection_key is None or key < selection_key
        assert row["checkpoint_selected"] == selected
        if selected:
            selection_key = key
    if len(history) < 20:
        assert bad_epochs == 4 and summary["stopped_early"]
    else:
        assert not summary["stopped_early"]
    assert summary["stopping_reference"] == loss_reference
    winner = min(history, key=lambda row: (row["validation_classification_error"], row["validation_loss"], row["epoch"]))
    assert winner["epoch"] == summary["best_epoch"]
    for partition in ("train", "validation"):
        selected_metrics = summary["selected_metrics"][partition]
        assert all(winner[partition + "_" + key] == value for key, value in selected_metrics.items())
    return {"epochs_completed": len(history), "selected_epoch": winner["epoch"],
            "minimum_loss_epoch": min(history, key=lambda row: row["validation_loss"])["epoch"],
            "patience_exhausted": bad_epochs >= 4, "stopping_and_selection_audited": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "part_b/outputs/b1")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if read_json(output / "summary.json").get("refit"):
        from scripts.faq_saved_checks import audit_classifier
        audit_classifier(output)
        return
    config, summary = read_json(output / "configuration.json"), read_json(output / "summary.json")
    frozen = read_json(output / "frozen_b1_settings.json")
    rows = read_json(output / "learning_rate_search.json")
    assert summary["status"] == read_json(output / "status.json")["status"] == "completed"
    assert config["selection_rule"] == summary["selection_rule"] == frozen["selection_rule"] == SELECTION_RULE
    assert config["learning_rate_grid"] == summary["learning_rate_grid"] == list(LEARNING_RATES)
    assert config["seed"] == summary["seed"] == frozen["selection_seed"] == 42
    assert config["batch_size"] == summary["batch_size"] == frozen["b1_batch_size"] == 128
    assert config["max_epochs"] == frozen["max_epochs"] == 20
    assert config["patience"] == frozen["patience"] == 4
    assert config["min_delta"] == frozen["min_delta"] == 1e-4
    assert config["stopping_monitor"] == "validation_loss" and not summary["refit"]
    assert config["probability_threshold"] == frozen["probability_threshold"] == .5
    selected = choose_learning_rate(rows)
    assert selected == summary["selected"]
    assert selected["learning_rate"] == frozen["selected_learning_rate"]
    assert selected["best_epoch"] == frozen["selected_epoch"]
    assert selected["checkpoint"] == frozen["checkpoint"]
    assert selected["checkpoint_sha256"] == frozen["checkpoint_sha256"]
    assert selected["validation_loss"] == frozen["validation_loss"]
    assert selected["validation_classification_error"] == frozen["validation_classification_error"]
    prepared = Path(config["prepared_directory"])
    bundle = load_data_bundle(prepared)
    assert bundle.source_sha256 == config["source_sha256"] == frozen["source_sha256"]
    assert sha256_file(prepared / "datasets.pt") == config["prepared_dataset_sha256"] == frozen["prepared_dataset_sha256"]
    assert sha256_file(prepared / "preprocessing.pt") == config["prepared_preprocessing_sha256"] == frozen["prepared_preprocessing_sha256"]
    assert torch.load(output / "preprocessing.pt", weights_only=True) == bundle.preprocessing.metadata()
    assert config["provided_partitions"] == {key: len(value) for key, value in bundle.datasets.items()}
    assert config["training_validation_overlap_unique_images"] == frozen["training_validation_overlap_unique_images"] == 8
    source_files = read_json(ROOT / "artifacts/setup/environment_check.json")["files"]
    for relative, recorded in source_files.items():
        assert sha256_file(ROOT / relative) == recorded["sha256"]
    previous = read_json(ROOT / "artifacts/b1_checks/before.json")["previous_output_hashes"]
    for relative, fingerprint in previous.items():
        assert sha256_file(ROOT / relative) == fingerprint
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce saved B1 training/validation metrics")
    torch.set_num_threads(4)
    seed_everything(42)
    initial_fingerprint = state_fingerprint(build_model("b1"))
    shuffle = shuffle_fingerprint(4708)
    checks = []
    counts = {"train": (4708, 3494, 1214), "validation": (524, 389, 135)}
    for row in rows:
        run = output / row["run_directory"]
        run_config, run_summary = read_json(run / "config.json"), read_json(run / "summary.json")
        assert run_summary["status"] == "completed" and not run_summary["test_assisted_stopping"]
        assert run_config["batch_size"] == 128 and run_config["trainable_parameters"] == 99569
        assert run_config["checkpoint_selection"] == "lowest_error_then_loss_then_earlier_epoch"
        assert run_config["stopping_metric"] == "validation_loss"
        assert run_config["task"] == {"threshold": .5, "kind": "binary_classification"}
        assert run_config["training"] == {"learning_rate": row["learning_rate"], "max_epochs": 20,
            "patience": 4, "min_delta": 1e-4, "seed": 42, "device": "cuda", "mode": "selection"}
        assert row["initial_state_sha256"] == initial_fingerprint
        assert row["first_epoch_shuffle_sha256"] == shuffle
        checkpoint_path = output / row["checkpoint"]
        assert sha256_file(checkpoint_path) == row["checkpoint_sha256"]
        history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
        history_check = audit_history(history, run_summary, counts)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert checkpoint["epoch"] == row["best_epoch"] == run_summary["best_epoch"]
        assert checkpoint["selected_metrics"] == run_summary["selected_metrics"]
        assert checkpoint["training_config"] == run_config["training"]
        assert checkpoint["preprocessing"] == bundle.preprocessing.metadata()
        assert checkpoint["optimizer_state"]["param_groups"][0]["lr"] == row["learning_rate"]
        assert checkpoint["model_config"] == build_model("b1").metadata()
        assert all(value.item() == 37 * row["best_epoch"] for key, value in checkpoint["model_state"].items()
                   if key.endswith("num_batches_tracked"))
        seed_everything(42)
        model = model_from_metadata(checkpoint["model_config"])
        restored = restore_checkpoint(checkpoint_path, model, device="cuda")
        errors = {}
        for split in ("train", "validation"):
            metrics = evaluate(model, make_loader(bundle.datasets[split], 128, seed=42, device="cuda"),
                               restored["task_object"], "cuda")
            errors[split] = compare_metrics(metrics, run_summary["selected_metrics"][split])
            assert all(row[f"{split}_{key}"] == value for key, value in run_summary["selected_metrics"][split].items())
        assert sha256_file(checkpoint_path) == row["checkpoint_sha256"]
        checks.append({"learning_rate": row["learning_rate"], "checkpoint": row["checkpoint"],
            "checkpoint_sha256": row["checkpoint_sha256"], "history": history_check,
            "maximum_restored_metric_differences": errors, "full_train_and_validation_reproduced": True,
            "initialization_and_first_epoch_order_match": True})
        print(f"Audited LR {row['learning_rate']:.0e}: checkpoint epoch {row['best_epoch']}, "
              f"full train/validation metric differences {errors}", flush=True)
        del model
    test_files = list((output / "runs").rglob("test_metrics.json"))
    assert test_files == [output / selected["run_directory"] / "test_metrics.json"]
    test_record = read_json(test_files[0])
    final = read_json(output / "final_test.json")
    assert test_record["checkpoint_epoch"] == final["checkpoint_epoch"] == selected["best_epoch"]
    assert test_record["seed"] == final["seed"] == 42
    assert not test_record["test_assisted_stopping"] and not final["test_assisted_stopping"]
    assert test_record["metrics"] == final["metrics"] == summary["final_test"]
    assert final["test_evaluations"] == summary["test_evaluations"] == 1
    predictions = torch.load(output / "final_test_predictions.pt", map_location="cpu", weights_only=True)
    assert predictions["checkpoint_sha256"] == selected["checkpoint_sha256"]
    assert predictions["source_sha256"] == bundle.source_sha256
    assert predictions["threshold"] == .5 and predictions["metrics"] == final["metrics"]
    torch.testing.assert_close(predictions["row_ids"], bundle.row_ids["test"], rtol=0, atol=0)
    torch.testing.assert_close(predictions["labels"], bundle.datasets["test"].tensors[1], rtol=0, atol=0)
    assert predictions["logits"].shape == predictions["probabilities"].shape == (624,)
    expected_metrics = binary_metrics(predictions["labels"], predictions["probabilities"], threshold=.5)
    assert {key: final["metrics"][key] for key in expected_metrics} == expected_metrics
    assert expected_metrics["tp"] + expected_metrics["fn"] == 390
    assert expected_metrics["tn"] + expected_metrics["fp"] == 234
    torch.testing.assert_close(predictions["probabilities"], torch.sigmoid(predictions["logits"]), rtol=1e-6, atol=2e-7)
    logits, labels = predictions["logits"].numpy().astype(np.float64), predictions["labels"].numpy().astype(np.float64)
    independently_computed_loss = float((np.maximum(logits, 0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))).mean())
    assert math.isclose(independently_computed_loss, final["metrics"]["loss"], rel_tol=1e-6, abs_tol=1e-7)
    assert (output / "frozen_b1_settings.json").stat().st_mtime_ns <= test_files[0].stat().st_mtime_ns
    with (output / "tables/learning_rate_selection.csv").open(newline="", encoding="utf-8") as stream:
        table_rows = list(csv.DictReader(stream))
    assert len(table_rows) == 5
    for saved, row in zip(table_rows, rows):
        for key in saved:
            assert math.isclose(float(saved[key]), row[key], rel_tol=1e-12, abs_tol=1e-12)
    for relative in ("figures/validation_loss.png", "figures/validation_loss.pdf",
                     "figures/validation_accuracy.png", "figures/validation_accuracy.pdf",
                     "tables/final_test.csv", "tables/learning_curves.csv", "analysis.md"):
        assert (output / relative).is_file() and (output / relative).stat().st_size > 0
    for relative, fingerprint in previous.items():
        assert sha256_file(ROOT / relative) == fingerprint
    report = {"status": "passed", "step": 10, "output_directory": str(output),
        "device": "cuda", "gpu": torch.cuda.get_device_name(), "trials_audited": 5,
        "selected_learning_rate": selected["learning_rate"], "selected_epoch": selected["best_epoch"],
        "checkpoints": checks, "full_training_validation_restoration": True,
        "test_metrics_recomputed_from_same_pass_saved_predictions": True,
        "independent_test_bce_loss": independently_computed_loss,
        "independent_test_bce_absolute_difference": abs(independently_computed_loss - final["metrics"]["loss"]),
        "test_prediction_sha256": sha256_file(output / "final_test_predictions.pt"),
        "additional_test_model_evaluations": 0, "selected_run_test_records": 1,
        "validation_selection_frozen_before_test": True,
        "all_previous_output_hashes_unchanged": True, "previous_outputs_checked": len(previous),
        "all_supplied_file_hashes_unchanged": True, "data_and_preprocessing_hashes_match": True,
        "tables_and_figure_files_checked": True}
    write_json(output / "verification.json", report)
    write_json(ROOT / "artifacts/b1_checks/latest.json", report)
    print("B1 audit passed. Final test metrics were recomputed from saved predictions; no new test inference.", flush=True)


if __name__ == "__main__":
    main()
