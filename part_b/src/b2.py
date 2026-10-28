"""Five fresh VGG-like batch-size trials with the frozen B1 learning rate."""

from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import platform
import time
from zoneinfo import ZoneInfo

import torch

from assignment_common import TrainingConfig, make_loader, seed_everything
from assignment_common.artifacts import write_checkpoint, write_json
from part_b.src.b1 import LEARNING_RATES, choose_learning_rate, shuffle_fingerprint, state_fingerprint
from part_b.src.data import CLASS_MAPPING, NORMALIZATION, load_data_bundle, sha256_file
from part_b.src.evaluation import evaluate_selected_classifier_once
from part_b.src.models import build_model
from part_b.src.training import train_classifier
from part_b.src.refit import attach_final_checkpoint, audit_final_refit, refit_final_classifier


BATCH_SIZES = (16, 32, 64, 128, 256)
SEED = 42
SELECTION_RULE = "lowest validation classification error, then lower validation loss, then earlier selected epoch, then smaller batch size"


def choose_batch_size(rows):
    batches = [row["batch_size"] for row in rows]
    if (len(rows) != 5 or any(type(value) is not int for value in batches)
            or set(batches) != set(BATCH_SIZES)):
        raise ValueError("B2 requires exactly one successful result for each of the five batch sizes")
    for row in rows:
        for key in ("validation_loss", "validation_accuracy", "validation_classification_error",
                    "validation_sensitivity", "validation_specificity"):
            value = row[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Validation results must be finite")
            if value < 0 or (key != "validation_loss" and value > 1):
                raise ValueError("Loss must be nonnegative and validation proportions in [0,1]")
        if (type(row["best_epoch"]) is not int or type(row["epochs_completed"]) is not int
                or not 1 <= row["best_epoch"] <= row["epochs_completed"] <= 20
                or not math.isclose(row["validation_accuracy"] + row["validation_classification_error"], 1., abs_tol=1e-12)):
            raise ValueError("Incoherent selected epoch or accuracy/error")
    return min(rows, key=lambda row: (row["validation_classification_error"], row["validation_loss"],
                                     row["best_epoch"], row["batch_size"]))


def validate_b1_frozen(frozen, *, source_sha256, dataset_sha256, preprocessing_sha256):
    expected = {"format_version": 1, "architecture": "b1", "selection_seed": 42,
                "b1_batch_size": 128, "probability_threshold": .5, "optimizer": "Adam",
                "max_epochs": 20, "patience": 4, "min_delta": 1e-4,
                "trainable_parameters": 99569, "refit": True,
                "class_mapping": {str(key): value for key, value in CLASS_MAPPING.items()},
                "class_mapping_status": "confirmed_by_instructor_QA",
                "source_sha256": source_sha256, "prepared_dataset_sha256": dataset_sha256,
                "prepared_preprocessing_sha256": preprocessing_sha256}
    if any(frozen.get(key) != value for key, value in expected.items()):
        raise ValueError("B1 frozen controls or data/preprocessing provenance disagree")
    if (frozen.get("selected_learning_rate") not in LEARNING_RATES
            or type(frozen.get("selected_epoch")) is not int or not 1 <= frozen["selected_epoch"] <= 20):
        raise ValueError("B1 must supply a valid selected learning rate and epoch")
    return frozen["selected_learning_rate"]


def check_settings(config):
    settings = config["settings"]
    expected = {"batch_size_grid": list(BATCH_SIZES), "probability_threshold": .5,
                "normalization": NORMALIZATION, "loss": "BCEWithLogitsLoss",
                "output": "unrestricted_binary_logit", "stopping_monitor": "validation_loss",
                "checkpoint_selection": "validation_error_then_loss_then_earlier_epoch",
                "class_mapping_assumption": {str(key): value for key, value in CLASS_MAPPING.items()}}
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError("B2 configuration disagrees with the brief or recorded protocol")
    if (config["selection_seed"] != SEED or config["device"] != "cuda"
            or config["training"] != {"optimizer": "Adam", "max_epochs": 20, "patience": 4, "min_delta": 1e-4}):
        raise ValueError("B2 uses seed 42, CUDA and Adam with 20/4/1e-4 stopping")


def result_row(batch_size, learning_rate, result, *, output, initial_sha256, shuffle_sha256, peak_memory, elapsed_seconds):
    return {"batch_size": batch_size, "learning_rate": learning_rate, "seed": SEED,
        "best_epoch": result.best_epoch, "epochs_completed": result.epochs_completed,
        "stopped_early": result.stopped_early,
        **{f"validation_{key}": value for key, value in result.selected_metrics["validation"].items()},
        **{f"train_{key}": value for key, value in result.selected_metrics["train"].items()},
        "trainable_parameters": 18185, "initial_state_sha256": initial_sha256,
        "first_epoch_shuffle_sha256": shuffle_sha256,
        "optimizer_steps_per_epoch": math.ceil(4708 / batch_size),
        "optimizer_steps_to_selected_epoch": math.ceil(4708 / batch_size) * result.best_epoch,
        "optimizer_steps_completed": math.ceil(4708 / batch_size) * result.epochs_completed,
        "last_training_batch_size": 4708 % batch_size or batch_size,
        "checkpoint": result.checkpoint_path.relative_to(output).as_posix(),
        "run_directory": result.checkpoint_path.parent.relative_to(output).as_posix(),
        "checkpoint_sha256": sha256_file(result.checkpoint_path),
        "elapsed_seconds": elapsed_seconds, "peak_allocated_gpu_mib": peak_memory}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_b2(output_directory, prepared_directory, b1_directory, config):
    check_settings(config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this B2 experiment")
    output, prepared, b1 = (Path(path).resolve() for path in (output_directory, prepared_directory, b1_directory))
    if output.exists():
        raise FileExistsError(f"Choose a new B2 output directory: {output}")
    torch.set_num_threads(4)
    bundle = load_data_bundle(prepared)
    dataset_hash, preprocessing_hash = sha256_file(prepared / "datasets.pt"), sha256_file(prepared / "preprocessing.pt")
    b1_frozen = _read(b1 / "frozen_b1_settings.json")
    learning_rate = validate_b1_frozen(b1_frozen, source_sha256=bundle.source_sha256,
        dataset_sha256=dataset_hash, preprocessing_sha256=preprocessing_hash)
    b1_summary = _read(b1 / "summary.json")
    if (b1_summary["status"] != "completed" or _read(b1 / "status.json")["status"] != "completed"
            or _read(b1 / "verification.json")["status"] != "passed"):
        raise ValueError("B2 requires the completed and verified B1 experiment")
    b1_selected = choose_learning_rate(_read(b1 / "learning_rate_search.json"))
    if (b1_selected != b1_summary["selected"] or b1_selected["learning_rate"] != learning_rate
            or b1_selected["best_epoch"] != b1_frozen["selected_epoch"]
            or b1_selected["checkpoint_sha256"] != b1_frozen["checkpoint_sha256"]
            or sha256_file(b1 / b1_frozen["checkpoint"]) != b1_frozen["checkpoint_sha256"]):
        raise ValueError("B1's frozen choice must agree with its validation results and checkpoint")
    overlap = next(item["shared_unique_images"] for item in bundle.report["duplicates"]["cross_split"]
                   if item["splits"] == ["train", "validation"])
    output.mkdir(parents=True, exist_ok=False)
    configuration = {"created_at_singapore": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "architecture": "b2", "learning_rate": learning_rate, "batch_size_grid": list(BATCH_SIZES),
        "seed": SEED, "optimizer": "Adam", "max_epochs": 20, "patience": 4, "min_delta": 1e-4,
        "probability_threshold": .5, "stopping_monitor": "validation_loss", "selection_rule": SELECTION_RULE,
        "prepared_directory": str(prepared), "source_sha256": bundle.source_sha256,
        "prepared_dataset_sha256": dataset_hash, "prepared_preprocessing_sha256": preprocessing_hash,
        "b1_directory": str(b1), "b1_frozen_settings": b1_frozen,
        "b1_frozen_settings_sha256": sha256_file(b1 / "frozen_b1_settings.json"),
        "provided_partitions": {split: len(dataset) for split, dataset in bundle.datasets.items()},
        "training_validation_overlap_unique_images": overlap,
        "class_mapping": dict(CLASS_MAPPING), "class_mapping_status": "confirmed_by_instructor_QA",
        "augmentation": None, "class_weighting": None, "refit": True,
        "fresh_model_and_loader_per_trial": True, "same_sample_order_different_batch_grouping": True,
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()}}
    write_json(output / "configuration.json", configuration)
    write_checkpoint(output / "preprocessing.pt", bundle.preprocessing.metadata())
    write_json(output / "status.json", {"status": "running", "stage": "batch_size_search"})
    rows, initial_reference, shuffle_reference = [], None, None
    started = time.perf_counter()
    try:
        for batch_size in BATCH_SIZES:
            seed_everything(SEED)
            model = build_model("b2")
            initial = state_fingerprint(model)
            order = shuffle_fingerprint(len(bundle.datasets["train"]), batch_size=batch_size)
            if initial_reference is None:
                initial_reference, shuffle_reference = initial, order
            if initial != initial_reference or order != shuffle_reference:
                raise RuntimeError("B2 candidates must share initial weights and underlying row order")
            training = make_loader(bundle.datasets["train"], batch_size, training=True, seed=SEED, device="cuda")
            validation = make_loader(bundle.datasets["validation"], batch_size, seed=SEED, device="cuda")
            run = output / "runs" / f"batch_{batch_size:03d}"
            def on_epoch(row):
                write_json(output / "progress.json", {"stage": "batch_size_search", "completed_trials": len(rows),
                    "total_trials": 5, "batch_size": batch_size, "epoch": row["epoch"],
                    "validation_loss": row["validation_loss"], "validation_accuracy": row["validation_accuracy"]})
                print(f"Batch {batch_size:3} epoch {row['epoch']:2}: train loss {row['train_loss']:.5f}, "
                      f"val loss {row['validation_loss']:.5f}, val accuracy {row['validation_accuracy']:.4%}" +
                      (" [checkpoint]" if row['checkpoint_selected'] else ""), flush=True)
            torch.cuda.reset_peak_memory_stats()
            result = train_classifier(model, training, config=TrainingConfig(learning_rate=learning_rate,
                max_epochs=20, patience=4, min_delta=1e-4, seed=SEED, device="cuda"),
                validation_loader=validation, run_directory=run, threshold=.5,
                preprocessing=bundle.preprocessing.metadata(), model_config=model.metadata(), on_epoch=on_epoch)
            run_summary = _read(run / "summary.json")
            row = result_row(batch_size, learning_rate, result, output=output, initial_sha256=initial,
                shuffle_sha256=order, peak_memory=torch.cuda.max_memory_allocated() / 1024**2,
                elapsed_seconds=run_summary["elapsed_seconds"])
            rows.append(row)
            write_json(output / "batch_size_search.json", rows)
            print(f"Completed {len(rows)}/5: batch {batch_size}, selected epoch {row['best_epoch']}/"
                  f"{row['epochs_completed']}, error {row['validation_classification_error']:.4%}", flush=True)
            del model
        selected = choose_batch_size(rows)
        frozen = {"format_version": 1, "architecture": "b2", "selected_learning_rate": learning_rate,
            "selected_batch_size": selected["batch_size"], "selection_seed": SEED, "probability_threshold": .5,
            "optimizer": "Adam", "max_epochs": 20, "patience": 4, "min_delta": 1e-4,
            "selection_rule": SELECTION_RULE, "selected_epoch": selected["best_epoch"],
            "validation_classification_error": selected["validation_classification_error"],
            "validation_loss": selected["validation_loss"], "trainable_parameters": 18185,
            "checkpoint": selected["checkpoint"], "checkpoint_sha256": selected["checkpoint_sha256"],
            "source_sha256": bundle.source_sha256, "prepared_dataset_sha256": dataset_hash,
            "prepared_preprocessing_sha256": preprocessing_hash,
            "b1_frozen_settings_sha256": configuration["b1_frozen_settings_sha256"],
            "b1_selected_learning_rate": learning_rate, "b1_checkpoint_sha256": b1_frozen["checkpoint_sha256"],
            "class_mapping": dict(CLASS_MAPPING), "class_mapping_status": "confirmed_by_instructor_QA",
            "training_validation_overlap_unique_images": overlap, "refit": False,
            "handoff": "B3 must reuse the B1-selected learning rate and B2-selected batch size"}
        write_json(output / "frozen_b2_settings.json", frozen)
        write_json(output / "status.json", {"status": "running", "stage": "final_test", "batch_size": selected["batch_size"]})
        final_record = refit_final_classifier(output, bundle, selected, architecture="b2", batch_size=selected["batch_size"])
        frozen = attach_final_checkpoint(frozen, final_record)
        write_json(output / "frozen_b2_settings.json", frozen)
        final_test = final_record["metrics"]
        summary = {"status": "completed", "architecture": "b2", "selected": selected,
            "selection_rule": SELECTION_RULE, "batch_size_grid": list(BATCH_SIZES), "trials": 5,
            "seed": SEED, "learning_rate": learning_rate, "trainable_parameters": 18185,
            "final_test": final_test, "test_evaluations": 1, "refit": True,
            "final_checkpoint": final_record["checkpoint"], "final_checkpoint_sha256": final_record["checkpoint_sha256"],
            "source_sha256": bundle.source_sha256, "training_validation_overlap_unique_images": overlap,
            "elapsed_seconds": time.perf_counter() - started}
        write_json(output / "summary.json", summary)
        from part_b.src.b2_exports import export_b2
        export_b2(output)
        write_json(output / "verification.json", audit_final_refit(output))
        write_json(output / "status.json", {"status": "completed", "stage": "completed"})
        write_json(output / "progress.json", {"stage": "completed", "completed_trials": 5, "total_trials": 5,
            "selected_batch_size": selected["batch_size"], "learning_rate": learning_rate})
        print(f"B2 complete: batch {selected['batch_size']}, LR {learning_rate:.0e}, epoch {selected['best_epoch']}; "
              f"test accuracy {final_test['accuracy']:.4%}, sensitivity {final_test['sensitivity']:.4%}, "
              f"specificity {final_test['specificity']:.4%}", flush=True)
        return summary
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise
