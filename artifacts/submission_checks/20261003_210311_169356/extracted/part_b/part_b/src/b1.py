"""Five fresh B1 learning-rate trials, validation selection, and one final test."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from zoneinfo import ZoneInfo

import torch
from torch.utils.data import TensorDataset

from assignment_common import TrainingConfig, evaluate_final_checkpoint, make_loader, seed_everything
from assignment_common.artifacts import write_checkpoint, write_json
from part_b.src.data import CLASS_MAPPING, NORMALIZATION, load_data_bundle, sha256_file
from part_b.src.models import build_model
from part_b.src.training import train_classifier


LEARNING_RATES = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
BATCH_SIZE = 128
SEED = 42
SELECTION_RULE = "lowest validation classification error, then lower validation loss, then earlier selected epoch, then smaller learning rate"


def choose_learning_rate(rows: list[dict]) -> dict:
    rates = [row["learning_rate"] for row in rows]
    if len(rates) != len(LEARNING_RATES) or set(rates) != set(LEARNING_RATES):
        raise ValueError("B1 requires exactly one successful result for each of the five learning rates")
    for row in rows:
        for key in ("validation_loss", "validation_accuracy", "validation_classification_error",
                    "validation_sensitivity", "validation_specificity"):
            value = row[key]
            if not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError("All validation results must be finite")
            if value < 0 or (key != "validation_loss" and value > 1):
                raise ValueError("Validation probabilities/error must be in [0,1] and loss nonnegative")
        if (not math.isclose(row["validation_accuracy"] + row["validation_classification_error"], 1., abs_tol=1e-12)
                or type(row["best_epoch"]) is not int or type(row["epochs_completed"]) is not int
                or not 1 <= row["best_epoch"] <= row["epochs_completed"] <= 20):
            raise ValueError("Incoherent validation accuracy/error or checkpoint epochs")
    return min(rows, key=lambda row: (row["validation_classification_error"], row["validation_loss"],
                                     row["best_epoch"], row["learning_rate"]))


def state_fingerprint(model) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(list(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def shuffle_fingerprint(count: int, batch_size: int = BATCH_SIZE, seed: int = SEED) -> str:
    """Use the actual DataLoader RNG sequence on row IDs, including base-seed draw."""
    rows = TensorDataset(torch.arange(count))
    batches = make_loader(rows, batch_size, training=True, seed=seed)
    order = torch.cat([batch[0] for batch in batches])
    return hashlib.sha256(order.numpy().tobytes()).hexdigest()


def check_settings(config: dict) -> None:
    settings = config["settings"]
    expected = {"learning_rate_grid": list(LEARNING_RATES), "b1_batch_size": BATCH_SIZE,
                "probability_threshold": .5, "output": "unrestricted_binary_logit",
                "loss": "BCEWithLogitsLoss", "normalization": NORMALIZATION,
                "stopping_monitor": "validation_loss",
                "checkpoint_selection": "validation_error_then_loss_then_earlier_epoch",
                "class_mapping_assumption": {str(key): value for key, value in CLASS_MAPPING.items()}}
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError("B1 configuration disagrees with the assignment or recorded protocol")
    if (config["selection_seed"] != SEED or config["device"] != "cuda"
            or config["training"] != {"optimizer": "Adam", "max_epochs": 20, "patience": 4, "min_delta": 1e-4}):
        raise ValueError("B1 uses seed 42, CUDA, Adam, and the assignment's 20/4/1e-4 rules")


def result_row(learning_rate: float, result, *, output: Path, initial_sha256: str,
               shuffle_sha256: str, peak_memory: float, elapsed_seconds: float) -> dict:
    metrics = result.selected_metrics["validation"]
    row = {"learning_rate": learning_rate, "best_epoch": result.best_epoch,
           "epochs_completed": result.epochs_completed, "stopped_early": result.stopped_early,
           **{f"validation_{key}": value for key, value in metrics.items()},
           **{f"train_{key}": value for key, value in result.selected_metrics["train"].items()},
           "seed": SEED, "batch_size": BATCH_SIZE, "trainable_parameters": 99569,
           "initial_state_sha256": initial_sha256, "first_epoch_shuffle_sha256": shuffle_sha256,
           "checkpoint": str(result.checkpoint_path.relative_to(output)),
           "run_directory": str(result.checkpoint_path.parent.relative_to(output)),
           "checkpoint_sha256": sha256_file(result.checkpoint_path),
           "peak_allocated_gpu_mib": peak_memory, "elapsed_seconds": elapsed_seconds}
    return row


def test_selected_model_once(run_directory: Path, model, bundle, destination: Path, *, device="cuda") -> dict:
    """Collect predictions in the one shared, guarded full-test evaluation pass."""
    if (run_directory / "test_metrics.json").exists() or destination.exists():
        raise FileExistsError("Reuse the saved final test result and predictions")
    outputs, captured_probabilities = [], []
    def capture(module, inputs, output):
        outputs.append(output.detach().cpu().clone())
        captured_probabilities.append(torch.sigmoid(output.detach()).cpu().clone())
    handle = model.register_forward_hook(capture)
    try:
        metrics = evaluate_final_checkpoint(run_directory, model,
            make_loader(bundle.datasets["test"], BATCH_SIZE, seed=SEED, device=device), device=device)
    finally:
        handle.remove()
    logits = torch.cat(outputs)
    labels = bundle.datasets["test"].tensors[1]
    if len(logits) != len(labels) or metrics["count"] != len(labels):
        raise RuntimeError("Final test prediction capture must cover every supplied row exactly once")
    # Same device, sigmoid implementation, and batch sizes as the metric accumulator.
    probabilities = torch.cat(captured_probabilities)
    write_checkpoint(destination, {"format_version": 1, "architecture": "b1",
        "checkpoint_sha256": sha256_file(run_directory / "best_model.pt"),
        "source_sha256": bundle.source_sha256, "row_ids": bundle.row_ids["test"],
        "labels": labels, "logits": logits, "probabilities": probabilities,
        "threshold": .5, "prediction_rule": "sigmoid(logit) >= 0.5", "metrics": metrics,
        "class_mapping": dict(CLASS_MAPPING), "class_mapping_status": "assumed_not_archive_metadata",
        "capture": "Same forward pass as the single guarded final test evaluation"})
    return metrics


def run_b1(output_directory: Path | str, prepared_directory: Path | str, config: dict) -> dict:
    check_settings(config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this B1 experiment")
    output, prepared = Path(output_directory).resolve(), Path(prepared_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new B1 output directory: {output}")
    torch.set_num_threads(4)
    bundle = load_data_bundle(prepared)
    training_validation_overlap = next(item["shared_unique_images"] for item in bundle.report["duplicates"]["cross_split"]
                                       if item["splits"] == ["train", "validation"])
    output.mkdir(parents=True, exist_ok=False)
    configuration = {"created_at_singapore": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "architecture": "b1", "learning_rate_grid": list(LEARNING_RATES), "batch_size": BATCH_SIZE,
        "seed": SEED, "optimizer": "Adam", "max_epochs": 20, "patience": 4, "min_delta": 1e-4,
        "probability_threshold": .5, "stopping_monitor": "validation_loss", "selection_rule": SELECTION_RULE,
        "within_run_checkpoint_selection": "validation error, loss, earlier epoch",
        "prepared_directory": str(prepared), "source_sha256": bundle.source_sha256,
        "prepared_dataset_sha256": sha256_file(prepared / "datasets.pt"),
        "prepared_preprocessing_sha256": sha256_file(prepared / "preprocessing.pt"),
        "provided_partitions": {split: len(dataset) for split, dataset in bundle.datasets.items()},
        "class_mapping": dict(CLASS_MAPPING), "class_mapping_status": "assumed_not_archive_metadata",
        "training_validation_overlap_unique_images": training_validation_overlap, "augmentation": None, "class_weighting": None,
        "fresh_model_and_loader_per_trial": True, "refit": False,
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()}}
    write_json(output / "configuration.json", configuration)
    write_checkpoint(output / "preprocessing.pt", bundle.preprocessing.metadata())
    write_json(output / "status.json", {"status": "running", "stage": "learning_rate_search"})
    started = time.perf_counter()
    rows = []
    initial_reference = None
    try:
        for learning_rate in LEARNING_RATES:
            seed_everything(SEED)
            model = build_model("b1")
            initial = state_fingerprint(model)
            if initial_reference is None:
                initial_reference = initial
            elif initial != initial_reference:
                raise RuntimeError("Every B1 candidate must start from identical weights")
            order = shuffle_fingerprint(len(bundle.datasets["train"]))
            training = make_loader(bundle.datasets["train"], BATCH_SIZE, training=True, seed=SEED, device="cuda")
            validation = make_loader(bundle.datasets["validation"], BATCH_SIZE, seed=SEED, device="cuda")
            run = output / "runs" / f"lr_{learning_rate:.0e}"
            def on_epoch(row):
                write_json(output / "progress.json", {"stage": "learning_rate_search", "completed_trials": len(rows),
                    "total_trials": 5, "learning_rate": learning_rate, "epoch": row["epoch"],
                    "validation_loss": row["validation_loss"], "validation_accuracy": row["validation_accuracy"]})
                print(f"LR {learning_rate:.0e} epoch {row['epoch']:2}: train loss {row['train_loss']:.5f}, "
                      f"val loss {row['validation_loss']:.5f}, val accuracy {row['validation_accuracy']:.4%}" +
                      (" [checkpoint]" if row['checkpoint_selected'] else ""), flush=True)
            torch.cuda.reset_peak_memory_stats()
            result = train_classifier(model, training, config=TrainingConfig(learning_rate=learning_rate,
                max_epochs=20, patience=4, min_delta=1e-4, seed=SEED, device="cuda"),
                run_directory=run, validation_loader=validation, threshold=.5,
                preprocessing=bundle.preprocessing.metadata(), model_config=model.metadata(), on_epoch=on_epoch)
            run_summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
            row = result_row(learning_rate, result, output=output, initial_sha256=initial, shuffle_sha256=order,
                peak_memory=torch.cuda.max_memory_allocated() / 1024**2, elapsed_seconds=run_summary["elapsed_seconds"])
            rows.append(row)
            write_json(output / "learning_rate_search.json", rows)
            print(f"Completed {len(rows)}/5: LR {learning_rate:.0e}, selected epoch "
                  f"{row['best_epoch']}/{row['epochs_completed']}, error {row['validation_classification_error']:.4%}", flush=True)
            del model
        selected = choose_learning_rate(rows)
        frozen = {"format_version": 1, "architecture": "b1", "selected_learning_rate": selected["learning_rate"],
            "selection_seed": SEED, "b1_batch_size": BATCH_SIZE, "probability_threshold": .5,
            "optimizer": "Adam", "max_epochs": 20, "patience": 4, "min_delta": 1e-4,
            "selection_rule": SELECTION_RULE, "selected_epoch": selected["best_epoch"],
            "validation_classification_error": selected["validation_classification_error"],
            "validation_loss": selected["validation_loss"], "trainable_parameters": 99569,
            "checkpoint": selected["checkpoint"], "checkpoint_sha256": selected["checkpoint_sha256"],
            "source_sha256": bundle.source_sha256, "prepared_dataset_sha256": configuration["prepared_dataset_sha256"],
            "prepared_preprocessing_sha256": configuration["prepared_preprocessing_sha256"],
            "class_mapping": dict(CLASS_MAPPING), "class_mapping_status": "assumed_not_archive_metadata",
            "training_validation_overlap_unique_images": training_validation_overlap, "refit": False,
            "handoff": "B2 and B3 must reuse this validation-selected learning rate"}
        # Freeze validation-based choice before constructing or accessing the test loader.
        write_json(output / "frozen_b1_settings.json", frozen)
        write_json(output / "status.json", {"status": "running", "stage": "final_test", "learning_rate": selected["learning_rate"]})
        final_model = build_model("b1")
        final_test = test_selected_model_once(output / selected["run_directory"], final_model, bundle,
                                             output / "final_test_predictions.pt")
        write_json(output / "final_test.json", {"selected_learning_rate": selected["learning_rate"],
            "checkpoint": selected["checkpoint"], "checkpoint_sha256": selected["checkpoint_sha256"],
            "checkpoint_epoch": selected["best_epoch"], "seed": SEED, "metrics": final_test,
            "test_evaluations": 1, "test_assisted_stopping": False,
            "prediction_artifact": "final_test_predictions.pt"})
        summary = {"status": "completed", "architecture": "b1", "selected": selected,
            "selection_rule": SELECTION_RULE, "learning_rate_grid": list(LEARNING_RATES), "trials": 5,
            "seed": SEED, "batch_size": BATCH_SIZE, "trainable_parameters": 99569, "final_test": final_test,
            "test_evaluations": 1, "refit": False, "elapsed_seconds": time.perf_counter() - started,
            "source_sha256": bundle.source_sha256, "training_validation_overlap_unique_images": training_validation_overlap}
        write_json(output / "summary.json", summary)
        from part_b.src.b1_exports import export_b1
        export_b1(output)
        write_json(output / "progress.json", {"stage": "completed", "completed_trials": 5, "total_trials": 5,
            "selected_learning_rate": selected["learning_rate"]})
        write_json(output / "status.json", {"status": "completed", "stage": "completed"})
        print(f"B1 complete: LR {selected['learning_rate']:.0e}, epoch {selected['best_epoch']}; "
              f"test accuracy {final_test['accuracy']:.4%}, sensitivity {final_test['sensitivity']:.4%}, "
              f"specificity {final_test['specificity']:.4%}", flush=True)
        return summary
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise
