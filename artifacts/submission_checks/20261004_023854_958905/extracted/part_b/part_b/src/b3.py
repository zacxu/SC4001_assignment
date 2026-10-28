"""Train the compact Figure B3 ResNet with frozen B1/B2 settings and compare all three."""

from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import platform
import time
from zoneinfo import ZoneInfo

import torch

from assignment_common import TrainingConfig, make_loader, restore_checkpoint, seed_everything
from assignment_common.artifacts import write_checkpoint, write_json
from part_b.src.b1 import choose_learning_rate, shuffle_fingerprint, state_fingerprint
from part_b.src.b2 import BATCH_SIZES, choose_batch_size, validate_b1_frozen
from part_b.src.complexity import FLOPS_CONVENTION, profile_model
from part_b.src.data import CLASS_MAPPING, load_data_bundle, sha256_file
from part_b.src.evaluation import evaluate_selected_classifier_once
from part_b.src.models import EXPECTED_PARAMETERS, build_model
from part_b.src.training import train_classifier


SEED = 42
SELECTION_RULE = "lowest validation classification error, then lower validation loss, then earlier epoch"
LABELS = {"b1": "B1 AlexNet-like", "b2": "B2 VGG-like", "b3": "B3 ResNet-like"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_b2_frozen(b2, b1, *, source_sha256, dataset_sha256, preprocessing_sha256):
    learning_rate = validate_b1_frozen(b1, source_sha256=source_sha256,
        dataset_sha256=dataset_sha256, preprocessing_sha256=preprocessing_sha256)
    expected = {"format_version": 1, "architecture": "b2", "selected_learning_rate": learning_rate,
        "selection_seed": SEED, "probability_threshold": .5, "optimizer": "Adam", "max_epochs": 20,
        "patience": 4, "min_delta": 1e-4, "trainable_parameters": 18185, "refit": False,
        "class_mapping": {str(key): value for key, value in CLASS_MAPPING.items()},
        "class_mapping_status": "assumed_not_archive_metadata", "source_sha256": source_sha256,
        "prepared_dataset_sha256": dataset_sha256, "prepared_preprocessing_sha256": preprocessing_sha256,
        "b1_selected_learning_rate": learning_rate, "b1_checkpoint_sha256": b1["checkpoint_sha256"]}
    if any(b2.get(key) != value for key, value in expected.items()):
        raise ValueError("B2 frozen controls or B1/data lineage disagree")
    if (type(b2.get("selected_batch_size")) is not int or b2["selected_batch_size"] not in BATCH_SIZES
            or type(b2.get("selected_epoch")) is not int or not 1 <= b2["selected_epoch"] <= 20):
        raise ValueError("B2 must provide a valid selected batch and epoch")
    return learning_rate, b2["selected_batch_size"]


def comparison_rows(records, complexities):
    """Combine saved final results with shape-checked complexity, without inference."""
    if set(records) != set(LABELS) or set(complexities) != set(LABELS):
        raise ValueError("Architecture comparison requires B1, B2 and B3")
    rows = []
    for architecture, label in LABELS.items():
        record, complexity = records[architecture], complexities[architecture]
        metrics = record["metrics"]
        if (record["test_evaluations"] != 1 or record["seed"] != SEED
                or complexity["architecture"] != architecture
                or complexity["trainable_parameters"] != EXPECTED_PARAMETERS[architecture]
                or complexity["convention"] != FLOPS_CONVENTION
                or not complexity["shape_formula_matches_profiler"]
                or complexity["inference_flops"] != complexity["profiler_flops"]):
            raise ValueError("Comparison records must share the verified evaluation/FLOPs convention")
        tp, tn, fp, fn = (metrics[key] for key in ("tp", "tn", "fp", "fn"))
        if (any(type(value) is not int or value < 0 for value in (tp, tn, fp, fn))
                or tp + fn != 390 or tn + fp != 234 or metrics["count"] != 624):
            raise ValueError("Final test metrics must cover the provided 624-image test partition")
        expected = {"accuracy": (tp + tn) / 624, "classification_error": 1 - (tp + tn) / 624,
                    "sensitivity": tp / 390, "specificity": tn / 234}
        if any(not math.isclose(metrics[key], value, rel_tol=0, abs_tol=1e-12) for key, value in expected.items()):
            raise ValueError("Incoherent saved final test metrics")
        rows.append({"architecture": architecture, "model": label, "learning_rate": record["learning_rate"],
            "batch_size": record["batch_size"], "seed": record["seed"], "checkpoint_epoch": record["checkpoint_epoch"],
            "accuracy": metrics["accuracy"], "sensitivity": metrics["sensitivity"], "specificity": metrics["specificity"],
            "classification_error": metrics["classification_error"], "trainable_parameters": complexity["trainable_parameters"],
            "inference_flops": complexity["inference_flops"], "convolution_and_linear_macs": complexity["convolution_and_linear_macs"],
            "residual_addition_flops": complexity["residual_addition_flops"], "flops_convention": FLOPS_CONVENTION,
            "test_loss": metrics["loss"], "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "checkpoint_sha256": record["checkpoint_sha256"]})
    return rows


def _upstream(directory, architecture, frozen):
    summary = read_json(directory / "summary.json")
    if (summary["status"] != "completed" or read_json(directory / "status.json")["status"] != "completed"
            or read_json(directory / "verification.json")["status"] != "passed"):
        raise ValueError(f"B3 requires completed, verified {architecture.upper()} artifacts")
    choose = choose_learning_rate if architecture == "b1" else choose_batch_size
    table = "learning_rate_search.json" if architecture == "b1" else "batch_size_search.json"
    selected = choose(read_json(directory / table))
    mapping = {"learning_rate": "selected_learning_rate", "best_epoch": "selected_epoch", "checkpoint": "checkpoint",
               "checkpoint_sha256": "checkpoint_sha256", "validation_loss": "validation_loss",
               "validation_classification_error": "validation_classification_error"}
    if architecture == "b2":
        mapping["batch_size"] = "selected_batch_size"
    if (selected != summary["selected"] or any(selected[key] != frozen[value] for key, value in mapping.items())
            or sha256_file(directory / frozen["checkpoint"]) != frozen["checkpoint_sha256"]):
        raise ValueError(f"{architecture.upper()}'s frozen selection must match its validation table/checkpoint")


def run_b3(output_directory, prepared_directory, b1_directory, b2_directory, config):
    if (config["selection_seed"] != SEED or config["device"] != "cuda"
            or config["training"] != {"optimizer": "Adam", "max_epochs": 20, "patience": 4, "min_delta": 1e-4}
            or config["settings"]["probability_threshold"] != .5
            or config["settings"]["loss"] != "BCEWithLogitsLoss"
            or config["settings"]["output"] != "unrestricted_binary_logit"):
        raise ValueError("B3 requires the recorded seed/CUDA/classification/stopping controls")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for B3 training")
    output, prepared, b1, b2 = (Path(path).resolve() for path in (output_directory, prepared_directory, b1_directory, b2_directory))
    if output.exists():
        raise FileExistsError(f"Choose a new B3 output directory: {output}")
    torch.set_num_threads(4)
    bundle = load_data_bundle(prepared)
    data_hash, preprocessing_hash = sha256_file(prepared / "datasets.pt"), sha256_file(prepared / "preprocessing.pt")
    frozen1, frozen2 = (read_json(directory / f"frozen_{architecture}_settings.json") for directory, architecture in ((b1, "b1"), (b2, "b2")))
    learning_rate, batch_size = validate_b2_frozen(frozen2, frozen1, source_sha256=bundle.source_sha256,
        dataset_sha256=data_hash, preprocessing_sha256=preprocessing_hash)
    hash1, hash2 = sha256_file(b1 / "frozen_b1_settings.json"), sha256_file(b2 / "frozen_b2_settings.json")
    if frozen2["b1_frozen_settings_sha256"] != hash1:
        raise ValueError("B2 must descend from the exact supplied B1 frozen settings")
    _upstream(b1, "b1", frozen1)
    _upstream(b2, "b2", frozen2)
    output.mkdir(parents=True, exist_ok=False)
    configuration = {"created_at_singapore": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "architecture": "b3", "learning_rate": learning_rate, "batch_size": batch_size, "seed": SEED,
        "optimizer": "Adam", "max_epochs": 20, "patience": 4, "min_delta": 1e-4, "probability_threshold": .5,
        "stopping_monitor": "validation_loss", "selection_rule": SELECTION_RULE,
        "prepared_directory": str(prepared), "source_sha256": bundle.source_sha256,
        "prepared_dataset_sha256": data_hash, "prepared_preprocessing_sha256": preprocessing_hash,
        "b1_directory": str(b1), "b2_directory": str(b2), "b1_frozen_settings": frozen1, "b2_frozen_settings": frozen2,
        "b1_frozen_settings_sha256": hash1, "b2_frozen_settings_sha256": hash2,
        "provided_partitions": {key: len(value) for key, value in bundle.datasets.items()},
        "training_validation_overlap_unique_images": 8, "class_mapping": dict(CLASS_MAPPING),
        "class_mapping_status": "assumed_not_archive_metadata", "augmentation": None, "class_weighting": None, "refit": False,
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()}}
    write_json(output / "configuration.json", configuration)
    write_checkpoint(output / "preprocessing.pt", bundle.preprocessing.metadata())
    write_json(output / "status.json", {"status": "running", "stage": "training"})
    started = time.perf_counter()
    try:
        seed_everything(SEED)
        model = build_model("b3")
        initial, order = state_fingerprint(model), shuffle_fingerprint(4708, batch_size=batch_size)
        training = make_loader(bundle.datasets["train"], batch_size, training=True, seed=SEED, device="cuda")
        validation = make_loader(bundle.datasets["validation"], batch_size, seed=SEED, device="cuda")
        def on_epoch(row):
            write_json(output / "progress.json", {"stage": "training", "epoch": row["epoch"],
                "validation_loss": row["validation_loss"], "validation_accuracy": row["validation_accuracy"]})
            print(f"B3 epoch {row['epoch']:2}: train loss {row['train_loss']:.5f}, val loss {row['validation_loss']:.5f}, "
                  f"val accuracy {row['validation_accuracy']:.4%}" + (" [checkpoint]" if row['checkpoint_selected'] else ""), flush=True)
        torch.cuda.reset_peak_memory_stats()
        result = train_classifier(model, training, config=TrainingConfig(learning_rate=learning_rate,
            max_epochs=20, patience=4, min_delta=1e-4, seed=SEED, device="cuda"), validation_loader=validation,
            run_directory=output / "run", threshold=.5, preprocessing=bundle.preprocessing.metadata(),
            model_config=model.metadata(), on_epoch=on_epoch)
        selected = {"architecture": "b3", "learning_rate": learning_rate, "batch_size": batch_size, "seed": SEED,
            "best_epoch": result.best_epoch, "epochs_completed": result.epochs_completed, "stopped_early": result.stopped_early,
            **{f"{split}_{key}": value for split, metrics in result.selected_metrics.items() for key, value in metrics.items()},
            "trainable_parameters": 77169, "initial_state_sha256": initial, "first_epoch_shuffle_sha256": order,
            "optimizer_steps_per_epoch": math.ceil(4708 / batch_size),
            "optimizer_steps_to_selected_epoch": math.ceil(4708 / batch_size) * result.best_epoch,
            "optimizer_steps_completed": math.ceil(4708 / batch_size) * result.epochs_completed,
            "last_training_batch_size": 4708 % batch_size or batch_size,
            "checkpoint": result.checkpoint_path.relative_to(output).as_posix(), "run_directory": "run",
            "checkpoint_sha256": sha256_file(result.checkpoint_path),
            "peak_allocated_gpu_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "elapsed_seconds": read_json(output / "run/summary.json")["elapsed_seconds"]}
        write_json(output / "selected.json", selected)
        frozen = {"format_version": 1, "architecture": "b3", "selected_learning_rate": learning_rate,
            "selected_batch_size": batch_size, "selection_seed": SEED, "probability_threshold": .5,
            "optimizer": "Adam", "max_epochs": 20, "patience": 4, "min_delta": 1e-4,
            "selected_epoch": result.best_epoch, "selection_rule": SELECTION_RULE, "trainable_parameters": 77169,
            "checkpoint": selected["checkpoint"], "checkpoint_sha256": selected["checkpoint_sha256"],
            "validation_loss": selected["validation_loss"], "validation_classification_error": selected["validation_classification_error"],
            "source_sha256": bundle.source_sha256, "prepared_dataset_sha256": data_hash,
            "prepared_preprocessing_sha256": preprocessing_hash, "b1_frozen_settings_sha256": hash1,
            "b2_frozen_settings_sha256": hash2, "class_mapping": dict(CLASS_MAPPING),
            "class_mapping_status": "assumed_not_archive_metadata", "training_validation_overlap_unique_images": 8,
            "refit": False, "last_convolution": "features.block64.conv2.conv",
            "handoff": "B4 uses this exact selected checkpoint alongside saved B1/B2 checkpoints"}
        write_json(output / "frozen_b3_settings.json", frozen)
        write_json(output / "status.json", {"status": "running", "stage": "final_test"})
        final_model = build_model("b3")
        final_metrics = evaluate_selected_classifier_once(output / "run", final_model, bundle,
            output / "final_test_predictions.pt", batch_size=batch_size, seed=SEED)
        final_record = {"learning_rate": learning_rate, "batch_size": batch_size, "checkpoint": selected["checkpoint"],
            "checkpoint_sha256": selected["checkpoint_sha256"], "checkpoint_epoch": result.best_epoch, "seed": SEED,
            "metrics": final_metrics, "test_evaluations": 1, "test_assisted_stopping": False,
            "prediction_artifact": "final_test_predictions.pt"}
        write_json(output / "final_test.json", final_record)
        del model, final_model
        records, complexities, lineage = {}, {}, {}
        for architecture, directory, inherited in (("b1", b1, frozen1), ("b2", b2, frozen2), ("b3", output, frozen)):
            final = read_json(directory / "final_test.json")
            records[architecture] = {"learning_rate": inherited["selected_learning_rate"],
                "batch_size": inherited["b1_batch_size"] if architecture == "b1" else inherited["selected_batch_size"],
                "checkpoint_epoch": final["checkpoint_epoch"], "seed": final["seed"], "metrics": final["metrics"],
                "test_evaluations": final["test_evaluations"], "checkpoint_sha256": final["checkpoint_sha256"]}
            if records[architecture]["checkpoint_sha256"] != inherited["checkpoint_sha256"] or final["test_assisted_stopping"]:
                raise ValueError("Compare the frozen final checkpoint without test-assisted reselection")
            lineage[architecture] = {"directory": str(directory), "final_test_sha256": sha256_file(directory / "final_test.json"),
                "frozen_settings_sha256": sha256_file(directory / f"frozen_{architecture}_settings.json"),
                "checkpoint": inherited["checkpoint"], "checkpoint_sha256": inherited["checkpoint_sha256"]}
            restored_model = build_model(architecture)
            restore_checkpoint(directory / inherited["checkpoint"], restored_model, device="cpu")
            complexities[architecture] = profile_model(restored_model)
            write_json(output / "complexity" / f"{architecture}.json", complexities[architecture])
            print(f"{LABELS[architecture]}: {complexities[architecture]['trainable_parameters']:,} parameters, "
                  f"{complexities[architecture]['inference_flops']:,} FLOPs/image (profiled scope)", flush=True)
        comparison = comparison_rows(records, complexities)
        write_json(output / "comparison_inputs.json", {"records": records, "lineage": lineage,
            "b1_b2_test_results_reused_without_inference": True})
        write_json(output / "architecture_comparison.json", comparison)
        summary = {"status": "completed", "architecture": "b3", "selected": selected, "selection_rule": SELECTION_RULE,
            "seed": SEED, "learning_rate": learning_rate, "batch_size": batch_size, "trainable_parameters": 77169,
            "final_test": final_metrics, "test_evaluations": 1, "refit": False, "source_sha256": bundle.source_sha256,
            "training_validation_overlap_unique_images": 8, "architecture_comparison": comparison,
            "b1_b2_test_results_reused_without_inference": True, "elapsed_seconds": time.perf_counter() - started}
        write_json(output / "summary.json", summary)
        from part_b.src.b3_exports import export_b3
        export_b3(output, source_directories={"b1": b1, "b2": b2, "b3": output})
        write_json(output / "status.json", {"status": "completed", "stage": "completed"})
        write_json(output / "progress.json", {"stage": "completed", "selected_epoch": result.best_epoch})
        print(f"B3 complete: epoch {result.best_epoch}/{result.epochs_completed}; test accuracy {final_metrics['accuracy']:.4%}, "
              f"sensitivity {final_metrics['sensitivity']:.4%}, specificity {final_metrics['specificity']:.4%}", flush=True)
        return summary
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise
