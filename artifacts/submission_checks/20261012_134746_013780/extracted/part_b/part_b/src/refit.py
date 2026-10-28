"""FAQ-mandated fresh, fixed-epoch final classifier refits and lineage checks."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from assignment_common import TrainingConfig, binary_metrics, make_loader, seed_everything
from assignment_common.artifacts import write_json
from part_b.src.data import CLASS_MAPPING, sha256_file
from part_b.src.evaluation import evaluate_selected_classifier_once
from part_b.src.models import build_model
from part_b.src.training import train_classifier


CLASS_MAPPING_STATUS = "confirmed_by_instructor_QA"
FINAL_PROTOCOL = "fresh_combined_train_validation_fixed_selected_epochs"


def state_sha256(model):
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(list(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def combined_training_data(bundle):
    """Preserve every supplied train/validation row, excluding test rows."""
    train, validation = (bundle.datasets[key] for key in ("train", "validation"))
    if len(train.tensors) != 2 or len(validation.tensors) != 2:
        raise ValueError("Part B requires image and binary-label tensors")
    return TensorDataset(*(torch.cat((first, second), dim=0)
                           for first, second in zip(train.tensors, validation.tensors)))


def refit_final_classifier(directory, bundle, selected, *, architecture,
                           batch_size, seed=42, device="cuda"):
    """Train from random initialization and evaluate the new final model once.

    The validation-selected checkpoint is used only as immutable provenance.
    Original training-only pixel normalization is deliberately reused; Q&A does
    not require changing Part B pixel statistics. No holdout loader is supplied
    during refitting, and only the last prescribed epoch can be saved.
    """
    output = Path(directory)
    if any((output / name).exists() for name in ("refit", "final_test.json", "final_test_predictions.pt")):
        raise FileExistsError("Final refit/test already exists; reuse saved evidence")
    epoch_count = selected["best_epoch"]
    if type(epoch_count) is not int or not 1 <= epoch_count <= 20:
        raise ValueError("Refit epoch count must come from the validated selection")
    selection_path = output / selected["checkpoint"]
    if sha256_file(selection_path) != selected["checkpoint_sha256"]:
        raise ValueError("Selection checkpoint changed before final refit")
    seed_everything(seed)
    model = build_model(architecture)
    initial_sha = state_sha256(model)
    if selected.get("initial_state_sha256", initial_sha) != initial_sha:
        raise ValueError("Fresh initialization disagrees with the selected architecture/seed")
    combined = combined_training_data(bundle)
    protocol = {"format_version": 1, "architecture": architecture,
        "protocol": FINAL_PROTOCOL, "refit": True,
        "learning_rate": selected["learning_rate"], "batch_size": batch_size,
        "seed": seed, "selected_epoch_count": epoch_count,
        "fresh_model": True, "fresh_optimizer": True,
        "initial_state_sha256": initial_sha,
        "training_rows": len(bundle.datasets["train"]),
        "validation_rows": len(bundle.datasets["validation"]),
        "combined_training_rows": len(combined),
        "selection_checkpoint": selected["checkpoint"],
        "selection_checkpoint_sha256": selected["checkpoint_sha256"],
        "source_sha256": bundle.source_sha256,
        "normalization": "reuse_original_training_only_pixel_statistics",
        "preprocessing": bundle.preprocessing.metadata(),
        "validation_monitor": False, "test_monitor": False,
        "class_mapping": dict(CLASS_MAPPING), "class_mapping_status": CLASS_MAPPING_STATUS,
        "clarification_source": "Q&A.md"}
    write_json(output / "refit_protocol.json", protocol)
    write_json(output / "status.json", {"status": "running", "stage": "fresh_final_refit"})
    def on_epoch(row):
        print(f"{architecture.upper()} refit epoch {row['epoch']}/{epoch_count}: "
              f"train loss {row['train_loss']:.6f}, accuracy {row['train_accuracy']:.4%}", flush=True)
        write_json(output / "progress.json", {"stage": "fresh_final_refit", **row})
    result = train_classifier(model,
        make_loader(combined, batch_size, training=True, seed=seed, device=device),
        config=TrainingConfig(learning_rate=selected["learning_rate"], max_epochs=epoch_count,
                              seed=seed, device=device, mode="fixed_refit"),
        run_directory=output / "refit", threshold=.5,
        preprocessing=bundle.preprocessing.metadata(), model_config=model.metadata(), on_epoch=on_epoch)
    if result.best_epoch != epoch_count or result.epochs_completed != epoch_count or result.stopped_early:
        raise RuntimeError("Fresh final refit did not complete the prescribed epoch budget")
    write_json(output / "status.json", {"status": "running", "stage": "refit_final_test"})
    metrics = evaluate_selected_classifier_once(output / "refit", model, bundle,
        output / "final_test_predictions.pt", batch_size=batch_size, seed=seed, device=device)
    checkpoint = result.checkpoint_path.relative_to(output).as_posix()
    final = {"architecture": architecture, "learning_rate": selected["learning_rate"],
        "batch_size": batch_size, "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(result.checkpoint_path),
        "checkpoint_epoch": epoch_count, "seed": seed, "metrics": metrics,
        "test_evaluations": 1, "test_assisted_stopping": False, "refit": True,
        "combined_training_rows": len(combined), "refit_protocol": "refit_protocol.json",
        "selection_checkpoint": selected["checkpoint"],
        "selection_checkpoint_sha256": selected["checkpoint_sha256"],
        "class_mapping_status": CLASS_MAPPING_STATUS,
        "prediction_artifact": "final_test_predictions.pt"}
    if architecture == "b1": final["selected_learning_rate"] = selected["learning_rate"]
    if architecture == "b2": final["selected_batch_size"] = batch_size
    if final["checkpoint_sha256"] == final["selection_checkpoint_sha256"]:
        raise RuntimeError("Final model must be a fresh combined-data fit")
    write_json(output / "final_test.json", final)
    if sha256_file(selection_path) != protocol["selection_checkpoint_sha256"]:
        raise RuntimeError("Selection evidence was modified during final refit")
    return final


def attach_final_checkpoint(frozen, final):
    """Keep selection lineage separate from final model lineage."""
    return {**frozen, "refit": True, "final_checkpoint": final["checkpoint"],
        "final_checkpoint_sha256": final["checkpoint_sha256"],
        "final_epoch": final["checkpoint_epoch"],
        "final_training_rows": final["combined_training_rows"],
        "final_training_protocol": FINAL_PROTOCOL,
        "class_mapping_status": CLASS_MAPPING_STATUS,
        "class_mapping_confirmation": "Q&A.md"}


def audit_final_refit(directory):
    """Validate saved model/data lineage and captured metrics without inference."""
    output = Path(directory)
    read = lambda path: json.loads((output / path).read_text(encoding="utf-8"))
    final, protocol, summary = (read(name) for name in ("final_test.json", "refit_protocol.json", "refit/summary.json"))
    checkpoint = torch.load(output / final["checkpoint"], map_location="cpu", weights_only=True)
    predictions = torch.load(output / "final_test_predictions.pt", map_location="cpu", weights_only=True)
    epoch_count = protocol["selected_epoch_count"]
    assert protocol["protocol"] == FINAL_PROTOCOL and protocol["fresh_model"] and protocol["fresh_optimizer"]
    assert protocol["combined_training_rows"] == protocol["training_rows"] + protocol["validation_rows"]
    assert not protocol["validation_monitor"] and not protocol["test_monitor"]
    assert checkpoint["training_config"]["mode"] == "fixed_refit"
    assert checkpoint["training_config"]["max_epochs"] == epoch_count
    assert checkpoint["training_config"]["seed"] == protocol["seed"]
    assert checkpoint["training_config"]["learning_rate"] == protocol["learning_rate"]
    steps = math.ceil(protocol["combined_training_rows"] / protocol["batch_size"]) * epoch_count
    assert all(state["step"].item() == steps for state in checkpoint["optimizer_state"]["state"].values())
    assert all(value.item() == steps for name, value in checkpoint["model_state"].items() if name.endswith("num_batches_tracked"))
    assert checkpoint["epoch"] == final["checkpoint_epoch"] == summary["epochs_completed"] == epoch_count
    assert summary["best_epoch"] == epoch_count and not summary["stopped_early"]
    assert checkpoint["selected_metrics"]["train"]["count"] == protocol["combined_training_rows"]
    assert json.dumps(checkpoint["preprocessing"], sort_keys=True) == json.dumps(protocol["preprocessing"], sort_keys=True)
    assert final["refit"] and final["test_evaluations"] == 1 and not final["test_assisted_stopping"]
    assert sha256_file(output / final["checkpoint"]) == final["checkpoint_sha256"] == predictions["checkpoint_sha256"]
    assert sha256_file(output / protocol["selection_checkpoint"]) == protocol["selection_checkpoint_sha256"]
    assert final["checkpoint_sha256"] != protocol["selection_checkpoint_sha256"]
    assert protocol["source_sha256"] == predictions["source_sha256"]
    expected = binary_metrics(predictions["labels"], predictions["probabilities"])
    assert all(final["metrics"][key] == value for key, value in expected.items())
    assert predictions["metrics"] == final["metrics"]
    assert predictions["class_mapping_status"] == CLASS_MAPPING_STATUS
    with (output / "refit/history.jsonl").open(encoding="utf-8") as stream:
        history = [json.loads(line) for line in stream]
    assert len(history) == epoch_count and all(not any(key.startswith(("validation_", "test_")) for key in row) for row in history)
    assert [row["epoch"] for row in history if row["checkpoint_selected"]] == [epoch_count]
    return {"status": "passed", "protocol": FINAL_PROTOCOL, "epochs": epoch_count,
        "combined_training_rows": protocol["combined_training_rows"],
        "test_rows": final["metrics"]["count"], "test_evaluations": 1,
        "saved_prediction_metrics_match": True, "new_model_inference": 0,
        "selection_checkpoint_unchanged": True, "fresh_final_checkpoint_sha256": final["checkpoint_sha256"]}
