"""Current FAQ saved-artifact checks; no model forward pass or retraining."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as functional

from assignment_common.artifacts import write_json
from part_b.src.data import sha256_file
from part_b.src.refit import audit_final_refit


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit_classifier(output):
    output = Path(output)
    summary = read(output / "summary.json")
    key = summary["architecture"]
    frozen = read(output / f"frozen_{key}_settings.json")
    selected = summary["selected"]
    assert summary["refit"] and frozen["refit"]
    assert frozen["checkpoint"] == selected["checkpoint"]
    assert frozen["checkpoint_sha256"] == selected["checkpoint_sha256"]
    assert frozen["final_checkpoint"] != frozen["checkpoint"]
    result = audit_final_refit(output)
    assert result["combined_training_rows"] == 5232 and result["test_rows"] == 624
    assert result["epochs"] == frozen["final_epoch"] == frozen["selected_epoch"] == selected["best_epoch"]
    rows = (read(output / ("learning_rate_search.json" if key == "b1" else "batch_size_search.json"))
            if key != "b3" else [selected])
    field = "learning_rate" if key == "b1" else "batch_size"
    if key != "b3":
        assert min(rows, key=lambda r: (r["validation_classification_error"], r["validation_loss"], r["best_epoch"], r[field])) == selected
        assert {r[field] for r in rows} == ({1e-5, 1e-4, 1e-3, 1e-2, 1e-1} if key == "b1" else {16, 32, 64, 128, 256})
    for record in rows:
        checkpoint = output / record["checkpoint"]
        assert sha256_file(checkpoint) == record["checkpoint_sha256"]
        history = [json.loads(line) for line in (checkpoint.parent / "history.jsonl").read_text().splitlines()]
        assert [row["epoch"] for row in history] == list(range(1, len(history) + 1))
        assert min(history, key=lambda r: (r["validation_classification_error"], r["validation_loss"], r["epoch"]))["epoch"] == record["best_epoch"]
        best, waiting = math.inf, 0
        for row in history:
            if row["validation_loss"] < best - 1e-4:
                best, waiting = row["validation_loss"], 0
            else:
                waiting += 1
            assert waiting < 4 or row is history[-1]
        assert len(history) == 20 or waiting >= 4
    result.update(architecture=key, selection_runs_checked=len(rows))
    write_json(output / "verification.json", result)
    print(json.dumps(result, indent=2))
    return result


def audit_gradcam(output):
    output = Path(output)
    config = read(output / "configuration.json")
    summary = read(output / "summary.json")
    payload = torch.load(output / "gradcam.pt", map_location="cpu", weights_only=True)
    assert summary["status"] == "completed" and summary["overlay_count"] == 15
    assert payload["row_ids"].tolist() == summary["sample_row_ids"] == [486, 279, 413, 53, 276]
    assert summary["full_test_performance_evaluations"] == 0
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    checks = {}
    for key, record in payload["models"].items():
        source = config["models"][key]
        directory = Path(source["directory"])
        frozen = read(directory / f"frozen_{key}_settings.json")
        assert frozen["refit"] and source["checkpoint"] == frozen["final_checkpoint"]
        assert sha256_file(directory / frozen["final_checkpoint"]) == record["checkpoint_sha256"] == frozen["final_checkpoint_sha256"]
        predictions = torch.load(directory / "final_test_predictions.pt", map_location="cpu", weights_only=True)
        assert predictions["checkpoint_sha256"] == record["checkpoint_sha256"]
        assert record["model_state_sha256_before"] == record["model_state_sha256_after"]
        assert torch.equal(record["logits"], record["saved_final_test_logits"])
        assert torch.equal(record["probabilities"], record["saved_final_test_probabilities"])
        indices = payload["row_ids"]
        assert torch.equal(predictions["logits"][indices], record["logits"])
        assert torch.equal(predictions["probabilities"][indices], record["probabilities"])
        count, batch = len(record["activations"]), max(record["forward_batch_sizes"])
        repeats = (batch + count - 1) // count
        # Reconstruct stored map arithmetic on its original CUDA backend/shape.
        activations = record["activations"].repeat(repeats, 1, 1, 1)[:batch].cuda()
        gradients = record["gradients"].repeat(repeats, 1, 1, 1)[:batch].cuda()
        weights = gradients.mean((2, 3), keepdim=True)
        coarse = (weights * activations).sum(1, keepdim=True)
        full = functional.interpolate(coarse[:count].relu(), (64, 64), mode="bilinear", align_corners=False)
        peaks = full.flatten(1).amax(1).reshape(-1, 1, 1, 1)
        normalized = full / torch.where(peaks > 0, peaks, torch.ones_like(peaks))
        for expected, actual in ((record["channel_weights"], weights[:count]),
                (record["signed_coarse_maps"], coarse[:count]),
                (record["positive_coarse_maps"], coarse.relu()[:count]),
                (record["upsampled_positive_maps"], full), (record["normalized_maps"], normalized)):
            torch.testing.assert_close(expected, actual.cpu(), rtol=0, atol=0)
        checks[key] = {"final_checkpoint_sha256": record["checkpoint_sha256"], "saved_map_arithmetic_exact": True}
    result = {"status": "passed", "protocol": "current_faq_final_refit_gradcam", "models": checks,
        "sample_row_ids": summary["sample_row_ids"], "overlays_checked": 15,
        "new_model_inference": 0, "new_training_runs": 0,
        "scope": "stored activations/gradients arithmetic and final-checkpoint lineage; no independent new backward pass"}
    write_json(output / "verification.json", result)
    print(json.dumps(result, indent=2))
    return result
