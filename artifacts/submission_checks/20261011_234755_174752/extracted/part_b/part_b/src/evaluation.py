"""One guarded final classifier test pass, with predictions retained for audits."""

from pathlib import Path

import torch

from assignment_common import evaluate_final_checkpoint, make_loader
from assignment_common.artifacts import write_checkpoint
from part_b.src.data import CLASS_MAPPING, sha256_file


def evaluate_selected_classifier_once(run_directory, model, bundle, destination, *, batch_size, seed=42, device="cuda"):
    run_directory, destination = Path(run_directory), Path(destination)
    if (run_directory / "test_metrics.json").exists() or destination.exists():
        raise FileExistsError("Reuse the saved final test result and predictions")
    logits, probabilities = [], []
    def capture(module, inputs, output):
        logits.append(output.detach().cpu().clone())
        probabilities.append(torch.sigmoid(output.detach()).cpu().clone())
    handle = model.register_forward_hook(capture)
    try:
        metrics = evaluate_final_checkpoint(run_directory, model,
            make_loader(bundle.datasets["test"], batch_size, seed=seed, device=device), device=device)
    finally:
        handle.remove()
    logits, probabilities = torch.cat(logits), torch.cat(probabilities)
    labels = bundle.datasets["test"].tensors[1]
    if logits.shape != labels.shape or metrics["count"] != len(labels):
        raise RuntimeError("Final prediction capture must cover every test row exactly once")
    write_checkpoint(destination, {"format_version": 1, "architecture": model.metadata()["architecture"],
        "checkpoint_sha256": sha256_file(run_directory / "best_model.pt"), "source_sha256": bundle.source_sha256,
        "row_ids": bundle.row_ids["test"], "labels": labels, "logits": logits, "probabilities": probabilities,
        "threshold": .5, "prediction_rule": "sigmoid(logit) >= 0.5", "metrics": metrics,
        "evaluation_batch_size": batch_size, "seed": seed,
        "class_mapping": dict(CLASS_MAPPING), "class_mapping_status": "confirmed_by_instructor_QA",
        "class_mapping_confirmation": "Q&A.md",
        "capture": "Same forward pass as the single guarded final test evaluation"})
    return metrics
