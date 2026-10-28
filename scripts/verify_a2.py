"""Audit 15 A2 runs and reproduce restored training metrics without rescoring test data."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn

from assignment_common import evaluate, make_loader, restore_checkpoint, seed_everything
from assignment_common.artifacts import write_json
from part_a.src.a2 import (
    EVALUATION_SEEDS, check_same_preprocessing, model_state_fingerprint,
    sha256_file, summarize_a2_records, validate_frozen_settings,
)
from part_a.src.ablations import VARIANTS, model_from_a2_metadata
from part_a.src.data import load_data_bundle


def compare_metrics(expected, actual):
    if expected.keys() != actual.keys():
        raise AssertionError("Metric names differ")
    for name, value in expected.items():
        if value is None:
            if actual[name] is not None:
                raise AssertionError(f"Undefined metric differs: {name}")
        elif not math.isclose(value, actual[name], rel_tol=1e-9, abs_tol=1e-8):
            raise AssertionError(f"Metric differs: {name}: {value} -> {actual[name]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "part_a/outputs/a2")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    config = json.loads((output / "configuration.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    if summary["status"] != "completed" or status["status"] != "completed":
        raise AssertionError("A2 is not complete")
    a1 = Path(config["a1_directory"])
    frozen = json.loads((a1 / "frozen_a1_settings.json").read_text(encoding="utf-8"))
    a1_summary = json.loads((a1 / "summary.json").read_text(encoding="utf-8"))
    validate_frozen_settings(frozen, a1_summary)
    if frozen != config["frozen_a1_settings"] or frozen != summary["frozen_a1_settings"]:
        raise AssertionError("A2 did not preserve A1's frozen settings")
    if (sha256_file(a1 / "frozen_a1_settings.json") != config["a1_frozen_settings_sha256"]
            or sha256_file(a1 / frozen["refit_preprocessing"]) != config["a1_preprocessing_sha256"]):
        raise AssertionError("A1 frozen settings/preprocessing changed")
    prepared = Path(config["prepared_directory"])
    if sha256_file(prepared / "refit/datasets.pt") != config["prepared_refit_tensors_sha256"]:
        raise AssertionError("Prepared dataset changed")
    bundle = load_data_bundle(prepared / "refit", expected_protocol="refit")
    metadata = torch.load(output / "preprocessing.pt", map_location="cpu", weights_only=True)
    check_same_preprocessing(metadata, bundle.preprocessing.metadata())
    check_same_preprocessing(metadata, torch.load(a1 / frozen["refit_preprocessing"], map_location="cpu", weights_only=True))
    if metadata["source_sha256"] != bundle.source_sha256 or bundle.source_sha256 != config["source_sha256"]:
        raise AssertionError("Source hashes differ")
    records = json.loads((output / "per_seed_results.json").read_text(encoding="utf-8"))
    summaries, paired = summarize_a2_records(records, epoch_budget=frozen["E_star"])
    if summaries != summary["variants"] or paired != summary["paired_differences"]:
        raise AssertionError("Saved aggregate summaries differ from per-seed records")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce saved CUDA training metrics")
    torch.set_num_threads(4)
    train = make_loader(bundle.datasets["train"], frozen["batch_size"], device="cuda")
    initialization, shuffle, audited = {}, {}, []
    for record in records:
        run = output / record["run_directory"]
        run_config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        run_summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        test = json.loads((run / "test_metrics.json").read_text(encoding="utf-8"))
        history = [json.loads(line) for line in (run / "history.jsonl").read_text(encoding="utf-8").splitlines()]
        if (run_summary["status"] != "completed" or run_config["training"]["mode"] != "fixed_refit"
                or run_config["training"]["seed"] != record["seed"]
                or run_config["training"]["max_epochs"] != frozen["E_star"]
                or run_config["training"]["learning_rate"] != .01 or run_config["batch_size"] != 512
                or run_config["stopping_metric"] is not None or run_summary["test_assisted_stopping"]
                or run_summary["stopped_early"] or run_summary["best_epoch"] != frozen["E_star"]
                or run_summary["epochs_completed"] != frozen["E_star"]):
            raise AssertionError("Run violates the fixed-epoch controls")
        if [row["epoch"] for row in history] != list(range(1, frozen["E_star"] + 1)):
            raise AssertionError("Missing or extra training epochs")
        if any(key.startswith(("validation_", "test_")) for row in history for key in row):
            raise AssertionError("A2 monitored validation/test during training")
        if (run / "internal_test_monitor.jsonl").exists():
            raise AssertionError("A2 contains a test stopping history")
        seed_everything(record["seed"])
        model = model_from_a2_metadata(run_config["model_config"])
        fingerprint = model_state_fingerprint(model)
        if fingerprint != record["initial_model_sha256"] or fingerprint != run_config["model_config"]["initial_model_sha256"]:
            raise AssertionError("Run was not initialized freshly from its recorded seed")
        initialization[(record["variant"], record["seed"])] = fingerprint
        shuffle.setdefault(record["seed"], set()).add(record["train_loader_generator_sha256"])
        if record["variant"] == "continuous_only":
            if any(isinstance(module, nn.Embedding) for module in model.modules()) or model.mlp[0].in_features != 6:
                raise AssertionError("Continuous-only architecture retains categorical inputs")
        else:
            expected_activation = nn.Sigmoid if record["variant"] == "sigmoid" else nn.ReLU
            if len(model.embeddings) != 4 or not isinstance(model.mlp[2], expected_activation):
                raise AssertionError("All-input ablation architecture is incorrect")
        if not isinstance(model.mlp[1], nn.LayerNorm) or model.mlp[0].out_features != frozen["hidden_width"]:
            raise AssertionError("Hidden width/LayerNorm changed")
        if sum(parameter.numel() for parameter in model.parameters()) != record["trainable_parameters"]:
            raise AssertionError("Parameter count differs")
        checkpoint = restore_checkpoint(output / record["checkpoint"], model, device="cuda")
        if (checkpoint["epoch"] != frozen["E_star"] or checkpoint["training_config"]["mode"] != "fixed_refit"
                or checkpoint["model_config"]["variant"] != record["variant"]
                or checkpoint["training_config"]["seed"] != record["seed"]):
            raise AssertionError("Checkpoint metadata differs from its run")
        check_same_preprocessing(checkpoint["preprocessing"], metadata)
        measured = evaluate(model, train, checkpoint["task_object"], "cuda")
        compare_metrics(checkpoint["selected_metrics"]["train"], measured)
        compare_metrics(run_summary["selected_metrics"]["train"], measured)
        if test["checkpoint_epoch"] != frozen["E_star"] or test["seed"] != record["seed"] or test["test_assisted_stopping"]:
            raise AssertionError("Final-test record uses an incorrect checkpoint or protocol")
        if test["metrics"]["count"] != 26702 or record["test_count"] != 26702:
            raise AssertionError("Final-test sample count differs")
        for name in ("rmse", "r2"):
            if not math.isclose(test["metrics"][name], record[name], rel_tol=1e-12):
                raise AssertionError("Per-seed table differs from the final-test record")
        last = history[-1]
        compare_metrics(measured, {key: last[f"train_{key}"] for key in measured})
        if not math.isclose(measured["rmse"], record["train_rmse"], rel_tol=1e-9):
            raise AssertionError("Recorded training metrics differ")
        audited.append({"variant": record["variant"], "seed": record["seed"],
                        "checkpoint_epoch": checkpoint["epoch"], "training_metrics_reproduced_on_cuda": True,
                        "checkpoint_sha256": sha256_file(output / record["checkpoint"])})
        del model
    for seed in EVALUATION_SEEDS:
        if initialization[("baseline", seed)] != initialization[("sigmoid", seed)] or len(shuffle[seed]) != 1:
            raise AssertionError("Matched seeds did not preserve initialization/batch-order controls")
    table = pd.read_csv(output / "tables/summary_metrics.csv").set_index("variant")
    for variant in VARIANTS:
        for metric in ("rmse", "r2"):
            values = np.array([row[metric] for row in records if row["variant"] == variant])
            prefix = "test_rmse" if metric == "rmse" else "test_r2"
            suffix = "_sgd" if metric == "rmse" else ""
            if not math.isclose(table.loc[variant, prefix + "_mean" + suffix], values.mean(), rel_tol=1e-12):
                raise AssertionError("CSV means differ from the original results")
            if not math.isclose(table.loc[variant, prefix + "_sample_std" + suffix], values.std(ddof=1), rel_tol=1e-12):
                raise AssertionError("CSV standard deviations do not use ddof=1")
    if len(pd.read_csv(output / "tables/per_seed_results.csv")) != 15 or len(pd.read_csv(output / "tables/paired_differences.csv")) != 10:
        raise AssertionError("Exported tables are incomplete")
    for name in ("categorical_ablation", "activation_ablation"):
        for extension in ("png", "pdf"):
            if (output / "figures" / f"{name}.{extension}").stat().st_size < 1000:
                raise AssertionError("An ablation figure is missing or empty")
    setup = json.loads((ROOT / "artifacts/setup/environment_check.json").read_text(encoding="utf-8"))
    for name, original in setup["files"].items():
        if sha256_file(ROOT / name) != original["sha256"]:
            raise AssertionError(f"Supplied file changed: {name}")
    a1_evidence = json.loads((a1 / "verification.json").read_text(encoding="utf-8"))
    for checked in a1_evidence["checkpoints"]:
        path = a1 / "grid" / f"h{checked['hidden_width']:03d}_d{checked['embedding_dimension']:02d}" / "training/best_model.pt"
        if sha256_file(path) != checked["checkpoint_sha256"]:
            raise AssertionError("A1 grid checkpoint changed")
    if sha256_file(a1 / "refit/best_model.pt") != a1_evidence["refit_checkpoint_sha256"]:
        raise AssertionError("A1 final checkpoint changed")
    baseline42 = next(row for row in records if row["variant"] == "baseline" and row["seed"] == 42)
    matches_a1 = all(math.isclose(baseline42[name], a1_summary["final_test_metrics"][name], rel_tol=1e-9, abs_tol=1e-8)
                     for name in ("rmse", "r2"))
    evidence = {"status": "passed", "checked_at_singapore": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
                "run_directory": str(output), "completed_runs": 15, "epochs_each": frozen["E_star"],
                "all_15_restored_training_metrics_reproduced_on_cuda": True,
                "no_additional_test_performance_evaluation": True, "no_test_or_validation_monitoring_during_a2": True,
                "exact_a1_preprocessing_reused": True, "fresh_seed_initializations_verified": True,
                "sigmoid_baseline_initial_weights_identical_for_each_seed": True,
                "matched_seed_loader_generator_states_identical": True,
                "sample_sd_ddof_1_and_paired_summaries_verified": True, "separate_figures_and_tables_checked": True,
                "supplied_files_and_a1_checkpoints_unchanged": True, "fresh_baseline_seed42_matches_a1_final_metrics": matches_a1,
                "checkpoints": audited}
    write_json(output / "verification.json", evidence)
    timestamp = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d_%H%M%S_%f")
    destination = ROOT / "artifacts/a2_checks" / timestamp
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "verification.json", evidence)
    write_json(ROOT / "artifacts/a2_checks/latest.json", {"status": "passed", "directory": str(destination),
                                                          "verification": str(destination / "verification.json")})
    print("A2 audit passed: all 15 fixed-epoch checkpoints, restored CUDA training metrics, controls, summaries, and unchanged source/A1 files.")


if __name__ == "__main__":
    main()
