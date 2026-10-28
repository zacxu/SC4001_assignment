"""Audit all 25 A1 checkpoints/results and reproduce validation metrics on CUDA."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from assignment_common import EarlyStopper, evaluate, make_loader, restore_checkpoint, seed_everything
from assignment_common.artifacts import write_json
from part_a.src.a1 import choose_configuration
from part_a.src.data import HousingPreprocessor, load_data_bundle
from part_a.src.model import HousingMLP, parameter_count


def checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare_metrics(expected, actual):
    if expected.keys() != actual.keys():
        raise AssertionError("Metric names differ")
    for name, value in expected.items():
        if value is None:
            if actual[name] is not None:
                raise AssertionError(f"Undefined metric differs: {name}")
        elif not math.isclose(value, actual[name], rel_tol=1e-9, abs_tol=1e-8):
            raise AssertionError(f"Metric differs: {name}: {value} -> {actual[name]}")


def read_history(directory):
    history = [json.loads(line) for line in (directory / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    if [row["epoch"] for row in history] != list(range(1, len(history) + 1)):
        raise AssertionError("Missing or repeated epoch rows")
    if any(any(key.startswith("test_") for key in row) for row in history):
        raise AssertionError("Test epoch metrics leaked into a public history")
    return history


def audit_stopping(values, summary):
    stopper = EarlyStopper(patience=4, min_delta=1e-4)
    for index, value in enumerate(values, 1):
        stopped = stopper.update(value)
        if stopped and index != len(values):
            raise AssertionError("Training continued after the stopping rule triggered")
    if len(values) != summary["epochs_completed"] or not 1 <= len(values) <= 20:
        raise AssertionError("Invalid completed epoch count")
    if len(values) < 20 and not stopped:
        raise AssertionError("Training ended before the cap without meeting the stopping rule")
    if summary["stopped_early"] != (stopped and len(values) < 20):
        raise AssertionError("Early-stopping flag differs from the actual history")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "part_a/outputs/a1")
    parser.add_argument("--prepared", type=Path, default=ROOT / "part_a/outputs/data")
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    torch.set_num_threads(4)
    seed_everything(42)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce the saved CUDA validation metrics")
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "completed":
        raise AssertionError("A1 is not complete")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    rows = json.loads((output / "grid_results.json").read_text(encoding="utf-8"))
    winner = choose_configuration(rows)
    if winner != summary["selected"] or winner["best_epoch"] != summary["E_val"]:
        raise AssertionError("Selected model differs from deterministic full-grid selection")
    selection = load_data_bundle(arguments.prepared / "selection", expected_protocol="selection")
    validation = make_loader(selection.datasets["validation"], 512, seed=42, device="cuda")
    audited = []
    for row in rows:
        run = output / row["run_directory"]
        run_summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        run_config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        if run_summary["status"] != "completed" or not row["ray_checkpoint_registered"]:
            raise AssertionError("Trial did not complete and register its checkpoint")
        if (run_config["training"]["mode"] != "selection" or run_config["test_assisted_stopping"]
                or run_config["batch_size"] != 512 or run_config["training"]["learning_rate"] != .01
                or run_config["training"]["seed"] != 42):
            raise AssertionError("Trial settings violate the selection protocol")
        if (run / "test_metrics.json").exists() or (run / "internal_test_monitor.jsonl").exists():
            raise AssertionError("A grid trial has test-evaluation artifacts")
        history = read_history(run)
        audit_stopping([item["validation_rmse"] for item in history], run_summary)
        selected_epoch = min(history, key=lambda item: (item["validation_rmse"], item["epoch"]))
        if selected_epoch["epoch"] != row["best_epoch"] or len(history) != row["epochs_completed"]:
            raise AssertionError("Row epochs do not match the selected checkpoint and history")
        for prefix in ("train", "validation"):
            compare_metrics(run_summary["selected_metrics"][prefix], {
                name: selected_epoch[f"{prefix}_{name}"] for name in run_summary["selected_metrics"][prefix]})
        pre = selection.preprocessing
        model = HousingMLP(pre.cardinalities, 0., 1., hidden_width=row["hidden_width"],
                           embedding_dimension=row["embedding_dimension"])
        checkpoint = restore_checkpoint(output / row["checkpoint"], model, device="cuda")
        saved_pre = HousingPreprocessor.from_metadata(checkpoint["preprocessing"])
        if (saved_pre.fit_years != (2017, 2018, 2019, 2020)
                or checkpoint["preprocessing"]["source_sha256"] != selection.source_sha256
                or checkpoint["epoch"] != row["best_epoch"]
                or row["trainable_parameters"] != parameter_count(pre.cardinalities, row["hidden_width"], row["embedding_dimension"])):
            raise AssertionError("Checkpoint metadata does not match this trial")
        torch.testing.assert_close(saved_pre.continuous_mean, pre.continuous_mean, rtol=0, atol=0)
        torch.testing.assert_close(saved_pre.continuous_std, pre.continuous_std, rtol=0, atol=0)
        if saved_pre.category_maps != pre.category_maps or model.target_mean != pre.target_mean or model.target_std != pre.target_std:
            raise AssertionError("Preprocessing/target statistics changed on restore")
        measured = evaluate(model, validation, checkpoint["task_object"], "cuda")
        compare_metrics(checkpoint["selected_metrics"]["validation"], measured)
        compare_metrics(run_summary["selected_metrics"]["validation"], measured)
        if not math.isclose(row["validation_rmse"], measured["rmse"], rel_tol=1e-9, abs_tol=1e-8):
            raise AssertionError("Grid table score differs from restored checkpoint")
        audited.append({"hidden_width": row["hidden_width"], "embedding_dimension": row["embedding_dimension"],
                        "best_epoch": row["best_epoch"], "validation_metrics_reproduced": True,
                        "checkpoint_sha256": checksum(output / row["checkpoint"])})
        del model
    selected_model = HousingMLP.from_metadata(summary["refit_model_config"])
    refit_checkpoint = restore_checkpoint(output / "refit/best_model.pt", selected_model, device="cuda")
    refit_pre = HousingPreprocessor.from_metadata(refit_checkpoint["preprocessing"])
    if refit_pre.fit_years != (2017, 2018, 2019, 2020, 2021) or refit_pre.fit_row_count != 116427:
        raise AssertionError("Refit preprocessing has incorrect fitting membership")
    saved_refit = load_data_bundle(arguments.prepared / "refit", expected_protocol="refit")
    torch.testing.assert_close(refit_pre.continuous_mean, saved_refit.preprocessing.continuous_mean, rtol=0, atol=0)
    torch.testing.assert_close(refit_pre.continuous_std, saved_refit.preprocessing.continuous_std, rtol=0, atol=0)
    if (refit_pre.category_maps != saved_refit.preprocessing.category_maps
            or selected_model.target_mean != saved_refit.preprocessing.target_mean
            or selected_model.target_std != saved_refit.preprocessing.target_std
            or refit_checkpoint["preprocessing"]["source_sha256"] != saved_refit.source_sha256):
        raise AssertionError("Refit checkpoint preprocessing differs from the saved combined-data transform")
    if refit_checkpoint["training_config"]["mode"] != "a1_refit" or refit_checkpoint["epoch"] != summary["E_star"]:
        raise AssertionError("Incorrect final-refit protocol or selected epoch")
    refit_summary = json.loads((output / "refit/summary.json").read_text(encoding="utf-8"))
    public_refit = read_history(output / "refit")
    monitor = [json.loads(line) for line in (output / "refit/internal_test_monitor.jsonl").read_text(encoding="utf-8").splitlines()]
    if [item["epoch"] for item in monitor] != [item["epoch"] for item in public_refit]:
        raise AssertionError("Refit monitoring/public epoch counts differ")
    audit_stopping([item["rmse"] for item in monitor], refit_summary)
    if min(monitor, key=lambda item: (item["rmse"], item["epoch"]))["epoch"] != summary["E_star"]:
        raise AssertionError("Refit did not restore the lowest monitored RMSE checkpoint")
    final = json.loads((output / "refit/test_metrics.json").read_text(encoding="utf-8"))
    if final["checkpoint_epoch"] != summary["E_star"] or not final["test_assisted_stopping"]:
        raise AssertionError("Final-test record does not identify the correct checkpoint/protocol")
    compare_metrics(final["metrics"], summary["final_test_metrics"])
    compare_metrics(final["metrics"], refit_checkpoint["selected_metrics"]["test_monitor"])
    compare_metrics(refit_checkpoint["selected_metrics"]["train"], summary["refit_selected_train_metrics"])
    frozen = json.loads((output / "frozen_a1_settings.json").read_text(encoding="utf-8"))
    if frozen["E_star"] != summary["E_star"] or frozen["hidden_width"] != winner["hidden_width"] or frozen["embedding_dimension"] != winner["embedding_dimension"]:
        raise AssertionError("Frozen A2/A3 settings differ from A1")
    table = pd.read_csv(output / "tables/grid_results.csv")
    if len(table) != 25 or table.selected.sum() != 1:
        raise AssertionError("Grid CSV has incomplete/ambiguous selection")
    final_table = pd.read_csv(output / "tables/final_test_metrics.csv").iloc[0]
    if not math.isclose(final_table.test_rmse_sgd, final["metrics"]["rmse"], rel_tol=1e-12) or not math.isclose(final_table.test_r2, final["metrics"]["r2"], rel_tol=1e-12):
        raise AssertionError("Final-test table changed the saved metrics")
    for name in ("baseline_architecture", "selected_learning_curve", "default_baseline_learning_curve",
                 "validation_grid_heatmap", "refit_training_curve"):
        for extension in ("png", "pdf"):
            if (output / "figures" / f"{name}.{extension}").stat().st_size < 1000:
                raise AssertionError("A report figure is missing or empty")
    setup = json.loads((ROOT / "artifacts/setup/environment_check.json").read_text(encoding="utf-8"))
    for name, original in setup["files"].items():
        if checksum(ROOT / name) != original["sha256"]:
            raise AssertionError(f"Supplied file changed: {name}")
    evidence = {"status": "passed", "checked_at_singapore": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
                "run_directory": str(output), "all_25_validation_checkpoints_reproduced_on_cuda": True,
                "epochs_selection_and_stopping_audited": True, "default_parameter_count": 3505,
                "selection_used_no_test_predictions": True, "public_epoch_histories_contain_no_test_metrics": True,
                "refit_metrics_match_restored_checkpoint_and_selected_internal_monitor": True,
                "no_additional_test_performance_evaluation": True, "tables_and_figures_checked": True,
                "supplied_files_unchanged": True, "E_val": summary["E_val"], "E_star": summary["E_star"],
                "selected": {key: winner[key] for key in ("hidden_width", "embedding_dimension", "validation_rmse", "trainable_parameters")},
                "final_test_metrics": final["metrics"], "checkpoints": audited,
                "refit_checkpoint_sha256": checksum(output / "refit/best_model.pt")}
    write_json(output / "verification.json", evidence)
    timestamp = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d_%H%M%S_%f")
    destination = ROOT / "artifacts/a1_checks" / timestamp
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "verification.json", evidence)
    write_json(ROOT / "artifacts/a1_checks/latest.json", {"status": "passed", "directory": str(destination),
                                                          "verification": str(destination / "verification.json")})
    print(f"A1 verification passed: all 25 restored CUDA validation scores, final artifact coherence, plots, and unchanged supplied files.")


if __name__ == "__main__":
    main()
