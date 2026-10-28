"""Full Ray grid, validation selection, and professor-authorized A1 final refit."""

from __future__ import annotations

from datetime import datetime
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
from zoneinfo import ZoneInfo

os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import ray
from ray import tune
import torch

from assignment_common import TrainingConfig, evaluate_final_checkpoint, make_loader, seed_everything
from assignment_common.artifacts import write_json
from part_a.src.data import load_data_bundle
from part_a.src.model import HousingMLP, parameter_count
from part_a.src.training import train_regressor

HIDDEN_WIDTHS = (16, 32, 64, 128, 256)
EMBEDDING_DIMENSIONS = (2, 4, 8, 12, 16)
TIE_BREAK_RULE = "lowest validation RMSE, then fewer parameters, then smaller hidden width, then smaller embedding dimension"


def choose_configuration(rows: list[dict]) -> dict:
    expected = {(width, dimension) for width in HIDDEN_WIDTHS for dimension in EMBEDDING_DIMENSIONS}
    pairs = [(row["hidden_width"], row["embedding_dimension"]) for row in rows]
    if len(pairs) != len(expected) or set(pairs) != expected:
        raise ValueError("A1 requires exactly one successful result for each of the 25 grid combinations")
    if any(not math.isfinite(row["validation_rmse"]) for row in rows):
        raise ValueError("Grid scores must be finite")
    return min(rows, key=lambda row: (row["validation_rmse"], row["trainable_parameters"],
                                     row["hidden_width"], row["embedding_dimension"]))


def selection_trainable(config: dict) -> None:
    """Each Ray worker constructs a fresh seeded model and reads the selection bundle."""
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("The A1 Ray trial requires CUDA")
    bundle = load_data_bundle(config["prepared_directory"], expected_protocol="selection")
    pre = bundle.preprocessing
    seed_everything(config["seed"])
    model = HousingMLP(pre.cardinalities, pre.target_mean, pre.target_std,
                       hidden_width=config["hidden_width"], embedding_dimension=config["embedding_dimension"])
    training = make_loader(bundle.datasets["train"], config["batch_size"], training=True,
                           seed=config["seed"], device="cuda")
    validation = make_loader(bundle.datasets["validation"], config["batch_size"], seed=config["seed"], device="cuda")
    run_directory = Path(tune.get_context().get_trial_dir()) / "training"
    best = {"rmse": float("inf"), "epoch": None}

    def report_epoch(row):
        if row["checkpoint_selected"]:
            best.update(rmse=row["validation_rmse"], epoch=row["epoch"])
        tune.report({"epoch": row["epoch"], "train_rmse": row["train_rmse"],
                     "validation_rmse": row["validation_rmse"], "best_validation_rmse": best["rmse"],
                     "best_epoch": best["epoch"], "epoch_seconds": row["epoch_seconds"],
                     "trainable_parameters": model.metadata()["trainable_parameters"], "complete": False})

    torch.cuda.reset_peak_memory_stats()
    result = train_regressor(
        model, training, target_mean=pre.target_mean, target_std=pre.target_std,
        config=TrainingConfig(learning_rate=config["learning_rate"], max_epochs=20, patience=4,
                              min_delta=1e-4, seed=config["seed"], device="cuda", mode="selection"),
        run_directory=run_directory, validation_loader=validation,
        preprocessing={**pre.metadata(), "source_sha256": bundle.source_sha256},
        model_config=model.metadata(), on_epoch=report_epoch,
    )
    # Report the restored checkpoint coherently. This final report is not another
    # epoch; the explicit epoch fields, not Ray's training_iteration, define epochs.
    summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
    tune.report({"epoch": result.epochs_completed, "epochs_completed": result.epochs_completed,
                 "best_epoch": result.best_epoch, "best_validation_rmse": result.selected_metrics["validation"]["rmse"],
                 "train_rmse": result.selected_metrics["train"]["rmse"],
                 "validation_rmse": result.selected_metrics["validation"]["rmse"],
                 "trainable_parameters": model.metadata()["trainable_parameters"],
                 "elapsed_seconds": summary["elapsed_seconds"],
                 "peak_allocated_gpu_mib": torch.cuda.max_memory_allocated() / 1024**2,
                 "device": torch.cuda.get_device_name(0), "complete": True},
                checkpoint=tune.Checkpoint.from_directory(run_directory))


def trial_directory_name(trial) -> str:
    return f"h{trial.config['hidden_width']:03d}_d{trial.config['embedding_dimension']:02d}"


class ProgressCallback(tune.Callback):
    def __init__(self, output: Path):
        self.output = output
        self.completed = 0

    def on_trial_result(self, iteration, trials, trial, result, **info):
        write_json(self.output / "progress.json", {"stage": "grid", "completed_trials": self.completed,
                   "total_trials": 25, "trial": trial_directory_name(trial), "epoch": result["epoch"],
                   "best_validation_rmse": result["best_validation_rmse"], "complete": result["complete"]})
        if result["epoch"] == 1 and not result["complete"]:
            print(f"Started {trial_directory_name(trial)}: epoch 1 took {result['epoch_seconds']:.2f}s, "
                  f"validation RMSE SGD {result['validation_rmse']:,.2f}", flush=True)

    def on_trial_complete(self, iteration, trials, trial, **info):
        self.completed += 1
        result = trial.last_result
        print(f"Completed {self.completed}/25: {trial_directory_name(trial)}, "
              f"best validation RMSE SGD {result['best_validation_rmse']:,.2f}, "
              f"epoch {result['best_epoch']}/{result['epochs_completed']}", flush=True)


def collect_grid(results, output: Path) -> list[dict]:
    rows = []
    for result in results:
        if result.error or not result.metrics.get("complete"):
            raise RuntimeError(f"Grid trial failed or incomplete: {result.path}: {result.error}")
        run_directory = Path(result.path) / "training"
        summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
        if summary["status"] != "completed":
            raise RuntimeError(f"Incomplete training summary: {run_directory}")
        metrics = summary["selected_metrics"]
        width, dimension = result.config["hidden_width"], result.config["embedding_dimension"]
        rows.append({"hidden_width": width, "embedding_dimension": dimension,
                     "input_width": 4 * dimension + 6,
                     "trainable_parameters": parameter_count([13, 27, 44, 18], width, dimension),
                     "best_epoch": summary["best_epoch"], "epochs_completed": summary["epochs_completed"],
                     "stopped_early": summary["stopped_early"],
                     "train_rmse": metrics["train"]["rmse"], "train_r2": metrics["train"]["r2"],
                     "validation_rmse": metrics["validation"]["rmse"], "validation_r2": metrics["validation"]["r2"],
                     "elapsed_seconds": summary["elapsed_seconds"],
                     "peak_allocated_gpu_mib": result.metrics["peak_allocated_gpu_mib"],
                     "device": result.metrics["device"],
                     "run_directory": str(run_directory.relative_to(output)),
                     "checkpoint": str((run_directory / "best_model.pt").relative_to(output)),
                     "ray_checkpoint_registered": result.checkpoint is not None})
    choose_configuration(rows)
    return sorted(rows, key=lambda row: (row["hidden_width"], row["embedding_dimension"]))


def run_a1(output_directory: Path | str, prepared_directory: Path | str, settings: dict, *, root: Path) -> dict:
    output, prepared = Path(output_directory).resolve(), Path(prepared_directory).resolve()
    if settings["hidden_width_grid"] != list(HIDDEN_WIDTHS) or settings["embedding_dimension_grid"] != list(EMBEDDING_DIMENSIONS):
        raise ValueError("A1 must use the assignment's complete 5 by 5 grid")
    if settings["learning_rate"] != .01 or settings["batch_size"] != 512:
        raise ValueError("A1 preserves baseline learning rate 0.01 and batch size 512")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this A1 run")
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    selection = load_data_bundle(prepared / "selection", expected_protocol="selection")
    if selection.preprocessing.cardinalities != [13, 27, 44, 18]:
        raise ValueError("Prepared data cardinalities differ from the supplied assignment")
    configuration = {
        "created_at_singapore": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "seed": 42, "learning_rate": .01, "batch_size": 512,
        "max_epochs": 20, "patience": 4, "min_delta": 1e-4,
        "hidden_width_grid": list(HIDDEN_WIDTHS), "embedding_dimension_grid": list(EMBEDDING_DIMENSIONS),
        "selection": TIE_BREAK_RULE, "refit": "2017-2021 training; 2022 test-RMSE stopping per professor clarification",
        "source_sha256": selection.source_sha256, "prepared_directory": str(prepared),
        "ray_resources_per_trial": {"cpu": 4, "gpu": 1}, "max_concurrent_trials": 1,
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "ray": ray.__version__, "cuda": torch.version.cuda,
                        "device": torch.cuda.get_device_name(0)},
    }
    write_json(output / "configuration.json", configuration)
    write_json(output / "status.json", {"status": "running", "stage": "grid"})
    try:
        # Local absolute paths and PYTHONPATH keep Windows workers independent of
        # their Ray trial working directories, without uploading the whole venv.
        ray.init(num_cpus=4, num_gpus=1, object_store_memory=256 * 1024**2,
                 include_dashboard=False, log_to_driver=False, _node_ip_address="127.0.0.1",
                 _temp_dir=str(Path(tempfile.gettempdir()) / "sc4001_ray_a1"),
                 runtime_env={"env_vars": {"PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1",
                                            "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "RAY_USAGE_STATS_ENABLED": "0"}})
        # Enumerate the supplied default first; it is one of the required 25 trials.
        tuner = tune.Tuner(
            tune.with_resources(selection_trainable, {"cpu": 4, "gpu": 1}),
            param_space={"hidden_width": tune.grid_search([64, 16, 32, 128, 256]),
                         "embedding_dimension": tune.grid_search([8, 2, 4, 12, 16]),
                         "learning_rate": .01, "batch_size": 512, "seed": 42,
                         "prepared_directory": str(prepared / "selection")},
            tune_config=tune.TuneConfig(metric="best_validation_rmse", mode="min", num_samples=1,
                                       max_concurrent_trials=1, trial_dirname_creator=trial_directory_name),
            run_config=tune.RunConfig(name="grid", storage_path=str(output), verbose=0,
                                     log_to_file=True, callbacks=[ProgressCallback(output)]),
        )
        results = tuner.fit()
        rows = collect_grid(results, output)
        selected = choose_configuration(rows)
        write_json(output / "grid_results.json", rows)
        write_json(output / "selection.json", {"selected": selected, "tie_break_rule": TIE_BREAK_RULE,
                   "E_val": selected["best_epoch"], "seed": 42, "learning_rate": .01, "batch_size": 512,
                   "source_sha256": selection.source_sha256, "selection_preprocessing": "2017-2020"})
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "stage": "grid", "error": str(error)})
        raise
    finally:
        ray.shutdown()
    try:
        write_json(output / "status.json", {"status": "running", "stage": "refit"})
        refit = load_data_bundle(prepared / "refit", expected_protocol="refit")
        if refit.source_sha256 != selection.source_sha256:
            raise ValueError("Selection and refit artifacts have different source data")
        pre = refit.preprocessing
        seed_everything(42)
        model = HousingMLP(pre.cardinalities, pre.target_mean, pre.target_std,
                           hidden_width=selected["hidden_width"], embedding_dimension=selected["embedding_dimension"])
        training = make_loader(refit.datasets["train"], 512, training=True, seed=42, device="cuda")
        test = make_loader(refit.datasets["test"], 512, seed=42, device="cuda")

        def refit_progress(row):
            write_json(output / "progress.json", {"stage": "refit", "completed_trials": 25,
                       "epoch": row["epoch"], "train_rmse": row["train_rmse"]})
            print(f"Refit epoch {row['epoch']:02d}: train RMSE SGD {row['train_rmse']:,.2f}", flush=True)

        result = train_regressor(
            model, training, target_mean=pre.target_mean, target_std=pre.target_std,
            config=TrainingConfig(learning_rate=.01, mode="a1_refit", seed=42, device="cuda"),
            run_directory=output / "refit", test_monitor_loader=test,
            preprocessing={**pre.metadata(), "source_sha256": refit.source_sha256},
            model_config=model.metadata(), on_epoch=refit_progress,
        )
        final_metrics = evaluate_final_checkpoint(output / "refit", model, test, device="cuda")
        baseline = next(row for row in rows if row["hidden_width"] == 64 and row["embedding_dimension"] == 8)
        summary = {"status": "completed", "default_baseline": baseline, "selected": selected,
                   "E_val": selected["best_epoch"], "E_star": result.best_epoch,
                   "refit_epochs_completed": result.epochs_completed,
                   "refit_stopped_early": result.stopped_early, "refit_checkpoint": "refit/best_model.pt",
                   "refit_model_config": model.metadata(), "final_test_metrics": final_metrics,
                   "refit_selected_train_metrics": result.selected_metrics["train"],
                   "test_assisted_stopping": True,
                   "test_independence_limitation": "The 2022 test partition selects the A1 stopping checkpoint; its score is not an untouched holdout estimate. The A2/A3 epoch budget inherits this test-assisted selection.",
                   "source_sha256": refit.source_sha256, "seed": 42,
                   "elapsed_seconds": time.perf_counter() - started}
        write_json(output / "summary.json", summary)
        write_json(output / "frozen_a1_settings.json", {
            "hidden_width": selected["hidden_width"], "embedding_dimension": selected["embedding_dimension"],
            "learning_rate": .01, "batch_size": 512, "E_val": selected["best_epoch"],
            "E_star": result.best_epoch, "evaluation_seeds": [42, 43, 44, 45, 46],
            "refit_preprocessing": "refit/preprocessing.pt", "refit_checkpoint": "refit/best_model.pt",
            "source_sha256": refit.source_sha256, "test_assisted_epoch_budget": True,
        })
        from part_a.src.a1_exports import export_a1
        export_a1(output, prepared_directory=prepared)
        write_json(output / "progress.json", {"stage": "complete", "completed_trials": 25,
                   "E_val": selected["best_epoch"], "E_star": result.best_epoch,
                   "test_rmse_sgd": final_metrics["rmse"], "test_r2": final_metrics["r2"]})
        write_json(output / "status.json", {"status": "completed", "stage": "complete"})
        print(f"A1 complete: H={selected['hidden_width']}, d={selected['embedding_dimension']}, "
              f"E_val={selected['best_epoch']}, E*={result.best_epoch}; "
              f"test RMSE SGD {final_metrics['rmse']:,.2f}, R2 {final_metrics['r2']:.6f}", flush=True)
        return summary
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "stage": "refit_or_export", "error": str(error)})
        raise
