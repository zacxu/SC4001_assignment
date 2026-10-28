"""Validation-only lambda selection, then five fixed-budget CUDA refits."""

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

from assignment_common import TrainingConfig, evaluate_final_checkpoint, make_loader, seed_everything
from assignment_common.artifacts import save_seed_results, write_checkpoint, write_json
from assignment_common.metrics import summarize_seed_results
from part_a.src.a2 import EVALUATION_SEEDS, check_same_preprocessing, model_state_fingerprint, sha256_file, validate_frozen_settings
from part_a.src.data import load_data_bundle
from part_a.src.model import HousingMLP
from part_a.src.training import train_regressor
from part_a.src.wide_deep import WideDeepHousingModel

LAMBDAS = (0., .25, .5, 1., 2.)
TIE_BREAK_RULE = "lowest best validation RMSE, then smaller lambda"


def lambda_name(value):
    return "lambda_" + format(value, "g").replace(".", "p")


def baseline_state_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if name.startswith("wide."):
            continue
        digest.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype}".encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def choose_lambda(records):
    values = [row["lambda"] for row in records]
    if len(values) != len(LAMBDAS) or set(values) != set(LAMBDAS):
        raise ValueError("A3 requires exactly one result for each of the five lambdas")
    if any(not math.isfinite(row["validation_rmse"]) for row in records):
        raise ValueError("Validation scores must be finite")
    return min(records, key=lambda row: (row["validation_rmse"], row["lambda"]))


def summarize_comparison(records, baseline_records, *, epoch_budget):
    expected = set(EVALUATION_SEEDS)
    if (len(records) != 5 or len(baseline_records) != 5
            or {row["seed"] for row in records} != expected
            or {row["seed"] for row in baseline_records} != expected):
        raise ValueError("A3 and baseline require exactly seeds 42-46")
    if any(row["epochs_completed"] != epoch_budget or row["checkpoint_epoch"] != epoch_budget
           for row in records + baseline_records):
        raise ValueError("A3 comparisons require the same fixed epoch and final checkpoint")
    records = sorted(records, key=lambda row: row["seed"])
    baseline_records = sorted(baseline_records, key=lambda row: row["seed"])
    paired = [{"seed": row["seed"], "rmse_delta": row["rmse"] - base["rmse"],
               "r2_delta": row["r2"] - base["r2"]} for row, base in zip(records, baseline_records)]
    return {"baseline": summarize_seed_results(baseline_records), "wide_deep": summarize_seed_results(records)}, summarize_seed_results(paired, metrics=("rmse_delta", "r2_delta"))


def run_a3(output_directory, prepared_directory, a1_directory, a2_directory):
    output, prepared, a1, a2 = [Path(path).resolve() for path in
                              (output_directory, prepared_directory, a1_directory, a2_directory)]
    if output.exists():
        raise FileExistsError(f"Choose a new A3 output directory: {output}")
    frozen = json.loads((a1 / "frozen_a1_settings.json").read_text())
    a1_summary = json.loads((a1 / "summary.json").read_text())
    a2_summary = json.loads((a2 / "summary.json").read_text())
    if a1_summary["status"] != "completed" or a2_summary["status"] != "completed":
        raise ValueError("A3 requires completed A1 and A2 experiments")
    validate_frozen_settings(frozen, a1_summary)
    if a2_summary["frozen_a1_settings"] != frozen:
        raise ValueError("A2 baseline controls differ from A1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for A3")
    torch.set_num_threads(4)
    selection = load_data_bundle(prepared / "selection", expected_protocol="selection")
    refit = load_data_bundle(prepared / "refit", expected_protocol="refit")
    if selection.source_sha256 != refit.source_sha256 or refit.source_sha256 != frozen["source_sha256"]:
        raise ValueError("A3 data sources differ from A1")
    selection_metadata = selection.preprocessing.metadata()
    refit_metadata = torch.load(a1 / frozen["refit_preprocessing"], map_location="cpu", weights_only=True)
    check_same_preprocessing(refit_metadata, refit.preprocessing.metadata())
    check_same_preprocessing(refit_metadata, torch.load(a2 / "preprocessing.pt", map_location="cpu", weights_only=True))
    baseline_records = a2_summary["variants"]["baseline"]["per_seed"]
    # Validate the saved comparison's seed/epoch coverage before any training.
    summarize_comparison(baseline_records, baseline_records, epoch_budget=frozen["E_star"])
    output.mkdir(parents=True, exist_ok=False)
    write_checkpoint(output / "selection_preprocessing.pt", selection_metadata)
    write_checkpoint(output / "refit_preprocessing.pt", refit_metadata)
    write_json(output / "baseline_results.json", baseline_records)
    config = {
        "created_at_singapore": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "prepared_directory": str(prepared), "a1_directory": str(a1), "a2_directory": str(a2),
        "frozen_a1_settings": frozen, "lambda_grid": list(LAMBDAS), "selection_seed": 42,
        "tie_break_rule": TIE_BREAK_RULE, "source_sha256": refit.source_sha256,
        "input_hashes": {"a1_frozen": sha256_file(a1 / "frozen_a1_settings.json"),
                         "a1_refit_preprocessing": sha256_file(a1 / frozen["refit_preprocessing"]),
                         "a2_summary": sha256_file(a2 / "summary.json"),
                         "selection_data": sha256_file(prepared / "selection/datasets.pt"),
                         "refit_data": sha256_file(prepared / "refit/datasets.pt")},
        "training": {"optimizer": "Adam", "learning_rate": .01, "weight_decay": 0., "batch_size": 512,
                     "selection_max_epochs": 20, "patience": 4, "min_delta": 1e-4,
                     "refit_mode": "fixed_refit", "refit_epochs": frozen["E_star"]},
        "rows": {"selection_train": len(selection.datasets["train"]), "validation": len(selection.datasets["validation"]),
                 "refit_train": len(refit.datasets["train"]), "test": len(refit.datasets["test"])},
        "device": "cuda", "loader_workers": 0, "torch_cpu_threads": 4,
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)},
    }
    write_json(output / "configuration.json", config)
    started = time.perf_counter()
    grid, records = [], []
    write_json(output / "status.json", {"status": "running", "search_completed": 0, "refits_completed": 0})

    def new_model(pre, value, seed):
        seed_everything(seed)
        baseline = HousingMLP(pre.cardinalities, pre.target_mean, pre.target_std,
                              hidden_width=frozen["hidden_width"], embedding_dimension=frozen["embedding_dimension"])
        reference_hash = model_state_fingerprint(baseline)
        seed_everything(seed)
        model = WideDeepHousingModel(pre.cardinalities, pre.target_mean, pre.target_std,
                                     hidden_width=frozen["hidden_width"], embedding_dimension=frozen["embedding_dimension"],
                                     lambda_value=value)
        if baseline_state_fingerprint(model) != reference_hash:
            raise AssertionError("Adding the wide branch changed baseline initialization")
        return model, reference_hash

    try:
        for value in LAMBDAS:
            pre = selection.preprocessing
            model, base_hash = new_model(pre, value, 42)
            initial_hash = model_state_fingerprint(model)
            train = make_loader(selection.datasets["train"], 512, training=True, seed=42, device="cuda")
            validation = make_loader(selection.datasets["validation"], 512, device="cuda")
            run = output / "search" / lambda_name(value)

            def report_epoch(row):
                write_json(output / "progress.json", {"stage": "lambda_search", "lambda": value,
                           "epoch": row["epoch"], "search_completed": len(grid), "refits_completed": 0})
                print(f"Lambda {value:g}, epoch {row['epoch']}: train RMSE {row['train_rmse']:,.2f}, "
                      f"validation RMSE {row['validation_rmse']:,.2f}", flush=True)

            result = train_regressor(model, train, target_mean=pre.target_mean, target_std=pre.target_std,
                config=TrainingConfig(learning_rate=.01, seed=42, device="cuda", mode="selection"),
                run_directory=run, validation_loader=validation, preprocessing=selection_metadata,
                model_config={**model.metadata(), "experiment": "A3", "stage": "selection",
                              "initial_model_sha256": initial_hash, "baseline_initial_sha256": base_hash}, on_epoch=report_epoch)
            row = {"lambda": value, "seed": 42, "best_epoch": result.best_epoch, "epochs_completed": result.epochs_completed,
                   "stopped_early": result.stopped_early, "trainable_parameters": model.metadata()["trainable_parameters"],
                   "active_parameters": model.metadata()["active_parameters"],
                   "train_rmse": result.selected_metrics["train"]["rmse"], "train_r2": result.selected_metrics["train"]["r2"],
                   "validation_rmse": result.selected_metrics["validation"]["rmse"],
                   "validation_r2": result.selected_metrics["validation"]["r2"],
                   "initial_model_sha256": initial_hash, "baseline_initial_sha256": base_hash,
                   "train_loader_generator_sha256": hashlib.sha256(train.generator.get_state().numpy().tobytes()).hexdigest(),
                   "run_directory": run.relative_to(output).as_posix(), "checkpoint": result.checkpoint_path.relative_to(output).as_posix()}
            if value == 0:
                expected = a1_summary["selected"]
                parity = {"initial_baseline_weights_matched": True, "best_epoch_matches_a1": result.best_epoch == frozen["E_val"],
                          "epochs_completed_match_a1": result.epochs_completed == expected["epochs_completed"],
                          "validation_metrics_match_a1": all(math.isclose(row["validation_" + key], expected["validation_" + key], rel_tol=1e-9, abs_tol=1e-8) for key in ("rmse", "r2")),
                          "training_metrics_match_a1": all(math.isclose(row["train_" + key], expected["train_" + key], rel_tol=1e-9, abs_tol=1e-8) for key in ("rmse", "r2"))}
                if not all(parity.values()):
                    raise AssertionError(f"Lambda-zero A1 parity failed: {parity}")
                write_json(output / "baseline_parity.json", parity)
            grid.append(row)
            write_json(output / "lambda_search.json", grid)
            print(f"Completed lambda {value:g}; best validation RMSE {row['validation_rmse']:,.2f} at epoch {result.best_epoch}", flush=True)
            del model
        selected = choose_lambda(grid)
        write_json(output / "selection.json", {"selected": selected, "tie_break_rule": TIE_BREAK_RULE,
                   "selection_partition": "2021 validation", "selection_seed": 42})
        write_json(output / "frozen_a3_settings.json", {**frozen, "lambda": selected["lambda"],
                   "lambda_selection_best_epoch": selected["best_epoch"], "lambda_selection_partition": "validation_2021",
                   "trainable_parameters": selected["trainable_parameters"],
                   "seed42_checkpoint": "refit/seed_42/best_model.pt", "preprocessing": "refit_preprocessing.pt"})
        for seed in EVALUATION_SEEDS:
            pre = refit.preprocessing
            model, base_hash = new_model(pre, selected["lambda"], seed)
            baseline = next(row for row in baseline_records if row["seed"] == seed)
            if base_hash != baseline["initial_model_sha256"]:
                raise AssertionError("Refit MLP initialization differs from A2 baseline")
            initial_hash = model_state_fingerprint(model)
            train = make_loader(refit.datasets["train"], 512, training=True, seed=seed, device="cuda")
            run = output / "refit" / f"seed_{seed}"

            def report_refit(row):
                write_json(output / "progress.json", {"stage": "fixed_refit", "seed": seed, "epoch": row["epoch"],
                           "search_completed": len(grid), "refits_completed": len(records)})
                print(f"Refit seed {seed}, epoch {row['epoch']}/{frozen['E_star']}: train RMSE {row['train_rmse']:,.2f}", flush=True)

            result = train_regressor(model, train, target_mean=pre.target_mean, target_std=pre.target_std,
                config=TrainingConfig(learning_rate=.01, max_epochs=frozen["E_star"], seed=seed, device="cuda", mode="fixed_refit"),
                run_directory=run, preprocessing=refit_metadata,
                model_config={**model.metadata(), "experiment": "A3", "stage": "fixed_refit",
                              "initial_model_sha256": initial_hash, "baseline_initial_sha256": base_hash}, on_epoch=report_refit)
            metrics = evaluate_final_checkpoint(run, model, make_loader(refit.datasets["test"], 512, device="cuda"), device="cuda")
            row = {"seed": seed, "lambda": selected["lambda"], "trainable_parameters": model.metadata()["trainable_parameters"],
                   "active_parameters": model.metadata()["active_parameters"], "epochs_completed": result.epochs_completed,
                   "checkpoint_epoch": result.best_epoch, "initial_model_sha256": initial_hash, "baseline_initial_sha256": base_hash,
                   "train_loader_generator_sha256": hashlib.sha256(train.generator.get_state().numpy().tobytes()).hexdigest(),
                   "rmse": metrics["rmse"], "r2": metrics["r2"], "test_count": metrics["count"],
                   "train_rmse": result.selected_metrics["train"]["rmse"], "train_r2": result.selected_metrics["train"]["r2"],
                   "run_directory": run.relative_to(output).as_posix(), "checkpoint": result.checkpoint_path.relative_to(output).as_posix(),
                   "test_metrics_file": (run / "test_metrics.json").relative_to(output).as_posix()}
            if row["train_loader_generator_sha256"] != baseline["train_loader_generator_sha256"]:
                raise AssertionError("Refit shuffling differs from A2 baseline")
            if selected["lambda"] == 0 and any(not math.isclose(row[key], baseline[key], rel_tol=1e-9, abs_tol=1e-8) for key in ("rmse", "r2")):
                raise AssertionError("Lambda-zero final refit differs from baseline")
            records.append(row)
            write_json(output / "per_seed_results.json", records)
            write_json(output / "status.json", {"status": "running", "search_completed": 5, "refits_completed": len(records)})
            print(f"Completed refit {len(records)}/5: seed {seed}, test RMSE SGD {metrics['rmse']:,.2f}, R2 {metrics['r2']:.6f}", flush=True)
            del model
        summaries, paired = summarize_comparison(records, baseline_records, epoch_budget=frozen["E_star"])
        for name in ("baseline", "wide_deep"):
            save_seed_results(output / "seed_results" / name, summaries[name]["per_seed"])
        summary = {"status": "completed", "search_runs": 5, "refit_runs": 5, "selected": selected,
                   "epoch_budget": frozen["E_star"], "frozen_a1_settings": frozen, "variants": summaries,
                   "paired_differences": paired, "test_assisted_epoch_budget": True, "test_monitoring_during_a3": False,
                   "baseline_origin": "A2 fresh fixed-budget five-seed baseline; reused without another test evaluation",
                   "a1_individual_final": a1_summary["final_test_metrics"], "elapsed_seconds": time.perf_counter() - started}
        write_json(output / "summary.json", summary)
        from part_a.src.a3_exports import export_a3
        export_a3(output)
        write_json(output / "status.json", {"status": "completed", "search_completed": 5, "refits_completed": 5})
        write_json(output / "progress.json", {"stage": "complete", "search_completed": 5, "refits_completed": 5})
        return summary
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "search_completed": len(grid), "refits_completed": len(records), "error": str(error)})
        raise
