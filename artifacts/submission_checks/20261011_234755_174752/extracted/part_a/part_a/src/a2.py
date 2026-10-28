"""Fifteen fresh, fixed-epoch, paired-seed A2 refits and single final test scores."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import time
from zoneinfo import ZoneInfo

import torch

from assignment_common import TrainingConfig, evaluate_final_checkpoint, make_loader, seed_everything
from assignment_common.artifacts import save_seed_results, write_checkpoint, write_json
from assignment_common.metrics import summarize_seed_results
from part_a.src.ablations import VARIANTS, VARIANT_LABELS, build_a2_model
from part_a.src.data import HousingPreprocessor, load_data_bundle
from part_a.src.training import train_regressor

EVALUATION_SEEDS = (42, 43, 44, 45, 46)


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_state_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype}".encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_frozen_settings(frozen, a1_summary):
    if (frozen["hidden_width"] != a1_summary["selected"]["hidden_width"]
            or frozen["embedding_dimension"] != a1_summary["selected"]["embedding_dimension"]
            or frozen["E_star"] != a1_summary["E_star"]
            or frozen["E_val"] != a1_summary["E_val"]
            or frozen["source_sha256"] != a1_summary["source_sha256"]):
        raise ValueError("Frozen settings do not match the completed A1 selection/refit")
    if (type(frozen["E_star"]) is not int or not 1 <= frozen["E_star"] <= 20
            or frozen["learning_rate"] != .01 or frozen["batch_size"] != 512
            or frozen["evaluation_seeds"] != list(EVALUATION_SEEDS)
            or frozen["test_assisted_epoch_budget"] is not True):
        raise ValueError("A2 requires the fixed A1 budget, baseline optimizer settings, and seeds 42-46")


def check_same_preprocessing(first, second):
    first_pre, second_pre = HousingPreprocessor.from_metadata(first), HousingPreprocessor.from_metadata(second)
    if (first_pre.category_maps != second_pre.category_maps
            or first_pre.fit_years != second_pre.fit_years or first_pre.fit_row_count != second_pre.fit_row_count
            or first_pre.target_mean != second_pre.target_mean or first_pre.target_std != second_pre.target_std):
        raise ValueError("A2 must reuse exactly the A1 combined-data preprocessing")
    torch.testing.assert_close(first_pre.continuous_mean, second_pre.continuous_mean, rtol=0, atol=0)
    torch.testing.assert_close(first_pre.continuous_std, second_pre.continuous_std, rtol=0, atol=0)


def summarize_a2_records(records, *, epoch_budget):
    pairs = [(row["variant"], row["seed"]) for row in records]
    expected = {(variant, seed) for variant in VARIANTS for seed in EVALUATION_SEEDS}
    if len(pairs) != 15 or set(pairs) != expected:
        raise ValueError("A2 requires all 15 unique variant/seed results")
    if any(row["epochs_completed"] != epoch_budget or row["checkpoint_epoch"] != epoch_budget for row in records):
        raise ValueError("Every A2 run must use the same fixed epoch budget and final checkpoint")
    grouped = {variant: sorted((row for row in records if row["variant"] == variant), key=lambda row: row["seed"])
               for variant in VARIANTS}
    summaries = {variant: summarize_seed_results(rows) for variant, rows in grouped.items()}
    baseline = {row["seed"]: row for row in grouped["baseline"]}
    paired = {}
    for variant in VARIANTS[1:]:
        differences = [{"seed": row["seed"], "rmse_delta": row["rmse"] - baseline[row["seed"]]["rmse"],
                        "r2_delta": row["r2"] - baseline[row["seed"]]["r2"]} for row in grouped[variant]]
        paired[variant] = summarize_seed_results(differences, metrics=("rmse_delta", "r2_delta"))
    return summaries, paired


def run_a2(output_directory, prepared_directory, a1_directory):
    output, prepared, a1 = map(lambda path: Path(path).resolve(), (output_directory, prepared_directory, a1_directory))
    if output.exists():
        raise FileExistsError(f"Choose a new A2 output directory: {output}")
    frozen = json.loads((a1 / "frozen_a1_settings.json").read_text(encoding="utf-8"))
    a1_summary = json.loads((a1 / "summary.json").read_text(encoding="utf-8"))
    if a1_summary["status"] != "completed":
        raise ValueError("A2 requires a completed A1 experiment")
    validate_frozen_settings(frozen, a1_summary)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full A2 experiment")
    torch.set_num_threads(4)
    bundle = load_data_bundle(prepared / "refit", expected_protocol="refit")
    frozen_pre_path = a1 / frozen["refit_preprocessing"]
    metadata = torch.load(frozen_pre_path, map_location="cpu", weights_only=True)
    check_same_preprocessing(metadata, bundle.preprocessing.metadata())
    if metadata["source_sha256"] != bundle.source_sha256 or frozen["source_sha256"] != bundle.source_sha256:
        raise ValueError("A1/A2 prepared data sources differ")
    pre = HousingPreprocessor.from_metadata(metadata)
    output.mkdir(parents=True, exist_ok=False)
    write_checkpoint(output / "preprocessing.pt", metadata)
    configuration = {
        "created_at_singapore": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "a1_directory": str(a1), "prepared_directory": str(prepared), "frozen_a1_settings": frozen,
        "a1_preprocessing_sha256": sha256_file(frozen_pre_path),
        "a1_frozen_settings_sha256": sha256_file(a1 / "frozen_a1_settings.json"),
        "prepared_refit_tensors_sha256": sha256_file(prepared / "refit/datasets.pt"),
        "source_sha256": bundle.source_sha256, "variants": list(VARIANTS), "seeds": list(EVALUATION_SEEDS),
        "training": {"optimizer": "Adam", "learning_rate": .01, "batch_size": 512,
                     "mode": "fixed_refit", "epochs": frozen["E_star"], "monitoring_loaders": None},
        "training_rows": len(bundle.datasets["train"]), "test_rows": len(bundle.datasets["test"]),
        "protocol": "Fresh combined-data runs; exactly E* epochs; one final test evaluation each.",
        "loader_workers": 0, "torch_cpu_threads": 4, "device": "cuda",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)},
    }
    write_json(output / "configuration.json", configuration)
    write_json(output / "status.json", {"status": "running", "completed_runs": 0, "total_runs": 15})
    records, initialization = [], {}
    started = time.perf_counter()
    try:
        for seed in EVALUATION_SEEDS:
            for variant in VARIANTS:
                seed_everything(seed)
                model = build_a2_model(variant, pre.cardinalities, pre.target_mean, pre.target_std,
                                       hidden_width=frozen["hidden_width"], embedding_dimension=frozen["embedding_dimension"])
                initial_hash = model_state_fingerprint(model)
                initialization[(variant, seed)] = initial_hash
                if variant == "sigmoid" and initial_hash != initialization[("baseline", seed)]:
                    raise AssertionError("Sigmoid and baseline initial weights are not identical")
                train_loader = make_loader(bundle.datasets["train"], 512, training=True, seed=seed, device="cuda")
                run_directory = output / "runs" / variant / f"seed_{seed}"
                model_config = {**model.metadata(), "experiment": "A2", "variant": variant,
                                "initial_model_sha256": initial_hash}
                count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
                if count != model_config["trainable_parameters"]:
                    raise AssertionError("Model parameter count disagrees with its metadata")

                def report_epoch(row):
                    write_json(output / "progress.json", {"stage": "training", "completed_runs": len(records),
                               "total_runs": 15, "variant": variant, "seed": seed, "epoch": row["epoch"],
                               "epoch_budget": frozen["E_star"], "train_rmse": row["train_rmse"]})
                    print(f"{variant} seed {seed}: epoch {row['epoch']}/{frozen['E_star']}, "
                          f"train RMSE SGD {row['train_rmse']:,.2f}", flush=True)

                result = train_regressor(
                    model, train_loader, target_mean=pre.target_mean, target_std=pre.target_std,
                    config=TrainingConfig(learning_rate=.01, max_epochs=frozen["E_star"], seed=seed,
                                          device="cuda", mode="fixed_refit"),
                    run_directory=run_directory, preprocessing=metadata, model_config=model_config,
                    on_epoch=report_epoch,
                )
                test_loader = make_loader(bundle.datasets["test"], 512, seed=seed, device="cuda")
                metrics = evaluate_final_checkpoint(run_directory, model, test_loader, device="cuda")
                run_summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
                record = {
                    "variant": variant, "label": VARIANT_LABELS[variant], "seed": seed,
                    "hidden_width": frozen["hidden_width"], "embedding_dimension": model_config["embedding_dimension"],
                    "input_width": model_config["input_width"], "trainable_parameters": count,
                    "epochs_completed": result.epochs_completed, "checkpoint_epoch": result.best_epoch,
                    "stopped_early": result.stopped_early, "initial_model_sha256": initial_hash,
                    "train_loader_generator_sha256": hashlib.sha256(train_loader.generator.get_state().numpy().tobytes()).hexdigest(),
                    "rmse": metrics["rmse"], "r2": metrics["r2"], "test_count": metrics["count"],
                    "train_rmse": result.selected_metrics["train"]["rmse"],
                    "train_r2": result.selected_metrics["train"]["r2"],
                    "fit_seconds": run_summary["elapsed_seconds"], "run_directory": run_directory.relative_to(output).as_posix(),
                    "checkpoint": (run_directory / "best_model.pt").relative_to(output).as_posix(),
                    "test_metrics_file": (run_directory / "test_metrics.json").relative_to(output).as_posix(),
                }
                records.append(record)
                write_json(run_directory / "a2_result.json", record)
                write_json(output / "per_seed_results.json", records)
                write_json(output / "status.json", {"status": "running", "completed_runs": len(records), "total_runs": 15})
                print(f"Completed {len(records)}/15: {variant}, seed {seed}; "
                      f"test RMSE SGD {metrics['rmse']:,.2f}, R2 {metrics['r2']:.6f}", flush=True)
                del model
        summaries, paired = summarize_a2_records(records, epoch_budget=frozen["E_star"])
        for variant in VARIANTS:
            save_seed_results(output / "seed_results" / variant, summaries[variant]["per_seed"])
        summary = {"status": "completed", "completed_runs": 15, "epoch_budget": frozen["E_star"],
                   "frozen_a1_settings": frozen, "variants": summaries, "paired_differences": paired,
                   "difference_definition": "ablation minus same-seed baseline; lower RMSE and higher R2 are better",
                   "test_assisted_epoch_budget": True, "test_monitoring_during_a2": False,
                   "elapsed_seconds": time.perf_counter() - started}
        write_json(output / "summary.json", summary)
        from part_a.src.a2_exports import export_a2
        export_a2(output)
        write_json(output / "status.json", {"status": "completed", "completed_runs": 15, "total_runs": 15})
        write_json(output / "progress.json", {"stage": "complete", "completed_runs": 15, "epoch_budget": frozen["E_star"]})
        return summary
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "completed_runs": len(records), "error": str(error)})
        raise
