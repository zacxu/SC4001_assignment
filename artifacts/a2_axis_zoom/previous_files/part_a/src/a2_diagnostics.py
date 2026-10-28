"""Held-out A2 learning curves without changing the completed final refits.

The FAQ requests train/validation RMSE and R2 against epochs. Validation is
already part of the combined-data final refits, so those histories cannot
provide held-out validation curves. These additional, explicitly diagnostic
runs use the original chronological split and the frozen A1 settings. They
never evaluate test data or choose new hyperparameters/epochs.
"""

from __future__ import annotations

from datetime import datetime
import json
import math
import os
from pathlib import Path
import platform
from zoneinfo import ZoneInfo

import torch
from torch.utils.data import TensorDataset

from assignment_common import TrainingConfig, make_loader, seed_everything
from assignment_common.artifacts import write_checkpoint, write_json
from part_a.src.a2 import model_state_fingerprint, sha256_file, validate_frozen_settings
from part_a.src.ablations import VARIANTS, VARIANT_LABELS, build_a2_model
from part_a.src.data import HousingPreprocessor, TRAIN_YEARS
from part_a.src.training import train_regressor


def load_diagnostic_partitions(prepared_directory):
    """Read only train/validation entries from the prepared selection payload."""
    directory = Path(prepared_directory) / "selection"
    metadata = torch.load(directory / "preprocessing.pt", map_location="cpu", weights_only=True)
    preprocessing = HousingPreprocessor.from_metadata(metadata)
    payload = torch.load(directory / "datasets.pt", map_location="cpu", weights_only=True)
    if (payload.get("format_version") != 1 or payload.get("protocol") != "selection"
            or preprocessing.fit_years != TRAIN_YEARS):
        raise ValueError("Diagnostics require the original train-only selection preprocessing")
    datasets, row_ids = {}, {}
    # Do not inspect, make a loader for, or evaluate the test partition.
    for name in ("train", "validation"):
        partition = payload["partitions"][name]
        categorical, continuous, targets, ids = (
            partition[key] for key in ("categorical", "continuous", "target_sgd", "row_ids")
        )
        count = len(targets)
        if (count == 0 or categorical.shape != (count, 4) or continuous.shape != (count, 6)
                or targets.shape != (count,) or ids.shape != (count,)
                or categorical.dtype != torch.long or ids.dtype != torch.long
                or continuous.dtype != torch.float32 or targets.dtype != torch.float32
                or not torch.isfinite(continuous).all() or not torch.isfinite(targets).all()):
            raise ValueError(f"Invalid diagnostic {name} tensors")
        for index, cardinality in enumerate(preprocessing.cardinalities):
            if not ((categorical[:, index] >= 0) & (categorical[:, index] < cardinality)).all():
                raise ValueError(f"Invalid diagnostic category IDs in {name}")
        datasets[name] = TensorDataset(categorical, continuous, targets)
        row_ids[name] = ids
    all_ids = torch.cat(list(row_ids.values()))
    if (all_ids < 0).any() or len(torch.unique(all_ids)) != len(all_ids):
        raise ValueError("Diagnostic training and held-out validation partitions overlap")
    if len(datasets["train"]) != preprocessing.fit_row_count:
        raise ValueError("Diagnostic training rows disagree with fitted preprocessing")
    return datasets, preprocessing, metadata, payload["source_sha256"]


def diagnostic_histories(directory):
    """Validate saved curves so exports cannot label training rows as validation."""
    output = Path(directory)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "completed" or summary.get("test_evaluated") is not False:
        raise ValueError("Exports require completed diagnostics without test evaluation")
    histories = {}
    for variant in VARIANTS:
        rows = [json.loads(line) for line in
                (output / "runs" / variant / "seed_42/history.jsonl").read_text(encoding="utf-8").splitlines()]
        if [row["epoch"] for row in rows] != list(range(1, summary["epoch_budget"] + 1)):
            raise ValueError("Diagnostic histories must contain every fixed-budget epoch")
        for row in rows:
            required = ("train_rmse", "train_r2", "validation_rmse", "validation_r2")
            if (any(name not in row for name in required)
                    or any(name.startswith("test_") for name in row)
                    or any(not math.isfinite(row[name]) for name in required)):
                raise ValueError("Diagnostics require train/validation metrics and forbid test epoch metrics")
            if (row["train_count"] != summary["training_rows"]
                    or row["validation_count"] != summary["validation_rows"]):
                raise ValueError("Curve metrics must evaluate all original train/validation rows")
        histories[variant] = rows
    return summary, histories


def export_a2_diagnostics(directory):
    """Regenerate separate categorical/activation epoch figures from saved history."""
    output = Path(directory)
    summary, histories = diagnostic_histories(output)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "artifacts/cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    # Share ranges across both comparisons so the baseline retains the same scale.
    # The explicit padding shows every train/validation point without starting at zero.
    limits = {}
    for metric in ("rmse", "r2"):
        values = [row[f"{partition}_{metric}"] for rows in histories.values()
                  for row in rows for partition in ("train", "validation")]
        lower, upper = min(values), max(values)
        padding = max((upper - lower) * .08, abs(lower) * .01, 1e-6)
        limits[metric] = (lower - padding, upper + padding)
    for variant, name, title in (
        ("continuous_only", "categorical_epoch_curves", "Categorical-input ablation"),
        ("sigmoid", "activation_epoch_curves", "Hidden-activation ablation"),
    ):
        figure, axes = plt.subplots(1, 2, figsize=(10, 4.4))
        for axis, metric, label in zip(axes, ("rmse", "r2"), ("RMSE (SGD)", "R²")):
            for model_variant, color in (("baseline", "#2775b6"), (variant, "#c57225")):
                rows = histories[model_variant]
                for partition, linestyle, marker in (("train", "-", "o"), ("validation", "--", "s")):
                    axis.plot([row["epoch"] for row in rows], [row[f"{partition}_{metric}"] for row in rows],
                              color=color, linestyle=linestyle, marker=marker, markersize=4,
                              label=f"{VARIANT_LABELS[model_variant]}: {partition}")
            axis.set(xlabel="Epoch", ylabel=label, xticks=range(1, summary["epoch_budget"] + 1))
            axis.set_ylim(*limits[metric])
            axis.grid(alpha=.22)
            if metric == "rmse":
                axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        figure.suptitle(f"A2: {title} — original train/validation split, seed 42", fontsize=12)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
        figure.text(.5, .015, "Zoomed y axes; identical ranges across both A2 comparisons.",
                    ha="center", fontsize=8)
        figure.tight_layout(rect=(0, .13, 1, .93))
        figure.savefig(figures / f"{name}.png", dpi=220, bbox_inches="tight")
        figure.savefig(figures / f"{name}.pdf", bbox_inches="tight")
        plt.close(figure)
    return summary


def verify_a2_diagnostics(directory):
    """Audit saved split, checkpoint and curve provenance without new inference."""
    output = Path(directory)
    summary, histories = diagnostic_histories(output)
    configuration = json.loads((output / "configuration.json").read_text(encoding="utf-8"))
    preprocessing = HousingPreprocessor.load(output / "preprocessing.pt")
    if (configuration["training_years"] != list(TRAIN_YEARS)
            or configuration["validation_years"] != [2021]
            or preprocessing.fit_years != TRAIN_YEARS
            or configuration["seed"] != 42 or configuration["test_evaluated"] is not False
            or summary["final_results_changed"] is not False):
        raise ValueError("Diagnostic split or final-result isolation is incorrect")
    if any(output.rglob("test_metrics.json")) or any(output.rglob("internal_test_monitor.jsonl")):
        raise ValueError("A2 diagnostics must not contain any test scoring")
    hashes = {"configuration.json": sha256_file(output / "configuration.json"),
              "preprocessing.pt": sha256_file(output / "preprocessing.pt")}
    for record in summary["records"]:
        variant = record["variant"]
        rows = histories[variant]
        run_directory = output / record["run_directory"]
        checkpoint = torch.load(run_directory / "best_model.pt", map_location="cpu", weights_only=True)
        config = checkpoint["training_config"]
        model = checkpoint["model_config"]
        expected_epoch = min(rows, key=lambda row: (row["validation_rmse"], row["epoch"]))["epoch"]
        if (config["learning_rate"] != .01 or config["max_epochs"] != 4
                or config["patience"] != 4 or config["min_delta"] != 1e-4 or config["seed"] != 42
                or config["mode"] != "selection" or checkpoint["epoch"] != expected_epoch
                or model["hidden_width"] != configuration["frozen_settings"]["hidden_width"]
                or model["embedding_dimension"] != (None if variant == "continuous_only" else configuration["frozen_settings"]["embedding_dimension"])
                or checkpoint["preprocessing"]["fit_years"] != list(TRAIN_YEARS)):
            raise ValueError("Diagnostic checkpoint disagrees with frozen settings or original split")
        if (record["last_epoch_train_rmse"] != rows[-1]["train_rmse"]
                or record["last_epoch_validation_rmse"] != rows[-1]["validation_rmse"]):
            raise ValueError("Diagnostic summary disagrees with saved full histories")
        for name in ("history.csv", "history.jsonl", "best_model.pt", "preprocessing.pt"):
            path = run_directory / name
            hashes[path.relative_to(output).as_posix()] = sha256_file(path)
    initial = {row["variant"]: row["initial_model_sha256"] for row in summary["records"]}
    if initial["baseline"] != initial["sigmoid"]:
        raise ValueError("Activation comparison must use identical initial parameter tensors")
    for name in ("categorical_epoch_curves", "activation_epoch_curves"):
        for extension in ("png", "pdf"):
            path = output / "figures" / f"{name}.{extension}"
            hashes[path.relative_to(output).as_posix()] = sha256_file(path)
    verification = {"status": "passed", "training_runs": 3, "epochs_per_run": 4,
                    "training_rows_per_epoch": summary["training_rows"],
                    "validation_rows_per_epoch": summary["validation_rows"],
                    "test_forward_passes": 0, "test_metrics_files": 0,
                    "preprocessing_fit_years": list(TRAIN_YEARS), "artifact_sha256": hashes,
                    "validation_checkpoint_role": "diagnostic evidence only; never used in final five-seed results"}
    write_json(output / "verification.json", verification)
    return verification


def run_a2_diagnostics(output_directory, prepared_directory, a1_directory, faq_path):
    """Run three original-split seed-42 diagnostics; preserve all final evidence."""
    output, prepared, a1 = map(Path, (output_directory, prepared_directory, a1_directory))
    if output.exists():
        raise FileExistsError(f"Choose a new diagnostic directory: {output}")
    frozen = json.loads((a1 / "frozen_a1_settings.json").read_text(encoding="utf-8"))
    a1_summary = json.loads((a1 / "summary.json").read_text(encoding="utf-8"))
    validate_frozen_settings(frozen, a1_summary)
    if frozen["E_star"] != 4:
        raise ValueError("This diagnostic protocol was frozen at the completed A1 E*=4 budget")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full A2 diagnostics")
    torch.set_num_threads(4)
    datasets, preprocessing, metadata, source_sha256 = load_diagnostic_partitions(prepared)
    if source_sha256 != frozen["source_sha256"]:
        raise ValueError("Diagnostic data must match the completed A1 source")
    output.mkdir(parents=True, exist_ok=False)
    write_checkpoint(output / "preprocessing.pt", metadata)
    configuration = {
        "experiment": "A2_original_split_diagnostics", "seed": 42,
        "created_at_singapore": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "source_sha256": source_sha256, "faq_sha256": sha256_file(faq_path),
        "a1_frozen_settings_sha256": sha256_file(a1 / "frozen_a1_settings.json"),
        "selection_preprocessing_sha256": sha256_file(prepared / "selection/preprocessing.pt"),
        "selection_tensors_sha256": sha256_file(prepared / "selection/datasets.pt"),
        "frozen_settings": frozen, "training_years": list(TRAIN_YEARS), "validation_years": [2021],
        "preprocessing_fit_years": list(preprocessing.fit_years), "training_rows": len(datasets["train"]),
        "validation_rows": len(datasets["validation"]), "test_evaluated": False,
        "training": {"optimizer": "Adam", "learning_rate": .01, "batch_size": 512,
                     "epochs": frozen["E_star"], "patience": 4, "min_delta": 1e-4},
        "purpose": "Held-out train/validation curves only; completed combined-data five-seed final results are unchanged.",
        "checkpoint_policy": "Shared selection trainer records a validation-best diagnostic checkpoint; it is never tested or used for final-model selection.",
        "epoch_budget_policy": "Frozen E*=4, with patience=4 so early stopping cannot shorten these four-epoch diagnostics; no retuning.",
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)},
    }
    write_json(output / "configuration.json", configuration)
    write_json(output / "status.json", {"status": "running", "completed_runs": 0, "total_runs": 3})
    records, initial_fingerprints = [], {}
    try:
        for variant in VARIANTS:
            seed_everything(42)
            model = build_a2_model(variant, preprocessing.cardinalities,
                                   preprocessing.target_mean, preprocessing.target_std,
                                   hidden_width=frozen["hidden_width"], embedding_dimension=frozen["embedding_dimension"])
            initial_fingerprints[variant] = model_state_fingerprint(model)
            if variant == "sigmoid" and initial_fingerprints[variant] != initial_fingerprints["baseline"]:
                raise AssertionError("Baseline and Sigmoid initial parameters must match")
            train_loader = make_loader(datasets["train"], 512, training=True, seed=42, device="cuda")
            validation_loader = make_loader(datasets["validation"], 512, seed=42, device="cuda")
            run_directory = output / "runs" / variant / "seed_42"

            def report_epoch(row):
                write_json(output / "progress.json", {"variant": variant, "epoch": row["epoch"],
                           "completed_runs": len(records), "total_runs": 3,
                           "train_rmse": row["train_rmse"], "validation_rmse": row["validation_rmse"]})
                print(f"A2 diagnostic {variant}, epoch {row['epoch']}/4: "
                      f"train RMSE {row['train_rmse']:,.2f}, validation RMSE {row['validation_rmse']:,.2f}", flush=True)

            result = train_regressor(
                model, train_loader, target_mean=preprocessing.target_mean, target_std=preprocessing.target_std,
                config=TrainingConfig(learning_rate=.01, max_epochs=4, patience=4, min_delta=1e-4,
                                      seed=42, device="cuda", mode="selection"),
                run_directory=run_directory, validation_loader=validation_loader,
                preprocessing=metadata, model_config={**model.metadata(), "experiment": "A2_diagnostic",
                               "variant": variant, "initial_model_sha256": initial_fingerprints[variant]},
                on_epoch=report_epoch,
            )
            if result.epochs_completed != 4:
                raise AssertionError("Diagnostics must complete the frozen four-epoch budget")
            record = {"variant": variant, "seed": 42, "epochs_completed": result.epochs_completed,
                      "diagnostic_checkpoint_epoch": result.best_epoch,
                      "initial_model_sha256": initial_fingerprints[variant],
                      "run_directory": run_directory.relative_to(output).as_posix(),
                      "last_epoch_train_rmse": result.history[-1]["train_rmse"],
                      "last_epoch_validation_rmse": result.history[-1]["validation_rmse"]}
            records.append(record)
            write_json(output / "status.json", {"status": "running", "completed_runs": len(records), "total_runs": 3})
            del model
        summary = {"status": "completed", "completed_runs": 3, "epoch_budget": 4, "seed": 42,
                   "training_rows": len(datasets["train"]), "validation_rows": len(datasets["validation"]),
                   "preprocessing_fit_years": list(TRAIN_YEARS), "test_evaluated": False,
                   "final_results_changed": False, "records": records}
        write_json(output / "summary.json", summary)
        export_a2_diagnostics(output)
        verify_a2_diagnostics(output)
        write_json(output / "status.json", {"status": "completed", "completed_runs": 3, "total_runs": 3})
        return summary
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "completed_runs": len(records), "error": str(error)})
        raise
