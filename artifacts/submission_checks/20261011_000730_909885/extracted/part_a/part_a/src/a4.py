"""Full-test, ten-input FeatureAblation for the paired seed-42 housing models."""

from __future__ import annotations

from datetime import datetime
import json
import math
import os
from pathlib import Path
import platform
import time
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "artifacts/cache/matplotlib"))

import captum
from captum.attr import FeatureAblation
import torch

from assignment_common import restore_checkpoint, seed_everything
from assignment_common.artifacts import write_checkpoint, write_json
from part_a.src.a2 import check_same_preprocessing, model_state_fingerprint, sha256_file
from part_a.src.data import CATEGORICAL_FEATURES, CONTINUOUS_FEATURES, REFIT_YEARS, load_data_bundle
from part_a.src.model import HousingMLP
from part_a.src.wide_deep import WideDeepHousingModel

FEATURE_NAMES = (*CATEGORICAL_FEATURES, *CONTINUOUS_FEATURES)
MODEL_NAMES = ("baseline", "wide_deep")


def make_references(preprocessing, training_categorical):
    """Modes from saved refit inputs; means already encoded as continuous zeros."""
    if preprocessing.fit_years != REFIT_YEARS:
        raise ValueError("A4 references require the saved 2017-2021 refit preprocessing")
    if (training_categorical.ndim != 2 or training_categorical.shape != (preprocessing.fit_row_count, 4)
            or training_categorical.dtype != torch.long or len(training_categorical) == 0):
        raise ValueError("Reference categories must cover every saved refit training row")
    modes, metadata = [], []
    for index, (feature, cardinality) in enumerate(zip(CATEGORICAL_FEATURES, preprocessing.cardinalities)):
        values = training_categorical[:, index].cpu()
        if not ((values > 0) & (values < cardinality)).all():
            raise ValueError("Refit training categories must be known, valid IDs")
        counts = torch.bincount(values, minlength=cardinality)
        mode = int(counts.argmax())  # Exact ties choose the smallest sorted encoded ID.
        reverse = {encoded: raw for raw, encoded in preprocessing.category_maps[feature].items()}
        modes.append(mode)
        metadata.append({"feature": feature, "kind": "categorical", "reference_raw": reverse[mode],
                         "reference_encoded": mode, "training_mode_count": int(counts[mode]),
                         "rule": "2017-2021 training mode; smallest encoded ID breaks ties"})
    for index, feature in enumerate(CONTINUOUS_FEATURES):
        metadata.append({"feature": feature, "kind": "continuous",
                         "reference_raw": float(preprocessing.continuous_mean[index]), "reference_encoded": 0.,
                         "training_mode_count": None, "rule": "saved 2017-2021 training mean (standardized zero)"})
    return {"categorical": torch.tensor([modes], dtype=torch.long),
            "continuous": torch.zeros((1, 6), dtype=torch.float32),
            "feature_names": list(FEATURE_NAMES), "fit_years": list(REFIT_YEARS),
            "fit_row_count": preprocessing.fit_row_count, "details": metadata}


def feature_masks(device="cpu"):
    # Globally distinct IDs across the input tuple: no accidental grouping of
    # categorical and continuous columns or attribution of embedding coordinates.
    return (torch.arange(4, device=device, dtype=torch.long).reshape(1, 4),
            torch.arange(4, 10, device=device, dtype=torch.long).reshape(1, 6))


def price_forward(model):
    mean, std = float(model.target_mean), float(model.target_std)
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise ValueError("Invalid restored target statistics")

    def forward(categorical, continuous):
        # Keep network computation in its saved float32 precision, then invert
        # the target transform once in float64 to reduce price cancellation error.
        output = model(categorical, continuous)
        if output.ndim != 1 or output.shape[0] != categorical.shape[0]:
            raise ValueError("FeatureAblation requires one scalar prediction per row")
        return output.double() * std + mean

    return forward


def compute_attributions(model, categorical, continuous, references, *, batch_size=512,
                         perturbations_per_eval=1, device="cpu", on_batch=None):
    if (categorical.ndim != 2 or categorical.shape[1] != 4 or categorical.dtype != torch.long
            or continuous.shape != (len(categorical), 6) or continuous.dtype != torch.float32
            or len(categorical) == 0 or not torch.isfinite(continuous).all()):
        raise ValueError("Explain nonempty aligned categorical/continuous tensors with ten original features")
    if (type(batch_size) is not int or batch_size < 1 or type(perturbations_per_eval) is not int
            or perturbations_per_eval < 1):
        raise ValueError("Use positive integer batching settings")
    if references["categorical"].shape != (1, 4) or references["continuous"].shape != (1, 6):
        raise ValueError("References must describe exactly ten original input columns")
    forward = price_forward(model)
    ablator = FeatureAblation(forward)
    baselines = (references["categorical"].to(device), references["continuous"].to(device))
    masks = feature_masks(device)
    previous_mode = model.training
    model.eval()
    attributions, predictions = [], []
    try:
        with torch.no_grad():
            for start in range(0, len(categorical), batch_size):
                stop = min(start + batch_size, len(categorical))
                inputs = (categorical[start:stop].to(device), continuous[start:stop].to(device))
                result = ablator.attribute(inputs, baselines=baselines, feature_mask=masks,
                                           perturbations_per_eval=perturbations_per_eval)
                values = torch.cat(result, dim=1).detach().cpu().double()
                prediction = forward(*inputs).detach().cpu().double()
                if values.shape != (stop-start, 10) or not torch.isfinite(values).all() or not torch.isfinite(prediction).all():
                    raise FloatingPointError("FeatureAblation returned invalid per-row SGD values")
                attributions.append(values)
                predictions.append(prediction)
                if on_batch is not None:
                    on_batch(stop, len(categorical))
    finally:
        model.train(previous_mode)
    return torch.cat(attributions), torch.cat(predictions)


def global_importance(attributions):
    if (attributions.ndim != 2 or attributions.shape[1] != 10 or len(attributions) == 0
            or not torch.isfinite(attributions).all()):
        raise ValueError("Aggregate a finite nonempty N-by-10 attribution matrix")
    return attributions.double().abs().mean(dim=0)


def rank_importance(scores):
    if scores.shape != (10,) or not torch.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Rank ten finite nonnegative global scores")
    order = sorted(range(10), key=lambda index: (-float(scores[index]), index))
    return [{"rank": rank, "feature": FEATURE_NAMES[index], "feature_index": index,
             "mean_absolute_attribution_sgd": float(scores[index])} for rank, index in enumerate(order, 1)]


def manual_ablation_differences(model, categorical, continuous, references):
    """Independent coordinate replacement, without Captum or feature masks."""
    forward = price_forward(model)
    with torch.no_grad():
        original = forward(categorical, continuous)
        changes = []
        for feature in range(10):
            cats, cont = categorical.clone(), continuous.clone()
            if feature < 4:
                cats[:, feature] = references["categorical"][0, feature].to(cats.device)
            else:
                cont[:, feature-4] = references["continuous"][0, feature-4].to(cont.device)
            changes.append(original - forward(cats, cont))
    return torch.stack(changes, dim=1)


def validate_checkpoint_pair(baseline, wide, preprocessing, a1_frozen, a3_frozen):
    expected_task = {"kind": "regression", "target_mean": preprocessing.target_mean, "target_std": preprocessing.target_std}
    for name, checkpoint in (("baseline", baseline), ("wide_deep", wide)):
        if checkpoint["training_config"]["seed"] != 42 or checkpoint["epoch"] != a1_frozen["E_star"]:
            raise ValueError("A4 requires matching seed-42, E* final checkpoints")
        if checkpoint["task"] != expected_task:
            raise ValueError("Checkpoint target scaling differs from shared refit preprocessing")
        check_same_preprocessing(checkpoint["preprocessing"], preprocessing.metadata())
        metadata = checkpoint["model_config"]
        if (metadata["hidden_width"] != a1_frozen["hidden_width"]
                or metadata["embedding_dimension"] != a1_frozen["embedding_dimension"]
                or metadata["cardinalities"] != preprocessing.cardinalities):
            raise ValueError("Checkpoint dimensions differ from the frozen A1 architecture")
        expected_architecture = "four_embedding_mlp" if name == "baseline" else "four_embedding_wide_and_deep"
        if metadata["architecture"] != expected_architecture:
            raise ValueError("A4 requires the A1 baseline and selected A3 wide-and-deep model")
    if (wide["model_config"]["lambda"] != a3_frozen["lambda"]
            or a3_frozen["E_star"] != a1_frozen["E_star"]
            or baseline["training_config"]["mode"] != "a1_refit"
            or wide["training_config"]["mode"] != "fixed_refit"):
        raise ValueError("Final checkpoint protocol or selected lambda differs")


def run_a4(output_directory, prepared_directory, a1_directory, a3_directory):
    output, prepared, a1, a3 = [Path(path).resolve() for path in
                              (output_directory, prepared_directory, a1_directory, a3_directory)]
    if output.exists():
        raise FileExistsError(f"Choose a new A4 output directory: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full A4 attribution run")
    seed_everything(42)
    torch.set_num_threads(4)
    bundle = load_data_bundle(prepared / "refit", expected_protocol="refit")
    a1_frozen = json.loads((a1 / "frozen_a1_settings.json").read_text(encoding="utf-8"))
    a3_frozen = json.loads((a3 / "frozen_a3_settings.json").read_text(encoding="utf-8"))
    if (json.loads((a1 / "summary.json").read_text())["status"] != "completed"
            or json.loads((a3 / "summary.json").read_text())["status"] != "completed"
            or bundle.source_sha256 != a1_frozen["source_sha256"] or bundle.source_sha256 != a3_frozen["source_sha256"]):
        raise ValueError("A4 requires completed A1/A3 with the same prepared source")
    paths = {"baseline": a1 / a1_frozen["refit_checkpoint"], "wide_deep": a3 / a3_frozen["seed42_checkpoint"]}
    payloads = {name: torch.load(path, map_location="cpu", weights_only=True) for name, path in paths.items()}
    validate_checkpoint_pair(payloads["baseline"], payloads["wide_deep"], bundle.preprocessing, a1_frozen, a3_frozen)
    references = make_references(bundle.preprocessing, bundle.datasets["train"].tensors[0])
    test_cat, test_cont, _ = bundle.datasets["test"].tensors
    if len(test_cat) != 26702:
        raise ValueError("A4 must explain every one of the 26,702 saved test rows")
    output.mkdir(parents=True, exist_ok=False)
    (output / "attributions").mkdir()
    write_checkpoint(output / "preprocessing.pt", bundle.preprocessing.metadata())
    write_checkpoint(output / "references.pt", references)
    write_json(output / "references.json", references)
    configuration = {
        "created_at_singapore": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "prepared_directory": str(prepared), "a1_directory": str(a1), "a3_directory": str(a3),
        "source_sha256": bundle.source_sha256, "seed": 42, "checkpoint_epoch": a1_frozen["E_star"],
        "a1_frozen_settings": a1_frozen, "a3_frozen_settings": a3_frozen,
        "checkpoints": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()},
        "prepared_refit_tensors_sha256": sha256_file(prepared / "refit/datasets.pt"),
        "method": "Captum FeatureAblation", "feature_names": list(FEATURE_NAMES), "test_rows": len(test_cat),
        "reference_fit_rows": len(bundle.datasets["train"]), "batch_size": 512, "perturbations_per_eval": 1,
        "feature_masks": [list(range(4)), list(range(4, 10))], "output_unit": "SGD",
        "inverse_transform_precision": "float64 after the saved float32 network output",
        "global_score": "mean(abs(per-sample signed attribution)) over all test rows",
        "device": "cuda", "torch_cpu_threads": 4,
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "captum": captum.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)},
    }
    write_json(output / "configuration.json", configuration)
    write_json(output / "status.json", {"status": "running", "models_completed": 0})
    summaries, checks = {}, {}
    started = time.perf_counter()
    try:
        for name in MODEL_NAMES:
            metadata = payloads[name]["model_config"]
            model = (HousingMLP.from_metadata(metadata) if name == "baseline" else WideDeepHousingModel.from_metadata(metadata))
            checkpoint = restore_checkpoint(paths[name], model, device="cuda")
            state_before = model_state_fingerprint(model)

            def report_batch(completed, total):
                write_json(output / "progress.json", {"stage": "attributing", "model": name,
                           "completed_rows": completed, "total_rows": total, "models_completed": len(summaries)})
                if completed == total or completed % (512 * 10) == 0:
                    print(f"{name}: explained {completed:,}/{total:,} rows", flush=True)

            attrs, predictions = compute_attributions(model, test_cat, test_cont, references,
                batch_size=512, perturbations_per_eval=1, device="cuda", on_batch=report_batch)
            first_count = min(512, len(test_cat))
            manual = manual_ablation_differences(model, test_cat[:first_count].cuda(), test_cont[:first_count].cuda(), references).cpu()
            torch.testing.assert_close(attrs[:first_count], manual, rtol=0, atol=1e-6)
            state_after = model_state_fingerprint(model)
            if state_before != state_after:
                raise AssertionError("Explanation changed the restored model state")
            checks[name] = {"manual_rows_checked": first_count, "features_checked": 10,
                           "max_absolute_error_sgd": float((attrs[:first_count] - manual).abs().max()),
                           "first_sample_source_row_id": int(bundle.row_ids["test"][0]),
                           "first_sample_signed_captum_sgd": attrs[0].tolist(),
                           "first_sample_manual_difference_sgd": manual[0].tolist(),
                           "model_state_unchanged": True}
            attribution_path = output / "attributions" / f"{name}.pt"
            write_checkpoint(attribution_path, {
                "format_version": 1, "model": name, "seed": 42, "checkpoint_epoch": checkpoint["epoch"],
                "checkpoint_sha256": configuration["checkpoints"][name]["sha256"],
                "feature_names": list(FEATURE_NAMES), "row_ids": bundle.row_ids["test"],
                "signed_attributions_sgd": attrs, "predictions_sgd": predictions,
                "output_unit": "SGD", "references_file": "references.pt",
            })
            ranked = rank_importance(global_importance(attrs))
            summaries[name] = {"test_rows": len(attrs), "features": 10, "seed": 42,
                               "checkpoint_epoch": checkpoint["epoch"], "ranked_importance": ranked,
                               "top_three": ranked[:3], "model_state_sha256": state_before,
                               "attributions": attribution_path.relative_to(output).as_posix(),
                               "attribution_sha256": sha256_file(attribution_path)}
            write_json(output / "manual_checks.json", checks)
            write_json(output / "status.json", {"status": "running", "models_completed": len(summaries)})
            print(f"{name} top three: " + "; ".join(f"{row['feature']} SGD {row['mean_absolute_attribution_sgd']:,.2f}" for row in ranked[:3]), flush=True)
            del model
        summary = {"status": "completed", "test_rows": len(test_cat), "features": list(FEATURE_NAMES),
                   "models": summaries, "reference_rule": "shared refit training categorical modes and continuous means",
                   "output_unit": "SGD", "test_assisted_models_or_budget": True,
                   "elapsed_seconds": time.perf_counter() - started}
        write_json(output / "summary.json", summary)
        from part_a.src.a4_exports import export_a4
        export_a4(output)
        write_json(output / "status.json", {"status": "completed", "models_completed": 2})
        write_json(output / "progress.json", {"stage": "complete", "models_completed": 2, "rows_per_model": len(test_cat)})
        return summary
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "models_completed": len(summaries), "error": str(error)})
        raise
