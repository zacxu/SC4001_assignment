"""Check shared trainers on small supplied-data subsets; no final test scoring."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Subset, TensorDataset

from assignment_common import TrainingConfig, evaluate, make_loader, restore_checkpoint, seed_everything
from assignment_common.artifacts import write_json
from part_a.src.training import train_regressor
from part_b.src.training import train_classifier


def compare_metrics(expected: dict, actual: dict) -> None:
    if expected.keys() != actual.keys():
        raise AssertionError("Checkpoint metric names differ")
    for name, value in expected.items():
        measured = actual[name]
        if value is None:
            if measured is not None:
                raise AssertionError(f"Undefined metric changed: {name}")
        elif not math.isclose(value, measured, rel_tol=1e-7, abs_tol=1e-7):
            raise AssertionError(f"Restored metric changed: {name}: {value} -> {measured}")


def classifier_probe() -> nn.Module:
    return nn.Sequential(nn.Conv2d(1, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8),
                         nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(.5), nn.Linear(8, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    arguments = parser.parse_args()
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        raise RuntimeError("Use the project .venv interpreter")
    settings = json.loads((ROOT / "configs/project.json").read_text(encoding="utf-8"))
    seed = settings["selection_seed"]
    seed_everything(seed)
    torch.set_num_threads(4)
    timestamp = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d_%H%M%S_%f")
    output = ROOT / "artifacts" / "training_checks" / timestamp
    output.mkdir(parents=True, exist_ok=False)
    project = ROOT / settings["housing_project"]
    sys.path.insert(0, str(project / "src"))
    from config import Config
    from data import prepare_data
    from model import PriceModel

    supplied_config = Config(seed=seed)
    data = prepare_data(ROOT / settings["housing_csv"], supplied_config)
    train_rows = pd.read_csv(ROOT / settings["housing_csv"])
    train_rows = train_rows[train_rows["year"].isin(supplied_config.train_years)]
    category_maps = {name: {value: index + 1 for index, value in enumerate(sorted(train_rows[name].unique()))}
                     for name in supplied_config.categorical_features}
    preprocessing = {"category_maps": category_maps, "cardinalities": data["cardinalities"],
                     "continuous_mean": data["continuous_mean"], "continuous_std": data["continuous_std"],
                     "target_mean": data["target_mean"], "target_std": data["target_std"],
                     "categorical_features": list(supplied_config.categorical_features),
                     "continuous_features": list(supplied_config.continuous_features),
                     "fit_years": list(supplied_config.train_years)}
    indices = torch.randperm(len(data["train"]), generator=torch.Generator().manual_seed(seed))[:257].tolist()
    train = make_loader(Subset(data["train"], indices), 64, training=True, seed=seed, device=arguments.device)
    validation = make_loader(Subset(data["validation"], range(65)), 32, seed=seed, device=arguments.device)
    seed_everything(seed)
    model = PriceModel(data["cardinalities"], data["target_mean"], data["target_std"], supplied_config)
    base = TrainingConfig(device=arguments.device, learning_rate=.01, max_epochs=2, seed=seed)
    regression = train_regressor(model, train, target_mean=data["target_mean"], target_std=data["target_std"],
                                 config=base, run_directory=output / "housing", validation_loader=validation,
                                 preprocessing=preprocessing, model_config={"hidden_width": 64, "embedding_dimension": 8,
                                                                          "verification_only": True})
    restored = PriceModel(data["cardinalities"], 0., 1., supplied_config)
    checkpoint = restore_checkpoint(regression.checkpoint_path, restored, device=arguments.device)
    regression_metrics = evaluate(restored, validation, checkpoint["task_object"], arguments.device)
    compare_metrics(regression.selected_metrics["validation"], regression_metrics)
    if not math.isclose(restored.target_mean, data["target_mean"]):
        raise AssertionError("Price-model target mean was not restored")

    with np.load(ROOT / settings["chest_xray_npz"], allow_pickle=False) as archive:
        scaled_training = archive["train_images"].astype(np.float32) / 255.
        mean, std = float(scaled_training.mean(dtype=np.float64)), float(scaled_training.std(dtype=np.float64))
        xray_train = TensorDataset(torch.from_numpy(((scaled_training[:129] - mean) / std)[:, None].copy()),
                                   torch.from_numpy(archive["train_labels"][:129].astype(np.float32)))
        scaled_validation = archive["val_images"][:65].astype(np.float32) / 255.
        xray_validation = TensorDataset(torch.from_numpy(((scaled_validation - mean) / std)[:, None].copy()),
                                        torch.from_numpy(archive["val_labels"][:65].astype(np.float32)))
    seed_everything(seed)
    probe = classifier_probe()
    class_result = train_classifier(probe, make_loader(xray_train, 64, training=True, seed=seed, device=arguments.device),
                                    config=replace(base, learning_rate=1e-3), run_directory=output / "xray",
                                    validation_loader=make_loader(xray_validation, 32, seed=seed, device=arguments.device),
                                    preprocessing={"mean": mean, "std": std, "intensity_divisor": 255.,
                                                   "class_mapping_assumption": {0: "healthy", 1: "pneumonia"}},
                                    model_config={"architecture": "setup_probe_not_B1_B2_B3", "verification_only": True})
    restored_probe = classifier_probe()
    class_checkpoint = restore_checkpoint(class_result.checkpoint_path, restored_probe, device=arguments.device)
    class_metrics = evaluate(restored_probe, make_loader(xray_validation, 32, seed=seed, device=arguments.device),
                             class_checkpoint["task_object"], arguments.device)
    compare_metrics(class_result.selected_metrics["validation"], class_metrics)
    report = {"status": "passed", "device": arguments.device, "seed": seed,
              "kind": "step_3_trainer_verification_not_assignment_results",
              "housing": {"train_samples": 257, "validation_samples": 65, "epochs_completed": regression.epochs_completed,
                           "best_epoch": regression.best_epoch, "restored_validation_metrics_match": True},
              "xray": {"train_samples": 129, "validation_samples": 65, "epochs_completed": class_result.epochs_completed,
                        "best_epoch": class_result.best_epoch, "restored_validation_metrics_match": True},
              "test_prediction_performance_evaluated": False, "output": str(output)}
    write_json(output / "verification.json", report)
    write_json(ROOT / "artifacts/training_checks/latest.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
