r"""Verify step-2 dependencies, supplied data, and small CUDA training probes.

This is an environment smoke check, not an assignment experiment or final model.
Run with: .\.venv\Scripts\python.exe scripts\verify_setup.py
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "setup"
OUTPUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT / "matplotlib_cache"))
sys.dont_write_bytecode = True

import captum
from captum.attr import FeatureAblation, LayerGradCam
import ipykernel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
from ray import tune
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_finite(tensor: torch.Tensor, description: str) -> None:
    if not torch.isfinite(tensor).all().item():
        raise RuntimeError(f"Nonfinite values in {description}")


def housing_smoke(settings: dict, device: torch.device) -> dict:
    project = ROOT / settings["housing_project"]
    # Import the supplied reference without editing it or copying its source.
    sys.path.insert(0, str(project / "src"))
    from config import Config
    from data import prepare_data
    from model import PriceModel

    config = Config(seed=settings["selection_seed"])
    data = prepare_data(ROOT / settings["housing_csv"], config)
    model = PriceModel(data["cardinalities"], data["target_mean"], data["target_std"], config).to(device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameter_count != 3505:
        raise RuntimeError(f"Baseline parameter count changed: {parameter_count}")
    for name in ("train", "validation", "test"):
        for index, tensor in enumerate(data[name].tensors):
            require_finite(tensor, f"housing {name} tensor {index}")

    generator = torch.Generator().manual_seed(config.seed)
    indices = torch.randperm(len(data["train"]), generator=generator)[:4096].tolist()
    loader = DataLoader(Subset(data["train"], indices), batch_size=config.batch_size,
                        shuffle=True, generator=generator, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    initial_weight = model.mlp[0].weight.detach().clone()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    losses = []
    model.train()
    for categorical, continuous, target in loader:
        categorical, continuous, target = (value.to(device) for value in (categorical, continuous, target))
        optimizer.zero_grad(set_to_none=True)
        prediction = model(categorical, continuous)
        standardized_target = (target - model.target_mean) / model.target_std
        loss = nn.functional.mse_loss(prediction, standardized_target)
        require_finite(loss, "housing smoke loss")
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if torch.equal(initial_weight, model.mlp[0].weight):
        raise RuntimeError("Housing optimizer did not update weights")

    model.eval()
    categorical = data["train"].tensors[0][indices[:2]].to(device)
    continuous = data["train"].tensors[1][indices[:2]].to(device)
    with torch.no_grad():
        attributions = FeatureAblation(model.predict_price).attribute(
            (categorical, continuous),
            baselines=(torch.zeros_like(categorical), torch.zeros_like(continuous)),
            feature_mask=(torch.arange(4, device=device).view(1, 4),
                          torch.arange(4, 10, device=device).view(1, 6)),
        )
        require_finite(attributions[0], "categorical FeatureAblation")
        require_finite(attributions[1], "continuous FeatureAblation")
        ablated = categorical.clone()
        ablated[:, 0] = 0
        manual = model.predict_price(categorical, continuous) - model.predict_price(ablated, continuous)
        torch.testing.assert_close(attributions[0][:, 0], manual)

    return {
        "kind": "setup_only_supplied_baseline_probe",
        "split_sizes": {name: len(data[name]) for name in ("train", "validation", "test")},
        "cardinalities": data["cardinalities"],
        "trainable_parameters": parameter_count,
        "training_samples": len(indices),
        "training_batches": len(losses),
        "epochs": 1,
        "epoch_seconds_including_first_cuda_use": elapsed,
        "standardized_mse_first_batch": losses[0],
        "standardized_mse_last_batch": losses[-1],
        "peak_allocated_gpu_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "optimizer_updated_weights": True,
        "feature_ablation_shapes": [list(value.shape) for value in attributions],
        "feature_ablation_manual_difference_check": "passed",
        "test_performance_evaluated": False,
    }


def xray_smoke(settings: dict, device: torch.device) -> dict:
    with np.load(ROOT / settings["chest_xray_npz"], allow_pickle=False) as archive:
        train_images = archive["train_images"]
        train_labels = archive["train_labels"]
        metadata = {}
        for split, prefix in (("train", "train"), ("validation", "val"), ("test", "test")):
            images, labels = archive[f"{prefix}_images"], archive[f"{prefix}_labels"]
            if images.shape[1:] != (64, 64) or len(images) != len(labels):
                raise RuntimeError(f"Unexpected {split} X-ray shapes")
            values, counts = np.unique(labels, return_counts=True)
            if set(values.tolist()) != {0, 1}:
                raise RuntimeError(f"Unexpected labels: {values}")
            metadata[split] = {"images": list(images.shape), "dtype": str(images.dtype),
                               "label_counts": {str(value): int(count) for value, count in zip(values, counts)}}

    scaled = train_images.astype(np.float32) / 255.0
    mean = float(scaled.mean(dtype=np.float64))
    std = float(scaled.std(dtype=np.float64))
    if not np.isfinite([mean, std]).all() or std <= 0:
        raise RuntimeError("Invalid X-ray training normalization")
    indices = np.random.default_rng(settings["selection_seed"]).permutation(len(scaled))[:512]
    images = torch.from_numpy(((scaled[indices] - mean) / std)[:, None, :, :].copy())
    labels = torch.from_numpy(train_labels[indices].astype(np.float32))
    require_finite(images, "normalized training X-rays")
    loader = DataLoader(TensorDataset(images, labels), batch_size=128, shuffle=True,
                        generator=torch.Generator().manual_seed(settings["selection_seed"]), num_workers=0)
    # This deliberately small CNN checks CUDA convolution/BatchNorm/BCE only;
    # it is not one of the assignment's B1/B2/B3 architectures.
    model = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8),
                          nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    initial_weight = model[0].weight.detach().clone()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    losses = []
    model.train()
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(inputs).squeeze(1), targets)
        require_finite(loss, "X-ray smoke loss")
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if torch.equal(initial_weight, model[0].weight):
        raise RuntimeError("X-ray optimizer did not update weights")
    model.eval()
    gradcam = LayerGradCam(model, model[0]).attribute(images[:1].to(device), target=0,
                                                    relu_attributions=True)
    require_finite(gradcam, "Grad-CAM smoke output")
    return {
        "kind": "setup_only_one_convolution_probe_not_B1_B2_B3",
        "partitions": metadata,
        "class_mapping": settings["part_b"]["class_mapping_assumption"],
        "class_mapping_status": "assumed_not_archive_metadata",
        "training_pixel_mean_after_divide_255": mean,
        "training_pixel_population_std_after_divide_255": std,
        "training_samples": len(indices),
        "training_batches": len(losses),
        "epochs": 1,
        "epoch_seconds_including_initial_convolution_setup": elapsed,
        "bce_first_batch": losses[0],
        "bce_last_batch": losses[-1],
        "peak_allocated_gpu_mib": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "optimizer_updated_weights": True,
        "gradcam_shape": list(gradcam.shape),
        "test_performance_evaluated": False,
    }


def main() -> None:
    settings = json.loads((ROOT / "configs" / "project.json").read_text(encoding="utf-8"))
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve() or sys.prefix == sys.base_prefix:
        raise RuntimeError("Run this script using the project's .venv Python")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this environment")
    seed_everything(settings["selection_seed"])
    device = torch.device("cuda:0")
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "numpy", "pandas", "matplotlib", "ray", "captum", "ipykernel", "pip")}
    pip_check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, check=True)
    report = {
        "checked_at_singapore": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "status": "passed",
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "venv_prefix": sys.prefix,
        "system_site_packages": False,
        "package_versions": versions,
        "pip_check": pip_check.stdout.strip(),
        "cuda": {"available": True, "runtime": torch.version.cuda,
                 "device": torch.cuda.get_device_name(device),
                 "compute_capability": list(torch.cuda.get_device_capability(device)),
                 "total_memory_mib": torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)},
        "seed": settings["selection_seed"],
        "housing": housing_smoke(settings, device),
        "xray": xray_smoke(settings, device),
        "ray_tune_grid_api": tune.grid_search([16, 32]),
        "files": {},
    }
    project = ROOT / settings["housing_project"]
    paths = [ROOT / "assignment.pdf", ROOT / "assignment_slides.pdf", ROOT / "clarifications.md",
             ROOT / settings["housing_csv"], ROOT / settings["chest_xray_npz"],
             project / "README.md", project / "pyproject.toml", project / "uv.lock",
             *(project / "src" / name for name in ("baseline.py", "config.py", "data.py", "model.py"))]
    report["files"] = {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in paths}
    figure, axes = plt.subplots(figsize=(4, 2))
    axes.plot([0, 1], [0, 1])
    axes.set(title="Matplotlib environment check", xlabel="Probe x", ylabel="Probe y")
    figure.tight_layout()
    figure.savefig(OUTPUT / "plot_smoke.png", dpi=100)
    plt.close(figure)
    destination = OUTPUT / "environment_check.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "versions": versions, "cuda": report["cuda"],
                      "housing_epoch_seconds": report["housing"]["epoch_seconds_including_first_cuda_use"],
                      "xray_epoch_seconds": report["xray"]["epoch_seconds_including_initial_convolution_setup"],
                      "report": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
