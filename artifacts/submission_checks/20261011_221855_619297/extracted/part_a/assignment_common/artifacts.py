"""Portable run artifacts and checkpoints loadable with weights_only=True."""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import platform
import numpy as np
import torch


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, torch.Tensor):
        return json_value(value.detach().cpu().tolist())
    if isinstance(value, (np.ndarray, np.generic)):
        return json_value(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported artifact value: {type(value).__name__}")


def checkpoint_value(value):
    """Clone tensor state onto CPU and keep category-map key types intact."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return checkpoint_value(torch.from_numpy(value.copy())) if value.dtype.kind in "biufc" else checkpoint_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {checkpoint_value(key): checkpoint_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [checkpoint_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}")


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_checkpoint(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint_value(value), temporary)
    temporary.replace(path)


class RunArtifacts:
    """Each run owns a new directory; existing results are never overwritten."""

    def __init__(self, directory: Path | str, configuration: dict, preprocessing: dict) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.checkpoint_path = self.directory / "best_model.pt"
        self.history = []
        write_json(self.directory / "config.json", {
            **configuration, "environment": {"python": platform.python_version(), "torch": str(torch.__version__)},
        })
        write_checkpoint(self.directory / "preprocessing.pt", preprocessing)

    def append_epoch(self, row: dict, internal_monitor: dict | None = None) -> None:
        if any(key.startswith("test_") for key in row):
            raise ValueError("Test epoch metrics do not belong in public learning curves")
        self.history.append(row)
        with (self.directory / "history.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(json_value(row), allow_nan=False) + "\n")
        with (self.directory / "history.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(self.history[0]))
            writer.writeheader()
            writer.writerows(self.history)
        if internal_monitor is not None:
            with (self.directory / "internal_test_monitor.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps(json_value(internal_monitor), allow_nan=False) + "\n")

    def save_checkpoint(self, payload: dict) -> None:
        write_checkpoint(self.checkpoint_path, payload)

    def finish(self, summary: dict) -> None:
        write_json(self.directory / "summary.json", summary)


def save_seed_results(directory: Path | str, records, metrics=("rmse", "r2")) -> dict:
    from .metrics import summarize_seed_results

    summary = summarize_seed_results(records, metrics)
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "summary.json", summary)
    with (destination / "per_seed.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["seed", *metrics], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["per_seed"])
    return summary
