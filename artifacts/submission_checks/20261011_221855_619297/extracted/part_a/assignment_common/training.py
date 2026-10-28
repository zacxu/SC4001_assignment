"""Adam training, separate stopping/selection rules, and final evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Callable, Literal

import torch
from torch import nn
from torch.utils.data import DataLoader

from .artifacts import RunArtifacts, write_json
from .metrics import BinaryAccumulator, RegressionAccumulator
from .reproducibility import make_loader, seed_everything


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-3
    max_epochs: int = 20
    patience: int = 4
    min_delta: float = 1e-4
    seed: int = 42
    device: str = "cuda"
    mode: Literal["selection", "a1_refit", "fixed_refit"] = "selection"

    def __post_init__(self) -> None:
        if isinstance(self.max_epochs, bool) or not isinstance(self.max_epochs, int) or not 1 <= self.max_epochs <= 20:
            raise ValueError("Epoch budget must be between 1 and 20")
        if self.patience < 1 or not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("Invalid early-stopping configuration")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive")
        if self.mode not in {"selection", "a1_refit", "fixed_refit"}:
            raise ValueError("Unknown training protocol")


class EarlyStopper:
    """Strict significant improvement, matching the supplied baseline rule."""

    def __init__(self, patience: int = 4, min_delta: float = 1e-4) -> None:
        if patience < 1 or not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("Invalid early-stopping configuration")
        self.patience, self.min_delta = patience, min_delta
        self.best = math.inf
        self.bad_epochs = 0

    def update(self, value: float) -> bool:
        if not math.isfinite(value):
            raise FloatingPointError("Stopping metric is not finite")
        if value < self.best - self.min_delta:
            self.best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def _column(output: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if output.ndim == 2 and output.shape[1] == 1:
        output = output[:, 0]
    if target.ndim == 2 and target.shape[1] == 1:
        target = target[:, 0]
    if output.ndim != 1 or output.shape != target.shape:
        raise ValueError("Expected one model output and target per sample")
    if not torch.isfinite(output).all().item() or not torch.isfinite(target).all().item():
        raise FloatingPointError("Outputs or targets are not finite")
    return output, target.to(dtype=output.dtype)


@dataclass(frozen=True)
class RegressionTask:
    target_mean: float = 0.0
    target_std: float = 1.0
    kind: str = field(default="regression", init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.target_mean) or not math.isfinite(self.target_std) or self.target_std <= 0:
            raise ValueError("Invalid target transformation")

    def loss(self, output, target):
        output, target = _column(output, target)
        return nn.functional.mse_loss(output, (target - self.target_mean) / self.target_std)

    def accumulator(self):
        return RegressionAccumulator()

    def accumulate(self, accumulator, output, target):
        output, target = _column(output, target)
        price = output * self.target_std + self.target_mean
        accumulator.update(target.detach().cpu().double().numpy(), price.detach().cpu().double().numpy())

    def monitor_value(self, metrics):
        return metrics["rmse"]

    def selection_key(self, metrics, epoch):
        return (metrics["rmse"], epoch)

    def bind_model(self, model):
        # The supplied model stores these outside state_dict; restore them too.
        for name in ("target_mean", "target_std"):
            if hasattr(model, name):
                attribute = getattr(model, name)
                if isinstance(attribute, torch.Tensor):
                    with torch.no_grad():
                        attribute.fill_(getattr(self, name))
                else:
                    setattr(model, name, getattr(self, name))


@dataclass(frozen=True)
class BinaryClassificationTask:
    threshold: float = 0.5
    kind: str = field(default="binary_classification", init=False)

    def __post_init__(self) -> None:
        if not 0 < self.threshold < 1:
            raise ValueError("Probability threshold must be between 0 and 1")

    def loss(self, output, target):
        output, target = _column(output, target)
        if not ((target == 0) | (target == 1)).all().item():
            raise ValueError("Binary targets must be 0 or 1")
        return nn.functional.binary_cross_entropy_with_logits(output, target)

    def accumulator(self):
        return BinaryAccumulator(self.threshold)

    def accumulate(self, accumulator, output, target):
        output, target = _column(output, target)
        accumulator.update(target.detach().cpu().numpy(), torch.sigmoid(output).detach().cpu().numpy())

    def monitor_value(self, metrics):
        return metrics["loss"]

    def selection_key(self, metrics, epoch):
        return (metrics["classification_error"], metrics["loss"], epoch)

    def bind_model(self, model):
        pass


Task = RegressionTask | BinaryClassificationTask


def _batch(batch, device):
    if not isinstance(batch, (tuple, list)) or len(batch) < 2 or not all(isinstance(value, torch.Tensor) for value in batch):
        raise TypeError("Batches must contain one or more input tensors followed by targets")
    inputs = [value.to(device, non_blocking=True) for value in batch[:-1]]
    target = batch[-1].to(device, non_blocking=True)
    return inputs, target


def evaluate(model: nn.Module, loader: DataLoader, task: Task, device: str | torch.device) -> dict:
    """Evaluate every sample with frozen dropout/BatchNorm; restore prior mode."""
    if loader.drop_last:
        raise ValueError("Evaluation loaders must not drop the last batch")
    accumulator = task.accumulator()
    loss_sum, count = 0.0, 0
    previous_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                inputs, target = _batch(batch, device)
                output = model(*inputs)
                loss = task.loss(output, target)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Evaluation loss is not finite")
                size = target.shape[0]
                loss_sum += loss.item() * size
                count += size
                task.accumulate(accumulator, output, target)
    finally:
        model.train(previous_mode)
    metrics = accumulator.compute()
    return {"loss": loss_sum / count, **metrics}


@dataclass
class FitResult:
    checkpoint_path: Path
    best_epoch: int
    epochs_completed: int
    stopped_early: bool
    selected_metrics: dict
    history: list[dict]


def restore_checkpoint(path: Path | str, model: nn.Module, *, device: str = "cpu",
                       optimizer: torch.optim.Optimizer | None = None) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    model.to(device)
    model.load_state_dict(checkpoint["model_state"])
    state = checkpoint["task"]
    if state["kind"] == "regression":
        task = RegressionTask(target_mean=state["target_mean"], target_std=state["target_std"])
    elif state["kind"] == "binary_classification":
        task = BinaryClassificationTask(threshold=state["threshold"])
    else:
        raise ValueError("Unknown checkpoint task")
    task.bind_model(model)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    model.eval()
    return {**checkpoint, "task_object": task}


def fit(model: nn.Module, train_loader: DataLoader, *, task: Task, config: TrainingConfig,
        run_directory: Path | str, validation_loader: DataLoader | None = None,
        test_monitor_loader: DataLoader | None = None, preprocessing: dict | None = None,
        model_config: dict | None = None, on_epoch: Callable[[dict], None] | None = None) -> FitResult:
    """Train, save the selected checkpoint, restore it, and return its metrics.

    Seed before creating the model. This function also seeds training randomness.
    Loader generators are separate; use make_loader for reproducible shuffling.
    """
    if config.mode == "selection" and (validation_loader is None or test_monitor_loader is not None):
        raise ValueError("Selection requires validation data and forbids test monitoring")
    if config.mode == "a1_refit" and (not isinstance(task, RegressionTask) or validation_loader is not None or test_monitor_loader is None):
        raise ValueError("Only the A1 regression refit may monitor test data")
    if config.mode == "fixed_refit" and (validation_loader is not None or test_monitor_loader is not None):
        raise ValueError("Fixed-epoch refits do not monitor validation or test data")
    if train_loader.drop_last or train_loader.batch_size is None:
        raise ValueError("Use a standard training loader that retains every sample")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed_everything(config.seed)
    model.to(device)
    task.bind_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_evaluation = make_loader(train_loader.dataset, train_loader.batch_size, seed=config.seed,
                                   num_workers=train_loader.num_workers, device=config.device)
    stopping_loader = test_monitor_loader if config.mode == "a1_refit" else validation_loader
    monitor_role = "test_monitor" if config.mode == "a1_refit" else "validation"
    artifacts = RunArtifacts(run_directory, {"training": asdict(config), "task": asdict(task),
                             "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
                             "model_config": model_config or {}, "batch_size": train_loader.batch_size,
                             "stopping_metric": (f"{monitor_role}_rmse_sgd" if isinstance(task, RegressionTask) else "validation_loss") if stopping_loader is not None else None,
                             "checkpoint_selection": "final_epoch" if config.mode == "fixed_refit" else ("lowest_rmse_then_earlier_epoch" if isinstance(task, RegressionTask) else "lowest_error_then_loss_then_earlier_epoch"),
                             "training_curve_metrics": "evaluation_of_completed_epoch_model_on_all_training_samples",
                             "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                             "test_assisted_stopping": config.mode == "a1_refit"}, preprocessing or {})
    stopper = EarlyStopper(config.patience, config.min_delta)
    best_key = None
    started = time.perf_counter()
    stopped_early = False
    try:
        for epoch in range(1, config.max_epochs + 1):
            epoch_started = time.perf_counter()
            model.train()
            online_loss, sample_count = 0.0, 0
            for batch in train_loader:
                inputs, target = _batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss = task.loss(model(*inputs), target)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Training loss is not finite")
                loss.backward()
                optimizer.step()
                online_loss += loss.item() * target.shape[0]
                sample_count += target.shape[0]
            if not sample_count:
                raise ValueError("Cannot train on an empty dataset")
            training_metrics = evaluate(model, train_evaluation, task, device)
            monitor_metrics = evaluate(model, stopping_loader, task, device) if stopping_loader is not None else None
            row = {"epoch": epoch, "train_loss_online": online_loss / sample_count,
                   **{f"train_{key}": value for key, value in training_metrics.items()}}
            if config.mode == "selection":
                row.update({f"validation_{key}": value for key, value in monitor_metrics.items()})
            row["epoch_seconds"] = time.perf_counter() - epoch_started
            should_stop = stopper.update(task.monitor_value(monitor_metrics)) if monitor_metrics is not None else False
            key = task.selection_key(monitor_metrics, epoch) if monitor_metrics is not None else None
            selected = config.mode == "fixed_refit" and epoch == config.max_epochs
            selected = selected or (monitor_metrics is not None and (best_key is None or key < best_key))
            row["checkpoint_selected"] = selected
            if selected:
                best_key = key
                selected_metrics = {"train": training_metrics}
                if monitor_metrics is not None:
                    selected_metrics[monitor_role] = monitor_metrics
                artifacts.save_checkpoint({"format_version": 1, "epoch": epoch, "model_state": model.state_dict(),
                                           "optimizer_state": optimizer.state_dict(), "task": asdict(task),
                                           "training_config": asdict(config), "model_config": model_config or {},
                                           "preprocessing": preprocessing or {}, "selected_metrics": selected_metrics})
            internal = {"epoch": epoch, **monitor_metrics} if config.mode == "a1_refit" else None
            artifacts.append_epoch(row, internal)
            if on_epoch is not None:
                on_epoch(dict(row))
            if should_stop:
                stopped_early = epoch < config.max_epochs
                break
        checkpoint = restore_checkpoint(artifacts.checkpoint_path, model, device=config.device)
        result = FitResult(artifacts.checkpoint_path, checkpoint["epoch"], epoch, stopped_early,
                           checkpoint["selected_metrics"], artifacts.history)
        artifacts.finish({"status": "completed", "best_epoch": result.best_epoch,
                          "epochs_completed": result.epochs_completed, "stopped_early": result.stopped_early,
                          "stopping_reference": stopper.best if monitor_metrics is not None else None,
                          "selected_metrics": result.selected_metrics, "elapsed_seconds": time.perf_counter() - started,
                          "test_assisted_stopping": config.mode == "a1_refit"})
        return result
    except Exception as error:
        artifacts.finish({"status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise


def evaluate_final_checkpoint(run_directory: Path | str, model: nn.Module, test_loader: DataLoader,
                              *, device: str = "cpu") -> dict:
    """Restore the selected model and persist one final test evaluation per run."""
    directory = Path(run_directory)
    destination = directory / "test_metrics.json"
    if destination.exists():
        raise FileExistsError("Final test metrics already exist; reuse them instead of reevaluating")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "completed":
        raise ValueError("Final evaluation requires a successfully completed training run")
    checkpoint = restore_checkpoint(directory / "best_model.pt", model, device=device)
    metrics = evaluate(model, test_loader, checkpoint["task_object"], device)
    write_json(destination, {"checkpoint_epoch": checkpoint["epoch"], "seed": checkpoint["training_config"]["seed"],
                             "test_assisted_stopping": checkpoint["training_config"]["mode"] == "a1_refit",
                             "metrics": metrics})
    return metrics
