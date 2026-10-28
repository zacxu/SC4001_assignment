"""Shared training utilities; include this package in both final code ZIPs."""

from .metrics import binary_metrics, regression_metrics, summarize_seed_results
from .artifacts import save_seed_results
from .reproducibility import make_loader, seed_everything
from .training import (
    BinaryClassificationTask, EarlyStopper, FitResult, RegressionTask,
    TrainingConfig, evaluate, evaluate_final_checkpoint, fit, restore_checkpoint,
)

__all__ = [
    "BinaryClassificationTask", "EarlyStopper", "FitResult", "RegressionTask",
    "TrainingConfig", "binary_metrics", "evaluate", "evaluate_final_checkpoint",
    "fit", "make_loader", "regression_metrics", "restore_checkpoint",
    "save_seed_results", "seed_everything", "summarize_seed_results",
]
