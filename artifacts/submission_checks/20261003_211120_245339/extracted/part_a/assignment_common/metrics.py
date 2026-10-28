"""Dataset-level metrics, including explicit undefined-denominator handling."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
import numpy as np


def _vector(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array


class RegressionAccumulator:
    """Accumulate SSE and stable target variance across unequal-size batches."""

    def __init__(self) -> None:
        self.count = 0
        self.sse = 0.0
        self.target_mean = 0.0
        self.target_m2 = 0.0

    def update(self, targets, predictions) -> None:
        target = _vector(targets, "targets")
        prediction = _vector(predictions, "predictions")
        if target.shape != prediction.shape:
            raise ValueError("Targets and predictions must have matching shapes")
        batch_count = len(target)
        batch_mean = float(target.mean())
        batch_m2 = float(np.square(target - batch_mean).sum())
        count = self.count + batch_count
        delta = batch_mean - self.target_mean
        self.target_m2 += batch_m2 + delta * delta * self.count * batch_count / count
        self.target_mean += delta * batch_count / count
        self.sse += float(np.square(prediction - target).sum())
        self.count = count

    def compute(self) -> dict:
        if not self.count:
            raise ValueError("Cannot evaluate an empty dataset")
        return {
            "count": self.count,
            "rmse": math.sqrt(self.sse / self.count),
            "r2": 1.0 - self.sse / self.target_m2 if self.count >= 2 and self.target_m2 > 0 else None,
        }


def regression_metrics(targets, predictions) -> dict:
    accumulator = RegressionAccumulator()
    accumulator.update(targets, predictions)
    return accumulator.compute()


class BinaryAccumulator:
    def __init__(self, threshold: float = 0.5) -> None:
        if not 0 < threshold < 1:
            raise ValueError("Probability threshold must be between 0 and 1")
        self.threshold = threshold
        self.tp = self.tn = self.fp = self.fn = 0

    def update(self, targets, probabilities) -> None:
        target = _vector(targets, "targets")
        probability = _vector(probabilities, "probabilities")
        if target.shape != probability.shape:
            raise ValueError("Targets and probabilities must have matching shapes")
        if not np.isin(target, [0, 1]).all():
            raise ValueError("Binary targets must be 0 or 1")
        if ((probability < 0) | (probability > 1)).any():
            raise ValueError("Probabilities must be in [0, 1]")
        predicted = probability >= self.threshold
        positive = target == 1
        self.tp += int(np.count_nonzero(predicted & positive))
        self.tn += int(np.count_nonzero(~predicted & ~positive))
        self.fp += int(np.count_nonzero(predicted & ~positive))
        self.fn += int(np.count_nonzero(~predicted & positive))

    def compute(self) -> dict:
        count = self.tp + self.tn + self.fp + self.fn
        if not count:
            raise ValueError("Cannot evaluate an empty dataset")
        accuracy = (self.tp + self.tn) / count
        return {
            "count": count, "accuracy": accuracy, "classification_error": 1.0 - accuracy,
            "sensitivity": self.tp / (self.tp + self.fn) if self.tp + self.fn else None,
            "specificity": self.tn / (self.tn + self.fp) if self.tn + self.fp else None,
            "tp": self.tp, "tn": self.tn, "fp": self.fp, "fn": self.fn,
        }


def binary_metrics(targets, probabilities, threshold: float = 0.5) -> dict:
    accumulator = BinaryAccumulator(threshold)
    accumulator.update(targets, probabilities)
    return accumulator.compute()


def summarize_seed_results(records: Iterable[Mapping], metrics=("rmse", "r2")) -> dict:
    """Require five distinct seeds and calculate sample, not population, SD."""
    rows = [dict(row) for row in records]
    if len(rows) != 5:
        raise ValueError("The assignment comparison requires five seed results")
    seeds = [row["seed"] for row in rows]
    if any(isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) for seed in seeds):
        raise ValueError("Seeds must be integers")
    if len(set(seeds)) != 5:
        raise ValueError("Evaluation seeds must be distinct")
    summary = {}
    for name in metrics:
        if any(row[name] is None for row in rows):
            raise ValueError(f"Cannot summarize undefined {name}")
        values = _vector([row[name] for row in rows], name)
        summary[name] = {"mean": float(values.mean()), "sample_std": float(values.std(ddof=1))}
    return {"n_seeds": 5, "seeds": [int(seed) for seed in seeds], "metrics": summary, "per_seed": rows}
