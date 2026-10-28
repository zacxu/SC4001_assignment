import math
import unittest
import numpy as np

from assignment_common.metrics import (
    RegressionAccumulator, binary_metrics, regression_metrics, summarize_seed_results,
)


class MetricsTests(unittest.TestCase):
    def test_regression_known_result_and_unequal_batches(self):
        targets, predictions = [100, 200, 300], [110, 190, 330]
        whole = regression_metrics(targets, predictions)
        batched = RegressionAccumulator()
        batched.update(targets[:2], predictions[:2])
        batched.update(targets[2:], predictions[2:])
        self.assertAlmostEqual(whole["rmse"], math.sqrt(1100 / 3))
        self.assertAlmostEqual(whole["r2"], 0.945)
        self.assertEqual(whole, batched.compute())

    def test_large_targets_use_stable_variance(self):
        target = np.array([1e12 + 1, 1e12 + 2, 1e12 + 3])
        accumulator = RegressionAccumulator()
        accumulator.update(target[:2], target[:2])
        accumulator.update(target[2:], target[2:])
        self.assertEqual(accumulator.compute()["r2"], 1.0)

    def test_constant_targets_have_undefined_r2(self):
        self.assertIsNone(regression_metrics([5, 5], [5, 5])["r2"])

    def test_known_confusion_matrix(self):
        metrics = binary_metrics([1, 1, 1, 0, 0, 0, 0], [.9, .6, .2, .7, .1, .4, .3])
        self.assertEqual((metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]), (2, 3, 1, 1))
        self.assertAlmostEqual(metrics["accuracy"], 5 / 7)
        self.assertAlmostEqual(metrics["classification_error"], 2 / 7)
        self.assertAlmostEqual(metrics["sensitivity"], 2 / 3)
        self.assertAlmostEqual(metrics["specificity"], 3 / 4)

    def test_threshold_and_missing_class_denominators(self):
        metrics = binary_metrics([0, 0], [.5, .1])
        self.assertEqual(metrics["fp"], 1)
        self.assertIsNone(metrics["sensitivity"])
        self.assertIsNone(binary_metrics([1], [.9])["specificity"])

    def test_five_seed_sample_standard_deviation(self):
        rows = [{"seed": seed, "rmse": value, "r2": value / 10} for seed, value in zip(range(42, 47), range(1, 6))]
        summary = summarize_seed_results(rows)
        self.assertEqual(summary["metrics"]["rmse"]["mean"], 3)
        self.assertAlmostEqual(summary["metrics"]["rmse"]["sample_std"], math.sqrt(2.5))
        self.assertEqual(summary["per_seed"], rows)
        rows[-1]["seed"] = 42
        with self.assertRaises(ValueError):
            summarize_seed_results(rows)
        with self.assertRaises(ValueError):
            summarize_seed_results(rows[:4])

    def test_invalid_metric_inputs_fail(self):
        for targets, predictions in (([], []), ([1, 2], [1]), ([1], [float("nan")])):
            with self.assertRaises(ValueError):
                regression_metrics(targets, predictions)
        with self.assertRaises(ValueError):
            binary_metrics([2], [.5])
        with self.assertRaises(ValueError):
            binary_metrics([1], [2])
