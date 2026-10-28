import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset

from assignment_common import (
    BinaryClassificationTask, EarlyStopper, RegressionTask, TrainingConfig,
    evaluate, evaluate_final_checkpoint, fit, make_loader, restore_checkpoint, seed_everything,
)
from assignment_common.artifacts import save_seed_results


class CountingDataset(Dataset):
    def __init__(self):
        self.accesses = 0

    def __len__(self):
        return 3

    def __getitem__(self, index):
        self.accesses += 1
        return torch.tensor([float(index)]), torch.tensor(float(index))


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        seed_everything(42)
        inputs = torch.arange(8, dtype=torch.float32).view(-1, 1) / 8
        self.regression = TensorDataset(inputs, inputs[:, 0] * 2 + 1)
        self.classification = TensorDataset(inputs, (inputs[:, 0] > .4).float())

    def tearDown(self):
        self.temporary.cleanup()

    def test_patience_and_significant_improvement(self):
        stopper = EarlyStopper()
        self.assertFalse(stopper.update(1))
        self.assertFalse(stopper.update(.99996))
        self.assertFalse(stopper.update(.99992))
        self.assertFalse(stopper.update(.99988))
        self.assertEqual(stopper.bad_epochs, 0)
        self.assertEqual([stopper.update(.99988) for _ in range(4)], [False, False, False, True])
        with self.assertRaises(FloatingPointError):
            stopper.update(float("nan"))

    def test_epoch_cap(self):
        for epochs in (0, 21, 1.5, True):
            with self.assertRaises(ValueError):
                TrainingConfig(max_epochs=epochs)

    def test_evaluation_weights_unequal_batches_and_unscales_price(self):
        data = TensorDataset(torch.tensor([[0.], [1.], [2.]]), torch.tensor([11., 14., 12.]))
        model = nn.Identity()
        model.train()
        metrics = evaluate(model, make_loader(data, 2), RegressionTask(10, 2), "cpu")
        self.assertAlmostEqual(metrics["loss"], .75)
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(3))
        self.assertAlmostEqual(metrics["r2"], -13 / 14)
        self.assertEqual(metrics["count"], 3)
        self.assertTrue(model.training)

    def test_evaluation_freezes_dropout_and_batchnorm(self):
        model = nn.Sequential(nn.BatchNorm1d(1), nn.Dropout(.9), nn.Linear(1, 1))
        model.train()
        mean = model[0].running_mean.clone()
        batches = model[0].num_batches_tracked.clone()
        loader = make_loader(self.regression, 3)
        first = evaluate(model, loader, RegressionTask(), "cpu")
        second = evaluate(model, loader, RegressionTask(), "cpu")
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(model[0].running_mean, mean))
        self.assertTrue(torch.equal(model[0].num_batches_tracked, batches))
        self.assertTrue(model.training)

    def test_classification_loss_weights_last_batch(self):
        logits = torch.tensor([[0.], [2.], [-1.]])
        targets = torch.tensor([0., 1., 1.])
        metrics = evaluate(nn.Identity(), make_loader(TensorDataset(logits, targets), 2), BinaryClassificationTask(), "cpu")
        expected_loss = float(np.mean(np.logaddexp(0, [0., 2., -1.]) - np.array([0., 1., 1.]) * [0., 2., -1.]))
        self.assertAlmostEqual(metrics["loss"], expected_loss, places=6)
        self.assertEqual((metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]), (1, 0, 1, 1))

    def test_loss_stopping_and_error_checkpoint_are_independent(self):
        training = make_loader(self.classification, 3, training=True)
        validation = make_loader(self.classification, 3)
        sequence = [(0.8, .4), (.7, .2), (.6, .3), (.5, .2), (.5, .2), (.5, .3), (.5, .3), (.5, .3)]
        real_evaluate = evaluate
        states = {}
        def measured(model, loader, task, device):
            if loader is validation:
                loss, error = sequence.pop(0)
                return {"loss": loss, "count": 10, "classification_error": error,
                        "accuracy": 1 - error, "sensitivity": .8, "specificity": .6,
                        "tp": 4, "tn": 3, "fp": 2, "fn": 1}
            return real_evaluate(model, loader, task, device)
        model = nn.Linear(1, 1)
        def capture(row):
            states[row["epoch"]] = model.weight.detach().clone()
        with patch("assignment_common.training.evaluate", side_effect=measured):
            result = fit(model, training, task=BinaryClassificationTask(),
                         config=TrainingConfig(device="cpu"), run_directory=self.root / "classification",
                         validation_loader=validation, on_epoch=capture)
        self.assertEqual(result.best_epoch, 4)
        self.assertEqual(result.epochs_completed, 8)
        self.assertTrue(result.stopped_early)
        self.assertTrue(torch.equal(model.weight, states[4]))
        self.assertFalse(torch.equal(model.weight, states[8]))
        self.assertEqual(result.selected_metrics["validation"]["loss"], .5)

    def test_small_rmse_improvement_saves_checkpoint_without_resetting_patience(self):
        validation = make_loader(self.regression, 4)
        values = iter([1., .99995, .99995, .99995, .99995])
        real_evaluate = evaluate
        def measured(model, loader, task, device):
            if loader is validation:
                return {"loss": 1., "count": 8, "rmse": next(values), "r2": 0.}
            return real_evaluate(model, loader, task, device)
        with patch("assignment_common.training.evaluate", side_effect=measured):
            result = fit(nn.Linear(1, 1), make_loader(self.regression, 4, training=True),
                         task=RegressionTask(), config=TrainingConfig(device="cpu"),
                         validation_loader=validation, run_directory=self.root / "regression")
        self.assertEqual((result.best_epoch, result.epochs_completed), (2, 5))

    def test_checkpoint_restores_predictions_metrics_and_preprocessing(self):
        class ScaledModel(nn.Linear):
            target_mean = 99.
            target_std = 99.
        validation = make_loader(self.regression, 3)
        model = ScaledModel(1, 1)
        preprocessing = {"category_maps": {"month": {1: 1, 12: 2}}, "mean": torch.tensor([2.]),
                         "std": np.array([3.]), "target_mean": 1., "target_std": 2.}
        result = fit(model, make_loader(self.regression, 3, training=True), task=RegressionTask(1, 2),
                     config=TrainingConfig(device="cpu", max_epochs=3), validation_loader=validation,
                     run_directory=self.root / "roundtrip", preprocessing=preprocessing,
                     model_config={"input_width": 1})
        restored = ScaledModel(1, 1)
        checkpoint = restore_checkpoint(result.checkpoint_path, restored)
        self.assertTrue(torch.equal(model(self.regression.tensors[0]), restored(self.regression.tensors[0])))
        self.assertEqual((restored.target_mean, restored.target_std), (1., 2.))
        self.assertEqual(checkpoint["preprocessing"]["category_maps"]["month"][12], 2)
        self.assertTrue(torch.equal(checkpoint["preprocessing"]["std"], torch.tensor([3.], dtype=torch.float64)))
        measured = evaluate(restored, validation, checkpoint["task_object"], "cpu")
        self.assertEqual(measured, result.selected_metrics["validation"])

    def test_fixed_epoch_refit_and_single_final_test_evaluation(self):
        model = nn.Linear(1, 1)
        test_data = CountingDataset()
        result = fit(model, make_loader(self.regression, 3, training=True), task=RegressionTask(),
                     config=TrainingConfig(device="cpu", mode="fixed_refit", max_epochs=3),
                     run_directory=self.root / "fixed")
        self.assertEqual((result.best_epoch, result.epochs_completed), (3, 3))
        self.assertEqual(test_data.accesses, 0)
        self.assertFalse(result.stopped_early)
        evaluate_final_checkpoint(self.root / "fixed", nn.Linear(1, 1), make_loader(test_data, 2))
        self.assertEqual(test_data.accesses, 3)
        with self.assertRaises(FileExistsError):
            evaluate_final_checkpoint(self.root / "fixed", model, make_loader(test_data, 2))
        self.assertEqual(test_data.accesses, 3)

    def test_a1_monitor_is_internal_and_requires_explicit_protocol(self):
        run = self.root / "a1_refit"
        result = fit(nn.Linear(1, 1), make_loader(self.regression, 4, training=True),
                     task=RegressionTask(), config=TrainingConfig(device="cpu", mode="a1_refit", max_epochs=2),
                     run_directory=run, test_monitor_loader=make_loader(self.regression, 3))
        self.assertFalse(any(key.startswith("test_") for row in result.history for key in row))
        self.assertFalse(any(key.startswith("validation_") for row in result.history for key in row))
        self.assertEqual(len((run / "internal_test_monitor.jsonl").read_text().splitlines()), 2)
        self.assertTrue(json.loads((run / "summary.json").read_text())["test_assisted_stopping"])
        for mode, task, validation, test in (
            ("selection", RegressionTask(), None, make_loader(self.regression, 4)),
            ("fixed_refit", RegressionTask(), make_loader(self.regression, 4), None),
            ("a1_refit", BinaryClassificationTask(), None, make_loader(self.classification, 4)),
        ):
            with self.assertRaises(ValueError):
                fit(nn.Linear(1, 1), make_loader(self.regression, 4), task=task,
                    config=TrainingConfig(device="cpu", mode=mode), run_directory=self.root / mode,
                    validation_loader=validation, test_monitor_loader=test)

    def test_training_metrics_use_full_final_epoch_model(self):
        model = nn.Linear(1, 1)
        result = fit(model, make_loader(self.regression, 3, training=True), task=RegressionTask(),
                     config=TrainingConfig(device="cpu", mode="fixed_refit", max_epochs=1, learning_rate=.1),
                     run_directory=self.root / "train_curves")
        measured = evaluate(model, make_loader(self.regression, 3), RegressionTask(), "cpu")
        self.assertEqual(result.history[0]["train_loss"], measured["loss"])
        self.assertNotAlmostEqual(result.history[0]["train_loss_online"], measured["loss"])

    def test_same_seed_reproduces_training_and_different_seed_changes_it(self):
        def run(seed, name):
            seed_everything(seed)
            model = nn.Linear(1, 1)
            fit(model, make_loader(self.regression, 3, training=True, seed=seed), task=RegressionTask(),
                config=TrainingConfig(device="cpu", seed=seed, mode="fixed_refit", max_epochs=2),
                run_directory=self.root / name)
            return model.weight.detach().clone()
        self.assertTrue(torch.equal(run(42, "first"), run(42, "repeat")))
        self.assertFalse(torch.equal(run(42, "third"), run(43, "different")))

    def test_existing_run_results_are_protected_and_seed_outputs_are_saved(self):
        directory = self.root / "existing"
        directory.mkdir()
        (directory / "keep.txt").write_text("existing result")
        with self.assertRaises(FileExistsError):
            fit(nn.Linear(1, 1), make_loader(self.regression, 4), task=RegressionTask(),
                config=TrainingConfig(device="cpu", mode="fixed_refit", max_epochs=1), run_directory=directory)
        self.assertEqual((directory / "keep.txt").read_text(), "existing result")
        rows = [{"seed": seed, "rmse": seed - 41, "r2": .5} for seed in range(42, 47)]
        save_seed_results(self.root / "seeds", rows)
        self.assertEqual(len((self.root / "seeds/per_seed.csv").read_text().splitlines()), 6)
        self.assertAlmostEqual(json.loads((self.root / "seeds/summary.json").read_text())["metrics"]["rmse"]["sample_std"], math.sqrt(2.5))

    def test_nonfinite_failure_is_logged(self):
        data = TensorDataset(torch.tensor([[float("nan")]]), torch.tensor([1.]))
        with self.assertRaises(FloatingPointError):
            fit(nn.Linear(1, 1), make_loader(data, 1), task=RegressionTask(),
                config=TrainingConfig(device="cpu", mode="fixed_refit", max_epochs=1), run_directory=self.root / "failed")
        self.assertEqual(json.loads((self.root / "failed/summary.json").read_text())["status"], "failed")

    def test_failed_run_cannot_be_evaluated_as_final(self):
        directory = self.root / "incomplete"
        fit(nn.Linear(1, 1), make_loader(self.regression, 4), task=RegressionTask(),
            config=TrainingConfig(device="cpu", mode="fixed_refit", max_epochs=1), run_directory=directory)
        (directory / "summary.json").write_text(json.dumps({"status": "failed"}))
        test_data = CountingDataset()
        with self.assertRaises(ValueError):
            evaluate_final_checkpoint(directory, nn.Linear(1, 1), make_loader(test_data, 2))
        self.assertEqual(test_data.accesses, 0)

    def test_min_delta_boundary_matches_supplied_strict_rule(self):
        stopper = EarlyStopper()
        stopper.update(1.0)
        stopper.update(1.0 - 1e-4)
        self.assertEqual(stopper.bad_epochs, 1)
