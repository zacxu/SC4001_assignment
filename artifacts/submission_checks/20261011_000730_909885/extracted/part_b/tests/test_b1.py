import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import TensorDataset

from assignment_common import FitResult, TrainingConfig, binary_metrics, make_loader, seed_everything
from part_b.src.b1 import (
    LEARNING_RATES, check_settings, choose_learning_rate, result_row, shuffle_fingerprint,
    state_fingerprint, test_selected_model_once as capture_final_test,
)
from part_b.src.models import build_model
from part_b.src.training import train_classifier


class B1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def rows(self):
        return [{"learning_rate": rate, "best_epoch": 3, "epochs_completed": 7,
                 "validation_loss": .4, "validation_accuracy": .9,
                 "validation_classification_error": .1,
                 "validation_sensitivity": .9, "validation_specificity": .9} for rate in LEARNING_RATES]

    def test_error_selects_learning_rate_even_when_loss_is_higher(self):
        rows = self.rows()
        rows[0]["validation_loss"] = .1
        rows[2].update(validation_accuracy=.95, validation_classification_error=.05, validation_loss=.5)
        self.assertIs(choose_learning_rate(rows), rows[2])

    def test_loss_epoch_and_learning_rate_break_exact_error_ties(self):
        rows = self.rows()
        rows[3]["validation_loss"] = .4 - 1e-8
        self.assertIs(choose_learning_rate(rows), rows[3])
        rows[4]["validation_loss"] = rows[3]["validation_loss"]
        rows[4]["best_epoch"] = 2
        self.assertIs(choose_learning_rate(rows), rows[4])
        rows[3]["best_epoch"] = 2
        self.assertIs(choose_learning_rate(list(reversed(rows))), rows[3])
        tied = self.rows()
        self.assertIs(choose_learning_rate(list(reversed(tied))), tied[0])

    def test_incomplete_duplicate_nonfinite_or_incoherent_results_fail(self):
        rows = self.rows()
        for invalid in (rows[:-1], rows + rows[:1], rows[:-1] + rows[:1]):
            with self.assertRaises(ValueError):
                choose_learning_rate(invalid)
        for key, value in (("validation_loss", float("nan")), ("validation_accuracy", 1.1),
                           ("best_epoch", 8), ("epochs_completed", 21), ("validation_classification_error", .2)):
            invalid = self.rows()
            invalid[0][key] = value
            with self.assertRaises(ValueError):
                choose_learning_rate(invalid)

    def test_seeded_initialization_and_real_loader_order_are_controlled(self):
        fingerprints = []
        for _ in LEARNING_RATES:
            seed_everything(42)
            fingerprints.append(state_fingerprint(build_model("b1")))
        self.assertEqual(len(set(fingerprints)), 1)
        self.assertEqual(shuffle_fingerprint(129), shuffle_fingerprint(129))
        self.assertNotEqual(shuffle_fingerprint(129), shuffle_fingerprint(129, seed=43))
        seed_everything(43)
        self.assertNotEqual(fingerprints[0], state_fingerprint(build_model("b1")))

    def test_assignment_controls_and_metrics_at_selected_checkpoint(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "part_b/configs/default.json").read_text())
        check_settings(config)
        for key, value in (("b1_batch_size", 64), ("probability_threshold", .6), ("learning_rate_grid", [1e-3])):
            changed = copy.deepcopy(config)
            changed["settings"][key] = value
            with self.assertRaises(ValueError):
                check_settings(changed)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = output / "model.pt"
            torch.save({}, checkpoint)
            metrics = {"loss": .3, **binary_metrics([0, 1, 1], [.2, .8, .4])}
            result = FitResult(checkpoint, 2, 6, True, {"validation": metrics, "train": metrics}, [])
            row = result_row(.001, result, output=output, initial_sha256="initial", shuffle_sha256="shuffle",
                             peak_memory=100., elapsed_seconds=1.)
            for name, value in metrics.items():
                self.assertEqual(row["validation_" + name], value)
            self.assertEqual(row["best_epoch"], 2)
            self.assertEqual(row["epochs_completed"], 6)

    def test_predictions_are_captured_in_single_guarded_test_pass(self):
        from types import SimpleNamespace
        seed_everything(42)
        train = TensorDataset(torch.randn(5, 1, 64, 64), torch.tensor([0., 1., 0., 1., 1.]))
        test = TensorDataset(torch.randn(3, 1, 64, 64), torch.tensor([0., 1., 1.]))
        bundle = SimpleNamespace(datasets={"test": test}, row_ids={"test": torch.arange(3)}, source_sha256="source")
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            model = build_model("b1")
            train_classifier(model, make_loader(train, 3, training=True), validation_loader=make_loader(train, 3),
                config=TrainingConfig(max_epochs=1, device="cpu"), run_directory=run, model_config=model.metadata())
            calls = []
            handle = model.register_forward_hook(lambda module, inputs, output: calls.append(len(output)))
            destination = Path(directory) / "predictions.pt"
            metrics = capture_final_test(run, model, bundle, destination, device="cpu")
            self.assertEqual(calls, [3])
            payload = torch.load(destination, weights_only=True)
            expected = binary_metrics(payload["labels"], payload["probabilities"], threshold=.5)
            self.assertEqual({key: metrics[key] for key in expected}, expected)
            torch.testing.assert_close(payload["row_ids"], torch.arange(3))
            with self.assertRaises(FileExistsError):
                capture_final_test(run, model, bundle, destination, device="cpu")
            self.assertEqual(calls, [3])
            handle.remove()


if __name__ == "__main__":
    unittest.main()
