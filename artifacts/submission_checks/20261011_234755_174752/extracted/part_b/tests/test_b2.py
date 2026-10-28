import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from torch.utils.data import TensorDataset

from assignment_common import FitResult, TrainingConfig, binary_metrics, make_loader, seed_everything
from part_b.src.b1 import shuffle_fingerprint, state_fingerprint
from part_b.src.b2 import BATCH_SIZES, check_settings, choose_batch_size, result_row, validate_b1_frozen
from part_b.src.evaluation import evaluate_selected_classifier_once
from part_b.src.models import build_model
from part_b.src.training import train_classifier


class B2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def rows(self):
        return [{"batch_size": batch, "best_epoch": 3, "epochs_completed": 7,
                 "validation_loss": .4, "validation_accuracy": .9,
                 "validation_classification_error": .1,
                 "validation_sensitivity": .9, "validation_specificity": .9} for batch in BATCH_SIZES]

    def test_error_then_loss_epoch_and_smaller_batch_select(self):
        rows = self.rows()
        rows[2].update(validation_accuracy=.95, validation_classification_error=.05, validation_loss=.8)
        self.assertIs(choose_batch_size(rows), rows[2])
        rows = self.rows()
        rows[3]["validation_loss"] -= 1e-8
        self.assertIs(choose_batch_size(rows), rows[3])
        rows[4]["validation_loss"] = rows[3]["validation_loss"]
        rows[4]["best_epoch"] = 2
        self.assertIs(choose_batch_size(rows), rows[4])
        rows[3]["best_epoch"] = 2
        self.assertIs(choose_batch_size(list(reversed(rows))), rows[3])
        tied = self.rows()
        self.assertIs(choose_batch_size(list(reversed(tied))), tied[0])

    def test_bad_grids_metrics_epochs_and_slide_typo_fail(self):
        rows = self.rows()
        for invalid in (rows[:-1], rows + rows[:1], rows[:-1] + rows[:1]):
            with self.assertRaises(ValueError):
                choose_batch_size(invalid)
        for key, value in (("batch_size", 258), ("validation_loss", float('nan')),
                           ("validation_specificity", 1.1), ("best_epoch", 8),
                           ("epochs_completed", 21), ("validation_classification_error", .2)):
            invalid = self.rows()
            invalid[0][key] = value
            with self.assertRaises(ValueError):
                choose_batch_size(invalid)

    def test_frozen_b1_controls_and_data_lineage_are_enforced(self):
        root = Path(__file__).resolve().parents[1]
        frozen = json.loads((root / 'part_b/outputs/b1/frozen_b1_settings.json').read_text())
        lineage = {"source_sha256": frozen['source_sha256'], "dataset_sha256": frozen['prepared_dataset_sha256'],
                   "preprocessing_sha256": frozen['prepared_preprocessing_sha256']}
        self.assertEqual(validate_b1_frozen(frozen, **lineage), frozen['selected_learning_rate'])
        for key, value in (("architecture", 'b2'), ("selected_learning_rate", .123), ("probability_threshold", .6),
                           ("selection_seed", 43), ("source_sha256", 'changed'), ("prepared_dataset_sha256", 'changed'),
                           ("prepared_preprocessing_sha256", 'changed'), ("refit", False)):
            invalid = copy.deepcopy(frozen)
            invalid[key] = value
            with self.assertRaises(ValueError):
                validate_b1_frozen(invalid, **lineage)
        config = json.loads((root / 'part_b/configs/default.json').read_text())
        check_settings(config)
        config['settings']['batch_size_grid'][-1] = 258
        with self.assertRaises(ValueError):
            check_settings(config)

    def test_batch_grouping_changes_but_initialization_and_row_order_do_not(self):
        states, orders, steps, tails = [], [], [], []
        for batch in BATCH_SIZES:
            seed_everything(42)
            model = build_model('b2')
            states.append(state_fingerprint(model))
            orders.append(shuffle_fingerprint(4708, batch_size=batch))
            loader = make_loader(TensorDataset(torch.arange(4708)), batch, training=True, seed=42)
            lengths = [len(items[0]) for items in loader]
            steps.append(len(lengths)); tails.append(lengths[-1])
        self.assertEqual(len(set(states)), 1)
        self.assertEqual(len(set(orders)), 1)
        self.assertEqual(steps, [295, 148, 74, 37, 19])
        self.assertEqual(tails, [4, 4, 36, 100, 100])

    def test_result_metrics_and_updates_belong_to_selected_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = output / 'model.pt'
            torch.save({}, checkpoint)
            metrics = {"loss": .3, **binary_metrics([0, 1, 1], [.2, .8, .4])}
            result = FitResult(checkpoint, 2, 6, True, {"validation": metrics, "train": metrics}, [])
            row = result_row(32, .001, result, output=output, initial_sha256='initial', shuffle_sha256='shuffle',
                             peak_memory=10., elapsed_seconds=1.)
            self.assertTrue(all(row['validation_' + key] == value for key, value in metrics.items()))
            self.assertEqual(row['optimizer_steps_per_epoch'], 148)
            self.assertEqual(row['optimizer_steps_to_selected_epoch'], 296)
            self.assertEqual(row['optimizer_steps_completed'], 888)
            self.assertEqual(row['last_training_batch_size'], 4)

    def test_actual_vgg_safe_checkpoint_test_capture_and_tail_once(self):
        seed_everything(42)
        training = TensorDataset(torch.randn(5, 1, 64, 64), torch.tensor([0., 1., 0., 1., 1.]))
        testing = TensorDataset(torch.randn(5, 1, 64, 64), torch.tensor([0., 1., 0., 1., 1.]))
        bundle = SimpleNamespace(datasets={'test': testing}, row_ids={'test': torch.arange(5)}, source_sha256='source')
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'run'
            model = build_model('b2')
            train_classifier(model, make_loader(training, 3, training=True), validation_loader=make_loader(training, 3),
                config=TrainingConfig(max_epochs=1, device='cpu'), run_directory=run, model_config=model.metadata())
            calls = []
            hook = model.register_forward_hook(lambda module, inputs, output: calls.append(len(output)))
            destination = Path(directory) / 'predictions.pt'
            metrics = evaluate_selected_classifier_once(run, model, bundle, destination, batch_size=3, device='cpu')
            self.assertEqual(calls, [3, 2])
            payload = torch.load(destination, weights_only=True)
            self.assertEqual(payload['architecture'], 'b2')
            self.assertEqual(payload['evaluation_batch_size'], 3)
            expected = binary_metrics(payload['labels'], payload['probabilities'])
            self.assertEqual({key: metrics[key] for key in expected}, expected)
            torch.testing.assert_close(payload['row_ids'], torch.arange(5))
            with self.assertRaises(FileExistsError):
                evaluate_selected_classifier_once(run, model, bundle, destination, batch_size=3, device='cpu')
            self.assertEqual(calls, [3, 2])
            hook.remove()


if __name__ == '__main__':
    unittest.main()
