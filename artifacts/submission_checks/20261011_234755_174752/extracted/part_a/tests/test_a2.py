import copy
import math
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset

from assignment_common import TrainingConfig, evaluate, evaluate_final_checkpoint, make_loader, restore_checkpoint, seed_everything
from part_a.src.a2 import model_state_fingerprint, summarize_a2_records, validate_frozen_settings
from part_a.src.ablations import VARIANTS, build_a2_model, model_from_a2_metadata
from part_a.src.training import train_regressor


class CountingHousingDataset(Dataset):
    def __init__(self, dataset):
        self.dataset, self.accesses = dataset, 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        self.accesses += 1
        return self.dataset[index]


class A2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cardinalities = [13, 27, 44, 18]

    def model(self, variant, width=256):
        return build_a2_model(variant, self.cardinalities, 450_000., 150_000.,
                              hidden_width=width, embedding_dimension=4)

    def test_sigmoid_changes_only_activation_and_preserves_all_seed_initializations(self):
        for seed in range(42, 47):
            seed_everything(seed)
            baseline = self.model("baseline")
            seed_everything(seed)
            sigmoid = self.model("sigmoid")
            self.assertEqual(model_state_fingerprint(baseline), model_state_fingerprint(sigmoid))
            self.assertEqual(sum(p.numel() for p in baseline.parameters()), 7065)
            self.assertEqual(sum(p.numel() for p in sigmoid.parameters()), 7065)
            self.assertIsInstance(baseline.mlp[2], nn.ReLU)
            self.assertIsInstance(sigmoid.mlp[2], nn.Sigmoid)
            self.assertIsInstance(sigmoid.mlp[-1], nn.Linear)
            self.assertIsInstance(sigmoid.mlp[1], nn.LayerNorm)
            self.assertEqual(baseline.state_dict().keys(), sigmoid.state_dict().keys())
            for name, parameter in baseline.state_dict().items():
                torch.testing.assert_close(parameter, sigmoid.state_dict()[name], rtol=0, atol=0)

    def test_continuous_only_has_no_embeddings_and_ignores_categories(self):
        model = self.model("continuous_only")
        self.assertFalse(any(isinstance(module, nn.Embedding) for module in model.modules()))
        self.assertEqual(model.mlp[0].in_features, 6)
        self.assertEqual(model.mlp[0].out_features, 256)
        self.assertIsInstance(model.mlp[1], nn.LayerNorm)
        self.assertIsInstance(model.mlp[2], nn.ReLU)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2561)
        continuous = torch.arange(18, dtype=torch.float32).reshape(3, 6) / 6
        first = model(torch.zeros((3, 4), dtype=torch.long), continuous)
        second = model(torch.full((3, 4), -999, dtype=torch.long), continuous)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(first.shape, (3,))

    def test_each_variant_fixed_refit_restore_and_single_final_test(self):
        continuous = torch.arange(48, dtype=torch.float32).reshape(8, 6) / 12
        categorical = torch.tensor([[index % 3 + 1] * 4 for index in range(8)], dtype=torch.long)
        target = 450_000 + continuous[:, 0] * 10_000
        dataset = TensorDataset(categorical, continuous, target)
        with tempfile.TemporaryDirectory() as temporary:
            for variant in VARIANTS:
                seed_everything(42)
                model = self.model(variant, width=8)
                test = CountingHousingDataset(TensorDataset(categorical[:3], continuous[:3], target[:3]))
                directory = Path(temporary) / variant
                result = train_regressor(model, make_loader(dataset, 3, training=True, seed=42),
                                         target_mean=450_000., target_std=150_000.,
                                         config=TrainingConfig(learning_rate=.01, max_epochs=4, mode="fixed_refit", device="cpu"),
                                         run_directory=directory, model_config=model.metadata())
                self.assertEqual(test.accesses, 0)
                self.assertEqual(result.epochs_completed, 4)
                self.assertEqual(result.best_epoch, 4)
                self.assertFalse(result.stopped_early)
                self.assertFalse(any(key.startswith(("validation_", "test_")) for row in result.history for key in row))
                fresh = model_from_a2_metadata(model.metadata())
                checkpoint = restore_checkpoint(result.checkpoint_path, fresh)
                self.assertEqual(fresh.target_mean, 450_000.)
                self.assertEqual(fresh.target_std, 150_000.)
                actual = evaluate(fresh, make_loader(dataset, 3), checkpoint["task_object"], "cpu")
                self.assertEqual(actual, result.selected_metrics["train"])
                metrics = evaluate_final_checkpoint(directory, fresh, make_loader(test, 2))
                self.assertEqual(metrics["count"], 3)
                self.assertEqual(test.accesses, 3)
                with self.assertRaises(FileExistsError):
                    evaluate_final_checkpoint(directory, fresh, make_loader(test, 2))
                self.assertEqual(test.accesses, 3)

    def test_frozen_settings_reject_changed_controls_or_a1_budget(self):
        frozen = {"hidden_width": 256, "embedding_dimension": 4, "E_star": 4, "E_val": 6,
                  "source_sha256": "example", "learning_rate": .01, "batch_size": 512,
                  "evaluation_seeds": [42, 43, 44, 45, 46], "test_assisted_epoch_budget": True}
        a1 = {"selected": {"hidden_width": 256, "embedding_dimension": 4},
              "E_star": 4, "E_val": 6, "source_sha256": "example"}
        validate_frozen_settings(frozen, a1)
        for name, value in (("hidden_width", 128), ("E_star", 6), ("E_star", True),
                            ("batch_size", 128), ("learning_rate", .001),
                            ("evaluation_seeds", [42, 43, 44, 45, 45]), ("test_assisted_epoch_budget", False)):
            changed = {**frozen, name: value}
            with self.subTest(field=name), self.assertRaises(ValueError):
                validate_frozen_settings(changed, a1)

    def example_records(self):
        return [{"variant": variant, "seed": seed, "epochs_completed": 4, "checkpoint_epoch": 4,
                 "rmse": 100. + seed - 42 + offset * 10, "r2": (seed - 42) / 10 - offset * .05}
                for offset, variant in enumerate(VARIANTS) for seed in range(42, 47)]

    def test_five_seed_sample_sd_and_paired_differences(self):
        summaries, paired = summarize_a2_records(self.example_records(), epoch_budget=4)
        self.assertEqual(summaries["baseline"]["metrics"]["rmse"]["mean"], 102.)
        self.assertAlmostEqual(summaries["baseline"]["metrics"]["rmse"]["sample_std"], math.sqrt(2.5))
        self.assertEqual(paired["continuous_only"]["metrics"]["rmse_delta"], {"mean": 10., "sample_std": 0.})
        self.assertEqual(paired["sigmoid"]["metrics"]["rmse_delta"], {"mean": 20., "sample_std": 0.})

    def test_partial_duplicate_or_different_epoch_results_are_rejected(self):
        records = self.example_records()
        for invalid in (records[:-1], records + records[:1], records[:-1] + records[:1]):
            with self.assertRaisesRegex(ValueError, "15 unique"):
                summarize_a2_records(invalid, epoch_budget=4)
        changed = copy.deepcopy(records)
        changed[0]["checkpoint_epoch"] = 3
        with self.assertRaisesRegex(ValueError, "fixed epoch budget"):
            summarize_a2_records(changed, epoch_budget=4)


if __name__ == "__main__":
    unittest.main()
