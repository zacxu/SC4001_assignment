import copy
import math
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import TensorDataset

from assignment_common import TrainingConfig, evaluate, make_loader, restore_checkpoint, seed_everything
from part_a.src.a2 import model_state_fingerprint
from part_a.src.a3 import LAMBDAS, baseline_state_fingerprint, choose_lambda, summarize_comparison
from part_a.src.model import HousingMLP
from part_a.src.training import train_regressor
from part_a.src.wide_deep import WideDeepHousingModel


class A3Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        self.cards = [13, 27, 44, 18]
        self.categories = torch.tensor([[0, 1, 2, 3], [12, 26, 43, 17]], dtype=torch.long)
        self.continuous = torch.arange(12, dtype=torch.float32).reshape(2, 6) / 6

    def model(self, value, width=256):
        return WideDeepHousingModel(self.cards, 450_000., 150_000., hidden_width=width,
                                    embedding_dimension=4, lambda_value=value)

    def test_zero_lambda_matches_baseline_initialization_and_predictions_all_seeds(self):
        for seed in range(42, 47):
            seed_everything(seed)
            baseline = HousingMLP(self.cards, 450_000., 150_000., hidden_width=256, embedding_dimension=4)
            seed_everything(seed)
            actual = self.model(0.)
            self.assertEqual(baseline_state_fingerprint(actual), model_state_fingerprint(baseline))
            for key, tensor in baseline.state_dict().items():
                torch.testing.assert_close(tensor, actual.state_dict()[key], rtol=0, atol=0)
            torch.testing.assert_close(actual(self.categories, self.continuous), baseline(self.categories, self.continuous), rtol=0, atol=0)
            torch.testing.assert_close(actual.predict_price(self.categories, self.continuous), baseline.predict_price(self.categories, self.continuous), rtol=0, atol=0)
            self.assertEqual(sum(p.numel() for p in actual.parameters()), 7088)
            self.assertEqual(actual.metadata()['active_parameters'], 7065)
            self.assertEqual(sum(isinstance(m, nn.Embedding) for m in actual.modules()), 4)

    def test_shared_input_manual_sum_single_inverse_transform_and_gradients(self):
        model = self.model(.5)
        seen = []
        handles = [branch.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0]))
                   for branch in (model.mlp, model.wide)]
        output = model(self.categories, self.continuous)
        for handle in handles:
            handle.remove()
        self.assertIs(seen[0], seen[1])
        features = model.encode_features(self.categories, self.continuous)
        manual = model.mlp(features).squeeze(1) + .5 * model.wide(features).squeeze(1)
        torch.testing.assert_close(output, manual, rtol=0, atol=0)
        torch.testing.assert_close(model.predict_price(self.categories, self.continuous), 450_000. + 150_000. * manual, rtol=0, atol=0)
        output.sum().backward()
        self.assertGreater(model.wide.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.mlp[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.embeddings[0].weight.grad.abs().sum().item(), 0)
        self.assertIsInstance(model.mlp[1], nn.LayerNorm)
        self.assertIsInstance(model.mlp[2], nn.ReLU)
        self.assertNotIn('lambda', dict(model.named_parameters()))

    def test_positive_lambdas_have_equivalent_functions_under_wide_rescaling(self):
        first = self.model(.5)
        second = self.model(2.)
        second.load_state_dict(first.state_dict())
        with torch.no_grad():
            second.wide.weight.mul_(.25)
            second.wide.bias.mul_(.25)
        torch.testing.assert_close(first(self.categories, self.continuous), second(self.categories, self.continuous), rtol=0, atol=0)

    def test_zero_lambda_training_parity_and_restored_target_statistics(self):
        cats = self.categories.repeat(4, 1)
        continuous = torch.arange(48, dtype=torch.float32).reshape(8, 6) / 12
        target = 450_000. + continuous[:, 0] * 10_000.
        dataset = TensorDataset(cats, continuous, target)
        with tempfile.TemporaryDirectory() as temporary:
            fitted = []
            for variant in ('baseline', 'wide_deep'):
                seed_everything(42)
                model = (HousingMLP(self.cards, 450_000., 150_000., hidden_width=8, embedding_dimension=4)
                         if variant == 'baseline' else self.model(0., width=8))
                original_wide = copy.deepcopy(model.wide.state_dict()) if variant == 'wide_deep' else None
                result = train_regressor(model, make_loader(dataset, 3, training=True, seed=42),
                    target_mean=450_000., target_std=150_000.,
                    config=TrainingConfig(learning_rate=.01, max_epochs=2, device='cpu', mode='fixed_refit'),
                    run_directory=Path(temporary)/variant, model_config=model.metadata())
                self.assertEqual(result.best_epoch, 2)
                self.assertEqual(result.epochs_completed, 2)
                fitted.append((model, result))
                if variant == 'wide_deep':
                    self.assertIsNone(model.wide.weight.grad)
                    for key, value in original_wide.items():
                        torch.testing.assert_close(value, model.wide.state_dict()[key], rtol=0, atol=0)
                    fresh = WideDeepHousingModel.from_metadata(model.metadata())
                    checkpoint = restore_checkpoint(result.checkpoint_path, fresh)
                    self.assertEqual(fresh.lambda_value, 0.)
                    self.assertEqual(fresh.target_mean, 450_000.)
                    self.assertEqual(fresh.target_std, 150_000.)
                    self.assertEqual(evaluate(fresh, make_loader(dataset, 3), checkpoint['task_object'], 'cpu'), result.selected_metrics['train'])
            self.assertEqual(fitted[0][1].selected_metrics, fitted[1][1].selected_metrics)
            for key, value in fitted[0][0].state_dict().items():
                torch.testing.assert_close(value, fitted[1][0].state_dict()[key], rtol=0, atol=0)

    def test_lambda_grid_selection_actual_minimum_ties_and_invalid_results(self):
        rows = [{'lambda': value, 'validation_rmse': 100.} for value in LAMBDAS]
        self.assertEqual(choose_lambda(list(reversed(rows)))['lambda'], 0.)
        rows[2]['validation_rmse'] = 99.99999
        self.assertIs(choose_lambda(rows), rows[2])
        for invalid in (rows[:-1], rows+rows[:1], rows[:-1]+rows[:1]):
            with self.assertRaises(ValueError):
                choose_lambda(invalid)
        rows[2]['validation_rmse'] = float('nan')
        with self.assertRaises(ValueError):
            choose_lambda(rows)

    def test_five_seed_sample_sd_pairing_and_fixed_budget_guards(self):
        baseline = [{'seed': seed, 'epochs_completed': 4, 'checkpoint_epoch': 4,
                     'rmse': 100. + seed - 42, 'r2': (seed - 42)/10} for seed in range(42,47)]
        wide = [{**row, 'rmse': row['rmse']-10, 'r2': row['r2']+.05} for row in reversed(baseline)]
        groups, paired = summarize_comparison(wide, baseline, epoch_budget=4)
        self.assertEqual(groups['wide_deep']['metrics']['rmse']['mean'], 92.)
        self.assertAlmostEqual(groups['wide_deep']['metrics']['rmse']['sample_std'], math.sqrt(2.5))
        self.assertEqual(paired['metrics']['rmse_delta'], {'mean': -10., 'sample_std': 0.})
        for invalid in (wide[:-1], wide[:-1]+wide[:1]):
            with self.assertRaises(ValueError):
                summarize_comparison(invalid, baseline, epoch_budget=4)
        wide[0]['checkpoint_epoch']=3
        with self.assertRaises(ValueError):
            summarize_comparison(wide, baseline, epoch_budget=4)

    def test_invalid_lambdas_and_wrong_checkpoint_architecture_are_rejected(self):
        for value in (-1., float('inf'), float('nan'), True, '0.5'):
            with self.assertRaises(ValueError):
                self.model(value)
        model=self.model(1.)
        with self.assertRaises(ValueError):
            WideDeepHousingModel.from_metadata({**model.metadata(), 'architecture':'four_embedding_mlp'})


if __name__ == '__main__':
    unittest.main()
