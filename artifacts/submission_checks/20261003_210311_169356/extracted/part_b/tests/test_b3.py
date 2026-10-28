import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import TensorDataset

from assignment_common import TrainingConfig, make_loader, seed_everything
from part_b.src.b1 import state_fingerprint
from part_b.src.b3 import comparison_rows, validate_b2_frozen
from part_b.src.complexity import FLOPS_CONVENTION, parameter_breakdown, profile_model
from part_b.src.evaluation import evaluate_selected_classifier_once
from part_b.src.models import build_model
from part_b.src.training import train_classifier


class B3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_frozen_b1_b2_controls_and_lineage(self):
        root = Path(__file__).resolve().parents[1]
        b1 = json.loads((root / 'part_b/outputs/b1/frozen_b1_settings.json').read_text())
        b2 = json.loads((root / 'part_b/outputs/b2/frozen_b2_settings.json').read_text())
        lineage = {'source_sha256': b1['source_sha256'], 'dataset_sha256': b1['prepared_dataset_sha256'],
                   'preprocessing_sha256': b1['prepared_preprocessing_sha256']}
        self.assertEqual(validate_b2_frozen(b2, b1, **lineage), (.001, 16))
        for key, value in (('selected_learning_rate', .01), ('selected_batch_size', 258), ('refit', True),
                           ('b1_checkpoint_sha256', 'changed'), ('prepared_dataset_sha256', 'changed'),
                           ('class_mapping_status', 'verified'), ('selected_epoch', 0)):
            invalid = copy.deepcopy(b2); invalid[key] = value
            with self.assertRaises(ValueError):
                validate_b2_frozen(invalid, b1, **lineage)

    def test_grouped_convolution_and_linear_shape_formulas(self):
        model = nn.Sequential(nn.Conv2d(1, 2, 3, padding=1, bias=False),
            nn.Conv2d(2, 2, 3, padding=1, groups=2, bias=False), nn.AvgPool2d(64), nn.Flatten(), nn.Linear(2, 1))
        result = profile_model(model)
        expected_macs = 64 * 64 * 2 * 9 + 64 * 64 * 2 * 9 + 2
        self.assertEqual(result['convolution_and_linear_macs'], expected_macs)
        self.assertEqual(result['inference_flops'], 2 * expected_macs)
        self.assertEqual(result['parameter_breakdown']['total'], 18 + 18 + 3)
        self.assertEqual(result['residual_addition_flops'], 0)
        self.assertIn('AvgPool2d', result['uncounted_executed_leaf_modules'])

    def test_three_architecture_counts_residual_operations_and_state_preservation(self):
        expected = {'b1': (99569, 31632, 224, 67713, 39063616, 0),
                    'b2': (18185, 17928, 224, 33, 19464256, 0),
                    'b3': (77169, 76432, 672, 65, 25325696, 28672)}
        for name, (total, conv, bn, linear, flops, additions) in expected.items():
            model = build_model(name).train()
            fingerprint = state_fingerprint(model)
            result = profile_model(model)
            self.assertEqual(parameter_breakdown(model), {'convolution': conv, 'batchnorm_affine': bn,
                'linear': linear, 'other': 0, 'total': total})
            self.assertEqual(result['inference_flops'], flops)
            self.assertEqual(result['residual_addition_flops'], additions)
            self.assertEqual(result['profiler_flops'], flops)
            self.assertEqual(state_fingerprint(model), fingerprint)
            self.assertTrue(model.training)
            self.assertEqual(result['output_shape'], [1])

    def test_unknown_profiled_leaf_fails_explicitly(self):
        with self.assertRaises(NotImplementedError):
            profile_model(nn.Sequential(nn.Tanh()))

    def test_comparison_combines_saved_metrics_without_model_evaluation(self):
        metrics = {'count': 624, 'tp': 387, 'tn': 147, 'fp': 87, 'fn': 3,
            'accuracy': 534 / 624, 'classification_error': 1 - 534 / 624,
            'sensitivity': 387 / 390, 'specificity': 147 / 234, 'loss': .5}
        records = {name: {'metrics': metrics, 'test_evaluations': 1, 'seed': 42, 'learning_rate': .001,
            'batch_size': 128 if name == 'b1' else 16, 'checkpoint_epoch': 3, 'checkpoint_sha256': name}
            for name in ('b1', 'b2', 'b3')}
        complexities = {name: profile_model(build_model(name)) for name in records}
        rows = comparison_rows(records, complexities)
        self.assertEqual([row['trainable_parameters'] for row in rows], [99569, 18185, 77169])
        self.assertTrue(all(row['accuracy'] == metrics['accuracy'] for row in rows))
        self.assertTrue(all(row['flops_convention'] == FLOPS_CONVENTION for row in rows))
        invalid = copy.deepcopy(records); invalid['b1']['test_evaluations'] = 2
        with self.assertRaises(ValueError):
            comparison_rows(invalid, complexities)
        invalid = copy.deepcopy(records); invalid['b3']['metrics']['accuracy'] = .9
        with self.assertRaises(ValueError):
            comparison_rows(invalid, complexities)

    def test_resnet_training_checkpoint_and_single_test_capture(self):
        seed_everything(42)
        data = TensorDataset(torch.randn(5, 1, 64, 64), torch.tensor([0., 1., 0., 1., 1.]))
        bundle = SimpleNamespace(datasets={'test': data}, row_ids={'test': torch.arange(5)}, source_sha256='source')
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'run'; model = build_model('b3')
            train_classifier(model, make_loader(data, 3, training=True), validation_loader=make_loader(data, 3),
                config=TrainingConfig(max_epochs=1, device='cpu'), run_directory=run, model_config=model.metadata())
            calls = []
            hook = model.register_forward_hook(lambda module, inputs, output: calls.append(len(output)))
            destination = Path(directory) / 'predictions.pt'
            metrics = evaluate_selected_classifier_once(run, model, bundle, destination, batch_size=3, device='cpu')
            self.assertEqual(calls, [3, 2]); self.assertEqual(metrics['count'], 5)
            payload = torch.load(destination, weights_only=True)
            self.assertEqual(payload['architecture'], 'b3')
            with self.assertRaises(FileExistsError):
                evaluate_selected_classifier_once(run, model, bundle, destination, batch_size=3, device='cpu')
            self.assertEqual(calls, [3, 2]); hook.remove()


if __name__ == '__main__':
    unittest.main()
