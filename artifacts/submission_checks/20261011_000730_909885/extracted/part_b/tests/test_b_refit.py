"""Behavioral checks for fresh refit separation, row coverage and single testing."""

import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from torch.utils.data import TensorDataset

from part_b.src.data import sha256_file
from part_b.src.refit import attach_final_checkpoint, audit_final_refit, combined_training_data, refit_final_classifier


class FinalRefitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def bundle(self):
        datasets = {key: TensorDataset(torch.full((count, 1, 64, 64), float(value)),
                                      torch.arange(count).remainder(2).float())
                    for key, count, value in (("train", 4, 1), ("validation", 3, 2), ("test", 5, 9))}
        return SimpleNamespace(datasets=datasets, source_sha256="synthetic",
            row_ids={"test": torch.arange(100, 105)},
            preprocessing=SimpleNamespace(metadata=lambda: {"normalization": "synthetic",
                "fit_partition": "train", "class_mapping": {0: "healthy", 1: "pneumonia"}}))

    def test_combined_training_contains_validation_but_excludes_test(self):
        bundle = self.bundle()
        combined = combined_training_data(bundle)
        self.assertEqual(len(combined), 7)
        for actual, index in zip(combined.tensors, (0, 1)):
            torch.testing.assert_close(actual, torch.cat((bundle.datasets['train'].tensors[index],
                                                        bundle.datasets['validation'].tensors[index])))
        self.assertFalse((combined.tensors[0] == 9).any())

    def test_fresh_fixed_fit_has_no_holdout_monitor_and_final_test_is_guarded(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            # An intentionally non-model selection file proves no model/optimizer
            # state can be resumed from it; it is only hashed as provenance.
            selection = output / 'selection.pt'
            torch.save({'historical_selection_only': True}, selection)
            selected = {'checkpoint': 'selection.pt', 'checkpoint_sha256': sha256_file(selection),
                        'best_epoch': 2, 'learning_rate': .001}
            final = refit_final_classifier(output, self.bundle(), selected,
                architecture='b2', batch_size=3, device='cpu')
            check = audit_final_refit(output)
            self.assertEqual(check['combined_training_rows'], 7)
            self.assertEqual(check['test_rows'], 5)
            self.assertEqual(check['epochs'], 2)
            checkpoint = torch.load(output / final['checkpoint'], weights_only=True)
            self.assertTrue(all(item['step'].item() == math.ceil(7 / 3) * 2
                                for item in checkpoint['optimizer_state']['state'].values()))
            self.assertEqual(set(checkpoint['selected_metrics']), {'train'})
            before = sha256_file(output / 'final_test_predictions.pt')
            with self.assertRaises(FileExistsError):
                refit_final_classifier(output, self.bundle(), selected, architecture='b2', batch_size=3, device='cpu')
            self.assertEqual(before, sha256_file(output / 'final_test_predictions.pt'))
            self.assertEqual(sha256_file(selection), selected['checkpoint_sha256'])
            frozen = attach_final_checkpoint({'checkpoint': selected['checkpoint'],
                                              'checkpoint_sha256': selected['checkpoint_sha256']}, final)
            self.assertEqual(frozen['checkpoint'], 'selection.pt')
            self.assertEqual(frozen['final_checkpoint'], 'refit/best_model.pt')
            self.assertNotEqual(frozen['checkpoint_sha256'], frozen['final_checkpoint_sha256'])
            audit_protocol = json.loads((output / 'refit_protocol.json').read_text())
            self.assertFalse(audit_protocol['test_monitor'])
            self.assertFalse(audit_protocol['validation_monitor'])


if __name__ == '__main__':
    unittest.main()
