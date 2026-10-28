from dataclasses import replace
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

from assignment_common import seed_everything
from part_a.src.a1 import EMBEDDING_DIMENSIONS, HIDDEN_WIDTHS, choose_configuration
from part_a.src.model import HousingMLP, parameter_count


class A1Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        self.cardinalities = [13, 27, 44, 18]

    def grid_rows(self):
        return [{"hidden_width": width, "embedding_dimension": dimension,
                 "validation_rmse": 100_000. + index,
                 "trainable_parameters": parameter_count(self.cardinalities, width, dimension)}
                for index, (width, dimension) in enumerate(
                    (width, dimension) for width in HIDDEN_WIDTHS for dimension in EMBEDDING_DIMENSIONS)]

    def test_parameter_formula_all_25_and_default_architecture(self):
        for width in HIDDEN_WIDTHS:
            for dimension in EMBEDDING_DIMENSIONS:
                model = HousingMLP(self.cardinalities, 450_000., 150_000.,
                                   hidden_width=width, embedding_dimension=dimension)
                self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),
                                 parameter_count(self.cardinalities, width, dimension))
                self.assertTrue(isinstance(model.mlp[1], nn.LayerNorm))
                self.assertTrue(isinstance(model.mlp[2], nn.ReLU))
                self.assertEqual(len({id(embedding.weight) for embedding in model.embeddings}), 4)
                self.assertTrue(all(embedding.padding_idx is None for embedding in model.embeddings))
        self.assertEqual(parameter_count(self.cardinalities, 64, 8), 3505)

    def test_exact_supplied_initialization_forward_and_sgd_predictions(self):
        reference = Path(__file__).resolve().parents[1] / "housing_baseline_data/hdb-price-prediction-assignment-main/src"
        sys.path.insert(0, str(reference))
        try:
            from config import Config
            from model import PriceModel
            for width, dimension in ((64, 8), (128, 12)):
                config = replace(Config(), hidden_width=width, embedding_dims=(dimension,) * 4)
                seed_everything(42)
                supplied = PriceModel(self.cardinalities, 450_000., 150_000., config)
                seed_everything(42)
                actual = HousingMLP(self.cardinalities, 450_000., 150_000.,
                                    hidden_width=width, embedding_dimension=dimension)
                self.assertEqual(actual.state_dict().keys(), supplied.state_dict().keys())
                for name, value in actual.state_dict().items():
                    torch.testing.assert_close(value, supplied.state_dict()[name], rtol=0, atol=0)
                categorical = torch.tensor([[0, 1, 2, 3], [12, 26, 43, 17]], dtype=torch.long)
                continuous = torch.arange(12, dtype=torch.float32).view(2, 6) / 6
                torch.testing.assert_close(actual(categorical, continuous), supplied(categorical, continuous), rtol=0, atol=0)
                torch.testing.assert_close(actual.predict_price(categorical, continuous), supplied.predict_price(categorical, continuous),
                                           rtol=0, atol=0)
        finally:
            sys.path.remove(str(reference))

    def test_metadata_constructs_matching_architecture(self):
        seed_everything(42)
        model = HousingMLP(self.cardinalities, 450_000., 150_000., hidden_width=256, embedding_dimension=4)
        reconstructed = HousingMLP.from_metadata(model.metadata(), target_mean=model.target_mean, target_std=model.target_std)
        reconstructed.load_state_dict(model.state_dict())
        self.assertEqual(reconstructed.metadata(), model.metadata())
        self.assertEqual(reconstructed.target_mean, 450_000.)
        self.assertEqual(reconstructed.target_std, 150_000.)
        for bad in (([13, 27, 44], 64, 8), (self.cardinalities, 0, 8), (self.cardinalities, 64, 1.5)):
            with self.assertRaises(ValueError):
                parameter_count(*bad)

    def test_selection_uses_lowest_actual_validation_score(self):
        rows = self.grid_rows()
        winner = rows[-1]
        winner["validation_rmse"] = 90_000.
        self.assertIs(choose_configuration(rows), winner)
        # The score difference is below the early-stopping min_delta; selection
        # still uses the actual minimum rather than treating it as an exact tie.
        rows[0]["validation_rmse"] = 90_000.00001
        self.assertIs(choose_configuration(rows), winner)

    def test_exact_ties_and_complete_grid_guard(self):
        rows = self.grid_rows()
        rows[0]["validation_rmse"] = rows[-1]["validation_rmse"] = 90_000.
        self.assertIs(choose_configuration(rows), rows[0])
        # Exercise the deterministic final ordering after the parameter tie.
        rows[-1]["trainable_parameters"] = rows[0]["trainable_parameters"]
        self.assertIs(choose_configuration(list(reversed(rows))), rows[0])
        for invalid in (rows[:-1], rows + rows[:1], rows[:-1] + rows[:1]):
            with self.assertRaisesRegex(ValueError, "25 grid combinations"):
                choose_configuration(invalid)
        rows[0]["validation_rmse"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            choose_configuration(rows)


if __name__ == "__main__":
    unittest.main()
