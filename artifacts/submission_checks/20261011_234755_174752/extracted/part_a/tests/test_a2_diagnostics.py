"""Guard held-out diagnostic provenance rather than repeating final-fit tests."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from assignment_common.artifacts import write_json
from part_a.src.a2_diagnostics import diagnostic_histories, load_diagnostic_partitions
from part_a.src.ablations import VARIANTS
from part_a.src.data import CATEGORICAL_FEATURES, HousingPreprocessor, TRAIN_YEARS


class ForbiddenTestPartition:
    def __getitem__(self, key):
        raise AssertionError("Diagnostic loader accessed test data")


class A2DiagnosticTests(unittest.TestCase):
    def payloads(self):
        preprocessing = HousingPreprocessor(
            {name: {1 if name == "month" else "example": 1} for name in CATEGORICAL_FEATURES},
            torch.zeros(6), torch.ones(6), 450_000., 150_000., TRAIN_YEARS, 2,
        )

        def partition(ids):
            return {"categorical": torch.ones((2, 4), dtype=torch.long),
                    "continuous": torch.ones((2, 6), dtype=torch.float32),
                    "target_sgd": torch.tensor([400_000., 500_000.]),
                    "row_ids": torch.tensor(ids, dtype=torch.long)}

        payload = {"format_version": 1, "protocol": "selection", "source_sha256": "example",
                   "partitions": {"train": partition([0, 1]), "validation": partition([2, 3]),
                                  "test": ForbiddenTestPartition()}}
        return preprocessing.metadata(), payload

    def test_loader_has_no_test_partition_access(self):
        metadata, payload = self.payloads()
        with patch("part_a.src.a2_diagnostics.torch.load", side_effect=[metadata, payload]):
            datasets, preprocessing, _, source = load_diagnostic_partitions("unused")
        self.assertEqual(set(datasets), {"train", "validation"})
        self.assertEqual(preprocessing.fit_years, TRAIN_YEARS)
        self.assertEqual(source, "example")

    def test_loader_rejects_validation_rows_inside_training(self):
        metadata, payload = self.payloads()
        payload["partitions"]["validation"]["row_ids"][0] = 1
        with patch("part_a.src.a2_diagnostics.torch.load", side_effect=[metadata, payload]):
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_diagnostic_partitions("unused")

    def test_curve_validator_rejects_missing_heldout_metrics_test_metrics_and_wrong_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_json(output / "summary.json", {"status": "completed", "test_evaluated": False,
                       "epoch_budget": 4, "training_rows": 2, "validation_rows": 2})
            history = [{"epoch": epoch, "train_rmse": 10., "train_r2": .8,
                        "validation_rmse": 12., "validation_r2": .6,
                        "train_count": 2, "validation_count": 2} for epoch in range(1, 5)]

            def save(rows):
                for variant in VARIANTS:
                    folder = output / "runs" / variant / "seed_42"
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / "history.jsonl").write_text(
                        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            save(history)
            _, valid = diagnostic_histories(output)
            self.assertEqual(set(valid), set(VARIANTS))
            invalid = []
            absent = copy.deepcopy(history)
            del absent[0]["validation_r2"]
            invalid.append(absent)
            test = copy.deepcopy(history)
            test[0]["test_rmse"] = 20.
            invalid.append(test)
            wrong_count = copy.deepcopy(history)
            wrong_count[0]["validation_count"] = 4
            invalid.append(wrong_count)
            invalid.append(history[:-1])
            for rows in invalid:
                save(rows)
                with self.assertRaises(ValueError):
                    diagnostic_histories(output)


if __name__ == "__main__":
    unittest.main()
