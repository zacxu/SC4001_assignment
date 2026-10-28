from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from part_a.src.data import (
    CATEGORICAL_FEATURES, CONTINUOUS_FEATURES, REFIT_YEARS, TRAIN_YEARS,
    HousingPreprocessor, load_data_bundle, prepare_housing, save_prepared_housing,
)


def example_rows():
    rows = []
    for year in range(2017, 2024):
        for index in range(2):
            offset = year - 2017 + index * .5
            row = {"year": year, "month": (1, 10)[index],
                   "town": "KNOWN" if year <= 2020 else f"TOWN_{year}",
                   "flat_model_type": ("B", "A")[index], "storey_range": "01 TO 03",
                   "resale_price": 100_000 + offset * 10_000,
                   "full_address": f"unused address {year}-{index}", "nearest_stn": "unused station"}
            row.update({name: offset * (i + 1) + 1 for i, name in enumerate(CONTINUOUS_FEATURES)})
            row["floor_area_sqm"] = 80.  # Constant training feature must be clamped, retained.
            rows.append(row)
    return pd.DataFrame(rows).sample(frac=1, random_state=7).reset_index(drop=True)


class HousingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.frame = example_rows()

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self, frame=None, name="housing.csv"):
        path = self.root / name
        (self.frame if frame is None else frame).to_csv(path, index=False)
        return prepare_housing(path)

    def assert_same_fit(self, first, second):
        self.assertEqual(first.category_maps, second.category_maps)
        self.assertEqual(first.fit_row_count, second.fit_row_count)
        self.assertEqual(first.target_mean, second.target_mean)
        self.assertEqual(first.target_std, second.target_std)
        torch.testing.assert_close(first.continuous_mean, second.continuous_mean, rtol=0, atol=0)
        torch.testing.assert_close(first.continuous_std, second.continuous_std, rtol=0, atol=0)

    def test_chronological_partitions_order_schema_and_raw_targets(self):
        prepared = self.prepare()
        expected = {"train": self.frame.year.isin(TRAIN_YEARS),
                    "validation": self.frame.year == 2021, "test": self.frame.year == 2022}
        self.assertEqual({name: len(ds) for name, ds in prepared.selection.datasets.items()},
                         {"train": 8, "validation": 2, "test": 2})
        self.assertEqual(len(prepared.refit.datasets["train"]), 10)
        for name, mask in expected.items():
            ids = prepared.selection.row_ids[name]
            self.assertEqual(ids.tolist(), self.frame.index[mask].tolist())
            categorical, continuous, targets = prepared.selection.datasets[name].tensors
            self.assertEqual(categorical.shape, (int(mask.sum()), 4))
            self.assertEqual(continuous.shape, (int(mask.sum()), 6))
            torch.testing.assert_close(targets, torch.tensor(self.frame.loc[mask, "resale_price"].tolist()))
            self.assertFalse((self.frame.loc[ids.tolist(), "year"] == 2023).any())
        self.assertEqual(prepared.report["input_features"], [*CATEGORICAL_FEATURES, *CONTINUOUS_FEATURES])
        self.assertEqual(prepared.report["excluded_rows"], 2)
        self.assertFalse(prepared.report["test_fitted_statistics"])

    def test_baseline_std_conventions_and_constant_feature(self):
        preprocessing = HousingPreprocessor.fit(self.frame)
        fitting = self.frame.loc[self.frame.year.isin(TRAIN_YEARS)]
        continuous = torch.tensor(fitting[list(CONTINUOUS_FEATURES)].to_numpy(), dtype=torch.float32)
        target = torch.tensor(fitting.resale_price.to_numpy(), dtype=torch.float32)
        torch.testing.assert_close(preprocessing.continuous_mean, continuous.mean(0), rtol=0, atol=0)
        torch.testing.assert_close(preprocessing.continuous_std, continuous.std(0, correction=1).clamp_min(1e-8),
                                   rtol=0, atol=0)
        self.assertEqual(preprocessing.target_std, target.std(correction=0).item())
        self.assertNotEqual(preprocessing.target_std, target.std(correction=1).item())
        self.assertEqual(preprocessing.continuous_std[-1].item(), torch.tensor(1e-8).item())
        self.assertTrue((preprocessing.transform(fitting).tensors[1][:, -1] == 0).all())

    def test_selection_statistics_ignore_validation_test_and_excluded_rows(self):
        original = HousingPreprocessor.fit(self.frame)
        changed = self.frame.copy()
        outside = ~changed.year.isin(TRAIN_YEARS)
        changed.loc[outside, list(CONTINUOUS_FEATURES)] = 1e6
        changed.loc[outside, "resale_price"] = 1e9
        changed.loc[outside, "town"] = "NEW_OUTSIDE_TRAINING"
        self.assert_same_fit(original, HousingPreprocessor.fit(changed))

    def test_refit_includes_validation_but_ignores_test_and_excluded_rows(self):
        selection = HousingPreprocessor.fit(self.frame)
        refit = HousingPreprocessor.fit(self.frame, fit_years=REFIT_YEARS)
        self.assertNotEqual(refit.target_mean, selection.target_mean)
        self.assertIn("TOWN_2021", refit.category_maps["town"])
        self.assertNotIn("TOWN_2022", refit.category_maps["town"])
        self.assertNotIn("TOWN_2023", refit.category_maps["town"])
        changed = self.frame.copy()
        outside = ~changed.year.isin(REFIT_YEARS)
        changed.loc[outside, list(CONTINUOUS_FEATURES)] = 1e6
        changed.loc[outside, "resale_price"] = 1e9
        changed.loc[outside, "town"] = "NEW_TEST_CATEGORY"
        self.assert_same_fit(refit, HousingPreprocessor.fit(changed, fit_years=REFIT_YEARS))

    def test_integer_category_sorting_unseen_zero_and_inference_inputs(self):
        preprocessing = HousingPreprocessor.fit(self.frame)
        self.assertEqual(preprocessing.category_maps["month"], {1: 1, 10: 2})
        self.assertEqual(preprocessing.category_maps["flat_model_type"], {"A": 1, "B": 2})
        row = self.frame.iloc[:1].copy()
        row["month"] = 12
        for name in CATEGORICAL_FEATURES[1:]:
            row[name] = "UNSEEN"
        inputs = row[[*CATEGORICAL_FEATURES, *CONTINUOUS_FEATURES]]
        categorical, continuous = preprocessing.transform_features(inputs)
        self.assertEqual(categorical.tolist(), [[0, 0, 0, 0]])
        self.assertEqual(continuous.shape, (1, 6))
        with self.assertRaisesRegex(ValueError, "Missing housing columns"):
            preprocessing.transform_features(inputs.drop(columns="floor_area_sqm"))

    def test_target_inverse_and_constant_target_clamp(self):
        preprocessing = HousingPreprocessor.fit(self.frame)
        target = torch.tensor(self.frame.resale_price.to_numpy(), dtype=torch.float32)
        restored = preprocessing.inverse_target(preprocessing.standardize_target(target))
        torch.testing.assert_close(restored, target, rtol=0, atol=.02)
        constant = self.frame.copy()
        constant["resale_price"] = 100_000.
        transform = HousingPreprocessor.fit(constant)
        self.assertEqual(transform.target_std, torch.tensor(1e-8).item())
        self.assertTrue((transform.standardize_target(torch.tensor([100_000.])) == 0).all())

    def test_invalid_required_values_fail_without_silent_cleanup(self):
        cases = [("resale_price", np.nan), ("dist_to_dhoby", np.inf),
                 ("dist_to_nearest_stn", "not a number"), ("town", "  "),
                 ("town", np.nan), ("month", 1.5), ("month", 13),
                 ("year", 2021.5), ("year", 2024), ("floor_area_sqm", 1e40)]
        for index, (name, value) in enumerate(cases):
            with self.subTest(column=name, value=value):
                frame = self.frame.copy()
                frame[name] = frame[name].astype(object)
                frame.loc[0, name] = value
                with self.assertRaises(ValueError):
                    self.prepare(frame, f"bad{index}.csv")
        with self.assertRaisesRegex(ValueError, "Missing housing columns"):
            self.prepare(self.frame.drop(columns="storey_range"), "missing.csv")

    def test_missing_partitions_unexpected_years_and_test_fit_are_rejected(self):
        for years in ((2017, 2018, 2019, 2020, 2022), (2017, 2018, 2019, 2020, 2021, 2022)):
            with self.assertRaisesRegex(ValueError, "never include test data"):
                HousingPreprocessor.fit(self.frame, fit_years=years)
        with self.assertRaisesRegex(ValueError, "Every fitting year"):
            HousingPreprocessor.fit(self.frame[self.frame.year != 2018])
        with self.assertRaisesRegex(ValueError, "validation partition is empty"):
            self.prepare(self.frame[self.frame.year != 2021])
        with self.assertRaisesRegex(ValueError, "test partition is empty"):
            self.prepare(self.frame[self.frame.year != 2022])

    def test_duplicates_are_reported_and_preserved(self):
        train_row = self.frame[self.frame.year == 2017].iloc[:1]
        prepared = self.prepare(pd.concat([self.frame, train_row], ignore_index=True))
        self.assertEqual(prepared.report["data_quality"]["exact_duplicate_rows"], 1)
        self.assertEqual(prepared.report["data_quality"]["rows_dropped_or_imputed"], 0)
        self.assertEqual(len(prepared.selection.datasets["train"]), 9)

    def test_safe_artifact_roundtrip_protocol_guard_and_existing_output(self):
        prepared = self.prepare()
        destination = save_prepared_housing(prepared, self.root / "prepared")
        for original in (prepared.selection, prepared.refit):
            loaded = load_data_bundle(destination / original.protocol, expected_protocol=original.protocol)
            self.assert_same_fit(original.preprocessing, loaded.preprocessing)
            self.assertEqual(loaded.source_sha256, original.source_sha256)
            self.assertTrue(all(type(key) is int for key in loaded.preprocessing.category_maps["month"]))
            for name, dataset in original.datasets.items():
                for expected, actual in zip(dataset.tensors, loaded.datasets[name].tensors):
                    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
                torch.testing.assert_close(original.row_ids[name], loaded.row_ids[name], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "protocol does not match"):
            load_data_bundle(destination / "refit", expected_protocol="selection")
        with self.assertRaises(FileExistsError):
            save_prepared_housing(prepared, destination)


if __name__ == "__main__":
    unittest.main()
