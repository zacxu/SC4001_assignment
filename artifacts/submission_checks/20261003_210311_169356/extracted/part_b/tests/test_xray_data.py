import copy
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from part_b.src.data import (
    XRayPreprocessor, load_data_bundle, prepare_xray, save_prepared_xray,
)


class XRayDataTests(unittest.TestCase):
    @staticmethod
    def arrays():
        ramp = (np.arange(4096).reshape(64, 64) % 256).astype(np.uint8)
        return {"train_images": np.stack([ramp, 255 - ramp, ramp]),
                "train_labels": np.array([0, 1, 0]),
                "val_images": np.stack([ramp, np.zeros((64, 64), dtype=np.uint8)]),
                "val_labels": np.array([0, 1]),
                "test_images": np.full((1, 64, 64), 255, dtype=np.uint8),
                "test_labels": np.array([1])}

    def test_training_only_population_statistics_and_no_double_scaling(self):
        arrays = self.arrays()
        expected = arrays["train_images"].astype(np.float32) / np.float32(255)
        preprocessing = XRayPreprocessor.fit(arrays["train_images"])
        self.assertEqual(preprocessing.mean, float(expected.mean(dtype=np.float64)))
        self.assertEqual(preprocessing.std, float(expected.std(dtype=np.float64)))
        self.assertNotEqual(preprocessing.std, float(expected.std(dtype=np.float64, ddof=1)))
        inputs = preprocessing.transform(arrays["test_images"])
        self.assertEqual(inputs.shape, (1, 1, 64, 64))
        self.assertEqual(inputs.dtype, torch.float32)
        torch.testing.assert_close(inputs, torch.full_like(inputs,
            (np.float32(1) - np.float32(preprocessing.mean)) / np.float32(preprocessing.std)), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            preprocessing.transform(expected)
        with self.assertRaises(ValueError):
            XRayPreprocessor.fit(np.zeros((2, 64, 64), dtype=np.uint8))

    def test_validation_and_test_changes_do_not_affect_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = self.arrays(), self.arrays()
            second["val_images"] = 255 - second["val_images"]
            second["test_images"][:] = 0
            paths = [Path(directory) / name for name in ("first.npz", "second.npz")]
            np.savez(paths[0], **first)
            np.savez(paths[1], **second)
            bundles = [prepare_xray(path, expected_counts=None) for path in paths]
            self.assertEqual(bundles[0].preprocessing, bundles[1].preprocessing)
            torch.testing.assert_close(bundles[0].datasets["train"].tensors[0],
                                       bundles[1].datasets["train"].tensors[0], rtol=0, atol=0)
            self.assertFalse(torch.equal(bundles[0].datasets["validation"].tensors[0],
                                        bundles[1].datasets["validation"].tensors[0]))

    def test_order_raw_pixels_labels_and_duplicates_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.npz"
            arrays = self.arrays()
            np.savez(path, **arrays)
            bundle = prepare_xray(path, expected_counts=None)
            for split, prefix in (("train", "train"), ("validation", "val"), ("test", "test")):
                np.testing.assert_array_equal(bundle.raw_images[split].numpy(), arrays[prefix + "_images"])
                torch.testing.assert_close(bundle.row_ids[split], torch.arange(len(arrays[prefix + "_images"])))
                np.testing.assert_array_equal(bundle.datasets[split].tensors[1].numpy(), arrays[prefix + "_labels"])
                self.assertEqual(bundle.datasets[split].tensors[1].dtype, torch.float32)
            duplicates = bundle.report["duplicates"]
            self.assertEqual(duplicates["within_split_extra_copies"]["train"], 1)
            self.assertEqual(duplicates["cross_split"][0]["shared_unique_images"], 1)
            self.assertEqual(duplicates["conflicting_label_groups"], 0)
            self.assertEqual(len(bundle.datasets["train"]), 3)
            self.assertFalse(bundle.report["test_performance_evaluated"])

    def test_bad_keys_sizes_labels_and_image_layout_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.npz"
            cases = []
            missing = self.arrays()
            del missing["val_labels"]
            cases.append(missing)
            invalid_label = self.arrays()
            invalid_label["test_labels"] = np.array([2])
            cases.append(invalid_label)
            float_label = self.arrays()
            float_label["train_labels"] = np.array([0., 1., 0.])
            cases.append(float_label)
            nchw = self.arrays()
            nchw["train_images"] = nchw["train_images"][:, None]
            cases.append(nchw)
            for arrays in cases:
                with self.subTest(keys=list(arrays)):
                    np.savez(path, **arrays)
                    with self.assertRaises(ValueError):
                        prepare_xray(path, expected_counts=None)
            np.savez(path, **self.arrays())
            with self.assertRaises(ValueError):
                prepare_xray(path)  # Production defaults require the supplied sizes.

    def test_safe_artifact_roundtrip_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.npz"
            np.savez(path, **self.arrays())
            original = prepare_xray(path, expected_counts=None)
            output = Path(directory) / "prepared"
            save_prepared_xray(original, output)
            restored = load_data_bundle(output, expected_counts=None)
            self.assertEqual(restored.preprocessing, original.preprocessing)
            self.assertEqual(restored.source_sha256, original.source_sha256)
            for split in original.datasets:
                for first, second in zip(original.datasets[split].tensors, restored.datasets[split].tensors):
                    torch.testing.assert_close(first, second, rtol=0, atol=0)
                torch.testing.assert_close(original.raw_images[split], restored.raw_images[split], rtol=0, atol=0)
            metadata = torch.load(output / "preprocessing.pt", weights_only=True)
            self.assertEqual(metadata["class_mapping"], {0: "healthy", 1: "pneumonia"})
            self.assertEqual(metadata["class_mapping_status"], "assumed_not_archive_metadata")
            with self.assertRaises(FileExistsError):
                save_prepared_xray(original, output)

    def test_changed_artifact_or_metadata_is_rejected(self):
        preprocessing = XRayPreprocessor.fit(self.arrays()["train_images"])
        for key, value in (("fit_partition", "test"), ("pixel_divisor", 1), ("std_correction", 1)):
            metadata = copy.deepcopy(preprocessing.metadata())
            metadata[key] = value
            with self.assertRaises(ValueError):
                XRayPreprocessor.from_metadata(metadata)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.npz"
            np.savez(path, **self.arrays())
            output = Path(directory) / "prepared"
            save_prepared_xray(prepare_xray(path, expected_counts=None), output)
            with (output / "datasets.pt").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                load_data_bundle(output, expected_counts=None)


if __name__ == "__main__":
    unittest.main()
