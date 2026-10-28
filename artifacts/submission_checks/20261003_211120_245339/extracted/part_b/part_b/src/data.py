"""Preserve supplied X-ray partitions and normalize using training pixels only."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from assignment_common.artifacts import write_checkpoint, write_json


FORMAT_VERSION = 1
SPLITS = {"train": "train", "validation": "val", "test": "test"}
EXPECTED_COUNTS = {"train": 4708, "validation": 524, "test": 624}
CLASS_MAPPING = {0: "healthy", 1: "pneumonia"}
NORMALIZATION = "training_pixel_mean_and_population_std_after_dividing_by_255"


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_images(images: np.ndarray) -> None:
    if (not isinstance(images, np.ndarray) or images.dtype != np.uint8
            or images.ndim != 3 or images.shape[1:] != (64, 64) or not len(images)):
        raise ValueError("Expected nonempty uint8 images with shape N x 64 x 64; do not rescale twice")


def validate_labels(labels: np.ndarray, count: int) -> None:
    if (labels.shape != (count,) or labels.dtype.kind not in "iu"
            or not np.isin(labels, [0, 1]).all()):
        raise ValueError("Expected one integer binary label (0 or 1) per image")


@dataclass(frozen=True)
class XRayPreprocessor:
    mean: float
    std: float
    fit_row_count: int

    def __post_init__(self):
        if (not math.isfinite(self.mean) or not 0 <= self.mean <= 1
                or not math.isfinite(self.std) or not 0 < self.std <= 0.5
                or type(self.fit_row_count) is not int or self.fit_row_count < 1):
            raise ValueError("Invalid training-pixel normalization statistics")

    @classmethod
    def fit(cls, training_images: np.ndarray) -> XRayPreprocessor:
        """Caller passes training images only; labels are never used to fit scaling."""
        validate_images(training_images)
        scaled = training_images.astype(np.float32) / np.float32(255.0)
        return cls(float(scaled.mean(dtype=np.float64)), float(scaled.std(dtype=np.float64)),
                   len(training_images))

    def transform(self, images: np.ndarray) -> torch.Tensor:
        validate_images(images)
        scaled = images.astype(np.float32) / np.float32(255.0)
        # Use the same float32 arithmetic for every partition and future inference.
        normalized = (scaled - np.float32(self.mean)) / np.float32(self.std)
        return torch.from_numpy(np.ascontiguousarray(normalized[:, None, :, :]))

    def metadata(self) -> dict:
        return {"format_version": FORMAT_VERSION, "normalization": NORMALIZATION,
                "mean": self.mean, "std": self.std, "std_correction": 0,
                "reduction_dtype": "float64", "transformation_dtype": "float32",
                "input_dtype": "uint8", "input_shape": [64, 64],
                "output_shape": [1, 64, 64], "pixel_divisor": 255.0,
                "fit_partition": "train", "fit_row_count": self.fit_row_count,
                "fit_pixel_count": self.fit_row_count * 64 * 64,
                "class_mapping": dict(CLASS_MAPPING),
                "class_mapping_status": "assumed_not_archive_metadata"}

    @classmethod
    def from_metadata(cls, metadata: dict) -> XRayPreprocessor:
        result = cls(metadata["mean"], metadata["std"], metadata["fit_row_count"])
        if metadata != result.metadata():
            raise ValueError("Preprocessing metadata does not match the training-only X-ray schema")
        return result

    @classmethod
    def load(cls, path: Path | str) -> XRayPreprocessor:
        return cls.from_metadata(torch.load(path, map_location="cpu", weights_only=True))


@dataclass
class XRayDataBundle:
    preprocessing: XRayPreprocessor
    datasets: dict[str, TensorDataset]
    raw_images: dict[str, torch.Tensor]
    row_ids: dict[str, torch.Tensor]
    source_sha256: str
    report: dict


def _duplicate_report(arrays: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """Inspect exact pixel equality, recording rows without changing any partition."""
    groups = defaultdict(list)
    for split, (images, labels) in arrays.items():
        for index, (image, label) in enumerate(zip(images, labels)):
            groups[hashlib.sha256(image.tobytes()).hexdigest()].append(
                {"split": split, "row_id": index, "label": int(label)})
    repeated = [{"pixel_sha256": digest, "rows": rows}
                for digest, rows in groups.items() if len(rows) > 1]
    within = {}
    for split in SPLITS:
        within[split] = sum(max(0, sum(row["split"] == split for row in group["rows"]) - 1)
                            for group in repeated)
    cross = []
    names = list(SPLITS)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            matches = [group for group in repeated
                       if {first, second}.issubset({row["split"] for row in group["rows"]})]
            cross.append({"splits": [first, second], "shared_unique_images": len(matches),
                          "rows_in_first": sum(row["split"] == first for g in matches for row in g["rows"]),
                          "rows_in_second": sum(row["split"] == second for g in matches for row in g["rows"])})
    conflicts = [group for group in repeated if len({row["label"] for row in group["rows"]}) > 1]
    return {"method": "SHA256 of exact uint8 64x64 pixel bytes; no perceptual matching",
            "within_split_extra_copies": within, "cross_split": cross,
            "conflicting_label_groups": len(conflicts), "repeated_image_groups": repeated,
            "action": "Preserve supplied rows and partitions; disclose training-validation overlap",
            "patient_identity": "No patient identifiers are supplied; patient overlap cannot be checked"}


def prepare_xray(path: Path | str, *, expected_counts: dict | None = EXPECTED_COUNTS) -> XRayDataBundle:
    source = Path(path)
    source_hash = sha256_file(source)
    arrays = {}
    with np.load(source, allow_pickle=False) as archive:
        required = {prefix + suffix for prefix in SPLITS.values() for suffix in ("_images", "_labels")}
        if set(archive.files) != required:
            raise ValueError(f"Expected exactly the six supplied NPZ keys: {sorted(required)}")
        for split, prefix in SPLITS.items():
            images, labels = archive[prefix + "_images"], archive[prefix + "_labels"]
            validate_images(images)
            validate_labels(labels, len(images))
            if expected_counts is not None and len(images) != expected_counts[split]:
                raise ValueError(f"Unexpected {split} size: {len(images)}")
            arrays[split] = (images, labels)
    preprocessing = XRayPreprocessor.fit(arrays["train"][0])
    datasets, raw_images, row_ids, partitions = {}, {}, {}, {}
    for split, (images, labels) in arrays.items():
        normalized = preprocessing.transform(images)
        targets = torch.from_numpy(labels.astype(np.float32))
        datasets[split] = TensorDataset(normalized, targets)
        raw_images[split] = torch.from_numpy(images.copy())
        row_ids[split] = torch.arange(len(images), dtype=torch.long)
        counts = {label: int((labels == label).sum()) for label in CLASS_MAPPING}
        partitions[split] = {"rows": len(images), "npz_prefix": SPLITS[split],
                             "class_counts": counts, "positive_fraction": counts[1] / len(images),
                             "raw_dtype": "uint8", "raw_shape": list(images.shape),
                             "raw_pixel_min": int(images.min()), "raw_pixel_max": int(images.max()),
                             "normalized_shape": list(normalized.shape), "normalized_dtype": "float32",
                             "normalized_pixel_mean": float(normalized.numpy().mean(dtype=np.float64)),
                             "normalized_pixel_population_std": float(normalized.numpy().std(dtype=np.float64)),
                             "source_row_order_preserved": True}
    if sha256_file(source) != source_hash:
        raise RuntimeError("Source archive changed during preparation")
    report = {"format_version": FORMAT_VERSION, "source_name": source.name,
              "source_sha256": source_hash, "preprocessing": preprocessing.metadata(),
              "partitions": partitions, "duplicates": _duplicate_report(arrays),
              "augmentation": None, "class_weighting": None,
              "partitioning": "Supplied train/val/test membership and order preserved",
              "test_performance_evaluated": False}
    return XRayDataBundle(preprocessing, datasets, raw_images, row_ids, source_hash, report)


def save_prepared_xray(bundle: XRayDataBundle, directory: Path | str) -> Path:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=False)
    write_checkpoint(destination / "preprocessing.pt", bundle.preprocessing.metadata())
    write_json(destination / "normalization.json", bundle.preprocessing.metadata())
    write_checkpoint(destination / "datasets.pt", {
        "format_version": FORMAT_VERSION, "protocol": "provided_partitions",
        "preprocessing": bundle.preprocessing.metadata(), "source_sha256": bundle.source_sha256,
        "splits": {split: {"images": dataset.tensors[0], "labels": dataset.tensors[1],
                           "raw_images": bundle.raw_images[split], "row_ids": bundle.row_ids[split]}
                   for split, dataset in bundle.datasets.items()},
    })
    write_json(destination / "data_report.json", bundle.report)
    write_json(destination / "manifest.json", {"format_version": FORMAT_VERSION,
               "source_sha256": bundle.source_sha256,
               "files": {name: sha256_file(destination / name) for name in
                         ("preprocessing.pt", "normalization.json", "datasets.pt", "data_report.json")}})
    return destination


def load_data_bundle(directory: Path | str, *, expected_counts: dict | None = EXPECTED_COUNTS) -> XRayDataBundle:
    import json

    source = Path(directory)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    for name in ("preprocessing.pt", "normalization.json", "datasets.pt", "data_report.json"):
        if sha256_file(source / name) != manifest["files"][name]:
            raise ValueError(f"Prepared X-ray artifact changed: {name}")
    payload = torch.load(source / "datasets.pt", map_location="cpu", weights_only=True)
    if (payload["format_version"] != FORMAT_VERSION or payload["protocol"] != "provided_partitions"
            or set(payload["splits"]) != set(SPLITS)):
        raise ValueError("Invalid prepared X-ray dataset schema")
    preprocessing = XRayPreprocessor.from_metadata(payload["preprocessing"])
    if (XRayPreprocessor.load(source / "preprocessing.pt") != preprocessing
            or manifest["source_sha256"] != payload["source_sha256"]):
        raise ValueError("Prepared dataset normalization or source provenance disagrees")
    datasets, raw_images, row_ids = {}, {}, {}
    for split, part in payload["splits"].items():
        images, labels, raw, ids = (part[key] for key in ("images", "labels", "raw_images", "row_ids"))
        validate_images(raw.numpy())
        count = len(raw)
        if (images.shape != (count, 1, 64, 64) or images.dtype != torch.float32
                or labels.shape != (count,) or labels.dtype != torch.float32
                or not torch.isfinite(images).all() or not ((labels == 0) | (labels == 1)).all()
                or ids.dtype != torch.long or not torch.equal(ids, torch.arange(count))
                or (expected_counts is not None and count != expected_counts[split])):
            raise ValueError(f"Invalid {split} tensors or original row IDs")
        datasets[split], raw_images[split], row_ids[split] = TensorDataset(images, labels), raw, ids
    if preprocessing.fit_row_count != len(datasets["train"]):
        raise ValueError("Training statistics must correspond to the supplied training partition")
    report = json.loads((source / "data_report.json").read_text(encoding="utf-8"))
    return XRayDataBundle(preprocessing, datasets, raw_images, row_ids, payload["source_sha256"], report)
