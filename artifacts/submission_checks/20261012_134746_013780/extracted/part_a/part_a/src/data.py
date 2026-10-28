"""Chronological housing splits and baseline-compatible, training-only transforms."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from assignment_common.artifacts import write_checkpoint, write_json


CATEGORICAL_FEATURES = ("month", "town", "flat_model_type", "storey_range")
CONTINUOUS_FEATURES = (
    "dist_to_nearest_stn", "dist_to_dhoby", "degree_centrality",
    "eigenvector_centrality", "remaining_lease_years", "floor_area_sqm",
)
TARGET = "resale_price"
TRAIN_YEARS = (2017, 2018, 2019, 2020)
REFIT_YEARS = (*TRAIN_YEARS, 2021)
VALIDATION_YEAR = 2021
TEST_YEAR = 2022
EXCLUDED_YEARS = (2023,)
MIN_STD = 1e-8
FORMAT_VERSION = 1


def _validated_frame(frame: pd.DataFrame, *, target: bool, year: bool) -> pd.DataFrame:
    """Fail on unusable required values; never silently drop, fill, or deduplicate."""
    if not frame.columns.is_unique:
        raise ValueError("Housing columns must be unique")
    required = [*CATEGORICAL_FEATURES, *CONTINUOUS_FEATURES]
    if target:
        required.append(TARGET)
    if year:
        required.append("year")
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing housing columns: {missing}")
    data = frame.copy()
    for name in required:
        if data[name].isna().any():
            raise ValueError(f"Missing values in required column {name}")
    for name in CATEGORICAL_FEATURES[1:]:
        if not all(isinstance(value, str) and value.strip() for value in data[name]):
            raise ValueError(f"{name} must contain nonempty categorical strings")
    numeric = ["month", *CONTINUOUS_FEATURES]
    if target:
        numeric.append(TARGET)
    if year:
        numeric.append("year")
    for name in numeric:
        try:
            values = pd.to_numeric(data[name], errors="raise").to_numpy(dtype=np.float64)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Nonnumeric values in {name}") from error
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite values in {name}")
        if name in ("month", "year"):
            if not np.equal(values, np.floor(values)).all():
                raise ValueError(f"{name} must contain integers")
            if name == "month" and not ((values >= 1) & (values <= 12)).all():
                raise ValueError("month must be between 1 and 12")
            if name == "year" and not ((values >= 1) & (values <= 9999)).all():
                raise ValueError("year must be between 1 and 9999")
            data[name] = values.astype(np.int64)
        else:
            # The supplied model consumes float32. Reject finite float64 values
            # that would overflow that representation, before fitting anything.
            if (np.abs(values) > np.finfo(np.float32).max).any():
                raise ValueError(f"Values in {name} cannot be represented as float32")
            data[name] = values
    return data


def _continuous_tensor(frame: pd.DataFrame) -> torch.Tensor:
    values = frame.loc[:, list(CONTINUOUS_FEATURES)].to_numpy(dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(values))


def _target_tensor(frame: pd.DataFrame) -> torch.Tensor:
    return torch.tensor(frame[TARGET].to_numpy(dtype=np.float32), dtype=torch.float32)


@dataclass(frozen=True)
class HousingPreprocessor:
    category_maps: dict
    continuous_mean: torch.Tensor
    continuous_std: torch.Tensor
    target_mean: float
    target_std: float
    fit_years: tuple[int, ...]
    fit_row_count: int

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.category_maps[name]) + 1 for name in CATEGORICAL_FEATURES]

    @classmethod
    def fit(cls, frame: pd.DataFrame, *, fit_years=TRAIN_YEARS) -> HousingPreprocessor:
        """Select permitted fitting years internally, even when given the full CSV."""
        years = tuple(fit_years)
        if years not in (TRAIN_YEARS, REFIT_YEARS):
            raise ValueError("Fit years must be 2017-2020 or 2017-2021; never include test data")
        data = _validated_frame(frame, target=True, year=True)
        fitting = data.loc[data["year"].isin(years)]
        if len(fitting) < 2 or set(fitting["year"].unique()) != set(years):
            raise ValueError("Every fitting year must be present and at least two rows are required")
        category_maps = {
            name: {value: index + 1 for index, value in enumerate(sorted(fitting[name].unique().tolist()))}
            for name in CATEGORICAL_FEATURES
        }
        continuous = _continuous_tensor(fitting)
        targets = _target_tensor(fitting)
        means = continuous.mean(dim=0)
        stds = continuous.std(dim=0, correction=1).clamp_min(MIN_STD)
        target_mean = targets.mean().item()
        target_std = targets.std(correction=0).clamp_min(MIN_STD).item()
        if not torch.isfinite(means).all() or not torch.isfinite(stds).all():
            raise ValueError("Nonfinite fitted continuous statistics")
        if not np.isfinite([target_mean, target_std]).all():
            raise ValueError("Nonfinite fitted target statistics")
        return cls(category_maps, means, stds, target_mean, target_std, years, len(fitting))

    def transform_features(self, frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode ten inputs without requiring year, address, station name, or target."""
        data = _validated_frame(frame, target=False, year=False)
        categorical = torch.tensor(
            np.column_stack([
                data[name].map(self.category_maps[name]).fillna(0).to_numpy(dtype=np.int64)
                for name in CATEGORICAL_FEATURES
            ]), dtype=torch.long,
        )
        continuous = (_continuous_tensor(data) - self.continuous_mean) / self.continuous_std
        if not torch.isfinite(continuous).all():
            raise ValueError("Nonfinite standardized inputs")
        return categorical, continuous

    def transform(self, frame: pd.DataFrame) -> TensorDataset:
        data = _validated_frame(frame, target=True, year=False)
        categorical, continuous = self.transform_features(data)
        # Raw SGD targets: RegressionTask performs the target standardization.
        return TensorDataset(categorical, continuous, _target_tensor(data))

    def standardize_target(self, prices: torch.Tensor) -> torch.Tensor:
        return (prices - self.target_mean) / self.target_std

    def inverse_target(self, standardized: torch.Tensor) -> torch.Tensor:
        return standardized * self.target_std + self.target_mean

    def metadata(self) -> dict:
        """Pass this exact payload to train_regressor(preprocessing=...)."""
        return {
            "format_version": FORMAT_VERSION,
            "categorical_features": list(CATEGORICAL_FEATURES),
            "continuous_features": list(CONTINUOUS_FEATURES),
            "target": TARGET, "target_units": "SGD", "stored_targets": "raw_sgd",
            "category_maps": self.category_maps, "unknown_category_id": 0,
            "cardinalities": self.cardinalities,
            "continuous_mean": self.continuous_mean, "continuous_std": self.continuous_std,
            "continuous_std_correction": 1, "target_mean": self.target_mean,
            "target_std": self.target_std, "target_std_correction": 0,
            "minimum_std": MIN_STD, "fit_years": list(self.fit_years),
            "fit_row_count": self.fit_row_count,
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> HousingPreprocessor:
        if metadata["format_version"] != FORMAT_VERSION:
            raise ValueError("Unsupported housing preprocessing format")
        if (tuple(metadata["categorical_features"]) != CATEGORICAL_FEATURES
                or tuple(metadata["continuous_features"]) != CONTINUOUS_FEATURES
                or metadata["target"] != TARGET or metadata["stored_targets"] != "raw_sgd"
                or metadata["unknown_category_id"] != 0
                or metadata["continuous_std_correction"] != 1
                or metadata["target_std_correction"] != 0):
            raise ValueError("Housing preprocessing schema does not match the assignment")
        years = tuple(metadata["fit_years"])
        if years not in (TRAIN_YEARS, REFIT_YEARS):
            raise ValueError("Preprocessing must be fitted on permitted years")
        maps = metadata["category_maps"]
        for name in CATEGORICAL_FEATURES:
            values = sorted(maps[name])
            if [maps[name][value] for value in values] != list(range(1, len(values) + 1)):
                raise ValueError(f"Invalid category IDs for {name}")
        if not all(type(value) is int for value in maps["month"]):
            raise ValueError("Month category keys must retain their integer type")
        mean, std = metadata["continuous_mean"], metadata["continuous_std"]
        if (mean.shape != (6,) or std.shape != (6,) or mean.dtype != torch.float32
                or std.dtype != torch.float32 or not torch.isfinite(mean).all()
                or not torch.isfinite(std).all() or not (std > 0).all()
                or not np.isfinite([metadata["target_mean"], metadata["target_std"]]).all()
                or metadata["target_std"] <= 0):
            raise ValueError("Invalid housing scaling statistics")
        result = cls(maps, mean, std, metadata["target_mean"], metadata["target_std"],
                     years, metadata["fit_row_count"])
        if result.cardinalities != metadata["cardinalities"]:
            raise ValueError("Category cardinalities do not match encoders")
        return result

    @classmethod
    def load(cls, path: Path | str) -> HousingPreprocessor:
        return cls.from_metadata(torch.load(path, map_location="cpu", weights_only=True))


@dataclass
class HousingDataBundle:
    protocol: str
    preprocessing: HousingPreprocessor
    datasets: dict[str, TensorDataset]
    row_ids: dict[str, torch.Tensor]
    source_sha256: str


@dataclass
class PreparedHousing:
    selection: HousingDataBundle
    refit: HousingDataBundle
    split_membership: pd.DataFrame
    report: dict


def _bundle(protocol: str, data: pd.DataFrame, source_sha256: str) -> HousingDataBundle:
    fit_years = TRAIN_YEARS if protocol == "selection" else REFIT_YEARS
    preprocessing = HousingPreprocessor.fit(data, fit_years=fit_years)
    partitions = {"train": data["year"].isin(fit_years), "test": data["year"] == TEST_YEAR}
    if protocol == "selection":
        partitions["validation"] = data["year"] == VALIDATION_YEAR
    datasets, row_ids = {}, {}
    for name, mask in partitions.items():
        rows = data.loc[mask]
        if rows.empty:
            raise ValueError(f"The {name} partition is empty")
        datasets[name] = preprocessing.transform(rows)
        row_ids[name] = torch.tensor(rows.index.to_numpy(), dtype=torch.long)
    return HousingDataBundle(protocol, preprocessing, datasets, row_ids, source_sha256)


def _bundle_report(bundle: HousingDataBundle, data: pd.DataFrame) -> dict:
    preprocessing = bundle.preprocessing
    metadata = preprocessing.metadata()
    # JSON object keys cannot preserve integer months: use typed value/ID records
    # in this human-readable report. The .pt metadata keeps real dictionary keys.
    metadata["category_maps"] = {
        name: [{"value": value, "id": index} for value, index in mapping.items()]
        for name, mapping in preprocessing.category_maps.items()
    }
    partitions = {}
    for name, dataset in bundle.datasets.items():
        rows = data.loc[bundle.row_ids[name].tolist()]
        categorical = dataset.tensors[0]
        partitions[name] = {
            "rows": len(dataset),
            "year_counts": rows["year"].value_counts().sort_index().to_dict(),
            "unknown_categories": {
                feature: {
                    "rows": int((categorical[:, index] == 0).sum()),
                    "values": sorted(set(rows[feature]) - set(preprocessing.category_maps[feature])),
                } for index, feature in enumerate(CATEGORICAL_FEATURES)
            },
        }
    return {"preprocessing": metadata, "partitions": partitions}


def prepare_housing(csv_path: Path | str) -> PreparedHousing:
    """Prepare both fixed protocols; no imputation, deduplication, or test fitting."""
    path = Path(csv_path).resolve()
    with path.open("rb") as source_file:
        source_sha256 = hashlib.file_digest(source_file, "sha256").hexdigest()
    source = pd.read_csv(path)
    data = _validated_frame(source, target=True, year=True).reset_index(drop=True)
    unexpected = set(data["year"].unique()) - {*REFIT_YEARS, TEST_YEAR, *EXCLUDED_YEARS}
    if unexpected:
        raise ValueError(f"Unexpected housing years: {sorted(unexpected)}")
    # Excluded rows cannot enter either fitting operation or exported dataset.
    included = data.loc[~data["year"].isin(EXCLUDED_YEARS)]
    selection = _bundle("selection", included, source_sha256)
    refit = _bundle("refit", included, source_sha256)
    labels = np.full(len(data), "excluded", dtype=object)
    labels[data["year"].isin(TRAIN_YEARS)] = "train"
    labels[data["year"] == VALIDATION_YEAR] = "validation"
    labels[data["year"] == TEST_YEAR] = "test"
    membership = pd.DataFrame({"row_id": data.index, "year": data["year"], "selection_partition": labels})
    report = {
        "format_version": FORMAT_VERSION,
        "source": {"path": str(path), "sha256": source_sha256, "rows": len(data),
                   "columns": source.columns.tolist()},
        "data_quality": {
            "missing_counts_all_source_columns": source.isna().sum().to_dict(),
            "exact_duplicate_rows": int(source.duplicated().sum()),
            "required_numeric_values_finite": True, "required_categories_nonempty": True,
            "rows_dropped_or_imputed": 0,
            "handling": "Reject invalid required values; preserve duplicates and original row order.",
        },
        "year_counts": data["year"].value_counts().sort_index().to_dict(),
        "excluded_years": list(EXCLUDED_YEARS), "excluded_rows": int((labels == "excluded").sum()),
        "input_features": [*CATEGORICAL_FEATURES, *CONTINUOUS_FEATURES],
        "not_model_inputs": ["year", "full_address", "nearest_stn"],
        "row_id_definition": "Zero-based data row in the original CSV, excluding its header.",
        "selection": _bundle_report(selection, data), "refit": _bundle_report(refit, data),
        "test_fitted_statistics": False,
        "refit_reuse": "Use the same saved 2017-2021 preprocessing for A1/A2/A3 refits.",
    }
    return PreparedHousing(selection, refit, membership, report)


def save_prepared_housing(prepared: PreparedHousing, directory: Path | str) -> Path:
    """Export to a new directory, preserving earlier prepared data."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=False)
    for bundle in (prepared.selection, prepared.refit):
        folder = destination / bundle.protocol
        folder.mkdir()
        write_checkpoint(folder / "preprocessing.pt", bundle.preprocessing.metadata())
        write_checkpoint(folder / "datasets.pt", {
            "format_version": FORMAT_VERSION, "protocol": bundle.protocol,
            "source_sha256": bundle.source_sha256,
            "partitions": {
                name: {"categorical": dataset.tensors[0], "continuous": dataset.tensors[1],
                       "target_sgd": dataset.tensors[2], "row_ids": bundle.row_ids[name]}
                for name, dataset in bundle.datasets.items()
            },
        })
    prepared.split_membership.to_csv(destination / "split_membership.csv", index=False)
    write_json(destination / "report.json", prepared.report)
    return destination


def load_data_bundle(directory: Path | str, *, expected_protocol: str) -> HousingDataBundle:
    """Load CPU data safely and explicitly guard selection versus refit transforms."""
    if expected_protocol not in ("selection", "refit"):
        raise ValueError("Expected protocol must be selection or refit")
    folder = Path(directory)
    preprocessing = HousingPreprocessor.load(folder / "preprocessing.pt")
    payload = torch.load(folder / "datasets.pt", map_location="cpu", weights_only=True)
    years = TRAIN_YEARS if expected_protocol == "selection" else REFIT_YEARS
    if (payload["format_version"] != FORMAT_VERSION or payload["protocol"] != expected_protocol
            or preprocessing.fit_years != years):
        raise ValueError("Prepared data protocol does not match the requested training stage")
    expected_names = {"train", "validation", "test"} if expected_protocol == "selection" else {"train", "test"}
    if set(payload["partitions"]) != expected_names:
        raise ValueError("Prepared data has incorrect partitions")
    datasets, row_ids = {}, {}
    for name, partition in payload["partitions"].items():
        cat, cont, target, ids = (partition[key] for key in ("categorical", "continuous", "target_sgd", "row_ids"))
        count = len(target)
        if (count == 0 or cat.shape != (count, 4) or cont.shape != (count, 6)
                or target.shape != (count,) or ids.shape != (count,)
                or cat.dtype != torch.long or ids.dtype != torch.long
                or cont.dtype != torch.float32 or target.dtype != torch.float32
                or not torch.isfinite(cont).all() or not torch.isfinite(target).all()):
            raise ValueError(f"Invalid prepared tensors in {name}")
        for index, cardinality in enumerate(preprocessing.cardinalities):
            if not ((cat[:, index] >= 0) & (cat[:, index] < cardinality)).all():
                raise ValueError(f"Invalid category IDs in {name}")
        datasets[name] = TensorDataset(cat, cont, target)
        row_ids[name] = ids
    if len(datasets["train"]) != preprocessing.fit_row_count:
        raise ValueError("Prepared training count does not match fitted preprocessing")
    all_ids = torch.cat(list(row_ids.values()))
    if (all_ids < 0).any() or len(torch.unique(all_ids)) != len(all_ids):
        raise ValueError("Prepared partitions overlap or have invalid row IDs")
    return HousingDataBundle(expected_protocol, preprocessing, datasets, row_ids, payload["source_sha256"])
