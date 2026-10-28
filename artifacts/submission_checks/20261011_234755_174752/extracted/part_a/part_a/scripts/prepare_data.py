"""Export Part A's chronological selection and combined-refit datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PART_A = ROOT / "part_a"
sys.path.insert(0, str(ROOT))

import torch

from part_a.src.data import (
    CATEGORICAL_FEATURES, CONTINUOUS_FEATURES, EXCLUDED_YEARS, TEST_YEAR,
    TRAIN_YEARS, VALIDATION_YEAR, prepare_housing, save_prepared_housing,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="Override the supplied housing CSV path")
    parser.add_argument("--output", type=Path, default=PART_A / "outputs/data",
                        help="New destination directory; existing data is never overwritten")
    arguments = parser.parse_args()
    config = json.loads((PART_A / "configs/default.json").read_text(encoding="utf-8"))
    settings = config["settings"]
    expected = {"train_years": list(TRAIN_YEARS), "validation_year": VALIDATION_YEAR,
                "test_year": TEST_YEAR, "excluded_years": list(EXCLUDED_YEARS),
                "categorical_features": list(CATEGORICAL_FEATURES),
                "continuous_features": list(CONTINUOUS_FEATURES)}
    for name, value in expected.items():
        if settings[name] != value:
            raise ValueError(f"Part A config {name} disagrees with the assignment data schema")
    if arguments.output.exists():
        raise FileExistsError(f"Choose a new --output directory: {arguments.output}")
    csv_path = arguments.csv or ROOT / config["data_path_from_assignment_root"]
    # Small tensors on CPU; keep reduction scheduling consistent across exports
    # and baseline parity verification. GPU training belongs to later steps.
    torch.set_num_threads(4)
    prepared = prepare_housing(csv_path)
    destination = save_prepared_housing(prepared, arguments.output)
    print(f"Saved housing data: {destination.resolve()}")
    for bundle in (prepared.selection, prepared.refit):
        counts = {name: len(data) for name, data in bundle.datasets.items()}
        print(f"{bundle.protocol}: {counts}; fit years={bundle.preprocessing.fit_years}; "
              f"cardinalities={bundle.preprocessing.cardinalities}")
    print(f"Excluded 2023 rows: {prepared.report['excluded_rows']}; no test-fitted statistics.")


if __name__ == "__main__":
    main()
