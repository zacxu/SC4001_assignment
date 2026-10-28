"""Verify saved Part A data against the complete supplied baseline preprocessing."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from assignment_common.artifacts import write_json
from part_a.src.data import CATEGORICAL_FEATURES, REFIT_YEARS, TRAIN_YEARS, load_data_bundle


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=ROOT / "part_a/outputs/data")
    arguments = parser.parse_args()
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        raise RuntimeError("Use the project .venv interpreter")
    torch.set_num_threads(4)
    settings = json.loads((ROOT / "configs/project.json").read_text(encoding="utf-8"))
    csv_path = ROOT / settings["housing_csv"]
    frame = pd.read_csv(csv_path)
    checksum = sha256(csv_path)
    report = json.loads((arguments.prepared / "report.json").read_text(encoding="utf-8"))
    if report["source"]["sha256"] != checksum:
        raise AssertionError("CSV differs from the source used to export prepared data")
    project = ROOT / settings["housing_project"]
    sys.path.insert(0, str(project / "src"))
    from config import Config
    from data import prepare_data as supplied_prepare_data

    expected_counts = {"selection": {"train": 87370, "validation": 29057, "test": 26702},
                       "refit": {"train": 116427, "test": 26702}}
    checked = {}
    for protocol, years in (("selection", TRAIN_YEARS), ("refit", REFIT_YEARS)):
        bundle = load_data_bundle(arguments.prepared / protocol, expected_protocol=protocol)
        baseline = supplied_prepare_data(csv_path, replace(Config(), train_years=years))
        if bundle.source_sha256 != checksum:
            raise AssertionError("Prepared tensors have an unexpected source checksum")
        if bundle.preprocessing.cardinalities != [13, 27, 44, 18]:
            raise AssertionError("Unexpected category cardinalities")
        if {name: len(dataset) for name, dataset in bundle.datasets.items()} != expected_counts[protocol]:
            raise AssertionError("Unexpected chronological split sizes")
        fitting = frame[frame.year.isin(years)]
        maps = {name: {value: index + 1 for index, value in enumerate(sorted(fitting[name].unique().tolist()))}
                for name in CATEGORICAL_FEATURES}
        if bundle.preprocessing.category_maps != maps:
            raise AssertionError("Encoders differ from the supplied sorted training-only mappings")
        for key in ("continuous_mean", "continuous_std"):
            torch.testing.assert_close(getattr(bundle.preprocessing, key), baseline[key], rtol=0, atol=0)
        for key in ("target_mean", "target_std"):
            if getattr(bundle.preprocessing, key) != baseline[key]:
                raise AssertionError(f"{protocol} {key} differs from the baseline")
        for name, dataset in bundle.datasets.items():
            for actual, expected in zip(dataset.tensors, baseline[name].tensors):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            years_in_partition = years if name == "train" else ((2021,) if name == "validation" else (2022,))
            expected_ids = frame.index[frame.year.isin(years_in_partition)].tolist()
            if bundle.row_ids[name].tolist() != expected_ids:
                raise AssertionError(f"{protocol}/{name} changed CSV row order or membership")
        checked[protocol] = {
            "counts": expected_counts[protocol], "fit_years": list(years),
            "cardinalities": bundle.preprocessing.cardinalities,
            "all_tensors_and_statistics_exactly_match_supplied_baseline": True,
            "row_order_and_split_membership": "passed",
            "safe_weights_only_artifact_loading": "passed",
            "integer_month_map_keys_preserved": True,
        }
        print(f"{protocol}: exact parity with supplied baseline on every row and all fitted statistics.")
    membership = pd.read_csv(arguments.prepared / "split_membership.csv")
    if membership.row_id.tolist() != frame.index.tolist() or membership.year.tolist() != frame.year.tolist():
        raise AssertionError("Split membership is not traceable to the original CSV")
    if int((membership.selection_partition == "excluded").sum()) != 16424:
        raise AssertionError("Unexpected excluded-row count")
    excluded = membership.selection_partition == "excluded"
    if not (membership.loc[excluded, "year"] == 2023).all():
        raise AssertionError("Excluded membership differs from the required year")
    evidence = {
        "status": "passed", "checked_at_singapore": datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "python_executable": sys.executable, "torch": str(torch.__version__),
        "prepared_directory": str(arguments.prepared.resolve()), "source_sha256": checksum,
        "selection": checked["selection"], "refit": checked["refit"], "excluded_2023_rows": 16424,
        "data_quality": report["data_quality"], "no_model_training_or_test_performance_evaluation": True,
    }
    setup = json.loads((ROOT / "artifacts/setup/environment_check.json").read_text(encoding="utf-8"))
    for name, original in setup["files"].items():
        if sha256(ROOT / name) != original["sha256"]:
            raise AssertionError(f"Supplied file was modified: {name}")
    evidence["supplied_files_unchanged"] = True
    timestamp = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d_%H%M%S_%f")
    output = ROOT / "artifacts/housing_checks" / timestamp
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "verification.json", evidence)
    write_json(ROOT / "artifacts/housing_checks/latest.json", {"status": "passed", "directory": str(output),
                                                              "verification": str(output / "verification.json")})
    print(f"Verification passed: {output}")


if __name__ == "__main__":
    main()
