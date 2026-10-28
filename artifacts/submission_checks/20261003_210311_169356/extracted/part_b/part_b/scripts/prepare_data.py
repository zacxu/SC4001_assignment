"""Export the supplied Part B partitions, training-only scaling, and raw overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from part_b.src.data import CLASS_MAPPING, NORMALIZATION, prepare_xray, save_prepared_xray


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, help="Override the supplied archive location")
    parser.add_argument("--output", type=Path, default=ROOT / "part_b/outputs/data",
                        help="New output directory; existing prepared data is never overwritten")
    arguments = parser.parse_args()
    config = json.loads((ROOT / "part_b/configs/default.json").read_text(encoding="utf-8"))
    settings = config["settings"]
    if (settings["normalization"] != NORMALIZATION
            or settings["class_mapping_assumption"] != {str(k): v for k, v in CLASS_MAPPING.items()}):
        raise ValueError("Part B settings disagree with the agreed data protocol")
    if arguments.output.exists():
        raise FileExistsError(f"Choose a new --output directory: {arguments.output}")
    torch.set_num_threads(4)
    source = arguments.npz or ROOT / config["data_path_from_assignment_root"]
    bundle = prepare_xray(source)
    save_prepared_xray(bundle, arguments.output)
    print(f"Saved X-ray data: {arguments.output.resolve()}")
    print(f"Training pixel mean={bundle.preprocessing.mean:.16f}; population std={bundle.preprocessing.std:.16f}")
    for split, partition in bundle.report["partitions"].items():
        print(f"{split}: {partition['rows']} rows; label counts={partition['class_counts']}; "
              f"normalized shape={partition['normalized_shape']}")
    print(f"Exact-image overlap: {bundle.report['duplicates']['cross_split']}")
    print("Preserved all source rows; label semantics assumed; no test performance evaluated.")


if __name__ == "__main__":
    main()
