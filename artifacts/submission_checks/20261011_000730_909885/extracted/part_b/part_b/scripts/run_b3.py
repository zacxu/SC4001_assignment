"""Run B3 with frozen B1/B2 hyperparameters and export the three-model comparison."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/cache/matplotlib"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from part_b.src.b3 import run_b3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "part_b/outputs/data")
    parser.add_argument("--b1", type=Path, default=ROOT / "part_b/outputs/b1")
    parser.add_argument("--b2", type=Path, default=ROOT / "part_b/outputs/b2")
    parser.add_argument("--output", type=Path, default=ROOT / "part_b/outputs/b3")
    parser.add_argument("--export-only", action="store_true", help="Rebuild saved assets without training/test inference")
    arguments = parser.parse_args()
    if arguments.export_only:
        from part_b.src.b3_exports import export_b3
        export_b3(arguments.output, source_directories={"b1": arguments.b1, "b2": arguments.b2, "b3": arguments.output})
        print(f"Rebuilt B3 assets: {arguments.output.resolve()}")
    else:
        config = json.loads((ROOT / "part_b/configs/default.json").read_text(encoding="utf-8"))
        run_b3(arguments.output, arguments.data, arguments.b1, arguments.b2, config)


if __name__ == "__main__":
    main()
