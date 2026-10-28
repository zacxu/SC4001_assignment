"""Run B2's five-batch CUDA search using B1's frozen learning rate."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/cache/matplotlib"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from part_b.src.b2 import run_b2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "part_b/outputs/data")
    parser.add_argument("--b1", type=Path, default=ROOT / "part_b/outputs/b1")
    parser.add_argument("--output", type=Path, default=ROOT / "part_b/outputs/b2",
                        help="New experiment destination; existing results are protected")
    parser.add_argument("--export-only", action="store_true", help="Rebuild assets from saved results without training/test inference")
    arguments = parser.parse_args()
    if arguments.export_only:
        from part_b.src.b2_exports import export_b2
        export_b2(arguments.output)
        print(f"Rebuilt B2 report assets: {arguments.output.resolve()}")
    else:
        config = json.loads((ROOT / "part_b/configs/default.json").read_text(encoding="utf-8"))
        run_b2(arguments.output, arguments.data, arguments.b1, config)


if __name__ == "__main__":
    main()
