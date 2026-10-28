"""Run B1's five-rate CUDA search, final test once, and report exports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/cache/matplotlib"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from part_b.src.b1 import run_b1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "part_b/outputs/data")
    parser.add_argument("--output", type=Path, default=ROOT / "part_b/outputs/b1",
                        help="New destination; existing experiments are never overwritten")
    parser.add_argument("--export-only", action="store_true", help="Rebuild tables/figures from saved results without training or test evaluation")
    arguments = parser.parse_args()
    if arguments.export_only:
        from part_b.src.b1_exports import export_b1
        export_b1(arguments.output)
        print(f"Rebuilt B1 report assets from saved results: {arguments.output.resolve()}")
    else:
        config = json.loads((ROOT / "part_b/configs/default.json").read_text(encoding="utf-8"))
        run_b1(arguments.output, arguments.data, config)


if __name__ == "__main__":
    main()
