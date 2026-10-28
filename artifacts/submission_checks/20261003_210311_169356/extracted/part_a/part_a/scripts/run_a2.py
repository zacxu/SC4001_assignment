"""Run all 15 A2 controlled CUDA refits, or regenerate their saved report assets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "part_a/outputs/a2")
    parser.add_argument("--prepared", type=Path, default=ROOT / "part_a/outputs/data")
    parser.add_argument("--a1", type=Path, default=ROOT / "part_a/outputs/a1")
    parser.add_argument("--export-only", action="store_true", help="Regenerate figures/tables without training or test scoring")
    arguments = parser.parse_args()
    if arguments.export_only:
        from part_a.src.a2_exports import export_a2
        export_a2(arguments.output)
        print(f"Regenerated A2 assets from {arguments.output.resolve()}")
    else:
        from part_a.src.a2 import run_a2
        run_a2(arguments.output, arguments.prepared, arguments.a1)


if __name__ == "__main__":
    main()
