"""Run A3 lambda selection/five refits, or regenerate completed report assets."""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "part_a/outputs/a3")
    parser.add_argument("--prepared", type=Path, default=ROOT / "part_a/outputs/data")
    parser.add_argument("--a1", type=Path, default=ROOT / "part_a/outputs/a1")
    parser.add_argument("--a2", type=Path, default=ROOT / "part_a/outputs/a2")
    parser.add_argument("--export-only", action="store_true", help="Rebuild assets from saved results")
    args = parser.parse_args()
    if args.export_only:
        from part_a.src.a3_exports import export_a3
        export_a3(args.output)
        print(f"Regenerated A3 assets from {args.output.resolve()}")
    else:
        from part_a.src.a3 import run_a3
        run_a3(args.output, args.prepared, args.a1, args.a2)


if __name__ == "__main__":
    main()
