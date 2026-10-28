"""Explain all test rows for A1/A3, or refresh saved FeatureAblation report assets."""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/cache/matplotlib"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "part_a/outputs/a4")
    parser.add_argument("--prepared", type=Path, default=ROOT / "part_a/outputs/data")
    parser.add_argument("--a1", type=Path, default=ROOT / "part_a/outputs/a1")
    parser.add_argument("--a3", type=Path, default=ROOT / "part_a/outputs/a3")
    parser.add_argument("--export-only", action="store_true", help="Rebuild tables/figures from saved attributions")
    args = parser.parse_args()
    if args.export_only:
        from part_a.src.a4_exports import export_a4
        export_a4(args.output)
        print(f"Regenerated A4 assets from {args.output.resolve()}")
    else:
        from part_a.src.a4 import run_a4
        run_a4(args.output, args.prepared, args.a1, args.a3)


if __name__ == "__main__":
    main()
