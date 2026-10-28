"""Create the fixed five-image, three-checkpoint Grad-CAM comparison."""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/cache/matplotlib"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from part_b.src.b4 import SAMPLE_SEED, run_b4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "part_b/outputs/data")
    for architecture in ("b1", "b2", "b3"):
        parser.add_argument(f"--{architecture}", type=Path, default=ROOT / "part_b/outputs" / architecture)
    parser.add_argument("--output", type=Path, default=ROOT / "part_b/outputs/b4")
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--export-only", action="store_true", help="Rebuild saved figures/tables without model inference")
    arguments = parser.parse_args()
    if arguments.export_only:
        from part_b.src.b4_exports import export_b4
        export_b4(arguments.output)
        print(f"Rebuilt B4 display assets: {arguments.output.resolve()}")
    else:
        run_b4(arguments.output, arguments.data, {key: getattr(arguments, key) for key in ("b1", "b2", "b3")}, seed=arguments.seed)


if __name__ == "__main__":
    main()
