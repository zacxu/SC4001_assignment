"""Run A1's full 25-trial CUDA Ray grid and final combined-data refit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "part_a/outputs/a1",
                        help="New run directory; previous results are protected")
    parser.add_argument("--prepared", type=Path, default=ROOT / "part_a/outputs/data")
    parser.add_argument("--export-only", action="store_true", help="Regenerate assets from existing results")
    arguments = parser.parse_args()
    if arguments.export_only:
        from part_a.src.a1_exports import export_a1
        export_a1(arguments.output, prepared_directory=arguments.prepared)
        print(f"Regenerated A1 assets from {arguments.output.resolve()}")
        return
    config = json.loads((ROOT / "part_a/configs/default.json").read_text(encoding="utf-8"))
    if config["selection_seed"] != 42 or config["device"] != "cuda":
        raise ValueError("A1 uses the documented seed 42 and CUDA")
    from part_a.src.a1 import run_a1
    run_a1(arguments.output, arguments.prepared, config["settings"], root=ROOT)


if __name__ == "__main__":
    main()
