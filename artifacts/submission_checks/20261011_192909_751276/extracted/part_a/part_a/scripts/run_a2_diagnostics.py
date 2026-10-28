"""Generate FAQ-required A2 original-split learning curves without final test scoring."""

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
    parser.add_argument("--output", type=Path, default=ROOT / "part_a/outputs/a2/diagnostics")
    parser.add_argument("--prepared", type=Path, default=ROOT / "part_a/outputs/data")
    parser.add_argument("--a1", type=Path, default=ROOT / "part_a/outputs/a1")
    parser.add_argument("--faq", type=Path, default=ROOT / "Q&A.md")
    parser.add_argument("--export-only", action="store_true", help="Regenerate saved figures without training")
    arguments = parser.parse_args()
    from part_a.src.a2_diagnostics import export_a2_diagnostics, run_a2_diagnostics, verify_a2_diagnostics
    if arguments.export_only:
        export_a2_diagnostics(arguments.output)
        verify_a2_diagnostics(arguments.output)
        print(f"Regenerated A2 diagnostic curves from {arguments.output.resolve()}")
    else:
        run_a2_diagnostics(arguments.output, arguments.prepared, arguments.a1, arguments.faq)


if __name__ == "__main__":
    main()
