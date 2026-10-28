"""Reproduce the packaged part into a new directory, preserving supplied evidence."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New destination; never overwrite bundled results")
    args = parser.parse_args()
    part = "part_a" if (ROOT / "part_a").is_dir() else "part_b"
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Run using a Python venv with requirements.txt installed")
    output.mkdir(parents=True)

    def run(script, *options):
        subprocess.run([sys.executable, "-B", str(ROOT / part / "scripts" / script), *map(str, options)], cwd=ROOT, check=True)

    run("prepare_data.py", "--output", output / "data")
    letter = "a" if part == "part_a" else "b"
    for number in range(1, 5):
        options = ["--output", output / f"{letter}{number}", "--prepared" if letter == "a" else "--data", output / "data"]
        dependencies = ({2: [1], 3: [1, 2], 4: [1, 3]} if letter == "a" else {2: [1], 3: [1, 2], 4: [1, 2, 3]})
        for prior in dependencies.get(number, []):
            options += [f"--{letter}{prior}", output / f"{letter}{prior}"]
        run(f"run_{letter}{number}.py", *options)
    print(f"Completed {part}; new results: {output}")


if __name__ == "__main__":
    main()
