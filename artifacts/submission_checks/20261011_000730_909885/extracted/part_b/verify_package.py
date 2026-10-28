"""Verify a standalone extracted code submission using only local files."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/cache/matplotlib"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare(a, b, label="root"):
    import torch
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=label)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), label
        for key in a:
            compare(a[key], b[key], f"{label}.{key}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), label
        for index, (left, right) in enumerate(zip(a, b)):
            compare(left, right, f"{label}.{index}")
    else:
        assert a == b, (label, a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--ray-smoke", action="store_true", help="Part A CUDA-worker import and synthetic forward check")
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Use a Python venv containing requirements.txt dependencies")
    part = "part_a" if (ROOT / "part_a").is_dir() else "part_b"
    other = "part_b" if part == "part_a" else "part_a"
    assert not (ROOT / other).exists(), "This package must not depend on the other part"
    manifest = read(ROOT / "PACKAGE_MANIFEST.json")
    for relative, expected in manifest["files"].items():
        path = (ROOT / relative).resolve()
        assert path.is_relative_to(ROOT) and digest(path) == expected, relative
    pins = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if "==" in line and not line.lstrip().startswith("#"):
            name, version = line.split("==", 1)
            actual = importlib.metadata.version(name)
            assert actual == version, (name, actual, version)
            pins[name] = actual
    subprocess.run([sys.executable, "-m", "pip", "check"], cwd=ROOT, check=True)

    import torch
    torch.set_num_threads(4)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu for CPU inspection")
    output = ROOT / "artifacts/package_checks" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT)))
    assert result.wasSuccessful(), "Packaged behavioral tests failed"
    data = ROOT / part / "outputs/data"
    preprocessing_compared = []
    if args.prepare:
        fresh = output / "prepared"
        subprocess.run([sys.executable, "-B", str(ROOT / part / "scripts/prepare_data.py"), "--output", str(fresh)], cwd=ROOT, check=True)
        for original in sorted(data.rglob("*.pt")):
            relative = original.relative_to(data)
            compare(torch.load(original, map_location="cpu", weights_only=True),
                    torch.load(fresh / relative, map_location="cpu", weights_only=True), str(relative))
            preprocessing_compared.append(str(relative))
        # Source paths vary across extractions; all exported tensors/statistics
        # and source-data hashes above remain exactly equal.

    checkpoints = []
    if part == "part_a":
        from part_a.src.model import HousingMLP
        from part_a.src.wide_deep import WideDeepHousingModel
        from part_a.src.data import load_data_bundle
        bundle = load_data_bundle(data / "refit", expected_protocol="refit")
        frozen1 = read(ROOT / part / "outputs/a1/frozen_a1_settings.json")
        frozen3 = read(ROOT / part / "outputs/a3/frozen_a3_settings.json")
        for relative, cls in (("a1/" + frozen1["refit_checkpoint"], HousingMLP),
                              ("a3/" + frozen3["seed42_checkpoint"], WideDeepHousingModel)):
            path = ROOT / part / "outputs" / relative
            payload = torch.load(path, map_location="cpu", weights_only=True)
            model = cls.from_metadata(payload["model_config"], target_mean=payload["task"]["target_mean"], target_std=payload["task"]["target_std"])
            model.load_state_dict(payload["model_state"])
            model.to(args.device).eval()
            with torch.no_grad():
                prediction = model(torch.zeros(2, 4, dtype=torch.long, device=args.device), torch.zeros(2, 6, device=args.device))
            assert prediction.shape == (2,) and torch.isfinite(prediction).all()
            assert payload["epoch"] == frozen1["E_star"] and bundle.source_sha256 == frozen1["source_sha256"]
            checkpoints.append(relative)
    else:
        from part_b.src.models import build_model
        from part_b.src.data import load_data_bundle
        from part_b.src.refit import audit_final_refit
        bundle = load_data_bundle(data)
        for key in ("b1", "b2", "b3"):
            directory = ROOT / part / "outputs" / key
            frozen = read(directory / f"frozen_{key}_settings.json")
            refit_check = audit_final_refit(directory)
            assert refit_check["combined_training_rows"] == 5232 and refit_check["test_rows"] == 624
            assert frozen["refit"] is True
            path = directory / frozen["final_checkpoint"].replace("\\", "/")
            assert digest(path) == frozen["final_checkpoint_sha256"]
            assert path.resolve() != (directory / frozen["checkpoint"].replace("\\", "/")).resolve()
            assert bundle.source_sha256 == frozen["source_sha256"]
            model = build_model(key)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(payload["model_state"])
            model.to(args.device).eval()
            with torch.no_grad():
                prediction = model(torch.zeros(2, 1, 64, 64, device=args.device))
            assert prediction.shape == (2,) and torch.isfinite(prediction).all()
            assert payload["epoch"] == frozen["final_epoch"] == frozen["selected_epoch"]
            run_config = read(path.parent / "config.json")
            assert run_config["training"]["mode"] == "fixed_refit"
            assert run_config["training"]["max_epochs"] == frozen["final_epoch"]
            run_summary = read(path.parent / "summary.json")
            assert run_summary["epochs_completed"] == frozen["final_epoch"]
            checkpoints.append(str(path.relative_to(ROOT)))

    ray_record = None
    if args.ray_smoke:
        assert part == "part_a" and args.device == "cuda"
        import ray
        import tempfile
        ray.init(num_cpus=4, num_gpus=1, object_store_memory=256 * 1024**2,
                 include_dashboard=False, log_to_driver=False, _node_ip_address="127.0.0.1",
                 _temp_dir=str(Path(tempfile.gettempdir()) / "sc4001_package_ray"),
                 runtime_env={"env_vars": {"PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}})
        try:
            @ray.remote(num_cpus=4, num_gpus=1)
            def probe():
                import torch
                import part_a.src.model as local
                torch.set_num_threads(4)
                model = local.HousingMLP([13, 27, 44, 18], 0., 1.).cuda().eval()
                with torch.no_grad():
                    value = model(torch.zeros(2, 4, dtype=torch.long, device="cuda"), torch.zeros(2, 6, device="cuda"))
                return {"module": local.__file__, "finite": bool(torch.isfinite(value).all()), "device": torch.cuda.get_device_name(0)}
            ray_record = ray.get(probe.remote())
            assert Path(ray_record["module"]).resolve().is_relative_to(ROOT) and ray_record["finite"]
        finally:
            ray.shutdown()

    generated = []
    if args.export:
        # Capture tables before regeneration and compare values, not PDF creation
        # timestamps. No source run/checkpoint/preprocessing may change.
        import pandas as pd
        tables = {p: pd.read_csv(p) for p in (ROOT / part / "outputs").rglob("tables/*.csv")}
        immutable = {p: digest(p) for p in (ROOT / part / "outputs").rglob("*") if p.is_file()
                     and "figures" not in p.parts and "tables" not in p.parts
                     and p.name not in {"analysis.md", "curve_diagnostics.json"}
                     and not ("diagnostics" in p.parts and p.name == "verification.json")}
        letter = "a" if part == "part_a" else "b"
        for number in range(1, 5):
            script = ROOT / part / "scripts" / f"run_{letter}{number}.py"
            subprocess.run([sys.executable, "-B", str(script), "--export-only"], cwd=ROOT, check=True)
            generated.append(f"{letter}{number}")
        if letter == "a":
            subprocess.run([sys.executable, "-B", str(ROOT / part / "scripts/run_a2_diagnostics.py"), "--export-only"], cwd=ROOT, check=True)
            generated.append("a2_diagnostics")
        for path, expected in immutable.items():
            assert digest(path) == expected, str(path)
        for path, expected in tables.items():
            pd.testing.assert_frame_equal(expected, pd.read_csv(path), check_exact=False, rtol=1e-12, atol=1e-12)

    record = {"part": part, "status": "passed", "package_files_verified": len(manifest["files"]),
              "dependency_pins_verified": len(pins), "python": sys.version, "venv": sys.prefix,
              "fresh_dependency_installation": False, "tests_passed": result.testsRun, "device": args.device,
              "prepared_tensors_compared": preprocessing_compared, "synthetic_forward_checkpoints": checkpoints,
              "exports_regenerated": generated, "ray_worker": ray_record,
              "full_experiment_training_runs": 0, "test_set_model_inference": 0}
    (output / "verification.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
