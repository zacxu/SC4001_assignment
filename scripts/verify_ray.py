r"""Run two tiny Ray Tune GPU trials to verify the local Windows runtime.

These trials test scheduling and CUDA; they are not A1 search results.
Run: .\.venv\Scripts\python.exe -B scripts\verify_ray.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import ray
from ray import tune

ROOT = Path(__file__).resolve().parents[1]


def gpu_probe(config: dict) -> None:
    import torch
    from ray import tune

    if not torch.cuda.is_available():
        raise RuntimeError("Ray worker cannot access CUDA")
    torch.manual_seed(config["seed"])
    tensor = torch.arange(config["width"], dtype=torch.float32, device="cuda")
    score = tensor.square().mean().item()
    tune.report({"probe_score": score, "cuda_available": True,
                 "device": torch.cuda.get_device_name(0)})


def main() -> None:
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        raise RuntimeError("Use the project .venv Python")
    settings = json.loads((ROOT / "configs/project.json").read_text(encoding="utf-8"))
    storage = ROOT / "artifacts" / "ray"
    temp_root = Path(tempfile.gettempdir()) / "sc4001_ray_setup"
    try:
        ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False,
                 _node_ip_address="127.0.0.1", _temp_dir=str(temp_root))
        resources = ray.cluster_resources()
        if resources.get("GPU", 0) < 1:
            raise RuntimeError(f"Ray did not discover the GPU: {resources}")
        tuner = tune.Tuner(
            tune.with_resources(gpu_probe, {"cpu": 1, "gpu": 1}),
            param_space={"width": tune.grid_search([16, 32]), "seed": settings["selection_seed"]},
            tune_config=tune.TuneConfig(max_concurrent_trials=1),
            run_config=tune.RunConfig(name="setup_gpu_grid", storage_path=str(storage), verbose=0),
        )
        results = tuner.fit()
        if len(results) != 2 or any(result.error for result in results):
            raise RuntimeError("The two-trial Ray Tune probe did not finish successfully")
        trials = [{"width": result.config["width"], "probe_score": result.metrics["probe_score"],
                   "device": result.metrics["device"], "cuda_available": result.metrics["cuda_available"]}
                  for result in results]
        if sorted(trial["width"] for trial in trials) != [16, 32]:
            raise RuntimeError("Ray grid did not enumerate both configurations")
        report = {"status": "passed", "kind": "setup_only_gpu_scheduling_probe",
                  "resources": resources, "trials": trials}
        destination = ROOT / "artifacts/setup/ray_check.json"
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
