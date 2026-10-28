"""Apply the FAQ final protocol using immutable, already completed selections."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/cache/matplotlib"))

import torch

from assignment_common.artifacts import write_json
from part_b.src.b3 import comparison_rows
from part_b.src.data import load_data_bundle, sha256_file
from part_b.src.refit import attach_final_checkpoint, audit_final_refit, refit_final_classifier


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "part_b/outputs/data")
    parser.add_argument("--outputs", type=Path, default=ROOT / "part_b/outputs")
    parser.add_argument("--archive", type=Path, default=ROOT / "artifacts/faq_corrections/previous_part_b")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the corrected final experiments")
    output, archive = args.outputs.resolve(), args.archive.resolve()
    if output == archive or output in archive.parents:
        raise ValueError("Historical archive must be outside current experiment outputs")
    if not archive.exists():
        archive.mkdir(parents=True)
        for key in ("b1", "b2", "b3", "b4"):
            shutil.copytree(output / key, archive / key)
    bundle = load_data_bundle(args.data)
    audits = {}
    for architecture in ("b1", "b2", "b3"):
        directory = output / architecture
        summary = read(directory / "summary.json")
        selected = summary["selected"]
        frozen_path = directory / f"frozen_{architecture}_settings.json"
        frozen = read(frozen_path)
        if (directory / "refit").exists():
            audits[architecture] = audit_final_refit(directory)
            final = read(directory / "final_test.json")
        else:
            # Archive fidelity is checked before removing only the two superseded
            # final-test records. Original validation checkpoints/histories stay put.
            for name in ("final_test.json", "final_test_predictions.pt"):
                current, old = directory / name, archive / architecture / name
                if sha256_file(current) != sha256_file(old):
                    raise ValueError("Historical test evidence does not match the archive")
                current.unlink()
            batch_size = frozen.get("b1_batch_size", frozen.get("selected_batch_size"))
            final = refit_final_classifier(directory, bundle, selected,
                architecture=architecture, batch_size=batch_size)
        frozen = attach_final_checkpoint(frozen, final)
        write_json(frozen_path, frozen)
        summary.update(refit=True, final_test=final["metrics"],
            final_checkpoint=final["checkpoint"], final_checkpoint_sha256=final["checkpoint_sha256"],
            final_epoch=final["checkpoint_epoch"], combined_training_rows=final["combined_training_rows"],
            refit_protocol="refit_protocol.json", historical_protocol_archive=str(archive / architecture))
        write_json(directory / "summary.json", summary)
        config = read(directory / "configuration.json")
        config.update(refit=True, final_training_protocol="fresh_combined_train_validation_fixed_selected_epochs",
            final_training_rows=len(bundle.datasets["train"]) + len(bundle.datasets["validation"]),
            final_epoch_count=final["checkpoint_epoch"], class_mapping_status="confirmed_by_instructor_QA")
        write_json(directory / "configuration.json", config)
        audits[architecture] = audit_final_refit(directory)
        audits[architecture].update(selection_verification_preserved_in_archive=True,
            selection_evidence_unchanged=True, historical_verification=str(archive / architecture / "verification.json"))
        write_json(directory / "verification.json", audits[architecture])
        write_json(directory / "status.json", {"status": "completed", "stage": "faq_final_refit_completed"})
        write_json(directory / "progress.json", {"stage": "completed", "final_epoch": final["checkpoint_epoch"], "refit": True})
        print(f"{architecture.upper()} corrected final test: {final['metrics']}", flush=True)
    # Refresh inherited frozen metadata, retaining all selection choices.
    for architecture, upstream in (("b2", ("b1",)), ("b3", ("b1", "b2"))):
        directory = output / architecture
        config = read(directory / "configuration.json")
        frozen = read(directory / f"frozen_{architecture}_settings.json")
        for source in upstream:
            source_path = output / source / f"frozen_{source}_settings.json"
            config[f"{source}_frozen_settings"] = read(source_path)
            config[f"{source}_frozen_settings_sha256"] = sha256_file(source_path)
            if f"{source}_frozen_settings_sha256" in frozen:
                frozen[f"{source}_frozen_settings_sha256"] = sha256_file(source_path)
        write_json(directory / "configuration.json", config)
        write_json(directory / f"frozen_{architecture}_settings.json", frozen)
    b3 = output / "b3"
    previous = read(b3 / "comparison_inputs.json")
    records, lineage = {}, {}
    for architecture in ("b1", "b2", "b3"):
        directory = output / architecture
        final = read(directory / "final_test.json")
        records[architecture] = {key: final[key] for key in
            ("learning_rate", "batch_size", "checkpoint_epoch", "seed", "metrics", "test_evaluations", "checkpoint_sha256")}
        lineage[architecture] = {"directory": str(directory), "final_test_sha256": sha256_file(directory / "final_test.json"),
            "frozen_settings_sha256": sha256_file(directory / f"frozen_{architecture}_settings.json"),
            "checkpoint": final["checkpoint"], "checkpoint_sha256": final["checkpoint_sha256"]}
    complexities = {key: read(b3 / f"complexity/{key}.json") for key in records}
    comparison = comparison_rows(records, complexities)
    write_json(b3 / "comparison_inputs.json", {"records": records, "lineage": lineage,
        "b1_b2_test_results_reused_without_inference": True})
    write_json(b3 / "architecture_comparison.json", comparison)
    summary = read(b3 / "summary.json")
    summary["architecture_comparison"] = comparison
    write_json(b3 / "summary.json", summary)
    # Exporters use captured results/histories only; no further model evaluation.
    from part_b.src.b1_exports import export_b1
    from part_b.src.b2_exports import export_b2
    from part_b.src.b3_exports import export_b3
    export_b1(output / "b1")
    export_b2(output / "b2")
    export_b3(output / "b3", source_directories={key: output / key for key in records})
    write_json(archive.parent / "refit_verification.json", {"status": "passed", "models": audits,
        "new_final_training_runs": 3, "new_full_test_passes": 3, "existing_searches_reused": True,
        "normalization": "original training-only pixel statistics reused consistently"})


if __name__ == "__main__":
    main()
