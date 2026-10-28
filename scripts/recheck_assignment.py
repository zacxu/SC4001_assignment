"""Recheck assignment protocols and saved numerical evidence without model inference."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as functional


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left, right):
    assert math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-8), (left, right)


def history(folder):
    rows = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in rows] == list(range(1, len(rows) + 1))
    assert 1 <= len(rows) <= 20
    assert not any(key.startswith("test_") for row in rows for key in row)
    return rows


def stopping(values):
    best, waiting = float("inf"), 0
    for index, value in enumerate(values):
        if value < best - 1e-4:
            best, waiting = value, 0
        else:
            waiting += 1
        assert waiting < 4 or index == len(values) - 1, "Continued after patience expired"
    assert len(values) == 20 or waiting >= 4, "Stopped without meeting patience or cap"


def main():
    assert Path(sys.prefix).resolve() == (ROOT / ".venv").resolve()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    checks = {}
    # Recheck the exact supplied partitions and normalization definitions.
    from part_a.src.data import load_data_bundle as housing_bundle
    from part_b.src.data import load_data_bundle as xray_bundle
    from part_a.src.model import parameter_count
    from part_b.src.models import build_model
    from part_b.src.refit import audit_final_refit
    selection = housing_bundle(ROOT / "part_a/outputs/data/selection", expected_protocol="selection")
    refit = housing_bundle(ROOT / "part_a/outputs/data/refit", expected_protocol="refit")
    assert {key: len(value) for key, value in selection.datasets.items()} == {"train": 87370, "validation": 29057, "test": 26702}
    assert {key: len(value) for key, value in refit.datasets.items()} == {"train": 116427, "test": 26702}
    csv = pd.read_csv(ROOT / "housing_baseline_data/hdb-price-prediction-assignment-main/hdb_price_prediction.csv")
    for bundle in (selection, refit):
        pre = bundle.preprocessing.metadata()
        assert pre["continuous_std_correction"] == 1 and pre["target_std_correction"] == 0
        fitted = csv[csv.year.isin(pre["fit_years"])]
        values = torch.tensor(fitted[pre["continuous_features"]].to_numpy(), dtype=torch.float32).contiguous()
        prices = torch.tensor(fitted.resale_price.to_numpy(), dtype=torch.float32)
        torch.testing.assert_close(pre["continuous_mean"], values.mean(0), rtol=0, atol=0)
        torch.testing.assert_close(pre["continuous_std"], values.std(0, correction=1).clamp_min(1e-8), rtol=0, atol=0)
        close(pre["target_mean"], prices.mean()); close(pre["target_std"], prices.std(correction=0))
        assert pre["cardinalities"] == [13, 27, 44, 18] and len(pre["continuous_features"]) == 6
    checks["housing_data"] = "Chronology, train-only fitting, sample input SD and population target SD reproduced"

    # All 25 Ray candidates, validation choice and the clarified A1 refit.
    a1 = read("part_a/outputs/a1/summary.json")
    grid = read("part_a/outputs/a1/grid_results.json")
    assert {(r["hidden_width"], r["embedding_dimension"]) for r in grid} == {(h, d) for h in (16,32,64,128,256) for d in (2,4,8,12,16)} and len(grid) == 25
    chosen = min(grid, key=lambda r: (r["validation_rmse"], r["trainable_parameters"], r["hidden_width"], r["embedding_dimension"]))
    assert chosen == a1["selected"] and (chosen["hidden_width"], chosen["embedding_dimension"]) == (256,4)
    assert parameter_count([13,27,44,18],64,8) == 3505
    runs = []
    for row in grid:
        directory = ROOT / "part_a/outputs/a1" / row["run_directory"].replace("\\", "/")
        rows = history(directory); stopping([r["validation_rmse"] for r in rows])
        assert min(rows, key=lambda r: (r["validation_rmse"],r["epoch"]))["epoch"] == row["best_epoch"]
        assert len(rows) == row["epochs_completed"] and parameter_count([13,27,44,18],row["hidden_width"],row["embedding_dimension"]) == row["trainable_parameters"]
        runs.append(directory)
    a1refit = ROOT / "part_a/outputs/a1/refit"
    monitor = [json.loads(line) for line in (a1refit / "internal_test_monitor.jsonl").read_text().splitlines()]
    stopping([r["rmse"] for r in monitor])
    assert min(monitor, key=lambda r: (r["rmse"],r["epoch"]))["epoch"] == a1["E_star"] == 4
    assert a1["E_val"] == 6 and len(history(a1refit)) == len(monitor) == 8 and a1["test_assisted_stopping"]
    runs.append(a1refit)
    checks["A1"] = "25 complete Ray candidates; default/selected counts; validation choice; refit E*=4 with private test monitor"

    a2 = read("part_a/outputs/a2/summary.json")
    records2 = read("part_a/outputs/a2/per_seed_results.json")
    a3 = read("part_a/outputs/a3/summary.json")
    records3 = read("part_a/outputs/a3/per_seed_results.json")
    lambda_rows = read("part_a/outputs/a3/lambda_search.json")
    assert len(lambda_rows)==5 and {r["lambda"] for r in lambda_rows} == {0.,.25,.5,1.,2.}
    assert min(lambda_rows,key=lambda r:(r["validation_rmse"],r["lambda"])) == a3["selected"]
    assert a3["selected"]["lambda"] == 2
    for row in lambda_rows:
        directory=ROOT/"part_a/outputs/a3"/row["run_directory"].replace("\\","/")
        rows=history(directory);stopping([r["validation_rmse"] for r in rows]);runs.append(directory)
    groups = {variant:[r for r in records2 if r["variant"]==variant] for variant in ("baseline","continuous_only","sigmoid")}
    assert len(records2)==15 and len(records3)==5
    comparisons=[(a2["variants"][key],rows) for key,rows in groups.items()]+[(a3["variants"]["wide_deep"],records3)]
    for summary,rows in comparisons:
        assert {r["seed"] for r in rows} == {42,43,44,45,46} and len(rows)==5
        for metric in ("rmse","r2"):
            values=np.array([r[metric] for r in rows]); close(values.mean(),summary["metrics"][metric]["mean"]);close(values.std(ddof=1),summary["metrics"][metric]["sample_std"])
        for row in rows:
            prefix="part_a/outputs/a2" if "variant" in row else "part_a/outputs/a3"
            directory=ROOT/prefix/row["run_directory"].replace("\\","/")
            saved=read((directory/"test_metrics.json").relative_to(ROOT))
            assert len(history(directory))==row["epochs_completed"]==row["checkpoint_epoch"]==4 and row["test_count"]==26702
            assert read((directory/"config.json").relative_to(ROOT))["training"]["mode"]=="fixed_refit"
            for metric in ("rmse","r2"):close(saved["metrics"][metric],row[metric])
            runs.append(directory)
    assert a3["variants"]["baseline"]==a2["variants"]["baseline"]
    for seed in range(42,47):
        baseline=next(r for r in groups["baseline"] if r["seed"]==seed)
        sigmoid=next(r for r in groups["sigmoid"] if r["seed"]==seed)
        wide=next(r for r in records3 if r["seed"]==seed)
        assert baseline["initial_model_sha256"]==sigmoid["initial_model_sha256"]==wide["baseline_initial_sha256"]
        assert baseline["train_loader_generator_sha256"]==sigmoid["train_loader_generator_sha256"]==wide["train_loader_generator_sha256"]
    checks["A2"]="All 15 fixed-budget records, one final test each, matched controls and sample SDs"
    diagnostics = read("part_a/outputs/a2/diagnostics/summary.json")
    assert diagnostics["completed_runs"] == 3 and diagnostics["epoch_budget"] == 4
    assert diagnostics["training_rows"] == 87370 and diagnostics["validation_rows"] == 29057
    assert diagnostics["seed"] == 42 and not diagnostics["test_evaluated"] and not diagnostics["final_results_changed"]
    for record in diagnostics["records"]:
        directory = ROOT / "part_a/outputs/a2/diagnostics" / record["run_directory"]
        rows = history(directory)
        assert len(rows) == 4
        assert all(all(f"{split}_{metric}" in row for split in ("train", "validation") for metric in ("rmse", "r2")) for row in rows)
        assert not (directory / "test_metrics.json").exists()
        runs.append(directory)
    checks["A2_epoch_curves"] = "Three original-split seed-42 fixed-four-epoch diagnostic histories; genuine train/validation RMSE/R2; no test scoring"
    checks["A3"]="Five lambdas, five final seeds, reused paired baseline and sample SDs"

    a4=read("part_a/outputs/a4/summary.json")
    for name,record in a4["models"].items():
        path=ROOT/"part_a/outputs/a4"/record["attributions"]
        assert fingerprint(path)==record["attribution_sha256"]
        payload=torch.load(path,map_location="cpu",weights_only=True)
        matrix=payload["signed_attributions_sgd"]
        assert matrix.shape==(26702,10) and payload["seed"]==42 and payload["checkpoint_epoch"]==4 and payload["output_unit"]=="SGD"
        assert torch.equal(payload["row_ids"],refit.row_ids["test"])
        scores=matrix.abs().mean(0)
        ordered=sorted(record["ranked_importance"],key=lambda r:r["rank"])
        assert len(ordered)==10 and [r["mean_absolute_attribution_sgd"] for r in ordered]==sorted([r["mean_absolute_attribution_sgd"] for r in ordered],reverse=True)
        for row in ordered:close(scores[payload["feature_names"].index(row["feature"])],row["mean_absolute_attribution_sgd"])
    checks["A4"]="Full 26702x10 SGD matrices, matching seed/epoch/rows, independent mean-absolute aggregation and descending ranks"

    bundle=xray_bundle(ROOT/"part_b/outputs/data")
    with np.load(ROOT/"chestxray_binary_64.npz",allow_pickle=False) as archive:
        scaled=archive["train_images"].astype(np.float32)/np.float32(255.)
        close(scaled.mean(dtype=np.float64),bundle.preprocessing.mean);close(scaled.std(dtype=np.float64),bundle.preprocessing.std)
        for split,prefix in (("train","train"),("validation","val"),("test","test")):
            raw=archive[prefix+"_images"];labels=archive[prefix+"_labels"]
            torch.testing.assert_close(torch.from_numpy(raw),bundle.raw_images[split],rtol=0,atol=0)
            torch.testing.assert_close(torch.tensor(labels,dtype=torch.float32),bundle.datasets[split].tensors[1],rtol=0,atol=0)
            expected=(raw.astype(np.float32)/np.float32(255.)-np.float32(bundle.preprocessing.mean))/np.float32(bundle.preprocessing.std)
            torch.testing.assert_close(torch.from_numpy(expected[:,None]),bundle.datasets[split].tensors[0],rtol=0,atol=0)
    checks["xray_data"]="Original partitions/pixels/labels preserved; training-only normalization independently reproduced"

    for key,expected in (("b1",99569),("b2",18185),("b3",77169)):
        model=build_model(key)
        assert sum(p.numel() for p in model.parameters() if p.requires_grad)==expected
        assert all(m.bias is None for m in model.modules() if isinstance(m,nn.Conv2d))
        assert all(m.bias is not None for m in model.modules() if isinstance(m,nn.Linear))
        summary=read(f"part_b/outputs/{key}/summary.json")
        assert summary["test_evaluations"]==1 and summary["refit"] is True
        refit_audit = audit_final_refit(ROOT / f"part_b/outputs/{key}")
        assert refit_audit["combined_training_rows"] == 5232
        assert refit_audit["epochs"] == {"b1": 9, "b2": 13, "b3": 9}[key]
        runs.append(ROOT / f"part_b/outputs/{key}/refit")
        if key in ("b1","b2"):
            rows=read(f"part_b/outputs/{key}/"+("learning_rate_search.json" if key=="b1" else "batch_size_search.json"))
            field="learning_rate" if key=="b1" else "batch_size"
            assert {r[field] for r in rows}==({1e-5,1e-4,1e-3,1e-2,1e-1} if key=="b1" else {16,32,64,128,256}) and len(rows)==5
            assert min(rows,key=lambda r:(r["validation_classification_error"],r["validation_loss"],r["best_epoch"],r[field]))==summary["selected"]
        else:rows=[summary["selected"]]
        for row in rows:
            directory=ROOT/f"part_b/outputs/{key}"/row["run_directory"].replace("\\","/")
            epoch_rows=history(directory);stopping([r["validation_loss"] for r in epoch_rows])
            assert min(epoch_rows,key=lambda r:(r["validation_classification_error"],r["validation_loss"],r["epoch"]))["epoch"]==row["best_epoch"]
            config=read((directory/"config.json").relative_to(ROOT));assert config["training"]["max_epochs"]==20 and config["training"]["patience"]==4 and config["training"]["min_delta"]==1e-4
            assert config["training"]["seed"]==42 and config["training"]["mode"]=="selection"
            runs.append(directory)
        payload=torch.load(ROOT/f"part_b/outputs/{key}/final_test_predictions.pt",map_location="cpu",weights_only=True)
        labels=payload["labels"].bool();positive=payload["probabilities"]>=.5
        assert len(labels)==624 and torch.equal(payload["row_ids"],torch.arange(624))
        counts={"tp":int((positive&labels).sum()),"tn":int((~positive&~labels).sum()),"fp":int((positive&~labels).sum()),"fn":int((~positive&labels).sum())}
        metrics=summary["final_test"]
        for name,value in counts.items():assert metrics[name]==value
        close(metrics["accuracy"],(counts["tp"]+counts["tn"])/624)
        close(metrics["classification_error"],(counts["fp"]+counts["fn"])/624)
        close(metrics["sensitivity"],counts["tp"]/(counts["tp"]+counts["fn"]));close(metrics["specificity"],counts["tn"]/(counts["tn"]+counts["fp"]))
        independent_loss=functional.binary_cross_entropy_with_logits(payload["logits"].double(),payload["labels"].double()).item()
        assert abs(independent_loss-metrics["loss"])<1e-6
        frozen = read(f"part_b/outputs/{key}/frozen_{key}_settings.json")
        model.load_state_dict(torch.load(ROOT/f"part_b/outputs/{key}"/frozen["final_checkpoint"].replace("\\","/"),map_location="cpu",weights_only=True)["model_state"])
        checks[key.upper()]="Validation selection retained; fresh combined-data fixed-epoch refit; one final test record; saved-prediction confusion metrics/BCE"
    assert read("part_b/outputs/b1/summary.json")["selected"]["learning_rate"]==read("part_b/outputs/b2/summary.json")["learning_rate"]==read("part_b/outputs/b3/summary.json")["learning_rate"]==.001
    assert read("part_b/outputs/b2/summary.json")["selected"]["batch_size"]==read("part_b/outputs/b3/summary.json")["batch_size"]==16
    for key,expected in (("b1",39063616),("b2",19464256),("b3",25325696)):
        complexity=read(f"part_b/outputs/b3/complexity/{key}.json")
        total=0
        for layer in complexity["layers"]:
            if layer["type"]=="Conv2d":total+=2*math.prod(layer["output_shape"])*(layer["in_channels"]//layer["groups"])*math.prod(layer["kernel_size"])
            elif layer["type"]=="Linear":total+=2*math.prod(layer["output_shape"])*layer["in_features"]
        if key=="b3":total+=28672
        assert total==complexity["inference_flops"]==expected
    checks["B3_complexity"]="Executed-shape convolution/FC formulas plus residual additions reproduce all FLOPs"

    b4=read("part_b/outputs/b4/summary.json")
    payload=torch.load(ROOT/"part_b/outputs/b4/gradcam.pt",map_location="cpu",weights_only=True)
    ids=np.random.default_rng(42).choice(np.flatnonzero(bundle.datasets["test"].tensors[1].numpy()==1),5,replace=False).tolist()
    assert ids==b4["sample_row_ids"]==payload["row_ids"].tolist()==[486,279,413,53,276]
    assert b4["overlay_count"]==15 and b4["full_test_performance_evaluations"]==0
    torch.testing.assert_close(payload["raw_images"],bundle.raw_images["test"][ids],rtol=0,atol=0)
    for key,record in payload["models"].items():
        frozen = read(f"part_b/outputs/{key}/frozen_{key}_settings.json")
        assert record["checkpoint_sha256"] == frozen["final_checkpoint_sha256"]
        assert record["model_state_sha256_before"]==record["model_state_sha256_after"] and record["relu_attributions"]
        # Reuse stored tensors only, on their original arithmetic backend and
        # batch shape. CPU/GPU reductions differ by float32 rounding; matching
        # the saved CUDA shape keeps every comparison exact without inference.
        count=len(record["activations"]); batch=max(record["forward_batch_sizes"])
        repeats=(batch+count-1)//count
        gradients=record["gradients"].repeat(repeats,1,1,1)[:batch].cuda()
        activations=record["activations"].repeat(repeats,1,1,1)[:batch].cuda()
        weights=gradients.mean((2,3),keepdim=True)
        torch.testing.assert_close(weights[:count].cpu(),record["channel_weights"],rtol=0,atol=0)
        coarse=(weights*activations).sum(1,keepdim=True)
        torch.testing.assert_close(coarse[:count].cpu(),record["signed_coarse_maps"],rtol=0,atol=0)
        torch.testing.assert_close(coarse.relu()[:count].cpu(),record["positive_coarse_maps"],rtol=0,atol=0)
        full=functional.interpolate(coarse[:count].relu(),(64,64),mode="bilinear",align_corners=False)
        torch.testing.assert_close(full[:count].cpu(),record["upsampled_positive_maps"],rtol=0,atol=0)
        peaks=full.flatten(1).amax(1).reshape(-1,1,1,1)
        normalized=full/torch.where(peaks>0,peaks,torch.ones_like(peaks))
        torch.testing.assert_close(normalized[:count].cpu(),record["normalized_maps"],rtol=0,atol=0)
        torch.testing.assert_close(record["logits"],record["saved_final_test_logits"],rtol=0,atol=0)
        torch.testing.assert_close(record["probabilities"],record["saved_final_test_probabilities"],rtol=0,atol=0)
    checks["B4"]="Same five independently sampled positives; 15 stored gradient/map reconstructions; exact saved logits and unchanged states"

    # The FAQ deliberately replaces final B records and A1/A2 exported plotting corrections;
    # preserve the original selection runs, preparation and Part A final scores.
    prior=read("artifacts/faq_corrections/baseline_snapshot.json")["experiment_outputs"]
    unchanged, corrected = [], []
    for relative, expected in prior.items():
        parts = Path(relative).parts
        allowed = ((len(parts) >= 3 and parts[:2] == ("part_b", "outputs") and parts[2] == "b4")
                   or (len(parts) >= 4 and parts[:2] == ("part_b", "outputs") and parts[2] in {"b1", "b2", "b3"}
                       and (len(parts) == 4 or parts[3] in {"figures", "tables", "complexity"}))
                   or (len(parts) >= 4 and parts[:3] in {("part_a", "outputs", "a1"), ("part_a", "outputs", "a2")} and parts[3] in {"analysis.md", "figures"}))
        if fingerprint(ROOT/relative) != expected:
            assert allowed, relative
            corrected.append(relative)
        else:
            unchanged.append(relative)
    report={"review_date":"2026-10-11","status":"passed_current_faq_with_identity_and_portal_pending",
            "checks":checks,"training_run_histories_checked":len(runs),"protected_original_outputs_unchanged":len(unchanged),
            "intentional_faq_replacements": corrected,
            "new_training_runs":0,"new_model_inference":0,
            "correction":"Fresh combined-data B1/B2/B3 refits, final-model B4 and original-split A2/A3 epoch evidence",
            "confirmed_by_faq":["label 0 healthy / 1 pneumonia", "B1/B2 raw final logits", "B1 hidden-FC dropout", "B2 no dropout", "strided padding", "A1 refitted preprocessing"],
            "assumptions":["B3 raw final logit uses the shared binary-loss interpretation; FAQ explicitly names B1/B2", "A3 final refits use A1 E*=4 budget for paired comparison", "Part B retains original training-only pixel statistics, following the brief"],
            "pending":["actual author name","NTULearn submission and receipt"]}
    destination=ROOT/"artifacts/faq_corrections/recheck_verification.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
