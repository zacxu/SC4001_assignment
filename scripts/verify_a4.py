"""Audit full-test A4 coverage, SGD aggregation, references, and manual differences."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"artifacts/cache/matplotlib"))

import numpy as np
import pandas as pd
import torch

from assignment_common import restore_checkpoint
from assignment_common.artifacts import write_json
from part_a.src.a2 import check_same_preprocessing, model_state_fingerprint, sha256_file
from part_a.src.a4 import (
    FEATURE_NAMES, MODEL_NAMES, make_references, manual_ablation_differences,
    price_forward, rank_importance, validate_checkpoint_pair,
)
from part_a.src.data import load_data_bundle
from part_a.src.model import HousingMLP
from part_a.src.wide_deep import WideDeepHousingModel


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"part_a/outputs/a4")
    output=parser.parse_args().output.resolve()
    config=json.loads((output/"configuration.json").read_text())
    summary=json.loads((output/"summary.json").read_text())
    assert summary["status"]==json.loads((output/"status.json").read_text())["status"]=="completed"
    assert config["seed"]==42 and config["checkpoint_epoch"]==4
    assert config["batch_size"]==512 and config["perturbations_per_eval"]==1
    assert config["output_unit"]==summary["output_unit"]=="SGD"
    assert config["feature_names"]==summary["features"]==list(FEATURE_NAMES)
    assert config["feature_masks"]==[list(range(4)),list(range(4,10))]
    prepared=Path(config["prepared_directory"])
    assert sha256_file(prepared/"refit/datasets.pt")==config["prepared_refit_tensors_sha256"]
    bundle=load_data_bundle(prepared/"refit",expected_protocol="refit")
    assert bundle.source_sha256==config["source_sha256"]
    check_same_preprocessing(torch.load(output/"preprocessing.pt",map_location="cpu",weights_only=True),bundle.preprocessing.metadata())
    references=torch.load(output/"references.pt",map_location="cpu",weights_only=True)
    expected=make_references(bundle.preprocessing,bundle.datasets["train"].tensors[0])
    for key in ("categorical","continuous"):
        torch.testing.assert_close(references[key],expected[key],rtol=0,atol=0)
    for key in ("feature_names","fit_years","fit_row_count","details"):
        assert references[key]==expected[key]
    refs_json=json.loads((output/"references.json").read_text())
    assert refs_json["details"]==references["details"] and refs_json["fit_row_count"]==116427
    assert refs_json["categorical"]==references["categorical"].tolist()
    assert refs_json["continuous"]==references["continuous"].tolist()
    paths={name:Path(config["checkpoints"][name]["path"]) for name in MODEL_NAMES}
    payloads={name:torch.load(path,map_location="cpu",weights_only=True) for name,path in paths.items()}
    validate_checkpoint_pair(payloads["baseline"],payloads["wide_deep"],bundle.preprocessing,
                             config["a1_frozen_settings"],config["a3_frozen_settings"])
    assert all(sha256_file(path)==config["checkpoints"][name]["sha256"] for name,path in paths.items())
    a1,a3=Path(config["a1_directory"]),Path(config["a3_directory"])
    assert json.loads((a1/"frozen_a1_settings.json").read_text())==config["a1_frozen_settings"]
    assert json.loads((a3/"frozen_a3_settings.json").read_text())==config["a3_frozen_settings"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce A4's manual SGD differences")
    torch.set_num_threads(4)
    cat,cont,_=bundle.datasets["test"].tensors
    assert len(cat)==summary["test_rows"]==config["test_rows"]==26702
    manual_records=json.loads((output/"manual_checks.json").read_text())
    models_audited=[]
    all_scores={}
    for name in MODEL_NAMES:
        result=summary["models"][name]
        attr_path=output/result["attributions"]
        assert sha256_file(attr_path)==result["attribution_sha256"]
        saved=torch.load(attr_path,map_location="cpu",weights_only=True)
        attrs,predictions=saved["signed_attributions_sgd"],saved["predictions_sgd"]
        assert saved["format_version"]==1 and saved["model"]==name and saved["seed"]==42
        assert saved["checkpoint_epoch"]==4 and saved["checkpoint_sha256"]==config["checkpoints"][name]["sha256"]
        assert saved["feature_names"]==list(FEATURE_NAMES) and saved["output_unit"]=="SGD"
        assert attrs.shape==(26702,10) and attrs.dtype==torch.float64 and torch.isfinite(attrs).all()
        assert predictions.shape==(26702,) and predictions.dtype==torch.float64 and torch.isfinite(predictions).all()
        assert torch.equal(saved["row_ids"],bundle.row_ids["test"])
        assert len(torch.unique(saved["row_ids"]))==26702
        # Recompute means independently from all rows, including the partial tail.
        scores=np.abs(attrs.numpy()).mean(axis=0)
        rows=rank_importance(torch.from_numpy(scores))
        for measured,reported in zip(rows,result["ranked_importance"]):
            assert measured["feature"]==reported["feature"] and measured["rank"]==reported["rank"]
            assert math.isclose(measured["mean_absolute_attribution_sgd"],reported["mean_absolute_attribution_sgd"],rel_tol=1e-12,abs_tol=1e-8)
        assert result["top_three"]==result["ranked_importance"][:3]
        assert result["test_rows"]==26702 and result["features"]==10
        table=pd.read_csv(output/"tables"/f"{name}_importance.csv")
        assert len(table)==10 and table["feature"].tolist()==[row["feature"] for row in rows]
        np.testing.assert_allclose(table["mean_absolute_attribution_sgd"], [row["mean_absolute_attribution_sgd"] for row in rows],rtol=1e-12,atol=1e-8)
        metadata=payloads[name]["model_config"]
        model=HousingMLP.from_metadata(metadata) if name=="baseline" else WideDeepHousingModel.from_metadata(metadata)
        restore_checkpoint(paths[name],model,device="cuda")
        original_state=model_state_fingerprint(model)
        assert original_state==result["model_state_sha256"]
        errors=[]
        for start in (0,(len(cat)-1)//512*512):
            stop=min(start+512,len(cat))
            cats,continuous=cat[start:stop].cuda(),cont[start:stop].cuda()
            manual=manual_ablation_differences(model,cats,continuous,references).cpu()
            torch.testing.assert_close(attrs[start:stop],manual,rtol=0,atol=1e-6)
            with torch.no_grad():
                restored_predictions=price_forward(model)(cats,continuous).cpu()
            torch.testing.assert_close(predictions[start:stop],restored_predictions,rtol=0,atol=1e-6)
            errors.append(float((attrs[start:stop]-manual).abs().max()))
        assert model_state_fingerprint(model)==original_state
        checked=manual_records[name]
        assert checked["manual_rows_checked"]==512 and checked["features_checked"]==10 and checked["model_state_unchanged"]
        assert checked["first_sample_source_row_id"]==int(bundle.row_ids["test"][0])
        np.testing.assert_allclose(checked["first_sample_signed_captum_sgd"],attrs[0].numpy(),rtol=0,atol=1e-6)
        np.testing.assert_allclose(checked["first_sample_manual_difference_sgd"],attrs[0].numpy(),rtol=0,atol=1e-6)
        for index in range(10):
            values=cat[:,index] if index<4 else cont[:,index-4]
            reference=references["categorical"][0,index] if index<4 else references["continuous"][0,index-4]
            same=values==reference
            if same.any():
                assert attrs[same,index].abs().max()<=1e-6
        for extension in ("png","pdf"):
            assert (output/"figures"/f"{name}_feature_importance.{extension}").stat().st_size>1000
        models_audited.append({"model":name,"rows":26702,"features":10,"first_and_last_batches_manual_max_error_sgd":max(errors),
            "model_state_unchanged":True,"attribution_sha256":sha256_file(attr_path)})
        all_scores[name]={row["feature"]:row for row in rows}
        print(f"Audited {name}: full matrix/row IDs, all ten scores, first/last CUDA manual differences.",flush=True)
        del model
    combined=pd.read_csv(output/"tables/global_importance.csv")
    assert len(combined)==20 and combined.groupby("model").size().to_dict()=={name:10 for name in MODEL_NAMES}
    comparison=pd.read_csv(output/"tables/feature_comparison.csv")
    assert comparison["feature"].tolist()==list(FEATURE_NAMES)
    for row in comparison.to_dict("records"):
        first,second=all_scores["baseline"][row["feature"]],all_scores["wide_deep"][row["feature"]]
        assert row["baseline_rank"]==first["rank"] and row["wide_deep_rank"]==second["rank"]
        for column,value in (("baseline_importance_sgd",first["mean_absolute_attribution_sgd"]),
                             ("wide_deep_importance_sgd",second["mean_absolute_attribution_sgd"]),
                             ("importance_delta_sgd",second["mean_absolute_attribution_sgd"]-first["mean_absolute_attribution_sgd"])):
            assert math.isclose(row[column],value,rel_tol=1e-12,abs_tol=1e-8)
    setup=json.loads((ROOT/"artifacts/setup/environment_check.json").read_text())
    for name,original in setup["files"].items():
        assert sha256_file(ROOT/name)==original["sha256"],f"Supplied file changed: {name}"
    a1_audit=json.loads((a1/"verification.json").read_text())
    for row in a1_audit["checkpoints"]:
        assert sha256_file(a1/f"grid/h{row['hidden_width']:03d}_d{row['embedding_dimension']:02d}/training/best_model.pt")==row["checkpoint_sha256"]
    assert sha256_file(a1/"refit/best_model.pt")==a1_audit["refit_checkpoint_sha256"]
    for name in ("a2","a3"):
        prior=ROOT/"part_a/outputs"/name
        for row in json.loads((prior/"verification.json").read_text())["checkpoints"]:
            relative=(f"runs/{row['variant']}/seed_{row['seed']}/best_model.pt" if name=="a2" else row["checkpoint"])
            assert sha256_file(prior/relative)==row["checkpoint_sha256"]
    evidence={"status":"passed","checked_at_singapore":datetime.now(ZoneInfo("Asia/Singapore")).isoformat(),
        "run_directory":str(output),"models":models_audited,"full_test_rows_per_model":26702,"features_per_model":10,
        "all_signed_attributions_finite_and_row_ids_aligned":True,"shared_training_only_references_verified":True,
        "same_seed_epoch_and_preprocessing_verified":True,"sgd_manual_first_and_last_batches_verified_on_cuda":True,
        "full_test_mean_absolute_aggregation_and_rankings_verified":True,"unchanged_reference_values_have_zero_effect":True,
        "model_weights_unchanged":True,"supplied_files_and_a1_a2_a3_checkpoints_unchanged":True,
        "separate_descending_plots_and_tables_present":True,"no_new_training_or_test_performance_scoring":True}
    evidence_dir=ROOT/"artifacts/a4_checks"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    evidence_dir.mkdir(parents=True,exist_ok=False)
    write_json(output/"verification.json",evidence)
    write_json(evidence_dir/"verification.json",evidence)
    write_json(ROOT/"artifacts/a4_checks/latest.json",{"status":"passed","directory":str(evidence_dir),"verification":str(evidence_dir/"verification.json")})
    print("A4 audit passed: both full-test ten-feature SGD matrices, references, manual checks, rankings, and unchanged prior files.")


if __name__=="__main__":
    main()
