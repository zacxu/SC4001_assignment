"""Ranked SGD importance plots, comparison tables, and A4 discussion."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "artifacts/cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from part_a.src.a4 import FEATURE_NAMES, MODEL_NAMES

DISPLAY_NAMES = {
    "month": "Month", "town": "Town", "flat_model_type": "Flat model type", "storey_range": "Storey range",
    "dist_to_nearest_stn": "Distance to nearest station", "dist_to_dhoby": "Distance to Dhoby Ghaut",
    "degree_centrality": "Degree centrality", "eigenvector_centrality": "Eigenvector centrality",
    "remaining_lease_years": "Remaining lease years", "floor_area_sqm": "Floor area (m²)",
}


def export_a4(directory):
    output=Path(directory)
    summary=json.loads((output/"summary.json").read_text(encoding="utf-8"))
    config=json.loads((output/"configuration.json").read_text(encoding="utf-8"))
    references=json.loads((output/"references.json").read_text(encoding="utf-8"))
    manual_checks=json.loads((output/"manual_checks.json").read_text(encoding="utf-8"))
    if summary["status"]!="completed":
        raise ValueError("A4 exports require completed full-test attributions")
    tables,figures=output/"tables",output/"figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    all_rows=[]
    common_max=max(row['mean_absolute_attribution_sgd'] for model in summary['models'].values() for row in model['ranked_importance'])
    for name in MODEL_NAMES:
        rows=summary["models"][name]["ranked_importance"]
        all_rows.extend({"model":name,**row} for row in rows)
        pd.DataFrame(rows).to_csv(tables/f"{name}_importance.csv",index=False)
        fig,axis=plt.subplots(figsize=(9,5))
        values=[row["mean_absolute_attribution_sgd"] for row in rows]
        axis.barh(range(10),values,color="#4b83b3" if name=="baseline" else "#b67940")
        axis.set(yticks=range(10),yticklabels=[DISPLAY_NAMES[row["feature"]] for row in rows],
                 xlabel="Mean absolute FeatureAblation attribution (SGD)")
        axis.invert_yaxis()
        axis.set_xlim(0,common_max*1.22 if common_max else 1.)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value,_:f"{value:,.0f}"))
        for index,value in enumerate(values):
            axis.text(value+common_max*.012,index,f"{value:,.2f}",va="center",fontsize=9)
        axis.grid(axis="x",alpha=.2)
        axis.set_axisbelow(True)
        label="A1 baseline" if name=="baseline" else f"A3 wide-and-deep, λ={config['a3_frozen_settings']['lambda']:g}"
        axis.set_title(f"{label}: seed 42, all {summary['test_rows']:,} test rows\nShared training-mode / training-mean references")
        fig.tight_layout()
        for extension in ("png","pdf"):
            fig.savefig(figures/f"{name}_feature_importance.{extension}",dpi=220,bbox_inches="tight")
        plt.close(fig)
    pd.DataFrame(all_rows).to_csv(tables/"global_importance.csv",index=False)
    pd.DataFrame(references["details"]).to_csv(tables/"reference_values.csv",index=False)
    by_model={name:{row["feature"]:row for row in summary["models"][name]["ranked_importance"]} for name in MODEL_NAMES}
    comparison=[]
    for feature in FEATURE_NAMES:
        base,wide=by_model["baseline"][feature],by_model["wide_deep"][feature]
        delta=wide["mean_absolute_attribution_sgd"]-base["mean_absolute_attribution_sgd"]
        comparison.append({"feature":feature,"baseline_rank":base["rank"],"wide_deep_rank":wide["rank"],
            "baseline_importance_sgd":base["mean_absolute_attribution_sgd"],
            "wide_deep_importance_sgd":wide["mean_absolute_attribution_sgd"],"importance_delta_sgd":delta,
            "importance_delta_percent":delta/base["mean_absolute_attribution_sgd"]*100 if base["mean_absolute_attribution_sgd"] else None})
    pd.DataFrame(comparison).to_csv(tables/"feature_comparison.csv",index=False)
    table="\n".join(f"| {DISPLAY_NAMES[row['feature']]} | {row['baseline_rank']} | {row['baseline_importance_sgd']:,.2f} | {row['wide_deep_rank']} | {row['wide_deep_importance_sgd']:,.2f} | {row['importance_delta_sgd']:+,.2f} |" for row in comparison)
    top_text={name:"; ".join(f"**{DISPLAY_NAMES[row['feature']]}** (SGD {row['mean_absolute_attribution_sgd']:,.2f})" for row in summary["models"][name]["top_three"]) for name in MODEL_NAMES}
    base_top={row["feature"] for row in summary["models"]["baseline"]["top_three"]}
    wide_top={row["feature"] for row in summary["models"]["wide_deep"]["top_three"]}
    common=sorted(base_top&wide_top,key=lambda feature:by_model["baseline"][feature]["rank"])
    common_text=", ".join(DISPLAY_NAMES[feature] for feature in common) if common else "none"
    largest_rank_change=max(abs(row["wide_deep_rank"]-row["baseline_rank"]) for row in comparison)
    changed=[row for row in comparison if abs(row["wide_deep_rank"]-row["baseline_rank"])==largest_rank_change]
    rank_change_text=(f"Ranks changed by at most {largest_rank_change} place(s): "+"; ".join(
        f"{DISPLAY_NAMES[row['feature']]} from {row['baseline_rank']} to {row['wide_deep_rank']}" for row in changed)+"."
        if largest_rank_change else "All ten feature ranks are unchanged.")
    magnitude_change=max(comparison,key=lambda row:abs(row["importance_delta_sgd"]))
    # Average tied ranks, without a SciPy dependency; undefined for constant scores.
    base_ranks=pd.Series([row["baseline_importance_sgd"] for row in comparison]).rank().to_numpy()
    wide_ranks=pd.Series([row["wide_deep_importance_sgd"] for row in comparison]).rank().to_numpy()
    rho=float(np.corrcoef(base_ranks,wide_ranks)[0,1]) if np.std(base_ranks) and np.std(wide_ranks) else None
    rho_text=f"{rho:.4f}" if rho is not None else "undefined (constant scores)"
    ref_table="\n".join(f"| {DISPLAY_NAMES[row['feature']]} | {row['reference_raw']} | {row['reference_encoded']} |" for row in references["details"])
    manual_table="\n".join(f"| {'Baseline' if name=='baseline' else 'Wide-and-deep'} | {manual_checks[name]['first_sample_source_row_id']} | {manual_checks[name]['first_sample_signed_captum_sgd'][9]:+,.2f} | {manual_checks[name]['first_sample_manual_difference_sgd'][9]:+,.2f} |" for name in MODEL_NAMES)
    text=f"""# A4 FeatureAblation results and discussion draft

All ten original features were explained for **every one of the {summary['test_rows']:,} 2022 test rows**, for both final seed-42 models.

## Models, references, and SGD calculation

The A1 baseline and validation-selected A3 wide-and-deep model (lambda {config['a3_frozen_settings']['lambda']:g}) were restored from their final epoch-{config['checkpoint_epoch']} combined-data checkpoints. Both use the exact saved 2017–2021 encoders/scalers. Their weights were kept fixed during explanation.

Captum FeatureAblation replaces one input with its reference while retaining the other nine. Signed attribution is `price(original) - price(feature replaced)`. Categorical IDs are replaced before embedding lookup, so each entire categorical variable is one feature. Unique mask IDs 0–3 identify categorical columns and 4–9 identify continuous columns. No embedding coordinates are presented as additional features. The explained forward function returns one SGD prediction per row; target scaling is applied once after the model output. The network stays float32, with the final inverse transform and subtraction in float64 to reduce numerical cancellation.

References are shared across models: each categorical feature's mode among all 116,427 refit training rows (ties favor the smallest encoded ID), and each continuous feature's saved training mean, represented by standardized zero. No test data determine these reference values. [Captum's FeatureAblation documentation](https://captum.ai/api/feature_ablation.html) describes these replacement baselines and feature masks; the installed Captum {config['environment']['captum']} implementation was checked as well.

| Feature | Raw reference | Encoded / standardized reference |
| --- | --- | ---: |
{ref_table}

Global importance is **mean(abs(signed attribution)) over all {summary['test_rows']:,} test rows**, yielding exactly ten SGD scores per model. The absolute value is taken before averaging. Batches of 512 include the final partial batch, with one perturbation per forward evaluation. Full signed N×10 matrices, SGD predictions, and original CSV row IDs are saved for reproducible aggregation.

## Ranked results and top three

**Baseline top three:** {top_text['baseline']}.

**Wide-and-deep top three:** {top_text['wide_deep']}.

Each separate plot (`figures/baseline_feature_importance` and `figures/wide_deep_feature_importance`) shows all ten scores in descending order, with a shared SGD axis scale and numeric labels. PNG and vector PDF versions are saved.

| Feature | Baseline rank | Baseline importance (SGD) | Wide-and-deep rank | Wide-and-deep importance (SGD) | Change (SGD) |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

The models share **{len(common)}/3** top-three features: **{common_text}**. The full ten-feature rank correlation is **{rho_text}** (Spearman, average ranks for any ties). {rank_change_text} The largest absolute score change is **{DISPLAY_NAMES[magnitude_change['feature']]}**, {magnitude_change['importance_delta_sgd']:+,.2f} SGD.

These magnitudes describe how strongly each fitted model's predictions respond to replacement by the chosen reference. Shared prominent features suggest similar predictive reliance under this reference scheme. Changes in ranking or magnitude can arise from the active linear branch, changes in jointly trained embeddings, and changes in the MLP; this comparison does not isolate a branch-specific causal effect. A larger attribution does not by itself establish better prediction accuracy.

## Verification and limitations

For both models, all ten Captum columns in the first 512-row batch were checked against independent manual prediction differences, including a documented first-row example. This validates categorical grouping, sign, feature order, and SGD scaling. Model-state fingerprints were unchanged. Saved attribution matrices cover every test row in the prepared-data order.

For the first test sample's floor-area replacement, the signed results agree as follows (row IDs are original zero-based CSV indices). A negative value means replacement by the training mean increased this model's prediction.

| Model | Source row ID | Captum signed attribution (SGD) | Manual original-minus-replaced price (SGD) |
| --- | ---: | ---: | ---: |
{manual_table}

Reference choice affects results: a mode-category/mean-continuous replacement answers a particular model-sensitivity question and another valid reference could change scores. Replacing one correlated feature while leaving others fixed may create unrealistic property/location combinations, for example between floor area and flat model type or among station distance and centrality variables. Mean absolute scores hide the direction and distribution of effects; signed per-row values are retained. Because the network is nonlinear, the ten separate effects need not add up to the prediction difference from replacing all features together. These are model explanations, not causal effects on resale prices.

Only the requested paired seed-42 models are explained, so no across-seed attribution stability claim is made. The baseline checkpoint and common epoch budget retain A1's professor-authorized test-assisted origin. A4's required perturbed forward passes are explanation computations; the saved A1/A3 performance records are reused, and no settings are tuned from attribution results.

## Saved evidence

- `attributions/baseline.pt` and `attributions/wide_deep.pt`: full signed SGD matrices, predictions, row IDs, checkpoint hashes, and feature ordering (safe `weights_only=True` loading).
- `references.pt`, `references.json`, and `tables/reference_values.csv`: shared legal category IDs, raw categories/means, mode counts, and fitting provenance.
- `tables/global_importance.csv`, individual ranked tables, and `tables/feature_comparison.csv`: all global scores, ranks, and model differences.
- `manual_checks.json`: all first-batch coordinate checks and the first-sample attribution/manual-difference comparison.
- `configuration.json`, `summary.json`, and `progress.json`: frozen models, source/runtime provenance, full coverage, ranked summaries, and completion status.
"""
    (output/"analysis.md").write_text(text,encoding="utf-8")
