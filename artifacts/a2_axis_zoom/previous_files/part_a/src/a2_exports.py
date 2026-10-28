"""A2 mean/sample-SD tables and two separate final-result ablation figures."""

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

from part_a.src.ablations import VARIANTS, VARIANT_LABELS


def _comparison_figure(baseline, ablation, variant, summaries, path, epoch_budget):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    colors = plt.cm.viridis(np.linspace(.1, .85, 5))
    labels = [f"Baseline (ReLU)\n{baseline[0]['trainable_parameters']:,} parameters",
              f"{'Continuous only' if variant == 'continuous_only' else 'Hidden Sigmoid'}\n{ablation[0]['trainable_parameters']:,} parameters"]
    for axis, metric, ylabel in zip(axes, ("rmse", "r2"), ("Test RMSE (SGD) — lower is better", "Test R² — higher is better")):
        means = [summaries[name]["metrics"][metric]["mean"] for name in ("baseline", variant)]
        stds = [summaries[name]["metrics"][metric]["sample_std"] for name in ("baseline", variant)]
        axis.bar([0, 1], means, color=["#488ac7", "#d19a4a"], alpha=.23, width=.6, zorder=1)
        axis.errorbar([0, 1], means, yerr=stds, fmt="D", color="black", markersize=6,
                       capsize=6, linewidth=1.7, zorder=4, label="Mean ± sample SD")
        for index, (first, second) in enumerate(zip(baseline, ablation)):
            offset = (index - 2) * .035
            axis.plot([offset, 1 + offset], [first[metric], second[metric]], "o-", color=colors[index],
                       markersize=4.5, linewidth=.9, alpha=.8, zorder=3, label=f"Seed {first['seed']}")
        axis.set(xticks=[0, 1], xticklabels=labels, ylabel=ylabel, xlim=(-.45, 1.45))
        axis.grid(axis="y", alpha=.22, zorder=0)
        if metric == "rmse":
            axis.set_ylim(bottom=0)
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        else:
            minimum = min([row[metric] for row in baseline + ablation] + [mean - std for mean, std in zip(means, stds)])
            axis.set_ylim(bottom=min(0, minimum - .04))
    title = "Categorical-input ablation" if variant == "continuous_only" else "Hidden-activation ablation"
    figure.suptitle(f"A2: {title} — five paired seeds, fixed {epoch_budget}-epoch refits", fontsize=12)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="lower center", ncol=6, frameon=False, fontsize=8)
    figure.tight_layout(rect=(0, .08, 1, .93))
    figure.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def export_a2(directory):
    output = Path(directory)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "completed":
        raise ValueError("Export requires completed A2 results")
    records = json.loads((output / "per_seed_results.json").read_text(encoding="utf-8"))
    summaries = summary["variants"]
    tables, figures = output / "tables", output / "figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    pd.DataFrame(records).sort_values(["variant", "seed"]).to_csv(tables / "per_seed_results.csv", index=False)
    metric_rows = []
    for variant in VARIANTS:
        values = summaries[variant]
        model = values["per_seed"][0]
        metric_rows.append({"variant": variant, "label": VARIANT_LABELS[variant], "seeds": 5,
                            "epochs": summary["epoch_budget"], "parameters": model["trainable_parameters"],
                            "test_rmse_mean_sgd": values["metrics"]["rmse"]["mean"],
                            "test_rmse_sample_std_sgd": values["metrics"]["rmse"]["sample_std"],
                            "test_r2_mean": values["metrics"]["r2"]["mean"],
                            "test_r2_sample_std": values["metrics"]["r2"]["sample_std"]})
    pd.DataFrame(metric_rows).to_csv(tables / "summary_metrics.csv", index=False)
    paired_rows = [{"ablation": variant, **row}
                   for variant, paired in summary["paired_differences"].items() for row in paired["per_seed"]]
    pd.DataFrame(paired_rows).to_csv(tables / "paired_differences.csv", index=False)
    baseline = summaries["baseline"]["per_seed"]
    for variant, name in (("continuous_only", "categorical_ablation"), ("sigmoid", "activation_ablation")):
        _comparison_figure(baseline, summaries[variant]["per_seed"], variant, summaries, figures / name, summary["epoch_budget"])

    comparison_text = {}
    for variant in VARIANTS[1:]:
        paired = summary["paired_differences"][variant]
        mean_delta = paired["metrics"]["rmse_delta"]["mean"]
        r2_delta = paired["metrics"]["r2_delta"]["mean"]
        improvements = sum(row["rmse_delta"] < 0 for row in paired["per_seed"])
        relative = abs(mean_delta) / summaries["baseline"]["metrics"]["rmse"]["mean"] * 100
        comparison_text[variant] = (
            f"{VARIANT_LABELS[variant]} had mean test RMSE SGD {abs(mean_delta):,.2f} "
            f"{'lower' if mean_delta < 0 else 'higher'} than baseline ({relative:.2f}%). "
            f"Its mean paired R² change was {r2_delta:+.6f}; RMSE improved in {improvements}/5 paired seeds. "
            f"The sample SD of the paired RMSE difference was SGD {paired['metrics']['rmse_delta']['sample_std']:,.2f}."
        )
    table_text = "\n".join(
        f"| {row['label']} | {row['parameters']:,} | {row['test_rmse_mean_sgd']:,.2f} ± {row['test_rmse_sample_std_sgd']:,.2f} | "
        f"{row['test_r2_mean']:.6f} ± {row['test_r2_sample_std']:.6f} |" for row in metric_rows)
    frozen = summary["frozen_a1_settings"]
    baseline_params = metric_rows[0]["parameters"]
    continuous_params = metric_rows[1]["parameters"]
    text = f"""# A2 results and discussion draft

This material is for the later Part A report. All results come from 15 completed saved runs.

## Controlled protocol

The A1-frozen hidden width {frozen['hidden_width']}, embedding dimension {frozen['embedding_dimension']} where applicable, Adam learning rate 0.01, batch size 512, and **E*={summary['epoch_budget']}** were used unchanged. Each model was initialized fresh for each of seeds 42,43,44,45,46 and trained on the same 116,427 rows from 2017–2021. The exact A1 combined-data encoders/scalers were reused without fitting again. Model loss was standardized-target MSE, while all RMSE values below are in SGD.

Every run completed exactly {summary['epoch_budget']} epochs, saved the final epoch, and received one final evaluation on the same 26,702 test rows from 2022. No validation/test monitoring loader was supplied during A2 training. The fresh seed-42 baseline was trained independently of A1's longer test-monitored run. The common epoch budget retains its origin in A1's professor-authorized test-assisted stopping.

All variants kept LayerNorm. The baseline and Sigmoid model use all ten inputs; only the hidden ReLU becomes Sigmoid, with a linear standardized-price output retained. Their initial parameter tensors are identical for every matched seed. Continuous-only has no embedding tables and a six-input first linear layer. Shuffling uses a separate, same-seed loader generator, so each matched seed sees the same training-row order across variants. Different continuous-only matrix shapes and parameter counts also change its random initialization; its tensors cannot be matched one-for-one to the all-input models.

## Mean and sample standard deviation

Each row summarizes five independent initialization/training seeds. Standard deviations use denominator 5−1 (`ddof=1`). These are seed-variability summaries, not confidence intervals or standard errors; each seed's R² is computed over the complete test partition before averaging.

| Model | Parameters | Test RMSE, SGD: mean ± sample SD | Test R²: mean ± sample SD |
| --- | ---: | ---: | ---: |
{table_text}

## Categorical-input ablation

{comparison_text['continuous_only']}

This comparison measures the observed contribution of categorical inputs under this fixed training recipe. Removing them also reduces the parameter count from {baseline_params:,} to {continuous_params:,} ({(baseline_params-continuous_params)/baseline_params*100:.2f}% fewer), so information loss and capacity changes are confounded. Town, flat model type, month, and storey range can carry predictive associations beyond the six continuous variables, but this experiment neither isolates their individual importance nor identifies causal price effects. A continuous-only model retuned independently could behave differently; retuning was deliberately outside the controlled comparison.

The separate `figures/categorical_ablation` figure shows baseline versus continuous-only, with RMSE and R² panels, mean ± sample SD, and connected matched-seed observations.

## Hidden-activation ablation

{comparison_text['sigmoid']}

This comparison isolates the hidden activation while preserving all inputs, parameter count, initial weights, preprocessing, batch order, optimizer settings, and epoch budget. Sigmoid is bounded, with derivative σ(x)(1−σ(x)) ≤ 1/4; gradients approach zero at extreme inputs. Saturation can slow optimization, while ReLU retains a unit derivative for positive inputs. LayerNorm affects the preactivation distribution. Saturation and gradient statistics were not measured here, so they are plausible mechanisms rather than demonstrated causes of the observed result. Four epochs and hyperparameters selected for the ReLU model may also affect relative optimization speed; either activation could win under another recipe or seed.

The separate `figures/activation_ablation` figure shows baseline versus Sigmoid, covering both final test metrics. No test-metric epoch plot is produced.

## Interpretation limits and next step

Five matched seeds quantify variability, and paired differences provide descriptive direction. No significance claim is made from these five averages. Chronological property/price-distribution changes can also affect generalization. Test results describe the requested comparisons; they were not used to retune architectures, learning rate, or stopping epochs.

Use this A2 **five-seed baseline** for the later A3 comparison. A1's individual final result remains separately identified. Preserve H={frozen['hidden_width']}, d={frozen['embedding_dimension']}, learning rate 0.01, batch size 512, the same refit preprocessing, and fixed E*={summary['epoch_budget']} for A3's controlled final refits.

## Saved evidence

- `tables/per_seed_results.csv`: all 15 results, parameter/epoch counts, initial-state fingerprints, loader-generator fingerprints, and checkpoint paths.
- `tables/summary_metrics.csv`: the required three-model mean/sample-SD table.
- `tables/paired_differences.csv`: same-seed ablation-minus-baseline differences.
- `figures/categorical_ablation.{{png,pdf}}` and `figures/activation_ablation.{{png,pdf}}`: separately presented final comparisons.
- `runs/<variant>/seed_<seed>/`: configuration, preprocessing, training history, final checkpoint, and single final-test record.
- `configuration.json`: frozen A1 source/settings/preprocessing provenance and this run's environment.

Activation definition: [PyTorch 2.11 Sigmoid documentation](https://docs.pytorch.org/docs/2.11/generated/torch.nn.Sigmoid.html). The derivative statement follows directly from that definition.
"""
    (output / "analysis.md").write_text(text, encoding="utf-8")
