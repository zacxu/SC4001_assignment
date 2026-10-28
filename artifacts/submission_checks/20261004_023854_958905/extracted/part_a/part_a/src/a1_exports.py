"""Regenerate A1 report assets from saved experiment results, without training."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "artifacts/cache/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd


def _save(figure, path: Path):
    figure.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _learning_curve(history: pd.DataFrame, row: dict, title: str, path: Path):
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(history.epoch, history.train_rmse, "o-", markersize=3, label="Training")
    axis.plot(history.epoch, history.validation_rmse, "s-", markersize=3, label="Validation")
    axis.axvline(row["best_epoch"], color="0.4", linestyle="--", linewidth=1,
                 label=f"Selected epoch {row['best_epoch']}")
    axis.scatter([row["best_epoch"]], [row["validation_rmse"]], s=65, facecolors="none", edgecolors="black", zorder=5)
    axis.set(xlabel="Epoch", ylabel="RMSE (SGD)", title=title)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axis.grid(alpha=.25)
    axis.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    _save(figure, path)


def _baseline_diagram(path: Path):
    figure, axis = plt.subplots(figsize=(12.8, 6))
    axis.set(xlim=(0, 16), ylim=(0, 8))
    axis.axis("off")

    def box(x, y, width, height, text, color="#e7eef8", fontsize=9):
        axis.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=.06",
                                      linewidth=1, edgecolor="#38506c", facecolor=color))
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)

    def arrow(start, end):
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                                      linewidth=1.1, color="#38506c"))

    for name, size, y in (("Month", 13, 6.1), ("Town", 27, 4.9),
                          ("Flat model type", 44, 3.7), ("Storey range", 18, 2.5)):
        box(.3, y, 3.2, .8, f"{name}: category ID\nEmbedding({size}, 8) · {size * 8} parameters")
        arrow((3.56, y + .4), (5, 4))
    box(.3, 1.0, 3.2, .9, "Six continuous inputs\nStandardized using fitting rows", color="#e9f4ed")
    arrow((3.56, 1.45), (5, 4))
    box(5.05, 3.5, 2.1, 1, "Concatenate\n4 × 8 + 6 = 38")
    arrow((7.22, 4), (8.1, 6.5))
    box(8.15, 6.0, 3.4, 1, "Linear(38, 64)\n38 × 64 + 64 = 2,496")
    arrow((9.85, 5.94), (9.85, 5.55))
    box(8.15, 4.5, 3.4, 1, "LayerNorm(64) → ReLU\n64 scale + 64 bias = 128")
    arrow((9.85, 4.44), (9.85, 4.05))
    box(8.15, 3.0, 3.4, 1, "Linear(64, 1)\n64 + 1 = 65")
    arrow((9.85, 2.94), (9.85, 2.55))
    box(8.15, 1.5, 3.4, 1, "Standardized price ẑ\nMSE against standardized target", color="#fff0d7")
    arrow((11.62, 2), (12.45, 2))
    box(12.5, 1.35, 3.1, 1.3, "SGD prediction\nŷ = μᵧ + σᵧ ẑ\nRMSE / R² / explanations", color="#e9f4ed")
    axis.text(8, 7.6, "Supplied baseline: four independent embeddings + LayerNorm MLP",
              ha="center", fontsize=13, weight="bold")
    axis.text(8, .35, "Trainable parameters: 816 embeddings + 2,496 hidden linear + 128 LayerNorm + 65 output = 3,505",
              ha="center", fontsize=10)
    figure.tight_layout()
    _save(figure, path)


def export_a1(directory: Path | str, *, prepared_directory: Path | str | None = None) -> None:
    output = Path(directory)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "completed":
        raise ValueError("Export requires completed A1 results")
    rows = json.loads((output / "grid_results.json").read_text(encoding="utf-8"))
    selected, baseline = summary["selected"], summary["default_baseline"]
    figures, tables = output / "figures", output / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    grid = pd.DataFrame(rows)
    grid["selected"] = ((grid.hidden_width == selected["hidden_width"])
                        & (grid.embedding_dimension == selected["embedding_dimension"]))
    grid.to_csv(tables / "grid_results.csv", index=False)
    test = summary["final_test_metrics"]
    pd.DataFrame([{
        "model": "A1 selected combined-data refit", "seed": summary["seed"],
        "hidden_width": selected["hidden_width"], "embedding_dimension": selected["embedding_dimension"],
        "trainable_parameters": selected["trainable_parameters"], "E_val": summary["E_val"],
        "E_star": summary["E_star"], "refit_epochs_completed": summary["refit_epochs_completed"],
        "test_rows": test["count"], "test_rmse_sgd": test["rmse"], "test_r2": test["r2"],
        "test_assisted_stopping": True,
    }]).to_csv(tables / "final_test_metrics.csv", index=False)
    pd.DataFrame([
        {"name": name, "hidden_width": row["hidden_width"], "embedding_dimension": row["embedding_dimension"],
         "parameters": row["trainable_parameters"], "best_epoch": row["best_epoch"],
         "epochs_completed": row["epochs_completed"], "train_rmse_sgd": row["train_rmse"],
         "validation_rmse_sgd": row["validation_rmse"], "validation_r2": row["validation_r2"]}
        for name, row in (("Supplied default", baseline), ("Validation-selected", selected))
    ]).to_csv(tables / "baseline_selection_comparison.csv", index=False)

    for row, title, filename in (
        (selected, f"Selected MLP: H={selected['hidden_width']}, d={selected['embedding_dimension']}", "selected_learning_curve"),
        (baseline, "Supplied default MLP: H=64, d=8", "default_baseline_learning_curve"),
    ):
        history = pd.read_csv(output / row["run_directory"] / "history.csv")
        _learning_curve(history, row, title, figures / filename)
    pivot = grid.pivot(index="hidden_width", columns="embedding_dimension", values="validation_rmse")
    figure, axis = plt.subplots(figsize=(7, 4.8))
    values = pivot.to_numpy()
    heatmap = axis.imshow(values, cmap="viridis_r", aspect="auto")
    axis.set(xticks=np.arange(len(pivot.columns)), xticklabels=pivot.columns,
             yticks=np.arange(len(pivot.index)), yticklabels=pivot.index,
             xlabel="Embedding dimension d (same dimension, independent tables)", ylabel="Hidden width H",
             title="A1: minimum validation RMSE across each trial (SGD)")
    threshold = (values.min() + values.max()) / 2
    for y, x in np.ndindex(values.shape):
        axis.text(x, y, f"{values[y, x]:,.0f}", ha="center", va="center", fontsize=10,
                  color="white" if values[y, x] > threshold else "black")
    x = list(pivot.columns).index(selected["embedding_dimension"])
    y = list(pivot.index).index(selected["hidden_width"])
    axis.add_patch(Rectangle((x - .49, y - .49), .98, .98, fill=False, edgecolor="red", linewidth=2.5))
    colorbar = figure.colorbar(heatmap, ax=axis, pad=.03)
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    figure.tight_layout()
    _save(figure, figures / "validation_grid_heatmap")
    _baseline_diagram(figures / "baseline_architecture")

    refit_history = pd.read_csv(output / "refit/history.csv")
    figure, axis = plt.subplots(figsize=(7, 3.8))
    axis.plot(refit_history.epoch, refit_history.train_rmse, "o-", markersize=3, label="Combined training")
    axis.axvline(summary["E_star"], color="0.4", linestyle="--", label=f"Selected epoch E*={summary['E_star']}")
    axis.set(xlabel="Epoch", ylabel="RMSE (SGD)", title="A1 combined-data refit: training only")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(alpha=.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    _save(figure, figures / "refit_training_curve")

    history = pd.read_csv(output / selected["run_directory"] / "history.csv")
    last = history.iloc[-1]
    reduction = (baseline["validation_rmse"] - selected["validation_rmse"]) / baseline["validation_rmse"] * 100
    gap = selected["validation_rmse"] - selected["train_rmse"]
    top = sorted(rows, key=lambda row: (row["validation_rmse"], row["trainable_parameters"],
                                       row["hidden_width"], row["embedding_dimension"]))[:5]
    ranking = "\n".join(f"| {i} | {row['hidden_width']} | {row['embedding_dimension']} | {row['trainable_parameters']:,} | {row['best_epoch']} | {row['validation_rmse']:,.2f} |"
                        for i, row in enumerate(top, 1))
    late_behavior = (
        "At the last epoch, training RMSE was lower but validation RMSE was higher than at the selected epoch. "
        "This is consistent with overfitting after the selected checkpoint, though one noisy trajectory cannot establish its cause."
        if last.train_rmse < selected["train_rmse"] and last.validation_rmse > selected["validation_rmse"]
        else "The late training/validation trajectory does not show a simple monotonic overfitting pattern; inspect the saved curve rather than inferring it solely from model size."
    )
    narrow = min((row for row in rows if row["embedding_dimension"] == selected["embedding_dimension"]),
                 key=lambda row: row["hidden_width"])
    other_embedding = min((row for row in rows if row["hidden_width"] == selected["hidden_width"]
                          and row["embedding_dimension"] != selected["embedding_dimension"]),
                         key=lambda row: row["validation_rmse"])
    capacity_discussion = (
        f"Holding d={selected['embedding_dimension']} fixed, H={narrow['hidden_width']} had training/validation RMSE "
        f"SGD {narrow['train_rmse']:,.2f}/{narrow['validation_rmse']:,.2f}, compared with "
        f"SGD {selected['train_rmse']:,.2f}/{selected['validation_rmse']:,.2f} for the selected width. "
        f"At H={selected['hidden_width']}, the best alternative dimension was d={other_embedding['embedding_dimension']} "
        f"with validation RMSE SGD {other_embedding['validation_rmse']:,.2f}. "
        "Thus embedding size and hidden width interact; increasing embedding dimension does not uniformly improve validation error."
    )
    if selected["hidden_width"] != narrow["hidden_width"] and selected["train_rmse"] < narrow["train_rmse"]:
        capacity_discussion += (
            f" The narrow model's training error is only SGD {narrow['train_rmse'] - selected['train_rmse']:,.2f} higher. "
            "This gives weak evidence of relative underfitting within the budget; the validation improvement is much larger, "
            "so optimization and generalization differences also matter."
        )

    # Post-fit distribution diagnostics use the saved raw targets only. They do
    # not fit a transform, change selection, or evaluate any test predictions.
    diagnostic_path = tables / "partition_price_summary.csv"
    # Historical absolute paths are provenance, not dependencies after copying.
    prepared_path = Path(prepared_directory) if prepared_directory is not None else output.parent / "data"
    if prepared_path.exists():
        from part_a.src.data import load_data_bundle
        diagnostic_selection = load_data_bundle(prepared_path / "selection", expected_protocol="selection")
        diagnostic_refit = load_data_bundle(prepared_path / "refit", expected_protocol="refit")
        records = []
        for name, dataset, years in (
            ("selection_train", diagnostic_selection.datasets["train"], "2017-2020"),
            ("validation", diagnostic_selection.datasets["validation"], "2021"),
            ("refit_train", diagnostic_refit.datasets["train"], "2017-2021"),
            ("test", diagnostic_refit.datasets["test"], "2022"),
        ):
            targets = dataset.tensors[-1].double()
            records.append({"partition": name, "years": years, "rows": len(dataset),
                            "observed_mean_price_sgd": targets.mean().item(),
                            "observed_population_std_sgd": targets.std(correction=0).item(),
                            "purpose": "post-fit interpretation only"})
        pd.DataFrame(records).to_csv(diagnostic_path, index=False)
    distribution_discussion = ""
    if diagnostic_path.exists():
        diagnostic = pd.read_csv(diagnostic_path).set_index("partition")
        distribution_discussion = (
            f"Observed mean prices in the saved data are SGD {diagnostic.loc['selection_train', 'observed_mean_price_sgd']:,.2f} "
            f"for 2017–2020, SGD {diagnostic.loc['validation', 'observed_mean_price_sgd']:,.2f} for 2021, "
            f"and SGD {diagnostic.loc['test', 'observed_mean_price_sgd']:,.2f} for 2022. "
            f"The refit training mean is SGD {diagnostic.loc['refit_train', 'observed_mean_price_sgd']:,.2f}. "
            "These post-fit diagnostics support considering temporal distribution shift when interpreting the higher test RMSE. "
            "They are unadjusted transaction averages and also reflect property composition, so they do not identify a causal market effect. "
            "Year is deliberately excluded from the ten model inputs. No configuration, transform, or checkpoint was chosen using these diagnostics."
        )
    text = f"""# A1 results and discussion draft

This is material for the Part A report, not the complete final report. All values below come from saved run artifacts.

## Baseline architecture and parameter count

Four independent category embeddings concatenate with six continuous inputs. The MLP is `Linear(4d+6,H) → LayerNorm(H) → ReLU → Linear(H,1)`. Its output is a standardized price; inverse target scaling gives SGD predictions. Category table sizes including unknown ID 0 are 13, 27, 44, and 18.

`P = d(13+27+44+18) + (4d+10)H + 1` includes all embedding weights, linear weights/biases, and LayerNorm scale/bias. At supplied defaults H=64,d=8, P=3,505. The selected model has **{selected['trainable_parameters']:,} parameters**. The architecture diagram describes the supplied default and labels each parameter contribution.

## Search and selection

Ray Tune completed all 25 combinations of H∈{{16,32,64,128,256}} and d∈{{2,4,8,12,16}} with fresh seed-42 initialization. Adam learning rate 0.01, batch size 512, standardized-target MSE, and LayerNorm stayed fixed. Maximum 20 epochs, patience 4, and minimum significant improvement 1e-4 used validation RMSE in SGD. Encoders/scalers were fitted only on 2017–2020; 2021 selected the checkpoint/configuration. No trial used test predictions or test scores.

The supplied-default trial achieved validation RMSE **SGD {baseline['validation_rmse']:,.2f}** at epoch {baseline['best_epoch']}. The grid selected **H={selected['hidden_width']}, d={selected['embedding_dimension']}**, with validation RMSE **SGD {selected['validation_rmse']:,.2f}**, validation R² **{selected['validation_r2']:.6f}**, and **E_val={summary['E_val']}**. Its validation RMSE is {reduction:.2f}% lower than the supplied default. Selection uses the lowest actual checkpoint RMSE, then fewer parameters, smaller H, and smaller d for exact ties.

| Rank | H | d | Parameters | Best epoch | Validation RMSE, SGD |
| --- | ---: | ---: | ---: | ---: | ---: |
{ranking}

The full table and heatmap retain every combination. The chosen width/dimension are supported by their joint measured score within this grid and seed, rather than a presumption that the largest model is best. Results do not establish a global optimum or stability across seeds.

{capacity_discussion}

## Learning behavior

At E_val, training RMSE was **SGD {selected['train_rmse']:,.2f}** and the validation–training gap was **SGD {gap:,.2f}**. The trial completed {selected['epochs_completed']} epochs. Its last completed epoch had training RMSE SGD {last.train_rmse:,.2f} and validation RMSE SGD {last.validation_rmse:,.2f}. {late_behavior}

Both curves evaluate the completed epoch's model in evaluation mode across the full partition, making their convention comparable. Chronological market changes between fitting and validation years may also contribute to a gap; these curves alone do not separate temporal shift from overfitting. Small configurations with higher errors on both partitions are compatible with limited model capacity under this training budget, but grid scores alone cannot prove underfitting without their learning curves.

## Final combined-data refit

A fresh seed-42 selected model trained on all 116,427 rows from 2017–2021, using the separately saved refit preprocessing. Following the professor's explicit answer, 2022 test RMSE monitored stopping under the same 20/4/1e-4 rule. The restored checkpoint is **E*={summary['E_star']}**; training completed {summary['refit_epochs_completed']} epochs. E_val and E* refer to different runs and must not be interchanged.

| H | d | Parameters | E_val | E* | Final test RMSE, SGD | Final test R² |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| {selected['hidden_width']} | {selected['embedding_dimension']} | {selected['trainable_parameters']:,} | {summary['E_val']} | {summary['E_star']} | {test['rmse']:,.2f} | {test['r2']:.6f} |

The final metrics cover all 26,702 test rows and come from one restored checkpoint. Public plots show training/validation metrics only; the optional refit curve shows combined training alone. Test stopping history is retained internally and is not an epoch plot.

**Limitation:** {summary['test_independence_limitation']} Validation and test scores also use different fitting periods and scaling statistics; their difference is not a controlled estimate of the benefit of retraining.

{distribution_discussion}

## Next-step frozen settings

A2/A3 refits inherit H={selected['hidden_width']}, d={selected['embedding_dimension']}, learning rate 0.01, batch size 512, and E*={summary['E_star']}. Reuse the saved refit preprocessing unchanged. A2 trains fresh baseline/ablation models with seeds 42–46 for exactly E* epochs each, with one final test evaluation per run.

## Artifacts and implementation references

- `tables/grid_results.csv`: every trial's selected metrics, epoch counts, parameter count, runtime, and checkpoint location.
- `figures/selected_learning_curve.{{png,pdf}}`: required training/validation curves.
- `figures/validation_grid_heatmap.{{png,pdf}}`: full grid support for selection.
- `figures/baseline_architecture.{{png,pdf}}`: supplied-default diagram and count.
- `tables/final_test_metrics.csv`: coherent restored refit result.
- `tables/partition_price_summary.csv`: post-fit target-distribution diagnostics; no additional predictions or test-based tuning.
- `refit/best_model.pt`: final A1 model plus optimizer/task/model/preprocessing metadata.
- `frozen_a1_settings.json`: settings and E* for later comparisons.

Ray's [grid_search documentation](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.grid_search.html) describes the Cartesian product. Its [report documentation](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.report.html) describes metric/checkpoint registration. One final Ray report records the restored checkpoint; explicit epoch fields are used because report-call counts are not epoch counts.
"""
    (output / "analysis.md").write_text(text, encoding="utf-8")
