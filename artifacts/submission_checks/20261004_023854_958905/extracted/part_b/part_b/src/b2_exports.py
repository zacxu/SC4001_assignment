"""B2 tables, curves, and numerical discussion from saved completed runs."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "artifacts/cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import pandas as pd

from assignment_common.artifacts import write_json
from part_b.src.b2 import choose_batch_size


def export_b2(directory):
    output = Path(directory)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "completed":
        raise ValueError("B2 exports require completed results")
    rows = json.loads((output / "batch_size_search.json").read_text(encoding="utf-8"))
    selected = choose_batch_size(rows)
    if selected != summary["selected"]:
        raise ValueError("Saved B2 selection disagrees with the validation results")
    tables, figures = output / "tables", output / "figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    required = ["batch_size", "best_epoch", "validation_loss", "validation_classification_error",
                "validation_sensitivity", "validation_specificity"]
    pd.DataFrame(rows)[required].to_csv(tables / "batch_size_selection.csv", index=False)
    pd.DataFrame(rows).to_csv(tables / "batch_size_search_details.csv", index=False)
    pd.DataFrame([{"model": "B2 VGG-like", "learning_rate": summary["learning_rate"],
        "batch_size": selected["batch_size"], "seed": 42, "checkpoint_epoch": selected["best_epoch"],
        "trainable_parameters": 18185, **summary["final_test"]}]).to_csv(tables / "final_test.csv", index=False)
    histories, diagnostics, curves = {}, [], []
    for row in rows:
        history = pd.read_csv(output / row["run_directory"] / "history.csv")
        if len(history) != row["epochs_completed"] or any(key.startswith("test_") for key in history):
            raise ValueError("Invalid B2 history or test epoch curves")
        histories[row["batch_size"]] = history
        curves.extend({"batch_size": row["batch_size"], **record} for record in history.to_dict(orient="records"))
        diagnostics.append({"batch_size": row["batch_size"], "best_epoch": row["best_epoch"],
            "epochs_completed": row["epochs_completed"], "stopped_early": row["stopped_early"],
            "optimizer_steps_per_epoch": row["optimizer_steps_per_epoch"],
            "optimizer_steps_to_selected_epoch": row["optimizer_steps_to_selected_epoch"],
            "first_validation_loss": float(history.iloc[0]["validation_loss"]),
            "last_validation_loss": float(history.iloc[-1]["validation_loss"]),
            "minimum_validation_loss": float(history["validation_loss"].min()),
            "minimum_validation_loss_epoch": int(history.loc[history["validation_loss"].idxmin(), "epoch"]),
            "first_validation_accuracy": float(history.iloc[0]["validation_accuracy"]),
            "last_validation_accuracy": float(history.iloc[-1]["validation_accuracy"]),
            "maximum_validation_accuracy": float(history["validation_accuracy"].max()),
            "selected_validation_loss": row["validation_loss"],
            "selected_validation_accuracy": row["validation_accuracy"],
            "selected_training_accuracy": row["train_accuracy"], "selected_training_loss": row["train_loss"],
            "maximum_absolute_epoch_validation_loss_change": float(history["validation_loss"].diff().abs().max()),
            "maximum_absolute_epoch_validation_accuracy_change": float(history["validation_accuracy"].diff().abs().max())})
    pd.DataFrame(curves).to_csv(tables / "learning_curves.csv", index=False)
    write_json(output / "curve_diagnostics.json", diagnostics)
    colors = ["#4477AA", "#EE7733", "#228833", "#CC3311", "#AA3377"]
    for metric, ylabel in (("loss", "Binary cross-entropy loss"), ("accuracy", "Accuracy")):
        figure, axes = plt.subplots(2, 1, figsize=(8.7, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [1.25, 1]})
        for row, color in zip(rows, colors):
            history = histories[row["batch_size"]]
            label = f"Batch {row['batch_size']}" + (" (selected)" if row == selected else "")
            for axis, partition in zip(axes, ("validation", "train")):
                axis.plot(history["epoch"], history[f"{partition}_{metric}"], "o-", color=color,
                          label=label, markersize=3, linewidth=1.7)
                axis.scatter(row["best_epoch"], row[f"{partition}_{metric}"], marker="*", color=color,
                             s=100, edgecolors="black", linewidths=.35, zorder=5)
        axes[0].set_title("Validation: all five batch sizes")
        axes[1].set_title("Training: completed-epoch model in evaluation mode")
        for axis in axes:
            axis.set_ylabel(ylabel)
            axis.grid(alpha=.22)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            if metric == "accuracy":
                minimum = min(float(history[f"{partition}_accuracy"].min())
                              for history in histories.values() for partition in ("train", "validation"))
                axis.set_ylim(max(0, int(minimum * 10) / 10), 1.01)
                axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            else:
                axis.set_ylim(bottom=0)
        axes[0].legend(fontsize=9, ncol=2, loc="best")
        axes[1].set_xlabel("Completed epoch (curves end at each run's actual stopping epoch)")
        figure.suptitle(f"B2 VGG-like: {metric}, B1-selected LR {summary['learning_rate']:.0e}, seed 42\n"
                        "Stars mark error-selected checkpoints; stopping monitors validation loss", fontsize=11)
        figure.tight_layout()
        figure.savefig(figures / f"validation_{metric}.png", dpi=220, bbox_inches="tight")
        figure.savefig(figures / f"validation_{metric}.pdf", bbox_inches="tight")
        plt.close(figure)
    table_rows = "\n".join(f"| {row['batch_size']} | {row['best_epoch']} | {row['validation_loss']:.6f} | "
        f"{row['validation_classification_error']:.4%} | {row['validation_sensitivity']:.4%} | {row['validation_specificity']:.4%} |" for row in rows)
    controls = "\n".join(f"| {row['batch_size']} | {row['best_epoch']}/{row['epochs_completed']} | "
        f"{row['optimizer_steps_per_epoch']} | {row['optimizer_steps_to_selected_epoch']} | "
        f"{row['train_accuracy']:.4%} | {row['validation_accuracy']:.4%} |" for row in rows)
    curve_descriptions = []
    for item in diagnostics:
        row = next(row for row in rows if row["batch_size"] == item["batch_size"])
        stopping = "loss patience exhausted" if row["stopped_early"] else "20-epoch cap reached"
        curve_descriptions.append(f"- **Batch {row['batch_size']}:** first/final validation loss "
            f"{item['first_validation_loss']:.4f}/{item['last_validation_loss']:.4f}; minimum loss "
            f"{item['minimum_validation_loss']:.6f} at epoch {item['minimum_validation_loss_epoch']}; "
            f"selected accuracy {row['validation_accuracy']:.4%} at epoch {row['best_epoch']}; "
            f"{row['epochs_completed']} completed epochs ({stopping}).")
    curves_text = "\n".join(curve_descriptions)
    test = summary["final_test"]
    selected_minimum = next(item for item in diagnostics if item["batch_size"] == selected["batch_size"])
    tied = [row["batch_size"] for row in rows if row["validation_classification_error"] == selected["validation_classification_error"]]
    tie_text = (f"Batches {tied} tie on the lowest error; lower selected validation loss breaks the tie."
                if len(tied) > 1 and len({row['validation_loss'] for row in rows if row['batch_size'] in tied}) > 1
                else "There is one lowest-error candidate." if len(tied) == 1
                else f"Batches {tied} tie on error; the recorded loss/epoch/batch ordering determines the choice.")
    text = f"""# B2 results and discussion draft

## Controlled protocol

Five fresh Figure B2 VGG-like networks (18,185 parameters) use the B1-frozen learning rate **{summary['learning_rate']:.0e}**, seed 42, identical initial weights and underlying shuffled sample order, batch sizes 16/32/64/128/256, the supplied 4,708 training/524 validation images, and saved training-only normalization. Adam, maximum 20 epochs, patience 4 and significant loss improvement 1e-4 are fixed. Models have six bias-free Conv/BN/ReLU stages, three max pools, global averaging, and a biased raw-logit FC1; there is no dropout. The brief's batch 256 overrides the slide's 258 typo. No augmentation, weighting, resampling, or combined-data refit is used.

Stopping monitors validation BCE loss; checkpoint selection minimizes validation classification error, then loss, then earlier epoch. Across candidates, smaller batch size is a final exact-tie break. Metrics use sigmoid >=0.5 and the assumed mapping 0=healthy, 1=pneumonia. The selected batch was frozen before accessing test.

## Required table and batch selection

Each row reports metrics from one restored selected checkpoint. Percentages are used here; CSV exports store proportions.

| Batch size | Best epoch | Validation loss | Classification error | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
{table_rows}

**Select batch {selected['batch_size']}**, epoch **{selected['best_epoch']}**: validation accuracy **{selected['validation_accuracy']:.4%}**, error **{selected['validation_classification_error']:.4%}**, loss **{selected['validation_loss']:.6f}**. {tie_text} The frozen B1 rate and B2 batch are the prescribed inputs to B3.

| Batch | Saved/completed epoch | Updates/epoch | Updates to saved checkpoint | Train accuracy | Validation accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
{controls}

## Curves and batch-size interpretation

Separate `figures/validation_loss` and `figures/validation_accuracy` PNG/vector-PDF figures show five validation curves above five training curves. Stars mark saved checkpoints and actual stopping lengths are preserved. Training metrics use evaluation mode at each completed epoch. No test epoch curves are plotted.

{curves_text}

The selected model has minimum validation loss at epoch **{selected_minimum['minimum_validation_loss_epoch']}**, compared with error-selected epoch **{selected['best_epoch']}**; loss stopping and classification-error selection are separate rules. Its train/validation accuracies are **{selected['train_accuracy']:.4%}/{selected['validation_accuracy']:.4%}**, with losses **{selected['train_loss']:.6f}/{selected['validation_loss']:.6f}**.

Batch size changes the gradient's sample group and BN's batch/spatial statistics. Smaller groups can give noisier gradient/statistic estimates, while larger groups average more samples; no separate gradient-variance or BN-ablation measurement was performed, so these mechanisms are interpretations rather than demonstrated causes of the ranking. It also changes optimizer updates per epoch: **295/148/74/37/19** for batches **16/32/64/128/256**. Final partial batches are retained: **4/4/36/100/100** images respectively. The running BN estimates are updated once per training batch, with the configured momentum; batch sizes therefore also change the number of running-statistic updates. Equal epoch limits and learning rate do not imply equal update budgets. This assignment compares those complete training recipes; it does not isolate batch size at a fixed number of optimizer steps. There is no rule that the smallest or largest batch must win.

## Final test, once

The exact chosen checkpoint was restored and evaluated once on all 624 test images, without refitting, at the chosen inference batch size. The same pass captured logits/probabilities/labels/row IDs in `final_test_predictions.pt` for auditing without further model test inference.

| Accuracy | Sensitivity | Specificity | Classification error | BCE loss |
| ---: | ---: | ---: | ---: | ---: |
| {test['accuracy']:.4%} | {test['sensitivity']:.4%} | {test['specificity']:.4%} | {test['classification_error']:.4%} | {test['loss']:.6f} |

Confusion: TP={test['tp']}, TN={test['tn']}, FP={test['fp']}, FN={test['fn']}; test contains 390 positive and 234 negative labels. Validation's positive fraction is 74.24%, compared with 62.5% in test, so sensitivity/specificity complement prevalence-weighted accuracy. Test specificity is **{test['specificity']:.4%}**, versus **{selected['validation_specificity']:.4%}** at validation; conditional-class rates clarify which class accounts for errors. Neither the B1 learning rate nor B2 selection/threshold was revised using test results.

## Limitations and handoff

The supplied splits share eight distinct pixel-identical images between train and validation (10 training rows, 8 validation rows), so validation may be optimistic. Three extra copies occur within test; no exact train/test or validation/test pixel overlaps were found, but patient IDs are absent. Membership is preserved. Class names are assumed rather than encoded in NPZ metadata. B2 uses one seed and does not estimate seed variability. Its learning rate was selected on B1, not optimized independently for VGG. Exact causes of validation/test differences and curve instability were not established.

`frozen_b2_settings.json` records **LR {summary['learning_rate']:.0e}, batch {selected['batch_size']}**, selected epoch/checkpoint hash and B1/data/preprocessing lineage for B3/B4. Further architecture comparison reuses saved B1/B2 test records; B3 and B4 remain to be run.
"""
    (output / "analysis.md").write_text(text, encoding="utf-8")
