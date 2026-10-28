"""B1 tables and separate loss/accuracy figures from completed saved runs only."""

from __future__ import annotations

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
from part_b.src.b1 import choose_learning_rate


def export_b1(directory: Path | str):
    output = Path(directory)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "completed":
        raise ValueError("B1 exports require completed experiment results")
    rows = json.loads((output / "learning_rate_search.json").read_text(encoding="utf-8"))
    selected = choose_learning_rate(rows)
    if selected != summary["selected"]:
        raise ValueError("Saved selection does not match the five-rate validation table")
    tables, figures = output / "tables", output / "figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    required_columns = ["learning_rate", "best_epoch", "validation_loss", "validation_classification_error",
                        "validation_sensitivity", "validation_specificity"]
    table = pd.DataFrame(rows)
    table[required_columns].to_csv(tables / "learning_rate_selection.csv", index=False)
    table.to_csv(tables / "learning_rate_search_details.csv", index=False)
    pd.DataFrame([{"model": "B1 AlexNet-like", "learning_rate": selected["learning_rate"],
        "batch_size": 128, "seed": 42, "checkpoint_epoch": selected["best_epoch"],
        "trainable_parameters": 99569, **summary["final_test"]}]).to_csv(tables / "final_test.csv", index=False)
    curve_records, curve_diagnostics = [], []
    histories = {}
    for row in rows:
        history = pd.read_csv(output / row["run_directory"] / "history.csv")
        if len(history) != row["epochs_completed"] or any(key.startswith("test_") for key in history):
            raise ValueError("Invalid B1 history length or test epoch curves")
        histories[row["learning_rate"]] = history
        for record in history.to_dict(orient="records"):
            curve_records.append({"learning_rate": row["learning_rate"], **record})
        curve_diagnostics.append({"learning_rate": row["learning_rate"],
            "first_validation_loss": float(history.iloc[0]["validation_loss"]),
            "last_validation_loss": float(history.iloc[-1]["validation_loss"]),
            "minimum_validation_loss": float(history["validation_loss"].min()),
            "minimum_validation_loss_epoch": int(history.loc[history["validation_loss"].idxmin(), "epoch"]),
            "minimum_validation_accuracy": float(history["validation_accuracy"].min()),
            "maximum_validation_accuracy": float(history["validation_accuracy"].max()),
            "first_validation_accuracy": float(history.iloc[0]["validation_accuracy"]),
            "last_validation_accuracy": float(history.iloc[-1]["validation_accuracy"]),
            "selected_validation_loss": row["validation_loss"],
            "selected_validation_accuracy": row["validation_accuracy"],
            "selected_training_accuracy": row["train_accuracy"],
            "selected_training_loss": row["train_loss"],
            "best_epoch": row["best_epoch"], "epochs_completed": row["epochs_completed"],
            "stopped_early": row["stopped_early"]})
    pd.DataFrame(curve_records).to_csv(tables / "learning_curves.csv", index=False)
    write_json(output / "curve_diagnostics.json", curve_diagnostics)
    colors = ["#4477AA", "#EE7733", "#228833", "#CC3311", "#AA3377"]
    for metric, ylabel in (("loss", "Binary cross-entropy loss"), ("accuracy", "Accuracy")):
        figure, axes = plt.subplots(2, 1, figsize=(8.7, 7.0), sharex=True,
                                    gridspec_kw={"height_ratios": [1.25, 1]})
        for row, color in zip(rows, colors):
            history = histories[row["learning_rate"]]
            label = f"LR {row['learning_rate']:.0e}" + (" (selected)" if row == selected else "")
            for axis, partition in zip(axes, ("validation", "train")):
                axis.plot(history["epoch"], history[f"{partition}_{metric}"], marker="o", markersize=3,
                          color=color, label=label, linewidth=1.7)
                axis.scatter(row["best_epoch"], row[f"{partition}_{metric}"], marker="*", s=100,
                             color=color, edgecolors="black", linewidths=.35, zorder=5)
        axes[0].set_title("Validation: all five learning rates")
        axes[1].set_title("Training: completed-epoch model in evaluation mode")
        for axis in axes:
            axis.set_ylabel(ylabel)
            axis.grid(alpha=.22)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            if metric == "accuracy":
                # Show all observations while making small accuracy differences visible.
                minimum = min(float(history[f"{partition}_accuracy"].min())
                              for history in histories.values() for partition in ("train", "validation"))
                axis.set_ylim(max(0., int(minimum * 10) / 10), 1.01)
                axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            else:
                # Loss magnitude is meaningful. Keep spikes visible instead of clipping.
                axis.set_ylim(bottom=0)
        axes[0].legend(fontsize=9, ncol=2, loc="best")
        axes[1].set_xlabel("Completed epoch (curves end at each run's actual stopping epoch)")
        figure.suptitle(f"B1 AlexNet-like: {metric}, batch 128, seed 42\n"
                        "Stars mark error-selected checkpoints; stopping monitors validation loss", fontsize=11)
        figure.tight_layout()
        target = figures / f"validation_{metric}"
        figure.savefig(target.with_suffix(".png"), dpi=220, bbox_inches="tight")
        figure.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)

    test = summary["final_test"]
    diagnostics = {row["learning_rate"]: row for row in curve_diagnostics}
    low, moderate, fast, high, highest = [diagnostics[rate] for rate in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)]
    markdown_rows = "\n".join(
        f"| {row['learning_rate']:.0e} | {row['best_epoch']} | {row['validation_loss']:.6f} | "
        f"{row['validation_classification_error']:.4%} | {row['validation_sensitivity']:.4%} | "
        f"{row['validation_specificity']:.4%} |" for row in rows)
    stopping_rows = "\n".join(f"| {row['learning_rate']:.0e} | {row['best_epoch']} | {row['epochs_completed']} | "
        f"{row['train_accuracy']:.4%} | {row['validation_accuracy']:.4%} | {row['stopped_early']} |" for row in rows)
    result_text = f"""# B1 results and discussion draft

## Controlled experiment

Five fresh Figure B1 AlexNet-like models were trained on the 4,708 supplied training images, with the 524 supplied validation images used for stopping and selection. All runs use the same seed 42, initial weights, first-epoch shuffle fingerprint, saved training-only pixel normalization, Adam, batch size 128, maximum 20 epochs, patience 4, and minimum loss improvement 1e-4. The model has 99,569 trainable parameters, BN/ReLU after all bias-free convolutions, FC bias, two hidden-FC dropout layers at 0.5, and an unrestricted final binary logit. Metrics use sigmoid probability >=0.5 under the recorded label assumption 0=healthy, 1=pneumonia. No augmentation, class weighting, resampling, or combined-data refit is used.

Early stopping monitors validation BCE loss. Each saved checkpoint minimizes validation classification error, then loss, then earlier epoch. Across the five runs, the same ordering is used, with smaller learning rate as a final deterministic tie-break. Loss minimum and saved epoch can therefore differ. Hyperparameters were frozen before the single final test evaluation.

## Required learning-rate table

All metrics in a row belong to that run's one restored selected checkpoint. Error, sensitivity, and specificity are displayed as percentages; CSV exports contain proportions.

| Learning rate | Best epoch | Validation loss | Classification error | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
{markdown_rows}

**Selected learning rate: {selected['learning_rate']:.0e}**, epoch **{selected['best_epoch']}**, validation classification error **{selected['validation_classification_error']:.4%}**, accuracy **{selected['validation_accuracy']:.4%}**, loss **{selected['validation_loss']:.6f}**. This is the validation-only selection used for B2 and B3.

| Learning rate | Saved epoch | Completed epochs | Selected train accuracy | Selected validation accuracy | Early stopped |
| --- | ---: | ---: | ---: | ---: | --- |
{stopping_rows}

## Learning curves

`figures/validation_loss` and `figures/validation_accuracy` are two separate PNG/vector-PDF figures, each with validation and training panels for all five rates. Stars identify the error-selected checkpoint; lines preserve actual stopping lengths. Training curves evaluate each completed-epoch model with dropout disabled and BN statistics frozen, which permits comparison with validation. Online training loss remains separately logged.

The lowest rate, 1e-5, improved slowly: validation loss moved from {low['first_validation_loss']:.4f} to {low['last_validation_loss']:.4f} over {low['epochs_completed']} epochs, with best accuracy {low['selected_validation_accuracy']:.2%}. Its loss was still falling at the cap, consistent with slow learning within the allowed budget. Rate 1e-4 made more gradual, substantial progress, moving from loss {moderate['first_validation_loss']:.4f} to {moderate['last_validation_loss']:.4f}, with best accuracy {moderate['selected_validation_accuracy']:.2%}; it also reached the cap.

Rate 1e-3 started at loss {fast['first_validation_loss']:.4f} and rapidly improved thereafter. Its minimum loss was {fast['minimum_validation_loss']:.6f} at epoch {fast['minimum_validation_loss_epoch']}, while its error-selected checkpoint was epoch {fast['best_epoch']}. It stopped at epoch {fast['epochs_completed']} after four epochs without a significant loss improvement. This illustrates why minimum loss and the reported best epoch differ.

Rate 1e-2 began with validation loss {high['first_validation_loss']:.4f}, which ended at {high['last_validation_loss']:.4f}. Training loss also deteriorated over this run; loss did not improve enough to reset patience, so it ended at epoch {high['epochs_completed']}. Nevertheless its accuracy-selected checkpoint was epoch {high['best_epoch']}, with accuracy {high['selected_validation_accuracy']:.2%}. This pattern indicates an unfavorable optimization trajectory at that rate; it does not by itself establish overfitting. Rate 1e-1 was more erratic, with repeated loss/accuracy reversals; its saved accuracy was {highest['selected_validation_accuracy']:.2%} at epoch {highest['best_epoch']}, below the selected configuration.

The selected checkpoint has training accuracy {selected['train_accuracy']:.4%} versus validation accuracy {selected['validation_accuracy']:.4%}, a gap of {(selected['train_accuracy']-selected['validation_accuracy'])*100:.4f} percentage points, and train/validation losses {selected['train_loss']:.6f}/{selected['validation_loss']:.6f}. The later loss fluctuations do not identify a unique cause. Early stopping ends this finite schedule; it is not proof of global convergence. All raw values and minima are in `curve_diagnostics.json`.

## Why sensitivity and specificity matter

The training and validation partitions contain 74.21% and 74.24% assumed pneumonia images. An all-positive classifier would achieve 74.24% validation accuracy while having sensitivity 100% and specificity 0%. Sensitivity is TP/(TP+FN), the fraction of pneumonia-labelled images detected; specificity is TN/(TN+FP), the fraction of healthy-labelled images rejected as pneumonia. They reveal false negatives and false positives that a single prevalence-weighted accuracy hides.

## One final test evaluation

The selected checkpoint was restored and evaluated once on all **{test['count']}** provided test images. It was not refitted or selected using test performance. Test results are a final table, with no test epoch curves.

| Accuracy | Sensitivity | Specificity | Classification error | BCE loss |
| ---: | ---: | ---: | ---: | ---: |
| {test['accuracy']:.4%} | {test['sensitivity']:.4%} | {test['specificity']:.4%} | {test['classification_error']:.4%} | {test['loss']:.6f} |

Confusion counts: TP={test['tp']}, TN={test['tn']}, FP={test['fp']}, FN={test['fn']}. Test label counts are 390 positive and 234 negative; its positive fraction is 62.5%, different from training/validation. Captured logits, probabilities, labels, and original test-row IDs are saved in `final_test_predictions.pt` from the same forward pass, so audits can recompute metrics without running test inference again.

Test accuracy is {(selected['validation_accuracy']-test['accuracy'])*100:.2f} percentage points below validation. Sensitivity remains high, but specificity falls from {selected['validation_specificity']:.2%} to {test['specificity']:.2%}: {test['fp']} of 234 healthy-labelled images are false positives, compared with only {test['fn']} false negative among 390 pneumonia-labelled images. Specificity is computed within the negative class, so prevalence change alone cannot explain that drop. Distribution differences or optimistic validation caused by supplied overlap are possible explanations; the exact cause was not established. The learning rate and 0.5 threshold remain the choices frozen before test access.

## Limitations and handoff

The supplied data contain eight distinct pixel-identical images shared by training and validation (10 training rows, 8 validation rows), so validation results may be optimistic. The provided partitions and repeated images were preserved. There are three extra pixel-identical copies within test and no exact train/test or validation/test pixel matches, but patient IDs are absent and patient overlap cannot be excluded. Class names remain an assumption because the archive lacks semantic metadata.

This is one seed's learning-rate comparison, as requested for B1; it does not establish variability over seeds. Final output/padding/pooling interpretations are recorded in the preparation report. `frozen_b1_settings.json` holds the selected learning rate, checkpoint hash, preprocessing/data hashes, threshold, and fixed training rules for B2/B3/B4. Future steps reuse the saved B1 test record rather than reevaluating it to choose a better result.
"""
    (output / "analysis.md").write_text(result_text, encoding="utf-8")
