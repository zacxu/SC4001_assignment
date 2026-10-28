"""B3 curves, saved-result comparison, and discussion assets without new inference."""

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
from part_b.src.b3 import comparison_rows
from part_b.src.data import sha256_file


def export_b3(directory, *, source_directories=None):
    output = Path(directory)
    read = lambda name: json.loads((output / name).read_text(encoding="utf-8"))
    summary, selected = read("summary.json"), read("selected.json")
    if summary["status"] != "completed" or selected != summary["selected"]:
        raise ValueError("B3 exports require coherent completed results")
    inputs = read("comparison_inputs.json")
    complexities = {key: read(f"complexity/{key}.json") for key in ("b1", "b2", "b3")}
    comparison = comparison_rows(inputs["records"], complexities)
    if comparison != read("architecture_comparison.json") or comparison != summary["architecture_comparison"]:
        raise ValueError("Comparison must match the saved final results and complexity")
    tables, figures = output / "tables", output / "figures"
    tables.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    pd.DataFrame(comparison).to_csv(tables / "architecture_comparison.csv", index=False)
    pd.DataFrame([selected]).to_csv(tables / "validation_selection.csv", index=False)
    pd.DataFrame([{"model": "B3 ResNet-like", "learning_rate": summary["learning_rate"],
        "batch_size": summary["batch_size"], "checkpoint_epoch": selected["best_epoch"],
        "seed": 42, **summary["final_test"]}]).to_csv(tables / "final_test.csv", index=False)
    pd.DataFrame([{"architecture": key, **value["parameter_breakdown"]} for key, value in complexities.items()]).to_csv(
        tables / "parameter_breakdown.csv", index=False)
    pd.DataFrame([{"architecture": key, **row} for key, value in complexities.items() for row in value["layers"]
                  if row["type"] in ("Conv2d", "Linear")]).to_csv(tables / "flops_layers.csv", index=False)
    pd.DataFrame([{"architecture": key, **row} for key, value in complexities.items() for row in value["nonzero_profiler_operators"]]).to_csv(
        tables / "flops_operator_counts.csv", index=False)
    pd.DataFrame([{"architecture": key, **row} for key, value in complexities.items() for row in value["residual_additions"]]).to_csv(
        tables / "residual_addition_flops.csv", index=False)
    all_curves = []
    sources = source_directories or {"b1": output.parent / "b1", "b2": output.parent / "b2", "b3": output}
    for key, lineage in inputs["lineage"].items():
        # Read current local folders, preserving original provenance unchanged.
        checkpoint = Path(sources[key]) / Path(lineage["checkpoint"].replace("\\", "/"))
        if sha256_file(checkpoint) != lineage["checkpoint_sha256"]:
            raise ValueError(f"{key} export requires its recorded final-refit checkpoint")
        source_summary = json.loads((Path(sources[key]) / "summary.json").read_text(encoding="utf-8"))
        selected_checkpoint = Path(sources[key]) / source_summary["selected"]["checkpoint"].replace("\\", "/")
        run = selected_checkpoint.parent
        history = pd.read_csv(run / "history.csv")
        if any(column.startswith("test_") for column in history):
            raise ValueError("No test epoch curves are permitted")
        all_curves.extend({"architecture": key, **row} for row in history.to_dict(orient="records"))
    pd.DataFrame(all_curves).to_csv(tables / "selected_pipeline_learning_curves.csv", index=False)
    history = pd.read_csv(output / "run/history.csv")
    if len(history) != selected["epochs_completed"]:
        raise ValueError("B3 curve length must match actual training")
    history.to_csv(tables / "learning_curves.csv", index=False)
    diagnostic = {"best_epoch": selected["best_epoch"], "epochs_completed": selected["epochs_completed"],
        "stopped_early": selected["stopped_early"], "minimum_validation_loss": float(history["validation_loss"].min()),
        "minimum_validation_loss_epoch": int(history.loc[history["validation_loss"].idxmin(), "epoch"]),
        "first_validation_loss": float(history.iloc[0]["validation_loss"]),
        "last_validation_loss": float(history.iloc[-1]["validation_loss"]),
        "first_validation_accuracy": float(history.iloc[0]["validation_accuracy"]),
        "last_validation_accuracy": float(history.iloc[-1]["validation_accuracy"]),
        "selected_training_accuracy": selected["train_accuracy"], "selected_training_loss": selected["train_loss"],
        "selected_validation_accuracy": selected["validation_accuracy"], "selected_validation_loss": selected["validation_loss"]}
    write_json(output / "curve_diagnostics.json", diagnostic)
    for metric, ylabel in (("loss", "Binary cross-entropy loss"), ("accuracy", "Accuracy")):
        figure, axis = plt.subplots(figsize=(7.5, 4.4))
        for split, color in (("train", "#4477AA"), ("validation", "#EE7733")):
            axis.plot(history["epoch"], history[f"{split}_{metric}"], "o-", color=color, markersize=4,
                      label="Training (evaluation mode)" if split == "train" else "Validation")
            axis.scatter(selected["best_epoch"], selected[f"{split}_{metric}"], s=135, marker="*", color=color,
                         edgecolors="black", linewidths=.5, zorder=5)
        axis.set(xlabel="Completed epoch", ylabel=ylabel,
                 title=f"B3 ResNet-like: LR {summary['learning_rate']:.0e}, batch {summary['batch_size']}, seed 42\n"
                       f"Stars: error-selected checkpoint at epoch {selected['best_epoch']}")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True)); axis.grid(alpha=.25)
        if metric == "accuracy":
            minimum = min(float(history[f"{split}_accuracy"].min()) for split in ("train", "validation"))
            axis.set_ylim(max(0, int(minimum * 10) / 10), 1.01)
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        else:
            axis.set_ylim(bottom=0)
        axis.legend(); figure.tight_layout()
        for extension in ("png", "pdf"):
            figure.savefig(figures / f"training_validation_{metric}.{extension}", dpi=220, bbox_inches="tight")
        plt.close(figure)
    test = summary["final_test"]
    required = "\n".join(f"| {row['model']} | {row['accuracy']:.4%} | {row['sensitivity']:.4%} | "
        f"{row['specificity']:.4%} | {row['classification_error']:.4%} | {row['trainable_parameters']:,} | "
        f"{row['inference_flops']:,} |" for row in comparison)
    components = "\n".join(f"| {key.upper()} | {value['convolution_and_linear_macs']:,} | "
        f"{value['convolution_and_linear_flops']:,} | {value['residual_addition_flops']:,} | {value['inference_flops']:,} |"
        for key, value in complexities.items())
    text = f"""# B3 and three-architecture comparison draft

## B3 executed protocol and results

The compact Figure B3 ResNet-like model has 77,169 trainable parameters. It uses the B1-frozen learning rate **{summary['learning_rate']:.0e}** and B2-frozen batch **{summary['batch_size']}**, seed 42, full supplied training/validation partitions, saved training-only normalization, Adam, maximum 20 epochs, patience 4, and significant loss improvement 1e-4. It has a Conv/BN/ReLU stem, max pooling, three residual blocks at widths 16/32/64, two learned stride-two projections, ReLU after residual additions, global averaging, and one biased raw-logit FC output. There is no dropout, augmentation or class weighting. The final model is a separate fresh combined-data refit required by Q&A.md.

Stopping monitors validation loss. The checkpoint minimizes validation classification error, then loss, then earlier epoch. Selected **epoch {selected['best_epoch']}**, after **{selected['epochs_completed']} completed epochs**: validation accuracy **{selected['validation_accuracy']:.4%}**, error **{selected['validation_classification_error']:.4%}**, loss **{selected['validation_loss']:.6f}**, sensitivity **{selected['validation_sensitivity']:.4%}**, specificity **{selected['validation_specificity']:.4%}**. The minimum validation loss is **{diagnostic['minimum_validation_loss']:.6f} at epoch {diagnostic['minimum_validation_loss_epoch']}**. Training/validation accuracy at selection is **{selected['train_accuracy']:.4%}/{selected['validation_accuracy']:.4%}**; losses are **{selected['train_loss']:.6f}/{selected['validation_loss']:.6f}**. Separate loss/accuracy figures plot both partitions, with no test epoch curves.

Selection froze the epoch count and hyperparameters before a fresh seed-42 model and Adam state trained on 5,232 combined training/validation images for exactly {selected['best_epoch']} epochs. The original training-only normalization is retained consistently under the brief. This final `refit/best_model.pt` checkpoint received one test evaluation on 624 images. Test **accuracy {test['accuracy']:.4%}, sensitivity {test['sensitivity']:.4%}, specificity {test['specificity']:.4%}**, classification error **{test['classification_error']:.4%}**, BCE **{test['loss']:.6f}**; TP/TN/FP/FN **{test['tp']}/{test['tn']}/{test['fp']}/{test['fn']}**. The same forward pass captured predictions for later auditing without new model test inference.

## Required architecture comparison

All performance values are saved final test results. B1/B2's existing results were read without another test pass. Percentages are shown here; CSVs use proportions. FLOPs are per float32 1×1×64×64 synthetic image in evaluation mode, within the profiled scope below.

| Model | Test accuracy | Sensitivity | Specificity | Classification error | Learnable parameters | FLOPs/image, profiled scope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{required}

## FLOPs convention and audit trail

`torch.profiler.profile(with_flops=True, record_shapes=True, acc_events=True)` profiles a CPU evaluation copy of each model on one synthetic zero image. One convolution/linear multiply-add is **two FLOPs**. Actual nonzero profiler events are convolution, linear matrix multiplication, and B3 residual elementwise addition; the latter adds one FLOP per output element. Layer-shape formulas reproduce both components exactly. Model state and BN buffers remain unchanged; no dataset is used for complexity profiling.

| Architecture | Conv/linear MACs | Conv/linear FLOPs = 2×MACs | Residual additions | Reported FLOPs |
| --- | ---: | ---: | ---: | ---: |
{components}

BN, ReLU, pooling, FC bias addition, views, data normalization/transfers, probability sigmoid/threshold, and backward/optimizer work are excluded. Uncounted operations are recorded explicitly; their omission does not mean zero computational cost. These are operation-count estimates, not measured latency, memory, throughput, or a complete scalar-operation inventory. See [PyTorch profiler documentation](https://docs.pytorch.org/docs/2.11/profiler.html). Detailed layer shapes, MACs, nonzero/zero profiler operators, residual additions, and parameter breakdowns are saved in `complexity/` and `tables/`.

## Architecture strengths and limitations

B1 has three convolution stages and larger 5×5 early kernels. Its spatial 4×4 output feeds FC64/FC32 with dropout; the linear layers contain 67,713 parameters, most of its 99,569 total. It can retain spatial combinations in the head, but has the highest parameter and profiled operation counts in this comparison.

B2's six smaller 3×3 convolutions stack nonlinear transformations before global averaging. Its 33-parameter output head yields the fewest parameters (18,185) and profiled FLOPs. Global averaging compresses spatial information; this economy does not itself prove superior generalization. Its assigned training recipe uses batch 16, with many more updates per epoch than B1's batch 128.

B3 adds identity/projection shortcut routes while preserving shape alignment and applying the required ReLU after sums. These routes are intended to ease optimization, as explained in [He et al.'s residual-learning paper](https://arxiv.org/abs/1512.03385); this single run does not isolate their causal effect. Its 77,169 parameters and profiled FLOPs lie between B1 and B2. Learned projections and wider convolution channels add cost relative to B2; global averaging keeps its head at 65 parameters. The assignment's ReLU placement follows Figure B rather than a library ResNet preset.

The table compares selected assignment pipelines. The common LR was tuned on B1, batch size was tuned on B2, B1 uses batch 128, and B2/B3 use batch {summary['batch_size']}. One seed per final model and differing checkpoint epochs/update budgets prevent attributing differences to architecture alone. High sensitivity must be read alongside specificity and false positives; all models use the frozen 0.5 threshold, and the comparison does not select a new model or tune settings from test results.

Q&A.md confirms 0=healthy/1=pneumonia; the NPZ itself lacks semantic class-name metadata. Eight distinct images overlap train/validation, potentially making validation optimistic; patient IDs are absent, and three extra copies occur within test. No exact train/test or validation/test pixel overlaps were found. Supplied partitions are preserved. The sources of validation/test differences are not established.

## Handoff

`frozen_b3_settings.json` records separate validation-selection and fresh final-refit B3 checkpoints and B1/B2/data lineage for B4. B4 should load the saved final-refit checkpoints and compare Grad-CAM on the same five sampled positive-class test images. Current B4 explanations and reports/packages use the canonical FAQ-corrected final results.
"""
    (output / "analysis.md").write_text(text, encoding="utf-8")
