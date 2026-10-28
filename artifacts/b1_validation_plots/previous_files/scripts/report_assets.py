"""Report-sized vector plots built exclusively from completed saved evidence."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import MaxNLocator, FuncFormatter

from report_layout import CONTENT_WIDTH

COLORS = ["#176087", "#cf741c", "#228064", "#9b4e87", "#6959a6"]
WIDTH = CONTENT_WIDTH / 72
HALF = (CONTENT_WIDTH - 12) / 144


def setup():
    for name in ("arial.ttf", "arialbd.ttf"):
        font_manager.fontManager.addfont(f"C:/Windows/Fonts/{name}")
    plt.rcParams.update({"font.family": "Arial", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#637785", "axes.labelcolor": "#253d4e", "text.color": "#253d4e",
        "pdf.fonttype": 42, "figure.facecolor": "white", "savefig.facecolor": "white"})


def save(fig, directory, name):
    directory = Path(directory)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_artists = list(fig.texts)
    for axis in fig.axes:
        text_artists.extend([axis.title, *axis.texts])
        if axis.axison:
            text_artists.extend([axis.xaxis.label, axis.yaxis.label])
            for locations, labels, limits in ((axis.get_xticks(), axis.get_xticklabels(), axis.get_xlim()),
                                             (axis.get_yticks(), axis.get_yticklabels(), axis.get_ylim())):
                low, high = sorted(limits)
                text_artists.extend(label for location, label in zip(locations, labels) if low <= location <= high)
    for artist in text_artists:
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.x1 + .5 or box.y1 > fig.bbox.y1 + .5:
                raise ValueError(f"Clipped plot text in {name}: {artist.get_text()!r}")
    fig.savefig(directory / f"{name}.pdf")
    fig.savefig(directory / f"{name}.png", dpi=200)
    plt.close(fig)


def baseline_diagram(directory):
    fig, axis = plt.subplots(figsize=(WIDTH, 1.38))
    axis.set(xlim=(0, 1), ylim=(0, 1)); axis.axis("off")
    def box(x, y, w, h, label, size=8.5):
        axis.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=.025",
            edgecolor="#6e8fa1", facecolor="#eef4f7", linewidth=.8))
        axis.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=size)
    box(.015, .56, .215, .30, "4 categorical IDs\nEmbeddings: 13 / 27 / 44 / 18\n4 × d = 32 outputs", 8)
    box(.015, .10, .215, .25, "6 continuous inputs\nTraining-standardized")
    xs = [.285, .435, .565, .695, .815]
    labels = ["Concat.\n38 inputs", "Linear\n38 → 64", "LayerNorm\n+ ReLU", "Linear\n64 → 1", "Inverse scale\nPrice (SGD)"]
    for x, label in zip(xs, labels):
        box(x, .32, .10 if x < .81 else .155, .36, label, 8.2)
    for y in (.70, .23):
        axis.annotate("", xy=(.283, .50), xytext=(.232, y), arrowprops={"arrowstyle": "->", "lw": .8})
    for start, end in zip(xs, xs[1:]):
        axis.annotate("", xy=(end - .007, .50), xytext=(start + .105, .50),
                      arrowprops={"arrowstyle": "->", "lw": .8})
    axis.text(.48, .005, "Supplied default: d=8, H=64. Shared embedding dimension; independent weights.",
              fontsize=8.5, ha="center", va="bottom")
    fig.subplots_adjust(left=.01, right=.99, top=.99, bottom=.01)
    save(fig, directory, "a1_baseline_diagram")


def grid_heatmap(frame, directory):
    pivot = frame.pivot(index="hidden_width", columns="embedding_dimension", values="validation_rmse") / 1000
    fig, axis = plt.subplots(figsize=(WIDTH, 2.04))
    values = pivot.to_numpy()
    plot = axis.imshow(values, cmap="Blues_r", aspect="auto", origin="upper")
    for i in range(5):
        for j in range(5):
            axis.text(j, i, f"{values[i,j]:.2f}", ha="center", va="center", fontsize=8.5,
                      color="white" if values[i,j] < (values.min() + values.max()) / 2 else "#182d3a")
    axis.add_patch(Rectangle((.5, 3.5), 1, 1, fill=False, edgecolor="#ed9c2d", linewidth=2))
    axis.set(xticks=range(5), xticklabels=pivot.columns.astype(int), yticks=range(5),
        yticklabels=pivot.index.astype(int), xlabel="Shared embedding dimension d", ylabel="Hidden width H",
        title="A1: best validation RMSE for all 25 Ray trials (SGD thousands)")
    fig.colorbar(plot, ax=axis, fraction=.035, pad=.025).ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=.10, right=.92, top=.85, bottom=.24)
    save(fig, directory, "a1_grid")


def a1_curves(selected, refit, test_monitor, directory):
    for name, frame, title in (("a1_selected_curves", selected, "A1: selected H=256, d=4"),
                              ("a1_refit_training", refit, "A1: combined-data refit")):
        fig, axis = plt.subplots(figsize=(HALF, 2.48))
        axis.plot(frame.epoch, frame.train_rmse / 1000, color=COLORS[0], marker="o", markersize=3, label="Training")
        if "validation_rmse" in frame:
            axis.plot(frame.epoch, frame.validation_rmse / 1000, color=COLORS[1], marker="o", markersize=3, label="Validation")
        else:
            assert frame.epoch.tolist() == test_monitor.epoch.tolist()
            axis.plot(test_monitor.epoch, test_monitor.rmse / 1000, color=COLORS[1], marker="s",
                      markersize=3, label="Test (stopping)")
        axis.axvline(6 if "validation_rmse" in frame else 4, color="#667b8a", linestyle="--", linewidth=.8,
                     label="Selected epoch")
        axis.set(xlabel="Epoch", ylabel="RMSE (SGD thousands)", title=title)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=.18); axis.legend(loc="best", fontsize=8)
        fig.subplots_adjust(left=.19, right=.97, top=.86, bottom=.21)
        save(fig, directory, name)


def paired_regression_limits(frame, left, right):
    """Paired-point and mean +/- SD zoom ranges, using report RMSE units."""
    key = "model" if "model" in frame.columns else "variant"
    limits = {}
    for metric, step in (("rmse", 5), ("r2", .025)):
        samples = [frame[frame[key] == model][metric].to_numpy() for model in (left, right)]
        if metric == "rmse":
            samples = [values / 1000 for values in samples]
        endpoints = np.concatenate([*samples, *[
            np.array([values.mean() - values.std(ddof=1), values.mean() + values.std(ddof=1)])
            for values in samples]])
        low, high = float(endpoints.min()), float(endpoints.max())
        padding = .08 * max(high - low, step)
        limits[metric] = (np.floor((low - padding) / step) * step,
                          np.ceil((high + padding) / step) * step)
    return limits


def paired_results(frame, directory, name, left, right, labels, *, y_limits=None):
    key = "model" if "model" in frame.columns else "variant"
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.10))
    for axis, metric, ylabel in zip(axes, ("rmse", "r2"), ("Test RMSE (SGD thousands)", "Test R²")):
        samples = [frame[frame[key] == model].sort_values("seed")[metric].to_numpy() for model in (left, right)]
        if metric == "rmse":
            samples = [values / 1000 for values in samples]
        for index, values in enumerate(zip(*samples)):
            axis.plot([0, 1], values, color="#80939e", alpha=.5, linewidth=.7, marker="o", markersize=3)
        for i, values in enumerate(samples):
            axis.errorbar(i + .10, values.mean(), yerr=values.std(ddof=1), color=COLORS[i],
                          marker="s", markersize=5, capsize=4, linewidth=1.3)
        axis.set(xticks=[0, 1], xticklabels=labels, xlim=(-.35, 1.4), ylabel=ylabel)
        if y_limits is not None:
            axis.set_ylim(*y_limits[metric])
            axis.set_title("Zoomed y axis", fontsize=8.5)
        axis.grid(axis="y", alpha=.20)
    fig.subplots_adjust(left=.105, right=.98, top=.91, bottom=.21, wspace=.33)
    save(fig, directory, name)


def regression_curve_limits(histories):
    """Padded ranges over all supplied train/validation histories, in plot units."""
    limits = {}
    for metric in ("rmse", "r2"):
        scale = 1000 if metric == "rmse" else 1
        values = np.concatenate([frame[f"{split}_{metric}"].to_numpy() / scale
                                 for frame in histories.values() for split in ("train", "validation")])
        lower, upper = float(values.min()), float(values.max())
        padding = max((upper - lower) * .08, abs(lower) * .01, 1e-6)
        limits[metric] = (lower - padding, upper + padding)
    return limits


def regression_epoch_curves(histories, directory, name, *, y_limits=None):
    """Training/validation diagnostics only; final test scores belong in tables."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.05))
    for index, (label, frame) in enumerate(histories.items()):
        assert "test_rmse" not in frame and "test_r2" not in frame
        for axis, metric, ylabel in zip(axes, ("rmse", "r2"), ("RMSE (SGD thousands)", "R²")):
            scale = 1000 if metric == "rmse" else 1
            for split, style in (("train", "-"), ("validation", "--")):
                axis.plot(frame.epoch, frame[f"{split}_{metric}"] / scale,
                    color=COLORS[index], linestyle=style, linewidth=1.2,
                    marker="o", markersize=2.6,
                    label=f"{label}: {'train' if split == 'train' else 'val.'}")
            axis.set(xlabel="Epoch", ylabel=ylabel)
            if y_limits is not None:
                axis.set_ylim(*y_limits[metric])
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(alpha=.18)
    axes[0].legend(loc="best", fontsize=8, handlelength=1.6)
    fig.subplots_adjust(left=.105, right=.98, top=.97, bottom=.24, wspace=.31)
    save(fig, directory, name)


def importance(frame, directory, name, title):
    labels = {"floor_area_sqm": "Floor area", "remaining_lease_years": "Remaining lease",
        "dist_to_dhoby": "Distance to Dhoby Ghaut", "flat_model_type": "Flat model type", "town": "Town",
        "dist_to_nearest_stn": "Distance to nearest station", "storey_range": "Storey range",
        "eigenvector_centrality": "Eigenvector centrality", "month": "Month", "degree_centrality": "Degree centrality"}
    frame = frame.sort_values("rank")
    fig, axis = plt.subplots(figsize=(WIDTH, 2.53))
    values = frame.mean_absolute_attribution_sgd.to_numpy()
    positions = np.arange(10)
    axis.barh(positions, values, color=COLORS[0] if "baseline" in name else COLORS[1], height=.7)
    axis.set(yticks=positions, yticklabels=[labels[key] for key in frame.feature],
        xlabel="Mean absolute FeatureAblation attribution (SGD)", title=title, xlim=(0, 85000))
    axis.invert_yaxis(); axis.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:,.0f}"))
    for i, value in enumerate(values):
        axis.text(value + 800, i, f"{value:,.2f}", va="center", fontsize=8.5)
    axis.grid(axis="x", alpha=.18); axis.set_axisbelow(True)
    fig.subplots_adjust(left=.285, right=.98, top=.84, bottom=.21)
    save(fig, directory, name)


def classification_curves(frame, selection, key, directory, prefix):
    for metric in ("loss", "accuracy"):
        fig, axes = plt.subplots(2, 1, figsize=(HALF, 3.10), sharex=True)
        groups = list(frame.groupby(key, sort=True))
        for index, (value, group) in enumerate(groups):
            label = f"{value:.0e}" if key == "learning_rate" else str(int(value))
            epoch = int(selection.loc[np.isclose(selection[key], value), "best_epoch"].iloc[0])
            scale = 100 if metric == "accuracy" else 1
            for axis, split in zip(axes, ("validation", "train")):
                axis.plot(group.epoch, group[f"{split}_{metric}"] * scale,
                    color=COLORS[index], linewidth=1, label=label)
                checkpoint = group.loc[group.epoch == epoch].iloc[0]
                axis.plot(epoch, checkpoint[f"{split}_{metric}"] * scale, marker="*", color=COLORS[index], markersize=6)
                axis.set_ylabel(("Val." if split == "validation" else "Train") + (" accuracy (%)" if metric == "accuracy" else " BCE loss"))
                axis.grid(alpha=.17)
                if metric == "accuracy":
                    axis.set_ylim(70, 101)
        fig.suptitle(f"{prefix.upper()}: {'accuracy' if metric == 'accuracy' else 'loss'}", fontsize=10, y=.97)
        axes[0].legend(ncol=3, loc="lower left", bbox_to_anchor=(-.04, 1.06),
                       fontsize=8.5, handlelength=1.2, columnspacing=.9, borderaxespad=0)
        axes[-1].set_xlabel("Epoch")
        axes[-1].set_xticks([1, 5, 10, 15, 20]); axes[-1].set_xlim(.5, 20.5)
        fig.subplots_adjust(left=.20, right=.97, top=.73, bottom=.16, hspace=.28)
        save(fig, directory, f"{prefix}_{metric}")


def b3_curves(frame, directory):
    for metric in ("loss", "accuracy"):
        fig, axis = plt.subplots(figsize=(HALF, 1.97))
        scale = 100 if metric == "accuracy" else 1
        for index, split in enumerate(("train", "validation")):
            axis.plot(frame.epoch, frame[f"{split}_{metric}"] * scale, color=COLORS[index], label=split.title(), linewidth=1)
        axis.axvline(9, color="#72808a", linestyle="--", linewidth=.8)
        axis.set(xlabel="Epoch", ylabel="Accuracy (%)" if metric == "accuracy" else "BCE loss",
                 title=f"B3: {metric}")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.legend(fontsize=8, loc="best"); axis.grid(alpha=.18)
        fig.subplots_adjust(left=.20, right=.97, top=.83, bottom=.26)
        save(fig, directory, f"b3_{metric}")


def gradcam_grid(payload, directory):
    fig, axes = plt.subplots(5, 4, figsize=(WIDTH, 5.95))
    for i, row_id in enumerate(payload["row_ids"].tolist()):
        raw = payload["raw_images"][i].numpy()
        axes[i, 0].imshow(raw, cmap="gray", vmin=0, vmax=255, interpolation="nearest", origin="upper")
        axes[i, 0].set_title(f"Row {row_id} · label 1", fontsize=8.5, pad=2)
        for j, model in enumerate(("b1", "b2", "b3"), 1):
            result = payload["models"][model]
            heat = result["normalized_maps"][i, 0].numpy()
            p = float(result["saved_final_test_probabilities"][i])
            axes[i, j].imshow(raw, cmap="gray", vmin=0, vmax=255, interpolation="nearest", origin="upper")
            axes[i, j].imshow(heat, cmap="inferno", vmin=0, vmax=1, alpha=.55 * heat,
                              interpolation="nearest", origin="upper")
            axes[i, j].set_title(f"{model.upper()} · p={p:.4f}", fontsize=8.5, pad=2)
        for axis in axes[i]:
            axis.axis("off")
    fig.subplots_adjust(left=.005, right=.875, top=.96, bottom=.005, hspace=.25, wspace=.04)
    color_axis = fig.add_axes([.91, .23, .014, .54])
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap="inferno"),
                          cax=color_axis)
    colorbar.set_label("Relative CAM / own maximum", fontsize=8, labelpad=5)
    colorbar.ax.tick_params(labelsize=8)
    save(fig, directory, "b4_grid")
