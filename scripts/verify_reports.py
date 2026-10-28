"""Audit the two rendered reports against saved experiment data and PDF layout."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import pymupdf
import torch


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokens(text):
    return " ".join(text.split())


def contains(text, value):
    assert tokens(str(value)) in tokens(text), value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    directory = parser.parse_args().output.resolve()
    manifest = read_json(directory / "manifest.json")
    assert manifest["step"] == 14 and manifest["pages_per_report"] == 5
    assert manifest["new_training_runs"] == manifest["new_model_inference"] == 0
    assert Path(sys.prefix).resolve() == (ROOT / ".venv").resolve()
    for relative, expected in manifest["inputs"].items():
        assert fingerprint(ROOT / relative) == expected, relative
    baseline = read_json(ROOT / "artifacts/faq_corrections/baseline_snapshot.json")
    previous = baseline["experiment_outputs"]
    protected, changed = 0, []
    for relative, expected in previous.items():
        parts = Path(relative).parts
        canonical_b_final = relative.startswith("part_b/outputs/b4/") or (
            len(parts) >= 4 and parts[:2] == ("part_b", "outputs") and parts[2] in {"b1", "b2", "b3"}
            and (len(parts) == 4 or parts[3] in {"figures", "tables", "complexity"}))
        a2_export = relative.startswith("part_a/outputs/a2/figures/") or relative == "part_a/outputs/a2/analysis.md"
        a1_export = relative.startswith("part_a/outputs/a1/figures/") or relative == "part_a/outputs/a1/analysis.md"
        a3_export = relative in {
            "part_a/outputs/a3/figures/baseline_comparison.png",
            "part_a/outputs/a3/figures/baseline_comparison.pdf",
            "part_a/outputs/a3/analysis.md"}
        current = fingerprint(ROOT / relative)
        if current != expected:
            assert canonical_b_final or a1_export or a2_export or a3_export, relative
            changed.append(relative)
        else:
            protected += 1
    supplied = read_json(ROOT / "artifacts/setup/environment_check.json")["files"]
    for relative, record in supplied.items():
        assert fingerprint(ROOT / relative) == record["sha256"], relative

    pages, checked, fonts = {}, {}, {}
    font = pymupdf.Font(fontfile="C:/Windows/Fonts/arial.ttf")
    for part, name in manifest["reports"].items():
        pdf = directory / name
        markdown = pdf.with_suffix(".md").read_text(encoding="utf-8")
        assert all(character.isspace() or font.has_glyph(ord(character)) for character in markdown)
        document = pymupdf.open(pdf)
        assert len(document) == 5 and document.metadata["author"] == manifest["name"]
        page_texts = []
        font_sizes, link_count = [], 0
        for i, page in enumerate(document):
            assert page.rect == pymupdf.Rect(0, 0, 595, 842)
            text = page.get_text()
            assert "\ufffd" not in text and "□" not in text
            assert len(text.split()) > 200
            contains(text, f"{i+1} / 5")
            spans = [span for block in page.get_text("dict")["blocks"] if block["type"] == 0
                     for line in block["lines"] for span in line["spans"]]
            for span in spans:
                rectangle = pymupdf.Rect(span["bbox"])
                assert rectangle.x0 >= 25 and rectangle.x1 <= 571, (part, i+1, span["text"], span["bbox"])
                assert rectangle.y0 >= 25 and rectangle.y1 <= 835, (part, i+1, span["text"], span["bbox"])
                assert span["size"] >= 7.99, (part, span["text"], span["size"])
                font_sizes.append(span["size"])
            link_count += len(page.get_links())
            page_texts.append(text)
        first = tokens(page_texts[0])
        assert first.index("Generative AI declaration") < first.index("Data")
        contains(first, "OpenAI Codex / ChatGPT")
        contains(first, "behavioral tests")
        assert link_count == (2 if part == "A" else 4)
        assert all(block["bottom"] <= 802 for block in manifest["layout"][part])
        for target in re.findall(r"\]\(([^)]+)\)", markdown):
            assert target.startswith("https://") or (directory / target).is_file(), target
        pages[part] = page_texts
        fonts[part] = min(font_sizes)
        checked[part] = {"pdf": name, "pages": 5, "pdf_sha256": fingerprint(pdf),
                        "markdown_sha256": fingerprint(pdf.with_suffix(".md")),
                        "embedded_reference_links": link_count, "minimum_font_size_pt": min(font_sizes)}
        document.close()

    a = pages["A"]
    contains(a[0], "sample SD for continuous inputs")
    contains(a[0], "population SD for price")
    for index, title in enumerate(("A1.", "A1 ·", "A2 ·", "A3 ·", "A4 ·")):
        contains(a[index], title)
    a1 = read_json(ROOT / "part_a/outputs/a1/summary.json")
    assert a1["test_assisted_stopping"] and a1["E_val"] == 6 and a1["E_star"] == 4
    for value in ("3,505", "7,065", f"{a1['selected']['validation_rmse']:,.2f}", f"{a1['selected']['validation_r2']:.6f}"):
        contains(a[0], value)
    contains(a[1], f"{a1['final_test_metrics']['rmse']:,.2f}")
    contains(a[1], f"{a1['final_test_metrics']['r2']:.6f}")
    contains(a[1], "not an untouched-holdout")
    contains(a[1], "A1 retains its expressly requested train/test curve")
    a1_monitor = pd.DataFrame(json.loads(line) for line in
        (ROOT / "part_a/outputs/a1/refit/internal_test_monitor.jsonl").read_text(encoding="utf-8").splitlines())
    a1_refit = pd.read_csv(ROOT / "part_a/outputs/a1/refit/history.csv")
    assert a1_monitor.epoch.tolist() == a1_refit.epoch.tolist() == list(range(1, 9))
    assert a1_monitor.loc[a1_monitor.rmse.idxmin(), "epoch"] == a1["E_star"] == 4
    assert a1_monitor.loc[a1_monitor.epoch == 4, "rmse"].item() == a1["final_test_metrics"]["rmse"]
    for row, expected in zip(a1_refit.itertuples(), manifest["a1_refit_curve_values"]):
        assert expected == {"epoch": row.epoch, "train_rmse_sgd": row.train_rmse,
            "test_rmse_sgd": a1_monitor.loc[a1_monitor.epoch == row.epoch, "rmse"].item()}
    a1_figure = pymupdf.open(directory / "assets/a1_refit_training.pdf")
    for required in ("Training", "Test (stopping)", "Epoch", "Selected epoch"):
        contains(a1_figure[0].get_text(), required)
    a1_figure.close()
    contains(a[2], "ddof=1")
    contains(a[2], "original-split")
    contains(a[2], "They do not select settings or use test data")
    for asset, page_index in (("a2_categorical", 2), ("a2_activation", 2), ("a3_learning_curves", 3)):
        figure = pymupdf.open(directory / f"assets/{asset}.pdf")
        text = figure[0].get_text()
        for required in ("Epoch", "RMSE", "R²", "train", "val."):
            contains(text, required)
        assert "Test" not in text
        figure.close()
    diagnostics = read_json(ROOT / "part_a/outputs/a2/diagnostics/verification.json")
    assert diagnostics["status"] == "passed"
    for variant in ("baseline", "continuous_only", "sigmoid"):
        history = pd.read_csv(ROOT / f"part_a/outputs/a2/diagnostics/runs/{variant}/seed_42/history.csv")
        assert history.epoch.tolist() == [1, 2, 3, 4]
        assert (history.train_count == 87370).all() and (history.validation_count == 29057).all()
        assert {"train_rmse", "train_r2", "validation_rmse", "validation_r2"} <= set(history.columns)
        assert not any(column.startswith("test_") for column in history.columns)
    grid = pd.read_csv(ROOT / "part_a/outputs/a1/tables/grid_results.csv")
    grid_pdf = pymupdf.open(directory / "assets/a1_grid.pdf")
    for value in grid.validation_rmse:
        contains(grid_pdf[0].get_text(), f"{value/1000:.2f}")
    grid_pdf.close()
    independent_seed_statistics = 0
    for experiment, key, page_index in (("a2", "variant", 2), ("a3", "model", 3)):
        seeds_file = "per_seed_results.csv" if experiment == "a2" else "per_seed_comparison.csv"
        seeds = pd.read_csv(ROOT / f"part_a/outputs/{experiment}/tables/{seeds_file}")
        aggregate = pd.read_csv(ROOT / f"part_a/outputs/{experiment}/tables/summary_metrics.csv")
        for group, subset in seeds.groupby(key):
            assert subset.seed.tolist() == [42, 43, 44, 45, 46]
            assert len(subset) == 5 and (subset.epochs_completed == 4).all()
            row = aggregate.loc[aggregate[key] == group].iloc[0]
            for metric, decimals in (("rmse", 2), ("r2", 6)):
                values = subset[metric].to_numpy()
                for stat, actual in (("mean", np.mean(values)), ("sample_std", np.std(values, ddof=1))):
                    column = f"test_{metric}_{stat}" + ("_sgd" if metric == "rmse" else "")
                    assert math.isclose(actual, row[column], rel_tol=1e-12)
                    contains(a[page_index], f"{actual:,.{decimals}f}")
                    independent_seed_statistics += 1
    lambdas = pd.read_csv(ROOT / "part_a/outputs/a3/tables/lambda_search.csv")
    assert lambdas["lambda"].tolist() == [0, .25, .5, 1, 2]
    assert lambdas.loc[lambdas.validation_rmse.idxmin(), "lambda"] == 2
    for row in lambdas.itertuples():
        contains(a[3], f"{row.train_rmse:,.2f}")
        contains(a[3], f"{row.validation_rmse:,.2f}")
        contains(a[3], f"{row.validation_r2:.6f}")
    importance = pd.read_csv(ROOT / "part_a/outputs/a4/tables/global_importance.csv")
    for model, subset in importance.groupby("model"):
        assert subset["rank"].tolist() == list(range(1, 11)) and len(subset) == 10
        figure = pymupdf.open(directory / f"assets/a4_{model}.pdf")
        for score in subset.mean_absolute_attribution_sgd:
            contains(figure[0].get_text(), f"{score:,.2f}")
            contains(a[4], f"{score:,.2f}")
        figure.close()
    for required in ("all ten", "26,702", "references", "not causal", "one matched seed"):
        contains(a[4], required)

    b = pages["B"]
    contains(b[0], "caption's 'all layers'")
    contains(b[0], "final FC1 has an unrestricted logit")
    for required in ("Q&A confirms 0=healthy, 1=pneumonia", "padding=kernel_size//2", "5,232",
                     "9/13/9 epochs", "Fixed refits use no validation/test stopping"):
        contains(b[0], required)
    assert "Q&A.md" in manifest["inputs"]
    final_checkpoint_checks = 0
    for architecture in ("b1", "b2", "b3"):
        base = ROOT / f"part_b/outputs/{architecture}"
        summary = read_json(base / "summary.json")
        final = read_json(base / "final_test.json")
        frozen = read_json(base / f"frozen_{architecture}_settings.json")
        assert summary["refit"] is True and final["refit"] is True
        assert final["combined_training_rows"] == 5232
        assert final["checkpoint_epoch"] == summary["selected"]["best_epoch"]
        assert Path(final["checkpoint"]).as_posix() == "refit/best_model.pt"
        assert fingerprint(base / final["checkpoint"]) == final["checkpoint_sha256"]
        assert final["checkpoint_sha256"] != summary["selected"]["checkpoint_sha256"]
        assert frozen["refit"] is True and frozen["final_checkpoint_sha256"] == final["checkpoint_sha256"]
        assert summary["test_evaluations"] == 1
        from part_b.src.refit import audit_final_refit
        assert audit_final_refit(base)["status"] == "passed"
        final_checkpoint_checks += 1
    for index, title in enumerate(("B1.", "B1 ·", "B2 ·", "B3 ·", "B4 ·")):
        contains(b[index], title)
    selection_values = 0
    for experiment, page_index in (("b1", 0), ("b2", 1)):
        filename = "learning_rate_selection.csv" if experiment == "b1" else "batch_size_selection.csv"
        table = pd.read_csv(ROOT / f"part_b/outputs/{experiment}/tables/{filename}")
        for row in table.itertuples():
            contains(b[page_index], f"{row.validation_loss:.6f}")
            for field in ("validation_classification_error", "validation_sensitivity", "validation_specificity"):
                contains(b[page_index], f"{100*getattr(row,field):.4f}%")
            selection_values += 4
        for metric in ("loss", "accuracy"):
            plot = pymupdf.open(directory / f"assets/{experiment}_{metric}.pdf")
            plot_text = plot[0].get_text()
            contains(plot_text, "Epoch")
            assert "Test" not in plot_text
            candidates = [f"{value:.0e}" for value in table.learning_rate] if experiment == "b1" else [str(v) for v in table.batch_size]
            for candidate in candidates:
                contains(plot_text, candidate)
            plot.close()
    comparison = pd.read_csv(ROOT / "part_b/outputs/b3/tables/architecture_comparison.csv")
    for row in comparison.itertuples():
        final = read_json(ROOT / f"part_b/outputs/{row.architecture}/final_test.json")
        for field in ("accuracy", "sensitivity", "specificity", "classification_error", "tp", "tn", "fp", "fn"):
            assert math.isclose(getattr(row, field), final["metrics"][field], rel_tol=1e-12)
        assert row.checkpoint_sha256 == final["checkpoint_sha256"]
        for field in ("accuracy", "sensitivity", "specificity", "classification_error"):
            contains(b[3], f"{100*getattr(row,field):.4f}%")
        contains(b[3], f"{row.trainable_parameters:,}")
        contains(b[3], f"{row.inference_flops:,}")
        if row.architecture in ("b1", "b2"):
            target = b[1 if row.architecture == "b1" else 2]
            contains(target, f"{100*row.accuracy:.4f}%")
            contains(target, f"{100*row.sensitivity:.4f}%")
            contains(target, f"{100*row.specificity:.4f}%")
    for required in ("2 per", "28,672", "Exclude BN", "not", "one", "different batches"):
        contains(b[3], required)
    maps = pd.read_csv(ROOT / "part_b/outputs/b4/tables/sample_predictions_and_maps.csv")
    assert len(maps) == 15
    payload = torch.load(ROOT / "part_b/outputs/b4/gradcam.pt", map_location="cpu", weights_only=True)
    assert payload["row_ids"].tolist() == [486, 279, 413, 53, 276]
    assert payload["labels"].tolist() == [1, 1, 1, 1, 1]
    for architecture in ("b1", "b2", "b3"):
        final = read_json(ROOT / f"part_b/outputs/{architecture}/final_test.json")
        assert payload["models"][architecture]["checkpoint_sha256"] == final["checkpoint_sha256"]
        assert payload["models"][architecture]["model_state_unchanged"]
        assert payload["models"][architecture]["maximum_probability_difference_from_saved_test"] == 0
    for row_id in (486, 279, 413, 53, 276):
        contains(b[4], f"Row {row_id}")
    for model in ("B1", "B2", "B3"):
        assert b[4].count(f"{model} · p=") == 5
    for row in maps.itertuples():
        contains(b[4], f"{row.architecture.upper()} · p={row.saved_final_test_probability:.4f}")
    for required in ("seed 42", "without replacement", "target=None", "before BN/ReLU", "Relative CAM / own maximum",
                     "0.55", "absent lesion annotations", "no full-test performance"):
        contains(b[4], required)

    for name in ("build_reports.py", "report_assets.py", "report_layout.py", "verify_reports.py"):
        ast.parse((ROOT / "scripts" / name).read_text(encoding="utf-8"), filename=name)
    report = {"status": "passed", "step": 14, "reports": checked, "author": manifest["name"],
        "author_placeholder_pending": manifest["name_placeholder"], "source_inputs_checked": len(manifest["inputs"]),
        "protected_pre_FAQ_output_hashes_unchanged": protected,
        "approved_FAQ_corrected_output_hash_changes": changed, "supplied_files_unchanged": len(supplied),
        "independent_five_seed_mean_sample_sd_checks": independent_seed_statistics,
        "b1_b2_selected_checkpoint_metric_values_checked": selection_values,
        "all_25_grid_scores_and_five_lambda_results_checked": True,
        "all_20_ranked_feature_scores_checked": True, "all_15_gradcam_titles_and_ids_checked": True,
        "instructor_QA_source_fingerprinted": True, "part_b_fresh_combined_refit_provenance_checks": final_checkpoint_checks,
        "a2_separate_original_split_diagnostic_curves_checked": True, "a3_train_validation_epoch_curves_checked": True,
        "a1_required_train_test_epoch_curve_checked": True, "a1_curve_points_checked": len(a1_monitor),
        "a2_a3_test_epoch_figures_absent": True, "required_sections_in_order": True,
        "glyphs_links_bounds_and_minimum_font_sizes_checked": True,
        "python_venv_isolated": True, "new_training_runs": 0, "new_model_inference": 0,
        "visual_review": "All 10 pages rendered; final visual inspection recorded separately in docs/REPORTS_COMPLETED.md"}
    for path in (directory / "verification.json", ROOT / "artifacts/report_checks/latest.json"):
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "pages": {part: 5 for part in checked},
        "seed_stat_checks": independent_seed_statistics, "protected_pre_FAQ_outputs_unchanged": protected,
        "minimum_font_sizes_pt": fonts, "name_placeholder": manifest["name_placeholder"]}, indent=2))


if __name__ == "__main__":
    main()
