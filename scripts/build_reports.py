"""Build both five-page reports from saved evidence, without training/inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

import report_assets as assets
from report_layout import Report


INPUTS = {}


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source(relative):
    path = ROOT / relative
    INPUTS[relative] = fingerprint(path)
    return path


def read_json(relative):
    return json.loads(source(relative).read_text(encoding="utf-8"))


def read_csv(relative):
    return pd.read_csv(source(relative))


def read_jsonl(relative):
    return pd.DataFrame(json.loads(line) for line in source(relative).read_text(encoding="utf-8").splitlines())


def number(value, decimals=2):
    return f"{value:,.{decimals}f}"


def percent(value):
    return f"{100 * value:.4f}%"


def summary_rows(frame, names, key):
    return [[names[row[key]], f"{int(row['parameters']):,}",
        f"{number(row['test_rmse_mean_sgd'])} ± {number(row['test_rmse_sample_std_sgd'])}",
        f"{row['test_r2_mean']:.6f} ± {row['test_r2_sample_std']:.6f}"]
        for row in frame.to_dict("records")]


def selection_rows(frame, key):
    return [[f"{row[key]:.0e}" if key == "learning_rate" else str(int(row[key])), str(int(row["best_epoch"])),
        f"{row['validation_loss']:.6f}", percent(row["validation_classification_error"]),
        percent(row["validation_sensitivity"]), percent(row["validation_specificity"])]
        for row in frame.to_dict("records")]


def test_table(report, row):
    report.table(["Test accuracy", "Sensitivity", "Specificity", "TP / TN / FP / FN"],
        [[percent(row.accuracy), percent(row.sensitivity), percent(row.specificity),
          f"{int(row.tp)} / {int(row.tn)} / {int(row.fp)} / {int(row.fn)}"]], widths=[1, 1, 1, 1.45])


def prepare_assets(directory):
    assets.setup()
    output = directory / "assets"
    output.mkdir(parents=True, exist_ok=True)
    data = {}
    csv_paths = {
        "a1_grid": "part_a/outputs/a1/tables/grid_results.csv",
        "a1_selected": "part_a/outputs/a1/grid/h256_d04/training/history.csv",
        "a1_refit": "part_a/outputs/a1/refit/history.csv",
        "a2_seeds": "part_a/outputs/a2/tables/per_seed_results.csv",
        "a2_metrics": "part_a/outputs/a2/tables/summary_metrics.csv",
        "a3_lambda": "part_a/outputs/a3/tables/lambda_search.csv",
        "a3_seeds": "part_a/outputs/a3/tables/per_seed_comparison.csv",
        "a3_metrics": "part_a/outputs/a3/tables/summary_metrics.csv",
        "a3_baseline_history": "part_a/outputs/a3/search/lambda_0/history.csv",
        "a3_selected_history": "part_a/outputs/a3/search/lambda_2/history.csv",
        "a4_importance": "part_a/outputs/a4/tables/global_importance.csv",
        "b1_curves": "part_b/outputs/b1/tables/learning_curves.csv",
        "b1_selection": "part_b/outputs/b1/tables/learning_rate_selection.csv",
        "b2_curves": "part_b/outputs/b2/tables/learning_curves.csv",
        "b2_selection": "part_b/outputs/b2/tables/batch_size_selection.csv",
        "b3_curves": "part_b/outputs/b3/tables/learning_curves.csv",
        "b_comparison": "part_b/outputs/b3/tables/architecture_comparison.csv",
        "b4_geometry": "part_b/outputs/b4/tables/geometric_diagnostics.csv",
        "b4_maps": "part_b/outputs/b4/tables/sample_predictions_and_maps.csv"}
    for key, path in csv_paths.items():
        data[key] = read_csv(path)
    data["a1_test_monitor"] = read_jsonl("part_a/outputs/a1/refit/internal_test_monitor.jsonl")
    for variant in ("baseline", "continuous_only", "sigmoid"):
        data[f"a2_{variant}_history"] = read_csv(
            f"part_a/outputs/a2/diagnostics/runs/{variant}/seed_42/history.csv")
    diagnostic = read_json("part_a/outputs/a2/diagnostics/verification.json")
    assert diagnostic["status"] == "passed"
    for name in ("a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4"):
        part = "part_a" if name.startswith("a") else "part_b"
        assert read_json(f"{part}/outputs/{name}/status.json")["status"] == "completed"
        assert read_json(f"{part}/outputs/{name}/verification.json")["status"] == "passed"
        data[f"{name}_summary"] = read_json(f"{part}/outputs/{name}/summary.json")
    for name in ("b1", "b2", "b3"):
        assert data[f"{name}_summary"]["refit"] is True, f"{name}: FAQ requires combined-data refit"
        data[f"{name}_final_test"] = read_json(f"part_b/outputs/{name}/final_test.json")
        final = data[f"{name}_final_test"]
        assert final["metrics"] == data[f"{name}_summary"]["final_test"]
        assert final["refit"] and final["combined_training_rows"] == 5232
        assert final["checkpoint_epoch"] == data[f"{name}_summary"]["selected"]["best_epoch"]
        source(f"part_b/outputs/{name}/{final['checkpoint']}")
        read_json(f"part_b/outputs/{name}/refit_protocol.json")
    assert len(data["a1_grid"]) == 25 and data["a1_grid"].selected.sum() == 1
    assert len(data["a2_seeds"]) == 15 and len(data["a3_seeds"]) == 10
    assert len(data["a4_importance"]) == 20 and len(data["b4_maps"]) == 15
    data["b4_payload"] = torch.load(source("part_b/outputs/b4/gradcam.pt"), map_location="cpu", weights_only=True)
    assert data["b4_payload"]["row_ids"].tolist() == [486, 279, 413, 53, 276]
    assets.baseline_diagram(output)
    assets.grid_heatmap(data["a1_grid"], output)
    assets.a1_curves(data["a1_selected"], data["a1_refit"], data["a1_test_monitor"], output)
    a2_limits = assets.regression_curve_limits({
        variant: data[f"a2_{variant}_history"]
        for variant in ("baseline", "continuous_only", "sigmoid")})
    assets.regression_epoch_curves({"Baseline": data["a2_baseline_history"],
        "Continuous": data["a2_continuous_only_history"]}, output, "a2_categorical", y_limits=a2_limits)
    assets.regression_epoch_curves({"ReLU": data["a2_baseline_history"],
        "Sigmoid": data["a2_sigmoid_history"]}, output, "a2_activation", y_limits=a2_limits)
    assets.regression_epoch_curves({"Baseline": data["a3_baseline_history"],
        "Wide + deep": data["a3_selected_history"]}, output, "a3_learning_curves")
    data["a3_comparison_limits"] = assets.paired_regression_limits(data["a3_seeds"], "baseline", "wide_deep")
    assets.paired_results(data["a3_seeds"], output, "a3_comparison", "baseline", "wide_deep",
        ["Baseline (ReLU)", "Wide + deep"], y_limits=data["a3_comparison_limits"])
    for model in ("baseline", "wide_deep"):
        assets.importance(data["a4_importance"][data["a4_importance"].model == model], output,
            f"a4_{model}", "A1 baseline" if model == "baseline" else "A3 wide-and-deep (λ=2)")
    for model, key in (("b1", "learning_rate"), ("b2", "batch_size")):
        assets.classification_curves(data[f"{model}_curves"], data[f"{model}_selection"], key,
            output, model, validation_only=True)
    assets.b3_curves(data["b3_curves"], output)
    assets.gradcam_grid(data["b4_payload"], output)
    return data


def part_a(directory, name, slug, data):
    report = Report(directory, "A", name)
    a1 = data["a1_summary"]
    chosen, default = a1["selected"], a1["default_baseline"]
    report.page("Regression: A1–A4", first=True)
    report.section("Generative AI declaration")
    report.text("I used OpenAI Codex / ChatGPT for planning, code generation/debugging, experiment execution, analysis, "
        "figures, report drafting and submission packaging. Outputs were checked against the brief/slides, Q&A and library source, behavioral "
        "tests, CUDA checkpoint restoration and artifact/hash audits. FeatureAblation scores were independently "
        "aggregated and checked by manual replacements. All results below come from completed saved experiments. "
        "I remain responsible for reviewing and understanding the submitted work.")
    report.section("Data and common training protocol")
    report.text("Chronological partitions: 2017–2020 train (87,370), 2021 validation (29,057), 2022 test (26,702); "
        "2023 is excluded. Inputs are month, town, flat model type, storey range and six continuous variables; year, "
        "address and station name are excluded. Category encoders and standardization statistics are fitted on training only "
        "(sample SD for continuous inputs; population SD for price), then refitted on 2017–2021 for combined-data runs. "
        "Adam uses LR 0.01, batch 512, "
        "standardized-target MSE. Selection uses seed 42, at most 20 epochs, patience 4 and RMSE improvement 0.0001 SGD.")
    report.section("A1. Baseline diagram, parameters and Ray grid search")
    report.figures(["a1_baseline_diagram"], caption="Figure A1a. Supplied baseline (H=64, d=8); the output before inverse scaling is linear.")
    report.text("P = d(13+27+44+18) + (4d+10)H + 1. Default: 816 embedding + 2,496 first-linear + 128 LayerNorm + "
        "65 output-linear = 3,505 trainable parameters. Unknown category ID 0 is an ordinary learned embedding.", size=10)
    report.figures(["a1_grid"], caption="Figure A1b. Minimum actual validation RMSE per fresh Ray Tune trial [1]; outlined cell is selected.")
    report.text(f"All H in {{16,32,64,128,256}} × d in {{2,4,8,12,16}} combinations completed. Select H=256, d=4: "
        f"{chosen['trainable_parameters']:,} parameters, validation RMSE SGD {number(chosen['validation_rmse'])}, "
        f"R² {chosen['validation_r2']:.6f}, epoch {chosen['best_epoch']}. This is "
        f"{100*(1-chosen['validation_rmse']/default['validation_rmse']):.2f}% lower RMSE than the default "
        f"(SGD {number(default['validation_rmse'])}). Exact ties favor fewer parameters, smaller H, then smaller d.")

    report.page("A1 · learning curves, final refit and interpretation")
    report.figures(["a1_selected_curves", "a1_refit_training"],
        caption="Figure A1c–d. Selected trial: train/validation RMSE on the same axes. A1 refit: train/test RMSE on the same axes, as the brief requests. Dashed lines mark epochs 6 and 4; the test-stopping history is disclosed.")
    last = data["a1_selected"].iloc[-1]
    width_means = data["a1_grid"].groupby("hidden_width").validation_rmse.mean()
    d2 = data["a1_grid"].query("hidden_width == 256 and embedding_dimension == 2").iloc[0]
    narrow = data["a1_grid"].query("hidden_width == 16 and embedding_dimension == 4").iloc[0]
    report.section("Evidence for the selected configuration")
    report.text(f"Mean best validation RMSE across embedding sizes falls from SGD {number(width_means.loc[16])} "
        f"at H=16 to {number(width_means.loc[256])} at H=256. The measured winner is H=256,d=4: "
        f"{number(chosen['validation_rmse'])} SGD, versus {number(d2.validation_rmse)} for H=256,d=2. "
        "Increasing d is not uniformly beneficial at each width; d=4 also beats every larger d at H=256. "
        "This supports the chosen 7,065-parameter configuration within this grid, rather than a universal optimum.")
    report.text(f"The selected trial's epoch-6 training RMSE is SGD {number(chosen['train_rmse'])}, versus validation "
        f"{number(chosen['validation_rmse'])}. At epoch 10, training falls to {number(last.train_rmse)} while validation "
        f"rises to {number(last.validation_rmse)}; every later validation checkpoint is worse than epoch 6. "
        "This is consistent with late overfitting and supports restoring epoch 6. The refit similarly improves "
        "training RMSE from 41,239.37 to 39,074.75 after epoch 4 while test RMSE worsens from 96,901.64 to 99,668.21 SGD.")
    report.text(f"At fixed d=4, H=16 has training RMSE {number(narrow.train_rmse)} SGD, only 181.96 above H=256. "
        "These similar training errors do not establish substantial underfitting. High test error or model width relative "
        "to sample count cannot alone diagnose underfitting or rule out overfitting. Optimization fluctuations "
        "and temporal shift may contribute; one seed and one validation year do not establish ranking stability.")
    report.section("Fresh combined-data refit and final test result")
    final = a1["final_test_metrics"]
    report.table(["H / d", "Parameters", "E_val / E*", "2022 test RMSE (SGD)", "Test R²"],
        [["256 / 4", "7,065", "6 / 4", number(final["rmse"]), f"{final['r2']:.6f}"]], widths=[.8, .9, .85, 1.65, .9])
    report.text("A fresh seed-42 model was trained on all 116,427 combined 2017–2021 rows using the saved refit "
        "preprocessing. As explicitly authorized by the professor, test RMSE monitored this refit's stopping under "
        "the same 20/4/0.0001 rule. Eight epochs completed; the restored best checkpoint is E*=4. E_val=6 belongs to "
        "the separate original-split selection run. Final scores above cover every test row.")
    report.section("Test-assisted stopping limitation")
    report.text("The 2022 test set selected A1's stopping checkpoint, so its final score is not an untouched-holdout "
        "estimate. A2/A3 inherit the selected four-epoch budget. The higher test RMSE can reflect temporal "
        "distribution changes and model error; it is not proof of overfitting alone. Validation and test scores use "
        "different fitting periods and transforms, so their difference does not isolate the benefit of refitting. "
        "Q&A's no-test-epoch plotting answer addresses A2/A3; their test scores appear only in final tables. "
        "A1 retains its expressly requested train/test curve and separately authorized test-assisted stopping.")
    report.reference("Sources: assignment brief pp. 2–4, slides pp. 2–3; supplied professor clarifications on plots and refit stopping.")
    report.reference("[1] Ray Tune 2.58.0 documentation: grid_search (Cartesian-product search).",
        "https://docs.ray.io/en/latest/tune/api/doc/ray.tune.grid_search.html")

    report.page("A2 · controlled ablations across five seeds")
    report.text("All three variants were initialized fresh for seeds 42–46 and trained for exactly four epochs on "
        "2017–2021. They reuse A1's refit preprocessing, H=256, LR 0.01, batch 512 and the same seeded row order. "
        "All-input variants use d=4 and matched initial weights. Each run receives one final test evaluation; "
        "there is no A2 validation/test stopping. SD is sample SD (ddof=1), not a confidence interval. "
        "Separately, the FAQ epoch diagnostics use fresh seed-42 original-split runs for four fixed epochs: "
        "2017–2020 training, 2021 validation and training-only transforms. They do not select settings or use test data.", size=10)
    report.table(["Model", "Parameters", "Test RMSE (SGD): mean ± SD", "Test R²: mean ± SD"],
        summary_rows(data["a2_metrics"], {"baseline": "Baseline (ReLU)", "continuous_only": "Continuous only", "sigmoid": "All inputs (Sigmoid)"}, "variant"),
        widths=[1.3, .65, 1.85, 1.6])
    report.section("A2a. Remove all four categorical embeddings")
    report.figures(["a2_categorical"], caption="Figure A2a. Categorical ablation: original-split seed-42 train/validation RMSE and R² against epochs. Zoomed y axes share ranges with A2b; all points are shown. Final test results are in the five-seed table.")
    report.text("Only six continuous inputs enter Linear(6,256) → LayerNorm → ReLU → Linear(256,1), giving 2,561 "
        "parameters. Mean test RMSE rises by 5,824.79 SGD (5.69%), worsening in four of five seeds; mean R² falls "
        "by 0.044107. The diagnostic also has higher train/validation error, consistent with reduced fit. "
        "Removal reduces capacity by 63.75%, confounding information loss with fewer parameters.", size=10)
    report.section("A2b. Replace hidden ReLU with Sigmoid")
    report.figures(["a2_activation"], caption="Figure A2b. Activation ablation: matched initialization, train/validation RMSE and R² against epochs. Zoomed y axes share ranges with A2a. Neither diagnostic uses test metrics or changes the four-epoch budget.")
    report.text("Sigmoid lowers mean RMSE by only 605.94 SGD (0.59%) and improves three of five seeds; paired RMSE "
        "difference SD is 2,024.47 SGD. Diagnostic validation RMSE rebounds from 60,460.29 to 68,030.75 SGD "
        "between epochs 3–4, showing instability under this budget. Sigmoid's bounded output and small derivatives "
        "can slow optimization relative to ReLU; saturation was not measured. The small, "
        "sign-changing final gain does not establish an advantage. The short budget was chosen for ReLU "
        "and inherited from test-assisted A1 stopping; no settings were retuned.", size=10)

    report.page("A3 · wide-and-deep comparison")
    report.text("Both branches receive the same 22-vector of shared embeddings and continuous inputs: "
        "ẑ = MLP(x) + λ·Linear(22,1)(x); apply the target inverse transform once after the sum. The MLP is "
        "A1's Linear(22,256) → LayerNorm → ReLU → Linear(256,1). The linear branch adds 23 parameters, "
        "for 7,088 total. Keep all A1 settings fixed and select λ using original train/validation partitions, "
        "seed 42 and validation-RMSE stopping. Exact ties favor smaller λ.")
    report.table(["λ", "Saved / completed epoch", "Train RMSE (SGD)", "Validation RMSE (SGD)", "Validation R²"],
        [[f"{row['lambda']:g}", f"{int(row['best_epoch'])} / {int(row['epochs_completed'])}", number(row["train_rmse"]),
          number(row["validation_rmse"]), f"{row['validation_r2']:.6f}"] for row in data["a3_lambda"].to_dict("records")],
        widths=[.4, 1.2, 1.2, 1.35, .85], highlight=4)
    report.text("Select λ=2: validation RMSE improves by 721.46 SGD (1.23%) over λ=0, with saved epoch 3. "
        "λ=0 reproduced A1's trained baseline tensors and scores exactly. For the controlled final comparison, "
        "fresh seeds 42–46 use the saved combined-data transforms and A1's four-epoch budget, matching baseline "
        "MLP initialization and row order; the five A2 baseline results are reused.")
    report.table(["Model", "Parameters", "Test RMSE (SGD): mean ± SD", "Test R²: mean ± SD"],
        summary_rows(data["a3_metrics"], {"baseline": "Baseline (ReLU)", "wide_deep": "Wide + deep, λ=2"}, "model"),
        widths=[1.3, .65, 1.85, 1.6])
    report.figures(["a3_learning_curves"], caption="Figure A3. Original-split, seed-42 baseline (λ=0) and selected wide-and-deep (λ=2): train/validation RMSE and R² against epochs. Five-seed final test scores appear only in the table.")
    comparison_limits = data["a3_comparison_limits"]
    report.text("Mean RMSE improves 1,014.15 SGD (0.99%), in four of five seeds; paired difference SD is "
        "2,811.40 SGD. Mean R² improves 0.006770. Seed 45 worsens by 3,246.47 SGD, so the gain is modest "
        "relative to seed variability and is not a statistical-significance claim. The packaged paired-comparison "
        f"plot uses zoomed axes: RMSE SGD {comparison_limits['rmse'][0]*1000:,.0f}–{comparison_limits['rmse'][1]*1000:,.0f} "
        f"and R² {comparison_limits['r2'][0]:.3f}–{comparison_limits['r2'][1]:.3f}; all points and sample-SD bars are visible.")
    report.text("A direct linear route can fit additive associations and shorten the gradient path to embeddings. "
        "The earlier/lower-error validation checkpoint is consistent with altered optimization or generalization, "
        "but branch-specific effects were not measured. Any positive λ can be absorbed into unrestricted linear "
        "weights/bias, so positive candidates share a function class but differ in effective initialization and "
        "finite-time optimization. Confounds include 23 extra parameters, jointly learned embeddings, one-seed "
        "λ selection and the short test-assisted epoch budget. The evidence does not isolate one mechanism.")

    report.page("A4 · FeatureAblation in Singapore dollars")
    report.text("Captum FeatureAblation [2] explains the matching seed-42, epoch-4 A1/A3 checkpoints on all "
        "26,702 test rows and all ten original inputs. For feature j, aᵢⱼ = f_SGD(xᵢ) − f_SGD(xᵢ with j replaced); "
        "importance is meanᵢ|aᵢⱼ|. Categorical IDs are replaced before embedding lookup, with ten distinct mask IDs. "
        "Shared combined-training (2017–2021) references are month 7, SENGKANG, flat type ‘4 ROOM, Model A’, "
        "storey ‘04 TO 06’, and each continuous variable's combined-training mean (standardized zero).")
    report.figures(["a4_baseline"], caption="Figure A4a. All ten baseline scores, ranked descending. Values are mean absolute prediction changes in SGD.")
    report.figures(["a4_wide_deep"], caption="Figure A4b. All ten wide-and-deep scores under the same replacement references and saved preprocessing.")
    report.text("Baseline top three: floor area (71,537.72 SGD), remaining lease (46,187.61), distance to Dhoby "
        "Ghaut (41,163.95). Wide-and-deep: floor area (73,119.34), remaining lease (48,364.21), flat model "
        "type (42,280.28). Flat type and Dhoby distance swap third/fourth; eight other ranks match. The similar "
        "rankings suggest broadly similar reliance under these references, with jointly trained branch/embedding differences.")
    report.text("These are reference-dependent model sensitivities, not causal price effects. Replacing one "
        "correlated input can create unrealistic feature combinations; mean absolute scores hide signs and sample "
        "variability. Nonlinear effects need not add to the total prediction change, and larger scores do not imply "
        "better accuracy. Only one matched seed was explained. Independent global aggregation and manual "
        "replacements matched, with maximum manual difference 0 SGD and unchanged checkpoints.")
    report.reference("[2] Captum 0.9.0 documentation: FeatureAblation (replacement references and feature masks).",
        "https://captum.ai/api/feature_ablation.html")
    return report.save(slug)


def part_b(directory, name, slug, data):
    report = Report(directory, "B", name)
    comparison = data["b_comparison"].set_index("architecture")
    report.page("Classification: B1–B4", first=True)
    report.section("Generative AI declaration")
    report.text("I used OpenAI Codex / ChatGPT for planning, code generation/debugging, experiment execution, "
        "analysis, figures, report drafting and submission packaging. Verification included assignment/Q&A/library-source checks, "
        "behavioral tests, CUDA checkpoint restoration, saved-prediction metric/BCE checks, independent "
        "parameter/FLOPs formulas and artifact hashes. Grad-CAM tensors were reproduced by independent "
        "PyTorch autograd. All results below come from completed saved experiments. I remain responsible for "
        "reviewing and understanding the submitted work.")
    report.section("Data, architecture interpretation and common protocol")
    report.text("Use the supplied 4,708/524/624 train/validation/test partitions and 64×64 grayscale images. "
        "Divide uint8 by 255 once, then normalize with training-pixel mean 0.5716994213 and population SD "
        "0.1740975398. Q&A confirms 0=healthy, 1=pneumonia. Retain these original-training normalization "
        "statistics during final refits, following the brief; the FAQ only requires preprocessing refits for A1. "
        "Eight distinct images overlap train/validation; three extra copies occur within test. No exact train/test "
        "or validation/test matches were found, but patient independence cannot be checked without IDs.")
    report.text("Convolutions have no bias, then BN/ReLU; FC layers have bias; residual sums use ReLU. "
        "Q&A confirms SAME padding=kernel_size//2, including stride 2. It corrects the caption's 'all layers': "
        "B1/B2 final FC1 has an unrestricted logit with BCEWithLogitsLoss, no ReLU/dropout or prior sigmoid. "
        "Apply the same binary-loss interpretation to B3 (not separately specified in that answer). "
        "Sigmoid≥0.5 predicts class 1; B1 dropout 0.5 follows both hidden FC ReLUs, B2/B3 use none.", size=10)
    report.text("Selection: Adam, seed 42, 20-epoch cap, patience 4, minimum validation-loss improvement 0.0001; "
        "save lowest validation error, then lower loss, then earlier epoch. Candidate ties prefer smaller rate/batch. "
        "Per Q&A, initialize each final model fresh and refit on all 5,232 train+validation images for its selected "
        "9/13/9 epochs (B1/B2/B3), with frozen hyperparameters, then evaluate test once. Fixed refits use no "
        "validation/test stopping. No augmentation, weighting or resampling; metrics use evaluation mode.", size=10)
    report.section("B1. AlexNet-like learning-rate selection")
    report.text("Figure B1 is implemented as 5×5/16 → max pool → 5×5/32 → max pool → 3×3/64 → max pool "
        "→ average output 4×4 → FC64 → FC32 → FC1, with dropout 0.5 only in hidden FC layers. "
        "All five rates use batch 128 and fresh initialization; 99,569 trainable parameters.")
    report.table(["Learning rate", "Best epoch", "Val. BCE loss", "Val. error", "Sensitivity", "Specificity"],
        selection_rows(data["b1_selection"], "learning_rate"), widths=[1, .85, 1.15, 1, 1.05, 1.05], highlight=2)
    report.text("Select LR=0.001, epoch 9: 11/524 validation errors (97.9008% accuracy), versus 16 for the "
        "next-best rate 0.0001. Every metric in a row belongs to that one saved checkpoint; stopping loss "
        "and error selection can prefer different epochs. The frozen learning rate supplies B2 and B3.")
    report.section("Why sensitivity and specificity are necessary")
    report.text("Validation is 74.24% positive: predicting pneumonia for everyone would yield 74.24% accuracy, "
        "100% sensitivity and 0% specificity. Sensitivity=TP/(TP+FN) measures detected positives; "
        "specificity=TN/(TN+FP) measures correctly rejected negatives. LR 0.01 has higher sensitivity "
        "than 0.001 but far lower specificity (80.74% versus 97.04%), a trade-off hidden by one aggregate score.")

    report.page("B1 · learning curves and test; B2 selection table")
    report.figures(["b1_loss", "b1_accuracy"], caption="Figure B1a–b. Validation loss and accuracy against epoch for all five learning rates. Stars mark saved checkpoints. Accuracy axes start at 70%; curves end at actual stopping epochs.")
    report.text("LR 1e-5 learns slowly: loss keeps falling at the 20-epoch cap, but best accuracy is only 90.46%. "
        "1e-4 improves more gradually to 96.95% at epoch 19. 1e-3 reaches minimum loss at epoch 8, but epoch 9 "
        "has lower error; patience stops at 12. Its selected train/validation losses are 0.035620/0.064911. "
        "1e-2 worsens in both train/validation loss and stops at 5, indicating an unfavorable optimization "
        "trajectory rather than a simple overfitting pattern. 1e-1 shows reversals and reaches only 95.42% "
        "best accuracy. Finite-budget stopping is not proof of global convergence; BN/dropout causes were not isolated.")
    report.section("B1. One final test evaluation")
    test_table(report, comparison.loc["b1"])
    b1_final = comparison.loc["b1"]
    b1_gap = data["b1_summary"]["selected"]["validation_accuracy"] - b1_final.accuracy
    report.text(f"The fresh nine-epoch combined-data refit misses {int(b1_final.fn)} positives and labels "
        f"{int(b1_final.fp)}/234 negatives positive. Test accuracy is {abs(100*b1_gap):.2f} percentage points "
        f"{'below' if b1_gap >= 0 else 'above'} selection validation accuracy; test specificity is "
        f"{percent(b1_final.specificity)}. Prevalence "
        "changes affect accuracy but cannot alone explain lower specificity. Supplied overlap/dataset differences "
        "and refitting may contribute; their effects are not isolated. No settings or threshold were changed after test.")
    report.section("B2. VGG-like batch-size selection, frozen LR=0.001")
    report.table(["Batch size", "Best epoch", "Val. BCE loss", "Val. error", "Sensitivity", "Specificity"],
        selection_rows(data["b2_selection"], "batch_size"), widths=[1, .85, 1.15, 1, 1.05, 1.05], highlight=0)
    report.text("Batch 16 wins uniquely with 11 errors, versus 15 for 32/64/256 and 25 for 128. Batch 32 has "
        "higher specificity, but the prescribed lowest-error criterion selects batch 16. LR/batch are frozen for B3.")

    report.page("B2 · batch-size curves, interpretation and final test")
    report.text("The VGG-like model has two 3×3 Conv/BN/ReLU layers per width 8,16,32, each pair followed by "
        "max pooling, then global average pooling and FC1: 18,185 parameters, no dropout. All candidates "
        "share initial tensors and the underlying shuffled row sequence; batch grouping changes.")
    report.figures(["b2_loss", "b2_accuracy"], caption="Figure B2a–b. Validation loss and accuracy against epoch for all five batch sizes. Stars mark error-selected checkpoints, not necessarily minimum loss; curves end at actual stopping epochs.")
    report.text("Batch 16 reaches minimum validation loss at epoch 9 (0.066596), but lowest error at 13 "
        "(0.069390); loss patience expires at 13. Its train/validation accuracies are 99.11%/97.90%. "
        "Batch 32 has a large epoch-6 loss spike/accuracy drop and then recovers, selecting 13. Batch 64 "
        "selects 6, then deteriorates through stopping at 10. Batch 128 begins near the positive-class majority "
        "rate, selects 5 and stops at 9. Batch 256 starts slowly, selects 16, and exhausts patience at the "
        "20-epoch cap. Simultaneous train/validation reversals do not justify attributing every spike to overfitting.")
    report.section("Batch size also changes the update budget")
    report.text("For batches 16/32/64/128/256, updates per epoch are 295/148/74/37/19. Selected checkpoints "
        "receive 3,835/1,924/444/185/304 updates; batch 16 receives 12.6× as many as batch 256. "
        "Smaller batches alter gradient/BN estimates, while larger batches make fewer updates per epoch. "
        "Gradient variance and BN effects were not separately measured. This one-seed comparison selects "
        "the required training recipe, not a universal batch-size ranking or an equal-update experiment.")
    report.section("B2. One final test evaluation")
    test_table(report, comparison.loc["b2"])
    b2_final = comparison.loc["b2"]
    b2_selected = data["b2_summary"]["selected"]
    report.text(f"The fresh thirteen-epoch combined-data refit misses {int(b2_final.fn)} positives and makes "
        f"{int(b2_final.fp)} false positives. Specificity is {percent(b2_final.specificity)}, versus "
        f"{percent(b2_selected['validation_specificity'])} for the selection checkpoint on validation. "
        "Conditional specificity is not explained by prevalence alone. The distinct refit weights, "
        "supplied train/validation overlap and absent patient IDs limit "
        "interpretation; partitions and frozen choices were preserved.")

    report.page("B3 · compact ResNet and architecture comparison")
    report.text("Use LR=0.001 and batch 16 exactly from B1/B2. Figure B3 has an initial 3×3/16 stage/max pool, "
        "then residual blocks at 16/32/64 channels; downsampling blocks use stride-2 main convolutions and "
        "1×1 projections, with ReLU after sums. Global averaging feeds FC1. Seed 42 selects epoch 9: "
        "validation accuracy 97.5191%, loss 0.061988; loss stopping ends at 13.")
    report.figures(["b3_loss", "b3_accuracy"], caption="Figure B3a–b. Training/validation curves; dashed line marks selected epoch 9. The growing later gap is consistent with overfitting; earlier simultaneous fluctuations have unisolated causes.")
    b3_final = comparison.loc["b3"]
    report.text("At selection epoch 9, train/validation accuracies are 99.43%/97.52%; by epoch 13 training reaches "
        "99.87% while validation does not improve. A fresh nine-epoch combined-data refit then detects "
        f"{int(b3_final.tp)}/390 positives and misclassifies {int(b3_final.fp)}/234 negatives; specificity is "
        f"{percent(b3_final.specificity)}. The table uses all three fresh refits' saved final tests without re-evaluation.")
    report.table(["Model", "Accuracy", "Sensitivity", "Specificity", "Class. error", "Parameters", "FLOPs/image"],
        [[row.model.replace("-like", ""), percent(row.accuracy), percent(row.sensitivity), percent(row.specificity),
          percent(row.classification_error), f"{int(row.trainable_parameters):,}", f"{int(row.inference_flops):,}"]
         for row in data["b_comparison"].itertuples()], widths=[1.5, .95, 1, 1, 1, 1, 1.35], size=9)
    report.text("FLOPs use one float32 1×1×64×64 evaluation input: 2 per convolution/FC multiply-add plus "
        "one per residual addition (28,672 for B3), checked against PyTorch profiler and executed-shape formulas [2]. "
        "Exclude BN, activations, pooling, FC bias addition, normalization/transfers, sigmoid/threshold and "
        "training work. These consistent scoped estimates are neither total scalar operations nor measured latency.", size=10)
    report.text("B1 retains a 4×4 spatial FC head and dropout, but has the most parameters/FLOPs (67,713 head "
        "parameters). B2 stacks small kernels with narrower channels/global averaging: fewest parameters/FLOPs, "
        "but compressed spatial information and no shortcuts. B3 provides residual gradient routes intended "
        "to ease deeper optimization [1], but wider channels/projections cost more than B2. Shortcut "
        "benefits were not isolated.")
    best_accuracy = comparison.accuracy.idxmax()
    best_specificity = comparison.specificity.idxmax()
    ranking = (f"{best_accuracy.upper()} has the highest observed accuracy and specificity. "
        if best_accuracy == best_specificity else
        f"{best_accuracy.upper()} has the highest observed accuracy; {best_specificity.upper()} the highest specificity. ")
    report.text(ranking + "High sensitivity with lower specificity indicates many healthy images "
        "classified positive. One seed, different batches/selected epochs/update budgets, B1-only dropout and inherited "
        "tuning prevent attributing performance differences to architecture alone. The table does not trigger retuning.", size=10)
    report.reference("[1] He et al. (2016), Deep Residual Learning for Image Recognition.", "https://arxiv.org/abs/1512.03385")
    report.reference("[2] PyTorch 2.11 profiler documentation (formula-based convolution/matrix FLOP estimates).",
        "https://docs.pytorch.org/docs/2.11/profiler.html")

    report.page("B4 · Grad-CAM on five shared random positive examples")
    report.text("NumPy PCG64 seed 42 uniformly samples five label-1 test rows without replacement; IDs were "
        "saved before maps, with no visual/prediction filtering. The fresh combined-data final checkpoints use main-path "
        "convolutions conv3.conv (B1), conv6.conv (B2), block64.conv2.conv (B3), all under features, "
        "before BN/ReLU; B3's shortcut is not separately mapped. Captum "
        "targets scalar pneumonia logits (target=None), evaluation mode with gradients enabled: spatially "
        "average gradients, weight channels, sum and ReLU [3,4]. Native grids 16×16/16×16/8×8 are bilinearly "
        "upsampled to 64×64 and divided by each map's own maximum.", size=10)
    report.figures(["b4_grid"], caption="Figure B4. Original pixels at left; shared inferno and opacity=0.55×heat, no flips. p is the saved final-test probability. Each map is normalized separately; colors are relative, not absolute evidence across models.")
    report.text("486: B1 is lower central; B2 peaks at the upper edge; B3 highlights upper image-right and lower "
        "image-left. 279: B1 is lower central; B2/B3 favor lower boundaries. 413: B1 is central, B2 scattered, "
        "B3 broadly lower image-left. 53: B1 is lower central; B2/B3 include the lower boundary. 276: B1 "
        "overlaps central brightness; B2 is peripheral/scattered, B3 favors lower corners. Directions refer "
        "to the displayed image. Bright regions can also contain normal structures.", size=10)
    frame_mass = data["b4_geometry"].set_index("architecture").mean_outer_8_pixel_frame_mass_fraction
    report.text("Mean CAM mass in the outer eight-pixel frame is "
        + "/".join(f"{100*frame_mass[model]:.2f}%" for model in ("b1", "b2", "b3"))
        + " (B1/B2/B3), supporting this descriptive spatial comparison; it does not establish lesion localization.", size=9.8)
    maps = data["b4_maps"]
    least = maps.loc[maps.saved_final_test_probability.idxmin()]
    positive_counts = "/".join(str(int((maps.loc[maps.architecture == architecture,
        "saved_predicted_label"] == 1).sum())) for architecture in ("b1", "b2", "b3"))
    report.text(f"Positive predictions among the five rows: {positive_counts} for B1/B2/B3. The lowest saved "
        f"probability is {least.architecture.upper()} row {int(least.row_id)}: "
        f"{least.saved_final_test_probability:.4f}. Coarse maps, "
        "normal structures within bright regions and absent lesion annotations prevent validating localization "
        "or causal border reliance. Five positive examples do not assess healthy/false-positive behavior. "
        "Duplicate selected-input padding matches final-test CUDA batch sizes; saved logits and independent "
        "autograd maps match exactly, with unchanged models and no full-test performance evaluation.", size=9.8)
    report.reference("[3] Selvaraju et al. (2017), Grad-CAM: Visual Explanations from Deep Networks.", "https://arxiv.org/abs/1610.02391")
    report.reference("[4] Captum 0.9.0 LayerGradCam documentation; positive-attribution ReLU is explicit.",
        "https://captum.ai/api/layer.html#captum.attr.LayerGradCam")
    return report.save(slug)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="YourName", help="Author display name; replace placeholder before submission")
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    name = args.name.strip()
    slug = re.sub(r"[^\w-]+", "_", name, flags=re.UNICODE).strip("_")
    if not name or not slug:
        raise ValueError("Author name must contain at least one letter or digit")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    for path in ("assignment.pdf", "assignment_slides.pdf", "Q&A.md", "clarifications.md", "docs/GENAI_USAGE.md",
                 "docs/ASSIGNMENT_DECISIONS.md", "docs/FINAL_REQUIREMENTS_RECHECK.md", "docs/FAQ_COMPLIANCE_REVIEW.md"):
        source(path)
    for script in ("build_reports.py", "report_assets.py", "report_layout.py", "verify_reports.py"):
        source(f"scripts/{script}")
    for experiment in ("A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"):
        source(f"docs/{experiment}_EXPERIMENT.md")
    data = prepare_assets(directory)
    outputs, layouts = {}, {}
    for part, builder in (("A", part_a), ("B", part_b)):
        path, layout = builder(directory, name, slug, data)
        outputs[part] = str(path.relative_to(directory))
        layouts[part] = layout
        print(f"Built Part {part}: {path.name} (5 pages)", flush=True)
    manifest = {"step": 14, "status": "built", "name": name, "name_placeholder": name == "YourName",
        "reports": outputs, "pages_per_report": 5, "inputs": INPUTS, "layout": layouts,
        "source_experiments_reused": ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4"],
        "new_training_runs": 0, "new_model_inference": 0,
        "requirements_revision": "2026-10-10 instructor Q&A",
        "part_b_final_protocol": "fresh train+validation refits, fixed selected epoch counts, one test pass per final refit",
        "a2_diagnostics": "original-split seed42, four epochs, train/validation RMSE and R2; distinct from five-seed combined refits",
        "a1_refit_curve_values": [{"epoch": int(row.epoch), "train_rmse_sgd": float(row.train_rmse),
            "test_rmse_sgd": float(data["a1_test_monitor"].loc[data["a1_test_monitor"].epoch == row.epoch, "rmse"].item())}
            for row in data["a1_refit"].itertuples()],
        "test_epoch_plot_scope": "A1 train/test RMSE explicitly required in brief; Q&A no-test epoch answer applies A2/A3",
        "builder": "scripts/build_reports.py", "python_executable": sys.executable,
        "fonts": "Windows Arial regular/bold embedded; plot text embedded as TrueType"}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
