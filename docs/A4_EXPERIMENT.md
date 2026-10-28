# A4 FeatureAblation — completed October 3, 2026

FeatureAblation is complete for **all ten original inputs and all 26,702 test rows**, for the final A1 baseline and A3 wide-and-deep model. Both are the matching **seed-42, epoch-4** combined-data checkpoints. The attribution computation took approximately **1.83 seconds** on CUDA, excluding loading, plotting, and the later audit. Step 8 is complete; Part A's experiments A1–A4 are complete. Both five-page final reports are complete; package verification remains step 15.

## Models and reference values

The exact saved 2017–2021 preprocessing is shared by both models. The A1 baseline has H=256, d=4 and 7,065 parameters; A3 has the same MLP and shared embeddings, lambda 2, and 7,088 parameters. Model weights remained fixed throughout explanation.

The four categorical inputs are ablated as integer category IDs before embedding lookup, giving one attribution for each original variable. Six separate continuous columns complete the ten inputs. Distinct mask IDs 0–9 prevent accidental grouping across the two input tensors.

References are shared and use only the 116,427 refit training rows:

| Categorical input | Training mode | Encoded ID | Training count |
| --- | --- | ---: | ---: |
| Month | 7 | 7 | 11,520 |
| Town | SENGKANG | 21 | 9,853 |
| Flat model type | 4 ROOM, Model A | 19 | 28,725 |
| Storey range | 04 TO 06 | 2 | 26,932 |

Each continuous reference is its saved training mean, represented by standardized **0**. Approximate raw means are 0.815478 for distance to the nearest station, 11.163990 for distance to Dhoby Ghaut, 0.0174987 for degree centrality, 0.00660629 for eigenvector centrality, 74.895813 remaining lease years, and 97.854713 m² floor area. Exact tensors/statistics and category-mode tie handling are saved in `references.pt`/JSON; no test data were used to choose references.

For each sample and feature, the signed result is:

```text
attribution_SGD = price(original inputs) - price(one feature replaced by its reference)
global_importance = mean(abs(attribution_SGD)) over every test sample
```

The explained forward function applies the saved target inverse transform **once**, after the baseline output or wide-and-deep branch sum. Network computation remains float32, while the final SGD inverse transform and subtraction use float64 to reduce cancellation. The absolute value is taken before averaging; signed per-row values are retained. The [Captum documentation](https://captum.ai/api/feature_ablation.html) supports explicit replacement baselines and feature masks; the installed 0.9.0 source was also inspected.

## Top-three features

These values are mean absolute prediction changes in SGD under the specified reference scheme.

| Rank | A1 baseline | Importance (SGD) | A3 wide-and-deep | Importance (SGD) |
| --- | --- | ---: | --- | ---: |
| 1 | Floor area | **71,537.72** | Floor area | **73,119.34** |
| 2 | Remaining lease years | **46,187.61** | Remaining lease years | **48,364.21** |
| 3 | Distance to Dhoby Ghaut | **41,163.95** | Flat model type | **42,280.28** |

For example, the baseline's floor-area score averages the absolute change when each test property's floor area is replaced by the training mean of approximately 97.85 m², keeping the other inputs fixed.

All ten scores and comparisons are in [analysis.md](../part_a/outputs/a4/analysis.md) and the CSV tables. The models share two of their top-three features, and their overall ranking correlation is **0.9879**. Flat model type and distance to Dhoby Ghaut swap third/fourth place; the other eight ranks match. The largest absolute importance change is flat model type, increasing **SGD 2,949.88 (7.50%)**. Floor area increases SGD 1,581.61, remaining lease increases SGD 2,176.60, and town decreases SGD 2,893.87.

These results suggest similar model reliance under the shared references. Changes may reflect the extra linear branch and the jointly trained MLP/embeddings; the comparison does not isolate the linear branch's contribution. Larger attribution scores do not establish higher prediction accuracy.

## Manual check and production verification

Both models' first 512-row attribution batches were compared with independent manual feature replacements for **all ten columns**. The audit repeated manual comparisons on the first and last batches, checking 512 + 78 rows per model. The maximum difference was **SGD 0.00** for both models.

For the first test sample, original zero-based CSV row ID **116427**, replacing floor area gives:

| Model | Captum signed attribution (SGD) | Manual original-minus-replaced price (SGD) |
| --- | ---: | ---: |
| Baseline | −200,394.63 | −200,394.63 |
| Wide-and-deep | −176,860.94 | −176,860.94 |

The negative sign indicates that replacing this sample's floor area by the training mean increased the fitted model's prediction.

The complete suite passed **57 tests**, including six A4 tests covering training-only references/ties, ten-feature masks, analytic embedded-model SGD effects, categorical grouping, partial batches, perturbation batching, model-mode preservation, actual-model manual checks, target mean/scale behavior, mean-absolute aggregation, stable ranking, and paired-checkpoint guards.

The CUDA production audit independently recomputed all global means from the full saved matrices using NumPy, checked every row ID and all rankings/tables, verified zero attribution where inputs already equalled references, restored both checkpoints for manual differences, and confirmed unchanged model states plus all supplied/A1/A2/A3 checkpoints. It used the existing performance records; the new forward passes computed explanations rather than new test-performance scores.

## Limitations

The references determine the sensitivity question: different valid references may change scores. Replacing one correlated input while keeping other inputs fixed can create unrealistic combinations, such as a typical floor area paired with an atypical flat type, or mean station distance paired with the original location-centrality values. Mean absolute scores conceal signs and sample variability. The nonlinear model's separate feature effects need not add up to the prediction difference from replacing all inputs together. Interpret these results as fitted-model explanations rather than causal price effects.

Only one matched seed was explained, so attribution stability across seeds remains unmeasured. The baseline checkpoint and common epoch budget also retain A1's test-assisted origin. These limitations are recorded in the report discussion.

## Saved deliverables and reproduction

All outputs are in [part_a/outputs/a4](../part_a/outputs/a4/):

- [analysis.md](../part_a/outputs/a4/analysis.md): report-ready method, references, top-three lists, full comparison, manual example, and limitations.
- [baseline feature plot](../part_a/outputs/a4/figures/baseline_feature_importance.png) and [wide-and-deep feature plot](../part_a/outputs/a4/figures/wide_deep_feature_importance.png): ten descending bars per model, SGD labels, and a shared axis scale; vector PDF versions are also saved. Both figures were visually inspected.
- [tables/global_importance.csv](../part_a/outputs/a4/tables/global_importance.csv), individual ranked tables, [feature_comparison.csv](../part_a/outputs/a4/tables/feature_comparison.csv), and [reference_values.csv](../part_a/outputs/a4/tables/reference_values.csv): all 20 global scores, rankings, differences, and references.
- `attributions/baseline.pt` and `attributions/wide_deep.pt`: full **26,702×10 signed SGD matrices** with predictions, original row IDs, feature order, and checkpoint hashes. Load with `torch.load(..., map_location="cpu", weights_only=True)`.
- [manual_checks.json](../part_a/outputs/a4/manual_checks.json): first-batch validation and first-sample differences.
- `references.pt`, `references.json`, `preprocessing.pt`, `configuration.json`, `summary.json`, and completion status/progress: exact references, transformations, source/runtime provenance, and results.
- [verification.json](../part_a/outputs/a4/verification.json) and [artifacts/a4_checks/latest.json](../artifacts/a4_checks/latest.json): successful audit and timestamped evidence.

The implementation is in [a4.py](../part_a/src/a4.py), exports in [a4_exports.py](../part_a/src/a4_exports.py), and audit in [verify_a4.py](../scripts/verify_a4.py). Run from the assignment root; existing outputs are protected:

```powershell
.\.venv\Scripts\python.exe -B -u part_a\scripts\run_a4.py --output part_a\outputs\a4_reproduction
.\.venv\Scripts\python.exe -B part_a\scripts\run_a4.py --export-only
.\.venv\Scripts\python.exe -B -u scripts\verify_a4.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

All packages remain in `.venv`; A4 uses the existing Captum installation and required no new dependencies. The next step is Part B data preparation and architecture verification.
