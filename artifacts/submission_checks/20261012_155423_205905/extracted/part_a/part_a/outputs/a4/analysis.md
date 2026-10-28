# A4 FeatureAblation results and discussion draft

All ten original features were explained for **every one of the 26,702 2022 test rows**, for both final seed-42 models.

## Models, references, and SGD calculation

The A1 baseline and validation-selected A3 wide-and-deep model (lambda 2) were restored from their final epoch-4 combined-data checkpoints. Both use the exact saved 2017–2021 encoders/scalers. Their weights were kept fixed during explanation.

Captum FeatureAblation replaces one input with its reference while retaining the other nine. Signed attribution is `price(original) - price(feature replaced)`. Categorical IDs are replaced before embedding lookup, so each entire categorical variable is one feature. Unique mask IDs 0–3 identify categorical columns and 4–9 identify continuous columns. No embedding coordinates are presented as additional features. The explained forward function returns one SGD prediction per row; target scaling is applied once after the model output. The network stays float32, with the final inverse transform and subtraction in float64 to reduce numerical cancellation.

References are shared across models: each categorical feature's mode among all 116,427 refit training rows (ties favor the smallest encoded ID), and each continuous feature's saved training mean, represented by standardized zero. No test data determine these reference values. [Captum's FeatureAblation documentation](https://captum.ai/api/feature_ablation.html) describes these replacement baselines and feature masks; the installed Captum 0.9.0 implementation was checked as well.

| Feature | Raw reference | Encoded / standardized reference |
| --- | --- | ---: |
| Month | 7 | 7 |
| Town | SENGKANG | 21 |
| Flat model type | 4 ROOM, Model A | 19 |
| Storey range | 04 TO 06 | 2 |
| Distance to nearest station | 0.8154776096343994 | 0.0 |
| Distance to Dhoby Ghaut | 11.163990020751953 | 0.0 |
| Degree centrality | 0.0174986831843853 | 0.0 |
| Eigenvector centrality | 0.00660629291087389 | 0.0 |
| Remaining lease years | 74.89581298828125 | 0.0 |
| Floor area (m²) | 97.8547134399414 | 0.0 |

Global importance is **mean(abs(signed attribution)) over all 26,702 test rows**, yielding exactly ten SGD scores per model. The absolute value is taken before averaging. Batches of 512 include the final partial batch, with one perturbation per forward evaluation. Full signed N×10 matrices, SGD predictions, and original CSV row IDs are saved for reproducible aggregation.

## Ranked results and top three

**Baseline top three:** **Floor area (m²)** (SGD 71,537.72); **Remaining lease years** (SGD 46,187.61); **Distance to Dhoby Ghaut** (SGD 41,163.95).

**Wide-and-deep top three:** **Floor area (m²)** (SGD 73,119.34); **Remaining lease years** (SGD 48,364.21); **Flat model type** (SGD 42,280.28).

Each separate plot (`figures/baseline_feature_importance` and `figures/wide_deep_feature_importance`) shows all ten scores in descending order, with a shared SGD axis scale and numeric labels. PNG and vector PDF versions are saved.

| Feature | Baseline rank | Baseline importance (SGD) | Wide-and-deep rank | Wide-and-deep importance (SGD) | Change (SGD) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Month | 9 | 4,622.20 | 9 | 5,086.60 | +464.40 |
| Town | 5 | 38,117.48 | 5 | 35,223.61 | -2,893.87 |
| Flat model type | 4 | 39,330.39 | 3 | 42,280.28 | +2,949.88 |
| Storey range | 7 | 16,839.62 | 7 | 16,254.52 | -585.10 |
| Distance to nearest station | 6 | 21,066.58 | 6 | 20,933.73 | -132.85 |
| Distance to Dhoby Ghaut | 3 | 41,163.95 | 4 | 40,061.64 | -1,102.31 |
| Degree centrality | 10 | 4,025.42 | 10 | 4,045.44 | +20.02 |
| Eigenvector centrality | 8 | 9,071.04 | 8 | 7,820.91 | -1,250.13 |
| Remaining lease years | 2 | 46,187.61 | 2 | 48,364.21 | +2,176.60 |
| Floor area (m²) | 1 | 71,537.72 | 1 | 73,119.34 | +1,581.61 |

The models share **2/3** top-three features: **Floor area (m²), Remaining lease years**. The full ten-feature rank correlation is **0.9879** (Spearman, average ranks for any ties). Ranks changed by at most 1 place(s): Flat model type from 4 to 3; Distance to Dhoby Ghaut from 3 to 4. The largest absolute score change is **Flat model type**, +2,949.88 SGD.

These magnitudes describe how strongly each fitted model's predictions respond to replacement by the chosen reference. Shared prominent features suggest similar predictive reliance under this reference scheme. Changes in ranking or magnitude can arise from the active linear branch, changes in jointly trained embeddings, and changes in the MLP; this comparison does not isolate a branch-specific causal effect. A larger attribution does not by itself establish better prediction accuracy.

## Verification and limitations

For both models, all ten Captum columns in the first 512-row batch were checked against independent manual prediction differences, including a documented first-row example. This validates categorical grouping, sign, feature order, and SGD scaling. Model-state fingerprints were unchanged. Saved attribution matrices cover every test row in the prepared-data order.

For the first test sample's floor-area replacement, the signed results agree as follows (row IDs are original zero-based CSV indices). A negative value means replacement by the training mean increased this model's prediction.

| Model | Source row ID | Captum signed attribution (SGD) | Manual original-minus-replaced price (SGD) |
| --- | ---: | ---: | ---: |
| Baseline | 116427 | -200,394.63 | -200,394.63 |
| Wide-and-deep | 116427 | -176,860.94 | -176,860.94 |

Reference choice affects results: a mode-category/mean-continuous replacement answers a particular model-sensitivity question and another valid reference could change scores. Replacing one correlated feature while leaving others fixed may create unrealistic property/location combinations, for example between floor area and flat model type or among station distance and centrality variables. Mean absolute scores hide the direction and distribution of effects; signed per-row values are retained. Because the network is nonlinear, the ten separate effects need not add up to the prediction difference from replacing all features together. These are model explanations, not causal effects on resale prices.

Only the requested paired seed-42 models are explained, so no across-seed attribution stability claim is made. The baseline checkpoint and common epoch budget retain A1's professor-authorized test-assisted origin. A4's required perturbed forward passes are explanation computations; the saved A1/A3 performance records are reused, and no settings are tuned from attribution results.

## Saved evidence

- `attributions/baseline.pt` and `attributions/wide_deep.pt`: full signed SGD matrices, predictions, row IDs, checkpoint hashes, and feature ordering (safe `weights_only=True` loading).
- `references.pt`, `references.json`, and `tables/reference_values.csv`: shared legal category IDs, raw categories/means, mode counts, and fitting provenance.
- `tables/global_importance.csv`, individual ranked tables, and `tables/feature_comparison.csv`: all global scores, ranks, and model differences.
- `manual_checks.json`: all first-batch coordinate checks and the first-sample attribution/manual-difference comparison.
- `configuration.json`, `summary.json`, and `progress.json`: frozen models, source/runtime provenance, full coverage, ranked summaries, and completion status.
