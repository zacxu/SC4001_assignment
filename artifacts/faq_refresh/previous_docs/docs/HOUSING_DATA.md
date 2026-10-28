# Part A housing preparation — step 4

Completed October 1, 2026 using the existing project `.venv`. Implementation: [part_a/src/data.py](../part_a/src/data.py). Preparation command: [part_a/scripts/prepare_data.py](../part_a/scripts/prepare_data.py).

## Partitions and inputs

The CSV contains 159,553 rows. The saved datasets preserve their original relative order and use the required chronological split.

| Dataset | Years | Rows | Preprocessing fitted on |
| --- | --- | ---: | --- |
| Selection training | 2017–2020 | 87,370 | 2017–2020 |
| Selection validation | 2021 | 29,057 | 2017–2020 |
| Selection test | 2022 | 26,702 | 2017–2020 |
| Combined refit training | 2017–2021 | 116,427 | 2017–2021 |
| Combined refit test | 2022 | 26,702 | 2017–2021 |
| Excluded | 2023 | 16,424 | Never fitted or exported as model samples |

Selection preprocessing and combined-refit preprocessing are separate artifacts. Reuse the saved refit artifact unchanged across A1's final model, A2's ablations, and A3's final comparisons. Neither transformation fits any statistic or category mapping on 2022 or 2023.

The exact input order is:

| Categorical tensor columns | Continuous tensor columns |
| --- | --- |
| `month` | `dist_to_nearest_stn` |
| `town` | `dist_to_dhoby` |
| `flat_model_type` | `degree_centrality` |
| `storey_range` | `eigenvector_centrality` |
| | `remaining_lease_years` |
| | `floor_area_sqm` |

The target is `resale_price` in SGD. `year` is only a split key. `full_address` and `nearest_stn` are excluded from model inputs.

## Encoding and scaling

Category values are sorted within the fitting partition and assigned IDs starting at 1; ID 0 is reserved for unseen values. Month keys remain integers, preserving numeric sorting. Embedding cardinalities, including ID 0, are **13, 27, 44, 18**, in categorical-feature order, for both protocols.

Continuous means and **sample** standard deviations (`correction=1`) use only fitting rows. Standard deviations are clamped to `1e-8`. Price scaling uses the fitting mean and **population** standard deviation (`correction=0`), also clamped to `1e-8`. Calculations use float32, matching the supplied baseline exactly.

| Price statistic, SGD | Selection | Combined refit |
| --- | ---: | ---: |
| Mean | 442,495.9375 | 459,696.8125 |
| Population standard deviation | 154,220.6875 | 159,183.8750 |

Datasets contain **raw SGD targets**. The shared `RegressionTask` standardizes them for MSE and inverse-transforms predictions for reported SGD metrics. Do not standardize saved targets a second time. `HousingPreprocessor.standardize_target` and `inverse_target` are available for checks and explanations. Model forward outputs remain standardized prices; explanation wrappers must inverse-transform those outputs to SGD.

There are five test rows whose `flat_model_type` is `5 ROOM, 3Gen`, absent from both fitting partitions. They correctly map to ID 0. All validation categories are known; the other three categorical inputs have no unseen test values. Detailed unknown counts and values are recorded for every partition in `report.json`.

## Data quality and artifacts

Checks found no missing values in any source column, no empty required categorical values, no exact duplicate rows, and no nonfinite required numeric values. No imputation, deduplication, or cleaning-based row removal was needed. The explicit 2023 exclusion is recorded separately. Invalid required values, unexpected years, missing required partitions, and invalid month/year values fail with a descriptive error. Constant features/targets retain the baseline's standard-deviation clamp.

Prepared files are in [part_a/outputs/data/](../part_a/outputs/data/):

```text
data/
  report.json                 # source SHA-256, quality, schema, maps, statistics, counts
  split_membership.csv        # original zero-based CSV row ID, year, selection partition
  selection/
    preprocessing.pt          # 2017–2020 encoders/scalers and feature metadata
    datasets.pt               # train/validation/test tensors and original row IDs
  refit/
    preprocessing.pt          # 2017–2021 encoders/scalers and feature metadata
    datasets.pt               # combined train/test tensors and original row IDs
```

Each dataset yields `(categorical, continuous, target_sgd)` with shapes `(N, 4)`, `(N, 6)`, and `(N,)`. Categorical tensors and row IDs are `torch.long`; continuous tensors and targets are `torch.float32`. Saved tensors load onto CPU; the shared trainer moves each batch to CUDA.

The `.pt` artifacts contain only tensors and basic values and load with `weights_only=True`. They preserve integer month keys. The human-readable JSON encoders use explicit value/ID records, avoiding JSON's conversion of dictionary keys to strings. `split_membership.csv` includes the excluded rows for provenance; neither dataset archive contains them. Refitting training membership is the union of the selection `train` and `validation` rows, retaining original CSV order.

## Commands and API

Run from the assignment root:

```powershell
# The default destination has already been populated.
.\.venv\Scripts\python.exe -B part_a\scripts\prepare_data.py --output part_a\outputs\data_reproduction
.\.venv\Scripts\python.exe -B scripts\verify_housing_data.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Omit `--output` on the first preparation in a fresh workspace. Existing destinations are protected; use a new output path for reproduction. `--csv PATH` supports a supplied CSV at another location. The verification script accepts `--prepared PATH` to check a reproduced export. It also needs the supplied reference project for the baseline comparison; the production module does not import that project.

Load prepared data in later Part A experiments:

```python
from part_a.src.data import load_data_bundle, HousingPreprocessor

selection = load_data_bundle("part_a/outputs/data/selection", expected_protocol="selection")
train_dataset = selection.datasets["train"]
validation_dataset = selection.datasets["validation"]
preprocessing = selection.preprocessing
cardinalities = preprocessing.cardinalities
target_mean, target_std = preprocessing.target_mean, preprocessing.target_std
checkpoint_preprocessing = preprocessing.metadata()

refit = load_data_bundle("part_a/outputs/data/refit", expected_protocol="refit")
combined_dataset = refit.datasets["train"]

# Reconstruct the same feature transforms from a trained checkpoint's metadata:
# restored = HousingPreprocessor.from_metadata(checkpoint["preprocessing"])
```

Pass `target_mean`, `target_std`, and `preprocessing=checkpoint_preprocessing` to `train_regressor`. Use the matching bundle for every model and loader in that training stage. The loader requires an explicit expected protocol and rejects mixing selection with refit artifacts. `transform_features(frame)` supports future inference using only the ten model inputs; no target or split key is required.

## Verification evidence

All **33 tests passed**, including 10 new housing behavior tests for chronological membership/order, sample versus population standard deviations, constant-feature/target clamps, raw SGD targets and inverse scaling, sorted integer categories and unknown ID 0, fitting leakage, strict value validation, duplicate preservation, safe artifact round trips, and protocol/output guards.

The complete exported datasets were compared against the unchanged supplied `src/data.py`: **every tensor element and every fitted statistic matched exactly** for selection. The refit export also matched the supplied function with fitting years changed to 2017–2021; its overlapping validation output was not used. Both safe artifact loads and original row IDs were checked, and all supplied-file hashes remained unchanged.

Latest evidence: [artifacts/housing_checks/latest.json](../artifacts/housing_checks/latest.json), pointing to the timestamped verification JSON. Data preparation evaluated no prediction performance and trained no assignment models. The next step is **5: A1 baseline and the 25-combination Ray grid search**.
