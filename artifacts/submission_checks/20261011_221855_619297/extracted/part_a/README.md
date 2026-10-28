# SC4001 Part A: HDB regression

This ZIP is independently runnable. Extract it into a new folder and run every command below **from that folder**, which contains this README, `assignment_common/`, `part_a/`, and `housing_baseline_data/`. Keep those directories together. Part B and the original assignment workspace are unnecessary.

## Environment and dependencies

Verified on Windows, Python **3.12.0**, NVIDIA GeForce RTX 3060 Ti **8 GB**, PyTorch **2.11.0+cu128** and CUDA runtime **12.8**. Full direct/transitive version pins are in `requirements.txt`; main libraries are NumPy 2.5.2, pandas 3.0.5, Matplotlib 3.11.1, Ray 2.58.0 and Captum 0.9.0. CUDA and a compatible NVIDIA driver are required for full training; CPU suffices for behavioral tests and asset regeneration. Different hardware/library kernels may change numeric results.

Create a Python venv and install all packages inside it (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -B verify_package.py
```

`requirements.txt` specifies the official CUDA 12.8 wheel index and verified Torch build. The instructor's supplied `uv.lock` requests a different Torch release; this submission uses the tested venv pins. The supplied source/lock/README are retained unchanged as reference, with no requirement to run `uv sync`.

## Included data and results

The unchanged source CSV is included at `housing_baseline_data/hdb-price-prediction-assignment-main/hdb_price_prediction.csv`, SHA-256 `6879f9f0484d390150a4ccf10d5f1f53894de62fb3a2e308d305a0145a53e42c`. If replacing it, place the identical supplied CSV there or use `part_a/scripts/prepare_data.py --csv PATH --output NEW_FOLDER`. Original baseline Python files are also included for exact parity tests.

`part_a/outputs/data/` contains both prepared protocols: 2017–2020 training (87,370 rows), 2021 validation (29,057), 2022 test (26,702); combined 2017–2021 refit training (116,427). 2023 is excluded. Encoders, standardizers, target scaling and reference values use the appropriate training rows only. Continuous input standardization uses **sample SD (`correction=1`)**; price target scaling uses **population SD (`correction=0`)**, matching the supplied baseline. Unknown categories use index zero. Inputs are four categorical features and six continuous features.

`part_a/outputs/a1/` through `a4/` retain all completed selection/refit checkpoints, histories, frozen settings, numerical evidence, verification records, tables and PNG/PDF figures. A1 includes all 25 Ray training folders; duplicate Ray checkpoints, serialized tuner state and event/progress logs are omitted because fresh training starts a new search. Virtual environments, caches, setup probes and unrelated files are excluded. Recorded absolute paths in JSON/PT provenance describe the original run; execution resolves files within this extraction, and those records are kept intact to preserve hashes.

## Verification and figures from saved evidence

```powershell
.\.venv\Scripts\python.exe -B verify_package.py --prepare --export --ray-smoke
```

This verifies the package's hashes/dependencies and behavioral tests, rebuilds preprocessing into a **new** check folder, compares every tensor/statistic with saved preprocessing, loads selected checkpoints and checks synthetic CUDA forwards, regenerates saved assets, and verifies Ray CUDA-worker imports from this folder. It performs no full experiment training or test-set model inference. `--ray-smoke` requires CUDA and is optional when inspecting on CPU (`--device cpu`). The initial hash check runs before regenerated PDF timestamps can change. To restore the exact archive bytes, extract again.

Individual regeneration commands (use saved data; no training/test inference):

```powershell
.\.venv\Scripts\python.exe -B part_a/scripts/run_a1.py --export-only
.\.venv\Scripts\python.exe -B part_a/scripts/run_a2.py --export-only
.\.venv\Scripts\python.exe -B part_a/scripts/run_a2_diagnostics.py --export-only
.\.venv\Scripts\python.exe -B part_a/scripts/run_a3.py --export-only
.\.venv\Scripts\python.exe -B part_a/scripts/run_a4.py --export-only
```

Assets appear in each `part_a/outputs/aN/figures/`, `tables/`, and `analysis.md`. A1 supplies the baseline diagram, 25-cell heatmap, training/validation curves and the brief-required authorized train/test refit RMSE curve. A2 keeps categorical and activation ablations separate, with genuine original-split train/validation RMSE and R2 epoch diagnostics at seed 42 and the frozen four-epoch budget. Diagnostics do not replace the required combined-data five-seed final results or tune their settings. A3 supplies lambda selection, saved training/validation RMSE/R2 epoch curves, and the paired final baseline comparison. A4 supplies ten-feature mean absolute SGD attribution rankings. A2/A3 epoch plots contain no test metric curves; A1's `internal_test_monitor.jsonl` records the professor-authorized stopping audit and supplies the brief's requested A1 train/test refit curve. Final RMSE/R² are in tables/barplots.

A2 final comparisons use paired points and mean ± sample-SD error bars with shared zoomed y axes: RMSE SGD 90,000–120,000 and R² 0.500–0.725. Its two train/validation epoch comparisons also share padded zoomed ranges. Every point/error-bar endpoint remains visible; captions disclose the zoom. Regeneration derives the limits from saved data, and all numerical results remain unchanged.

## Reproduce all experiments from the source CSV

```powershell
.\.venv\Scripts\python.exe -B run_all.py --output part_a/outputs/reproduction
```

The destination must not exist. This runs data preparation → A1 → A2 → A3 → A4, with explicit local data/handoff paths. New results stay beneath that destination; the bundled results are preserved. A1 runs 25 Ray trials with one CUDA GPU/four CPU threads per worker and one trial at a time; A2 runs 15 fresh refits and three separate original-split epoch diagnostics; A3 runs five lambda selections/five fresh refits. Full execution can take substantial time. For individual runs:

```powershell
.\.venv\Scripts\python.exe -B part_a/scripts/prepare_data.py --output part_a/outputs/new_data
.\.venv\Scripts\python.exe -B part_a/scripts/run_a1.py --prepared part_a/outputs/new_data --output part_a/outputs/new_a1
.\.venv\Scripts\python.exe -B part_a/scripts/run_a2.py --prepared part_a/outputs/new_data --a1 part_a/outputs/new_a1 --output part_a/outputs/new_a2
.\.venv\Scripts\python.exe -B part_a/scripts/run_a3.py --prepared part_a/outputs/new_data --a1 part_a/outputs/new_a1 --a2 part_a/outputs/new_a2 --output part_a/outputs/new_a3
.\.venv\Scripts\python.exe -B part_a/scripts/run_a4.py --prepared part_a/outputs/new_data --a1 part_a/outputs/new_a1 --a3 part_a/outputs/new_a3 --output part_a/outputs/new_a4
```

## Controls and interpretation

All searches use seed **42**; paired A2/A3 final runs use **42,43,44,45,46**. Adam LR **0.01**, batch **512**, maximum **20** epochs, patience **4**, significant RMSE improvement **1e-4 SGD**. A1 validation selected **H=256,d=4**, then its fresh combined refit selected **E*=4** using **test RMSE under the instructor clarification**. A2/A3 inherit that fixed four-epoch budget. Consequently these Part A final scores are test-assisted and are not untouched-holdout estimates. Initialization and shuffled sample order are paired across controlled comparisons; standard deviations use `ddof=1`.

A3 uses the shared encoded input in both branches, selects **λ=2** on validation, and retains LayerNorm. A4 uses full-test FeatureAblation for seed-42 final A1/A3 checkpoints, shared categorical modes/continuous means and ten original-feature masks, reporting mean absolute **SGD** output differences. Attribution depends on references and correlated inputs; it is not causal evidence.

Code map: `assignment_common/` provides seeded loaders, metrics, stopping/checkpoint rules and artifacts; `part_a/src/` provides data, model, ablations, wide-and-deep and explanation modules; `part_a/scripts/` provides CLI entry points; `tests/` contains Part A/common behavioral tests. GenAI-assisted implementation, documentation and verification are disclosed in `GENAI_USAGE.md` and the separate report. Package checks are scoped verification, not newly reported experiment results.

The supplied brief, slides, clarifications and latest Q&A.md are included at this extraction root. Selection and combined-data preprocessing are fitted separately; Q&A.md explicitly confirms refitting A1 scaler/encoder statistics on the combined set.
