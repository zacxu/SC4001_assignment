# SC4001 assignment workspace

**October 10 FAQ review: corrections are required before submission.** [Q&A.md](Q&A.md) confirms fresh train+validation refits for all three Part B models and train/validation RMSE/R² epoch plots for A2/A3. Current Part B final results use the selection checkpoints; A2 lacks validation curves and A3's saved epoch curves were omitted from the report. B4 must be regenerated after the refits. See the [FAQ comparison and correction checklist](docs/FAQ_COMPLIANCE_REVIEW.md).

**Historical completion before the new FAQ review:** steps 1–14 and step 15's local verification/packaging had completed. The [four packaged files](submission/README.md) passed fresh extracted-folder checks: 57 Part A / 62 Part B tests, exact preprocessing, local checkpoint loading, all eight asset exporters, and a CUDA Ray worker. The [Part A PDF](reports/PartA_report_YourName.pdf) and [Part B PDF](reports/PartB_report_YourName.pdf) are **five pages each**, including GenAI declarations and references. **FAQ corrections, student-name replacement and NTULearn upload/receipt remain pending.** See the [submission verification record](docs/SUBMISSION_VERIFICATION.md) and [current manifest](submission/manifest.json). All dependencies stay in `.venv`; package validation reused that environment without fresh installation or full retraining.

**October 4 final recheck:** reread all 13 brief/slide pages and visually reviewed all 10 report pages. Corrected Part A's standardization wording (sample input SD/population price SD) and made the unconfirmed final-FC ReLU interpretation explicit. All 96 behavioral tests passed again; a new audit checked62 run histories and all questions' numerical evidence with938 experiment files unchanged. The [requirement-by-requirement review](docs/FINAL_REQUIREMENTS_RECHECK.md) records the remaining assumptions and identity/portal actions.

See [ASSIGNMENT_PLAN.md](ASSIGNMENT_PLAN.md) for the complete plan, [assignment decisions](docs/ASSIGNMENT_DECISIONS.md) for the source/assumption record, and [setup report](docs/SETUP_REPORT.md) for verification evidence.

The [report completion record](docs/REPORTS_COMPLETED.md) maps every question to its report page. [reports/](reports/README.md) contains the PDFs, matching editable Markdown, vector assets, source manifest and independent verification. Rebuild with `.\.venv\Scripts\python.exe -B scripts\build_reports.py --name "Your Full Name"`, then run `.\.venv\Scripts\python.exe -B scripts\verify_reports.py`. Report generation performs no training/inference; all 938 prior experiment-output hashes remain unchanged.

The [training rules](docs/TRAINING_RULES.md) describe Adam/early stopping, task-specific checkpoint selection, sample-weighted metrics, artifact formats, and final test evaluation. The canonical code is in `assignment_common/`, with entry points in each part's `src/training.py`. It passed 23 behavioral tests and CUDA checks on supplied-data subsets.

The [housing preparation report](docs/HOUSING_DATA.md) describes the saved chronological datasets and separate selection/refit transformations in `part_a/outputs/data/`. All exported tensors and fitted statistics match the supplied baseline exactly; 2023 is excluded and no statistics are fitted on test data. The expanded suite passes 57 tests.

The [A1 experiment report](docs/A1_EXPERIMENT.md) records all 25 Ray trials and the fresh final refit. Selected settings: **H=256, d=4, 7,065 parameters**, validation RMSE **SGD 58,627.10**, `E_val=6`. The final test-assisted refit selected **E*=4** and achieved test RMSE **SGD 96,901.64**, R² **0.676082**. Its test score uses the professor-authorized stopping protocol and is not an untouched holdout estimate. Results, diagrams, plots, tables, checkpoints, and discussion are saved under `part_a/outputs/a1/`; all 25 validation checkpoints passed a CUDA restoration audit.

The [A2 experiment report](docs/A2_EXPERIMENT.md) records 15 fresh runs, each using the frozen four-epoch budget and one final test evaluation. Test RMSE mean ± sample SD is **SGD 102,427.90 ± 5,289.17** for baseline, **108,252.68 ± 9,605.81** for continuous-only, and **101,821.96 ± 4,082.21** for Sigmoid. Removing categorical inputs worsened RMSE in four of five seeds; Sigmoid's small average gain occurred in three of five. Separate figures, RMSE/R² tables, per-seed checkpoints, and discussion are in `part_a/outputs/a2/`. All 15 restored checkpoints passed the CUDA audit. The fixed budget inherits A1's test-assisted selection.

The [A3 experiment report](docs/A3_EXPERIMENT.md) records five validation-only lambda candidates and five fresh four-epoch refits. Validation selected **λ=2**, RMSE **SGD 57,905.65**, R² **0.873239**. Its five-seed test RMSE is **SGD 101,413.74 ± 6,407.58**, R² **0.644081 ± 0.044905**, with a modest **0.99%** average RMSE gain over the saved A2 baseline. All ten checkpoints passed the CUDA restoration audit; lambda 0 reproduced A1's trained baseline weights exactly. Tables, four figures, checkpoints, discussion, and A4 settings are in `part_a/outputs/a3/`. The final budget retains the inherited test-assisted limitation.

The [A4 experiment report](docs/A4_EXPERIMENT.md) records full-test FeatureAblation for all ten original inputs and both seed-42 final models. Baseline top three: **floor area (SGD 71,537.72)**, **remaining lease (46,187.61)**, **distance to Dhoby Ghaut (41,163.95)**. Wide-and-deep top three: **floor area (73,119.34)**, **remaining lease (48,364.21)**, **flat model type (42,280.28)**. Scores are mean absolute attribution over all 26,702 test rows using shared training-mode/mean references. Full signed matrices, ranked plots/tables, manual checks, and discussion are in `part_a/outputs/a4/`. The CUDA audit verified full coverage, SGD differences, aggregation, and unchanged checkpoints.

The [Part B preparation report](docs/PART_B_PREPARATION.md) records the preserved 4,708/524/624 partitions, training-only normalization, raw overlay images, and Figure B CNNs. B1/B2/B3 have **99,569 / 18,185 / 77,169** trainable parameters and passed CUDA forward/backward, Adam, shape, residual-alignment, and checkpoint-restoration checks. Step 9 passed **70 tests**. Eight distinct images occur in both training and validation; this supplied-data limitation is documented without changing the partitions. All B1–B4 experiments are now complete.

The [B1 experiment report](docs/B1_EXPERIMENT.md) records all five learning-rate trials at batch 128. Validation selected **LR 0.001, epoch 9**, accuracy **97.9008%**, error **2.0992%**. Its single final test evaluation gave accuracy **85.4167%**, sensitivity **99.7436%**, specificity **61.5385%** (90 false positives, one false negative). Two separate loss/accuracy figures, the required table, checkpoints, captured test predictions, discussion, and frozen B2/B3 settings are in `part_b/outputs/b1/`. All **76 tests** passed; every candidate's full train/validation metrics reproduced exactly on CUDA, and final test metrics were audited from saved predictions without another test inference.

The [B2 experiment report](docs/B2_EXPERIMENT.md) records all five batch sizes using B1's **LR 0.001**. Validation selected **batch 16, epoch 13**, accuracy **97.9008%**, error **2.0992%**. Its single final test evaluation gave accuracy **85.5769%**, sensitivity **99.2308%**, specificity **62.8205%** (87 false positives, three false negatives). The required table, separate loss/accuracy figures, checkpoints, same-pass test predictions, discussion of curve fluctuations/update budgets, and frozen B3 settings are in `part_b/outputs/b2/`. All **82 tests** passed; all five checkpoints reproduced full train/validation metrics exactly on CUDA. The audit recomputed final metrics from saved predictions, checked Adam/BN updates and B1/data lineage, and preserved all supplied files and 833 previous outputs.

The [B3 experiment and comparison report](docs/B3_EXPERIMENT.md) records the compact ResNet run using **LR 0.001, batch 16**. Validation selected **epoch 9**, accuracy **97.5191%**; training stopped at epoch 13. Its single final test gave accuracy **82.8526%**, sensitivity **99.7436%**, specificity **54.7009%** (106 false positives, one false negative). The required B1–B3 table, B3 loss/accuracy figures, layer/parameter/profiler records, captured predictions, discussion, and B4 settings are in `part_b/outputs/b3/`. Profiled inference FLOPs per image are **39,063,616 / 19,464,256 / 25,325,696**, counting convolution/FC multiply-adds as two FLOPs plus residual additions; uncounted operations are explicitly documented. All **88 tests** passed. B3's full train/validation metrics reproduced exactly on CUDA; all complexity counts matched executed shapes. B1/B2 test records were reused without inference, and supplied/884 prior output hashes were preserved.

The [B4 Grad-CAM report](docs/B4_EXPERIMENT.md) compares the same five randomly drawn positive-class test rows **486, 279, 413, 53, 276**, seed **42**, across the exact selected B1/B2/B3 models. [The figure](part_b/outputs/b4/figures/gradcam_comparison.png) contains five originals and **15 overlays** with a shared display protocol. B1 has compact central hotspots, B2 more upper/peripheral patches, and B3 broader/coarser regions in this sample; these are qualitative observations without lesion annotations. Saved logits/probabilities match exactly after padding with duplicates of selected inputs to match original CUDA inference batch sizes. All **96 tests** passed; independent PyTorch autograd reproduced every map exactly, source checkpoints/data were unchanged, and **922 prior output hashes** were preserved. No full-test performance evaluation or retraining was performed.

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B scripts\verify_training.py
.\.venv\Scripts\python.exe -B scripts\verify_housing_data.py
.\.venv\Scripts\python.exe -B scripts\verify_a1.py
.\.venv\Scripts\python.exe -B scripts\verify_a2.py
.\.venv\Scripts\python.exe -B scripts\verify_a3.py
.\.venv\Scripts\python.exe -B scripts\verify_a4.py
.\.venv\Scripts\python.exe -B scripts\verify_part_b.py
.\.venv\Scripts\python.exe -B scripts\verify_b1.py
.\.venv\Scripts\python.exe -B scripts\verify_b2.py
.\.venv\Scripts\python.exe -B scripts\verify_b3.py
.\.venv\Scripts\python.exe -B scripts\verify_b4.py
```

## Python environment

All assignment libraries are installed in **`.venv/`**, created with Python's `venv` module. System site-packages are disabled. Use `.venv\Scripts\python.exe` for this project; activation is optional.

| Component | Installed version |
| --- | --- |
| Python | 3.12.0 |
| PyTorch | 2.11.0+cu128 |
| CUDA runtime provided by PyTorch | 12.8 |
| NumPy | 2.5.2 |
| pandas | 3.0.5 |
| Matplotlib | 3.11.1 |
| Ray with Tune dependencies | 2.58.0 |
| Captum | 0.9.0 |
| ipykernel | 7.3.0 |
| PyMuPDF (PDF inspection) | 1.28.2 |

CUDA execution was verified on the **NVIDIA GeForce RTX 3060 Ti, 8 GB**. The supplied lock requests PyTorch 2.13.0; that version was unavailable on the selected official CUDA 12.8 wheel index, so the project uses the verified 2.11.0+cu128 build. All other direct assignment dependencies match the supplied lock for Python 3.12. The supplied source and lock remain unchanged.

The project uses `venv` and `pip` as requested. `uv` is not required for these commands. Full installed dependency pins, including the CUDA build and its index, are in [requirements.txt](requirements.txt); each part has a copy for later independent packaging. [requirements.in](requirements.in) records the direct non-PyTorch dependency choices.

## Recreate the environment

Run these PowerShell commands from this assignment folder, using Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Alternatively, run `scripts\setup_venv.ps1`, which creates the venv if needed and installs the pinned packages using its interpreter.

To activate the existing environment when your PowerShell policy permits it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Directly invoking `.venv\Scripts\python.exe` works without activation. In an editor or notebook, select that interpreter as the Python environment/kernel.

## Run the setup checks

```powershell
.\.venv\Scripts\python.exe -B scripts\verify_setup.py
.\.venv\Scripts\python.exe -B scripts\verify_ray.py
```

The first check loads the supplied housing baseline and data, verifies 3,505 trainable parameters, performs one small CUDA training epoch, and checks FeatureAblation. It also verifies the X-ray arrays and normalization, trains a small one-convolution probe, checks Grad-CAM, exports a test plot, checks dependency consistency, and saves file hashes.

The second check runs a two-value Ray Tune grid, with one GPU worker at a time, and verifies CUDA access inside each worker. Ray's local services are shut down at the end. These are **setup probes**, not A1 search results or B1–B3 model implementations. No test-set prediction performance is evaluated in these probes.

Evidence is saved in:

- `artifacts/setup/environment_check.json`
- `artifacts/setup/ray_check.json`
- `artifacts/setup/config_snapshot.json`
- `artifacts/setup/pip_freeze.txt`
- `artifacts/setup/plot_smoke.png`

## Data and folders

The original housing project is under `housing_baseline_data/hdb-price-prediction-assignment-main/`; the X-ray archive is `chestxray_binary_64.npz` at the workspace root. [data/README.md](data/README.md) documents these locations.

`part_a/` and `part_b/` each have `src/`, `scripts/`, `configs/`, and `outputs/` folders plus their own README and pinned requirements. Shared training utilities, housing/X-ray preparation, A1–A4, all three CNN implementations, B1/B2/B3, and their architecture comparison are complete; B4 follows. Include `assignment_common/` in each final code ZIP for independent execution. `reports/` and `submission/` hold the final deliverables.

Shared settings are in [configs/project.json](configs/project.json). Selection seed: **42**. Five-seed evaluation: **42, 43, 44, 45, 46**. Start with one Ray GPU trial at a time and data-loader workers 0. The smoke script seeds Python/NumPy/PyTorch, sets deterministic options, and explicitly moves models and batches onto CUDA.

The [GenAI usage log](docs/GENAI_USAGE.md) records planning, setup, code/data assistance, A1–A4 and B1–B3 execution, Part B preparation/architecture verification, parameter/FLOPs comparison, and interpretation support for the required report declarations.
