# SC4001 Part B: chest X-ray classification

This ZIP is independently runnable. Extract into a new folder and run every command below **from that folder**, which contains this README, `assignment_common/`, `part_b/` and `chestxray_binary_64.npz`. Keep that layout. Part A and the original assignment workspace are unnecessary.

## Environment and dependencies

Verified on Windows, Python **3.12.0**, NVIDIA GeForce RTX 3060 Ti **8 GB**, PyTorch **2.11.0+cu128**, CUDA runtime **12.8**, NumPy 2.5.2, pandas 3.0.5, Matplotlib 3.11.1 and Captum 0.9.0. Full direct/transitive pins are in `requirements.txt`; CUDA-compatible NVIDIA hardware/driver is required for production training. CPU can inspect saved evidence and run behavioral tests. Different hardware/library kernels may change numeric results.

Create a venv and install packages only into it (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -B verify_package.py
```

The requirements include the official CUDA 12.8 wheel index and tested Torch version. The same complete tested environment is pinned in both submissions; Ray is needed by Part A and remains in this environment snapshot.

## Included data and results

The unchanged supplied NPZ is included at `chestxray_binary_64.npz`, SHA-256 `55ab6e2c7167c451c3bbaa47727ecf810fc2d6285e0d8090a1078c67cbfa3cd8`. If replacing it, use that exact supplied archive or `part_b/scripts/prepare_data.py --npz PATH --output NEW_FOLDER`.

`part_b/outputs/data/` preserves the supplied **4,708 train / 524 validation / 624 test** rows and raw overlay images. Scaling is uint8/255 followed by training-only mean **0.5716994212632378**, population SD **0.17409753982645407**. There is no resplitting, augmentation, resampling or class weighting. **Q&A.md confirms 0=healthy,1=pneumonia**; the archive itself lacks semantic class-name metadata. Eight distinct images overlap training/validation; no exact training/test or validation/test overlap was found, but patient-level independence cannot be verified.

`part_b/outputs/b1/` through `b4/` retain the validation-search checkpoints/histories and separate fresh final-refit checkpoints/histories, frozen settings, captured single-pass test predictions, tables, PNG/PDF figures and prior verification records. Earlier pre-FAQ test records and the initial failed attribution attempt remain in the workspace historical archive and are excluded from this current-answer package. Setup architecture probes, venvs, caches and console logs are omitted. Historical absolute paths in JSON/PT describe the original runs and remain unchanged to preserve hashes; execution uses paths inside this extraction.

## Verification and saved asset regeneration

```powershell
.\.venv\Scripts\python.exe -B verify_package.py --prepare --export
```

This verifies package integrity/dependencies and all Part B/common tests, rebuilds preprocessing into a **new** check folder, compares every tensor/statistic with saved preparation, loads three fresh final-refit checkpoints and checks synthetic CUDA forwards, then regenerates all saved figures/tables. It performs no full training or test-set model inference. For CPU inspection use `--device cpu`. The first hash check precedes regenerated PDF timestamp changes; re-extract to restore original archive bytes.

Individual asset commands:

```powershell
.\.venv\Scripts\python.exe -B part_b/scripts/run_b1.py --export-only
.\.venv\Scripts\python.exe -B part_b/scripts/run_b2.py --export-only
.\.venv\Scripts\python.exe -B part_b/scripts/run_b3.py --export-only
.\.venv\Scripts\python.exe -B part_b/scripts/run_b4.py --export-only
```

Each `part_b/outputs/bN/figures/`, `tables/`, and `analysis.md` contains the corresponding answer evidence. B1/B2 have the required selection tables and **separate loss and accuracy** training/validation plots. B3 adds the architecture/performance/parameter/FLOPs comparison. B4 supplies five originals and fifteen labelled Grad-CAM overlays. No test metric epoch plots are produced. B3's `--b1`/`--b2` arguments can locate alternate saved source directories during export.

## Reproduce all experiments from the NPZ

```powershell
.\.venv\Scripts\python.exe -B run_all.py --output part_b/outputs/reproduction
```

The destination must not exist. This runs preparation → B1 → B2 → B3 → B4 with explicit local handoff paths; bundled evidence remains intact. B1/B2 require five selection runs plus one fresh combined-data refit each; B3 requires one selection run plus one fresh combined-data refit; full execution takes substantial time. Individual commands:

```powershell
.\.venv\Scripts\python.exe -B part_b/scripts/prepare_data.py --output part_b/outputs/new_data
.\.venv\Scripts\python.exe -B part_b/scripts/run_b1.py --data part_b/outputs/new_data --output part_b/outputs/new_b1
.\.venv\Scripts\python.exe -B part_b/scripts/run_b2.py --data part_b/outputs/new_data --b1 part_b/outputs/new_b1 --output part_b/outputs/new_b2
.\.venv\Scripts\python.exe -B part_b/scripts/run_b3.py --data part_b/outputs/new_data --b1 part_b/outputs/new_b1 --b2 part_b/outputs/new_b2 --output part_b/outputs/new_b3
.\.venv\Scripts\python.exe -B part_b/scripts/run_b4.py --data part_b/outputs/new_data --b1 part_b/outputs/new_b1 --b2 part_b/outputs/new_b2 --b3 part_b/outputs/new_b3 --output part_b/outputs/new_b4
```

## Controls and interpretation

Training/search and the independent B4 sampler all use seed **42**. Models output an unrestricted binary logit with BCEWithLogitsLoss; sigmoid≥0.5 predicts positive. The caption's “all layers use ReLU” is interpreted to apply to hidden/convolutional layers and residual sums, **excluding final FC1**. Q&A.md explicitly confirms this output/dropout handling for B1/B2; B3 uses the same binary-loss interpretation, which is not separately named in that answer. ReLU on that final logit with the stated sigmoid threshold would predict every input positive. Hidden Conv/BN/ReLU operations and post-residual ReLU otherwise follow the diagrams. Adam, at most **20** epochs, validation-loss patience **4**, significant improvement **1e-4**, selected checkpoint by lowest validation error then loss then earlier epoch. After validation selection, initialize a fresh model and Adam state with seed 42, combine the original 4,708 train and 524 validation rows, and train for exactly the selected 9/13/9 epochs without monitoring validation or test metrics. Retain original training-only pixel normalization in both phases, following the brief. The fixed refit budget follows the explicit FAQ instead of applying early stopping without a holdout. Each new final classifier receives one final test pass; captured predictions support independent metric recomputation without another inference pass.

B1 AlexNet-like (99,569 parameters) uses batch128, rate grid 1e-5/1e-4/1e-3/1e-2/1e-1; selects **LR0.001,epoch9**. B2 VGG-like (18,185 parameters) inherits LR0.001 and selects **batch16,epoch13** among16/32/64/128/256. B3 compact ResNet-like (77,169 parameters) inherits LR0.001/batch16 and selects **epoch9**. Reported FLOPs/image **39,063,616 / 19,464,256 / 25,325,696** count two FLOPs per convolution/FC MAC plus residual additions; BN/activations/pooling/bias/scaling/threshold and training are excluded. Different recipes, epochs/update budgets and one seed limit causal architecture conclusions; test specificity must be assessed alongside sensitivity.

B4 uniformly samples the same five instructor-confirmed pneumonia test rows **486,279,413,53,276**, retaining order across the exact fresh final-refit checkpoints. Captum LayerGradCam explains the positive raw logit at `features.conv3.conv`, `features.conv6.conv`, and `features.block64.conv2.conv` before their BN/ReLU. Positive maps have native16×16/16×16/8×8 grids, bilinear64×64 upsampling, per-map maximum scaling, inferno colors and opacity0.55×heat. Duplicate selected images pad to original inference batch sizes128/16/16; padding maps are discarded. This reproduced saved logits exactly without redrawing rows or relaxing guards. Maps are qualitative, reference no lesion annotations, and cannot validate causal/anatomical localization or absolute cross-model strength.

Code map: `assignment_common/` provides shared training/metrics/reproducibility/artifacts; `part_b/src/` provides data, the three CNNs, evaluation, complexity and explanations; `part_b/scripts/` provides CLI entry points; `tests/` contains Part B/common tests. GenAI assistance is declared in `GENAI_USAGE.md` and the separate report. Package smoke checks are scoped verification and are not additional final experiment results.

The supplied brief, slides, clarifications and latest Q&A.md are included at this extraction root. Original no-refit test results and pre-FAQ Grad-CAM are archived outside this package as historical provenance; current summary.json, final_test.json, tables and B4 use final_checkpoint keys. These historical results must not be reported as the current final answers.
