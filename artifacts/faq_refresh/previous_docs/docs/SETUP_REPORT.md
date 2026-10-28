# Steps 1–2 completion report

Completed October 1, 2026. The next planned task is step 3.

## Step 1: requirements and assumptions

- Read both PDFs, the supplied professor clarifications, and the housing baseline source.
- Recorded source precedence, train/validation/test protocols, the professor's A1 stopping/plotting changes, separate ablation presentation, correct search grids, and report/submission rules.
- Recorded assumptions for fixed-epoch A2/A3 refits, CNN final logits, checkpoint ties, and X-ray class semantics.
- Confirmed data locations and local RTX 3060 Ti hardware.
- Saved human-readable decisions in [ASSIGNMENT_DECISIONS.md](ASSIGNMENT_DECISIONS.md) and machine-readable defaults in [project.json](../configs/project.json).

## Step 2: reproducible environment and project

Created the root `.venv` with Python 3.12.0. Its `pyvenv.cfg` disables system site-packages; installations used `.venv\Scripts\python.exe -m pip`. No assignment packages were installed into the global Python environment during this setup.

| Dependency | Supplied lock for Python 3.12 | Installed |
| --- | --- | --- |
| PyTorch | 2.13.0 | **2.11.0+cu128** |
| NumPy | 2.5.2 | 2.5.2 |
| pandas | 3.0.5 | 3.0.5 |
| Matplotlib | 3.11.1 | 3.11.1 |
| Ray / Tune | 2.58.0 | 2.58.0 |
| Captum | 0.9.0 | 0.9.0 |
| ipykernel | 7.3.0 | 7.3.0 |

The supplied PyTorch pin was unavailable from the selected official CUDA 12.8 wheel index. The installed CUDA build satisfies the supplied project's `torch>=2,<3` range and Captum's requirements. Baseline, convolution, attribution, and Ray-worker checks passed with this build. The original `pyproject.toml`, `uv.lock`, and source files were preserved. Installation followed the official [PyTorch local-installation guidance](https://pytorch.org/get-started/locally/); the actual wheel availability was checked through pip against [the CUDA 12.8 index](https://download.pytorch.org/whl/cu128).

Created separate Part A / Part B folders, READMEs, default configurations, and identical complete requirements snapshots. Root scripts and README document environment recreation and verification. Future production scripts will be implemented in the corresponding plan steps.

## Verification evidence

| Check | Result |
| --- | --- |
| Virtual environment | Interpreter and prefix point to the assignment's `.venv`; system site-packages disabled. |
| Dependency consistency | `pip check`: **No broken requirements found.** |
| CUDA | Available; PyTorch CUDA runtime 12.8; RTX 3060 Ti; compute capability 8.6; approximately 8 GB VRAM. |
| Supplied housing model | **3,505 trainable parameters**, cross-checked on the actual instantiated model. All loaded split tensors were finite. |
| Housing GPU smoke epoch | 4,096 randomly selected training rows, 8 batches of 512, Adam/MSE; finite loss and updated weights; approximately **0.215 s**. |
| Housing FeatureAblation | Two training samples; output shapes 2×4 categorical and 2×6 continuous; finite SGD attributions; first-feature attribution matched a manual prediction difference. |
| X-ray data | Verified six array keys, 64×64 uint8 images, 4,708/524/624 split sizes, and binary label counts. Semantic class mapping remains explicitly assumed. |
| X-ray training normalization | Mean **0.5716994213**, population standard deviation **0.1740975398**, after dividing training pixels by 255. |
| X-ray GPU smoke epoch | 512 training images, 4 batches of 128, setup-only one-convolution CNN, Adam/BCEWithLogitsLoss; finite loss and updated weights; approximately **0.127 s**. |
| Grad-CAM | Finite 1×1×64×64 attribution from the probe CNN on a training sample. |
| Matplotlib | Saved a labelled PNG with the noninteractive backend. |
| Ray Tune | Two successful grid trials, widths 16/32; Ray discovered one GPU, and both workers verified CUDA on the RTX 3060 Ti. |
| Provenance | Saved supplied document/data/source SHA-256 hashes, full installed versions, and a configuration snapshot. |

The timing figures cover small setup subsets and include initial CUDA/convolution setup. They are not full-dataset or full-architecture runtime estimates. Peak PyTorch allocated GPU memory was approximately 65.7 MiB for the housing probe and 131.0 MiB for the X-ray probe; these are not total device-memory measurements and do not justify increasing production trial concurrency yet.

No test-set prediction performance was evaluated. The setup CNN is not B1, B2, or B3, and the two Ray trials are not A1 hyperparameter-search results. Production metrics, early stopping, checkpoint export, and complete experiment code remain work for step 3 onward.

Saved evidence:

- [environment_check.json](../artifacts/setup/environment_check.json)
- [ray_check.json](../artifacts/setup/ray_check.json)
- [config_snapshot.json](../artifacts/setup/config_snapshot.json)
- [pip_freeze.txt](../artifacts/setup/pip_freeze.txt)
- [plot_smoke.png](../artifacts/setup/plot_smoke.png)

