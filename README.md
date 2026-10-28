# SC4001 assignment workspace

**Current answers follow the assignment brief, slides, clarifications and latest Q&A.** On October 11, 2026, completed the FAQ corrections: three fresh Part B combined-data refits, genuine A2 train/validation epoch diagnostics, A3 epoch-curve inclusion, final-model B4 regeneration and updated documentation/reports/packages. All **101 distinct behavioral tests passed**. The saved-evidence recheck verified **68 histories** and all protected original data/checkpoint/selection evidence, with no new model inference during the audit. See the [FAQ review](docs/FAQ_COMPLIANCE_REVIEW.md) and [requirements matrix](docs/FINAL_REQUIREMENTS_RECHECK.md).

The [reports folder](reports/README.md) contains both five-page PDFs, matching editable Markdown, generated figures and source/verification manifests. The [submission folder](submission/README.md) contains the four current upload files; its manifest/checksums and [latest extraction audit](artifacts/submission_checks/latest.json) are the authority for current hashes, counts and status. Earlier payloads are archived under artifacts/faq_refresh/previous_submission. **Student-name replacement and NTULearn upload/receipt remain pending.** No portal submission is claimed.

A1 selected H=256/d=4, validation RMSE SGD 58,627.10, then fresh combined-data refit E*=4: test RMSE SGD 96,901.64 and R2 0.676082. Its instructor-authorized test-assisted stopping and the inherited Part A budget are disclosed. A2/A3 five-seed final scores are unchanged; separate original-split A2 diagnostics and saved A3 curves meet the epoch-plot FAQ. A4 uses both full 26,702-by-10 signed attribution matrices in SGD.

Part B preserves all supplied rows and original training-only pixel statistics. Searches select LR 0.001, batch 128/16/16 and epochs 9/13/9; final models are freshly initialized and trained on all 5,232 train/validation images without holdout/test monitoring. Current final accuracy/sensitivity/specificity:

| Model | Accuracy | Sensitivity | Specificity |
| --- | ---: | ---: | ---: |
| B1 AlexNet-like | 90.7051% | 97.6923% | 79.0598% |
| B2 VGG-like | 85.2564% | 100.0000% | 60.6838% |
| B3 ResNet-like | 81.0897% | 100.0000% | 49.5726% |

B4 explains the same preselected pneumonia rows 486, 279, 413, 53, 276 with the fresh final checkpoints. Maps and displayed probabilities match current final-test evidence. Labels, B1 dropout, B2 no dropout and B1/B2 raw logits are instructor-confirmed; B3 output retains the documented binary-loss interpretation. Duplicate images, missing patient/lesion metadata and differing training recipes limit conclusions.

All packages belong in a Python venv. The existing `.venv` uses Python 3.12, Torch 2.11.0+cu128 and local RTX 3060 Ti CUDA; exact pins and extraction-root commands are in each code ZIP's standalone README. Verification reuses this environment and does not claim a fresh dependency installation or a rerun of the entire training workload.

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B scripts/recheck_assignment.py
.\.venv\Scripts\python.exe -B scripts/verify_reports.py
.\.venv\Scripts\python.exe -B scripts/verify_submission.py
```

The [step plan](ASSIGNMENT_PLAN.md), [decisions](docs/ASSIGNMENT_DECISIONS.md), [training rules](docs/TRAINING_RULES.md), [report completion record](docs/REPORTS_COMPLETED.md), [GenAI log](docs/GENAI_USAGE.md) and [submission verification](docs/SUBMISSION_VERIFICATION.md) document evidence, assumptions and remaining identity/portal actions.
