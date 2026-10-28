# Part B: chest X-ray classification

The current B1-B4 evidence follows [Q&A.md](../Q&A.md), the [brief](../assignment.pdf), slides and professor clarifications. See the [standalone package README](../submission_support/README_B.md) for venv setup, exact dependency pins, controls, seeds, data hashes, and extraction-root commands.

B1 retains its five-rate validation search and selectsLR0.001/batch128/epoch9. B2 retains the five-batch search at that LR and selectsbatch16/epoch13. B3 retains its original-split training and selectsLR0.001/batch16/epoch9. Each final model is **initialized from scratch** and trained on5,232 combined training/validation rows for9/13/9epochs without a monitoring holdout. The original training-only pixel normalization remains fixed under the brief. Each fresh final checkpoint receives one final624-row test pass; final predictions are saved for inference-free metric audits. Labels0=healthy/1=pneumonia, B1 dropout and B1/B2 raw FC1 logits are instructor-confirmed; B3 uses the documented same-loss output interpretation.

Current final accuracy/sensitivity/specificity is **90.7051%/97.6923%/79.0598%** forB1, **85.2564%/100%/60.6838%** forB2, and **81.0897%/100%/49.5726%** forB3. The [architecture comparison](outputs/b3/tables/architecture_comparison.csv) includes classification error,99,569/18,185/77,169parameters and scoped39,063,616/19,464,256/25,325,696FLOPs/image. Distinct update budgets and one seed limit architecture causality.

B4 compares the same fixed pneumonia rows486,279,413,53,276 across the three final-refit checkpoints in [five originals/fifteen overlays](outputs/b4/figures/gradcam_comparison.png). Saved probabilities and checkpoint hashes match final-test evidence. Coarse maps/per-map scaling and absent lesion annotations limit localization claims.

```powershell
# From the workspace root, regenerate saved evidence without model inference.
.\.venv\Scripts\python.exe -B part_b/scripts/run_b1.py --export-only
.\.venv\Scripts\python.exe -B part_b/scripts/run_b2.py --export-only
.\.venv\Scripts\python.exe -B part_b/scripts/run_b3.py --export-only
.\.venv\Scripts\python.exe -B part_b/scripts/run_b4.py --export-only
# Current audit branches check stored predictions/maps and final-refit lineage.
.\.venv\Scripts\python.exe -B scripts/verify_b1.py
.\.venv\Scripts\python.exe -B scripts/verify_b2.py
.\.venv\Scripts\python.exe -B scripts/verify_b3.py
.\.venv\Scripts\python.exe -B scripts/verify_b4.py
```

Full reproduction uses each run_bN entry point into new destinations or the bundled run_all.py. It preserves supplied partitions during search, then performs fresh combined-data refits before final testing/Grad-CAM. Historical earlier scores, original audits and old maps are retained under artifacts/faq_corrections/previous_part_b; current summaries/tables use FAQ-corrected finals. Author replacement and NTULearn upload/receipt remain pending; [manifest](../submission/manifest.json) and [latest extraction checks](../artifacts/submission_checks/latest.json) provide current packaging status.
