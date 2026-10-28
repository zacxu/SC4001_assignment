# Assignment reports — FAQ revision, updated October 12, 2026

Both reports are **five A4 pages**, including declarations, answers, figures, limitations and references. They follow [assignment.pdf](../assignment.pdf), [assignment_slides.pdf](../assignment_slides.pdf), [Q&A.md](../Q&A.md) and the supplied [A1 stopping/plot clarifications](../clarifications.md). The detailed experiment documents in `../docs/` remain the full evidence record; these PDFs are the concise assessed reports.

- [Part A PDF](PartA_report_YourName.pdf) and [editable Markdown](PartA_report_YourName.md).
- [Part B PDF](PartB_report_YourName.pdf) and [editable Markdown](PartB_report_YourName.md).
- [Input/layout manifest](manifest.json) and [independent verification](verification.json).

**YourName is a placeholder** because the report author name has not been supplied. The four submission files and their current verification status are recorded in [submission/README.md](../submission/README.md) and [submission/manifest.json](../submission/manifest.json); actual-name replacement and NTULearn upload remain pending. `scripts/finalize_submission.py --name "Your Full Name"` sets visible author, PDF metadata, footers and all four filenames, rebuilds packages and verifies integrity. Spaces become underscores in filenames. The latest report filenames are also recorded in `manifest.json`.

The FAQ revision adds separate train/validation RMSE and R² epoch plots for A2's categorical and activation ablations, plus A3's baseline/wide-and-deep epoch comparison. A2's additional original-split, seed-42 diagnostics are explicitly distinguished from its unchanged five-seed combined-data final results. B1/B2/B3 final results now come from fresh models trained on all 5,232 training+validation images for the frozen 9/13/9 epochs; the B3 comparison and B4 heatmaps use those same final checkpoints. Labels and B1/B2 output/dropout rules are instructor-confirmed; extending the raw-logit loss convention to B3 remains a stated interpretation.

B1's separate loss and accuracy figures now show validation curves only against epoch for all five learning rates, with saved-checkpoint stars and actual stopping lengths. Standalone figures, report assets, captions and packaged generators are synchronized. Saved training histories and numerical results are unchanged; B2/B3 retain their existing plots.

Run from the workspace root, using the existing project virtual environment and completed experiment outputs:

```powershell
.\.venv\Scripts\python.exe -B scripts\build_reports.py --name "Your Full Name"
.\.venv\Scripts\python.exe -B scripts\verify_reports.py
```

The builder regenerates the report PDFs, their corresponding Markdown and report-sized assets under `reports/assets/`. It performs **no training or model inference** and preserves source experiments. Text/tables and Markdown are generated from the same content in [build_reports.py](../scripts/build_reports.py); change that source to keep edited PDFs and Markdown synchronized. Plot code is in [report_assets.py](../scripts/report_assets.py), with the PDF layout in [report_layout.py](../scripts/report_layout.py).

The renderer uses installed PyMuPDF 1.28.2, Matplotlib 3.11.1, NumPy/pandas and CPU loading of saved Grad-CAM tensors through PyTorch. All libraries remain in `.venv`; no new package was needed. It embeds the existing Windows Arial regular/bold fonts (`C:/Windows/Fonts/arial.ttf` and `arialbd.ttf`) and TrueType plot text. Body text is principally 10–10.5 pt; the smallest captions/diagram labels/footer are 8 pt. Vector plots, searchable Unicode text and clickable primary references are retained; the X-rays/heatmaps are raster images at their supplied resolution.

Verification checks five-page limits, declarations and section order, bounds/glyphs/font sizes/links, 25 grid scores, five lambda rows, independent five-seed statistics, the complete B1/B2 tables and B3 comparison, all twenty feature scores, and all fifteen Grad-CAM titles/IDs. It also checks the FAQ source fingerprint, separate A2/A3 epoch plots, fresh-final-refit data/epoch provenance, and exact checkpoint hashes shared by final metrics, the architecture table and B4. Original selection runs, preprocessing and Part A final evidence are protected against [the pre-revision snapshot](../artifacts/faq_corrections/baseline_snapshot.json); only the documented FAQ corrections may change. [The completion record](../docs/REPORTS_COMPLETED.md) maps requirements to pages and links the current validation evidence.
