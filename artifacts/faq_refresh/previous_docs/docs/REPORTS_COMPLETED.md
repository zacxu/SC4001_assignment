# Step 14 — two concise reports completed

Completed **October 3, 2026**, using the project `.venv` and existing verified A1–A4/B1–B4 results. Both PDFs contain **exactly five A4 pages**, satisfying the brief's stricter **less than six pages each** requirement. Declarations and references count within those five pages. The brief, all slides, professor clarifications, detailed reports and saved CSV/JSON/tensor evidence were rechecked before drafting. No new training or model inference was performed.

| Deliverable | PDF | Editable source |
| --- | --- | --- |
| Part A | [PartA_report_YourName.pdf](../reports/PartA_report_YourName.pdf) | [PartA_report_YourName.md](../reports/PartA_report_YourName.md) |
| Part B | [PartB_report_YourName.pdf](../reports/PartB_report_YourName.pdf) | [PartB_report_YourName.md](../reports/PartB_report_YourName.md) |

**Author name remains YourName**, the documented placeholder. It is the only report identity item awaiting the user's name; it can be replaced throughout both reports using the generator's `--name` option before step 15 packaging. Report writing, numerical/layout checks and page-limit verification are complete. Final code ZIPs, fresh extracted-package checks, portal upload and receipt verification belong to step 15 and have not been performed here.

## Coverage within the page limit

| Page | Part A | Part B |
| --- | --- | --- |
| 1 | GenAI declaration; splits/preprocessing/Adam/stopping; A1 baseline diagram and parameter count; all 25 Ray results and selected settings. | GenAI declaration; normalization/partitions/output interpretation; B1 architecture, complete five-rate table, selected rate and why sensitivity/specificity matter. |
| 2 | A1 selected train/validation RMSE on common axes and refit training curve; final test RMSE/R²; fit/search justification and explicit test-assisted stopping limitation. | B1's separate loss/accuracy figures for all five rates, convergence discussion and final test metrics; complete B2 five-batch table and selection. |
| 3 | A2's five-seed mean/sample-SD table; two separate ablation figures, each including RMSE and R²; matched controls, interpretation and confounds. | B2's separate loss/accuracy figures, selection/curve interpretation, optimizer/BN update-budget confound and final test metrics. |
| 4 | A3 architecture/formula and complete lambda table; five-seed baseline comparison, both metric plots/statistics, mechanism and limitations. | B3 inherited settings, training/validation figures, final test performance/parameter/FLOPs comparison, counting convention and architecture pros/cons. |
| 5 | A4 all-ten-feature ranked SGD plots for both seed-42 models, training-only references, top threes, comparison, independent checks and attribution limitations. | B4 five originals/fifteen labelled overlays, fixed sampling/target/layers/display description, all-five image comparison, saved probabilities and absent-lesion-annotation limitation. |

The professor's updated presentation rules are applied: **no test metrics against epochs**, the two A2 ablations are shown separately, and final test RMSE/R² are reported in tables and final-seed comparisons. B1/B2 loss and accuracy remain separate figures. All B1/B2 table columns use the same saved error-selected checkpoint, rather than mixing error-selected and minimum-loss epochs. All B3 performance columns are final test values, and its FLOPs definition/exclusions are explicit.

The declarations accurately describe OpenAI Codex / ChatGPT assistance with planning, code, debugging, execution, interpretation, figures and report drafting. They state the actual automated/source/gradient checks and do not claim that the student has already personally reviewed all generated material. Reports retain Part A's test-assisted stopping/budget limitation, Part B's low test specificity and class/duplicate/patient assumptions, A3's modest seed-variable gain/confounds, A4's reference dependence, and B4's coarse/relative, noncausal localization limits.

## Rendering and independent verification

[build_reports.py](../scripts/build_reports.py) reads saved results directly, produces report-sized Matplotlib vector assets, and renders consistent PDFs/Markdown using [report_layout.py](../scripts/report_layout.py). Windows Arial and plot fonts are embedded. Main text is principally **10–10.5 pt**, with an **8 pt** minimum for captions, diagram labels, references and footers. An explicit Unicode mapping preserves copy/search spelling where the font's default glyph aliases would otherwise turn hyphens into soft hyphens. Primary references have clickable links within the page limit.

All **ten pages were rendered and visually inspected**. The review corrected overlapping chart legends/titles, a baseline-diagram caption overlap, missing mathematical glyphs, and clipped Grad-CAM first-row/colorbar labels. Final figures retain readable axes, units, all five curve legends, ranked score labels and the shared relative Grad-CAM scale.

The independent [verify_reports.py](../scripts/verify_reports.py) passed:

- Exact five-page counts, declaration placement, section/page coverage, searchable text, glyphs, font sizes, bounds and primary-reference links.
- **56 source-input hashes**, all **938 saved experiment-output hashes**, and all **12 supplied-file hashes**, with no changes to results/checkpoints/data.
- All **25** A1 grid scores and **five** A3 lambda rows; **20 independent mean/sample-SD checks** from the raw per-seed CSVs.
- **40 B1/B2 selected-checkpoint metric values**, all B3 performance/parameter/FLOPs values and final-test records.
- All **20 ranked A4 importance values** and all **15 Grad-CAM probability titles**, with the same five IDs and documented display method.

Evidence: [reports/verification.json](../reports/verification.json), [audit mirror](../artifacts/report_checks/latest.json), [source/layout manifest](../reports/manifest.json), [pre-report output hashes](../artifacts/report_checks/before.json), and [all page previews](../artifacts/report_checks/previews/). These are report checks; the earlier **96 behavioral tests** and CUDA experiment audits remain the model/code validation evidence.

## Reproduction and next step

On October 4, [the complete requirements review](FINAL_REQUIREMENTS_RECHECK.md) reread both assignment PDFs/diagrams and all ten report pages. Corrected the report's housing SD description to sample SD for inputs/population SD for price and made the caption's final-FC ReLU interpretation explicit. The revised reports remain five pages; their metrics and all original experiment evidence remain unchanged.

```powershell
# From the workspace root; uses existing completed outputs and the project venv.
.\.venv\Scripts\python.exe -B scripts\build_reports.py --name "Your Full Name"
.\.venv\Scripts\python.exe -B scripts\verify_reports.py
```

The builder updates report artifacts only. It does not retune models, repeat final performance evaluation, or regenerate source experiment outputs. Sources and detailed discussions remain in the existing experiment directories. [reports/README.md](../reports/README.md) explains editable content and the Windows font dependency. Step 15's [local verification/packaging](SUBMISSION_VERIFICATION.md) is complete; the two reports now also disclose packaging assistance. Actual student-name replacement and NTULearn upload/receipt remain pending.
