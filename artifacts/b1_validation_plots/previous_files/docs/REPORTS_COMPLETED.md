# Two concise current reports

Updated October 11, 2026 against both assignment PDFs, clarifications and all six FAQ answers. The reports contain exactlyfive pages each, including GenAI declarations first and references within the page limit. Current PartB final results use new combined-data models; A2/A3 include genuine train/validation epoch evidence. Historical report versions are archived separately.

A subsequent [A1 interpretation review](A1_INTERPRETATION_REVIEW.md) strengthens the selection justification: width-averaged grid evidence, the measured d=4 versus d=2 comparison, controlled width comparison at fixed d=4, and both learning histories support late overfitting. High test error and width relative to sample count do not establish underfitting. This changes explanation only; experiment settings, model weights and numeric results are preserved.

A2's categorical and activation figures now use common zoomed y-axis ranges to show differences clearly. Both report epoch plots retain every training/validation point and disclose the zoom; the packaged final five-seed comparisons retain every paired point and mean ± SD endpoint. The saved numbers and conclusions are unchanged.

A3's final baseline comparison also uses zoomed paired-point/error-bar plots, covering RMSE SGD 90,000–115,000 and R² 0.575–0.725. Its standalone PNG/PDF and report asset use matching ranges. The report keeps the required A3 epoch plot and final table, with a description of the packaged comparison. All saved results are unchanged.

| Page | PartA | PartB |
| --- | --- | --- |
|1|Declaration/methods, A1 baseline diagram/count and full25Ray grid|Declaration/methods and completeB1selectiontable|
|2|A1train/validation RMSE, train/test refit curve, result and justification/limits|B1separate loss/accuracyfigures/currentfinaltest and B2selectiontable|
|3|A2five-seed finalsummary; separate original-split categorical/activationRMSE/R2curves and conclusions|B2separate loss/accuracyfigures, batchselection/confound and correctedfinaltest|
|4|A3lambda search/formula; savedtrain/validationRMSE/R2curves; five-seedcomparison/mechanism/limitations|B3training/validationcurves; correctedthree-final-model performance/parameter/FLOPstable, pros/cons|
|5|A4bothrankedten-featureSGDplots/topthree/references/limits|B4samefiveoriginals/fifteenfinal-modeloverlays and qualitative comparison/limits|

[PartA PDF](../reports/PartA_report_YourName.pdf), [editableA Markdown](../reports/PartA_report_YourName.md), [PartB PDF](../reports/PartB_report_YourName.pdf), and [editableB Markdown](../reports/PartB_report_YourName.md) are generated from canonical saved evidence. [Manifest](../reports/manifest.json) records the controlling source hashes includingQ&A.md. [Verification](../reports/verification.json) checks all values, epochplotprotocol, freshfinalcheckpoint lineage, pagecount/glyphs/fonts/textbounds and references. Rendered previews support visual checking.

The declarations include actual FAQ correction, new refit/diagnostic execution, Grad-CAM regeneration, report and package assistance. Original held-out searches remain valid; current final metrics are saved separately from historical earlier scores. No A2/A3 test metric epochplots appear; A1 alone includes the required authorized train/test refit curve. B1/B2validation tables use coherent error-selected checkpointmetrics; finaltest rows use separatefreshmodels. Label0 = healthy/1 = pneumonia and B1/B2output/dropout are instructor-confirmed. B3output remains the documented same-loss interpretation, with test-assisted PartA and other limitations retained.

```powershell
.\.venv\Scripts\python.exe -B scripts/build_reports.py --name "Your Full Name"
.\.venv\Scripts\python.exe -B scripts/verify_reports.py
```

A1's plot includes the original saved eight-epoch train/test refit RMSE histories because its explicit brief requirement is separate from the A2/A3 FAQ. Generation from saved evidence performs no retraining or new model inference. Actualname replacement and NTULearn upload/receipt remainpending; the [submission verification record](SUBMISSION_VERIFICATION.md) and currentmanifest provide final packagingstatus.
