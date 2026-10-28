# Final assignment requirements recheck

Reviewed **October 4, 2026**, at the student's request. Reread **all seven pages of assignment.pdf and all six slides**, including the original Figure A/Figure B diagrams, then reread both final reports and visually inspected every report page. Compared the diagrams with implemented modules and compared each required answer with saved full-precision evidence. The written brief, supplied baseline and [professor clarifications](../clarifications.md) are the controlling sources.

**All eight questions have answer coverage and their saved numerical evidence passes verification.** One factual report sentence was corrected; the existing output-ReLU interpretation is now more explicit. Literal compliance with the ambiguous “all layers use ReLU” caption is not claimed. Student-name replacement and actual NTULearn submission/receipt remain incomplete.

## Correction and interpretation

**Corrected Part A, page 1:** the report previously described both input and price standardization using population SD. The supplied baseline, working implementation and saved preprocessing actually use **sample SD (`correction=1`) for continuous housing inputs** and **population SD (`correction=0`) for the price target**. The report now states both accurately. Recomputed the original raw-source statistics with the same float32 tensor layout and confirmed exact saved input means/SDs and matching price statistics. The implementation, preprocessing and experiment outputs were already correct and remain unchanged. Added report-verification assertions for the corrected wording.

**Clarified Part B, page 1:** the diagram caption says “all layers” use ReLU. Existing models interpret this as hidden/convolutional layers and residual sums; final FC1 returns an unrestricted binary logit for BCEWithLogitsLoss. This is a disclosed, user-authorized assumption, **not an instructor-confirmed exception**. Under the current sigmoid≥0.5 classifier, applying ReLU to the final logit would force every prediction positive because sigmoid(nonnegative)≥0.5. The report now explicitly names the caption and the final-FC exception. The implemented convolutional/hidden/residual ReLUs otherwise match the diagrams. This remaining ambiguity prevents certifying strict literal compliance with that caption.

Other recorded assumptions remain explicit: A3 final refits use A1's fixed four-epoch budget for a paired comparison; NPZ labels are interpreted as 0=healthy/1=pneumonia because semantic metadata is absent. A1's test-RMSE stopping is **explicit professor guidance**, rather than an unresolved assumption. Its final performance and later shared budget retain the disclosed test-assisted limitation.

## Every question and subrequirement

| Requirement | Report evidence | Check/result |
| --- | --- | --- |
| A1: baseline block diagram | A p.1, Figure A1a | Supplied four independent embeddings, six continuous inputs, concatenation, Linear/LayerNorm/ReLU/output/inverse scale match code. |
| A1: baseline trainable parameters | A p.1 | Default 3,505; formula includes learned unknown-category rows, biases and LayerNorm affine parameters. Selected count 7,065 also correct. |
| A1: Ray grid H={16,32,64,128,256}, shared d={2,4,8,12,16} | A p.1, Figure A1b | All 25 completed Ray candidates/checkpoints and histories exist; every combination appears once. |
| A1: choose lowest best validation RMSE and report settings | A pp.1–2 | H=256,d=4; RMSE SGD58,627.10 at epoch6; independent selection from the full grid matches. |
| A1: training/validation RMSE on same axes | A p.2, Figure A1c | Correct selected-run history; no test epoch metrics in public history/plot. |
| A1: combined-data retraining, final RMSE/R² | A p.2, refit curve/table | 116,427 fitting rows; E*=4 of eight completed epochs; SGD96,901.64/R²0.676082. Private test monitor matches clarified stopping. |
| A1: justify settings and discuss under/overfitting | A p.2 | Uses full-grid alternatives, actual train/validation gaps, late deterioration, parameter counts and temporal/one-seed limitations. |
| A2: baseline and two separate ablations over five seeds | A p.3, Figures A2a/A2b | All 15 fresh runs, seeds42–46. Separate categorical and activation figures. |
| A2: remove all embeddings, six inputs, fixed hidden width | A p.3 | Linear(6,256), LayerNorm/ReLU/output; 2,561 parameters. |
| A2: replace only hidden ReLU with Sigmoid | A p.3 | Same inputs, d=4,H=256, LayerNorm and 7,065 parameters; same-seed initial tensors matched. |
| A2: same preprocessing/hyperparameters/epochs, one final test each | A p.3 | Exact A1 refit preprocessing, LR0.01,batch512,four epochs; all run records/histories checked. |
| A2: RMSE/R² means/SDs, plots and conclusions | A p.3 | Independently recomputed every mean/sample SD from five records; paired trends, capacity confound and small Sigmoid gain discussed. |
| A3: shared concatenated input, selected MLP, Linear(...,1), weighted sum | A p.4 | Both branches use same22-vector; output scaling applied once; extra23/total7,088 parameters. |
| A3: λ grid {0,0.25,0.5,1,2}, fixed A1 settings | A p.4 | Five validation candidates; independent choice λ2 matches saved selection. Zero-lambda trained baseline parity is preserved. |
| A3: five seeds, sample SD, relevant plots and baseline comparison | A p.4 | Seeds42–46, matched initialization/order/budget; same five A2 baseline records reused. A1 individual final score remains separately visible. |
| A3: mechanism, consistency of evidence, limitation/confound | A p.4 | Additive route/gradient path, positive-λ reparameterization, finite optimization, modest gains and extra-parameter/one-seed/test-budget confounds explained. |
| A4: Captum FeatureAblation, same training seed, every sample/ten original features | A p.5 | A1/A3 seed42,epoch4; both saved signed matrices are26702×10 with exact original test row IDs. |
| A4: mean absolute SGD attribution, ten descending labelled bars/model | A p.5, Figures A4a/A4b | Independently recomputed all 20 global scores from signed matrices and checked descending ranks/SGD units. |
| A4: top three, compare and one limitation | A p.5 | Correct top-three lists; third/fourth swap; shared references, correlated replacements, signs/nonadditivity and one-seed limits. |
| B1: exact AlexNet-like diagram, batch128 and five learning rates | B pp.1–2 | Conv/pool/FC/dropout layout and99,569 parameters match. All five rates1e-5…1e-1 and run histories verified. Final-logit interpretation disclosed above. |
| B1: separate validation loss/accuracy figures | B p.2, Figures B1a/B1b | Both separately generated figures contain all five curves; train panels also included; stopping endpoints/checkpoint stars checked. |
| B1: required six-column selection table | B p.1 | LR,best epoch,validation loss,error,sensitivity,specificity all from the same saved checkpoint; report audit matches all values. |
| B1: lowest-error choice, convergence, sensitivity/specificity reasoning | B pp.1–2 | LR0.001,epoch9; curves/slow or unstable rates, imbalance and conditional-metric tradeoffs supported by saved evidence. |
| B1: one final test accuracy/sensitivity/specificity | B p.2 | 85.4167%/99.7436%/61.5385%; saved same-pass predictions reproduce confusion counts and metrics without inference. |
| B2: exact VGG-like diagram, frozen B1 rate, required five batches | B pp.2–3 | Six Conv/BN/ReLU layers, three pools/global average/FC;18,185 parameters. LR0.001, batches16/32/64/128/256 verified. |
| B2: separate loss/accuracy figures and required table | B pp.2–3, Figures B2a/B2b | All five validation curves/rows plus train panels; table metrics match saved selected checkpoints. |
| B2: lowest-error selection and explanation | B pp.2–3 | Batch16,epoch13;11 errors versus15/25 alternatives. Update-budget/BN/gradient interpretations are appropriately limited. |
| B2: one final test accuracy/sensitivity/specificity | B p.3 | 85.5769%/99.2308%/62.8205%; independent saved-prediction recomputation passes. |
| B3: compact ResNet diagram, inherited LR/batch and final performance | B p.4 | Stem, three residual blocks, two projections, ReLU after sums, global average/FC; LR0.001,batch16,77,169 parameters. Saved test metrics independently reproduce. |
| B3: learning curves as additionally requested by slides | B p.4, Figures B3a/B3b | Train/validation loss and accuracy curves, saved epoch9 marker and interpretation present. |
| B3: full three-model comparison table | B p.4 | Accuracy,sensitivity,specificity,error,parameters,FLOPs for all three; saved B1/B2 final tests reused. |
| B3: FLOPs convention and architecture pros/cons | B p.4 | Independent executed-shape formulas reproduce39,063,616/19,464,256/25,325,696; two FLOPs/MAC plus28,672 B3 residual additions; exclusions/recipe confounds stated. |
| B4: randomly select five pneumonia examples, same three models | B p.5 | Independent PCG64 seed42 draw reproduces486/279/413/53/276, with positive-label semantics explicitly assumed. |
| B4: Captum Grad-CAM overlays and comparison | B p.5, Figure B4 | Five originals plus15 overlays; exact selected checkpoints/layers, positive-logit target, per-map scaling and shared display. Every image compared with localization limitations. |

## General and clarified requirements

| Requirement | Result |
| --- | --- |
| Individual work / ownership | No copying from others introduced; the supplied baseline is clearly identified. Both reports declare GenAI assistance and student review/understanding responsibility. Actual ownership/understanding cannot be certified by an automated audit. |
| Two separate reports, two Python code ZIPs | Four current upload files listed in [submission manifest](../submission/manifest.json). Each ZIP has an independent root README and full Python implementation. |
| Reports strictly less than six pages, concise answers | Exactly **five pages each**, including GenAI declarations/references. All ten pages visually inspected again; font/bounds/glyph/table audits pass. |
| GenAI declaration at beginning: tool, purpose, verification | Both p.1 declarations cover Codex/ChatGPT, code/debugging/execution/analysis/figures/report/packaging and actual verification. Full log includes this recheck. |
| Three-way data partitions and training-only preprocessing | Housing chronology and both fitting protocols checked; supplied X-ray partitions/raw pixels/labels unchanged and train-only normalization independently reproduced. |
| Adam, maximum20,patience4,min improvement1e-4 | All selection/refit stopping trajectories checked. A2 follows its explicit fixed-A1-epoch requirement; A3's same fixed budget is an explicit controlled-comparison assumption. |
| Updated plots: train/validation against epochs, no test epoch plots | Public run histories/figures comply. A1 combined refit plots training only; final test RMSE/R² appear in tables/barplots. Private A1 stopping monitor is retained for audit. |
| Separate A2 ablation presentations | Two labelled independent figures/sections, each comparing its ablation with baseline and showing both final metrics. |
| Brief/slides conflicts resolved | Brief's λ={0,0.25,0.5,1,2}, batch256 instead of258, strict five-page reports and clarified plot rules are used. Both late-penalty wordings differ; submit on time. |
| READMEs: environment, dependencies, seeds, commands, data | Both standalone READMEs cover every item plus raw-source hashes, output locations, fresh-result pipelines and export-only commands. All packages remain installed in a Python venv. |
| Clean independently runnable ZIPs | Integrity/extraction audits pass; venvs/caches/opposite-part code excluded.57 A/62 B extracted tests previously passed;96 distinct root behavioral tests passed again on October 4. |
| Correct actual author in reports/filenames | **Pending.** `YourName` remains because the student's actual name has not been supplied. |
| NTULearn upload before October12 and successful receipt | **Pending.** No authenticated NTULearn connection or receipt is available. Check its actual cutoff and retain final confirmation. |

## Verification evidence and scope

[Recheck JSON](../artifacts/final_recheck/verification.json) verifies **62 saved run histories**, both preprocessing protocols, all required grids/seeds and selection/stopping rules, matching controls, independent mean/sample SDs, full-test SGD aggregation, three saved-prediction confusion matrices/BCE values, consistent FLOPs formulas and all 15 stored Grad-CAM map reconstructions. Grad-CAM arithmetic reused saved tensors with the original CUDA deterministic settings/batch geometry, retaining exact equality; no model forward/backward was repeated. Original artifacts/hashes preserve the earlier independent autograd/FeatureAblation inference audits.

All **938 earlier experiment output hashes** remain intact. [Report verification](../reports/verification.json) additionally checks supplied inputs, report metrics/content/fonts/layout and the corrected normalization/output-assumption wording. [Latest package audit](../artifacts/submission_checks/latest.json) verifies the four rebuilt files and independent extractions; [checksums](../submission/SHA256SUMS.txt) match current files. No full training runs or test-set model inference were repeated during this review.

The review confirms answer coverage and saved evidence under the disclosed assumptions. The final-activation interpretation remains unconfirmed; the student still needs to review/understand the work, supply their actual name and complete the portal submission.
