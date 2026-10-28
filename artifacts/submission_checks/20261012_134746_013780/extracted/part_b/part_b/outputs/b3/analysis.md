# B3 and three-architecture comparison draft

## B3 executed protocol and results

The compact Figure B3 ResNet-like model has 77,169 trainable parameters. It uses the B1-frozen learning rate **1e-03** and B2-frozen batch **16**, seed 42, full supplied training/validation partitions, saved training-only normalization, Adam, maximum 20 epochs, patience 4, and significant loss improvement 1e-4. It has a Conv/BN/ReLU stem, max pooling, three residual blocks at widths 16/32/64, two learned stride-two projections, ReLU after residual additions, global averaging, and one biased raw-logit FC output. There is no dropout, augmentation or class weighting. The final model is a separate fresh combined-data refit required by Q&A.md.

Stopping monitors validation loss. The checkpoint minimizes validation classification error, then loss, then earlier epoch. Selected **epoch 9**, after **13 completed epochs**: validation accuracy **97.5191%**, error **2.4809%**, loss **0.061988**, sensitivity **99.2288%**, specificity **92.5926%**. The minimum validation loss is **0.061988 at epoch 9**. Training/validation accuracy at selection is **99.4265%/97.5191%**; losses are **0.021046/0.061988**. Separate loss/accuracy figures plot both partitions, with no test epoch curves.

Selection froze the epoch count and hyperparameters before a fresh seed-42 model and Adam state trained on 5,232 combined training/validation images for exactly 9 epochs. The original training-only normalization is retained consistently under the brief. This final `refit/best_model.pt` checkpoint received one test evaluation on 624 images. Test **accuracy 81.0897%, sensitivity 100.0000%, specificity 49.5726%**, classification error **18.9103%**, BCE **0.823993**; TP/TN/FP/FN **390/116/118/0**. The same forward pass captured predictions for later auditing without new model test inference.

## Required architecture comparison

All performance values are saved final test results. B1/B2's existing results were read without another test pass. Percentages are shown here; CSVs use proportions. FLOPs are per float32 1×1×64×64 synthetic image in evaluation mode, within the profiled scope below.

| Model | Test accuracy | Sensitivity | Specificity | Classification error | Learnable parameters | FLOPs/image, profiled scope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 AlexNet-like | 90.7051% | 97.6923% | 79.0598% | 9.2949% | 99,569 | 39,063,616 |
| B2 VGG-like | 85.2564% | 100.0000% | 60.6838% | 14.7436% | 18,185 | 19,464,256 |
| B3 ResNet-like | 81.0897% | 100.0000% | 49.5726% | 18.9103% | 77,169 | 25,325,696 |

## FLOPs convention and audit trail

`torch.profiler.profile(with_flops=True, record_shapes=True, acc_events=True)` profiles a CPU evaluation copy of each model on one synthetic zero image. One convolution/linear multiply-add is **two FLOPs**. Actual nonzero profiler events are convolution, linear matrix multiplication, and B3 residual elementwise addition; the latter adds one FLOP per output element. Layer-shape formulas reproduce both components exactly. Model state and BN buffers remain unchanged; no dataset is used for complexity profiling.

| Architecture | Conv/linear MACs | Conv/linear FLOPs = 2×MACs | Residual additions | Reported FLOPs |
| --- | ---: | ---: | ---: | ---: |
| B1 | 19,531,808 | 39,063,616 | 0 | 39,063,616 |
| B2 | 9,732,128 | 19,464,256 | 0 | 19,464,256 |
| B3 | 12,648,512 | 25,297,024 | 28,672 | 25,325,696 |

BN, ReLU, pooling, FC bias addition, views, data normalization/transfers, probability sigmoid/threshold, and backward/optimizer work are excluded. Uncounted operations are recorded explicitly; their omission does not mean zero computational cost. These are operation-count estimates, not measured latency, memory, throughput, or a complete scalar-operation inventory. See [PyTorch profiler documentation](https://docs.pytorch.org/docs/2.11/profiler.html). Detailed layer shapes, MACs, nonzero/zero profiler operators, residual additions, and parameter breakdowns are saved in `complexity/` and `tables/`.

## Architecture strengths and limitations

B1 has three convolution stages and larger 5×5 early kernels. Its spatial 4×4 output feeds FC64/FC32 with dropout; the linear layers contain 67,713 parameters, most of its 99,569 total. It can retain spatial combinations in the head, but has the highest parameter and profiled operation counts in this comparison.

B2's six smaller 3×3 convolutions stack nonlinear transformations before global averaging. Its 33-parameter output head yields the fewest parameters (18,185) and profiled FLOPs. Global averaging compresses spatial information; this economy does not itself prove superior generalization. Its assigned training recipe uses batch 16, with many more updates per epoch than B1's batch 128.

B3 adds identity/projection shortcut routes while preserving shape alignment and applying the required ReLU after sums. These routes are intended to ease optimization, as explained in [He et al.'s residual-learning paper](https://arxiv.org/abs/1512.03385); this single run does not isolate their causal effect. Its 77,169 parameters and profiled FLOPs lie between B1 and B2. Learned projections and wider convolution channels add cost relative to B2; global averaging keeps its head at 65 parameters. The assignment's ReLU placement follows Figure B rather than a library ResNet preset.

The table compares selected assignment pipelines. The common LR was tuned on B1, batch size was tuned on B2, B1 uses batch 128, and B2/B3 use batch 16. One seed per final model and differing checkpoint epochs/update budgets prevent attributing differences to architecture alone. High sensitivity must be read alongside specificity and false positives; all models use the frozen 0.5 threshold, and the comparison does not select a new model or tune settings from test results.

Q&A.md confirms 0=healthy/1=pneumonia; the NPZ itself lacks semantic class-name metadata. Eight distinct images overlap train/validation, potentially making validation optimistic; patient IDs are absent, and three extra copies occur within test. No exact train/test or validation/test pixel overlaps were found. Supplied partitions are preserved. The sources of validation/test differences are not established.

## Handoff

`frozen_b3_settings.json` records separate validation-selection and fresh final-refit B3 checkpoints and B1/B2/data lineage for B4. B4 should load the saved final-refit checkpoints and compare Grad-CAM on the same five sampled positive-class test images. Current B4 explanations and reports/packages use the canonical FAQ-corrected final results.
