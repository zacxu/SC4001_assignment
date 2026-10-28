# A1 configuration and fitting interpretation review

Reviewed October 11, 2026 against the A1 question in assignment.pdf, the briefing slides, instructor clarifications/Q&A, all 25 saved grid scores and both selected-model learning histories. This is an explanation revision, not a new experiment. The previous reports, packages and explanation are preserved in `artifacts/a1_interpretation_review/previous_files`.

The friend's explanation identifies a useful width trend, but its conclusion of underfitting does not follow from a high test plateau or the ratio of hidden units to training rows. Their H=256,d=2 configuration could be the winner of their own run; their grid and curves have not been supplied. Our answer must use our saved H=256,d=4 winner.

## Selection evidence

| Comparison | Best validation RMSE (SGD) |
| --- | ---: |
| Selected H=256,d=4 | 58,627.10 |
| H=256,d=2 | 63,213.60 |
| Supplied default H=64,d=8 | 62,109.01 |
| Next-best H=128,d=12 | 59,688.99 |

The selected configuration improves on the default by 5.61% and on H=256,d=2 by 7.26%, using the comparator's RMSE as denominator. It has 7,065 parameters, versus 20,577 at H=256,d=16. Higher complexity does not automatically improve validation performance.

Across all five embedding choices, the mean of each trial's best validation RMSE decreases with H: 65,100.41, 63,728.59, 62,756.66, 61,534.31 and 60,991.97 SGD for widths 16,32,64,128,256. This supports choosing a wider hidden layer within the tested grid. It does not show why that improvement occurs, establish that width256 cannot memorize data, or prove that still wider layers would help.

At fixed H, increasing d has mixed effects. At H=256, both d=2 and every d above4 score worse than d=4. Averaging over H nevertheless gives improving marginal averages for d=2,4,8,12,16: 63,975.13, 62,980.52, 62,505.78, 62,374.37 and 62,276.14 SGD. Therefore distinguish a marginal average trend from the best joint configuration. A single seed cannot establish whether a ranking is stable or whether a difference is chance; it also cannot justify claiming the winner won by chance.

## Fitting evidence

| Run/checkpoint | Training RMSE (SGD) | Held-out RMSE (SGD) |
| --- | ---: | ---: |
| Original split, epoch6 (selected) | 34,372.58 | 58,627.10 validation |
| Original split, epoch10 (last) | 31,678.96 | 70,715.09 validation |
| Combined-data refit, epoch4 (selected) | 41,239.37 | 96,901.64 test |
| Combined-data refit, epoch8 (last) | 39,074.75 | 99,668.21 test |

Every original-split validation checkpoint after epoch6 is worse than the selected minimum while training error is lower. Further fitting improves training performance without improving future-year performance, consistent with late overfitting. The refit shows a similar endpoint pattern, although its test values fluctuate rather than rise monotonically. This supports checkpoint restoration, not a claim that the selected epoch is completely free of model error.

A high or plateauing held-out error alone cannot diagnose underfitting. Underfitting requires evidence of inadequate training fit relative to an achievable reference, alongside poor generalization. At fixed d=4, H=16 has training RMSE 34,554.54, only 181.96 SGD (0.53%) above H=256; its validation RMSE is 64,672.61. These similar training errors do not establish substantial underfitting. These values are from each trial's best validation checkpoint (epochs 10 and 6), so optimization duration also differs.

Overfitting does not require memorizing all 87,370 training examples. Width and sample count alone ignore parameter count, feature structure, noise, optimization and regularization. Thus a relatively small model can still fit relationships that fail on a future-year partition. Limited capacity and late overfitting can also coexist; the plots do not identify a full bias/variance decomposition.

## Appropriate conclusion and boundaries

Choose H=256,d=4 and epoch6 because these minimize the observed validation RMSE. The strongest fitting conclusion is evidence consistent with late overfitting after that checkpoint, rather than persistent underfitting inferred from the test plateau. Wider H generally helps within the tested grid, while a modest d wins the measured joint search.

The curves are noisy and chronology matters. Unadjusted observed price means are 442,495.91 SGD in 2017–2020, 511,417.39 in 2021 and 549,701.07 in 2022; property composition also changes, so those means do not isolate a causal market effect. Temporal shift and optimization fluctuations may contribute to the gap. One seed and one validation year do not establish a stable ranking or a global optimum.

The professor authorized A1 test-based refit stopping, but its selected final test score is consequently not an untouched-holdout estimate. The supplied FAQ's train/validation-only epoch rule addresses A2/A3 and does not remove A1's express train/test plot requirement. These protocol limitations remain disclosed in the report.

Evidence: [grid scores](../part_a/outputs/a1/tables/grid_results.csv), [selected learning history](../part_a/outputs/a1/grid/h256_d04/training/history.csv), [refit training history](../part_a/outputs/a1/refit/history.csv), [authorized test monitor](../part_a/outputs/a1/refit/internal_test_monitor.jsonl), and [post-fit price summaries](../part_a/outputs/a1/tables/partition_price_summary.csv).
