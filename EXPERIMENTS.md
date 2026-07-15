# Short experimental exploration (2026-07-15)

This note records bounded pilot runs used during the mathematical refactor. No
full-data or long-duration training was attempted.

## Common pilot protocol

Unless stated otherwise:

- city: Beijing;
- source files: `small` train/validation/test;
- first 512 records inspected in each split;
- complete edge paths, with positive-cost cycles dropped;
- accepted train/validation/test samples: 409/398/405;
- projected solver, six epochs, validation every epoch;
- four Rayon workers, release build;
- multiplier box `[0.1, 10]`;
- final test metrics computed after restoring the validation-selected checkpoint.

These are ordered, small pilot subsets rather than randomized replicates. Test
results were inspected while exploring configurations, so they are diagnostics,
not an unbiased final generalization estimate. A formal study must tune only on
validation data, freeze the protocol, and evaluate an untouched test set once.

## Optimizer and step-size exploration

Raw regret is in the integer metric's cost units.

| Method | eta0 | lambda | Train regret | Validation regret | Test regret | Test exact | Test F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline `q=1` | — | — | 575,867 | 736,817 | 688,406 | .3630 | .5945 |
| Projected | 1e-6 | 1e5 | 547,658 | 726,637 | 678,381 | .3333 | .5870 |
| Projected | 3e-6 | 1e5 | 501,516 | 711,813 | 662,343 | .3185 | .5890 |
| Projected | 1e-5 | 1e5 | 398,580 | 683,912 | 631,451 | .3086 | .5866 |
| Projected | 3e-5 | 1e5 | 267,699 | 678,852 | 621,220 | .2914 | .5954 |
| Legacy Adam | — | 0 | 370,237 | 702,708 | 662,218 | .2593 | .5659 |

The `3e-5` projected pilot gave the lowest validation/test regret and the best
F1 in this tiny sweep. It was not made the default: `1e-5` is a more conservative
starting point for other sample sizes, because the raw gradient scale depends on
baseline costs and sample frequencies.

An aggressive `eta0=1e-4, lambda=1e7` run demonstrated overfitting and
projection:

- unselected last train regret fell by 71.8%, to 162,616;
- validation regret worsened from 736,817 to 773,243;
- several edges repeatedly hit the `q=0.1` lower bound;
- validation correctly selected epoch 0 and restored the initial weights.

A 20-epoch `eta0=3e-5, lambda=1e7` run stopped at epoch 13 after eight stale
validation evaluations. The selected epoch was 5: train/validation/test regret
was 268,490/678,793/620,870, and test F1 was .5957. Training regret continued
falling after epoch 5 while validation degraded, showing why validation-based
checkpointing is necessary.

## Legacy Adam behavior

Within six epochs, legacy Adam reduced train regret and count residual quickly,
but validation/test results were worse than both projected `1e-5` and `3e-5`.
Its inferred multipliers spread from approximately `[0.00035, 1.99]` after one
step to `[0.000067, 4.86]`; 261 edges were already below `q=0.1`. This is the
scale/boundary degeneration that the baseline-relative box and regularizer are
designed to prevent. No shock was triggered in this short run.

## Regularization scale

At `eta0=1e-5`, increasing lambda from `1e7` to `1e9` made anchoring materially
visible:

| lambda | Selected q range | Selected regularizer | Test regret | Test F1 |
|---:|---|---:|---:|---:|
| 1e7 | about `[0.657, 1.219]` | 384 | 631,502 | .5872 |
| 1e9 | `[0.705, 1.187]` | 27,908 | 636,804 | .5885 |

The stronger setting narrowed the selected multiplier range by about 14%, but
increased test regret by about 0.84%. It illustrates the intended tradeoff: lambda
chooses a more baseline-like representative among behaviorally similar costs;
it is not a free improvement in data fit.

## Count residual is not the objective

Several projected runs had an epoch where `count_residual_l1` increased while
both train and validation regret continued to decrease. Conversely, Adam could
obtain a smaller residual and lower train regret than projected `1e-5` while
generalizing worse. This is direct experimental evidence for keeping the count
residual as a diagnostic and selecting checkpoints by regret.

## Full versus partial CCH customization

For projected `eta0=1e-5`, full and partial customization produced byte-identical
weight and multiplier checkpoints and identical route metrics. At roughly
9%–11% changed original edges, the measured customization times were:

| Mode | Mean customization per updated epoch |
|---|---:|
| Partial | about 50 ms |
| Full | about 12–25 ms |

Thus partial customization was about 2–4 times slower in these dense-update
pilots. Full customization is now the default; partial remains available for
experiments where integer changes become genuinely sparse. These timings are
pilot measurements, not a calibrated crossover curve.

## Data-policy ablation

One-epoch runs make no optimizer update and expose only the baseline task.

| Policy | Accepted train/val/test | Train regret | Validation regret | Test regret | Test exact |
|---|---|---:|---:|---:|---:|
| Complete path, drop cycles | 409/398/405 | 575,867 | 736,817 | 688,406 | .3630 |
| Complete path, keep cycles | 512/512/512 | 922,097 | 1,081,808 | 1,006,136 | .2871 |
| Trim first/last edge, drop cycles | 452/442/444 | 432,313 | 593,220 | 554,557 | .4527 |

Keeping cycles adds observations that cannot be positive-cost shortest paths,
mechanically raising regret. Trimming appears better only because it shortens
the route, changes the OD, and removes some cycles; it is a different and easier
prediction task. These rows must not be interpreted as an algorithm ranking.
Relative regret is now also logged to make comparisons across cost/path-length
scales less misleading.

## Runtime and scaling observations

- A 512-record six-epoch release pilot took roughly 1.1–1.4 seconds after
  compilation and used about 40–50 MiB in the measured process.
- The old debug implementation processed about 62 epochs on 10,000 records in
  15 seconds, but logged only count residual and repeatedly evaluated test data.
- Full Beijing contains about 786k/309k/322k train/validation/test records.
  Deserializing the full collections was observed to require roughly 1.4 GiB in
  a separate read-only audit.
- Full-path OD grouping saves almost nothing on the first 512 ordered records,
  but about 9.8% of train queries on the full Beijing split.

The next defensible experiments are randomized multi-seed validation sweeps,
changed-edge-ratio crossover measurements for CCH updates, and then staged runs
on the existing 1%/5%/10% partial training files.
