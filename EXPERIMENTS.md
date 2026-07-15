# Beijing convergence and error-attribution follow-up (2026-07-15)

This follow-up asks whether the 20-epoch edge-only result below was an
optimization-budget artifact or a stable representation limit. The protocol
was frozen before reading the new development metrics. It uses full Beijing
train, a time-blocked development set, and two later, source-index-disjoint
one-shot confirmation blocks. Every training command had a hard 900-second
per-run timeout; the smaller turn-capacity probe has a 180-second timeout. This
follow-up kept test contents sealed: formal training, diagnostics, confirmation,
and the capacity probe did not load or evaluate test, and no test contents or
metrics were used.

The new development block contains 163,614 raw / 129,033 valid routes after the
unchanged positive-cycle filter. The learning-rate grid was
`{1e-4, 3e-4, 1e-3}` at 10% and full train, with `lambda=1e5`, multiplier box
`[0.1, 10]`, at most 100 epochs, validation every five epochs, and four-event
early stopping with absolute minimum delta `1e-5`. The deterministic first 20
epochs of the long `eta0=1e-4` run exactly matched the separate 20-epoch
control.

| Development model | Relative regret | Edge F1 | Exact | Best epoch |
|---|---:|---:|---:|---:|
| Unlearned `q=1` | 0.09352278 | 0.591144 | 0.337542 | 0 |
| Full, `eta0=1e-4`, 20 epochs | 0.06826350 | 0.656391 | 0.346694 | 19 |
| Full, `eta0=1e-4`, up to 100 | 0.06357497 | 0.677356 | 0.367921 | 99 |
| Full, `eta0=3e-4`, up to 100 (selected) | **0.06348409** | **0.681488** | **0.371068** | 99 |

The 100-epoch checkpoint clearly improves on the former budget, but both viable
full-data trajectories selected epoch 99. Therefore the result is evidence
that 20 epochs was too short, not a proof that optimization has converged. This
horizon conclusion comes from the exact same-eta `1e-4` prefix comparison
`0.06826350` to `0.06357497`; the ultimately selected `3e-4` checkpoint is a
separate model-selection gain. The aggressive `eta0=1e-3` runs stopped after 20
epochs and selected the unlearned epoch-0 checkpoint. No automatic extension
beyond 100 epochs was allowed.

After the primary checkpoint was frozen, `q=1`, the 20-epoch control, and the
primary model were each evaluated exactly once on AM and PM confirmation
blocks. After cycle filtering they contain 15,776 and 15,886 routes.

| Confirmation scope | Model | Relative regret | Edge F1 | Exact |
|---|---|---:|---:|---:|
| AM | `q=1` | 0.09210652 | 0.599664 | 0.353892 |
| AM | 20 epochs | 0.06724488 | 0.662546 | 0.356237 |
| AM | 100 epochs | **0.06307110** | **0.686270** | **0.376331** |
| PM | `q=1` | 0.09285387 | 0.593518 | 0.343573 |
| PM | 20 epochs | 0.06767404 | 0.659278 | 0.353204 |
| PM | 100 epochs | **0.06298240** | **0.682767** | **0.376684** |
| Pooled | `q=1` | 0.09246791 | 0.596580 | 0.348715 |
| Pooled | 20 epochs | 0.06745259 | 0.660906 | 0.354715 |
| Pooled | 100 epochs | **0.06302821** | **0.684512** | **0.376508** |

For the pooled paired comparison, 100 versus 20 epochs improved relative regret
by `0.00442438` (95% stratified-bootstrap interval
`[0.00405114, 0.00483054]`), F1 by `0.023606`
(`[0.021152, 0.025967]`), and exact match by `0.021793`
(`[0.017971, 0.025551]`). For this 20-to-100 comparison, each metric improved
in all 2,000 paired replicates.

This confirmation compares the complete frozen candidates, so it jointly
reflects their different horizon and eta; it is not a same-eta causal estimate
of the epoch budget.

These blocks are source-index-disjoint confirmation within the validation
source, not an untouched final test estimate, and they are now spent:
loop-policy and turn-feature work may not use them for selection or
confirmation.

The selected model still diverges from the observed route on 62.89% of
development routes, but this is lower than `q=1` (66.25%) and the 20-epoch model
(65.33%). Its first-edge accuracy is 81.94%, mean common-prefix ratio is 50.13%,
and mean edge F1 is 68.15%. Of divergent routes, 69.32% first diverge at a
complex junction, 82.26% rejoin before the target, and the median relative
suffix-cost gap is only 4.06%. The model predicts 9,855 left turns versus 8,531
observed at classifiable first divergences. Route length remains a major
difficulty axis. After conditioning on its first quartile, the 0 and 1--2
complex-junction bins are a small nonmonotone exception, followed by much worse
rates at higher complexity; the second quartile is monotone. This motivates a
minimal turn-cost probe, but does not establish a causal turn-cost omission.

A separate read-only audit found 162,434 cyclic records among 785,709
structurally valid full-train records (20.67%). Default dropping retains 79.33%
of records and 77.49% of graph edges. Chronological loop erasure retains 99.94%
of records and 81.03% of graph edges while deleting 4.25% of edge occurrences.
The dropped population is systematically longer and slower, so dropping induces
a real selection effect. However, frozen-hyperparameter training with erased
loops was not a stable improvement: at 10% it changed regret from `0.06430115`
to `0.06412951` with essentially flat/slightly lower F1, while at full scale it
worsened regret from `0.06348409` to `0.06486248`. `drop` therefore remains the
default; `erase` is implemented for a future preregistered, matched-retuning
study rather than promoted from this ablation.

The final bounded capacity experiment froze the selected edge weights and added
one nonnegative global left-turn penalty on an expanded edge-state graph. A
deterministic 20,000-route development subset tuned
`r={0,.05,.1,.2,.4}` and the next 10,000 routes were reserved for audit. All
30,000 `r=0` expanded distances and observed costs exactly matched the original
edge-only oracle, and every correctness gate passed. Tune regret was
`0.06341452/0.06378618/0.06430391/0.06559292/0.06892272`; the primary rule
therefore selected `r=0`. The predeclared regret and F1 improvement gates failed
and no new confirmation block was opened. This is evidence against a single
city-wide nonnegative left-turn penalty with fixed edge weights, not against
conditional, node-specific, temporal, or jointly optimized turn models.

Full protocol details, artifact paths, limitations, and bounded reproduction
commands are in
[`experiments/convergence_study/RESULTS.md`](experiments/convergence_study/RESULTS.md).
The accompanying reproducibility bundle contains the already-created six
one-shot route exports, three frozen checkpoints, and ten training logs; it
contains no raw/test data and did not require rerunning confirmation.

---

# Randomized Beijing scale study (2026-07-15)

This study supersedes the small ordered pilots below for statistical claims. It
uses manifest-backed random subsets, three train seeds, one fixed 20,000-record
validation subset, validation-only selection, release builds, and a hard
15-minute timeout per run. No model was evaluated on test.

## Reproducibility and data provenance

Beijing has 31,199 nodes, 72,156 directed edges, and 785,709 raw train records.
The full train pickle SHA-256 is
`d7fdfb5870c54df79d1044ecb12a076e0244dbd5d3bc74fd67d1bdcc2b7c0fce`.
The new generator uses `rand 0.8.5 StdRng(seed_from_u64)`, shuffles all full-file
indices without replacement, takes a prefix, and records the selected indices
and source/output SHA-256 values. Therefore the 1%, 5%, and 10% subsets for one
seed are nested, while seeds 42/43/44 are independent.

The validation subset contains 20,000 raw records sampled with seed 20260715;
15,812 remain after dropping 4,188 positive-cost cycles. Every formal run uses
this same validation set and logs `TEST_SKIPPED`. Exact manifests are in
`experiments/scale_study/subset_manifest.json`.

The pre-existing `all_partial_1.0/5.0/10.0` files were also audited. They are
exact no-replacement subsets of full and statistically resemble independent
random samples, but they have no generator, seed, or manifest, so they were not
used as named-seed replicates. Directly taking the first N full records is not a
valid substitute: full is ordered in date blocks.

One transparency caveat: during preliminary data-schema/count auditing, a
read-only helper deserialized the test pickle to inspect count and schema. It
did not build a graph, run an oracle, load a checkpoint, or compute any model
metric. This nevertheless violates a literal “never read test” rule; all formal
training and selection after that audit avoided test completely.

## Coverage audit before optimization

All paths below are complete edge paths with cycles dropped. “Unseen validation
routes” means at least one observed route edge has zero train count.

| Raw train | Valid | Cycles | Unique OD | Graph-edge coverage | Observed edges seen once | Unseen validation routes |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 783 | 21.70% | 783 | 16.14% | 51.61% | 84.21% |
| 5,000 | 3,930 | 21.40% | 3,923 | 31.51% | 31.48% | 45.48% |
| 7,857 (1%) | 6,147 | 21.76% | 6,132 | 36.13% | 27.81% | 35.01% |
| 39,285 (5%) | 31,167 | 20.66% | 30,878 | 51.98% | 19.16% | 10.53% |
| 78,570 (10%) | 62,348 | 20.65% | 61,253 | 58.61% | 15.99% | 5.53% |
| 785,709 (full metadata) | 623,275 | 20.67% | 561,131 | 77.49% | 8.65% | 0.58% |

This directly validates the concern behind the rerun: a few hundred valid
routes leave most of the graph and most validation routes unsupported. Across
all three formal seeds, unseen-validation-route rates were 35.18% at 1%, 10.44%
at 5%, and 5.70% at 10%. The full audit took 4.22 seconds and peaked at about
1.42 GiB RSS. After the 10% timing established that a full run was safe, one
already-selected configuration was also trained on full.

## Baseline and selection metric

Checkpoint selection now has an explicit
`--selection-metric relative-regret` mode. The study minimizes aggregate

```text
sum(route regret) / sum(observed route cost)
```

on validation, with edge F1 secondary. This is different from the mean of the
per-route relative regrets, which remains a distribution diagnostic.

For `q=1` on all 15,812 valid validation routes:

| Mean raw regret | Aggregate relative regret | Exact | Edge F1 | Jaccard |
|---:|---:|---:|---:|---:|
| 650,449.94 | 0.09455969 | 0.336643 | 0.589902 | 0.519793 |

Raw regret median/p75/p90/p95 were
85,426 / 670,428 / 1,740,892 / 2,897,846. Per-route relative-regret median and
p90 were 0.01652 and 0.19999. Baseline epsilon-optimal rates were:

| epsilon | 0% | 1% | 5% | 10% |
|---:|---:|---:|---:|---:|
| Rate | 33.677% | 45.788% | 63.819% | 76.973% |

## One-percent grid and three-seed selection

The prescribed 15-point grid used 20 epochs, validation every epoch, patience
5, full customization, `[0.1, 10]`, and seed 42. All runs completed in about
6.2--6.5 seconds and none timed out or became nonfinite. The top seed-42 rows
were:

| eta0 | lambda | Relative regret | F1 | Exact | Best epoch | q range |
|---:|---:|---:|---:|---:|---:|---|
| 1e-4 | 1e5 | 0.07222536 | 0.643864 | 0.329939 | 18 | [0.260, 1.402] |
| 1e-4 | 1e7 | 0.07245321 | 0.642009 | 0.327916 | 18 | [0.262, 1.389] |
| 3e-5 | 1e5 | 0.07548939 | 0.636210 | 0.334809 | 19 | [0.446, 1.289] |

Replicating those configurations on seeds 42/43/44 produced:

| eta0 | lambda | Relative regret | Mean regret | F1 | Exact |
|---:|---:|---:|---:|---:|---:|
| 1e-4 | 1e5 | 0.0722676 ± 0.0005561 | 452,828 ± 4,838 | 0.644525 ± 0.001748 | 0.328864 ± 0.001221 |
| 1e-4 | 1e7 | 0.0723962 ± 0.0006139 | 454,948 ± 5,343 | 0.642088 ± 0.004921 | 0.326904 ± 0.005884 |
| 3e-5 | 1e5 | 0.0753638 ± 0.0003546 | 492,864 ± 3,600 | 0.634840 ± 0.002203 | 0.332385 ± 0.002501 |

Only one edge in one seed of the leading configuration reached `q=0.1`; no
edge reached `q=10`. Mean q p05/median/p95 for the leading 1% configuration was
0.946/1.000/1.027, so the isolated minimum is not broad saturation.

The representative legacy Adam run attained relative regret 0.07250979 and F1
0.642682, but its multipliers spread to `[0.000018, 65.20]`; 1,337 edges were
below the projected solver's lower bound and 16 exceeded its upper bound. Its
exact rate was also lower (0.318682). Similar regret therefore does not make it
an equally well-identified or stable inverse solution.

## Three-seed scale curve and full endpoint

The leading two configurations were run at 5% and 10%. The table below shows
the better mean configuration (`eta0=1e-4, lambda=1e5`); `±` is sample standard
deviation across seeds 42/43/44.

| Raw scale | Valid train | Validation relative regret | F1 | Exact | Relative train-val gap | Unseen val routes |
|---:|---:|---:|---:|---:|---:|---:|
| 1% | 6,208 ± 53 | 0.0722676 ± 0.0005561 | 0.644525 ± 0.001748 | 0.328864 ± 0.001221 | 0.012374 ± 0.002078 | 35.18% ± 0.20% |
| 5% | 31,147 ± 67 | 0.0692878 ± 0.0003575 | 0.657164 ± 0.002198 | 0.347036 ± 0.002009 | 0.004492 ± 0.000656 | 10.44% ± 0.11% |
| 10% | 62,286 ± 180 | 0.0688698 ± 0.0001485 | 0.658569 ± 0.001078 | 0.349397 ± 0.000958 | 0.003112 ± 0.000391 | 5.70% ± 0.14% |
| Full† | 623,275 | 0.06811979 | 0.658107 | 0.348722 | 0.001703 | 0.58% |

† Full is a single deterministic all-train endpoint with the already selected
configuration, so it has no across-seed standard deviation and was not used to
retune the hyperparameters.

Thus the baseline-to-1% relative-regret reduction was 23.6%. From 1% to 10%,
relative regret fell another 4.7%, F1 rose 1.40 percentage points, exact rose
2.05 points, and the train-validation gap shrank by about 75%. Gains from 5% to
10% were positive but much smaller. Full reduced relative regret a further
1.09% versus the 10% mean and reduced the relative train-validation gap again,
but F1/exact stayed essentially flat. This is strong evidence for diminishing
returns rather than an abrupt failure to generalize. The `lambda=1e7`
alternative was very close at
10% (0.0690618 ± 0.0000856 relative regret and 0.658439 ± 0.001402 F1).

Relative to `q=1`, full training reduced validation relative regret by 28.0%,
raised F1 by 6.82 percentage points, and raised exact by 1.21 points. Relative
to 1% training, full reduced relative regret by 5.74%; most of that extra gain
was already present at 5%.

Mean per-run wall times were 6.29/11.41/17.73 seconds for 1%/5%/10%. Mean epoch
times were 252/502/807 ms and peak RSS was 61.6/91.8/151.8 MiB. These epoch
times were measured while three four-thread processes shared the machine, so
they are capacity measurements rather than isolated single-process latency.
Among epochs that applied an update, 32.9%/46.8%/51.8% of integer edge weights
changed; mean full-customization time stayed near 14--15 ms.

The single full run finished in 90.24 seconds, averaged 4.27 seconds per epoch
(4.12 seconds in the train oracle), changed 62.1% of integer weights per update,
and peaked at 1.41 GiB RSS. Its best checkpoint was again the final epoch, so
the 20-epoch budget remains a declared limitation rather than proof of
convergence.

## Best-checkpoint route diagnostics

The lowest individual validation result was the full-train
`eta0=1e-4, lambda=1e5` checkpoint. It is a validation diagnostic, not an
untouched final-test estimate.

- aggregate relative regret 0.06811979; mean raw regret 425,958.71;
- exact 0.348722, F1 0.658107, Jaccard 0.580961;
- epsilon-optimal rates at 0/1/5/10%: 34.879/54.901/73.356/83.247%;
- correlation between per-route relative regret and `1 - F1`: 0.5072;
- zero-regret but non-exact: 1 of 15,812 routes (0.0063%), confirming that ties
  exist but do not explain the broad F1/regret relationship here.

Seen/unseen stratification shows a remaining data-coverage effect:

| Validation group | Routes | Aggregate relative regret | F1 | Exact |
|---|---:|---:|---:|---:|
| All route edges seen in train | 15,721 | 0.06707 | 0.65872 | 0.34991 |
| At least one unseen train edge | 91 | 0.19034 | 0.55289 | 0.14286 |

The unseen full-train group is only 91 routes, so its large gap is directionally
important but has much higher sampling uncertainty than the overall metrics.

Route length is an even stronger difficulty axis:

| Observed edge length | Routes | Aggregate relative regret | F1 | Exact |
|---|---:|---:|---:|---:|
| 3--15 | 4,207 | 0.03439 | 0.82505 | 0.68861 |
| 16--27 | 3,906 | 0.04579 | 0.70665 | 0.40092 |
| 28--45 | 3,810 | 0.05355 | 0.60990 | 0.21601 |
| 46--259 | 3,889 | 0.08804 | 0.47599 | 0.05863 |

Regret and route overlap are therefore aligned moderately, not “fundamentally
misaligned.” The remaining errors concentrate on unseen-edge and long-route
cases, which is evidence for coverage and model-capacity follow-ups.

## Controlled CCH customization measurement

Nine repeats used one nested random edge permutation and changed weights by
about 1%. Median timings with four Rayon threads were:

| Changed original edges | Full | Partial | Partial/full |
|---:|---:|---:|---:|
| 1.00% | 11.58 ms | 21.47 ms | 1.85x |
| 5.00% | 11.72 ms | 45.80 ms | 3.91x |
| 10.00% | 11.67 ms | 54.87 ms | 4.70x |

Full customization is decisively appropriate for the naturally dense updates
in this study. Partial remains useful only if a future optimizer produces much
sparser integer changes than the 1% controlled case.

## Interpretation and next scientific steps

The user's hypothesis is supported: conclusions from a few hundred routes were
not trustworthy, and increasing randomized train coverage improves regret, F1,
exact match, seed variance, and the train-validation gap. An edge-only inverse
shortest-path model is a credible strong baseline. However, the 5%--10%--full
curve is flattening, especially for F1/exact, so data volume is not the whole
story.

The next defensible improvements are:

1. add multiple fixed validation seeds or time/OD-blocked cross-validation;
2. extend eta beyond `1e-4` and/or epochs beyond 20 only in a new declared
   validation protocol, because the selected eta and epoch are both at the
   search boundary;
3. use count-aware hierarchical shrinkage for sparse edges instead of one
   global lambda, while preserving convexity and baseline anchoring;
4. inspect the dropped 20.7% cyclic observations as a separate noisy-route
   model rather than silently forcing them into a positive-cost shortest-path
   objective;
5. test turn costs or route-choice heterogeneity specifically on the long-route
   residuals, and require improvement over this larger edge-only baseline;
6. after freezing every choice, perform exactly one model evaluation on an
   untouched held-out split.

Exact per-run data, aggregate tables, coverage, CCH timings, and diagnostics are
under `experiments/scale_study/`. Full per-route diagnostic files were written
to `/tmp/edge-weight-scale-study/` to avoid committing large derived artifacts.

---

# Earlier short experimental exploration (2026-07-15)

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

The randomized multi-seed validation sweep, scale curve, and controlled CCH
measurements proposed by these pilots are now reported in the study above.
