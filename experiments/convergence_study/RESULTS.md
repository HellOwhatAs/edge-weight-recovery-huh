# Beijing edge-only convergence and error-attribution study

Date: 2026-07-15

Status: the preregistered convergence study, one-shot AM/PM confirmation,
first-divergence diagnostics, loop-policy audit/ablation, and fixed-edge global
left-turn probe are complete. The selected edge-only model is a stronger
baseline, but its best development checkpoint is still at epoch 99 of 100. The
turn probe selected zero penalty and did not pass its scientific gate.

## Executive result

The previous 20-epoch budget was too short. On 129,033 valid time-blocked
development routes, the exact same-eta (`1e-4`) deterministic trajectory
improved from relative regret `0.06826350`, F1 `0.656391`, and exact match
`0.346694` at epoch 19 to `0.06357497`, `0.677356`, and `0.367921` at epoch 99.
The ultimately selected checkpoint used `eta0=3e-4`, `lambda=1e5`, multiplier
box `[0.1, 10]`, and epoch 99; it reached `0.06348409`, `0.681488`, and
`0.371068`.

The final selected candidate's advantage over the old frozen control survived
both one-shot temporal confirmation blocks. Across 31,662 pooled valid routes,
the selected model attained relative regret
`0.06302821`, F1 `0.684512`, and exact match `0.376508`. Relative to the frozen
20-epoch control, paired improvement was:

- relative regret: `0.00442438` absolute, 95% interval
  `[0.00405114, 0.00483054]` (about 6.56% of the 20-epoch value);
- mean edge F1: `0.023606`, interval `[0.021152, 0.025967]`;
- exact match: `0.021793`, interval `[0.017971, 0.025551]`.

For the 20-to-100 comparison, each metric improved in all 2,000 paired
bootstrap replicates. This confirmation compares candidates that differ in
both eta and horizon, not a same-eta causal epoch effect. The development
optimum remains at the maximum epoch, however, so 100 epochs is a
stronger bounded endpoint rather than proof of convergence.

The remaining error is structured. It rises sharply with route length and
junction complexity, 82.26% of divergent routes rejoin before the target, and
the median suffix regret after first divergence is only 4.06% of observed
suffix cost. A preregistered fixed-edge global left-turn penalty nevertheless
selected `r=0`: every positive penalty increased tune regret. This rules out the
tested *uniform nonnegative left-turn penalty* as the next improvement; it does
not rule out richer, conditional, or jointly learned turn models.

## Data firewall and estimands

This study used only full train and validation-derived blocks. Formal training,
diagnostics, confirmation, and the capacity probe did not load or evaluate
test; no test contents or metrics were used. This statement applies to this
follow-up; historical pilot/audit caveats remain documented in the repository's
top-level `EXPERIMENTS.md`.

The validation source was partitioned by source index and local-time windows:

| Role | Local time (Asia/Shanghai) | Raw | Valid after `drop` | Sampling |
|---|---|---:|---:|---|
| Development | 2009-05-13 00:00--24:00 | 163,614 | 129,033 | all eligible, excluding the prior fixed validation indices |
| Confirmation A | 2009-05-16 00:00--12:00 | 20,000 | 15,776 | uniform without replacement, seed 20260716 |
| Confirmation B | 2009-05-16 12:00--24:00 | 20,000 | 15,886 | uniform without replacement, seed 20260717 |

Development, A, B, and the previous 20,000-record validation sample have
disjoint source indices. Exact indices, RNG identity, timestamps, and source and
output SHA-256 values are in `validation_blocks.json`.

The development set supports checkpoint, eta, diagnostic, loop-ablation, and
capacity-probe decisions. A/B were evaluated only after the three models were
frozen. They are now spent and must not be reused to select or confirm any loop
or turn model. They provide one-shot temporal confirmation within the
validation source, not a final test estimate.

## What changed in the implementation

The experiment required several behavior changes with direct methodological
roles:

- `--eval-path-metrics` measures F1/exact/Jaccard at every scheduled validation
  event, so regret convergence can be compared with route recovery rather than
  inferred from the count residual.
- `--early-stop-min-delta` always saves a genuinely better checkpoint but resets
  patience only for an improvement larger than the declared delta. This avoids
  losing the best observed state while preventing negligible numerical changes
  from extending a run indefinitely.
- `scripts/run_experiment_matrix.py` supports scheduled evaluation, cycle
  policy, content-addressed completed checkpoints, and a per-run timeout capped
  at 900 seconds. Cache reuse requires the exact matrix row, command, runner
  concurrency/timeout context, binary SHA-256, graph/train/validation input
  SHA-256 set, and unchanged checkpoint/log SHA-256 outputs; the test split is
  not opened or fingerprinted. Full-data jobs were serial to bound memory.
- `examples/generate_validation_blocks.rs` constructs manifest-backed disjoint
  time blocks rather than using ordered prefixes as statistical samples.
- `scripts/analyze_convergence.py` rebuilds trajectories without opening data,
  applies the frozen decision rule, and checks that the standalone 20-epoch
  control is an exact deterministic prefix of its 100-epoch counterpart.
- `examples/analyze_divergence.rs` attributes the first route disagreement by
  prefix, rejoin, route length, train-edge visibility, junction degree, and
  planar turn class. These are descriptive associations, not causal estimates.
- training now supports `--train-cycle-policy drop|erase|keep` while validation
  and test remain fixed at `drop`. Chronological loop erasure is deterministic
  and preserves the original edge order outside removed loops.
- the turn probe builds a line graph with one state per directed edge and costs
  transitions as the next edge cost plus a fixed left-turn penalty. Its `r=0`
  equivalence and path-continuity checks are explicit correctness gates.

## Bounded convergence study

The frozen grid used train variants `scale_10pct_seed42` and `all`, eta values
`{1e-4, 3e-4, 1e-3}`, `lambda=1e5`, up to 100 epochs, validation every five
epochs, four stale validation events, absolute minimum delta `1e-5`, and full
CCH customization. Every run had a 900-second hard timeout. Ten-percent runs
could run concurrently; full-data runs were serial.

| Train | eta0 | Executed | Best epoch | Dev relative regret | F1 | Exact | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| Full baseline | — | 1 | 0 | 0.09352278 | 0.591144 | 0.337542 | `q=1` control |
| Full control | 1e-4 | 20 | 19 | 0.06826350 | 0.656391 | 0.346694 | prior budget |
| 10% | 1e-4 | 100 | 99 | 0.06430115 | 0.676683 | 0.369115 | best 10% regret |
| 10% | 3e-4 | 100 | 89 | 0.06440743 | 0.675646 | 0.362303 | close alternative |
| 10% | 1e-3 | 20 | 0 | 0.09352278 | 0.591144 | 0.337542 | early stop; no useful update |
| Full | 1e-4 | 100 | 99 | 0.06357497 | 0.677356 | 0.367921 | viable endpoint |
| Full | 3e-4 | 100 | 99 | **0.06348409** | **0.681488** | **0.371068** | selected |
| Full | 1e-3 | 20 | 0 | 0.09352278 | 0.591144 | 0.337542 | early stop; no useful update |

The separate full `eta0=1e-4`, 20-epoch control exactly matches epochs 0--19
of the 100-epoch log for every scientific state token. Timing is excluded, as is
the long run's post-evaluation update at the terminal control epoch. Thus the
horizon comparison is not explained by a changed trajectory or runner.

At horizon 20, `eta0=1e-4` was best. By horizon 100, `eta0=3e-4` narrowly led in
regret and more clearly led in F1/exact. Only seven edges were at the lower box
bound and none at the upper bound in the selected checkpoint, well below the
predeclared 1% failure threshold. The selected run completed in about 272
seconds; no formal run approached its timeout.

Interpretation: the exact-prefix, same-eta comparison shows that extending the
optimization budget is a real improvement and the old 20-epoch baseline
understated edge-only capacity. Since both useful
full trajectories still select epoch 99, a future extension beyond 100 epochs
requires a new protocol; it must not be justified retrospectively with A/B.

## One-shot A/B confirmation

The three frozen checkpoints were `q=1`, full `eta0=1e-4` at 20 epochs, and the
development-selected full `eta0=3e-4` at 100 epochs. Their hashes were validated
before analysis, and `(route_index, source, target)` aligned exactly across all
three models within each block.

| Scope | Valid routes | Model | Relative regret | Mean edge F1 | Exact |
|---|---:|---|---:|---:|---:|
| AM | 15,776 | `q=1` | 0.09210652 | 0.599664 | 0.353892 |
| AM | 15,776 | 20 epochs | 0.06724488 | 0.662546 | 0.356237 |
| AM | 15,776 | 100 epochs | **0.06307110** | **0.686270** | **0.376331** |
| PM | 15,886 | `q=1` | 0.09285387 | 0.593518 | 0.343573 |
| PM | 15,886 | 20 epochs | 0.06767404 | 0.659278 | 0.353204 |
| PM | 15,886 | 100 epochs | **0.06298240** | **0.682767** | **0.376684** |
| Pooled | 31,662 | `q=1` | 0.09246791 | 0.596580 | 0.348715 |
| Pooled | 31,662 | 20 epochs | 0.06745259 | 0.660906 | 0.354715 |
| Pooled | 31,662 | 100 epochs | **0.06302821** | **0.684512** | **0.376508** |

Pooled paired-bootstrap effects are shown as improvement from the row model to
the 100-epoch model:

| Comparison | Relative-regret improvement (95% CI) | F1 improvement (95% CI) | Exact improvement (95% CI) |
|---|---:|---:|---:|
| `q=1` to 100 | 0.02943970 [0.02854511, 0.03037834] | 0.087932 [0.084253, 0.091456] | 0.027794 [0.023024, 0.032248] |
| 20 to 100 | 0.00442438 [0.00405114, 0.00483054] | 0.023606 [0.021152, 0.025967] | 0.021793 [0.017971, 0.025551] |

The 2,000-replicate bootstrap is paired within a block and stratified across AM
and PM, preserving both accepted block sizes. It quantifies route-sampling
uncertainty for these fixed checkpoints; it does not include training-data,
hyperparameter-search, city, or temporal-regime uncertainty.

## First-divergence and complexity diagnostics

Optimization improves more than aggregate regret:

| Model | Divergence rate | First-edge accuracy | Prefix ratio | Edge F1 | Exact |
|---|---:|---:|---:|---:|---:|
| `q=1` | 0.662458 | 0.782451 | 0.446082 | 0.591144 | 0.337542 |
| 20 epochs | 0.653306 | 0.805701 | 0.473814 | 0.656391 | 0.346694 |
| 100 epochs | **0.628932** | **0.819380** | **0.501290** | **0.681488** | **0.371068** |

For the selected model, 81,153 of 129,033 development routes diverge. Median
first divergence is edge index 3, or 9.86% through the observed route. Among
divergent routes:

- 69.32% first diverge at a complex junction (directed indegree and outdegree
  both at least two);
- 82.26% rejoin the observed route before the common target;
- median relative observed-minus-predicted suffix cost is 0.04056;
- when both turn classes are available, only 33.63% match; observed/predicted
  first-divergence counts are 8,531/9,855 left, 12,369/11,382 right, and
  36,834/36,501 straight.

Difficulty rises strongly by route-length quartile:

| Observed-length quartile | Routes | Divergence | Mean F1 |
|---|---:|---:|---:|
| Q1 | 33,648 | 0.2896 | 0.8409 |
| Q2 | 31,520 | 0.5734 | 0.7208 |
| Q3 | 32,544 | 0.7613 | 0.6342 |
| Q4 | 31,321 | 0.9118 | 0.5198 |

Complexity is not only a proxy for length. Within Q1 the 0 to 1--2 bins are a
small nonmonotone exception (`0.144` to `0.133`), after which divergence rises
to `0.195` for 3--5 and `0.377` for at least six complex junctions. Q2 is
monotone at `0.242/0.390/0.434/0.590`. Higher-quartile low-complexity cells are
small, so their individual rates should not be overinterpreted. The at-least-six
group reaches 0.769 in Q3 and 0.913 in Q4.

Only 621 routes (0.481%) contain an edge unseen in full train. Their divergence
is 0.7746 and F1 is 0.6072 versus 0.6282 and 0.6818 on fully seen routes. This
coverage effect is real but too rare to explain the aggregate residual. The
combination of early divergence, frequent rejoin, small suffix gaps, and
length-controlled junction effects suggests many close alternatives and a
remaining representation/choice-noise problem.

## Cyclic observations and chronological loop erasure

The full-train audit is read-only and performs no model training. Among 785,709
structurally valid original records, 162,434 (20.67%) repeat a node.

| Policy | Source records retained | Output paths | Removed source-edge occurrences | Graph-edge coverage |
|---|---:|---:|---:|---:|
| Drop cyclic originals | 623,275 (79.33%) | 623,275 | 27.39% | 77.49% |
| Chronological loop erasure | 785,263 (99.94%) | 785,263 | 4.25% | 81.03% |
| Greedy split at repeats | 785,709 (100%) | 1,007,666 | 0.0019% | 85.02% |

Dropped cyclic routes average 47.68 edges and 1,272 seconds versus 32.94 edges
and 812 seconds for acyclic routes. Dropping loses 5,514 graph edges and
147,651 original OD pairs found only in cyclic records. These are descriptive
population differences, not causal effects, but they establish selection bias.
Greedy splitting was audited but not trained because flattening a variable
number of pieces into equally weighted observations changes each source
record's influence.

The matched fixed-hyperparameter erase ablation used the same development task:

| Train scale/configuration | Drop regret/F1/exact | Erase regret/F1/exact | Interpretation |
|---|---|---|---|
| 10%, `eta0=1e-4`, 100 epochs | 0.06430115 / 0.676683 / 0.369115 | 0.06412951 / 0.676313 / 0.369045 | tiny regret gain; overlap flat/slightly lower |
| Full, `eta0=3e-4`, 100 epochs | 0.06348409 / 0.681488 / 0.371068 | 0.06486248 / 0.681446 / 0.368813 | regret worsens 0.001378; F1 flat |

Chronological erasure improves retention, not robust predictive performance at
the frozen optimizer settings. It changes route counts and gradient scale, so
this single comparison does not prove erasure is intrinsically worse. It does
show that it should not replace `drop` by default without a new, predeclared
matched-retuning study and new confirmation data.

## Fixed-edge global left-turn probe

The probe freezes the selected edge weights. Its line graph has 72,156 states,
188,071 transitions, and 39,436 classified left-turn transitions. Transition
cost is `c_f + round(kappa * r)` for a left turn and `c_f` otherwise, with
`kappa=127625`, the median selected edge weight. The tune/audit partitions are
deterministic disjoint 20,000/10,000-route samples from development.

All correctness gates passed: expanded paths were continuous and connected the
OD; every transition cost was positive and finite; and all 30,000 `r=0`
expanded distances and observed costs exactly matched the original edge-only
oracle, with zero mismatches and zero maximum difference.

| r | Left penalty | Tune relative regret | F1 | Exact |
|---:|---:|---:|---:|---:|
| **0.00** | **0** | **0.06341452** | 0.682378 | **0.369850** |
| 0.05 | 6,381 | 0.06378618 | 0.682863 | 0.369350 |
| 0.10 | 12,763 | 0.06430391 | **0.682914** | 0.369200 |
| 0.20 | 25,525 | 0.06559292 | 0.681636 | 0.357600 |
| 0.40 | 51,050 | 0.06892272 | 0.676582 | 0.345000 |

The frozen primary rule selected `r=0`. Audit therefore compared `r=0` with
itself: regret/F1/exact changes and their bootstrap intervals are all zero. The
scientific gate required at least 0.0005 regret improvement and 0.005 F1
improvement with no more than 0.002 exact decline, so its formal verdict is
false. No confirmation C is warranted. The full run took 5.10 seconds and
peaked at 315.2 MiB. Estimated raw expanded arrays used 3.31 MiB, below the
512-MiB correctness limit, and the run was well below its 180-second timeout.

This negative result is narrow but useful. A single city-wide nonnegative
left-turn penalty cannot explain the first-divergence signal under fixed edge
weights. It does not test node-specific turn costs, right/straight effects,
road-class interactions, time-of-day effects, joint edge/turn optimization, or
probabilistic route choice.

## Reproduction with bounded commands

Build and test once:

```bash
cargo build --release --locked --all-targets
cargo test --locked --all-targets
```

In a clean reproduction checkout, regenerate the three blocks with a bounded
command (the generator deliberately refuses to overwrite existing outputs):

```bash
timeout 180s cargo run --release --locked --example generate_validation_blocks -- \
  --city beijing \
  --source-variant all \
  --exclude-manifest experiments/scale_study/subsets/validation_scale_fixed_seed20260715.json \
  --development-variant time_dev_20090513_excl_previous \
  --development-label 2009-05-13_full_day_Asia-Shanghai \
  --development-start 1242144000 \
  --development-end-exclusive 1242230400 \
  --confirmation-a-variant time_confirm_am_20090516_seed20260716 \
  --confirmation-a-label 2009-05-16_00-12_Asia-Shanghai \
  --confirmation-a-start 1242403200 \
  --confirmation-a-end-exclusive 1242446400 \
  --confirmation-a-count 20000 \
  --confirmation-a-seed 20260716 \
  --confirmation-b-variant time_confirm_pm_20090516_seed20260717 \
  --confirmation-b-label 2009-05-16_12-24_Asia-Shanghai \
  --confirmation-b-start 1242446400 \
  --confirmation-b-end-exclusive 1242489600 \
  --confirmation-b-count 20000 \
  --confirmation-b-seed 20260717 \
  --manifest /tmp/edge-weight-convergence-repro/validation_blocks.json
```

Run the development matrices. The runner enforces at most 900 seconds per row;
full jobs remain serial. Use a new run root so the recorded study is not
overwritten:

```bash
python3 scripts/run_experiment_matrix.py \
  --matrix experiments/convergence_study/matrix_10pct.csv \
  --run-root /tmp/edge-weight-convergence-repro/runs \
  --validation-variant time_dev_20090513_excl_previous \
  --summary-csv /tmp/edge-weight-convergence-repro/results_10pct.csv \
  --summary-json /tmp/edge-weight-convergence-repro/results_10pct.json \
  --jobs 3 --rayon-threads 4 --timeout-seconds 900

python3 scripts/run_experiment_matrix.py \
  --matrix experiments/convergence_study/matrix_full.csv \
  --run-root /tmp/edge-weight-convergence-repro/runs \
  --validation-variant time_dev_20090513_excl_previous \
  --summary-csv /tmp/edge-weight-convergence-repro/results_full.csv \
  --summary-json /tmp/edge-weight-convergence-repro/results_full.json \
  --jobs 1 --rayon-threads 8 --timeout-seconds 900

python3 scripts/analyze_convergence.py \
  --run-root /tmp/edge-weight-convergence-repro/runs \
  --protocol experiments/convergence_study/protocol.json \
  --trajectory-csv /tmp/edge-weight-convergence-repro/trajectories.csv \
  --summary-csv /tmp/edge-weight-convergence-repro/convergence_summary.csv \
  --summary-json /tmp/edge-weight-convergence-repro/convergence_summary.json
```

Reproduce the fixed-hyperparameter loop-erasure comparisons separately; these
remain development-only and use the same per-row timeout:

```bash
python3 scripts/run_experiment_matrix.py \
  --matrix experiments/convergence_study/matrix_loop_erase_10pct.csv \
  --run-root /tmp/edge-weight-convergence-repro/runs \
  --validation-variant time_dev_20090513_excl_previous \
  --summary-csv /tmp/edge-weight-convergence-repro/results_loop_erase_10pct.csv \
  --summary-json /tmp/edge-weight-convergence-repro/results_loop_erase_10pct.json \
  --jobs 1 --rayon-threads 8 --timeout-seconds 900

python3 scripts/run_experiment_matrix.py \
  --matrix experiments/convergence_study/matrix_loop_erase_full.csv \
  --run-root /tmp/edge-weight-convergence-repro/runs \
  --validation-variant time_dev_20090513_excl_previous \
  --summary-csv /tmp/edge-weight-convergence-repro/results_loop_erase_full.csv \
  --summary-json /tmp/edge-weight-convergence-repro/results_loop_erase_full.json \
  --jobs 1 --rayon-threads 8 --timeout-seconds 900
```

The audit, divergence analysis, and turn probe are independently bounded:

```bash
timeout 180s target/release/examples/audit_loop_policies \
  --city beijing --train-variant all \
  --output /tmp/edge-weight-convergence-repro/loop_policy_audit.json

timeout 180s target/release/examples/analyze_divergence \
  --city beijing \
  --train-variant all \
  --validation-variant time_dev_20090513_excl_previous \
  --checkpoint /tmp/edge-weight-convergence-repro/runs/convergence_full/conv_full_eta3e4/model_checkpoint.json \
  --summary-output /tmp/edge-weight-convergence-repro/divergence_summary.json

tar -xzf experiments/convergence_study/evidence/reproducibility_bundle.tar.gz -C /tmp

timeout 180s target/release/examples/probe_turn_penalty \
  --protocol experiments/convergence_study/turn_probe_protocol.json \
  --output /tmp/edge-weight-convergence-repro/turn_probe_results.json
```

The frozen turn protocol contains the original study checkpoint path and hash.
The evidence extraction immediately above restores that exact layout and
byte-identical checkpoint before the probe runs. The program rejects a hash
mismatch.

Do not rerun A/B to make further decisions. The 6,790,676-byte evidence archive
(SHA-256
`e8ba23b62e356afd3fdc8d2b5412e65cb62ba52e03a41b1322faa39e76f665d0`)
contains the six *existing* route-level confirmation exports, three
hash-matched checkpoints, and ten training logs. It contains no raw trip or
test data, and confirmation was not rerun to create it. Restore the original
absolute-path layout and recheck the paired analysis without opening a pickle:

```bash
sha256sum experiments/convergence_study/evidence/reproducibility_bundle.tar.gz
tar -xzf experiments/convergence_study/evidence/reproducibility_bundle.tar.gz -C /tmp

python3 scripts/analyze_confirmation.py \
  --plan experiments/convergence_study/confirmation_plan.json \
  --summary-dir experiments/convergence_study/confirmation \
  --routes-dir /tmp/edge-weight-convergence-study/confirmation \
  --output-json /tmp/confirmation_summary_rechecked.json \
  --output-csv /tmp/confirmation_summary_rechecked.csv
```

## Artifact map

| Artifact | Purpose |
|---|---|
| `protocol.json` | frozen convergence question, grid, decision rules, and data firewall |
| `validation_blocks.json` | source indices, disjointness evidence, seeds, timestamps, and hashes |
| `matrix_10pct.csv`, `matrix_full.csv` | bounded convergence runs and controls |
| `convergence_summary.json`, `trajectories.csv` | selected candidate, horizon rankings, and exact-prefix check |
| `confirmation_plan.json` | frozen models, checkpoint hashes, one-shot rule, and bootstrap seed |
| `confirmation/`, `confirmation_summary.json` | six compact evaluations and paired AM/PM analysis |
| `divergence_*.json` | baseline/20/100 first-divergence diagnostics |
| `loop_policy_audit.json` | full-train drop/erase/split retention and population audit |
| `loop_erasure_protocol.json`, `results_loop_erase_*.json` | frozen fixed-hyperparameter erase comparisons |
| `turn_probe_protocol.json`, `turn_probe_results.json` | fixed-edge capacity hypothesis, gates, correctness, and negative result |
| `evidence/manifest.json`, `evidence/reproducibility_bundle.tar.gz` | hashes and minimum route/checkpoint/log evidence for offline reanalysis |

## Limitations and next defensible steps

- Epoch 99 remains best, so optimization and representation error are not yet
  cleanly separated. A longer schedule needs a newly frozen development-only
  protocol.
- Confirmation A/B share one validation source and one city. Paired intervals
  are route-level rather than vehicle/time-clustered and are not multi-city or
  multi-period uncertainty estimates.
- The evidence bundle omits `runner_result.json`. Reanalyzing convergence from
  the archived logs can therefore report only a log-derived lower bound for
  wall time, rather than the runner's end-to-end duration. Scientific states,
  route metrics, candidate selection, and the exact-prefix check are unaffected.
- The loop-erasure comparison holds optimizer hyperparameters fixed even though
  it changes sample count and gradient composition. It is a policy ablation,
  not a completed retuning study.
- First-divergence associations can be caused by route length, choice noise,
  map ambiguity, temporal heterogeneity, or omitted costs. They motivate
  hypotheses but do not identify causality.
- The turn tune/audit source indices are disjoint, but the single-left-feature
  hypothesis was proposed after inspecting divergence summaries over all of
  development. The audit isolates penalty-grid selection conditional on that
  feature; it is not untouched confirmation of the feature hypothesis.
- The global left penalty failed. If turn modeling continues, the smallest
  defensible next model is a regularized conditional feature (for example by
  junction/road class) evaluated on development only. Any positive result needs
  a new source-index-disjoint confirmation C; A/B and test remain unavailable.
