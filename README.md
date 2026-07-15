# Edge-weight recovery from observed routes

This repository learns one globally shared road-cost metric from historical
routes. Its production method is a regularized edge-only inverse shortest-path
model trained with projected subgradient descent and a batched Customizable
Contraction Hierarchy (CCH) oracle.

The repository is intentionally narrow: it contains one training path, one
standard evaluation path, and a generic edge-state graph for the next planned
turn-aware model. Historical exploratory branches are recoverable from the
pre-cleanup archive rather than exposed through the main CLI.

## Problem

Let `G=(V,E)` be a directed road network. Each edge `e` has a positive baseline
cost `b_e`, and each observation is a complete, continuous sequence of original
edge IDs from an origin `s_i` to a destination `t_i`.

The goal is to learn shared costs under which the observed routes are close to
shortest paths. The method does not learn a separate metric for each route,
traveller, or time period.

## Mathematical objective

For learned edge costs `w`, the regret of observation `i` is

```text
regret_i(w)
  = cost_w(observed_path_i) - shortest_distance_w(s_i, t_i).
```

The edge-only model learns dimensionless multipliers `q` around the baseline:

```text
w_e = b_e q_e

J(q)
  = (1/N) sum_i regret_i(b .* q)
  + lambda_edge / (2|E|) ||q - 1||^2.
```

If `h` is the aggregate observed edge count and `h_hat(q)` is the aggregate
count on current shortest paths, their difference supplies a data subgradient.
The optimizer applies

```text
q[t+1] = project_[q_min,q_max](q[t] - eta0/sqrt(t+1) * g[t]).
```

`count_residual_l1 = ||h_hat-h||_1` is reported only as a diagnostic. It is not
the loss: shortest-path ties can give zero regret while deterministic path
reconstruction still produces a nonzero count residual.

## Edge-only method

`EdgeOnlyModel` owns continuous `f64` multipliers. CCH requires positive integer
weights, so quantization is explicit and the continuous latent state remains
separate from the integer oracle metric. No value may reach the CCH infinity
sentinel.

Training uses full CCH metric customization. In each epoch, observations with
the same OD pair share one oracle query; predicted distances and edge counts are
then weighted by the number of observations in that OD group. Checkpoints are
selected only by aggregate validation relative regret:

```text
sum(validation regret) / sum(validation observed-path cost).
```

The selected, already-evaluated state is saved atomically as one
`checkpoint.json` containing `q`, quantized weights, epoch, complete
configuration, selection value, train regret, and data/baseline identities.
The training process never reads test data.

## Planned turn-aware extension

The retained edge-state expansion assigns one state to each original directed
edge and one stable transition ID to every legal adjacent pair `(e,f)`. It
supports multi-source/multi-target OD queries and maps transition IDs back to
their original edge pairs.

The next planned model is

```text
kappa_(e,f) = b_f q_f + scale r_(e,f),    r_(e,f) >= 0,
```

with `q` anchored at one and per-transition residuals `r` anchored at zero.
At `r=0`, expanded distances, observed-path costs, path continuity, and OD
endpoints must match the edge-only model exactly.

The graph and zero-residual model placeholder are implemented. Learning the
per-transition residuals is not implemented in this cleanup. A historical
fixed global left-turn probe is not part of the formal method and is not
exposed by the current training path.

## Architecture

- `src/data.rs` loads the graph and complete paths, validates paths, groups ODs,
  and computes observed edge counts.
- `src/objective.rs` computes shortest-path regret and the count-residual
  diagnostic; the model owns its normalized anchoring term.
- `src/optimizer.rs` implements the projected subgradient update.
- `src/oracle/cch.rs` provides the production batched CCH oracle;
  `src/oracle/dijkstra.rs` is a small exact correctness oracle.
- `src/model/edge_only.rs` contains the production parameterization;
  `src/model/turn_aware.rs` is the zero-residual structural placeholder.
- `src/turn_graph.rs` owns generic transition indexing, metric construction,
  source/target states, and expanded-path decoding.
- `src/training.rs` runs validation-selected training and structured logging.
- `src/evaluation.rs` reports relative and mean regret, exact match, edge
  precision/recall/F1, and edge Jaccard.
- `src/config.rs` validates compact experiment configurations and the atomic
  checkpoint schema.
- `src/bin/train.rs` and `src/bin/evaluate.rs` are the two user-facing binaries.

## Data contract

Route pickle entries must contain the complete sequence of original directed
edge IDs. The loader never removes the first or last real edge.

Every path is checked for an empty sequence, invalid edge IDs, discontinuous
adjacent edges, and repeated nodes. Because a positive-cost cycle cannot belong
to a shortest path, cyclic observations are dropped. This is the sole training
data policy; alternative cycle policies are not CLI options.

The OD pair is the tail of the first edge and the head of the last edge. Train
and validation file identities are declared in the experiment JSON. The two
provided configurations expect the corresponding preprocessed pickle files
under `data/<city>_data/`; generated data remains outside version control.

## Quick start

The bounded smoke configuration uses the existing deterministic Beijing 1%
training subset and fixed validation subset:

```bash
cargo run --release --locked --bin train -- \
  --config experiments/configs/smoke_1pct.json \
  --output-dir /tmp/edge-weight-recovery-smoke
```

The output directory contains the atomic checkpoint and structured training
log. Inspect the intentionally small CLI with:

```bash
cargo run --release --locked --bin train -- --help
```

## Reproducible full baseline

The frozen full-Beijing configuration records the selected solver parameters
and input identities:

```bash
cargo run --release --locked --bin train -- \
  --config experiments/configs/beijing_edge_only_full.json \
  --output-dir /tmp/edge-weight-recovery-beijing-full
```

This is a full 100-epoch reproduction, not the recommended smoke check for a
routine code change. It requires the declared full training and development
pickle files.

## Current validated result

The frozen strong baseline trained on 623,275 valid acyclic Beijing routes with
`eta0=3e-4`, `lambda_edge=1e5`, box `[0.1,10]`, and at most 100 epochs. Aggregate
validation relative regret selected epoch 99.

| Evaluation scope | Routes | Relative regret | Edge F1 | Exact match |
|---|---:|---:|---:|---:|
| Time-blocked development | 129,033 | 0.06348409 | 0.681488 | 0.371068 |
| Pooled one-shot AM/PM confirmation | 31,662 | 0.06302821 | 0.684512 | 0.376508 |

The two confirmation blocks were source-index-disjoint temporal blocks from the
validation source. They are spent confirmation data, not an untouched final
test estimate. Formal training and confirmation did not evaluate test.

For the same `eta0=1e-4` trajectory, development relative regret improved from
`0.06826350` at epoch 19 to `0.06357497` at epoch 99. This establishes that the
old 20-epoch horizon was too short. Because useful runs still selected the
100-epoch boundary, the result does not establish full optimization convergence.

The generic expanded graph has also passed exact zero-residual equivalence
checks against the edge-only oracle. The earlier uniform nonnegative global
left-turn probe selected zero penalty; that narrow negative result does not
test the planned learned per-transition residual model.

## Repository structure

```text
src/
  bin/                 train and evaluate entry points
  model/               edge-only model and turn-aware placeholder
  oracle/              production CCH and exact small-graph Dijkstra
  data.rs              graph/trip contract and OD grouping
  objective.rs         regret and diagnostics
  optimizer.rs         projected subgradient update
  turn_graph.rs        generic edge-state transition graph
  training.rs          reusable training loop
  evaluation.rs        standard route metrics
  config.rs            experiment and checkpoint contracts
experiments/
  configs/             compact reproducible configurations
  summaries/           concise trusted results
docs/
  research_status.md   proved, open, and next-step claims
  repository_cleanup_inventory.md
tools/                 bounded preprocessing or benchmark utilities
```

Large generated checkpoints, logs, route-level outputs, and temporary subsets
are intentionally ignored rather than committed.

## Limitations

- The validated result covers one city and a limited temporal source; it is not
  a multi-city or untouched final-test estimate.
- Epoch 99 is a bounded strong checkpoint, not evidence of convergence.
- Integer quantization can change tie-breaking relative to the continuous
  objective, so continuous convex statements do not transfer without this
  qualification.
- Dropping cyclic observations is mathematically consistent with the
  positive-cost shortest-path model but excludes a meaningful part of the raw
  trajectory population.
- The production model is edge-only. Per-transition residual learning remains
  the next scientific milestone.

## Status and citation

The project is research code supporting a manuscript in preparation. Citation
metadata will be added when the manuscript is released. See
[`docs/research_status.md`](docs/research_status.md) for the current claim
boundary and next milestone.

The complete pre-cleanup convergence-study evidence remains available without
history rewriting at tag `archive/pre-cleanup-convergence-study`, commit
`8aacf2e8020bae13c6fad58f22ccb369f249e029`. For example:

```bash
git show archive/pre-cleanup-convergence-study:experiments/convergence_study/RESULTS.md
```
