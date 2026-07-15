# Edge-weight recovery from observed routes

This repository learns one globally shared road-cost metric from complete
historical routes. The reliable main method is an edge-only inverse
shortest-path model. A generic nonnegative per-transition residual model is
also implemented and correctness-checked, but it has not yet been compared
fairly enough to select a final turn-aware training method.

## Problem and objective

For a directed graph `G=(V,E)`, edge `e` has positive baseline cost `b_e` and
learned multiplier `q_e`:

```text
w_e = b_e q_e
```

For each observed complete edge sequence `P_i` from `s_i` to `t_i`, the route
term is true shortest-path regret:

```text
regret_i(w) = cost_w(P_i) - distance_w(s_i, t_i).

J(q) = mean_i regret_i(b .* q)
     + lambda_edge / (2|E|) ||q - 1||^2.
```

The continuous objective is convex. Its data subgradient is obtained from
observed minus predicted edge counts. The optimizer applies a square-root
schedule and projects each multiplier to a positive box:

```text
q[t+1] = project_[q_min,q_max](q[t] - eta0/sqrt(t+1) * g[t]).
```

`count_residual_l1` is a tie-dependent diagnostic only; it is not the loss.

## Established edge-only baseline

`EdgeOnlyModel` keeps the latent `f64` multipliers separate from positive
integer CCH weights. Training uses full CCH customization. Observations sharing
one OD pair use one query, with distances and predicted counts weighted by OD
multiplicity.

The training path has the following enforced contracts:

- complete original edge-ID sequences, with continuity and endpoint checks;
- a single positive-cost policy that drops cyclic observations;
- validation-only checkpoint selection and no test-data read;
- atomic checkpoints containing `q`, quantized weights, configuration,
  selection state, and data/baseline identity;
- exact reconstruction of integer metric state on checkpoint restore.

The frozen Beijing baseline used 623,275 accepted training routes and selected
epoch 99 within a bounded 100-epoch run. Its established development result is:

| Scope | Routes | Relative regret | Mean regret | Edge F1 | Exact match |
|---|---:|---:|---:|---:|---:|
| Time-blocked development | 129,033 | 0.06348409 | 339,523.40 | 0.681488 | 0.371068 |
| Spent AM/PM confirmation | 31,662 | 0.06302821 | — | 0.684512 | 0.376508 |

The AM/PM blocks were validation-derived, source-index-disjoint confirmation
blocks. They are spent evidence, not an untouched final test. The run selected
its epoch-99 boundary, so full optimization convergence is not established.
The authoritative baseline records remain
[`experiments/configs/beijing_edge_only_full.json`](experiments/configs/beijing_edge_only_full.json)
and
[`experiments/summaries/beijing_edge_only.json`](experiments/summaries/beijing_edge_only.json).

## Generic turn-expanded model

The expanded topology uses each original directed edge as a state and every
legal adjacent edge pair `(e,f)` as a transition. It provides stable transition
IDs, bidirectional ID/pair mapping, and multi-source/multi-target handling for
original-node OD queries.

The retained nested model is:

```text
kappa_(e,f) = b_f q_f + scale r_(e,f),    r_(e,f) >= 0

J(q,r) = mean route regret
       + lambda_edge / (2|E|) ||q - 1||^2
       + lambda_turn / (2|T|) ||r||^2.
```

`TurnAwareModel` retains continuous residuals, projection to `[0,r_max]`, and
regularization toward zero. The training implementation can update `r` with
`q` frozen or update both blocks from the same pre-update predicted counts.
Turn-aware checkpoints bind `q`, `r`, edge and transition weights, independent
update clocks, data identity, and expanded-topology identity.

At `r=0`, expanded shortest distances and observed-path costs equal the
edge-only values; reconstructed paths may differ only at ties. This is covered
on synthetic graphs and by an ignored, validation-only Beijing correctness
audit. The model implementation, transition counts, regret accounting,
quantization, and checkpoint restore are retained.

## Turn-aware evidence boundary

Current status:

> implemented, correctness-checked, promising but not yet fairly validated

The archived Beijing development study observed higher edge F1 and exact match
for one frozen-edge residual run than for its expanded-edge continuation
control (`0.693069` versus `0.682444`, and `0.390234` versus `0.369874`). This
suggests possible route-reproduction benefit. It is not a model-selection or
generalization result. In the same comparison, raw mean regret increased from
`317,952.34` to `327,845.80`.

The old A/B/C conclusion is retired because:

- joint learning received only a fixed 30-step simultaneous-update budget,
  although every frozen-edge turn-only state is also feasible for the joint
  continuous model;
- the selection ratio used each model's own observed-path cost as denominator,
  so residuals could change the scale being compared;
- the 10% screen retained a full-data `q*` for turn-only but let joint updates
  modify that representation using only the 10% subset;
- no full-data joint endpoint was run;
- development data selected checkpoints and supplied the reported metrics.

Therefore the repository does not establish that turn-only is better than
joint, that joint is ineffective, that residuals reduce raw regret, that the
effect generalizes independently, or that learned residuals are physical or
causal turn costs.

## Metric interpretation

Historical `relative_regret` is

```text
sum(model regret) / sum(model observed-path cost).
```

It remains useful for reproducing results within a fixed metric convention,
but it is model-relative: both numerator and denominator use the current
model. It must not be the sole fair ranking metric across edge-only,
turn-only, and joint models. A future comparison should report raw mean regret
or a denominator fixed across all models. Edge F1 and exact match are useful
cost-scale-independent route-reproduction metrics.

## Repository guide

- `src/model/edge_only.rs`: reliable edge-only parameterization.
- `src/model/turn_aware.rs`: nonnegative per-transition residual state.
- `src/objective.rs` and `src/optimizer.rs`: regret, diagnostics, and projected
  updates.
- `src/turn_graph.rs` and `src/oracle/expanded.rs`: generic expansion and bound
  expanded queries.
- `src/training.rs` and `src/turn_training.rs`: validation-selected training.
- `tests/`: behavioral, correctness, identity, and theoretical contracts.
- `experiments/configs/`: active baseline and bounded smoke configurations.
- `experiments/archive/turn_residual_abc_v1/`: original inconclusive A/B/C
  protocol, configurations, and machine-readable results.

The CLIs can be inspected without reading data:

```bash
cargo run --locked --bin train -- --help
cargo run --locked --bin evaluate -- --help
```

## Historical recovery

The A/B/C files are preserved byte-for-byte under
[`experiments/archive/turn_residual_abc_v1/`](experiments/archive/turn_residual_abc_v1/README.md).
The immutable pre-audit commit is
`6b66eae329b0beea3546550292a4efd789276159`; this workspace also has the local
annotated tag `pre-turn-abc-audit-20260715` pointing to it. For example:

```bash
git show 6b66eae329b0beea3546550292a4efd789276159:experiments/turn_residual_protocol.json
```

Earlier convergence, scale, and exploratory work remains recoverable as
described in [`experiments/archive/README.md`](experiments/archive/README.md).
See [`docs/research_status.md`](docs/research_status.md) for the exact current
claim boundary and requirements for a future fair evaluation.
