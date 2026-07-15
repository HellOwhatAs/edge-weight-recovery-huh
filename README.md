# Edge-weight recovery from observed routes

This repository studies inverse shortest paths on road networks. The active
scientific abstraction has two model classes:

1. an edge-only baseline; and
2. an expanded road model with a nonnegative cost for every legal directed-edge
   transition.

The expanded model strictly contains the edge-only model. There are no active
fixed-block or staged model arms: expanded training always optimizes the full
parameter pair `(q,r)`.

## Models and objectives

For a directed graph `G=(V,E)`, edge `e` has positive baseline cost `b_e` and
learned multiplier `q_e`. The edge-only metric is

```text
w_e = b_e q_e.
```

In the implementation, `b_e` is the fixed `metric_baseline` obtained as
`round(original_baseline_e * quantization_scale)`. The optimizer never replaces
this coordinate scale with the changing quantized weight `round(b_e q_e)`.

For each observed complete edge sequence `P_i` from `s_i` to `t_i`, the data
term is true shortest-path regret:

```text
regret_i = cost(P_i) - distance(s_i,t_i).

J_edge(q) = mean_i regret_i(b .* q)
          + lambda_edge / (2|E|) ||q - 1||^2.
```

The expanded topology uses original directed edges as states and legal
adjacent edge pairs `(e,f)` as transitions. Its continuous transition cost and
objective are

```text
kappa_(e,f) = b_f q_f + residual_scale r_(e,f),    r_(e,f) >= 0

J_expanded(q,r) = mean_i regret_i(q,r)
                + lambda_edge / (2|E|) ||q - 1||^2
                + lambda_transition / (2|T|) ||r||^2.
```

Setting every transition residual to zero reproduces the edge-only metric and
objective exactly. Thus every feasible edge-only state `(q,0)` is also a
feasible expanded state; adding transitions does not create a separate arm or
training mechanism.

`count_residual_l1` remains a tie-dependent diagnostic only. It is not the
optimization objective.

## Unified expanded optimizer

Expanded training uses one projected-subgradient optimizer, one learning-rate
parameter, and one update clock. One optimizer update performs exactly one
expanded training-set shortest-path batch query. Both edge and transition
count subgradients come from that same pre-update metric, after which `q` and
`r` are updated together and both integer metrics are rebuilt. Scheduled
validation queries are selection-only and never supply an optimizer gradient.

The two parameter blocks have different raw cost scales. To give one `eta0` a
consistent meaning, optimization is expressed in additive cost coordinates:

```text
u_e = b_e (q_e - 1)
v_t = residual_scale r_t.
```

Let

```text
delta_edge_e       = (observed_edge_e - predicted_edge_e) / N
delta_transition_t = (observed_transition_t - predicted_transition_t) / N
eta_k              = eta0 / sqrt(completed_updates + 1).
```

One update is

```text
u_e' = project_[b_e(q_min-1), b_e(q_max-1)](
         u_e - eta_k [delta_edge_e
                      + lambda_edge u_e / (|E| b_e^2)])

v_t' = project_[0, residual_scale r_max](
         v_t - eta_k [delta_transition_t
                      + lambda_transition v_t
                        / (|T| residual_scale^2)]).
```

Equivalently, this is deterministic diagonal preconditioning of the original
`q/r` subgradients:

```text
g_q,e = b_e delta_edge_e
        + lambda_edge (q_e - 1) / |E|
g_r,t = residual_scale delta_transition_t
        + lambda_transition r_t / |T|

q_e' = project_[q_min,q_max](q_e - eta_k g_q,e / b_e^2)
r_t' = project_[0,r_max](r_t - eta_k g_r,t / residual_scale^2).
```

This changes the optimization geometry, not the continuous objective,
regularizers, or feasible set. Equal count imbalance produces equal additive
cost movement in `u` and `v` before regularization and projection. It does not
introduce a second tunable learning rate or a second clock.

The expanded-specific active configuration surface is correspondingly small:

```text
model.kind = expanded
eta0
lambda_edge
lambda_transition
q_min
q_max
r_max
quantization_scale
residual_scale
updates
validation_every
```

Data identity, oracle, runtime, and validation-selection metadata remain
ordinary run metadata. There is no arm, frozen-block flag, staged-protocol
field, or block-specific learning rate.

Expanded checkpoint selection is fixed to validation mean regret plus the two
current regularization terms. Model-relative regret remains a diagnostic and
does not drive the active expanded selection state.

Continuous latent parameters remain separate from positive integer CCH
weights. Expanded runs start deterministically from `q=1, r=0`. Checkpoints
bind `q`, `r`, both quantized metrics, configuration and data identities, that
initialization identity, expanded-topology identity, validation-selection
state, and one `completed_updates`. Restore is strict: latent state must
reproduce the saved integer metrics exactly, without implicit clamping or
repair.

## Established

The following claims are supported by implementation contracts and existing
evidence:

- the edge-only inverse shortest-path baseline and its normalized L2 anchor;
- full CCH customization, unique-OD batching, and correct OD multiplicity;
- complete-path validation and the single active policy of dropping cyclic
  observations;
- validation-only checkpoint selection and no test-data read during training;
- stable expanded transition IDs and reversible transition-to-edge mapping;
- correct expanded source/target handling and observed-path cost accounting;
- exact equality of edge-only and expanded distances and observed costs when
  `r=0`, allowing different path representations at shortest-path ties;
- strict nesting of every edge-only state `(q,0)` inside the expanded model;
- strictly greater representational capacity on a synthetic conflict graph,
  where edge-only costs cannot make two conditionally opposed routes both
  uniquely shortest but nonnegative transition costs can.

The frozen Beijing edge-only baseline used 623,275 accepted training routes
and selected epoch 99 within a bounded 100-epoch run. Its established
development evidence is:

| Scope | Routes | Relative regret | Mean regret | Edge F1 | Exact match |
|---|---:|---:|---:|---:|---:|
| Time-blocked development | 129,033 | 0.06348409 | 339,523.40 | 0.681488 | 0.371068 |
| Spent AM/PM confirmation | 31,662 | 0.06302821 | — | 0.684512 | 0.376508 |

The AM/PM blocks were validation-derived, source-index-disjoint confirmation
blocks. They are spent evidence, not an untouched final test. The selected
epoch was the budget boundary, so convergence is not established. The
authoritative records are
[`experiments/configs/beijing_edge_only_full.json`](experiments/configs/beijing_edge_only_full.json)
and
[`experiments/summaries/beijing_edge_only.json`](experiments/summaries/beijing_edge_only.json).

## Not yet established

The repository does not yet establish:

- that a fairly and sufficiently optimized expanded road model outperforms the
  edge-only baseline on real data;
- improvement on independent data in raw mean objective, Edge F1, or Exact
  Match;
- that either active model has been optimized sufficiently close to its global
  optimum;
- that a learned transition residual has a physical, behavioral, or causal
  interpretation;
- generalization to another city, time period, or context.

`relative_regret` divides regret by observed-path cost under the current
model. The frozen edge-only baseline still uses it for checkpoint selection to
preserve that established run, while expanded training logs it only as a
diagnostic. Because the denominator changes with the metric, it must not be the
sole cross-model ranking criterion. A future fair comparison must evaluate only

```text
edge-only baseline
vs.
fully optimized expanded road model
```

and should report raw mean regret or a denominator fixed identically across
both models, together with Edge F1 and Exact Match on independent data. This is
a future validation requirement, not authorization to read test data or start
a new large experiment.

## Historical archive

The former Beijing A/B/C study is preserved under
[`experiments/archive/turn_residual_abc_v1/`](experiments/archive/turn_residual_abc_v1/README.md).
Its configurations, summaries, protocol decisions, and numerical fields are
historical audit material only. The study's model-ranking conclusion was
withdrawn because its finite optimization budgets, model-relative selection
ratio, subset asymmetry, and lack of an independent endpoint did not support
the claimed ranking.

Those historical arms are not active model categories and are not future
research questions. Strings such as `winner`, `promoted`, or `continue` inside
the archived JSON describe execution-time protocol decisions, not current
scientific conclusions. The immutable pre-audit recovery point is
`6b66eae329b0beea3546550292a4efd789276159`.

## Repository guide

- `src/model/edge_only.rs`: edge-only parameterization.
- `src/model/expanded_road.rs`: `ExpandedRoadModel` and continuous transition
  residual state.
- `src/objective.rs` and `src/optimizer.rs`: regret, diagnostics, and unified
  projected updates.
- `src/turn_graph.rs` and `src/oracle/expanded.rs`: expanded topology and
  metric-bound expanded queries.
- `src/training.rs` and `src/expanded_training.rs`: validation-selected
  training paths.
- `tests/`: behavioral, correctness, identity, nesting, and checkpoint
  contracts.
- `experiments/configs/`: active baseline and bounded configurations.
- `experiments/archive/turn_residual_abc_v1/`: immutable historical A/B/C
  audit material.

The CLIs can be inspected without reading data:

```bash
cargo run --locked --bin train -- --help
cargo run --locked --bin evaluate -- --help
```

See [`docs/research_status.md`](docs/research_status.md) for the precise claim
boundary and [`EXPERIMENTS.md`](EXPERIMENTS.md) for the evidence index.
