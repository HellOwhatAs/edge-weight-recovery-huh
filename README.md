# Edge-weight recovery

This project learns one nonnegative graph-weight vector from observed road
trajectories. Its architecture separates two concerns:

1. a graph-representation layer chooses `original_edges` or
   `edge_transition_arcs`, maps road trajectories into that topology, and owns
   route decoding; and
2. one inverse-shortest-path trainer learns the coordinates supplied by the
   representation with a common relative-weight optimizer.

The trainer does not inspect whether a coordinate is an original road edge or
a legal road-to-road transition. Optimization is identical in both cases.

## Graph representations

Let an original trajectory be

```text
(e1, e2, ..., eN).
```

### `original_edges`

The original directed road graph is used directly. Each original road edge is
one learned coordinate, and the mapped trajectory is unchanged:

```text
(e1, e2, ..., eN).
```

### `edge_transition_arcs`

This is the directed line graph (also called a line digraph or edge-based
graph) of the original road graph:

- every original directed edge `e` is one routing node;
- every legal consecutive transition `(e,f)`, with
  `head(e) = tail(f)`, is one routing arc `e -> f`; and
- every routing arc has one directly learned coordinate `w[e,f]`.

The original road trajectory is therefore the line-graph node path

```text
e1 -> e2 -> ... -> eN
```

and its learned coordinates are obtained with `windows(2)`:

```text
(e1,e2), (e2,e3), ..., (e{N-1},eN).
```

Its cost has exactly `N - 1` terms:

```text
C_w(P) = sum_i w[e_i,e_{i+1}].
```

Source states are original edges leaving the requested source vertex; target
states are original edges entering the requested target vertex. Both endpoint
offsets are zero. A returned line-graph node path already is the decoded
original-edge sequence.

The representation contains only those original-edge routing nodes and
transition coordinates. All experiments filter trajectories with `N < 2`,
including `original_edges`, so the two representations use the same data
boundary without introducing a special start structure or first-edge cost.

## One inverse-shortest-path optimizer

For either representation, let `m` be its number of learned coordinates, `w0`
the initial direct-weight vector, `q = w / w0`, and `N` the number of
observations. The active optimizer minimizes

```text
J(q) = average[
         observed_path_cost(w0 * q)
         - predicted_shortest_path_cost(w0 * q)
       ]
       + lambda / (2m) * ||q - 1||^2.
```

With coordinate counts from the mapped observed and predicted paths, one
subgradient is

```text
g_q = w0 * (observed_counts - predicted_counts) / N
      + lambda / m * (q - 1).
```

Update `k` uses one global clock and one projection box:

```text
eta_k = eta0 / sqrt(k + 1)
q <- project(q - eta_k * g_q)
w <- w0 * q.
```

The configured lower and upper factors are the projection bounds in `q`. Only
the mapped direct vector `w` is stored and checkpointed. In direct-weight
space, the same update is a generic `diag(w0^2)` preconditioner paired with
relative regularization. There is one learning rate, regularization
coefficient, clock, projection rule, training loop, and checkpoint format for
both representations; no optimizer state depends on whether a coordinate is
an original edge or a transition arc.

The explicit `projected_subgradient` kind retains the earlier direct-weight
Euclidean semantics for reproducibility. Active training uses
`relative_projected_subgradient`; old configurations are never silently
reinterpreted.

The current RoutingKit CCH binding accepts `u32` metrics, so direct `f64`
weights are rounded for route selection and selected paths are evaluated under
the direct vector. Exact continuous-weight CCH routing remains a deliberately
deferred oracle limitation; it does not introduce another learned vector or a
representation-specific optimizer.

## Checkpoints

A checkpoint records the representation and topology identity, configuration
and runtime data identity, current direct weights, and `completed_updates`.
Restoring it rebuilds `w0`, bounds, and the representation-specific oracle from
the verified configuration and topology, restores the configured optimizer
geometry, then resumes the same square-root learning-rate clock. Both
representations use the same checkpoint structure.

## Current evidence boundary

Synthetic tests and two short Beijing 1% technical smokes establish the
mapping, optimization, CCH, and checkpoint contracts:

- [`original_edges_smoke_1pct.json`](experiments/configs/original_edges_smoke_1pct.json)
- [`edge_transition_arcs_smoke_1pct.json`](experiments/configs/edge_transition_arcs_smoke_1pct.json)

The first deterministic Beijing 10% calibration exposed an optimizer
regression: direct-weight Euclidean updates improved Edge F1 by only about
`2e-5`. A generic relative-coordinate recovery then reproduced the historical
`original_edges` result and established meaningful learning for both graph
representations. At their minimum-objective checkpoints, decoded Edge F1 is
0.685404 for `original_edges` and 0.694125 for
`edge_transition_arcs`; Exact Match is 0.373640 and 0.377245. The line graph
therefore remains the recommended representation for a later NeuroMLR
comparison, now with learning gain rather than initialization alone as
evidence. Its best checkpoint is the registered update-299 boundary, so
convergence remains unconfirmed.

See the [optimizer-recovery report](experiments/optimizer_recovery/report.md)
and [machine-readable summary](experiments/optimizer_recovery/summary.json).
The earlier [direct-weight calibration](experiments/line_graph_10pct_calibration/report.md)
is retained as the diagnostic baseline. No test split was read, so this remains
development evidence rather than a test-set claim.

## Development checks

```bash
cargo fmt --check
cargo build --release --locked
cargo test --locked --all-targets
cargo clippy --locked --all-targets -- -D warnings
git diff --check
```

Input loading, deterministic subset generation, CCH infrastructure, reference
shortest paths, and route metrics remain shared infrastructure.
