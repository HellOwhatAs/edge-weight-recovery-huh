# Edge-weight recovery

This project learns one direct nonnegative weight vector from observed road
trajectories. Its architecture separates two concerns:

1. a graph-representation layer chooses `original_edges` or
   `edge_transition_arcs`, maps road trajectories into that topology, and owns
   route decoding; and
2. one inverse-shortest-path trainer learns the direct coordinates supplied by
   the representation.

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
the initial direct-weight vector, and `N` the number of observations. Training
minimizes

```text
J(w) = average[
         observed_path_cost(w)
         - predicted_shortest_path_cost(w)
       ]
       + lambda / (2m) * ||w - w0||^2.
```

With coordinate counts from the mapped observed and predicted paths, one
subgradient is

```text
g = (observed_counts - predicted_counts) / N
    + lambda / m * (w - w0).
```

Update `k` uses one global clock and one projection box:

```text
eta_k = eta0 / sqrt(k + 1)
w <- project(w - eta_k * g).
```

The configured lower and upper factors are applied coordinate-wise to `w0`.
There is one direct-weight vector, learning rate, regularization coefficient,
clock, projection rule, training loop, and checkpoint format for both
representations.

The current RoutingKit CCH binding accepts `u32` metrics, so direct `f64`
weights are rounded for route selection and selected paths are evaluated under
the direct vector. Exact continuous-weight CCH routing remains a deliberately
deferred oracle limitation; it does not introduce another learned vector or a
representation-specific optimizer.

## Checkpoints

A checkpoint records the representation and topology identity, configuration
and runtime data identity, current direct weights, and `completed_updates`.
Restoring it rebuilds `w0`, bounds, and the representation-specific oracle from
the verified configuration and topology, then resumes the same square-root
learning-rate clock. Both representations use the same checkpoint structure.

## Current verification boundary

This revision is an architecture and correctness pass. Verification consists
of synthetic tests plus two short technical smokes on the deterministic
Beijing 1% training subset and fixed validation subset:

- [`original_edges_smoke_1pct.json`](experiments/configs/original_edges_smoke_1pct.json)
- [`edge_transition_arcs_smoke_1pct.json`](experiments/configs/edge_transition_arcs_smoke_1pct.json)

Both configurations completed three updates on 2026-07-16. The original-edge
run used 31,199 routing nodes and 72,156 direct coordinates; the directed line
graph used 72,156 routing nodes and 188,249 direct transition-arc coordinates.
Both produced finite objectives, changed weights, completed shortest-path
queries, and resumed from update 0 to byte-identical update-3 checkpoints.
These smokes are independent health checks, not a model comparison.
This revision does not run larger training, tune learning rates, evaluate a
test split, or make route-quality claims.

## Development checks

```bash
cargo fmt --check
cargo build --locked --all-targets
cargo test --locked --all-targets
cargo clippy --locked --all-targets -- -D warnings
git diff --check
```

Input loading, deterministic subset generation, CCH infrastructure, reference
shortest paths, and route metrics remain shared infrastructure.
