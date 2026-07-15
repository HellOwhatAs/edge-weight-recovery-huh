# Edge-weight recovery

This project learns direct nonnegative weights for a graph from observed road
trajectories. The architecture has two deliberately separate layers:

1. a graph-representation layer chooses a first-order or second-order graph,
   maps each road trajectory into that graph, and owns route decoding; and
2. one inverse-shortest-path trainer learns a single weight vector on the
   graph it receives.

The trainer does not inspect what a graph coordinate means. A coordinate may
represent a road in the first-order graph or a legal road-to-road transition
in the second-order graph; optimization is identical in both cases.

## Graph representations

Let an original trajectory be

```text
(e1, e2, ..., eN).
```

### First order

The original directed road graph is used directly. Each road is one learned
coordinate, and the mapped trajectory is unchanged:

```text
(e1, e2, ..., eN).
```

### Second order

Every legal consecutive road pair is one graph node:

```text
(e1, e2).
```

Two such nodes are connected exactly when they overlap:

```text
(e1, e2) -> (e2, e3).
```

A road trajectory of length `N >= 2` maps to the `N - 1` transition nodes

```text
(e1,e2), (e2,e3), ..., (e{N-1},eN).
```

A one-road observation has no second-order pair coordinate and is rejected by
that representation. No artificial start state or start cost is introduced.

Decoding reverses this overlap rule: emit both roads from the first pair, then
append the second road from every following pair. Source and target handling,
topology identity, mapped-path validation, and conversion from learned node
weights to the arc weights required by CCH stay inside the representation and
oracle layer. They do not introduce trainable coordinates.

## One inverse-shortest-path optimizer

For either graph, let `m` be its number of learned coordinates, `w0` the
initial direct-weight vector, and `N` the number of observations. Training
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
clock, projection rule, training loop, and checkpoint format for both graph
orders.

The current RoutingKit CCH binding accepts `u32` metrics, so the representation
layer rounds the direct `f64` weights for route selection and then evaluates
the selected coordinates under the direct vector. Exact continuous-weight CCH
routing is a known deferred oracle limitation; it does not introduce a second
learned vector or graph-specific optimizer state.

## Checkpoints

A checkpoint records the graph order and topology identity, configuration and
runtime data identity, the current direct weights, and `completed_updates`.
Restoring it rebuilds `w0`, bounds, and the graph-specific oracle from the
verified configuration and topology, then resumes the same square-root
learning-rate clock. The checkpoint structure is independent of graph order.

## Current verification boundary

This revision is an architecture and correctness pass. Verification consists
of synthetic tests plus two three-update technical smokes on the deterministic
Beijing 1% training subset and fixed validation subset:

- [`first_order_smoke_1pct.json`](experiments/configs/first_order_smoke_1pct.json)
- [`second_order_smoke_1pct.json`](experiments/configs/second_order_smoke_1pct.json)

Both smokes passed: objectives stayed finite, direct weights changed,
shortest-path queries succeeded, and fresh runs resumed from each
`checkpoint-0.json` to bit-identical final checkpoints. They are not a model
comparison. This revision does not run larger training, tune learning rates,
evaluate a test split, or make route-quality claims.

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
