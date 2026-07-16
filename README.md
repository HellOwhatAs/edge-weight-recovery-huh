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

## Coarse departure-time conditioning

The temporal extension retains the pickle's whole-trip `start_time` and
`end_time` alongside every path. Beijing timestamps are interpreted as Unix
seconds in `Asia/Shanghai` only after checking the full-train `MMDD` keys:
all 785,709 match UTC+8 civil dates, whereas 695,129 match UTC dates. Inference
selects one of five train-derived departure-hour buckets (`00-06`, `06-10`,
`10-16`, `16-20`, `20-24`). Validation timestamps are used only to select the
already-defined bucket and report metrics.

For bucket `b`, the relative coordinate is

```text
q_b = q_global + r_b
w_b = w0_b * q_b.
```

`q_global` is shared by every bucket and each `r_b` is a bounded residual
regularized toward zero. The full-batch objective is

```text
(1/N) sum_b sum_(trip in b)[observed_cost(w_b) - shortest_path_cost(w_b)]
+ lambda_global / (2m)  * ||q_global - 1||^2
+ lambda_residual / (2mB) * sum_b ||r_b||^2.
```

It is convex in the global and residual coordinates. Independent boxes on
`q_global` and `r_b` guarantee every effective multiplier remains in the
configured `[0.1, 10]` range. The graph problem still supplies only topology,
trajectory mapping, coordinate counts, CCH customization, and route decoding;
it does not inspect temporal optimizer state. Line-graph source states,
zero endpoint offsets, transition meanings, and the absence of a first-edge
cost are unchanged.

The optional travel-time baseline uses only accepted training trajectories.
For a complete trip `t`, it first computes the proxy

```text
v_t = full_path_length_t / (end_time_t - start_time_t).
```

This is explicitly a whole-trip average, not an observed per-edge speed. After
clipping implausible extremes, road-global speeds shrink to the train-wide
mean and road-bucket speeds shrink to their road-global values. The two prior
counts are train-support quantiles. An edge baseline is proportional to
`length / smoothed_speed`; one train-derived positive global fixed-point scale
reduces CCH `u32` rounding without changing route order. For line-graph
coordinates, the baseline remains the entered edge's value, so no new first
edge term is introduced.

## Checkpoints

A static checkpoint records the representation and topology identity,
configuration and runtime data identity, current direct weights, and
`completed_updates`. A temporal checkpoint records the global relative vector,
all bucket residuals, the complete bucket definition, train-derived baseline
vectors and diagnostics, configuration, runtime identity, topology identity,
and update clock. Restore verifies those identities and resumes the same
square-root learning-rate clock; inference selects the stored metric using the
departure timestamp.

## Current evidence boundary

Synthetic tests and two short Beijing 1% technical smokes establish the
mapping, optimization, CCH, and checkpoint contracts:

- [`original_edges_smoke_1pct.json`](experiments/configs/original_edges_smoke_1pct.json)
- [`edge_transition_arcs_smoke_1pct.json`](experiments/configs/edge_transition_arcs_smoke_1pct.json)

The relative-coordinate recovery first established a 10% Beijing line-graph
baseline at F1 0.694125 and Exact Match 0.377245. Full training data moves the
same static model to F1 0.700554 and Exact Match 0.388186 at interior update
400; all five departure buckets improve over the 10% model.

The shared departure-time model with a length baseline reaches F1 0.702280
and Exact Match 0.388629 at update 400. Replacing its anchor with the
train-only smoothed whole-trip-speed proxy reaches F1 0.703176 and Exact Match
0.389704 at update 425. The best temporal model is only `+0.002623` F1 and
`+0.001518` Exact over the full static model, with F1 losses in two of five
buckets. This is a small, mixed development gain rather than evidence of a
significant and stable improvement. Decoded metrics peak inside the bounded
budget, although regularized objectives are still falling at update 500, so
numerical objective convergence is not confirmed.

See the [full-data time-conditioning report](experiments/full_data_time_conditioning/report.md)
and its [machine-readable summary](experiments/full_data_time_conditioning/summary.json).
The [optimizer-recovery report](experiments/optimizer_recovery/report.md) and
earlier [direct-weight calibration](experiments/line_graph_10pct_calibration/report.md)
remain as historical evidence. No test split was read, so all results are
development evidence rather than test-set claims.

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
