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

## Independent departure-time partitions

Whole-trip `start_time` and `end_time` remain aligned with every accepted path.
The full-train `MMDD` audit identifies Unix seconds in `Asia/Shanghai`: all
785,709 raw train records match the UTC+8 civil date. Five coarse departure
buckets (`00-06`, `06-10`, `10-16`, `16-20`, `20-24`) were frozen from that
train audit.

Time is only a data-selection and checkpoint-dispatch concern. For bucket `b`,
the ordinary static trainer receives

```text
D_b = {trip_i : bucket(start_time_i) = b}
```

and learns one independent direct vector with the unchanged
`edge_transition_arcs`, length baseline, relative projected-subgradient
objective, regularizer, projection, CCH oracle, and static checkpoint. There is
no shared global parameter, bucket residual, special optimizer, travel-time
baseline, or temporal checkpoint. At inference, departure time selects one of
the ordinary static checkpoints before the usual CCH query.

## Checkpoints

A checkpoint records the representation and topology identity, configuration
and runtime data identity, current direct weights, and `completed_updates`.
Restoring it rebuilds the same length anchor, bounds, optimizer geometry, and
CCH topology, then resumes the square-root learning-rate clock. Bucketed runs
use this same checkpoint shape; the outer dispatcher chooses which checkpoint
to load.

## Current evidence boundary

Synthetic tests and two short Beijing 1% technical smokes establish the
mapping, optimization, CCH, and checkpoint contracts:

- [`original_edges_smoke_1pct.json`](experiments/configs/original_edges_smoke_1pct.json)
- [`edge_transition_arcs_smoke_1pct.json`](experiments/configs/edge_transition_arcs_smoke_1pct.json)

The relative-coordinate recovery first established a 10% Beijing line-graph
baseline at F1 0.694125 and Exact Match 0.377245. Full training data moves the
same static model to F1 0.700554 and Exact Match 0.388186 at interior update
400; all five departure buckets improve over the 10% model.

The active departure-time comparison fits five independent ordinary static
checkpoints with the same eta, regularization, bounds, and 500-update budget as
the full static reference. The five disjoint validation results are aggregated
with their fixed sample counts. No bucket-specific hyperparameter search is
used. The result is negative overall: independent buckets reach F1 0.699198,
Exact Match 0.384202, and Edge Jaccard 0.626007, versus 0.700554, 0.388186,
and 0.627908 for one full static model. Mean regret also rises from 303,899.5
to 320,424.2 mm. Only two of five buckets improve F1, and only the 20-24
bucket improves Exact Match. All five selected checkpoints are before update
500, so extending the fixed search solely to rescue this result is not
justified. The full static line graph remains the recommended model.

The final held-out comparison adds a fixed-first-edge/fixed-last-edge query for
fair comparison with NeuroMLR-Greedy. On the common 500-path Beijing test set,
the project reaches Edge F1 0.766015 versus 0.768496 for NeuroMLR-Greedy. The
quality is close but does not exceed the baseline. Internally, CCH is 1.68×
faster than binary-heap Dijkstra on the strictly path-stable 4,971-OD,
20-update training workload when CCH preprocessing is included, and 8.05×
faster for query-only inference on 500 one-thread node-to-node queries. See the
[final benchmark report](experiments/neuromlr_cch_dijkstra_benchmarks/report.md)
and [machine-readable summary](experiments/neuromlr_cch_dijkstra_benchmarks/summary.json).

The former shared-residual and trip-average travel-time study is preserved in
the [historical archive](experiments/archive/full_data_shared_temporal_residual)
but its model, optimizer, checkpoint, evaluator, and binaries are no longer
active. See the [independent-bucket report](experiments/independent_time_buckets/report.md)
and [machine-readable summary](experiments/independent_time_buckets/summary.json).
The [optimizer-recovery report](experiments/optimizer_recovery/report.md) and
earlier [direct-weight calibration](experiments/line_graph_10pct_calibration/report.md)
remain as historical evidence. No test split was read for those historical
studies, so their results remain development evidence. The final benchmark
above used its hash-gated test manifest exactly once after validation protocol
freeze.

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
