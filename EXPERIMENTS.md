# Experiments

## Active invariant and calibration

The architecture retains this invariant:

```text
graph representation affects only topology construction, trajectory mapping,
and route decoding; inverse optimization learns one direct-weight vector on
the supplied graph.
```

With that architecture frozen, the active Beijing 10% calibration compares
decoded route quality for the two representations, screens only
`eta0={300,1000,3000}` plus the permitted adjacent boundary candidate, and
runs one 200-update development configuration per representation. The full
protocol, results, audit, and risk boundary are in
[`experiments/line_graph_10pct_calibration/report.md`](experiments/line_graph_10pct_calibration/report.md).

## Representation contract

`original_edges` uses the original directed road graph and learns one weight
per original edge.

`edge_transition_arcs` uses the directed line graph (line digraph / edge-based
graph): routing nodes are original directed edges, routing arcs are legal
consecutive transitions `e -> f`, and each such arc is one directly learned
coordinate `w[e,f]`. A trajectory `(e1,...,eN)` is the routing-node path
`e1 -> ... -> eN`; its coordinate path is `windows(2)` and has `N - 1` arcs.
Its cost is

```text
sum_i w[e_i,e_{i+1}].
```

Line-graph queries use all original edges leaving the requested source as
source states and all original edges entering the requested target as target
states. Every endpoint offset is zero. Decoding a returned routing-node path
means interpreting those node IDs directly as original-edge IDs.

The topology contains only the original-edge routing nodes and directly
learned transition arcs described above. All active experiments filter
observations with `N < 2` for both representations; no start node, start cost,
or first-edge parameter is added.

## Mathematical contract

For mapped trajectories `P_i`, initial weights `w0`, and `m` graph
coordinates, both representations use

```text
J(w) = (1/N) sum_i [cost_w(P_i) - dist_w(s_i,t_i)]
       + lambda / (2m) * ||w - w0||^2,

g = (observed_counts - predicted_counts) / N
    + lambda / m * (w - w0),

eta_k = eta0 / sqrt(k + 1),
w <- project(w - eta_k * g).
```

The graph representation supplies the topology, mapped observations, initial
weights, bounds, coordinate counts, topology identity, and route decoder. The
trainer receives only this graph-problem contract.

The active RoutingKit binding accepts integer CCH weights. Production queries
therefore select routes after rounding the direct `f64` vector, while reported
direct path costs use that vector. Strict continuous-weight oracle equivalence
is deferred; the synthetic CCH/reference gate checks the integer metric that
the CCH actually receives.

## Synthetic correctness gates

The correctness suite must establish:

1. identity mapping for `original_edges` road trajectories;
2. directed-line-graph node mapping, `windows(2)` arc mapping, and decoding;
3. use of the same direct-weight optimizer by both representations;
4. the update formula, regularization, projection, and global clock;
5. shortest-path cost agreement between CCH and reference Dijkstra on small
   graphs; and
6. identical final state for uninterrupted training and checkpoint resume.

## Prior Beijing 1% technical smokes

These configurations established the technical baseline before calibration:

| Configuration | Representation | Updates | Validation cadence | Threads |
|---|---:|---:|---:|---:|
| `experiments/configs/original_edges_smoke_1pct.json` | `original_edges` | 3 | 3 | 4 |
| `experiments/configs/edge_transition_arcs_smoke_1pct.json` | `edge_transition_arcs` | 3 | 3 | 4 |

Both use the deterministic Beijing `scale_1pct_seed42` training subset and the
fixed `scale_fixed_seed20260715` validation subset. The test split is never
read.

A smoke is healthy only if all of the following hold:

- every reported objective is finite;
- at least one direct weight changes from initialization;
- shortest-path customization and queries complete normally; and
- a saved checkpoint restores and can continue on the original update clock.

The corrected smokes completed on 2026-07-16:

| Representation | Routing nodes | Routing arcs / coordinates | Final train objective | Final validation objective | Changed coordinates | Wall time |
|---|---:|---:|---:|---:|---:|---:|
| `original_edges` | 31,199 | 72,156 | 595,624.236207 | 650,361.204608 | 25,825 | 1.61 s |
| `edge_transition_arcs` | 72,156 | 188,249 | 586,188.223920 | 636,885.624068 | 33,018 | 4.04 s |

Every health check passed: objectives were finite, weights changed,
shortest-path queries completed, and `test_read` remained false. Independently
resuming each run from `checkpoint-0.json` produced a byte-identical final
checkpoint after update 3. The two rows remain independent technical health
checks and must not be interpreted as comparable endpoint-quality
measurements.

## Beijing 10% route-quality calibration

The fixed `scale_10pct_seed42` train and `scale_fixed_seed20260715` validation
calibration selected eta 300 for `original_edges` and eta 100 for
`edge_transition_arcs`. At their minimum-objective 200-update checkpoints,
decoded validation Edge F1 was 0.589923 and 0.603495 respectively; Exact Match
was 0.335947 and 0.346762. Both minima occurred at update 200, so convergence
is not confirmed and no longer run was added. Raw objectives are not compared
across representations. No test split was read.

The tracked machine-readable evidence is
[`summary.json`](experiments/line_graph_10pct_calibration/summary.json). Full
logs and checkpoints remain under the ignored local artifact tree.
