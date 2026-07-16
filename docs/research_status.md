# Research status

## Current claim

The project treats graph representation and inverse shortest-path optimization
as separate concerns. `original_edges` and `edge_transition_arcs` share the
same relative-coordinate objective, optimizer, update clock, projection logic,
training loop, and direct-weight checkpoint state. Their intended differences
are topology, trajectory mapping, coordinate interpretation, and route
decoding.

The direct-weight Euclidean calibration exposed an optimization regression and
is no longer the active training geometry. A generic `q=w/w0` recovery has now
reproduced the historical edge-only result and produced substantial learning
gain for both representations. On that Beijing 10% development protocol,
`edge_transition_arcs` has higher decoded Edge F1 and Exact Match than
`original_edges`; this is a development result, not a test-set or general
superiority claim.

## Representation definitions

In `original_edges`, the original directed road graph is routed directly.
Every original road edge is one learned coordinate, and an observed road
sequence maps to itself.

`edge_transition_arcs` is the original graph's directed line graph (line
digraph / edge-based graph):

- one routing node represents one original directed edge `e`;
- one routing arc `e -> f` represents each legal transition satisfying
  `head(e) = tail(f)`; and
- that routing arc carries the directly learned weight `w[e,f]`.

Thus `(e1,...,eN)` is a line-graph node path with the `N - 1` coordinate arcs
returned by `windows(2)`. Its cost is

```text
w[e1,e2] + w[e2,e3] + ... + w[e{N-1},eN].
```

Source states are the original directed edges leaving the source vertex;
target states are the original directed edges entering the target vertex.
Both endpoint offsets are zero, and a returned routing-node sequence decodes
directly to the corresponding original-road sequence.

The representation contains only the original-edge routing nodes and directly
learned transition arcs described above. All experiments filter trajectories
with `N < 2` for both representations rather than introducing a start node,
first-edge cost, or other special parameter.

## Optimization invariant

For either representation the trainer owns one stored direct vector `w`
initialized at `w0`, optimizes `q=w/w0`, and applies

```text
J(q) = average[observed path cost under w0*q
               - predicted shortest-path cost under w0*q]
       + lambda / (2m) * ||q - 1||^2,

g_q = w0 * (observed_counts - predicted_counts) / N
      + lambda / m * (q - 1),

eta_k = eta0 / sqrt(k + 1),
q <- project(q - eta_k * g_q),
w <- w0 * q.
```

A single global `completed_updates` value controls the schedule and is stored
with the direct weights in the representation-independent checkpoint. This is
equivalent to `diag(w0^2)` preconditioning in direct-weight space and introduces
no representation-specific optimizer state.

The prior direct geometry is retained behind the explicit
`projected_subgradient` configuration value for reproducibility; the active
geometry is `relative_projected_subgradient`. Configurations are never silently
reinterpreted.

## Known oracle boundary

`routingkit-cch` currently accepts only `u32` metric weights. The graph layer
rounds the direct `f64` weights for route selection, then evaluates the returned
coordinate path under the direct vector. When rounding changes path ordering,
this is not a strictly exact oracle for the written continuous objective and
its count subgradient. Replacing it with an exact continuous-weight oracle is
explicitly deferred; there is still only one learned/checkpointed vector and
one optimizer.

## Verification and calibration status

Synthetic mapping, decoding, optimizer, projection, clock, CCH/reference, and
checkpoint-resume tests remain the correctness gate. The prior Beijing 1%
technical smokes established healthy execution for both representations.

The first bounded Beijing 10% calibration used the same 62,348 filtered train
trajectories and 15,812 fixed validation trajectories for both representations.
Its direct-weight geometry improved F1 by only about `2e-5`, exposing the scale
regression described in its [diagnostic report](../experiments/line_graph_10pct_calibration/report.md).

The recovery then used the same data and one common configuration:
`eta0=0.0002`, `lambda=100000`, relative bounds `[0.1,10]`, 299 updates,
validation cadence 10, and four threads. `original_edges` recovered historical
edge-only performance: Edge F1 0.685404 and Exact Match 0.373640 at update 290,
versus the old 0.682145 and 0.368454. The same optimizer brought
`edge_transition_arcs` to F1 0.694125 and Exact Match 0.377245 at update 299.
These are gains of 0.095503 and 0.090658 F1 from their respective update-0
states, so meaningful learning is now established.

Under the matched recovery protocol, line graph exceeds original edges by
0.008720 F1 and 0.003605 Exact Match, at 2.42x training wall time and 1.36x
peak RSS. Its best checkpoint is the final registered update, so convergence is
still not confirmed. The complete evidence is in the
[optimizer-recovery report](../experiments/optimizer_recovery/report.md). The
test split was never read. A later NeuroMLR comparison should carry
`edge_transition_arcs` with `relative_projected_subgradient`, while preserving
the single-edge zero-cost, integer-quantization, single-seed, and convergence
risks.
