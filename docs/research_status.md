# Research status

## Current claim

The project treats graph representation and inverse shortest-path optimization
as separate concerns. `original_edges` and `edge_transition_arcs` share the
same objective, optimizer, update clock, projection logic, training loop, and
checkpoint state. Their intended differences are topology, trajectory
mapping, coordinate interpretation, and route decoding.

The architecture is now held fixed while representation quality is evaluated.
On the registered Beijing 10% development protocol, `edge_transition_arcs`
has higher decoded Edge F1 and Exact Match than `original_edges`; this is a
development result, not a test-set or general superiority claim.

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

For either representation the trainer owns one direct vector `w` initialized
at `w0` and applies

```text
J(w) = average[observed path cost - predicted shortest-path cost]
       + lambda / (2m) * ||w - w0||^2,

g = (observed_counts - predicted_counts) / N
    + lambda / m * (w - w0),

eta_k = eta0 / sqrt(k + 1),
w <- project(w - eta_k * g).
```

A single global `completed_updates` value controls the schedule and is stored
with the direct weights in the representation-independent checkpoint.

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

The subsequent bounded Beijing 10% calibration used the same 62,348 filtered
train trajectories and 15,812 fixed validation trajectories for both
representations. It selected eta 300 for `original_edges` and eta 100 for
`edge_transition_arcs`. At the minimum-objective 200-update checkpoints,
decoded Edge F1 was 0.589923 versus 0.603495 and Exact Match was 0.335947
versus 0.346762. Both checkpoints landed at the final registered update, so
convergence is not confirmed. The line graph required 2.42x training wall time
and 1.36x peak RSS.

The complete audit and result tables are in the
[calibration report](../experiments/line_graph_10pct_calibration/report.md).
The test split was never read. The next authorized comparison should carry
`edge_transition_arcs` into NeuroMLR while preserving the reported single-edge
zero-cost, integer-quantization, and convergence risks.
