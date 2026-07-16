# Research status

## Current claim

The project treats graph representation and inverse shortest-path optimization
as separate concerns. `original_edges` and `edge_transition_arcs` share the
same objective, optimizer, update clock, projection logic, training loop, and
checkpoint state. Their intended differences are topology, trajectory
mapping, coordinate interpretation, and route decoding.

This is an architectural claim. No empirical superiority claim has been
established in this revision.

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

## Verification status

The required evidence for this pass is deliberately narrow:

- synthetic mapping, decoding, optimizer, projection, clock, CCH/reference,
  and checkpoint-resume tests; and
- one short Beijing 1% technical smoke per representation.

The active smoke configurations are
[`original_edges_smoke_1pct.json`](../experiments/configs/original_edges_smoke_1pct.json)
and
[`edge_transition_arcs_smoke_1pct.json`](../experiments/configs/edge_transition_arcs_smoke_1pct.json).
They share the same data identities and optimizer settings. On 2026-07-16 both
completed three updates with finite objectives, changed direct weights, and
healthy shortest-path queries. The corrected topologies were 31,199 nodes /
72,156 arcs for `original_edges` and 72,156 nodes / 188,249 arcs for
`edge_transition_arcs`; in each case the arc count is also the learned
coordinate count. Resuming from update 0 reproduced the uninterrupted final
checkpoint byte for byte. No test split was read, and these technical checks
establish no representation-quality ranking.

## Claim boundary and next step

The immediate completion gate is finite objectives, changed weights, healthy
shortest-path queries, and successful checkpoint continuation in both smokes,
together with the full synthetic correctness suite. Larger training,
hyperparameter calibration, formal representation comparison, and all
test-split evaluation remain out of scope until this architecture is verified.
