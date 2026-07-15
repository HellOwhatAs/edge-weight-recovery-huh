# Research status

## Current claim

The project now treats graph representation and inverse shortest-path
optimization as separate concerns. First-order and second-order problems
share the same objective, optimizer, update clock, projection logic, training
loop, and checkpoint state. Their only intended differences are topology,
trajectory mapping, metric conversion inside the oracle, and route decoding.

This is an architectural claim. No new empirical superiority claim has been
established in this revision.

## Representation definitions

In the first-order graph, each learned coordinate is an original directed
road, and an observed road sequence maps to itself.

In the second-order graph, each learned coordinate is a legal adjacent road
pair `(e_i,e_{i+1})`. Nodes `(e1,e2)` and `(e2,e3)` are adjacent by overlap,
and a road sequence of length `N >= 2` maps to `N-1` such nodes. Decoding the
overlapping nodes reconstructs the original road sequence. A single-road
observation has no pair coordinate and is rejected instead of adding an
artificial start state.

Source and target construction, graph-weight conversion for CCH, topology
identity, observed-path validation, and decoded route construction are owned
by the representation/oracle layer. These details do not enter the objective
or create learned coordinates.

## Optimization invariant

For either representation the trainer owns a direct vector `w` initialized at
`w0` and applies

```text
J(w) = average[observed path cost - predicted shortest-path cost]
       + lambda / (2m) * ||w - w0||^2,

g = (observed_counts - predicted_counts) / N
    + lambda / m * (w - w0),

eta_k = eta0 / sqrt(k + 1),
w <- project(w - eta_k * g).
```

A single global `completed_updates` value controls the schedule and is stored
with the direct weights in the graph-order-independent checkpoint.

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
- one three-update Beijing 1% technical smoke per graph order.

The active smoke configurations are
[`first_order_smoke_1pct.json`](../experiments/configs/first_order_smoke_1pct.json)
and
[`second_order_smoke_1pct.json`](../experiments/configs/second_order_smoke_1pct.json).
They share the same data identities and optimizer settings. Both three-update
release runs passed on 2026-07-16: objectives were finite, 25,825 first-order
and 33,020 second-order direct coordinates changed, every grouped shortest-path
query completed, and strict checkpoint reload succeeded. Independent resumes
from each state-0 checkpoint reproduced the complete final checkpoint exactly.
No test split was read, and these technical checks establish no graph-order
quality ranking.

## Claim boundary and next step

The immediate completion gate is finite objectives, changed weights, healthy
shortest-path queries, and successful checkpoint continuation in both smokes,
together with the full synthetic correctness suite. Larger training,
hyperparameter calibration, formal graph-order comparison, and all test-split
evaluation remain out of scope until this architecture is verified.
