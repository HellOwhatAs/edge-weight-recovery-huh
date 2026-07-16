# Research status

## Current claim

The project treats graph representation and inverse shortest-path optimization
as separate concerns. `original_edges` and `edge_transition_arcs` share the
same relative-coordinate objective, optimizer, update clock, projection logic,
training loop, and direct-weight checkpoint state. Their intended differences
are topology, trajectory mapping, coordinate interpretation, and route
decoding.

The direct-weight Euclidean calibration exposed an optimization regression and
is no longer the active training geometry. A generic `q=w/w0` recovery
reproduced the historical edge-only result and established
`edge_transition_arcs` as the current representation. The subsequent full
Beijing development study shows that using all training trajectories is the
clear improvement; coarse departure-time residuals and a train-only
travel-time proxy add only small, mixed-bucket gains. These are fixed
validation results, not test-set, statistical-significance, or general
superiority claims.

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
explicitly deferred. The static model still has one learned/checkpointed
vector and one optimizer.

For the temporal model there is one effective vector per registered bucket,
constructed from a shared global relative vector plus a bounded residual. CCH
quantization is diagnosed independently for every bucket; the selected formal
checkpoints have no zero integer weights and maximum relative rounding errors
between `3.92e-4` (length anchor) and `8.76e-4` (travel-time anchor).

## Temporal extension and train-only baseline

Accepted paths retain aligned whole-trip `start_time` and `end_time` values.
The full-train `MMDD` keys match the timestamps' UTC+8 civil date for all
785,709 raw records, so the registered interpretation is Unix seconds in
`Asia/Shanghai`. Five train-derived departure buckets cover 00-06, 06-10,
10-16, 16-20, and 20-24; validation timestamps only assign already-frozen
buckets.

The convex temporal parameterization is

```text
q_b = q_global + residual_b,
w_b = w0_b * q_b.
```

All buckets share `q_global`; bounded `L2`-regularized residuals shrink toward
zero. The optional travel-time anchor estimates only a complete trip's average
speed, clips it to `[1, 33.333...]` m/s, then shrinks road-bucket speed to
road-global and train-network means. It is not a per-edge speed observation.
Validation and test contribute to none of its statistics.

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
peak RSS. The complete historical evidence is in the
[optimizer-recovery report](../experiments/optimizer_recovery/report.md).

The full-data study then used 623,275 filtered train trajectories and the same
15,812 fixed validation trajectories:

| Stage | Selected update | Edge F1 | Exact Match | Edge Jaccard |
|---|---:|---:|---:|---:|
| Existing 10% static line graph | 299 | 0.694125 | 0.377245 | 0.620388 |
| Full static line graph | 400 | 0.700554 | 0.388186 | 0.627908 |
| Full temporal, length anchor | 400 | 0.702280 | 0.388629 | 0.629917 |
| Full temporal, travel-time anchor | 425 | 0.703176 | 0.389704 | 0.631127 |

Full data adds 0.006429 F1 and 0.010941 Exact, with F1 gains in all five
buckets. Time conditioning adds only 0.001726 F1 and 0.000443 Exact over the
full static model, with three bucket F1 losses. The travel-time anchor adds
another 0.000896 F1 and 0.001075 Exact over the temporal length model, with
only two bucket F1 gains. All formal selections are before update 500, so
decoded route quality is no longer budget-boundary selected. Objectives still
fall at the final update, hence numerical objective convergence is not
confirmed.

The [full report](../experiments/full_data_time_conditioning/report.md),
[machine-readable summary](../experiments/full_data_time_conditioning/summary.json),
and [time audit](../experiments/full_data_time_conditioning/time_audit.json)
retain the complete evidence. The test split was never read. Remaining risks
are the single fixed date split, sparse road-time support, whole-trip rather
than per-edge time estimates, zero first-edge cost for line-graph queries, and
integer CCH quantization.
