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
clear improvement. The active time experiment now asks the narrower question:
does splitting those trajectories by departure time and fitting five ordinary
static models beat one full static model? The former shared-residual and
travel-time proxy extension is archived rather than part of the active model.
All results are fixed validation evidence, not test-set,
statistical-significance, or general superiority claims.

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

For independent departure partitions, each bucket has an ordinary static
checkpoint and its own integer metric. Quantization is diagnosed per
checkpoint; there is no shared or residual parameter block.

## Departure-time data partitions

Accepted paths retain aligned whole-trip `start_time` and `end_time` values.
The full-train `MMDD` keys match the timestamps' UTC+8 civil date for all
785,709 raw records, so the registered interpretation is Unix seconds in
`Asia/Shanghai`. Five train-derived departure buckets cover 00-06, 06-10,
10-16, 16-20, and 20-24. Validation timestamps only assign already-frozen
buckets and never determine their boundaries.

For each bucket `b`, the loader constructs

```text
D_b = {trip_i : bucket(start_time_i) = b}.
```

The unchanged static trainer independently fits one length-anchored direct
weight vector on each `D_b`. Time does not enter the graph problem, objective,
optimizer, regularizer, projection, CCH oracle, or checkpoint. Inference maps
departure time to one of those static checkpoints. The earlier shared-residual
and trip-average travel-time experiment is retained only in
[`experiments/archive/full_data_shared_temporal_residual`](../experiments/archive/full_data_shared_temporal_residual).

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

The full-data static study then used 623,275 filtered train trajectories and
the same 15,812 fixed validation trajectories. The narrower departure-time
study uses exactly the same paths and evaluator:

| Stage | Selected update | Edge F1 | Exact Match | Edge Jaccard | Mean regret (mm) |
|---|---:|---:|---:|---:|---:|
| Existing 10% static line graph | 299 | 0.694125 | 0.377245 | 0.620388 | 321,414.6 |
| Full static line graph | 400 | **0.700554** | **0.388186** | **0.627908** | **303,899.5** |
| Five independent bucket static models | 475/275/200/275/150 | 0.699198 | 0.384202 | 0.626007 | 320,424.2 |

Using full data adds 0.006429 F1 and 0.010941 Exact over the existing 10%
result. Splitting that data into five independent models instead loses
0.001356 F1, 0.003984 Exact, and 0.001900 Jaccard, while mean regret increases
by 16,524.8 mm. Bucket-specific F1 improves only at 10-16 (+0.001586) and
20-24 (+0.001192); Exact improves only at 20-24 (+0.002082). The 00-06 bucket,
with just 35,201 train trajectories, loses 0.010841 F1 and 0.021804 Exact.
This is consistent with lost cross-time statistical sharing outweighing the
route-preference difference captured by coarse departure buckets.

All five independent selections precede update 500, although the sparse night
model selects update 475. This establishes an interior development peak but
does not prove numerical convergence of each convex objective. There was no
bucket-specific hyperparameter search and no formal significance test. The
full static line graph therefore remains the active recommendation.

The [independent-bucket report](../experiments/independent_time_buckets/report.md),
[machine-readable summary](../experiments/independent_time_buckets/summary.json),
and [time audit](../experiments/independent_time_buckets/time_audit.json) retain
the active evidence. The former shared-residual/trip-average travel-time study
remains reproducible in the
[historical archive](../experiments/archive/full_data_shared_temporal_residual)
but none of its special model code remains active. The test split was never
read. Remaining risks are the single fixed date split, sparse transition
support after partitioning, zero first-edge cost for line-graph queries, and
integer CCH quantization.
