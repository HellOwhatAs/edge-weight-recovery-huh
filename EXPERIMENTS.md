# Experiments

## Active invariant and calibration

The architecture retains this invariant:

```text
graph representation affects only topology construction, trajectory mapping,
and route decoding; inverse optimization uses one common relative-coordinate
geometry and stores one direct-weight vector on the supplied graph.
```

The first Beijing 10% representation calibration intentionally froze the
direct-weight optimizer and exposed a severe optimization-scale regression.
The active recovery uses `q=w/w0` for both representations, first reproduces
the historical `original_edges` result, then applies the identical optimizer
to `edge_transition_arcs`. The full diagnosis, results, and risk boundary are
in [`experiments/optimizer_recovery/report.md`](experiments/optimizer_recovery/report.md).

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

For mapped trajectories `P_i`, initial weights `w0`, dimensionless coordinates
`q=w/w0`, and `m` graph coordinates, both representations actively use

```text
J(q) = (1/N) sum_i [cost_(w0*q)(P_i) - dist_(w0*q)(s_i,t_i)]
       + lambda / (2m) * ||q - 1||^2,

g_q = w0 * (observed_counts - predicted_counts) / N
      + lambda / m * (q - 1),

eta_k = eta0 / sqrt(k + 1),
q <- project(q - eta_k * g_q),
w <- w0 * q.
```

The graph representation supplies the topology, mapped observations, initial
weights, bounds, coordinate counts, topology identity, and route decoder. The
trainer receives only this graph-problem contract. It checkpoints `w`, not an
additional model block. In direct coordinates the update is equivalent to the
generic diagonal preconditioner `diag(w0^2)` paired with relative
regularization.

The old `projected_subgradient` kind remains available with its direct-weight
Euclidean semantics only to reproduce historical configurations. New formal
training uses the explicit `relative_projected_subgradient` kind.

The active RoutingKit binding accepts integer CCH weights. Production queries
therefore select routes after rounding the direct `f64` vector, while reported
direct path costs use that vector. Strict continuous-weight oracle equivalence
is deferred; the synthetic CCH/reference gate checks the integer metric that
the CCH actually receives.

## Synthetic correctness gates

The correctness suite must establish:

1. identity mapping for `original_edges` road trajectories;
2. directed-line-graph node mapping, `windows(2)` arc mapping, and decoding;
3. use of the same selected optimizer geometry by both representations;
4. direct and relative update formulas, matching regularizers, projection,
   and the global clock;
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

## Beijing 10% direct-weight diagnostic

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

This calibration is now a diagnostic baseline, not the optimizer used for a
formal model comparison. Its near-zero learning gain motivated the recovery
below.

## Beijing 10% relative-optimizer recovery

The recovery configurations differ only by graph representation and use
`relative_projected_subgradient`, `eta0=0.0002`, `lambda=100000`, multiplier
bounds `[0.1,10]`, 299 updates, validation cadence 10, and four Rayon threads.
These settings reproduce the historical edge-only multiplier geometry without
restoring any q/r model block.

At the minimum-objective checkpoints:

| Representation | Selected update | Update-0 F1 | Selected F1 | F1 gain | Update-0 Exact | Selected Exact | Training wall | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `original_edges` | 290 | 0.589902 | 0.685404 | +0.095503 | 0.336643 | 0.373640 | 125.95 s | 162,044 KiB |
| `edge_transition_arcs` | 299 | 0.603467 | 0.694125 | +0.090658 | 0.346952 | 0.377245 | 304.26 s | 220,580 KiB |

The original-edge regression gate passed: historical Edge F1 0.682145 and
Exact Match 0.368454 were recovered to 0.685404 and 0.373640, while update-299
mean regret differs from the historical trajectory by only 0.0194. On common
decoded route metrics, the trained line graph exceeds original edges by
0.008720 F1 and 0.003605 Exact Match. Raw objectives are not compared across
representations. The line-graph best is at the registered budget boundary, so
convergence remains unconfirmed. No test split was read.

Tracked evidence:

- [recovery report](experiments/optimizer_recovery/report.md)
- [machine-readable recovery summary](experiments/optimizer_recovery/summary.json)
- [two exact recovery configurations](experiments/optimizer_recovery/configs)

## Full-data independent departure-bucket study

The active study keeps `edge_transition_arcs`, the length anchor, and
`relative_projected_subgradient` fixed. It compares one full-train static model
with five ordinary static models trained independently after a coarse
departure-time data partition. Time does not enter the model, optimizer, graph
problem, or checkpoint. The fixed validation variant is unchanged, and no test
file is read.

### Timestamp and bucket audit

The pickle timestamps are Unix seconds. Full-train keys contain an `MMDD`
field: all `785,709 / 785,709` agree with the start timestamp's UTC+8 date,
while `695,129 / 785,709` agree with UTC. The study therefore records
`Asia/Shanghai`, fixed UTC+8 for these 2009 dates, and assigns metrics by trip
departure time.

Five coarse buckets were frozen from the train hourly profile before model
training:

| Local departure bucket | Filtered train | Fixed validation |
|---|---:|---:|
| 00:00-06:00 | 35,201 | 1,009 |
| 06:00-10:00 | 106,282 | 2,399 |
| 10:00-16:00 | 220,543 | 5,456 |
| 16:00-20:00 | 151,360 | 4,066 |
| 20:00-24:00 | 109,889 | 2,882 |
| **Total** | **623,275** | **15,812** |

The common path filter drops 162,434 cyclic full-train trajectories and 4,188
cyclic validation trajectories. There are no empty, short, discontinuous, or
out-of-bounds paths. The exact audit and bucket file are
[`time_audit.json`](experiments/independent_time_buckets/time_audit.json)
and
[`time_buckets.json`](experiments/independent_time_buckets/time_buckets.json).

### Independent static models

For each registered bucket `b`, the data layer constructs

```text
D_b = {trip_i : bucket(start_time_i) = b}
```

and invokes the existing static trainer unchanged. Every bucket uses the same
`eta0=0.0002`, `lambda=100000`, relative bounds `[0.1,10]`, 500 updates, and
checkpoint cadence 25. No per-bucket hyperparameter search is allowed. Each
run stores the same schema-2 direct-weight checkpoint used by the full static
model.

The only new active mechanism is an optional schema-3
`data.departure_time_filter`, verified against the train-derived bucket file
and expected train/validation counts before graph mapping. Overall bucketed
metrics are validation-sample-weighted; additive regret totals preserve exact
relative regret.

The former shared-global-plus-residual and trip-average travel-time experiment
is historical only. Its configurations, audit, report, and summary are kept in
[`experiments/archive/full_data_shared_temporal_residual`](experiments/archive/full_data_shared_temporal_residual),
while its special optimizer, checkpoint, evaluator, binaries, and runner have
been removed from the active implementation.

### Formal comparison protocol

The full static reference uses its previously registered update-400
checkpoint, whose configuration exactly matches the active reference JSON.
The five independent models each select maximum decoded validation Edge F1
from updates `0,25,...,500`; maximum Exact Match and then the earlier update
break exact ties. Their five validation partitions are disjoint and sum to the
same 15,812 paths used by the full static evaluator.

This deliberately gives the independent approach five checkpoint-selection
opportunities versus one overall selection for the full static model. No
per-bucket learning-rate or regularization search is added. Raw objectives
across different bucket datasets are not compared.

### Formal results

| Model | Edge Precision | Edge Recall | Edge F1 | Exact Match | Edge Jaccard | Mean regret (mm) |
|---|---:|---:|---:|---:|---:|---:|
| One full static model | **0.713366** | **0.695423** | **0.700554** | **0.388186** | **0.627908** | **303,899.5** |
| Five independent static models | 0.711458 | 0.694377 | 0.699198 | 0.384202 | 0.626007 | 320,424.2 |
| Independent minus full | -0.001908 | -0.001046 | -0.001356 | -0.003984 | -0.001900 | +16,524.8 |

The independent models select updates 475, 275, 200, 275, and 150 for the
five chronological buckets, so none is selected at the 500-update boundary.
Bucketed F1 improves only for 10-16 (+0.001586) and 20-24 (+0.001192); Exact
Match improves only for 20-24 (+0.002082). The sparsest 00-06 bucket loses
0.010841 F1 and 0.021804 Exact Match. Thus simple departure partitioning does
not offset the loss of shared training data on this fixed validation split,
and the full static model remains the recommendation. No formal significance
test was performed.

Tracked evidence:

- [full report](experiments/independent_time_buckets/report.md)
- [machine-readable summary](experiments/independent_time_buckets/summary.json)
- [time audit](experiments/independent_time_buckets/time_audit.json)
- [full static and five bucket configurations](experiments/independent_time_buckets/configs)

Ignored local logs, checkpoints, and per-checkpoint evaluations remain under
`artifacts/independent_time_buckets/`.
