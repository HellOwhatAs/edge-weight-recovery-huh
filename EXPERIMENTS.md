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

## Full-data and departure-time study

The next registered study keeps `edge_transition_arcs` and
`relative_projected_subgradient` fixed. It compares the existing 10% result,
one full-train static model, one full-train coarse departure-time model with
the length anchor, and the same temporal model with a train-only travel-time
anchor. The fixed validation variant is unchanged, and no test file is read.

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
[`time_audit.json`](experiments/full_data_time_conditioning/time_audit.json)
and
[`time_buckets.json`](experiments/full_data_time_conditioning/time_buckets.json).

### Shared temporal model

For bucket `b`, the effective relative vector is

```text
q_b = q_global + residual_b
w_b = w0_b * q_b.
```

The global vector is shared by every bucket. Residuals have a common bounded
box and an `L2` penalty toward zero; their configured box combined with the
global box guarantees effective multipliers remain in `[0.1,10]`. The data
term is the sample-weighted sum of bucket regrets, so this remains a convex
projected-subgradient problem. The graph representation still owns topology,
path mapping, CCH, and decoding only. Departure-time state does not add a
start cost, first-edge parameter, or higher-order graph state.

The full static learning-rate screen used only `0.0002`, `0.0004`, and
`0.0008`. The `0.0008` arm was stopped after its registered update-10
validation objective rose from 636,965.750 to 1,135,726.472. At update 60,
`0.0002` produced F1 0.694756 and Exact Match 0.383569; `0.0004` produced F1
0.689208 and Exact Match 0.371174. Thus `0.0002` was frozen for all formal
runs. Residual step multipliers 1 and 2 were effectively tied at update 40
(F1 0.692307 versus 0.692333), but multiplier 1 had higher Exact Match
(0.383569 versus 0.381229) and a more stable early trajectory. Multiplier 5
was rejected after a bounded update-25 instability check (F1 0.675727).

### Train-only travel-time anchor

For every accepted training trip, the estimator computes full-path length
divided by whole-trip duration. It clips the proxy speed to `[1, 33.333...]`
m/s, affecting 401 low and 307 high observations. The clipped train-wide mean
is 8.768182 m/s. A road-global mean shrinks to that network mean with 42
pseudo-observations, and each road-bucket mean shrinks to its road-global mean
with 74 pseudo-observations; both counts are derived from train support
quantiles. Validation contributes to none of these values.

The resulting baseline is proportional to `length / smoothed_speed`. A common
train-derived scale of 8.768182 is applied before `u32` CCH customization and
recorded with the checkpoint; dividing by it recovers milliseconds. This
positive global scale cannot change a route ordering, but reduces initial
maximum relative fixed-point error to about `8.8e-4`, with no coordinate
quantized to zero. Whole-trip proxy speeds are not claimed to be true per-edge
travel times.

### Formal development result

Each of the three new formal runs used 500 updates with checkpoint cadence 25;
selection used
maximum decoded validation Edge F1, with Exact Match and earlier update only
as exact tie-breaks. All selected checkpoints are inside the budget:

| Stage | Selected update | Precision | Recall | F1 | Exact | Jaccard | Mean regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| Existing 10% static line graph | 299 | 0.707199 | 0.688856 | 0.694125 | 0.377245 | 0.620388 | 321,414.6 mm |
| Full static line graph | 400 | 0.713366 | 0.695423 | 0.700554 | 0.388186 | 0.627908 | 303,899.5 mm |
| Full temporal, length baseline | 400 | 0.715654 | 0.696363 | 0.702280 | 0.388629 | 0.629917 | 295,214.9 mm |
| Full temporal, travel-time baseline | 425 | 0.716090 | 0.697769 | 0.703176 | 0.389704 | 0.631127 | 290,476.5 scaled-ms |

Full data contributes the clear gain: `+0.006429` F1 and `+0.010941` Exact
over the 10% model, with positive F1 changes in all five departure buckets.
Length-based time conditioning adds only `+0.001726` F1 and `+0.000443`
Exact over the full static model, with F1 losses in three buckets. The
travel-time anchor adds another `+0.000896` F1 and `+0.001075` Exact over the
length temporal model, but improves F1 in only two of five buckets. Its
positive global fixed-point scale means its regret units cannot be compared
with the length rows; dividing by 8.7681816625 gives 33,128.5 ms overall.

The selected checkpoints are not budget-boundary artifacts, and decoded route
metrics plateau before update 500. Regularized validation objectives continue
to decrease through update 500, however, so numerical objective convergence
is not confirmed. On this single fixed development split, the temporal
changes are a small mixed-bucket improvement, not evidence of a significant
and stable gain over the full static line graph. No test data was read.

Tracked evidence:

- [full report](experiments/full_data_time_conditioning/report.md)
- [machine-readable summary](experiments/full_data_time_conditioning/summary.json)
- [time and baseline audit](experiments/full_data_time_conditioning/time_audit.json)
- [formal and screening configurations](experiments/full_data_time_conditioning/configs)

Ignored local logs, checkpoints, and per-checkpoint evaluations remain under
`artifacts/full_data_time_conditioning/`.
