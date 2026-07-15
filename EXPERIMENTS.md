# Experiments

## Scope of this revision

The current work validates an architectural invariant:

```text
graph order affects only graph construction and trajectory mapping;
inverse optimization learns one direct-weight vector on the supplied graph.
```

No comparative performance conclusion is carried forward. Historical
large-run configurations and generated summaries are intentionally absent
from the active tree; Git history remains the recovery authority.

## Mathematical contract

For mapped trajectories `P_i`, initial weights `w0`, and `m` graph
coordinates, both graph orders use

```text
J(w) = (1/N) sum_i [cost_w(P_i) - dist_w(s_i,t_i)]
       + lambda / (2m) * ||w - w0||^2,

g = (observed_counts - predicted_counts) / N
    + lambda / m * (w - w0),

eta_k = eta0 / sqrt(k + 1),
w <- project(w - eta_k * g).
```

The graph representation supplies the topology, mapped observations, initial
weights, bounds, coordinate counts, topology identity, route decoder, and the
metric conversion used by its oracle. The trainer receives only this graph
problem contract.

The active RoutingKit binding accepts integer CCH weights. Production queries
therefore select routes after rounding the direct `f64` vector, while reported
direct path costs use that vector. Strict continuous-weight oracle equivalence
is deferred; the synthetic CCH/reference gate checks the integer metric that
the CCH actually receives.

## Synthetic correctness gates

The correctness suite must establish:

1. identity mapping for first-order road trajectories;
2. overlapping-pair mapping and decoding for second-order trajectories;
3. use of the same direct-weight optimizer by both graph orders;
4. the update formula, regularization, projection, and global clock;
5. shortest-path cost agreement between CCH and reference Dijkstra on small
   graphs; and
6. identical final state for uninterrupted training and checkpoint resume.

## Beijing 1% technical smokes

Only these active experiment configurations belong to this revision:

| Configuration | Graph order | Updates | Validation cadence | Threads |
|---|---:|---:|---:|---:|
| `experiments/configs/first_order_smoke_1pct.json` | first | 3 | 3 | 4 |
| `experiments/configs/second_order_smoke_1pct.json` | second | 3 | 3 | 4 |

Both use the deterministic Beijing `scale_1pct_seed42` training subset and
the fixed `scale_fixed_seed20260715` validation subset. Both set
`eta0=1000.0`, `lambda=0.001`, and coordinate bounds to `[0.1*w0, 10*w0]`.
Validation runs at the initial and final states only. The test split is never
read.

A smoke is healthy only if all of the following hold:

- every reported objective is finite;
- at least one direct weight changes from initialization;
- shortest-path customization and queries complete normally; and
- a saved checkpoint restores and can continue on the original update clock.

Both release smokes completed successfully on 2026-07-16:

| Graph order | Routing topology | Train objective, state 0 -> 3 | Changed direct coordinates | Wall time |
|---|---:|---:|---:|---:|
| first | 31,199 nodes / 72,156 arcs | 595,722.8061 -> 595,624.2362 | 25,825 / 72,156 | 1.57 s |
| second | 188,249 nodes / 511,079 arcs | 586,169.2582 -> 586,078.5724 | 33,020 / 188,249 | 11.26 s |

Final validation objectives were finite (`650361.2046` and `636831.1501`,
respectively), all 6,132 train and 15,730 validation unique-OD queries
completed, and both final checkpoints passed the trainer's strict reload.
Fresh runs from each `checkpoint-0.json` reproduced the corresponding final
checkpoint exactly, including all direct-weight values and
`completed_updates=3`. These are independent health checks, not comparable
endpoint-quality measurements. No test data was read.

## Explicitly out of scope

This revision does not authorize larger-subset or full-data training,
learning-rate searches, formal first-versus-second-order comparison, or test
split evaluation. Smoke values must not be interpreted as endpoint quality.
