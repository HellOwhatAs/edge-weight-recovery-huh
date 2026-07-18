# Production and research architecture

Status: implemented.

## Objective

The production library has one responsibility: accept a typed directed road
network, accepted historical trajectories, and fit options, then return a
transition-weight model whose coordinate identity is explicit. It must not
depend on experiment protocols, dataset naming conventions, benchmark
backends, or baseline implementations.

The repository may continue to contain research code, but dependencies always
point toward the production crates. Production crates never depend on a
research crate, script, manifest, or report.

## Target production crates

- `ewr-core`: road-network and trajectory types, directed-line-graph mapping,
  the inverse-shortest-path objective, relative projected-subgradient state,
  and the sole reusable trainer.
- `ewr-cch`: the RoutingKit CCH implementation of the training and query oracle
  contracts exposed by `ewr-core`.
- `ewr-io`: direct Shapefile and pickle adapters, strict typed configuration,
  and one model/resume artifact. File paths and schema versions stop here; this
  crate does not depend on the CCH adapter.
- `ewr-cli`: composition root for the production `train` command. Comparative
  evaluation belongs to the method-neutral research protocol.

`objective`, `optimizer`, and line-graph implementation details remain modules
inside `ewr-core`; they are not separate crates. Objective and optimizer types
are not part of the public facade. The line-graph/routing values that remain
public form the narrow SPI required to implement and test an oracle adapter.

The production dependency direction is:

```text
ewr-cli ----> ewr-io ----> ewr-core
    |                         ^
    +-------> ewr-cch --------+
```

The file adapter makes the original-edge identity contract explicit: every
edge Shapefile `fid` must be a unique unsigned integer equal to its record
index. Pickle and research-protocol edge IDs therefore cannot silently drift
from the network coordinate order.

## Research boundary

Rust experiments live in a separate `research` Cargo workspace with its own
lockfile. Project ablations and oracle benchmarks may depend on the production
crates by path, but the production workspace excludes the research workspace.

Python baselines are independent packages under `baselines`. In particular,
NeuroMLR owns its Python environment and does not add dependencies to any Rust
crate.

Cross-method experiments communicate through versioned files rather than Rust
library internals:

- a dataset manifest identifies every sample and its complete original-edge
  sequence;
- a prediction file maps the same sample IDs to complete predicted edge
  sequences; and
- a run receipt records the method, method version, configuration, dataset
  identity, source revision, and environment identity.

One method-independent evaluator computes comparison metrics from those files.

## Core API boundary

The concrete high-level API has this shape:

```text
ewr_cch::fit(&RoadNetwork, &[Trajectory], &FitOptions) -> FitResult
```

Internally, `ewr_core::fit` takes one additional `&mut impl RoutingOracle`.
That dependency-inversion point lets research backends reuse the trainer
without adding backend switches or branches to production.

`FitResult` returns a model, not a bare floating-point vector. The model binds
each learned weight to a stable `(previous_edge, next_edge)` transition and a
topology identity. The core also exposes a serialization-independent training
state for resume, while `ewr-io` owns one on-disk artifact that atomically
commits model and state together.

The core quantizes direct weights under the frozen v1 policy and gives one
coarse oracle port the integer metric and ordered routing queries. The oracle
returns stable routing-node paths, coordinate paths, and integer distances.
Core validates their topology, OD endpoints, and distance sums, then aggregates
predicted counts and evaluates paths under the direct `f64` vector. Oracle
customization, search buffers, and backend-specific path objects do not leak
into the optimizer, and alternative backends cannot reimplement training
semantics differently.

Every oracle must declare a versioned semantics identity. A training-state
problem identity covers topology, routing geometry, aggregate observed counts,
and stable OD multiplicities. Resume checks both identities, the exact
baseline, optimizer geometry, bounds, and clock before the first oracle call.
Thus two raw trajectory orders with identical consumed statistics are
equivalent, while changed data, geometry, or tie/path semantics are rejected.

The reusable trainer may expose immutable snapshots at a caller-selected
cadence, but it retains the sole update loop. The production CLI writes those
snapshots as one `ewr.training-artifact/v1` file, so model and resume state can
never be published from different runs.

## Frozen v1 semantics

The structural refactor must preserve the following behavior before any model
change is considered:

1. The active representation is the directed line graph. Original road edges
   are routing nodes, legal consecutive edge transitions are routing arcs and
   learned coordinates, and an observed path with `N` edges has `N - 1`
   coordinates.
2. A transition coordinate is initialized from the baseline weight of the
   entered edge. There is no learned first-edge coordinate or source offset.
3. Node-to-node queries use every original edge leaving the source as a
   zero-offset source state and every original edge entering the target as a
   zero-offset target state.
4. Direct `f64` weights are rounded to positive `u32` weights for v1 route
   selection. The selected coordinate path is evaluated with the direct `f64`
   vector. This known numerical limitation is preserved during the structural
   migration and addressed only by a separately versioned model change.
5. Training uses relative coordinates `q = w / w0`, the schedule
   `eta_k = eta0 / sqrt(k + 1)`, relative regularization, multiplicative bounds,
   and one global `completed_updates` clock.
6. Observed counts use every accepted trajectory. Predicted paths are queried
   once per unique OD and weighted by that OD's sample count.
7. Resume restores the direct vector and global clock only after verifying the
   exact optimizer/baseline and the versioned problem/oracle identities; it
   must then produce the same final state as uninterrupted training.

Golden tests, not experiment reports, enforce these invariants.

## Promotion rule

New representations, temporal models, optimizers, baseline-specific adapters,
and diagnostic oracles begin in the research workspace. They are promoted into
production only after they become an accepted product capability. Experiments
must not enter production through feature flags or additional branches in the
active trainer.

## Completed migration

The workspace replacement passed the frozen v1 gates before the monolithic
package was removed. Historical Rust source remains recoverable from the Git
revision recorded in `research/archive/legacy-rust.md`; historical experiment
evidence is retained under `research/archive/experiments`. Active research
backends consume the public oracle API, and external baselines exchange only
the versioned dataset, prediction, and run-receipt files.
