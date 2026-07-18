# Edge-weight recovery

This repository learns stable road-transition weights from a directed road
network and historical trajectories. The production path has one algorithm:
directed-line-graph inverse shortest-path fitting with a relative projected
subgradient optimizer and a CCH routing oracle.

The former experiment-driven monolith has been replaced by four production
crates:

| Crate | Responsibility |
| --- | --- |
| `ewr-core` | Typed network/trajectory/model values, line graph, objective, optimizer, backend-neutral oracle port, and the sole resumable trainer |
| `ewr-cch` | RoutingKit CCH preprocessing, customization, and parallel queries; also the three-argument production `fit` facade |
| `ewr-io` | Strict config, direct Shapefile/Pickle adapters, and one versioned training artifact |
| `ewr-cli` | Thin `ewr train` composition root |

Dependencies point inward: production crates never depend on an experiment,
baseline, report, or dataset naming convention. Research code has its own
workspace and lockfile under [`research`](research/README.md); NeuroMLR is an
independent Python package under [`baselines/neuromlr`](baselines/neuromlr/README.md).

## Production CLI

Create a strict schema-v1 config containing only explicit inputs and active
algorithm parameters:

```json
{
  "schema_version": 1,
  "dataset": {
    "nodes": "data/map/nodes.shp",
    "edges": "data/map/edges.shp",
    "trajectories": "data/train_trips.pkl"
  },
  "fit": {
    "eta0": 0.0002,
    "lambda": 100000.0,
    "lower_factor": 0.1,
    "upper_factor": 10.0,
    "updates": 500
  },
  "runtime": { "threads": 8, "checkpoint_every": 25 }
}
```

Trajectory edge IDs are Shapefile record indices. Loading fails unless every
edge record has a unique unsigned `fid` equal to that index; the adapter never
silently remaps IDs.

Run training:

```console
cargo run --release -p ewr-cli --bin ewr -- \
  train --config train.json --output-dir outputs/model
```

The command periodically replaces one `training-artifact.json`. Each
replacement is a single atomic commit containing:

- every weight bound to `(previous_edge, next_edge)`;
- the exact baseline anchor, optimizer identity, and global update clock;
- a versioned identity of the training sufficient statistics and routing
  geometry;
- the versioned CCH semantics needed to make resume safe; and
- the objective evaluated at that snapshot.

`runtime.checkpoint_every` controls the maximum number of successful updates
lost to an interruption. It is an operational CLI concern, not an algorithm
branch.

Continue to a larger `fit.updates` target with:

```console
cargo run --release -p ewr-cli --bin ewr -- \
  train --config train-extended.json --output-dir outputs/model \
  --resume outputs/model/training-artifact.json
```

## Library boundary

Applications that already hold typed values can use the concrete production
facade directly:

```rust,ignore
let result = ewr_cch::fit(&network, &trajectories, &fit_options)?;
for (transition, weight) in result
    .model
    .transitions()
    .iter()
    .zip(result.model.weights())
{
    // transition.previous, transition.next, weight
}
```

`ewr-core` additionally exposes `Trainer`, `TrainingState`, and the narrow
`RoutingOracle` SPI for storage and routing adapters. Objective and optimizer
implementations remain private. File paths and serialization never cross the
core boundary.

## Frozen v1 behavior

- Original road edges are routing nodes; legal consecutive transitions are
  learned coordinates.
- An observed path with `N` edges contributes `N - 1` coordinates.
- A transition starts at the baseline weight of the entered edge. The first
  road edge and all query endpoint offsets cost zero.
- Core rounds positive direct `f64` weights to positive `u32` values for CCH
  route selection, then evaluates the chosen path under the direct `f64`
  vector. This known numerical limitation is intentionally versioned as v1.
- Training uses `q = w / w0`, `eta_k = eta0 / sqrt(k + 1)`, relative
  regularization, multiplicative projection bounds, and one global update
  clock.
- Queries are grouped by unique OD, while counts and costs retain each OD's
  observation multiplicity.
- A resumed run is required to be bitwise equal to uninterrupted training.
  State validation binds the exact baseline, optimizer parameters, routing
  geometry, oracle semantics, aggregate observed counts, and OD
  multiplicities before the first resumed query. Reordering trajectories with
  identical sufficient statistics remains valid; changing any consumed input
  is rejected.

See [`docs/architecture.md`](docs/architecture.md) for the full boundary and
promotion rules.

## Verification

```console
cargo fmt --all --check
cargo test --workspace --locked --all-targets
cargo clippy --workspace --locked --all-targets -- -D warnings
cargo fmt --manifest-path research/Cargo.toml --all --check
cargo test --manifest-path research/Cargo.toml --workspace --locked --all-targets
cargo clippy --manifest-path research/Cargo.toml --workspace --locked --all-targets -- -D warnings
python3 -m unittest discover -s baselines/neuromlr/tests -v
```

Historical reports and configurations are retained under `research/archive`.
The removed monolithic Rust implementation remains recoverable from the Git
revision documented in
[`research/archive/legacy-rust.md`](research/archive/legacy-rust.md).
