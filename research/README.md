# Research workspace

This directory is a Cargo workspace independent of the production workspace.
It owns project experiments, ablations, oracle benchmarks, evaluators, and
adapters for external baselines. An experiment may depend on production crates
with a relative path (for example `../../../crates/ewr-core`), but no crate in
the repository root may depend on anything under `research` or `baselines`.

`ewr-research-protocol` is the only cross-method contract. Version 1 uses:

- a JSON dataset descriptor with schema `ewr.dataset-manifest/v1`, pointing to
  JSONL rows shaped as `{"sample_id":"...","original_edge_ids":[...]}`;
- JSONL predictions shaped only as
  `{"sample_id":"...","predicted_edge_ids":[...]}`; and
- a JSON run receipt with schema `ewr.run-receipt/v1`, which binds the method,
  configuration, source revision, environment, dataset manifest hash, and
  prediction-record schema.

Edge IDs always refer to unmodified original road records. Each sequence is a
complete route; observed dataset rows contain at least two edges, while a
prediction may contain one edge. Method-specific coordinates, truth copies,
metrics, timing, and implementation names do not belong in prediction rows.
The common evaluator aligns rows by `sample_id` and rejects duplicates,
omissions, and extras before computing metrics.

Run this workspace on its own:

```console
cargo test --manifest-path research/Cargo.toml --workspace --locked --all-targets
```

Future Rust experiments should be added as workspace members here. Future
non-Rust baselines remain independent packages under `baselines` and exchange
only these versioned files.

`ewr-research-dijkstra` is the transparent shortest-path baseline adapter. It
implements the production core's oracle port with deterministic binary-heap
Dijkstra over the same quantized line-graph metric. It contains no optimizer,
training loop, dataset I/O, or production feature switch.

`ewr-research-static-baselines` owns the untrained road-length shortest path
and first-order Markov shortest-path controls. Both fix the true first and last
raw roads, preserve directed parallel-road identity, use 16 CPU workers by
default, and publish only protocol-v1 prediction rows. Its README documents the
validation-only smoothing selection and timing boundary.

`ewr-research-evaluator` is the method-neutral quality gate. It strictly reads
the protocol JSONL files, aligns every sample ID, and macro-averages exact
sequence match plus set-based edge precision, recall, F1, and Jaccard. It has
no production or baseline dependency:

```console
cargo run --locked --manifest-path research/Cargo.toml --bin ewr-evaluate -- \
  --dataset-jsonl test.jsonl \
  --predictions-jsonl predictions.jsonl \
  --output summary.json
```

The command always prints the strict `ewr.evaluation-summary/v1` JSON document
to stdout. `--output` additionally replaces that artifact atomically.

## Current route-baseline study

The canonical Beijing comparison is
[`experiments/route_baselines_full_test_20260719/`](experiments/route_baselines_full_test_20260719/README.md).
It evaluates Project, SP-Length, Markov-SP, NeuroMLR-G, DRNCS-LG,
DRPK-static, and DRP-TP on all 248,233 eligible test routes. That directory
contains the frozen protocol, the resumable sequential pipeline, and the
tracked paper-style result.
