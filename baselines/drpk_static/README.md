# DRPK-static protocol adapter

This directory is a clean-room implementation of DRPK for the EWR version-one
raw-road protocol. It imports no code from the upstream repository. The design
was audited against the official MIT-licensed repository at commit
`2f65eaeb784d7266b591196c795d72cf909294d8`; machine-readable provenance and
audited file hashes are in `upstream.json`.

This baseline must be reported as **DRPK-static**, not as an unqualified DRPK
reproduction. Full DRPK needs departure time, per-road entry times, typical
travel time, and within-road origin/destination offsets. The current EWR v1
dataset intentionally has no such fields. DRPK-static makes one declared,
testable adaptation:

- count road occurrences using the training split only;
- min-max normalize that one global popularity vector to `[0, 1]`, as specified
  by the paper, and replicate the normalized vector into 48 identical
  weekday/weekend-hour slots;
- fix the origin offset to `0.0` and destination offset to `1.0`; and
- omit query time entirely, so changing a timestamp outside this protocol could
  not change a prediction.

The package also exposes **DRP-TP**, DRPK's native non-learned sanity baseline:
`key_num=0` and the upstream `wotp` tie behavior. It needs preprocessing but no
checkpoint.

## Preserved algorithm boundary

The adapter uses original shapefile record IDs as directed road identities,
including distinct parallel roads. It constructs the directed raw-road line
graph, a sparse directed-association matrix from every ordered road pair in
each training trajectory, the top-100 KSD candidate pool, and a 64-dimensional
Node2Vec initialization. With the formal `p=q=1` setting, walks are uniform
topological walks on that conjugate graph: this matches the official
`RandomWalker._simulate_walks` path into `deepwalk_walk`. Training transition
counts never weight a Node2Vec arc; the formal preprocessor does not collect
this unused statistic (synthetic audit helpers can still compute it).
Candidate labels are the highest-ranked pool members
that are interior roads of the observed training or validation trajectory, up
to `ceil(0.2 * route_length)`.

KSD follows paper Eq. (6)--(9): separate trainable, Node2Vec-initialized
64-dimensional source and destination embeddings plus two offset ratios form a
130-dimensional query in the exact order `[e_s, r_s, e_d, r_d]`; its MLP is
`130 -> 2048 -> 256`. Each candidate has a
separate trainable 64-dimensional segment-ID embedding concatenated with its
one-dimensional static popularity; its MLP is `65 -> 512 -> 256`. The logit is
the dot product of the query and candidate representations. Both MLPs are
Linear-ReLU-Linear exactly as printed in Eq. (7) and (9), with no BatchNorm or
dropout in the formal configuration. KSD has no
direction-cosine feature. Positive BCE weights follow the paper as
`exp(normalized_importance)`, where positive association strengths are
normalized to sum to one; a non-positive has weight one. Eq. (3) sums these
weighted candidate losses within each route and Eq. (4) averages route losses;
the implementation does not divide by the number of candidate entries.
Validation checkpoint
selection uses native top-1 key-segment accuracy, then weighted BCE, then the
earlier epoch. The test descriptor is not accepted by `preprocess` or `train`.

### Paper versus official release

The published formula and the pinned official release conflict. The paper uses
the candidate-ID embedding/candidate MLP/dot-product architecture above and
reports 300 training epochs. In contrast, release `conf.py` and
`train_keyseg.py` default to 1000 epochs, while release `models/key_segs.py`
does not implement the paper's candidate embedding or candidate MLP: it indexes
rows of a road-wide linear classifier, projects direction cosine plus traffic
popularity to four features, concatenates those features to each classifier
row, and adds a per-road bias. The release trainer also uses base-ten positive
weights. These alternatives cannot both be the same KSD model. This study uses
the published Eq. (6)--(9), exponential weighting, and 300-epoch experimental
setting as the primary specification; the release remains the source for
unambiguous preprocessing, endpoint, candidate-pool, and planning behavior.

Planning starts from the observed first raw edge and targets the observed last
raw edge, exactly as the official inference code does. Each DRP leg is capped
at 300 roads. A unique maximum destination-association successor is selected
directly. For a positive-association tie, DRPK-static uses global popularity
and then direction cosine; a zero-association tie uses direction cosine. Once
the one-use filter is exhausted, all outgoing roads become eligible. Dead ends,
cycles that hit the cap, and a failed first key leg return the partial route.
The adapter never appends or otherwise repairs the true destination.

Two implementation hygiene changes are explicit: padding has a real `-1` mask
instead of colliding with raw road ID zero, and randomness is seeded and made
independent of worker completion order. These changes do not add test
information or change any non-padding model feature.

## Environment

The pinned environment is Python 3.12, Gensim 4.4.0, NumPy 2.2.6, SciPy
1.15.3, PyShp 2.3.1, and PyTorch 2.7.1. Install the correct PyTorch build for
the machine first when CUDA is used, then install the local package:

```console
cd baselines/drpk_static
python3.12 -m venv .venv
.venv/bin/python -m pip install \
  torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -e '.[test]'
```

All commands default to seed `20260718` and 16 CPU threads/workers. Use
`--device cpu` for a CPU-only run, `--device cuda:N` for a specific GPU, or
`--device auto` for the only mode allowed to choose CUDA/CPU automatically. An
explicit CUDA request fails if that device is unavailable; it never silently
changes the hardware efficiency condition. The resolved device is recorded in
the run receipt.
`DRP-TP` is always a NumPy/standard-library CPU planner: its prediction path
loads only configuration, graph, and DA artifacts, does not import or initialize
PyTorch/Gensim, ignores `--device`, and records `cpu` as the resolved device.

## Inputs

`--map-dir` contains `nodes.shp` and `edges.shp`. The edge `fid`, when present,
must equal the unmodified shapefile record order. Dataset manifests conform to
`ewr.dataset-manifest/v1` and contain exactly:

```json
{
  "schema": "ewr.dataset-manifest/v1",
  "dataset_id": "beijing/train",
  "network_id": "beijing-roads-v1",
  "records_schema": "ewr.dataset-record/v1",
  "records_file": "train.jsonl"
}
```

Every referenced JSONL row is exactly
`{"sample_id":"...","original_edge_ids":[...]}`. Routes must be continuous
and contain at least two raw edges. Training and validation IDs must be
disjoint. Use the already frozen NeuroMLR-provided split manifests; do not
resample them for this baseline.

## Preprocess

```console
.venv/bin/ewr-drpk-static preprocess \
  --map-dir /path/to/data/beijing_data/map \
  --train-manifest /path/to/train.manifest.json \
  --validation-manifest /path/to/validation.manifest.json \
  --output-dir /path/to/run/drpk-static/preprocess \
  --source-revision COMMIT-PLUS-WORKTREE-LABEL
```

This writes the graph, sparse DA matrix, replicated static popularity table,
Node2Vec embedding, KSD train/validation candidates, configuration, and
component timing diagnostics. Every routing/model artifact is size- and
SHA-256-bound by a manifest. `routing-configuration.json` and
`routing-artifacts.json` are complete before the DRP-TP timing snapshot; the
later full `configuration.json` binds that routing configuration and all
KSD-only artifacts. When reporting end-to-end training cost, add
preprocessing time to KSD optimization time; both boundaries remain separate
so the paper can also report them individually.

DRP-TP (`wotp`) does not use popularity. It becomes runnable immediately after
graph loading, split validation, and DA construction. Diagnostics snapshot time
and peak RSS at exactly that boundary, before popularity counting/normalization
and the KSD-only candidate tables and Node2Vec.
The shared artifact's full preprocessing total therefore must not be charged to
DRP-TP; use `timing.drp_tp_ready_seconds` and
`timing.drp_tp_ready_peak_rss_kib` for that sanity baseline.

DA construction never allocates an edge-count-squared matrix. It sorts bounded
one-million-pair runs on disk, externally merges them into memory-mapped CSR
and CSC views, and deletes the temporary runs after a successful merge. KSD
candidate artifacts are preallocated as compact `int32` IDs, `uint8` labels,
and `float32` weights in individual `.npy` memory maps. Training opens them in
copy-on-write mmap mode and only materializes the current shuffled microbatch; it
never copies the formal 605k-by-100 table into a Python list or a second dense
array. The
formal Node2Vec engine streams deterministic walks into Gensim's multicore C
Word2Vec implementation; it does not materialize 1.8 million Python walk
lists. `--node2vec-engine torch` is deliberately limited to tiny smoke graphs.

## Train

```console
.venv/bin/ewr-drpk-static train \
  --preprocess-dir /path/to/run/drpk-static/preprocess \
  --validation-manifest /path/to/validation.manifest.json \
  --output-dir /path/to/run/drpk-static/train \
  --source-revision COMMIT-PLUS-WORKTREE-LABEL \
  --device cuda:0
```

The formal paper schedule is exactly 300 epochs, optimizer batch size 8192,
Adam at `1e-3`, and ReduceLROnPlateau. Gradients are accumulated in
memory-bounded 512-route microbatches and the optimizer steps once per complete
8192-route macro batch (or its final remainder), so this changes memory use but
not the Eq. (4) update. Learning-rate early stopping is disabled by
default; `--early-stop-learning-rate POSITIVE_VALUE` is an explicit non-formal
option. Use
`checkpoint-best-accuracy.pt` for formal prediction. Every checkpoint binds
the network, raw graph identity, exact preprocessing configuration hash,
adapter SHA-256, and source-revision label. Loading fails if the adapter source
has changed since preprocessing.

## Predict

```console
.venv/bin/ewr-drpk-static predict \
  --preprocess-dir /path/to/run/drpk-static/preprocess \
  --checkpoint /path/to/run/drpk-static/train/checkpoint-best-accuracy.pt \
  --dataset-manifest /path/to/test.manifest.json \
  --method drpk_static \
  --predictions /path/to/run/drpk-static/test.predictions.jsonl \
  --run-receipt /path/to/run/drpk-static/test.run.json \
  --diagnostics /path/to/run/drpk-static/test.diagnostics.json \
  --source-revision COMMIT-OF-THIS-ADAPTER \
  --device cuda:0 \
  --warmup-repetitions 1 --measured-repetitions 5
```

For the sanity baseline, omit `--checkpoint` and use `--method drp_tp`.
Prediction defaults to zero warm-ups and one measured full-dataset repetition;
set `--warmup-repetitions` and `--measured-repetitions` explicitly for the
formal efficiency table. Every repetition reruns candidate construction, key
scoring, and route planning. Diagnostics report each full duration and the
candidate-pool, KSD, and planner component totals. Repetitions must produce
identical complete routes.

Predictions contain only
`{"sample_id":"...","predicted_edge_ids":[...]}`. They contain no truth,
metric, or timing. Accuracy is computed by the common evaluator, while
method-specific resource and timing measurements remain in diagnostics.

### Memory-bounded full-test quality prediction

The normal adapter reads every test trip into Python lists. DRPK-static also
constructs candidate pools for the whole supplied batch and retains every
planned route until output, so the complete test manifest must use the
crash-safe sharded runner. It verifies the configuration and every core/routing
artifact against their manifest, then binds those hashes together with the
checkpoint, adapter source/executable, dataset, and query settings.

DRPK-static uses CUDA when available:

```console
python3 research/scripts/run_sharded_route_predictions.py \
  --method drpk_static \
  --dataset-manifest /path/to/full-test.manifest.json \
  --output-dir /path/to/full-test/drpk-static \
  --adapter-executable baselines/neuromlr/.venv/bin/ewr-drpk-static \
  --adapter-source baselines/drpk_static/drpk_static_adapter.py \
  --preprocess-dir /path/to/run/drpk-static/preprocess \
  --checkpoint /path/to/run/drpk-static/train/checkpoint-best-accuracy.pt \
  --source-revision "$(git rev-parse HEAD)" \
  --device cuda:0 --cuda-visible-devices 0 \
  --workers 16 --inference-batch-size 32 --shard-size 4096
```

DRP-TP uses the same routing artifacts but stays on CPU and has no checkpoint:

```console
python3 research/scripts/run_sharded_route_predictions.py \
  --method drp_tp \
  --dataset-manifest /path/to/full-test.manifest.json \
  --output-dir /path/to/full-test/drp-tp \
  --adapter-executable baselines/neuromlr/.venv/bin/ewr-drpk-static \
  --adapter-source baselines/drpk_static/drpk_static_adapter.py \
  --preprocess-dir /path/to/run/drpk-static/preprocess \
  --source-revision "$(git rev-parse HEAD)" \
  --device cpu --workers 16 --shard-size 4096
```

For both methods, first add `--shard-limit 1`, inspect the committed shard, and
then rerun without the limit. `progress.json` shows committed sample/shard
counts and ETA. SIGINT/SIGTERM discards only the uncommitted current shard; the
same command resumes after revalidating every marker and output hash. A changed
dataset, artifact, source, checkpoint, or setting is rejected rather than
silently mixed with old results.
Shard size defaults to 4096 and is hard-limited to 8192, preventing an
accidental full-split single batch from restoring the original memory risk.

The joined diagnostics set `efficiency_comparable=false`, because every shard
reloads the immutable DA/model artifacts. Its internal adapter timing is a
decomposition only. The paper's primary operational inference cost comes from
GNU time around the complete full-test task, including materialization,
per-shard reloads, prediction, validation, and final assembly.

## Tests

Protocol and planner tests require only Python's standard library; model tests
skip explicitly when the scientific dependencies are absent:

```console
python3 -m unittest discover -s baselines/drpk_static/tests -v
```

With the pinned environment installed:

```console
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pytest
```

The suite uses only synthetic small graphs. It does not download upstream code,
read the formal test split, or run full-data preprocessing/training.

For an end-to-end six-road CLI smoke test, first create the disposable fixture
and then use deliberately tiny model settings:

```console
SMOKE_DIR=/tmp/ewr-drpk-static-smoke
.venv/bin/python tests/make_cli_fixture.py "$SMOKE_DIR"
.venv/bin/ewr-drpk-static preprocess \
  --map-dir "$SMOKE_DIR/map" \
  --train-manifest "$SMOKE_DIR/train.manifest.json" \
  --validation-manifest "$SMOKE_DIR/validation.manifest.json" \
  --output-dir "$SMOKE_DIR/preprocess" \
  --source-revision smoke \
  --candidate-pool-size 4 --node2vec-dim 4 \
  --walk-length 4 --walks-per-edge 2 --node2vec-epochs 1 \
  --node2vec-negative-samples 2 --node2vec-batch-size 32 \
  --node2vec-engine torch
.venv/bin/ewr-drpk-static train \
  --preprocess-dir "$SMOKE_DIR/preprocess" \
  --validation-manifest "$SMOKE_DIR/validation.manifest.json" \
  --output-dir "$SMOKE_DIR/train" --source-revision smoke \
  --device cpu --epochs 2 --batch-size 2 \
  --query-hidden-size 8 --representation-size 6 \
  --candidate-embedding-size 3 --candidate-hidden-size 7
.venv/bin/ewr-drpk-static predict \
  --preprocess-dir "$SMOKE_DIR/preprocess" \
  --checkpoint "$SMOKE_DIR/train/checkpoint-best-accuracy.pt" \
  --dataset-manifest "$SMOKE_DIR/test.manifest.json" --method drpk_static \
  --predictions "$SMOKE_DIR/predictions.jsonl" \
  --run-receipt "$SMOKE_DIR/run.json" \
  --diagnostics "$SMOKE_DIR/predict-diagnostics.json" \
  --source-revision smoke --device cpu
```
