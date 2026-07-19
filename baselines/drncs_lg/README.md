# DRNCS-LG baseline adapter

This directory contains a clean-room implementation of DRNCS adapted to the
project's raw-road protocol. It is intentionally isolated from both the
production Rust workspace and the released DRNCS scripts. The implementation
was audited against upstream commit
`8847482eb507785ee5b4e145f1a5144d1737fbe0` and the ECML-PKDD 2025 preprint.
It does not claim to be the authors' official implementation.

## Upstream audit boundary

The pinned public repository cannot be executed as a like-for-like reproduction
under this study's protocol. Its `train_hierarchy.py` optimization loop
constructs batches from `test_data`, so this adapter does not reuse that
training path. The released evaluation also uses a 50 metre endpoint tolerance
and truth-destination-aware failure truncation, whereas this study requires
exact raw-edge endpoints and retains failed generated prefixes.

There are additional paper/code mismatches: the paper reports 200 epochs and a
0.5 contraction ratio, while released scripts hard-code 50 epochs and conflicting
0.1/0.3 ratios; the stated Python 3.6 plus PyTorch 2.4.1 environment is not a
supported combination.  `upstream.json` records the audited files, line-level
findings, and the research-integrity boundary.

`upstream.json` records the concrete code locations and paper/release
differences. DRNCS-LG follows the paper where the specification is explicit,
uses train/validation-only model selection, and reports a fresh result under
the common strict protocol rather than importing a published number.

## Why this is named DRNCS-LG

The paper and released scripts use intersections as prediction states, while
NeuroMLR and this project's version-1 protocol use unmodified directed raw
road IDs. Collapsing each road to `(u, v)` would lose parallel-road identity
and make exact route evaluation ill-defined. DRNCS-LG therefore runs the DRNCS
algorithm on the directed line graph:

- one state is one raw road record;
- `e_i -> e_j` exists exactly when the head of `e_i` equals the tail of `e_j`;
- a query is the observed first raw edge and observed last raw edge; and
- an output is the model-generated raw-edge sequence.

This preserves parallel edges and matches the query convention already used
by the project's NeuroMLR reproduction. The query endpoints are inputs, not an
evaluation repair: prediction never reads the truth interior, appends the true
destination, truncates to the closest truth point, or replaces a failed route
with the observed route.

## Reproducibility choices

Several paper/repository ambiguities need explicit choices. They are fixed here
instead of being silently tuned:

- Static inputs are used. Although the paper's prose mentions query time, the
  released transition model consumes only current, destination, and candidate
  Node2Vec embeddings and has no time feature.
- Node2Vec uses dimension 64, directed walks of length 30, 200 walks per state,
  window 10, skip-gram, `p=q=1`, five Word2Vec epochs, and the release's explicit
  `batch_words=4`. The embeddings remain fixed during MLP training. Walks are
  exposed through a re-iterable stream rather than the release library's
  materialized Python list; this preserves the formal corpus and parameters
  without attempting to retain 14.4 million Beijing walks in RAM.
- The transition MLP has two 128-unit hidden affine layers and a scalar output,
  with ReLU after every affine layer including the last, matching the release.
- Categorical masking uses `-inf` for padded candidates. The release multiplies
  padded logits by zero, which can make padding beat a negative valid logit;
  that is treated as an implementation defect rather than reproduced.
- Contraction defaults to ratio 0.5 and uses shortcut-edge differential:
  `new shortcuts - active indegree - active outdegree`. Ties use ascending raw
  edge ID. This follows the paper definition; released preprocessing files use
  mutually inconsistent criteria and ratios.
- SC1 candidates are collected from the training split only. A deterministic
  paper-Eq.-(11) average pairwise-precision medoid is stored. Historical path
  multiplicity is preserved exactly. Unique path counts are aggregated in a
  temporary SQLite database, and the medoid score is algebraically reduced to
  per-edge historical mass, avoiding both an all-observation Python object graph
  and quadratic pair materialization. SC2 fills remaining final shortcuts using
  original-model negative-log-probability cost and reverse Dijkstra grouped by
  destination.
- The original and sparse MLPs are both trained. Checkpoints are chosen only by
  validation macro edge F1, then exact match, then earliest epoch. The test
  manifest is not accepted by either preprocessing or training. The paper's
  comparative setting of 200 epochs and batch size 512 is the formal default;
  the released scripts' hard-coded 50 epochs are not used for paper runs.
- When both query endpoints survive contraction, sparse inference and shortcut
  expansion are used. Otherwise inference falls back to the original model.
  A model dead end, repeated state, or step limit returns its generated prefix;
  there is no endpoint repair or truth-aware fallback.
- Validation and test generation use a 300-transition cap. This is the
  `MAX_ITERS = 300` value in the pinned upstream `valid_hierarchy.py` and the
  common NeuroMLR study cap. Formal commands pass `--max-steps 300`
  explicitly so checkpoint selection and frozen-test decoding use the same
  stopping rule.
- Training uses 512 trajectories per batch, matching the paper and release.
  All transitions from those routes are flattened for the categorical loss;
  only each transition's actual outgoing candidates are padded within the
  batch, so the release's incorrect zero-valued padding logits are avoided.
  To bound memory without changing optimizer-step boundaries, transition loss
  is accumulated in chunks of 8,192 and divided by the number of routes in the
  complete batch before the single Adam update. This is the paper's Eq. (10):
  the summed transition NLL is averaged over routes. The released script instead
  obtains a transition mean from PyTorch's default cross-entropy reduction; that
  release/paper discrepancy is recorded in every formal training artifact.

Training JSONL is parsed line by line into contiguous uint32 edge IDs plus
uint64 route offsets. Sparse training routes are generated lazily from compact
base-route indices, rather than copied into a second nested Python list. The
training split is deliberately loaded only after Node2Vec and contraction, and
pickle artifacts are streamed directly to their atomic temporary files.
Preprocessing imports only NumPy, PyShp, SciPy/Gensim, and never imports or
probes PyTorch/CUDA; PyTorch is loaded only by training and prediction. These
are storage changes only: they do not reduce the corpus, epochs, routes, batch
size, model dimensions, or worker limit.

The pinned default seed is `20260716`. CPU pools are capped at 16 workers, as
requested. On the 72,156-state Beijing line graph, the formal Node2Vec maximum
per corpus traversal is `72,156 * 200 * 30 = 432,936,000` walk tokens (dead ends
can shorten it). Gensim first traverses the corpus once to build vocabulary and
then traverses it for five training epochs: at most 432.9 million tokens for
the vocabulary scan plus 2.165 billion training-token visits, or 2.598 billion
total visits. Thus "432.9 million" is not the total five-epoch workload.
Gensim's multi-worker Word2Vec is repeatable at the experimental
seed/configuration level but is not guaranteed bit-for-bit identical across
operating systems or thread schedules; use `--workers 1` for bitwise-focused
audits and report that deviation.

## Environment

Use an isolated Python 3.12 environment. Dependencies are exactly pinned in
`pyproject.toml`.

```console
cd baselines/drncs_lg
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest
```

## Three-stage protocol

Preprocessing has no validation or test argument. It requires a dataset ID whose
split role is `train`, validates the separately pinned manifest and records
hashes and full raw
edge continuity of the training records, builds the line graph, trains
Node2Vec, contracts the graph, and constructs train-only SC1 storage.

```console
.venv/bin/ewr-drncs-lg preprocess \
  --train-manifest /path/to/manifests/train.manifest.json \
  --map-dir /path/to/data/beijing_data/map \
  --output-dir /path/to/run/preprocess \
  --source-revision "$(git rev-parse HEAD)" \
  --expected-train-manifest-sha256 805d587fbf38613df77915aa06680dd3100daf75038e4c754c3ce52abb9c28f4 \
  --expected-train-records-sha256 244496f31e906ebbde8d60c8cac34bb018230410e6d3c718d117a1c20cf752df \
  --workers 16
```

Training rereads and hash-checks that exact training manifest, requires the
second dataset ID to have role `validation`, rejects sample overlap, accepts no
test argument, builds SC2 with the selected original model, and trains/selects
the sparse model. Formal runs use the separate resumable runner below. Keeping
it in a separate source file preserves the adapter SHA-256 already bound into
completed preprocessing artifacts. It writes the same version-2
`checkpoint.pt` and `training_diagnostics.json` schemas consumed by prediction
and result summarization.

```console
.venv/bin/ewr-drncs-lg-resumable \
  --preprocess-dir /path/to/run/preprocess \
  --train-manifest /path/to/manifests/train.manifest.json \
  --validation-manifest /path/to/manifests/validation.manifest.json \
  --output-dir /path/to/run/train \
  --source-revision "$(git rev-parse HEAD)" \
  --expected-train-manifest-sha256 805d587fbf38613df77915aa06680dd3100daf75038e4c754c3ce52abb9c28f4 \
  --expected-train-records-sha256 244496f31e906ebbde8d60c8cac34bb018230410e6d3c718d117a1c20cf752df \
  --expected-validation-manifest-sha256 0307f86e2f11db981b85876c424e344d1cb17440fbf0fb9f55411e60a50365c9 \
  --expected-validation-records-sha256 186fc142edd1cf1eb32d5cfb1542467ccfbc3a7d4987dcfe286167ea25a24596 \
  --device cuda \
  --workers 16 \
  --max-steps 300 \
  --resume auto \
  --progress /path/to/run/train/progress.json
```

The runner creates these atomic recovery files:

- `resume/original.pt`: current model and Adam state, selected best state/rank,
  complete history, and retained compute time after every original-model epoch;
- `resume/sc2.pt`: the completed destination prefix and partial SC2 database,
  saved every 25 destinations by default and immediately at a safe signal
  boundary;
- `resume/sparse.pt`: the corresponding per-epoch sparse-model state;
- `progress.json`: an atomic, human-readable heartbeat (rate-limited to at
  most one ordinary SC2 write every two seconds) with current stage,
  completed/total units, stage percentage and ETA, current-process wall time,
  cumulative process wall time, retained compute time, device, and the nearest
  recovery checkpoint hash. `completed_units` is live progress, while
  `recoverable_completed_units` and `maximum_redo_units` make any gap to the
  most recent SC2 checkpoint explicit; and
- `resume_provenance.json`: runner/adapter hashes, resume binding, input and
  configuration bindings, hashes of all stage/final artifacts, and a split of
  final-invocation, prior-invocation, retained-compute, and active-overhead time.

`SIGINT` and `SIGTERM` finish or discard only the current incomplete unit, write
an `interrupted` heartbeat, and exit nonzero. Repeating the identical command
with `--resume auto` restores the latest complete epoch or SC2 destination.
`--resume require` refuses to start without existing state; `--resume never`
refuses to overwrite any existing training artifact. A resume binding covers
the runner hash, unchanged adapter hash, preprocessing artifact hash, both
dataset hashes, device, and every model/training setting, so a changed formal
configuration cannot silently reuse incompatible state. Monitor a detached run
without attaching a debugger:

```console
watch -n 5 'jq . /path/to/run/train/progress.json'
```

The `total_process_seconds` in final diagnostics is cumulative end-to-end
active process wall time across invocations through the final checkpoint. It
includes loading, validation, checkpoint I/O, orchestration overhead, and work
discarded by a cleanly interrupted incomplete unit, but excludes downtime.
Original/SC2/sparse `wall_seconds` remain the sum of successfully retained
complete units. The progress heartbeat and provenance sidecar expose both
boundaries. Formal GPU scripts should additionally bind the GPU explicitly
with `CUDA_VISIBLE_DEVICES=0`; using `--device cuda` fails fast if CUDA is
unavailable instead of silently falling back to CPU.

Prediction may target validation or the frozen test manifest. It checks the
checkpoint, map identity, network ID, route continuity, and produces strict
version-1 predictions plus a run receipt and component diagnostics.

```console
.venv/bin/ewr-drncs-lg predict \
  --checkpoint /path/to/run/train/checkpoint.pt \
  --map-dir /path/to/data/beijing_data/map \
  --dataset-manifest /path/to/manifests/test.manifest.json \
  --predictions /path/to/run/test/predictions.jsonl \
  --run-receipt /path/to/run/test/run.json \
  --diagnostics /path/to/run/test/diagnostics.json \
  --source-revision "$(git rev-parse HEAD)" \
  --expected-dataset-manifest-sha256 2e7c6c390bedd279ff302d36a2b4658a3c0a385d1a16402e6deb7727254c4ece \
  --expected-dataset-records-sha256 60d3fe13b74b9c7ad63eb1613580e2bdf2e0e1dc22317861388d63a96b2300e4 \
  --device cuda \
  --workers 16 \
  --max-steps 300 \
  --warmup-repetitions 1 \
  --measured-repetitions 5 \
  --latency-samples 100
```

The common evaluator remains the sole source of paper quality metrics. Version-2
adapter artifacts bind the exact adapter SHA-256 and source revision. Prediction
loads full protocol records for evaluator compatibility, but route generation
uses only their first and last raw-edge IDs; diagnostics therefore report
`truth_interior_used_for_route_generation=false` rather than claiming that the
records were never read. The
prediction diagnostics separately report end-to-end throughput, optional
batch-size-one latency percentiles, original/sparse rollout counts, shortcut
expansion time, endpoint failures, continuity, non-simple generated routes,
peak RSS, and peak CUDA allocation. Preprocessing, training,
SC2 generation, and inference are timed separately, so preprocessing cost is
not hidden inside query latency.

### Memory-bounded full-test quality prediction

Do not pass the complete test manifest directly to `ewr-drncs-lg predict`.
Although protocol records use compact numeric storage, the decoder still holds
all generated Python route lists until it writes the final JSONL; a large number
of 300-step failures can therefore make full-test memory scale with the entire
split.  Use the independent sharded runner instead.  It launches a fresh
adapter process per shard, atomically commits its predictions, and binds the
exact full manifest/records, map files, checkpoint, adapter executable/source,
and decoding settings:

```console
python3 research/scripts/run_sharded_route_predictions.py \
  --method drncs_lg \
  --dataset-manifest /path/to/full-test.manifest.json \
  --output-dir /path/to/full-test/drncs-lg \
  --adapter-executable baselines/neuromlr/.venv/bin/ewr-drncs-lg \
  --adapter-source baselines/drncs_lg/drncs_lg_adapter.py \
  --checkpoint /path/to/run/drncs-lg/train/checkpoint.pt \
  --map-dir data/beijing_data/map \
  --source-revision "$(git rev-parse HEAD)" \
  --device cuda --cuda-visible-devices 0 \
  --workers 16 --inference-batch-size 1000 --max-steps 300 \
  --shard-size 4096
```

Run once with `--shard-limit 1` as a formal-input smoke test, then repeat the
same command without that option.  A normal rerun verifies and skips committed
shards. `progress.json` reports committed samples/shards and a shard-process
ETA; `predictions.jsonl`, `diagnostics.json`, `run-receipt.json`, and
`complete.json` appear after all shards validate. SIGINT/SIGTERM stops the
current child and leaves the last completed shard as the recovery boundary.
The default shard size is 4096 and the runner rejects values above 8192, so an
accidental full-split single batch cannot bypass the memory boundary.
Changing any bound dataset, artifact, source, or decoding setting requires a
new output directory.

The aggregate diagnostics are explicitly marked
`efficiency_comparable=false`: repeated model/map loads are a recovery cost;
the paper uses GNU time around the complete full-test task as the uniform
operational boundary, while these internal sums remain decomposition evidence,
not stand-alone direct-batch inference latency.

Do not compare the paper's published node-state numbers directly with this
edge-state adaptation. All paper tables must use fresh runs on the same pinned
Beijing train/validation/test manifests and the common evaluator.
