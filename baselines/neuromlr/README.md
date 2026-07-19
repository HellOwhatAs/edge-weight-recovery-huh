# NeuroMLR protocol adapter

This directory is an independent Python package for the NeuroMLR baseline. It
does not import the production Rust workspace or any project experiment module.
Its only shared boundary is the versioned file protocol documented in
`../../research/README.md`.

The model implementation comes from the official NeuroMLR checkout pinned at
commit `c45e3b5811e5a59b36e4682307d2196c02dac360`. The adapter preserves the
existing Lipschitz embeddings, next-road training targets, Greedy rollout, and
destination-conditioned Dijkstra algorithms while replacing upstream data and
artifact plumbing.

The complete source identity is recorded in `upstream.json`. Prediction and
training verify the commit, Git tree, clean tracked worktree, and the SHA-256 of
all six recorded upstream files. Use a persistent checkout, not a temporary
directory:

```console
git clone https://github.com/idea-iitd/NeuroMLR /path/to/persistent/neuromlr
git -C /path/to/persistent/neuromlr checkout --detach \
  c45e3b5811e5a59b36e4682307d2196c02dac360
```

## Environment boundary

The tested environment is Python 3.12 with NumPy 2.2.6, SciPy 1.15.3,
PyShp 2.3.1, PyTorch 2.7.1, and PyG 2.6.1. CUDA installations should install
the appropriate PyTorch wheel before installing this package; the remaining
versions are fixed in `pyproject.toml`.

```console
cd baselines/neuromlr
python3.12 -m venv .venv
.venv/bin/python -m pip install \
  torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -e '.[test]'
```

The caller owns four external boundaries:

- `--upstream-dir`: the pinned, otherwise unmodified NeuroMLR checkout;
- `--map-dir`: `nodes.shp` and `edges.shp` in the raw-road ID order;
- dataset descriptors and their relative JSONL record files; and
- output directories for checkpoints, predictions, receipts, and diagnostics.

Every dataset descriptor has exactly these fields:

```json
{
  "schema": "ewr.dataset-manifest/v1",
  "dataset_id": "beijing/train",
  "network_id": "beijing-roads-v1",
  "records_schema": "ewr.dataset-record/v1",
  "records_file": "train.jsonl"
}
```

Each referenced JSONL row is exactly
`{"sample_id":"...","original_edge_ids":[...]}`. Paths are complete and
all edge IDs address the unmodified shapefile record order. Training and
validation descriptors must have the same `network_id` and disjoint sample
IDs. The adapter accepts every continuous, cycle-free path with at least two
edges; it does not retain the historical five-edge benchmark filter. It also
does not expose a record-limit flag: create a separate descriptor and JSONL for
an explicit subset so predictions always cover the manifest bound by the run
receipt.

## Train

```console
.venv/bin/ewr-neuromlr train \
  --upstream-dir /path/to/NeuroMLR \
  --map-dir /path/to/map \
  --train-manifest /path/to/train.manifest.json \
  --validation-manifest /path/to/validation.manifest.json \
  --output-dir /path/to/run/train
```

Training writes checkpoints plus method-local configuration, epoch logs, and
checkpoint-selection diagnostics. The checkpoint binds both the raw graph hash
and the descriptor `network_id`.

## Predict

```console
CUDA_VISIBLE_DEVICES=0 .venv/bin/ewr-neuromlr predict \
  --upstream-dir /path/to/NeuroMLR \
  --map-dir /path/to/map \
  --checkpoint /path/to/run/train/checkpoint-epoch-50.pt \
  --dataset-manifest /path/to/test.manifest.json \
  --method greedy \
  --predictions /path/to/run/predictions.jsonl \
  --run-receipt /path/to/run/run.json \
  --diagnostics /path/to/run/neuromlr-diagnostics.json \
  --source-revision COMMIT-OF-THIS-ADAPTER
```

Both prediction methods accept `--warmup-repetitions` and
`--measured-repetitions`. Their defaults are zero warm-ups and one measured
repetition. For example, a repeated NeuroMLR-Dijkstra measurement can use:

```console
CUDA_VISIBLE_DEVICES=0 .venv/bin/ewr-neuromlr predict \
  --upstream-dir /path/to/NeuroMLR \
  --map-dir /path/to/map \
  --checkpoint /path/to/run/train/checkpoint-epoch-50.pt \
  --dataset-manifest /path/to/test.manifest.json \
  --method dijkstra \
  --predictions /path/to/run/predictions.jsonl \
  --run-receipt /path/to/run/run.json \
  --diagnostics /path/to/run/neuromlr-diagnostics.json \
  --source-revision COMMIT-OF-THIS-ADAPTER \
  --warmup-repetitions 1 \
  --measured-repetitions 5
```

Every Dijkstra repetition covers the complete ordered query batch. For every
query it recomputes destination-conditioned transition scores over the road
graph and then runs Dijkstra; neither transition scoring nor search is reused
between repetitions. Warm-ups execute the same boundary but are excluded from
the measured mean. The adapter compares every warm-up and measured prediction
against one common route reference and fails if any complete raw-edge sequence
changes.

Diagnostics retain each full-batch duration in
`prediction_repetition_seconds`, each warm-up duration in
`warmup_repetition_seconds`, and per-repetition component totals in
`component_totals_per_repetition`. `prediction_seconds` and
`component_totals` are the corresponding means over measured repetitions.

### Resumable full-test NeuroMLR-G prediction

The default `--route-chunk-size 0` retains the original full-batch behavior.
For a large quality-only test, bound GPU memory and make every completed route
chunk recoverable by selecting a positive chunk size:

```console
CUDA_VISIBLE_DEVICES=0 .venv/bin/ewr-neuromlr predict \
  --upstream-dir /path/to/NeuroMLR \
  --map-dir /path/to/map \
  --checkpoint /path/to/run/train/checkpoint-epoch-45.pt \
  --dataset-manifest /path/to/full-test.manifest.json \
  --method greedy \
  --device cuda:0 \
  --predictions /path/to/run/full-test/predictions.jsonl \
  --run-receipt /path/to/run/full-test/run.json \
  --diagnostics /path/to/run/full-test/diagnostics.json \
  --source-revision COMMIT-OF-THIS-ADAPTER \
  --route-chunk-size 500 \
  --resume auto \
  --resume-dir /path/to/run/full-test/resume \
  --progress /path/to/run/full-test/progress.json
```

This mode is deliberately limited to NeuroMLR-Greedy with zero warm-up
repetitions and one measured pass. It is the full-test quality workload; its
accumulated prediction time, throughput, RSS, and CUDA peak are retained as
workload-specific execution costs. Each shard is written by atomic rename
before the progress file advances. Re-running the identical command with
`--resume auto` validates and skips committed shards; `--resume require`
additionally fails unless recovery metadata already exists. A chunk interrupted
before its progress commit is recomputed.

Every prediction that requests a CUDA device fails if CUDA is unavailable
instead of silently falling back to CPU. Pass `--device cpu` only for an
intentional CPU run. CUDA prediction additionally requires logical `cuda:0`
and `CUDA_VISIBLE_DEVICES=0`; that mapping is recorded in the resume identity.

The recovery binding includes SHA-256 identities for the checkpoint, manifest,
the manifest's JSONL records, adapter source, graph topology and coordinates,
plus the model configuration, Python/PyTorch/NumPy and CUDA environment,
device, seed, output path, and source revision. Changing any of these rejects
the resume instead of mixing predictions from different experiments. Every
retained shard is also re-read to verify its hash, exact sample order, row
schema, raw-graph edge range, fixed true first edge, every directed edge
transition, and endpoint-failure count. Final `predictions.jsonl` is an atomic
byte concatenation of the ordered shards.

`route_chunk_size` is part of the registered inference protocol, not merely a
performance knob. Neural matrix kernels can change an `argmax` at a near tie
when the active batch shape changes. A real-checkpoint CPU gate found one route
difference among 500 when comparing chunks of 128 against a 500-route full
batch; a chunk of 500 was byte-identical to the same-device full batch.
Therefore freeze `500` before the full test to retain the registered 500-route
batch boundary as closely as possible, keep the same ordered manifest and
device, and do not mix outputs from other chunk sizes.
The resume mechanism guarantees equivalence to an uninterrupted run of that
fixed chunk protocol. It does not claim that a multi-chunk full-test run is
bitwise identical to an infeasible all-routes-at-once batch.

Progress is human-readable JSON. For example:

```console
python3 -m json.tool /path/to/run/full-test/progress.json
```

It reports committed samples/chunks, accumulated prediction time, endpoint
failures, estimated remaining prediction time, session count, maximum observed
RSS/CUDA allocation across committed sessions, and final output identity.
Standard output emits the same core fields in one compact event per committed
chunk. After a resumed run, diagnostics label timing as the sum of atomic chunk
measurements and resources as the maximum observed across committed sessions.
Report that boundary explicitly; it is not interchangeable with the earlier
repeated 500-query timing.

`predictions.jsonl` contains only
`{"sample_id":"...","predicted_edge_ids":[...]}`. It contains no truth
copy, method name, metric, or timing. `run.json` conforms to
`ewr.run-receipt/v1`; model-specific timing and resource measurements live in
the separate diagnostics file. Evaluation and sample alignment belong to the
common research workspace.

## Tests

The minimal repository-root command needs only Python's standard library. It
runs all protocol tests and explicitly skips model-algorithm tests when the
scientific stack is absent:

```console
python3 -m unittest discover -s baselines/neuromlr/tests -v
```

After installing the pinned environment, run the complete model-side suite
from this directory:

```console
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pytest
```

The tests lock the original model-side algorithms and the strict descriptor,
dataset row, prediction row, and run-receipt shapes. They do not download the
upstream checkout or execute full model training.
