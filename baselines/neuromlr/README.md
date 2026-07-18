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
.venv/bin/ewr-neuromlr predict \
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
