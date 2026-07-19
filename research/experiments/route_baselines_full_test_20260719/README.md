# Full-scale route-baseline study

This directory is the single tracked entry point for the Beijing route-
recommendation comparison. The formal result is [RESULTS.md](RESULTS.md); the
ignored `research/generated/` tree contains checkpoints, route-level
predictions, timing reports, and hash-bound receipts and must not be committed.

## Experiment

The study compares the project method with six baselines under one raw-edge
contract:

- **Project**: inverse shortest-path transition-weight recovery;
- **SP-Length**: untrained shortest path with physical road length;
- **Markov-SP**: first-order transition cost learned on training data, with
  additive smoothing selected on validation data;
- **NeuroMLR-G**: the pinned NeuroMLR model with greedy raw-edge decoding;
- **DRNCS-LG**: a clean-room DRNCS edge-state adaptation on the directed
  raw-edge line graph;
- **DRPK-static**: a clean-room, equal-information DRPK adaptation with time
  features collapsed because the dataset does not provide them; and
- **DRP-TP**: DRPK's non-learned trajectory-planning component.

The NeuroMLR-provided Beijing split is retained. The common structural filter
keeps continuous routes with at least five raw edges and no repeated original
node, while preserving directed parallel-edge identity. It leaves 605,935
training routes, 500 validation routes, and 248,233 full-test routes. Training,
preprocessing, checkpoint selection, and hyperparameter selection use only the
training and validation splits.

Every query fixes the true first and last raw edge. A prediction is the complete
generated raw-edge sequence. Failed generation remains a failure: no adapter or
evaluator appends the destination or repairs the route with ground truth.

Quality is the unweighted route-level macro average of edge precision, recall,
F1, Jaccard, and exact sequence match. Reported 95% intervals use route-level
sample variance; differences against Project are paired by route. They quantify
test-route sampling uncertainty for fixed checkpoints, not training-seed or
cross-city uncertainty.

Training and preprocessing costs reuse the frozen run records. Inference cost
comes from the same single 248,233-query task used for quality evaluation:
zero warm-up runs and one measured production pass. `/usr/bin/time -v` encloses
input and model loading, shard materialization and reloads, prediction,
validation, and final output assembly. CPU stages use at most 16 threads;
CUDA-capable neural stages are pinned to logical GPU 0.

## Reproduction

The declarative experiment is in `pipeline.json`; `protocol.json` fixes the
data, method, metric, and timing contract; `summary-input.json` declares the
artifacts used to build the paper tables.

```console
# Configuration only: no model or dataset is opened.
research/scripts/route_baseline_pipeline.sh validate

# CUDA and seven-method regression gates.
research/scripts/route_baseline_pipeline.sh smoke

# Sequential full prediction, evaluation, confidence intervals, and report.
research/scripts/route_baseline_pipeline.sh start

# Monitoring, receipt verification, and recoverable interruption.
research/scripts/route_baseline_pipeline.sh status
research/scripts/route_baseline_pipeline.sh verify
research/scripts/route_baseline_pipeline.sh stop
research/scripts/route_baseline_pipeline.sh start
```

Long tasks are sequential. Neural predictions commit bounded chunks or shards,
so an interrupted run resumes without changing the registered inference batch
shape. Runtime outputs are accepted only when their command, prerequisites,
and outputs match the recorded SHA-256 identities.

## Interpretation boundaries

DRNCS-LG is not the published node-state DRNCS implementation, and
DRPK-static is not the full time-aware DRPK model. Their names must remain
qualified in papers and tables. CPU and CUDA measurements are reported beside
their device; they are operational measurements, not hardware-independent
algorithmic speedups.
