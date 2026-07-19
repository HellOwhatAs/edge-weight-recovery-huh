# Static route baselines

This research-only crate implements two equal-information baselines on the
same directed line graph and fixed raw-road endpoint protocol as the project:

- `sp_length`: the transition cost is the physical length of the road being
  entered. The fixed first road contributes no route-dependent cost.
- `markov_sp`: training counts adjacent raw-road transitions and uses additive
  smoothing,
  `cost(previous,next) = -log P(next | previous)`. The smoothing value is
  selected on validation only, by macro edge F1, exact match, then the smaller
  alpha.

Both methods route with the production CCH backend. Predictions contain only
the versioned protocol fields and are evaluated by `ewr-evaluate`; diagnostics
keep preprocessing, selection, warm-up, measured fixed-batch timing, thread
count, endpoint mismatches, and peak RSS out of the prediction rows.

The default CPU worker count is 16. Markov costs are multiplied by `100000`
and rounded to positive `u32` values before routing. This quantization rule and
the candidate alpha grid are recorded in the artifact and diagnostics. During
SP-Length prediction, the current road-length metric is quantized again and
must exactly match the artifact, so a same-topology network with different
lengths cannot reuse the artifact.

```console
cargo run --release --locked --manifest-path research/Cargo.toml \
  --bin ewr-static-baseline -- train \
  --method markov \
  --nodes data/beijing_data/map/nodes.shp \
  --edges data/beijing_data/map/edges.shp \
  --train-jsonl /path/to/train.jsonl \
  --validation-jsonl /path/to/validation.jsonl \
  --alpha-candidates 0.01,0.1,1,10 \
  --artifact /path/to/markov.artifact.json \
  --diagnostics /path/to/markov.train.json \
  --threads 16
```

```console
cargo run --release --locked --manifest-path research/Cargo.toml \
  --bin ewr-static-baseline -- predict \
  --artifact /path/to/markov.artifact.json \
  --nodes data/beijing_data/map/nodes.shp \
  --edges data/beijing_data/map/edges.shp \
  --dataset-jsonl /path/to/test.jsonl \
  --predictions /path/to/predictions.jsonl \
  --diagnostics /path/to/predict.json \
  --threads 16 --warmup-repetitions 1 --measured-repetitions 5
```
