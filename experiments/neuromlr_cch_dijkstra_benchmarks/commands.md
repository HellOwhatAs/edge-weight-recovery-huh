# Actual command registry

All commands run from the repository root. Large outputs are under
`artifacts/neuromlr_cch_dijkstra_benchmarks/` or this study's ignored
`generated/` directory.

## Upstream and environment

```bash
git clone https://github.com/idea-iitd/NeuroMLR.git /tmp/NeuroMLR
git -C /tmp/NeuroMLR rev-parse HEAD

python3 -m virtualenv artifacts/neuromlr_cch_dijkstra_benchmarks/.venv
artifacts/neuromlr_cch_dijkstra_benchmarks/.venv/bin/pip install \
  torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
artifacts/neuromlr_cch_dijkstra_benchmarks/.venv/bin/pip install \
  numpy==2.2.6 scipy==1.15.3 pyshp==2.3.1 \
  torch-geometric==2.6.1 termcolor==1.1.0
```

The system lacked `ensurepip`; the actual bootstrap used PyPA `get-pip.py` to
install `virtualenv` into `/tmp`, then created the same repository-local venv.

## Common train and validation

```bash
cargo build --release --locked --bin build_common_manifest

target/release/build_common_manifest \
  --city beijing --split train --variant all \
  --minimum-edges 5 --maximum-selected all \
  --manifest experiments/neuromlr_cch_dijkstra_benchmarks/generated/manifests/train.jsonl \
  --pickle data/beijing_data/preprocessed_train_trips_neuromlr_common.pkl \
  --audit experiments/neuromlr_cch_dijkstra_benchmarks/train_audit.json \
  --protocol experiments/neuromlr_cch_dijkstra_benchmarks/protocol.json

target/release/build_common_manifest \
  --city beijing --split validation --variant scale_fixed_seed20260715 \
  --minimum-edges 5 --maximum-selected 500 \
  --manifest experiments/neuromlr_cch_dijkstra_benchmarks/generated/manifests/validation.jsonl \
  --pickle data/beijing_data/preprocessed_validation_trips_neuromlr_common.pkl \
  --audit experiments/neuromlr_cch_dijkstra_benchmarks/validation_audit.json \
  --protocol experiments/neuromlr_cch_dijkstra_benchmarks/protocol.json
```

## NeuroMLR train/validation

```bash
artifacts/neuromlr_cch_dijkstra_benchmarks/.venv/bin/python \
  scripts/neuromlr_fair.py train \
  --upstream-dir /tmp/NeuroMLR \
  --map-dir data/beijing_data/map \
  --train-pickle data/beijing_data/preprocessed_train_trips_neuromlr_common.pkl \
  --validation-manifest experiments/neuromlr_cch_dijkstra_benchmarks/generated/manifests/validation.jsonl \
  --output-dir artifacts/neuromlr_cch_dijkstra_benchmarks/neuromlr_formal \
  --epochs 50 --validation-every 5 --batch-size 32 --learning-rate 0.001 \
  --seed 20260716 --device cuda:0
```

The training driver evaluates NeuroMLR-Greedy on the same 500 validation paths
every five epochs. After training, all ten saved Greedy checkpoints are replayed
and ranked by the Rust common evaluator using maximum Edge F1, then maximum
Exact Match, then the earliest epoch:

```bash
python3 scripts/select_neuromlr_greedy_checkpoint.py \
  --python artifacts/neuromlr_cch_dijkstra_benchmarks/.venv/bin/python \
  --driver scripts/neuromlr_fair.py --upstream-dir /tmp/NeuroMLR \
  --map-dir data/beijing_data/map \
  --run-dir artifacts/neuromlr_cch_dijkstra_benchmarks/neuromlr_formal \
  --manifest experiments/neuromlr_cch_dijkstra_benchmarks/generated/manifests/validation.jsonl \
  --common-evaluator target/release/evaluate_predictions \
  --output artifacts/neuromlr_cch_dijkstra_benchmarks/neuromlr_formal/greedy_common_selection.json \
  --seed 20260716 --device cuda:0
```

Per the final experiment scope, NeuroMLR-Dijkstra is neither used for
checkpoint selection nor run on validation/test. NeuroMLR-Greedy is the sole
external quality baseline.

## Project common-data training and checkpoint selection

The initial four-thread formal attempt was stopped before protocol freeze. A
full-data ten-update pilot then compared 8 and 16 Rayon threads sequentially;
the latter was selected by median training-oracle time. Raw identities and the
decision are in `thread_scaling_audit.json`.

```bash
RAYON_NUM_THREADS=8 /usr/bin/time -v \
  -o artifacts/neuromlr_cch_dijkstra_benchmarks/thread_scaling_8_u10.time.txt \
  target/release/train \
  --config experiments/neuromlr_cch_dijkstra_benchmarks/configs/project_thread_scaling_8_u10.json \
  --output-dir artifacts/neuromlr_cch_dijkstra_benchmarks/thread_scaling_8_u10

RAYON_NUM_THREADS=16 /usr/bin/time -v \
  -o artifacts/neuromlr_cch_dijkstra_benchmarks/thread_scaling_16_u10.time.txt \
  target/release/train \
  --config experiments/neuromlr_cch_dijkstra_benchmarks/configs/project_thread_scaling_16_u10.json \
  --output-dir artifacts/neuromlr_cch_dijkstra_benchmarks/thread_scaling_16_u10
```

```bash
RAYON_NUM_THREADS=16 target/release/train \
  --config experiments/neuromlr_cch_dijkstra_benchmarks/configs/project_common_train_u500.json \
  --output-dir artifacts/neuromlr_cch_dijkstra_benchmarks/project_common_train_u500

python3 scripts/select_edge_to_edge_checkpoint.py \
  --run-dir artifacts/neuromlr_cch_dijkstra_benchmarks/project_common_train_u500 \
  --manifest experiments/neuromlr_cch_dijkstra_benchmarks/generated/manifests/validation.jsonl \
  --benchmark-binary target/release/benchmark_routes \
  --output artifacts/neuromlr_cch_dijkstra_benchmarks/project_common_train_u500/edge_to_edge_selection.json \
  --threads 16
```

## Frozen training-oracle workload

The original 50,000-OD run proved that initial route equality was insufficient:
new equal-distance ties appeared after updates and caused optimizer divergence.
That run and a roughly 20-minute fixed-point attempt are excluded. The
post-test efficiency-only amendment freezes the fully converged workload below.
It starts with 5,000 initially matching OD groups and repeatedly removes every
OD whose CCH/Dijkstra path differs in any of 21 shared-weight states. Eight
passes leave 4,971 groups / 4,979 observations.

```bash
RAYON_NUM_THREADS=16 target/release/benchmark_training_oracles \
  --config experiments/neuromlr_cch_dijkstra_benchmarks/configs/project_common_train_u500.json \
  --manifest experiments/neuromlr_cch_dijkstra_benchmarks/generated/manifests/train.jsonl \
  --output experiments/neuromlr_cch_dijkstra_benchmarks/generated/benchmarks/training-oracle-stabilization-smoke-5k.json \
  --oracle both_cch_first --candidate-samples 6000 --maximum-groups 5000 \
  --require-path-match true --stabilize-path-matches true \
  --frozen-workload none --updates 20 --threads 16
```

The four formal commands load that exact OD list; fixed-point selection is not
inside the timing boundary. Repetitions 1 and 3 use `both_cch_first` and 2 and
4 use `both_dijkstra_first`:

```bash
RAYON_NUM_THREADS=16 target/release/benchmark_training_oracles \
  --config experiments/neuromlr_cch_dijkstra_benchmarks/configs/project_common_train_u500.json \
  --manifest experiments/neuromlr_cch_dijkstra_benchmarks/generated/manifests/train.jsonl \
  --output experiments/neuromlr_cch_dijkstra_benchmarks/generated/benchmarks/training-oracles-5k-rep1-cch-first.json \
  --oracle both_cch_first --candidate-samples 6000 --maximum-groups 5000 \
  --require-path-match true --stabilize-path-matches false \
  --frozen-workload experiments/neuromlr_cch_dijkstra_benchmarks/generated/benchmarks/training-oracle-stabilization-smoke-5k.json \
  --updates 20 --threads 16

RAYON_NUM_THREADS=16 target/release/benchmark_training_oracles \
  --config experiments/neuromlr_cch_dijkstra_benchmarks/configs/project_common_train_u500.json \
  --manifest experiments/neuromlr_cch_dijkstra_benchmarks/generated/manifests/train.jsonl \
  --output experiments/neuromlr_cch_dijkstra_benchmarks/generated/benchmarks/training-oracles-5k-rep2-dijkstra-first.json \
  --oracle both_dijkstra_first --candidate-samples 6000 --maximum-groups 5000 \
  --require-path-match true --stabilize-path-matches false \
  --frozen-workload experiments/neuromlr_cch_dijkstra_benchmarks/generated/benchmarks/training-oracle-stabilization-smoke-5k.json \
  --updates 20 --threads 16

# Repetitions 3 and 4 use the same arguments and frozen workload, with
# outputs training-oracles-5k-rep3-cch-first.json and
# training-oracles-5k-rep4-dijkstra-first.json respectively.
```

The claim adds each method's one-time topology/adjacency setup to 21
customizations/query batches and 20 optimizer updates. Common data loading,
graph mapping, initial OD screening, and fixed-point stabilization are excluded.

## Final test and inference

After validation evidence was hashed, these commands created the unlock and
performed the sole test-pickle decode. The completed receipt prevents another
decode:

```bash
python3 scripts/create_test_unlock.py \
  --protocol experiments/neuromlr_cch_dijkstra_benchmarks/protocol.json \
  --validation-evidence experiments/neuromlr_cch_dijkstra_benchmarks/validation_evidence.json \
  --output experiments/neuromlr_cch_dijkstra_benchmarks/test_unlock.json

target/release/build_common_manifest \
  --city beijing --split test --variant small \
  --minimum-edges 5 --maximum-selected 500 \
  --manifest experiments/neuromlr_cch_dijkstra_benchmarks/generated/manifests/test.jsonl \
  --pickle data/beijing_data/preprocessed_test_trips_neuromlr_common.pkl \
  --audit experiments/neuromlr_cch_dijkstra_benchmarks/test_audit.json \
  --protocol experiments/neuromlr_cch_dijkstra_benchmarks/protocol.json \
  --test-unlock experiments/neuromlr_cch_dijkstra_benchmarks/test_unlock.json \
  --test-receipt experiments/neuromlr_cch_dijkstra_benchmarks/test_access_receipt.json
```

Final fair quality predictions:

```bash
RAYON_NUM_THREADS=16 target/release/benchmark_routes \
  --checkpoint artifacts/neuromlr_cch_dijkstra_benchmarks/project_common_train_u500/checkpoint-200.json \
  --manifest experiments/neuromlr_cch_dijkstra_benchmarks/generated/manifests/test.jsonl \
  --predictions experiments/neuromlr_cch_dijkstra_benchmarks/generated/predictions/project-edge-to-edge-test.jsonl \
  --summary experiments/neuromlr_cch_dijkstra_benchmarks/generated/results/project-edge-to-edge-test.json \
  --oracle cch --query-protocol edge_to_edge --threads 16 \
  --warmup-repetitions 0 --measured-repetitions 1

artifacts/neuromlr_cch_dijkstra_benchmarks/.venv/bin/python \
  scripts/neuromlr_fair.py predict \
  --upstream-dir /tmp/NeuroMLR --map-dir data/beijing_data/map \
  --checkpoint artifacts/neuromlr_cch_dijkstra_benchmarks/neuromlr_formal/checkpoint-epoch-45.pt \
  --manifest experiments/neuromlr_cch_dijkstra_benchmarks/generated/manifests/test.jsonl \
  --method greedy \
  --predictions experiments/neuromlr_cch_dijkstra_benchmarks/generated/predictions/neuromlr-greedy-test.jsonl \
  --summary experiments/neuromlr_cch_dijkstra_benchmarks/generated/results/neuromlr-greedy-test.json \
  --seed 20260716 --device cuda:0 \
  --warmup-repetitions 1 --measured-repetitions 5

target/release/evaluate_predictions \
  --predictions experiments/neuromlr_cch_dijkstra_benchmarks/generated/predictions/neuromlr-greedy-test.jsonl \
  --output experiments/neuromlr_cch_dijkstra_benchmarks/generated/results/neuromlr-greedy-test-common.json
```

The final project CCH/Dijkstra inference commands use the same selected
checkpoint, the same node-to-node test manifest, one warm-up, five measured
repetitions, and one thread. They differ only in `--oracle cch|dijkstra` and
write `inference-cch-test.json` / `inference-dijkstra-test.json`; the paired raw
predictions are checked by `compare_oracle_predictions.py`. The final aggregate
is produced by `summarize_oracle_benchmarks.py` from four training records, two
isolated memory records, and the two inference records.

## Quality checks

```bash
cargo fmt --check
cargo build --release --locked
cargo test --locked --all-targets
cargo clippy --locked --all-targets -- -D warnings
git diff --check
artifacts/neuromlr_cch_dijkstra_benchmarks/.venv/bin/python -m py_compile \
  scripts/neuromlr_fair.py scripts/select_edge_to_edge_checkpoint.py \
  scripts/select_neuromlr_greedy_checkpoint.py \
  scripts/compare_oracle_predictions.py scripts/summarize_oracle_benchmarks.py \
  scripts/create_test_unlock.py
artifacts/neuromlr_cch_dijkstra_benchmarks/.venv/bin/python -m unittest \
  tests/test_neuromlr_fair.py
```
