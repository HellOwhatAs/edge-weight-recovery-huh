# Regularized inverse shortest-path learning

This project learns road-edge costs from observed routes. The main solver is a
batch projected-subgradient method backed by a CCH shortest-path oracle. The old
integer Adam + random-shock routine is retained as an explicit ablation.

## Objective and update

For observed route `i`, let `x_obs[i]` be its edge-incidence vector and let
`d_w(s_i, t_i)` be the current shortest-path distance. The empirical regret is

```text
F(w) = (1/N) sum_i [w^T x_obs[i] - d_w(s_i, t_i)].
```

If `h` is the observed aggregate edge count and `h_hat(w)` is the aggregate
count on current shortest paths, then

```text
g_w = (h - h_hat(w)) / N
```

is a subgradient. The implementation learns dimensionless multipliers
`q = w / b` relative to baseline costs `b`, with the regularized objective

```text
J(q) = F(b .* q) + lambda / (2m) * ||q - 1||^2
```

and update

```text
q[t+1] = project_[q_min,q_max](q[t] - eta0/sqrt(t+1) * g_q[t]).
```

The logged `count_residual_l1 = ||h_hat - h||_1` is a diagnostic, not the loss.
With an exact oracle, a zero residual does imply zero unregularized regret,
because `F(w) = w^T(h - h_hat)`. The converse fails under shortest-path ties:
regret can be zero even when deterministic tie-breaking selects a different
equal-cost path and the residual is nonzero. The residual magnitude is therefore
not a sound progress measure, and it does not include the regularization term.

## Continuous theory versus the CCH implementation

The convex statement is exact for continuous weights `b .* q`. CCH consumes
positive integer weights, so the program deliberately keeps two states:

- `best_multipliers.json`: continuous latent `q`;
- `best_weights.json`: quantized integer weights used by CCH.
- `checkpoint.json`: the authoritative atomic pairing of both arrays with the
  selected epoch, losses, and experiment metadata.

Because rounding can change which route is shortest, the CCH route is an
approximate oracle for the continuous objective. The program reports maximum
quantization error, and the test suite includes both an exact `f64` subgradient
inequality test and a counterexample showing that quantization can change the
shortest path. Convergence claims for the continuous projected method must not
be transferred unqualified to the quantized training loop.

## Data policy

The pickle route field is already a complete vector of original edge IDs. The
old implementation removed its first and last entries as though they were node
sentinels; this changed almost every OD pair. Complete paths are now the default.
`--trim-boundary-edges` remains available only as an explicit ablation for a
possible partial-edge interpretation.

Every loaded route is checked for:

- empty paths and invalid edge IDs;
- discontinuous adjacent edges;
- repeated nodes (positive-cost cycles).

Cyclic routes cannot be shortest under positive costs, so they are dropped by
default and counted in the log. `--keep-cycles` is a noisy-observation ablation.

## Safe quick start

Defaults intentionally use the 10,000-route `small` splits and 20 epochs rather
than the old hard-coded 4,000-epoch full-data run.

```bash
cargo test --locked

# Tiny smoke experiment
RAYON_NUM_THREADS=4 cargo run --release --locked -- \
  --city beijing \
  --epochs 3 \
  --max-train-samples 512 \
  --max-validation-samples 512 \
  --max-test-samples 512 \
  --eval-every 1 \
  --solver projected \
  --metric-update full \
  --eta0 0.00001 \
  --lambda 10000000 \
  --run-test \
  --output-prefix /tmp/beijing_pilot
```

Use `cargo run --release -- --help` for all options. In particular:

- `--solver projected|adam-shock` selects the scientific method or legacy
  ablation;
- `--metric-update full|partial` supports measured CCH customization studies;
- `--train-variant all_partial_1.0` uses an existing partial training file;
- `--eval-every 0` disables validation during training;
- `--run-test` explicitly enables the one final test evaluation after the
  experiment protocol is frozen;
- `--seed` makes the legacy shock reproducible.

`--quantization-scale` scales the integer baseline. It consequently scales raw
regret and the data gradient relative to `lambda`; changing it requires retuning
both `eta0` and `lambda`, so it is not merely a harmless precision switch.

The validation split selects checkpoints using validation regret alone. Test is
skipped by default; with `--run-test`, it is loaded and queried once after the
chosen checkpoint is restored. This makes validation-only tuning the default
and reduces accidental test leakage across repeated runs.

## Logged measurements

Each epoch reports:

- mean train regret, regularization, and their objective sum;
- the count-residual diagnostic;
- unique OD queries and total oracle time (query, path reconstruction, counting,
  consistency checks, and reduction);
- changed integer edges and percentage;
- optimizer and full/partial customization time;
- latent multiplier range, box-boundary counts, and quantization error.

Final validation/test output includes mean regret, exact path match, edge
precision/recall/F1, and edge Jaccard. Standard set metrics replace the old
one-sided baseline-weighted overlap score. `relative_regret` is the aggregate
ratio `sum(regret) / sum(observed_cost)`, not a mean of per-route ratios.

## Scaling protocol

Do not jump directly to the full Beijing split. A defensible sequence is:

1. run unit tests and 128–512 route smoke experiments;
2. tune `eta0`, `lambda`, and the multiplier box on validation data;
3. compare full and partial customization at measured changed-edge ratios;
4. move through `all_partial_1.0`, `all_partial_5.0`, and larger subsets;
5. run the full split only after per-epoch time and memory are understood.

Full Beijing train/validation/test contain roughly 786k/309k/322k samples, and
deserializing the full collections can require well over one GiB of memory.
The current pickle reader must deserialize a whole file before applying
`--max-*-samples`, and a limit takes the first records rather than a random
sample. Use the existing prebuilt partial/small files for memory-bounded runs;
use randomized, pre-generated subsets for formal statistical comparisons.
