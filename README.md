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

Cyclic routes cannot be shortest under positive costs. Training therefore uses
`--train-cycle-policy drop` by default; `erase` applies chronological loop
erasure before learning, and `keep` is a deliberately misspecified
noisy-observation ablation. The legacy `--keep-cycles` flag is a training-only
alias for `keep`. Validation and test always use `drop`, so changing the training
policy cannot silently change the evaluation task.

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
  --eval-every 1 \
  --solver projected \
  --metric-update full \
  --selection-metric relative-regret \
  --eta0 0.00001 \
  --lambda 10000000 \
  --output-prefix /tmp/beijing_pilot
```

Use `cargo run --release -- --help` for all options. In particular:

- `--solver projected|adam-shock` selects the scientific method or legacy
  ablation;
- `--metric-update full|partial` supports measured CCH customization studies;
- `--selection-metric mean-regret|relative-regret` controls validation-only
  checkpoint selection;
- `--eval-path-metrics` reports exact-path and edge-overlap metrics at each
  validation event, rather than only at final evaluation;
- `--early-stop-min-delta` separates checkpoint improvements from the larger
  improvement required to reset early-stopping patience;
- `--train-cycle-policy drop|erase|keep` controls training observations only;
- `--train-variant all_partial_1.0` uses an existing partial training file;
- `--eval-every 0` disables validation during training;
- `--run-test` explicitly enables the one final test evaluation after the
  experiment protocol is frozen;
- `--seed` makes the legacy shock reproducible.

`--quantization-scale` scales the integer baseline. It consequently scales raw
regret and the data gradient relative to `lambda`; changing it requires retuning
both `eta0` and `lambda`, so it is not merely a harmless precision switch.

The validation split selects checkpoints using validation regret alone. The
randomized scale study uses aggregate relative regret so routes are weighted by
their observed cost. Test is skipped by default; with `--run-test`, it is loaded
and queried once after the chosen checkpoint is restored. This makes
validation-only tuning the default and reduces accidental test leakage across
repeated runs.

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

## Bounded convergence and error-attribution follow-up

The latest Beijing study extended the edge-only solver from 20 to at most 100
epochs on 10% and full train, with validation every five epochs, validation-path
metrics, minimum-delta early stopping, and a hard 900-second limit per run. It
used one complete time-blocked development set and two source-index-disjoint,
one-shot AM/PM confirmation blocks. Formal training, diagnostics, confirmation,
and the capacity probe did not load or evaluate test; no test contents or
metrics were used.

On the 129,033 valid development routes, the selected full-train checkpoint
(`eta0=3e-4`, epoch 99) reached relative regret `0.06348409`, edge F1
`0.681488`, and exact match `0.371068`. On the pooled 31,662 one-shot
confirmation routes, it reached `0.06302821`, `0.684512`, and `0.376508`.
The exact same-eta development trajectory improved from `0.06826350` at epoch
19 to `0.06357497` at epoch 99, which establishes that the old horizon was too
short. The final selected candidate also changed eta; relative to the frozen
20-epoch control, its paired confirmation improvement was `0.00442438` absolute
regret (95% bootstrap interval
`[0.00405114, 0.00483054]`), `0.023606` F1, and `0.021793` exact match. The
selected epoch is still the 100-epoch boundary, so the study does not establish
full convergence.

First-divergence diagnostics show that the remaining errors concentrate on
long, junction-complex routes and frequently rejoin the observed route. A
full-train audit also shows that dropping all cyclic records removes 20.67% of
otherwise structurally valid observations, but chronological loop erasure did
not improve the full-data checkpoint under the frozen edge-only hyperparameters.
A preregistered fixed-edge single-left-turn-penalty probe then tested one minimal
capacity extension on disjoint development tune/audit routes. All correctness
gates passed, but the grid selected zero penalty: every positive penalty raised
the primary tune regret. This rejects a uniform nonnegative left-turn penalty,
not richer conditional or jointly learned turn models. The completed
confirmation blocks were not reused.

See [`experiments/convergence_study/RESULTS.md`](experiments/convergence_study/RESULTS.md)
for the protocol, complete tables, artifact map, limitations, and bounded
reproduction commands. Machine-readable results are in the same directory; a
hash-manifested evidence bundle preserves the six existing one-shot route
exports, three frozen checkpoints, and ten training logs without raw or test
data.

The matrix runner invalidates stale caches unless the complete matrix row,
command, runner concurrency/timeout context, executable SHA-256,
graph/train/validation input SHA-256 values, and checkpoint/log output SHA-256
values all match. It never fingerprints the test split.

## Reproducible randomized scale study

The formal validation study is specified by CSV matrices in
`experiments/scale_study/`. Generated pickle subsets remain ignored, while their
source indices, fixed RNG algorithm, seeds, sizes, and SHA-256 hashes are kept
in versioned manifests. Subsets with the same seed are nested, so 1%, 5%, and
10% comparisons are paired by construction.

```bash
# Build once and generate the seeded subsets serially (avoids concurrent full-file loads).
cargo build --release --locked --all-targets
python3 scripts/generate_scale_subsets.py \
  --plan experiments/scale_study/subset_plan.csv \
  --manifest-dir experiments/scale_study/subsets \
  --aggregate-manifest experiments/scale_study/subset_manifest.json

# Run any matrix. Completed checkpoints are cached; no --run-test is passed.
python3 scripts/run_experiment_matrix.py \
  --matrix experiments/scale_study/matrix_grid_1pct.csv \
  --run-root /tmp/edge-weight-scale-study/runs \
  --validation-variant scale_fixed_seed20260715 \
  --summary-csv experiments/scale_study/results.csv \
  --summary-json experiments/scale_study/results.json \
  --jobs 3 --rayon-threads 4 --timeout-seconds 900

# Rebuild aggregate grid and scale tables.
python3 scripts/analyze_scale_results.py \
  --results experiments/scale_study/results.json \
  --coverage experiments/scale_study/coverage_all_seeds.json \
  --coverage-audit experiments/scale_study/coverage_audit.json \
  --output-json experiments/scale_study/aggregate_results.json \
  --scale-csv experiments/scale_study/scale_curve.csv \
  --grid-csv experiments/scale_study/grid_ranking.csv
```

The study followed this sequence:

1. coverage audit at 1,000, 5,000, 1%, 5%, 10%, and full metadata;
2. a fixed 20,000-record validation subset and nested train subsets for seeds
   42, 43, and 44;
3. a 15-point 1% grid, followed by top-three multi-seed replication;
4. top-two configurations at 5% and 10%;
5. one fixed-configuration full-train endpoint after its runtime was shown safe;
6. per-route epsilon, seen/unseen, and route-length diagnostics;
7. controlled full/partial customization timings at 1%, 5%, and 10% changed
   edges.

See [EXPERIMENTS.md](EXPERIMENTS.md) for interpretation and
`experiments/scale_study/aggregate_results.json` for exact values. Full Beijing
was trained once with the already selected primary configuration; it was not a
new hyperparameter-search stage.

Full Beijing train/validation/test contain roughly 786k/309k/322k samples, and
deserializing the full collections can require well over one GiB of memory.
The current pickle reader must deserialize a whole file before applying
`--max-*-samples`, and a limit takes the first records rather than a random
sample. Use the existing prebuilt partial/small files for memory-bounded runs;
use the manifest-backed randomized subsets for formal statistical comparisons.
