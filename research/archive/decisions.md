# Historical research decisions

This index is the durable part of the pre-workspace experiment history. It is
descriptive, not an execution guide. Historical paths and commands in linked
reports refer to the repository layout at the time of each run.

## Adopted

### Directed line graph as the production representation

The production problem uses original directed roads as routing nodes and legal
road-to-road transitions as directly learned coordinates. An observed sequence
of `N` roads maps to `N - 1` transition coordinates. Source and target offsets
are zero, so the first road has no learned coordinate cost.

The first 10% calibration showed only negligible learning under direct-weight
Euclidean optimization. That result was diagnostic, not evidence against the
line graph. See the [calibration report](experiments/line_graph_10pct_calibration/report.md).

### Relative-coordinate projected subgradient

The optimizer recovery restored the dimensionless parameterization `q = w/w0`
with one direct learned-weight vector stored in checkpoints. It recovered the
historical original-edge result and improved the line-graph model. This is the
experimental basis for the frozen v1 algorithm. See the
[recovery report](experiments/optimizer_recovery/report.md) and
[summary](experiments/optimizer_recovery/summary.json).

### One full-data static model

Training one static line-graph model on all accepted trajectories outperformed
five separately selected departure-time bucket models. The full-data static
model therefore remained the recommendation. See the
[independent-bucket report](experiments/independent_time_buckets/report.md).

## Rejected or superseded

### Direct-weight Euclidean calibration

The direct optimizer was retained only long enough to diagnose its scale
regression. Its eta-screening scripts and intermediate configs are not supported
inputs after the workspace split. Exact run details remain embedded in the
calibration machine summary and can also be recovered from Git.

### Shared temporal residual and travel-time proxy

The `q_bucket = q_global + residual_bucket` model and the train-only trip-average
travel-time proxy produced small, unstable gains and added a special optimizer,
checkpoint schema, and evaluator. The implementation was removed. The
[archived report](experiments/archive/full_data_shared_temporal_residual/report.md)
and [summary](experiments/archive/full_data_shared_temporal_residual/summary.json)
retain the conclusion and full configuration records.

### Independent departure-time buckets

Five ordinary static models lost overall Edge F1, Exact Match, and Jaccard
against one full-data static model. The sparse night bucket degraded most. This
negative result is retained to prevent repeating the same experiment without a
materially different hypothesis.

### Extra CCH decimal precision

One decimal place was technically viable but did not consistently improve route
quality over integer route selection. Two decimal places exceeded the CCH
infinity sentinel for a valid upper bound and could not start. The production
oracle contract therefore remains integer quantization. See the
[fixed-point report](experiments/fixed_point_cch_precision/report.md).

## Final historical benchmark

The frozen NeuroMLR comparison found close, not superior, quality: project
Edge F1 `0.766015` versus NeuroMLR-Greedy `0.768496` on the registered common
test protocol. CCH was `1.68x` faster than Dijkstra on the fixed-point training
workload including preprocessing and `8.05x` faster on query-only inference.
These claims are valid only with the workload and timing boundaries in the
[benchmark report](experiments/neuromlr_cch_dijkstra_benchmarks/report.md).

## Reuse policy

- Use reports and summaries to understand a decision, not old scripts to start
  a new run.
- Implement new studies against the current `research` protocol.
- Recover retired source with `git show 0f4bd55:<path>` when historical code is
  genuinely required.
- Do not interpret an archived experiment's use of "active" as a statement
  about the current tree.
