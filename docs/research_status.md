# Research status

This document is the claim boundary for the active repository. The project has
two nested model classes: an edge-only inverse shortest-path baseline and an
expanded road model with nonnegative transition residuals. The expanded model
is trained through one joint `(q,r)` optimization path; historical training
arms are not active scientific abstractions.

## 1. Established

### Edge-only inverse shortest paths

- The edge-only metric is `w_e = b_e q_e`, with `q` projected to a positive
  box and regularized toward one.
- The data objective is observed-path cost minus true shortest distance.
  Aggregate count residual is a tie-dependent subgradient diagnostic, not the
  loss.
- Continuous latent state is separate from the positive integer CCH metric;
  quantization is explicit and checked against the CCH infinity sentinel.
- Production queries use full CCH customization and unique-OD batching. OD
  multiplicity weights distances and predicted edge counts correctly.
- Complete paths are checked for edge IDs, continuity, endpoints, and cycles.
  The only active cycle policy drops cyclic observations.
- Checkpoint selection is validation-only. Formal training does not read test
  data.
- Edge-only checkpoints bind configuration, `q`, quantized weights, selection
  state, and data/baseline identity. The edge model's strict constructor can
  reconstruct weights from finite positive `q` without implicit repair.

The frozen full-Beijing baseline selected epoch 99 at the boundary of a
bounded 100-epoch run. On 129,033 accepted development routes it reported
relative regret `0.06348409`, raw mean regret `339523.40`, Edge F1 `0.681488`,
and Exact Match `0.371068`. On pooled validation-derived AM/PM confirmation
blocks it reported relative regret `0.06302821`, Edge F1 `0.684512`, and Exact
Match `0.376508`. Those confirmation blocks are spent evidence, not untouched
test data, and boundary selection does not establish convergence.

### Expanded-road correctness and nesting

The active expanded model is

```text
kappa_(e,f) = b_f q_f + residual_scale r_(e,f),    r_(e,f) >= 0

J(q,r) = mean route regret
       + lambda_edge / (2|E|) ||q - 1||^2
       + lambda_transition / (2|T|) ||r||^2.
```

Here the implementation's `b_e` is its fixed, quantized `metric_baseline`, not
the changing integer weight reconstructed from the current `q`.

The following properties are established:

- original directed edges are expanded states, while legal adjacent edge
  pairs are transitions with stable IDs and reversible pair mapping;
- source and target state handling pays the first edge exactly once and
  preserves original OD endpoints;
- each expanded query is bound to the edge weights, transition weights,
  topology, coordinates, and CCH order used to create its metric;
- expanded observed-path cost counts each edge cost and each transition
  residual exactly once;
- at `r=0`, expanded shortest distances and observed-path costs equal their
  edge-only values; reconstructed shortest paths may differ only at ties;
- every feasible edge-only state `(q,0)` is exactly the same state inside the
  expanded feasible set, with the same metric and objective value;
- on the synthetic conflict graph, edge-only costs cannot make two
  conditionally opposed observed routes both strictly shortest, while
  nonnegative transition residuals can.

Synthetic tests enforce these contracts. An ignored, fixed validation-only
Beijing audit remains available for a larger `q=1, r=0` check, but it was not
run during this cleanup. No real-data correctness number or test-data result
is introduced here.

### Unified optimization contract

`ExpandedRoadModel` is optimized as one parameter vector. There is one
expanded training batch query, one pre-update subgradient snapshot, one
`eta0`, one square-root schedule, and one `completed_updates` per joint step.
Scheduled validation queries affect selection only. Both integer metrics are
reconstructed after `q` and `r` have been updated.

The fixed coordinate normalization is

```text
u_e = b_e (q_e - 1)
v_t = residual_scale r_t
eta_k = eta0 / sqrt(completed_updates + 1).
```

For normalized observed-minus-predicted counts `delta_edge_e` and
`delta_transition_t`, the cost-coordinate update is

```text
u_e' = project_[b_e(q_min-1), b_e(q_max-1)](
         u_e - eta_k [delta_edge_e
                      + lambda_edge u_e / (|E| b_e^2)])

v_t' = project_[0, residual_scale r_max](
         v_t - eta_k [delta_transition_t
                      + lambda_transition v_t
                        / (|T| residual_scale^2)]).
```

Equivalently, the raw `q` subgradient is divided by `b_e^2` and the raw `r`
subgradient by `residual_scale^2`. This deterministic diagonal
preconditioning makes `eta0` control movement in common additive cost units;
it does not change the objective, regularization anchors, or projection set.

Expanded checkpoints retain `q`, `r`, both quantized metrics, configuration,
data and initialization identities, expanded-topology identity,
validation-selection state, and one optimizer clock. Restore requires exact
continuous-to-integer metric reconstruction and rejects identity mismatches,
clock discontinuity, implicit clamping, and repair.

Expanded selection uses validation mean regret plus edge and transition
regularization, and logs model-relative regret only as a diagnostic. The
frozen edge-only baseline retains its historical selection convention for
reproducibility; that convention is not a future cross-model gate.

## 2. Not yet established

The repository does not establish that:

- a fairly and sufficiently optimized expanded road model improves over the
  edge-only baseline on real data;
- the expanded model improves independent-data raw objective, Edge F1, or
  Exact Match;
- either active optimization run reached, or came sufficiently close to, a
  global optimum;
- a learned transition residual represents a physical turn cost, a behavioral
  preference, or a causal mechanism;
- any expanded-model gain generalizes to untouched test data, another city,
  another period, or another context;
- the 100-epoch edge-only budget reached optimization convergence;
- dropping cyclic observations is empirically superior to every alternative
  path policy.

No new real-data result is created by the code and concept cleanup. In
particular, expanded-graph correctness and synthetic representational capacity
do not establish a learned real-data advantage.

## 3. Requirements for a future fair evaluation

The only model comparison that remains scientifically relevant is

```text
edge-only baseline
vs.
fully optimized expanded road model.
```

A future preregistered evaluation must:

1. optimize the complete expanded parameter pair `(q,r)` with a defensible
   procedure and budget;
2. use raw mean regret or a denominator fixed identically across both models,
   rather than model-relative regret as the sole ranking metric;
3. report Edge F1 and Exact Match as scale-independent route-reproduction
   measures;
4. separate validation checkpoint selection from independent reporting data;
5. distinguish finite-optimizer failure from a limitation of the expanded
   model class;
6. freeze the full protocol before any untouched-test access.

These are future evidence requirements, not authorization to launch a new
grid, full-data endpoint, or test evaluation.

## 4. Historical archive

The previous Beijing A/B/C experiment is preserved byte-for-byte under
[`experiments/archive/turn_residual_abc_v1/`](../experiments/archive/turn_residual_abc_v1/README.md).
It used separate finite-budget update arms and a model-relative validation
ratio. A later audit found that its optimization budgets, 10-percent subset
asymmetry, changing denominator, and absence of an independent endpoint did
not support ranking those arms. Its former model-selection conclusion is
withdrawn.

The archived configurations, protocol, summaries, hashes, numerical fields,
and execution-time decision strings remain useful only for historical audit.
They do not constrain the active configuration schema, optimizer, checkpoint,
or future research question. No historical arm is an active model class, and
the project will not revisit their ranking as a scientific objective.
