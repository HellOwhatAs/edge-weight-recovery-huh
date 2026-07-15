# Research status

This document is the claim boundary for the active repository. It separates
the reliable edge-only method, the implemented turn-aware extension, and an
inconclusive historical A/B/C experiment whose raw evidence is retained only
for audit.

## 1. Established

### Edge-only method

- The main objective uses shortest-path regret: observed-path cost minus the
  shortest distance under the current metric. Count residual is diagnostic
  only.
- The model is `w_e = b_e q_e`. Continuous `q` is regularized toward one,
  updated by projected subgradient descent, and kept in a positive box.
- Continuous latent state is separate from the positive integer CCH metric;
  quantization is explicit and checked against the CCH infinity sentinel.
- Production queries use full CCH customization and unique-OD batching. OD
  multiplicity weights distances and predicted counts correctly.
- Checkpoint selection is validation-only. Training never reads test data.
- Checkpoints bind configuration, `q`, quantized weights, selection state, and
  data/baseline identity, and restore checks reproduce the integer metric.
- Complete input paths are validated for edge IDs, continuity, endpoints, and
  cycles. The sole active policy drops cyclic observations.

The frozen full-Beijing baseline selected epoch 99 in a bounded 100-epoch run.
On 129,033 accepted development routes it reported relative regret
`0.06348409`, mean regret `339523.40`, edge F1 `0.681488`, and exact match
`0.371068`. On pooled validation-derived AM/PM confirmation blocks it reported
relative regret `0.06302821`, edge F1 `0.684512`, and exact match `0.376508`.
Those blocks are spent confirmation evidence, not untouched test data.

### Expanded-graph correctness

- Original directed edges are expanded states; legal adjacent edge pairs are
  transitions with stable IDs and reversible `(previous_edge,next_edge)`
  mapping.
- Source and target state handling pays the first edge exactly once and
  preserves original OD endpoints.
- An expanded query is bound to the exact edge weights, transition weights,
  topology, coordinates, and CCH order used to create its metric.
- At zero transition residual, expanded shortest distances and observed-path
  costs equal edge-only values. Synthetic tests cover the invariant. A fixed
  validation-only Beijing audit found zero distance and observed-cost
  mismatches on 15,812 accepted routes; four reconstructed paths differed at
  shortest-path ties.

## 2. Implemented but not yet fairly validated

The generic per-transition model is implemented as

```text
kappa_(e,f) = b_f q_f + scale r_(e,f),    r_(e,f) >= 0.
```

The repository retains:

- `TurnAwareModel` and continuous per-transition residuals;
- residual regularization toward zero and projection to `[0,r_max]`;
- observed and predicted edge and transition counts;
- correct expanded observed cost and turn-aware regret;
- turn-only and simultaneous joint block updates using one pre-update count
  snapshot;
- independent edge and residual update clocks;
- atomic checkpoints containing `q`, `r`, both integer metrics, configuration,
  data identity, initialization identity, and expanded-topology identity;
- frozen-block, clock, quantization, topology, and checkpoint round-trip tests.

For the same continuous objective and constraints, every frozen-edge turn-only
state is a feasible joint state: joint permits `q` to remain at `q*` while
using the same nonnegative `r`. This is a feasible-set statement, not a claim
that a particular finite-step joint optimizer must find the turn-only state.

One historical development comparison observed higher edge F1 and exact match
after learning residuals with frozen `q`. This suggests possible improvement in
route reproduction. The appropriate current status is:

> implemented, correctness-checked, promising but not yet fairly validated

## 3. Inconclusive historical experiment

The previous A/B/C study was executed as declared, but later audit found that
its model-relative selection metric and 10-percent fine-tuning design do not
support ranking frozen-edge turn-only against joint learning. Its raw results
are preserved for audit, but its model-selection conclusion is retired.

The historical arms were:

- A: expanded-graph edge continuation with `r=0`;
- B: frozen-edge turn-only residual updates;
- C: simultaneous joint edge and residual updates.

The 10% screen ran exactly 13 declared cells for 30 updates: one A, six B, and
six C. Under the then-preregistered gate, six B cells and zero C cells passed.
This is an execution fact, not evidence that the joint model failed. It says
only that the tested 30-step simultaneous-update configurations did not pass
that historical gate.

Only full A and full B were run; there is no full-data joint result. Both
selected their fixed step-50 boundary:

| Historical development endpoint | Relative regret | Raw mean regret | Edge F1 | Exact match |
|---|---:|---:|---:|---:|
| A: expanded edge continuation | 0.06203214 | 317,952.34 | 0.682444 | 0.369874 |
| B: frozen-edge turn-only | 0.06041708 | 327,845.80 | 0.693069 | 0.390234 |

The selection ratio was

```text
sum(model regret) / sum(model observed-path cost).
```

Its denominator changes with the model. Nonnegative residuals can increase
observed-path costs, so this model-relative ratio is not a fair sole ranking
metric across A/B/C. The lower historical ratio for B coincided with higher raw
mean regret, demonstrating that the old gate did not establish lower absolute
shortest-path regret.

The 10% design was also asymmetric. Its `q*` came from the full training data.
B retained that full-data representation while fitting only new residuals on
the subset; C modified the full-data `q*` using only the subset. That design
can damage an existing edge representation and cannot adjudicate a
sufficiently optimized joint model.

Finally, the development split served both checkpoint selection and reported
evaluation. The result is development evidence, not independent confirmation
or untouched-test evidence.

All original numeric fields, configurations, provenance hashes, gates, and
historical decision strings remain byte-for-byte in
[`experiments/archive/turn_residual_abc_v1/`](../experiments/archive/turn_residual_abc_v1/README.md).
Strings such as `winner`, `promoted`, or `continue` inside those raw JSON files
describe the protocol's decision at execution time; they are not current
scientific conclusions.

## 4. Not established

The repository does not currently establish that:

- frozen-edge turn-only is better than joint learning;
- joint learning is ineffective or worse than turn-only;
- transition residuals reduce raw mean shortest-path regret;
- either turn-aware optimizer was run to convergence or a global optimum;
- transition residual gains generalize to independent data, untouched test,
  another city, or another context;
- learned residuals are physical, behavioral, or causal turn costs;
- the 100-epoch edge-only budget reached optimization convergence;
- dropping cyclic paths is empirically superior to every alternative path
  policy.

## 5. Future evaluation requirements

Before ranking turn-only against joint learning, a future preregistered study
must:

1. use raw mean regret or a denominator fixed identically across all models;
2. report edge F1 and exact match as cost-scale-independent route-reproduction
   metrics rather than treating model-relative regret as the sole gate;
3. give joint learning a defensible optimization procedure and budget, and
   distinguish optimizer failure from model-class failure;
4. remove the 10% asymmetry around a full-data `q*`, or otherwise control edge
   drift fairly across arms;
5. include a full-data joint endpoint or another comparison that genuinely
   evaluates the nested feasible sets;
6. separate checkpoint selection from independent reporting data;
7. freeze the complete protocol before any untouched-test access.

These requirements are a future evaluation agenda, not authorization to run a
new grid or endpoint. The reliable main method remains the frozen edge-only
baseline. The turn-aware implementation remains available for correctness work
and a later fair validation design.
