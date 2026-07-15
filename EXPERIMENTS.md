# Experimental evidence

This file indexes the evidence that supports the active two-model research
program. It is not an experiment diary and does not authorize new data access
or runs.

## Established: edge-only baseline

The edge-only baseline learns

```text
w_e = b_e q_e

J_edge(q) = mean route regret
          + lambda_edge / (2|E|) ||q - 1||^2
```

with projected subgradient descent, `q in [0.1,10]`, full CCH customization,
and one query per unique OD. The route loss is observed-path cost minus true
shortest distance; count residual is diagnostic only.

The frozen full-Beijing configuration used `eta0=3e-4`,
`lambda_edge=1e5`, and at most 100 epochs. Of 785,709 available training
records, 623,275 complete continuous acyclic paths were accepted. Validation
relative regret selected epoch 99.

| Development routes | Relative regret | Raw mean regret | Edge F1 | Exact Match |
|---:|---:|---:|---:|---:|
| 129,033 | 0.06348409 | 339,523.40 | 0.681488 | 0.371068 |

For the same `eta0=1e-4` trajectory, development relative regret improved from
`0.06826350` at epoch 19 to `0.06357497` at epoch 99. This establishes that the
old 20-epoch horizon was too short, but not that 100 epochs reached
convergence.

After model selection, two validation-derived, source-index-disjoint AM/PM
blocks were evaluated once. The selected checkpoint reported pooled relative
regret `0.06302821`, Edge F1 `0.684512`, and Exact Match `0.376508` on 31,662
accepted routes. These blocks are spent confirmation evidence, not an
untouched test estimate.

Authoritative active records:

- [full baseline configuration](experiments/configs/beijing_edge_only_full.json)
- [machine-readable baseline summary](experiments/summaries/beijing_edge_only.json)
- [bounded smoke configuration](experiments/configs/smoke_1pct.json)

## Established: expanded-road correctness and expressiveness

The expanded model is

```text
kappa_(e,f) = b_f q_f + residual_scale r_(e,f),    r_(e,f) >= 0

J_expanded(q,r) = mean route regret
                + lambda_edge / (2|E|) ||q - 1||^2
                + lambda_transition / (2|T|) ||r||^2.
```

In code, `b_e` is the fixed `metric_baseline` after the configured first-stage
quantization; it is not the changing reconstructed edge weight.

Existing synthetic tests establish stable transition mapping, source/target
handling, metric binding, observed-cost accounting, and exact nesting of the
edge-only state `(q,0)`. At `r=0`, edge-only and expanded observed costs and
shortest distances are identical; decoded routes may differ at ties.

An ignored, fixed validation-only Beijing audit remains available for a larger
`q=1, r=0` correctness check. It was not run during this cleanup and therefore
contributes no new real-data number. Its split and variant are compile-time
validation constants, and it cannot be redirected to test data.

The synthetic conflict graph also establishes a strict representation result:
one shared set of edge costs cannot make two conditionally opposed observed
routes both uniquely shortest, while nonnegative transition residuals can.

These are correctness, nesting, and representational-capacity results. They do
not establish that learned transition residuals improve a real-data endpoint.

## Active expanded training contract

`ExpandedRoadModel` has one training path. Each optimizer update makes one
expanded training batch query, derives edge and transition subgradients from
the same pre-update metric, updates `(q,r)` jointly, reconstructs both integer
metrics, and advances one global clock. Scheduled validation queries are used
only for checkpoint selection:

```text
eta_k = eta0 / sqrt(completed_updates + 1).
```

To express both blocks in the same additive cost units, the optimizer uses

```text
u_e = b_e (q_e - 1)
v_t = residual_scale r_t.
```

With normalized observed-minus-predicted counts `delta_edge_e` and
`delta_transition_t`, its update is

```text
u_e' = project_[b_e(q_min-1), b_e(q_max-1)](
         u_e - eta_k [delta_edge_e
                      + lambda_edge u_e / (|E| b_e^2)])

v_t' = project_[0, residual_scale r_max](
         v_t - eta_k [delta_transition_t
                      + lambda_transition v_t
                        / (|T| residual_scale^2)]).
```

This fixed coordinate normalization is equivalent to dividing the raw `q`
gradient by `b_e^2` and the raw `r` gradient by `residual_scale^2`. It preserves
the stated objective and box constraints while avoiding two tunable learning
rates or two decay clocks.

The active expanded checkpoint selector is validation mean regret plus both
regularization terms. Model-relative regret is logged only as a diagnostic.

## Current bounded 10-percent development comparison

The first active two-model development comparison used exactly the same
deterministic Beijing 10-percent train subset and fixed validation split for
both models:

```text
train       scale_10pct_seed42       78,570 raw / 62,348 accepted
validation  scale_fixed_seed20260715 20,000 raw / 15,812 accepted
cycle policy                         drop
Rayon threads                        4
test read                            false
```

Three learning rates were reported for each model. Raw validation mean regret
selected `eta0=2e-4` for edge-only and `eta0=16000` for expanded. Each selected
configuration ran a main budget and, because its best state was at the
boundary, exactly one extension from the default regularization center. The
hard caps were 300 edge epochs and 600 expanded updates.

| Model | Best state | Raw mean regret | Edge F1 | Exact Match |
|---|---:|---:|---:|---:|
| edge-only | 289 | 310,343.73 | 0.682145 | 0.368454 |
| expanded | 600 | 619,093.64 | 0.590588 | 0.314318 |

The finite edge-only checkpoint is better on all three fixed metrics. This is
not a model-class win: expanded selected the hard-cap boundary and improved at
every cadence through update 600. Its q and r blocks and both quantized metrics
continued to change without numerical failure, upper-bound saturation, or
quantization stall. The evidence therefore falls in category F—expanded is
still underoptimized, so this run cannot distinguish optimizer shortfall from
model value.

At the expanded best state, setting r to zero while retaining learned q made
raw mean regret 9,102.34 worse and reduced both F1 and Exact Match. This
evaluation-only diagnostic shows that the learned residuals help their current
q state; it is not a third training model.

Active machine-readable records:

- [finite calibration](experiments/summaries/beijing_10pct_calibration.json)
- [edge-only final run](experiments/summaries/beijing_edge_only_10pct.json)
- [expanded final run](experiments/summaries/beijing_expanded_10pct.json)
- [unified comparison](experiments/summaries/beijing_10pct_model_comparison.json)
- [edge-only configuration](experiments/configs/beijing_edge_only_10pct.json)
- [expanded configuration](experiments/configs/beijing_expanded_10pct.json)

## Not yet established

The bounded development result does not show that either model class is
intrinsically better after sufficient optimization. In particular, the
repository has not established:

- lower independent-data raw mean objective;
- higher independent-data Edge F1 or Exact Match;
- optimization sufficiently close to a global optimum;
- physical, behavioral, or causal meaning for learned residuals;
- generalization beyond the development setting.

`relative_regret` is

```text
sum(regret under current model) / sum(observed cost under current model).
```

Its denominator changes with the model, so it is not a fair sole cross-model
gate. The frozen edge-only baseline retains this selection convention for
reproducibility; expanded training uses validation mean regret plus
regularization and logs the ratio only as a diagnostic. The only future
comparison is

```text
edge-only baseline
vs.
fully optimized expanded road model.
```

That comparison must use raw mean regret or a shared fixed denominator, report
Edge F1 and Exact Match, and separate checkpoint selection from independent
reporting. No test result belongs to the bounded development comparison above.

## Historical archive

The former A/B/C development study is retained without changes under
[`experiments/archive/turn_residual_abc_v1/`](experiments/archive/turn_residual_abc_v1/README.md).
Its sixteen configurations, protocol, two summaries, provenance hashes,
numerical results, and execution-time decision fields are historical audit
material only.

The study's model-ranking conclusion was withdrawn after audit. Its
finite-budget arms, model-relative gate, asymmetric 10-percent fine-tuning,
lack of a complete independent endpoint, and reuse of development data for
selection and reporting cannot adjudicate the active nested model comparison.
Historical arm outcomes are therefore not current evidence, active model
categories, or future research questions.

The immutable pre-audit recovery point is
`6b66eae329b0beea3546550292a4efd789276159`. Earlier convergence, scale, loop,
and exploratory work remains recoverable through
[the archive index](experiments/archive/README.md).
