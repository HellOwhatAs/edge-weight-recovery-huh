# Experimental evidence

This file distinguishes the trusted edge-only result from the archived,
inconclusive turn-residual study. It is a compact evidence index, not an
experiment diary or authorization for new runs.

## Trusted edge-only baseline

The active baseline learns `w_e = b_e q_e` with projected subgradient descent,
`q in [0.1,10]`, normalized L2 anchoring toward one, full CCH customization,
and one query per unique OD. The route objective is observed-path cost minus
the true shortest distance; count residual is diagnostic only.

The frozen full-Beijing configuration used `eta0=3e-4`,
`lambda_edge=1e5`, and at most 100 epochs. Of 785,709 available training
records, 623,275 complete continuous acyclic paths were accepted. Validation
relative regret selected epoch 99.

| Development routes | Relative regret | Mean regret | Edge F1 | Exact match |
|---:|---:|---:|---:|---:|
| 129,033 | 0.06348409 | 339,523.40 | 0.681488 | 0.371068 |

For the same `eta0=1e-4` trajectory, development relative regret improved from
`0.06826350` at epoch 19 to `0.06357497` at epoch 99. This establishes that the
old 20-epoch horizon was too short. It does not establish convergence: viable
runs selected the epoch-99 budget boundary.

After model selection, two validation-derived, source-index-disjoint AM/PM
blocks were evaluated once. The selected edge checkpoint reported pooled
relative regret `0.06302821`, edge F1 `0.684512`, and exact match `0.376508` on
31,662 accepted routes. These blocks are spent confirmation data, not an
untouched test estimate.

Authoritative active records:

- [full baseline configuration](experiments/configs/beijing_edge_only_full.json)
- [machine-readable baseline summary](experiments/summaries/beijing_edge_only.json)
- [bounded smoke configuration](experiments/configs/smoke_1pct.json)

## Expanded-graph correctness evidence

The generic directed-edge-state expansion was checked on synthetic graphs and
on a fixed validation-only Beijing audit. With all transition residuals zero,
the real-data audit accepted 15,812 routes and found:

- zero shortest-distance mismatches;
- zero observed-path-cost mismatches;
- four reconstructed-path differences at shortest-path ties;
- no test-data read.

This supports the expansion, transition mapping, endpoint handling, metric
binding, and zero-residual equivalence. It does not validate a learned
turn-aware model.

## Archived A/B/C development experiment

The historical study executed its declared 13-cell 10% screen and two
authorized full-data endpoints. Its configurations, raw summaries, protocol,
hashes, and decision fields are preserved without numeric edits under
[`experiments/archive/turn_residual_abc_v1/`](experiments/archive/turn_residual_abc_v1/README.md).

Historical execution facts include:

- the screen ran one expanded-edge continuation cell, six frozen-edge
  turn-only cells, and six simultaneous joint cells for 30 updates;
- six turn-only cells and zero joint cells passed the then-declared gate;
- only full A and full B were run; no full joint endpoint exists;
- both full endpoints selected the step-50 budget boundary;
- all recorded training and development evaluation reported `test_read=false`.

The full development numbers were:

| Historical endpoint | Relative regret | Raw mean regret | Edge F1 | Exact match |
|---|---:|---:|---:|---:|
| A: expanded edge continuation | 0.06203214 | 317,952.34 | 0.682444 | 0.369874 |
| B: frozen-edge turn-only | 0.06041708 | 327,845.80 | 0.693069 | 0.390234 |

The higher F1 and exact match for B are a promising development observation
about route reproduction. They do not establish that B is a winning model or
that transition residuals reduce raw regret; raw mean regret was higher.

The old model-selection interpretation is retired for five reasons:

1. Joint used a particular simultaneous update and only a fixed 30-step screen.
   Since a turn-only state is feasible for the joint continuous model, finite
   optimizer performance cannot show that the joint model class is worse.
2. Historical relative regret divided by each model's own observed-path cost.
   The denominator changes when residuals change the metric, so the ratio is
   not a fair sole cross-model ranking criterion.
3. The 10% screen started from a full-data edge checkpoint. Turn-only preserved
   `q*`, whereas joint modified it using only the subset.
4. No full-data joint run was performed.
5. Development data was used both for checkpoint selection and for reporting.

Consequently, `6/6` versus `0/6`, `winner`, `promoted`, and `continue` remain
auditable protocol facts or raw historical strings, not current scientific
conclusions.

## Metric and data boundaries

Historical relative regret is

```text
sum(regret under current model) / sum(observed cost under current model).
```

It can reproduce selection within one fixed convention, but it is
model-relative. A future fair comparison must use raw mean regret or a shared
fixed denominator, and should report edge F1 and exact match as
cost-scale-independent route-reproduction measures.

Formal training reads train and validation only. The turn study used
development evidence and did not access test. No archived result should be
promoted into an untouched-test claim.

## Recovery points

The immutable commit immediately before this audit is
`6b66eae329b0beea3546550292a4efd789276159`. The local annotated tag
`pre-turn-abc-audit-20260715` points to it. Earlier convergence, scale, loop,
and fixed-turn work remains recoverable through the separate archive described
in [the archive index](experiments/archive/README.md).
