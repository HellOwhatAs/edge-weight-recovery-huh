# Research status

This document separates verified facts from open scientific questions. The
active repository is organized around one paper line: learn globally shared
road costs from complete historical routes, then extend the same objective with
nonnegative per-transition residuals.

## What is established

### Mathematical and implementation invariants

- The optimized route term is observed-path cost minus current shortest-path
  distance. Count residual is diagnostic only.
- The edge-only parameterization is `w_e = b_e q_e`, with `q` anchored at one,
  updated by projected subgradient descent, and kept as continuous `f64` state.
- Quantized positive integer weights are a separate, explicit CCH metric state.
- Repeated observations with one OD share one oracle query and contribute their
  multiplicity to distances and predicted feature counts.
- Input paths are complete original edge-ID sequences. The mainline validates
  them and drops cyclic observations; it does not trim real first or last
  edges.
- Checkpoint selection is validation-only. The training path does not read
  test.

### Empirical findings

- The projected edge-only method is effective, and anchoring plus box projection
  keep the learned multiplier scale controlled.
- On full Beijing train, the frozen development checkpoint at epoch 99 reaches
  aggregate relative regret `0.06348409`, edge F1 `0.681488`, and exact match
  `0.371068`.
- On the pooled spent AM/PM confirmation blocks it reaches `0.06302821`,
  `0.684512`, and `0.376508`, respectively.
- The same-`eta0=1e-4` development run improves from `0.06826350` at epoch 19
  to `0.06357497` at epoch 99. Twenty epochs were therefore too short.
- With zero turn residuals, the expanded edge-state oracle exactly matched
  edge-only distances and observed costs on all 30,000 checked routes, with
  continuity and endpoint gates passing.
- A frozen-edge, city-wide nonnegative left-turn penalty selected zero. This is
  evidence against that single shared penalty only.

## What is not established

- The 100-epoch budget is not proven converged: the viable runs selected their
  epoch-99 boundary checkpoints.
- No untouched final-test performance claim has been made. The AM/PM blocks
  were validation-derived and are spent.
- Dropping cyclic paths can change the trip population. Existing loop-erasure
  evidence does not establish a superior replacement policy.
- The remaining route errors have not been shown to be caused by omitted turn
  costs.
- The fixed-left result says nothing decisive about heterogeneous,
  transition-specific, contextual, or jointly learned turn effects.
- No formal per-transition residual optimizer or result exists yet.

## Frozen baseline

The paper baseline is the full-train Beijing `EdgeOnlyModel` with:

- projected subgradient descent;
- `eta0=3e-4` and `eta_t = eta0 / sqrt(t)`;
- `lambda_edge=1e5` with `lambda_edge/(2|E|) ||q-1||^2`;
- `q in [0.1, 10]`;
- full CCH customization and unique-OD batching;
- aggregate validation relative regret selection;
- epoch 99 as the selected checkpoint from the bounded 100-epoch run.

Its authoritative compact records are the
[configuration](../experiments/configs/beijing_edge_only_full.json) and
[result summary](../experiments/summaries/beijing_edge_only.json). Retired
studies and complete evidence are recoverable from
`archive/pre-cleanup-convergence-study` at commit
`8aacf2e8020bae13c6fad58f22ccb369f249e029`.

The baseline is frozen for comparison. A future turn-aware study should not
retune it opportunistically or reuse the spent AM/PM blocks.

## Next scientific milestone

The next planned method is a per-transition residual model on the generic
edge-state graph:

```text
kappa_(e,f) = b_f q_f + scale r_(e,f),    r_(e,f) >= 0.
```

The edge multipliers remain anchored at one and transition residuals are
anchored at zero. The edge-only baseline is the exact `r=0` special case. The
next session may implement joint learning of `q` and `r`, using stable
transition IDs and the existing zero-residual equivalence contract.

That milestone should remain narrow: define the residual regularizer and
projected update, preserve validation-only selection, add bounded correctness
and behavior tests, and preregister any new evaluation split. It should not add
fixed turn classes, route-choice heterogeneity, contextual models, broad
statistical tooling, or a new experiment framework.
