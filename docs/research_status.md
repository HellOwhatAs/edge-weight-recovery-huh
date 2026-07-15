# Research status

This document separates verified facts from open scientific questions. The
active repository follows one paper line: learn globally shared road costs from
complete historical routes, then test the nested addition of nonnegative
per-transition residuals.

## What is established

### Mathematical and implementation invariants

- The optimized route term is observed-path cost minus current shortest-path
  distance. Count residual is diagnostic only.
- The edge-only parameterization is `w_e = b_e q_e`, with `q` anchored at one,
  updated by projected subgradient descent, and kept as continuous `f64` state.
- The turn extension uses
  `kappa_(e,f) = b_f q_f + scale r_(e,f)`, with `r in [0,r_max]` and normalized
  L2 anchoring around zero.
- Quantized positive integer weights are a separate, explicit CCH metric state.
- Repeated observations with one OD share one oracle query and contribute their
  multiplicity to distances and predicted edge/transition counts.
- Input paths are complete original edge-ID sequences. The mainline validates
  them and drops cyclic observations; it does not trim real first or last edges.
- Checkpoint selection is validation-only. The training path does not read test.
- Expanded queries are bound to the exact transition metric and topology used
  by the query. Turn-aware checkpoints atomically bind both latent blocks, both
  integer metrics, both update clocks, configuration, and data identities.

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
  edge-only distances and observed costs on all 15,812 accepted routes in the
  fixed real-data audit. Four predicted paths differed at shortest-path ties;
  continuity and endpoint gates passed.
- A frozen-edge, city-wide nonnegative left-turn penalty selected zero. This is
  evidence against that single shared penalty only.
- The preregistered 10% screen completed exactly 13 cells. All six turn-only
  cells passed the fixed gate; none of the six joint edge-turn cells passed.
- The only authorized full-data endpoints were the expanded-edge continuation
  control and the promoted frozen-edge turn-only configuration. On the same
  development set and fixed 50-update budget, turn-only changed relative regret
  from `0.06203214` to `0.06041708`, edge F1 from `0.682444` to `0.693069`, and
  exact match from `0.369874` to `0.390234`. The corresponding improvements
  (`0.00161506`, `0.01062468`, and `0.02035913`) passed every preregistered gate.

## What is not established

- The 100-epoch edge budget is not proven converged: viable runs selected their
  epoch-99 boundary checkpoints.
- No untouched final-test performance claim has been made. The turn experiment
  used the development split for checkpoint selection and evaluation; the
  earlier AM/PM blocks were validation-derived and are spent.
- Dropping cyclic paths can change the trip population. Existing loop-erasure
  evidence does not establish a superior replacement policy.
- Turn-only development gains show useful transition-specific capacity under
  this protocol; they do not identify physical or causal turn costs or prove
  that turn costs explain most remaining route error.
- Mean regret increased from `317952.34` to `327845.80` even though aggregate
  relative regret and the two path-overlap gates improved. It is therefore not
  an all-metrics improvement.
- Joint edge-turn learning was not promoted, so no positive joint-model claim
  is supported.
- Both full endpoints selected the step-50 budget boundary. The turn study does
  not establish convergence or a global optimum.
- No cross-city, contextual, user-specific, or time-specific result exists.

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
studies and complete evidence are recoverable from immutable commit
`8aacf2e8020bae13c6fad58f22ccb369f249e029`. The annotated
`archive/pre-cleanup-convergence-study` tag is local convenience, not a remote
recovery guarantee.

The baseline remains frozen for comparison. It must not be opportunistically
retuned or evaluated again on the spent AM/PM blocks.

## Current turn candidate and next scientific milestone

The implemented per-transition residual model is:

```text
kappa_(e,f) = b_f q_f + scale r_(e,f),    r_(e,f) >= 0.
```

The development-selected candidate freezes `q=q*`, initializes `r=0`, and uses
`eta_r0=3e-4`, `lambda_turn=1e3`, `scale=127625`, and `r in [0,10]`. It is
recorded in the
[full result summary](../experiments/summaries/beijing_turn_residual_full.json).
The present A/B/C protocol is closed; it authorizes no additional run or
step-50 extension.

The next milestone is evidence separation, not another feature branch: freeze
this candidate and preregister an independent evaluation before any untouched
test access. A future one-shot final evaluation or genuinely external
replication should compare only the frozen edge baseline and frozen turn-only
candidate. It must not reopen the grid, reuse the spent AM/PM blocks, or add
fixed turn classes, route-choice heterogeneity, contextual models, broad
statistical tooling, or a new experiment framework.
