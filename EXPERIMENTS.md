# Current trusted experimental results

This file records only the evidence used by the paper mainline. It is not an
experiment diary. The complete pre-cleanup studies, generated evidence, and
retired executables remain available at immutable archive commit
`8aacf2e8020bae13c6fad58f22ccb369f249e029`. The local annotated tag
`archive/pre-cleanup-convergence-study` is a convenience, not a published
recovery guarantee.

History was not rewritten. See the [archive pointer](experiments/archive/README.md)
for commands that inspect or restore historical material.

## Frozen edge-only baseline

The formal baseline learns one global multiplier per directed road edge,

```text
w_e = b_e q_e,
```

with projected subgradient descent, `eta_t = eta0 / sqrt(t)`,
`q in [0.1, 10]`, and anchoring
`lambda_edge / (2 |E|) ||q - 1||^2`. The route term is true shortest-path
regret. Count residual is a diagnostic, not the optimization loss. Latent
`q` values remain `f64`; positive integer CCH weights are produced by an
explicit quantization step. The oracle uses full CCH customization and queries
each unique OD once per evaluation pass.

The frozen Beijing configuration uses all 785,709 available training records;
623,275 complete, continuous, acyclic paths remain after validation and the
positive-cost cycle filter. It uses `eta0=3e-4`, `lambda_edge=1e5`, at most 100
epochs, and aggregate validation relative regret for checkpoint selection.

On the 129,033 accepted routes in the development block, the selected
checkpoint was epoch 99 (zero-based):

| Relative regret | Mean regret | Exact match | Edge precision | Edge recall | Edge F1 | Edge Jaccard |
|---:|---:|---:|---:|---:|---:|---:|
| **0.06348409** | 339,523.40 | 0.371068 | 0.692053 | 0.679015 | **0.681488** | 0.606961 |

Here relative regret is the aggregate ratio
`sum(route regret) / sum(observed route cost)`.

## Bounded horizon evidence

The exact same-`eta0=1e-4` full-data trajectory improved from relative regret
`0.06826350` at epoch 19 to `0.06357497` at epoch 99. This establishes that the
old 20-epoch budget was insufficient. The ultimately selected checkpoint used
`eta0=3e-4` and reached `0.06348409`, so its advantage over the frozen 20-epoch
control is not a same-learning-rate estimate of the epoch effect.

Both viable 100-epoch trajectories selected the final checkpoint at epoch 99.
Consequently, the study does **not** establish convergence at 100 epochs; it
only freezes the strongest validated edge-only checkpoint obtained under the
bounded protocol.

## Spent AM/PM confirmation

After model selection was frozen, `q=1`, the 20-epoch control, and the selected
100-epoch model were evaluated once on two later, source-index-disjoint blocks.
After the unchanged cycle filter, AM contained 15,776 routes and PM contained
15,886 routes.

| Block | Candidate | Relative regret | Edge F1 | Exact match |
|---|---|---:|---:|---:|
| AM | `q=1` | 0.09210652 | 0.599664 | 0.353892 |
| AM | 20 epochs | 0.06724488 | 0.662546 | 0.356237 |
| AM | selected 100 epochs | **0.06307110** | **0.686270** | **0.376331** |
| PM | `q=1` | 0.09285387 | 0.593518 | 0.343573 |
| PM | 20 epochs | 0.06767404 | 0.659278 | 0.353204 |
| PM | selected 100 epochs | **0.06298240** | **0.682767** | **0.376684** |
| Pooled | `q=1` | 0.09246791 | 0.596580 | 0.348715 |
| Pooled | 20 epochs | 0.06745259 | 0.660906 | 0.354715 |
| Pooled | selected 100 epochs | **0.06302821** | **0.684512** | **0.376508** |

The selected checkpoint improved over the 20-epoch candidate on both blocks.
These are validation-derived confirmation blocks, not an untouched test set,
and they are now spent. They must not be reused for model selection or for a
new confirmation claim.

## Turn-graph evidence and scope

The generic edge-state expansion was checked on a fixed 20,000-record
development sample, of which 15,812 complete acyclic routes were accepted.
With every transition residual set to zero, all expanded shortest distances and
observed-path costs exactly matched the edge-only oracle. Four reconstructed
paths differed at shortest-path ties; continuity and endpoint gates passed.

A narrow historical probe then froze the learned edge weights and added one
shared nonnegative left-turn penalty. On its 20,000-route tuning subset, the
penalty grid `0, 0.05, 0.1, 0.2, 0.4` produced relative regrets
`0.06341452, 0.06378618, 0.06430391, 0.06559292, 0.06892272`; the predefined
rule selected zero. The improvement gates failed, so the reserved audit block
was not promoted to a new confirmation claim.

This result rejects only a single city-wide nonnegative left-turn penalty with
frozen edge weights. The fixed-left probe is archived and is not part of the
formal method or active training path.

## Nonnegative per-transition residual study

The formal study used the same expanded graph for all arms:

- A continued the edge multipliers with residuals fixed at zero;
- B froze the epoch-99 edge multipliers and learned transition residuals;
- C updated both blocks from the same pre-update counts.

All arms started from the same frozen `q*` and fresh `r=0`. Edge updates
continued the original step-size clock; residual updates used an independent
clock. Checkpoints were selected only by aggregate development relative regret.

The fixed 10% screen completed exactly 13 declared cells: one A control, six B
cells, and six C cells. All six B cells passed the preregistered gate; no C cell
passed. The winning B cell used `eta_r0=3e-4` and `lambda_turn=1e3`. Therefore
only A and this B configuration were authorized for the full-train endpoint.

| Full-train development endpoint | Best step | Relative regret | Mean regret | Edge F1 | Exact match |
|---|---:|---:|---:|---:|---:|
| A: expanded edge continuation | 50 | 0.06203214 | 317,952.34 | 0.682444 | 0.369874 |
| B: frozen-edge turn-only | 50 | **0.06041708** | 327,845.80 | **0.693069** | **0.390234** |

B passed all three preregistered full gates relative to A: relative-regret gain
`0.00161506 >= 0.0005`, edge-F1 gain `0.01062468 >= 0.003`, and exact-match
change `+0.02035913 >= -0.002`. Mean regret increased rather than decreased, so
the result must not be summarized as improvement on every metric. Both selected
step 50, leaving convergence unresolved. The protocol is complete and
authorizes no further run.

The supported claim is narrow: on this Beijing development protocol, learned
nonnegative per-transition residuals with frozen edge multipliers provide
additional route-fit capacity beyond the equal-budget expanded-edge control.
This is not an untouched-test, independent-confirmation, causal-turn-cost,
joint-model, or cross-city result.

## Data and test firewall

- Pickle paths are complete original edge-ID sequences; real boundary edges
  are never trimmed.
- The sole mainline policy validates continuity and edge IDs, then drops cyclic
  observations under the positive-cost shortest-path assumption.
- Checkpoint selection reads validation only and minimizes aggregate validation
  relative regret.
- Formal training, diagnostics, the one-shot confirmations, the historical turn
  probe, and the per-transition study did not load or evaluate test. No test
  metric is reported here.
- Confirmation artifacts are spent; future method development requires a new,
  preregistered validation protocol before any final held-out evaluation.

## Limitations

- The frozen result is a strong Beijing validation baseline, not a final
  held-out generalization estimate.
- Selecting epoch 99 leaves optimization convergence unresolved.
- Dropping cyclic observations changes the represented trip population; loop
  erasure was not established as a better frozen-policy alternative.
- The full turn endpoints both selected their fixed step-50 boundary, so
  optimization convergence is not established.
- Turn-only passed its development gates, but mean regret increased and no
  joint edge-turn cell passed screening.
- Learned residuals are predictive parameters, not identified physical or
  causal turn costs.
- The fixed-left negative result has deliberately narrow scope.

## Compact reproducibility pointers

- [Frozen full configuration](experiments/configs/beijing_edge_only_full.json)
- [Machine-readable baseline summary](experiments/summaries/beijing_edge_only.json)
- [Turn-study protocol](experiments/turn_residual_protocol.json)
- [10% turn screen summary](experiments/summaries/beijing_turn_residual_10pct.json)
- [Full turn endpoint summary](experiments/summaries/beijing_turn_residual_full.json)
- [Bounded 1% smoke configuration](experiments/configs/smoke_1pct.json)
- [Cleanup inventory and protected-state record](docs/repository_cleanup_inventory.md)
- [Research status and next milestone](docs/research_status.md)

Large generated JSON, route-level derivatives, detailed historical reports,
and retired experiment code are intentionally absent from the active tree. Use
the immutable archive commit when historical audit detail is required.
