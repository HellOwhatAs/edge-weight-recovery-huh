# Current trusted experimental results

This file records only the evidence used by the paper mainline. It is not an
experiment diary. The complete pre-cleanup studies, generated evidence, and
retired executables remain available at:

- archive tag: `archive/pre-cleanup-convergence-study`
- archive commit: `8aacf2e8020bae13c6fad58f22ccb369f249e029`

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

The generic edge-state expansion was checked on a deterministic 30,000-route
development subset. With every transition residual set to zero, all expanded
shortest distances and observed-path costs exactly matched the edge-only
oracle; path continuity and endpoint correctness gates also passed. This is
the retained equivalence evidence needed for a future per-transition residual
model.

A narrow historical probe then froze the learned edge weights and added one
shared nonnegative left-turn penalty. On its 20,000-route tuning subset, the
penalty grid `0, 0.05, 0.1, 0.2, 0.4` produced relative regrets
`0.06341452, 0.06378618, 0.06430391, 0.06559292, 0.06892272`; the predefined
rule selected zero. The improvement gates failed, so the reserved audit block
was not promoted to a new confirmation claim.

This result rejects only a single city-wide nonnegative left-turn penalty with
frozen edge weights. It does not reject per-transition residuals, joint
edge/turn learning, or other conditional turn models. The fixed-left probe is
archived and is not part of the formal method or active training path.

## Data and test firewall

- Pickle paths are complete original edge-ID sequences; real boundary edges
  are never trimmed.
- The sole mainline policy validates continuity and edge IDs, then drops cyclic
  observations under the positive-cost shortest-path assumption.
- Checkpoint selection reads validation only and minimizes aggregate validation
  relative regret.
- Formal training, diagnostics, the one-shot confirmations, and the turn probe
  did not load or evaluate test. No test metric is reported here.
- Confirmation artifacts are spent; future method development requires a new,
  preregistered validation protocol before any final held-out evaluation.

## Limitations

- The frozen result is a strong Beijing validation baseline, not a final
  held-out generalization estimate.
- Selecting epoch 99 leaves optimization convergence unresolved.
- Dropping cyclic observations changes the represented trip population; loop
  erasure was not established as a better frozen-policy alternative.
- Edge-only weights cannot represent transition-specific costs.
- Zero-residual equivalence verifies the expanded oracle boundary case, not the
  effectiveness of a learned turn-aware model.
- The fixed-left negative result has deliberately narrow scope.

## Compact reproducibility pointers

- [Frozen full configuration](experiments/configs/beijing_edge_only_full.json)
- [Machine-readable baseline summary](experiments/summaries/beijing_edge_only.json)
- [Bounded 1% smoke configuration](experiments/configs/smoke_1pct.json)
- [Cleanup inventory and protected-state record](docs/repository_cleanup_inventory.md)
- [Research status and next milestone](docs/research_status.md)

Large generated JSON, route-level derivatives, detailed historical reports,
and retired experiment code are intentionally absent from
the active tree. Use the archive tag when historical audit detail is required.
