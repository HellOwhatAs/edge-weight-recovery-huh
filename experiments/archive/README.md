# Archived research branches

The active branch contains only compact baseline configurations and results.
The complete pre-cleanup convergence study, scale study, generated summaries,
route-level evidence bundle, loop-policy work, partial-customization benchmark,
legacy Adam ablation, and fixed global left-turn probe remain recoverable at
immutable commit `8aacf2e8020bae13c6fad58f22ccb369f249e029`.

The earlier cleanup recorded an annotated tag named
`archive/pre-cleanup-convergence-study` in its original workspace. That local
tag is not guaranteed to exist in another clone, so recovery instructions use
the immutable commit instead.

## Nonnegative per-transition residual A/B/C v1

The later Beijing A/B/C protocol, all sixteen associated turn configurations,
and both result summaries are preserved in the
[turn-residual A/B/C v1 archive](turn_residual_abc_v1/README.md). That study was
executed as declared, but a subsequent audit found that its model-relative
selection metric and 10-percent fine-tuning design do not support ranking
frozen-edge turn-only against joint learning. Its raw results remain available
for audit, while its model-selection conclusion is retired.

The immutable pre-audit recovery point for that study is
`6b66eae329b0beea3546550292a4efd789276159`. The local annotated tag
`pre-turn-abc-audit-20260715` is a convenience; recovery instructions use the
commit SHA.

Inspect a historical file without restoring it:

```bash
git show 8aacf2e8020bae13c6fad58f22ccb369f249e029:experiments/convergence_study/RESULTS.md
```

Create a separate worktree when executable historical code or the complete
evidence tree is needed:

```bash
git worktree add /tmp/edge-weight-recovery-archive 8aacf2e8020bae13c6fad58f22ccb369f249e029
```

History was not rewritten or squashed. The active summary at
`experiments/summaries/beijing_edge_only.json` preserves the scientific
conclusions needed by the paper mainline.
