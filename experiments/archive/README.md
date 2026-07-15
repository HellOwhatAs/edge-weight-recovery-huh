# Archived research branches

The active branch contains only compact baseline configurations and results.
The complete pre-cleanup convergence study, scale study, generated summaries,
route-level evidence bundle, loop-policy work, partial-customization benchmark,
legacy Adam ablation, and fixed global left-turn probe remain recoverable at:

- tag: `archive/pre-cleanup-convergence-study`
- commit: `8aacf2e8020bae13c6fad58f22ccb369f249e029`

Inspect a historical file without restoring it:

```bash
git show archive/pre-cleanup-convergence-study:experiments/convergence_study/RESULTS.md
```

Create a separate worktree when executable historical code or the complete
evidence tree is needed:

```bash
git worktree add /tmp/edge-weight-recovery-archive archive/pre-cleanup-convergence-study
```

History was not rewritten or squashed. The active summary at
`experiments/summaries/beijing_edge_only.json` preserves the scientific
conclusions needed by the paper mainline.
