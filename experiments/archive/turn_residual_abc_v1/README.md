# Archived nonnegative per-transition residual A/B/C study

This directory preserves the first Beijing A/B/C study of nonnegative
per-transition residuals. The protocol was preregistered and executed as
declared. Its machine-readable protocol, configurations, and summaries are
retained without changing their JSON contents.

The immutable pre-audit recovery point is commit
`6b66eae329b0beea3546550292a4efd789276159`. The local annotated tag
`pre-turn-abc-audit-20260715` points to that commit as a convenience; the
commit SHA is the recovery authority because a local tag is not guaranteed to
exist in another clone.

## Original path mapping

| Original active path | Archived path |
|---|---|
| `experiments/turn_residual_protocol.json` | `experiments/archive/turn_residual_abc_v1/turn_residual_protocol.json` |
| `experiments/summaries/beijing_turn_residual_10pct.json` | `experiments/archive/turn_residual_abc_v1/summaries/beijing_turn_residual_10pct.json` |
| `experiments/summaries/beijing_turn_residual_full.json` | `experiments/archive/turn_residual_abc_v1/summaries/beijing_turn_residual_full.json` |
| `experiments/configs/turn_*.json` | `experiments/archive/turn_residual_abc_v1/configs/turn_*.json` |

The archived configuration set contains one correctness smoke configuration,
thirteen 10-percent screening configurations, and two full-data endpoint
configurations. Paths embedded inside the preserved JSON record their original
execution-time locations; they are historical provenance, not active
recommendations.

## What was run

- The correctness gates and checkpoint-identity checks recorded in the
  protocol passed before screening.
- The 10-percent screen ran exactly thirteen declared cells for 30 updates
  each: one expanded-edge continuation control, six frozen-edge turn-only
  cells, and six simultaneous joint edge-turn cells.
- Under the declared gate, all six turn-only cells passed and none of the six
  joint cells passed. These are protocol outcomes, not a valid ranking of the
  two model classes.
- The protocol consequently ran two 50-update full-data endpoints: expanded
  edge continuation A and frozen-edge turn-only B. No full-data joint endpoint
  was run.
- Both full endpoints selected the step-50 budget boundary. On development
  data, B had lower model-relative regret and higher edge F1 and exact match
  than A, while its raw mean regret was higher (`327845.7964009207` versus
  `317952.3393472988`).
- Training and endpoint evaluation did not read test data.

## Why the model-selection conclusion is retired

A later audit found that the study supports preservation of its observations,
but not its original model ranking or final-candidate decision:

1. The continuous joint model contains every frozen-edge turn-only state as a
   feasible special case. The fixed 30-step simultaneous block-update result
   shows only that this particular finite-budget optimization did not pass the
   gate; it does not show that the joint model is worse or ineffective.
2. Checkpoint selection and the primary gate used model-relative regret. Its
   observed-cost denominator is recomputed under each model, so nonnegative
   residuals can change the denominator. It is not a fair sole metric for
   ranking edge-only, turn-only, and joint models. The lower relative regret
   for full B coincided with higher raw mean regret.
3. The initialization `q*` came from full-data edge-only training, but screening
   used only a 10-percent training subset. Turn-only retained that full-data
   representation while joint fine-tuning modified it using the subset, making
   the screen asymmetric for judging joint learning.
4. Only A and B were run on full training data. There is no full-data joint
   result from which to infer that joint learning is worse than turn-only or
   ineffective on full data.
5. The development split was used both for checkpoint selection and reported
   endpoint metrics. The study is development evidence, not independent
   confirmation, untouched-test evidence, or a generalization result.

Accordingly, the former conclusions that frozen-edge turn-only was the winning
or final candidate, that joint learning failed, or that the protocol established
a definitive improvement in shortest-path regret are retired. The preserved
results may be cited only as an inconclusive historical experiment and as a
development observation that transition residuals may improve scale-independent
route-reproduction metrics such as edge F1 and exact match.

Future fair evaluation would need raw mean regret or a denominator shared by
all models, scale-independent route metrics, adequate and symmetric optimization
of joint learning, and an independent evaluation split. This archive does not
authorize or recommend another run.

Inspect the original active tree without changing the current worktree:

```bash
git show 6b66eae329b0beea3546550292a4efd789276159:experiments/turn_residual_protocol.json
```

Create a separate worktree when the original paths are required:

```bash
git worktree add /tmp/edge-weight-recovery-turn-abc-v1 6b66eae329b0beea3546550292a4efd789276159
```
