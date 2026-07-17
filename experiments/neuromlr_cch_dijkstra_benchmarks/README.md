# NeuroMLR quality and CCH/Dijkstra efficiency benchmark

This directory is the final benchmark study. Its quality comparison fixes the
true first and last raw road IDs for both methods and scores the complete road
sequence. It never compares the project's native node-to-node recommendation
against NeuroMLR's edge-to-edge task.

Final results are in [`report.md`](report.md); the machine-readable counterpart
is [`summary.json`](summary.json).

The quality protocol is [`protocol.json`](protocol.json), frozen before test at
SHA-256 `c386dab9…`. A hash-bound unlock and completed one-time receipt prove
that the source test pickle was decoded once. The original 50,000-OD efficiency
workload later failed the stronger all-update tie-consistency audit; the quality
protocol and results were not changed. The training workload correction is
isolated in [`efficiency_protocol_amendment.json`](efficiency_protocol_amendment.json).

Project checkpoints are ranked by common edge-to-edge F1 every 25 updates. By
the user's final scope decision, NeuroMLR-Dijkstra is not run; the registered
NeuroMLR checkpoints are ranked by common Greedy F1 every five epochs, and
NeuroMLR-Greedy is the sole external quality baseline.

Project training and the internal CCH/Dijkstra training benchmark use 16 Rayon
threads. This was selected before protocol freeze by sequential, full-data,
ten-update 8-vs-16-thread runs; 16 threads reduced median training-oracle time
from 5289.132 ms to 3514.573 ms while producing identical final weights. See
[`thread_scaling_audit.json`](thread_scaling_audit.json).

Both methods receive the true first and last road IDs. Project edge-to-edge
routing necessarily reaches both fixed states. Greedy rollout keeps the
upstream 300-step and closest-point truncation behavior when it fails to reach
the destination; no destination edge is appended afterward. Such endpoint
failures remain in the common quality metrics and are counted separately.

Large manifests, aligned pickles, checkpoints, raw predictions, and timing
samples live under `generated/` and are intentionally ignored by Git. Their
paths, byte sizes, record counts, and SHA-256 identities are retained in the
tracked audit and summary files.

On the 500-path common test set, project edge-to-edge F1 is `0.766015` and
NeuroMLR-Greedy F1 is `0.768496`; this is close quality, not an improvement.
For the fully consistent 4,971-OD / 20-update workload, CCH is `1.68×` faster
than Dijkstra including CCH preprocessing. On 500 one-thread node-to-node test
queries, query-only inference is `8.05×` faster.

The common route filter is method-independent:

- preserve shapefile-record raw road IDs one-to-one;
- require at least five roads;
- require every adjacent road transition to be continuous;
- drop a trajectory when any original graph node repeats; and
- reject every out-of-map road ID.

The official upstream commit is
`c45e3b5811e5a59b36e4682307d2196c02dac360`. Compatibility adaptations and
their rationale are recorded separately; model dimensions, GCN/MLP structure,
optimizer defaults, and the `L-1` next-road targets are not redefined.
