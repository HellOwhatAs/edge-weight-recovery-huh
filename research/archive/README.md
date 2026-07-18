# Historical research archive

This directory preserves reports, configurations, evidence, and orchestration
from the pre-workspace repository. They document past decisions but are not
members of either Cargo workspace and are not production entry points.

Paths written inside historical reports reflect their original repository
locations. The corresponding files now sit below `research/archive` with the
same relative `experiments/` or `scripts/` suffix. Large generated artifacts
remain untracked under their original ignored locations.

New experiments should be ordinary crates in the research workspace or
independent packages under `baselines`, and should exchange only the v1 dataset,
prediction, and run-receipt protocol. Do not revive archived configuration
switches in a production crate.
