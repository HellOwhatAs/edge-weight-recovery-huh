# Retired monolithic Rust implementation

The pre-workspace implementation is preserved by Git at commit `0f4bd55` on
the former `neuromlr-cch-dijkstra-benchmarks` branch. It mixed production
training, two graph representations, two routing backends, evaluation,
benchmark loops, time-bucket studies, manifest generation, and CLI concerns in
one package.

Use `git show 0f4bd55:src/<file>` when historical source is needed. The active
replacement is split across `crates/ewr-core`, `crates/ewr-cch`,
`crates/ewr-io`, and `crates/ewr-cli`; research-only Dijkstra and interchange
protocol code live in the independent `research` workspace.

The old files were removed only after the production workspace passed frozen
v1 tests for line-graph mapping, integer route selection with direct-float path
evaluation, the relative one-step update, CCH routing, and bitwise-equivalent
resume.
