# Historical research archive

This is a thin, read-only record of decisions made before the Cargo workspace
split. Nothing below this directory is a Cargo workspace member, supported CLI
input, or current experiment entry point. Words such as "current" or "active"
inside a historical report describe the state at the report's original commit.

Start with [`decisions.md`](decisions.md). It records which ideas were adopted,
superseded, or rejected and links to the evidence that is still useful.

| Study | Status | Retained evidence |
| --- | --- | --- |
| 10% direct-weight line-graph calibration | superseded | report and machine summary |
| relative-optimizer recovery | adopted by the frozen v1 algorithm | report, summary, and final configs |
| independent departure-time buckets | rejected | report, summary, audit, and final configs |
| shared temporal residual/travel-time proxy | rejected | report, summary, and audit |
| fixed-point CCH decimal precision | superseded | report and machine summary |
| NeuroMLR quality and CCH/Dijkstra efficiency | final historical benchmark | report, protocol, audits, environment, and summary |

Old orchestration scripts, screening configs, copied raw artifacts, and stale
commands are deliberately not retained here. They targeted the retired
monolithic CLI and old checkpoint schemas and would mislead new development.
The complete pre-workspace tree remains recoverable from Git commit `0f4bd55`;
earlier scale and convergence studies remain in the history preceding retirement
commit `f0ef941`.

Large checkpoints, manifests, predictions, and timing samples are runtime
artifacts and stay outside Git. Only selected compatibility fixtures may remain
under ignored `artifacts/retained/`. Their presence is optional and must never be
assumed by production code or tests.

New experiments belong in the independent `research` workspace. Non-Rust
baselines belong under `baselines`. Methods exchange only the versioned dataset,
prediction, and run-receipt protocol described in [`../README.md`](../README.md).
Do not restore archived switches or experiment-only behavior to a production
crate.
