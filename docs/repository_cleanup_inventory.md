# Repository cleanup inventory

This inventory was written before the cleanup changed or removed tracked
scientific code or evidence. It is the Phase 0 decision record for the
behavior-preserving repository contraction.

## Protected state

- Pre-cleanup commit: `8aacf2e8020bae13c6fad58f22ccb369f249e029`
- Archive tag: `archive/pre-cleanup-convergence-study`
- Annotated tag object: `fb327ebef49363c19ec5cf14a553fa86af2854db`
- The tag peels to the pre-cleanup commit above.
- The tag is local because publishing it to origin was not authenticated; the
  immutable commit is the recovery authority.
- The worktree was clean when the audit began. History is not rewritten or
  squashed.

The immutable pre-cleanup commit is the authoritative location for removed
experiment code, complete result trees, and the convergence-study evidence
bundle. Important scientific conclusions are summarized in the cleaned branch
rather than being silently discarded.

## Pre-cleanup size

The Git-tracked tree contained 95 files and 16,172,414 bytes. A line count over
all tracked text and generated files reported 668,250 lines, including 9,839
lines of Rust and 3,399 lines of Python. The two experiment directories alone
contained 67 files and 15,631,120 bytes.

The ignored local `data/` directory is not part of these Git figures and is not
deleted by this cleanup. It contains required local map and pickle inputs as
well as historical local derivatives.

The largest tracked expansion sources were:

| Group | Files | Bytes | Decision |
|---|---:|---:|---|
| `experiments/convergence_study/evidence/reproducibility_bundle.tar.gz` | 1 | 6,790,676 | DELETE from active branch; archive commit retains it |
| `experiments/convergence_study/validation_blocks.json` | 1 | 3,159,614 | ARCHIVE |
| `experiments/scale_study/subsets/*.json` | 12 | 4,789,178 | DELETE after extracting compact smoke identity |
| Remaining convergence-study outputs | 34 | 638,539 | ARCHIVE or replace with one summary |
| Remaining scale-study outputs | 18 | 253,113 | ARCHIVE or replace with compact configs |

## Classification

`ARCHIVE` means removal from the active branch with recovery through the
protected commit. It does not mean copying the same bulk into another directory.

### KEEP_CORE

| Current path | Cleaned responsibility |
|---|---|
| `Cargo.toml`, `Cargo.lock` | Minimal build definition; remove dependencies used only by retired branches |
| `src/lib.rs` | Export only the paper mainline modules |
| `src/config.rs` | Compact edge-only configuration and atomic checkpoint schema |
| `src/graph.rs` | Split into data, objective, oracle, and evaluation responsibilities |
| `src/optimizer.rs` | Projected subgradient and quantization only |
| `src/main.rs` | Move reusable work to `training.rs`; replace with a small `src/bin/train.rs` |
| `src/turn_graph.rs` | Generic edge-state transition graph with stable transition IDs |
| `src/evaluation.rs` | Replace route-level analysis with the standard paper metrics |
| `README.md` | Rewrite as a project guide rather than an experiment diary |
| `EXPERIMENTS.md` | Rewrite as the compact set of currently trusted results |

New core records are this inventory, `docs/research_status.md`, compact configs
under `experiments/configs/`, a compact result under
`experiments/summaries/`, and an archive pointer.

### KEEP_TOOL

| Current path | Decision |
|---|---|
| `examples/evaluate_checkpoint.rs` | Replace with a minimal `src/bin/evaluate.rs` using standard metrics only |
| `examples/generate_subsets.rs` | Move to `tools/`; retain the deterministic bounded-reproduction preprocessor |
| `scripts/run_experiment_matrix.py` | Keep only if reduced to the current edge-only configuration and output schema |
| scale subset manifest content | Extract only seed/count/hash identity for the fixed smoke inputs |

### ARCHIVE

| Paths | Reason |
|---|---|
| `src/divergence.rs`, `examples/analyze_divergence.rs` | First-divergence and route-level attribution are not core evaluation |
| `src/turn.rs` | Geometric turn classes currently serve only historical analyses and the fixed-left probe |
| `examples/audit_loop_policies.rs` | Loop-policy audit branch |
| `examples/audit_coverage.rs` | One-time coverage/scale audit |
| `examples/benchmark_customization.rs` | Partial-customization benchmark |
| `examples/generate_validation_blocks.rs` | Spent confirmation-block generator |
| `examples/probe_turn_penalty.rs` | Fixed global left-turn grid, bootstrap, and gates |
| `scripts/analyze_confirmation.py` | Spent confirmation/bootstrap analysis |
| `scripts/analyze_convergence.py` | Historical convergence trajectory analysis |
| `scripts/analyze_scale_results.py` | Superseded scale-study analysis |
| `scripts/generate_scale_subsets.py` | Historical matrix-specific wrapper; the deterministic Rust tool is sufficient |
| `experiments/convergence_study/protocol.json`, `matrix_*.csv`, `results_*`, `convergence_summary.*` | Full convergence experiment history |
| `experiments/convergence_study/confirmation_plan.json`, `confirmation/*`, `confirmation_summary.*` | Spent AM/PM confirmation evidence |
| `experiments/convergence_study/divergence_*`, `loop_*`, `turn_probe_*` | Historical diagnostic and narrow fixed-left branches |
| `experiments/convergence_study/RESULTS.md`, `evidence/manifest.json` | Detailed historical narrative and evidence map; conclusions are extracted first |
| `experiments/scale_study/environment.json`, `subset_plan.csv`, `matrix_*.csv` | Superseded scale-study protocol and grids |
| `experiments/scale_study/results.*`, `aggregate_results.json`, `grid_ranking.csv`, `scale_curve.csv` | Superseded or duplicate generated summaries |
| `experiments/scale_study/*diagnostics.json`, `coverage*.json`, `customization_benchmark.json` | One-time statistics and partial-customization evidence |

### DELETE FROM ACTIVE BRANCH

| Paths | Reason |
|---|---|
| `src/utils.rs` | Used only by legacy Adam random shock |
| `experiments/convergence_study/evidence/reproducibility_bundle.tar.gz` | Binary generated bundle; 6.79 MiB; available at archive commit |
| `experiments/convergence_study/validation_blocks.json` | 3.16 MiB source-index dump; compact identities and conclusions are sufficient |
| `experiments/convergence_study/trajectories.csv` | Regenerable trajectory table |
| `experiments/scale_study/subsets/*.json` | 4.79 MiB detailed source-index dumps; local pickle plus compact hashes drive smoke reproduction |
| ignored `scripts/__pycache__/` | Generated bytecode |

Within files that remain, the following branches are deleted from the active
training path:

- `LegacyAdamShock`, `AdamOptimizer`, random shock, restart, and seed flags;
- partial CCH customization and its updater;
- keep/erase cycle policies and loop erasure;
- boundary-edge trimming of already-complete pickle paths;
- mean-regret or train-objective checkpoint selection;
- optional test loading from the training binary;
- detailed route distributions, percentiles, strata, correlations, bootstrap,
  and first-divergence analysis;
- left-turn flags, fixed-left penalties, and geometry dependencies in the
  expanded graph core.

## Target tree

The cleanup targets this direct, non-framework structure:

```text
src/
  lib.rs
  config.rs
  data.rs
  objective.rs
  optimizer.rs
  evaluation.rs
  training.rs
  turn_graph.rs
  oracle/
    mod.rs
    cch.rs
    dijkstra.rs
  model/
    mod.rs
    edge_only.rs
    turn_aware.rs
  bin/
    train.rs
    evaluate.rs
tools/
  generate_subsets.rs
experiments/
  configs/
  summaries/
  archive/README.md
scripts/
  run_experiment_matrix.py
  summarize_results.py
tests/
  behavior_preservation.rs
```

Files may be omitted when a separate layer would add no responsibility. No
plugin system, trait hierarchy, workflow engine, or turn-residual optimizer is
introduced.

## Scientific behavior snapshot

Before refactoring, a bounded deterministic run used the existing Beijing 1%
seed-42 train subset and fixed validation subset. It used one Rayon thread,
five evaluated epochs, full customization, aggregate validation relative
regret selection, `eta0=1e-4`, `lambda=1e5`, and `q in [0.1,10]`. Test was not
read.

The command's scientific outputs were:

| Epoch | Train mean regret | Train relative regret | Validation relative regret |
|---:|---:|---:|---:|
| 0 | 595722.806084 | 0.08756428 | 0.09455969 |
| 1 | 580497.733041 | 0.09159398 | 0.10187987 |
| 2 | 515470.049130 | 0.07988029 | 0.09014614 |
| 3 | 482549.376444 | 0.07607343 | 0.08736639 |
| 4 | 447898.900602 | 0.07036808 | 0.08142399 |

Selected epoch was 4. The selected validation metrics were mean regret
`525996.588983`, aggregate relative regret `0.081423992787`, exact match
`0.301417`, edge precision `0.634683`, edge recall `0.621491`, edge F1
`0.623727`, and edge Jaccard `0.541459`. The selected multiplier range was
`[0.227425, 1.544847]`.

Pre-cleanup artifacts are retained outside the repository at
`/tmp/edge-weight-recovery-pre-cleanup/` for the Phase 5 comparison. Their
SHA-256 values are:

- training log: `2f09cb0e4d2b378db1f2f1810f69abda6899b497d526d5560ee58c0cf722ddac`
- atomic checkpoint: `e1ba252a24894320d3b7eae4946421ad5784ee25748aa9c781db181cb23bf81d`
- integer weights: `ba7f9bc6e1467a3b898b0688d62f2d6fa1280a639cbfcbe3439be34f042149e2`
- latent multipliers: `94409066d5f8179679dc5d3d7e9fe06f7136bb2ed7175ad10c77395005b928d1`

Timing and memory tokens are not scientific comparison fields. The refactor
must preserve the evaluated-before-update epoch order and the existing
two-stage quantization (`round(baseline * scale)`, then `round(metric_baseline
* q)`), because changing either would alter checkpoints.
