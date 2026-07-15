use edge_weight_recovery::config::{
    ExperimentConfig, TrainingConfig, TrainingState, TurnExperimentArm, load_checkpoint,
};
use serde_json::json;
use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn train_help_exposes_only_the_mainline_inputs() {
    let output = Command::new(env!("CARGO_BIN_EXE_train"))
        .arg("--help")
        .output()
        .expect("run train --help");
    assert!(output.status.success());
    let help = String::from_utf8(output.stdout).expect("UTF-8 help");
    assert!(help.contains("--config PATH"));
    assert!(help.contains("--output-dir PATH"));
    for retired in [
        "--solver",
        "--metric-update",
        "--selection-metric",
        "--train-cycle-policy",
        "--trim-boundary-edges",
        "--run-test",
        "--test-variant",
    ] {
        assert!(!help.contains(retired), "retired option leaked: {retired}");
    }
}

#[test]
fn atomic_checkpoint_pairs_model_state_config_and_identity() {
    let config = TrainingConfig::load(Path::new("experiments/configs/smoke_1pct.json"))
        .expect("load smoke config");
    let mut state = TrainingState::new(&[10, 20], &[1.0, 1.0]);
    assert!(state.update(3, 0.25, 4.0, &[9, 21], &[0.9, 1.05], 0.0));
    let identity = json!({
        "baseline": {"fingerprint": "fixture"},
        "train": {"identity": "train-fixture"},
        "validation": {"identity": "validation-fixture"}
    });
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock")
        .as_nanos();
    let output_dir = std::env::temp_dir().join(format!(
        "edge-weight-recovery-checkpoint-{}-{nonce}",
        std::process::id()
    ));

    let checkpoint_path = state
        .save_checkpoint(&output_dir, &config, &identity)
        .expect("save checkpoint");
    let checkpoint = load_checkpoint(&checkpoint_path).expect("load checkpoint");
    assert_eq!(checkpoint["epoch"], 3);
    assert_eq!(
        checkpoint["selection"]["metric"],
        "aggregate_relative_regret"
    );
    assert_eq!(checkpoint["selection"]["value"], 0.25);
    assert_eq!(checkpoint["q"], json!([0.9, 1.05]));
    assert_eq!(checkpoint["quantized_metric_weights"], json!([9, 21]));
    assert_eq!(checkpoint["configuration"]["run_id"], "smoke_1pct");
    assert_eq!(checkpoint["runtime_identity"], identity);

    std::fs::remove_dir_all(output_dir).expect("remove checkpoint fixture");
}

#[test]
fn turn_screen_is_exactly_the_preregistered_thirteen_cells() {
    let mut paths = std::fs::read_dir("experiments/configs")
        .expect("read experiment configs")
        .map(|entry| entry.expect("read config entry").path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("turn_screen_") && name.ends_with(".json"))
        })
        .collect::<Vec<_>>();
    paths.sort();
    assert_eq!(paths.len(), 13);

    let mut arm_counts = [0usize; 3];
    let mut turn_grid = BTreeSet::new();
    let mut joint_grid = BTreeSet::new();
    for path in paths {
        let ExperimentConfig::TurnAware(config) =
            ExperimentConfig::load(&path).expect("load preregistered turn config")
        else {
            panic!("{} is not turn-aware", path.display());
        };
        assert_eq!(config.stage, "screen_10pct");
        assert_eq!(config.updates, 30);
        assert_eq!(config.validation_every, 10);
        assert_eq!(config.rayon_threads, 4);
        assert_eq!(config.residual_scale, 127_625.0);
        assert_eq!(config.r_max, 10.0);
        let cell = || {
            (
                (config.eta_r0.expect("turn eta") * 10_000.0).round() as i32,
                config.lambda_turn.expect("turn lambda") as u64,
            )
        };
        match config.arm {
            TurnExperimentArm::ExpandedEdgeContinuation => arm_counts[0] += 1,
            TurnExperimentArm::TurnOnly => {
                arm_counts[1] += 1;
                turn_grid.insert(cell());
            }
            TurnExperimentArm::JointEdgeTurn => {
                arm_counts[2] += 1;
                joint_grid.insert(cell());
            }
        }
    }
    let expected_grid = BTreeSet::from([
        (1, 1_000),
        (1, 100_000),
        (1, 10_000_000),
        (3, 1_000),
        (3, 100_000),
        (3, 10_000_000),
    ]);
    assert_eq!(arm_counts, [1, 6, 6]);
    assert_eq!(turn_grid, expected_grid);
    assert_eq!(joint_grid, expected_grid);
}
