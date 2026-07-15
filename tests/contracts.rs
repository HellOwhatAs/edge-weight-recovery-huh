use edge_weight_recovery::config::{TrainingConfig, TrainingState, load_checkpoint};
use serde_json::json;
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
