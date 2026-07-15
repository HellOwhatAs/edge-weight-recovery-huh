use edge_weight_recovery::config::{
    ExperimentConfig, TrainingConfig, TrainingState, TurnExperimentArm, load_checkpoint,
};
use serde_json::{Value, json};
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
fn active_configs_do_not_expose_the_archived_turn_study() {
    for entry in std::fs::read_dir("experiments/configs").expect("read active experiment configs") {
        let path = entry.expect("read active config entry").path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        assert!(
            !(name.starts_with("turn_") && name.ends_with(".json")),
            "archived turn-study config remains active: {}",
            path.display()
        );
    }
}

#[test]
fn archived_turn_study_preserves_execution_facts_without_ranking_models() {
    let archive = Path::new("experiments/archive/turn_residual_abc_v1");
    let mut paths = std::fs::read_dir(archive.join("configs"))
        .expect("read archived turn-study configs")
        .map(|entry| entry.expect("read config entry").path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(".json"))
        })
        .collect::<Vec<_>>();
    paths.sort();
    assert_eq!(paths.len(), 16);

    let mut stage_counts = [0usize; 3];
    let mut screen_arm_counts = [0usize; 3];
    let mut full_arm_counts = [0usize; 3];
    for path in paths {
        let ExperimentConfig::TurnAware(config) =
            ExperimentConfig::load(&path).expect("parse archived turn-study config")
        else {
            panic!("{} is not turn-aware", path.display());
        };
        assert_eq!(config.as_json()["test_policy"], "never_read");
        let arm_index = match config.arm {
            TurnExperimentArm::ExpandedEdgeContinuation => 0,
            TurnExperimentArm::TurnOnly => 1,
            TurnExperimentArm::JointEdgeTurn => 2,
        };
        match config.stage.as_str() {
            "correctness" => stage_counts[0] += 1,
            "screen_10pct" => {
                stage_counts[1] += 1;
                screen_arm_counts[arm_index] += 1;
            }
            "full_endpoint" => {
                stage_counts[2] += 1;
                full_arm_counts[arm_index] += 1;
            }
            stage => panic!("unexpected archived stage {stage:?}"),
        }
    }
    assert_eq!(stage_counts, [1, 13, 2]);
    assert_eq!(screen_arm_counts, [1, 6, 6]);
    assert_eq!(full_arm_counts, [1, 1, 0]);

    let protocol = read_json(&archive.join("turn_residual_protocol.json"));
    assert_eq!(
        protocol["decision"]["layer_2_outcome"]["completed_cells"],
        13
    );
    assert_eq!(protocol["layer_3_full_endpoint"]["completed_runs"], 2);
    assert_eq!(protocol["layer_3_full_endpoint"]["test_read"], false);
    assert_eq!(
        protocol["decision"]["layer_3_outcome"]["joint_edge_turn_full_run"],
        false
    );

    let screen_summary = read_json(
        &archive
            .join("summaries")
            .join("beijing_turn_residual_10pct.json"),
    );
    assert_eq!(screen_summary["integrity"]["completed_cells"], 13);
    assert_eq!(
        screen_summary["results"]
            .as_array()
            .expect("screen results")
            .len(),
        13
    );
    assert_eq!(screen_summary["data"]["test_read"], false);

    let full_summary = read_json(
        &archive
            .join("summaries")
            .join("beijing_turn_residual_full.json"),
    );
    assert_eq!(full_summary["integrity"]["completed_runs"], 2);
    assert_eq!(full_summary["integrity"]["all_test_read_false"], true);
    assert_eq!(full_summary["data"]["test_read"], false);
    let results = full_summary["results"]
        .as_array()
        .expect("full endpoint results");
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|result| {
        result["selected_at_budget_boundary"].as_bool() == Some(true)
            && result["best_step"].as_u64() == Some(50)
    }));
    assert!(
        results
            .iter()
            .all(|result| result["arm"] != "joint_edge_turn")
    );

    let mean_regret = |arm: &str| {
        results
            .iter()
            .find(|result| result["arm"] == arm)
            .and_then(|result| result.pointer("/validation/mean_regret"))
            .and_then(Value::as_f64)
            .unwrap_or_else(|| panic!("missing mean regret for archived arm {arm}"))
    };
    assert!(mean_regret("turn_only") > mean_regret("expanded_edge_continuation"));
}

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).expect("read archived JSON"))
        .expect("parse archived JSON")
}
