use edge_weight_recovery::config::{TrainingConfig, TrainingState, load_checkpoint};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
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
fn archived_json_tree_is_parseable_and_bitwise_unchanged() {
    let archive = Path::new("experiments/archive/turn_residual_abc_v1");
    let mut paths = Vec::new();
    collect_files_with_extension(archive, "json", &mut paths);
    let mut entries = paths
        .into_iter()
        .map(|path| {
            let relative = relative_unix_path(archive, &path);
            (relative, path)
        })
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    assert_eq!(entries.len(), 19, "archived JSON inventory changed");

    let mut hasher = Sha256::new();
    for (relative, path) in entries {
        let raw = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("read archived JSON {}: {error}", path.display()));
        let parsed: serde_json::Value = serde_json::from_slice(&raw)
            .unwrap_or_else(|error| panic!("parse archived JSON {}: {error}", path.display()));
        assert!(
            parsed.is_object(),
            "archived JSON root is not an object: {}",
            path.display()
        );
        hasher.update(relative.as_bytes());
        hasher.update([0]);
        hasher.update(&raw);
        hasher.update([0]);
    }
    assert_eq!(
        format!("{:x}", hasher.finalize()),
        "c7f4df925e9b443645f160dc63a3a094686bf252e3be41a729953d276d68f0ac",
        "archived JSON bytes or relative paths changed"
    );
}

#[test]
fn retired_training_terms_do_not_reappear_in_active_code_or_configs() {
    let forbidden = [
        ["TurnExperiment", "Arm"].concat(),
        ["expanded_edge", "_continuation"].concat(),
        ["turn", "_only"].concat(),
        ["joint_edge", "_turn"].concat(),
        ["q_completed", "_updates"].concat(),
        ["r_completed", "_updates"].concat(),
        ["ExpandedEdge", "Continuation"].concat(),
        ["Turn", "Only"].concat(),
        ["JointEdge", "Turn"].concat(),
        ["updates", "_q"].concat(),
        ["updates", "_residuals"].concat(),
        ["eta", "_q0"].concat(),
        ["eta", "_r0"].concat(),
        ["lambda", "_turn"].concat(),
        ["turn", "_aware"].concat(),
        ["turn", "_training"].concat(),
    ];
    let mut active_files = Vec::new();
    for root in ["src", "tests", "tools"] {
        collect_files_with_extension(Path::new(root), "rs", &mut active_files);
    }
    collect_files_with_extension(Path::new("experiments/configs"), "json", &mut active_files);
    active_files.sort();

    for path in active_files {
        let contents = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read active file {}: {error}", path.display()));
        for retired in &forbidden {
            assert!(
                !contents.contains(retired),
                "retired term {retired:?} appears in active file {}",
                path.display()
            );
        }
    }
}

fn collect_files_with_extension(root: &Path, extension: &str, output: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(root)
        .unwrap_or_else(|error| panic!("read directory {}: {error}", root.display()))
    {
        let path = entry.expect("read directory entry").path();
        if path.is_dir() {
            collect_files_with_extension(&path, extension, output);
        } else if path.extension().and_then(|value| value.to_str()) == Some(extension) {
            output.push(path);
        }
    }
}

fn relative_unix_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or_else(|_| panic!("{} is not below {}", path.display(), root.display()))
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}
