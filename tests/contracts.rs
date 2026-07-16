use edge_weight_recovery::checkpoint::TrainingCheckpoint;
use edge_weight_recovery::config::TrainingConfig;
use edge_weight_recovery::temporal::{TemporalTrainingConfig, TimeBucketSpec};
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const ORIGINAL_EDGES_SMOKE: &str = "experiments/configs/original_edges_smoke_1pct.json";
const EDGE_TRANSITION_ARCS_SMOKE: &str = "experiments/configs/edge_transition_arcs_smoke_1pct.json";
const ORIGINAL_RELATIVE_RECOVERY: &str =
    "experiments/optimizer_recovery/configs/original_edges_relative_10pct_u299.json";
const TRANSITION_RELATIVE_RECOVERY: &str =
    "experiments/optimizer_recovery/configs/edge_transition_arcs_relative_10pct_u299.json";
const FULL_TIME_BUCKETS: &str = "experiments/full_data_time_conditioning/time_buckets.json";
const FULL_TEMPORAL_LENGTH: &str =
    "experiments/full_data_time_conditioning/configs/temporal_length_full_eta0002_u500.json";
const FULL_TEMPORAL_TRAVEL_TIME: &str =
    "experiments/full_data_time_conditioning/configs/temporal_travel_time_full_eta0002_u500.json";

#[test]
fn full_temporal_configs_share_coarse_train_derived_buckets_and_never_read_test() {
    let buckets = TimeBucketSpec::load(Path::new(FULL_TIME_BUCKETS)).unwrap();
    assert_eq!(buckets.buckets.len(), 5);
    assert_eq!(buckets.buckets.first().unwrap().start_hour, 0);
    assert_eq!(buckets.buckets.last().unwrap().end_hour, 24);

    let length = TemporalTrainingConfig::load(Path::new(FULL_TEMPORAL_LENGTH)).unwrap();
    let travel = TemporalTrainingConfig::load(Path::new(FULL_TEMPORAL_TRAVEL_TIME)).unwrap();
    for config in [&length, &travel] {
        assert_eq!(config.train_variant, "all");
        assert_eq!(config.validation_variant, "scale_fixed_seed20260715");
        assert_eq!(config.graph_representation, "edge_transition_arcs");
        assert_eq!(
            config
                .as_json()
                .pointer("/test_policy")
                .and_then(Value::as_str),
            Some("never_read")
        );
        assert_eq!(config.load_bucket_spec().unwrap(), buckets);
    }
    assert_eq!(length.baseline_kind.as_str(), "length");
    assert_eq!(travel.baseline_kind.as_str(), "trip_average_travel_time");
}

#[test]
fn train_help_exposes_only_the_unified_inputs() {
    let output = Command::new(env!("CARGO_BIN_EXE_train"))
        .arg("--help")
        .output()
        .expect("run train --help");
    assert!(output.status.success());
    let help = String::from_utf8(output.stdout).expect("UTF-8 help");
    assert!(help.contains("--config PATH"));
    assert!(help.contains("--output-dir PATH"));
    assert!(help.contains("[--resume CHECKPOINT]"));

    let exposed_long_options = help
        .split_whitespace()
        .map(|token| token.trim_matches(['[', ']', ',']))
        .filter(|token| token.starts_with("--"))
        .map(str::to_string)
        .collect::<BTreeSet<_>>();
    assert_eq!(
        exposed_long_options,
        ["--config", "--help", "--output-dir", "--resume"]
            .into_iter()
            .map(str::to_string)
            .collect()
    );

    for retired in [
        "--solver",
        "--metric-update",
        "--selection-metric",
        "--train-cycle-policy",
        "--trim-boundary-edges",
        "--run-test",
        "--test-variant",
        "--graph-order",
        "--model",
    ] {
        assert!(!help.contains(retired), "retired option leaked: {retired}");
    }
}

#[test]
fn common_direct_weight_checkpoint_is_saved_and_loaded_atomically() {
    let config = TrainingConfig::load(Path::new(EDGE_TRANSITION_ARCS_SMOKE))
        .expect("load transition-arc smoke config through the common schema");
    let checkpoint = TrainingCheckpoint {
        graph_representation: config.graph_representation.clone(),
        completed_updates: 3,
        weights: vec![12.5, f64::from_bits(0x4034_5555_5555_5555), 91.25],
        configuration: config.as_json().clone(),
        runtime_identity: json!({
            "data": "fixture",
            "graph": "fixture",
        }),
        topology_identity: "fnv1a64:common-checkpoint-fixture".to_string(),
    };
    let output_dir = temporary_directory("common-direct-checkpoint");

    let checkpoint_path = checkpoint
        .save(&output_dir)
        .expect("atomically save common checkpoint");
    assert_eq!(checkpoint_path, output_dir.join("checkpoint.json"));
    assert_eq!(
        directory_file_names(&output_dir),
        vec!["checkpoint.json".to_string()],
        "atomic save must not leave a temporary file behind"
    );

    let restored = TrainingCheckpoint::load(&checkpoint_path).expect("load common checkpoint");
    assert_eq!(restored, checkpoint);
    assert_eq!(restored.graph_representation, "edge_transition_arcs");
    assert_eq!(restored.completed_updates, 3);
    assert_eq!(
        restored
            .weights
            .iter()
            .map(|weight| weight.to_bits())
            .collect::<Vec<_>>(),
        checkpoint
            .weights
            .iter()
            .map(|weight| weight.to_bits())
            .collect::<Vec<_>>()
    );

    let raw: Value = serde_json::from_slice(
        &std::fs::read(&checkpoint_path).expect("read serialized common checkpoint"),
    )
    .expect("parse serialized common checkpoint");
    assert_eq!(
        raw.as_object()
            .expect("checkpoint root object")
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>(),
        [
            "completed_updates",
            "configuration",
            "graph_representation",
            "runtime_identity",
            "schema_version",
            "topology_identity",
            "weights",
        ]
        .into_iter()
        .collect()
    );

    std::fs::remove_dir_all(output_dir).expect("remove checkpoint fixture");
}

#[test]
fn active_smokes_share_one_schema_and_differ_only_by_representation() {
    let original = TrainingConfig::load(Path::new(ORIGINAL_EDGES_SMOKE))
        .expect("load original-edge smoke config");
    let transitions = TrainingConfig::load(Path::new(EDGE_TRANSITION_ARCS_SMOKE))
        .expect("load transition-arc smoke config");

    assert_eq!(original.graph_representation, "original_edges");
    assert_eq!(transitions.graph_representation, "edge_transition_arcs");
    assert_eq!(original.city, transitions.city);
    assert_eq!(original.train_variant, transitions.train_variant);
    assert_eq!(original.validation_variant, transitions.validation_variant);
    assert_eq!(
        original.weight_lower_factor,
        transitions.weight_lower_factor
    );
    assert_eq!(
        original.weight_upper_factor,
        transitions.weight_upper_factor
    );
    assert_eq!(original.eta0, transitions.eta0);
    assert_eq!(original.optimizer_kind, transitions.optimizer_kind);
    assert_eq!(original.lambda, transitions.lambda);
    assert_eq!(original.updates, transitions.updates);
    assert_eq!(original.validation_every, transitions.validation_every);
    assert_eq!(original.rayon_threads, transitions.rayon_threads);
    assert_eq!(original.updates, 3);
    assert_eq!(original.validation_every, 3);
    assert_eq!(original.rayon_threads, 4);

    assert_eq!(
        normalized_smoke_configuration(&original),
        normalized_smoke_configuration(&transitions),
        "apart from run metadata, representation must be the only configuration difference"
    );
}

#[test]
fn registered_calibration_matrix_is_bounded_and_frozen() {
    let expected = BTreeSet::from([
        ("edge_transition_arcs".to_string(), 100, 50),
        ("edge_transition_arcs".to_string(), 300, 50),
        ("edge_transition_arcs".to_string(), 1000, 50),
        ("edge_transition_arcs".to_string(), 3000, 50),
        ("edge_transition_arcs".to_string(), 100, 200),
        ("original_edges".to_string(), 100, 50),
        ("original_edges".to_string(), 300, 50),
        ("original_edges".to_string(), 1000, 50),
        ("original_edges".to_string(), 3000, 50),
        ("original_edges".to_string(), 300, 200),
    ]);
    let config_paths = directory_file_names(Path::new("experiments/configs"))
        .into_iter()
        .filter(|name| name.contains("_10pct_u"))
        .map(|name| Path::new("experiments/configs").join(name))
        .collect::<Vec<_>>();
    assert_eq!(config_paths.len(), expected.len());

    let mut actual = BTreeSet::new();
    for path in config_paths {
        let config = TrainingConfig::load(&path)
            .unwrap_or_else(|error| panic!("load calibration config {}: {error}", path.display()));
        assert_eq!(config.city, "beijing");
        assert_eq!(config.train_variant, "scale_10pct_seed42");
        assert_eq!(config.validation_variant, "scale_fixed_seed20260715");
        assert_eq!(config.weight_lower_factor, 0.1);
        assert_eq!(config.weight_upper_factor, 10.0);
        assert_eq!(config.lambda, 0.001);
        assert_eq!(config.validation_every, 10);
        assert_eq!(config.rayon_threads, 4);
        assert_eq!(
            config
                .as_json()
                .pointer("/data/path_contract")
                .and_then(Value::as_str),
            Some("complete_original_edge_id_sequence_min_2_edges")
        );
        assert_eq!(
            config
                .as_json()
                .pointer("/data/cycle_policy")
                .and_then(Value::as_str),
            Some("drop")
        );
        assert_eq!(
            config
                .as_json()
                .pointer("/test_policy")
                .and_then(Value::as_str),
            Some("never_read")
        );
        assert_eq!(
            config
                .as_json()
                .pointer("/data/train_identity/sha256")
                .and_then(Value::as_str),
            Some("8943d8958f3b4fadd7d3eb2f351b97268543961e441436e0ad68408cee45cc0a")
        );
        assert_eq!(
            config
                .as_json()
                .pointer("/data/validation_identity/sha256")
                .and_then(Value::as_str),
            Some("c855d1ebc396576463c363cf2b94480569938de77908aac560df2573d75a1ade")
        );

        actual.insert((
            config.graph_representation.clone(),
            config.eta0 as u64,
            config.updates,
        ));
    }
    assert_eq!(actual, expected);

    assert_eq!(
        directory_file_names(Path::new("experiments/configs")),
        vec![
            "edge_transition_arcs_eta1000_10pct_u50.json".to_string(),
            "edge_transition_arcs_eta100_10pct_u200.json".to_string(),
            "edge_transition_arcs_eta100_10pct_u50.json".to_string(),
            "edge_transition_arcs_eta3000_10pct_u50.json".to_string(),
            "edge_transition_arcs_eta300_10pct_u50.json".to_string(),
            "edge_transition_arcs_smoke_1pct.json".to_string(),
            "original_edges_eta1000_10pct_u50.json".to_string(),
            "original_edges_eta100_10pct_u50.json".to_string(),
            "original_edges_eta3000_10pct_u50.json".to_string(),
            "original_edges_eta300_10pct_u200.json".to_string(),
            "original_edges_eta300_10pct_u50.json".to_string(),
            "original_edges_smoke_1pct.json".to_string(),
        ],
        "only the two technical smokes and bounded calibration matrix are active"
    );
}

#[test]
fn relative_optimizer_recovery_is_representation_neutral() {
    let original = TrainingConfig::load(Path::new(ORIGINAL_RELATIVE_RECOVERY))
        .expect("load original-edge relative recovery config");
    let transitions = TrainingConfig::load(Path::new(TRANSITION_RELATIVE_RECOVERY))
        .expect("load transition-arc relative recovery config");

    for config in [&original, &transitions] {
        assert_eq!(config.optimizer_kind, "relative_projected_subgradient");
        assert_eq!(config.eta0, 0.0002);
        assert_eq!(config.lambda, 100000.0);
        assert_eq!(config.weight_lower_factor, 0.1);
        assert_eq!(config.weight_upper_factor, 10.0);
        assert_eq!(config.updates, 299);
        assert_eq!(config.validation_every, 10);
        assert_eq!(config.rayon_threads, 4);
        assert_eq!(config.train_variant, "scale_10pct_seed42");
        assert_eq!(config.validation_variant, "scale_fixed_seed20260715");
    }
    assert_eq!(original.graph_representation, "original_edges");
    assert_eq!(transitions.graph_representation, "edge_transition_arcs");
    assert_eq!(
        normalized_smoke_configuration(&original),
        normalized_smoke_configuration(&transitions),
        "representation must be the only mathematical recovery-config difference"
    );
}

#[test]
fn active_tree_contains_no_retired_qr_terms() {
    let forbidden = [
        ["residual", "_scale"].concat(),
        ["lambda", "_transition"].concat(),
        ["r", "_max"].concat(),
        ["transition", "_residual"].concat(),
        ["ExpandedProjected", "SubgradientOptimizer"].concat(),
        ["ExpandedRoad", "Model"].concat(),
        ["ExpandedTraining", "Config"].concat(),
        ["expanded", "_training"].concat(),
        ["pair", "_state"].concat(),
        ["pair", "-state"].concat(),
        ["overlap", "_arcs"].concat(),
        ["overlap", "-arc"].concat(),
        ["node", "_weight"].concat(),
        ["node", "-weight"].concat(),
        ["coordinate", "_to_arc_weights"].concat(),
        ["Graph", "Order"].concat(),
        ["graph", "_order"].concat(),
        ["first", "_order"].concat(),
        ["second", "_order"].concat(),
    ];
    let mut active_files = Vec::new();
    for root in ["src", "tests", "tools"] {
        collect_files_with_extensions(Path::new(root), &["rs"], &mut active_files);
    }
    collect_files_with_extensions(Path::new("scripts"), &["py"], &mut active_files);
    collect_files_with_extensions(
        Path::new("experiments/configs"),
        &["json"],
        &mut active_files,
    );
    for path in ["README.md", "EXPERIMENTS.md", "docs/research_status.md"] {
        let path = PathBuf::from(path);
        if path.is_file() {
            active_files.push(path);
        }
    }
    active_files.sort();
    active_files.dedup();

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

fn normalized_smoke_configuration(config: &TrainingConfig) -> Value {
    let mut normalized = config.as_json().clone();
    let root = normalized
        .as_object_mut()
        .expect("validated configuration root is an object");
    root.remove("run_id");
    root.remove("description");
    normalized
        .pointer_mut("/graph")
        .and_then(Value::as_object_mut)
        .expect("validated graph configuration is an object")
        .remove("representation");
    normalized
}

fn temporary_directory(label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "edge-weight-recovery-{label}-{}-{nonce}",
        std::process::id()
    ))
}

fn directory_file_names(root: &Path) -> Vec<String> {
    let mut names = std::fs::read_dir(root)
        .unwrap_or_else(|error| panic!("read directory {}: {error}", root.display()))
        .map(|entry| {
            entry
                .expect("read directory entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect::<Vec<_>>();
    names.sort();
    names
}

fn collect_files_with_extensions(root: &Path, extensions: &[&str], output: &mut Vec<PathBuf>) {
    if !root.exists() {
        return;
    }
    if root.is_file() {
        if root
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|extension| extensions.contains(&extension))
        {
            output.push(root.to_path_buf());
        }
        return;
    }
    for entry in std::fs::read_dir(root)
        .unwrap_or_else(|error| panic!("read directory {}: {error}", root.display()))
    {
        let path = entry.expect("read directory entry").path();
        collect_files_with_extensions(&path, extensions, output);
    }
}
