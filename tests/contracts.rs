use edge_weight_recovery::checkpoint::TrainingCheckpoint;
use edge_weight_recovery::config::TrainingConfig;
use edge_weight_recovery::time_buckets::TimeBucketSpec;
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
const INDEPENDENT_TIME_BUCKETS: &str = "experiments/independent_time_buckets/time_buckets.json";
const FULL_STATIC_REFERENCE: &str =
    "experiments/independent_time_buckets/configs/static_full_reference_u500.json";
const FINAL_BENCHMARK_PROTOCOL: &str = "experiments/neuromlr_cch_dijkstra_benchmarks/protocol.json";
const FINAL_PROJECT_CONFIG: &str =
    "experiments/neuromlr_cch_dijkstra_benchmarks/configs/project_common_train_u500.json";
const FINAL_TRAIN_AUDIT: &str = "experiments/neuromlr_cch_dijkstra_benchmarks/train_audit.json";
const FINAL_VALIDATION_AUDIT: &str =
    "experiments/neuromlr_cch_dijkstra_benchmarks/validation_audit.json";
const FINAL_TEST_AUDIT: &str = "experiments/neuromlr_cch_dijkstra_benchmarks/test_audit.json";
const FINAL_EFFICIENCY_AMENDMENT: &str =
    "experiments/neuromlr_cch_dijkstra_benchmarks/efficiency_protocol_amendment.json";
const FINAL_BENCHMARK_SUMMARY: &str = "experiments/neuromlr_cch_dijkstra_benchmarks/summary.json";
const INDEPENDENT_BUCKET_CONFIGS: [&str; 5] = [
    "experiments/independent_time_buckets/configs/static_night_00_06_u500.json",
    "experiments/independent_time_buckets/configs/static_morning_06_10_u500.json",
    "experiments/independent_time_buckets/configs/static_day_10_16_u500.json",
    "experiments/independent_time_buckets/configs/static_evening_16_20_u500.json",
    "experiments/independent_time_buckets/configs/static_late_20_24_u500.json",
];

#[test]
fn independent_bucket_configs_are_ordinary_static_models_with_data_filters() {
    let buckets = TimeBucketSpec::load(Path::new(INDEPENDENT_TIME_BUCKETS)).unwrap();
    assert_eq!(buckets.buckets.len(), 5);
    assert_eq!(buckets.buckets.first().unwrap().start_hour, 0);
    assert_eq!(buckets.buckets.last().unwrap().end_hour, 24);

    let mut ids = BTreeSet::new();
    let mut train_samples = 0;
    let mut validation_samples = 0;
    let full_static = TrainingConfig::load(Path::new(FULL_STATIC_REFERENCE)).unwrap();
    assert!(full_static.departure_time_filter.is_none());
    for path in INDEPENDENT_BUCKET_CONFIGS {
        let config = TrainingConfig::load(Path::new(path)).unwrap();
        assert_eq!(config.train_variant, "all");
        assert_eq!(config.validation_variant, "scale_fixed_seed20260715");
        assert_eq!(config.graph_representation, "edge_transition_arcs");
        assert_eq!(config.optimizer_kind, "relative_projected_subgradient");
        assert_eq!(config.eta0, 0.0002);
        assert_eq!(config.lambda, 100000.0);
        assert_eq!(config.optimizer_kind, full_static.optimizer_kind);
        assert_eq!(config.eta0, full_static.eta0);
        assert_eq!(config.lambda, full_static.lambda);
        assert_eq!(config.updates, full_static.updates);
        assert_eq!(
            config
                .as_json()
                .pointer("/test_policy")
                .and_then(Value::as_str),
            Some("never_read")
        );
        let filter = config.departure_time_filter.as_ref().unwrap();
        assert_eq!(filter.load_spec().unwrap(), buckets);
        assert!(ids.insert(filter.bucket_id.clone()));
        train_samples += filter.expected_train_samples;
        validation_samples += filter.expected_validation_samples;
    }
    assert_eq!(ids.len(), 5);
    assert_eq!(train_samples, 623275);
    assert_eq!(validation_samples, 15812);
}

#[test]
fn final_quality_protocol_freezes_common_raw_edge_inputs_and_metrics() {
    let protocol: Value = serde_json::from_slice(
        &std::fs::read(FINAL_BENCHMARK_PROTOCOL).expect("read final benchmark protocol"),
    )
    .expect("decode final benchmark protocol");
    assert_eq!(
        protocol.pointer("/schema_version").and_then(Value::as_u64),
        Some(1)
    );
    assert_eq!(
        protocol
            .pointer("/data/road_id_space")
            .and_then(Value::as_str),
        Some("unaltered_shapefile_record_index")
    );
    assert_eq!(
        protocol
            .pointer("/data/minimum_edges")
            .and_then(Value::as_u64),
        Some(5)
    );
    assert_eq!(
        protocol
            .pointer("/data/splits/train/maximum_selected")
            .unwrap(),
        &Value::Null
    );
    assert_eq!(
        protocol
            .pointer("/data/splits/validation/maximum_selected")
            .and_then(Value::as_u64),
        Some(500)
    );
    assert_eq!(
        protocol
            .pointer("/data/splits/test/maximum_selected")
            .and_then(Value::as_u64),
        Some(500)
    );
    assert_eq!(
        protocol
            .pointer("/quality/query_protocol")
            .and_then(Value::as_str),
        Some("true_first_edge_to_true_last_edge_complete_sequence")
    );
    assert_eq!(
        protocol
            .pointer("/neuromlr/upstream_commit")
            .and_then(Value::as_str),
        Some("c45e3b5811e5a59b36e4682307d2196c02dac360")
    );
    assert_eq!(
        protocol.pointer("/quality/primary_methods").unwrap(),
        &json!(["project_edge_to_edge", "neuromlr_greedy"])
    );
    assert_eq!(
        protocol
            .pointer("/neuromlr/dijkstra_validation")
            .and_then(Value::as_str),
        Some("not_run_by_user_request")
    );
    assert_eq!(
        protocol
            .pointer("/oracle_efficiency/training_formal_workload/maximum_unique_od_groups")
            .and_then(Value::as_u64),
        Some(50000)
    );
}

#[test]
fn final_project_config_uses_the_aligned_common_training_pickle() {
    let config = TrainingConfig::load(Path::new(FINAL_PROJECT_CONFIG)).unwrap();
    assert_eq!(config.train_variant, "neuromlr_common");
    assert_eq!(config.validation_variant, "neuromlr_common");
    assert_eq!(config.graph_representation, "edge_transition_arcs");
    assert_eq!(config.optimizer_kind, "relative_projected_subgradient");
    assert_eq!(config.eta0, 0.0002);
    assert_eq!(config.lambda, 100000.0);
    assert_eq!(config.updates, 500);
    assert_eq!(config.validation_every, 25);
    assert_eq!(config.rayon_threads, 16);
    assert_eq!(
        config
            .as_json()
            .pointer("/data/train_identity/sample_count")
            .and_then(Value::as_u64),
        Some(605935)
    );
    assert_eq!(
        config
            .as_json()
            .pointer("/data/validation_identity/sample_count")
            .and_then(Value::as_u64),
        Some(500)
    );
}

#[test]
fn final_summary_separates_fair_quality_from_node_to_node_and_freezes_actual_workloads() {
    let summary: Value = serde_json::from_slice(&std::fs::read(FINAL_BENCHMARK_SUMMARY).unwrap())
        .expect("decode final benchmark summary");
    assert_eq!(
        summary
            .pointer("/scope/external_quality_baseline")
            .and_then(Value::as_str),
        Some("neuromlr_greedy")
    );
    assert_eq!(
        summary
            .pointer("/quality/test/project_edge_to_edge/metrics/samples")
            .and_then(Value::as_u64),
        Some(500)
    );
    assert_eq!(
        summary
            .pointer("/quality/test/neuromlr_greedy/metrics/samples")
            .and_then(Value::as_u64),
        Some(500)
    );
    assert!(
        summary
            .pointer("/quality/test/project_minus_neuromlr_greedy/edge_f1")
            .and_then(Value::as_f64)
            .unwrap()
            < 0.0
    );
    assert_eq!(
        summary
            .pointer("/oracle_efficiency/training/workload/fixed_point_od_groups")
            .and_then(Value::as_u64),
        Some(4971)
    );
    assert_eq!(
        summary
            .pointer("/oracle_efficiency/inference/workload/query_protocol")
            .and_then(Value::as_str),
        Some("node_to_node")
    );
}

#[test]
fn efficiency_amendment_requires_all_state_model_equivalence() {
    let amendment: Value =
        serde_json::from_slice(&std::fs::read(FINAL_EFFICIENCY_AMENDMENT).unwrap())
            .expect("decode efficiency amendment");
    let workload = amendment.pointer("/frozen_training_workload").unwrap();
    assert_eq!(
        workload
            .pointer("/fixed_point_groups")
            .and_then(Value::as_u64),
        Some(4971)
    );
    assert_eq!(
        workload
            .pointer("/selected_observations")
            .and_then(Value::as_u64),
        Some(4979)
    );
    for key in [
        "all_distance_sums_equal",
        "all_predicted_counts_equal",
        "all_objectives_equal",
        "final_weights_bitwise_equal",
    ] {
        assert_eq!(
            workload
                .pointer(&format!("/{key}"))
                .and_then(Value::as_bool),
            Some(true)
        );
    }
}

#[test]
fn final_test_audit_is_bound_to_the_single_frozen_manifest() {
    let audit: Value = serde_json::from_slice(&std::fs::read(FINAL_TEST_AUDIT).unwrap()).unwrap();
    assert_eq!(
        audit
            .pointer("/audit/common_manifest/test_read")
            .and_then(Value::as_bool),
        Some(true)
    );
    assert_eq!(
        audit
            .pointer("/outputs/manifest/records")
            .and_then(Value::as_u64),
        Some(500)
    );
    assert_eq!(
        audit
            .pointer("/outputs/manifest/sha256")
            .and_then(Value::as_str),
        Some("d340e0715853f3245538f00525f4edeed6edca19c5e5326253f160baace1c5a9")
    );
}

#[test]
fn common_manifest_audits_balance_and_prove_test_was_not_read() {
    for (path, source, eligible, selected) in [
        (FINAL_TRAIN_AUDIT, 785709, 605935, 605935),
        (FINAL_VALIDATION_AUDIT, 20000, 15399, 500),
    ] {
        let audit: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let root = audit.pointer("/audit/common_manifest").unwrap();
        assert_eq!(
            root.pointer("/source/records").and_then(Value::as_u64),
            Some(source)
        );
        assert_eq!(
            root.pointer("/filtering/eligible").and_then(Value::as_u64),
            Some(eligible)
        );
        assert_eq!(
            root.pointer("/filtering/selected").and_then(Value::as_u64),
            Some(selected)
        );
        assert_eq!(
            root.pointer("/test_read").and_then(Value::as_bool),
            Some(false)
        );
        assert_eq!(
            root.pointer("/filtering/dropped")
                .and_then(Value::as_u64)
                .unwrap()
                + eligible,
            source
        );
    }
}

#[test]
fn test_manifest_requires_a_hash_bound_unlock_and_receipt() {
    let output_dir = temporary_directory("forbidden-test-manifest");
    let output = Command::new(env!("CARGO_BIN_EXE_build_common_manifest"))
        .args([
            "--city",
            "beijing",
            "--split",
            "test",
            "--variant",
            "small",
            "--minimum-edges",
            "5",
            "--maximum-selected",
            "500",
            "--manifest",
        ])
        .arg(output_dir.join("test.jsonl"))
        .arg("--pickle")
        .arg(output_dir.join("test.pkl"))
        .arg("--audit")
        .arg(output_dir.join("test-audit.json"))
        .arg("--protocol")
        .arg(FINAL_BENCHMARK_PROTOCOL)
        .arg("--test-unlock")
        .arg(output_dir.join("missing-unlock.json"))
        .arg("--test-receipt")
        .arg(output_dir.join("receipt.json"))
        .output()
        .expect("run guarded test manifest command");
    assert!(!output.status.success());
    assert!(!output_dir.join("receipt.json").exists());
    assert!(!output_dir.join("test.jsonl").exists());
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
        ["Temporal", "ProjectedSubgradientOptimizer"].concat(),
        ["Temporal", "TrainingConfig"].concat(),
        ["Temporal", "Checkpoint"].concat(),
        ["temporal", "_training"].concat(),
        ["train", "_temporal"].concat(),
        ["evaluate", "_temporal"].concat(),
        ["trip_average", "_travel_time"].concat(),
        ["q", "_global"].concat(),
        ["bucket", "_residual"].concat(),
        ["lambda", "_residual"].concat(),
        ["residual", "_eta_multiplier"].concat(),
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
