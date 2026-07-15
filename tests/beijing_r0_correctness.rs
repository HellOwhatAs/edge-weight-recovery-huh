use edge_weight_recovery::data::{GraphData, load_graph, load_trips};
use edge_weight_recovery::oracle::{CchOracle, ExpandedCchOracle};
use edge_weight_recovery::turn_graph::ExpandedTurnGraph;
use serde_json::{Value, json};
use std::collections::HashSet;
use std::path::Path;

const CITY: &str = "beijing";
const VALIDATION_VARIANT: &str = "scale_fixed_seed20260715";
const EXPECTED_RAW_SAMPLES: usize = 20_000;
const EXPECTED_VALID_SAMPLES: usize = 15_812;
const EXPECTED_NODES: usize = 31_199;
const EXPECTED_EDGES: usize = 72_156;
const EXPECTED_BASELINE_FINGERPRINT: &str = "ad08fec01f56dd3c";
const RESIDUAL_SCALE: f64 = 127_625.0;

#[derive(Default)]
struct RouteMetricSums {
    exact: usize,
    edge_f1: f64,
}

/// Real-data correctness audit for the nested `r=0` model.
///
/// This test is deliberately ignored: it builds both Beijing CCH topologies
/// and evaluates the fixed validation subset. It never constructs a test-split
/// path. Run it explicitly in release mode after importing the frozen edge
/// initialization:
///
/// `EDGE_INITIALIZATION=/path/to/edge_initialization.json cargo test --release
///  --locked --test beijing_r0_correctness -- --ignored --nocapture`
#[test]
#[ignore = "requires real Beijing data and EDGE_INITIALIZATION; never reads test"]
fn beijing_r0_original_and_expanded_cch_audit() {
    let initialization = std::env::var_os("EDGE_INITIALIZATION").expect(
        "EDGE_INITIALIZATION is required and must point to the imported frozen Beijing \
         edge_initialization JSON; this ignored audit reads validation only and never test",
    );
    run_audit(Path::new(&initialization))
        .unwrap_or_else(|error| panic!("Beijing r=0 correctness audit failed: {error}"));
}

fn run_audit(initialization_path: &Path) -> Result<(), String> {
    let graph = load_graph(CITY)?;
    validate_graph_identity(&graph)?;
    let edge_weights = load_edge_initialization(initialization_path, &graph)?;

    // This is the only trip-loading call in the audit. The split and variant
    // are constants so an environment value cannot redirect the audit to test.
    let validation = load_trips(CITY, "validation", VALIDATION_VARIANT, &graph, None)?;
    if validation.report.available_samples != EXPECTED_RAW_SAMPLES
        || validation.report.inspected_samples != EXPECTED_RAW_SAMPLES
        || validation.report.accepted_samples != EXPECTED_VALID_SAMPLES
    {
        return Err(format!(
            "fixed validation identity mismatch: available={}, inspected={}, accepted={}, expected {EXPECTED_RAW_SAMPLES}/{EXPECTED_RAW_SAMPLES}/{EXPECTED_VALID_SAMPLES}",
            validation.report.available_samples,
            validation.report.inspected_samples,
            validation.report.accepted_samples
        ));
    }

    let original_oracle = CchOracle::build(&graph)?;
    let original_metric = original_oracle.customize(&edge_weights)?;

    let expanded = ExpandedTurnGraph::build(&graph)?;
    let zero_residuals = vec![0.0; expanded.transition_count()];
    let transition_weights =
        expanded.transition_metric_weights(&edge_weights, &zero_residuals, RESIDUAL_SCALE)?;
    let expanded_oracle = ExpandedCchOracle::build(&graph, &expanded)?;
    let expanded_metric = expanded_oracle.customize(&edge_weights, &transition_weights)?;
    let mut expanded_query = expanded_metric.new_query();

    let mut original_metrics = RouteMetricSums::default();
    let mut expanded_metrics = RouteMetricSums::default();
    let mut predicted_path_tie_mismatches = 0usize;

    for (sample, ((source, target), observed_path)) in validation.paths.iter().enumerate() {
        let original = original_oracle
            .shortest_path(&original_metric, *source, *target)
            .map_err(|error| format!("sample {sample} original OD ({source},{target}): {error}"))?;
        let expanded_path = expanded_query
            .query(*source, *target)
            .map_err(|error| format!("sample {sample} expanded OD ({source},{target}): {error}"))?;

        if original.distance != expanded_path.distance {
            return Err(format!(
                "sample {sample} OD ({source},{target}) shortest-distance mismatch: original={}, expanded={}",
                original.distance, expanded_path.distance
            ));
        }

        let original_observed_cost = original_path_cost(&edge_weights, observed_path)?;
        let expanded_observed_cost = expanded_metric
            .observed_path_cost(observed_path)
            .map_err(|error| format!("sample {sample} observed expanded cost: {error}"))?;
        if original_observed_cost != expanded_observed_cost {
            return Err(format!(
                "sample {sample} OD ({source},{target}) observed-cost mismatch: original={original_observed_cost}, expanded={expanded_observed_cost}"
            ));
        }

        if original.original_edges != expanded_path.original_edges {
            // Both decoded routes have the same strictly checked shortest
            // distance under r=0, so a path difference is a tie-breaking
            // mismatch rather than a metric mismatch.
            predicted_path_tie_mismatches += 1;
        }
        accumulate_route_metrics(
            &mut original_metrics,
            &original.original_edges,
            observed_path,
        );
        accumulate_route_metrics(
            &mut expanded_metrics,
            &expanded_path.original_edges,
            observed_path,
        );
    }

    let samples = validation.paths.len();
    let denominator = samples as f64;
    let original_f1 = original_metrics.edge_f1 / denominator;
    let expanded_f1 = expanded_metrics.edge_f1 / denominator;
    let original_exact = original_metrics.exact as f64 / denominator;
    let expanded_exact = expanded_metrics.exact as f64 / denominator;
    let report = json!({
        "audit": "beijing_r0_original_vs_expanded_cch",
        "split": "validation",
        "variant": VALIDATION_VARIANT,
        "test_read": false,
        "raw_samples": validation.report.available_samples,
        "valid_samples": samples,
        "shortest_distance_mismatches": 0,
        "observed_cost_mismatches": 0,
        "predicted_path_tie_mismatches": predicted_path_tie_mismatches,
        "predicted_path_tie_mismatch_rate": predicted_path_tie_mismatches as f64 / denominator,
        "original": {
            "edge_f1": original_f1,
            "exact_match": original_exact,
        },
        "expanded_r0": {
            "edge_f1": expanded_f1,
            "exact_match": expanded_exact,
        },
        "expanded_minus_original": {
            "edge_f1": expanded_f1 - original_f1,
            "exact_match": expanded_exact - original_exact,
        },
        "expanded_topology_identity": expanded_metric.topology_identity(),
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&report)
            .map_err(|error| format!("failed to serialize audit report: {error}"))?
    );
    Ok(())
}

fn load_edge_initialization(path: &Path, graph: &GraphData) -> Result<Vec<u32>, String> {
    let bytes = std::fs::read(path).map_err(|error| {
        format!(
            "failed to read EDGE_INITIALIZATION {}: {error}",
            path.display()
        )
    })?;
    let initialization: Value = serde_json::from_slice(&bytes).map_err(|error| {
        format!(
            "failed to decode EDGE_INITIALIZATION {}: {error}",
            path.display()
        )
    })?;
    for (pointer, expected) in [
        ("/schema", "edge_initialization"),
        ("/model", "edge_only"),
        ("/status", "frozen_validated"),
        ("/baseline_identity/city", CITY),
        ("/baseline_identity/fnv1a64", EXPECTED_BASELINE_FINGERPRINT),
    ] {
        let actual = initialization
            .pointer(pointer)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("EDGE_INITIALIZATION field {pointer} is not a string"))?;
        if actual != expected {
            return Err(format!(
                "EDGE_INITIALIZATION field {pointer} mismatch: got {actual:?}, expected {expected:?}"
            ));
        }
    }
    if initialization
        .pointer("/schema_version")
        .and_then(Value::as_u64)
        != Some(1)
        || initialization
            .pointer("/completed_q_updates")
            .and_then(Value::as_u64)
            != Some(99)
        || initialization
            .pointer("/baseline_identity/nodes")
            .and_then(Value::as_u64)
            != Some(EXPECTED_NODES as u64)
        || initialization
            .pointer("/baseline_identity/edges")
            .and_then(Value::as_u64)
            != Some(EXPECTED_EDGES as u64)
    {
        return Err("EDGE_INITIALIZATION numeric identity mismatch".to_string());
    }

    let values = initialization
        .pointer("/quantized_metric_weights")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            "EDGE_INITIALIZATION is missing quantized_metric_weights array".to_string()
        })?;
    if values.len() != graph.tail.len() {
        return Err(format!(
            "EDGE_INITIALIZATION has {} weights but graph has {} edges",
            values.len(),
            graph.tail.len()
        ));
    }
    values
        .iter()
        .enumerate()
        .map(|(edge, value)| {
            let weight = value
                .as_u64()
                .ok_or_else(|| format!("quantized_metric_weights[{edge}] is not an integer"))?;
            let weight = u32::try_from(weight)
                .map_err(|_| format!("quantized_metric_weights[{edge}] does not fit in u32"))?;
            if weight == 0 || weight >= i32::MAX as u32 {
                return Err(format!(
                    "quantized_metric_weights[{edge}]={weight} is outside the positive CCH range"
                ));
            }
            Ok(weight)
        })
        .collect()
}

fn validate_graph_identity(graph: &GraphData) -> Result<(), String> {
    if graph.x.len() != EXPECTED_NODES || graph.tail.len() != EXPECTED_EDGES {
        return Err(format!(
            "Beijing graph size mismatch: nodes={}, edges={}",
            graph.x.len(),
            graph.tail.len()
        ));
    }
    let actual = baseline_fingerprint(graph);
    if actual != EXPECTED_BASELINE_FINGERPRINT {
        return Err(format!(
            "Beijing baseline fingerprint mismatch: got {actual}, expected {EXPECTED_BASELINE_FINGERPRINT}"
        ));
    }
    Ok(())
}

fn baseline_fingerprint(graph: &GraphData) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for value in graph
        .tail
        .iter()
        .chain(&graph.head)
        .chain(&graph.baseline_weights)
    {
        for byte in value.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    format!("{hash:016x}")
}

fn original_path_cost(weights: &[u32], path: &[usize]) -> Result<u64, String> {
    path.iter().try_fold(0u64, |sum, &edge| {
        let weight = weights
            .get(edge)
            .ok_or_else(|| format!("observed edge {edge} is out of bounds"))?;
        sum.checked_add(*weight as u64)
            .ok_or_else(|| "observed original path cost overflow".to_string())
    })
}

fn accumulate_route_metrics(sums: &mut RouteMetricSums, predicted: &[usize], observed: &[usize]) {
    sums.exact += usize::from(predicted == observed);
    let predicted_set: HashSet<usize> = predicted.iter().copied().collect();
    let observed_set: HashSet<usize> = observed.iter().copied().collect();
    let intersection = predicted_set.intersection(&observed_set).count() as f64;
    let precision = intersection / predicted_set.len().max(1) as f64;
    let recall = intersection / observed_set.len().max(1) as f64;
    sums.edge_f1 += if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
}
