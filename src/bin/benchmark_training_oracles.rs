use edge_weight_recovery::config::{TrainingConfig, atomic_write};
use edge_weight_recovery::data::load_graph;
use edge_weight_recovery::graph_problem::{
    GraphMetric, GraphProblem, GraphRepresentation, MappedPath, OracleKind, QueryGroup,
};
use edge_weight_recovery::objective::{compute_regret, count_difference_l1};
use edge_weight_recovery::optimizer::{OptimizerGeometry, ProjectedSubgradientOptimizer};
use rayon::prelude::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let Some(arguments) = Arguments::from_args()? else {
        return Ok(());
    };
    let actual_threads = rayon::current_num_threads().max(1);
    if actual_threads != arguments.threads {
        return Err(format!(
            "--threads={} but RAYON_NUM_THREADS resolved to {actual_threads}",
            arguments.threads
        ));
    }
    let total_started = Instant::now();
    let config = TrainingConfig::load(&arguments.config)?;
    let graph = load_graph(&config.city)?;
    let representation = GraphRepresentation::parse(&config.graph_representation)?;
    let problem = GraphProblem::build(
        &graph,
        representation,
        config.weight_lower_factor,
        config.weight_upper_factor,
    )?;
    let raw_paths = load_manifest_paths(&arguments.manifest, arguments.candidate_samples)?;
    let candidate_paths = raw_paths
        .iter()
        .map(|edges| problem.map_path(edges))
        .collect::<Result<Vec<_>, _>>()?;
    let candidate_groups = GraphProblem::group_paths(&candidate_paths)?;
    let selection = select_matching_groups(
        &problem,
        &candidate_groups,
        arguments.maximum_groups,
        arguments.require_path_match,
    )?;
    let selected_keys = selection
        .selected_groups
        .iter()
        .map(|group| (group.source, group.target))
        .collect::<BTreeSet<_>>();
    let selected_paths = candidate_paths
        .into_iter()
        .filter(|path| selected_keys.contains(&(path.source, path.target)))
        .collect::<Vec<_>>();
    let selected_groups = GraphProblem::group_paths(&selected_paths)?;
    if selected_groups != selection.selected_groups {
        return Err("selected workload groups changed while retaining observations".to_string());
    }
    let geometry = OptimizerGeometry::parse(&config.optimizer_kind)?;
    let (selected_paths, selected_groups, stabilization) = if let Some(path) =
        &arguments.frozen_workload
    {
        let frozen_groups = load_frozen_workload(path, arguments.updates)?;
        let frozen_keys = frozen_groups
            .iter()
            .map(|group| (group.source, group.target))
            .collect::<BTreeSet<_>>();
        let frozen_paths = selected_paths
            .into_iter()
            .filter(|path| frozen_keys.contains(&(path.source, path.target)))
            .collect::<Vec<_>>();
        let recomputed_groups = GraphProblem::group_paths(&frozen_paths)?;
        if recomputed_groups != frozen_groups {
            return Err("frozen workload differs from the current candidate manifest".to_string());
        }
        (
            frozen_paths,
            recomputed_groups,
            json!({
                "status": "loaded_frozen_fixed_point",
                "source": path,
                "source_sha256": file_checksum(path)?,
            }),
        )
    } else if arguments.stabilize_path_matches {
        stabilize_path_matching_workload(
            &problem,
            selected_paths,
            geometry,
            config.eta0,
            config.lambda,
            arguments.updates,
            arguments.threads,
        )?
    } else {
        (selected_paths, selected_groups, Value::Null)
    };
    let observed = problem.observed_counts(&selected_paths)?;

    let requested_oracles = match arguments.oracle {
        RequestedOracle::One(kind) => vec![kind],
        RequestedOracle::BothCchFirst => vec![OracleKind::Cch, OracleKind::Dijkstra],
        RequestedOracle::BothDijkstraFirst => vec![OracleKind::Dijkstra, OracleKind::Cch],
    };
    let mut runs = Vec::new();
    for oracle in requested_oracles {
        runs.push(run_training_workload(
            &problem,
            &selected_groups,
            &observed,
            geometry,
            config.eta0,
            config.lambda,
            arguments.updates,
            arguments.threads,
            oracle,
        )?);
    }
    let consistency = if runs.len() == 2 {
        compare_runs(&runs[0], &runs[1])?
    } else {
        Value::Null
    };
    let build_stats = problem.oracle_build_stats();
    let output = json!({
        "schema_version": 1,
        "configuration": arguments.config,
        "manifest": arguments.manifest,
        "candidate_samples": raw_paths.len(),
        "candidate_unique_od": candidate_groups.len(),
        "oracle_run_order": runs.iter().map(|run| run.oracle.as_str()).collect::<Vec<_>>(),
        "workload": {
            "selection": "sorted_OD_after_source_order_candidate_prefix",
            "require_initial_path_match": arguments.require_path_match,
            "stabilize_path_matches_across_all_states": arguments.stabilize_path_matches,
            "maximum_groups": arguments.maximum_groups,
            "selected_groups": selected_groups.len(),
            "selected_observations": selected_paths.len(),
            "selected_sample_weight": selected_groups.iter().map(|group| group.sample_count).sum::<u64>(),
            "initial_audit": selection.audit,
            "stabilization_audit": stabilization,
            "selected_od_groups": selected_groups.iter().map(|group| json!({
                "source": group.source,
                "target": group.target,
                "sample_count": group.sample_count,
            })).collect::<Vec<_>>(),
        },
        "invariants": {
            "graph_representation": representation.as_str(),
            "initial_weights": "problem_length_baseline",
            "optimizer": geometry.as_str(),
            "eta0": config.eta0,
            "lambda": config.lambda,
            "updates": arguments.updates,
            "threads": arguments.threads,
            "integer_weight_semantics": "same_rounded_positive_u32_vector",
            "observed_count_sha256": u64_checksum(&observed),
        },
        "oracle_setup": {
            "cch_topology_preprocessing_seconds": build_stats.cch_topology.as_secs_f64(),
            "dijkstra_adjacency_setup_seconds": build_stats.dijkstra_topology.as_secs_f64(),
        },
        "runs": runs.iter().map(TrainingRun::as_json).collect::<Vec<_>>(),
        "consistency": consistency,
        "total_process_seconds": total_started.elapsed().as_secs_f64(),
        "peak_rss_kib": process_peak_rss_kib().unwrap_or(0),
        "test_read": false,
    });
    let encoded = serde_json::to_vec_pretty(&output)
        .map_err(|error| format!("failed to encode training benchmark: {error}"))?;
    atomic_write(&arguments.output, &encoded)?;
    println!(
        "benchmarked {} observations / {} groups for {} updates",
        selected_paths.len(),
        selected_groups.len(),
        arguments.updates
    );
    Ok(())
}

#[derive(Debug)]
struct GroupSelection {
    selected_groups: Vec<QueryGroup>,
    audit: Value,
}

fn select_matching_groups(
    problem: &GraphProblem,
    groups: &[QueryGroup],
    maximum_groups: usize,
    require_path_match: bool,
) -> Result<GroupSelection, String> {
    let cch = problem.customize_with_oracle(problem.initial_weights(), OracleKind::Cch)?;
    let dijkstra =
        problem.customize_with_oracle(problem.initial_weights(), OracleKind::Dijkstra)?;
    let mut cch_query = cch.new_query();
    let mut dijkstra_query = dijkstra.new_query();
    let mut selected = Vec::new();
    let mut distance_mismatches = 0usize;
    let mut equal_distance_path_ties = 0usize;
    let mut audited = 0usize;
    for group in groups {
        audited += 1;
        let cch_path = cch_query.shortest_path(group.source, group.target)?;
        let dijkstra_path = dijkstra_query.shortest_path(group.source, group.target)?;
        if cch_path.distance != dijkstra_path.distance {
            distance_mismatches += 1;
            continue;
        }
        if cch_path.original_edges != dijkstra_path.original_edges {
            equal_distance_path_ties += 1;
            if require_path_match {
                continue;
            }
        }
        selected.push(*group);
        if selected.len() == maximum_groups {
            break;
        }
    }
    if selected.len() < maximum_groups {
        return Err(format!(
            "only {} eligible groups found, fewer than requested {maximum_groups}",
            selected.len()
        ));
    }
    Ok(GroupSelection {
        selected_groups: selected,
        audit: json!({
            "audited_groups_until_selection_complete": audited,
            "distance_mismatches": distance_mismatches,
            "equal_distance_path_differences": equal_distance_path_ties,
            "selected_initial_path_matches": require_path_match,
        }),
    })
}

#[derive(Debug)]
struct GroupPathComparison {
    distance_mismatches: usize,
    equal_distance_path_differences: BTreeSet<(u32, u32)>,
}

#[allow(clippy::too_many_arguments)]
fn stabilize_path_matching_workload(
    problem: &GraphProblem,
    mut selected_paths: Vec<MappedPath>,
    geometry: OptimizerGeometry,
    eta0: f64,
    lambda: f64,
    updates: u64,
    threads: usize,
) -> Result<(Vec<MappedPath>, Vec<QueryGroup>, Value), String> {
    const MAXIMUM_PASSES: usize = 64;
    let initial_groups = GraphProblem::group_paths(&selected_paths)?.len();
    let mut pass_rows = Vec::new();

    for pass in 1..=MAXIMUM_PASSES {
        let groups = GraphProblem::group_paths(&selected_paths)?;
        let observed = problem.observed_counts(&selected_paths)?;
        let mut weights = problem.initial_weights().to_vec();
        let mut optimizer = ProjectedSubgradientOptimizer::new(geometry, eta0, lambda)?;
        let mut unstable = BTreeSet::new();
        let mut state_rows = Vec::with_capacity(updates as usize + 1);

        for completed_updates in 0..=updates {
            let cch = problem.customize_with_oracle(&weights, OracleKind::Cch)?;
            let dijkstra = problem.customize_with_oracle(&weights, OracleKind::Dijkstra)?;
            let comparison = compare_group_paths(&cch, &dijkstra, &groups, threads)?;
            if comparison.distance_mismatches != 0 {
                return Err(format!(
                    "CCH and Dijkstra had {} distance mismatches during stabilization pass {pass}, state {completed_updates}",
                    comparison.distance_mismatches
                ));
            }
            let path_differences = comparison.equal_distance_path_differences.len();
            unstable.extend(comparison.equal_distance_path_differences);
            state_rows.push(json!({
                "completed_updates": completed_updates,
                "distance_mismatches": 0,
                "equal_distance_path_differences": path_differences,
            }));

            if completed_updates < updates {
                let stats = cch.batch_stats(&groups, threads)?;
                optimizer.step(
                    &mut weights,
                    problem.initial_weights(),
                    problem.lower_bounds(),
                    problem.upper_bounds(),
                    &observed,
                    &stats.predicted_counts,
                    stats.sample_count,
                )?;
            }
        }

        let input_groups = groups.len();
        let removed_groups = unstable.len();
        pass_rows.push(json!({
            "pass": pass,
            "input_groups": input_groups,
            "removed_equal_distance_path_groups": removed_groups,
            "output_groups": input_groups - removed_groups,
            "states": state_rows,
        }));
        if unstable.is_empty() {
            return Ok((
                selected_paths,
                groups,
                json!({
                    "status": "fixed_point",
                    "maximum_passes": MAXIMUM_PASSES,
                    "passes": pass_rows,
                    "initial_groups": initial_groups,
                    "final_groups": input_groups,
                    "removed_groups": initial_groups - input_groups,
                }),
            ));
        }
        selected_paths.retain(|path| !unstable.contains(&(path.source, path.target)));
        if selected_paths.is_empty() {
            return Err("path-match stabilization removed the entire workload".to_string());
        }
    }

    Err(format!(
        "path-match workload did not stabilize within {MAXIMUM_PASSES} passes"
    ))
}

fn compare_group_paths(
    cch: &GraphMetric<'_>,
    dijkstra: &GraphMetric<'_>,
    groups: &[QueryGroup],
    threads: usize,
) -> Result<GroupPathComparison, String> {
    if groups.is_empty() {
        return Ok(GroupPathComparison {
            distance_mismatches: 0,
            equal_distance_path_differences: BTreeSet::new(),
        });
    }
    let chunk_size = groups.len().div_ceil(threads).max(1);
    let locals = groups
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut cch_query = cch.new_query();
            let mut dijkstra_query = dijkstra.new_query();
            let mut distance_mismatches = 0usize;
            let mut path_differences = Vec::new();
            for group in chunk {
                let cch_path = cch_query.shortest_path(group.source, group.target)?;
                let dijkstra_path = dijkstra_query.shortest_path(group.source, group.target)?;
                if cch_path.distance != dijkstra_path.distance {
                    distance_mismatches += 1;
                } else if cch_path.coordinates != dijkstra_path.coordinates {
                    path_differences.push((group.source, group.target));
                }
            }
            Ok((distance_mismatches, path_differences))
        })
        .collect::<Vec<Result<_, String>>>();

    let mut comparison = GroupPathComparison {
        distance_mismatches: 0,
        equal_distance_path_differences: BTreeSet::new(),
    };
    for local in locals {
        let (distance_mismatches, path_differences) = local?;
        comparison.distance_mismatches += distance_mismatches;
        comparison
            .equal_distance_path_differences
            .extend(path_differences);
    }
    Ok(comparison)
}

#[derive(Debug)]
struct TrainingState {
    completed_updates: u64,
    objective: f64,
    mean_regret: f64,
    regularization: f64,
    count_difference_l1: u128,
    weighted_distance_sum: u128,
    predicted_count_checksum: String,
    quantized_weight_checksum: String,
    customization_seconds: f64,
    query_seconds: f64,
    optimizer_seconds: f64,
}

#[derive(Debug)]
struct TrainingRun {
    oracle: OracleKind,
    states: Vec<TrainingState>,
    final_weights: Vec<f64>,
    core_seconds: f64,
    peak_rss_kib: u64,
}

impl TrainingRun {
    fn as_json(&self) -> Value {
        json!({
            "oracle": self.oracle.as_str(),
            "states": self.states.iter().map(|state| json!({
                "completed_updates": state.completed_updates,
                "objective": state.objective,
                "mean_regret": state.mean_regret,
                "regularization": state.regularization,
                "count_difference_l1": state.count_difference_l1.to_string(),
                "weighted_quantized_distance_sum": state.weighted_distance_sum.to_string(),
                "predicted_count_sha256": state.predicted_count_checksum,
                "quantized_weight_sha256": state.quantized_weight_checksum,
                "customization_seconds": state.customization_seconds,
                "query_seconds": state.query_seconds,
                "optimizer_seconds": state.optimizer_seconds,
            })).collect::<Vec<_>>(),
            "timing_totals": {
                "customization_seconds": self.states.iter().map(|state| state.customization_seconds).sum::<f64>(),
                "query_seconds": self.states.iter().map(|state| state.query_seconds).sum::<f64>(),
                "optimizer_seconds": self.states.iter().map(|state| state.optimizer_seconds).sum::<f64>(),
                "core_end_to_end_seconds": self.core_seconds,
            },
            "final_weight_sha256": f64_checksum(&self.final_weights),
            "peak_rss_kib": self.peak_rss_kib,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn run_training_workload(
    problem: &GraphProblem,
    groups: &[QueryGroup],
    observed: &[u64],
    geometry: OptimizerGeometry,
    eta0: f64,
    lambda: f64,
    updates: u64,
    threads: usize,
    oracle: OracleKind,
) -> Result<TrainingRun, String> {
    let mut weights = problem.initial_weights().to_vec();
    let mut optimizer = ProjectedSubgradientOptimizer::new(geometry, eta0, lambda)?;
    let core_started = Instant::now();
    let mut states = Vec::with_capacity(updates as usize + 1);
    for completed_updates in 0..=updates {
        let customization_started = Instant::now();
        let metric = problem.customize_with_oracle(&weights, oracle)?;
        let customization_seconds = customization_started.elapsed().as_secs_f64();
        let stats = metric.batch_stats(groups, threads)?;
        let regret = compute_regret(
            metric.direct_weights(),
            observed,
            stats.weighted_direct_path_cost_sum,
            stats.sample_count,
        )?;
        let regularization = optimizer.regularization(&weights, problem.initial_weights())?;
        let objective = regret.mean_data_loss + regularization;
        let difference = count_difference_l1(&stats.predicted_counts, observed)?;
        let quantized_weight_checksum = u32_checksum(metric.quantized_weights());
        let predicted_count_checksum = u64_checksum(&stats.predicted_counts);
        let optimizer_started = Instant::now();
        if completed_updates < updates {
            optimizer.step(
                &mut weights,
                problem.initial_weights(),
                problem.lower_bounds(),
                problem.upper_bounds(),
                observed,
                &stats.predicted_counts,
                stats.sample_count,
            )?;
        }
        let optimizer_seconds = optimizer_started.elapsed().as_secs_f64();
        states.push(TrainingState {
            completed_updates,
            objective,
            mean_regret: regret.mean_data_loss,
            regularization,
            count_difference_l1: difference,
            weighted_distance_sum: stats.weighted_shortest_distance_sum,
            predicted_count_checksum,
            quantized_weight_checksum,
            customization_seconds,
            query_seconds: stats.oracle_duration.as_secs_f64(),
            optimizer_seconds,
        });
    }
    Ok(TrainingRun {
        oracle,
        states,
        final_weights: weights,
        core_seconds: core_started.elapsed().as_secs_f64(),
        peak_rss_kib: process_peak_rss_kib().unwrap_or(0),
    })
}

fn compare_runs(left: &TrainingRun, right: &TrainingRun) -> Result<Value, String> {
    if left.states.len() != right.states.len()
        || left.final_weights.len() != right.final_weights.len()
    {
        return Err("oracle runs have incompatible state dimensions".to_string());
    }
    let state_rows = left
        .states
        .iter()
        .zip(&right.states)
        .map(|(left, right)| {
            json!({
                "completed_updates": left.completed_updates,
                "distance_sum_equal": left.weighted_distance_sum == right.weighted_distance_sum,
                "predicted_counts_equal": left.predicted_count_checksum == right.predicted_count_checksum,
                "objective_abs_difference": (left.objective - right.objective).abs(),
            })
        })
        .collect::<Vec<_>>();
    let mut max_abs_weight_difference = 0.0f64;
    let mut different_weights = 0usize;
    for (&left, &right) in left.final_weights.iter().zip(&right.final_weights) {
        let difference = (left - right).abs();
        max_abs_weight_difference = max_abs_weight_difference.max(difference);
        different_weights += usize::from(left.to_bits() != right.to_bits());
    }
    Ok(json!({
        "states": state_rows,
        "all_distance_sums_equal": left.states.iter().zip(&right.states).all(|(left, right)| left.weighted_distance_sum == right.weighted_distance_sum),
        "all_predicted_counts_equal": left.states.iter().zip(&right.states).all(|(left, right)| left.predicted_count_checksum == right.predicted_count_checksum),
        "final_weights_bitwise_equal": different_weights == 0,
        "different_final_weights": different_weights,
        "max_abs_final_weight_difference": max_abs_weight_difference,
    }))
}

fn load_manifest_paths(path: &PathBuf, limit: usize) -> Result<Vec<Vec<usize>>, String> {
    let file =
        File::open(path).map_err(|error| format!("failed to open {}: {error}", path.display()))?;
    let mut paths = Vec::new();
    for (line_number, line) in BufReader::new(file).lines().enumerate() {
        if paths.len() == limit {
            break;
        }
        let value: Value = serde_json::from_str(
            &line.map_err(|error| format!("failed to read manifest: {error}"))?,
        )
        .map_err(|error| format!("manifest line {}: {error}", line_number + 1))?;
        let edges = value
            .pointer("/edges")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("manifest line {} lacks edges", line_number + 1))?
            .iter()
            .map(|edge| {
                edge.as_u64()
                    .and_then(|edge| usize::try_from(edge).ok())
                    .ok_or_else(|| format!("manifest line {} has invalid edge", line_number + 1))
            })
            .collect::<Result<Vec<_>, _>>()?;
        paths.push(edges);
    }
    if paths.len() < limit {
        return Err(format!(
            "manifest has {} paths, fewer than requested {limit}",
            paths.len()
        ));
    }
    Ok(paths)
}

fn load_frozen_workload(path: &Path, updates: u64) -> Result<Vec<QueryGroup>, String> {
    let value: Value =
        serde_json::from_slice(&std::fs::read(path).map_err(|error| {
            format!("failed to read frozen workload {}: {error}", path.display())
        })?)
        .map_err(|error| {
            format!(
                "failed to decode frozen workload {}: {error}",
                path.display()
            )
        })?;
    if value.pointer("/test_read").and_then(Value::as_bool) != Some(false) {
        return Err("frozen training workload must have test_read=false".to_string());
    }
    if value.pointer("/invariants/updates").and_then(Value::as_u64) != Some(updates) {
        return Err("frozen workload update count differs from the command".to_string());
    }
    if value
        .pointer("/workload/stabilization_audit/status")
        .and_then(Value::as_str)
        != Some("fixed_point")
    {
        return Err("frozen workload was not produced by fixed-point stabilization".to_string());
    }
    for pointer in [
        "/consistency/all_distance_sums_equal",
        "/consistency/all_predicted_counts_equal",
        "/consistency/final_weights_bitwise_equal",
    ] {
        if value.pointer(pointer).and_then(Value::as_bool) != Some(true) {
            return Err(format!("frozen workload failed invariant {pointer}"));
        }
    }
    let rows = value
        .pointer("/workload/selected_od_groups")
        .and_then(Value::as_array)
        .ok_or_else(|| "frozen workload lacks selected_od_groups".to_string())?;
    if rows.is_empty() {
        return Err("frozen workload contains no OD groups".to_string());
    }
    rows.iter()
        .enumerate()
        .map(|(index, row)| {
            let source = row
                .pointer("/source")
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| format!("frozen workload group {index} has invalid source"))?;
            let target = row
                .pointer("/target")
                .and_then(Value::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| format!("frozen workload group {index} has invalid target"))?;
            let sample_count = row
                .pointer("/sample_count")
                .and_then(Value::as_u64)
                .filter(|&value| value > 0)
                .ok_or_else(|| format!("frozen workload group {index} has invalid sample_count"))?;
            Ok(QueryGroup {
                source,
                target,
                sample_count,
            })
        })
        .collect()
}

fn file_checksum(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("failed to hash {}: {error}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn u32_checksum(values: &[u32]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn u64_checksum(values: &[u64]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn f64_checksum(values: &[f64]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn process_peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        line.strip_prefix("VmHWM:")?
            .split_whitespace()
            .next()?
            .parse()
            .ok()
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RequestedOracle {
    One(OracleKind),
    BothCchFirst,
    BothDijkstraFirst,
}

impl RequestedOracle {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "both" | "both_cch_first" => Ok(Self::BothCchFirst),
            "both_dijkstra_first" => Ok(Self::BothDijkstraFirst),
            value => Ok(Self::One(OracleKind::parse(value)?)),
        }
    }
}

struct Arguments {
    config: PathBuf,
    manifest: PathBuf,
    output: PathBuf,
    oracle: RequestedOracle,
    candidate_samples: usize,
    maximum_groups: usize,
    require_path_match: bool,
    stabilize_path_matches: bool,
    frozen_workload: Option<PathBuf>,
    updates: u64,
    threads: usize,
}

impl Arguments {
    fn from_args() -> Result<Option<Self>, String> {
        let arguments = std::env::args().skip(1).collect::<Vec<_>>();
        if arguments
            .iter()
            .any(|argument| argument == "--help" || argument == "-h")
        {
            println!(
                "Usage: benchmark_training_oracles --config PATH --manifest PATH --output PATH \\\n                 --oracle cch|dijkstra|both_cch_first|both_dijkstra_first \\\n                 --candidate-samples N --maximum-groups N --require-path-match true|false \\\n                 --stabilize-path-matches true|false --frozen-workload none|PATH \\\n                 --updates N --threads N"
            );
            return Ok(None);
        }
        let mut values = BTreeMap::new();
        let mut index = 0;
        while index < arguments.len() {
            let flag = arguments[index].clone();
            let value = arguments
                .get(index + 1)
                .ok_or_else(|| format!("missing value for {flag}"))?
                .clone();
            if values.insert(flag.clone(), value).is_some() {
                return Err(format!("{flag} was provided more than once"));
            }
            index += 2;
        }
        let get = |flag: &str| {
            values
                .get(flag)
                .cloned()
                .ok_or_else(|| format!("missing {flag}"))
        };
        let parse_usize = |flag: &str| -> Result<usize, String> {
            get(flag)?
                .parse()
                .map_err(|_| format!("{flag} must be an integer"))
        };
        let parse_u64 = |flag: &str| -> Result<u64, String> {
            get(flag)?
                .parse()
                .map_err(|_| format!("{flag} must be an integer"))
        };
        let candidate_samples = parse_usize("--candidate-samples")?;
        let maximum_groups = parse_usize("--maximum-groups")?;
        let threads = parse_usize("--threads")?;
        if candidate_samples == 0 || maximum_groups == 0 || threads == 0 {
            return Err("sample, group, and thread counts must be positive".to_string());
        }
        Ok(Some(Self {
            config: PathBuf::from(get("--config")?),
            manifest: PathBuf::from(get("--manifest")?),
            output: PathBuf::from(get("--output")?),
            oracle: RequestedOracle::parse(&get("--oracle")?)?,
            candidate_samples,
            maximum_groups,
            require_path_match: match get("--require-path-match")?.as_str() {
                "true" => true,
                "false" => false,
                _ => return Err("--require-path-match must be true or false".to_string()),
            },
            stabilize_path_matches: match get("--stabilize-path-matches")?.as_str() {
                "true" => true,
                "false" => false,
                _ => return Err("--stabilize-path-matches must be true or false".to_string()),
            },
            frozen_workload: match get("--frozen-workload")?.as_str() {
                "none" => None,
                path => Some(PathBuf::from(path)),
            },
            updates: parse_u64("--updates")?,
            threads,
        }))
    }
}
