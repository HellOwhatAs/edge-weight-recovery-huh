use crate::checkpoint::TrainingCheckpoint;
use crate::config::TrainingConfig;
use crate::data::{GraphData, PathValidationReport};
use crate::evaluation::{PathMetrics, evaluate_paths};
use crate::graph_problem::{GraphOrder, GraphProblem};
use crate::objective::{compute_regret, count_difference_l1, regularization};
use crate::optimizer::ProjectedSubgradientOptimizer;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct TrainingOutcome {
    pub checkpoint_path: PathBuf,
    pub log_path: PathBuf,
    pub graph_order: String,
    pub completed_updates: u64,
    pub train_objective: f64,
    pub validation_objective: f64,
    pub validation_metrics: PathMetrics,
}

/// Run the sole training loop for either graph representation.
pub fn run_training(
    config: &TrainingConfig,
    output_dir: &Path,
    resume_path: Option<&Path>,
) -> Result<TrainingOutcome, String> {
    let actual_threads = rayon::current_num_threads().max(1);
    if actual_threads != config.rayon_threads {
        return Err(format!(
            "configuration requires {} Rayon threads, process has {actual_threads}; set RAYON_NUM_THREADS before launch",
            config.rayon_threads
        ));
    }
    let graph_order = GraphOrder::parse(&config.graph_order)?;
    let mut logger = JsonlLogger::new(output_dir)?;
    logger.log(json!({
        "event": "configuration",
        "run_id": config.run_id,
        "graph_order": graph_order.as_str(),
        "configuration": config.as_json(),
        "resume_path": resume_path,
        "test_read": false,
    }))?;

    let load_started = Instant::now();
    verify_declared_data(config, "train", &config.train_variant)?;
    verify_declared_data(config, "validation", &config.validation_variant)?;
    let graph = crate::data::load_graph(&config.city)?;
    let train =
        crate::data::load_trips(&config.city, "train", &config.train_variant, &graph, None)?;
    let validation = crate::data::load_trips(
        &config.city,
        "validation",
        &config.validation_variant,
        &graph,
        None,
    )?;
    if train.paths.is_empty() || validation.paths.is_empty() {
        return Err("train and validation must both contain valid paths".to_string());
    }
    verify_declared_sample_count(config, "train", &train.report)?;
    verify_declared_sample_count(config, "validation", &validation.report)?;
    log_data_report(&mut logger, "train", &config.train_variant, &train.report)?;
    log_data_report(
        &mut logger,
        "validation",
        &config.validation_variant,
        &validation.report,
    )?;

    let build_started = Instant::now();
    let problem = GraphProblem::build(
        &graph,
        graph_order,
        config.weight_lower_factor,
        config.weight_upper_factor,
    )?;
    let train_paths = problem.map_paths(&train.paths)?;
    let validation_paths = problem.map_paths(&validation.paths)?;
    let train_observed = problem.observed_counts(&train_paths)?;
    let validation_observed = problem.observed_counts(&validation_paths)?;
    let train_groups = GraphProblem::group_paths(&train_paths)?;
    let validation_groups = GraphProblem::group_paths(&validation_paths)?;
    let runtime_identity = runtime_identity(
        config,
        &graph,
        &train.report,
        &validation.report,
        &problem,
        train_paths.len(),
        validation_paths.len(),
    );
    logger.log(json!({
        "event": "graph_problem",
        "graph_order": graph_order.as_str(),
        "original_nodes": graph.x.len(),
        "original_edges": graph.tail.len(),
        "routing_nodes": problem.routing_node_count(),
        "routing_arcs": problem.routing_arc_count(),
        "coordinates": problem.coordinate_count(),
        "train_mapped_paths": train_paths.len(),
        "validation_mapped_paths": validation_paths.len(),
        "train_unique_od": train_groups.len(),
        "validation_unique_od": validation_groups.len(),
        "topology_identity": problem.topology_identity(),
        "build_and_map_ms": milliseconds(build_started),
        "load_and_build_ms": milliseconds(load_started),
        "threads": actual_threads,
    }))?;

    let (mut weights, restored_updates, resumed) = if let Some(path) = resume_path {
        let checkpoint = TrainingCheckpoint::load(path)?;
        validate_resume_checkpoint(&checkpoint, config, &runtime_identity, &problem)?;
        (checkpoint.weights, checkpoint.completed_updates, true)
    } else {
        (problem.initial_weights().to_vec(), 0, false)
    };
    if restored_updates > config.updates {
        return Err(format!(
            "checkpoint has {restored_updates} completed updates but configuration target is {}",
            config.updates
        ));
    }
    // Strictly validate a restored or initialized direct vector before any
    // training query or optimizer mutation.
    problem.customize(&weights)?;
    let mut optimizer = ProjectedSubgradientOptimizer::with_completed_updates(
        config.eta0,
        config.lambda,
        restored_updates,
    )?;
    let initial_metric = problem.customize(problem.initial_weights())?;
    let initial_quantized = initial_metric.quantized_weights().to_vec();

    let mut last_train_objective = f64::NAN;
    let mut last_validation_objective = f64::NAN;
    for completed_updates in restored_updates..=config.updates {
        if optimizer.completed_updates() != completed_updates {
            return Err(format!(
                "optimizer clock mismatch: loop={completed_updates}, optimizer={}",
                optimizer.completed_updates()
            ));
        }
        let state_started = Instant::now();
        let customization_started = Instant::now();
        let metric = problem.customize(&weights)?;
        let customization_ms = milliseconds(customization_started);
        let train_oracle = metric.batch_stats(&train_groups, actual_threads)?;
        let train_regret = compute_regret(
            metric.direct_weights(),
            &train_observed,
            train_oracle.weighted_direct_path_cost_sum,
            train_oracle.sample_count,
        )?;
        let penalty = regularization(&weights, problem.initial_weights(), config.lambda)?;
        let train_objective = train_regret.mean_data_loss + penalty;
        if !train_objective.is_finite() {
            return Err(format!(
                "training objective is not finite at update {completed_updates}"
            ));
        }
        last_train_objective = train_objective;
        let count_difference =
            count_difference_l1(&train_oracle.predicted_counts, &train_observed)?;

        let should_validate = completed_updates == restored_updates
            || completed_updates % config.validation_every == 0
            || completed_updates == config.updates;
        let mut validation_event = Value::Null;
        if should_validate {
            let validation_oracle = metric.batch_stats(&validation_groups, actual_threads)?;
            let validation_regret = compute_regret(
                metric.direct_weights(),
                &validation_observed,
                validation_oracle.weighted_direct_path_cost_sum,
                validation_oracle.sample_count,
            )?;
            last_validation_objective = validation_regret.mean_data_loss + penalty;
            if !last_validation_objective.is_finite() {
                return Err(format!(
                    "validation objective is not finite at update {completed_updates}"
                ));
            }
            validation_event = json!({
                "mean_regret": validation_regret.mean_data_loss,
                "relative_regret": validation_regret.relative_data_loss,
                "objective": last_validation_objective,
                "queries": validation_oracle.num_queries,
                "oracle_ms": validation_oracle.oracle_duration.as_secs_f64() * 1_000.0,
            });

            let checkpoint = make_checkpoint(
                config,
                &runtime_identity,
                &problem,
                completed_updates,
                &weights,
            );
            checkpoint.save(output_dir)?;
            checkpoint.save_to(&output_dir.join(format!("checkpoint-{completed_updates}.json")))?;
        }

        let current_summary = weight_summary(
            &weights,
            problem.initial_weights(),
            problem.lower_bounds(),
            problem.upper_bounds(),
            metric.quantized_weights(),
        );
        let mut update_event = json!({
            "status": "final_skipped",
            "completed_updates_before": completed_updates,
            "completed_updates_after": completed_updates,
        });
        if completed_updates < config.updates {
            let before = weights.clone();
            let optimizer_started = Instant::now();
            let step = optimizer.step(
                &mut weights,
                problem.initial_weights(),
                problem.lower_bounds(),
                problem.upper_bounds(),
                &train_observed,
                &train_oracle.predicted_counts,
                train_oracle.sample_count,
            )?;
            let next_updates = completed_updates
                .checked_add(1)
                .ok_or_else(|| "training update clock overflow".to_string())?;
            if optimizer.completed_updates() != next_updates {
                return Err("optimizer failed to advance the global clock exactly once".to_string());
            }
            let changed_coordinates = changed_f64_count(&before, &weights);
            update_event = json!({
                "status": if changed_coordinates == 0 { "no_direct_change" } else { "applied" },
                "eta": step.eta,
                "max_abs_delta": step.max_abs_delta,
                "projected_coordinates": step.projected_coordinates,
                "changed_coordinates": changed_coordinates,
                "completed_updates_before": completed_updates,
                "completed_updates_after": next_updates,
                "optimizer_ms": milliseconds(optimizer_started),
            });
        }

        logger.log(json!({
            "event": "state",
            "graph_order": graph_order.as_str(),
            "completed_updates": completed_updates,
            "train_mean_regret": train_regret.mean_data_loss,
            "train_relative_regret": train_regret.relative_data_loss,
            "regularization": penalty,
            "train_objective": train_objective,
            "count_difference_l1_diagnostic": count_difference,
            "train_queries": train_oracle.num_queries,
            "train_oracle_ms": train_oracle.oracle_duration.as_secs_f64() * 1_000.0,
            "quantized_shortest_distance_sum": train_oracle.weighted_shortest_distance_sum.to_string(),
            "customization_ms": customization_ms,
            "weights": current_summary,
            "validation": validation_event,
            "update": update_event,
            "state_ms": milliseconds(state_started),
        }))?;
    }

    if !last_validation_objective.is_finite() {
        return Err("final validation objective was not evaluated".to_string());
    }
    let expected_checkpoint = make_checkpoint(
        config,
        &runtime_identity,
        &problem,
        config.updates,
        &weights,
    );
    // The final state is always a validation state, but save once more to make
    // this invariant explicit if cadence logic changes later.
    let checkpoint_path = expected_checkpoint.save(output_dir)?;
    let restored = TrainingCheckpoint::load(&checkpoint_path)?;
    if restored != expected_checkpoint {
        return Err("saved checkpoint does not round-trip the final training state".to_string());
    }
    let restored_metric = problem.customize(&restored.weights)?;
    let validation_metrics = evaluate_paths(&restored_metric, &validation_paths, actual_threads)?;
    logger.log(json!({
        "event": "evaluation",
        "split": "validation_final",
        "metrics": metrics_json(&validation_metrics),
    }))?;

    let changed_coordinates = changed_f64_count(problem.initial_weights(), &restored.weights);
    let changed_quantized_coordinates = initial_quantized
        .iter()
        .zip(restored_metric.quantized_weights())
        .filter(|(initial, final_weight)| initial != final_weight)
        .count();
    logger.log(json!({
        "event": "finished",
        "graph_order": graph_order.as_str(),
        "completed_updates": restored.completed_updates,
        "train_objective": last_train_objective,
        "validation_objective": last_validation_objective,
        "changed_coordinates": changed_coordinates,
        "changed_quantized_coordinates": changed_quantized_coordinates,
        "checkpoint_restore_verified": true,
        "shortest_path_queries_ok": true,
        "resumed": resumed,
        "checkpoint_path": checkpoint_path,
        "topology_identity": problem.topology_identity(),
        "peak_rss_kib": process_peak_rss_kib().unwrap_or(0),
        "test_read": false,
    }))?;

    Ok(TrainingOutcome {
        checkpoint_path,
        log_path: logger.path,
        graph_order: graph_order.as_str().to_string(),
        completed_updates: restored.completed_updates,
        train_objective: last_train_objective,
        validation_objective: last_validation_objective,
        validation_metrics,
    })
}

fn make_checkpoint(
    config: &TrainingConfig,
    runtime_identity: &Value,
    problem: &GraphProblem,
    completed_updates: u64,
    weights: &[f64],
) -> TrainingCheckpoint {
    TrainingCheckpoint {
        graph_order: problem.order().as_str().to_string(),
        completed_updates,
        weights: weights.to_vec(),
        configuration: config.as_json().clone(),
        runtime_identity: runtime_identity.clone(),
        topology_identity: problem.topology_identity().to_string(),
    }
}

fn validate_resume_checkpoint(
    checkpoint: &TrainingCheckpoint,
    config: &TrainingConfig,
    runtime_identity: &Value,
    problem: &GraphProblem,
) -> Result<(), String> {
    if checkpoint.graph_order != problem.order().as_str() {
        return Err("checkpoint graph order does not match the configured graph".to_string());
    }
    if checkpoint.configuration != *config.as_json() {
        return Err("checkpoint configuration does not match the requested run".to_string());
    }
    if checkpoint.runtime_identity != *runtime_identity {
        return Err("checkpoint runtime data identity does not match this run".to_string());
    }
    if checkpoint.topology_identity != problem.topology_identity() {
        return Err("checkpoint topology identity does not match this graph problem".to_string());
    }
    problem.customize(&checkpoint.weights)?;
    Ok(())
}

fn runtime_identity(
    config: &TrainingConfig,
    graph: &GraphData,
    train: &PathValidationReport,
    validation: &PathValidationReport,
    problem: &GraphProblem,
    mapped_train: usize,
    mapped_validation: usize,
) -> Value {
    json!({
        "baseline": {
            "city": config.city,
            "nodes": graph.x.len(),
            "edges": graph.tail.len(),
            "fingerprint": baseline_fingerprint(graph),
        },
        "graph_problem": {
            "order": problem.order().as_str(),
            "coordinates": problem.coordinate_count(),
            "routing_nodes": problem.routing_node_count(),
            "routing_arcs": problem.routing_arc_count(),
            "topology_identity": problem.topology_identity(),
        },
        "train": {
            "variant": config.train_variant,
            "declared": config.as_json().pointer("/data/train_identity").cloned().unwrap_or(Value::Null),
            "available": train.available_samples,
            "accepted": train.accepted_samples,
            "mapped": mapped_train,
        },
        "validation": {
            "variant": config.validation_variant,
            "declared": config.as_json().pointer("/data/validation_identity").cloned().unwrap_or(Value::Null),
            "available": validation.available_samples,
            "accepted": validation.accepted_samples,
            "mapped": mapped_validation,
        },
    })
}

fn verify_declared_data(config: &TrainingConfig, split: &str, variant: &str) -> Result<(), String> {
    let pointer = format!("/data/{split}_identity");
    let identity = config
        .as_json()
        .pointer(&pointer)
        .ok_or_else(|| format!("configuration is missing {pointer}"))?;
    let expected_path = format!(
        "data/{}_data/preprocessed_{split}_trips_{variant}.pkl",
        config.city
    );
    let declared_path = identity
        .pointer("/path")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("configuration is missing {pointer}/path"))?;
    if declared_path != expected_path {
        return Err(format!(
            "declared {split} path {declared_path:?} does not match {expected_path:?}"
        ));
    }
    let declared_bytes = identity
        .pointer("/bytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("configuration is missing {pointer}/bytes"))?;
    let metadata = std::fs::metadata(&expected_path)
        .map_err(|error| format!("failed to inspect {expected_path}: {error}"))?;
    if metadata.len() != declared_bytes {
        return Err(format!(
            "{expected_path} has {} bytes, expected {declared_bytes}",
            metadata.len()
        ));
    }
    let declared_hash = identity
        .pointer("/sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("configuration is missing {pointer}/sha256"))?;
    let actual_hash = sha256_file(Path::new(&expected_path))?;
    if actual_hash != declared_hash {
        return Err(format!(
            "{expected_path} SHA-256 mismatch: expected {declared_hash}, got {actual_hash}"
        ));
    }
    Ok(())
}

fn verify_declared_sample_count(
    config: &TrainingConfig,
    split: &str,
    report: &PathValidationReport,
) -> Result<(), String> {
    let pointer = format!("/data/{split}_identity/sample_count");
    if let Some(declared) = config.as_json().pointer(&pointer).and_then(Value::as_u64)
        && declared != report.available_samples as u64
    {
        return Err(format!(
            "{split} sample count mismatch: declared {declared}, loaded {}",
            report.available_samples
        ));
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path)
        .map_err(|error| format!("failed to open {} for hashing: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("failed to hash {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn log_data_report(
    logger: &mut JsonlLogger,
    split: &str,
    variant: &str,
    report: &PathValidationReport,
) -> Result<(), String> {
    logger.log(json!({
        "event": "data",
        "split": split,
        "variant": variant,
        "available": report.available_samples,
        "inspected": report.inspected_samples,
        "accepted": report.accepted_samples,
        "dropped": report.dropped_samples(),
        "cyclic": report.cyclic,
        "empty": report.empty,
        "out_of_bounds": report.out_of_bounds,
        "discontinuous": report.discontinuous,
        "policy": "complete_paths_drop_cycles",
    }))
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

fn weight_summary(
    weights: &[f64],
    initial: &[f64],
    lower: &[f64],
    upper: &[f64],
    quantized: &[u32],
) -> Value {
    let min = weights.iter().copied().fold(f64::INFINITY, f64::min);
    let max = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let max_abs_drift = weights
        .iter()
        .zip(initial)
        .map(|(&weight, &anchor)| (weight - anchor).abs())
        .fold(0.0, f64::max);
    let max_quantization_error = weights
        .iter()
        .zip(quantized)
        .map(|(&weight, &integer)| (weight - integer as f64).abs())
        .fold(0.0, f64::max);
    json!({
        "min": min,
        "max": max,
        "changed_from_initial": changed_f64_count(weights, initial),
        "max_abs_drift": max_abs_drift,
        "at_lower_bound": weights.iter().zip(lower).filter(|(weight, bound)| *weight <= *bound).count(),
        "at_upper_bound": weights.iter().zip(upper).filter(|(weight, bound)| *weight >= *bound).count(),
        "max_quantization_error": max_quantization_error,
    })
}

fn changed_f64_count(left: &[f64], right: &[f64]) -> usize {
    left.iter()
        .zip(right)
        .filter(|(left, right)| left.to_bits() != right.to_bits())
        .count()
}

fn metrics_json(metrics: &PathMetrics) -> Value {
    json!({
        "samples": metrics.sample_count,
        "mean_regret": metrics.mean_regret,
        "relative_regret": metrics.relative_regret,
        "exact_match": metrics.exact_match,
        "edge_precision": metrics.edge_precision,
        "edge_recall": metrics.edge_recall,
        "edge_f1": metrics.edge_f1,
        "edge_jaccard": metrics.edge_jaccard,
    })
}

fn process_peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

fn milliseconds(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

struct JsonlLogger {
    path: PathBuf,
    file: File,
}

impl JsonlLogger {
    fn new(output_dir: &Path) -> Result<Self, String> {
        std::fs::create_dir_all(output_dir)
            .map_err(|error| format!("failed to create {}: {error}", output_dir.display()))?;
        let path = output_dir.join("training.jsonl");
        let file = File::create(&path)
            .map_err(|error| format!("failed to create {}: {error}", path.display()))?;
        Ok(Self { path, file })
    }

    fn log(&mut self, event: Value) -> Result<(), String> {
        let line = serde_json::to_string(&event)
            .map_err(|error| format!("failed to encode structured log event: {error}"))?;
        println!("{line}");
        writeln!(self.file, "{line}")
            .map_err(|error| format!("failed to write {}: {error}", self.path.display()))?;
        self.file
            .flush()
            .map_err(|error| format!("failed to flush {}: {error}", self.path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TripPath;

    fn graph() -> GraphData {
        GraphData {
            tail: vec![0, 1, 0, 2],
            head: vec![1, 3, 2, 3],
            baseline_weights: vec![5, 5, 2, 2],
            x: vec![0.0, 1.0, 1.0, 2.0],
            y: vec![0.0, 0.0, 1.0, 0.0],
        }
    }

    fn advance(
        problem: &GraphProblem,
        paths: &[TripPath],
        weights: &mut [f64],
        optimizer: &mut ProjectedSubgradientOptimizer,
        steps: usize,
    ) {
        let mapped = problem.map_paths(paths).unwrap();
        let observed = problem.observed_counts(&mapped).unwrap();
        let groups = GraphProblem::group_paths(&mapped).unwrap();
        for _ in 0..steps {
            let metric = problem.customize(weights).unwrap();
            let oracle = metric.batch_stats(&groups, 2).unwrap();
            optimizer
                .step(
                    weights,
                    problem.initial_weights(),
                    problem.lower_bounds(),
                    problem.upper_bounds(),
                    &observed,
                    &oracle.predicted_counts,
                    oracle.sample_count,
                )
                .unwrap();
        }
    }

    #[test]
    fn both_graph_orders_use_the_same_direct_weight_optimizer() {
        let graph = graph();
        let paths = vec![((0, 3), vec![0, 1]), ((0, 3), vec![0, 1])];
        for order in [GraphOrder::First, GraphOrder::Second] {
            let problem = GraphProblem::build(&graph, order, 0.1, 10.0).unwrap();
            let initial = problem.initial_weights().to_vec();
            let mut weights = initial.clone();
            let mut optimizer = ProjectedSubgradientOptimizer::new(0.5, 0.1).unwrap();
            advance(&problem, &paths, &mut weights, &mut optimizer, 1);
            assert_eq!(optimizer.completed_updates(), 1);
            assert_ne!(weights, initial);
        }
    }

    #[test]
    fn checkpoint_resume_matches_uninterrupted_training_for_both_orders() {
        let graph = graph();
        let paths = vec![((0, 3), vec![0, 1]), ((0, 3), vec![0, 1])];
        for order in [GraphOrder::First, GraphOrder::Second] {
            let problem = GraphProblem::build(&graph, order, 0.1, 10.0).unwrap();
            let mut uninterrupted_weights = problem.initial_weights().to_vec();
            let mut uninterrupted = ProjectedSubgradientOptimizer::new(0.5, 0.1).unwrap();
            advance(
                &problem,
                &paths,
                &mut uninterrupted_weights,
                &mut uninterrupted,
                4,
            );

            let mut resumed_weights = problem.initial_weights().to_vec();
            let mut first_half = ProjectedSubgradientOptimizer::new(0.5, 0.1).unwrap();
            advance(&problem, &paths, &mut resumed_weights, &mut first_half, 2);
            let checkpoint = TrainingCheckpoint {
                graph_order: order.as_str().to_string(),
                completed_updates: first_half.completed_updates(),
                weights: resumed_weights,
                configuration: json!({"fixture": true}),
                runtime_identity: json!({"fixture": true}),
                topology_identity: problem.topology_identity().to_string(),
            };
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path =
                std::env::temp_dir().join(format!("direct-resume-{}-{nonce}.json", order.as_str()));
            checkpoint.save_to(&path).unwrap();
            let restored = TrainingCheckpoint::load(&path).unwrap();
            std::fs::remove_file(path).unwrap();

            let mut resumed_weights = restored.weights;
            let mut resumed = ProjectedSubgradientOptimizer::with_completed_updates(
                0.5,
                0.1,
                restored.completed_updates,
            )
            .unwrap();
            advance(&problem, &paths, &mut resumed_weights, &mut resumed, 2);

            assert_eq!(resumed_weights, uninterrupted_weights);
            assert_eq!(
                resumed.completed_updates(),
                uninterrupted.completed_updates()
            );
        }
    }
}
