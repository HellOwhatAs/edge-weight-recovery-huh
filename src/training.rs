use crate::config::{TrainingConfig, TrainingState};
use crate::data::{
    GraphData, PathValidationReport, compute_observed_edge_counts, group_paths_by_od,
};
use crate::evaluation::{PathMetrics, evaluate_paths};
use crate::model::EdgeOnlyModel;
use crate::objective::{compute_regret, count_residual_l1};
use crate::optimizer::ProjectedSubgradientOptimizer;
use crate::oracle::CchOracle;
use serde_json::{Value, json};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct TrainingOutcome {
    pub checkpoint_path: PathBuf,
    pub log_path: PathBuf,
    pub best_epoch: u64,
    pub selection_value: f64,
    pub validation_metrics: PathMetrics,
}

/// Run the sole production training path: edge-only projected subgradient,
/// full CCH customization, and validation-only checkpoint selection.
pub fn run_training(config: &TrainingConfig, output_dir: &Path) -> Result<TrainingOutcome, String> {
    let mut logger = JsonlLogger::new(output_dir)?;
    logger.log(json!({
        "event": "configuration",
        "run_id": config.run_id,
        "configuration": config.as_json(),
        "test_read": false,
    }))?;

    let load_started = Instant::now();
    let graph = crate::data::load_graph(&config.city)?;
    let train =
        crate::data::load_trips(&config.city, "train", &config.train_variant, &graph, None)?;
    if train.paths.is_empty() {
        return Err("no valid training paths remain after validation".to_string());
    }
    log_data_report(&mut logger, "train", &config.train_variant, &train.report)?;

    let validation = crate::data::load_trips(
        &config.city,
        "validation",
        &config.validation_variant,
        &graph,
        None,
    )?;
    if validation.paths.is_empty() {
        return Err("no valid validation paths remain after validation".to_string());
    }
    log_data_report(
        &mut logger,
        "validation",
        &config.validation_variant,
        &validation.report,
    )?;

    let runtime_identity = runtime_identity(config, &graph, &train.report, &validation.report);
    logger.log(json!({
        "event": "loaded",
        "nodes": graph.x.len(),
        "edges": graph.tail.len(),
        "valid_train": train.paths.len(),
        "valid_validation": validation.paths.len(),
        "wall_ms": milliseconds(load_started),
        "runtime_identity": &runtime_identity,
    }))?;

    let thread_count = rayon::current_num_threads().max(1);
    let edge_count = graph.tail.len();
    let train_observed = compute_observed_edge_counts(&train.paths, edge_count, thread_count);
    let validation_observed =
        compute_observed_edge_counts(&validation.paths, edge_count, thread_count);
    let train_groups = group_paths_by_od(&train.paths);
    let validation_groups = group_paths_by_od(&validation.paths);
    logger.log(json!({
        "event": "od_grouping",
        "train_samples": train.paths.len(),
        "train_unique_od": train_groups.len(),
        "validation_samples": validation.paths.len(),
        "validation_unique_od": validation_groups.len(),
    }))?;

    let build_started = Instant::now();
    let oracle = CchOracle::build(&graph)?;
    let build_ms = milliseconds(build_started);
    let mut model = EdgeOnlyModel::new(&graph.baseline_weights, config.quantization_scale)?;
    let initial_weights = model.quantized_weights()?;
    let customization_started = Instant::now();
    let mut metric = oracle.customize(&initial_weights)?;
    logger.log(json!({
        "event": "oracle_built",
        "kind": "cch",
        "customization": "full",
        "build_ms": build_ms,
        "initial_customization_ms": milliseconds(customization_started),
        "threads": thread_count,
    }))?;

    let mut optimizer = ProjectedSubgradientOptimizer::new(
        config.eta0,
        config.lambda_edge,
        config.q_min,
        config.q_max,
    )?;
    let mut state = TrainingState::new(metric.weights(), model.q());
    let mut initial_train_mean_regret = None;

    for epoch in 0..config.epochs {
        let epoch_started = Instant::now();
        let train_oracle = oracle.batch_stats(&metric, &train_groups, thread_count)?;
        let train_regret = compute_regret(metric.weights(), &train_observed, &train_oracle)?;
        initial_train_mean_regret.get_or_insert(train_regret.mean_data_loss);
        let count_residual =
            count_residual_l1(&train_oracle.predicted_edge_counts, &train_observed)?;
        let regularization = model.regularization(config.lambda_edge);
        let train_objective = train_regret.mean_data_loss + regularization;
        let current_q = q_summary(model.q(), config.q_min, config.q_max);
        let current_quantization_error =
            max_quantization_error(metric.weights(), model.metric_baseline(), model.q());

        let should_validate =
            epoch == 0 || (epoch + 1) % config.validation_every == 0 || epoch + 1 == config.epochs;
        let mut validation_event = Value::Null;
        let mut is_best = false;
        let mut should_stop = false;
        if should_validate {
            let validation_oracle =
                oracle.batch_stats(&metric, &validation_groups, thread_count)?;
            let validation_regret =
                compute_regret(metric.weights(), &validation_observed, &validation_oracle)?;
            is_best = state.update(
                epoch,
                validation_regret.relative_data_loss,
                train_regret.mean_data_loss,
                metric.weights(),
                model.q(),
                config.early_stop_min_delta,
            );
            if is_best {
                state.save_checkpoint(output_dir, config, &runtime_identity)?;
            }
            should_stop = state.stale_evaluations >= config.early_stop_patience;
            validation_event = json!({
                "mean_regret": validation_regret.mean_data_loss,
                "relative_regret": validation_regret.relative_data_loss,
                "selection_value": validation_regret.relative_data_loss,
                "queries": validation_oracle.num_queries,
                "oracle_ms": validation_oracle.oracle_duration.as_secs_f64() * 1_000.0,
                "stale_evaluations": state.stale_evaluations,
                "is_best": is_best,
            });
        }

        let mut update = json!({"status": "not_scheduled"});
        if epoch + 1 == config.epochs {
            update = json!({"status": "final_skipped"});
        } else if should_stop {
            update = json!({"status": "early_stop_skipped"});
        } else {
            let optimizer_started = Instant::now();
            let step = model.projected_step(
                &mut optimizer,
                &train_observed,
                &train_oracle.predicted_edge_counts,
                train_oracle.sample_count,
            );
            let optimizer_ms = milliseconds(optimizer_started);
            let next_weights = model.quantized_weights()?;
            let changed_edges = metric
                .weights()
                .iter()
                .zip(&next_weights)
                .filter(|(current, next)| current != next)
                .count();
            let customization_started = Instant::now();
            let customization_ms = if changed_edges == 0 {
                0.0
            } else {
                metric = oracle.customize(&next_weights)?;
                milliseconds(customization_started)
            };
            update = json!({
                "status": if changed_edges == 0 { "latent_only_no_integer_change" } else { "applied" },
                "eta": step.eta,
                "latent_max_delta": step.max_abs_delta,
                "projected_edges": step.projected_edges,
                "changed_edges": changed_edges,
                "changed_pct": 100.0 * changed_edges as f64 / edge_count as f64,
                "optimizer_ms": optimizer_ms,
                "customization_ms": customization_ms,
            });
        }

        let next_q = q_summary(model.q(), config.q_min, config.q_max);
        let next_quantization_error =
            max_quantization_error(metric.weights(), model.metric_baseline(), model.q());
        logger.log(json!({
            "event": "epoch",
            "epoch": epoch,
            "train_mean_regret": train_regret.mean_data_loss,
            "train_relative_regret": train_regret.relative_data_loss,
            "regularization": regularization,
            "train_objective": train_objective,
            "count_residual_l1_diagnostic": count_residual,
            "train_queries": train_oracle.num_queries,
            "train_oracle_ms": train_oracle.oracle_duration.as_secs_f64() * 1_000.0,
            "current_q": current_q,
            "current_max_quantization_error": current_quantization_error,
            "validation": validation_event,
            "update": update,
            "next_q": next_q,
            "next_max_quantization_error": next_quantization_error,
            "epoch_ms": milliseconds(epoch_started),
            "is_best": is_best,
        }))?;
        if should_stop {
            logger.log(json!({
                "event": "early_stop",
                "epoch": epoch,
                "stale_evaluations": state.stale_evaluations,
                "patience": config.early_stop_patience,
                "min_delta": config.early_stop_min_delta,
            }))?;
            break;
        }
    }

    let checkpoint_path = state.save_checkpoint(output_dir, config, &runtime_identity)?;
    let restore_started = Instant::now();
    metric = oracle.customize(&state.best_weights)?;
    let validation_metrics = evaluate_paths(&metric, &validation.paths, thread_count)?;
    logger.log(json!({
        "event": "evaluation",
        "split": "validation_best",
        "metrics": metrics_json(&validation_metrics),
    }))?;

    let initial = initial_train_mean_regret.unwrap_or(0.0);
    let improvement = if initial > 0.0 {
        100.0 * (initial - state.best_train_mean_regret) / initial
    } else {
        0.0
    };
    let best_q = q_summary(&state.best_q, config.q_min, config.q_max);
    logger.log(json!({
        "event": "finished",
        "best_epoch": state.best_epoch,
        "selection_metric": "aggregate_validation_relative_regret",
        "selection_value": state.best_selection_value,
        "best_train_mean_regret": state.best_train_mean_regret,
        "best_regularization": regularization(&state.best_q, config.lambda_edge),
        "best_q": best_q,
        "train_regret_improvement_pct": improvement,
        "restore_full_customization_ms": milliseconds(restore_started),
        "peak_rss_kib": process_peak_rss_kib().unwrap_or(0),
        "checkpoint_path": checkpoint_path,
        "test_read": false,
    }))?;

    Ok(TrainingOutcome {
        checkpoint_path,
        log_path: logger.path,
        best_epoch: state.best_epoch,
        selection_value: state.best_selection_value,
        validation_metrics,
    })
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

fn runtime_identity(
    config: &TrainingConfig,
    graph: &GraphData,
    train: &PathValidationReport,
    validation: &PathValidationReport,
) -> Value {
    json!({
        "baseline": {
            "city": config.city,
            "nodes": graph.x.len(),
            "edges": graph.tail.len(),
            "fnv1a64": baseline_fingerprint(graph),
        },
        "train": {
            "variant": config.train_variant,
            "declared": config.as_json().pointer("/data/train_identity").cloned().unwrap_or(Value::Null),
            "available": train.available_samples,
            "inspected": train.inspected_samples,
            "accepted": train.accepted_samples,
        },
        "validation": {
            "variant": config.validation_variant,
            "declared": config.as_json().pointer("/data/validation_identity").cloned().unwrap_or(Value::Null),
            "available": validation.available_samples,
            "inspected": validation.inspected_samples,
            "accepted": validation.accepted_samples,
        }
    })
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

fn q_summary(q: &[f64], lower: f64, upper: f64) -> Value {
    let min = q.iter().copied().fold(f64::INFINITY, f64::min);
    let max = q.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    json!({
        "min": min,
        "max": max,
        "at_min": q.iter().filter(|&&value| value <= lower + 1e-12).count(),
        "at_max": q.iter().filter(|&&value| value >= upper - 1e-12).count(),
    })
}

fn max_quantization_error(weights: &[u32], baseline: &[u32], q: &[f64]) -> f64 {
    weights
        .iter()
        .zip(baseline)
        .zip(q)
        .map(|((&weight, &base), &multiplier)| (weight as f64 - base as f64 * multiplier).abs())
        .fold(0.0, f64::max)
}

fn regularization(q: &[f64], lambda: f64) -> f64 {
    if q.is_empty() {
        return 0.0;
    }
    lambda * q.iter().map(|value| (value - 1.0).powi(2)).sum::<f64>() / (2.0 * q.len() as f64)
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
