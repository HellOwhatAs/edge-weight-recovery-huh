use crate::data::{GraphData, PathValidationReport, TimestampEvidence};
use crate::evaluation::{PathMetrics, combine_path_metrics, evaluate_paths};
use crate::graph_problem::{GraphProblem, GraphRepresentation, MappedPath, QueryGroup};
use crate::objective::{count_difference_l1, observed_cost};
use crate::temporal::{
    BaselineModel, TemporalCheckpoint, TemporalParameters, TemporalProjectedSubgradientOptimizer,
    TemporalTrainingConfig, TimeBucketSpec, estimate_baseline_model, sha256_file,
};
use serde_json::{Value, json};
use std::fs::File;
use std::io::{Write, stdout};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct TemporalTrainingOutcome {
    pub checkpoint_path: PathBuf,
    pub log_path: PathBuf,
    pub completed_updates: u64,
    pub train_objective: f64,
    pub validation_objective: f64,
    pub validation_metrics: PathMetrics,
}

struct BucketWorkload {
    paths: Vec<MappedPath>,
    observed: Vec<u64>,
    groups: Vec<QueryGroup>,
}

struct BatchSnapshot {
    predicted_by_bucket: Vec<Vec<u64>>,
    observed_cost_sum: f64,
    predicted_cost_sum: f64,
    sample_count: u64,
    queries: usize,
    oracle_ms: f64,
    count_difference_l1: u128,
    quantization: Vec<Value>,
}

impl BatchSnapshot {
    fn mean_regret(&self) -> f64 {
        (self.observed_cost_sum - self.predicted_cost_sum) / self.sample_count as f64
    }

    fn relative_regret(&self) -> f64 {
        if self.observed_cost_sum == 0.0 {
            0.0
        } else {
            (self.observed_cost_sum - self.predicted_cost_sum) / self.observed_cost_sum
        }
    }
}

pub fn run_temporal_training(
    config: &TemporalTrainingConfig,
    output_dir: &Path,
    resume_path: Option<&Path>,
) -> Result<TemporalTrainingOutcome, String> {
    let actual_threads = rayon::current_num_threads().max(1);
    if actual_threads != config.rayon_threads {
        return Err(format!(
            "configuration requires {} Rayon threads, process has {actual_threads}; set RAYON_NUM_THREADS before launch",
            config.rayon_threads
        ));
    }
    let mut logger = JsonlLogger::new(output_dir)?;
    let bucket_spec = config.load_bucket_spec()?;
    logger.log(json!({
        "event": "configuration",
        "run_id": config.run_id,
        "graph_representation": config.graph_representation,
        "optimizer_kind": "relative_projected_subgradient",
        "model_kind": "global_plus_bucket_residual",
        "parameterization": "q_bucket = q_global + residual_bucket",
        "configuration": config.as_json(),
        "time_bucket_specification": bucket_spec.as_json(),
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
    validate_retained_times("train", &train.times)?;
    validate_retained_times("validation", &validation.times)?;
    log_data_report(
        &mut logger,
        "train",
        &config.train_variant,
        &train.report,
        &train.timestamp_evidence,
    )?;
    log_data_report(
        &mut logger,
        "validation",
        &config.validation_variant,
        &validation.report,
        &validation.timestamp_evidence,
    )?;

    // This is the only baseline-estimation call. Its API accepts training
    // paths/timestamps only, preventing validation statistics from entering
    // either speed estimates or smoothing pseudocounts.
    let baseline_model =
        estimate_baseline_model(&graph, &train.paths, &train.times, &bucket_spec, config)?;
    logger.log(json!({
        "event": "baseline_model",
        "kind": baseline_model.kind.as_str(),
        "diagnostics": baseline_model.diagnostics,
        "time_bucket_specification": bucket_spec.as_json(),
        "validation_used": false,
        "test_used": false,
    }))?;

    let build_started = Instant::now();
    let problem = GraphProblem::build(
        &graph,
        GraphRepresentation::EdgeTransitionArcs,
        config.weight_lower_factor,
        config.weight_upper_factor,
    )?;
    let coordinate_initials = baseline_model
        .edge_weights_by_bucket
        .iter()
        .map(|weights| problem.coordinate_weights_from_original(weights))
        .collect::<Result<Vec<_>, _>>()?;
    for (bucket, initial) in coordinate_initials.iter().enumerate() {
        let configured_upper = initial
            .iter()
            .map(|weight| weight * config.weight_upper_factor)
            .collect::<Vec<_>>();
        problem
            .customize_external(&configured_upper)
            .map_err(|error| {
                format!(
                    "time bucket {bucket} configured upper weights exceed CCH capacity: {error}"
                )
            })?;
    }
    let initial_quantized = coordinate_initials
        .iter()
        .map(|weights| {
            problem
                .customize_external(weights)
                .map(|metric| metric.quantized_weights().to_vec())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mapped_train = problem.map_paths(&train.paths)?;
    let mapped_validation = problem.map_paths(&validation.paths)?;
    let train_workloads = partition_workloads(&problem, mapped_train, &train.times, &bucket_spec)?;
    let validation_workloads =
        partition_workloads(&problem, mapped_validation, &validation.times, &bucket_spec)?;
    if train_workloads
        .iter()
        .any(|workload| workload.paths.is_empty())
        || validation_workloads
            .iter()
            .any(|workload| workload.paths.is_empty())
    {
        return Err(
            "every pre-registered time bucket must have train and validation paths".to_string(),
        );
    }
    let runtime_identity = runtime_identity(
        config,
        &graph,
        &train.report,
        &validation.report,
        &problem,
        &bucket_spec,
        &baseline_model,
        &train_workloads,
        &validation_workloads,
    );
    logger.log(json!({
        "event": "graph_problem",
        "graph_representation": problem.representation().as_str(),
        "model_kind": "global_plus_bucket_residual",
        "original_nodes": graph.x.len(),
        "original_edges": graph.tail.len(),
        "routing_nodes": problem.routing_node_count(),
        "routing_arcs": problem.routing_arc_count(),
        "coordinates": problem.coordinate_count(),
        "train_mapped_paths": train_workloads.iter().map(|workload| workload.paths.len()).sum::<usize>(),
        "validation_mapped_paths": validation_workloads.iter().map(|workload| workload.paths.len()).sum::<usize>(),
        "train_buckets": workload_summary(&bucket_spec, &train_workloads),
        "validation_buckets": workload_summary(&bucket_spec, &validation_workloads),
        "topology_identity": problem.topology_identity(),
        "build_and_map_ms": milliseconds(build_started),
        "load_and_build_ms": milliseconds(load_started),
        "threads": actual_threads,
    }))?;

    let (mut parameters, restored_updates, resumed) = if let Some(path) = resume_path {
        let checkpoint = TemporalCheckpoint::load(path)?;
        validate_resume_checkpoint(
            &checkpoint,
            config,
            &runtime_identity,
            &problem,
            &bucket_spec,
            &baseline_model,
        )?;
        (checkpoint.parameters, checkpoint.completed_updates, true)
    } else {
        (
            TemporalParameters::initial(problem.coordinate_count(), bucket_spec.buckets.len())?,
            0,
            false,
        )
    };
    if restored_updates > config.updates {
        return Err(format!(
            "checkpoint has {restored_updates} updates but target is {}",
            config.updates
        ));
    }
    parameters.validate_for_config(
        config,
        problem.coordinate_count(),
        bucket_spec.buckets.len(),
    )?;
    let mut optimizer =
        TemporalProjectedSubgradientOptimizer::with_completed_updates(config, restored_updates);

    let mut last_train_objective = f64::NAN;
    let mut last_validation_objective = f64::NAN;
    for completed_updates in restored_updates..=config.updates {
        if optimizer.completed_updates() != completed_updates {
            return Err(format!(
                "temporal optimizer clock mismatch: loop={completed_updates}, optimizer={}",
                optimizer.completed_updates()
            ));
        }
        let state_started = Instant::now();
        let train_snapshot = evaluate_batch(
            &problem,
            &parameters,
            &coordinate_initials,
            &initial_quantized,
            &train_workloads,
            actual_threads,
        )?;
        let penalty = optimizer.regularization(&parameters)?;
        let train_objective = train_snapshot.mean_regret() + penalty;
        if !train_objective.is_finite() {
            return Err(format!(
                "training objective is not finite at update {completed_updates}"
            ));
        }
        last_train_objective = train_objective;

        let should_validate = completed_updates == restored_updates
            || completed_updates % config.validation_every == 0
            || completed_updates == config.updates;
        let mut validation_event = Value::Null;
        if should_validate {
            let snapshot = evaluate_batch(
                &problem,
                &parameters,
                &coordinate_initials,
                &initial_quantized,
                &validation_workloads,
                actual_threads,
            )?;
            last_validation_objective = snapshot.mean_regret() + penalty;
            if !last_validation_objective.is_finite() {
                return Err(format!(
                    "validation objective is not finite at update {completed_updates}"
                ));
            }
            validation_event = json!({
                "mean_regret": snapshot.mean_regret(),
                "relative_regret": snapshot.relative_regret(),
                "objective": last_validation_objective,
                "queries": snapshot.queries,
                "oracle_ms": snapshot.oracle_ms,
                "quantization": snapshot.quantization,
            });
            let checkpoint = make_checkpoint(
                config,
                &runtime_identity,
                &problem,
                completed_updates,
                &parameters,
                &bucket_spec,
                &baseline_model,
            );
            checkpoint.save(output_dir)?;
            checkpoint.save_to(&output_dir.join(format!("checkpoint-{completed_updates}.json")))?;
        }

        let mut update_event = json!({
            "status": "final_skipped",
            "completed_updates_before": completed_updates,
            "completed_updates_after": completed_updates,
        });
        if completed_updates < config.updates {
            let step = optimizer.step(
                &mut parameters,
                &coordinate_initials,
                &train_workloads
                    .iter()
                    .map(|workload| workload.observed.clone())
                    .collect::<Vec<_>>(),
                &train_snapshot.predicted_by_bucket,
                train_snapshot.sample_count,
                config,
            )?;
            update_event = json!({
                "status": "applied",
                "global_eta": step.global_eta,
                "residual_eta": step.residual_eta,
                "max_abs_global_delta": step.max_abs_global_delta,
                "max_abs_residual_delta": step.max_abs_residual_delta,
                "projected_global_coordinates": step.projected_global_coordinates,
                "projected_residual_coordinates": step.projected_residual_coordinates,
                "completed_updates_before": completed_updates,
                "completed_updates_after": optimizer.completed_updates(),
            });
        }

        logger.log(json!({
            "event": "state",
            "graph_representation": problem.representation().as_str(),
            "optimizer_kind": "relative_projected_subgradient",
            "model_kind": "global_plus_bucket_residual",
            "completed_updates": completed_updates,
            "train_mean_regret": train_snapshot.mean_regret(),
            "train_relative_regret": train_snapshot.relative_regret(),
            "regularization": penalty,
            "train_objective": train_objective,
            "count_difference_l1_diagnostic": train_snapshot.count_difference_l1,
            "train_queries": train_snapshot.queries,
            "train_oracle_ms": train_snapshot.oracle_ms,
            "parameters": parameter_summary(&parameters, config),
            "quantization": train_snapshot.quantization,
            "validation": validation_event,
            "update": update_event,
            "state_ms": milliseconds(state_started),
        }))?;
    }

    if !last_validation_objective.is_finite() {
        return Err("final temporal validation objective was not evaluated".to_string());
    }
    let expected_checkpoint = make_checkpoint(
        config,
        &runtime_identity,
        &problem,
        config.updates,
        &parameters,
        &bucket_spec,
        &baseline_model,
    );
    let checkpoint_path = expected_checkpoint.save(output_dir)?;
    let restored = TemporalCheckpoint::load(&checkpoint_path)?;
    if restored != expected_checkpoint {
        return Err("temporal checkpoint did not round-trip exactly".to_string());
    }
    let (validation_metrics, per_bucket_metrics) = evaluate_decoded_paths(
        &problem,
        &restored.parameters,
        &coordinate_initials,
        &validation_workloads,
        actual_threads,
    )?;
    logger.log(json!({
        "event": "evaluation",
        "split": "validation_final",
        "metrics": metrics_json(&validation_metrics),
        "time_buckets": bucket_spec.buckets.iter().zip(&per_bucket_metrics).map(|(bucket, metrics)| json!({
            "id": bucket.id,
            "start_hour": bucket.start_hour,
            "end_hour": bucket.end_hour,
            "metrics": metrics_json(metrics),
        })).collect::<Vec<_>>(),
    }))?;
    let changed_global = restored
        .parameters
        .global_relative
        .iter()
        .filter(|&&value| value.to_bits() != 1.0f64.to_bits())
        .count();
    let changed_residual = restored
        .parameters
        .bucket_residuals
        .iter()
        .flatten()
        .filter(|&&value| value.to_bits() != 0.0f64.to_bits())
        .count();
    logger.log(json!({
        "event": "finished",
        "graph_representation": problem.representation().as_str(),
        "optimizer_kind": "relative_projected_subgradient",
        "model_kind": "global_plus_bucket_residual",
        "completed_updates": restored.completed_updates,
        "train_objective": last_train_objective,
        "validation_objective": last_validation_objective,
        "changed_global_coordinates": changed_global,
        "changed_residual_coordinates": changed_residual,
        "checkpoint_restore_verified": true,
        "shortest_path_queries_ok": true,
        "baseline_train_only": true,
        "resumed": resumed,
        "checkpoint_path": checkpoint_path,
        "topology_identity": problem.topology_identity(),
        "peak_rss_kib": process_peak_rss_kib().unwrap_or(0),
        "test_read": false,
    }))?;

    Ok(TemporalTrainingOutcome {
        checkpoint_path,
        log_path: logger.path,
        completed_updates: restored.completed_updates,
        train_objective: last_train_objective,
        validation_objective: last_validation_objective,
        validation_metrics,
    })
}

fn partition_workloads(
    problem: &GraphProblem,
    mapped: Vec<MappedPath>,
    times: &[crate::data::TripTime],
    bucket_spec: &TimeBucketSpec,
) -> Result<Vec<BucketWorkload>, String> {
    if mapped.len() != times.len() {
        return Err("mapped paths and timestamps are not aligned".to_string());
    }
    let mut paths_by_bucket = (0..bucket_spec.buckets.len())
        .map(|_| Vec::new())
        .collect::<Vec<_>>();
    for (path, time) in mapped.into_iter().zip(times) {
        paths_by_bucket[bucket_spec.bucket_index(time.start_time)].push(path);
    }
    paths_by_bucket
        .into_iter()
        .map(|paths| {
            Ok(BucketWorkload {
                observed: problem.observed_counts(&paths)?,
                groups: GraphProblem::group_paths(&paths)?,
                paths,
            })
        })
        .collect()
}

fn evaluate_batch(
    problem: &GraphProblem,
    parameters: &TemporalParameters,
    initial_weights_by_bucket: &[Vec<f64>],
    initial_quantized: &[Vec<u32>],
    workloads: &[BucketWorkload],
    threads: usize,
) -> Result<BatchSnapshot, String> {
    let mut snapshot = BatchSnapshot {
        predicted_by_bucket: Vec::with_capacity(workloads.len()),
        observed_cost_sum: 0.0,
        predicted_cost_sum: 0.0,
        sample_count: 0,
        queries: 0,
        oracle_ms: 0.0,
        count_difference_l1: 0,
        quantization: Vec::with_capacity(workloads.len()),
    };
    for (bucket, workload) in workloads.iter().enumerate() {
        let weights = parameters.effective_weights(bucket, &initial_weights_by_bucket[bucket])?;
        let metric = problem.customize_external(&weights)?;
        let oracle = metric.batch_stats(&workload.groups, threads)?;
        snapshot.observed_cost_sum += observed_cost(&weights, &workload.observed)?;
        snapshot.predicted_cost_sum += oracle.weighted_direct_path_cost_sum;
        snapshot.sample_count = snapshot
            .sample_count
            .checked_add(oracle.sample_count)
            .ok_or_else(|| "temporal sample count overflow".to_string())?;
        snapshot.queries = snapshot
            .queries
            .checked_add(oracle.num_queries)
            .ok_or_else(|| "temporal query count overflow".to_string())?;
        snapshot.oracle_ms += oracle.oracle_duration.as_secs_f64() * 1_000.0;
        snapshot.count_difference_l1 = snapshot
            .count_difference_l1
            .checked_add(count_difference_l1(
                &oracle.predicted_counts,
                &workload.observed,
            )?)
            .ok_or_else(|| "temporal count-difference diagnostic overflow".to_string())?;
        snapshot.quantization.push(quantization_summary(
            &weights,
            metric.quantized_weights(),
            &initial_quantized[bucket],
        ));
        snapshot.predicted_by_bucket.push(oracle.predicted_counts);
    }
    if snapshot.sample_count == 0
        || !snapshot.observed_cost_sum.is_finite()
        || !snapshot.predicted_cost_sum.is_finite()
    {
        return Err("invalid aggregate temporal batch costs".to_string());
    }
    Ok(snapshot)
}

fn evaluate_decoded_paths(
    problem: &GraphProblem,
    parameters: &TemporalParameters,
    initial_weights_by_bucket: &[Vec<f64>],
    workloads: &[BucketWorkload],
    threads: usize,
) -> Result<(PathMetrics, Vec<PathMetrics>), String> {
    let mut parts = Vec::with_capacity(workloads.len());
    for (bucket, workload) in workloads.iter().enumerate() {
        let weights = parameters.effective_weights(bucket, &initial_weights_by_bucket[bucket])?;
        let metric = problem.customize_external(&weights)?;
        parts.push(evaluate_paths(&metric, &workload.paths, threads)?);
    }
    Ok((combine_path_metrics(&parts), parts))
}

fn make_checkpoint(
    config: &TemporalTrainingConfig,
    runtime_identity: &Value,
    problem: &GraphProblem,
    completed_updates: u64,
    parameters: &TemporalParameters,
    bucket_spec: &TimeBucketSpec,
    baseline_model: &BaselineModel,
) -> TemporalCheckpoint {
    TemporalCheckpoint {
        graph_representation: problem.representation().as_str().to_string(),
        completed_updates,
        parameters: parameters.clone(),
        bucket_edge_baselines: baseline_model.edge_weights_by_bucket.clone(),
        bucket_specification: bucket_spec.as_json().clone(),
        baseline_diagnostics: baseline_model.diagnostics.clone(),
        configuration: config.as_json().clone(),
        runtime_identity: runtime_identity.clone(),
        topology_identity: problem.topology_identity().to_string(),
    }
}

fn validate_resume_checkpoint(
    checkpoint: &TemporalCheckpoint,
    config: &TemporalTrainingConfig,
    runtime_identity: &Value,
    problem: &GraphProblem,
    bucket_spec: &TimeBucketSpec,
    baseline_model: &BaselineModel,
) -> Result<(), String> {
    if checkpoint.graph_representation != problem.representation().as_str()
        || checkpoint.configuration != *config.as_json()
        || checkpoint.runtime_identity != *runtime_identity
        || checkpoint.topology_identity != problem.topology_identity()
        || checkpoint.bucket_specification != *bucket_spec.as_json()
        || checkpoint.bucket_edge_baselines != baseline_model.edge_weights_by_bucket
        || checkpoint.baseline_diagnostics != baseline_model.diagnostics
    {
        return Err("temporal checkpoint identity does not match this run".to_string());
    }
    checkpoint.parameters.validate_for_config(
        config,
        problem.coordinate_count(),
        bucket_spec.buckets.len(),
    )
}

#[allow(clippy::too_many_arguments)]
fn runtime_identity(
    config: &TemporalTrainingConfig,
    graph: &GraphData,
    train: &PathValidationReport,
    validation: &PathValidationReport,
    problem: &GraphProblem,
    bucket_spec: &TimeBucketSpec,
    baseline_model: &BaselineModel,
    train_workloads: &[BucketWorkload],
    validation_workloads: &[BucketWorkload],
) -> Value {
    json!({
        "graph": {
            "city": config.city,
            "nodes": graph.x.len(),
            "edges": graph.tail.len(),
            "map_fingerprint": graph_fingerprint(graph),
            "representation": problem.representation().as_str(),
            "coordinates": problem.coordinate_count(),
            "routing_nodes": problem.routing_node_count(),
            "routing_arcs": problem.routing_arc_count(),
            "topology_identity": problem.topology_identity(),
        },
        "time_bucket_specification": bucket_spec.as_json(),
        "baseline": {
            "kind": baseline_model.kind.as_str(),
            "fingerprint": baseline_fingerprint(&baseline_model.edge_weights_by_bucket),
            "diagnostics": baseline_model.diagnostics,
            "estimated_from_split": "train",
        },
        "train": {
            "variant": config.train_variant,
            "declared": config.as_json().pointer("/data/train_identity"),
            "available": train.available_samples,
            "accepted": train.accepted_samples,
            "too_short": train.too_short,
            "bucket_samples": train_workloads.iter().map(|workload| workload.paths.len()).collect::<Vec<_>>(),
        },
        "validation": {
            "variant": config.validation_variant,
            "declared": config.as_json().pointer("/data/validation_identity"),
            "available": validation.available_samples,
            "accepted": validation.accepted_samples,
            "too_short": validation.too_short,
            "bucket_samples": validation_workloads.iter().map(|workload| workload.paths.len()).collect::<Vec<_>>(),
        },
    })
}

fn verify_declared_data(
    config: &TemporalTrainingConfig,
    split: &str,
    variant: &str,
) -> Result<(), String> {
    if split != "train" && split != "validation" {
        return Err("temporal training may only verify train or validation".to_string());
    }
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
    let actual_bytes = std::fs::metadata(&expected_path)
        .map_err(|error| format!("failed to inspect {expected_path}: {error}"))?
        .len();
    let declared_bytes = identity
        .pointer("/bytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("configuration is missing {pointer}/bytes"))?;
    if actual_bytes != declared_bytes {
        return Err(format!(
            "{expected_path} has {actual_bytes} bytes, expected {declared_bytes}"
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
    config: &TemporalTrainingConfig,
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

fn validate_retained_times(split: &str, times: &[crate::data::TripTime]) -> Result<(), String> {
    if times.iter().all(|time| time.duration_seconds().is_some()) {
        Ok(())
    } else {
        Err(format!(
            "{split} contains a nonpositive retained trip interval"
        ))
    }
}

fn log_data_report(
    logger: &mut JsonlLogger,
    split: &str,
    variant: &str,
    report: &PathValidationReport,
    timestamps: &TimestampEvidence,
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
        "too_short": report.too_short,
        "out_of_bounds": report.out_of_bounds,
        "discontinuous": report.discontinuous,
        "timestamp_evidence": {
            "samples": timestamps.timestamp_samples,
            "invalid_intervals": timestamps.invalid_intervals,
            "minimum_start_time": timestamps.minimum_start_time,
            "maximum_end_time": timestamps.maximum_end_time,
            "mmdd_keys": timestamps.mmdd_keys,
            "mmdd_matches_utc": timestamps.mmdd_matches_utc,
            "mmdd_matches_utc_plus_8": timestamps.mmdd_matches_utc_plus_8,
        },
        "policy": "complete_paths_min_2_edges_drop_cycles_retain_trip_timestamps",
    }))
}

fn workload_summary(spec: &TimeBucketSpec, workloads: &[BucketWorkload]) -> Vec<Value> {
    spec.buckets
        .iter()
        .zip(workloads)
        .map(|(bucket, workload)| {
            json!({
                "id": bucket.id,
                "start_hour": bucket.start_hour,
                "end_hour": bucket.end_hour,
                "samples": workload.paths.len(),
                "unique_od": workload.groups.len(),
            })
        })
        .collect()
}

fn parameter_summary(parameters: &TemporalParameters, config: &TemporalTrainingConfig) -> Value {
    let global_min = parameters
        .global_relative
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let global_max = parameters
        .global_relative
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let max_abs_residual = parameters
        .bucket_residuals
        .iter()
        .flatten()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max);
    let effective_min = parameters
        .bucket_residuals
        .iter()
        .flat_map(|residuals| parameters.global_relative.iter().zip(residuals))
        .map(|(&global, &residual)| global + residual)
        .fold(f64::INFINITY, f64::min);
    let effective_max = parameters
        .bucket_residuals
        .iter()
        .flat_map(|residuals| parameters.global_relative.iter().zip(residuals))
        .map(|(&global, &residual)| global + residual)
        .fold(f64::NEG_INFINITY, f64::max);
    json!({
        "global_min": global_min,
        "global_max": global_max,
        "changed_global_coordinates": parameters.global_relative.iter().filter(|&&value| value.to_bits() != 1.0f64.to_bits()).count(),
        "global_at_lower": parameters.global_relative.iter().filter(|&&value| value <= config.global_lower_factor).count(),
        "global_at_upper": parameters.global_relative.iter().filter(|&&value| value >= config.global_upper_factor).count(),
        "changed_residual_coordinates": parameters.bucket_residuals.iter().flatten().filter(|&&value| value.to_bits() != 0.0f64.to_bits()).count(),
        "max_abs_residual": max_abs_residual,
        "minimum_effective_multiplier": effective_min,
        "maximum_effective_multiplier": effective_max,
    })
}

fn quantization_summary(weights: &[f64], quantized: &[u32], initial: &[u32]) -> Value {
    json!({
        "max_abs_error": weights.iter().zip(quantized).map(|(&weight, &integer)| (weight - integer as f64).abs()).fold(0.0, f64::max),
        "maximum_relative_error": weights.iter().zip(quantized).map(|(&weight, &integer)| (weight - integer as f64).abs() / weight).fold(0.0, f64::max),
        "changed_from_initial": quantized.iter().zip(initial).filter(|(left, right)| left != right).count(),
        "zero_quantized_weights": quantized.iter().filter(|&&weight| weight == 0).count(),
    })
}

fn graph_fingerprint(graph: &GraphData) -> String {
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

fn baseline_fingerprint(buckets: &[Vec<f64>]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for value in buckets.iter().flatten() {
        for byte in value.to_bits().to_le_bytes() {
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
            .map_err(|error| format!("failed to encode temporal log event: {error}"))?;
        println!("{line}");
        writeln!(self.file, "{line}")
            .map_err(|error| format!("failed to write {}: {error}", self.path.display()))?;
        self.file
            .flush()
            .map_err(|error| format!("failed to flush {}: {error}", self.path.display()))?;
        stdout()
            .flush()
            .map_err(|error| format!("failed to flush stdout: {error}"))
    }
}
