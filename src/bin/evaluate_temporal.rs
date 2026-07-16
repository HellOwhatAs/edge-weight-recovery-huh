use edge_weight_recovery::config::atomic_write;
use edge_weight_recovery::data::load_graph;
use edge_weight_recovery::evaluation::{PathMetrics, combine_path_metrics, evaluate_paths};
use edge_weight_recovery::graph_problem::{GraphProblem, GraphRepresentation};
use edge_weight_recovery::temporal::{
    TemporalCheckpoint, TemporalTrainingConfig, TimeBucketSpec, sha256_file,
};
use serde_json::{Value, json};
use std::path::PathBuf;

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
    let checkpoint = TemporalCheckpoint::load(&arguments.checkpoint)?;
    let config = TemporalTrainingConfig::from_value(checkpoint.configuration.clone())?;
    let configured_spec = config.load_bucket_spec()?;
    let checkpoint_spec = TimeBucketSpec::from_value(checkpoint.bucket_specification.clone())?;
    if configured_spec != checkpoint_spec {
        return Err("checkpoint time buckets differ from the verified configuration".to_string());
    }

    // This evaluator intentionally exposes no split or variant argument. It
    // can read only the fixed validation file declared by the checkpoint.
    let validation_path = PathBuf::from(format!(
        "data/{}_data/preprocessed_validation_trips_{}.pkl",
        config.city, config.validation_variant
    ));
    let validation_identity = config
        .as_json()
        .pointer("/data/validation_identity")
        .ok_or_else(|| "configuration lacks validation identity".to_string())?;
    let actual_bytes = std::fs::metadata(&validation_path)
        .map_err(|error| format!("failed to inspect {}: {error}", validation_path.display()))?
        .len();
    let actual_hash = sha256_file(&validation_path)?;
    if validation_identity.pointer("/path").and_then(Value::as_str) != validation_path.to_str()
        || validation_identity
            .pointer("/bytes")
            .and_then(Value::as_u64)
            != Some(actual_bytes)
        || validation_identity
            .pointer("/sha256")
            .and_then(Value::as_str)
            != Some(actual_hash.as_str())
    {
        return Err("fixed validation identity does not match the checkpoint".to_string());
    }
    let graph = load_graph(&config.city)?;
    let trips = edge_weight_recovery::data::load_trips(
        &config.city,
        "validation",
        &config.validation_variant,
        &graph,
        None,
    )?;
    if trips.paths.is_empty() || trips.paths.len() != trips.times.len() {
        return Err("validation paths and timestamps must be nonempty and aligned".to_string());
    }
    if let Some(declared) = config
        .as_json()
        .pointer("/data/validation_identity/sample_count")
        .and_then(Value::as_u64)
        && declared != trips.report.available_samples as u64
    {
        return Err(format!(
            "validation sample count mismatch: declared {declared}, loaded {}",
            trips.report.available_samples
        ));
    }
    let problem = GraphProblem::build(
        &graph,
        GraphRepresentation::EdgeTransitionArcs,
        config.weight_lower_factor,
        config.weight_upper_factor,
    )?;
    if problem.topology_identity() != checkpoint.topology_identity {
        return Err("checkpoint topology identity differs from the loaded graph".to_string());
    }
    checkpoint.parameters.validate_for_config(
        &config,
        problem.coordinate_count(),
        checkpoint_spec.buckets.len(),
    )?;
    if checkpoint.bucket_edge_baselines.len() != checkpoint_spec.buckets.len() {
        return Err("checkpoint baseline bucket count is inconsistent".to_string());
    }
    let coordinate_initials = checkpoint
        .bucket_edge_baselines
        .iter()
        .map(|weights| problem.coordinate_weights_from_original(weights))
        .collect::<Result<Vec<_>, _>>()?;
    let mapped = problem.map_paths(&trips.paths)?;
    let mut paths_by_bucket = (0..checkpoint_spec.buckets.len())
        .map(|_| Vec::new())
        .collect::<Vec<_>>();
    for (path, time) in mapped.into_iter().zip(&trips.times) {
        paths_by_bucket[checkpoint_spec.bucket_index(time.start_time)].push(path);
    }

    let threads = rayon::current_num_threads().max(1);
    let mut per_bucket = Vec::with_capacity(checkpoint_spec.buckets.len());
    let mut rows = Vec::with_capacity(checkpoint_spec.buckets.len());
    let mut quantization = Vec::with_capacity(checkpoint_spec.buckets.len());
    for (bucket_index, (bucket, paths)) in checkpoint_spec
        .buckets
        .iter()
        .zip(&paths_by_bucket)
        .enumerate()
    {
        let weights = checkpoint
            .parameters
            .effective_weights(bucket_index, &coordinate_initials[bucket_index])?;
        let initial_metric = problem.customize_external(&coordinate_initials[bucket_index])?;
        let metric = problem.customize_external(&weights)?;
        let metrics = evaluate_paths(&metric, paths, threads)?;
        rows.push(json!({
            "id": bucket.id,
            "start_hour": bucket.start_hour,
            "end_hour": bucket.end_hour,
            "metrics": metrics_json(&metrics),
        }));
        quantization.push(json!({
            "id": bucket.id,
            "max_abs_error": weights.iter().zip(metric.quantized_weights()).map(|(&weight, &integer)| (weight - integer as f64).abs()).fold(0.0, f64::max),
            "maximum_relative_error": weights.iter().zip(metric.quantized_weights()).map(|(&weight, &integer)| (weight - integer as f64).abs() / weight).fold(0.0, f64::max),
            "zero_quantized_weights": metric.quantized_weights().iter().filter(|&&weight| weight == 0).count(),
            "changed_from_baseline": metric.quantized_weights().iter().zip(initial_metric.quantized_weights()).filter(|(left, right)| left != right).count(),
        }));
        per_bucket.push(metrics);
    }
    let metrics = combine_path_metrics(&per_bucket);
    let output = json!({
        "schema_version": 1,
        "checkpoint": arguments.checkpoint,
        "checkpoint_completed_updates": checkpoint.completed_updates,
        "model_kind": "global_plus_bucket_residual",
        "graph_representation": checkpoint.graph_representation,
        "topology_identity": problem.topology_identity(),
        "split": "validation",
        "variant": config.validation_variant,
        "path_report": {
            "available": trips.report.available_samples,
            "accepted": trips.report.accepted_samples,
            "dropped": trips.report.dropped_samples(),
            "too_short": trips.report.too_short,
            "cyclic": trips.report.cyclic,
        },
        "timestamp_evidence": {
            "samples": trips.timestamp_evidence.timestamp_samples,
            "invalid_intervals": trips.timestamp_evidence.invalid_intervals,
            "minimum_start_time": trips.timestamp_evidence.minimum_start_time,
            "maximum_end_time": trips.timestamp_evidence.maximum_end_time,
        },
        "metrics": metrics_json(&metrics),
        "time_bucket_evaluation": {
            "specification": checkpoint_spec.as_json(),
            "selection_timestamp": "departure_time",
            "buckets": rows,
        },
        "baseline": {
            "kind": config.baseline_kind.as_str(),
            "diagnostics": checkpoint.baseline_diagnostics,
            "estimated_from_split": "train",
            "validation_used": false,
        },
        "quantization": quantization,
        "test_read": false,
    });
    let bytes = serde_json::to_vec_pretty(&output)
        .map_err(|error| format!("failed to encode temporal evaluation: {error}"))?;
    if let Some(path) = arguments.output {
        atomic_write(&path, &bytes)
    } else {
        println!(
            "{}",
            serde_json::to_string_pretty(&output)
                .map_err(|error| format!("failed to print temporal evaluation: {error}"))?
        );
        Ok(())
    }
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

struct Arguments {
    checkpoint: PathBuf,
    output: Option<PathBuf>,
}

impl Arguments {
    fn from_args() -> Result<Option<Self>, String> {
        let arguments = std::env::args().skip(1).collect::<Vec<_>>();
        if arguments
            .iter()
            .any(|argument| argument == "--help" || argument == "-h")
        {
            println!(
                "Usage: evaluate_temporal --checkpoint PATH [--output PATH]\n\n\
                 Evaluates only the fixed validation split recorded in the checkpoint."
            );
            return Ok(None);
        }
        let mut checkpoint = None;
        let mut output = None;
        let mut index = 0;
        while index < arguments.len() {
            let flag = &arguments[index];
            let value = arguments
                .get(index + 1)
                .ok_or_else(|| format!("missing value for {flag}"))?;
            let slot = match flag.as_str() {
                "--checkpoint" => &mut checkpoint,
                "--output" => &mut output,
                _ => return Err(format!("unknown argument {flag:?}")),
            };
            if slot.replace(PathBuf::from(value)).is_some() {
                return Err(format!("{flag} was provided more than once"));
            }
            index += 2;
        }
        Ok(Some(Self {
            checkpoint: checkpoint.ok_or_else(|| "missing --checkpoint PATH".to_string())?,
            output,
        }))
    }
}
