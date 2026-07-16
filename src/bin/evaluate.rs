use edge_weight_recovery::checkpoint::TrainingCheckpoint;
use edge_weight_recovery::config::{TrainingConfig, atomic_write};
use edge_weight_recovery::data::{load_graph, load_trips};
use edge_weight_recovery::evaluation::{PathMetrics, combine_path_metrics, evaluate_paths};
use edge_weight_recovery::graph_problem::{GraphProblem, GraphRepresentation};
use edge_weight_recovery::time_buckets::{TimeBucketSpec, retain_departure_bucket};
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
    let checkpoint = TrainingCheckpoint::load(&arguments.checkpoint)?;
    let config = TrainingConfig::from_value(checkpoint.configuration.clone())?;
    let variant = match (&arguments.variant, arguments.split.as_str()) {
        (Some(variant), _) => variant.clone(),
        (None, "validation") => config.validation_variant.clone(),
        (None, "test") => {
            return Err("--variant is required when explicitly evaluating test".to_string());
        }
        _ => return Err("--split must be validation or test".to_string()),
    };
    validate_component(&variant, "variant")?;

    let graph = load_graph(&config.city)?;
    let representation = GraphRepresentation::parse(&checkpoint.graph_representation)?;
    if representation.as_str() != config.graph_representation {
        return Err("checkpoint graph representation differs from its configuration".to_string());
    }
    let problem = GraphProblem::build(
        &graph,
        representation,
        config.weight_lower_factor,
        config.weight_upper_factor,
    )?;
    if problem.topology_identity() != checkpoint.topology_identity {
        return Err("checkpoint topology identity does not match the loaded graph".to_string());
    }
    let mut trips = load_trips(&config.city, &arguments.split, &variant, &graph, None)?;
    let applied_filter = if let Some(filter) = &config.departure_time_filter {
        if arguments.time_buckets.is_some() {
            return Err(
                "--time-buckets cannot repartition a checkpoint already trained on one bucket"
                    .to_string(),
            );
        }
        let spec = filter.load_spec()?;
        let selection = retain_departure_bucket(&mut trips, &spec, &filter.bucket_id)?;
        if arguments.split == "validation"
            && selection.selected != filter.expected_validation_samples
        {
            return Err(format!(
                "validation bucket {:?} selected {}, expected {}",
                filter.bucket_id, selection.selected, filter.expected_validation_samples
            ));
        }
        Some((spec, selection))
    } else {
        None
    };
    if trips.paths.is_empty() {
        return Err("no evaluation paths remain after structural and time filtering".to_string());
    }
    let mapped = problem.map_paths(&trips.paths)?;
    let metric = problem.customize(&checkpoint.weights)?;
    let initial_metric = problem.customize(problem.initial_weights())?;
    let threads = rayon::current_num_threads().max(1);
    let (metrics, time_bucket_output) = if let Some((spec, selection)) = &applied_filter {
        let metrics = evaluate_paths(&metric, &mapped, threads)?;
        let (_, bucket) = spec.bucket(&selection.bucket_id)?;
        let row = json!({
            "id": bucket.id,
            "start_hour": bucket.start_hour,
            "end_hour": bucket.end_hour,
            "metrics": metrics_json(&metrics),
            "metric_totals": metric_totals_json(&metrics),
        });
        (
            metrics,
            json!({
                "specification": spec.as_json(),
                "selection_timestamp": "departure_time",
                "filter_mode": "single_registered_bucket",
                "buckets": [row],
            }),
        )
    } else if let Some(path) = &arguments.time_buckets {
        let spec = TimeBucketSpec::load(path)?;
        if mapped.len() != trips.times.len() {
            return Err("evaluation paths and timestamps are not aligned".to_string());
        }
        let mut bucket_paths = vec![Vec::new(); spec.buckets.len()];
        for (path, time) in mapped.into_iter().zip(&trips.times) {
            bucket_paths[spec.bucket_index(time.start_time)].push(path);
        }
        let mut bucket_metrics = Vec::with_capacity(spec.buckets.len());
        let mut rows = Vec::with_capacity(spec.buckets.len());
        for (bucket, paths) in spec.buckets.iter().zip(&bucket_paths) {
            let part = evaluate_paths(&metric, paths, threads)?;
            rows.push(json!({
                "id": bucket.id,
                "start_hour": bucket.start_hour,
                "end_hour": bucket.end_hour,
                "metrics": metrics_json(&part),
                "metric_totals": metric_totals_json(&part),
            }));
            bucket_metrics.push(part);
        }
        (
            combine_path_metrics(&bucket_metrics),
            json!({
                "spec_path": path,
                "specification": spec.as_json(),
                "selection_timestamp": "departure_time",
                "buckets": rows,
            }),
        )
    } else {
        (evaluate_paths(&metric, &mapped, threads)?, Value::Null)
    };

    let test_read = arguments.split == "test";
    let output = json!({
        "schema_version": 2,
        "checkpoint": arguments.checkpoint,
        "checkpoint_completed_updates": checkpoint.completed_updates,
        "graph_representation": representation.as_str(),
        "topology_identity": problem.topology_identity(),
        "split": arguments.split,
        "variant": variant,
        "path_report": {
            "available": trips.report.available_samples,
            "accepted": trips.report.accepted_samples,
            "selected": trips.paths.len(),
            "dropped": trips.report.dropped_samples(),
            "too_short": trips.report.too_short,
        },
        "metrics": metrics_json(&metrics),
        "metric_totals": metric_totals_json(&metrics),
        "time_bucket_evaluation": time_bucket_output,
        "departure_time_filter": applied_filter.as_ref().map(|(spec, selection)| json!({
            "selection_timestamp": "start_time",
            "bucket_id": selection.bucket_id,
            "source_accepted": selection.source_accepted,
            "selected": selection.selected,
            "specification": spec.as_json(),
        })),
        "quantization": {
            "max_abs_error": checkpoint.weights.iter().zip(metric.quantized_weights()).map(|(&weight, &integer)| (weight - integer as f64).abs()).fold(0.0, f64::max),
            "maximum_relative_error": checkpoint.weights.iter().zip(metric.quantized_weights()).map(|(&weight, &integer)| (weight - integer as f64).abs() / weight).fold(0.0, f64::max),
            "changed_from_baseline": metric.quantized_weights().iter().zip(initial_metric.quantized_weights()).filter(|(left, right)| left != right).count(),
            "zero_quantized_weights": metric.quantized_weights().iter().filter(|&&weight| weight == 0).count(),
        },
        "test_read": test_read,
    });
    let encoded = serde_json::to_vec_pretty(&output)
        .map_err(|error| format!("failed to encode evaluation output: {error}"))?;
    if let Some(path) = arguments.output {
        atomic_write(&path, &encoded)?;
    } else {
        println!(
            "{}",
            serde_json::to_string_pretty(&output)
                .map_err(|error| format!("failed to print evaluation output: {error}"))?
        );
    }
    Ok(())
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

fn metric_totals_json(metrics: &PathMetrics) -> Value {
    json!({
        "regret_sum": metrics.regret_sum,
        "observed_cost_sum": metrics.observed_cost_sum,
    })
}

struct Arguments {
    checkpoint: PathBuf,
    split: String,
    variant: Option<String>,
    output: Option<PathBuf>,
    time_buckets: Option<PathBuf>,
}

impl Arguments {
    fn from_args() -> Result<Option<Self>, String> {
        let arguments = std::env::args().skip(1).collect::<Vec<_>>();
        if arguments
            .iter()
            .any(|argument| argument == "--help" || argument == "-h")
        {
            println!(
                "Usage: evaluate --checkpoint PATH [--split validation|test] [--variant NAME] [--time-buckets PATH] [--output PATH]\n\n\
                 Validation defaults to the checkpoint configuration's fixed validation variant.\n\
                 Test requires an explicit --split test and --variant NAME. Time buckets select by retained departure time."
            );
            return Ok(None);
        }
        let mut checkpoint = None;
        let mut split = "validation".to_string();
        let mut variant = None;
        let mut output = None;
        let mut time_buckets = None;
        let mut index = 0;
        while index < arguments.len() {
            let flag = &arguments[index];
            let value = arguments
                .get(index + 1)
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--checkpoint" => checkpoint = Some(PathBuf::from(value)),
                "--split" => split = value.clone(),
                "--variant" => variant = Some(value.clone()),
                "--output" => output = Some(PathBuf::from(value)),
                "--time-buckets" => time_buckets = Some(PathBuf::from(value)),
                _ => return Err(format!("unknown argument {flag:?}")),
            }
            index += 2;
        }
        if split != "validation" && split != "test" {
            return Err("--split must be validation or test".to_string());
        }
        Ok(Some(Self {
            checkpoint: checkpoint.ok_or_else(|| "missing --checkpoint PATH".to_string())?,
            split,
            variant,
            output,
            time_buckets,
        }))
    }
}

fn validate_component(value: &str, label: &str) -> Result<(), String> {
    if value.is_empty() || value.contains('/') || value.contains("..") {
        return Err(format!("{label} contains an unsafe path component"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_component;

    #[test]
    fn explicit_variants_must_be_safe_path_components() {
        assert!(validate_component("scale_fixed_seed20260715", "variant").is_ok());
        assert!(validate_component("", "variant").is_err());
        assert!(validate_component("../test", "variant").is_err());
        assert!(validate_component("nested/test", "variant").is_err());
    }
}
