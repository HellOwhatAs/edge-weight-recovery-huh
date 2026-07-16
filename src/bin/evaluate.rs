use edge_weight_recovery::checkpoint::TrainingCheckpoint;
use edge_weight_recovery::config::{TrainingConfig, atomic_write};
use edge_weight_recovery::data::{load_graph, load_trips};
use edge_weight_recovery::evaluation::{PathMetrics, evaluate_paths};
use edge_weight_recovery::graph_problem::{GraphProblem, GraphRepresentation};
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
    let trips = load_trips(&config.city, &arguments.split, &variant, &graph, None)?;
    if trips.paths.is_empty() {
        return Err("no valid evaluation paths remain after validation".to_string());
    }
    let mapped = problem.map_paths(&trips.paths)?;
    let metric = problem.customize(&checkpoint.weights)?;
    let metrics = evaluate_paths(&metric, &mapped, rayon::current_num_threads().max(1))?;

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
            "dropped": trips.report.dropped_samples(),
            "too_short": trips.report.too_short,
        },
        "metrics": metrics_json(&metrics),
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

struct Arguments {
    checkpoint: PathBuf,
    split: String,
    variant: Option<String>,
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
                "Usage: evaluate --checkpoint PATH [--split validation|test] [--variant NAME] [--output PATH]\n\n\
                 Validation defaults to the checkpoint configuration's fixed validation variant.\n\
                 Test requires an explicit --split test and --variant NAME."
            );
            return Ok(None);
        }
        let mut checkpoint = None;
        let mut split = "validation".to_string();
        let mut variant = None;
        let mut output = None;
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
