use edge_weight_recovery::config::{atomic_write, load_checkpoint};
use edge_weight_recovery::data::{load_graph, load_trips};
use edge_weight_recovery::evaluation::{evaluate_expanded_paths, evaluate_paths};
use edge_weight_recovery::expanded_training::restore_expanded_metric;
use edge_weight_recovery::oracle::{CchOracle, ExpandedCchOracle};
use edge_weight_recovery::turn_graph::ExpandedTurnGraph;
use serde_json::{Value, json};
use std::path::PathBuf;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let Some(arguments) = Arguments::parse()? else {
        return Ok(());
    };
    let checkpoint = load_checkpoint(&arguments.checkpoint)?;
    let city = required_str(&checkpoint, "/configuration/data/city")?;
    let variant = match arguments.variant {
        Some(variant) => variant,
        None if arguments.split == "validation" => {
            required_str(&checkpoint, "/configuration/data/validation_variant")?.to_string()
        }
        None => return Err("--variant is required when --split test is explicit".to_string()),
    };
    validate_component(&variant, "variant")?;

    let graph = load_graph(city)?;
    let trips = load_trips(city, &arguments.split, &variant, &graph, None)?;
    if trips.paths.is_empty() {
        return Err("no valid evaluation paths remain after validation".to_string());
    }
    let model = required_str(&checkpoint, "/model")?;
    let threads = rayon::current_num_threads().max(1);
    let metrics = match model {
        "edge_only" => {
            let weights = checkpoint_weights(&checkpoint, "/quantized_metric_weights")?;
            if weights.len() != graph.tail.len() {
                return Err(format!(
                    "checkpoint has {} edge weights but {city} graph has {} edges",
                    weights.len(),
                    graph.tail.len()
                ));
            }
            let oracle = CchOracle::build(&graph)?;
            let metric = oracle.customize(&weights)?;
            evaluate_paths(&metric, &trips.paths, threads)?
        }
        "expanded" => {
            let expanded = ExpandedTurnGraph::build(&graph)?;
            let oracle = ExpandedCchOracle::build(&graph, &expanded)?;
            let weights = restore_expanded_metric(
                &checkpoint,
                &graph,
                &expanded,
                oracle.topology_identity(),
            )?;
            let metric = oracle.customize(weights.edge_weights(), weights.transition_weights())?;
            evaluate_expanded_paths(&metric, &trips.paths, threads)?
        }
        _ => return Err(format!("unsupported checkpoint model {model:?}")),
    };
    let result = json!({
        "schema_version": 1,
        "checkpoint": arguments.checkpoint,
        "checkpoint_epoch": checkpoint.pointer("/epoch").and_then(Value::as_u64),
        "checkpoint_completed_updates": checkpoint
            .pointer("/completed_updates")
            .and_then(Value::as_u64),
        "model": model,
        "city": city,
        "split": arguments.split,
        "variant": variant,
        "path_policy": "complete_paths_drop_cycles",
        "data": {
            "available": trips.report.available_samples,
            "inspected": trips.report.inspected_samples,
            "accepted": trips.report.accepted_samples,
            "cyclic": trips.report.cyclic,
        },
        "metrics": {
            "samples": metrics.sample_count,
            "mean_regret": metrics.mean_regret,
            "relative_regret": metrics.relative_regret,
            "exact_match": metrics.exact_match,
            "edge_precision": metrics.edge_precision,
            "edge_recall": metrics.edge_recall,
            "edge_f1": metrics.edge_f1,
            "edge_jaccard": metrics.edge_jaccard,
        }
    });
    let bytes = serde_json::to_vec_pretty(&result)
        .map_err(|error| format!("failed to encode evaluation: {error}"))?;
    if let Some(path) = arguments.output {
        atomic_write(&path, &bytes)?;
    } else {
        println!("{}", String::from_utf8(bytes).expect("JSON is UTF-8"));
    }
    Ok(())
}

struct Arguments {
    checkpoint: PathBuf,
    split: String,
    variant: Option<String>,
    output: Option<PathBuf>,
}

impl Arguments {
    fn parse() -> Result<Option<Self>, String> {
        let arguments = std::env::args().skip(1).collect::<Vec<_>>();
        if arguments
            .iter()
            .any(|value| value == "--help" || value == "-h")
        {
            println!(
                "Usage: evaluate --checkpoint PATH [--split validation|test] [--variant NAME] [--output PATH]\n\n\
                 Defaults to the checkpoint's validation variant. Test is read only\n\
                 when --split test and --variant are both explicitly supplied."
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
                _ => return Err(format!("unknown argument {flag:?}; use --help")),
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

fn checkpoint_weights(checkpoint: &Value, pointer: &str) -> Result<Vec<u32>, String> {
    checkpoint
        .pointer(pointer)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("checkpoint is missing {pointer}"))?
        .iter()
        .enumerate()
        .map(|(edge, value)| {
            let weight = value
                .as_u64()
                .ok_or_else(|| format!("checkpoint weight {edge} is not an integer"))?;
            u32::try_from(weight).map_err(|_| format!("checkpoint weight {edge} does not fit u32"))
        })
        .collect()
}

fn required_str<'a>(value: &'a Value, pointer: &str) -> Result<&'a str, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("checkpoint is missing string {pointer}"))
}

fn validate_component(value: &str, label: &str) -> Result<(), String> {
    if value.is_empty() || value.contains('/') || value.contains("..") {
        return Err(format!("{label} contains an unsafe path component"));
    }
    Ok(())
}
