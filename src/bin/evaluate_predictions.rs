use edge_weight_recovery::config::atomic_write;
use edge_weight_recovery::evaluation::{RouteMetrics, evaluate_raw_paths};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
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
    let file = File::open(&arguments.predictions).map_err(|error| {
        format!(
            "failed to open predictions {}: {error}",
            arguments.predictions.display()
        )
    })?;
    let mut ids = HashSet::new();
    let mut observed = Vec::new();
    let mut predicted = Vec::new();
    let mut endpoint_mismatches = 0usize;
    let mut manifest_hash = Sha256::new();
    let mut method = None::<String>;
    let mut query_protocol = None::<String>;
    for (line_number, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|error| format!("failed to read predictions: {error}"))?;
        let row: Value = serde_json::from_str(&line)
            .map_err(|error| format!("prediction line {}: {error}", line_number + 1))?;
        let id = row
            .pointer("/manifest_id")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("prediction line {} lacks manifest_id", line_number + 1))?;
        if !ids.insert(id.to_string()) {
            return Err(format!("duplicate prediction manifest_id {id:?}"));
        }
        manifest_hash.update((id.len() as u64).to_le_bytes());
        manifest_hash.update(id.as_bytes());
        let truth = parse_edges(&row, "/observed_edges", line_number + 1)?;
        let prediction = parse_edges(&row, "/predicted_edges", line_number + 1)?;
        endpoint_mismatches +=
            usize::from(truth.first() != prediction.first() || truth.last() != prediction.last());
        merge_stable_field(&mut method, &row, "/method")?;
        merge_stable_field(&mut query_protocol, &row, "/query_protocol")?;
        observed.push(truth);
        predicted.push(prediction);
    }
    let metrics = evaluate_raw_paths(&observed, &predicted)?;
    let output = json!({
        "schema_version": 1,
        "predictions": arguments.predictions,
        "method": method,
        "query_protocol": query_protocol,
        "manifest_id_order_sha256": format!("{:x}", manifest_hash.finalize()),
        "endpoint_mismatches": endpoint_mismatches,
        "metrics": metrics_json(&metrics),
        "test_read": ids.iter().any(|id| id.starts_with("test:")),
    });
    let encoded = serde_json::to_vec_pretty(&output)
        .map_err(|error| format!("failed to encode common evaluation: {error}"))?;
    atomic_write(&arguments.output, &encoded)?;
    println!(
        "{} paths: F1 {:.6}, Exact {:.6}",
        metrics.sample_count, metrics.edge_f1, metrics.exact_match
    );
    Ok(())
}

fn parse_edges(row: &Value, pointer: &str, line_number: usize) -> Result<Vec<usize>, String> {
    let edges = row
        .pointer(pointer)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("prediction line {line_number} lacks {pointer}"))?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|edge| usize::try_from(edge).ok())
                .ok_or_else(|| format!("prediction line {line_number} has invalid {pointer}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if edges.is_empty() {
        return Err(format!("prediction line {line_number} has empty {pointer}"));
    }
    Ok(edges)
}

fn merge_stable_field(
    target: &mut Option<String>,
    row: &Value,
    pointer: &str,
) -> Result<(), String> {
    let Some(value) = row.pointer(pointer).and_then(Value::as_str) else {
        return Ok(());
    };
    if target.as_deref().is_some_and(|existing| existing != value) {
        return Err(format!("prediction rows disagree on {pointer}"));
    }
    target.get_or_insert_with(|| value.to_string());
    Ok(())
}

fn metrics_json(metrics: &RouteMetrics) -> Value {
    json!({
        "samples": metrics.sample_count,
        "edge_precision": metrics.edge_precision,
        "edge_recall": metrics.edge_recall,
        "edge_f1": metrics.edge_f1,
        "exact_match": metrics.exact_match,
        "edge_jaccard": metrics.edge_jaccard,
    })
}

struct Arguments {
    predictions: PathBuf,
    output: PathBuf,
}

impl Arguments {
    fn from_args() -> Result<Option<Self>, String> {
        let arguments = std::env::args().skip(1).collect::<Vec<_>>();
        if arguments
            .iter()
            .any(|argument| argument == "--help" || argument == "-h")
        {
            println!("Usage: evaluate_predictions --predictions PATH --output PATH");
            return Ok(None);
        }
        let mut predictions = None;
        let mut output = None;
        let mut index = 0;
        while index < arguments.len() {
            let value = arguments
                .get(index + 1)
                .ok_or_else(|| format!("missing value for {}", arguments[index]))?;
            let target = match arguments[index].as_str() {
                "--predictions" => &mut predictions,
                "--output" => &mut output,
                flag => return Err(format!("unknown argument {flag}")),
            };
            if target.replace(PathBuf::from(value)).is_some() {
                return Err(format!("{} was provided twice", arguments[index]));
            }
            index += 2;
        }
        Ok(Some(Self {
            predictions: predictions.ok_or_else(|| "missing --predictions".to_string())?,
            output: output.ok_or_else(|| "missing --output".to_string())?,
        }))
    }
}
