use edge_weight_recovery::checkpoint::TrainingCheckpoint;
use edge_weight_recovery::config::{TrainingConfig, atomic_write};
use edge_weight_recovery::data::load_graph;
use edge_weight_recovery::evaluation::{RouteMetrics, evaluate_raw_paths};
use edge_weight_recovery::graph_problem::{
    GraphProblem, GraphRepresentation, OracleKind, ShortestPath,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::hint::black_box;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
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
    if rayon::current_num_threads().max(1) != arguments.threads {
        return Err(format!(
            "--threads={} but RAYON_NUM_THREADS resolved to {}",
            arguments.threads,
            rayon::current_num_threads().max(1)
        ));
    }

    let total_started = Instant::now();
    let checkpoint_started = Instant::now();
    let checkpoint = TrainingCheckpoint::load(&arguments.checkpoint)?;
    let config = TrainingConfig::from_value(checkpoint.configuration.clone())?;
    let checkpoint_load = checkpoint_started.elapsed();

    let manifest_started = Instant::now();
    let records = load_manifest(&arguments.manifest, arguments.limit)?;
    let manifest_load = manifest_started.elapsed();

    let graph_load_started = Instant::now();
    let graph = load_graph(&config.city)?;
    let graph_load = graph_load_started.elapsed();
    let problem_build_started = Instant::now();
    let representation = GraphRepresentation::parse(&checkpoint.graph_representation)?;
    let problem = GraphProblem::build(
        &graph,
        representation,
        config.weight_lower_factor,
        config.weight_upper_factor,
    )?;
    let problem_build = problem_build_started.elapsed();
    if problem.topology_identity() != checkpoint.topology_identity {
        return Err("checkpoint topology identity differs from the loaded graph".to_string());
    }
    if arguments.query_protocol == QueryProtocol::EdgeToEdge
        && representation != GraphRepresentation::EdgeTransitionArcs
    {
        return Err("edge-to-edge benchmark requires edge_transition_arcs".to_string());
    }
    let mapped = records
        .iter()
        .map(|record| problem.map_path(&record.edges))
        .collect::<Result<Vec<_>, _>>()?;

    let customization_started = Instant::now();
    let metric = problem.customize_with_oracle(&checkpoint.weights, arguments.oracle)?;
    let customization = customization_started.elapsed();

    for _ in 0..arguments.warmup_repetitions {
        let paths = run_queries(&metric, &mapped, arguments.query_protocol)?;
        black_box(paths);
    }

    let mut measured_seconds = Vec::with_capacity(arguments.measured_repetitions);
    let mut first_predictions = None;
    let mut expected_checksum = None;
    for repetition in 0..arguments.measured_repetitions {
        let started = Instant::now();
        let predictions = run_queries(&metric, &mapped, arguments.query_protocol)?;
        let elapsed = started.elapsed();
        let checksum = route_checksum(&predictions);
        if let Some(expected) = &expected_checksum {
            if expected != &checksum {
                return Err(format!(
                    "query repetition {repetition} produced nondeterministic route checksum"
                ));
            }
        } else {
            expected_checksum = Some(checksum);
        }
        if first_predictions.is_none() {
            first_predictions = Some(predictions);
        }
        measured_seconds.push(elapsed.as_secs_f64());
    }
    let predictions = first_predictions.expect("positive measured repetitions validated");
    let observed = records
        .iter()
        .map(|record| record.edges.clone())
        .collect::<Vec<_>>();
    let predicted_edges = predictions
        .iter()
        .map(|prediction| prediction.original_edges.clone())
        .collect::<Vec<_>>();
    let metrics = evaluate_raw_paths(&observed, &predicted_edges)?;
    write_predictions(&arguments, &records, &predictions)?;

    let query_count = records.len();
    let mean_seconds = mean(&measured_seconds);
    let build_stats = problem.oracle_build_stats();
    let output = json!({
        "schema_version": 1,
        "checkpoint": arguments.checkpoint,
        "checkpoint_completed_updates": checkpoint.completed_updates,
        "manifest": arguments.manifest,
        "manifest_records": query_count,
        "predictions": arguments.predictions,
        "graph_representation": representation.as_str(),
        "query_protocol": arguments.query_protocol.as_str(),
        "oracle": arguments.oracle.as_str(),
        "threads": arguments.threads,
        "warmup_repetitions": arguments.warmup_repetitions,
        "measured_repetitions": arguments.measured_repetitions,
        "metrics": route_metrics_json(&metrics),
        "timing": {
            "checkpoint_load_seconds": checkpoint_load.as_secs_f64(),
            "manifest_load_seconds": manifest_load.as_secs_f64(),
            "graph_load_seconds": graph_load.as_secs_f64(),
            "graph_problem_build_seconds": problem_build.as_secs_f64(),
            "cch_topology_preprocessing_seconds": build_stats.cch_topology.as_secs_f64(),
            "dijkstra_adjacency_setup_seconds": build_stats.dijkstra_topology.as_secs_f64(),
            "customization_seconds": customization.as_secs_f64(),
            "query_repetition_seconds": measured_seconds,
            "mean_total_query_seconds": mean_seconds,
            "mean_query_latency_seconds": mean_seconds / query_count as f64,
            "mean_throughput_queries_per_second": query_count as f64 / mean_seconds,
            "total_process_seconds": total_started.elapsed().as_secs_f64(),
        },
        "quantized_weight_sha256": u32_checksum(metric.quantized_weights()),
        "route_checksum_sha256": expected_checksum,
        "peak_rss_kib": process_peak_rss_kib().unwrap_or(0),
        "test_read": records.iter().any(|record| record.manifest_id.starts_with("test:")),
    });
    let encoded = serde_json::to_vec_pretty(&output)
        .map_err(|error| format!("failed to encode benchmark summary: {error}"))?;
    atomic_write(&arguments.summary, &encoded)?;
    println!(
        "{} {} queries: F1 {:.6}, Exact {:.6}, {:.3} queries/s",
        arguments.oracle.as_str(),
        query_count,
        metrics.edge_f1,
        metrics.exact_match,
        query_count as f64 / mean_seconds
    );
    Ok(())
}

fn run_queries(
    metric: &edge_weight_recovery::graph_problem::GraphMetric<'_>,
    paths: &[edge_weight_recovery::graph_problem::MappedPath],
    protocol: QueryProtocol,
) -> Result<Vec<ShortestPath>, String> {
    let mut query = metric.new_query();
    paths
        .iter()
        .map(|path| match protocol {
            QueryProtocol::NodeToNode => query.shortest_path(path.source, path.target),
            QueryProtocol::EdgeToEdge => query.shortest_path_edges(
                *path
                    .original_edges
                    .first()
                    .expect("mapped paths are nonempty"),
                *path
                    .original_edges
                    .last()
                    .expect("mapped paths are nonempty"),
            ),
        })
        .collect()
}

#[derive(Clone, Debug)]
struct ManifestRecord {
    manifest_id: String,
    original_trip_id: String,
    source_index: u64,
    edges: Vec<usize>,
}

fn load_manifest(path: &PathBuf, limit: Option<usize>) -> Result<Vec<ManifestRecord>, String> {
    let file = File::open(path)
        .map_err(|error| format!("failed to open manifest {}: {error}", path.display()))?;
    let mut records = Vec::new();
    let mut ids = std::collections::HashSet::new();
    for (line_number, line) in BufReader::new(file).lines().enumerate() {
        if limit.is_some_and(|limit| records.len() >= limit) {
            break;
        }
        let line = line.map_err(|error| {
            format!(
                "failed to read {} line {}: {error}",
                path.display(),
                line_number + 1
            )
        })?;
        let value: Value = serde_json::from_str(&line).map_err(|error| {
            format!(
                "failed to decode {} line {}: {error}",
                path.display(),
                line_number + 1
            )
        })?;
        let manifest_id = value
            .pointer("/manifest_id")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("manifest line {} lacks manifest_id", line_number + 1))?
            .to_string();
        if !ids.insert(manifest_id.clone()) {
            return Err(format!("duplicate manifest_id {manifest_id:?}"));
        }
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
        if edges.is_empty() {
            return Err(format!(
                "manifest line {} has an empty path",
                line_number + 1
            ));
        }
        records.push(ManifestRecord {
            manifest_id,
            original_trip_id: value
                .pointer("/original_trip_id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            source_index: value
                .pointer("/source_index")
                .and_then(Value::as_u64)
                .ok_or_else(|| format!("manifest line {} lacks source_index", line_number + 1))?,
            edges,
        });
    }
    if records.is_empty() {
        return Err("manifest contains no selected records".to_string());
    }
    Ok(records)
}

fn write_predictions(
    arguments: &Arguments,
    records: &[ManifestRecord],
    predictions: &[ShortestPath],
) -> Result<(), String> {
    if let Some(parent) = arguments.predictions.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let temporary = arguments
        .predictions
        .with_extension(format!("jsonl.{}.tmp", std::process::id()));
    let mut writer = BufWriter::new(
        File::create(&temporary)
            .map_err(|error| format!("failed to create {}: {error}", temporary.display()))?,
    );
    for (record, prediction) in records.iter().zip(predictions) {
        serde_json::to_writer(
            &mut writer,
            &json!({
                "manifest_id": record.manifest_id,
                "source_index": record.source_index,
                "original_trip_id": record.original_trip_id,
                "observed_edges": record.edges,
                "predicted_edges": prediction.original_edges,
                "distance_u32": prediction.distance,
                "direct_cost": prediction.direct_cost,
                "method": format!("project_{}", arguments.query_protocol.as_str()),
                "query_protocol": arguments.query_protocol.as_str(),
                "oracle": arguments.oracle.as_str(),
            }),
        )
        .map_err(|error| format!("failed to encode prediction: {error}"))?;
        writer
            .write_all(b"\n")
            .map_err(|error| format!("failed to write {}: {error}", temporary.display()))?;
    }
    writer
        .flush()
        .map_err(|error| format!("failed to flush {}: {error}", temporary.display()))?;
    drop(writer);
    std::fs::rename(&temporary, &arguments.predictions).map_err(|error| {
        format!(
            "failed to move {} to {}: {error}",
            temporary.display(),
            arguments.predictions.display()
        )
    })
}

fn route_metrics_json(metrics: &RouteMetrics) -> Value {
    json!({
        "samples": metrics.sample_count,
        "edge_precision": metrics.edge_precision,
        "edge_recall": metrics.edge_recall,
        "edge_f1": metrics.edge_f1,
        "exact_match": metrics.exact_match,
        "edge_jaccard": metrics.edge_jaccard,
    })
}

fn route_checksum(paths: &[ShortestPath]) -> String {
    let mut hash = Sha256::new();
    for path in paths {
        hash.update(path.distance.to_le_bytes());
        hash.update((path.original_edges.len() as u64).to_le_bytes());
        for &edge in &path.original_edges {
            hash.update((edge as u64).to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}

fn u32_checksum(values: &[u32]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
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
enum QueryProtocol {
    NodeToNode,
    EdgeToEdge,
}

impl QueryProtocol {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "node_to_node" => Ok(Self::NodeToNode),
            "edge_to_edge" => Ok(Self::EdgeToEdge),
            _ => Err("--query-protocol must be node_to_node or edge_to_edge".to_string()),
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::NodeToNode => "node_to_node",
            Self::EdgeToEdge => "edge_to_edge",
        }
    }
}

struct Arguments {
    checkpoint: PathBuf,
    manifest: PathBuf,
    predictions: PathBuf,
    summary: PathBuf,
    oracle: OracleKind,
    query_protocol: QueryProtocol,
    threads: usize,
    warmup_repetitions: usize,
    measured_repetitions: usize,
    limit: Option<usize>,
}

impl Arguments {
    fn from_args() -> Result<Option<Self>, String> {
        let arguments = std::env::args().skip(1).collect::<Vec<_>>();
        if arguments
            .iter()
            .any(|argument| argument == "--help" || argument == "-h")
        {
            println!(
                "Usage: benchmark_routes --checkpoint PATH --manifest PATH --predictions PATH \\\n                 --summary PATH --oracle cch|dijkstra --query-protocol node_to_node|edge_to_edge \\\n                 --threads N --warmup-repetitions N --measured-repetitions N [--limit N]"
            );
            return Ok(None);
        }
        let mut values = std::collections::BTreeMap::new();
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
        let take = |flag: &str| {
            values
                .get(flag)
                .cloned()
                .ok_or_else(|| format!("missing {flag}"))
        };
        let known = [
            "--checkpoint",
            "--manifest",
            "--predictions",
            "--summary",
            "--oracle",
            "--query-protocol",
            "--threads",
            "--warmup-repetitions",
            "--measured-repetitions",
            "--limit",
        ];
        if let Some(unknown) = values.keys().find(|flag| !known.contains(&flag.as_str())) {
            return Err(format!("unknown argument {unknown}"));
        }
        let parse_usize = |flag: &str| -> Result<usize, String> {
            take(flag)?
                .parse()
                .map_err(|_| format!("{flag} must be an integer"))
        };
        let threads = parse_usize("--threads")?;
        let measured_repetitions = parse_usize("--measured-repetitions")?;
        if threads == 0 || measured_repetitions == 0 {
            return Err("threads and measured repetitions must be positive".to_string());
        }
        Ok(Some(Self {
            checkpoint: PathBuf::from(take("--checkpoint")?),
            manifest: PathBuf::from(take("--manifest")?),
            predictions: PathBuf::from(take("--predictions")?),
            summary: PathBuf::from(take("--summary")?),
            oracle: OracleKind::parse(&take("--oracle")?)?,
            query_protocol: QueryProtocol::parse(&take("--query-protocol")?)?,
            threads,
            warmup_repetitions: parse_usize("--warmup-repetitions")?,
            measured_repetitions,
            limit: values
                .get("--limit")
                .map(|value| {
                    value
                        .parse::<usize>()
                        .map_err(|_| "--limit must be an integer".to_string())
                })
                .transpose()?,
        }))
    }
}
