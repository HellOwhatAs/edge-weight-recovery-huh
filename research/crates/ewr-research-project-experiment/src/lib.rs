//! Research-only orchestration for full-data project training and prediction.
//!
//! This crate composes the stable production trainer, artifact adapter, and CCH
//! oracle without adding experiment switches to any production crate.

use ewr_cch::{CCH_ORACLE_IDENTITY_V1, CchOracle};
use ewr_core::{
    EdgeId, LineGraph, OracleQuery, QueryEndpoint, ROUTING_INFINITY, RoadNetwork, RoutingOracle,
    Trainer, TrainingOutcome,
};
use ewr_io::{
    LoadedDataset, TrainConfig, load_dataset, load_network, load_training_artifact,
    save_training_artifact,
};
use ewr_research_protocol::{
    DatasetRecordV1, PredictionRecordV1, read_dataset_jsonl, write_prediction_jsonl,
};
use rayon::ThreadPoolBuilder;
use serde::Serialize;
use std::collections::BTreeMap;
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fmt::{self, Display, Formatter};
use std::fs::{File, OpenOptions};
use std::hint::black_box;
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub const TRAINING_SUMMARY_SCHEMA_V1: &str = "ewr.project-training-summary/v1";
pub const PREDICTION_DIAGNOSTICS_SCHEMA_V1: &str = "ewr.project-prediction-diagnostics/v1";
pub const TRAINING_SUMMARY_FILENAME: &str = "training-summary.json";
pub const USAGE: &str = "\
Usage:
  ewr-project-experiment train --config PATH --output-dir DIR
  ewr-project-experiment predict --artifact PATH --dataset-jsonl PATH \\
    --nodes PATH --edges PATH \\
    --predictions PATH --diagnostics PATH [--threads N] \\
    [--warmup-repetitions N] [--measured-repetitions N]
";

static TEMPORARY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// One independently saved core snapshot.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SnapshotSummary {
    pub completed_updates: u64,
    pub objective: f64,
    pub artifact: String,
}

/// Training timing with input adaptation separate from the fitted loop.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct TrainingTiming {
    pub input_load_seconds: f64,
    pub setup_training_and_snapshot_seconds: f64,
    pub total_before_summary_write_seconds: f64,
}

/// Concise record emitted by the research training command.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TrainingSummary {
    pub schema: String,
    pub accepted: usize,
    pub dropped: usize,
    pub coordinates: usize,
    pub completed_updates: u64,
    pub objective: f64,
    pub threads: usize,
    pub checkpoint_every: u64,
    pub snapshots: Vec<SnapshotSummary>,
    pub timing: TrainingTiming,
    pub peak_rss_kib: Option<u64>,
}

/// Fixed edge-to-edge prediction timing.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PredictionTiming {
    pub input_and_network_adapter_load_seconds: f64,
    pub line_graph_and_query_preparation_seconds: f64,
    pub warmup_metric_and_query_seconds: Vec<f64>,
    pub measured_metric_and_query_seconds: Vec<f64>,
    pub mean_metric_and_query_seconds: f64,
    pub mean_seconds_per_query: f64,
    pub queries_per_second: f64,
    pub total_before_diagnostics_write_seconds: f64,
    pub timing_boundary: String,
}

/// Method-local diagnostics kept out of protocol prediction rows.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PredictionDiagnostics {
    pub schema: String,
    pub method: String,
    pub query_protocol: String,
    pub samples: usize,
    pub threads: usize,
    pub warmup_repetitions: usize,
    pub measured_repetitions: usize,
    pub completed_updates: u64,
    pub objective: f64,
    pub topology_id: String,
    pub oracle_identity: String,
    pub deterministic_repetitions: bool,
    pub timing: PredictionTiming,
    pub peak_rss_kib: Option<u64>,
}

/// In-memory prediction output used by the CLI and tests.
#[derive(Clone, Debug, PartialEq)]
pub struct PredictionRun {
    pub predictions: Vec<PredictionRecordV1>,
    pub diagnostics: PredictionDiagnostics,
}

/// Train already-adapted values and retain every requested snapshot.
pub fn train_loaded(
    config: &TrainConfig,
    dataset: &LoadedDataset,
    output_dir: &Path,
) -> Result<TrainingSummary, ExperimentError> {
    validate_runtime(config.threads, config.checkpoint_every, config.fit.updates)?;
    let training_started = Instant::now();
    let pool = ThreadPoolBuilder::new()
        .num_threads(config.threads)
        .build()
        .map_err(|error| failure(format!("failed to build Rayon pool: {error}")))?;
    let mut snapshots = Vec::new();
    let outcome = pool
        .install(|| {
            let mut oracle = CchOracle::new();
            let mut trainer = Trainer::new(
                &dataset.network,
                &dataset.trajectories,
                &config.fit,
                &mut oracle,
            )?;
            trainer.fit_with_snapshots(config.checkpoint_every, |snapshot| {
                let artifact = output_dir.join(format!(
                    "checkpoint-{}.json",
                    snapshot.state.completed_updates()
                ));
                save_training_artifact(&artifact, snapshot)?;
                snapshots.push(SnapshotSummary {
                    completed_updates: snapshot.state.completed_updates(),
                    objective: snapshot.result.diagnostics.objective,
                    artifact: artifact.display().to_string(),
                });
                Ok::<(), ewr_io::ArtifactError>(())
            })
        })
        .map_err(|error| failure(format!("training failed: {error}")))?;
    let training_seconds = training_started.elapsed().as_secs_f64();
    validate_snapshots(config, &snapshots)?;

    Ok(TrainingSummary {
        schema: TRAINING_SUMMARY_SCHEMA_V1.into(),
        accepted: dataset.report.accepted,
        dropped: dataset.report.dropped(),
        coordinates: outcome.result.model.weights().len(),
        completed_updates: outcome.result.diagnostics.completed_updates,
        objective: outcome.result.diagnostics.objective,
        threads: config.threads,
        checkpoint_every: config.checkpoint_every,
        snapshots,
        timing: TrainingTiming {
            input_load_seconds: 0.0,
            setup_training_and_snapshot_seconds: training_seconds,
            total_before_summary_write_seconds: training_seconds,
        },
        peak_rss_kib: peak_rss_kib(),
    })
}

/// Load production inputs, run training, and atomically publish its summary.
pub fn run_train(arguments: &TrainArguments) -> Result<TrainingSummary, ExperimentError> {
    let total_started = Instant::now();
    let input_started = Instant::now();
    let config = TrainConfig::load(&arguments.config)
        .map_err(|error| failure(format!("configuration failed: {error}")))?;
    let dataset = load_dataset(&config.dataset)
        .map_err(|error| failure(format!("dataset loading failed: {error}")))?;
    let input_seconds = input_started.elapsed().as_secs_f64();
    let mut summary = train_loaded(&config, &dataset, &arguments.output_dir)?;
    summary.timing.input_load_seconds = input_seconds;
    summary.timing.total_before_summary_write_seconds = total_started.elapsed().as_secs_f64();
    summary.peak_rss_kib = peak_rss_kib();
    write_json_atomic(
        &arguments.output_dir.join(TRAINING_SUMMARY_FILENAME),
        &summary,
    )?;
    Ok(summary)
}

/// Predict complete raw-edge paths with fixed true first and last edges.
pub fn predict_loaded(
    network: &RoadNetwork,
    dataset: &[DatasetRecordV1],
    artifact: &TrainingOutcome,
    threads: usize,
    warmup_repetitions: usize,
    measured_repetitions: usize,
) -> Result<PredictionRun, ExperimentError> {
    if threads == 0 {
        return Err(failure("threads must be positive"));
    }
    if measured_repetitions == 0 {
        return Err(failure("measured repetitions must be positive"));
    }
    if dataset.is_empty() {
        return Err(failure("prediction dataset must not be empty"));
    }
    if artifact.state.oracle_identity() != CCH_ORACLE_IDENTITY_V1 {
        return Err(failure(format!(
            "artifact oracle identity {:?} is not the production CCH identity {:?}",
            artifact.state.oracle_identity(),
            CCH_ORACLE_IDENTITY_V1
        )));
    }

    let preparation_started = Instant::now();
    let graph = LineGraph::build(
        network,
        artifact.state.lower_factor(),
        artifact.state.upper_factor(),
    )
    .map_err(|error| failure(format!("line-graph construction failed: {error}")))?;
    validate_artifact_against_graph(&graph, artifact)?;
    let quantized = quantize_weights(artifact.result.model.weights())?;
    let queries = build_edge_to_edge_queries(network, dataset)?;
    let preparation_seconds = preparation_started.elapsed().as_secs_f64();

    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|error| failure(format!("failed to build Rayon pool: {error}")))?;
    let (warmup_seconds, measured_seconds, paths, oracle_identity) = pool
        .install(|| {
            let mut oracle = CchOracle::new();
            let mut warmup_seconds = Vec::with_capacity(warmup_repetitions);
            for _ in 0..warmup_repetitions {
                let started = Instant::now();
                let paths =
                    oracle.shortest_paths(graph.routing_topology(), &quantized, &queries)?;
                warmup_seconds.push(started.elapsed().as_secs_f64());
                black_box(paths);
            }

            let mut measured_seconds = Vec::with_capacity(measured_repetitions);
            let mut first_paths = None;
            for repetition in 0..measured_repetitions {
                let started = Instant::now();
                let candidate =
                    oracle.shortest_paths(graph.routing_topology(), &quantized, &queries)?;
                measured_seconds.push(started.elapsed().as_secs_f64());
                if let Some(expected) = &first_paths {
                    if expected != &candidate {
                        return Err(ewr_core::OracleError::new(format!(
                            "measured repetition {repetition} produced different paths"
                        )));
                    }
                } else {
                    first_paths = Some(candidate);
                }
            }
            Ok::<_, ewr_core::OracleError>((
                warmup_seconds,
                measured_seconds,
                first_paths.expect("positive measured repetitions were validated"),
                oracle.identity().to_string(),
            ))
        })
        .map_err(|error| failure(format!("CCH prediction failed: {error}")))?;

    let predictions = dataset
        .iter()
        .zip(&paths)
        .map(|(record, path)| PredictionRecordV1 {
            sample_id: record.sample_id.clone(),
            predicted_edge_ids: path
                .nodes()
                .iter()
                .map(|edge| edge.index() as u32)
                .collect(),
        })
        .collect::<Vec<_>>();
    let mean_seconds = mean(&measured_seconds);
    let samples = dataset.len();
    Ok(PredictionRun {
        predictions,
        diagnostics: PredictionDiagnostics {
            schema: PREDICTION_DIAGNOSTICS_SCHEMA_V1.into(),
            method: "project_cch".into(),
            query_protocol: "fixed_true_first_edge_to_true_last_edge_complete_sequence".into(),
            samples,
            threads,
            warmup_repetitions,
            measured_repetitions,
            completed_updates: artifact.result.diagnostics.completed_updates,
            objective: artifact.result.diagnostics.objective,
            topology_id: artifact.result.model.topology_id().as_str().into(),
            oracle_identity,
            deterministic_repetitions: true,
            timing: PredictionTiming {
                input_and_network_adapter_load_seconds: 0.0,
                line_graph_and_query_preparation_seconds: preparation_seconds,
                warmup_metric_and_query_seconds: warmup_seconds,
                measured_metric_and_query_seconds: measured_seconds,
                mean_metric_and_query_seconds: mean_seconds,
                mean_seconds_per_query: mean_seconds / samples as f64,
                queries_per_second: samples as f64 / mean_seconds,
                total_before_diagnostics_write_seconds: preparation_seconds + mean_seconds,
                timing_boundary: "each repetition includes one CCH metric customization and the complete ordered query batch; the first oracle call additionally includes lazy CCH topology preprocessing".into(),
            },
            peak_rss_kib: peak_rss_kib(),
        },
    })
}

/// Load files, write strict v1 predictions, and write method-local diagnostics.
pub fn run_predict(arguments: &PredictArguments) -> Result<PredictionRun, ExperimentError> {
    let total_started = Instant::now();
    let input_started = Instant::now();
    let network = load_network(&arguments.nodes, &arguments.edges)
        .map_err(|error| failure(format!("network loading failed: {error}")))?;
    let dataset_file = File::open(&arguments.dataset_jsonl).map_err(|error| {
        failure(format!(
            "failed to open dataset {}: {error}",
            arguments.dataset_jsonl.display()
        ))
    })?;
    let dataset = read_dataset_jsonl(BufReader::new(dataset_file))
        .map_err(|error| failure(format!("invalid dataset JSONL: {error}")))?;
    let artifact = load_training_artifact(&arguments.artifact)
        .map_err(|error| failure(format!("artifact loading failed: {error}")))?;
    let input_seconds = input_started.elapsed().as_secs_f64();

    let mut run = predict_loaded(
        &network,
        &dataset,
        &artifact,
        arguments.threads,
        arguments.warmup_repetitions,
        arguments.measured_repetitions,
    )?;
    let mut encoded_predictions = Vec::new();
    write_prediction_jsonl(&mut encoded_predictions, &run.predictions)
        .map_err(|error| failure(format!("failed to encode predictions: {error}")))?;
    atomic_write(&arguments.predictions, &encoded_predictions)?;
    run.diagnostics
        .timing
        .input_and_network_adapter_load_seconds = input_seconds;
    run.diagnostics
        .timing
        .total_before_diagnostics_write_seconds = total_started.elapsed().as_secs_f64();
    run.diagnostics.peak_rss_kib = peak_rss_kib();
    write_json_atomic(&arguments.diagnostics, &run.diagnostics)?;
    Ok(run)
}

fn validate_runtime(threads: usize, cadence: u64, updates: u64) -> Result<(), ExperimentError> {
    if threads == 0 {
        return Err(failure("threads must be positive"));
    }
    if cadence == 0 || cadence > updates {
        return Err(failure(format!(
            "checkpoint cadence {cadence} must be in 1..={updates}"
        )));
    }
    Ok(())
}

fn validate_snapshots(
    config: &TrainConfig,
    snapshots: &[SnapshotSummary],
) -> Result<(), ExperimentError> {
    let Some(first) = snapshots.first() else {
        return Err(failure("trainer produced no snapshots"));
    };
    let last = snapshots
        .last()
        .expect("a nonempty snapshot list has a final element");
    if first.completed_updates != 0 || last.completed_updates != config.fit.updates {
        return Err(failure(format!(
            "snapshot clocks must include 0 and {}, got {} and {}",
            config.fit.updates, first.completed_updates, last.completed_updates
        )));
    }
    if snapshots
        .windows(2)
        .any(|pair| pair[0].completed_updates >= pair[1].completed_updates)
    {
        return Err(failure("snapshot clocks are not strictly increasing"));
    }
    Ok(())
}

fn validate_artifact_against_graph(
    graph: &LineGraph,
    artifact: &TrainingOutcome,
) -> Result<(), ExperimentError> {
    let model = &artifact.result.model;
    if model.topology_id() != graph.topology_id() {
        return Err(failure(format!(
            "artifact topology {:?} differs from network topology {:?}",
            model.topology_id().as_str(),
            graph.topology_id().as_str()
        )));
    }
    if model.transitions() != graph.transitions() {
        return Err(failure(
            "artifact transition coordinate order differs from the rebuilt line graph",
        ));
    }
    if !same_f64_bits(artifact.state.initial_weights(), graph.initial_weights()) {
        return Err(failure(
            "artifact baseline differs from the rebuilt line-graph baseline",
        ));
    }
    Ok(())
}

fn build_edge_to_edge_queries(
    network: &RoadNetwork,
    dataset: &[DatasetRecordV1],
) -> Result<Vec<OracleQuery>, ExperimentError> {
    dataset
        .iter()
        .map(|record| {
            let edges = record
                .original_edge_ids
                .iter()
                .copied()
                .map(EdgeId::new)
                .collect::<Vec<_>>();
            let trajectory = ewr_core::Trajectory::new(edges);
            network.validate_trajectory(&trajectory).map_err(|error| {
                failure(format!(
                    "sample {:?} is not a valid common trajectory: {error}",
                    record.sample_id
                ))
            })?;
            let first = *trajectory
                .edges()
                .first()
                .expect("protocol datasets contain at least two edges");
            let last = *trajectory
                .edges()
                .last()
                .expect("protocol datasets contain at least two edges");
            Ok(OracleQuery::new(
                vec![QueryEndpoint::zero(first)],
                vec![QueryEndpoint::zero(last)],
            ))
        })
        .collect()
}

fn quantize_weights(weights: &[f64]) -> Result<Vec<u32>, ExperimentError> {
    weights
        .iter()
        .copied()
        .enumerate()
        .map(|(coordinate, weight)| {
            if !weight.is_finite() || weight <= 0.0 {
                return Err(failure(format!(
                    "weight[{coordinate}] cannot be quantized: {weight}"
                )));
            }
            let rounded = weight.round().max(1.0);
            if rounded >= f64::from(ROUTING_INFINITY) {
                return Err(failure(format!(
                    "weight[{coordinate}] exceeds the routing limit: {weight}"
                )));
            }
            Ok(rounded as u32)
        })
        .collect()
}

fn same_f64_bits(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(&left, &right)| left.to_bits() == right.to_bits())
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<(), ExperimentError> {
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| failure(format!("failed to encode JSON: {error}")))?;
    bytes.push(b'\n');
    atomic_write(path, &bytes)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), ExperimentError> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).map_err(|error| {
            failure(format!(
                "failed to create output directory {}: {error}",
                parent.display()
            ))
        })?;
    }
    let filename = path
        .file_name()
        .ok_or_else(|| failure(format!("output path has no filename: {}", path.display())))?;
    let mut temporary = None;
    for _ in 0..32 {
        let mut name = OsString::from(".");
        name.push(filename);
        name.push(format!(
            ".{}.{}.tmp",
            std::process::id(),
            TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let candidate = path.with_file_name(name);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                temporary = Some((candidate, file));
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(failure(format!(
                    "failed to create temporary output {}: {error}",
                    candidate.display()
                )));
            }
        }
    }
    let (temporary_path, mut file) = temporary.ok_or_else(|| {
        failure(format!(
            "could not reserve temporary output for {}",
            path.display()
        ))
    })?;
    let result = (|| {
        file.write_all(bytes).map_err(|error| {
            failure(format!(
                "failed to write temporary output {}: {error}",
                temporary_path.display()
            ))
        })?;
        file.sync_all().map_err(|error| {
            failure(format!(
                "failed to sync temporary output {}: {error}",
                temporary_path.display()
            ))
        })?;
        std::fs::rename(&temporary_path, path).map_err(|error| {
            failure(format!(
                "failed to replace {} with {}: {error}",
                path.display(),
                temporary_path.display()
            ))
        })
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary_path);
    }
    result
}

fn peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        line.strip_prefix("VmHWM:")?
            .split_whitespace()
            .next()?
            .parse()
            .ok()
    })
}

/// Exact actions accepted by the experiment binary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CliAction {
    Help,
    Train(TrainArguments),
    Predict(PredictArguments),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainArguments {
    pub config: PathBuf,
    pub output_dir: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PredictArguments {
    pub artifact: PathBuf,
    pub dataset_jsonl: PathBuf,
    pub nodes: PathBuf,
    pub edges: PathBuf,
    pub predictions: PathBuf,
    pub diagnostics: PathBuf,
    pub threads: usize,
    pub warmup_repetitions: usize,
    pub measured_repetitions: usize,
}

/// Strictly parse arguments after the binary name.
pub fn parse_cli<I, S>(arguments: I) -> Result<CliAction, ExperimentError>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let arguments = arguments.into_iter().map(Into::into).collect::<Vec<_>>();
    let Some(command) = arguments.first() else {
        return Err(failure("missing command"));
    };
    if command == OsStr::new("--help") || command == OsStr::new("-h") {
        return if arguments.len() == 1 {
            Ok(CliAction::Help)
        } else {
            Err(failure("help does not accept additional arguments"))
        };
    }
    if arguments.len() == 2
        && (arguments[1] == OsStr::new("--help") || arguments[1] == OsStr::new("-h"))
    {
        return Ok(CliAction::Help);
    }
    let values = parse_flag_values(&arguments[1..])?;
    if command == OsStr::new("train") {
        parse_train(values).map(CliAction::Train)
    } else if command == OsStr::new("predict") {
        parse_predict(values).map(CliAction::Predict)
    } else {
        Err(failure(format!("unknown command {command:?}")))
    }
}

fn parse_flag_values(
    arguments: &[OsString],
) -> Result<BTreeMap<String, OsString>, ExperimentError> {
    if !arguments.len().is_multiple_of(2) {
        return Err(failure("every option requires a separate value"));
    }
    let mut values = BTreeMap::new();
    for pair in arguments.chunks_exact(2) {
        let flag = pair[0]
            .to_str()
            .ok_or_else(|| failure(format!("non-Unicode option {:?}", pair[0])))?;
        if !flag.starts_with("--") {
            return Err(failure(format!("unexpected positional argument {flag:?}")));
        }
        if pair[1].is_empty() || pair[1].as_encoded_bytes().starts_with(b"-") {
            return Err(failure(format!("invalid value {:?} for {flag}", pair[1])));
        }
        if values.insert(flag.to_string(), pair[1].clone()).is_some() {
            return Err(failure(format!("duplicate option {flag}")));
        }
    }
    Ok(values)
}

fn parse_train(mut values: BTreeMap<String, OsString>) -> Result<TrainArguments, ExperimentError> {
    let arguments = TrainArguments {
        config: required_path(&mut values, "--config")?,
        output_dir: required_path(&mut values, "--output-dir")?,
    };
    reject_unknown(values)?;
    Ok(arguments)
}

fn parse_predict(
    mut values: BTreeMap<String, OsString>,
) -> Result<PredictArguments, ExperimentError> {
    let arguments = PredictArguments {
        artifact: required_path(&mut values, "--artifact")?,
        dataset_jsonl: required_path(&mut values, "--dataset-jsonl")?,
        nodes: required_path(&mut values, "--nodes")?,
        edges: required_path(&mut values, "--edges")?,
        predictions: required_path(&mut values, "--predictions")?,
        diagnostics: required_path(&mut values, "--diagnostics")?,
        threads: optional_usize(&mut values, "--threads", 16)?,
        warmup_repetitions: optional_usize(&mut values, "--warmup-repetitions", 1)?,
        measured_repetitions: optional_usize(&mut values, "--measured-repetitions", 5)?,
    };
    reject_unknown(values)?;
    if arguments.threads == 0 || arguments.measured_repetitions == 0 {
        return Err(failure(
            "threads and measured repetitions must both be positive",
        ));
    }
    Ok(arguments)
}

fn required_path(
    values: &mut BTreeMap<String, OsString>,
    flag: &str,
) -> Result<PathBuf, ExperimentError> {
    values
        .remove(flag)
        .map(PathBuf::from)
        .ok_or_else(|| failure(format!("missing required option {flag}")))
}

fn optional_usize(
    values: &mut BTreeMap<String, OsString>,
    flag: &str,
    default: usize,
) -> Result<usize, ExperimentError> {
    let Some(value) = values.remove(flag) else {
        return Ok(default);
    };
    value
        .to_str()
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| failure(format!("{flag} must be a nonnegative integer")))
}

fn reject_unknown(values: BTreeMap<String, OsString>) -> Result<(), ExperimentError> {
    if let Some(flag) = values.keys().next() {
        Err(failure(format!("unknown option {flag}")))
    } else {
        Ok(())
    }
}

/// One concise, cloneable orchestration failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExperimentError(String);

fn failure(message: impl Into<String>) -> ExperimentError {
    ExperimentError(message.into())
}

impl Display for ExperimentError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ExperimentError {}

#[cfg(test)]
mod tests {
    use super::*;
    use ewr_core::{FitOptions, NodeId, Trajectory};
    use ewr_io::LoadReport;

    struct TemporaryDirectory(PathBuf);

    impl TemporaryDirectory {
        fn new(label: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "ewr-project-experiment-{label}-{}-{}",
                std::process::id(),
                TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
            Self(path)
        }
    }

    impl Drop for TemporaryDirectory {
        fn drop(&mut self) {
            if let Err(error) = std::fs::remove_dir_all(&self.0)
                && error.kind() != std::io::ErrorKind::NotFound
            {
                panic!("failed to remove temporary directory: {error}");
            }
        }
    }

    fn network() -> RoadNetwork {
        RoadNetwork::new(
            vec![
                NodeId::new(0),
                NodeId::new(1),
                NodeId::new(2),
                NodeId::new(1),
                NodeId::new(4),
            ],
            vec![
                NodeId::new(1),
                NodeId::new(2),
                NodeId::new(3),
                NodeId::new(4),
                NodeId::new(3),
            ],
            vec![2.0, 3.0, 5.0, 1.0, 1.0],
            vec![0.0, 1.0, 2.0, 3.0, 2.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap()
    }

    fn trajectories() -> Vec<Trajectory> {
        vec![
            Trajectory::new(vec![EdgeId::new(0), EdgeId::new(1), EdgeId::new(2)]),
            Trajectory::new(vec![EdgeId::new(0), EdgeId::new(3), EdgeId::new(4)]),
        ]
    }

    fn config(updates: u64, checkpoint_every: u64) -> TrainConfig {
        TrainConfig {
            dataset: ewr_io::DatasetPaths {
                nodes: "unused-nodes.shp".into(),
                edges: "unused-edges.shp".into(),
                trajectories: "unused.pkl".into(),
            },
            fit: FitOptions {
                eta0: 0.01,
                lambda: 1.0,
                lower_factor: 0.1,
                upper_factor: 10.0,
                updates,
            },
            threads: 2,
            checkpoint_every,
        }
    }

    fn loaded_dataset() -> LoadedDataset {
        LoadedDataset {
            network: network(),
            trajectories: trajectories(),
            report: LoadReport {
                available: 2,
                accepted: 2,
                ..LoadReport::default()
            },
        }
    }

    #[test]
    fn train_retains_initial_cadence_and_final_snapshots() {
        let directory = TemporaryDirectory::new("train");
        let summary = train_loaded(&config(3, 2), &loaded_dataset(), &directory.0).unwrap();
        assert_eq!(
            summary
                .snapshots
                .iter()
                .map(|snapshot| snapshot.completed_updates)
                .collect::<Vec<_>>(),
            vec![0, 2, 3]
        );
        for snapshot in &summary.snapshots {
            let restored = load_training_artifact(&snapshot.artifact).unwrap();
            assert_eq!(
                restored.state.completed_updates(),
                snapshot.completed_updates
            );
        }
    }

    #[test]
    fn predict_is_fixed_edge_to_edge_complete_and_deterministic() {
        let mut oracle = CchOracle::new();
        let artifact = Trainer::new(&network(), &trajectories(), &config(1, 1).fit, &mut oracle)
            .unwrap()
            .fit()
            .unwrap();
        let dataset = vec![DatasetRecordV1 {
            sample_id: "test:0".into(),
            original_edge_ids: vec![0, 1, 2],
        }];
        let run = predict_loaded(&network(), &dataset, &artifact, 2, 1, 2).unwrap();
        assert_eq!(run.predictions.len(), 1);
        assert_eq!(run.predictions[0].predicted_edge_ids, vec![0, 1, 2]);
        assert!(run.diagnostics.deterministic_repetitions);
        assert_eq!(
            run.diagnostics
                .timing
                .measured_metric_and_query_seconds
                .len(),
            2
        );
    }

    #[test]
    fn predict_rejects_a_network_with_a_different_baseline() {
        let mut oracle = CchOracle::new();
        let source_network = network();
        let artifact = Trainer::new(
            &source_network,
            &trajectories(),
            &config(1, 1).fit,
            &mut oracle,
        )
        .unwrap()
        .fit()
        .unwrap();
        let changed = RoadNetwork::new(
            source_network.tails().to_vec(),
            source_network.heads().to_vec(),
            vec![9.0; source_network.edge_count()],
            source_network.x().to_vec(),
            source_network.y().to_vec(),
        )
        .unwrap();
        let dataset = vec![DatasetRecordV1 {
            sample_id: "test:0".into(),
            original_edge_ids: vec![0, 1, 2],
        }];
        assert!(
            predict_loaded(&changed, &dataset, &artifact, 1, 0, 1)
                .unwrap_err()
                .to_string()
                .contains("baseline")
        );
    }

    #[test]
    fn cli_is_strict_and_predict_defaults_to_registered_timing() {
        assert_eq!(
            parse_cli(["train", "--config", "train.json", "--output-dir", "out"]).unwrap(),
            CliAction::Train(TrainArguments {
                config: "train.json".into(),
                output_dir: "out".into(),
            })
        );
        let action = parse_cli([
            "predict",
            "--artifact",
            "model.json",
            "--dataset-jsonl",
            "test.jsonl",
            "--nodes",
            "nodes.shp",
            "--edges",
            "edges.shp",
            "--predictions",
            "predictions.jsonl",
            "--diagnostics",
            "diagnostics.json",
        ])
        .unwrap();
        let CliAction::Predict(arguments) = action else {
            panic!("expected predict action");
        };
        assert_eq!(arguments.threads, 16);
        assert_eq!(arguments.warmup_repetitions, 1);
        assert_eq!(arguments.measured_repetitions, 5);

        assert!(
            parse_cli([
                "predict",
                "--artifact",
                "model.json",
                "--dataset-jsonl",
                "test.jsonl",
                "--nodes",
                "nodes.shp",
                "--edges",
                "edges.shp",
                "--network-trajectories",
                "small.pkl",
                "--predictions",
                "predictions.jsonl",
                "--diagnostics",
                "diagnostics.json",
            ])
            .unwrap_err()
            .to_string()
            .contains("unknown option --network-trajectories")
        );

        assert!(parse_cli(["train", "--config", "x"]).is_err());
        assert!(parse_cli(["train", "--config", "x", "--wat", "y"]).is_err());
        assert!(
            parse_cli([
                "train",
                "--config",
                "x",
                "--config",
                "y",
                "--output-dir",
                "out"
            ])
            .is_err()
        );
    }
}
