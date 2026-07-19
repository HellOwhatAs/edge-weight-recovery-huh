//! Equal-information road-length and first-order Markov shortest-path baselines.

use ewr_cch::CchOracle;
use ewr_core::{
    EdgeId, LineGraph, OracleQuery, QueryEndpoint, ROUTING_INFINITY, RoadNetwork, RoutingOracle,
    Trajectory,
};
use ewr_io::load_network;
use ewr_research_evaluator::{EvaluationSummaryV1, evaluate_records};
use ewr_research_protocol::{
    DatasetRecordV1, PredictionRecordV1, read_dataset_jsonl, write_prediction_jsonl,
};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fmt::{self, Display, Formatter};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const ARTIFACT_SCHEMA_V1: &str = "ewr.static-route-baseline-artifact/v1";
pub const TRAINING_DIAGNOSTICS_SCHEMA_V1: &str =
    "ewr.static-route-baseline-training-diagnostics/v1";
pub const PREDICTION_DIAGNOSTICS_SCHEMA_V1: &str =
    "ewr.static-route-baseline-prediction-diagnostics/v1";
pub const MARKOV_QUANTIZATION_SCALE: f64 = 100_000.0;
pub const USAGE: &str = "Usage:\n  ewr-static-baseline train --method length|markov --nodes PATH --edges PATH \\\n    --validation-jsonl PATH --artifact PATH --diagnostics PATH \\\n    [--train-jsonl PATH] [--alpha-candidates CSV] [--threads N]\n  ewr-static-baseline predict --artifact PATH --nodes PATH --edges PATH \\\n    --dataset-jsonl PATH --predictions PATH --diagnostics PATH [--threads N] \\\n    [--warmup-repetitions N] [--measured-repetitions N]";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    Length,
    Markov,
}

impl Method {
    fn parse(value: &OsStr) -> Result<Self, BaselineError> {
        match value.to_str() {
            Some("length") => Ok(Self::Length),
            Some("markov") => Ok(Self::Markov),
            _ => fail(format!(
                "unsupported method {value:?}; expected length or markov"
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Length => "sp_length",
            Self::Markov => "markov_sp",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ValidationCandidate {
    pub alpha: Option<f64>,
    pub quantization_scale: f64,
    pub routing_seconds: f64,
    pub exact_match: f64,
    pub edge_precision: f64,
    pub edge_recall: f64,
    pub edge_f1: f64,
    pub edge_jaccard: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainingTiming {
    pub network_and_topology_seconds: f64,
    pub training_records_load_seconds: f64,
    pub transition_counting_seconds: f64,
    pub validation_selection_seconds: f64,
    pub total_before_artifact_write_seconds: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StaticArtifact {
    pub schema: String,
    pub method: Method,
    pub topology_id: String,
    pub coordinate_count: usize,
    pub quantized_weights: Vec<u32>,
    pub selected_alpha: Option<f64>,
    pub quantization_scale: f64,
    pub training_samples: usize,
    pub transition_observations: u64,
    pub observed_coordinates: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TrainingDiagnostics {
    pub schema: String,
    pub method: String,
    pub query_protocol: String,
    pub threads: usize,
    pub training_samples: usize,
    pub validation_samples: usize,
    pub coordinate_count: usize,
    pub transition_observations: u64,
    pub observed_coordinates: usize,
    pub selected_alpha: Option<f64>,
    pub validation_candidates: Vec<ValidationCandidate>,
    pub timing: TrainingTiming,
    pub peak_rss_kib: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PredictionTiming {
    pub input_and_network_load_seconds: f64,
    pub topology_and_query_preparation_seconds: f64,
    pub warmup_metric_and_query_seconds: Vec<f64>,
    pub measured_metric_and_query_seconds: Vec<f64>,
    pub mean_metric_and_query_seconds: f64,
    pub mean_seconds_per_query: f64,
    pub queries_per_second: f64,
    pub total_before_diagnostics_write_seconds: f64,
    pub timing_boundary: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PredictionDiagnostics {
    pub schema: String,
    pub method: String,
    pub query_protocol: String,
    pub samples: usize,
    pub threads: usize,
    pub warmup_repetitions: usize,
    pub measured_repetitions: usize,
    pub selected_alpha: Option<f64>,
    pub topology_id: String,
    pub deterministic_repetitions: bool,
    pub endpoint_mismatches: usize,
    pub timing: PredictionTiming,
    pub peak_rss_kib: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainArguments {
    pub method: Method,
    pub nodes: PathBuf,
    pub edges: PathBuf,
    pub train_jsonl: Option<PathBuf>,
    pub validation_jsonl: PathBuf,
    pub artifact: PathBuf,
    pub diagnostics: PathBuf,
    pub alpha_candidates: Vec<AlphaBits>,
    pub threads: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AlphaBits(u64);

impl AlphaBits {
    fn get(self) -> f64 {
        f64::from_bits(self.0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PredictArguments {
    pub artifact: PathBuf,
    pub nodes: PathBuf,
    pub edges: PathBuf,
    pub dataset_jsonl: PathBuf,
    pub predictions: PathBuf,
    pub diagnostics: PathBuf,
    pub threads: usize,
    pub warmup_repetitions: usize,
    pub measured_repetitions: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CliAction {
    Help,
    Train(TrainArguments),
    Predict(PredictArguments),
}

pub fn run_train(arguments: &TrainArguments) -> Result<TrainingDiagnostics, BaselineError> {
    let total_started = Instant::now();
    let topology_started = Instant::now();
    let network = load_network(&arguments.nodes, &arguments.edges)
        .map_err(|error| failure(format!("network loading failed: {error}")))?;
    let graph = LineGraph::build(&network, 0.1, 10.0)
        .map_err(|error| failure(format!("line-graph construction failed: {error}")))?;
    let topology_seconds = topology_started.elapsed().as_secs_f64();
    let validation = read_dataset_file(&arguments.validation_jsonl)?;
    let validation_queries = build_queries(&network, &validation)?;
    let pool = ThreadPoolBuilder::new()
        .num_threads(arguments.threads)
        .build()
        .map_err(|error| {
            failure(format!(
                "failed to build {0}-thread pool: {error}",
                arguments.threads
            ))
        })?;

    let training_load_started = Instant::now();
    let training = match (&arguments.method, &arguments.train_jsonl) {
        (Method::Markov, Some(path)) => read_dataset_file(path)?,
        (Method::Markov, None) => return fail("markov training requires --train-jsonl"),
        (Method::Length, Some(_)) => return fail("length training does not accept --train-jsonl"),
        (Method::Length, None) => Vec::new(),
    };
    let training_load_seconds = training_load_started.elapsed().as_secs_f64();

    let counting_started = Instant::now();
    let counts = if arguments.method == Method::Markov {
        pool.install(|| transition_counts(&network, &graph, &training))?
    } else {
        vec![0; graph.coordinate_count()]
    };
    let counting_seconds = counting_started.elapsed().as_secs_f64();
    let transition_observations = counts.iter().copied().sum();
    let observed_coordinates = counts.iter().filter(|&&count| count > 0).count();

    let selection_started = Instant::now();
    let mut validation_candidates = Vec::new();
    let mut oracle = CchOracle::new();
    let candidate_alphas = if arguments.method == Method::Length {
        vec![None]
    } else {
        arguments
            .alpha_candidates
            .iter()
            .map(|alpha| Some(alpha.get()))
            .collect()
    };
    let mut candidate_weights = Vec::new();
    for alpha in candidate_alphas {
        let weights = match alpha {
            Some(alpha) => markov_weights(&graph, &counts, alpha)?,
            None => length_weights(&graph)?,
        };
        let routing_started = Instant::now();
        let paths = pool
            .install(|| {
                oracle.shortest_paths(graph.routing_topology(), &weights, &validation_queries)
            })
            .map_err(|error| failure(format!("validation routing failed: {error}")))?;
        let routing_seconds = routing_started.elapsed().as_secs_f64();
        let predictions = predictions_from_paths(&validation, &paths);
        let evaluation = evaluate_records(&validation, &predictions)
            .map_err(|error| failure(format!("validation evaluation failed: {error}")))?;
        validation_candidates.push(validation_candidate(
            alpha,
            if alpha.is_some() {
                MARKOV_QUANTIZATION_SCALE
            } else {
                1.0
            },
            routing_seconds,
            &evaluation,
        ));
        candidate_weights.push(weights);
    }
    let selected_index = validation_candidates
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| compare_candidates(left, right))
        .map(|(index, _)| index)
        .ok_or_else(|| failure("no validation candidate was evaluated"))?;
    let selected = &validation_candidates[selected_index];
    let artifact = StaticArtifact {
        schema: ARTIFACT_SCHEMA_V1.into(),
        method: arguments.method,
        topology_id: graph.topology_id().as_str().into(),
        coordinate_count: graph.coordinate_count(),
        quantized_weights: candidate_weights.swap_remove(selected_index),
        selected_alpha: selected.alpha,
        quantization_scale: selected.quantization_scale,
        training_samples: training.len(),
        transition_observations,
        observed_coordinates,
    };
    let validation_selection_seconds = selection_started.elapsed().as_secs_f64();
    let timing = TrainingTiming {
        network_and_topology_seconds: topology_seconds,
        training_records_load_seconds: training_load_seconds,
        transition_counting_seconds: counting_seconds,
        validation_selection_seconds,
        total_before_artifact_write_seconds: total_started.elapsed().as_secs_f64(),
    };
    let diagnostics = TrainingDiagnostics {
        schema: TRAINING_DIAGNOSTICS_SCHEMA_V1.into(),
        method: arguments.method.name().into(),
        query_protocol: "fixed_true_first_edge_to_true_last_edge".into(),
        threads: arguments.threads,
        training_samples: training.len(),
        validation_samples: validation.len(),
        coordinate_count: graph.coordinate_count(),
        transition_observations,
        observed_coordinates,
        selected_alpha: artifact.selected_alpha,
        validation_candidates,
        timing,
        peak_rss_kib: peak_rss_kib(),
    };
    write_json(&arguments.artifact, &artifact)?;
    write_json(&arguments.diagnostics, &diagnostics)?;
    Ok(diagnostics)
}

pub fn run_predict(arguments: &PredictArguments) -> Result<PredictionDiagnostics, BaselineError> {
    let total_started = Instant::now();
    let input_started = Instant::now();
    let artifact: StaticArtifact = read_json(&arguments.artifact)?;
    validate_artifact(&artifact)?;
    let network = load_network(&arguments.nodes, &arguments.edges)
        .map_err(|error| failure(format!("network loading failed: {error}")))?;
    let dataset = read_dataset_file(&arguments.dataset_jsonl)?;
    let input_seconds = input_started.elapsed().as_secs_f64();

    let preparation_started = Instant::now();
    let graph = LineGraph::build(&network, 0.1, 10.0)
        .map_err(|error| failure(format!("line-graph construction failed: {error}")))?;
    validate_artifact_against_graph(&artifact, &graph)?;
    let queries = build_queries(&network, &dataset)?;
    let preparation_seconds = preparation_started.elapsed().as_secs_f64();
    let pool = ThreadPoolBuilder::new()
        .num_threads(arguments.threads)
        .build()
        .map_err(|error| {
            failure(format!(
                "failed to build {0}-thread pool: {error}",
                arguments.threads
            ))
        })?;
    let mut oracle = CchOracle::new();
    let mut warmup_seconds = Vec::with_capacity(arguments.warmup_repetitions);
    for _ in 0..arguments.warmup_repetitions {
        let started = Instant::now();
        let paths = pool
            .install(|| {
                oracle.shortest_paths(
                    graph.routing_topology(),
                    &artifact.quantized_weights,
                    &queries,
                )
            })
            .map_err(|error| failure(format!("warm-up routing failed: {error}")))?;
        std::hint::black_box(paths);
        warmup_seconds.push(started.elapsed().as_secs_f64());
    }
    let mut measured_seconds = Vec::with_capacity(arguments.measured_repetitions);
    let mut selected_paths = None;
    let mut deterministic = true;
    for _ in 0..arguments.measured_repetitions {
        let started = Instant::now();
        let paths = pool
            .install(|| {
                oracle.shortest_paths(
                    graph.routing_topology(),
                    &artifact.quantized_weights,
                    &queries,
                )
            })
            .map_err(|error| failure(format!("measured routing failed: {error}")))?;
        measured_seconds.push(started.elapsed().as_secs_f64());
        if let Some(first) = &selected_paths {
            deterministic &= first == &paths;
        } else {
            selected_paths = Some(paths);
        }
    }
    let paths = selected_paths.expect("measured repetitions are positive");
    let predictions = predictions_from_paths(&dataset, &paths);
    let endpoint_mismatches = predictions
        .iter()
        .zip(&dataset)
        .filter(|(prediction, truth)| {
            prediction.predicted_edge_ids.first() != truth.original_edge_ids.first()
                || prediction.predicted_edge_ids.last() != truth.original_edge_ids.last()
        })
        .count();
    write_predictions(&arguments.predictions, &predictions)?;
    let mean_seconds = mean(&measured_seconds);
    let diagnostics = PredictionDiagnostics {
        schema: PREDICTION_DIAGNOSTICS_SCHEMA_V1.into(),
        method: artifact.method.name().into(),
        query_protocol: "fixed_true_first_edge_to_true_last_edge".into(),
        samples: dataset.len(),
        threads: arguments.threads,
        warmup_repetitions: arguments.warmup_repetitions,
        measured_repetitions: arguments.measured_repetitions,
        selected_alpha: artifact.selected_alpha,
        topology_id: artifact.topology_id,
        deterministic_repetitions: deterministic,
        endpoint_mismatches,
        timing: PredictionTiming {
            input_and_network_load_seconds: input_seconds,
            topology_and_query_preparation_seconds: preparation_seconds,
            warmup_metric_and_query_seconds: warmup_seconds,
            measured_metric_and_query_seconds: measured_seconds,
            mean_metric_and_query_seconds: mean_seconds,
            mean_seconds_per_query: mean_seconds / dataset.len() as f64,
            queries_per_second: dataset.len() as f64 / mean_seconds,
            total_before_diagnostics_write_seconds: total_started.elapsed().as_secs_f64(),
            timing_boundary: "CCH metric customization plus 16-thread fixed-batch query and path decode; input, topology construction, and warm-up reported separately".into(),
        },
        peak_rss_kib: peak_rss_kib(),
    };
    write_json(&arguments.diagnostics, &diagnostics)?;
    Ok(diagnostics)
}

fn transition_counts(
    network: &RoadNetwork,
    graph: &LineGraph,
    records: &[DatasetRecordV1],
) -> Result<Vec<u64>, BaselineError> {
    records
        .par_iter()
        .try_fold(
            || vec![0_u64; graph.coordinate_count()],
            |mut counts, record| {
                let trajectory = record_trajectory(network, record)?;
                for pair in trajectory.edges().windows(2) {
                    let coordinate = graph.transition_id(pair[0], pair[1]).ok_or_else(|| {
                        failure(format!(
                            "sample {:?} contains missing transition {:?}->{:?}",
                            record.sample_id, pair[0], pair[1]
                        ))
                    })?;
                    counts[coordinate] = counts[coordinate]
                        .checked_add(1)
                        .ok_or_else(|| failure("transition count overflow"))?;
                }
                Ok(counts)
            },
        )
        .try_reduce(
            || vec![0_u64; graph.coordinate_count()],
            |mut left, right| {
                for (target, value) in left.iter_mut().zip(right) {
                    *target = target
                        .checked_add(value)
                        .ok_or_else(|| failure("transition count reduction overflow"))?;
                }
                Ok(left)
            },
        )
}

fn length_weights(graph: &LineGraph) -> Result<Vec<u32>, BaselineError> {
    graph
        .initial_weights()
        .iter()
        .copied()
        .enumerate()
        .map(|(coordinate, weight)| quantize(weight, coordinate))
        .collect()
}

fn markov_weights(
    graph: &LineGraph,
    counts: &[u64],
    alpha: f64,
) -> Result<Vec<u32>, BaselineError> {
    if !alpha.is_finite() || alpha <= 0.0 {
        return fail(format!(
            "Markov alpha must be positive and finite, got {alpha}"
        ));
    }
    if counts.len() != graph.coordinate_count() {
        return fail("Markov count vector length differs from the line graph");
    }
    let edge_count = graph.routing_topology().node_count();
    let mut totals = vec![0_u64; edge_count];
    let mut degrees = vec![0_usize; edge_count];
    for (transition, &count) in graph.transitions().iter().zip(counts) {
        totals[transition.previous.index()] = totals[transition.previous.index()]
            .checked_add(count)
            .ok_or_else(|| failure("Markov outgoing total overflow"))?;
        degrees[transition.previous.index()] += 1;
    }
    graph
        .transitions()
        .iter()
        .zip(counts)
        .enumerate()
        .map(|(coordinate, (transition, &count))| {
            let previous = transition.previous.index();
            let denominator = totals[previous] as f64 + alpha * degrees[previous] as f64;
            let probability = (count as f64 + alpha) / denominator;
            quantize(-probability.ln() * MARKOV_QUANTIZATION_SCALE, coordinate)
        })
        .collect()
}

fn quantize(weight: f64, coordinate: usize) -> Result<u32, BaselineError> {
    if !weight.is_finite() || weight < 0.0 {
        return fail(format!(
            "weight[{coordinate}] cannot be quantized: {weight}"
        ));
    }
    let rounded = weight.round().max(1.0);
    if rounded >= f64::from(ROUTING_INFINITY) {
        return fail(format!(
            "weight[{coordinate}] exceeds the routing limit: {weight}"
        ));
    }
    Ok(rounded as u32)
}

fn build_queries(
    network: &RoadNetwork,
    records: &[DatasetRecordV1],
) -> Result<Vec<OracleQuery>, BaselineError> {
    records
        .iter()
        .map(|record| {
            let trajectory = record_trajectory(network, record)?;
            Ok(OracleQuery::new(
                vec![QueryEndpoint::zero(trajectory.edges()[0])],
                vec![QueryEndpoint::zero(
                    *trajectory.edges().last().expect("at least two edges"),
                )],
            ))
        })
        .collect()
}

fn record_trajectory(
    network: &RoadNetwork,
    record: &DatasetRecordV1,
) -> Result<Trajectory, BaselineError> {
    let trajectory = Trajectory::new(
        record
            .original_edge_ids
            .iter()
            .copied()
            .map(EdgeId::new)
            .collect(),
    );
    network.validate_trajectory(&trajectory).map_err(|error| {
        failure(format!(
            "sample {:?} is not a valid common trajectory: {error}",
            record.sample_id
        ))
    })?;
    Ok(trajectory)
}

fn predictions_from_paths(
    records: &[DatasetRecordV1],
    paths: &[ewr_core::OraclePath],
) -> Vec<PredictionRecordV1> {
    records
        .iter()
        .zip(paths)
        .map(|(record, path)| PredictionRecordV1 {
            sample_id: record.sample_id.clone(),
            predicted_edge_ids: path
                .nodes()
                .iter()
                .map(|edge| edge.index() as u32)
                .collect(),
        })
        .collect()
}

fn validation_candidate(
    alpha: Option<f64>,
    quantization_scale: f64,
    routing_seconds: f64,
    evaluation: &EvaluationSummaryV1,
) -> ValidationCandidate {
    ValidationCandidate {
        alpha,
        quantization_scale,
        routing_seconds,
        exact_match: evaluation.metrics.exact_match,
        edge_precision: evaluation.metrics.edge_precision,
        edge_recall: evaluation.metrics.edge_recall,
        edge_f1: evaluation.metrics.edge_f1,
        edge_jaccard: evaluation.metrics.edge_jaccard,
    }
}

fn compare_candidates(
    left: &ValidationCandidate,
    right: &ValidationCandidate,
) -> std::cmp::Ordering {
    left.edge_f1
        .total_cmp(&right.edge_f1)
        .then_with(|| left.exact_match.total_cmp(&right.exact_match))
        .then_with(|| {
            right
                .alpha
                .unwrap_or(0.0)
                .total_cmp(&left.alpha.unwrap_or(0.0))
        })
}

fn validate_artifact(artifact: &StaticArtifact) -> Result<(), BaselineError> {
    if artifact.schema != ARTIFACT_SCHEMA_V1 {
        return fail(format!("unsupported artifact schema {:?}", artifact.schema));
    }
    if artifact.coordinate_count == 0
        || artifact.quantized_weights.len() != artifact.coordinate_count
        || artifact
            .quantized_weights
            .iter()
            .any(|&weight| weight == 0 || weight >= ROUTING_INFINITY)
    {
        return fail("artifact has an invalid quantized metric");
    }
    match (artifact.method, artifact.selected_alpha) {
        (Method::Length, None) | (Method::Markov, Some(_)) => Ok(()),
        _ => fail("artifact method and selected alpha are inconsistent"),
    }
}

fn validate_artifact_against_graph(
    artifact: &StaticArtifact,
    graph: &LineGraph,
) -> Result<(), BaselineError> {
    if artifact.topology_id != graph.topology_id().as_str()
        || artifact.coordinate_count != graph.coordinate_count()
        || artifact.quantized_weights.len() != graph.coordinate_count()
    {
        return fail("artifact does not match the rebuilt road-transition topology");
    }
    if artifact.method == Method::Length && artifact.quantized_weights != length_weights(graph)? {
        return fail("length artifact metric differs from the rebuilt road-network lengths");
    }
    Ok(())
}

fn read_dataset_file(path: &Path) -> Result<Vec<DatasetRecordV1>, BaselineError> {
    let file = File::open(path)
        .map_err(|error| failure(format!("failed to open {}: {error}", path.display())))?;
    read_dataset_jsonl(BufReader::new(file))
        .map_err(|error| failure(format!("invalid dataset {}: {error}", path.display())))
}

fn write_predictions(path: &Path, predictions: &[PredictionRecordV1]) -> Result<(), BaselineError> {
    create_parent(path)?;
    let file = File::create(path)
        .map_err(|error| failure(format!("failed to create {}: {error}", path.display())))?;
    let mut writer = BufWriter::new(file);
    write_prediction_jsonl(&mut writer, predictions)
        .map_err(|error| failure(format!("failed to encode predictions: {error}")))?;
    writer
        .flush()
        .map_err(|error| failure(format!("failed to flush {}: {error}", path.display())))
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, BaselineError> {
    let bytes = std::fs::read(path)
        .map_err(|error| failure(format!("failed to read {}: {error}", path.display())))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| failure(format!("failed to decode {}: {error}", path.display())))
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), BaselineError> {
    create_parent(path)?;
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| failure(format!("failed to encode JSON: {error}")))?;
    bytes.push(b'\n');
    std::fs::write(path, bytes)
        .map_err(|error| failure(format!("failed to write {}: {error}", path.display())))
}

fn create_parent(path: &Path) -> Result<(), BaselineError> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).map_err(|error| {
            failure(format!(
                "failed to create directory {}: {error}",
                parent.display()
            ))
        })?;
    }
    Ok(())
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
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

pub fn parse_cli<I, S>(arguments: I) -> Result<CliAction, BaselineError>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let arguments = arguments.into_iter().map(Into::into).collect::<Vec<_>>();
    let Some(command) = arguments.first() else {
        return fail("missing command");
    };
    if command == OsStr::new("--help") || command == OsStr::new("-h") {
        return Ok(CliAction::Help);
    }
    if arguments.len() == 2
        && (arguments[1] == OsStr::new("--help") || arguments[1] == OsStr::new("-h"))
    {
        return Ok(CliAction::Help);
    }
    let values = parse_flag_values(&arguments[1..])?;
    match command.to_str() {
        Some("train") => parse_train(values).map(CliAction::Train),
        Some("predict") => parse_predict(values).map(CliAction::Predict),
        _ => fail(format!("unknown command {command:?}")),
    }
}

fn parse_flag_values(arguments: &[OsString]) -> Result<BTreeMap<String, OsString>, BaselineError> {
    if !arguments.len().is_multiple_of(2) {
        return fail("every option requires a separate value");
    }
    let mut values = BTreeMap::new();
    for pair in arguments.chunks_exact(2) {
        let flag = pair[0]
            .to_str()
            .ok_or_else(|| failure(format!("non-Unicode option {:?}", pair[0])))?;
        if !flag.starts_with("--") {
            return fail(format!("unexpected positional argument {flag:?}"));
        }
        if values.insert(flag.into(), pair[1].clone()).is_some() {
            return fail(format!("duplicate option {flag}"));
        }
    }
    Ok(values)
}

fn parse_train(mut values: BTreeMap<String, OsString>) -> Result<TrainArguments, BaselineError> {
    let method_value = required(&mut values, "--method")?;
    let method = Method::parse(&method_value)?;
    let train_jsonl = values.remove("--train-jsonl").map(PathBuf::from);
    let alpha_candidates = parse_alphas(
        values
            .remove("--alpha-candidates")
            .unwrap_or_else(|| OsString::from("0.01,0.1,1,10")),
    )?;
    let arguments = TrainArguments {
        method,
        nodes: required_path(&mut values, "--nodes")?,
        edges: required_path(&mut values, "--edges")?,
        train_jsonl,
        validation_jsonl: required_path(&mut values, "--validation-jsonl")?,
        artifact: required_path(&mut values, "--artifact")?,
        diagnostics: required_path(&mut values, "--diagnostics")?,
        alpha_candidates,
        threads: optional_usize(&mut values, "--threads", 16)?,
    };
    reject_unknown(values)?;
    if arguments.threads == 0 {
        return fail("threads must be positive");
    }
    Ok(arguments)
}

fn parse_predict(
    mut values: BTreeMap<String, OsString>,
) -> Result<PredictArguments, BaselineError> {
    let arguments = PredictArguments {
        artifact: required_path(&mut values, "--artifact")?,
        nodes: required_path(&mut values, "--nodes")?,
        edges: required_path(&mut values, "--edges")?,
        dataset_jsonl: required_path(&mut values, "--dataset-jsonl")?,
        predictions: required_path(&mut values, "--predictions")?,
        diagnostics: required_path(&mut values, "--diagnostics")?,
        threads: optional_usize(&mut values, "--threads", 16)?,
        warmup_repetitions: optional_usize(&mut values, "--warmup-repetitions", 1)?,
        measured_repetitions: optional_usize(&mut values, "--measured-repetitions", 5)?,
    };
    reject_unknown(values)?;
    if arguments.threads == 0 || arguments.measured_repetitions == 0 {
        return fail("threads and measured repetitions must be positive");
    }
    Ok(arguments)
}

fn parse_alphas(value: OsString) -> Result<Vec<AlphaBits>, BaselineError> {
    let value = value
        .to_str()
        .ok_or_else(|| failure("alpha candidates must be Unicode"))?;
    let candidates = value
        .split(',')
        .map(|item| {
            let alpha = item
                .parse::<f64>()
                .map_err(|_| failure(format!("invalid alpha candidate {item:?}")))?;
            if !alpha.is_finite() || alpha <= 0.0 {
                return fail(format!("alpha candidate must be positive: {alpha}"));
            }
            Ok(AlphaBits(alpha.to_bits()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if candidates.is_empty() {
        return fail("at least one alpha candidate is required");
    }
    Ok(candidates)
}

fn required(
    values: &mut BTreeMap<String, OsString>,
    flag: &str,
) -> Result<OsString, BaselineError> {
    values
        .remove(flag)
        .ok_or_else(|| failure(format!("missing required option {flag}")))
}

fn required_path(
    values: &mut BTreeMap<String, OsString>,
    flag: &str,
) -> Result<PathBuf, BaselineError> {
    required(values, flag).map(PathBuf::from)
}

fn optional_usize(
    values: &mut BTreeMap<String, OsString>,
    flag: &str,
    default: usize,
) -> Result<usize, BaselineError> {
    let Some(value) = values.remove(flag) else {
        return Ok(default);
    };
    value
        .to_str()
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| failure(format!("{flag} must be a nonnegative integer")))
}

fn reject_unknown(values: BTreeMap<String, OsString>) -> Result<(), BaselineError> {
    if let Some(flag) = values.keys().next() {
        fail(format!("unknown option {flag}"))
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BaselineError(String);

fn failure(message: impl Into<String>) -> BaselineError {
    BaselineError(message.into())
}

fn fail<T>(message: impl Into<String>) -> Result<T, BaselineError> {
    Err(failure(message))
}

impl Display for BaselineError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for BaselineError {}

#[cfg(test)]
mod tests {
    use super::*;
    use ewr_core::{NodeId, RoadNetwork};

    fn network() -> RoadNetwork {
        RoadNetwork::new(
            vec![
                NodeId::new(0),
                NodeId::new(1),
                NodeId::new(1),
                NodeId::new(2),
            ],
            vec![
                NodeId::new(1),
                NodeId::new(3),
                NodeId::new(2),
                NodeId::new(3),
            ],
            vec![1.0; 4],
            vec![0.0, 1.0, 2.0, 3.0],
            vec![0.0, 0.0, 1.0, 0.0],
        )
        .unwrap()
    }

    #[test]
    fn markov_cost_prefers_the_more_observed_transition() {
        let graph = LineGraph::build(&network(), 0.1, 10.0).unwrap();
        let mut counts = vec![0; graph.coordinate_count()];
        let preferred = graph.transition_id(EdgeId::new(0), EdgeId::new(1)).unwrap();
        let other = graph.transition_id(EdgeId::new(0), EdgeId::new(2)).unwrap();
        counts[preferred] = 9;
        counts[other] = 1;
        let weights = markov_weights(&graph, &counts, 1.0).unwrap();
        assert!(weights[preferred] < weights[other]);
        assert!(weights.iter().all(|&weight| weight > 0));
    }

    #[test]
    fn query_protocol_fixes_the_true_first_and_last_edges() {
        let network = network();
        let records = vec![DatasetRecordV1 {
            sample_id: "x".into(),
            original_edge_ids: vec![0, 2, 3],
        }];
        let queries = build_queries(&network, &records).unwrap();
        assert_eq!(queries[0].sources()[0].node(), EdgeId::new(0));
        assert_eq!(queries[0].targets()[0].node(), EdgeId::new(3));
    }

    #[test]
    fn length_artifact_is_bound_to_the_rebuilt_length_metric() {
        let source_network = network();
        let source_graph = LineGraph::build(&source_network, 0.1, 10.0).unwrap();
        let artifact = StaticArtifact {
            schema: ARTIFACT_SCHEMA_V1.into(),
            method: Method::Length,
            topology_id: source_graph.topology_id().as_str().into(),
            coordinate_count: source_graph.coordinate_count(),
            quantized_weights: length_weights(&source_graph).unwrap(),
            selected_alpha: None,
            quantization_scale: 1.0,
            training_samples: 0,
            transition_observations: 0,
            observed_coordinates: 0,
        };

        let same_metric_network = RoadNetwork::new(
            source_network.tails().to_vec(),
            source_network.heads().to_vec(),
            vec![1.0; 4],
            vec![10.0, 11.0, 12.0, 13.0],
            vec![7.0, 7.0, 8.0, 7.0],
        )
        .unwrap();
        let same_metric_graph = LineGraph::build(&same_metric_network, 0.1, 10.0).unwrap();
        assert_eq!(source_graph.topology_id(), same_metric_graph.topology_id());
        validate_artifact_against_graph(&artifact, &same_metric_graph).unwrap();

        let changed_metric_network = RoadNetwork::new(
            source_network.tails().to_vec(),
            source_network.heads().to_vec(),
            vec![2.0; 4],
            source_network.x().to_vec(),
            source_network.y().to_vec(),
        )
        .unwrap();
        let changed_metric_graph = LineGraph::build(&changed_metric_network, 0.1, 10.0).unwrap();
        assert_eq!(
            source_graph.topology_id(),
            changed_metric_graph.topology_id()
        );
        let error = validate_artifact_against_graph(&artifact, &changed_metric_graph).unwrap_err();
        assert_eq!(
            error.to_string(),
            "length artifact metric differs from the rebuilt road-network lengths"
        );
    }

    #[test]
    fn cli_defaults_to_sixteen_threads() {
        let CliAction::Predict(arguments) = parse_cli([
            "predict",
            "--artifact",
            "a.json",
            "--nodes",
            "nodes.shp",
            "--edges",
            "edges.shp",
            "--dataset-jsonl",
            "test.jsonl",
            "--predictions",
            "predictions.jsonl",
            "--diagnostics",
            "diagnostics.json",
        ])
        .unwrap() else {
            panic!("expected prediction action")
        };
        assert_eq!(arguments.threads, 16);
        assert_eq!(arguments.warmup_repetitions, 1);
        assert_eq!(arguments.measured_repetitions, 5);
    }

    #[test]
    fn cli_rejects_a_trajectory_carrier_input() {
        let error = parse_cli([
            "predict",
            "--artifact",
            "a.json",
            "--nodes",
            "nodes.shp",
            "--edges",
            "edges.shp",
            "--network-trajectories",
            "test.pkl",
            "--dataset-jsonl",
            "test.jsonl",
            "--predictions",
            "predictions.jsonl",
            "--diagnostics",
            "diagnostics.json",
        ])
        .unwrap_err();
        assert_eq!(error.to_string(), "unknown option --network-trajectories");
    }
}
