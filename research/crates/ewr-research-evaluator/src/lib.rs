//! Method-neutral quality evaluation for research prediction artifacts.
//!
//! The evaluator accepts only the versioned research protocol's complete
//! original-edge sequences. It deliberately has no production, routing, or
//! method-specific dependency.

use ewr_research_protocol::{
    DatasetRecordV1, PredictionRecordV1, ProtocolError, align_predictions, read_dataset_jsonl,
    read_prediction_jsonl,
};
use serde_json::{Value, json};
use std::collections::HashSet;
use std::ffi::{OsStr, OsString};
use std::fmt::{self, Display, Formatter};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const EVALUATION_SUMMARY_SCHEMA_V1: &str = "ewr.evaluation-summary/v1";
pub const USAGE: &str =
    "Usage: ewr-evaluate --dataset-jsonl PATH --predictions-jsonl PATH [--output PATH]";

static TEMPORARY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Per-sample macro averages over complete original-edge sequences.
#[derive(Clone, Debug, PartialEq)]
pub struct RouteMetrics {
    pub exact_match: f64,
    pub edge_precision: f64,
    pub edge_recall: f64,
    pub edge_f1: f64,
    pub edge_jaccard: f64,
}

/// Strict, versioned evaluator output.
#[derive(Clone, Debug, PartialEq)]
pub struct EvaluationSummaryV1 {
    pub schema: String,
    pub sample_count: usize,
    pub metrics: RouteMetrics,
}

impl EvaluationSummaryV1 {
    fn new(sample_count: usize, metrics: RouteMetrics) -> Self {
        Self {
            schema: EVALUATION_SUMMARY_SCHEMA_V1.into(),
            sample_count,
            metrics,
        }
    }

    fn validate(&self) -> Result<(), EvaluationError> {
        if self.schema != EVALUATION_SUMMARY_SCHEMA_V1 {
            return Err(EvaluationError::InvalidSummary(format!(
                "unsupported evaluation summary schema {:?}; expected {:?}",
                self.schema, EVALUATION_SUMMARY_SCHEMA_V1
            )));
        }
        if self.sample_count == 0 {
            return Err(EvaluationError::InvalidSummary(
                "evaluation summary must contain at least one sample".into(),
            ));
        }
        for (name, value) in [
            ("exact_match", self.metrics.exact_match),
            ("edge_precision", self.metrics.edge_precision),
            ("edge_recall", self.metrics.edge_recall),
            ("edge_f1", self.metrics.edge_f1),
            ("edge_jaccard", self.metrics.edge_jaccard),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(EvaluationError::InvalidSummary(format!(
                    "metric {name} must be finite and in [0, 1], got {value}"
                )));
            }
        }
        Ok(())
    }

    /// Convert to the exact version-one JSON shape, with no incidental fields.
    pub fn to_json(&self) -> Result<Value, EvaluationError> {
        self.validate()?;
        Ok(json!({
            "schema": self.schema,
            "sample_count": self.sample_count,
            "metrics": {
                "exact_match": self.metrics.exact_match,
                "edge_precision": self.metrics.edge_precision,
                "edge_recall": self.metrics.edge_recall,
                "edge_f1": self.metrics.edge_f1,
                "edge_jaccard": self.metrics.edge_jaccard,
            },
        }))
    }
}

/// Evaluate already-decoded protocol rows after exact sample-ID alignment.
pub fn evaluate_records(
    dataset: &[DatasetRecordV1],
    predictions: &[PredictionRecordV1],
) -> Result<EvaluationSummaryV1, EvaluationError> {
    let aligned = align_predictions(dataset, predictions).map_err(EvaluationError::Alignment)?;
    let mut totals = MetricTotals::default();
    for sample in &aligned {
        totals.add(
            &sample.dataset.original_edge_ids,
            &sample.prediction.predicted_edge_ids,
        );
    }
    let sample_count = aligned.len();
    let denominator = sample_count as f64;
    Ok(EvaluationSummaryV1::new(
        sample_count,
        RouteMetrics {
            exact_match: totals.exact_match / denominator,
            edge_precision: totals.edge_precision / denominator,
            edge_recall: totals.edge_recall / denominator,
            edge_f1: totals.edge_f1 / denominator,
            edge_jaccard: totals.edge_jaccard / denominator,
        },
    ))
}

/// Read, strictly validate, align, and evaluate a dataset/prediction pair.
pub fn evaluate_files(
    dataset_path: &Path,
    predictions_path: &Path,
) -> Result<EvaluationSummaryV1, EvaluationError> {
    let dataset_file = File::open(dataset_path).map_err(|source| EvaluationError::OpenDataset {
        path: dataset_path.to_path_buf(),
        source,
    })?;
    let dataset = read_dataset_jsonl(BufReader::new(dataset_file))
        .map_err(EvaluationError::InvalidDataset)?;

    let prediction_file =
        File::open(predictions_path).map_err(|source| EvaluationError::OpenPredictions {
            path: predictions_path.to_path_buf(),
            source,
        })?;
    let predictions = read_prediction_jsonl(BufReader::new(prediction_file))
        .map_err(EvaluationError::InvalidPredictions)?;
    evaluate_records(&dataset, &predictions)
}

#[derive(Default)]
struct MetricTotals {
    exact_match: f64,
    edge_precision: f64,
    edge_recall: f64,
    edge_f1: f64,
    edge_jaccard: f64,
}

impl MetricTotals {
    fn add(&mut self, truth: &[u32], prediction: &[u32]) {
        let truth_set = truth.iter().copied().collect::<HashSet<_>>();
        let prediction_set = prediction.iter().copied().collect::<HashSet<_>>();
        let intersection = truth_set.intersection(&prediction_set).count() as f64;
        let precision = intersection / prediction_set.len() as f64;
        let recall = intersection / truth_set.len() as f64;
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        let union = truth_set.union(&prediction_set).count() as f64;

        self.exact_match += f64::from(truth == prediction);
        self.edge_precision += precision;
        self.edge_recall += recall;
        self.edge_f1 += f1;
        self.edge_jaccard += intersection / union;
    }
}

/// Encode the strict summary as pretty JSON followed by one newline.
pub fn encode_summary(summary: &EvaluationSummaryV1) -> Result<Vec<u8>, EvaluationError> {
    let mut encoded =
        serde_json::to_vec_pretty(&summary.to_json()?).map_err(EvaluationError::EncodeSummary)?;
    encoded.push(b'\n');
    Ok(encoded)
}

/// Write the strict summary to a stream, normally stdout.
pub fn write_summary(
    mut writer: impl Write,
    summary: &EvaluationSummaryV1,
) -> Result<(), EvaluationError> {
    writer
        .write_all(&encode_summary(summary)?)
        .map_err(EvaluationError::WriteSummary)
}

/// Atomically replace an optional summary artifact in its destination directory.
pub fn write_summary_atomic(
    path: &Path,
    summary: &EvaluationSummaryV1,
) -> Result<(), EvaluationError> {
    let bytes = encode_summary(summary)?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty());
    if let Some(parent) = parent {
        std::fs::create_dir_all(parent).map_err(|source| {
            EvaluationError::CreateOutputDirectory {
                path: parent.to_path_buf(),
                source,
            }
        })?;
    }

    let mut temporary = TemporaryOutput::create_for(path)?;
    temporary
        .file
        .write_all(&bytes)
        .map_err(|source| EvaluationError::WriteTemporary {
            path: temporary.path.clone(),
            source,
        })?;
    temporary
        .file
        .sync_all()
        .map_err(|source| EvaluationError::SyncTemporary {
            path: temporary.path.clone(),
            source,
        })?;
    std::fs::rename(&temporary.path, path).map_err(|source| EvaluationError::ReplaceOutput {
        temporary: temporary.path.clone(),
        destination: path.to_path_buf(),
        source,
    })?;
    temporary.committed = true;
    Ok(())
}

fn temporary_path(destination: &Path) -> Result<PathBuf, EvaluationError> {
    let filename = destination
        .file_name()
        .ok_or_else(|| EvaluationError::InvalidOutputPath(destination.to_path_buf()))?;
    let mut temporary_name = OsString::from(".");
    temporary_name.push(filename);
    temporary_name.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    Ok(destination.with_file_name(temporary_name))
}

struct TemporaryOutput {
    path: PathBuf,
    file: File,
    committed: bool,
}

impl TemporaryOutput {
    fn create_for(destination: &Path) -> Result<Self, EvaluationError> {
        for _ in 0..100 {
            let path = temporary_path(destination)?;
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => {
                    return Ok(Self {
                        path,
                        file,
                        committed: false,
                    });
                }
                Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(source) => return Err(EvaluationError::CreateTemporary { path, source }),
            }
        }
        Err(EvaluationError::TemporaryNameExhausted(
            destination.to_path_buf(),
        ))
    }
}

impl Drop for TemporaryOutput {
    fn drop(&mut self) {
        if !self.committed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Successfully parsed command-line action.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CliAction {
    Run(CliArguments),
    Help,
}

/// File arguments accepted by `ewr-evaluate`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CliArguments {
    pub dataset_jsonl: PathBuf,
    pub predictions_jsonl: PathBuf,
    pub output: Option<PathBuf>,
}

/// Strictly parse arguments after the program name.
pub fn parse_cli<I, S>(arguments: I) -> Result<CliAction, CliError>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let arguments = arguments.into_iter().map(Into::into).collect::<Vec<_>>();
    if arguments
        .iter()
        .any(|argument| argument == OsStr::new("--help") || argument == OsStr::new("-h"))
    {
        return Ok(CliAction::Help);
    }

    let mut dataset_jsonl = None;
    let mut predictions_jsonl = None;
    let mut output = None;
    let mut index = 0;
    while index < arguments.len() {
        let flag = arguments[index]
            .to_str()
            .ok_or_else(|| CliError::NonUnicodeFlag(arguments[index].clone()))?;
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| CliError::MissingValue(flag.into()))?;
        let slot = match flag {
            "--dataset-jsonl" => &mut dataset_jsonl,
            "--predictions-jsonl" => &mut predictions_jsonl,
            "--output" => &mut output,
            _ => return Err(CliError::UnknownArgument(flag.into())),
        };
        if slot.replace(PathBuf::from(value)).is_some() {
            return Err(CliError::DuplicateArgument(flag.into()));
        }
        index += 2;
    }

    Ok(CliAction::Run(CliArguments {
        dataset_jsonl: dataset_jsonl.ok_or(CliError::MissingRequired("--dataset-jsonl"))?,
        predictions_jsonl: predictions_jsonl
            .ok_or(CliError::MissingRequired("--predictions-jsonl"))?,
        output,
    }))
}

/// A typed file, protocol, alignment, encoding, or output failure.
#[derive(Debug)]
pub enum EvaluationError {
    OpenDataset {
        path: PathBuf,
        source: std::io::Error,
    },
    InvalidDataset(ProtocolError),
    OpenPredictions {
        path: PathBuf,
        source: std::io::Error,
    },
    InvalidPredictions(ProtocolError),
    Alignment(ProtocolError),
    InvalidSummary(String),
    EncodeSummary(serde_json::Error),
    WriteSummary(std::io::Error),
    InvalidOutputPath(PathBuf),
    CreateOutputDirectory {
        path: PathBuf,
        source: std::io::Error,
    },
    CreateTemporary {
        path: PathBuf,
        source: std::io::Error,
    },
    TemporaryNameExhausted(PathBuf),
    WriteTemporary {
        path: PathBuf,
        source: std::io::Error,
    },
    SyncTemporary {
        path: PathBuf,
        source: std::io::Error,
    },
    ReplaceOutput {
        temporary: PathBuf,
        destination: PathBuf,
        source: std::io::Error,
    },
}

impl Display for EvaluationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::OpenDataset { path, source } => {
                write!(
                    formatter,
                    "failed to open dataset {}: {source}",
                    path.display()
                )
            }
            Self::InvalidDataset(source) => write!(formatter, "invalid dataset JSONL: {source}"),
            Self::OpenPredictions { path, source } => write!(
                formatter,
                "failed to open predictions {}: {source}",
                path.display()
            ),
            Self::InvalidPredictions(source) => {
                write!(formatter, "invalid predictions JSONL: {source}")
            }
            Self::Alignment(source) => write!(formatter, "prediction alignment failed: {source}"),
            Self::InvalidSummary(message) => write!(formatter, "invalid summary: {message}"),
            Self::EncodeSummary(source) => write!(formatter, "failed to encode summary: {source}"),
            Self::WriteSummary(source) => write!(formatter, "failed to write summary: {source}"),
            Self::InvalidOutputPath(path) => {
                write!(
                    formatter,
                    "output path has no file name: {}",
                    path.display()
                )
            }
            Self::CreateOutputDirectory { path, source } => write!(
                formatter,
                "failed to create output directory {}: {source}",
                path.display()
            ),
            Self::CreateTemporary { path, source } => write!(
                formatter,
                "failed to create temporary output {}: {source}",
                path.display()
            ),
            Self::TemporaryNameExhausted(path) => write!(
                formatter,
                "failed to allocate a temporary name for {}",
                path.display()
            ),
            Self::WriteTemporary { path, source } => write!(
                formatter,
                "failed to write temporary output {}: {source}",
                path.display()
            ),
            Self::SyncTemporary { path, source } => write!(
                formatter,
                "failed to sync temporary output {}: {source}",
                path.display()
            ),
            Self::ReplaceOutput {
                temporary,
                destination,
                source,
            } => write!(
                formatter,
                "failed to replace {} with {}: {source}",
                destination.display(),
                temporary.display()
            ),
        }
    }
}

impl std::error::Error for EvaluationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::OpenDataset { source, .. }
            | Self::OpenPredictions { source, .. }
            | Self::WriteSummary(source)
            | Self::CreateOutputDirectory { source, .. }
            | Self::CreateTemporary { source, .. }
            | Self::WriteTemporary { source, .. }
            | Self::SyncTemporary { source, .. }
            | Self::ReplaceOutput { source, .. } => Some(source),
            Self::InvalidDataset(source)
            | Self::InvalidPredictions(source)
            | Self::Alignment(source) => Some(source),
            Self::EncodeSummary(source) => Some(source),
            Self::InvalidSummary(_)
            | Self::InvalidOutputPath(_)
            | Self::TemporaryNameExhausted(_) => None,
        }
    }
}

/// A typed command-line syntax failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CliError {
    NonUnicodeFlag(OsString),
    UnknownArgument(String),
    MissingValue(String),
    DuplicateArgument(String),
    MissingRequired(&'static str),
}

impl Display for CliError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonUnicodeFlag(flag) => write!(formatter, "argument {flag:?} is not Unicode"),
            Self::UnknownArgument(argument) => write!(formatter, "unknown argument {argument}"),
            Self::MissingValue(argument) => write!(formatter, "missing value for {argument}"),
            Self::DuplicateArgument(argument) => {
                write!(formatter, "{argument} was provided more than once")
            }
            Self::MissingRequired(argument) => write!(formatter, "missing required {argument}"),
        }
    }
}

impl std::error::Error for CliError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_DIRECTORY_ID: AtomicU64 = AtomicU64::new(0);

    fn dataset() -> Vec<DatasetRecordV1> {
        vec![
            DatasetRecordV1 {
                sample_id: "test:1".into(),
                original_edge_ids: vec![1, 2, 3],
            },
            DatasetRecordV1 {
                sample_id: "test:2".into(),
                original_edge_ids: vec![7, 8],
            },
        ]
    }

    fn predictions() -> Vec<PredictionRecordV1> {
        vec![
            PredictionRecordV1 {
                sample_id: "test:2".into(),
                predicted_edge_ids: vec![7, 8],
            },
            PredictionRecordV1 {
                sample_id: "test:1".into(),
                predicted_edge_ids: vec![1, 2, 4, 5],
            },
        ]
    }

    #[test]
    fn synthetic_macro_metrics_are_golden() {
        let summary = evaluate_records(&dataset(), &predictions()).unwrap();
        assert_eq!(summary.schema, EVALUATION_SUMMARY_SCHEMA_V1);
        assert_eq!(summary.sample_count, 2);
        assert_eq!(summary.metrics.exact_match, 0.5);
        assert!((summary.metrics.edge_precision - 0.75).abs() < 1e-12);
        assert!((summary.metrics.edge_recall - (5.0 / 6.0)).abs() < 1e-12);
        assert!((summary.metrics.edge_f1 - (11.0 / 14.0)).abs() < 1e-12);
        assert!((summary.metrics.edge_jaccard - 0.7).abs() < 1e-12);
    }

    #[test]
    fn exact_match_uses_sequences_while_overlap_uses_sets() {
        let dataset = vec![DatasetRecordV1 {
            sample_id: "x".into(),
            original_edge_ids: vec![1, 1, 2],
        }];
        let predictions = vec![PredictionRecordV1 {
            sample_id: "x".into(),
            predicted_edge_ids: vec![2, 1],
        }];
        let summary = evaluate_records(&dataset, &predictions).unwrap();
        assert_eq!(summary.metrics.exact_match, 0.0);
        assert_eq!(summary.metrics.edge_precision, 1.0);
        assert_eq!(summary.metrics.edge_recall, 1.0);
        assert_eq!(summary.metrics.edge_f1, 1.0);
        assert_eq!(summary.metrics.edge_jaccard, 1.0);
    }

    #[test]
    fn protocol_alignment_rejects_missing_and_extra_predictions() {
        let mut predictions = predictions();
        predictions.pop();
        assert!(matches!(
            evaluate_records(&dataset(), &predictions),
            Err(EvaluationError::Alignment(_))
        ));

        predictions.push(PredictionRecordV1 {
            sample_id: "extra".into(),
            predicted_edge_ids: vec![9],
        });
        assert!(matches!(
            evaluate_records(&dataset(), &predictions),
            Err(EvaluationError::Alignment(_))
        ));
    }

    #[test]
    fn encoded_summary_has_only_the_versioned_shape() {
        let summary = evaluate_records(&dataset(), &predictions()).unwrap();
        let encoded = encode_summary(&summary).unwrap();
        assert!(encoded.ends_with(b"\n"));
        let value: Value = serde_json::from_slice(&encoded).unwrap();
        let object = value.as_object().unwrap();
        assert_eq!(object.len(), 3);
        assert_eq!(value["schema"], EVALUATION_SUMMARY_SCHEMA_V1);
        assert_eq!(value["sample_count"], 2);
        assert_eq!(value["metrics"].as_object().unwrap().len(), 5);
    }

    #[test]
    fn optional_output_is_written_atomically_and_can_be_replaced() {
        let directory = test_directory();
        std::fs::create_dir_all(&directory).unwrap();
        let output = directory.join("nested/summary.json");
        let summary = evaluate_records(&dataset(), &predictions()).unwrap();
        write_summary_atomic(&output, &summary).unwrap();
        write_summary_atomic(&output, &summary).unwrap();
        assert_eq!(
            std::fs::read(&output).unwrap(),
            encode_summary(&summary).unwrap()
        );
        assert_eq!(
            std::fs::read_dir(output.parent().unwrap()).unwrap().count(),
            1
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn cli_parser_accepts_optional_output_and_help() {
        assert_eq!(
            parse_cli([
                "--dataset-jsonl",
                "dataset.jsonl",
                "--predictions-jsonl",
                "predictions.jsonl",
                "--output",
                "summary.json",
            ])
            .unwrap(),
            CliAction::Run(CliArguments {
                dataset_jsonl: "dataset.jsonl".into(),
                predictions_jsonl: "predictions.jsonl".into(),
                output: Some("summary.json".into()),
            })
        );
        assert_eq!(parse_cli(["--help"]).unwrap(), CliAction::Help);
    }

    #[test]
    fn cli_parser_rejects_missing_duplicate_and_unknown_arguments() {
        assert!(matches!(
            parse_cli(["--dataset-jsonl", "dataset.jsonl"]),
            Err(CliError::MissingRequired("--predictions-jsonl"))
        ));
        assert!(matches!(
            parse_cli([
                "--dataset-jsonl",
                "one.jsonl",
                "--dataset-jsonl",
                "two.jsonl",
                "--predictions-jsonl",
                "predictions.jsonl",
            ]),
            Err(CliError::DuplicateArgument(_))
        ));
        assert!(matches!(
            parse_cli([
                "--dataset-jsonl",
                "dataset.jsonl",
                "--predictions-jsonl",
                "predictions.jsonl",
                "--method",
                "forbidden",
            ]),
            Err(CliError::UnknownArgument(_))
        ));
    }

    fn test_directory() -> PathBuf {
        let id = TEST_DIRECTORY_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "ewr-research-evaluator-test-{}-{id}",
            std::process::id()
        ))
    }
}
