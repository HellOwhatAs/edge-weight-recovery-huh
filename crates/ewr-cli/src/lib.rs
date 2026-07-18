//! Thin composition root for production edge-weight training.
//!
//! This crate deliberately owns only process concerns: strict command-line
//! parsing, loading typed production inputs, selecting the production CCH
//! adapter, constraining Rayon to an explicit pool, and saving artifacts.

use ewr_cch::CchOracle;
use ewr_core::{Trainer, TrainerError, TrainingOutcome, TrainingState};
use ewr_io::{
    ArtifactError, ConfigError, DatasetError, LoadedDataset, TrainConfig, load_dataset,
    load_training_artifact, save_training_artifact,
};
use rayon::{ThreadPoolBuildError, ThreadPoolBuilder};
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};

/// Exact production command surface.
pub const USAGE: &str = "Usage: ewr train --config PATH --output-dir DIR [--resume ARTIFACT]\n";

/// Filename of the atomically published model and exact resume state.
pub const TRAINING_ARTIFACT_FILENAME: &str = "training-artifact.json";

/// One action accepted by the strict argument parser.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CliAction {
    /// Print [`USAGE`] and exit successfully.
    Help,
    /// Run the sole production training command.
    Train(TrainArgs),
}

/// Paths supplied to `ewr train`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TrainArgs {
    /// Strict production training configuration.
    pub config: PathBuf,
    /// Directory that receives the single stable training artifact.
    pub output_dir: PathBuf,
    /// Optional complete artifact from which to continue.
    pub resume: Option<PathBuf>,
}

/// Parse arguments after the executable name.
///
/// Only separate flag values are accepted; positional values, duplicate
/// options, and unknown options are errors.
pub fn parse_args<I, S>(args: I) -> Result<CliAction, ParseError>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let mut args = args.into_iter().map(Into::into);
    let Some(command) = args.next() else {
        return Err(ParseError::MissingCommand);
    };
    if command == OsStr::new("--help") || command == OsStr::new("-h") {
        if let Some(argument) = args.next() {
            return Err(ParseError::UnexpectedArgument(argument));
        }
        return Ok(CliAction::Help);
    }
    if command != OsStr::new("train") {
        return Err(ParseError::UnknownCommand(command));
    }

    let mut config = None;
    let mut output_dir = None;
    let mut resume = None;
    while let Some(flag) = args.next() {
        if flag == OsStr::new("--help") || flag == OsStr::new("-h") {
            if config.is_some() || output_dir.is_some() || resume.is_some() {
                return Err(ParseError::UnexpectedArgument(flag));
            }
            if let Some(argument) = args.next() {
                return Err(ParseError::UnexpectedArgument(argument));
            }
            return Ok(CliAction::Help);
        }
        let slot = if flag == OsStr::new("--config") {
            &mut config
        } else if flag == OsStr::new("--output-dir") {
            &mut output_dir
        } else if flag == OsStr::new("--resume") {
            &mut resume
        } else {
            return Err(ParseError::UnknownOption(flag));
        };
        if slot.is_some() {
            return Err(ParseError::DuplicateOption(flag));
        }
        let Some(value) = args.next() else {
            return Err(ParseError::MissingValue(flag));
        };
        if value.is_empty() || looks_like_option(&value) {
            return Err(ParseError::InvalidValue {
                option: flag,
                value,
            });
        }
        *slot = Some(PathBuf::from(value));
    }

    Ok(CliAction::Train(TrainArgs {
        config: config.ok_or(ParseError::MissingOption("--config"))?,
        output_dir: output_dir.ok_or(ParseError::MissingOption("--output-dir"))?,
        resume,
    }))
}

fn looks_like_option(value: &OsStr) -> bool {
    value.as_encoded_bytes().starts_with(b"-")
}

/// Load the on-disk inputs and execute one production training command.
pub fn run_train(args: &TrainArgs) -> Result<TrainSummary, CliError> {
    let config = TrainConfig::load(&args.config).map_err(CliError::Config)?;
    let dataset = load_dataset(&config.dataset).map_err(CliError::Dataset)?;
    let resume = args
        .resume
        .as_ref()
        .map(load_training_artifact)
        .transpose()
        .map_err(CliError::LoadResumeArtifact)?;
    train_loaded(
        &config,
        &dataset,
        resume.as_ref().map(|artifact| &artifact.state),
        args.output_dir.as_path(),
    )
}

/// Train already-loaded production values and publish one stable artifact.
///
/// The explicit pool makes `runtime.threads` authoritative for all parallel
/// CCH queries made during this run. The snapshot cadence is operational: core
/// owns the loop and exposes immutable model/state pairs, while `ewr-io`
/// atomically replaces one complete artifact.
pub fn train_loaded(
    config: &TrainConfig,
    dataset: &LoadedDataset,
    resume: Option<&TrainingState>,
    output_dir: &Path,
) -> Result<TrainSummary, CliError> {
    if config.threads == 0 {
        return Err(CliError::Config(ConfigError::InvalidThreads(0)));
    }
    if config.checkpoint_every == 0 || config.checkpoint_every > config.fit.updates {
        return Err(CliError::Config(ConfigError::InvalidCheckpointEvery {
            cadence: config.checkpoint_every,
            updates: config.fit.updates,
        }));
    }
    let artifact_path = output_dir.join(TRAINING_ARTIFACT_FILENAME);
    let pool = ThreadPoolBuilder::new()
        .num_threads(config.threads)
        .build()
        .map_err(CliError::ThreadPool)?;
    let outcome = pool
        .install(|| {
            let mut oracle = CchOracle::new();
            let mut trainer = Trainer::new(
                &dataset.network,
                &dataset.trajectories,
                &config.fit,
                &mut oracle,
            )?;
            let mut save_snapshot =
                |snapshot: &TrainingOutcome| save_training_artifact(&artifact_path, snapshot);
            match resume {
                Some(state) => trainer.resume_with_snapshots(
                    state,
                    config.checkpoint_every,
                    &mut save_snapshot,
                ),
                None => trainer.fit_with_snapshots(config.checkpoint_every, &mut save_snapshot),
            }
        })
        .map_err(CliError::Trainer)?;
    let restored =
        load_training_artifact(&artifact_path).map_err(CliError::VerifyTrainingArtifact)?;
    if restored != outcome {
        return Err(CliError::TrainingArtifactRoundTripMismatch);
    }

    Ok(TrainSummary {
        accepted: dataset.report.accepted,
        dropped: dataset.report.dropped(),
        coordinates: outcome.result.model.weights().len(),
        completed_updates: outcome.result.diagnostics.completed_updates,
        objective: outcome.result.diagnostics.objective,
    })
}

/// Concise, experiment-independent outcome printed by the executable.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainSummary {
    /// Structurally valid observations supplied to the trainer.
    pub accepted: usize,
    /// Raw observations rejected while loading.
    pub dropped: usize,
    /// Number of stable transition-weight coordinates.
    pub coordinates: usize,
    /// Global optimizer clock in the saved state.
    pub completed_updates: u64,
    /// Final full-batch training objective.
    pub objective: f64,
}

impl Display for TrainSummary {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "accepted={} dropped={} coordinates={} completed_updates={} objective={:.6}",
            self.accepted, self.dropped, self.coordinates, self.completed_updates, self.objective
        )
    }
}

/// Invalid command-line structure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ParseError {
    /// No subcommand was supplied.
    MissingCommand,
    /// The subcommand is not part of the production surface.
    UnknownCommand(OsString),
    /// A flag is not accepted by `ewr train`.
    UnknownOption(OsString),
    /// An option occurred more than once.
    DuplicateOption(OsString),
    /// A required option was omitted.
    MissingOption(&'static str),
    /// An option was not followed by a value.
    MissingValue(OsString),
    /// A value was empty or looked like another option.
    InvalidValue { option: OsString, value: OsString },
    /// Extra input followed a complete help request.
    UnexpectedArgument(OsString),
}

impl Display for ParseError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingCommand => formatter.write_str("missing command"),
            Self::UnknownCommand(command) => {
                write!(formatter, "unknown command {:?}", command)
            }
            Self::UnknownOption(option) => write!(formatter, "unknown option {:?}", option),
            Self::DuplicateOption(option) => write!(formatter, "duplicate option {:?}", option),
            Self::MissingOption(option) => write!(formatter, "missing required option {option}"),
            Self::MissingValue(option) => write!(formatter, "missing value for {:?}", option),
            Self::InvalidValue { option, value } => {
                write!(formatter, "invalid value {:?} for {:?}", value, option)
            }
            Self::UnexpectedArgument(argument) => {
                write!(formatter, "unexpected argument {:?}", argument)
            }
        }
    }
}

impl Error for ParseError {}

/// Failure while composing production adapters around the core trainer.
#[derive(Debug)]
pub enum CliError {
    /// Strict config loading failed.
    Config(ConfigError),
    /// Dataset adaptation failed.
    Dataset(DatasetError),
    /// Optional resume-artifact loading failed.
    LoadResumeArtifact(ArtifactError),
    /// The requested fixed-size pool could not be created.
    ThreadPool(ThreadPoolBuildError),
    /// Core validation or fitting failed.
    Trainer(TrainerError),
    /// The final published artifact could not be read back.
    VerifyTrainingArtifact(ArtifactError),
    /// The atomically restored artifact differed from the returned outcome.
    TrainingArtifactRoundTripMismatch,
}

impl Display for CliError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Config(error) => write!(formatter, "configuration failed: {error}"),
            Self::Dataset(error) => write!(formatter, "dataset loading failed: {error}"),
            Self::LoadResumeArtifact(error) => {
                write!(formatter, "resume-artifact loading failed: {error}")
            }
            Self::ThreadPool(error) => write!(formatter, "thread-pool creation failed: {error}"),
            Self::Trainer(error) => write!(formatter, "training failed: {error}"),
            Self::VerifyTrainingArtifact(error) => {
                write!(
                    formatter,
                    "final training-artifact verification failed: {error}"
                )
            }
            Self::TrainingArtifactRoundTripMismatch => {
                formatter.write_str("restored training artifact differs from the completed outcome")
            }
        }
    }
}

impl Error for CliError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Config(error) => Some(error),
            Self::Dataset(error) => Some(error),
            Self::LoadResumeArtifact(error) | Self::VerifyTrainingArtifact(error) => Some(error),
            Self::ThreadPool(error) => Some(error),
            Self::Trainer(error) => Some(error),
            Self::TrainingArtifactRoundTripMismatch => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ewr_core::{EdgeId, FitOptions, NodeId, RoadNetwork, Trajectory};
    use ewr_io::{DatasetPaths, LoadReport, load_training_artifact};

    struct TemporaryDirectory(PathBuf);

    impl TemporaryDirectory {
        fn new(label: &str) -> Self {
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir()
                .join(format!("ewr-cli-{label}-{}-{nonce}", std::process::id()));
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
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

    fn parse(values: &[&str]) -> Result<CliAction, ParseError> {
        parse_args(values.iter().copied())
    }

    #[test]
    fn parser_accepts_only_the_train_contract() {
        assert_eq!(
            parse(&[
                "train",
                "--config",
                "train.json",
                "--output-dir",
                "artifacts",
                "--resume",
                "old-artifact.json",
            ]),
            Ok(CliAction::Train(TrainArgs {
                config: PathBuf::from("train.json"),
                output_dir: PathBuf::from("artifacts"),
                resume: Some(PathBuf::from("old-artifact.json")),
            }))
        );
        assert_eq!(parse(&["--help"]), Ok(CliAction::Help));
        assert_eq!(parse(&["train", "--help"]), Ok(CliAction::Help));
    }

    #[test]
    fn parser_rejects_experiment_switches_duplicates_and_missing_values() {
        assert!(matches!(
            parse(&[
                "train",
                "--config",
                "train.json",
                "--output-dir",
                "out",
                "--oracle",
                "dijkstra",
            ]),
            Err(ParseError::UnknownOption(_))
        ));
        assert!(matches!(
            parse(&[
                "train",
                "--config",
                "one.json",
                "--config",
                "two.json",
                "--output-dir",
                "out",
            ]),
            Err(ParseError::DuplicateOption(_))
        ));
        assert!(matches!(
            parse(&["train", "--config", "--output-dir", "out"]),
            Err(ParseError::InvalidValue { .. })
        ));
        assert_eq!(
            parse(&["train", "--config", "train.json"]),
            Err(ParseError::MissingOption("--output-dir"))
        );
    }

    fn synthetic_config(updates: u64) -> TrainConfig {
        TrainConfig {
            dataset: DatasetPaths {
                nodes: PathBuf::from("unused-nodes.shp"),
                edges: PathBuf::from("unused-edges.shp"),
                trajectories: PathBuf::from("unused-trajectories.pkl"),
            },
            fit: FitOptions {
                eta0: 0.5,
                lambda: 0.1,
                lower_factor: 0.1,
                upper_factor: 10.0,
                updates,
            },
            threads: 2,
            checkpoint_every: 1,
        }
    }

    fn synthetic_dataset() -> LoadedDataset {
        let network = RoadNetwork::new(
            vec![
                NodeId::new(0),
                NodeId::new(1),
                NodeId::new(0),
                NodeId::new(2),
            ],
            vec![
                NodeId::new(1),
                NodeId::new(3),
                NodeId::new(2),
                NodeId::new(3),
            ],
            vec![5.0, 5.0, 2.0, 2.0],
            vec![0.0, 1.0, 1.0, 2.0],
            vec![0.0, 0.0, 1.0, 0.0],
        )
        .unwrap();
        LoadedDataset {
            network,
            trajectories: vec![
                Trajectory::new(vec![EdgeId::new(0), EdgeId::new(1)]),
                Trajectory::new(vec![EdgeId::new(0), EdgeId::new(1)]),
            ],
            report: LoadReport {
                available: 3,
                accepted: 2,
                too_short: 1,
                ..LoadReport::default()
            },
        }
    }

    #[test]
    fn in_memory_training_uses_cch_and_round_trips_one_atomic_artifact() {
        let directory = TemporaryDirectory::new("train-loaded");
        let dataset = synthetic_dataset();
        let summary = train_loaded(&synthetic_config(1), &dataset, None, directory.path()).unwrap();

        let artifact =
            load_training_artifact(directory.path().join(TRAINING_ARTIFACT_FILENAME)).unwrap();
        assert_eq!(summary.accepted, 2);
        assert_eq!(summary.dropped, 1);
        assert_eq!(summary.coordinates, 2);
        assert_eq!(summary.completed_updates, 1);
        assert!(summary.objective.is_finite());
        assert_eq!(artifact.result.model.weights(), artifact.state.weights());
        assert_eq!(
            artifact.result.model.topology_id(),
            artifact.state.topology_id()
        );
        assert_eq!(artifact.state.completed_updates(), 1);
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);

        let resumed = train_loaded(
            &synthetic_config(2),
            &dataset,
            Some(&artifact.state),
            directory.path(),
        )
        .unwrap();
        let resumed_artifact =
            load_training_artifact(directory.path().join(TRAINING_ARTIFACT_FILENAME)).unwrap();
        assert_eq!(resumed.completed_updates, 2);
        assert_eq!(
            resumed_artifact.result.model.weights(),
            resumed_artifact.state.weights()
        );
        assert_eq!(resumed_artifact.state.completed_updates(), 2);
    }

    #[test]
    fn in_memory_composition_still_rejects_an_invalid_pool_size() {
        let directory = TemporaryDirectory::new("zero-threads");
        let mut config = synthetic_config(1);
        config.threads = 0;
        let error =
            train_loaded(&config, &synthetic_dataset(), None, directory.path()).unwrap_err();
        assert!(matches!(
            error,
            CliError::Config(ConfigError::InvalidThreads(0))
        ));
    }
}
