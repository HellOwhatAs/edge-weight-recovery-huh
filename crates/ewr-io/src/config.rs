use ewr_core::FitOptions;
use serde::Deserialize;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};

/// Explicit input files for one production training run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DatasetPaths {
    /// Road-network node Shapefile.
    pub nodes: PathBuf,
    /// Road-network edge Shapefile.
    pub edges: PathBuf,
    /// Pickle containing complete original-edge trajectories.
    pub trajectories: PathBuf,
}

/// Minimal production training configuration.
///
/// Experiment identity, dataset naming conventions, validation policy, and
/// algorithm/backend selection intentionally do not belong to this type.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainConfig {
    /// Explicit production input paths.
    pub dataset: DatasetPaths,
    /// Options of the sole production fitting algorithm.
    pub fit: FitOptions,
    /// Number of worker threads requested by the composition root.
    pub threads: usize,
    /// Number of successful optimizer updates between published snapshots.
    pub checkpoint_every: u64,
}

impl TrainConfig {
    /// Current on-disk configuration schema.
    pub const SCHEMA_VERSION: u32 = 1;

    /// Load and strictly validate a JSON configuration file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_json_slice(&bytes).map_err(|error| error.with_path(path))
    }

    /// Strictly decode a configuration from JSON bytes.
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, ConfigError> {
        let raw: RawTrainConfig = serde_json::from_slice(bytes)
            .map_err(|source| ConfigError::Decode { path: None, source })?;

        if raw.schema_version != Self::SCHEMA_VERSION {
            return Err(ConfigError::UnsupportedSchema {
                expected: Self::SCHEMA_VERSION,
                actual: raw.schema_version,
            });
        }
        for (field, path) in [
            ("dataset.nodes", &raw.dataset.nodes),
            ("dataset.edges", &raw.dataset.edges),
            ("dataset.trajectories", &raw.dataset.trajectories),
        ] {
            if path.as_os_str().is_empty() {
                return Err(ConfigError::EmptyPath(field));
            }
        }
        if raw.runtime.threads == 0 {
            return Err(ConfigError::InvalidThreads(raw.runtime.threads));
        }

        let fit = FitOptions {
            eta0: raw.fit.eta0,
            lambda: raw.fit.lambda,
            lower_factor: raw.fit.lower_factor,
            upper_factor: raw.fit.upper_factor,
            updates: raw.fit.updates,
        }
        .validate()
        .map_err(|error| ConfigError::InvalidFitOptions {
            message: error.to_string(),
        })?;
        if raw.runtime.checkpoint_every == 0 || raw.runtime.checkpoint_every > fit.updates {
            return Err(ConfigError::InvalidCheckpointEvery {
                cadence: raw.runtime.checkpoint_every,
                updates: fit.updates,
            });
        }

        Ok(Self {
            dataset: DatasetPaths {
                nodes: raw.dataset.nodes,
                edges: raw.dataset.edges,
                trajectories: raw.dataset.trajectories,
            },
            fit,
            threads: raw.runtime.threads,
            checkpoint_every: raw.runtime.checkpoint_every,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTrainConfig {
    schema_version: u32,
    dataset: RawDatasetPaths,
    fit: RawFitOptions,
    runtime: RawRuntime,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDatasetPaths {
    nodes: PathBuf,
    edges: PathBuf,
    trajectories: PathBuf,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFitOptions {
    eta0: f64,
    lambda: f64,
    lower_factor: f64,
    upper_factor: f64,
    updates: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRuntime {
    threads: usize,
    checkpoint_every: u64,
}

/// Failure to read or validate a production training configuration.
#[derive(Debug)]
pub enum ConfigError {
    /// The configuration file could not be read.
    Read {
        /// Input path.
        path: PathBuf,
        /// Underlying filesystem error.
        source: std::io::Error,
    },
    /// JSON syntax or strict schema decoding failed.
    Decode {
        /// Input path when decoding a file.
        path: Option<PathBuf>,
        /// Underlying JSON error.
        source: serde_json::Error,
    },
    /// The declared schema is not supported.
    UnsupportedSchema {
        /// Supported schema version.
        expected: u32,
        /// Version found in the input.
        actual: u32,
    },
    /// A required input path was empty.
    EmptyPath(&'static str),
    /// The worker count must be positive.
    InvalidThreads(usize),
    /// Snapshot cadence must be positive and no larger than the target clock.
    InvalidCheckpointEvery { cadence: u64, updates: u64 },
    /// Core fit-option validation failed.
    InvalidFitOptions {
        /// Core validation failure without exposing an implementation type.
        message: String,
    },
}

impl ConfigError {
    fn with_path(self, path: &Path) -> Self {
        match self {
            Self::Decode { source, .. } => Self::Decode {
                path: Some(path.to_path_buf()),
                source,
            },
            error => error,
        }
    }
}

impl Display for ConfigError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read { path, source } => {
                write!(formatter, "failed to read {}: {source}", path.display())
            }
            Self::Decode {
                path: Some(path),
                source,
            } => write!(formatter, "failed to decode {}: {source}", path.display()),
            Self::Decode { path: None, source } => {
                write!(formatter, "failed to decode training config: {source}")
            }
            Self::UnsupportedSchema { expected, actual } => write!(
                formatter,
                "unsupported training-config schema {actual}; expected {expected}"
            ),
            Self::EmptyPath(field) => write!(formatter, "{field} must not be empty"),
            Self::InvalidThreads(threads) => {
                write!(formatter, "runtime.threads must be positive, got {threads}")
            }
            Self::InvalidCheckpointEvery { cadence, updates } => write!(
                formatter,
                "runtime.checkpoint_every must be in 1..={updates}, got {cadence}"
            ),
            Self::InvalidFitOptions { message } => {
                write!(formatter, "invalid fit options: {message}")
            }
        }
    }
}

impl Error for ConfigError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Read { source, .. } => Some(source),
            Self::Decode { source, .. } => Some(source),
            Self::UnsupportedSchema { .. }
            | Self::EmptyPath(_)
            | Self::InvalidThreads(_)
            | Self::InvalidCheckpointEvery { .. }
            | Self::InvalidFitOptions { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_CONFIG: &str = r#"
        {
          "schema_version": 1,
          "dataset": {
            "nodes": "/inputs/nodes.shp",
            "edges": "/inputs/edges.shp",
            "trajectories": "/inputs/trajectories.pkl"
          },
          "fit": {
            "eta0": 1000.0,
            "lambda": 0.01,
            "lower_factor": 0.1,
            "upper_factor": 10.0,
            "updates": 200
          },
          "runtime": { "threads": 8, "checkpoint_every": 25 }
        }
    "#;

    #[test]
    fn decodes_only_the_minimal_v1_contract() {
        let config = TrainConfig::from_json_slice(VALID_CONFIG.as_bytes()).unwrap();
        assert_eq!(config.dataset.nodes, PathBuf::from("/inputs/nodes.shp"));
        assert_eq!(config.fit.updates, 200);
        assert_eq!(config.threads, 8);
        assert_eq!(config.checkpoint_every, 25);
    }

    #[test]
    fn rejects_experiment_and_unknown_nested_fields() {
        let with_oracle = VALID_CONFIG.replace(
            "\"runtime\": { \"threads\": 8, \"checkpoint_every\": 25 }",
            "\"runtime\": { \"threads\": 8, \"checkpoint_every\": 25 }, \"oracle\": { \"kind\": \"cch\" }",
        );
        let error = TrainConfig::from_json_slice(with_oracle.as_bytes()).unwrap_err();
        assert!(matches!(error, ConfigError::Decode { .. }));

        let with_city = VALID_CONFIG.replace(
            "\"nodes\": \"/inputs/nodes.shp\"",
            "\"nodes\": \"/inputs/nodes.shp\", \"city\": \"porto\"",
        );
        let error = TrainConfig::from_json_slice(with_city.as_bytes()).unwrap_err();
        assert!(matches!(error, ConfigError::Decode { .. }));
    }

    #[test]
    fn validates_schema_runtime_and_core_fit_options() {
        let wrong_schema = VALID_CONFIG.replace("\"schema_version\": 1", "\"schema_version\": 2");
        assert!(matches!(
            TrainConfig::from_json_slice(wrong_schema.as_bytes()),
            Err(ConfigError::UnsupportedSchema { .. })
        ));

        let no_threads = VALID_CONFIG.replace("\"threads\": 8", "\"threads\": 0");
        assert!(matches!(
            TrainConfig::from_json_slice(no_threads.as_bytes()),
            Err(ConfigError::InvalidThreads(0))
        ));

        let no_checkpoints =
            VALID_CONFIG.replace("\"checkpoint_every\": 25", "\"checkpoint_every\": 0");
        assert!(matches!(
            TrainConfig::from_json_slice(no_checkpoints.as_bytes()),
            Err(ConfigError::InvalidCheckpointEvery { .. })
        ));

        let late_checkpoints =
            VALID_CONFIG.replace("\"checkpoint_every\": 25", "\"checkpoint_every\": 201");
        assert!(matches!(
            TrainConfig::from_json_slice(late_checkpoints.as_bytes()),
            Err(ConfigError::InvalidCheckpointEvery { .. })
        ));

        let invalid_eta = VALID_CONFIG.replace("\"eta0\": 1000.0", "\"eta0\": 0.0");
        assert!(matches!(
            TrainConfig::from_json_slice(invalid_eta.as_bytes()),
            Err(ConfigError::InvalidFitOptions { .. })
        ));
    }
}
