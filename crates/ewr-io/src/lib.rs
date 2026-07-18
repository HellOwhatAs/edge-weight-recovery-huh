//! Production input, configuration, and atomic training-artifact adapters.
//!
//! File formats terminate at this crate and are converted into `ewr-core`
//! values before the algorithm runs.

mod artifact;
mod config;
mod dataset;

pub use artifact::{
    ArtifactError, TRAINING_ARTIFACT_SCHEMA_V1, load_training_artifact, save_training_artifact,
};
pub use config::{ConfigError, DatasetPaths, TrainConfig};
pub use dataset::{DatasetError, LoadReport, LoadedDataset, load_dataset};
