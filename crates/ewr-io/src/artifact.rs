use ewr_core::{
    EdgeId, FitDiagnostics, FitResult, ModelError, TopologyId, TrainerError, TrainingOutcome,
    TrainingState, Transition, TransitionWeightModel,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::ffi::OsString;
use std::fmt::{Display, Formatter};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Stable schema of one self-consistent model and exact resume snapshot.
pub const TRAINING_ARTIFACT_SCHEMA_V1: &str = "ewr.training-artifact/v1";

static TEMPORARY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Atomically save one model/state pair as a single commit unit.
pub fn save_training_artifact(
    path: impl AsRef<Path>,
    outcome: &TrainingOutcome,
) -> Result<(), ArtifactError> {
    validate_outcome(outcome)?;
    let model = &outcome.result.model;
    let state = &outcome.state;
    let coordinates = model
        .transitions()
        .iter()
        .zip(model.weights())
        .map(|(transition, &weight)| CoordinateDocument {
            previous_edge: transition.previous.index() as u32,
            next_edge: transition.next.index() as u32,
            weight,
        })
        .collect();
    let document = TrainingArtifactDocument {
        schema: TRAINING_ARTIFACT_SCHEMA_V1.to_string(),
        topology_id: state.topology_id().as_str().to_string(),
        training_problem_id: state.training_problem_id().to_string(),
        oracle_identity: state.oracle_identity().to_string(),
        optimizer: OptimizerDocument {
            eta0: state.eta0(),
            lambda: state.lambda(),
            lower_factor: state.lower_factor(),
            upper_factor: state.upper_factor(),
        },
        completed_updates: state.completed_updates(),
        objective: outcome.result.diagnostics.objective,
        initial_weights: state.initial_weights().to_vec(),
        coordinates,
    };
    save_json(path.as_ref(), &document)
}

/// Load and validate one complete model and exact resume snapshot.
pub fn load_training_artifact(path: impl AsRef<Path>) -> Result<TrainingOutcome, ArtifactError> {
    let path = path.as_ref();
    let document: TrainingArtifactDocument = load_json(path)?;
    if document.schema != TRAINING_ARTIFACT_SCHEMA_V1 {
        return Err(ArtifactError::UnsupportedSchema {
            expected: TRAINING_ARTIFACT_SCHEMA_V1,
            actual: document.schema,
        });
    }
    if !document.objective.is_finite() {
        return Err(ArtifactError::InvalidDiagnostics(format!(
            "objective must be finite, got {}",
            document.objective
        )));
    }

    let topology_id = TopologyId::new(document.topology_id).map_err(ArtifactError::InvalidModel)?;
    let (transitions, weights): (Vec<_>, Vec<_>) = document
        .coordinates
        .into_iter()
        .map(|coordinate| {
            (
                Transition {
                    previous: EdgeId::new(coordinate.previous_edge),
                    next: EdgeId::new(coordinate.next_edge),
                },
                coordinate.weight,
            )
        })
        .unzip();
    let model = TransitionWeightModel::new(topology_id.clone(), transitions, weights.clone())
        .map_err(ArtifactError::InvalidModel)?;
    let state = TrainingState::from_parts(
        topology_id,
        document.training_problem_id,
        document.oracle_identity,
        document.initial_weights,
        weights,
        document.completed_updates,
        document.optimizer.eta0,
        document.optimizer.lambda,
        document.optimizer.lower_factor,
        document.optimizer.upper_factor,
    )
    .map_err(ArtifactError::InvalidState)?;
    let outcome = TrainingOutcome {
        result: FitResult {
            model,
            diagnostics: FitDiagnostics {
                completed_updates: document.completed_updates,
                objective: document.objective,
            },
        },
        state,
    };
    validate_outcome(&outcome)?;
    Ok(outcome)
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrainingArtifactDocument {
    schema: String,
    topology_id: String,
    training_problem_id: String,
    oracle_identity: String,
    optimizer: OptimizerDocument,
    completed_updates: u64,
    objective: f64,
    initial_weights: Vec<f64>,
    coordinates: Vec<CoordinateDocument>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OptimizerDocument {
    eta0: f64,
    lambda: f64,
    lower_factor: f64,
    upper_factor: f64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CoordinateDocument {
    previous_edge: u32,
    next_edge: u32,
    weight: f64,
}

fn validate_outcome(outcome: &TrainingOutcome) -> Result<(), ArtifactError> {
    let model = &outcome.result.model;
    let state = &outcome.state;
    if model.topology_id() != state.topology_id() {
        return Err(ArtifactError::InconsistentOutcome(
            "model and state topology IDs differ".to_string(),
        ));
    }
    if model.weights().len() != state.weights().len()
        || !model
            .weights()
            .iter()
            .zip(state.weights())
            .all(|(&model, &state)| model.to_bits() == state.to_bits())
    {
        return Err(ArtifactError::InconsistentOutcome(
            "model and state weights differ".to_string(),
        ));
    }
    if outcome.result.diagnostics.completed_updates != state.completed_updates() {
        return Err(ArtifactError::InconsistentOutcome(
            "diagnostic and state clocks differ".to_string(),
        ));
    }
    if !outcome.result.diagnostics.objective.is_finite() {
        return Err(ArtifactError::InvalidDiagnostics(format!(
            "objective must be finite, got {}",
            outcome.result.diagnostics.objective
        )));
    }
    Ok(())
}

fn load_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, ArtifactError> {
    let bytes = std::fs::read(path).map_err(|source| ArtifactError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| ArtifactError::Decode {
        path: path.to_path_buf(),
        source,
    })
}

fn save_json(path: &Path, value: &impl Serialize) -> Result<(), ArtifactError> {
    let mut bytes = serde_json::to_vec(value).map_err(ArtifactError::Encode)?;
    bytes.push(b'\n');
    atomic_write(path, &bytes)
}

fn atomic_write(path: &Path, contents: &[u8]) -> Result<(), ArtifactError> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty());
    if let Some(parent) = parent {
        std::fs::create_dir_all(parent).map_err(|source| ArtifactError::CreateDirectory {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let filename = path
        .file_name()
        .ok_or_else(|| ArtifactError::InvalidPath(path.to_path_buf()))?;

    let mut temporary = None;
    for _ in 0..16 {
        let sequence = TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let mut temporary_name = OsString::from(".");
        temporary_name.push(filename);
        temporary_name.push(format!(".{}.{sequence}.tmp", std::process::id()));
        let temporary_path = path.with_file_name(temporary_name);
        match TemporaryArtifact::create(temporary_path) {
            Ok(file) => {
                temporary = Some(file);
                break;
            }
            Err(ArtifactError::CreateTemporary { source, .. })
                if source.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error),
        }
    }
    let mut temporary = temporary.ok_or_else(|| ArtifactError::TemporaryNameExhausted {
        destination: path.to_path_buf(),
    })?;
    temporary
        .file
        .write_all(contents)
        .map_err(|source| ArtifactError::Write {
            path: temporary.path.clone(),
            source,
        })?;
    temporary
        .file
        .sync_all()
        .map_err(|source| ArtifactError::Sync {
            path: temporary.path.clone(),
            source,
        })?;
    std::fs::rename(&temporary.path, path).map_err(|source| ArtifactError::Replace {
        temporary: temporary.path.clone(),
        destination: path.to_path_buf(),
        source,
    })?;
    temporary.committed = true;
    Ok(())
}

struct TemporaryArtifact {
    path: PathBuf,
    file: File,
    committed: bool,
}

impl TemporaryArtifact {
    fn create(path: PathBuf) -> Result<Self, ArtifactError> {
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|source| ArtifactError::CreateTemporary {
                path: path.clone(),
                source,
            })?;
        Ok(Self {
            path,
            file,
            committed: false,
        })
    }
}

impl Drop for TemporaryArtifact {
    fn drop(&mut self) {
        if !self.committed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Failure to decode, validate, or atomically publish a training artifact.
#[derive(Debug)]
pub enum ArtifactError {
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    Decode {
        path: PathBuf,
        source: serde_json::Error,
    },
    Encode(serde_json::Error),
    UnsupportedSchema {
        expected: &'static str,
        actual: String,
    },
    InvalidModel(ModelError),
    InvalidState(TrainerError),
    InvalidDiagnostics(String),
    InconsistentOutcome(String),
    InvalidPath(PathBuf),
    CreateDirectory {
        path: PathBuf,
        source: std::io::Error,
    },
    CreateTemporary {
        path: PathBuf,
        source: std::io::Error,
    },
    TemporaryNameExhausted {
        destination: PathBuf,
    },
    Write {
        path: PathBuf,
        source: std::io::Error,
    },
    Sync {
        path: PathBuf,
        source: std::io::Error,
    },
    Replace {
        temporary: PathBuf,
        destination: PathBuf,
        source: std::io::Error,
    },
}

impl Display for ArtifactError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read { path, source } => {
                write!(formatter, "failed to read {}: {source}", path.display())
            }
            Self::Decode { path, source } => {
                write!(formatter, "failed to decode {}: {source}", path.display())
            }
            Self::Encode(source) => write!(formatter, "failed to encode artifact: {source}"),
            Self::UnsupportedSchema { expected, actual } => write!(
                formatter,
                "unsupported training-artifact schema {actual:?}; expected {expected:?}"
            ),
            Self::InvalidModel(source) => write!(formatter, "invalid model artifact: {source}"),
            Self::InvalidState(source) => write!(formatter, "invalid resume state: {source}"),
            Self::InvalidDiagnostics(reason) => {
                write!(formatter, "invalid fit diagnostics: {reason}")
            }
            Self::InconsistentOutcome(reason) => {
                write!(formatter, "inconsistent model/state outcome: {reason}")
            }
            Self::InvalidPath(path) => {
                write!(
                    formatter,
                    "artifact path has no filename: {}",
                    path.display()
                )
            }
            Self::CreateDirectory { path, source } => write!(
                formatter,
                "failed to create artifact directory {}: {source}",
                path.display()
            ),
            Self::CreateTemporary { path, source } => write!(
                formatter,
                "failed to create temporary artifact {}: {source}",
                path.display()
            ),
            Self::TemporaryNameExhausted { destination } => write!(
                formatter,
                "could not reserve a temporary name for {}",
                destination.display()
            ),
            Self::Write { path, source } => {
                write!(formatter, "failed to write {}: {source}", path.display())
            }
            Self::Sync { path, source } => {
                write!(formatter, "failed to sync {}: {source}", path.display())
            }
            Self::Replace {
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

impl Error for ArtifactError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Read { source, .. }
            | Self::CreateDirectory { source, .. }
            | Self::CreateTemporary { source, .. }
            | Self::Write { source, .. }
            | Self::Sync { source, .. }
            | Self::Replace { source, .. } => Some(source),
            Self::Decode { source, .. } | Self::Encode(source) => Some(source),
            Self::InvalidModel(source) => Some(source),
            Self::InvalidState(source) => Some(source),
            Self::UnsupportedSchema { .. }
            | Self::InvalidDiagnostics(_)
            | Self::InconsistentOutcome(_)
            | Self::InvalidPath(_)
            | Self::TemporaryNameExhausted { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary_path(label: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("ewr-{label}-{nonce}/artifact.json"))
    }

    fn outcome() -> TrainingOutcome {
        let topology_id = TopologyId::new("line-graph-v1:fixture").unwrap();
        let weights = vec![0.5, f64::from_bits(0x4010_0000_0000_0000)];
        let model = TransitionWeightModel::new(
            topology_id.clone(),
            vec![
                Transition {
                    previous: EdgeId::new(0),
                    next: EdgeId::new(1),
                },
                Transition {
                    previous: EdgeId::new(2),
                    next: EdgeId::new(3),
                },
            ],
            weights.clone(),
        )
        .unwrap();
        let state = TrainingState::from_parts(
            topology_id,
            "training-problem-v1:fixture".to_string(),
            "ewr-cch:fixture:v1".to_string(),
            vec![5.0, 2.0],
            weights,
            7,
            0.5,
            0.1,
            0.1,
            10.0,
        )
        .unwrap();
        TrainingOutcome {
            result: FitResult {
                model,
                diagnostics: FitDiagnostics {
                    completed_updates: 7,
                    objective: 12.5,
                },
            },
            state,
        }
    }

    #[test]
    fn single_artifact_round_trips_model_state_and_float_bits() {
        let path = temporary_path("round-trip");
        let expected = outcome();
        save_training_artifact(&path, &expected).unwrap();
        let actual = load_training_artifact(&path).unwrap();
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();

        assert_eq!(actual, expected);
        assert_eq!(
            actual.state.weights()[1].to_bits(),
            expected.state.weights()[1].to_bits()
        );
        assert_eq!(actual.result.model.weights(), actual.state.weights());
    }

    #[test]
    fn literal_v1_fixture_is_stable_and_strict() {
        let path = temporary_path("literal");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let fixture = br#"{
          "schema": "ewr.training-artifact/v1",
          "topology_id": "line-graph-v1:fixture",
          "training_problem_id": "training-problem-v1:fixture",
          "oracle_identity": "ewr-cch:fixture:v1",
          "optimizer": {
            "eta0": 0.5,
            "lambda": 0.1,
            "lower_factor": 0.1,
            "upper_factor": 10.0
          },
          "completed_updates": 1,
          "objective": 3.25,
          "initial_weights": [5.0, 2.0],
          "coordinates": [
            {"previous_edge": 0, "next_edge": 1, "weight": 0.5},
            {"previous_edge": 2, "next_edge": 3, "weight": 4.0}
          ]
        }"#;
        std::fs::write(&path, fixture).unwrap();
        let restored = load_training_artifact(&path).unwrap();
        assert_eq!(restored.state.completed_updates(), 1);
        assert_eq!(restored.result.model.weights(), &[0.5, 4.0]);

        let unknown = String::from_utf8(fixture.to_vec()).unwrap().replace(
            "\"completed_updates\": 1",
            "\"extra\": 1, \"completed_updates\": 1",
        );
        std::fs::write(&path, unknown).unwrap();
        assert!(matches!(
            load_training_artifact(&path),
            Err(ArtifactError::Decode { .. })
        ));
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn load_rejects_a_snapshot_outside_its_declared_optimizer_bounds() {
        let path = temporary_path("outside-bounds");
        let expected = outcome();
        save_training_artifact(&path, &expected).unwrap();
        let document = std::fs::read_to_string(&path).unwrap();
        let outside = document.replace("\"weight\":0.5", "\"weight\":0.49");
        assert_ne!(outside, document);
        std::fs::write(&path, outside).unwrap();

        assert!(matches!(
            load_training_artifact(&path),
            Err(ArtifactError::InvalidState(
                TrainerError::StateWeightOutsideBounds { .. }
            ))
        ));
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn save_rejects_a_cross_run_model_state_pair() {
        let path = temporary_path("inconsistent");
        let mut inconsistent = outcome();
        inconsistent.result.model = TransitionWeightModel::new(
            inconsistent.state.topology_id().clone(),
            inconsistent.result.model.transitions().to_vec(),
            vec![0.5, 3.0],
        )
        .unwrap();

        assert!(matches!(
            save_training_artifact(&path, &inconsistent),
            Err(ArtifactError::InconsistentOutcome(_))
        ));
        assert!(!path.exists());
    }

    #[test]
    fn atomic_save_replaces_one_complete_document_without_temp_files() {
        let path = temporary_path("replace");
        let first = outcome();
        save_training_artifact(&path, &first).unwrap();
        let mut second = first.clone();
        second.result.diagnostics.objective = 9.0;
        save_training_artifact(&path, &second).unwrap();

        assert_eq!(load_training_artifact(&path).unwrap(), second);
        assert_eq!(
            std::fs::read_dir(path.parent().unwrap()).unwrap().count(),
            1
        );
        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }
}
