use serde_json::{Value, json};
use std::path::{Path, PathBuf};

/// Validated configuration for the single edge-only training path.
#[derive(Clone, Debug)]
pub struct TrainingConfig {
    raw: Value,
    pub run_id: String,
    pub city: String,
    pub train_variant: String,
    pub validation_variant: String,
    pub epochs: u64,
    pub validation_every: u64,
    pub early_stop_patience: usize,
    pub early_stop_min_delta: f64,
    pub eta0: f64,
    pub lambda_edge: f64,
    pub q_min: f64,
    pub q_max: f64,
    pub quantization_scale: f64,
}

impl TrainingConfig {
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let raw: Value = serde_json::from_slice(&bytes)
            .map_err(|error| format!("failed to decode {}: {error}", path.display()))?;
        Self::from_value(raw).map_err(|error| format!("{}: {error}", path.display()))
    }

    fn from_value(raw: Value) -> Result<Self, String> {
        if require_u64(&raw, "/schema_version")? != 1 {
            return Err("schema_version must be 1".to_string());
        }
        for (pointer, expected) in [
            ("/data/path_contract", "complete_original_edge_id_sequence"),
            ("/data/cycle_policy", "drop"),
            ("/model/kind", "edge_only"),
            ("/model/solver", "projected_subgradient"),
            ("/oracle/kind", "cch"),
            ("/oracle/customization", "full"),
            ("/selection/split", "validation"),
            ("/selection/metric", "aggregate_relative_regret"),
            ("/test_policy", "never_read"),
        ] {
            let actual = require_str(&raw, pointer)?;
            if actual != expected {
                return Err(format!("{pointer} must be {expected:?}, got {actual:?}"));
            }
        }
        if raw
            .pointer("/oracle/group_unique_od")
            .and_then(Value::as_bool)
            != Some(true)
        {
            return Err("/oracle/group_unique_od must be true".to_string());
        }

        let run_id = require_str(&raw, "/run_id")?.to_string();
        let city = require_str(&raw, "/data/city")?.to_string();
        let train_variant = require_str(&raw, "/data/train_variant")?.to_string();
        let validation_variant = require_str(&raw, "/data/validation_variant")?.to_string();
        for (label, value) in [
            ("run_id", run_id.as_str()),
            ("city", city.as_str()),
            ("train_variant", train_variant.as_str()),
            ("validation_variant", validation_variant.as_str()),
        ] {
            if value.is_empty() || value.contains('/') || value.contains("..") {
                return Err(format!(
                    "{label} is empty or contains an unsafe path component"
                ));
            }
        }

        let epochs = require_u64(&raw, "/training/epochs")?;
        let validation_every = require_u64(&raw, "/training/validation_every")?;
        let early_stop_patience =
            usize::try_from(require_u64(&raw, "/training/early_stop_patience")?)
                .map_err(|_| "early_stop_patience does not fit usize".to_string())?;
        let early_stop_min_delta = require_f64(&raw, "/training/early_stop_min_delta")?;
        let eta0 = require_f64(&raw, "/model/eta0")?;
        let lambda_edge = require_f64(&raw, "/model/lambda_edge")?;
        let q_min = require_f64(&raw, "/model/q_min")?;
        let q_max = require_f64(&raw, "/model/q_max")?;
        let quantization_scale = require_f64(&raw, "/model/quantization_scale")?;

        if epochs == 0 || validation_every == 0 || early_stop_patience == 0 {
            return Err(
                "epochs, validation_every, and early_stop_patience must be positive".into(),
            );
        }
        if !eta0.is_finite() || eta0 <= 0.0 {
            return Err("eta0 must be finite and positive".into());
        }
        if !lambda_edge.is_finite() || lambda_edge < 0.0 {
            return Err("lambda_edge must be finite and nonnegative".into());
        }
        if !q_min.is_finite() || !q_max.is_finite() || q_min <= 0.0 || q_max < q_min {
            return Err("q_min/q_max must define a finite positive box".into());
        }
        if !quantization_scale.is_finite() || quantization_scale <= 0.0 {
            return Err("quantization_scale must be finite and positive".into());
        }
        if !early_stop_min_delta.is_finite() || early_stop_min_delta < 0.0 {
            return Err("early_stop_min_delta must be finite and nonnegative".into());
        }

        Ok(Self {
            raw,
            run_id,
            city,
            train_variant,
            validation_variant,
            epochs,
            validation_every,
            early_stop_patience,
            early_stop_min_delta,
            eta0,
            lambda_edge,
            q_min,
            q_max,
            quantization_scale,
        })
    }

    pub fn as_json(&self) -> &Value {
        &self.raw
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunOptions {
    pub config_path: PathBuf,
    pub output_dir: PathBuf,
}

impl RunOptions {
    pub fn from_args() -> Result<Option<Self>, String> {
        Self::from_iter(std::env::args().skip(1))
    }

    fn from_iter<I, S>(arguments: I) -> Result<Option<Self>, String>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let arguments = arguments.into_iter().map(Into::into).collect::<Vec<_>>();
        if arguments
            .iter()
            .any(|argument| argument == "--help" || argument == "-h")
        {
            print_help();
            return Ok(None);
        }
        let mut config_path = None;
        let mut output_dir = None;
        let mut index = 0;
        while index < arguments.len() {
            let flag = &arguments[index];
            let value = arguments
                .get(index + 1)
                .ok_or_else(|| format!("missing value for {flag}"))?;
            let slot = match flag.as_str() {
                "--config" => &mut config_path,
                "--output-dir" => &mut output_dir,
                _ => return Err(format!("unknown argument {flag:?}; use --help")),
            };
            if slot.replace(PathBuf::from(value)).is_some() {
                return Err(format!("{flag} was provided more than once"));
            }
            index += 2;
        }
        Ok(Some(Self {
            config_path: config_path.ok_or_else(|| "missing --config PATH".to_string())?,
            output_dir: output_dir.ok_or_else(|| "missing --output-dir PATH".to_string())?,
        }))
    }
}

fn print_help() {
    println!(
        "edge-weight-recovery train\n\
         Learn one shared edge-only metric with projected subgradient descent.\n\
         Training always drops cyclic complete paths, uses full CCH customization,\n\
         selects by validation aggregate relative regret, and never reads test.\n\n\
         Usage:\n\
           train --config PATH --output-dir PATH\n\n\
         Options:\n\
           --config PATH      compact experiment JSON\n\
           --output-dir PATH  checkpoint.json and training.jsonl destination\n\
           -h, --help         show this help"
    );
}

#[derive(Clone, Debug)]
pub struct TrainingState {
    pub best_selection_value: f64,
    pub best_train_mean_regret: f64,
    pub best_weights: Vec<u32>,
    pub best_q: Vec<f64>,
    pub best_epoch: u64,
    pub stale_evaluations: usize,
    early_stop_reference: f64,
}

impl TrainingState {
    pub fn new(initial_weights: &[u32], initial_q: &[f64]) -> Self {
        Self {
            best_selection_value: f64::INFINITY,
            best_train_mean_regret: f64::INFINITY,
            best_weights: initial_weights.to_vec(),
            best_q: initial_q.to_vec(),
            best_epoch: 0,
            stale_evaluations: 0,
            early_stop_reference: f64::INFINITY,
        }
    }

    pub fn update(
        &mut self,
        epoch: u64,
        selection_value: f64,
        train_mean_regret: f64,
        weights: &[u32],
        q: &[f64],
        early_stop_min_delta: f64,
    ) -> bool {
        let is_best = selection_value < self.best_selection_value;
        if is_best {
            self.best_selection_value = selection_value;
            self.best_train_mean_regret = train_mean_regret;
            self.best_weights.clear();
            self.best_weights.extend_from_slice(weights);
            self.best_q.clear();
            self.best_q.extend_from_slice(q);
            self.best_epoch = epoch;
        }
        if selection_value < self.early_stop_reference - early_stop_min_delta {
            self.early_stop_reference = selection_value;
            self.stale_evaluations = 0;
        } else {
            self.stale_evaluations += 1;
        }
        is_best
    }

    pub fn save_checkpoint(
        &self,
        output_dir: &Path,
        config: &TrainingConfig,
        runtime_identity: &Value,
    ) -> Result<PathBuf, String> {
        let checkpoint = json!({
            "schema_version": 2,
            "model": "edge_only",
            "epoch": self.best_epoch,
            "configuration": config.as_json(),
            "selection": {
                "split": "validation",
                "metric": "aggregate_relative_regret",
                "value": self.best_selection_value,
            },
            "train_mean_regret": self.best_train_mean_regret,
            "runtime_identity": runtime_identity,
            "q": &self.best_q,
            "quantized_metric_weights": &self.best_weights,
        });
        let bytes = serde_json::to_vec(&checkpoint)
            .map_err(|error| format!("failed to serialize checkpoint: {error}"))?;
        let path = output_dir.join("checkpoint.json");
        atomic_write(&path, &bytes)?;
        Ok(path)
    }
}

pub fn load_checkpoint(path: &Path) -> Result<Value, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let checkpoint: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("failed to decode {}: {error}", path.display()))?;
    if checkpoint
        .pointer("/schema_version")
        .and_then(Value::as_u64)
        != Some(2)
        || checkpoint.pointer("/model").and_then(Value::as_str) != Some("edge_only")
    {
        return Err(format!(
            "{} is not an edge-only schema-2 checkpoint",
            path.display()
        ));
    }
    Ok(checkpoint)
}

pub fn atomic_write(path: &Path, contents: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    let temporary = path.with_extension(format!("{extension}.{}.tmp", std::process::id()));
    std::fs::write(&temporary, contents)
        .map_err(|error| format!("failed to write {}: {error}", temporary.display()))?;
    std::fs::rename(&temporary, path).map_err(|error| {
        format!(
            "failed to atomically replace {} with {}: {error}",
            path.display(),
            temporary.display()
        )
    })
}

fn require_str<'a>(value: &'a Value, pointer: &str) -> Result<&'a str, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string {pointer}"))
}

fn require_u64(value: &Value, pointer: &str) -> Result<u64, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("missing nonnegative integer {pointer}"))
}

fn require_f64(value: &Value, pointer: &str) -> Result<f64, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("missing number {pointer}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_value() -> Value {
        json!({
            "schema_version": 1,
            "run_id": "smoke",
            "data": {
                "city": "beijing",
                "train_variant": "small",
                "validation_variant": "small",
                "path_contract": "complete_original_edge_id_sequence",
                "cycle_policy": "drop"
            },
            "model": {
                "kind": "edge_only", "solver": "projected_subgradient",
                "eta0": 1e-4, "lambda_edge": 1e5, "q_min": 0.1,
                "q_max": 10.0, "quantization_scale": 1.0
            },
            "oracle": {"kind": "cch", "customization": "full", "group_unique_od": true},
            "training": {
                "epochs": 5, "validation_every": 1, "early_stop_patience": 4,
                "early_stop_min_delta": 0.0
            },
            "selection": {"split": "validation", "metric": "aggregate_relative_regret"},
            "test_policy": "never_read"
        })
    }

    #[test]
    fn accepts_only_the_single_mainline_configuration() {
        let config = TrainingConfig::from_value(config_value()).unwrap();
        assert_eq!(config.city, "beijing");
        let mut invalid = config_value();
        invalid["oracle"]["customization"] = json!("partial");
        assert!(TrainingConfig::from_value(invalid).is_err());
    }

    #[test]
    fn cli_has_only_config_and_output_directory() {
        let options = RunOptions::from_iter(["--config", "run.json", "--output-dir", "/tmp/run"])
            .unwrap()
            .unwrap();
        assert_eq!(options.config_path, PathBuf::from("run.json"));
        assert!(RunOptions::from_iter(["--solver", "adam-shock"]).is_err());
        assert!(RunOptions::from_iter(["--run-test"]).is_err());
    }

    #[test]
    fn selection_and_patience_track_different_improvement_thresholds() {
        let mut state = TrainingState::new(&[10], &[1.0]);
        assert!(state.update(0, 1.0, 4.0, &[10], &[1.0], 0.01));
        assert!(state.update(1, 0.995, 3.0, &[9], &[0.9], 0.01));
        assert_eq!(state.best_epoch, 1);
        assert_eq!(state.stale_evaluations, 1);
        assert!(state.update(2, 0.98, 2.0, &[8], &[0.8], 0.01));
        assert_eq!(state.stale_evaluations, 0);
    }
}
