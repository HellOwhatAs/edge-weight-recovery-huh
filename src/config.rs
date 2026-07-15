use serde_json::{Value, json};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub enum ExperimentConfig {
    EdgeOnly(TrainingConfig),
    TurnAware(TurnTrainingConfig),
}

impl ExperimentConfig {
    pub fn load(path: &Path) -> Result<Self, String> {
        let raw = read_json(path)?;
        let kind = raw
            .pointer("/model/kind")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{}: missing string /model/kind", path.display()))?
            .to_string();
        let parsed = match kind.as_str() {
            "edge_only" => TrainingConfig::from_value(raw).map(Self::EdgeOnly),
            "turn_aware" => TurnTrainingConfig::from_value(raw).map(Self::TurnAware),
            _ => Err(format!("unsupported /model/kind {kind:?}")),
        };
        parsed.map_err(|error| format!("{}: {error}", path.display()))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TurnExperimentArm {
    ExpandedEdgeContinuation,
    TurnOnly,
    JointEdgeTurn,
}

impl TurnExperimentArm {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ExpandedEdgeContinuation => "expanded_edge_continuation",
            Self::TurnOnly => "turn_only",
            Self::JointEdgeTurn => "joint_edge_turn",
        }
    }

    pub const fn updates_q(self) -> bool {
        matches!(self, Self::ExpandedEdgeContinuation | Self::JointEdgeTurn)
    }

    pub const fn updates_residuals(self) -> bool {
        matches!(self, Self::TurnOnly | Self::JointEdgeTurn)
    }

    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "expanded_edge_continuation" => Ok(Self::ExpandedEdgeContinuation),
            "turn_only" => Ok(Self::TurnOnly),
            "joint_edge_turn" => Ok(Self::JointEdgeTurn),
            _ => Err(format!("unsupported turn-aware arm {value:?}")),
        }
    }
}

/// Strict configuration for a turn-residual expanded-graph run.
///
/// This also supports exact replay of the archived historical A/B/C protocol;
/// accepting one of its update modes does not establish a model ranking.
#[derive(Clone, Debug)]
pub struct TurnTrainingConfig {
    raw: Value,
    pub run_id: String,
    pub protocol_id: String,
    pub stage: String,
    pub city: String,
    pub train_variant: String,
    pub validation_variant: String,
    pub initialization_path: PathBuf,
    pub initialization_sha256: String,
    pub arm: TurnExperimentArm,
    pub updates: u64,
    pub validation_every: u64,
    pub eta_q0: f64,
    pub lambda_edge: f64,
    pub q_min: f64,
    pub q_max: f64,
    pub quantization_scale: f64,
    pub eta_r0: Option<f64>,
    pub lambda_turn: Option<f64>,
    pub residual_scale: f64,
    pub r_max: f64,
    pub rayon_threads: usize,
}

impl TurnTrainingConfig {
    pub fn load(path: &Path) -> Result<Self, String> {
        Self::from_value(read_json(path)?).map_err(|error| format!("{}: {error}", path.display()))
    }

    fn from_value(raw: Value) -> Result<Self, String> {
        reject_unknown_keys(
            &raw,
            "",
            &[
                "schema_version",
                "run_id",
                "description",
                "protocol_id",
                "stage",
                "archive_commit",
                "initialization",
                "data",
                "model",
                "oracle",
                "training",
                "runtime",
                "selection",
                "test_policy",
            ],
        )?;
        reject_unknown_keys(&raw, "/initialization", &["path", "sha256"])?;
        reject_unknown_keys(
            &raw,
            "/data",
            &[
                "city",
                "train_variant",
                "validation_variant",
                "path_contract",
                "cycle_policy",
                "train_identity",
                "validation_identity",
            ],
        )?;
        for pointer in ["/data/train_identity", "/data/validation_identity"] {
            reject_unknown_keys(
                &raw,
                pointer,
                &[
                    "path",
                    "bytes",
                    "sha256",
                    "source_sha256",
                    "sample_count",
                    "seed",
                ],
            )?;
        }
        reject_unknown_keys(
            &raw,
            "/model",
            &[
                "kind",
                "arm",
                "solver",
                "eta_q0",
                "lambda_edge",
                "q_min",
                "q_max",
                "quantization_scale",
                "eta_r0",
                "lambda_turn",
                "residual_scale_policy",
                "residual_scale",
                "r_max",
            ],
        )?;
        reject_unknown_keys(
            &raw,
            "/oracle",
            &["kind", "customization", "group_unique_od"],
        )?;
        reject_unknown_keys(
            &raw,
            "/training",
            &["updates", "validation_every", "early_stop"],
        )?;
        reject_unknown_keys(&raw, "/runtime", &["rayon_threads"])?;
        reject_unknown_keys(&raw, "/selection", &["split", "metric"])?;

        if require_u64(&raw, "/schema_version")? != 1 {
            return Err("schema_version must be 1".to_string());
        }
        for (pointer, expected) in [
            ("/data/path_contract", "complete_original_edge_id_sequence"),
            ("/data/cycle_policy", "drop"),
            ("/model/kind", "turn_aware"),
            ("/model/solver", "blockwise_projected_subgradient"),
            ("/model/residual_scale_policy", "frozen_edge_metric_median"),
            ("/oracle/kind", "expanded_cch"),
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
        if raw.pointer("/training/early_stop").and_then(Value::as_bool) != Some(false) {
            return Err("/training/early_stop must be false".to_string());
        }

        let run_id = require_safe_component(&raw, "/run_id", "run_id")?;
        let protocol_id = require_safe_component(&raw, "/protocol_id", "protocol_id")?;
        let stage = require_str(&raw, "/stage")?.to_string();
        if !matches!(
            stage.as_str(),
            "correctness" | "screen_10pct" | "full_endpoint"
        ) {
            return Err("stage must be correctness, screen_10pct, or full_endpoint".to_string());
        }
        let city = require_safe_component(&raw, "/data/city", "city")?;
        let train_variant = require_safe_component(&raw, "/data/train_variant", "train_variant")?;
        let validation_variant =
            require_safe_component(&raw, "/data/validation_variant", "validation_variant")?;
        let initialization_path = PathBuf::from(require_str(&raw, "/initialization/path")?);
        let initialization_sha256 = require_str(&raw, "/initialization/sha256")?.to_string();
        validate_sha256(&initialization_sha256, "/initialization/sha256")?;
        let arm = TurnExperimentArm::parse(require_str(&raw, "/model/arm")?)?;
        let updates = require_u64(&raw, "/training/updates")?;
        let validation_every = require_u64(&raw, "/training/validation_every")?;
        let eta_q0 = require_f64(&raw, "/model/eta_q0")?;
        let lambda_edge = require_f64(&raw, "/model/lambda_edge")?;
        let q_min = require_f64(&raw, "/model/q_min")?;
        let q_max = require_f64(&raw, "/model/q_max")?;
        let quantization_scale = require_f64(&raw, "/model/quantization_scale")?;
        let residual_scale = require_f64(&raw, "/model/residual_scale")?;
        let r_max = require_f64(&raw, "/model/r_max")?;
        let rayon_threads = usize::try_from(require_u64(&raw, "/runtime/rayon_threads")?)
            .map_err(|_| "rayon_threads does not fit usize".to_string())?;
        let eta_r0 = optional_f64(&raw, "/model/eta_r0")?;
        let lambda_turn = optional_f64(&raw, "/model/lambda_turn")?;

        if updates == 0 || validation_every == 0 || validation_every > updates {
            return Err(
                "updates and validation_every must be positive, with cadence <= updates".into(),
            );
        }
        if rayon_threads == 0 {
            return Err("rayon_threads must be positive".into());
        }
        if !eta_q0.is_finite() || eta_q0 <= 0.0 {
            return Err("eta_q0 must be finite and positive".into());
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
        if !residual_scale.is_finite() || residual_scale <= 0.0 {
            return Err("residual_scale must be finite and positive".into());
        }
        if !r_max.is_finite() || r_max <= 0.0 {
            return Err("r_max must be finite and positive".into());
        }
        match arm.updates_residuals() {
            true => {
                if eta_r0.is_none_or(|value| !value.is_finite() || value <= 0.0) {
                    return Err("turn-only/joint arms require finite positive eta_r0".into());
                }
                if lambda_turn.is_none_or(|value| !value.is_finite() || value < 0.0) {
                    return Err(
                        "turn-only/joint arms require finite nonnegative lambda_turn".into(),
                    );
                }
            }
            false if eta_r0.is_some() || lambda_turn.is_some() => {
                return Err("expanded-edge arm must omit eta_r0 and lambda_turn".into());
            }
            false => {}
        }

        Ok(Self {
            raw,
            run_id,
            protocol_id,
            stage,
            city,
            train_variant,
            validation_variant,
            initialization_path,
            initialization_sha256,
            arm,
            updates,
            validation_every,
            eta_q0,
            lambda_edge,
            q_min,
            q_max,
            quantization_scale,
            eta_r0,
            lambda_turn,
            residual_scale,
            r_max,
            rayon_threads,
        })
    }

    pub fn as_json(&self) -> &Value {
        &self.raw
    }
}

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
        reject_unknown_keys(
            &raw,
            "",
            &[
                "schema_version",
                "run_id",
                "description",
                "archive_commit",
                "data",
                "model",
                "oracle",
                "training",
                "selection",
                "test_policy",
            ],
        )?;
        reject_unknown_keys(
            &raw,
            "/data",
            &[
                "city",
                "train_variant",
                "validation_variant",
                "path_contract",
                "cycle_policy",
                "train_identity",
                "validation_identity",
            ],
        )?;
        for pointer in ["/data/train_identity", "/data/validation_identity"] {
            reject_unknown_keys(
                &raw,
                pointer,
                &[
                    "path",
                    "bytes",
                    "sha256",
                    "source_sha256",
                    "sample_count",
                    "seed",
                ],
            )?;
        }
        reject_unknown_keys(
            &raw,
            "/model",
            &[
                "kind",
                "solver",
                "eta0",
                "lambda_edge",
                "q_min",
                "q_max",
                "quantization_scale",
            ],
        )?;
        reject_unknown_keys(
            &raw,
            "/oracle",
            &["kind", "customization", "group_unique_od"],
        )?;
        reject_unknown_keys(
            &raw,
            "/training",
            &[
                "epochs",
                "validation_every",
                "early_stop_patience",
                "early_stop_min_delta",
            ],
        )?;
        reject_unknown_keys(&raw, "/selection", &["split", "metric"])?;

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
         Train the edge-only baseline or run a configured turn-residual job.\n\
         Training always drops cyclic complete paths, uses full CCH customization,\n\
         selects checkpoints on validation only, and never reads test. Historical\n\
         turn runs retain their model-relative metric for replay, not model ranking.\n\n\
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
    let checkpoint_kind = (
        checkpoint
            .pointer("/schema_version")
            .and_then(Value::as_u64),
        checkpoint.pointer("/model").and_then(Value::as_str),
    );
    if !matches!(
        checkpoint_kind,
        (Some(2), Some("edge_only")) | (Some(3), Some("turn_aware"))
    ) {
        return Err(format!(
            "{} is not a supported edge-only schema-2 or turn-aware schema-3 checkpoint",
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

fn optional_f64(value: &Value, pointer: &str) -> Result<Option<f64>, String> {
    match value.pointer(pointer) {
        None => Ok(None),
        Some(value) => value
            .as_f64()
            .map(Some)
            .ok_or_else(|| format!("{pointer} must be a number when present")),
    }
}

fn require_safe_component(value: &Value, pointer: &str, label: &str) -> Result<String, String> {
    let component = require_str(value, pointer)?;
    if component.is_empty() || component.contains('/') || component.contains("..") {
        return Err(format!(
            "{label} is empty or contains an unsafe path component"
        ));
    }
    Ok(component.to_string())
}

fn validate_sha256(value: &str, pointer: &str) -> Result<(), String> {
    if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err(format!("{pointer} must be 64 hexadecimal characters"))
    }
}

fn read_json(path: &Path) -> Result<Value, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("failed to decode {}: {error}", path.display()))
}

fn reject_unknown_keys(value: &Value, pointer: &str, allowed: &[&str]) -> Result<(), String> {
    let Some(candidate) = (if pointer.is_empty() {
        Some(value)
    } else {
        value.pointer(pointer)
    }) else {
        return Ok(());
    };
    let object = candidate.as_object().ok_or_else(|| {
        format!(
            "{} must be an object",
            if pointer.is_empty() { "/" } else { pointer }
        )
    })?;
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(format!("unknown configuration key {}/{}", pointer, key));
    }
    Ok(())
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

    fn turn_config_value(arm: &str) -> Value {
        let mut value = json!({
            "schema_version": 1,
            "run_id": "turn_screen",
            "protocol_id": "turn_residual_v1",
            "stage": "screen_10pct",
            "initialization": {
                "path": "/tmp/frozen_edge_initialization.json",
                "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            },
            "data": {
                "city": "beijing",
                "train_variant": "scale_10pct_seed42",
                "validation_variant": "time_dev_20090513_excl_previous",
                "path_contract": "complete_original_edge_id_sequence",
                "cycle_policy": "drop"
            },
            "model": {
                "kind": "turn_aware",
                "arm": arm,
                "solver": "blockwise_projected_subgradient",
                "eta_q0": 3e-4,
                "lambda_edge": 1e5,
                "q_min": 0.1,
                "q_max": 10.0,
                "quantization_scale": 1.0,
                "eta_r0": 3e-4,
                "lambda_turn": 1e5,
                "residual_scale_policy": "frozen_edge_metric_median",
                "residual_scale": 127625.0,
                "r_max": 10.0
            },
            "oracle": {
                "kind": "expanded_cch",
                "customization": "full",
                "group_unique_od": true
            },
            "training": {
                "updates": 30,
                "validation_every": 10,
                "early_stop": false
            },
            "runtime": {"rayon_threads": 4},
            "selection": {
                "split": "validation",
                "metric": "aggregate_relative_regret"
            },
            "test_policy": "never_read"
        });
        if arm == "expanded_edge_continuation" {
            value["model"]
                .as_object_mut()
                .expect("model object")
                .remove("eta_r0");
            value["model"]
                .as_object_mut()
                .expect("model object")
                .remove("lambda_turn");
        }
        value
    }

    #[test]
    fn accepts_only_the_single_mainline_configuration() {
        let config = TrainingConfig::from_value(config_value()).unwrap();
        assert_eq!(config.city, "beijing");
        let mut invalid = config_value();
        invalid["oracle"]["customization"] = json!("partial");
        assert!(TrainingConfig::from_value(invalid).is_err());
        let mut unknown = config_value();
        unknown["model"]["shock"] = json!(true);
        assert_eq!(
            TrainingConfig::from_value(unknown).unwrap_err(),
            "unknown configuration key /model/shock"
        );
    }

    #[test]
    fn turn_config_accepts_the_three_implemented_update_modes() {
        let turn_only = TurnTrainingConfig::from_value(turn_config_value("turn_only")).unwrap();
        assert_eq!(turn_only.arm, TurnExperimentArm::TurnOnly);
        assert_eq!(turn_only.updates, 30);
        assert_eq!(turn_only.eta_r0, Some(3e-4));

        let edge = TurnTrainingConfig::from_value(turn_config_value("expanded_edge_continuation"))
            .unwrap();
        assert!(edge.arm.updates_q());
        assert!(!edge.arm.updates_residuals());
        assert_eq!(edge.eta_r0, None);

        let mut invalid = turn_config_value("turn_only");
        invalid["model"]["arm"] = json!("signed_turn_residual");
        assert!(TurnTrainingConfig::from_value(invalid).is_err());
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
