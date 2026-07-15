use serde_json::Value;
use std::path::{Path, PathBuf};

/// Strict, representation-neutral training configuration.
#[derive(Clone, Debug)]
pub struct TrainingConfig {
    raw: Value,
    pub run_id: String,
    pub city: String,
    pub train_variant: String,
    pub validation_variant: String,
    pub graph_order: String,
    pub weight_lower_factor: f64,
    pub weight_upper_factor: f64,
    pub eta0: f64,
    pub lambda: f64,
    pub updates: u64,
    pub validation_every: u64,
    pub rayon_threads: usize,
}

impl TrainingConfig {
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let raw: Value = serde_json::from_slice(&bytes)
            .map_err(|error| format!("failed to decode {}: {error}", path.display()))?;
        Self::from_value(raw).map_err(|error| format!("{}: {error}", path.display()))
    }

    pub fn from_value(raw: Value) -> Result<Self, String> {
        reject_unknown_keys(
            &raw,
            "",
            &[
                "schema_version",
                "run_id",
                "description",
                "data",
                "graph",
                "optimizer",
                "oracle",
                "training",
                "runtime",
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
            "/graph",
            &["order", "weight_lower_factor", "weight_upper_factor"],
        )?;
        reject_unknown_keys(&raw, "/optimizer", &["kind", "eta0", "lambda"])?;
        reject_unknown_keys(
            &raw,
            "/oracle",
            &["kind", "customization", "group_unique_od"],
        )?;
        reject_unknown_keys(&raw, "/training", &["updates", "validation_every"])?;
        reject_unknown_keys(&raw, "/runtime", &["rayon_threads"])?;

        if require_u64(&raw, "/schema_version")? != 2 {
            return Err("schema_version must be 2".to_string());
        }
        for (pointer, expected) in [
            ("/data/path_contract", "complete_original_edge_id_sequence"),
            ("/data/cycle_policy", "drop"),
            ("/optimizer/kind", "projected_subgradient"),
            ("/oracle/kind", "cch"),
            ("/oracle/customization", "full"),
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

        let run_id = require_safe_component(&raw, "/run_id", "run_id")?;
        let city = require_safe_component(&raw, "/data/city", "city")?;
        let train_variant = require_safe_component(&raw, "/data/train_variant", "train_variant")?;
        let validation_variant =
            require_safe_component(&raw, "/data/validation_variant", "validation_variant")?;
        let graph_order = require_str(&raw, "/graph/order")?.to_string();
        if graph_order != "first" && graph_order != "second" {
            return Err(format!(
                "/graph/order must be \"first\" or \"second\", got {graph_order:?}"
            ));
        }

        let weight_lower_factor = require_f64(&raw, "/graph/weight_lower_factor")?;
        let weight_upper_factor = require_f64(&raw, "/graph/weight_upper_factor")?;
        if !weight_lower_factor.is_finite()
            || !weight_upper_factor.is_finite()
            || weight_lower_factor <= 0.0
            || weight_lower_factor > 1.0
            || weight_upper_factor < 1.0
        {
            return Err(
                "graph weight factors must be finite with 0 < lower <= 1 <= upper".to_string(),
            );
        }

        let eta0 = require_f64(&raw, "/optimizer/eta0")?;
        let lambda = require_f64(&raw, "/optimizer/lambda")?;
        if !eta0.is_finite() || eta0 <= 0.0 {
            return Err("eta0 must be finite and positive".to_string());
        }
        if !lambda.is_finite() || lambda < 0.0 {
            return Err("lambda must be finite and nonnegative".to_string());
        }

        let updates = require_u64(&raw, "/training/updates")?;
        let validation_every = require_u64(&raw, "/training/validation_every")?;
        if updates == 0 || validation_every == 0 || validation_every > updates {
            return Err(
                "updates and validation_every must be positive, with cadence <= updates"
                    .to_string(),
            );
        }
        let rayon_threads = usize::try_from(require_u64(&raw, "/runtime/rayon_threads")?)
            .map_err(|_| "rayon_threads does not fit usize".to_string())?;
        if rayon_threads == 0 {
            return Err("rayon_threads must be positive".to_string());
        }

        Ok(Self {
            raw,
            run_id,
            city,
            train_variant,
            validation_variant,
            graph_order,
            weight_lower_factor,
            weight_upper_factor,
            eta0,
            lambda,
            updates,
            validation_every,
            rayon_threads,
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
    pub resume: Option<PathBuf>,
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
        let mut resume = None;
        let mut index = 0;
        while index < arguments.len() {
            let flag = &arguments[index];
            let value = arguments
                .get(index + 1)
                .ok_or_else(|| format!("missing value for {flag}"))?;
            let slot = match flag.as_str() {
                "--config" => &mut config_path,
                "--output-dir" => &mut output_dir,
                "--resume" => &mut resume,
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
            resume,
        }))
    }
}

fn print_help() {
    println!(
        "edge-weight-recovery train\n\
         Train direct weights on a first- or second-order graph with one optimizer.\n\
         Training uses full CCH customization, validation diagnostics, and never reads test.\n\n\
         Usage:\n\
           train --config PATH --output-dir PATH [--resume CHECKPOINT]\n\n\
         Options:\n\
           --config PATH       unified experiment JSON\n\
           --output-dir PATH   checkpoint.json and training.jsonl destination\n\
           --resume PATH       continue weights and the global update clock\n\
           -h, --help          show this help"
    );
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

fn reject_unknown_keys(value: &Value, pointer: &str, allowed: &[&str]) -> Result<(), String> {
    let object = value
        .pointer(pointer)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("missing object {pointer:?}"))?;
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(format!(
            "unknown configuration field {}/{key}",
            pointer.trim_end_matches('/')
        ));
    }
    Ok(())
}

fn require_str<'a>(value: &'a Value, pointer: &str) -> Result<&'a str, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string {pointer}"))
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
    use serde_json::json;

    fn config_value(order: &str) -> Value {
        json!({
            "schema_version": 2,
            "run_id": "smoke",
            "data": {
                "city": "beijing",
                "train_variant": "train",
                "validation_variant": "validation",
                "path_contract": "complete_original_edge_id_sequence",
                "cycle_policy": "drop",
                "train_identity": {},
                "validation_identity": {}
            },
            "graph": {
                "order": order,
                "weight_lower_factor": 0.1,
                "weight_upper_factor": 10.0
            },
            "optimizer": {
                "kind": "projected_subgradient",
                "eta0": 1000.0,
                "lambda": 0.001
            },
            "oracle": {
                "kind": "cch",
                "customization": "full",
                "group_unique_od": true
            },
            "training": {"updates": 3, "validation_every": 3},
            "runtime": {"rayon_threads": 4},
            "test_policy": "never_read"
        })
    }

    #[test]
    fn both_orders_share_one_strict_schema() {
        let first = TrainingConfig::from_value(config_value("first")).unwrap();
        let second = TrainingConfig::from_value(config_value("second")).unwrap();
        assert_eq!(first.eta0, second.eta0);
        assert_eq!(first.lambda, second.lambda);
        assert_eq!(first.updates, second.updates);
    }

    #[test]
    fn retired_or_unknown_model_fields_are_rejected() {
        let mut raw = config_value("first");
        raw["optimizer"]["second_lambda"] = json!(1.0);
        assert!(TrainingConfig::from_value(raw).is_err());
    }

    #[test]
    fn command_line_accepts_an_optional_resume_checkpoint() {
        let options = RunOptions::from_iter([
            "--config",
            "config.json",
            "--output-dir",
            "out",
            "--resume",
            "saved.json",
        ])
        .unwrap()
        .unwrap();
        assert_eq!(options.resume, Some(PathBuf::from("saved.json")));
    }
}
