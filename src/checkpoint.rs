use crate::config::atomic_write;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

pub const CHECKPOINT_SCHEMA_VERSION: u64 = 1;

/// The sole checkpoint shape for both graph orders.
///
/// Only the current direct learned vector and the optimizer clock are stateful.
/// Initial weights, bounds, and CCH arc weights are reconstructed by the graph
/// problem and are intentionally not serialized as additional model blocks.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingCheckpoint {
    pub graph_order: String,
    pub completed_updates: u64,
    pub weights: Vec<f64>,
    pub configuration: Value,
    pub runtime_identity: Value,
    pub topology_identity: String,
}

impl TrainingCheckpoint {
    pub fn save(&self, output_dir: &Path) -> Result<PathBuf, String> {
        let path = output_dir.join("checkpoint.json");
        self.save_to(&path)?;
        Ok(path)
    }

    pub fn save_to(&self, path: &Path) -> Result<(), String> {
        self.validate()?;
        let value = json!({
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "graph_order": self.graph_order,
            "completed_updates": self.completed_updates,
            "weights": self.weights,
            "configuration": self.configuration,
            "runtime_identity": self.runtime_identity,
            "topology_identity": self.topology_identity,
        });
        let bytes = serde_json::to_vec(&value)
            .map_err(|error| format!("failed to serialize checkpoint: {error}"))?;
        atomic_write(path, &bytes)
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|error| format!("failed to decode {}: {error}", path.display()))?;
        if value.pointer("/schema_version").and_then(Value::as_u64)
            != Some(CHECKPOINT_SCHEMA_VERSION)
        {
            return Err(format!(
                "{} is not a schema-{CHECKPOINT_SCHEMA_VERSION} direct-weight checkpoint",
                path.display()
            ));
        }
        let graph_order = required_str(&value, "/graph_order")?.to_string();
        let completed_updates = value
            .pointer("/completed_updates")
            .and_then(Value::as_u64)
            .ok_or_else(|| "checkpoint is missing /completed_updates".to_string())?;
        let weights = value
            .pointer("/weights")
            .and_then(Value::as_array)
            .ok_or_else(|| "checkpoint is missing /weights".to_string())?
            .iter()
            .enumerate()
            .map(|(coordinate, value)| {
                value.as_f64().ok_or_else(|| {
                    format!("checkpoint weight {coordinate} is not a finite JSON number")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let checkpoint = Self {
            graph_order,
            completed_updates,
            weights,
            configuration: value
                .pointer("/configuration")
                .cloned()
                .ok_or_else(|| "checkpoint is missing /configuration".to_string())?,
            runtime_identity: value
                .pointer("/runtime_identity")
                .cloned()
                .ok_or_else(|| "checkpoint is missing /runtime_identity".to_string())?,
            topology_identity: required_str(&value, "/topology_identity")?.to_string(),
        };
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    fn validate(&self) -> Result<(), String> {
        if self.graph_order != "first" && self.graph_order != "second" {
            return Err(format!(
                "checkpoint has invalid graph order {:?}",
                self.graph_order
            ));
        }
        if self.weights.is_empty() {
            return Err("checkpoint direct weight vector must not be empty".to_string());
        }
        if let Some((coordinate, weight)) = self
            .weights
            .iter()
            .copied()
            .enumerate()
            .find(|(_, weight)| !weight.is_finite())
        {
            return Err(format!(
                "checkpoint weight {coordinate} is not finite: {weight}"
            ));
        }
        if self.topology_identity.is_empty() {
            return Err("checkpoint topology identity must not be empty".to_string());
        }
        Ok(())
    }
}

fn required_str<'a>(value: &'a Value, pointer: &str) -> Result<&'a str, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("checkpoint is missing string {pointer}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_checkpoint_round_trips_direct_weights_exactly() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("direct-checkpoint-{nonce}.json"));
        let checkpoint = TrainingCheckpoint {
            graph_order: "second".to_string(),
            completed_updates: 7,
            weights: vec![0.1, f64::from_bits(0x3fd5_5555_5555_5555), 9_999.25],
            configuration: json!({"run_id": "fixture"}),
            runtime_identity: json!({"data": "fixture"}),
            topology_identity: "fnv1a64:fixture".to_string(),
        };
        checkpoint.save_to(&path).unwrap();
        let restored = TrainingCheckpoint::load(&path).unwrap();
        assert_eq!(restored, checkpoint);
        std::fs::remove_file(path).unwrap();
    }
}
