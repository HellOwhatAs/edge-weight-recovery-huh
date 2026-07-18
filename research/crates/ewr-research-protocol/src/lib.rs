//! Small, method-neutral interchange contract for research experiments.
//!
//! Descriptors carry file-level versions. JSONL rows contain only sample IDs
//! and complete paths in the original-road ID space.

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::{self, Display, Formatter};
use std::io::{BufRead, Write};
use std::path::{Component, Path};

pub const DATASET_MANIFEST_SCHEMA_V1: &str = "ewr.dataset-manifest/v1";
pub const DATASET_RECORD_SCHEMA_V1: &str = "ewr.dataset-record/v1";
pub const PREDICTION_RECORD_SCHEMA_V1: &str = "ewr.prediction-record/v1";
pub const RUN_RECEIPT_SCHEMA_V1: &str = "ewr.run-receipt/v1";

/// File-level descriptor for a version-one dataset JSONL file.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetManifestV1 {
    pub schema: String,
    pub dataset_id: String,
    pub network_id: String,
    pub records_schema: String,
    pub records_file: String,
}

impl DatasetManifestV1 {
    pub fn new(
        dataset_id: impl Into<String>,
        network_id: impl Into<String>,
        records_file: impl Into<String>,
    ) -> Self {
        Self {
            schema: DATASET_MANIFEST_SCHEMA_V1.into(),
            dataset_id: dataset_id.into(),
            network_id: network_id.into(),
            records_schema: DATASET_RECORD_SCHEMA_V1.into(),
            records_file: records_file.into(),
        }
    }

    pub fn validate(&self) -> Result<(), ProtocolError> {
        schema("dataset manifest", &self.schema, DATASET_MANIFEST_SCHEMA_V1)?;
        schema(
            "dataset records",
            &self.records_schema,
            DATASET_RECORD_SCHEMA_V1,
        )?;
        nonempty("dataset_id", &self.dataset_id)?;
        nonempty("network_id", &self.network_id)?;
        nonempty("records_file", &self.records_file)?;
        let records_file = Path::new(&self.records_file);
        if records_file.is_absolute()
            || records_file.components().any(|component| {
                matches!(
                    component,
                    Component::ParentDir | Component::RootDir | Component::Prefix(_)
                )
            })
        {
            return fail("records_file must be a safe path relative to its manifest");
        }
        Ok(())
    }
}

/// One sample and its complete observed path.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetRecordV1 {
    pub sample_id: String,
    pub original_edge_ids: Vec<u32>,
}

/// One method's complete predicted path for a sample.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictionRecordV1 {
    pub sample_id: String,
    pub predicted_edge_ids: Vec<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MethodIdentity {
    pub name: String,
    pub version: String,
}

/// Reproduction and attribution metadata for one prediction artifact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunReceiptV1 {
    pub schema: String,
    pub method: MethodIdentity,
    pub dataset_id: String,
    pub dataset_manifest_sha256: String,
    pub prediction_records_schema: String,
    pub configuration: Map<String, Value>,
    pub source_revision: String,
    pub environment: BTreeMap<String, String>,
}

impl RunReceiptV1 {
    pub fn validate(&self) -> Result<(), ProtocolError> {
        schema("run receipt", &self.schema, RUN_RECEIPT_SCHEMA_V1)?;
        schema(
            "prediction records",
            &self.prediction_records_schema,
            PREDICTION_RECORD_SCHEMA_V1,
        )?;
        for (field, value) in [
            ("method.name", self.method.name.as_str()),
            ("method.version", self.method.version.as_str()),
            ("dataset_id", self.dataset_id.as_str()),
            ("source_revision", self.source_revision.as_str()),
        ] {
            nonempty(field, value)?;
        }
        if self.dataset_manifest_sha256.len() != 64
            || !self
                .dataset_manifest_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return fail("dataset_manifest_sha256 must be 64 hexadecimal characters");
        }
        if self.environment.is_empty() {
            return fail("environment must identify at least one property");
        }
        for (key, value) in &self.environment {
            nonempty("environment key", key)?;
            nonempty("environment value", value)?;
        }
        Ok(())
    }
}

trait SampleRecord {
    fn sample_id(&self) -> &str;
    fn edge_field(&self) -> (&'static str, &[u32]);
    fn minimum_edges(&self) -> usize;
}

impl SampleRecord for DatasetRecordV1 {
    fn sample_id(&self) -> &str {
        &self.sample_id
    }

    fn edge_field(&self) -> (&'static str, &[u32]) {
        ("original_edge_ids", &self.original_edge_ids)
    }

    fn minimum_edges(&self) -> usize {
        2
    }
}

impl SampleRecord for PredictionRecordV1 {
    fn sample_id(&self) -> &str {
        &self.sample_id
    }

    fn edge_field(&self) -> (&'static str, &[u32]) {
        ("predicted_edge_ids", &self.predicted_edge_ids)
    }

    fn minimum_edges(&self) -> usize {
        1
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AlignedSample<'a> {
    pub dataset: &'a DatasetRecordV1,
    pub prediction: &'a PredictionRecordV1,
}

/// Align in dataset order and reject duplicates, omissions, and extras.
pub fn align_predictions<'a>(
    dataset: &'a [DatasetRecordV1],
    predictions: &'a [PredictionRecordV1],
) -> Result<Vec<AlignedSample<'a>>, ProtocolError> {
    validate_records(dataset)?;
    validate_records(predictions)?;
    let by_id = predictions
        .iter()
        .map(|record| (record.sample_id.as_str(), record))
        .collect::<HashMap<_, _>>();
    let dataset_ids = dataset
        .iter()
        .map(|record| record.sample_id.as_str())
        .collect::<HashSet<_>>();
    if let Some(extra) = predictions
        .iter()
        .find(|record| !dataset_ids.contains(record.sample_id.as_str()))
    {
        return fail(format!("unexpected prediction for {:?}", extra.sample_id));
    }
    dataset
        .iter()
        .map(|record| {
            by_id
                .get(record.sample_id.as_str())
                .map(|prediction| AlignedSample {
                    dataset: record,
                    prediction,
                })
                .ok_or_else(|| error(format!("missing prediction for {:?}", record.sample_id)))
        })
        .collect()
}

pub fn read_dataset_jsonl(reader: impl BufRead) -> Result<Vec<DatasetRecordV1>, ProtocolError> {
    read_jsonl(reader, "dataset")
}

pub fn write_dataset_jsonl(
    writer: impl Write,
    records: &[DatasetRecordV1],
) -> Result<(), ProtocolError> {
    write_jsonl(writer, records)
}

pub fn read_prediction_jsonl(
    reader: impl BufRead,
) -> Result<Vec<PredictionRecordV1>, ProtocolError> {
    read_jsonl(reader, "prediction")
}

pub fn write_prediction_jsonl(
    writer: impl Write,
    records: &[PredictionRecordV1],
) -> Result<(), ProtocolError> {
    write_jsonl(writer, records)
}

fn read_jsonl<T: DeserializeOwned + SampleRecord>(
    reader: impl BufRead,
    kind: &str,
) -> Result<Vec<T>, ProtocolError> {
    let mut records = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line =
            line.map_err(|source| error(format!("failed to read line {}: {source}", index + 1)))?;
        if line.trim().is_empty() {
            return fail(format!("blank {kind} JSONL line {}", index + 1));
        }
        records.push(serde_json::from_str(&line).map_err(|source| {
            error(format!("invalid {kind} JSONL line {}: {source}", index + 1))
        })?);
    }
    validate_records(&records)?;
    Ok(records)
}

fn write_jsonl<T: Serialize + SampleRecord>(
    mut writer: impl Write,
    records: &[T],
) -> Result<(), ProtocolError> {
    validate_records(records)?;
    for record in records {
        serde_json::to_writer(&mut writer, record)
            .map_err(|source| error(format!("failed to encode JSONL: {source}")))?;
        writer
            .write_all(b"\n")
            .map_err(|source| error(format!("failed to write JSONL: {source}")))?;
    }
    Ok(())
}

fn validate_records<T: SampleRecord>(records: &[T]) -> Result<(), ProtocolError> {
    if records.is_empty() {
        return fail("JSONL contains no records");
    }
    let mut ids = HashSet::with_capacity(records.len());
    for record in records {
        let id = record.sample_id();
        nonempty("sample_id", id)?;
        if id.chars().any(char::is_control) {
            return fail(format!("sample_id {id:?} contains a control character"));
        }
        let (field, edges) = record.edge_field();
        if edges.len() < record.minimum_edges() {
            return fail(format!(
                "sample {id:?} has {} {field}; at least {} required",
                edges.len(),
                record.minimum_edges()
            ));
        }
        if !ids.insert(id) {
            return fail(format!("duplicate sample_id {id:?}"));
        }
    }
    Ok(())
}

fn schema(kind: &str, actual: &str, expected: &str) -> Result<(), ProtocolError> {
    if actual == expected {
        Ok(())
    } else {
        fail(format!(
            "unsupported {kind} schema {actual:?}; expected {expected:?}"
        ))
    }
}

fn nonempty(field: &str, value: &str) -> Result<(), ProtocolError> {
    if value.trim().is_empty() {
        fail(format!("{field} must not be empty"))
    } else {
        Ok(())
    }
}

fn fail<T>(message: impl Into<String>) -> Result<T, ProtocolError> {
    Err(error(message))
}

fn error(message: impl Into<String>) -> ProtocolError {
    ProtocolError(message.into())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProtocolError(String);

impl Display for ProtocolError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ProtocolError {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn dataset() -> Vec<DatasetRecordV1> {
        [("test:1", vec![4, 8, 15]), ("test:2", vec![16, 23, 42])]
            .into_iter()
            .map(|(id, edges)| DatasetRecordV1 {
                sample_id: id.into(),
                original_edge_ids: edges,
            })
            .collect()
    }

    fn predictions() -> Vec<PredictionRecordV1> {
        [("test:1", vec![4, 8, 15]), ("test:2", vec![16, 42])]
            .into_iter()
            .map(|(id, edges)| PredictionRecordV1 {
                sample_id: id.into(),
                predicted_edge_ids: edges,
            })
            .collect()
    }

    #[test]
    fn jsonl_round_trips_and_predictions_stay_minimal() {
        let records = dataset();
        let mut encoded = Vec::new();
        write_dataset_jsonl(&mut encoded, &records).unwrap();
        assert_eq!(read_dataset_jsonl(Cursor::new(encoded)).unwrap(), records);

        let mut encoded = Vec::new();
        write_prediction_jsonl(&mut encoded, &predictions()[..1]).unwrap();
        assert_eq!(
            String::from_utf8(encoded).unwrap(),
            "{\"sample_id\":\"test:1\",\"predicted_edge_ids\":[4,8,15]}\n"
        );
    }

    #[test]
    fn strict_jsonl_validation_rejects_unknown_blank_empty_and_duplicate_rows() {
        for input in [
            "{\"sample_id\":\"x\",\"original_edge_ids\":[1],\"method\":\"x\"}\n",
            "{\"sample_id\":\"x\",\"original_edge_ids\":[1]}\n\n",
            "",
            "{\"sample_id\":\"x\",\"original_edge_ids\":[]}\n",
            "{\"sample_id\":\"x\",\"original_edge_ids\":[1]}\n",
            "{\"sample_id\":\"x\",\"original_edge_ids\":[1]}\n{\"sample_id\":\"x\",\"original_edge_ids\":[2]}\n",
        ] {
            assert!(read_dataset_jsonl(Cursor::new(input)).is_err(), "{input:?}");
        }
    }

    #[test]
    fn alignment_uses_dataset_order_and_requires_exact_ids() {
        let dataset = dataset();
        let mut predictions = predictions();
        predictions.reverse();
        let aligned = align_predictions(&dataset, &predictions).unwrap();
        assert_eq!(aligned[0].prediction.sample_id, "test:1");

        assert!(align_predictions(&dataset, &predictions[..1]).is_err());
        predictions.push(PredictionRecordV1 {
            sample_id: "extra".into(),
            predicted_edge_ids: vec![1],
        });
        assert!(align_predictions(&dataset, &predictions).is_err());
    }

    #[test]
    fn descriptor_and_receipt_validate_file_level_versions() {
        let manifest = DatasetManifestV1::new("chengdu/test", "roads-v1", "test.jsonl");
        manifest.validate().unwrap();
        for records_file in ["/tmp/test.jsonl", "../test.jsonl"] {
            let mut invalid = manifest.clone();
            invalid.records_file = records_file.into();
            assert!(invalid.validate().is_err());
        }
        let mut receipt = RunReceiptV1 {
            schema: RUN_RECEIPT_SCHEMA_V1.into(),
            method: MethodIdentity {
                name: "project-cch".into(),
                version: "0.1.0".into(),
            },
            dataset_id: manifest.dataset_id,
            dataset_manifest_sha256: "a".repeat(64),
            prediction_records_schema: PREDICTION_RECORD_SCHEMA_V1.into(),
            configuration: Map::new(),
            source_revision: "0123456789abcdef".into(),
            environment: BTreeMap::from([("rust".into(), "1.93.1".into())]),
        };
        receipt.validate().unwrap();
        receipt.prediction_records_schema = "ewr.prediction-record/v2".into();
        assert!(receipt.validate().is_err());
    }
}
