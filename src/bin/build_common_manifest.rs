use edge_weight_recovery::benchmark_data::{
    CommonManifestPolicy, build_common_manifest, write_common_audit, write_common_manifest,
};
use edge_weight_recovery::config::atomic_write;
use edge_weight_recovery::data::load_graph;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let Some(arguments) = Arguments::from_args()? else {
        return Ok(());
    };
    let protocol_bytes = std::fs::read(&arguments.protocol).map_err(|error| {
        format!(
            "failed to read protocol {}: {error}",
            arguments.protocol.display()
        )
    })?;
    let protocol: Value = serde_json::from_slice(&protocol_bytes)
        .map_err(|error| format!("failed to decode protocol: {error}"))?;
    let protocol_sha256 = sha256_bytes(&protocol_bytes);
    validate_protocol(&arguments, &protocol)?;

    if arguments.split == "test" {
        validate_test_unlock(&arguments, &protocol, &protocol_sha256)?;
        let receipt = arguments
            .test_receipt
            .as_ref()
            .ok_or_else(|| "--test-receipt is required for split=test".to_string())?;
        if receipt.exists() {
            return Err(format!(
                "test access receipt {} already exists; the frozen test list may only be decoded once",
                receipt.display()
            ));
        }
        write_receipt(
            receipt,
            &json!({
                "schema_version": 1,
                "status": "started",
                "protocol_sha256": protocol_sha256,
                "split": "test",
                "test_read": true,
            }),
        )?;
    }

    let graph = load_graph(&arguments.city)?;
    let policy = CommonManifestPolicy {
        minimum_edges: arguments.minimum_edges,
        maximum_selected: arguments.maximum_selected,
    };
    let (trips, audit) = build_common_manifest(
        &arguments.city,
        &arguments.split,
        &arguments.variant,
        &graph,
        policy,
    )?;
    let outputs = write_common_manifest(&trips, &arguments.manifest, &arguments.pickle)?;
    let audit_json = audit.as_json(
        &arguments.city,
        &arguments.split,
        &arguments.variant,
        policy,
    );
    let envelope = json!({
        "schema_version": 1,
        "protocol": {
            "path": arguments.protocol,
            "sha256": protocol_sha256,
            "status": protocol.pointer("/status"),
        },
        "common_manifest": audit_json,
    });
    write_common_audit(&arguments.audit, &envelope, &outputs)?;

    if arguments.split == "test" {
        let receipt = arguments.test_receipt.as_ref().unwrap();
        write_receipt(
            receipt,
            &json!({
                "schema_version": 1,
                "status": "completed",
                "protocol_sha256": protocol_sha256,
                "split": "test",
                "selected_records": trips.len(),
                "manifest": outputs.pointer("/manifest"),
                "audit_path": arguments.audit,
                "test_read": true,
            }),
        )?;
    }
    println!(
        "selected {} of {} eligible {} records into {}",
        trips.len(),
        audit.eligible_records,
        arguments.split,
        arguments.manifest.display()
    );
    Ok(())
}

fn validate_protocol(arguments: &Arguments, protocol: &Value) -> Result<(), String> {
    if protocol.pointer("/schema_version").and_then(Value::as_u64) != Some(1) {
        return Err("protocol schema_version must be one".to_string());
    }
    if protocol.pointer("/data/city").and_then(Value::as_str) != Some(arguments.city.as_str()) {
        return Err("command city differs from the protocol".to_string());
    }
    if protocol
        .pointer("/data/minimum_edges")
        .and_then(Value::as_u64)
        != Some(arguments.minimum_edges as u64)
    {
        return Err("command minimum_edges differs from the protocol".to_string());
    }
    let split_pointer = format!("/data/splits/{}", arguments.split);
    let split = protocol
        .pointer(&split_pointer)
        .ok_or_else(|| format!("protocol is missing {split_pointer}"))?;
    if split.pointer("/variant").and_then(Value::as_str) != Some(arguments.variant.as_str()) {
        return Err("command variant differs from the protocol".to_string());
    }
    let expected_maximum = split.pointer("/maximum_selected");
    let maximum_matches = match (expected_maximum, arguments.maximum_selected) {
        (Some(Value::Null), None) => true,
        (Some(value), Some(maximum)) => value.as_u64() == Some(maximum as u64),
        _ => false,
    };
    if !maximum_matches {
        return Err("command maximum_selected differs from the protocol".to_string());
    }
    Ok(())
}

fn validate_test_unlock(
    arguments: &Arguments,
    protocol: &Value,
    protocol_sha256: &str,
) -> Result<(), String> {
    if protocol.pointer("/status").and_then(Value::as_str) != Some("frozen_after_validation") {
        return Err("test requires protocol status=frozen_after_validation".to_string());
    }
    let path = arguments
        .test_unlock
        .as_ref()
        .ok_or_else(|| "--test-unlock is required for split=test".to_string())?;
    let unlock: Value = serde_json::from_slice(
        &std::fs::read(path)
            .map_err(|error| format!("failed to read test unlock {}: {error}", path.display()))?,
    )
    .map_err(|error| format!("failed to decode test unlock: {error}"))?;
    if unlock.pointer("/status").and_then(Value::as_str) != Some("test_unlocked")
        || unlock.pointer("/protocol_sha256").and_then(Value::as_str) != Some(protocol_sha256)
    {
        return Err("test unlock status or protocol hash is invalid".to_string());
    }
    let evidence_path = unlock
        .pointer("/validation_evidence/path")
        .and_then(Value::as_str)
        .ok_or_else(|| "test unlock lacks validation_evidence.path".to_string())?;
    let expected_hash = unlock
        .pointer("/validation_evidence/sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| "test unlock lacks validation_evidence.sha256".to_string())?;
    if sha256_file(Path::new(evidence_path))? != expected_hash {
        return Err("validation evidence hash does not match the test unlock".to_string());
    }
    Ok(())
}

fn write_receipt(path: &Path, value: &Value) -> Result<(), String> {
    let encoded = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("failed to encode test receipt: {error}"))?;
    atomic_write(path, &encoded)
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file =
        File::open(path).map_err(|error| format!("failed to open {}: {error}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

struct Arguments {
    city: String,
    split: String,
    variant: String,
    minimum_edges: usize,
    maximum_selected: Option<usize>,
    manifest: PathBuf,
    pickle: PathBuf,
    audit: PathBuf,
    protocol: PathBuf,
    test_unlock: Option<PathBuf>,
    test_receipt: Option<PathBuf>,
}

impl Arguments {
    fn from_args() -> Result<Option<Self>, String> {
        let arguments = std::env::args().skip(1).collect::<Vec<_>>();
        if arguments
            .iter()
            .any(|argument| argument == "--help" || argument == "-h")
        {
            println!(
                "Usage: build_common_manifest --city CITY --split train|validation|test --variant NAME \\\n                 --minimum-edges N --maximum-selected N|all --manifest PATH --pickle PATH \\\n                 --audit PATH --protocol PATH [--test-unlock PATH --test-receipt PATH]"
            );
            return Ok(None);
        }
        let mut values = std::collections::BTreeMap::new();
        let mut index = 0;
        while index < arguments.len() {
            let flag = arguments[index].clone();
            let value = arguments
                .get(index + 1)
                .ok_or_else(|| format!("missing value for {flag}"))?
                .clone();
            if !flag.starts_with("--") {
                return Err(format!("unexpected positional argument {flag:?}"));
            }
            if values.insert(flag.clone(), value).is_some() {
                return Err(format!("{flag} was provided more than once"));
            }
            index += 2;
        }
        let required = |flag: &str| {
            values
                .get(flag)
                .cloned()
                .ok_or_else(|| format!("missing {flag}"))
        };
        let split = required("--split")?;
        if !["train", "validation", "test"].contains(&split.as_str()) {
            return Err("--split must be train, validation, or test".to_string());
        }
        let maximum = required("--maximum-selected")?;
        let maximum_selected =
            if maximum == "all" {
                None
            } else {
                Some(maximum.parse::<usize>().map_err(|_| {
                    "--maximum-selected must be a positive integer or all".to_string()
                })?)
            };
        let optional_path = |flag: &str| values.get(flag).map(PathBuf::from);
        let known = [
            "--city",
            "--split",
            "--variant",
            "--minimum-edges",
            "--maximum-selected",
            "--manifest",
            "--pickle",
            "--audit",
            "--protocol",
            "--test-unlock",
            "--test-receipt",
        ];
        if let Some(unknown) = values.keys().find(|flag| !known.contains(&flag.as_str())) {
            return Err(format!("unknown argument {unknown}"));
        }
        Ok(Some(Self {
            city: required("--city")?,
            split,
            variant: required("--variant")?,
            minimum_edges: required("--minimum-edges")?
                .parse()
                .map_err(|_| "--minimum-edges must be an integer".to_string())?,
            maximum_selected,
            manifest: PathBuf::from(required("--manifest")?),
            pickle: PathBuf::from(required("--pickle")?),
            audit: PathBuf::from(required("--audit")?),
            protocol: PathBuf::from(required("--protocol")?),
            test_unlock: optional_path("--test-unlock"),
            test_receipt: optional_path("--test-receipt"),
        }))
    }
}
