use edge_weight_recovery::config::atomic_write;
use edge_weight_recovery::data::{GraphData, load_graph};
use edge_weight_recovery::optimizer::quantize_weights;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::env;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::process;

const LEGACY_SHA256: &str = "e9589ec899cd909430d3bf995e104cfeacb6f2463207b9ca7f4e54b666a2165c";
const ARCHIVE_COMMIT: &str = "8aacf2e8020bae13c6fad58f22ccb369f249e029";
const ARCHIVE_TAG: &str = "archive/pre-cleanup-convergence-study";
const BUNDLE_PATH: &str = "experiments/convergence_study/evidence/reproducibility_bundle.tar.gz";
const BUNDLE_SHA256: &str = "e8ba23b62e356afd3fdc8d2b5412e65cb62ba52e03a41b1322faa39e76f665d0";
const MEMBER_PATH: &str =
    "edge-weight-convergence-study/runs/convergence_full/conv_full_eta3e4/model_checkpoint.json";
const EDGE_COUNT: usize = 72_156;
const NODE_COUNT: usize = 31_199;
const BASELINE_FINGERPRINT: &str = "ad08fec01f56dd3c";
const Q_MIN: f64 = 0.1;
const Q_MAX: f64 = 10.0;

const USAGE: &str = "Usage:
  import-frozen-edge-checkpoint \\
    --legacy-checkpoint PATH \\
    --output PATH

Validates the byte-exact archived Beijing epoch-99 edge checkpoint against its
legacy configuration, latent q, quantized weights, and current road baseline,
then writes an atomic edge_initialization artifact. No training is performed.";

#[derive(Debug)]
struct Arguments {
    legacy_checkpoint: PathBuf,
    output: PathBuf,
}

enum ParseOutcome {
    Run(Arguments),
    Help,
}

fn main() {
    match parse_arguments().and_then(|outcome| match outcome {
        ParseOutcome::Run(arguments) => run(arguments),
        ParseOutcome::Help => {
            println!("{USAGE}");
            Ok(())
        }
    }) {
        Ok(()) => {}
        Err(error) => {
            eprintln!("error: {error}\n\n{USAGE}");
            process::exit(2);
        }
    }
}

fn parse_arguments() -> Result<ParseOutcome, String> {
    let mut legacy_checkpoint = None;
    let mut output = None;
    let mut arguments = env::args().skip(1);
    while let Some(flag) = arguments.next() {
        if flag == "--help" || flag == "-h" {
            return Ok(ParseOutcome::Help);
        }
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value after {flag}"))?;
        match flag.as_str() {
            "--legacy-checkpoint" => set_once(&mut legacy_checkpoint, PathBuf::from(value), &flag)?,
            "--output" => set_once(&mut output, PathBuf::from(value), &flag)?,
            _ => return Err(format!("unknown argument {flag:?}")),
        }
    }

    let legacy_checkpoint = legacy_checkpoint
        .ok_or_else(|| "required argument --legacy-checkpoint is missing".to_string())?;
    let output = output.ok_or_else(|| "required argument --output is missing".to_string())?;
    if legacy_checkpoint == output {
        return Err("--output must differ from --legacy-checkpoint".to_string());
    }
    Ok(ParseOutcome::Run(Arguments {
        legacy_checkpoint,
        output,
    }))
}

fn set_once<T>(slot: &mut Option<T>, value: T, flag: &str) -> Result<(), String> {
    if slot.replace(value).is_some() {
        return Err(format!("argument {flag} was provided more than once"));
    }
    Ok(())
}

fn run(arguments: Arguments) -> Result<(), String> {
    let legacy_bytes = std::fs::read(&arguments.legacy_checkpoint).map_err(|error| {
        format!(
            "failed to read {}: {error}",
            arguments.legacy_checkpoint.display()
        )
    })?;
    let actual_sha256 = sha256_hex(&legacy_bytes);
    require_equal(
        "legacy checkpoint SHA-256",
        actual_sha256.as_str(),
        LEGACY_SHA256,
    )?;

    let legacy: Value = serde_json::from_slice(&legacy_bytes).map_err(|error| {
        format!(
            "failed to decode {}: {error}",
            arguments.legacy_checkpoint.display()
        )
    })?;
    validate_legacy_contract(&legacy)?;

    let q = parse_q(&legacy)?;
    let saved_weights = parse_weights(&legacy)?;
    let graph = load_graph("beijing")?;
    validate_baseline(&graph)?;
    if q.len() != graph.baseline_weights.len()
        || saved_weights.len() != graph.baseline_weights.len()
    {
        return Err(format!(
            "checkpoint/graph length mismatch: q={}, weights={}, graph_edges={}",
            q.len(),
            saved_weights.len(),
            graph.baseline_weights.len()
        ));
    }

    let recomputed_weights = quantize_weights(&graph.baseline_weights, &q, 1.0)?;
    if let Some(edge) = recomputed_weights
        .iter()
        .zip(&saved_weights)
        .position(|(recomputed, saved)| recomputed != saved)
    {
        return Err(format!(
            "saved weight mismatch at edge {edge}: recomputed {}, saved {}",
            recomputed_weights[edge], saved_weights[edge]
        ));
    }

    let artifact = json!({
        "schema": "edge_initialization",
        "schema_version": 1,
        "model": "edge_only",
        "status": "frozen_validated",
        "completed_q_updates": 99,
        "source": {
            "archive_commit": ARCHIVE_COMMIT,
            "archive_tag": ARCHIVE_TAG,
            "bundle_path": BUNDLE_PATH,
            "bundle_sha256": BUNDLE_SHA256,
            "member_path": MEMBER_PATH,
            "legacy_checkpoint_sha256": LEGACY_SHA256,
        },
        "baseline_identity": {
            "city": "beijing",
            "nodes": NODE_COUNT,
            "edges": EDGE_COUNT,
            "fnv1a64": BASELINE_FINGERPRINT,
        },
        "source_training_data_identity": {
            "train": {
                "variant": "all",
                "path": "data/beijing_data/preprocessed_train_trips_all.pkl",
                "bytes": 120236049,
                "sha256": "d7fdfb5870c54df79d1044ecb12a076e0244dbd5d3bc74fd67d1bdcc2b7c0fce",
            },
            "validation": {
                "variant": "time_dev_20090513_excl_previous",
                "path": "data/beijing_data/preprocessed_validation_trips_time_dev_20090513_excl_previous.pkl",
                "bytes": 34101105,
                "sha256": "8dd462469b8c890944e31ce0856cb233af22644af8f985c5d4ea113d1736da03",
            },
            "test_read": false,
        },
        "source_optimizer": {
            "solver": "projected_subgradient",
            "eta0": 0.0003,
            "lambda_edge": 100000.0,
            "q_min": Q_MIN,
            "q_max": Q_MAX,
            "quantization_scale": 1.0,
        },
        "selection": {
            "split": "validation",
            "metric": "aggregate_relative_regret",
            "epoch": 99,
            "value": 0.06348409082193338,
        },
        "q": q,
        "quantized_metric_weights": saved_weights,
    });
    let output_bytes = serde_json::to_vec_pretty(&artifact)
        .map_err(|error| format!("failed to encode edge_initialization: {error}"))?;
    atomic_write(&arguments.output, &output_bytes)?;
    println!(
        "validated {} and wrote {} ({} q values, completed_q_updates=99)",
        arguments.legacy_checkpoint.display(),
        arguments.output.display(),
        EDGE_COUNT
    );
    Ok(())
}

fn validate_legacy_contract(legacy: &Value) -> Result<(), String> {
    for (pointer, expected) in [
        ("/schema_version", 1),
        ("/best_epoch", 99),
        ("/num_epochs", 100),
        ("/eval_every", 5),
        ("/patience", 4),
        ("/random_seed", 42),
    ] {
        require_u64(legacy, pointer, expected)?;
    }
    for (pointer, expected) in [
        ("/city", "beijing"),
        ("/train_variant", "all"),
        ("/validation_variant", "time_dev_20090513_excl_previous"),
        ("/solver", "ProjectedSubgradient"),
        ("/metric_update_mode", "Full"),
        ("/selection_metric", "RelativeRegret"),
        ("/cycle_policy", "Drop"),
        ("/test_variant", "all"),
    ] {
        require_str(legacy, pointer, expected)?;
    }
    for (pointer, expected) in [
        ("/eta0", 0.0003),
        ("/lambda", 100000.0),
        ("/q_min", Q_MIN),
        ("/q_max", Q_MAX),
        ("/quantization_scale", 1.0),
        ("/early_stop_min_delta", 0.00001),
        ("/selection_loss", 0.06348409082193338),
        ("/train_data_loss", 322081.5822566283),
    ] {
        require_f64(legacy, pointer, expected)?;
    }
    for (pointer, expected) in [
        ("/run_test", false),
        ("/trim_boundary_edges", false),
        ("/eval_path_metrics", true),
    ] {
        require_bool(legacy, pointer, expected)?;
    }
    for pointer in [
        "/max_train_samples",
        "/max_validation_samples",
        "/max_test_samples",
    ] {
        if legacy.pointer(pointer) != Some(&Value::Null) {
            return Err(format!("legacy field {pointer} must be null"));
        }
    }
    Ok(())
}

fn parse_q(legacy: &Value) -> Result<Vec<f64>, String> {
    let values = legacy
        .pointer("/multipliers")
        .and_then(Value::as_array)
        .ok_or_else(|| "legacy checkpoint is missing multipliers array".to_string())?;
    if values.len() != EDGE_COUNT {
        return Err(format!(
            "legacy checkpoint has {} multipliers, expected {EDGE_COUNT}",
            values.len()
        ));
    }
    values
        .iter()
        .enumerate()
        .map(|(edge, value)| {
            let q = value
                .as_f64()
                .ok_or_else(|| format!("multipliers[{edge}] is not a number"))?;
            if !q.is_finite() || !(Q_MIN..=Q_MAX).contains(&q) {
                return Err(format!(
                    "multipliers[{edge}]={q} is outside the finite [{Q_MIN},{Q_MAX}] box"
                ));
            }
            Ok(q)
        })
        .collect()
}

fn parse_weights(legacy: &Value) -> Result<Vec<u32>, String> {
    let values = legacy
        .pointer("/weights")
        .and_then(Value::as_array)
        .ok_or_else(|| "legacy checkpoint is missing weights array".to_string())?;
    if values.len() != EDGE_COUNT {
        return Err(format!(
            "legacy checkpoint has {} weights, expected {EDGE_COUNT}",
            values.len()
        ));
    }
    values
        .iter()
        .enumerate()
        .map(|(edge, value)| {
            let weight = value
                .as_u64()
                .ok_or_else(|| format!("weights[{edge}] is not an unsigned integer"))?;
            let weight =
                u32::try_from(weight).map_err(|_| format!("weights[{edge}] does not fit u32"))?;
            if weight == 0 || weight >= i32::MAX as u32 {
                return Err(format!(
                    "weights[{edge}]={weight} is not a positive finite CCH weight"
                ));
            }
            Ok(weight)
        })
        .collect()
}

fn validate_baseline(graph: &GraphData) -> Result<(), String> {
    if graph.x.len() != NODE_COUNT || graph.tail.len() != EDGE_COUNT {
        return Err(format!(
            "Beijing baseline size mismatch: nodes={}, edges={}, expected nodes={NODE_COUNT}, edges={EDGE_COUNT}",
            graph.x.len(),
            graph.tail.len()
        ));
    }
    let fingerprint = baseline_fingerprint(graph);
    require_equal(
        "Beijing baseline FNV-1a fingerprint",
        fingerprint.as_str(),
        BASELINE_FINGERPRINT,
    )
}

fn baseline_fingerprint(graph: &GraphData) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for value in graph
        .tail
        .iter()
        .chain(&graph.head)
        .chain(&graph.baseline_weights)
    {
        for byte in value.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    format!("{hash:016x}")
}

fn require_u64(document: &Value, pointer: &str, expected: u64) -> Result<(), String> {
    let actual = document
        .pointer(pointer)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("legacy field {pointer} is not an unsigned integer"))?;
    require_equal(pointer, &actual, &expected)
}

fn require_f64(document: &Value, pointer: &str, expected: f64) -> Result<(), String> {
    let actual = document
        .pointer(pointer)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("legacy field {pointer} is not a number"))?;
    require_equal(pointer, &actual, &expected)
}

fn require_bool(document: &Value, pointer: &str, expected: bool) -> Result<(), String> {
    let actual = document
        .pointer(pointer)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("legacy field {pointer} is not a Boolean"))?;
    require_equal(pointer, &actual, &expected)
}

fn require_str(document: &Value, pointer: &str, expected: &str) -> Result<(), String> {
    let actual = document
        .pointer(pointer)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("legacy field {pointer} is not a string"))?;
    require_equal(pointer, actual, expected)
}

fn require_equal<T>(label: &str, actual: &T, expected: &T) -> Result<(), String>
where
    T: std::fmt::Debug + PartialEq + ?Sized,
{
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "{label} mismatch: got {actual:?}, expected {expected:?}"
        ))
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut result = String::with_capacity(64);
    for byte in digest {
        write!(&mut result, "{byte:02x}").expect("writing into String cannot fail");
    }
    result
}
