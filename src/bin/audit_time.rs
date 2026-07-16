use edge_weight_recovery::config::atomic_write;
use edge_weight_recovery::data::{LoadedTrips, load_graph, load_trips};
use edge_weight_recovery::temporal::{
    TemporalTrainingConfig, TimeBucketSpec, civil_date_from_unix_day, estimate_baseline_model,
    local_hour, local_unix_day, sha256_file,
};
use serde_json::{Value, json};
use std::collections::BTreeSet;
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
    let config = TemporalTrainingConfig::load(&arguments.config)?;
    let bucket_spec = config.load_bucket_spec()?;
    verify_data(&config, "train", &config.train_variant)?;
    verify_data(&config, "validation", &config.validation_variant)?;
    let graph = load_graph(&config.city)?;
    let train = load_trips(&config.city, "train", &config.train_variant, &graph, None)?;
    let validation = load_trips(
        &config.city,
        "validation",
        &config.validation_variant,
        &graph,
        None,
    )?;
    for (split, available) in [
        ("train", train.report.available_samples),
        ("validation", validation.report.available_samples),
    ] {
        let pointer = format!("/data/{split}_identity/sample_count");
        if config.as_json().pointer(&pointer).and_then(Value::as_u64) != Some(available as u64) {
            return Err(format!(
                "{split} sample count does not match the declared identity"
            ));
        }
    }
    let baseline =
        estimate_baseline_model(&graph, &train.paths, &train.times, &bucket_spec, &config)?;

    let output = json!({
        "schema_version": 1,
        "purpose": "train_only_departure_time_and_trip_average_speed_audit",
        "configuration": arguments.config,
        "configuration_sha256": sha256_file(&arguments.config)?,
        "data_identity": {
            "city": config.city,
            "train": config.as_json().pointer("/data/train_identity"),
            "validation": config.as_json().pointer("/data/validation_identity"),
        },
        "timestamp_interpretation": {
            "unit": "unix_seconds",
            "timezone": bucket_spec.timezone,
            "utc_offset_seconds": bucket_spec.utc_offset_seconds,
            "evidence": "full-train MMDD keys are compared against UTC and UTC+8 civil dates",
            "time_bucket_selection_field": "start_time",
        },
        "time_bucket_specification": bucket_spec.as_json(),
        "splits": {
            "train": split_audit(&train, &bucket_spec),
            "validation": split_audit(&validation, &bucket_spec),
        },
        "travel_time_baseline": {
            "diagnostics": baseline.diagnostics,
            "estimated_from_split": "train",
            "validation_used": false,
            "test_used": false,
        },
        "interpretation_limits": [
            "timestamps cover only whole trips",
            "trip-average speed is a proxy assigned to traversed roads, not a per-edge speed observation",
            "the line graph still omits the first-edge cost by design"
        ],
        "test_read": false,
    });
    let bytes = serde_json::to_vec_pretty(&output)
        .map_err(|error| format!("failed to encode time audit: {error}"))?;
    atomic_write(&arguments.output, &bytes)
}

fn split_audit(loaded: &LoadedTrips, spec: &TimeBucketSpec) -> Value {
    let mut hour_counts = vec![0usize; 24];
    let mut bucket_counts = vec![0usize; spec.buckets.len()];
    let mut durations = Vec::with_capacity(loaded.times.len());
    let mut dates = BTreeSet::new();
    for &time in &loaded.times {
        let hour = local_hour(time.start_time, spec.utc_offset_seconds);
        hour_counts[hour as usize] += 1;
        bucket_counts[spec.bucket_index(time.start_time)] += 1;
        if let Some(duration) = time.duration_seconds() {
            durations.push(duration);
        }
        dates.insert(civil_date_from_unix_day(local_unix_day(
            time.start_time,
            spec.utc_offset_seconds,
        )));
    }
    durations.sort_unstable();
    let bucket_rows = spec
        .buckets
        .iter()
        .zip(bucket_counts)
        .map(|(bucket, samples)| {
            json!({
                "id": bucket.id,
                "start_hour": bucket.start_hour,
                "end_hour": bucket.end_hour,
                "samples": samples,
            })
        })
        .collect::<Vec<_>>();
    json!({
        "filtering": {
            "available": loaded.report.available_samples,
            "inspected": loaded.report.inspected_samples,
            "accepted": loaded.report.accepted_samples,
            "dropped": loaded.report.dropped_samples(),
            "empty": loaded.report.empty,
            "too_short": loaded.report.too_short,
            "out_of_bounds": loaded.report.out_of_bounds,
            "discontinuous": loaded.report.discontinuous,
            "cyclic": loaded.report.cyclic,
        },
        "raw_timestamp_evidence": {
            "samples": loaded.timestamp_evidence.timestamp_samples,
            "invalid_intervals": loaded.timestamp_evidence.invalid_intervals,
            "minimum_start_time": loaded.timestamp_evidence.minimum_start_time,
            "maximum_end_time": loaded.timestamp_evidence.maximum_end_time,
            "mmdd_keys": loaded.timestamp_evidence.mmdd_keys,
            "mmdd_matches_utc": loaded.timestamp_evidence.mmdd_matches_utc,
            "mmdd_matches_utc_plus_8": loaded.timestamp_evidence.mmdd_matches_utc_plus_8,
        },
        "accepted_timestamp_samples": loaded.times.len(),
        "local_dates": dates.iter().map(|&(year, month, day)| format!("{year:04}-{month:02}-{day:02}")).collect::<Vec<_>>(),
        "local_hour_counts": hour_counts,
        "duration_seconds": duration_summary(&durations),
        "time_buckets": bucket_rows,
    })
}

fn duration_summary(sorted: &[u64]) -> Value {
    if sorted.is_empty() {
        return Value::Null;
    }
    let quantile =
        |probability: f64| sorted[((sorted.len() - 1) as f64 * probability).round() as usize];
    json!({
        "minimum": quantile(0.0),
        "p01": quantile(0.01),
        "p10": quantile(0.1),
        "p50": quantile(0.5),
        "p90": quantile(0.9),
        "p99": quantile(0.99),
        "maximum": quantile(1.0),
        "mean": sorted.iter().map(|&value| value as f64).sum::<f64>() / sorted.len() as f64,
    })
}

fn verify_data(config: &TemporalTrainingConfig, split: &str, variant: &str) -> Result<(), String> {
    if split != "train" && split != "validation" {
        return Err("time audit may only read train or validation".to_string());
    }
    let pointer = format!("/data/{split}_identity");
    let identity = config
        .as_json()
        .pointer(&pointer)
        .ok_or_else(|| format!("configuration lacks {pointer}"))?;
    let expected = format!(
        "data/{}_data/preprocessed_{split}_trips_{variant}.pkl",
        config.city
    );
    if identity.pointer("/path").and_then(Value::as_str) != Some(expected.as_str()) {
        return Err(format!("declared {split} path does not match {expected}"));
    }
    let bytes = std::fs::metadata(&expected)
        .map_err(|error| format!("failed to inspect {expected}: {error}"))?
        .len();
    if identity.pointer("/bytes").and_then(Value::as_u64) != Some(bytes) {
        return Err(format!(
            "declared {split} byte length does not match {expected}"
        ));
    }
    let hash = sha256_file(Path::new(&expected))?;
    if identity.pointer("/sha256").and_then(Value::as_str) != Some(hash.as_str()) {
        return Err(format!(
            "declared {split} SHA-256 does not match {expected}"
        ));
    }
    Ok(())
}

struct Arguments {
    config: PathBuf,
    output: PathBuf,
}

impl Arguments {
    fn from_args() -> Result<Option<Self>, String> {
        let arguments = std::env::args().skip(1).collect::<Vec<_>>();
        if arguments
            .iter()
            .any(|argument| argument == "--help" || argument == "-h")
        {
            println!("Usage: audit_time --config PATH --output PATH");
            return Ok(None);
        }
        let mut config = None;
        let mut output = None;
        let mut index = 0;
        while index < arguments.len() {
            let flag = &arguments[index];
            let value = arguments
                .get(index + 1)
                .ok_or_else(|| format!("missing value for {flag}"))?;
            let slot = match flag.as_str() {
                "--config" => &mut config,
                "--output" => &mut output,
                _ => return Err(format!("unknown argument {flag:?}")),
            };
            if slot.replace(PathBuf::from(value)).is_some() {
                return Err(format!("{flag} was supplied more than once"));
            }
            index += 2;
        }
        Ok(Some(Self {
            config: config.ok_or_else(|| "missing --config PATH".to_string())?,
            output: output.ok_or_else(|| "missing --output PATH".to_string())?,
        }))
    }
}
