use edge_weight_recovery::divergence::analyze_first_divergence;
use edge_weight_recovery::graph::{
    CyclePolicy, PathValidationReport, compute_observed_edge_counts, load_graph, load_trips,
};
use routingkit_cch::{CCH, CCHMetric, compute_order_inertial};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug)]
struct Args {
    city: String,
    train_variant: String,
    train_cycle_policy: CyclePolicy,
    validation_variant: String,
    checkpoint: PathBuf,
    summary_output: PathBuf,
    routes_output: Option<PathBuf>,
    max_train_samples: Option<usize>,
    max_validation_samples: Option<usize>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let Some(args) = parse_args()? else {
        print_help();
        return Ok(());
    };
    let started = Instant::now();
    let graph = load_graph(&args.city)?;
    let train = load_trips(
        &args.city,
        "train",
        &args.train_variant,
        &graph,
        args.max_train_samples,
        false,
        args.train_cycle_policy,
    )?;
    if train.paths.is_empty() {
        return Err("no valid training paths remain after validation".to_string());
    }
    let validation = load_trips(
        &args.city,
        "validation",
        &args.validation_variant,
        &graph,
        args.max_validation_samples,
        false,
        CyclePolicy::Drop,
    )?;
    if validation.paths.is_empty() {
        return Err("no valid validation paths remain after validation".to_string());
    }

    let checkpoint: Value = serde_json::from_slice(
        &std::fs::read(&args.checkpoint)
            .map_err(|error| format!("failed to read {}: {error}", args.checkpoint.display()))?,
    )
    .map_err(|error| format!("failed to decode {}: {error}", args.checkpoint.display()))?;
    if let Some(checkpoint_city) = checkpoint.get("city").and_then(Value::as_str)
        && checkpoint_city != args.city
    {
        return Err(format!(
            "checkpoint city {checkpoint_city:?} does not match requested city {:?}",
            args.city
        ));
    }
    let weights: Vec<u32> = serde_json::from_value(
        checkpoint
            .get("weights")
            .cloned()
            .ok_or_else(|| "checkpoint has no weights field".to_string())?,
    )
    .map_err(|error| format!("invalid checkpoint weights: {error}"))?;
    if weights.len() != graph.tail.len() {
        return Err(format!(
            "checkpoint has {} weights but graph has {} edges",
            weights.len(),
            graph.tail.len()
        ));
    }

    let checkpoint_train_variant = checkpoint.get("train_variant").and_then(Value::as_str);
    let train_variant_matches_checkpoint = checkpoint_train_variant
        .map(|variant| variant == args.train_variant)
        .unwrap_or(false);
    if checkpoint_train_variant.is_some() && !train_variant_matches_checkpoint {
        eprintln!(
            "warning: requested train variant {:?} differs from checkpoint train variant {:?}; unseen-edge strata describe the requested variant",
            args.train_variant, checkpoint_train_variant
        );
    }

    let threads = rayon::current_num_threads().max(1);
    let train_counts = compute_observed_edge_counts(&train.paths, graph.tail.len(), threads);
    let order = compute_order_inertial(
        graph.x.len() as u32,
        &graph.tail,
        &graph.head,
        &graph.x,
        &graph.y,
    );
    let cch = CCH::new(&order, &graph.tail, &graph.head, |_| {}, false);
    let metric = CCHMetric::new(&cch, weights);
    let analysis =
        analyze_first_divergence(&metric, &graph, &validation.paths, &train_counts, threads)?;

    let common_metadata = json!({
        "schema_version": 1,
        "analysis_kind": "first_route_divergence",
        "data_policy": {
            "training_performed": false,
            "loaded_splits": ["train", "validation"],
            "test_loaded": false,
        },
        "city": args.city,
        "train_variant": args.train_variant,
        "train_cycle_policy": format!("{:?}", args.train_cycle_policy),
        "evaluation_cycle_policy": "Drop",
        "validation_variant": args.validation_variant,
        "max_train_samples": args.max_train_samples,
        "max_validation_samples": args.max_validation_samples,
        "checkpoint_path": args.checkpoint,
        "checkpoint_metadata": checkpoint_metadata(&checkpoint),
        "train_variant_matches_checkpoint": train_variant_matches_checkpoint,
        "train_validation_report": report_json(&train.report),
        "validation_validation_report": report_json(&validation.report),
        "rayon_threads": threads,
    });

    if let Some(path) = &args.routes_output {
        write_json(
            path,
            &json!({
                "metadata": common_metadata,
                "analysis": analysis.to_json(true),
            }),
        )?;
    }
    let result = json!({
        "metadata": common_metadata,
        "analysis": analysis.to_json(false),
        "runtime": {
            "wall_seconds": started.elapsed().as_secs_f64(),
            "peak_rss_kib": process_peak_rss_kib(),
        },
        "routes_output": args.routes_output,
    });
    write_json(&args.summary_output, &result)?;

    let divergent = analysis.overall.divergent_routes.as_ref();
    println!(
        "DIVERGENCE samples={} divergence_rate={:.6} first_choice_accuracy={:.6} prefix_match_ratio={:.6} edge_f1={:.6} complex_at_divergence_rate={} rejoin_rate={} wall_seconds={:.3}",
        analysis.overall.sample_count,
        analysis.overall.divergence_rate,
        analysis.overall.first_choice_accuracy,
        analysis.overall.mean_prefix_match_ratio,
        analysis.overall.mean_edge_f1,
        optional_decimal(divergent.map(|summary| summary.complex_junction_rate)),
        optional_decimal(divergent.map(|summary| summary.rejoin_before_target_rate)),
        started.elapsed().as_secs_f64(),
    );
    println!("WROTE_SUMMARY {}", args.summary_output.display());
    if let Some(path) = &args.routes_output {
        println!("WROTE_ROUTES {}", path.display());
    }
    Ok(())
}

fn checkpoint_metadata(checkpoint: &Value) -> Value {
    let keys = [
        "schema_version",
        "city",
        "train_variant",
        "validation_variant",
        "solver",
        "selection_metric",
        "metric_update_mode",
        "train_cycle_policy",
        "evaluation_cycle_policy",
        "eta0",
        "lambda",
        "q_min",
        "q_max",
        "best_epoch",
        "selection_loss",
    ];
    Value::Object(
        keys.into_iter()
            .filter_map(|key| {
                checkpoint
                    .get(key)
                    .cloned()
                    .map(|value| (key.to_string(), value))
            })
            .collect(),
    )
}

fn report_json(report: &PathValidationReport) -> Value {
    json!({
        "available": report.available_samples,
        "inspected": report.inspected_samples,
        "accepted": report.accepted_samples,
        "trimmed_boundary_edges": report.trimmed_boundary_edges,
        "cyclic": report.cyclic,
        "cycle_erased_records": report.cycle_erased_records,
        "empty_after_cycle_transform": report.empty_after_cycle_transform,
        "cycle_edges_removed": report.cycle_edges_removed,
        "discontinuous": report.discontinuous,
        "out_of_bounds": report.out_of_bounds,
        "empty_or_too_short": report.empty_or_too_short,
    })
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("failed to encode JSON: {error}"))?;
    let temporary = PathBuf::from(format!("{}.{}.tmp", path.display(), std::process::id()));
    std::fs::write(&temporary, bytes)
        .map_err(|error| format!("failed to write {}: {error}", temporary.display()))?;
    std::fs::rename(&temporary, path).map_err(|error| {
        format!(
            "failed to atomically replace {} with {}: {error}",
            path.display(),
            temporary.display()
        )
    })
}

fn process_peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

fn optional_decimal(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.6}"))
        .unwrap_or_else(|| "null".to_string())
}

fn parse_args() -> Result<Option<Args>, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw
        .iter()
        .any(|argument| argument == "--help" || argument == "-h")
    {
        return Ok(None);
    }
    let mut city = None;
    let mut train_variant = None;
    let mut train_cycle_policy = CyclePolicy::Drop;
    let mut validation_variant = None;
    let mut checkpoint = None;
    let mut summary_output = None;
    let mut routes_output = None;
    let mut max_train_samples = None;
    let mut max_validation_samples = None;
    let mut index = 0;
    while index < raw.len() {
        let flag = &raw[index];
        let value = raw
            .get(index + 1)
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--city" => city = Some(value.clone()),
            "--train-variant" => train_variant = Some(value.clone()),
            "--train-cycle-policy" => train_cycle_policy = value.parse::<CyclePolicy>()?,
            "--validation-variant" => validation_variant = Some(value.clone()),
            "--checkpoint" => checkpoint = Some(PathBuf::from(value)),
            "--summary-output" => summary_output = Some(PathBuf::from(value)),
            "--routes-output" => routes_output = Some(PathBuf::from(value)),
            "--max-train-samples" => max_train_samples = Some(parse_positive(value, flag)?),
            "--max-validation-samples" => {
                max_validation_samples = Some(parse_positive(value, flag)?)
            }
            _ => return Err(format!("unknown argument {flag}; use --help")),
        }
        index += 2;
    }
    Ok(Some(Args {
        city: city.ok_or_else(|| "missing --city".to_string())?,
        train_variant: train_variant.ok_or_else(|| "missing --train-variant".to_string())?,
        train_cycle_policy,
        validation_variant: validation_variant
            .ok_or_else(|| "missing --validation-variant".to_string())?,
        checkpoint: checkpoint.ok_or_else(|| "missing --checkpoint".to_string())?,
        summary_output: summary_output.ok_or_else(|| "missing --summary-output".to_string())?,
        routes_output,
        max_train_samples,
        max_validation_samples,
    }))
}

fn parse_positive(value: &str, flag: &str) -> Result<usize, String> {
    let parsed: usize = value
        .parse()
        .map_err(|error| format!("invalid value {value:?} for {flag}: {error}"))?;
    if parsed == 0 {
        return Err(format!("{flag} must be positive"));
    }
    Ok(parsed)
}

fn print_help() {
    println!(
        "Usage: cargo run --release --example analyze_divergence -- \\\n  --city CITY --train-variant VARIANT --validation-variant VARIANT \\\n  --checkpoint PATH --summary-output PATH [--routes-output PATH] \\\n  [--train-cycle-policy drop|keep|erase] \\\n  [--max-train-samples N] [--max-validation-samples N]\n\n\
This is a read-only checkpoint diagnostic. It loads only train and validation, \
never test, and performs no optimization. The cycle-policy option affects only \
training seen-edge counts; validation always drops cyclic records. Use \
--routes-output to retain every observed/predicted next-edge decision at the \
first divergence."
    );
}
