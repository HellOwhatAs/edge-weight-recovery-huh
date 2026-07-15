use edge_weight_recovery::evaluation::evaluate_detailed_paths;
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
    validation_variant: String,
    checkpoint: PathBuf,
    summary_output: PathBuf,
    routes_output: Option<PathBuf>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let started = Instant::now();
    let graph = load_graph(&args.city)?;
    let train = load_trips(
        &args.city,
        "train",
        &args.train_variant,
        &graph,
        None,
        false,
        CyclePolicy::Drop,
    )?;
    let validation = load_trips(
        &args.city,
        "validation",
        &args.validation_variant,
        &graph,
        None,
        false,
        CyclePolicy::Drop,
    )?;
    let checkpoint: Value = serde_json::from_slice(
        &std::fs::read(&args.checkpoint)
            .map_err(|error| format!("failed to read {}: {error}", args.checkpoint.display()))?,
    )
    .map_err(|error| format!("failed to decode {}: {error}", args.checkpoint.display()))?;
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
    let evaluation = evaluate_detailed_paths(&metric, &validation.paths, &train_counts, threads)?;

    let detailed = evaluation.to_json();
    if let Some(path) = &args.routes_output {
        write_json(
            path,
            &json!({
                "schema_version": 1,
                "city": args.city,
                "train_variant": args.train_variant,
                "validation_variant": args.validation_variant,
                "checkpoint_path": args.checkpoint,
                "checkpoint_metadata": checkpoint_metadata(&checkpoint),
                "train_validation_report": report_json(&train.report),
                "validation_validation_report": report_json(&validation.report),
                "evaluation": detailed,
            }),
        )?;
    }

    let mut summary = evaluation.to_json();
    if let Value::Object(object) = &mut summary {
        object.remove("routes");
    }
    let result = json!({
        "schema_version": 1,
        "city": args.city,
        "train_variant": args.train_variant,
        "validation_variant": args.validation_variant,
        "checkpoint_path": args.checkpoint,
        "checkpoint_metadata": checkpoint_metadata(&checkpoint),
        "train_validation_report": report_json(&train.report),
        "validation_validation_report": report_json(&validation.report),
        "evaluation": summary,
        "wall_seconds": started.elapsed().as_secs_f64(),
        "peak_rss_kib": process_peak_rss_kib(),
        "routes_output": args.routes_output,
    });
    write_json(&args.summary_output, &result)?;
    println!(
        "DIAGNOSTIC samples={} aggregate_relative_regret={:.8} mean_relative_regret={:.8} edge_f1={:.6} exact={:.6} wall_seconds={:.3}",
        evaluation.overall.sample_count,
        evaluation.overall.aggregate_relative_regret,
        evaluation.overall.mean_relative_regret,
        evaluation.overall.mean_edge_f1,
        evaluation.overall.exact_match_rate,
        started.elapsed().as_secs_f64(),
    );
    println!("WROTE {}", args.summary_output.display());
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
        "eta0",
        "lambda",
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
        "cyclic": report.cyclic,
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
    std::fs::write(path, bytes)
        .map_err(|error| format!("failed to write {}: {error}", path.display()))
}

fn process_peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut city = None;
    let mut train_variant = None;
    let mut validation_variant = None;
    let mut checkpoint = None;
    let mut summary_output = None;
    let mut routes_output = None;
    let mut index = 0;
    while index < raw.len() {
        let flag = &raw[index];
        let value = raw
            .get(index + 1)
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--city" => city = Some(value.clone()),
            "--train-variant" => train_variant = Some(value.clone()),
            "--validation-variant" => validation_variant = Some(value.clone()),
            "--checkpoint" => checkpoint = Some(PathBuf::from(value)),
            "--summary-output" => summary_output = Some(PathBuf::from(value)),
            "--routes-output" => routes_output = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown argument {flag}")),
        }
        index += 2;
    }
    Ok(Args {
        city: city.ok_or_else(|| "missing --city".to_string())?,
        train_variant: train_variant.ok_or_else(|| "missing --train-variant".to_string())?,
        validation_variant: validation_variant
            .ok_or_else(|| "missing --validation-variant".to_string())?,
        checkpoint: checkpoint.ok_or_else(|| "missing --checkpoint".to_string())?,
        summary_output: summary_output.ok_or_else(|| "missing --summary-output".to_string())?,
        routes_output,
    })
}
