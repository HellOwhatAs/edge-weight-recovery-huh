use edge_weight_recovery::graph::{
    CyclePolicy, PathValidationReport, TripPath, compute_observed_edge_counts, group_paths_by_od,
    load_graph, load_trips,
};
use serde_json::{Value, json};
use std::path::Path;

#[derive(Debug)]
struct Args {
    city: String,
    validation_variant: String,
    train_variants: Vec<String>,
    output: String,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let graph = load_graph(&args.city)?;
    let threads = rayon::current_num_threads().max(1);
    let validation = load_trips(
        &args.city,
        "validation",
        &args.validation_variant,
        &graph,
        None,
        false,
        CyclePolicy::Drop,
    )?;
    if validation.paths.is_empty() {
        return Err("validation subset has no valid paths".to_string());
    }
    let validation_counts =
        compute_observed_edge_counts(&validation.paths, graph.tail.len(), threads);

    let mut scales = Vec::with_capacity(args.train_variants.len());
    for variant in &args.train_variants {
        let loaded = load_trips(
            &args.city,
            "train",
            variant,
            &graph,
            None,
            false,
            CyclePolicy::Drop,
        )?;
        if loaded.paths.is_empty() {
            return Err(format!("training variant {variant:?} has no valid paths"));
        }
        let observed = compute_observed_edge_counts(&loaded.paths, graph.tail.len(), threads);
        let stats = coverage_stats(
            variant,
            &loaded.paths,
            &loaded.report,
            &observed,
            &validation.paths,
            &validation_counts,
            graph.tail.len(),
        );
        println!(
            "AUDIT variant={} valid={} observed_edges={} coverage={:.3}% unseen_validation_routes={:.3}%",
            variant,
            stats["valid_routes"],
            stats["unique_observed_edges"],
            stats["graph_edge_coverage"].as_f64().unwrap_or(0.0) * 100.0,
            stats["validation"]["routes_with_unseen_edge_rate"]
                .as_f64()
                .unwrap_or(0.0)
                * 100.0,
        );
        scales.push(stats);
    }

    let result = json!({
        "schema_version": 1,
        "city": args.city,
        "graph": {
            "nodes": graph.x.len(),
            "directed_edges": graph.tail.len(),
        },
        "validation_variant": args.validation_variant,
        "validation_validation_report": report_json(&validation.report),
        "scales": scales,
    });
    if let Some(parent) = Path::new(&args.output).parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let bytes = serde_json::to_vec_pretty(&result)
        .map_err(|error| format!("failed to encode coverage JSON: {error}"))?;
    std::fs::write(&args.output, bytes)
        .map_err(|error| format!("failed to write {}: {error}", args.output))?;
    println!("WROTE {}", args.output);
    Ok(())
}

fn coverage_stats(
    variant: &str,
    paths: &[TripPath],
    report: &PathValidationReport,
    observed: &[u64],
    validation_paths: &[TripPath],
    validation_counts: &[u64],
    edge_count: usize,
) -> Value {
    let groups = group_paths_by_od(paths);
    let unique_observed_edges = observed.iter().filter(|&&count| count > 0).count();
    let mut positive_counts: Vec<u64> = observed
        .iter()
        .copied()
        .filter(|&count| count > 0)
        .collect();
    positive_counts.sort_unstable();
    let observed_le = |threshold| {
        positive_counts
            .iter()
            .filter(|&&count| count <= threshold)
            .count()
    };
    let graph_le = |threshold| observed.iter().filter(|&&count| count <= threshold).count();

    let mut lengths: Vec<usize> = paths.iter().map(|(_, path)| path.len()).collect();
    lengths.sort_unstable();
    let total_edges: usize = lengths.iter().sum();

    let validation_unique_edges = validation_counts.iter().filter(|&&count| count > 0).count();
    let unseen_validation_unique_edges = validation_counts
        .iter()
        .zip(observed)
        .filter(|(validation_count, train_count)| **validation_count > 0 && **train_count == 0)
        .count();
    let validation_edge_occurrences: u128 =
        validation_counts.iter().map(|&count| count as u128).sum();
    let unseen_validation_occurrences: u128 = validation_counts
        .iter()
        .zip(observed)
        .filter(|(_, train_count)| **train_count == 0)
        .map(|(&validation_count, _)| validation_count as u128)
        .sum();
    let routes_with_unseen = validation_paths
        .iter()
        .filter(|(_, path)| path.iter().any(|&edge| observed[edge] == 0))
        .count();

    json!({
        "train_variant": variant,
        "available_routes": report.available_samples,
        "inspected_routes": report.inspected_samples,
        "valid_routes": paths.len(),
        "cyclic_routes": report.cyclic,
        "discontinuous_routes": report.discontinuous,
        "out_of_bounds_routes": report.out_of_bounds,
        "unique_od": groups.len(),
        "od_query_reduction": 1.0 - groups.len() as f64 / paths.len() as f64,
        "total_observed_edge_occurrences": total_edges,
        "unique_observed_edges": unique_observed_edges,
        "graph_edge_coverage": unique_observed_edges as f64 / edge_count as f64,
        "positive_edge_count_quantiles": {
            "min": quantile_u64(&positive_counts, 0.0),
            "median": quantile_u64(&positive_counts, 0.5),
            "p75": quantile_u64(&positive_counts, 0.75),
            "p90": quantile_u64(&positive_counts, 0.90),
            "p95": quantile_u64(&positive_counts, 0.95),
            "max": quantile_u64(&positive_counts, 1.0),
        },
        "sparse_edges": {
            "observed_le_1": observed_le(1),
            "observed_le_2": observed_le(2),
            "observed_le_5": observed_le(5),
            "observed_le_1_rate_of_observed": ratio(observed_le(1), unique_observed_edges),
            "observed_le_2_rate_of_observed": ratio(observed_le(2), unique_observed_edges),
            "observed_le_5_rate_of_observed": ratio(observed_le(5), unique_observed_edges),
            "graph_edges_le_1_including_unseen": graph_le(1),
            "graph_edges_le_2_including_unseen": graph_le(2),
            "graph_edges_le_5_including_unseen": graph_le(5),
        },
        "route_length_edges": {
            "mean": total_edges as f64 / paths.len() as f64,
            "median": quantile_usize(&lengths, 0.5),
            "p90": quantile_usize(&lengths, 0.90),
            "max": quantile_usize(&lengths, 1.0),
        },
        "validation": {
            "valid_routes": validation_paths.len(),
            "unique_edges": validation_unique_edges,
            "unseen_unique_edges": unseen_validation_unique_edges,
            "unseen_unique_edge_rate": ratio(unseen_validation_unique_edges, validation_unique_edges),
            "edge_occurrences": validation_edge_occurrences.to_string(),
            "unseen_edge_occurrences": unseen_validation_occurrences.to_string(),
            "unseen_edge_occurrence_rate": if validation_edge_occurrences == 0 { 0.0 } else { unseen_validation_occurrences as f64 / validation_edge_occurrences as f64 },
            "routes_with_unseen_edge": routes_with_unseen,
            "routes_with_unseen_edge_rate": ratio(routes_with_unseen, validation_paths.len()),
        },
    })
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

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn quantile_u64(sorted: &[u64], probability: f64) -> u64 {
    sorted
        .get(quantile_index(sorted.len(), probability))
        .copied()
        .unwrap_or(0)
}

fn quantile_usize(sorted: &[usize], probability: f64) -> usize {
    sorted
        .get(quantile_index(sorted.len(), probability))
        .copied()
        .unwrap_or(0)
}

fn quantile_index(len: usize, probability: f64) -> usize {
    if len <= 1 {
        0
    } else {
        ((len - 1) as f64 * probability).round() as usize
    }
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut city = None;
    let mut validation_variant = None;
    let mut train_variants = Vec::new();
    let mut output = None;
    let mut index = 0;
    while index < raw.len() {
        let flag = &raw[index];
        let value = raw
            .get(index + 1)
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--city" => city = Some(value.clone()),
            "--validation-variant" => validation_variant = Some(value.clone()),
            "--train-variant" => train_variants.push(value.clone()),
            "--output" => output = Some(value.clone()),
            _ => return Err(format!("unknown argument {flag}")),
        }
        index += 2;
    }
    if train_variants.is_empty() {
        return Err("provide at least one --train-variant".to_string());
    }
    Ok(Args {
        city: city.ok_or_else(|| "missing --city".to_string())?,
        validation_variant: validation_variant
            .ok_or_else(|| "missing --validation-variant".to_string())?,
        train_variants,
        output: output.ok_or_else(|| "missing --output".to_string())?,
    })
}
