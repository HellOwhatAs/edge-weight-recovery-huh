//! Frozen, fixed-edge capacity probe for one non-negative global left-turn penalty.
//!
//! The executable accepts data/model identities only through the protocol. It
//! deliberately has no split, variant, city, or checkpoint CLI overrides.

use edge_weight_recovery::graph::{CyclePolicy, GraphData, PathValidationReport, TripPath};
use edge_weight_recovery::turn_graph::{
    CCH_INFINITY, ExpandedTurnGraph, MAX_EXPANDED_ARCS, MAX_RAW_EXPANDED_BYTES, expanded_path_cost,
    median_weight, query_expanded_path, scaled_left_penalty,
};
use rayon::prelude::*;
use routingkit_cch::{CCH, CCHMetric, CCHQuery, compute_order_inertial};
use serde_json::{Value, json};
use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const SELECTION_TIE: f64 = 1e-5;

#[derive(Debug)]
struct Args {
    protocol: PathBuf,
    output: PathBuf,
    max_routes: Option<usize>,
    max_grid_values: Option<usize>,
}

#[derive(Debug)]
struct Protocol {
    raw: Value,
    raw_sha256: String,
    checkpoint: PathBuf,
    checkpoint_sha256: String,
    validation_variant: String,
    selection_seed: u64,
    tune_routes: usize,
    audit_routes: usize,
    r_grid: Vec<f64>,
    bootstrap_replicates: usize,
    bootstrap_seed: u64,
    regret_improvement_threshold: f64,
    f1_improvement_threshold: f64,
    exact_decrease_limit: f64,
    timeout: Duration,
}

#[derive(Clone, Debug)]
struct RouteMetric {
    valid_route_index: usize,
    observed_cost: u64,
    shortest_cost: u32,
    raw_regret: u64,
    relative_regret: f64,
    edge_f1: f64,
    exact_match: bool,
}

#[derive(Clone, Copy, Debug)]
struct MetricSummary {
    sample_count: usize,
    total_observed_cost: u128,
    total_shortest_cost: u128,
    total_regret: u128,
    aggregate_relative_regret: f64,
    mean_relative_regret: f64,
    mean_edge_f1: f64,
    exact_match_rate: f64,
}

impl MetricSummary {
    fn from_routes(routes: &[RouteMetric]) -> Self {
        let count = routes.len();
        let total_observed_cost = routes.iter().map(|route| route.observed_cost as u128).sum();
        let total_shortest_cost = routes.iter().map(|route| route.shortest_cost as u128).sum();
        let total_regret = routes.iter().map(|route| route.raw_regret as u128).sum();
        Self {
            sample_count: count,
            total_observed_cost,
            total_shortest_cost,
            total_regret,
            aggregate_relative_regret: ratio(total_regret, total_observed_cost),
            mean_relative_regret: mean(routes.iter().map(|route| route.relative_regret), count),
            mean_edge_f1: mean(routes.iter().map(|route| route.edge_f1), count),
            exact_match_rate: mean(
                routes
                    .iter()
                    .map(|route| if route.exact_match { 1.0 } else { 0.0 }),
                count,
            ),
        }
    }

    fn to_json(self) -> Value {
        json!({
            "sample_count": self.sample_count,
            "total_observed_cost": self.total_observed_cost.to_string(),
            "total_shortest_cost": self.total_shortest_cost.to_string(),
            "total_regret": self.total_regret.to_string(),
            "aggregate_relative_regret": self.aggregate_relative_regret,
            "mean_relative_regret": self.mean_relative_regret,
            "mean_edge_f1": self.mean_edge_f1,
            "exact_match_rate": self.exact_match_rate,
        })
    }
}

#[derive(Debug)]
struct GridReport {
    r: f64,
    penalty: u32,
    weight_generation_seconds: f64,
    customization_seconds: f64,
    query_seconds: f64,
    min_arc_weight: u32,
    max_arc_weight: u32,
    rss_after_mib: Option<f64>,
    peak_rss_mib: Option<f64>,
    summary: MetricSummary,
}

impl GridReport {
    fn to_json(&self) -> Value {
        json!({
            "r": self.r,
            "left_penalty_u32": self.penalty,
            "metrics": self.summary.to_json(),
            "timing_seconds": {
                "weight_generation": self.weight_generation_seconds,
                "full_customization": self.customization_seconds,
                "tune_queries_and_decoding": self.query_seconds,
            },
            "expanded_arc_weight_range": [self.min_arc_weight, self.max_arc_weight],
            "rss_after_mib": self.rss_after_mib,
            "peak_rss_mib": self.peak_rss_mib,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct Interval {
    point: f64,
    lower: f64,
    upper: f64,
}

impl Interval {
    fn to_json(self) -> Value {
        json!({
            "point_estimate": self.point,
            "percentile_95_interval": [self.lower, self.upper],
        })
    }
}

#[derive(Debug)]
struct AlphaZeroGate {
    routes_checked: usize,
    distance_mismatches: usize,
    observed_cost_mismatches: usize,
    maximum_distance_absolute_difference: u64,
    maximum_observed_cost_absolute_difference: u64,
    query_seconds: f64,
}

impl AlphaZeroGate {
    fn passed(&self) -> bool {
        self.distance_mismatches == 0 && self.observed_cost_mismatches == 0
    }

    fn to_json(&self) -> Value {
        json!({
            "passed": self.passed(),
            "routes_checked": self.routes_checked,
            "expanded_vs_original_distance_mismatches": self.distance_mismatches,
            "expanded_vs_original_observed_cost_mismatches": self.observed_cost_mismatches,
            "maximum_distance_absolute_difference": self.maximum_distance_absolute_difference,
            "maximum_observed_cost_absolute_difference": self.maximum_observed_cost_absolute_difference,
            "original_cch_query_seconds": self.query_seconds,
        })
    }
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
    reject_test_text("protocol path", &args.protocol.to_string_lossy())?;
    let started = Instant::now();
    let protocol = load_protocol(&args.protocol)?;
    let formal_run = args.max_routes.is_none() && args.max_grid_values.is_none();
    let tune_count = args
        .max_routes
        .unwrap_or(protocol.tune_routes)
        .min(protocol.tune_routes);
    let audit_count = args
        .max_routes
        .unwrap_or(protocol.audit_routes)
        .min(protocol.audit_routes);
    if tune_count == 0 || audit_count == 0 {
        return Err("tune and audit route counts must both be positive".to_string());
    }
    let grid_count = args
        .max_grid_values
        .unwrap_or(protocol.r_grid.len())
        .min(protocol.r_grid.len());
    if grid_count == 0 {
        return Err("at least one r grid value is required".to_string());
    }
    let r_grid = &protocol.r_grid[..grid_count];
    if r_grid[0] != 0.0 {
        return Err("the executed grid must start with r=0".to_string());
    }

    progress("verify checkpoint");
    let checkpoint_bytes = fs::read(&protocol.checkpoint).map_err(|error| {
        format!(
            "failed to read frozen checkpoint {}: {error}",
            protocol.checkpoint.display()
        )
    })?;
    let actual_checkpoint_sha256 = hex_digest(&sha256(&checkpoint_bytes));
    if actual_checkpoint_sha256 != protocol.checkpoint_sha256 {
        return Err(format!(
            "frozen checkpoint SHA-256 mismatch: expected {}, got {}",
            protocol.checkpoint_sha256, actual_checkpoint_sha256
        ));
    }
    let checkpoint: Value = serde_json::from_slice(&checkpoint_bytes)
        .map_err(|error| format!("failed to decode frozen checkpoint JSON: {error}"))?;
    let city = required_str(&checkpoint, "/city")?.to_string();
    reject_test_text("checkpoint city", &city)?;
    let weights: Vec<u32> = serde_json::from_value(
        checkpoint
            .get("weights")
            .cloned()
            .ok_or_else(|| "checkpoint has no weights array".to_string())?,
    )
    .map_err(|error| format!("invalid checkpoint weights: {error}"))?;
    check_timeout(started, protocol.timeout, "checkpoint verification")?;

    progress("load graph and validation only");
    let load_started = Instant::now();
    let graph = edge_weight_recovery::graph::load_graph(&city)?;
    if weights.len() != graph.tail.len() {
        return Err(format!(
            "checkpoint has {} weights but graph has {} edges",
            weights.len(),
            graph.tail.len()
        ));
    }
    let validation = edge_weight_recovery::graph::load_trips(
        &city,
        "validation",
        &protocol.validation_variant,
        &graph,
        None,
        false,
        CyclePolicy::Drop,
    )?;
    let data_load_seconds = load_started.elapsed().as_secs_f64();
    if validation.paths.len() < tune_count + audit_count {
        return Err(format!(
            "only {} valid routes are available, fewer than requested {} tune + {} audit",
            validation.paths.len(),
            tune_count,
            audit_count
        ));
    }
    let ranked = ranked_route_indices(validation.paths.len(), protocol.selection_seed);
    let tune_indices = ranked[..tune_count].to_vec();
    let audit_indices = ranked[tune_count..tune_count + audit_count].to_vec();
    check_timeout(
        started,
        protocol.timeout,
        "validation loading and route selection",
    )?;

    progress("build original and expanded CCH indices");
    let original_order_started = Instant::now();
    let original_order = compute_order_inertial(
        graph.x.len() as u32,
        &graph.tail,
        &graph.head,
        &graph.x,
        &graph.y,
    );
    let original_order_seconds = original_order_started.elapsed().as_secs_f64();
    let original_cch_started = Instant::now();
    let original_cch = CCH::new(&original_order, &graph.tail, &graph.head, |_| {}, false);
    let original_cch_seconds = original_cch_started.elapsed().as_secs_f64();
    let original_customize_started = Instant::now();
    let original_metric = CCHMetric::new(&original_cch, weights.clone());
    let original_customize_seconds = original_customize_started.elapsed().as_secs_f64();

    let line_graph_started = Instant::now();
    let expanded = ExpandedTurnGraph::build(&graph)?;
    let line_graph_seconds = line_graph_started.elapsed().as_secs_f64();
    let expanded_order_started = Instant::now();
    let expanded_order = compute_order_inertial(
        expanded.stats.expanded_nodes as u32,
        &expanded.tail,
        &expanded.head,
        &expanded.state_x,
        &expanded.state_y,
    );
    let expanded_order_seconds = expanded_order_started.elapsed().as_secs_f64();
    let expanded_cch_started = Instant::now();
    let expanded_cch = CCH::new(
        &expanded_order,
        &expanded.tail,
        &expanded.head,
        |_| {},
        false,
    );
    let expanded_cch_seconds = expanded_cch_started.elapsed().as_secs_f64();
    check_timeout(started, protocol.timeout, "CCH construction")?;

    let kappa = median_weight(&weights)?;
    let mut grid_reports = Vec::with_capacity(r_grid.len());
    let mut r0_tune_routes = None;
    for &r in r_grid {
        progress(&format!("customize and evaluate tune grid r={r}"));
        let penalty = scaled_left_penalty(kappa, r)?;
        let weight_started = Instant::now();
        let arc_weights = expanded.customized_arc_weights(&weights, penalty)?;
        let weight_generation_seconds = weight_started.elapsed().as_secs_f64();
        let min_arc_weight = arc_weights.iter().copied().min().unwrap_or(0);
        let max_arc_weight = arc_weights.iter().copied().max().unwrap_or(0);
        let customize_started = Instant::now();
        let metric = CCHMetric::new(&expanded_cch, arc_weights);
        let customization_seconds = customize_started.elapsed().as_secs_f64();
        let query_started = Instant::now();
        let routes = evaluate_routes(
            &metric,
            &expanded,
            &graph,
            &weights,
            penalty,
            &validation.paths,
            &tune_indices,
        )?;
        let query_seconds = query_started.elapsed().as_secs_f64();
        let summary = MetricSummary::from_routes(&routes);
        if r == 0.0 {
            r0_tune_routes = Some(routes);
        }
        grid_reports.push(GridReport {
            r,
            penalty,
            weight_generation_seconds,
            customization_seconds,
            query_seconds,
            min_arc_weight,
            max_arc_weight,
            rss_after_mib: memory_mib("VmRSS:"),
            peak_rss_mib: memory_mib("VmHWM:"),
            summary,
        });
        check_timeout(started, protocol.timeout, &format!("tune grid r={r}"))?;
    }
    let selected_index = select_grid(&grid_reports);
    let selected_r = grid_reports[selected_index].r;
    let selected_penalty = grid_reports[selected_index].penalty;

    progress(&format!("one-shot audit r=0 and selected r={selected_r}"));
    let audit_r0 = evaluate_model(
        &expanded_cch,
        &expanded,
        &graph,
        &weights,
        0,
        &validation.paths,
        &audit_indices,
    )?;
    let audit_selected = if selected_penalty == 0 {
        EvaluatedModel {
            routes: audit_r0.routes.clone(),
            weight_generation_seconds: 0.0,
            customization_seconds: 0.0,
            query_seconds: 0.0,
        }
    } else {
        evaluate_model(
            &expanded_cch,
            &expanded,
            &graph,
            &weights,
            selected_penalty,
            &validation.paths,
            &audit_indices,
        )?
    };
    let audit_r0_summary = MetricSummary::from_routes(&audit_r0.routes);
    let audit_selected_summary = MetricSummary::from_routes(&audit_selected.routes);

    let mut all_indices = tune_indices.clone();
    all_indices.extend_from_slice(&audit_indices);
    let mut all_r0 = r0_tune_routes.ok_or_else(|| "r=0 tune result is missing".to_string())?;
    all_r0.extend(audit_r0.routes.iter().cloned());
    let alpha_zero_gate = alpha_zero_gate(
        &original_metric,
        &weights,
        &validation.paths,
        &all_indices,
        &all_r0,
    )?;

    let bootstrap = paired_bootstrap(
        &audit_r0.routes,
        &audit_selected.routes,
        protocol.bootstrap_replicates,
        protocol.bootstrap_seed,
    )?;
    let regret_pass = bootstrap.regret.point >= protocol.regret_improvement_threshold;
    let f1_pass = bootstrap.f1.point >= protocol.f1_improvement_threshold;
    let exact_pass = bootstrap.exact.point >= -protocol.exact_decrease_limit;
    let correctness_pass = expanded.stats.expanded_arcs <= MAX_EXPANDED_ARCS
        && expanded.stats.estimated_raw_expanded_bytes <= MAX_RAW_EXPANDED_BYTES
        && grid_reports
            .iter()
            .all(|report| report.min_arc_weight > 0 && report.max_arc_weight < CCH_INFINITY)
        && alpha_zero_gate.passed();
    let scientific_pass = regret_pass && f1_pass && exact_pass;
    let formal_scientific_verdict = formal_run.then_some(scientific_pass && correctness_pass);
    check_timeout(started, protocol.timeout, "audit and bootstrap")?;

    let output = json!({
        "schema_version": 1,
        "analysis_kind": "fixed_edge_single_left_turn_penalty_probe",
        "metadata": {
            "protocol_path": args.protocol,
            "protocol_sha256": protocol.raw_sha256,
            "protocol_schema_version": protocol.raw["schema_version"],
            "protocol_study": protocol.raw["study"],
            "protocol_status": protocol.raw["status"],
            "formal_protocol_run": formal_run,
            "smoke_overrides": {
                "max_routes_per_partition": args.max_routes,
                "max_grid_values": args.max_grid_values,
            },
            "data_policy": {
                "loaded_splits": ["validation"],
                "validation_variant": protocol.validation_variant,
                "test_loaded": false,
                "training_loaded": false,
                "edge_parameters_updated": false,
            },
            "checkpoint": {
                "path": protocol.checkpoint,
                "expected_sha256": protocol.checkpoint_sha256,
                "actual_sha256": actual_checkpoint_sha256,
                "sha256_matches": true,
                "city": city,
                "best_epoch": checkpoint["best_epoch"],
                "run_id": protocol.raw["fixed_edge_model"]["run_id"],
            },
            "validation_report": validation_report_json(&validation.report),
            "valid_route_selection": {
                "algorithm": "sort valid-route indices by (SplitMix64(index XOR seed), index)",
                "seed": protocol.selection_seed,
                "available_valid_routes": validation.paths.len(),
                "tune_routes": tune_indices.len(),
                "audit_routes": audit_indices.len(),
                "partitions_disjoint": indices_disjoint(&tune_indices, &audit_indices),
                "tune_index_sha256": index_digest(&tune_indices),
                "audit_index_sha256": index_digest(&audit_indices),
            },
            "rayon_threads": rayon::current_num_threads(),
        },
        "model": {
            "expanded_state": "one node per original directed edge",
            "transition": "e->f iff head(e)=tail(f), excluding e==f state self-arcs",
            "left_definition": "signed angle in (30, 150) degrees; positive is counter-clockwise",
            "first_edge_cost": "multi-source offset c_e",
            "target": "multi-target over all states e with head(e)=OD target",
            "median_kappa": kappa,
            "median_definition": "middle value for odd m; arithmetic mean of two middle values for even m",
            "executed_r_grid": r_grid,
        },
        "graph": {
            "original_nodes": expanded.stats.original_nodes,
            "original_edges": expanded.stats.original_edges,
            "expanded_nodes": expanded.stats.expanded_nodes,
            "expanded_arcs": expanded.stats.expanded_arcs,
            "skipped_state_self_arcs": expanded.stats.skipped_state_self_arcs,
            "left_turn_arcs": expanded.stats.left_turn_arcs,
            "unclassifiable_turn_arcs": expanded.stats.unclassifiable_turn_arcs,
            "estimated_raw_expanded_bytes": expanded.stats.estimated_raw_expanded_bytes,
            "hard_arc_limit": MAX_EXPANDED_ARCS,
            "hard_raw_byte_limit": MAX_RAW_EXPANDED_BYTES,
        },
        "timing_seconds": {
            "data_load_and_validation": data_load_seconds,
            "original_order": original_order_seconds,
            "original_cch_build": original_cch_seconds,
            "original_full_customization": original_customize_seconds,
            "line_graph_build": line_graph_seconds,
            "expanded_order": expanded_order_seconds,
            "expanded_cch_build": expanded_cch_seconds,
            "total": started.elapsed().as_secs_f64(),
        },
        "memory": {
            "rss_at_output_mib": memory_mib("VmRSS:"),
            "peak_rss_mib": memory_mib("VmHWM:"),
        },
        "tune_grid": grid_reports.iter().map(GridReport::to_json).collect::<Vec<_>>(),
        "selection": {
            "primary": "minimum tune aggregate relative regret",
            "tie_rule": "within absolute 1e-5 choose higher mean edge F1, then smaller r",
            "selected_grid_index": selected_index,
            "selected_r": selected_r,
            "selected_left_penalty_u32": selected_penalty,
            "selected_tune_metrics": grid_reports[selected_index].summary.to_json(),
        },
        "audit": {
            "r0": {
                "metrics": audit_r0_summary.to_json(),
                "timing_seconds": audit_r0.timing_json(),
            },
            "selected": {
                "r": selected_r,
                "left_penalty_u32": selected_penalty,
                "metrics": audit_selected_summary.to_json(),
                "timing_seconds": audit_selected.timing_json(),
                "timing_note": if selected_penalty == 0 { "reused r0 audit routes" } else { "independent full customization and audit queries" },
            },
            "paired_effects": {
                "regret_absolute_improvement_r0_minus_selected": bootstrap.regret.to_json(),
                "mean_edge_f1_improvement_selected_minus_r0": bootstrap.f1.to_json(),
                "exact_match_change_selected_minus_r0": bootstrap.exact.to_json(),
                "bootstrap_replicates": protocol.bootstrap_replicates,
                "bootstrap_seed": protocol.bootstrap_seed,
                "interval_method": "paired route bootstrap with replacement; empirical 2.5/97.5 percentiles",
            },
        },
        "correctness_gates": {
            "expanded_arc_count_within_limit": expanded.stats.expanded_arcs <= MAX_EXPANDED_ARCS,
            "estimated_raw_arrays_within_limit": expanded.stats.estimated_raw_expanded_bytes <= MAX_RAW_EXPANDED_BYTES,
            "all_expanded_arc_costs_positive_and_below_cch_infinity": grid_reports.iter().all(|report| report.min_arc_weight > 0 && report.max_arc_weight < CCH_INFINITY),
            "decoded_paths_continuous_and_connect_od": true,
            "alpha_zero_equivalence": alpha_zero_gate.to_json(),
            "passed": correctness_pass,
        },
        "scientific_gate": {
            "threshold_basis": "audit point estimates; bootstrap intervals are reported uncertainty diagnostics",
            "regret": {
                "threshold": protocol.regret_improvement_threshold,
                "passed": regret_pass,
            },
            "mean_edge_f1": {
                "threshold": protocol.f1_improvement_threshold,
                "passed": f1_pass,
            },
            "exact_match": {
                "maximum_allowed_decrease": protocol.exact_decrease_limit,
                "passed": exact_pass,
            },
            "point_estimate_passed": scientific_pass,
            "formal_overall_verdict": formal_scientific_verdict,
            "verdict_note": if formal_run {
                "overall requires every correctness gate and all three predeclared audit thresholds"
            } else {
                "smoke overrides make the scientific verdict non-evaluable"
            },
        },
    });
    write_json_atomic(&args.output, &output)?;
    println!(
        "TURN_PROBE output={} selected_r={} correctness_pass={} scientific_point_pass={} formal={}",
        args.output.display(),
        selected_r,
        correctness_pass,
        scientific_pass,
        formal_run
    );
    Ok(())
}

#[derive(Debug)]
struct EvaluatedModel {
    routes: Vec<RouteMetric>,
    weight_generation_seconds: f64,
    customization_seconds: f64,
    query_seconds: f64,
}

impl EvaluatedModel {
    fn timing_json(&self) -> Value {
        json!({
            "weight_generation": self.weight_generation_seconds,
            "full_customization": self.customization_seconds,
            "queries_and_decoding": self.query_seconds,
        })
    }
}

fn evaluate_model(
    cch: &CCH,
    expanded: &ExpandedTurnGraph,
    graph: &GraphData,
    weights: &[u32],
    penalty: u32,
    paths: &[TripPath],
    indices: &[usize],
) -> Result<EvaluatedModel, String> {
    let weight_started = Instant::now();
    let arc_weights = expanded.customized_arc_weights(weights, penalty)?;
    let weight_generation_seconds = weight_started.elapsed().as_secs_f64();
    let customize_started = Instant::now();
    let metric = CCHMetric::new(cch, arc_weights);
    let customization_seconds = customize_started.elapsed().as_secs_f64();
    let query_started = Instant::now();
    let routes = evaluate_routes(&metric, expanded, graph, weights, penalty, paths, indices)?;
    let query_seconds = query_started.elapsed().as_secs_f64();
    Ok(EvaluatedModel {
        routes,
        weight_generation_seconds,
        customization_seconds,
        query_seconds,
    })
}

fn evaluate_routes(
    metric: &CCHMetric<'_>,
    expanded: &ExpandedTurnGraph,
    graph: &GraphData,
    weights: &[u32],
    penalty: u32,
    paths: &[TripPath],
    indices: &[usize],
) -> Result<Vec<RouteMetric>, String> {
    let chunks = rayon::current_num_threads().max(1) * 4;
    let chunk_size = indices.len().div_ceil(chunks).max(1);
    let partials: Vec<Result<Vec<RouteMetric>, String>> = indices
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut query = CCHQuery::new(metric);
            chunk
                .iter()
                .map(|&valid_route_index| {
                    let ((source, target), observed) = paths.get(valid_route_index).ok_or_else(|| {
                        format!("selected valid-route index {valid_route_index} is out of bounds")
                    })?;
                    let predicted = query_expanded_path(
                        &mut query,
                        metric,
                        expanded,
                        graph,
                        weights,
                        penalty,
                        *source,
                        *target,
                    )?;
                    let observed_cost = expanded_path_cost(graph, weights, observed, penalty)?;
                    let raw_regret = observed_cost
                        .checked_sub(predicted.distance as u64)
                        .ok_or_else(|| {
                            format!(
                                "negative regret for valid route {valid_route_index}: observed={observed_cost}, shortest={}",
                                predicted.distance
                            )
                        })?;
                    Ok(RouteMetric {
                        valid_route_index,
                        observed_cost,
                        shortest_cost: predicted.distance,
                        raw_regret,
                        relative_regret: raw_regret as f64 / observed_cost as f64,
                        edge_f1: edge_f1(observed, &predicted.original_edges),
                        exact_match: observed == &predicted.original_edges,
                    })
                })
                .collect()
        })
        .collect();
    let mut routes = Vec::with_capacity(indices.len());
    for partial in partials {
        routes.extend(partial?);
    }
    Ok(routes)
}

fn edge_f1(observed: &[usize], predicted: &[usize]) -> f64 {
    let observed = observed.iter().copied().collect::<HashSet<_>>();
    let predicted = predicted.iter().copied().collect::<HashSet<_>>();
    let intersection = observed.intersection(&predicted).count() as f64;
    let precision = intersection / predicted.len().max(1) as f64;
    let recall = intersection / observed.len().max(1) as f64;
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

fn select_grid(reports: &[GridReport]) -> usize {
    let best_loss = reports
        .iter()
        .map(|report| report.summary.aggregate_relative_regret)
        .min_by(f64::total_cmp)
        .expect("executed grid is nonempty");
    reports
        .iter()
        .enumerate()
        .filter(|(_, report)| report.summary.aggregate_relative_regret <= best_loss + SELECTION_TIE)
        .max_by(|(_, left), (_, right)| {
            left.summary
                .mean_edge_f1
                .total_cmp(&right.summary.mean_edge_f1)
                .then_with(|| right.r.total_cmp(&left.r))
        })
        .map(|(index, _)| index)
        .unwrap()
}

fn alpha_zero_gate(
    original_metric: &CCHMetric<'_>,
    weights: &[u32],
    paths: &[TripPath],
    indices: &[usize],
    expanded_r0: &[RouteMetric],
) -> Result<AlphaZeroGate, String> {
    if indices.len() != expanded_r0.len() {
        return Err("alpha-zero gate input lengths differ".to_string());
    }
    let query_started = Instant::now();
    let chunks = rayon::current_num_threads().max(1) * 4;
    let chunk_size = indices.len().div_ceil(chunks).max(1);
    let partials: Vec<Result<AlphaZeroGate, String>> = indices
        .par_chunks(chunk_size)
        .zip(expanded_r0.par_chunks(chunk_size))
        .map(|(index_chunk, expanded_chunk)| {
            let mut query = CCHQuery::new(original_metric);
            let mut gate = AlphaZeroGate {
                routes_checked: 0,
                distance_mismatches: 0,
                observed_cost_mismatches: 0,
                maximum_distance_absolute_difference: 0,
                maximum_observed_cost_absolute_difference: 0,
                query_seconds: 0.0,
            };
            for (&valid_route_index, expanded) in index_chunk.iter().zip(expanded_chunk) {
                if expanded.valid_route_index != valid_route_index {
                    return Err("alpha-zero route ordering mismatch".to_string());
                }
                let ((source, target), observed) = &paths[valid_route_index];
                query.add_source(*source, 0);
                query.add_target(*target, 0);
                let result = query.run();
                let original_distance = result.distance().ok_or_else(|| {
                    format!("original OD ({source}, {target}) is unreachable")
                })?;
                let original_path = result.arc_path();
                drop(result);
                let reconstructed = original_path.iter().try_fold(0u64, |sum, &edge| {
                    sum.checked_add(weights[edge as usize] as u64)
                        .ok_or_else(|| "original path cost overflow".to_string())
                })?;
                if reconstructed != original_distance as u64 {
                    return Err(format!(
                        "original CCH distance/path mismatch: {original_distance} != {reconstructed}"
                    ));
                }
                let original_observed = observed.iter().try_fold(0u64, |sum, &edge| {
                    sum.checked_add(weights[edge] as u64)
                        .ok_or_else(|| "original observed cost overflow".to_string())
                })?;
                let distance_difference =
                    original_distance.abs_diff(expanded.shortest_cost) as u64;
                let observed_difference = original_observed.abs_diff(expanded.observed_cost);
                gate.distance_mismatches += usize::from(distance_difference != 0);
                gate.observed_cost_mismatches += usize::from(observed_difference != 0);
                gate.maximum_distance_absolute_difference = gate
                    .maximum_distance_absolute_difference
                    .max(distance_difference);
                gate.maximum_observed_cost_absolute_difference = gate
                    .maximum_observed_cost_absolute_difference
                    .max(observed_difference);
                gate.routes_checked += 1;
            }
            Ok(gate)
        })
        .collect();
    let mut combined = AlphaZeroGate {
        routes_checked: 0,
        distance_mismatches: 0,
        observed_cost_mismatches: 0,
        maximum_distance_absolute_difference: 0,
        maximum_observed_cost_absolute_difference: 0,
        query_seconds: 0.0,
    };
    for partial in partials {
        let partial = partial?;
        combined.routes_checked += partial.routes_checked;
        combined.distance_mismatches += partial.distance_mismatches;
        combined.observed_cost_mismatches += partial.observed_cost_mismatches;
        combined.maximum_distance_absolute_difference = combined
            .maximum_distance_absolute_difference
            .max(partial.maximum_distance_absolute_difference);
        combined.maximum_observed_cost_absolute_difference = combined
            .maximum_observed_cost_absolute_difference
            .max(partial.maximum_observed_cost_absolute_difference);
    }
    combined.query_seconds = query_started.elapsed().as_secs_f64();
    Ok(combined)
}

#[derive(Debug)]
struct BootstrapResult {
    regret: Interval,
    f1: Interval,
    exact: Interval,
}

fn paired_bootstrap(
    baseline: &[RouteMetric],
    selected: &[RouteMetric],
    replicates: usize,
    seed: u64,
) -> Result<BootstrapResult, String> {
    if baseline.is_empty() || baseline.len() != selected.len() || replicates == 0 {
        return Err(
            "paired bootstrap requires equal nonempty routes and positive replicates".into(),
        );
    }
    if baseline
        .iter()
        .zip(selected)
        .any(|(left, right)| left.valid_route_index != right.valid_route_index)
    {
        return Err("paired bootstrap route indices do not align".to_string());
    }
    let baseline_summary = MetricSummary::from_routes(baseline);
    let selected_summary = MetricSummary::from_routes(selected);
    let regret_point =
        baseline_summary.aggregate_relative_regret - selected_summary.aggregate_relative_regret;
    let f1_point = selected_summary.mean_edge_f1 - baseline_summary.mean_edge_f1;
    let exact_point = selected_summary.exact_match_rate - baseline_summary.exact_match_rate;
    let mut rng = SplitMix64::new(seed);
    let mut regret_samples = Vec::with_capacity(replicates);
    let mut f1_samples = Vec::with_capacity(replicates);
    let mut exact_samples = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut baseline_regret = 0u128;
        let mut baseline_observed = 0u128;
        let mut selected_regret = 0u128;
        let mut selected_observed = 0u128;
        let mut f1_difference = 0.0;
        let mut exact_difference = 0.0;
        for _ in 0..baseline.len() {
            let index = (rng.next_u64() % baseline.len() as u64) as usize;
            let left = &baseline[index];
            let right = &selected[index];
            baseline_regret += left.raw_regret as u128;
            baseline_observed += left.observed_cost as u128;
            selected_regret += right.raw_regret as u128;
            selected_observed += right.observed_cost as u128;
            f1_difference += right.edge_f1 - left.edge_f1;
            exact_difference += if right.exact_match { 1.0 } else { 0.0 };
            exact_difference -= if left.exact_match { 1.0 } else { 0.0 };
        }
        regret_samples.push(
            ratio(baseline_regret, baseline_observed) - ratio(selected_regret, selected_observed),
        );
        f1_samples.push(f1_difference / baseline.len() as f64);
        exact_samples.push(exact_difference / baseline.len() as f64);
    }
    Ok(BootstrapResult {
        regret: interval(regret_point, regret_samples),
        f1: interval(f1_point, f1_samples),
        exact: interval(exact_point, exact_samples),
    })
}

fn interval(point: f64, mut samples: Vec<f64>) -> Interval {
    samples.sort_by(f64::total_cmp);
    let last = samples.len() - 1;
    let lower = samples[((last as f64 * 0.025).floor() as usize).min(last)];
    let upper = samples[((last as f64 * 0.975).ceil() as usize).min(last)];
    Interval {
        point,
        lower,
        upper,
    }
}

fn ranked_route_indices(count: usize, seed: u64) -> Vec<usize> {
    let mut indices = (0..count).collect::<Vec<_>>();
    indices.sort_unstable_by_key(|&index| (splitmix64_hash(index as u64 ^ seed), index));
    indices
}

fn splitmix64_hash(value: u64) -> u64 {
    let mut value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }
}

fn load_protocol(path: &Path) -> Result<Protocol, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("failed to read protocol {}: {error}", path.display()))?;
    let raw_sha256 = hex_digest(&sha256(&bytes));
    let raw: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("failed to decode protocol {}: {error}", path.display()))?;
    if required_u64(&raw, "/schema_version")? != 1 {
        return Err("turn probe protocol schema_version must be 1".to_string());
    }
    if required_str(&raw, "/study")? != "fixed_edge_single_left_turn_penalty_probe" {
        return Err("unexpected turn probe study identifier".to_string());
    }
    if required_str(&raw, "/status")? != "frozen_before_turn_probe_metrics" {
        return Err("turn probe protocol is not frozen before metrics".to_string());
    }
    if raw.pointer("/fixed_edge_model/edge_parameters_updated") != Some(&Value::Bool(false)) {
        return Err("protocol must freeze edge_parameters_updated=false".to_string());
    }
    let checkpoint = PathBuf::from(required_str(&raw, "/fixed_edge_model/checkpoint")?);
    let checkpoint_sha256 =
        required_str(&raw, "/fixed_edge_model/checkpoint_sha256")?.to_ascii_lowercase();
    if checkpoint_sha256.len() != 64
        || !checkpoint_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("checkpoint_sha256 must be 64 hexadecimal characters".to_string());
    }
    let validation_variant = required_str(&raw, "/data/variant")?.to_string();
    reject_test_text("checkpoint path", &checkpoint.to_string_lossy())?;
    reject_test_text("validation variant", &validation_variant)?;
    if required_str(&raw, "/data/cycle_policy")? != "drop" {
        return Err("turn probe requires cycle_policy=drop".to_string());
    }
    let r_grid = required_f64_array(&raw, "/turn_model/r_grid")?;
    if r_grid.is_empty()
        || r_grid[0] != 0.0
        || r_grid.iter().any(|r| !r.is_finite() || *r < 0.0)
        || r_grid.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(
            "r_grid must be strictly increasing, finite, nonnegative, and start at 0".into(),
        );
    }
    if raw.pointer("/turn_model/nonnegative_penalty") != Some(&Value::Bool(true)) {
        return Err("protocol must require a nonnegative penalty".to_string());
    }
    let selection_seed = required_u64(&raw, "/data/selection_seed")?;
    let tune_routes = required_usize(&raw, "/data/tune_routes")?;
    let audit_routes = required_usize(&raw, "/data/audit_routes")?;
    let bootstrap_replicates =
        required_usize(&raw, "/scientific_gate/paired_bootstrap_replicates")?;
    let bootstrap_seed = required_u64(&raw, "/scientific_gate/bootstrap_seed")?;
    let regret_improvement_threshold = required_f64(
        &raw,
        "/scientific_gate/audit_aggregate_relative_regret_absolute_improvement_at_least",
    )?;
    let f1_improvement_threshold = required_f64(
        &raw,
        "/scientific_gate/audit_mean_edge_f1_improvement_at_least",
    )?;
    let exact_decrease_limit =
        required_f64(&raw, "/scientific_gate/audit_exact_match_decrease_at_most")?;
    let timeout = Duration::from_secs(required_u64(&raw, "/runtime/hard_timeout_seconds")?);
    Ok(Protocol {
        raw,
        raw_sha256,
        checkpoint,
        checkpoint_sha256,
        validation_variant,
        selection_seed,
        tune_routes,
        audit_routes,
        r_grid,
        bootstrap_replicates,
        bootstrap_seed,
        regret_improvement_threshold,
        f1_improvement_threshold,
        exact_decrease_limit,
        timeout,
    })
}

fn required_str<'a>(value: &'a Value, pointer: &str) -> Result<&'a str, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("protocol field {pointer} must be a string"))
}

fn required_u64(value: &Value, pointer: &str) -> Result<u64, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("protocol field {pointer} must be a nonnegative integer"))
}

fn required_usize(value: &Value, pointer: &str) -> Result<usize, String> {
    usize::try_from(required_u64(value, pointer)?)
        .map_err(|_| format!("protocol field {pointer} does not fit usize"))
}

fn required_f64(value: &Value, pointer: &str) -> Result<f64, String> {
    let number = value
        .pointer(pointer)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("protocol field {pointer} must be numeric"))?;
    if !number.is_finite() {
        return Err(format!("protocol field {pointer} must be finite"));
    }
    Ok(number)
}

fn required_f64_array(value: &Value, pointer: &str) -> Result<Vec<f64>, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("protocol field {pointer} must be an array"))?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| format!("protocol field {pointer}[{index}] must be finite numeric"))
        })
        .collect()
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut protocol = None;
    let mut output = None;
    let mut max_routes = None;
    let mut max_grid_values = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--help" | "-h" => return Ok(None),
            "--protocol" => protocol = Some(next_path(&mut arguments, "--protocol")?),
            "--output" => output = Some(next_path(&mut arguments, "--output")?),
            "--max-routes" => max_routes = Some(next_usize(&mut arguments, "--max-routes")?),
            "--max-grid-values" => {
                max_grid_values = Some(next_usize(&mut arguments, "--max-grid-values")?)
            }
            _ => return Err(format!("unknown argument {argument:?}; use --help")),
        }
    }
    Ok(Some(Args {
        protocol: protocol.ok_or_else(|| "missing required --protocol PATH".to_string())?,
        output: output.ok_or_else(|| "missing required --output PATH".to_string())?,
        max_routes,
        max_grid_values,
    }))
}

fn next_path(arguments: &mut impl Iterator<Item = String>, flag: &str) -> Result<PathBuf, String> {
    arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| format!("missing value after {flag}"))
}

fn next_usize(arguments: &mut impl Iterator<Item = String>, flag: &str) -> Result<usize, String> {
    let value = arguments
        .next()
        .ok_or_else(|| format!("missing value after {flag}"))?;
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid {flag} value {value:?}: {error}"))
}

fn print_help() {
    println!(
        "Usage: cargo run --release --example probe_turn_penalty -- \\\n+  --protocol experiments/convergence_study/turn_probe_protocol.json \\\n+  --output experiments/convergence_study/turn_probe_results.json [OPTIONS]\n\n\
Options:\n\
  --max-routes N       Smoke-only cap applied independently to tune and audit\n\
  --max-grid-values N  Smoke-only prefix of the frozen r grid\n\n\
No city, split, variant, or checkpoint override is accepted."
    );
}

fn reject_test_text(label: &str, value: &str) -> Result<(), String> {
    if value.to_ascii_lowercase().contains("test") {
        Err(format!(
            "refusing {label} containing forbidden text 'test': {value:?}"
        ))
    } else {
        Ok(())
    }
}

fn check_timeout(started: Instant, timeout: Duration, stage: &str) -> Result<(), String> {
    if started.elapsed() > timeout {
        Err(format!(
            "protocol hard timeout of {} seconds exceeded after {stage}; use an external process timeout to enforce the bound during blocking CCH calls",
            timeout.as_secs()
        ))
    } else {
        Ok(())
    }
}

fn progress(stage: &str) {
    eprintln!("TURN_PROBE_STAGE {stage}");
}

fn ratio(numerator: u128, denominator: u128) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn mean(values: impl Iterator<Item = f64>, count: usize) -> f64 {
    if count == 0 {
        0.0
    } else {
        values.sum::<f64>() / count as f64
    }
}

fn indices_disjoint(left: &[usize], right: &[usize]) -> bool {
    let left = left.iter().copied().collect::<HashSet<_>>();
    right.iter().all(|index| !left.contains(index))
}

fn index_digest(indices: &[usize]) -> String {
    let mut bytes = Vec::with_capacity(indices.len() * 8);
    for &index in indices {
        bytes.extend_from_slice(&(index as u64).to_le_bytes());
    }
    hex_digest(&sha256(&bytes))
}

fn validation_report_json(report: &PathValidationReport) -> Value {
    json!({
        "available_samples": report.available_samples,
        "inspected_samples": report.inspected_samples,
        "accepted_samples": report.accepted_samples,
        "dropped_samples": report.dropped_samples(),
        "empty_or_too_short": report.empty_or_too_short,
        "out_of_bounds": report.out_of_bounds,
        "discontinuous": report.discontinuous,
        "cyclic": report.cyclic,
    })
}

fn memory_mib(label: &str) -> Option<f64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let kib = status.lines().find_map(|line| {
        let rest = line.strip_prefix(label)?;
        rest.split_whitespace().next()?.parse::<u64>().ok()
    })?;
    Some(kib as f64 / 1024.0)
}

fn write_json_atomic(path: &Path, value: &Value) -> Result<(), String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let temporary = path.with_extension(format!(
        "{}.tmp-{}",
        path.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("json"),
        std::process::id()
    ));
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("failed to encode probe JSON: {error}"))?;
    let mut file = fs::File::create(&temporary)
        .map_err(|error| format!("failed to create {}: {error}", temporary.display()))?;
    file.write_all(&bytes)
        .map_err(|error| format!("failed to write {}: {error}", temporary.display()))?;
    file.write_all(b"\n")
        .map_err(|error| format!("failed to finish {}: {error}", temporary.display()))?;
    file.sync_all()
        .map_err(|error| format!("failed to sync {}: {error}", temporary.display()))?;
    fs::rename(&temporary, path).map_err(|error| {
        format!(
            "failed to rename {} to {}: {error}",
            temporary.display(),
            path.display()
        )
    })
}

fn hex_digest(digest: &[u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

// Minimal dependency-free SHA-256, shared in spirit with the validation-block
// generator so a frozen checkpoint can be verified without network access.
fn sha256(input: &[u8]) -> [u8; 32] {
    const INITIAL: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let bit_length = (input.len() as u64)
        .checked_mul(8)
        .expect("SHA-256 input bit length overflow");
    let mut padded = Vec::with_capacity(input.len() + 72);
    padded.extend_from_slice(input);
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_length.to_be_bytes());
    let mut state = INITIAL;
    let mut schedule = [0u32; 64];
    for chunk in padded.chunks_exact(64) {
        for (index, word) in chunk.chunks_exact(4).enumerate() {
            schedule[index] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for index in 16..64 {
            let s0 = schedule[index - 15].rotate_right(7)
                ^ schedule[index - 15].rotate_right(18)
                ^ (schedule[index - 15] >> 3);
            let s1 = schedule[index - 2].rotate_right(17)
                ^ schedule[index - 2].rotate_right(19)
                ^ (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(s0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ ((!e) & g);
            let temporary_1 = h
                .wrapping_add(s1)
                .wrapping_add(choice)
                .wrapping_add(K[index])
                .wrapping_add(schedule[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temporary_2 = s0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temporary_1);
            d = c;
            c = b;
            b = a;
            a = temporary_1.wrapping_add(temporary_2);
        }
        for (value, compressed) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *value = value.wrapping_add(compressed);
        }
    }
    let mut digest = [0u8; 32];
    for (chunk, value) in digest.chunks_exact_mut(4).zip(state) {
        chunk.copy_from_slice(&value.to_be_bytes());
    }
    digest
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_standard_vectors() {
        assert_eq!(
            hex_digest(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn route_ranking_is_deterministic_and_unique() {
        let first = ranked_route_indices(100, 42);
        let second = ranked_route_indices(100, 42);
        assert_eq!(first, second);
        assert_eq!(first.iter().copied().collect::<HashSet<_>>().len(), 100);
    }

    #[test]
    fn forbidden_test_text_is_rejected_case_insensitively() {
        assert!(reject_test_text("variant", "validation_dev").is_ok());
        assert!(reject_test_text("variant", "TEST_all").is_err());
    }
}
