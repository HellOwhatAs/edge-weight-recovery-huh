use edge_weight_recovery::config::{TrainingConfig, atomic_write};
use edge_weight_recovery::data::{
    OdGroup, PathValidationReport, TripPath, group_paths_by_od, load_graph, load_trips,
};
use edge_weight_recovery::graph_problem::{GraphMetric, GraphProblem, GraphRepresentation};
use rayon::prelude::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

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
    let started = Instant::now();
    let config = TrainingConfig::load(&arguments.config)?;
    let actual_threads = rayon::current_num_threads().max(1);
    if actual_threads != config.rayon_threads {
        return Err(format!(
            "configuration requires {} Rayon threads, process has {actual_threads}; set RAYON_NUM_THREADS before launch",
            config.rayon_threads
        ));
    }

    // The executable has no arbitrary split argument: it reads exactly the
    // configured training and validation variants and can never select test.
    let train_identity = verify_data_identity(&config, "train", &config.train_variant)?;
    let validation_identity =
        verify_data_identity(&config, "validation", &config.validation_variant)?;
    let graph = load_graph(&config.city)?;
    let train = load_trips(&config.city, "train", &config.train_variant, &graph, None)?;
    let validation = load_trips(
        &config.city,
        "validation",
        &config.validation_variant,
        &graph,
        None,
    )?;
    verify_sample_count(&config, "train", train.report.available_samples)?;
    verify_sample_count(&config, "validation", validation.report.available_samples)?;

    let direct_edges = graph
        .tail
        .iter()
        .copied()
        .zip(graph.head.iter().copied())
        .collect::<HashSet<_>>();
    let problem = GraphProblem::build(
        &graph,
        GraphRepresentation::EdgeTransitionArcs,
        config.weight_lower_factor,
        config.weight_upper_factor,
    )?;
    let metric = problem.customize(problem.initial_weights())?;
    let train_audit = audit_split(
        &train.paths,
        &train.report,
        &direct_edges,
        &metric,
        actual_threads,
    )?;
    let validation_audit = audit_split(
        &validation.paths,
        &validation.report,
        &direct_edges,
        &metric,
        actual_threads,
    )?;

    let output = json!({
        "schema_version": 1,
        "purpose": "edge_transition_arcs_zero_cost_single_edge_impact_audit",
        "configuration": arguments.config,
        "configuration_sha256": sha256_file(&arguments.config)?,
        "data": {
            "city": config.city,
            "path_contract": config.as_json().pointer("/data/path_contract"),
            "cycle_policy": config.as_json().pointer("/data/cycle_policy"),
            "train": train_identity,
            "validation": validation_identity,
        },
        "line_graph_initial_metric": {
            "routing_nodes": problem.routing_node_count(),
            "routing_arcs": problem.routing_arc_count(),
            "coordinates": problem.coordinate_count(),
            "topology_identity": problem.topology_identity(),
            "endpoint_offsets": 0,
            "has_start_cost": false,
            "has_first_edge_parameter": false,
        },
        "splits": {
            "train": train_audit,
            "validation": validation_audit,
        },
        "rayon_threads": actual_threads,
        "wall_seconds": started.elapsed().as_secs_f64(),
        "model_semantics_changed": false,
        "test_read": false,
    });
    let encoded = serde_json::to_vec_pretty(&output)
        .map_err(|error| format!("failed to encode audit output: {error}"))?;
    atomic_write(&arguments.output, &encoded)
}

fn audit_split(
    paths: &[TripPath],
    report: &PathValidationReport,
    direct_edges: &HashSet<(u32, u32)>,
    metric: &GraphMetric<'_>,
    threads: usize,
) -> Result<Value, String> {
    let groups = group_paths_by_od(paths);
    let direct = direct_edge_impact(paths, &groups, direct_edges);
    let (zero_cost_single_edge_ods, zero_cost_single_edge_samples) =
        zero_cost_single_edge_predictions(metric, &groups, threads)?;
    if zero_cost_single_edge_ods != direct.direct_edge_unique_ods
        || zero_cost_single_edge_samples != direct.direct_edge_samples
    {
        return Err(format!(
            "zero-cost predictions ({zero_cost_single_edge_ods} ODs, {zero_cost_single_edge_samples} samples) disagree with direct-edge exposure ({} ODs, {} samples)",
            direct.direct_edge_unique_ods, direct.direct_edge_samples
        ));
    }

    Ok(json!({
        "filtering": report_json(report),
        "effective_trajectories": paths.len(),
        "unique_ods": groups.len(),
        "direct_original_edge": {
            "samples": direct.direct_edge_samples,
            "proportion_of_effective_trajectories": ratio(direct.direct_edge_samples, paths.len()),
            "unique_ods": direct.direct_edge_unique_ods,
            "proportion_of_unique_ods": ratio(direct.direct_edge_unique_ods, groups.len()),
        },
        "direct_edge_but_observed_route_longer_than_one_edge": {
            "samples": direct.direct_edge_observed_long_samples,
            "proportion_of_effective_trajectories": ratio(direct.direct_edge_observed_long_samples, paths.len()),
            "proportion_conditional_on_direct_edge": ratio(direct.direct_edge_observed_long_samples, direct.direct_edge_samples),
        },
        "initial_edge_transition_prediction": {
            "zero_cost_single_edge_ods": zero_cost_single_edge_ods,
            "proportion_of_unique_ods": ratio(zero_cost_single_edge_ods, groups.len()),
            "affected_samples": zero_cost_single_edge_samples,
            "proportion_of_effective_trajectories": ratio(zero_cost_single_edge_samples, paths.len()),
            "all_affected_observations_have_more_than_one_edge": direct.direct_edge_observed_long_samples == zero_cost_single_edge_samples,
        },
    }))
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DirectEdgeImpact {
    direct_edge_samples: usize,
    direct_edge_observed_long_samples: usize,
    direct_edge_unique_ods: usize,
}

fn direct_edge_impact(
    paths: &[TripPath],
    groups: &[OdGroup],
    direct_edges: &HashSet<(u32, u32)>,
) -> DirectEdgeImpact {
    DirectEdgeImpact {
        direct_edge_samples: paths
            .iter()
            .filter(|(od, _)| direct_edges.contains(od))
            .count(),
        direct_edge_observed_long_samples: paths
            .iter()
            .filter(|(od, path)| direct_edges.contains(od) && path.len() > 1)
            .count(),
        direct_edge_unique_ods: groups
            .iter()
            .filter(|group| direct_edges.contains(&(group.source, group.target)))
            .count(),
    }
}

fn zero_cost_single_edge_predictions(
    metric: &GraphMetric<'_>,
    groups: &[OdGroup],
    threads: usize,
) -> Result<(usize, usize), String> {
    if groups.is_empty() {
        return Ok((0, 0));
    }
    let locals = groups
        .par_chunks(groups.len().div_ceil(threads.max(1)).max(1))
        .map(|chunk| {
            let mut query = metric.new_query();
            let mut od_count = 0usize;
            let mut sample_count = 0usize;
            for group in chunk {
                let prediction = query.shortest_path(group.source, group.target)?;
                let zero_cost_single_edge = prediction.distance == 0
                    && prediction.direct_cost == 0.0
                    && prediction.coordinates.is_empty()
                    && prediction.original_edges.len() == 1;
                if prediction.distance == 0 && !zero_cost_single_edge {
                    return Err(format!(
                        "OD ({}, {}) has a zero distance without a decoded single-edge route",
                        group.source, group.target
                    ));
                }
                if zero_cost_single_edge {
                    od_count = od_count
                        .checked_add(1)
                        .ok_or_else(|| "zero-cost OD count overflow".to_string())?;
                    sample_count = sample_count
                        .checked_add(
                            usize::try_from(group.sample_count)
                                .map_err(|_| "sample count does not fit usize".to_string())?,
                        )
                        .ok_or_else(|| "zero-cost sample count overflow".to_string())?;
                }
            }
            Ok((od_count, sample_count))
        })
        .collect::<Vec<Result<_, String>>>();

    let mut total = (0usize, 0usize);
    for local in locals {
        let (ods, samples) = local?;
        total.0 = total
            .0
            .checked_add(ods)
            .ok_or_else(|| "zero-cost OD count overflow".to_string())?;
        total.1 = total
            .1
            .checked_add(samples)
            .ok_or_else(|| "zero-cost sample count overflow".to_string())?;
    }
    Ok(total)
}

fn report_json(report: &PathValidationReport) -> Value {
    json!({
        "available": report.available_samples,
        "inspected": report.inspected_samples,
        "accepted": report.accepted_samples,
        "dropped": report.dropped_samples(),
        "empty": report.empty,
        "too_short": report.too_short,
        "out_of_bounds": report.out_of_bounds,
        "discontinuous": report.discontinuous,
        "cyclic": report.cyclic,
    })
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn verify_data_identity(
    config: &TrainingConfig,
    split: &str,
    variant: &str,
) -> Result<Value, String> {
    if split != "train" && split != "validation" {
        return Err("audit split must be train or validation".to_string());
    }
    let pointer = format!("/data/{split}_identity");
    let declared = config
        .as_json()
        .pointer(&pointer)
        .ok_or_else(|| format!("configuration is missing {pointer}"))?;
    let expected_path = format!(
        "data/{}_data/preprocessed_{split}_trips_{variant}.pkl",
        config.city
    );
    let declared_path = declared
        .pointer("/path")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("configuration is missing {pointer}/path"))?;
    if declared_path != expected_path {
        return Err(format!(
            "declared {split} path {declared_path:?} does not match {expected_path:?}"
        ));
    }
    let actual_bytes = std::fs::metadata(&expected_path)
        .map_err(|error| format!("failed to inspect {expected_path}: {error}"))?
        .len();
    let declared_bytes = declared
        .pointer("/bytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("configuration is missing {pointer}/bytes"))?;
    if actual_bytes != declared_bytes {
        return Err(format!(
            "{expected_path} has {actual_bytes} bytes, expected {declared_bytes}"
        ));
    }
    let actual_sha256 = sha256_file(Path::new(&expected_path))?;
    let declared_sha256 = declared
        .pointer("/sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("configuration is missing {pointer}/sha256"))?;
    if actual_sha256 != declared_sha256 {
        return Err(format!(
            "{expected_path} SHA-256 mismatch: expected {declared_sha256}, got {actual_sha256}"
        ));
    }
    Ok(json!({
        "variant": variant,
        "path": expected_path,
        "bytes": actual_bytes,
        "sha256": actual_sha256,
        "source_sha256": declared.pointer("/source_sha256"),
        "declared_sample_count": declared.pointer("/sample_count"),
        "seed": declared.pointer("/seed"),
        "verified": true,
    }))
}

fn verify_sample_count(
    config: &TrainingConfig,
    split: &str,
    available_samples: usize,
) -> Result<(), String> {
    let pointer = format!("/data/{split}_identity/sample_count");
    if let Some(declared) = config.as_json().pointer(&pointer).and_then(Value::as_u64)
        && declared != available_samples as u64
    {
        return Err(format!(
            "{split} sample count mismatch: declared {declared}, loaded {available_samples}"
        ));
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file =
        File::open(path).map_err(|error| format!("failed to open {}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("failed to hash {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
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
            println!("Usage: audit_single_edge --config PATH --output PATH");
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
                return Err(format!("{flag} was provided more than once"));
            }
            index += 2;
        }
        Ok(Some(Self {
            config: config.ok_or_else(|| "missing --config PATH".to_string())?,
            output: output.ok_or_else(|| "missing --output PATH".to_string())?,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_edge_impact_counts_samples_and_unique_ods() {
        let paths = vec![
            ((0, 3), vec![0, 1]),
            ((0, 3), vec![2, 3, 4]),
            ((0, 2), vec![0, 5]),
        ];
        let groups = group_paths_by_od(&paths);
        let direct_edges = HashSet::from([(0, 3)]);
        assert_eq!(
            direct_edge_impact(&paths, &groups, &direct_edges),
            DirectEdgeImpact {
                direct_edge_samples: 2,
                direct_edge_observed_long_samples: 2,
                direct_edge_unique_ods: 1,
            }
        );
    }

    #[test]
    fn ratios_are_safe_for_empty_denominators() {
        assert_eq!(ratio(0, 0), 0.0);
        assert_eq!(ratio(1, 4), 0.25);
    }
}
