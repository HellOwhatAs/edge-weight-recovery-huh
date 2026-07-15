use rayon::prelude::*;
use routingkit_cch::shp_utils;
use routingkit_cch::{CCHMetric, CCHQuery};
use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::time::{Duration, Instant};

pub type TripPath = ((u32, u32), Vec<usize>);

#[derive(Debug)]
pub struct GraphData {
    pub tail: Vec<u32>,
    pub head: Vec<u32>,
    pub baseline_weights: Vec<u32>,
    pub x: Vec<f32>,
    pub y: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CyclePolicy {
    Drop,
    Keep,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PathValidationReport {
    pub available_samples: usize,
    pub inspected_samples: usize,
    pub accepted_samples: usize,
    pub trimmed_boundary_edges: usize,
    pub empty_or_too_short: usize,
    pub out_of_bounds: usize,
    pub discontinuous: usize,
    pub cyclic: usize,
}

impl PathValidationReport {
    pub fn dropped_samples(&self) -> usize {
        self.inspected_samples - self.accepted_samples
    }
}

#[derive(Debug)]
pub struct LoadedTrips {
    pub paths: Vec<TripPath>,
    pub report: PathValidationReport,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OdGroup {
    pub source: u32,
    pub target: u32,
    pub sample_count: u64,
}

#[derive(Clone, Debug)]
pub struct OracleStats {
    pub predicted_edge_counts: Vec<u64>,
    pub weighted_shortest_distance_sum: u128,
    pub sample_count: u64,
    pub num_queries: usize,
    /// Wall time for shortest-path queries, path reconstruction, edge counting,
    /// consistency checks, and parallel reduction.
    pub oracle_duration: Duration,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RegretStats {
    pub observed_cost_sum: u128,
    pub shortest_distance_sum: u128,
    pub data_loss_sum: u128,
    pub mean_data_loss: f64,
    pub relative_data_loss: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PathMetrics {
    pub sample_count: usize,
    pub exact_match: f64,
    pub edge_precision: f64,
    pub edge_recall: f64,
    pub edge_f1: f64,
    pub edge_jaccard: f64,
    pub mean_regret: f64,
    pub relative_regret: f64,
}

/// Load only the fixed road topology. Trip splits are loaded separately so a full
/// test set need not stay resident during training.
pub fn load_graph(city: &str) -> Result<GraphData, String> {
    let edges_path = format!("data/{city}_data/map/edges.shp");
    let nodes_path = format!("data/{city}_data/map/nodes.shp");
    let edges = shp_utils::load_edges(&edges_path)
        .map_err(|error| format!("failed to load {edges_path}: {error}"))?;
    let nodes = shp_utils::load_nodes(&nodes_path)
        .map_err(|error| format!("failed to load {nodes_path}: {error}"))?;
    let arrays = shp_utils::build_graph_arrays(&nodes, &edges)
        .map_err(|error| format!("failed to construct graph arrays: {error}"))?;

    let mut baseline_weights = Vec::with_capacity(arrays.weight.len());
    for (edge_id, length) in arrays.weight.into_iter().enumerate() {
        let scaled = length * 1_000.0;
        if !scaled.is_finite() || scaled <= 0.0 || scaled >= i32::MAX as f64 {
            return Err(format!(
                "edge {edge_id} has invalid scaled baseline cost {scaled}"
            ));
        }
        baseline_weights.push(scaled.round().max(1.0) as u32);
    }

    Ok(GraphData {
        tail: arrays.tail.into_iter().map(|node| node as u32).collect(),
        head: arrays.head.into_iter().map(|node| node as u32).collect(),
        baseline_weights,
        x: arrays.xs.into_iter().map(|value| value as f32).collect(),
        y: arrays.ys.into_iter().map(|value| value as f32).collect(),
    })
}

/// Load and validate one split. The pickle schema is
/// `(trip_key, Vec<original_edge_id>, (start_time, end_time))`; the edge vector
/// already contains the complete path and has no node sentinels.
pub fn load_trips(
    city: &str,
    split: &str,
    variant: &str,
    graph: &GraphData,
    max_samples: Option<usize>,
    trim_boundary_edges: bool,
    cycle_policy: CyclePolicy,
) -> Result<LoadedTrips, String> {
    let path = format!("data/{city}_data/preprocessed_{split}_trips_{variant}.pkl");
    let raw: Vec<(serde_pickle::Value, Vec<usize>, (usize, usize))> = serde_pickle::from_reader(
        File::open(&path).map_err(|error| format!("failed to open {path}: {error}"))?,
        Default::default(),
    )
    .map_err(|error| format!("failed to decode {path}: {error}"))?;

    let mut report = PathValidationReport {
        available_samples: raw.len(),
        ..PathValidationReport::default()
    };
    let inspect_count = max_samples.unwrap_or(raw.len()).min(raw.len());
    let mut paths = Vec::with_capacity(inspect_count);

    for (_, mut edge_path, _) in raw.into_iter().take(inspect_count) {
        report.inspected_samples += 1;
        if trim_boundary_edges {
            if edge_path.len() <= 2 {
                report.empty_or_too_short += 1;
                continue;
            }
            edge_path = edge_path[1..edge_path.len() - 1].to_vec();
            report.trimmed_boundary_edges += 2;
        }

        match validate_edge_path(&edge_path, &graph.tail, &graph.head) {
            PathValidity::Valid => {}
            PathValidity::Empty => {
                report.empty_or_too_short += 1;
                continue;
            }
            PathValidity::OutOfBounds => {
                report.out_of_bounds += 1;
                continue;
            }
            PathValidity::Discontinuous => {
                report.discontinuous += 1;
                continue;
            }
            PathValidity::Cyclic => {
                report.cyclic += 1;
                if cycle_policy == CyclePolicy::Drop {
                    continue;
                }
            }
        }

        let first_edge = edge_path[0];
        let last_edge = *edge_path.last().expect("validated nonempty path");
        paths.push(((graph.tail[first_edge], graph.head[last_edge]), edge_path));
        report.accepted_samples += 1;
    }

    Ok(LoadedTrips { paths, report })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PathValidity {
    Valid,
    Empty,
    OutOfBounds,
    Discontinuous,
    Cyclic,
}

fn validate_edge_path(path: &[usize], tail: &[u32], head: &[u32]) -> PathValidity {
    if path.is_empty() {
        return PathValidity::Empty;
    }
    if path
        .iter()
        .any(|&edge_id| edge_id >= tail.len() || edge_id >= head.len())
    {
        return PathValidity::OutOfBounds;
    }
    if path.windows(2).any(|pair| head[pair[0]] != tail[pair[1]]) {
        return PathValidity::Discontinuous;
    }

    let mut visited_nodes = HashSet::with_capacity(path.len() + 1);
    visited_nodes.insert(tail[path[0]]);
    for &edge_id in path {
        if !visited_nodes.insert(head[edge_id]) {
            return PathValidity::Cyclic;
        }
    }
    PathValidity::Valid
}

fn chunk_size(len: usize, num_chunks: usize) -> usize {
    len.div_ceil(num_chunks.max(1)).max(1)
}

pub fn compute_observed_edge_counts(
    paths: &[TripPath],
    edge_count: usize,
    num_chunks: usize,
) -> Vec<u64> {
    if paths.is_empty() {
        return vec![0; edge_count];
    }
    paths
        .par_chunks(chunk_size(paths.len(), num_chunks))
        .map(|chunk| {
            let mut local = vec![0u64; edge_count];
            for (_, path) in chunk {
                for &edge_id in path {
                    local[edge_id] = local[edge_id]
                        .checked_add(1)
                        .expect("observed edge count overflow");
                }
            }
            local
        })
        .reduce(
            || vec![0; edge_count],
            |mut left, right| {
                left.iter_mut().zip(right).for_each(|(value, addend)| {
                    *value = value
                        .checked_add(addend)
                        .expect("observed edge count overflow")
                });
                left
            },
        )
}

pub fn group_paths_by_od(paths: &[TripPath]) -> Vec<OdGroup> {
    let mut counts = BTreeMap::<(u32, u32), u64>::new();
    for &((source, target), _) in paths {
        let count = counts.entry((source, target)).or_default();
        *count = count.checked_add(1).expect("OD sample count overflow");
    }
    counts
        .into_iter()
        .map(|((source, target), sample_count)| OdGroup {
            source,
            target,
            sample_count,
        })
        .collect()
}

pub fn compute_oracle_stats(
    metric: &CCHMetric<'_>,
    groups: &[OdGroup],
    edge_count: usize,
    num_chunks: usize,
) -> Result<OracleStats, String> {
    let started = Instant::now();
    if groups.is_empty() {
        return Ok(OracleStats {
            predicted_edge_counts: vec![0; edge_count],
            weighted_shortest_distance_sum: 0,
            sample_count: 0,
            num_queries: 0,
            oracle_duration: started.elapsed(),
        });
    }

    type LocalOracle = (Vec<u64>, u128, u64, usize);
    let locals: Vec<Result<LocalOracle, String>> = groups
        .par_chunks(chunk_size(groups.len(), num_chunks))
        .map(|chunk| {
            let mut query = CCHQuery::new(metric);
            let mut counts = vec![0u64; edge_count];
            let mut distance_sum = 0u128;
            let mut sample_count = 0u64;
            for group in chunk {
                query.add_source(group.source, 0);
                query.add_target(group.target, 0);
                let result = query.run();
                let distance = result.distance().ok_or_else(|| {
                    format!(
                        "OD ({}, {}) is unreachable in the current metric",
                        group.source, group.target
                    )
                })?;
                let arc_path = result.arc_path();
                let reconstructed_distance = arc_path.iter().try_fold(0u128, |sum, &edge_id| {
                    let weight = metric.weights().get(edge_id as usize).ok_or_else(|| {
                        format!("CCH returned invalid original edge id {edge_id}")
                    })?;
                    sum.checked_add(*weight as u128)
                        .ok_or_else(|| "reconstructed path cost overflow".to_string())
                })?;
                if reconstructed_distance != distance as u128 {
                    return Err(format!(
                        "CCH path/distance mismatch for OD ({}, {}): path={}, distance={distance}",
                        group.source, group.target, reconstructed_distance
                    ));
                }
                for edge_id in arc_path {
                    counts[edge_id as usize] = counts[edge_id as usize]
                        .checked_add(group.sample_count)
                        .ok_or_else(|| "predicted edge count overflow".to_string())?;
                }
                distance_sum = distance_sum
                    .checked_add(distance as u128 * group.sample_count as u128)
                    .ok_or_else(|| "shortest-distance sum overflow".to_string())?;
                sample_count = sample_count
                    .checked_add(group.sample_count)
                    .ok_or_else(|| "oracle sample count overflow".to_string())?;
            }
            Ok((counts, distance_sum, sample_count, chunk.len()))
        })
        .collect();

    let mut stats = OracleStats {
        predicted_edge_counts: vec![0; edge_count],
        weighted_shortest_distance_sum: 0,
        sample_count: 0,
        num_queries: 0,
        oracle_duration: Duration::ZERO,
    };
    for local in locals {
        let (counts, distance_sum, sample_count, queries) = local?;
        for (total, addend) in stats.predicted_edge_counts.iter_mut().zip(counts) {
            *total = total
                .checked_add(addend)
                .ok_or_else(|| "predicted edge count overflow".to_string())?;
        }
        stats.weighted_shortest_distance_sum = stats
            .weighted_shortest_distance_sum
            .checked_add(distance_sum)
            .ok_or_else(|| "shortest-distance sum overflow".to_string())?;
        stats.sample_count = stats
            .sample_count
            .checked_add(sample_count)
            .ok_or_else(|| "oracle sample count overflow".to_string())?;
        stats.num_queries += queries;
    }
    stats.oracle_duration = started.elapsed();
    Ok(stats)
}

pub fn observed_cost(weights: &[u32], observed_counts: &[u64]) -> Result<u128, String> {
    if weights.len() != observed_counts.len() {
        return Err("weight and observed-count lengths differ".to_string());
    }
    weights
        .iter()
        .zip(observed_counts)
        .try_fold(0u128, |sum, (&weight, &count)| {
            sum.checked_add(weight as u128 * count as u128)
                .ok_or_else(|| "observed path-cost sum overflow".to_string())
        })
}

pub fn compute_regret(
    weights: &[u32],
    observed_counts: &[u64],
    oracle: &OracleStats,
) -> Result<RegretStats, String> {
    let observed_cost_sum = observed_cost(weights, observed_counts)?;
    let data_loss_sum = observed_cost_sum
        .checked_sub(oracle.weighted_shortest_distance_sum)
        .ok_or_else(|| {
            format!(
                "negative regret: observed cost {observed_cost_sum} < shortest-distance sum {}; \
                 verify path endpoints and original-edge mapping",
                oracle.weighted_shortest_distance_sum
            )
        })?;
    let mean_data_loss = if oracle.sample_count == 0 {
        0.0
    } else {
        data_loss_sum as f64 / oracle.sample_count as f64
    };
    let relative_data_loss = if observed_cost_sum == 0 {
        0.0
    } else {
        data_loss_sum as f64 / observed_cost_sum as f64
    };
    Ok(RegretStats {
        observed_cost_sum,
        shortest_distance_sum: oracle.weighted_shortest_distance_sum,
        data_loss_sum,
        mean_data_loss,
        relative_data_loss,
    })
}

/// The L1 norm of one selected aggregate data subgradient. This is not the
/// regret objective: its magnitude is tie-breaking dependent and it omits the
/// regularizer. For an exact oracle, a value of zero does certify zero data
/// regret, but nonzero residual can coexist with zero regret under path ties.
pub fn count_residual_l1(predicted: &[u64], observed: &[u64]) -> Result<u128, String> {
    if predicted.len() != observed.len() {
        return Err("predicted and observed count lengths differ".to_string());
    }
    predicted
        .iter()
        .zip(observed)
        .try_fold(0u128, |sum, (&left, &right)| {
            sum.checked_add(left.abs_diff(right) as u128)
                .ok_or_else(|| "count residual overflow".to_string())
        })
}

/// Standard held-out route metrics. Test data should be passed here only after
/// training/hyperparameter selection has finished.
pub fn evaluate_paths(
    metric: &CCHMetric<'_>,
    paths: &[TripPath],
    num_chunks: usize,
) -> Result<PathMetrics, String> {
    type LocalPathMetrics = (usize, f64, f64, f64, f64, f64, u128, u128);
    if paths.is_empty() {
        return Ok(PathMetrics::default());
    }
    let locals: Vec<Result<LocalPathMetrics, String>> = paths
        .par_chunks(chunk_size(paths.len(), num_chunks))
        .map(|chunk| {
            let mut query = CCHQuery::new(metric);
            let mut exact = 0.0;
            let mut precision = 0.0;
            let mut recall = 0.0;
            let mut f1 = 0.0;
            let mut jaccard = 0.0;
            let mut regret = 0u128;
            let mut observed_cost = 0u128;
            for ((source, target), observed_path) in chunk {
                query.add_source(*source, 0);
                query.add_target(*target, 0);
                let result = query.run();
                let distance = result
                    .distance()
                    .ok_or_else(|| format!("held-out OD ({source}, {target}) is unreachable"))?;
                let predicted_path: Vec<usize> = result
                    .arc_path()
                    .into_iter()
                    .map(|edge| edge as usize)
                    .collect();
                exact += f64::from(predicted_path == *observed_path);

                let predicted_set: HashSet<usize> = predicted_path.iter().copied().collect();
                let observed_set: HashSet<usize> = observed_path.iter().copied().collect();
                let intersection = predicted_set.intersection(&observed_set).count() as f64;
                let sample_precision = intersection / predicted_set.len().max(1) as f64;
                let sample_recall = intersection / observed_set.len().max(1) as f64;
                precision += sample_precision;
                recall += sample_recall;
                f1 += if sample_precision + sample_recall == 0.0 {
                    0.0
                } else {
                    2.0 * sample_precision * sample_recall / (sample_precision + sample_recall)
                };
                let union = predicted_set.union(&observed_set).count();
                jaccard += intersection / union.max(1) as f64;

                let observed_sample_cost = observed_path.iter().try_fold(0u128, |sum, &edge| {
                    let weight = metric
                        .weights()
                        .get(edge)
                        .ok_or_else(|| format!("held-out path edge {edge} is out of bounds"))?;
                    sum.checked_add(*weight as u128)
                        .ok_or_else(|| "held-out path cost overflow".to_string())
                })?;
                observed_cost = observed_cost
                    .checked_add(observed_sample_cost)
                    .ok_or_else(|| "held-out observed-cost sum overflow".to_string())?;
                regret = regret
                    .checked_add(
                        observed_sample_cost
                            .checked_sub(distance as u128)
                            .ok_or_else(|| {
                                format!("negative held-out regret for OD ({source}, {target})")
                            })?,
                    )
                    .ok_or_else(|| "held-out regret overflow".to_string())?;
            }
            Ok((
                chunk.len(),
                exact,
                precision,
                recall,
                f1,
                jaccard,
                regret,
                observed_cost,
            ))
        })
        .collect();

    let mut total = (0usize, 0.0, 0.0, 0.0, 0.0, 0.0, 0u128, 0u128);
    for local in locals {
        let local = local?;
        total.0 += local.0;
        total.1 += local.1;
        total.2 += local.2;
        total.3 += local.3;
        total.4 += local.4;
        total.5 += local.5;
        total.6 = total
            .6
            .checked_add(local.6)
            .ok_or_else(|| "held-out regret overflow".to_string())?;
        total.7 = total
            .7
            .checked_add(local.7)
            .ok_or_else(|| "held-out observed-cost sum overflow".to_string())?;
    }
    let denominator = total.0 as f64;
    Ok(PathMetrics {
        sample_count: total.0,
        exact_match: total.1 / denominator,
        edge_precision: total.2 / denominator,
        edge_recall: total.3 / denominator,
        edge_f1: total.4 / denominator,
        edge_jaccard: total.5 / denominator,
        mean_regret: total.6 as f64 / denominator,
        relative_regret: if total.7 == 0 {
            0.0
        } else {
            total.6 as f64 / total.7 as f64
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use routingkit_cch::{CCH, CCHMetric, compute_order_degree};

    #[test]
    fn validates_empty_bounds_continuity_and_cycles() {
        let tail = [0, 1, 0, 2, 1];
        let head = [1, 3, 2, 3, 0];
        assert_eq!(validate_edge_path(&[], &tail, &head), PathValidity::Empty);
        assert_eq!(
            validate_edge_path(&[99], &tail, &head),
            PathValidity::OutOfBounds
        );
        assert_eq!(
            validate_edge_path(&[0, 3], &tail, &head),
            PathValidity::Discontinuous
        );
        assert_eq!(
            validate_edge_path(&[0, 4], &tail, &head),
            PathValidity::Cyclic
        );
        assert_eq!(
            validate_edge_path(&[0, 1], &tail, &head),
            PathValidity::Valid
        );
    }

    #[test]
    fn grouped_and_ungrouped_oracles_are_equivalent() {
        let tail = vec![0, 1, 0, 2];
        let head = vec![1, 3, 2, 3];
        let weights = vec![5, 5, 2, 2];
        let order = compute_order_degree(4, &tail, &head);
        let cch = CCH::new(&order, &tail, &head, |_| {}, false);
        let metric = CCHMetric::new(&cch, weights.clone());
        let paths = vec![((0, 3), vec![0, 1]), ((0, 3), vec![0, 1])];
        let grouped = group_paths_by_od(&paths);
        let ungrouped = vec![
            OdGroup {
                source: 0,
                target: 3,
                sample_count: 1,
            },
            OdGroup {
                source: 0,
                target: 3,
                sample_count: 1,
            },
        ];
        let grouped_stats = compute_oracle_stats(&metric, &grouped, 4, 16).unwrap();
        let ungrouped_stats = compute_oracle_stats(&metric, &ungrouped, 4, 16).unwrap();
        assert_eq!(grouped_stats.predicted_edge_counts, vec![0, 0, 2, 2]);
        assert_eq!(
            grouped_stats.predicted_edge_counts,
            ungrouped_stats.predicted_edge_counts
        );
        assert_eq!(
            grouped_stats.weighted_shortest_distance_sum,
            ungrouped_stats.weighted_shortest_distance_sum
        );
        assert_eq!(grouped_stats.num_queries, 1);
        assert_eq!(ungrouped_stats.num_queries, 2);

        let observed = compute_observed_edge_counts(&paths, 4, 16);
        let regret = compute_regret(&weights, &observed, &grouped_stats).unwrap();
        assert_eq!(regret.observed_cost_sum, 20);
        assert_eq!(regret.shortest_distance_sum, 8);
        assert_eq!(regret.data_loss_sum, 12);
        assert_eq!(regret.mean_data_loss, 6.0);
        assert_eq!(regret.relative_data_loss, 0.6);
    }

    #[test]
    fn zero_regret_can_have_nonzero_count_residual_under_a_tie() {
        // Observed and selected paths are different but have the same cost.
        // A deterministic oracle may select [0, 1] while [2, 3] is observed.
        let predicted = [1, 1, 0, 0];
        let observed = [0, 0, 1, 1];
        assert_eq!(count_residual_l1(&predicted, &observed).unwrap(), 4);
        let oracle = OracleStats {
            predicted_edge_counts: predicted.to_vec(),
            weighted_shortest_distance_sum: 4,
            sample_count: 1,
            num_queries: 1,
            oracle_duration: Duration::ZERO,
        };
        let regret = compute_regret(&[2, 2, 2, 2], &observed, &oracle).unwrap();
        assert_eq!(regret.data_loss_sum, 0);
        assert_eq!(regret.relative_data_loss, 0.0);
    }

    #[test]
    fn empty_workloads_do_not_create_zero_sized_chunks() {
        assert_eq!(compute_observed_edge_counts(&[], 3, 64), vec![0; 3]);
        let one = vec![((0, 1), vec![0])];
        assert_eq!(compute_observed_edge_counts(&one, 1, 64), vec![1]);
    }

    #[test]
    fn tie_breaking_is_reproducible() {
        let tail = vec![0, 1, 0, 2];
        let head = vec![1, 3, 2, 3];
        let weights = vec![2, 2, 2, 2];
        let order = compute_order_degree(4, &tail, &head);
        let cch = CCH::new(&order, &tail, &head, |_| {}, false);
        let metric = CCHMetric::new(&cch, weights);
        let mut paths = Vec::new();
        for _ in 0..20 {
            let mut query = CCHQuery::new(&metric);
            query.add_source(0, 0);
            query.add_target(3, 0);
            paths.push(query.run().arc_path());
        }
        assert!(paths.windows(2).all(|pair| pair[0] == pair[1]));
    }
}
