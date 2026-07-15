use crate::data::TripPath;
use crate::oracle::ExpandedMetric;
use rayon::prelude::*;
use routingkit_cch::{CCHMetric, CCHQuery};
use std::collections::HashSet;

type LocalPathMetrics = (usize, f64, f64, f64, f64, f64, u128, u128);

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PathMetrics {
    pub sample_count: usize,
    pub exact_match: f64,
    pub edge_precision: f64,
    pub edge_recall: f64,
    pub edge_f1: f64,
    pub edge_jaccard: f64,
    pub mean_regret: f64,
    /// Aggregate regret divided by aggregate observed path cost under the
    /// current model. This model-relative denominator changes with learned
    /// cost scale, so the value must not be the sole cross-model ranking metric
    /// for edge-only, frozen-edge turn-only, and joint edge-turn models.
    pub relative_regret: f64,
}

/// Evaluate the standard held-out route metrics for one fully customized CCH
/// metric. Test data must be supplied only after validation-based model
/// selection is complete.
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

                let reconstructed_shortest = path_cost(metric.weights(), &predicted_path)?;
                if reconstructed_shortest != distance as u128 {
                    return Err(format!(
                        "CCH path/distance mismatch for held-out OD ({source}, {target}): path={reconstructed_shortest}, distance={distance}"
                    ));
                }
                let observed_sample_cost = path_cost(metric.weights(), observed_path)?;
                observed_cost = observed_cost
                    .checked_add(observed_sample_cost)
                    .ok_or_else(|| "held-out observed-cost sum overflow".to_string())?;
                regret = regret
                    .checked_add(observed_sample_cost.checked_sub(distance as u128).ok_or_else(
                        || format!("negative held-out regret for OD ({source}, {target})"),
                    )?)
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

/// Evaluate the same standard route metrics on a bound expanded metric.
/// Predicted state-node paths are decoded to original edge IDs before route
/// overlap is measured.
pub fn evaluate_expanded_paths(
    metric: &ExpandedMetric<'_, '_>,
    paths: &[TripPath],
    num_chunks: usize,
) -> Result<PathMetrics, String> {
    if paths.is_empty() {
        return Ok(PathMetrics::default());
    }

    let locals: Vec<Result<LocalPathMetrics, String>> = paths
        .par_chunks(chunk_size(paths.len(), num_chunks))
        .map(|chunk| {
            let mut query = metric.new_query();
            let mut exact = 0.0;
            let mut precision = 0.0;
            let mut recall = 0.0;
            let mut f1 = 0.0;
            let mut jaccard = 0.0;
            let mut regret = 0u128;
            let mut observed_cost = 0u128;

            for ((source, target), observed_path) in chunk {
                let predicted = query.query(*source, *target)?;
                exact += f64::from(predicted.original_edges == *observed_path);

                let predicted_set: HashSet<usize> =
                    predicted.original_edges.iter().copied().collect();
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

                let observed_sample_cost = metric.observed_path_cost(observed_path)? as u128;
                observed_cost = observed_cost
                    .checked_add(observed_sample_cost)
                    .ok_or_else(|| "expanded held-out observed-cost sum overflow".to_string())?;
                regret = regret
                    .checked_add(
                        observed_sample_cost
                            .checked_sub(predicted.distance as u128)
                            .ok_or_else(|| {
                                format!(
                                    "negative expanded held-out regret for OD ({source}, {target})"
                                )
                            })?,
                    )
                    .ok_or_else(|| "expanded held-out regret overflow".to_string())?;
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

    aggregate_path_metrics(locals, "expanded held-out")
}

fn aggregate_path_metrics(
    locals: Vec<Result<LocalPathMetrics, String>>,
    label: &str,
) -> Result<PathMetrics, String> {
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
            .ok_or_else(|| format!("{label} regret overflow"))?;
        total.7 = total
            .7
            .checked_add(local.7)
            .ok_or_else(|| format!("{label} observed-cost sum overflow"))?;
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

fn path_cost(weights: &[u32], path: &[usize]) -> Result<u128, String> {
    path.iter().try_fold(0u128, |sum, &edge| {
        let weight = weights
            .get(edge)
            .ok_or_else(|| format!("held-out path edge {edge} is out of bounds"))?;
        sum.checked_add(*weight as u128)
            .ok_or_else(|| "held-out path cost overflow".to_string())
    })
}

fn chunk_size(len: usize, num_chunks: usize) -> usize {
    len.div_ceil(num_chunks.max(1)).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::GraphData;
    use crate::oracle::ExpandedCchOracle;
    use crate::turn_graph::ExpandedTurnGraph;
    use routingkit_cch::{CCH, CCHMetric, compute_order_degree};

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn reports_only_the_standard_aggregate_metrics() {
        let tail = vec![0, 1, 0, 2];
        let head = vec![1, 3, 2, 3];
        let weights = vec![5, 5, 2, 2];
        let order = compute_order_degree(4, &tail, &head);
        let cch = CCH::new(&order, &tail, &head, |_| {}, false);
        let metric = CCHMetric::new(&cch, weights);
        let paths = vec![((0, 3), vec![0, 1]), ((0, 3), vec![2, 3])];

        let metrics = evaluate_paths(&metric, &paths, 16).unwrap();
        assert_eq!(metrics.sample_count, 2);
        assert_close(metrics.mean_regret, 3.0);
        assert_close(metrics.relative_regret, 6.0 / 14.0);
        assert_close(metrics.exact_match, 0.5);
        assert_close(metrics.edge_precision, 0.5);
        assert_close(metrics.edge_recall, 0.5);
        assert_close(metrics.edge_f1, 0.5);
        assert_close(metrics.edge_jaccard, 0.5);
    }

    #[test]
    fn empty_evaluation_is_well_defined() {
        let tail = vec![0];
        let head = vec![1];
        let order = compute_order_degree(2, &tail, &head);
        let cch = CCH::new(&order, &tail, &head, |_| {}, false);
        let metric = CCHMetric::new(&cch, vec![1]);
        assert_eq!(
            evaluate_paths(&metric, &[], 8).unwrap(),
            PathMetrics::default()
        );
    }

    #[test]
    fn expanded_evaluation_decodes_to_the_same_standard_edge_metrics() {
        let graph = GraphData {
            tail: vec![0, 1, 0, 2],
            head: vec![1, 3, 2, 3],
            baseline_weights: vec![5, 5, 2, 2],
            x: vec![0.0, 1.0, 1.0, 2.0],
            y: vec![0.0, 0.0, 1.0, 0.0],
        };
        let expanded = ExpandedTurnGraph::build(&graph).unwrap();
        let oracle = ExpandedCchOracle::build(&graph, &expanded).unwrap();
        let transition_weights = expanded
            .transition_metric_weights(
                &graph.baseline_weights,
                &vec![0.0; expanded.transition_count()],
                1.0,
            )
            .unwrap();
        let metric = oracle
            .customize(&graph.baseline_weights, &transition_weights)
            .unwrap();
        let paths = vec![((0, 3), vec![0, 1]), ((0, 3), vec![2, 3])];

        let metrics = evaluate_expanded_paths(&metric, &paths, 16).unwrap();
        assert_eq!(metrics.sample_count, 2);
        assert_close(metrics.mean_regret, 3.0);
        assert_close(metrics.relative_regret, 6.0 / 14.0);
        assert_close(metrics.exact_match, 0.5);
        assert_close(metrics.edge_precision, 0.5);
        assert_close(metrics.edge_recall, 0.5);
        assert_close(metrics.edge_f1, 0.5);
        assert_close(metrics.edge_jaccard, 0.5);
    }
}
