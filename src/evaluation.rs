use crate::graph_problem::{GraphMetric, MappedPath};
use rayon::prelude::*;
use std::collections::HashSet;

type LocalPathMetrics = (usize, f64, f64, f64, f64, f64, f64, f64);

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RouteMetrics {
    pub sample_count: usize,
    pub exact_match: f64,
    pub edge_precision: f64,
    pub edge_recall: f64,
    pub edge_f1: f64,
    pub edge_jaccard: f64,
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
    /// Additive totals retained so disjoint time-bucket evaluations can be
    /// combined exactly under the same per-path metric implementation.
    pub regret_sum: f64,
    pub observed_cost_sum: f64,
}

/// Evaluate decoded original-road paths for either graph representation.
///
/// Graph-specific routing coordinates never enter the overlap metrics. The
/// bound metric decodes every prediction before this common evaluator sees it.
pub fn evaluate_paths(
    metric: &GraphMetric<'_>,
    paths: &[MappedPath],
    num_chunks: usize,
) -> Result<PathMetrics, String> {
    evaluate_bound_paths(metric, paths, num_chunks, false)
}

/// Evaluate the fair NeuroMLR protocol: the true first and last original
/// roads are fixed and the complete road sequence, including both, is scored.
pub fn evaluate_edge_to_edge_paths(
    metric: &GraphMetric<'_>,
    paths: &[MappedPath],
    num_chunks: usize,
) -> Result<PathMetrics, String> {
    evaluate_bound_paths(metric, paths, num_chunks, true)
}

fn evaluate_bound_paths(
    metric: &GraphMetric<'_>,
    paths: &[MappedPath],
    num_chunks: usize,
    edge_to_edge: bool,
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
            let mut regret = 0.0;
            let mut observed_cost = 0.0;

            for observed in chunk {
                let predicted = if edge_to_edge {
                    let source_edge = *observed.original_edges.first().ok_or_else(|| {
                        "edge-to-edge evaluation received an empty observed path".to_string()
                    })?;
                    let target_edge = *observed.original_edges.last().ok_or_else(|| {
                        "edge-to-edge evaluation received an empty observed path".to_string()
                    })?;
                    query.shortest_path_edges(source_edge, target_edge)?
                } else {
                    query.shortest_path(observed.source, observed.target)?
                };
                let sample =
                    sample_route_metrics(&observed.original_edges, &predicted.original_edges);
                exact += sample.0;
                precision += sample.1;
                recall += sample.2;
                f1 += sample.3;
                jaccard += sample.4;

                let sample_observed_cost =
                    path_cost(metric.direct_weights(), &observed.coordinates)?;
                observed_cost += sample_observed_cost;
                regret += sample_observed_cost - predicted.direct_cost;
                if !observed_cost.is_finite() || !regret.is_finite() {
                    return Err("evaluation cost aggregate is not finite".to_string());
                }
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

    let mut total = (0usize, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for local in locals {
        let local = local?;
        total.0 += local.0;
        total.1 += local.1;
        total.2 += local.2;
        total.3 += local.3;
        total.4 += local.4;
        total.5 += local.5;
        total.6 += local.6;
        total.7 += local.7;
    }
    let denominator = total.0 as f64;
    Ok(PathMetrics {
        sample_count: total.0,
        exact_match: total.1 / denominator,
        edge_precision: total.2 / denominator,
        edge_recall: total.3 / denominator,
        edge_f1: total.4 / denominator,
        edge_jaccard: total.5 / denominator,
        mean_regret: total.6 / denominator,
        relative_regret: if total.7 == 0.0 {
            0.0
        } else {
            total.6 / total.7
        },
        regret_sum: total.6,
        observed_cost_sum: total.7,
    })
}

/// Method-independent macro evaluator over complete original-road ID
/// sequences. It is the sole quality aggregation used for exported project
/// and NeuroMLR predictions.
pub fn evaluate_raw_paths(
    observed: &[Vec<usize>],
    predicted: &[Vec<usize>],
) -> Result<RouteMetrics, String> {
    if observed.len() != predicted.len() {
        return Err(format!(
            "raw evaluator length mismatch: observed={}, predicted={}",
            observed.len(),
            predicted.len()
        ));
    }
    if observed.is_empty() {
        return Ok(RouteMetrics::default());
    }
    let mut total = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (truth, prediction) in observed.iter().zip(predicted) {
        if truth.is_empty() || prediction.is_empty() {
            return Err("raw evaluator requires nonempty complete road sequences".to_string());
        }
        let sample = sample_route_metrics(truth, prediction);
        total.0 += sample.0;
        total.1 += sample.1;
        total.2 += sample.2;
        total.3 += sample.3;
        total.4 += sample.4;
    }
    let denominator = observed.len() as f64;
    Ok(RouteMetrics {
        sample_count: observed.len(),
        exact_match: total.0 / denominator,
        edge_precision: total.1 / denominator,
        edge_recall: total.2 / denominator,
        edge_f1: total.3 / denominator,
        edge_jaccard: total.4 / denominator,
    })
}

fn sample_route_metrics(observed: &[usize], predicted: &[usize]) -> (f64, f64, f64, f64, f64) {
    let predicted_set = predicted.iter().copied().collect::<HashSet<_>>();
    let observed_set = observed.iter().copied().collect::<HashSet<_>>();
    let intersection = predicted_set.intersection(&observed_set).count() as f64;
    let precision = intersection / predicted_set.len().max(1) as f64;
    let recall = intersection / observed_set.len().max(1) as f64;
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    let union = predicted_set.union(&observed_set).count();
    (
        f64::from(predicted == observed),
        precision,
        recall,
        f1,
        intersection / union.max(1) as f64,
    )
}

/// Combine metrics from disjoint path partitions, such as departure-time
/// buckets. All reported quantities remain sample-macro averages, and regret
/// uses its additive numerator and denominator rather than averaging ratios.
pub fn combine_path_metrics(parts: &[PathMetrics]) -> PathMetrics {
    let sample_count = parts.iter().map(|part| part.sample_count).sum::<usize>();
    if sample_count == 0 {
        return PathMetrics::default();
    }
    let weighted = |field: fn(&PathMetrics) -> f64| {
        parts
            .iter()
            .map(|part| field(part) * part.sample_count as f64)
            .sum::<f64>()
            / sample_count as f64
    };
    let regret_sum = parts.iter().map(|part| part.regret_sum).sum::<f64>();
    let observed_cost_sum = parts.iter().map(|part| part.observed_cost_sum).sum::<f64>();
    PathMetrics {
        sample_count,
        exact_match: weighted(|part| part.exact_match),
        edge_precision: weighted(|part| part.edge_precision),
        edge_recall: weighted(|part| part.edge_recall),
        edge_f1: weighted(|part| part.edge_f1),
        edge_jaccard: weighted(|part| part.edge_jaccard),
        mean_regret: regret_sum / sample_count as f64,
        relative_regret: if observed_cost_sum == 0.0 {
            0.0
        } else {
            regret_sum / observed_cost_sum
        },
        regret_sum,
        observed_cost_sum,
    }
}

fn path_cost(weights: &[f64], path: &[usize]) -> Result<f64, String> {
    let mut sum = 0.0;
    for &coordinate in path {
        let weight = weights
            .get(coordinate)
            .ok_or_else(|| format!("mapped coordinate {coordinate} is out of bounds"))?;
        sum += weight;
        if !sum.is_finite() {
            return Err("mapped path cost is not finite".to_string());
        }
    }
    Ok(sum)
}

fn chunk_size(len: usize, num_chunks: usize) -> usize {
    len.div_ceil(num_chunks.max(1)).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::GraphData;
    use crate::graph_problem::{GraphProblem, GraphRepresentation};

    fn graph() -> GraphData {
        GraphData {
            tail: vec![0, 1, 0, 2],
            head: vec![1, 3, 2, 3],
            baseline_weights: vec![5, 5, 2, 2],
            x: vec![0.0, 1.0, 1.0, 2.0],
            y: vec![0.0, 0.0, 1.0, 0.0],
        }
    }

    #[test]
    fn both_representations_use_the_same_decoded_path_metrics() {
        let graph = graph();
        let raw_paths = vec![((0, 3), vec![0, 1]), ((0, 3), vec![2, 3])];
        for representation in [
            GraphRepresentation::OriginalEdges,
            GraphRepresentation::EdgeTransitionArcs,
        ] {
            let problem = GraphProblem::build(&graph, representation, 0.1, 10.0).unwrap();
            let mapped = problem.map_paths(&raw_paths).unwrap();
            let metric = problem.customize(problem.initial_weights()).unwrap();
            let metrics = evaluate_paths(&metric, &mapped, 4).unwrap();
            assert_eq!(metrics.sample_count, 2);
            assert_eq!(metrics.exact_match, 0.5);
            assert_eq!(metrics.edge_f1, 0.5);
        }
    }

    #[test]
    fn edge_to_edge_evaluation_scores_complete_paths_with_fixed_endpoint_edges() {
        let graph = graph();
        let problem =
            GraphProblem::build(&graph, GraphRepresentation::EdgeTransitionArcs, 0.1, 10.0)
                .unwrap();
        let mapped = problem
            .map_paths(&[((0, 3), vec![0, 1]), ((0, 3), vec![2, 3])])
            .unwrap();
        let metric = problem.customize(problem.initial_weights()).unwrap();
        let metrics = evaluate_edge_to_edge_paths(&metric, &mapped, 4).unwrap();
        assert_eq!(metrics.sample_count, 2);
        assert_eq!(metrics.exact_match, 1.0);
        assert_eq!(metrics.edge_f1, 1.0);
    }

    #[test]
    fn raw_sequence_evaluator_uses_per_trip_macro_set_metrics() {
        let observed = vec![vec![1, 2, 3], vec![7, 8]];
        let predicted = vec![vec![1, 2, 4, 5], vec![7, 8]];
        let metrics = evaluate_raw_paths(&observed, &predicted).unwrap();
        assert_eq!(metrics.sample_count, 2);
        assert_eq!(metrics.exact_match, 0.5);
        assert!((metrics.edge_precision - 0.75).abs() < 1e-12);
        assert!((metrics.edge_recall - (5.0 / 6.0)).abs() < 1e-12);
        assert!((metrics.edge_f1 - (11.0 / 14.0)).abs() < 1e-12);
        assert!((metrics.edge_jaccard - 0.7).abs() < 1e-12);
        assert!(evaluate_raw_paths(&observed, &predicted[..1]).is_err());
    }

    #[test]
    fn empty_evaluation_is_well_defined() {
        let graph = graph();
        let problem =
            GraphProblem::build(&graph, GraphRepresentation::OriginalEdges, 0.1, 10.0).unwrap();
        let metric = problem.customize(problem.initial_weights()).unwrap();
        assert_eq!(
            evaluate_paths(&metric, &[], 4).unwrap(),
            PathMetrics::default()
        );
    }

    #[test]
    fn disjoint_time_bucket_metrics_combine_with_additive_regret() {
        let first = PathMetrics {
            sample_count: 1,
            exact_match: 1.0,
            edge_precision: 1.0,
            edge_recall: 1.0,
            edge_f1: 1.0,
            edge_jaccard: 1.0,
            mean_regret: 2.0,
            relative_regret: 0.2,
            regret_sum: 2.0,
            observed_cost_sum: 10.0,
        };
        let second = PathMetrics {
            sample_count: 3,
            exact_match: 0.0,
            edge_precision: 0.5,
            edge_recall: 0.25,
            edge_f1: 1.0 / 3.0,
            edge_jaccard: 0.2,
            mean_regret: 4.0,
            relative_regret: 0.4,
            regret_sum: 12.0,
            observed_cost_sum: 30.0,
        };
        let combined = combine_path_metrics(&[first, second]);
        assert_eq!(combined.sample_count, 4);
        assert_eq!(combined.exact_match, 0.25);
        assert_eq!(combined.edge_precision, 0.625);
        assert_eq!(combined.mean_regret, 3.5);
        assert_eq!(combined.relative_regret, 0.35);
    }
}
