use crate::graph_problem::{GraphMetric, MappedPath};
use rayon::prelude::*;
use std::collections::HashSet;

type LocalPathMetrics = (usize, f64, f64, f64, f64, f64, f64, f64);

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
                let predicted = query.shortest_path(observed.source, observed.target)?;
                exact += f64::from(predicted.original_edges == observed.original_edges);

                let predicted_set = predicted
                    .original_edges
                    .iter()
                    .copied()
                    .collect::<HashSet<_>>();
                let observed_set = observed
                    .original_edges
                    .iter()
                    .copied()
                    .collect::<HashSet<_>>();
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
