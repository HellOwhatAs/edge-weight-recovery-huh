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
}

/// Evaluate decoded original-road paths for either graph order.
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
    })
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
    use crate::graph_problem::{GraphOrder, GraphProblem};

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
    fn both_orders_use_the_same_decoded_path_metrics() {
        let graph = graph();
        let raw_paths = vec![((0, 3), vec![0, 1]), ((0, 3), vec![2, 3])];
        for order in [GraphOrder::First, GraphOrder::Second] {
            let problem = GraphProblem::build(&graph, order, 0.1, 10.0).unwrap();
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
        let problem = GraphProblem::build(&graph, GraphOrder::First, 0.1, 10.0).unwrap();
        let metric = problem.customize(problem.initial_weights()).unwrap();
        assert_eq!(
            evaluate_paths(&metric, &[], 4).unwrap(),
            PathMetrics::default()
        );
    }
}
