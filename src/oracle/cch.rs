use crate::data::{GraphData, OdGroup};
use rayon::prelude::*;
use routingkit_cch::{CCH, CCHMetric, CCHQuery, compute_order_inertial};
use std::time::{Duration, Instant};

pub const CCH_INFINITY: u32 = i32::MAX as u32;

#[derive(Clone, Debug)]
pub struct OracleStats {
    pub predicted_edge_counts: Vec<u64>,
    pub weighted_shortest_distance_sum: u128,
    pub sample_count: u64,
    pub num_queries: usize,
    pub oracle_duration: Duration,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShortestPath {
    pub distance: u32,
    pub original_edges: Vec<usize>,
}

/// CCH topology for the original road graph.
///
/// Metrics are fully customized from a complete positive integer edge-weight
/// vector. Partial customization is intentionally not part of the training
/// oracle API.
pub struct CchOracle {
    cch: CCH,
    edge_count: usize,
}

impl CchOracle {
    pub fn build(graph: &GraphData) -> Result<Self, String> {
        validate_graph(graph)?;
        let order = compute_order_inertial(
            graph.x.len() as u32,
            &graph.tail,
            &graph.head,
            &graph.x,
            &graph.y,
        );
        let cch = CCH::new(&order, &graph.tail, &graph.head, |_| {}, false);
        Ok(Self {
            cch,
            edge_count: graph.tail.len(),
        })
    }

    /// Fully customize the CCH with one complete positive integer metric.
    pub fn customize<'a>(&'a self, weights: &[u32]) -> Result<CCHMetric<'a>, String> {
        validate_metric(weights, self.edge_count)?;
        Ok(CCHMetric::new(&self.cch, weights.to_vec()))
    }

    pub fn shortest_path<'a>(
        &self,
        metric: &'a CCHMetric<'a>,
        source: u32,
        target: u32,
    ) -> Result<ShortestPath, String> {
        self.validate_metric_shape(metric)?;
        let mut query = CCHQuery::new(metric);
        query.add_source(source, 0);
        query.add_target(target, 0);
        let result = query.run();
        let distance = result
            .distance()
            .ok_or_else(|| format!("OD ({source}, {target}) is unreachable"))?;
        let original_edges = result
            .arc_path()
            .into_iter()
            .map(|edge| edge as usize)
            .collect::<Vec<_>>();
        validate_reconstructed_distance(
            metric.weights(),
            &original_edges,
            distance,
            source,
            target,
        )?;
        Ok(ShortestPath {
            distance,
            original_edges,
        })
    }

    /// Query every unique OD once and weight its path counts and distance by
    /// the number of observations in that OD group.
    pub fn batch_stats(
        &self,
        metric: &CCHMetric<'_>,
        groups: &[OdGroup],
        num_chunks: usize,
    ) -> Result<OracleStats, String> {
        self.validate_metric_shape(metric)?;
        let started = Instant::now();
        if groups.is_empty() {
            return Ok(OracleStats {
                predicted_edge_counts: vec![0; self.edge_count],
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
                let mut counts = vec![0u64; self.edge_count];
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
                    let original_edges = result.arc_path();
                    let reconstructed_distance = original_edges.iter().try_fold(
                        0u128,
                        |sum, &edge_id| {
                            let edge = edge_id as usize;
                            let weight = metric.weights().get(edge).ok_or_else(|| {
                                format!("CCH returned invalid original edge id {edge_id}")
                            })?;
                            sum.checked_add(*weight as u128)
                                .ok_or_else(|| "reconstructed path cost overflow".to_string())
                        },
                    )?;
                    if reconstructed_distance != distance as u128 {
                        return Err(format!(
                            "CCH path/distance mismatch for OD ({}, {}): path={}, distance={distance}",
                            group.source, group.target, reconstructed_distance
                        ));
                    }
                    for edge_id in original_edges {
                        let edge = edge_id as usize;
                        counts[edge] = counts[edge]
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
            predicted_edge_counts: vec![0; self.edge_count],
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

    fn validate_metric_shape(&self, metric: &CCHMetric<'_>) -> Result<(), String> {
        if metric.weights().len() != self.edge_count {
            return Err(format!(
                "metric has {} weights but oracle expects {}",
                metric.weights().len(),
                self.edge_count
            ));
        }
        Ok(())
    }
}

fn validate_graph(graph: &GraphData) -> Result<(), String> {
    let edge_count = graph.tail.len();
    if edge_count == 0
        || graph.head.len() != edge_count
        || graph.baseline_weights.len() != edge_count
    {
        return Err(format!(
            "invalid graph edge arrays: tail={edge_count}, head={}, baseline={}",
            graph.head.len(),
            graph.baseline_weights.len()
        ));
    }
    let node_count = graph.x.len();
    if node_count == 0 || graph.y.len() != node_count || node_count > u32::MAX as usize {
        return Err(format!(
            "invalid graph coordinate arrays: x={node_count}, y={}",
            graph.y.len()
        ));
    }
    for (edge, (&tail, &head)) in graph.tail.iter().zip(&graph.head).enumerate() {
        if tail as usize >= node_count || head as usize >= node_count {
            return Err(format!(
                "edge {edge} endpoint out of bounds for {node_count} nodes: {tail}->{head}"
            ));
        }
    }
    validate_metric(&graph.baseline_weights, edge_count)
}

fn validate_metric(weights: &[u32], expected: usize) -> Result<(), String> {
    if weights.len() != expected {
        return Err(format!(
            "metric has {} weights but graph has {expected} edges",
            weights.len()
        ));
    }
    if let Some((edge, weight)) = weights
        .iter()
        .copied()
        .enumerate()
        .find(|(_, weight)| *weight == 0 || *weight >= CCH_INFINITY)
    {
        return Err(format!("edge {edge} has invalid CCH weight {weight}"));
    }
    Ok(())
}

fn validate_reconstructed_distance(
    weights: &[u32],
    path: &[usize],
    distance: u32,
    source: u32,
    target: u32,
) -> Result<(), String> {
    let reconstructed = path.iter().try_fold(0u128, |sum, &edge| {
        let weight = weights
            .get(edge)
            .ok_or_else(|| format!("CCH returned invalid original edge id {edge}"))?;
        sum.checked_add(*weight as u128)
            .ok_or_else(|| "reconstructed path cost overflow".to_string())
    })?;
    if reconstructed != distance as u128 {
        return Err(format!(
            "CCH path/distance mismatch for OD ({source}, {target}): path={reconstructed}, distance={distance}"
        ));
    }
    Ok(())
}

fn chunk_size(len: usize, num_chunks: usize) -> usize {
    len.div_ceil(num_chunks.max(1)).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn grouped_batch_matches_repeated_queries_and_weights_counts() {
        let graph = graph();
        let oracle = CchOracle::build(&graph).unwrap();
        let metric = oracle.customize(&graph.baseline_weights).unwrap();
        let grouped = vec![OdGroup {
            source: 0,
            target: 3,
            sample_count: 2,
        }];
        let repeated = vec![
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
        let grouped_stats = oracle.batch_stats(&metric, &grouped, 16).unwrap();
        let repeated_stats = oracle.batch_stats(&metric, &repeated, 16).unwrap();
        assert_eq!(grouped_stats.predicted_edge_counts, vec![0, 0, 2, 2]);
        assert_eq!(
            grouped_stats.predicted_edge_counts,
            repeated_stats.predicted_edge_counts
        );
        assert_eq!(
            grouped_stats.weighted_shortest_distance_sum,
            repeated_stats.weighted_shortest_distance_sum
        );
        assert_eq!(grouped_stats.num_queries, 1);
        assert_eq!(repeated_stats.num_queries, 2);
    }

    #[test]
    fn reconstructs_original_edge_path() {
        let graph = graph();
        let oracle = CchOracle::build(&graph).unwrap();
        let metric = oracle.customize(&graph.baseline_weights).unwrap();
        let path = oracle.shortest_path(&metric, 0, 3).unwrap();
        assert_eq!(path.distance, 4);
        assert_eq!(path.original_edges, vec![2, 3]);
    }

    #[test]
    fn full_customization_rejects_the_infinity_sentinel() {
        let graph = graph();
        let oracle = CchOracle::build(&graph).unwrap();
        assert!(oracle.customize(&[5, 5, 2, CCH_INFINITY]).is_err());
    }
}
