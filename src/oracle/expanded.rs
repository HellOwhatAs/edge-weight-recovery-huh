use super::cch::CCH_INFINITY;
use crate::data::{GraphData, OdGroup};
use crate::turn_graph::{ExpandedPath, ExpandedTurnGraph, validate_decoded_path};
use rayon::prelude::*;
use routingkit_cch::{CCH, CCHMetric, CCHQuery, compute_order_inertial};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Aggregate predictions from one batch of unique original-node OD queries on
/// the expanded edge-state graph.
#[derive(Clone, Debug)]
pub struct ExpandedOracleStats {
    pub predicted_edge_counts: Vec<u64>,
    pub predicted_transition_counts: Vec<u64>,
    pub weighted_shortest_distance_sum: u128,
    pub sample_count: u64,
    pub num_queries: usize,
    pub oracle_duration: Duration,
}

/// Full-customization CCH topology for one specific expanded turn graph.
///
/// The original graph and expansion remain bound to this oracle so a metric
/// cannot later be queried with an unrelated edge-state mapping.
pub struct ExpandedCchOracle<'g> {
    graph: &'g GraphData,
    expanded: &'g ExpandedTurnGraph,
    cch: CCH,
    topology_identity: String,
}

/// One fully bound expanded metric.
///
/// The CCH metric owns the transition weights. The separate edge weights are
/// the source-state offsets used to pay the first original edge. Both are
/// private so queries and observed costs cannot accidentally mix metric states.
pub struct ExpandedMetric<'a, 'g> {
    cch_metric: CCHMetric<'a>,
    graph: &'g GraphData,
    expanded: &'g ExpandedTurnGraph,
    edge_weights: Vec<u32>,
    topology_identity: String,
}

/// Reusable query state tied to one [`ExpandedMetric`].
///
/// This is the allocation-efficient interface for route-level evaluation. It
/// deliberately exposes neither the raw CCH query nor independently supplied
/// weight vectors.
pub struct ExpandedQuery<'a, 'g> {
    cch_query: CCHQuery<'a>,
    graph: &'g GraphData,
    expanded: &'g ExpandedTurnGraph,
    edge_weights: &'a [u32],
    transition_weights: &'a [u32],
}

impl<'g> ExpandedCchOracle<'g> {
    pub fn build(graph: &'g GraphData, expanded: &'g ExpandedTurnGraph) -> Result<Self, String> {
        validate_bound_topology(graph, expanded)?;
        let order = compute_order_inertial(
            expanded.stats.expanded_nodes as u32,
            &expanded.tail,
            &expanded.head,
            &expanded.state_x,
            &expanded.state_y,
        );
        let topology_identity = topology_identity(graph, expanded, &order);
        let cch = CCH::new(&order, &expanded.tail, &expanded.head, |_| {}, false);
        Ok(Self {
            graph,
            expanded,
            cch,
            topology_identity,
        })
    }

    /// Fully customize a metric from one inseparable pair of original-edge
    /// source offsets and expanded transition weights.
    pub fn customize<'a>(
        &'a self,
        edge_weights: &[u32],
        transition_weights: &[u32],
    ) -> Result<ExpandedMetric<'a, 'g>, String> {
        validate_weights(self.expanded, edge_weights, transition_weights)?;
        Ok(ExpandedMetric {
            cch_metric: CCHMetric::new(&self.cch, transition_weights.to_vec()),
            graph: self.graph,
            expanded: self.expanded,
            edge_weights: edge_weights.to_vec(),
            topology_identity: self.topology_identity.clone(),
        })
    }

    pub fn topology_identity(&self) -> &str {
        &self.topology_identity
    }
}

impl<'metric, 'g> ExpandedMetric<'metric, 'g> {
    pub fn edge_weights(&self) -> &[u32] {
        &self.edge_weights
    }

    pub fn transition_weights(&self) -> &[u32] {
        self.cch_metric.weights()
    }

    pub fn topology_identity(&self) -> &str {
        &self.topology_identity
    }

    pub fn new_query(&self) -> ExpandedQuery<'_, 'g> {
        ExpandedQuery {
            cch_query: CCHQuery::new(&self.cch_metric),
            graph: self.graph,
            expanded: self.expanded,
            edge_weights: &self.edge_weights,
            transition_weights: self.cch_metric.weights(),
        }
    }

    pub fn query(&self, source: u32, target: u32) -> Result<ExpandedPath, String> {
        self.new_query().query(source, target)
    }

    /// Cost an observed complete original-edge path with exactly the metric
    /// used by this metric's CCH queries.
    pub fn observed_path_cost(&self, path: &[usize]) -> Result<u64, String> {
        self.expanded.observed_path_cost(
            self.graph,
            &self.edge_weights,
            self.cch_metric.weights(),
            path,
        )
    }

    /// Query every supplied OD group once and weight its decoded edge and
    /// transition counts by the number of observations in that group.
    ///
    /// Edge counts are dense per worker. Transition counts are sparse per
    /// worker so memory does not scale as `threads * transition_count`.
    pub fn batch_stats(
        &self,
        groups: &[OdGroup],
        num_chunks: usize,
    ) -> Result<ExpandedOracleStats, String> {
        let started = Instant::now();
        let edge_count = self.expanded.stats.original_edges;
        let transition_count = self.expanded.transition_count();
        if groups.is_empty() {
            return Ok(ExpandedOracleStats {
                predicted_edge_counts: vec![0; edge_count],
                predicted_transition_counts: vec![0; transition_count],
                weighted_shortest_distance_sum: 0,
                sample_count: 0,
                num_queries: 0,
                oracle_duration: started.elapsed(),
            });
        }

        type SparseTransitionCounts = HashMap<usize, u64>;
        type LocalStats = (Vec<u64>, SparseTransitionCounts, u128, u64, usize);
        let locals: Vec<Result<LocalStats, String>> = groups
            .par_chunks(chunk_size(groups.len(), num_chunks))
            .map(|chunk| {
                let mut query = self.new_query();
                let mut edge_counts = vec![0u64; edge_count];
                let mut transition_counts = HashMap::<usize, u64>::new();
                let mut distance_sum = 0u128;
                let mut sample_count = 0u64;

                for group in chunk {
                    let path = query.query(group.source, group.target)?;
                    for &edge in &path.original_edges {
                        edge_counts[edge] = edge_counts[edge]
                            .checked_add(group.sample_count)
                            .ok_or_else(|| "predicted edge count overflow".to_string())?;
                    }
                    for pair in path.original_edges.windows(2) {
                        let transition = self
                            .expanded
                            .transition_id(pair[0], pair[1])
                            .ok_or_else(|| {
                                format!(
                                    "decoded path contains missing transition {} -> {}",
                                    pair[0], pair[1]
                                )
                            })?
                            .index();
                        let count = transition_counts.entry(transition).or_default();
                        *count = count
                            .checked_add(group.sample_count)
                            .ok_or_else(|| "predicted transition count overflow".to_string())?;
                    }
                    let weighted_distance = (path.distance as u128)
                        .checked_mul(group.sample_count as u128)
                        .ok_or_else(|| "shortest-distance product overflow".to_string())?;
                    distance_sum = distance_sum
                        .checked_add(weighted_distance)
                        .ok_or_else(|| "shortest-distance sum overflow".to_string())?;
                    sample_count = sample_count
                        .checked_add(group.sample_count)
                        .ok_or_else(|| "oracle sample count overflow".to_string())?;
                }
                Ok((
                    edge_counts,
                    transition_counts,
                    distance_sum,
                    sample_count,
                    chunk.len(),
                ))
            })
            .collect();

        let mut stats = ExpandedOracleStats {
            predicted_edge_counts: vec![0; edge_count],
            predicted_transition_counts: vec![0; transition_count],
            weighted_shortest_distance_sum: 0,
            sample_count: 0,
            num_queries: 0,
            oracle_duration: Duration::ZERO,
        };
        for local in locals {
            let (edge_counts, transition_counts, distance_sum, sample_count, queries) = local?;
            for (total, addend) in stats.predicted_edge_counts.iter_mut().zip(edge_counts) {
                *total = total
                    .checked_add(addend)
                    .ok_or_else(|| "predicted edge count overflow".to_string())?;
            }
            for (transition, addend) in transition_counts {
                let total = stats
                    .predicted_transition_counts
                    .get_mut(transition)
                    .ok_or_else(|| format!("invalid predicted transition {transition}"))?;
                *total = total
                    .checked_add(addend)
                    .ok_or_else(|| "predicted transition count overflow".to_string())?;
            }
            stats.weighted_shortest_distance_sum = stats
                .weighted_shortest_distance_sum
                .checked_add(distance_sum)
                .ok_or_else(|| "shortest-distance sum overflow".to_string())?;
            stats.sample_count = stats
                .sample_count
                .checked_add(sample_count)
                .ok_or_else(|| "oracle sample count overflow".to_string())?;
            stats.num_queries = stats
                .num_queries
                .checked_add(queries)
                .ok_or_else(|| "oracle query count overflow".to_string())?;
        }
        stats.oracle_duration = started.elapsed();
        Ok(stats)
    }
}

impl ExpandedQuery<'_, '_> {
    pub fn query(&mut self, source: u32, target: u32) -> Result<ExpandedPath, String> {
        query_with(
            &mut self.cch_query,
            self.graph,
            self.expanded,
            self.edge_weights,
            self.transition_weights,
            source,
            target,
        )
    }
}

fn query_with(
    query: &mut CCHQuery<'_>,
    graph: &GraphData,
    expanded: &ExpandedTurnGraph,
    edge_weights: &[u32],
    transition_weights: &[u32],
    source: u32,
    target: u32,
) -> Result<ExpandedPath, String> {
    let sources = expanded.source_states(source)?;
    let targets = expanded.target_states(target)?;
    if sources.is_empty() || targets.is_empty() {
        return Err(format!(
            "expanded OD ({source}, {target}) has {} source states and {} target states",
            sources.len(),
            targets.len()
        ));
    }
    for &state in sources {
        query.add_source(state, edge_weights[state as usize]);
    }
    for &state in targets {
        query.add_target(state, 0);
    }

    let result = query.run();
    let distance = result
        .distance()
        .ok_or_else(|| format!("expanded OD ({source}, {target}) is unreachable"))?;
    let original_edges = result
        .node_path()
        .into_iter()
        .map(|edge| edge as usize)
        .collect::<Vec<_>>();
    drop(result);

    validate_decoded_path(graph, &original_edges, source, target)?;
    let reconstructed =
        expanded.observed_path_cost(graph, edge_weights, transition_weights, &original_edges)?;
    if reconstructed != distance as u64 {
        return Err(format!(
            "expanded OD ({source}, {target}) distance/path mismatch: distance={distance}, reconstructed={reconstructed}"
        ));
    }
    Ok(ExpandedPath {
        distance,
        original_edges,
    })
}

fn validate_bound_topology(graph: &GraphData, expanded: &ExpandedTurnGraph) -> Result<(), String> {
    let edge_count = graph.tail.len();
    let node_count = graph.x.len();
    if edge_count == 0
        || graph.head.len() != edge_count
        || graph.baseline_weights.len() != edge_count
    {
        return Err("invalid original graph edge arrays".to_string());
    }
    if node_count == 0 || graph.y.len() != node_count || node_count > u32::MAX as usize {
        return Err("invalid original graph coordinate arrays".to_string());
    }
    if expanded.stats.original_nodes != node_count
        || expanded.stats.original_edges != edge_count
        || expanded.stats.expanded_nodes != edge_count
        || expanded.stats.expanded_arcs != expanded.tail.len()
        || expanded.head.len() != expanded.tail.len()
        || expanded.state_x.len() != edge_count
        || expanded.state_y.len() != edge_count
    {
        return Err("expanded graph shape does not match original graph".to_string());
    }
    if expanded
        .state_x
        .iter()
        .chain(&expanded.state_y)
        .any(|coordinate| !coordinate.is_finite())
    {
        return Err("expanded graph contains a non-finite state coordinate".to_string());
    }
    for (transition, previous, next) in expanded.transitions() {
        if graph.head.get(previous) != graph.tail.get(next) {
            return Err(format!(
                "expanded transition {} is discontinuous: {previous} -> {next}",
                transition.index()
            ));
        }
        if expanded.transition_id(previous, next) != Some(transition) {
            return Err(format!(
                "expanded transition {} is inconsistent with the stable transition index",
                transition.index()
            ));
        }
    }
    Ok(())
}

fn validate_weights(
    expanded: &ExpandedTurnGraph,
    edge_weights: &[u32],
    transition_weights: &[u32],
) -> Result<(), String> {
    if edge_weights.len() != expanded.stats.original_edges {
        return Err(format!(
            "edge weight count {} does not match expanded state count {}",
            edge_weights.len(),
            expanded.stats.original_edges
        ));
    }
    if transition_weights.len() != expanded.transition_count() {
        return Err(format!(
            "transition weight count {} does not match expanded transition count {}",
            transition_weights.len(),
            expanded.transition_count()
        ));
    }
    if let Some((edge, weight)) = edge_weights
        .iter()
        .copied()
        .enumerate()
        .find(|(_, weight)| *weight == 0 || *weight >= CCH_INFINITY)
    {
        return Err(format!("edge {edge} has invalid CCH weight {weight}"));
    }
    for (transition, _, next) in expanded.transitions() {
        let weight = transition_weights[transition.index()];
        if weight == 0 || weight >= CCH_INFINITY {
            return Err(format!(
                "transition {} has invalid CCH weight {weight}",
                transition.index()
            ));
        }
        if weight < edge_weights[next] {
            return Err(format!(
                "transition {} weight {weight} is below next-edge {next} weight {}",
                transition.index(),
                edge_weights[next]
            ));
        }
    }
    Ok(())
}

fn topology_identity(graph: &GraphData, expanded: &ExpandedTurnGraph, order: &[u32]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    hash_u64(&mut hash, graph.x.len() as u64);
    hash_u32_slice(&mut hash, &graph.tail);
    hash_u32_slice(&mut hash, &graph.head);
    hash_f32_slice(&mut hash, &graph.x);
    hash_f32_slice(&mut hash, &graph.y);
    hash_u32_slice(&mut hash, &expanded.tail);
    hash_u32_slice(&mut hash, &expanded.head);
    hash_f32_slice(&mut hash, &expanded.state_x);
    hash_f32_slice(&mut hash, &expanded.state_y);
    hash_u32_slice(&mut hash, order);
    format!("fnv1a64:{hash:016x}")
}

fn hash_u32_slice(hash: &mut u64, values: &[u32]) {
    hash_u64(hash, values.len() as u64);
    for &value in values {
        hash_bytes(hash, &value.to_le_bytes());
    }
}

fn hash_f32_slice(hash: &mut u64, values: &[f32]) {
    hash_u64(hash, values.len() as u64);
    for &value in values {
        hash_bytes(hash, &value.to_bits().to_le_bytes());
    }
}

fn hash_u64(hash: &mut u64, value: u64) {
    hash_bytes(hash, &value.to_le_bytes());
}

fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *hash ^= byte as u64;
        *hash = hash.wrapping_mul(0x100000001b3);
    }
}

fn chunk_size(len: usize, num_chunks: usize) -> usize {
    len.div_ceil(num_chunks.max(1)).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph() -> GraphData {
        GraphData {
            tail: vec![0, 1, 0, 3, 1],
            head: vec![1, 2, 3, 2, 3],
            baseline_weights: vec![2, 2, 2, 3, 10],
            x: vec![0.0, 1.0, 1.0, 0.0],
            y: vec![0.0, 0.0, 1.0, 1.0],
        }
    }

    #[test]
    fn binds_query_and_observed_cost_to_one_metric() {
        let graph = graph();
        let expanded = ExpandedTurnGraph::build(&graph).unwrap();
        let oracle = ExpandedCchOracle::build(&graph, &expanded).unwrap();

        let zero = vec![0.0; expanded.transition_count()];
        let zero_weights = expanded
            .transition_metric_weights(&graph.baseline_weights, &zero, 1.0)
            .unwrap();
        let zero_metric = oracle
            .customize(&graph.baseline_weights, &zero_weights)
            .unwrap();
        let mut reusable = zero_metric.new_query();
        assert_eq!(reusable.query(0, 2).unwrap().original_edges, vec![0, 1]);
        assert_eq!(reusable.query(1, 2).unwrap().original_edges, vec![1]);
        assert_eq!(zero_metric.observed_path_cost(&[0, 1]).unwrap(), 4);

        let mut penalized = zero;
        penalized[expanded.transition_id(0, 1).unwrap().index()] = 2.0;
        let penalized_weights = expanded
            .transition_metric_weights(&graph.baseline_weights, &penalized, 1.0)
            .unwrap();
        let penalized_metric = oracle
            .customize(&graph.baseline_weights, &penalized_weights)
            .unwrap();
        let path = penalized_metric.query(0, 2).unwrap();
        assert_eq!(path.distance, 5);
        assert_eq!(path.original_edges, vec![2, 3]);
        assert_eq!(penalized_metric.observed_path_cost(&[0, 1]).unwrap(), 6);
    }

    #[test]
    fn grouped_batch_counts_edges_and_transitions_with_sample_weights() {
        let graph = graph();
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
        let groups = vec![
            OdGroup {
                source: 0,
                target: 2,
                sample_count: 2,
            },
            OdGroup {
                source: 1,
                target: 2,
                sample_count: 1,
            },
        ];

        let stats = metric.batch_stats(&groups, 16).unwrap();
        assert_eq!(stats.predicted_edge_counts, vec![2, 3, 0, 0, 0]);
        let mut expected_transitions = vec![0; expanded.transition_count()];
        expected_transitions[expanded.transition_id(0, 1).unwrap().index()] = 2;
        assert_eq!(stats.predicted_transition_counts, expected_transitions);
        assert_eq!(stats.weighted_shortest_distance_sum, 10);
        assert_eq!(stats.sample_count, 3);
        assert_eq!(stats.num_queries, 2);
    }

    #[test]
    fn topology_identity_is_stable_and_includes_coordinates() {
        let original_graph = graph();
        let expanded = ExpandedTurnGraph::build(&original_graph).unwrap();
        let first = ExpandedCchOracle::build(&original_graph, &expanded).unwrap();
        let second = ExpandedCchOracle::build(&original_graph, &expanded).unwrap();
        assert_eq!(first.topology_identity(), second.topology_identity());

        let mut moved_graph = graph();
        moved_graph.x[1] = 1.25;
        let moved_expanded = ExpandedTurnGraph::build(&moved_graph).unwrap();
        let moved = ExpandedCchOracle::build(&moved_graph, &moved_expanded).unwrap();
        assert_ne!(first.topology_identity(), moved.topology_identity());
    }

    #[test]
    fn rejects_transition_weights_below_their_next_edge_cost() {
        let graph = graph();
        let expanded = ExpandedTurnGraph::build(&graph).unwrap();
        let oracle = ExpandedCchOracle::build(&graph, &expanded).unwrap();
        let mut weights = expanded
            .transition_metric_weights(
                &graph.baseline_weights,
                &vec![0.0; expanded.transition_count()],
                1.0,
            )
            .unwrap();
        let transition = expanded.transition_id(0, 1).unwrap();
        weights[transition.index()] = graph.baseline_weights[1] - 1;
        assert!(oracle.customize(&graph.baseline_weights, &weights).is_err());
    }

    #[test]
    fn empty_batch_has_well_defined_dense_counts() {
        let graph = graph();
        let expanded = ExpandedTurnGraph::build(&graph).unwrap();
        let oracle = ExpandedCchOracle::build(&graph, &expanded).unwrap();
        let weights = expanded
            .transition_metric_weights(
                &graph.baseline_weights,
                &vec![0.0; expanded.transition_count()],
                1.0,
            )
            .unwrap();
        let metric = oracle.customize(&graph.baseline_weights, &weights).unwrap();
        let stats = metric.batch_stats(&[], 0).unwrap();
        assert_eq!(stats.predicted_edge_counts, vec![0; graph.tail.len()]);
        assert_eq!(
            stats.predicted_transition_counts,
            vec![0; expanded.transition_count()]
        );
        assert_eq!(stats.sample_count, 0);
        assert_eq!(stats.num_queries, 0);
    }
}
