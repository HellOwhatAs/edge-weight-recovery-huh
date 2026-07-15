//! Edge-state (line-graph) expansion for a single global left-turn penalty.

use crate::graph::GraphData;
use crate::turn::{TurnCategory, turn_geometry};
use routingkit_cch::{CCHMetric, CCHQuery};

pub const MAX_EXPANDED_ARCS: usize = 12_000_000;
pub const MAX_RAW_EXPANDED_BYTES: u64 = 512 * 1024 * 1024;
pub const CCH_INFINITY: u32 = i32::MAX as u32;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpandedGraphStats {
    pub original_nodes: usize,
    pub original_edges: usize,
    pub expanded_nodes: usize,
    pub expanded_arcs: usize,
    pub skipped_state_self_arcs: usize,
    pub left_turn_arcs: usize,
    pub unclassifiable_turn_arcs: usize,
    /// Tail/head, one-byte turn flag, one metric, coordinates, and order.
    pub estimated_raw_expanded_bytes: u64,
}

/// One expanded node per original directed edge. An expanded arc `e -> f`
/// exists exactly when `head(e) == tail(f)`, except that `e == f` self-arcs
/// are omitted because positive weights make them irrelevant to shortest paths.
#[derive(Debug)]
pub struct ExpandedTurnGraph {
    pub tail: Vec<u32>,
    pub head: Vec<u32>,
    /// One byte per expanded arc; `1` means the transition is a left turn.
    pub left_turn: Vec<u8>,
    pub state_x: Vec<f32>,
    pub state_y: Vec<f32>,
    edges_by_tail_offsets: Vec<usize>,
    edges_by_tail: Vec<u32>,
    edges_by_head_offsets: Vec<usize>,
    edges_by_head: Vec<u32>,
    pub stats: ExpandedGraphStats,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpandedPath {
    pub distance: u32,
    pub original_edges: Vec<usize>,
}

impl ExpandedTurnGraph {
    pub fn build(graph: &GraphData) -> Result<Self, String> {
        let edge_count = graph.tail.len();
        if graph.head.len() != edge_count || graph.baseline_weights.len() != edge_count {
            return Err(format!(
                "original edge-array length mismatch: tail={edge_count}, head={}, weights={}",
                graph.head.len(),
                graph.baseline_weights.len()
            ));
        }
        if edge_count == 0 || edge_count > u32::MAX as usize {
            return Err(format!(
                "original edge count {edge_count} must be in 1..={}",
                u32::MAX
            ));
        }
        let node_count = graph.x.len();
        if graph.y.len() != node_count || node_count == 0 {
            return Err(format!(
                "coordinate-array length mismatch or empty graph: x={node_count}, y={}",
                graph.y.len()
            ));
        }
        for (edge, (&tail, &head)) in graph.tail.iter().zip(&graph.head).enumerate() {
            if tail as usize >= node_count || head as usize >= node_count {
                return Err(format!(
                    "original edge {edge} endpoint out of bounds for {node_count} nodes: {tail}->{head}"
                ));
            }
        }

        let (edges_by_tail_offsets, edges_by_tail) =
            build_incidence(node_count, &graph.tail, edge_count)?;
        let (edges_by_head_offsets, edges_by_head) =
            build_incidence(node_count, &graph.head, edge_count)?;

        let mut arc_count = 0usize;
        let mut skipped_state_self_arcs = 0usize;
        for edge in 0..edge_count {
            let junction = graph.head[edge] as usize;
            for &next in incidence_slice(&edges_by_tail_offsets, &edges_by_tail, junction) {
                if next as usize == edge {
                    skipped_state_self_arcs += 1;
                } else {
                    arc_count = arc_count
                        .checked_add(1)
                        .ok_or_else(|| "expanded arc count overflow".to_string())?;
                }
            }
        }
        if arc_count > MAX_EXPANDED_ARCS {
            return Err(format!(
                "expanded graph would contain {arc_count} arcs, exceeding hard limit {MAX_EXPANDED_ARCS}"
            ));
        }
        let estimated_raw_expanded_bytes = estimate_raw_bytes(arc_count, edge_count)?;
        if estimated_raw_expanded_bytes > MAX_RAW_EXPANDED_BYTES {
            return Err(format!(
                "estimated raw expanded arrays require {estimated_raw_expanded_bytes} bytes, exceeding hard limit {MAX_RAW_EXPANDED_BYTES}"
            ));
        }

        let mut tail = Vec::with_capacity(arc_count);
        let mut head = Vec::with_capacity(arc_count);
        let mut left_turn = Vec::with_capacity(arc_count);
        let mut left_turn_arcs = 0usize;
        let mut unclassifiable_turn_arcs = 0usize;
        for edge in 0..edge_count {
            let junction = graph.head[edge] as usize;
            for &next in incidence_slice(&edges_by_tail_offsets, &edges_by_tail, junction) {
                let next = next as usize;
                if next == edge {
                    continue;
                }
                let is_left = match turn_geometry(graph, edge, next) {
                    Some(turn) => turn.category == TurnCategory::Left,
                    None => {
                        unclassifiable_turn_arcs += 1;
                        false
                    }
                };
                left_turn_arcs += usize::from(is_left);
                tail.push(edge as u32);
                head.push(next as u32);
                left_turn.push(u8::from(is_left));
            }
        }
        debug_assert_eq!(tail.len(), arc_count);

        let mut state_x = Vec::with_capacity(edge_count);
        let mut state_y = Vec::with_capacity(edge_count);
        for edge in 0..edge_count {
            let tail_node = graph.tail[edge] as usize;
            let head_node = graph.head[edge] as usize;
            let x = (graph.x[tail_node] + graph.x[head_node]) * 0.5;
            let y = (graph.y[tail_node] + graph.y[head_node]) * 0.5;
            if !x.is_finite() || !y.is_finite() {
                return Err(format!("expanded state {edge} has non-finite coordinates"));
            }
            state_x.push(x);
            state_y.push(y);
        }

        Ok(Self {
            tail,
            head,
            left_turn,
            state_x,
            state_y,
            edges_by_tail_offsets,
            edges_by_tail,
            edges_by_head_offsets,
            edges_by_head,
            stats: ExpandedGraphStats {
                original_nodes: node_count,
                original_edges: edge_count,
                expanded_nodes: edge_count,
                expanded_arcs: arc_count,
                skipped_state_self_arcs,
                left_turn_arcs,
                unclassifiable_turn_arcs,
                estimated_raw_expanded_bytes,
            },
        })
    }

    pub fn source_states(&self, source: u32) -> Result<&[u32], String> {
        let source = source as usize;
        if source + 1 >= self.edges_by_tail_offsets.len() {
            return Err(format!("source node {source} is out of bounds"));
        }
        Ok(incidence_slice(
            &self.edges_by_tail_offsets,
            &self.edges_by_tail,
            source,
        ))
    }

    pub fn target_states(&self, target: u32) -> Result<&[u32], String> {
        let target = target as usize;
        if target + 1 >= self.edges_by_head_offsets.len() {
            return Err(format!("target node {target} is out of bounds"));
        }
        Ok(incidence_slice(
            &self.edges_by_head_offsets,
            &self.edges_by_head,
            target,
        ))
    }

    pub fn customized_arc_weights(
        &self,
        original_weights: &[u32],
        left_penalty: u32,
    ) -> Result<Vec<u32>, String> {
        validate_original_weights(original_weights, self.stats.original_edges)?;
        self.head
            .iter()
            .zip(&self.left_turn)
            .enumerate()
            .map(|(arc, (&next_edge, &is_left))| {
                let base = original_weights[next_edge as usize];
                let weight = base
                    .checked_add(left_penalty.saturating_mul(is_left as u32))
                    .ok_or_else(|| format!("expanded arc {arc} weight overflow"))?;
                if weight == 0 || weight >= CCH_INFINITY {
                    return Err(format!(
                        "expanded arc {arc} has invalid CCH weight {weight}"
                    ));
                }
                Ok(weight)
            })
            .collect()
    }
}

pub fn median_weight(weights: &[u32]) -> Result<f64, String> {
    validate_original_weights(weights, weights.len())?;
    let mut sorted = weights.to_vec();
    sorted.sort_unstable();
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        Ok((sorted[middle - 1] as f64 + sorted[middle] as f64) * 0.5)
    } else {
        Ok(sorted[middle] as f64)
    }
}

pub fn scaled_left_penalty(kappa: f64, r: f64) -> Result<u32, String> {
    if !kappa.is_finite() || kappa <= 0.0 || !r.is_finite() || r < 0.0 {
        return Err(format!("invalid left-turn scale kappa={kappa}, r={r}"));
    }
    let penalty = (kappa * r).round();
    if penalty < 0.0 || penalty >= CCH_INFINITY as f64 {
        return Err(format!(
            "left-turn penalty {penalty} is outside the CCH range"
        ));
    }
    Ok(penalty as u32)
}

pub fn expanded_path_cost(
    graph: &GraphData,
    original_weights: &[u32],
    path: &[usize],
    left_penalty: u32,
) -> Result<u64, String> {
    if original_weights.len() != graph.tail.len() {
        return Err(format!(
            "original weight count {} does not match graph edge count {}",
            original_weights.len(),
            graph.tail.len()
        ));
    }
    let Some(&first) = path.first() else {
        return Err("expanded path cannot be empty".to_string());
    };
    let mut cost = *original_weights
        .get(first)
        .ok_or_else(|| format!("path edge {first} is out of bounds"))? as u64;
    for pair in path.windows(2) {
        if graph.head.get(pair[0]) != graph.tail.get(pair[1]) {
            return Err(format!(
                "discontinuous original-edge transition {} -> {}",
                pair[0], pair[1]
            ));
        }
        let next_cost = *original_weights
            .get(pair[1])
            .ok_or_else(|| format!("path edge {} is out of bounds", pair[1]))?
            as u64;
        let turn_cost = if turn_geometry(graph, pair[0], pair[1])
            .is_some_and(|turn| turn.category == TurnCategory::Left)
        {
            left_penalty as u64
        } else {
            0
        };
        cost = cost
            .checked_add(next_cost + turn_cost)
            .ok_or_else(|| "expanded path cost overflow".to_string())?;
    }
    Ok(cost)
}

#[allow(clippy::too_many_arguments)]
pub fn query_expanded_path<'a>(
    query: &mut CCHQuery<'a>,
    metric: &'a CCHMetric<'a>,
    expanded: &ExpandedTurnGraph,
    graph: &GraphData,
    original_weights: &[u32],
    left_penalty: u32,
    source: u32,
    target: u32,
) -> Result<ExpandedPath, String> {
    if original_weights.len() != expanded.stats.original_edges {
        return Err(format!(
            "original weight count {} does not match expanded state count {}",
            original_weights.len(),
            expanded.stats.original_edges
        ));
    }
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
        query.add_source(state, original_weights[state as usize]);
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
    let reconstructed = expanded_path_cost(graph, original_weights, &original_edges, left_penalty)?;
    if reconstructed != distance as u64 {
        return Err(format!(
            "expanded OD ({source}, {target}) distance/path mismatch: distance={distance}, reconstructed={reconstructed}"
        ));
    }
    // The argument catches accidentally pairing a query with another metric.
    if metric.weights().len() != expanded.stats.expanded_arcs {
        return Err("expanded metric arc count mismatch".to_string());
    }
    Ok(ExpandedPath {
        distance,
        original_edges,
    })
}

pub fn validate_decoded_path(
    graph: &GraphData,
    path: &[usize],
    source: u32,
    target: u32,
) -> Result<(), String> {
    let Some((&first, rest)) = path.split_first() else {
        return Err(format!("decoded OD ({source}, {target}) path is empty"));
    };
    if graph.tail.get(first).copied() != Some(source) {
        return Err(format!(
            "decoded OD ({source}, {target}) starts with edge {first} from {:?}",
            graph.tail.get(first)
        ));
    }
    let mut previous = first;
    for &edge in rest {
        if graph.head.get(previous) != graph.tail.get(edge) {
            return Err(format!(
                "decoded OD ({source}, {target}) is discontinuous at {previous}->{edge}"
            ));
        }
        previous = edge;
    }
    if graph.head.get(previous).copied() != Some(target) {
        return Err(format!(
            "decoded OD ({source}, {target}) ends with edge {previous} at {:?}",
            graph.head.get(previous)
        ));
    }
    Ok(())
}

fn validate_original_weights(weights: &[u32], expected: usize) -> Result<(), String> {
    if weights.len() != expected || weights.is_empty() {
        return Err(format!(
            "original weight count {} does not match expected {expected}",
            weights.len()
        ));
    }
    if let Some((edge, &weight)) = weights
        .iter()
        .enumerate()
        .find(|(_, weight)| **weight == 0 || **weight >= CCH_INFINITY)
    {
        return Err(format!(
            "original edge {edge} has invalid CCH weight {weight}"
        ));
    }
    Ok(())
}

fn build_incidence(
    node_count: usize,
    endpoints: &[u32],
    edge_count: usize,
) -> Result<(Vec<usize>, Vec<u32>), String> {
    let mut offsets = vec![0usize; node_count + 1];
    for &node in endpoints {
        offsets[node as usize + 1] = offsets[node as usize + 1]
            .checked_add(1)
            .ok_or_else(|| "incidence degree overflow".to_string())?;
    }
    for node in 1..offsets.len() {
        offsets[node] = offsets[node]
            .checked_add(offsets[node - 1])
            .ok_or_else(|| "incidence offset overflow".to_string())?;
    }
    let mut cursor = offsets[..node_count].to_vec();
    let mut edges = vec![0u32; edge_count];
    for (edge, &node) in endpoints.iter().enumerate() {
        let position = &mut cursor[node as usize];
        edges[*position] = edge as u32;
        *position += 1;
    }
    Ok((offsets, edges))
}

fn incidence_slice<'a>(offsets: &[usize], edges: &'a [u32], node: usize) -> &'a [u32] {
    &edges[offsets[node]..offsets[node + 1]]
}

fn estimate_raw_bytes(arcs: usize, states: usize) -> Result<u64, String> {
    // Per arc: tail/head (8), turn flag (1), one customized weight (4).
    // Per state: x/y coordinates (8) and one order entry (4).
    let arc_bytes = (arcs as u64)
        .checked_mul(13)
        .ok_or_else(|| "expanded arc byte estimate overflow".to_string())?;
    let state_bytes = (states as u64)
        .checked_mul(12)
        .ok_or_else(|| "expanded state byte estimate overflow".to_string())?;
    arc_bytes
        .checked_add(state_bytes)
        .ok_or_else(|| "expanded raw byte estimate overflow".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use routingkit_cch::{CCH, CCHMetric, compute_order_degree};

    fn graph() -> GraphData {
        // 0->1->2 is cheaper without a turn penalty, while 0->3->2 wins
        // after penalizing the left turn at node 1. Edge 4 is an original
        // self-loop and must not induce expanded state 4 -> state 4.
        GraphData {
            tail: vec![0, 1, 0, 3, 1],
            head: vec![1, 2, 3, 2, 1],
            baseline_weights: vec![2, 2, 2, 3, 7],
            x: vec![0.0, 1.0, 1.0, 0.0],
            y: vec![0.0, 0.0, 1.0, 1.0],
        }
    }

    #[test]
    fn builds_line_graph_and_skips_state_self_arc() {
        let graph = graph();
        let expanded = ExpandedTurnGraph::build(&graph).unwrap();
        assert_eq!(expanded.stats.expanded_nodes, graph.tail.len());
        assert_eq!(expanded.stats.skipped_state_self_arcs, 1);
        assert!(
            expanded
                .tail
                .iter()
                .zip(&expanded.head)
                .all(|(tail, head)| tail != head)
        );
        assert_eq!(expanded.source_states(0).unwrap(), &[0, 2]);
        assert_eq!(expanded.target_states(2).unwrap(), &[1, 3]);
    }

    #[test]
    fn multi_source_target_query_applies_left_penalty() {
        let graph = graph();
        let expanded = ExpandedTurnGraph::build(&graph).unwrap();
        let order = compute_order_degree(
            expanded.stats.expanded_nodes as u32,
            &expanded.tail,
            &expanded.head,
        );
        let cch = CCH::new(&order, &expanded.tail, &expanded.head, |_| {}, false);

        let zero_weights = expanded
            .customized_arc_weights(&graph.baseline_weights, 0)
            .unwrap();
        let zero_metric = CCHMetric::new(&cch, zero_weights);
        let mut zero_query = CCHQuery::new(&zero_metric);
        let zero = query_expanded_path(
            &mut zero_query,
            &zero_metric,
            &expanded,
            &graph,
            &graph.baseline_weights,
            0,
            0,
            2,
        )
        .unwrap();
        assert_eq!(zero.distance, 4);
        assert_eq!(zero.original_edges, vec![0, 1]);

        let direct = query_expanded_path(
            &mut zero_query,
            &zero_metric,
            &expanded,
            &graph,
            &graph.baseline_weights,
            0,
            0,
            1,
        )
        .unwrap();
        assert_eq!(direct.distance, 2);
        assert_eq!(direct.original_edges, vec![0]);

        let penalty_weights = expanded
            .customized_arc_weights(&graph.baseline_weights, 10)
            .unwrap();
        let penalty_metric = CCHMetric::new(&cch, penalty_weights);
        let mut penalty_query = CCHQuery::new(&penalty_metric);
        let penalized = query_expanded_path(
            &mut penalty_query,
            &penalty_metric,
            &expanded,
            &graph,
            &graph.baseline_weights,
            10,
            0,
            2,
        )
        .unwrap();
        assert_eq!(penalized.distance, 5);
        assert_eq!(penalized.original_edges, vec![2, 3]);
    }
}
