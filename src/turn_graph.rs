//! Generic edge-state expansion for transition-aware shortest paths.
//!
//! Each original directed edge is one expanded state. Every topologically
//! legal adjacent edge pair `(previous_edge, next_edge)` is one expanded arc.
//! Transition IDs are stable: arcs are ordered lexicographically by original
//! edge ID, first by `previous_edge` and then by `next_edge`.

use crate::data::GraphData;

pub const MAX_EXPANDED_ARCS: usize = 12_000_000;
pub const MAX_RAW_EXPANDED_BYTES: u64 = 512 * 1024 * 1024;
pub const CCH_INFINITY: u32 = i32::MAX as u32;

/// Stable index of a legal `(previous_edge, next_edge)` transition.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TransitionId(usize);

impl TransitionId {
    pub const fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpandedGraphStats {
    pub original_nodes: usize,
    pub original_edges: usize,
    pub expanded_nodes: usize,
    pub expanded_arcs: usize,
    /// Legal transitions whose previous and next state are the same original
    /// edge. These are retained, not silently filtered.
    pub state_self_transitions: usize,
    /// Tail/head, one customized metric, state coordinates, and one order.
    pub estimated_raw_expanded_bytes: u64,
}

/// One expanded state per original directed edge.
///
/// Expanded arc `TransitionId(i)` is `tail[i] -> head[i]`, where both values
/// are original edge IDs. The public arc arrays can be passed directly to CCH.
#[derive(Debug)]
pub struct ExpandedTurnGraph {
    pub tail: Vec<u32>,
    pub head: Vec<u32>,
    pub state_x: Vec<f32>,
    pub state_y: Vec<f32>,
    transition_offsets: Vec<usize>,
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
        for previous_edge in 0..edge_count {
            let junction = graph.head[previous_edge] as usize;
            arc_count = arc_count
                .checked_add(
                    incidence_slice(&edges_by_tail_offsets, &edges_by_tail, junction).len(),
                )
                .ok_or_else(|| "expanded arc count overflow".to_string())?;
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
        let mut transition_offsets = Vec::with_capacity(edge_count + 1);
        transition_offsets.push(0);
        for previous_edge in 0..edge_count {
            let junction = graph.head[previous_edge] as usize;
            for &next_edge in incidence_slice(&edges_by_tail_offsets, &edges_by_tail, junction) {
                tail.push(previous_edge as u32);
                head.push(next_edge);
            }
            transition_offsets.push(tail.len());
        }
        debug_assert_eq!(tail.len(), arc_count);
        debug_assert!(
            transition_offsets
                .windows(2)
                .all(|range| head[range[0]..range[1]].is_sorted())
        );
        let state_self_transitions = tail
            .iter()
            .zip(&head)
            .filter(|(previous, next)| previous == next)
            .count();

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
            state_x,
            state_y,
            transition_offsets,
            edges_by_tail_offsets,
            edges_by_tail,
            edges_by_head_offsets,
            edges_by_head,
            stats: ExpandedGraphStats {
                original_nodes: node_count,
                original_edges: edge_count,
                expanded_nodes: edge_count,
                expanded_arcs: arc_count,
                state_self_transitions,
                estimated_raw_expanded_bytes,
            },
        })
    }

    pub fn transition_count(&self) -> usize {
        self.stats.expanded_arcs
    }

    /// Return the stable ID of a legal transition, or `None` if the pair is
    /// out of bounds or topologically discontinuous.
    pub fn transition_id(&self, previous_edge: usize, next_edge: usize) -> Option<TransitionId> {
        let range = self.transition_range(previous_edge)?;
        let relative = self.head[range.clone()]
            .binary_search(&(u32::try_from(next_edge).ok()?))
            .ok()?;
        Some(TransitionId(range.start + relative))
    }

    /// Map a stable transition ID back to its original adjacent edge pair.
    pub fn transition_edges(&self, transition: TransitionId) -> Option<(usize, usize)> {
        let index = transition.index();
        Some((
            *self.tail.get(index)? as usize,
            *self.head.get(index)? as usize,
        ))
    }

    pub fn transitions(&self) -> impl ExactSizeIterator<Item = (TransitionId, usize, usize)> + '_ {
        self.tail
            .iter()
            .zip(&self.head)
            .enumerate()
            .map(|(index, (&previous, &next))| {
                (TransitionId(index), previous as usize, next as usize)
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

    /// Construct the integer expanded metric
    /// `edge_weights[next_edge] + round(residual_scale * residual[id])`.
    ///
    /// Residuals are continuous, finite, and nonnegative. The returned metric
    /// remains strictly below the CCH infinity sentinel.
    pub fn transition_metric_weights(
        &self,
        edge_weights: &[u32],
        transition_residuals: &[f64],
        residual_scale: f64,
    ) -> Result<Vec<u32>, String> {
        validate_edge_weights(edge_weights, self.stats.original_edges)?;
        if transition_residuals.len() != self.transition_count() {
            return Err(format!(
                "transition residual count {} does not match transition count {}",
                transition_residuals.len(),
                self.transition_count()
            ));
        }
        if !residual_scale.is_finite() || residual_scale < 0.0 {
            return Err(format!(
                "transition residual scale must be finite and nonnegative, got {residual_scale}"
            ));
        }

        self.head
            .iter()
            .zip(transition_residuals)
            .enumerate()
            .map(|(transition, (&next_edge, &residual))| {
                if !residual.is_finite() || residual < 0.0 {
                    return Err(format!(
                        "transition {transition} has invalid nonnegative residual {residual}"
                    ));
                }
                let scaled = residual_scale * residual;
                if !scaled.is_finite() {
                    return Err(format!(
                        "transition {transition} residual cannot be represented after scaling"
                    ));
                }
                let quantized_residual = scaled.round();
                if quantized_residual < 0.0 || quantized_residual >= CCH_INFINITY as f64 {
                    return Err(format!(
                        "transition {transition} quantized residual {quantized_residual} is outside the CCH range"
                    ));
                }
                let weight = edge_weights[next_edge as usize]
                    .checked_add(quantized_residual as u32)
                    .ok_or_else(|| format!("transition {transition} metric weight overflow"))?;
                if weight == 0 || weight >= CCH_INFINITY {
                    return Err(format!(
                        "transition {transition} has invalid CCH weight {weight}"
                    ));
                }
                Ok(weight)
            })
            .collect()
    }

    /// Cost an observed original-edge path under an already quantized expanded
    /// metric. The first edge is paid as a source-state offset; every remaining
    /// edge and its turn residual are paid by the corresponding transition.
    pub(crate) fn observed_path_cost(
        &self,
        graph: &GraphData,
        edge_weights: &[u32],
        transition_weights: &[u32],
        path: &[usize],
    ) -> Result<u64, String> {
        validate_edge_weights(edge_weights, self.stats.original_edges)?;
        validate_transition_weights(transition_weights, self.transition_count())?;
        let Some(&first) = path.first() else {
            return Err("expanded path cannot be empty".to_string());
        };
        let mut cost = *edge_weights
            .get(first)
            .ok_or_else(|| format!("path edge {first} is out of bounds"))?
            as u64;

        for pair in path.windows(2) {
            let previous = pair[0];
            let next = pair[1];
            if graph.head.get(previous) != graph.tail.get(next) {
                return Err(format!(
                    "discontinuous original-edge transition {previous} -> {next}"
                ));
            }
            let transition = self
                .transition_id(previous, next)
                .ok_or_else(|| format!("missing expanded transition {previous} -> {next}"))?;
            let transition_weight = transition_weights[transition.index()];
            if transition_weight < edge_weights[next] {
                return Err(format!(
                    "transition {} weight {transition_weight} is below next-edge weight {}",
                    transition.index(),
                    edge_weights[next]
                ));
            }
            cost = cost
                .checked_add(transition_weight as u64)
                .ok_or_else(|| "expanded path cost overflow".to_string())?;
        }
        Ok(cost)
    }

    fn transition_range(&self, previous_edge: usize) -> Option<std::ops::Range<usize>> {
        Some(
            *self.transition_offsets.get(previous_edge)?
                ..*self.transition_offsets.get(previous_edge + 1)?,
        )
    }
}

pub(crate) fn validate_decoded_path(
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

fn validate_edge_weights(weights: &[u32], expected: usize) -> Result<(), String> {
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

fn validate_transition_weights(weights: &[u32], expected: usize) -> Result<(), String> {
    if weights.len() != expected {
        return Err(format!(
            "transition weight count {} does not match expected {expected}",
            weights.len()
        ));
    }
    if let Some((transition, &weight)) = weights
        .iter()
        .enumerate()
        .find(|(_, weight)| **weight == 0 || **weight >= CCH_INFINITY)
    {
        return Err(format!(
            "transition {transition} has invalid CCH weight {weight}"
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
    // Per transition: tail/head (8) and one customized weight (4).
    // Per state: x/y coordinates (8) and one order entry (4).
    let arc_bytes = (arcs as u64)
        .checked_mul(12)
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
    use crate::oracle::ExpandedCchOracle;
    use routingkit_cch::{CCH, CCHMetric, CCHQuery, compute_order_degree};

    fn graph() -> GraphData {
        // Two 0->2 alternatives plus a longer connector 1->3.
        GraphData {
            tail: vec![0, 1, 0, 3, 1],
            head: vec![1, 2, 3, 2, 3],
            baseline_weights: vec![2, 2, 2, 3, 10],
            x: vec![0.0, 1.0, 1.0, 0.0],
            y: vec![0.0, 0.0, 1.0, 1.0],
        }
    }

    #[test]
    fn transition_ids_are_stable_and_reversible() {
        let graph = graph();
        let expanded = ExpandedTurnGraph::build(&graph).unwrap();

        let transitions = expanded
            .transitions()
            .map(|(id, previous, next)| (id.index(), previous, next))
            .collect::<Vec<_>>();
        assert_eq!(
            transitions,
            vec![(0, 0, 1), (1, 0, 4), (2, 2, 3), (3, 4, 3)]
        );
        for &(index, previous, next) in &transitions {
            let id = expanded.transition_id(previous, next).unwrap();
            assert_eq!(id.index(), index);
            assert_eq!(expanded.transition_edges(id), Some((previous, next)));
        }
        assert_eq!(expanded.transition_id(0, 3), None);
        assert_eq!(expanded.transition_id(99, 1), None);
        assert_eq!(expanded.source_states(0).unwrap(), &[0, 2]);
        assert_eq!(expanded.target_states(2).unwrap(), &[1, 3]);
    }

    #[test]
    fn retains_and_reports_legal_state_self_transitions() {
        let graph = GraphData {
            tail: vec![0],
            head: vec![0],
            baseline_weights: vec![2],
            x: vec![0.0],
            y: vec![0.0],
        };
        let expanded = ExpandedTurnGraph::build(&graph).unwrap();
        assert_eq!(expanded.transition_count(), 1);
        assert_eq!(expanded.stats.state_self_transitions, 1);
        assert_eq!(
            expanded.transition_edges(expanded.transition_id(0, 0).unwrap()),
            Some((0, 0))
        );
    }

    #[test]
    fn generic_metric_preserves_continuity_od_and_observed_cost() {
        let graph = graph();
        let expanded = ExpandedTurnGraph::build(&graph).unwrap();
        let mut residuals = vec![0.0; expanded.transition_count()];
        residuals[expanded.transition_id(0, 1).unwrap().index()] = 1.5;
        let weights = expanded
            .transition_metric_weights(&graph.baseline_weights, &residuals, 2.0)
            .unwrap();

        assert_eq!(weights[expanded.transition_id(0, 1).unwrap().index()], 5);
        assert_eq!(
            expanded
                .observed_path_cost(&graph, &graph.baseline_weights, &weights, &[0, 1])
                .unwrap(),
            7
        );
        validate_decoded_path(&graph, &[0, 1], 0, 2).unwrap();
        assert!(validate_decoded_path(&graph, &[1], 0, 2).is_err());
        assert!(
            expanded
                .observed_path_cost(&graph, &graph.baseline_weights, &weights, &[0, 3])
                .is_err()
        );
    }

    #[test]
    fn zero_residual_expansion_matches_original_multi_source_target_query() {
        let graph = graph();
        let original_order = compute_order_degree(graph.x.len() as u32, &graph.tail, &graph.head);
        let original_cch = CCH::new(&original_order, &graph.tail, &graph.head, |_| {}, false);
        let original_metric = CCHMetric::new(&original_cch, graph.baseline_weights.clone());
        let mut original_query = CCHQuery::new(&original_metric);
        original_query.add_source(0, 0);
        original_query.add_target(2, 0);
        let original_result = original_query.run();
        let original_distance = original_result.distance().unwrap();
        drop(original_result);

        let expanded = ExpandedTurnGraph::build(&graph).unwrap();
        assert_eq!(expanded.source_states(0).unwrap(), &[0, 2]);
        assert_eq!(expanded.target_states(2).unwrap(), &[1, 3]);
        let zero_residuals = vec![0.0; expanded.transition_count()];
        let expanded_weights = expanded
            .transition_metric_weights(&graph.baseline_weights, &zero_residuals, 1.0)
            .unwrap();
        let expanded_oracle = ExpandedCchOracle::build(&graph, &expanded).unwrap();
        let expanded_metric = expanded_oracle
            .customize(&graph.baseline_weights, &expanded_weights)
            .unwrap();
        let result = expanded_metric.query(0, 2).unwrap();

        assert_eq!(original_distance, 4);
        assert_eq!(result.distance, original_distance);
        assert_eq!(result.original_edges, vec![0, 1]);
        assert_eq!(
            expanded
                .observed_path_cost(&graph, &graph.baseline_weights, &expanded_weights, &[0, 1],)
                .unwrap(),
            graph.baseline_weights[0] as u64 + graph.baseline_weights[1] as u64
        );
    }
}
