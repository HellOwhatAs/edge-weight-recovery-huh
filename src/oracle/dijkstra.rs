use crate::oracle::CCH_INFINITY;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

#[derive(Clone, Debug, PartialEq)]
pub struct DijkstraPath {
    pub distance: f64,
    pub node_path: Vec<usize>,
    pub arc_path: Vec<usize>,
}

/// Representation-neutral adjacency index for production-style integer
/// Dijkstra queries. Arc IDs retain the routing graph's original ordering so
/// decoded coordinates have exactly the same meaning as CCH arc IDs.
pub(crate) struct DijkstraTopology {
    node_count: usize,
    arc_count: usize,
    outgoing_offsets: Vec<usize>,
    outgoing_arcs: Vec<usize>,
    head: Vec<u32>,
}

/// One integer metric on a [`DijkstraTopology`].
pub(crate) struct DijkstraMetric<'a> {
    topology: &'a DijkstraTopology,
    weights: Vec<u32>,
}

/// Reusable queue, distance, and predecessor storage for repeated queries.
pub(crate) struct DijkstraReusableQuery<'metric, 'topology> {
    metric: &'metric DijkstraMetric<'topology>,
    distance: Vec<u64>,
    predecessor: Vec<Option<(usize, usize)>>,
    target_offset: Vec<u64>,
    touched_nodes: Vec<usize>,
    touched_targets: Vec<usize>,
    queue: BinaryHeap<Reverse<(u64, u32)>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DijkstraU32Path {
    pub distance: u32,
    pub node_path: Vec<usize>,
    pub arc_path: Vec<usize>,
}

impl DijkstraTopology {
    pub(crate) fn build(node_count: usize, tail: &[u32], head: &[u32]) -> Result<Self, String> {
        if node_count == 0 || node_count > u32::MAX as usize {
            return Err(format!(
                "Dijkstra node count {node_count} must be in 1..={}",
                u32::MAX
            ));
        }
        if tail.len() != head.len() {
            return Err(format!(
                "Dijkstra arc arrays must have equal length: tail={}, head={}",
                tail.len(),
                head.len()
            ));
        }

        let mut outgoing_offsets = vec![0usize; node_count + 1];
        for (arc, (&tail_node, &head_node)) in tail.iter().zip(head).enumerate() {
            if tail_node as usize >= node_count || head_node as usize >= node_count {
                return Err(format!(
                    "Dijkstra arc {arc} endpoint out of bounds for {node_count} nodes: {tail_node}->{head_node}"
                ));
            }
            outgoing_offsets[tail_node as usize + 1] = outgoing_offsets[tail_node as usize + 1]
                .checked_add(1)
                .ok_or_else(|| "Dijkstra out-degree overflow".to_string())?;
        }
        for node in 1..outgoing_offsets.len() {
            outgoing_offsets[node] = outgoing_offsets[node]
                .checked_add(outgoing_offsets[node - 1])
                .ok_or_else(|| "Dijkstra adjacency offset overflow".to_string())?;
        }
        let mut cursor = outgoing_offsets[..node_count].to_vec();
        let mut outgoing_arcs = vec![0usize; tail.len()];
        for (arc, &tail_node) in tail.iter().enumerate() {
            let position = &mut cursor[tail_node as usize];
            outgoing_arcs[*position] = arc;
            *position += 1;
        }

        Ok(Self {
            node_count,
            arc_count: tail.len(),
            outgoing_offsets,
            outgoing_arcs,
            head: head.to_vec(),
        })
    }

    pub(crate) fn customize(&self, weights: &[u32]) -> Result<DijkstraMetric<'_>, String> {
        if weights.len() != self.arc_count {
            return Err(format!(
                "Dijkstra metric has {} arc weights but topology has {} arcs",
                weights.len(),
                self.arc_count
            ));
        }
        if let Some((arc, weight)) = weights
            .iter()
            .copied()
            .enumerate()
            .find(|(_, weight)| *weight == 0 || *weight >= CCH_INFINITY)
        {
            return Err(format!("Dijkstra arc {arc} has invalid weight {weight}"));
        }
        Ok(DijkstraMetric {
            topology: self,
            weights: weights.to_vec(),
        })
    }
}

impl<'topology> DijkstraMetric<'topology> {
    pub(crate) fn new_query(&self) -> DijkstraReusableQuery<'_, 'topology> {
        DijkstraReusableQuery {
            metric: self,
            distance: vec![u64::MAX; self.topology.node_count],
            predecessor: vec![None; self.topology.node_count],
            target_offset: vec![u64::MAX; self.topology.node_count],
            touched_nodes: Vec::new(),
            touched_targets: Vec::new(),
            queue: BinaryHeap::new(),
        }
    }
}

impl DijkstraReusableQuery<'_, '_> {
    pub(crate) fn shortest_path(
        &mut self,
        sources: &[(u32, u32)],
        targets: &[(u32, u32)],
    ) -> Result<DijkstraU32Path, String> {
        validate_integer_endpoints(sources, self.metric.topology.node_count, "source")?;
        validate_integer_endpoints(targets, self.metric.topology.node_count, "target")?;

        while let Some(node) = self.touched_nodes.pop() {
            self.distance[node] = u64::MAX;
            self.predecessor[node] = None;
        }
        while let Some(node) = self.touched_targets.pop() {
            self.target_offset[node] = u64::MAX;
        }
        self.queue.clear();

        for &(target, offset) in targets {
            let node = target as usize;
            if self.target_offset[node] == u64::MAX {
                self.touched_targets.push(node);
            }
            self.target_offset[node] = self.target_offset[node].min(offset as u64);
        }
        for &(source, offset) in sources {
            let node = source as usize;
            if u64::from(offset) < self.distance[node] {
                if self.distance[node] == u64::MAX {
                    self.touched_nodes.push(node);
                }
                self.distance[node] = u64::from(offset);
                self.predecessor[node] = None;
                self.queue.push(Reverse((u64::from(offset), source)));
            }
        }

        let mut selected_target = None::<(usize, u64)>;
        while let Some(Reverse((distance, node_u32))) = self.queue.pop() {
            let node = node_u32 as usize;
            if distance != self.distance[node] {
                continue;
            }
            if selected_target.is_some_and(|(_, best)| distance > best) {
                break;
            }

            let offset = self.target_offset[node];
            if offset != u64::MAX {
                let total = distance
                    .checked_add(offset)
                    .ok_or_else(|| "Dijkstra target distance overflow".to_string())?;
                if selected_target.is_none_or(|(best_node, best)| {
                    total < best || (total == best && node < best_node)
                }) {
                    selected_target = Some((node, total));
                }
            }

            let start = self.metric.topology.outgoing_offsets[node];
            let end = self.metric.topology.outgoing_offsets[node + 1];
            for &arc in &self.metric.topology.outgoing_arcs[start..end] {
                let next = self.metric.topology.head[arc] as usize;
                let candidate = distance
                    .checked_add(u64::from(self.metric.weights[arc]))
                    .ok_or_else(|| "Dijkstra path distance overflow".to_string())?;
                let improve_tie = candidate == self.distance[next]
                    && self.predecessor[next].is_some_and(|(_, previous_arc)| arc < previous_arc);
                if candidate < self.distance[next] || improve_tie {
                    if self.distance[next] == u64::MAX {
                        self.touched_nodes.push(next);
                    }
                    self.distance[next] = candidate;
                    self.predecessor[next] = Some((node, arc));
                    self.queue.push(Reverse((candidate, next as u32)));
                }
            }
        }

        let Some((target, total_distance)) = selected_target else {
            return Err("Dijkstra query is unreachable".to_string());
        };
        if total_distance >= u64::from(CCH_INFINITY) {
            return Err(format!(
                "Dijkstra distance {total_distance} reaches the shared infinity sentinel"
            ));
        }

        let mut node_path = vec![target];
        let mut arc_path = Vec::new();
        let mut node = target;
        while let Some((previous, arc)) = self.predecessor[node] {
            arc_path.push(arc);
            node = previous;
            node_path.push(node);
        }
        node_path.reverse();
        arc_path.reverse();

        Ok(DijkstraU32Path {
            distance: total_distance as u32,
            node_path,
            arc_path,
        })
    }
}

fn validate_integer_endpoints(
    endpoints: &[(u32, u32)],
    node_count: usize,
    kind: &str,
) -> Result<(), String> {
    if endpoints.is_empty() {
        return Err(format!("Dijkstra query has no {kind} states"));
    }
    for &(node, offset) in endpoints {
        if node as usize >= node_count {
            return Err(format!(
                "Dijkstra {kind} node {node} is out of bounds for {node_count} nodes"
            ));
        }
        if offset >= CCH_INFINITY {
            return Err(format!(
                "Dijkstra {kind} offset {offset} reaches the infinity sentinel"
            ));
        }
    }
    Ok(())
}

/// Exact dense Dijkstra with arbitrary nonnegative source and target offsets.
///
/// This intentionally uses an O(|V||E|) implementation. Production routing
/// belongs to CCH; this routine stays simple enough to audit and is used as a
/// correctness reference for any representation whose learned coordinates are
/// routed as ordinary arc weights.
pub fn shortest_path_multi_source_f64(
    node_count: usize,
    tail: &[u32],
    head: &[u32],
    weights: &[f64],
    sources: &[(u32, f64)],
    targets: &[(u32, f64)],
) -> Result<Option<DijkstraPath>, String> {
    validate_inputs(node_count, tail, head, weights, sources, targets)?;

    let mut distance = vec![f64::INFINITY; node_count];
    let mut settled = vec![false; node_count];
    let mut predecessor = vec![None::<(usize, usize)>; node_count];
    for &(source, offset) in sources {
        let source = source as usize;
        if offset < distance[source] {
            distance[source] = offset;
            predecessor[source] = None;
        }
    }

    for _ in 0..node_count {
        let next = (0..node_count)
            .filter(|&node| !settled[node])
            .min_by(|&left, &right| {
                distance[left]
                    .total_cmp(&distance[right])
                    .then_with(|| left.cmp(&right))
            });
        let Some(node) = next else {
            break;
        };
        if !distance[node].is_finite() {
            break;
        }
        settled[node] = true;

        for arc in 0..tail.len() {
            if tail[arc] as usize != node {
                continue;
            }
            let next_node = head[arc] as usize;
            if settled[next_node] {
                continue;
            }
            let candidate = distance[node] + weights[arc];
            if candidate < distance[next_node] {
                distance[next_node] = candidate;
                predecessor[next_node] = Some((node, arc));
            }
        }
    }

    let selected_target = targets
        .iter()
        .copied()
        .filter_map(|(node, offset)| {
            let total = distance[node as usize] + offset;
            total.is_finite().then_some((node as usize, total))
        })
        .min_by(|(left_node, left_cost), (right_node, right_cost)| {
            left_cost
                .total_cmp(right_cost)
                .then_with(|| left_node.cmp(right_node))
        });
    let Some((target, total_distance)) = selected_target else {
        return Ok(None);
    };

    let mut node_path = vec![target];
    let mut arc_path = Vec::new();
    let mut node = target;
    while let Some((previous, arc)) = predecessor[node] {
        arc_path.push(arc);
        node = previous;
        node_path.push(node);
    }
    node_path.reverse();
    arc_path.reverse();

    Ok(Some(DijkstraPath {
        distance: total_distance,
        node_path,
        arc_path,
    }))
}

/// Exact dense single-source/single-target Dijkstra.
pub fn shortest_path_f64(
    node_count: usize,
    tail: &[u32],
    head: &[u32],
    weights: &[f64],
    source: u32,
    target: u32,
) -> Result<Option<DijkstraPath>, String> {
    shortest_path_multi_source_f64(
        node_count,
        tail,
        head,
        weights,
        &[(source, 0.0)],
        &[(target, 0.0)],
    )
}

fn validate_inputs(
    node_count: usize,
    tail: &[u32],
    head: &[u32],
    weights: &[f64],
    sources: &[(u32, f64)],
    targets: &[(u32, f64)],
) -> Result<(), String> {
    if node_count == 0 {
        return Err("Dijkstra requires at least one node".to_string());
    }
    if tail.len() != head.len() || tail.len() != weights.len() {
        return Err(format!(
            "Dijkstra edge-array length mismatch: tail={}, head={}, weights={}",
            tail.len(),
            head.len(),
            weights.len()
        ));
    }
    if sources.is_empty() || targets.is_empty() {
        return Err("Dijkstra requires at least one source and one target".to_string());
    }
    for (arc, ((&tail_node, &head_node), &weight)) in tail.iter().zip(head).zip(weights).enumerate()
    {
        if tail_node as usize >= node_count || head_node as usize >= node_count {
            return Err(format!(
                "Dijkstra arc {arc} endpoint out of bounds: {tail_node}->{head_node}"
            ));
        }
        if !weight.is_finite() || weight < 0.0 {
            return Err(format!(
                "Dijkstra arc {arc} has invalid nonnegative f64 weight {weight}"
            ));
        }
    }
    for (kind, endpoints) in [("source", sources), ("target", targets)] {
        for &(node, offset) in endpoints {
            if node as usize >= node_count {
                return Err(format!(
                    "Dijkstra {kind} node {node} is out of bounds for {node_count} nodes"
                ));
            }
            if !offset.is_finite() || offset < 0.0 {
                return Err(format!(
                    "Dijkstra {kind} node {node} has invalid nonnegative offset {offset}"
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reusable_integer_query_preserves_arc_ids_and_endpoint_offsets() {
        let topology = DijkstraTopology::build(4, &[0, 1, 0, 2], &[1, 3, 2, 3]).unwrap();
        let metric = topology.customize(&[5, 5, 2, 2]).unwrap();
        let mut query = metric.new_query();
        let first = query.shortest_path(&[(0, 0)], &[(3, 0)]).unwrap();
        assert_eq!(first.distance, 4);
        assert_eq!(first.node_path, vec![0, 2, 3]);
        assert_eq!(first.arc_path, vec![2, 3]);

        let second = query
            .shortest_path(&[(0, 10), (2, 3)], &[(1, 8), (3, 0)])
            .unwrap();
        assert_eq!(second.distance, 5);
        assert_eq!(second.node_path, vec![2, 3]);
        assert_eq!(second.arc_path, vec![3]);
    }

    #[test]
    fn integer_query_rejects_unreachable_and_invalid_metrics() {
        let topology = DijkstraTopology::build(3, &[0], &[1]).unwrap();
        assert!(topology.customize(&[0]).is_err());
        let metric = topology.customize(&[1]).unwrap();
        assert!(
            metric
                .new_query()
                .shortest_path(&[(0, 0)], &[(2, 0)])
                .is_err()
        );
    }

    #[test]
    fn finds_the_exact_f64_shortest_path() {
        let result =
            shortest_path_f64(4, &[0, 1, 0, 2], &[1, 3, 2, 3], &[5.0, 5.0, 2.0, 2.0], 0, 3)
                .unwrap()
                .unwrap();
        assert_eq!(result.distance, 4.0);
        assert_eq!(result.node_path, vec![0, 2, 3]);
        assert_eq!(result.arc_path, vec![2, 3]);
    }

    #[test]
    fn supports_multiple_sources_and_target_offsets() {
        let result = shortest_path_multi_source_f64(
            4,
            &[0, 1, 2],
            &[1, 3, 3],
            &[2.0, 2.0, 1.0],
            &[(0, 10.0), (2, 3.0)],
            &[(1, 8.0), (3, 0.0)],
        )
        .unwrap()
        .unwrap();
        assert_eq!(result.distance, 4.0);
        assert_eq!(result.node_path, vec![2, 3]);
        assert_eq!(result.arc_path, vec![2]);
    }

    #[test]
    fn reports_unreachable_and_rejects_invalid_weights() {
        assert_eq!(
            shortest_path_f64(3, &[0], &[1], &[1.0], 0, 2).unwrap(),
            None
        );
        assert!(shortest_path_f64(2, &[0], &[1], &[f64::NAN], 0, 1).is_err());
        assert!(shortest_path_f64(2, &[0], &[1], &[-1.0], 0, 1).is_err());
        assert!(shortest_path_multi_source_f64(2, &[0], &[1], &[1.0], &[], &[(1, 0.0)]).is_err());
    }
}
