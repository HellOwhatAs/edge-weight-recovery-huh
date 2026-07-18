//! Auditable research-only Dijkstra routing oracle.
//!
//! This crate deliberately lives outside the production workspace. It is a
//! baseline adapter for the stable [`ewr_core::RoutingOracle`] boundary, not a
//! second training implementation.

use ewr_core::{
    EdgeId, OracleError, OraclePath, OracleQuery, QueryEndpoint, ROUTING_INFINITY, RoutingOracle,
    RoutingTopology,
};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Checkpoint identity of the research Dijkstra routing semantics.
pub const DIJKSTRA_ORACLE_IDENTITY_V1: &str =
    "ewr-research-dijkstra:stable-binary-heap:u32-metric:v1";

/// A Dijkstra oracle with a lazily cached adjacency list.
///
/// The cache contains topology only. Every call consumes the exact integer
/// metric supplied by core, so no learned weight can leak between updates.
#[derive(Debug, Default)]
pub struct DijkstraOracle {
    cached: Option<CachedAdjacency>,
}

#[derive(Debug)]
struct CachedAdjacency {
    topology_fingerprint: u64,
    outgoing_coordinates: Vec<Vec<usize>>,
}

impl DijkstraOracle {
    /// Construct an oracle without preprocessed topology.
    pub const fn new() -> Self {
        Self { cached: None }
    }

    /// Run one metric and ordered query batch while preserving typed errors.
    ///
    /// This is the research-facing equivalent of [`RoutingOracle::shortest_paths`].
    /// The trait implementation converts the same errors to core's opaque
    /// backend error at the production boundary.
    pub fn shortest_paths_checked(
        &mut self,
        topology: &RoutingTopology,
        quantized_weights: &[u32],
        queries: &[OracleQuery],
    ) -> Result<Vec<OraclePath>, DijkstraError> {
        validate_topology(topology)?;
        validate_metric(quantized_weights, topology.arc_count())?;
        for (query_index, query) in queries.iter().enumerate() {
            validate_query(query, topology.node_count(), query_index)?;
        }

        let adjacency = self.adjacency_for(topology);
        queries
            .iter()
            .enumerate()
            .map(|(query_index, query)| {
                run_query(
                    topology,
                    quantized_weights,
                    &adjacency.outgoing_coordinates,
                    query,
                    query_index,
                )
            })
            .collect()
    }

    fn adjacency_for(&mut self, topology: &RoutingTopology) -> &CachedAdjacency {
        let topology_fingerprint = topology.fingerprint();
        let must_rebuild = self.cached.as_ref().is_none_or(|cached| {
            cached.topology_fingerprint != topology_fingerprint
                || cached.outgoing_coordinates.len() != topology.node_count()
        });
        if must_rebuild {
            let mut outgoing_coordinates = vec![Vec::new(); topology.node_count()];
            for (coordinate, &tail) in topology.tails().iter().enumerate() {
                outgoing_coordinates[tail.index()].push(coordinate);
            }
            self.cached = Some(CachedAdjacency {
                topology_fingerprint,
                outgoing_coordinates,
            });
        }
        self.cached
            .as_ref()
            .expect("a missing or stale adjacency cache was rebuilt above")
    }
}

impl RoutingOracle for DijkstraOracle {
    fn identity(&self) -> &'static str {
        DIJKSTRA_ORACLE_IDENTITY_V1
    }

    fn shortest_paths(
        &mut self,
        topology: &RoutingTopology,
        quantized_weights: &[u32],
        queries: &[OracleQuery],
    ) -> Result<Vec<OraclePath>, OracleError> {
        self.shortest_paths_checked(topology, quantized_weights, queries)
            .map_err(|error| OracleError::new(error.to_string()))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct QueueState {
    distance: u64,
    node: EdgeId,
}

impl Ord for QueueState {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap. Reversing both fields pops the smallest
        // (distance, node ID), making equal-cost exploration deterministic.
        other
            .distance
            .cmp(&self.distance)
            .then_with(|| other.node.cmp(&self.node))
    }
}

impl PartialOrd for QueueState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Predecessor {
    node: EdgeId,
    coordinate: usize,
}

fn run_query(
    topology: &RoutingTopology,
    quantized_weights: &[u32],
    outgoing_coordinates: &[Vec<usize>],
    query: &OracleQuery,
    query_index: usize,
) -> Result<OraclePath, DijkstraError> {
    let mut distances = vec![u64::MAX; topology.node_count()];
    let mut predecessors = vec![None; topology.node_count()];
    let mut queue = BinaryHeap::new();

    for source in query.sources() {
        let node = source.node();
        let distance = u64::from(source.offset());
        if distance < distances[node.index()] {
            distances[node.index()] = distance;
            predecessors[node.index()] = None;
            queue.push(QueueState { distance, node });
        }
    }

    while let Some(state) = queue.pop() {
        if state.distance != distances[state.node.index()] {
            continue;
        }
        for &coordinate in &outgoing_coordinates[state.node.index()] {
            let head = topology.heads()[coordinate];
            let increment = quantized_weights[coordinate];
            let candidate = state.distance.checked_add(u64::from(increment)).ok_or(
                DijkstraError::DistanceOverflow {
                    query: query_index,
                    node: state.node,
                    coordinate,
                    distance: state.distance,
                    increment,
                },
            )?;

            // ROUTING_INFINITY is the backend-neutral sentinel. Paths reaching
            // it are unavailable, matching production CCH semantics.
            if candidate >= u64::from(ROUTING_INFINITY) || candidate >= distances[head.index()] {
                continue;
            }
            distances[head.index()] = candidate;
            predecessors[head.index()] = Some(Predecessor {
                node: state.node,
                coordinate,
            });
            queue.push(QueueState {
                distance: candidate,
                node: head,
            });
        }
    }

    let mut best_target = None::<(u64, EdgeId, u32, usize)>;
    for (target_index, target) in query.targets().iter().enumerate() {
        let distance = distances[target.node().index()];
        if distance == u64::MAX {
            continue;
        }
        let total = distance.checked_add(u64::from(target.offset())).ok_or(
            DijkstraError::TargetDistanceOverflow {
                query: query_index,
                node: target.node(),
                distance,
                offset: target.offset(),
            },
        )?;
        if total >= u64::from(ROUTING_INFINITY) {
            continue;
        }
        let candidate = (total, target.node(), target.offset(), target_index);
        if best_target.is_none_or(|best| candidate < best) {
            best_target = Some(candidate);
        }
    }

    let (distance, target, _, _) =
        best_target.ok_or(DijkstraError::Unreachable { query: query_index })?;
    let (nodes, coordinates) = reconstruct_path(target, &predecessors, query_index)?;
    let distance = u32::try_from(distance).map_err(|_| DijkstraError::DistanceOutOfRange {
        query: query_index,
        distance,
    })?;
    Ok(OraclePath::new(distance, nodes, coordinates))
}

fn reconstruct_path(
    target: EdgeId,
    predecessors: &[Option<Predecessor>],
    query_index: usize,
) -> Result<(Vec<EdgeId>, Vec<usize>), DijkstraError> {
    let mut nodes = vec![target];
    let mut coordinates = Vec::new();
    let mut current = target;

    while let Some(predecessor) = predecessors[current.index()] {
        if nodes.len() > predecessors.len() {
            return Err(DijkstraError::PredecessorCycle { query: query_index });
        }
        coordinates.push(predecessor.coordinate);
        current = predecessor.node;
        nodes.push(current);
    }
    nodes.reverse();
    coordinates.reverse();
    Ok((nodes, coordinates))
}

fn validate_topology(topology: &RoutingTopology) -> Result<(), DijkstraError> {
    if topology.node_count() == 0 {
        return Err(DijkstraError::EmptyTopology);
    }
    if topology.tails().is_empty() || topology.tails().len() != topology.heads().len() {
        return Err(DijkstraError::InvalidArcArrays {
            tails: topology.tails().len(),
            heads: topology.heads().len(),
        });
    }
    for (coordinate, (&tail, &head)) in topology.tails().iter().zip(topology.heads()).enumerate() {
        if tail.index() >= topology.node_count() || head.index() >= topology.node_count() {
            return Err(DijkstraError::ArcEndpointOutOfBounds {
                coordinate,
                tail,
                head,
                node_count: topology.node_count(),
            });
        }
    }
    Ok(())
}

fn validate_metric(weights: &[u32], expected: usize) -> Result<(), DijkstraError> {
    if weights.len() != expected {
        return Err(DijkstraError::MetricLength {
            actual: weights.len(),
            expected,
        });
    }
    if let Some((coordinate, weight)) = weights
        .iter()
        .copied()
        .enumerate()
        .find(|(_, weight)| *weight == 0 || *weight >= ROUTING_INFINITY)
    {
        return Err(DijkstraError::InvalidMetricWeight { coordinate, weight });
    }
    Ok(())
}

fn validate_query(
    query: &OracleQuery,
    node_count: usize,
    query_index: usize,
) -> Result<(), DijkstraError> {
    validate_endpoints(
        query.sources(),
        node_count,
        query_index,
        EndpointKind::Source,
    )?;
    validate_endpoints(
        query.targets(),
        node_count,
        query_index,
        EndpointKind::Target,
    )
}

fn validate_endpoints(
    endpoints: &[QueryEndpoint],
    node_count: usize,
    query: usize,
    kind: EndpointKind,
) -> Result<(), DijkstraError> {
    if endpoints.is_empty() {
        return Err(DijkstraError::EmptyEndpoints { query, kind });
    }
    for endpoint in endpoints {
        if endpoint.node().index() >= node_count {
            return Err(DijkstraError::EndpointOutOfBounds {
                query,
                kind,
                node: endpoint.node(),
                node_count,
            });
        }
        if endpoint.offset() >= ROUTING_INFINITY {
            return Err(DijkstraError::InvalidEndpointOffset {
                query,
                kind,
                offset: endpoint.offset(),
            });
        }
    }
    Ok(())
}

/// Endpoint side associated with an invalid research query.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EndpointKind {
    /// Multi-source endpoint.
    Source,
    /// Multi-target endpoint.
    Target,
}

/// Invalid topology, metric, query, or Dijkstra result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DijkstraError {
    /// The routing topology has no nodes.
    EmptyTopology,
    /// Arc tail/head arrays are empty or have different lengths.
    InvalidArcArrays { tails: usize, heads: usize },
    /// One arc references a node outside the topology.
    ArcEndpointOutOfBounds {
        coordinate: usize,
        tail: EdgeId,
        head: EdgeId,
        node_count: usize,
    },
    /// Metric length differs from the topology arc count.
    MetricLength { actual: usize, expected: usize },
    /// One metric value is zero or reaches the routing sentinel.
    InvalidMetricWeight { coordinate: usize, weight: u32 },
    /// A source or target list is empty.
    EmptyEndpoints { query: usize, kind: EndpointKind },
    /// A source or target node is outside the topology.
    EndpointOutOfBounds {
        query: usize,
        kind: EndpointKind,
        node: EdgeId,
        node_count: usize,
    },
    /// A source or target offset reaches the routing sentinel.
    InvalidEndpointOffset {
        query: usize,
        kind: EndpointKind,
        offset: u32,
    },
    /// Checked arc relaxation exceeded the internal distance type.
    DistanceOverflow {
        query: usize,
        node: EdgeId,
        coordinate: usize,
        distance: u64,
        increment: u32,
    },
    /// Checked target-offset addition exceeded the internal distance type.
    TargetDistanceOverflow {
        query: usize,
        node: EdgeId,
        distance: u64,
        offset: u32,
    },
    /// A finite result did not fit the core distance type.
    DistanceOutOfRange { query: usize, distance: u64 },
    /// No source can reach any target below the routing sentinel.
    Unreachable { query: usize },
    /// An internal predecessor chain was cyclic.
    PredecessorCycle { query: usize },
}

impl Display for DijkstraError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for DijkstraError {}

#[cfg(test)]
mod tests {
    use super::*;
    use ewr_cch::CchOracle;
    use ewr_core::{LineGraph, NodeId, RoadNetwork};

    fn branching_line_graph() -> LineGraph {
        let network = RoadNetwork::new(
            vec![
                NodeId::new(0),
                NodeId::new(1),
                NodeId::new(0),
                NodeId::new(2),
            ],
            vec![
                NodeId::new(1),
                NodeId::new(3),
                NodeId::new(2),
                NodeId::new(3),
            ],
            vec![1.0, 5.0, 1.0, 2.0],
            vec![0.0, 1.0, 1.0, 2.0],
            vec![0.0, 0.0, 1.0, 0.5],
        )
        .unwrap();
        LineGraph::build(&network, 0.1, 10.0).unwrap()
    }

    #[test]
    fn one_original_edge_has_zero_transition_cost() {
        let network = RoadNetwork::new(
            vec![NodeId::new(0), NodeId::new(1)],
            vec![NodeId::new(1), NodeId::new(2)],
            vec![7.0, 11.0],
            vec![0.0, 1.0, 2.0],
            vec![0.0, 0.0, 0.0],
        )
        .unwrap();
        let graph = LineGraph::build(&network, 0.1, 10.0).unwrap();
        let (sources, targets) = graph
            .node_query_endpoints(NodeId::new(0), NodeId::new(1))
            .unwrap();
        let mut oracle = DijkstraOracle::new();

        let path = oracle
            .shortest_paths_checked(
                graph.routing_topology(),
                &[11],
                &[OracleQuery::new(sources, targets)],
            )
            .unwrap()
            .remove(0);

        assert_eq!(path.distance(), 0);
        assert_eq!(path.nodes(), &[EdgeId::new(0)]);
        assert!(path.coordinates().is_empty());
    }

    #[test]
    fn multi_source_multi_target_path_matches_production_cch() {
        let graph = branching_line_graph();
        let (sources, targets) = graph
            .node_query_endpoints(NodeId::new(0), NodeId::new(3))
            .unwrap();
        assert_eq!(sources.len(), 2);
        assert_eq!(targets.len(), 2);
        let queries = [OracleQuery::new(sources, targets)];
        let metric = [5, 2];
        let mut dijkstra = DijkstraOracle::new();
        let mut cch = CchOracle::new();

        let dijkstra_path = dijkstra
            .shortest_paths(graph.routing_topology(), &metric, &queries)
            .unwrap();
        let cch_path = cch
            .shortest_paths(graph.routing_topology(), &metric, &queries)
            .unwrap();

        assert_eq!(dijkstra_path, cch_path);
        assert_eq!(dijkstra_path[0].distance(), 2);
        assert_eq!(dijkstra_path[0].nodes(), &[EdgeId::new(2), EdgeId::new(3)]);
        assert_eq!(dijkstra_path[0].coordinates(), &[1]);
    }

    #[test]
    fn ties_choose_the_smallest_stable_target_node() {
        let graph = branching_line_graph();
        let (sources, targets) = graph
            .node_query_endpoints(NodeId::new(0), NodeId::new(3))
            .unwrap();
        let mut oracle = DijkstraOracle::new();

        let path = oracle
            .shortest_paths_checked(
                graph.routing_topology(),
                &[3, 3],
                &[OracleQuery::new(sources, targets)],
            )
            .unwrap()
            .remove(0);

        assert_eq!(path.nodes(), &[EdgeId::new(0), EdgeId::new(1)]);
        assert_eq!(path.coordinates(), &[0]);
    }

    #[test]
    fn invalid_metric_is_a_typed_error() {
        let graph = branching_line_graph();
        let error = DijkstraOracle::new()
            .shortest_paths_checked(graph.routing_topology(), &[5, 0], &[])
            .unwrap_err();

        assert_eq!(
            error,
            DijkstraError::InvalidMetricWeight {
                coordinate: 1,
                weight: 0,
            }
        );
    }
}
