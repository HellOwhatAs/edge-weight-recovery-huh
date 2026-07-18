//! RoutingKit CCH adapter for the backend-neutral production trainer.
//!
//! Elimination orders, customized metrics, query buffers, and backend
//! lifetimes stop at this crate boundary. The algorithm crate only observes
//! stable line-graph node IDs, transition-coordinate IDs, and integer costs.

use ewr_core::{
    EdgeId, FitOptions, FitResult, OracleError, OraclePath, OracleQuery, ROUTING_INFINITY,
    RoadNetwork, RoutingOracle, RoutingTopology, TrainerError, Trajectory,
};
use rayon::prelude::*;
use routingkit_cch::{CCH, CCHMetric, CCHQuery, compute_order_inertial};
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Checkpoint identity of the production CCH routing semantics.
pub const CCH_ORACLE_IDENTITY_V1: &str =
    "ewr-cch:routingkit-cch-0.1.3:inertial-order:u32-metric:v1";

/// Lazily preprocessed CCH oracle for backend-independent line graphs.
pub struct CchOracle {
    cached: Option<CachedCch>,
}

struct CachedCch {
    routing_fingerprint: u64,
    cch: CCH,
}

impl CchOracle {
    /// Construct an empty oracle. Preprocessing occurs on its first query.
    pub const fn new() -> Self {
        Self { cached: None }
    }

    fn cch_for(&mut self, topology: &RoutingTopology) -> Result<&CCH, CchError> {
        let routing_fingerprint = topology.fingerprint();
        let must_rebuild = self
            .cached
            .as_ref()
            .is_none_or(|cached| cached.routing_fingerprint != routing_fingerprint);
        if must_rebuild {
            self.cached = Some(CachedCch {
                routing_fingerprint,
                cch: build_cch(topology)?,
            });
        }
        Ok(&self
            .cached
            .as_ref()
            .expect("a missing or stale CCH cache was rebuilt above")
            .cch)
    }
}

impl Default for CchOracle {
    fn default() -> Self {
        Self::new()
    }
}

impl RoutingOracle for CchOracle {
    fn identity(&self) -> &'static str {
        CCH_ORACLE_IDENTITY_V1
    }

    fn shortest_paths(
        &mut self,
        topology: &RoutingTopology,
        quantized_weights: &[u32],
        queries: &[OracleQuery],
    ) -> Result<Vec<OraclePath>, OracleError> {
        validate_metric(quantized_weights, topology.arc_count())
            .map_err(|error| OracleError::new(error.to_string()))?;
        let node_count = topology.node_count();
        let cch = self
            .cch_for(topology)
            .map_err(|error| OracleError::new(error.to_string()))?;

        let metric = CCHMetric::new(cch, quantized_weights.to_vec());
        queries
            .par_iter()
            .map(|query| run_query(&metric, node_count, query))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| OracleError::new(error.to_string()))
    }
}

/// Fit the production model with the default CCH backend.
///
/// The core trainer builds the line graph once. [`CchOracle`] then lazily
/// preprocesses the exact routing topology passed through the oracle port.
pub fn fit(
    network: &RoadNetwork,
    trajectories: &[Trajectory],
    options: &FitOptions,
) -> Result<FitResult, TrainerError> {
    let mut oracle = CchOracle::new();
    ewr_core::fit(network, trajectories, options, &mut oracle)
}

fn build_cch(topology: &RoutingTopology) -> Result<CCH, CchError> {
    validate_topology(topology)?;
    let node_count = u32::try_from(topology.node_count())
        .map_err(|_| CchError::TooManyNodes(topology.node_count()))?;
    let tails = edge_ids_to_u32(topology.tails());
    let heads = edge_ids_to_u32(topology.heads());
    let order = compute_order_inertial(node_count, &tails, &heads, topology.x(), topology.y());
    if order.len() != topology.node_count() {
        return Err(CchError::InvalidOrderLength {
            actual: order.len(),
            expected: topology.node_count(),
        });
    }
    Ok(CCH::new(&order, &tails, &heads, |_| {}, false))
}

fn run_query(
    metric: &CCHMetric<'_>,
    node_count: usize,
    query: &OracleQuery,
) -> Result<OraclePath, CchError> {
    validate_endpoints(query.sources(), node_count, EndpointKind::Source)?;
    validate_endpoints(query.targets(), node_count, EndpointKind::Target)?;

    let mut cch_query = CCHQuery::new(metric);
    for endpoint in query.sources() {
        cch_query.add_source(endpoint.node().index() as u32, endpoint.offset());
    }
    for endpoint in query.targets() {
        cch_query.add_target(endpoint.node().index() as u32, endpoint.offset());
    }
    let result = cch_query.run();
    let distance = result.distance().ok_or(CchError::Unreachable)?;
    let nodes = result.node_path().into_iter().map(EdgeId::new).collect();
    let coordinates = result
        .arc_path()
        .into_iter()
        .map(|coordinate| coordinate as usize)
        .collect();
    Ok(OraclePath::new(distance, nodes, coordinates))
}

fn edge_ids_to_u32(ids: &[EdgeId]) -> Vec<u32> {
    ids.iter().map(|id| id.index() as u32).collect()
}

fn validate_topology(topology: &RoutingTopology) -> Result<(), CchError> {
    if topology.node_count() == 0 {
        return Err(CchError::EmptyTopology);
    }
    if topology.node_count() > u32::MAX as usize {
        return Err(CchError::TooManyNodes(topology.node_count()));
    }
    if topology.tails().is_empty() || topology.tails().len() != topology.heads().len() {
        return Err(CchError::InvalidArcArrays {
            tails: topology.tails().len(),
            heads: topology.heads().len(),
        });
    }
    if topology.x().len() != topology.node_count() || topology.y().len() != topology.node_count() {
        return Err(CchError::InvalidCoordinateArrays {
            nodes: topology.node_count(),
            x: topology.x().len(),
            y: topology.y().len(),
        });
    }
    if let Some((node, (&x, &y))) = topology
        .x()
        .iter()
        .zip(topology.y())
        .enumerate()
        .find(|(_, (x, y))| !x.is_finite() || !y.is_finite())
    {
        return Err(CchError::InvalidCoordinates { node, x, y });
    }
    for (coordinate, (&tail, &head)) in topology.tails().iter().zip(topology.heads()).enumerate() {
        if tail.index() >= topology.node_count() || head.index() >= topology.node_count() {
            return Err(CchError::ArcEndpointOutOfBounds {
                coordinate,
                tail,
                head,
                node_count: topology.node_count(),
            });
        }
    }
    Ok(())
}

fn validate_metric(weights: &[u32], expected: usize) -> Result<(), CchError> {
    if weights.len() != expected {
        return Err(CchError::MetricLength {
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
        return Err(CchError::InvalidMetricWeight { coordinate, weight });
    }
    Ok(())
}

fn validate_endpoints(
    endpoints: &[ewr_core::QueryEndpoint],
    node_count: usize,
    kind: EndpointKind,
) -> Result<(), CchError> {
    if endpoints.is_empty() {
        return Err(CchError::EmptyEndpoints(kind));
    }
    for endpoint in endpoints {
        if endpoint.node().index() >= node_count {
            return Err(CchError::EndpointOutOfBounds {
                kind,
                node: endpoint.node(),
                node_count,
            });
        }
        if endpoint.offset() >= ROUTING_INFINITY {
            return Err(CchError::InvalidEndpointOffset {
                kind,
                offset: endpoint.offset(),
            });
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EndpointKind {
    Source,
    Target,
}

/// Invalid topology, metric, query, or CCH result.
#[derive(Clone, Debug, PartialEq)]
enum CchError {
    EmptyTopology,
    TooManyNodes(usize),
    InvalidArcArrays {
        tails: usize,
        heads: usize,
    },
    InvalidCoordinateArrays {
        nodes: usize,
        x: usize,
        y: usize,
    },
    InvalidCoordinates {
        node: usize,
        x: f32,
        y: f32,
    },
    ArcEndpointOutOfBounds {
        coordinate: usize,
        tail: EdgeId,
        head: EdgeId,
        node_count: usize,
    },
    InvalidOrderLength {
        actual: usize,
        expected: usize,
    },
    MetricLength {
        actual: usize,
        expected: usize,
    },
    InvalidMetricWeight {
        coordinate: usize,
        weight: u32,
    },
    EmptyEndpoints(EndpointKind),
    EndpointOutOfBounds {
        kind: EndpointKind,
        node: EdgeId,
        node_count: usize,
    },
    InvalidEndpointOffset {
        kind: EndpointKind,
        offset: u32,
    },
    Unreachable,
}

impl Display for CchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for CchError {}

#[cfg(test)]
mod tests {
    use super::*;
    use ewr_core::{FitOptions, LineGraph, NodeId, RoadNetwork, Trajectory};

    fn network() -> RoadNetwork {
        RoadNetwork::new(
            vec![
                NodeId::new(0),
                NodeId::new(1),
                NodeId::new(2),
                NodeId::new(1),
                NodeId::new(4),
            ],
            vec![
                NodeId::new(1),
                NodeId::new(2),
                NodeId::new(3),
                NodeId::new(4),
                NodeId::new(3),
            ],
            vec![2.0, 3.0, 5.0, 1.0, 1.0],
            vec![0.0, 1.0, 2.0, 3.0, 2.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap()
    }

    fn line_graph() -> LineGraph {
        LineGraph::build(&network(), 0.1, 10.0).unwrap()
    }

    #[test]
    fn cch_returns_stable_line_nodes_and_coordinate_ids() {
        let graph = line_graph();
        let (sources, targets) = graph
            .node_query_endpoints(NodeId::new(0), NodeId::new(3))
            .unwrap();
        let query = OracleQuery::new(sources, targets);
        let mut oracle = CchOracle::new();

        let paths = oracle
            .shortest_paths(graph.routing_topology(), &[3, 1, 5, 1], &[query])
            .unwrap();

        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].distance(), 2);
        assert_eq!(
            paths[0].nodes(),
            &[EdgeId::new(0), EdgeId::new(3), EdgeId::new(4)]
        );
        assert_eq!(paths[0].coordinates(), &[1, 3]);
    }

    #[test]
    fn cch_uses_the_frozen_quantized_selection_metric() {
        let graph = line_graph();
        let (sources, targets) = graph
            .node_query_endpoints(NodeId::new(0), NodeId::new(3))
            .unwrap();
        let mut oracle = CchOracle::new();
        let path = oracle
            .shortest_paths(
                graph.routing_topology(),
                &[2, 1, 2, 2],
                &[OracleQuery::new(sources, targets)],
            )
            .unwrap()
            .remove(0);

        assert_eq!(path.distance(), 3);
        assert_eq!(path.coordinates(), &[1, 3]);
    }

    #[test]
    fn invalid_metric_is_rejected_before_routingkit() {
        let graph = line_graph();
        let mut oracle = CchOracle::new();
        let error = oracle
            .shortest_paths(graph.routing_topology(), &[3, 0, 5, 1], &[])
            .unwrap_err();
        assert!(error.message().contains("InvalidMetricWeight"));
        assert!(oracle.cached.is_none());
    }

    #[test]
    fn cache_rebuilds_when_routing_geometry_identity_changes() {
        let first = line_graph();
        let changed_geometry = RoadNetwork::new(
            network().tails().to_vec(),
            network().heads().to_vec(),
            network().baseline_weights().to_vec(),
            vec![10.0, 11.0, 12.0, 13.0, 12.0],
            vec![7.0, 7.0, 7.0, 7.0, 8.0],
        )
        .unwrap();
        let second = LineGraph::build(&changed_geometry, 0.1, 10.0).unwrap();
        assert_ne!(
            first.routing_topology().fingerprint(),
            second.routing_topology().fingerprint()
        );

        let mut oracle = CchOracle::new();
        oracle
            .shortest_paths(first.routing_topology(), &[3, 1, 5, 1], &[])
            .unwrap();
        assert_eq!(
            oracle.cached.as_ref().unwrap().routing_fingerprint,
            first.routing_topology().fingerprint()
        );
        oracle
            .shortest_paths(second.routing_topology(), &[3, 1, 5, 1], &[])
            .unwrap();
        assert_eq!(
            oracle.cached.as_ref().unwrap().routing_fingerprint,
            second.routing_topology().fingerprint()
        );
    }

    #[test]
    fn three_argument_facade_runs_the_core_trainer() {
        let result = fit(
            &network(),
            &[Trajectory::new(vec![
                EdgeId::new(0),
                EdgeId::new(1),
                EdgeId::new(2),
            ])],
            &FitOptions {
                eta0: 0.01,
                lambda: 0.1,
                lower_factor: 0.1,
                upper_factor: 10.0,
                updates: 1,
            },
        )
        .unwrap();

        assert_eq!(result.model.transitions().len(), 4);
        assert_eq!(result.diagnostics.completed_updates, 1);
    }
}
