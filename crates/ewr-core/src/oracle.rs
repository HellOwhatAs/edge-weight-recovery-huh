use crate::{EdgeId, RoutingTopology};
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Largest exclusive value accepted by the active v1 integer routing metric.
///
/// RoutingKit uses `i32::MAX` as its infinity sentinel. Keeping the limit in
/// core makes every oracle backend consume the same byte-identical metric.
pub const ROUTING_INFINITY: u32 = i32::MAX as u32;

/// One multi-source or multi-target routing state and its initial cost.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueryEndpoint {
    node: EdgeId,
    offset: u32,
}

impl QueryEndpoint {
    /// Construct an endpoint. Production line-graph queries use [`Self::zero`].
    pub const fn new(node: EdgeId, offset: u32) -> Self {
        Self { node, offset }
    }

    /// Construct the frozen v1 zero-offset edge state.
    pub const fn zero(node: EdgeId) -> Self {
        Self::new(node, 0)
    }

    /// Stable line-graph node, equal to an original road-edge ID.
    pub const fn node(self) -> EdgeId {
        self.node
    }

    /// Initial endpoint cost in the quantized routing metric.
    pub const fn offset(self) -> u32 {
        self.offset
    }
}

/// One ordered multi-source/multi-target request given to a routing backend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OracleQuery {
    sources: Box<[QueryEndpoint]>,
    targets: Box<[QueryEndpoint]>,
}

impl OracleQuery {
    /// Construct a routing request from representation-owned endpoints.
    pub fn new(sources: Vec<QueryEndpoint>, targets: Vec<QueryEndpoint>) -> Self {
        Self {
            sources: sources.into_boxed_slice(),
            targets: targets.into_boxed_slice(),
        }
    }

    /// Candidate source edge states in stable edge-ID order.
    pub fn sources(&self) -> &[QueryEndpoint] {
        &self.sources
    }

    /// Candidate target edge states in stable edge-ID order.
    pub fn targets(&self) -> &[QueryEndpoint] {
        &self.targets
    }
}

/// Stable path returned by a routing backend.
///
/// Routing nodes are original [`EdgeId`] values and coordinates are line-graph
/// arc IDs. Neither vector contains a backend-specific CCH object.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OraclePath {
    distance: u32,
    nodes: Box<[EdgeId]>,
    coordinates: Box<[usize]>,
}

impl OraclePath {
    /// Construct a path for validation and aggregation by core.
    pub fn new(distance: u32, nodes: Vec<EdgeId>, coordinates: Vec<usize>) -> Self {
        Self {
            distance,
            nodes: nodes.into_boxed_slice(),
            coordinates: coordinates.into_boxed_slice(),
        }
    }

    /// Quantized multi-source/multi-target distance reported by the backend.
    pub const fn distance(&self) -> u32 {
        self.distance
    }

    /// Stable line-graph node path, including both endpoint edge states.
    pub fn nodes(&self) -> &[EdgeId] {
        &self.nodes
    }

    /// Stable transition-coordinate path.
    pub fn coordinates(&self) -> &[usize] {
        &self.coordinates
    }
}

/// Coarse backend-neutral shortest-path port used by the sole core trainer.
///
/// A call represents one complete metric customization and one ordered batch.
/// Backends may reuse internal query buffers, but optimizer state, direct-cost
/// evaluation, sample multiplicities, and count aggregation stay in core.
pub trait RoutingOracle {
    /// Stable, explicitly versioned identity of this backend's routing semantics.
    ///
    /// The value becomes part of every training checkpoint. Implementations
    /// must change it whenever preprocessing, tie-breaking, metric handling,
    /// or path reconstruction can change a training result. There is
    /// deliberately no default: an anonymous backend cannot produce a
    /// resumable checkpoint.
    fn identity(&self) -> &'static str;

    /// Return exactly one path for each request, preserving request order.
    fn shortest_paths(
        &mut self,
        topology: &RoutingTopology,
        quantized_weights: &[u32],
        queries: &[OracleQuery],
    ) -> Result<Vec<OraclePath>, OracleError>;
}

/// Failure reported by a concrete routing backend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OracleError {
    message: String,
}

impl OracleError {
    /// Wrap a backend failure without leaking its implementation types.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Backend-provided diagnostic text.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl Display for OracleError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for OracleError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_query_and_path_values_preserve_order() {
        let query = OracleQuery::new(
            vec![QueryEndpoint::zero(EdgeId::new(3))],
            vec![QueryEndpoint::zero(EdgeId::new(9))],
        );
        assert_eq!(query.sources()[0].node(), EdgeId::new(3));
        assert_eq!(query.sources()[0].offset(), 0);
        assert_eq!(query.targets()[0].node(), EdgeId::new(9));

        let path = OraclePath::new(17, vec![EdgeId::new(3), EdgeId::new(9)], vec![4]);
        assert_eq!(path.distance(), 17);
        assert_eq!(path.nodes(), &[EdgeId::new(3), EdgeId::new(9)]);
        assert_eq!(path.coordinates(), &[4]);
    }
}
