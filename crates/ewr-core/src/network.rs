use std::collections::HashSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Stable ID of an original road-network node.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct NodeId(u32);

impl NodeId {
    /// Construct a node ID from its zero-based storage index.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Return the zero-based storage index.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Stable ID of an original directed road edge.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EdgeId(u32);

impl EdgeId {
    /// Construct an edge ID from its zero-based storage index.
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Return the zero-based storage index.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Typed directed road network consumed by the production algorithm.
#[derive(Clone, Debug, PartialEq)]
pub struct RoadNetwork {
    tail: Box<[NodeId]>,
    head: Box<[NodeId]>,
    baseline_weights: Box<[f64]>,
    x: Box<[f32]>,
    y: Box<[f32]>,
}

impl RoadNetwork {
    /// Validate and construct one directed road network.
    pub fn new(
        tail: Vec<NodeId>,
        head: Vec<NodeId>,
        baseline_weights: Vec<f64>,
        x: Vec<f32>,
        y: Vec<f32>,
    ) -> Result<Self, NetworkError> {
        if x.is_empty() || x.len() != y.len() {
            return Err(NetworkError::InvalidNodeArrays {
                x: x.len(),
                y: y.len(),
            });
        }
        if x.len() > u32::MAX as usize {
            return Err(NetworkError::TooManyNodes(x.len()));
        }
        if tail.is_empty() || tail.len() != head.len() || tail.len() != baseline_weights.len() {
            return Err(NetworkError::InvalidEdgeArrays {
                tail: tail.len(),
                head: head.len(),
                weights: baseline_weights.len(),
            });
        }
        if tail.len() > u32::MAX as usize {
            return Err(NetworkError::TooManyEdges(tail.len()));
        }
        if let Some((node, (&node_x, &node_y))) = x
            .iter()
            .zip(&y)
            .enumerate()
            .find(|(_, (node_x, node_y))| !node_x.is_finite() || !node_y.is_finite())
        {
            return Err(NetworkError::InvalidCoordinates {
                node: NodeId::new(node as u32),
                x: node_x,
                y: node_y,
            });
        }
        for (edge, ((&edge_tail, &edge_head), &weight)) in
            tail.iter().zip(&head).zip(&baseline_weights).enumerate()
        {
            if edge_tail.index() >= x.len() || edge_head.index() >= x.len() {
                return Err(NetworkError::EndpointOutOfBounds {
                    edge: EdgeId::new(edge as u32),
                    tail: edge_tail,
                    head: edge_head,
                    node_count: x.len(),
                });
            }
            if !weight.is_finite() || weight <= 0.0 {
                return Err(NetworkError::InvalidBaselineWeight {
                    edge: EdgeId::new(edge as u32),
                    weight,
                });
            }
        }

        Ok(Self {
            tail: tail.into_boxed_slice(),
            head: head.into_boxed_slice(),
            baseline_weights: baseline_weights.into_boxed_slice(),
            x: x.into_boxed_slice(),
            y: y.into_boxed_slice(),
        })
    }

    /// Number of original road nodes.
    pub fn node_count(&self) -> usize {
        self.x.len()
    }

    /// Number of original directed road edges.
    pub fn edge_count(&self) -> usize {
        self.tail.len()
    }

    /// Original tail node of an edge.
    pub fn tail(&self, edge: EdgeId) -> Option<NodeId> {
        self.tail.get(edge.index()).copied()
    }

    /// Original head node of an edge.
    pub fn head(&self, edge: EdgeId) -> Option<NodeId> {
        self.head.get(edge.index()).copied()
    }

    /// Positive baseline weight of an edge.
    pub fn baseline_weight(&self, edge: EdgeId) -> Option<f64> {
        self.baseline_weights.get(edge.index()).copied()
    }

    /// Original edge tails in stable edge-ID order.
    pub fn tails(&self) -> &[NodeId] {
        &self.tail
    }

    /// Original edge heads in stable edge-ID order.
    pub fn heads(&self) -> &[NodeId] {
        &self.head
    }

    /// Original baseline weights in stable edge-ID order.
    pub fn baseline_weights(&self) -> &[f64] {
        &self.baseline_weights
    }

    /// X coordinates in stable node-ID order.
    pub fn x(&self) -> &[f32] {
        &self.x
    }

    /// Y coordinates in stable node-ID order.
    pub fn y(&self) -> &[f32] {
        &self.y
    }

    /// Validate a complete original-edge trajectory and return its OD pair.
    pub fn validate_trajectory(
        &self,
        trajectory: &Trajectory,
    ) -> Result<(NodeId, NodeId), NetworkError> {
        validate_trajectory_arrays(&self.tail, &self.head, trajectory)
    }
}

pub(crate) fn validate_trajectory_arrays(
    tail: &[NodeId],
    head: &[NodeId],
    trajectory: &Trajectory,
) -> Result<(NodeId, NodeId), NetworkError> {
    let edges = trajectory.edges();
    if edges.len() < 2 {
        return Err(NetworkError::TrajectoryTooShort(edges.len()));
    }
    if let Some(&edge) = edges.iter().find(|&&edge| edge.index() >= tail.len()) {
        return Err(NetworkError::TrajectoryEdgeOutOfBounds(edge));
    }
    for pair in edges.windows(2) {
        if head[pair[0].index()] != tail[pair[1].index()] {
            return Err(NetworkError::DiscontinuousTrajectory {
                previous: pair[0],
                next: pair[1],
            });
        }
    }

    let first = edges[0];
    let mut visited = HashSet::with_capacity(edges.len() + 1);
    visited.insert(tail[first.index()]);
    for &edge in edges {
        let node = head[edge.index()];
        if !visited.insert(node) {
            return Err(NetworkError::CyclicTrajectory(node));
        }
    }
    let last = edges[edges.len() - 1];
    Ok((tail[first.index()], head[last.index()]))
}

/// One complete observed path in original directed edge IDs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Trajectory {
    edges: Box<[EdgeId]>,
}

impl Trajectory {
    /// Construct a trajectory. Network-dependent validation occurs at the
    /// `RoadNetwork` boundary.
    pub fn new(edges: Vec<EdgeId>) -> Self {
        Self {
            edges: edges.into_boxed_slice(),
        }
    }

    /// Complete original-edge sequence.
    pub fn edges(&self) -> &[EdgeId] {
        &self.edges
    }
}

/// Structural input error at the stable core boundary.
#[derive(Clone, Debug, PartialEq)]
pub enum NetworkError {
    /// Node coordinate arrays are empty or mismatched.
    InvalidNodeArrays { x: usize, y: usize },
    /// Edge arrays are empty or mismatched.
    InvalidEdgeArrays {
        tail: usize,
        head: usize,
        weights: usize,
    },
    /// Node IDs cannot be represented by the stable ID type.
    TooManyNodes(usize),
    /// Edge IDs cannot be represented by the stable ID type.
    TooManyEdges(usize),
    /// A node has invalid coordinates.
    InvalidCoordinates { node: NodeId, x: f32, y: f32 },
    /// An edge endpoint is outside the node arrays.
    EndpointOutOfBounds {
        edge: EdgeId,
        tail: NodeId,
        head: NodeId,
        node_count: usize,
    },
    /// An edge baseline is nonpositive or nonfinite.
    InvalidBaselineWeight { edge: EdgeId, weight: f64 },
    /// A complete observation must have at least two edges under v1 semantics.
    TrajectoryTooShort(usize),
    /// A trajectory references an unknown edge.
    TrajectoryEdgeOutOfBounds(EdgeId),
    /// Consecutive original edges do not connect.
    DiscontinuousTrajectory { previous: EdgeId, next: EdgeId },
    /// A trajectory repeats an original node.
    CyclicTrajectory(NodeId),
}

impl Display for NetworkError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for NetworkError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn network() -> RoadNetwork {
        RoadNetwork::new(
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
            vec![5.0, 5.0, 2.0, 2.0],
            vec![0.0, 1.0, 1.0, 2.0],
            vec![0.0, 0.0, 1.0, 0.0],
        )
        .unwrap()
    }

    #[test]
    fn validates_complete_original_edge_trajectories() {
        let network = network();
        assert_eq!(
            network
                .validate_trajectory(&Trajectory::new(vec![EdgeId::new(0), EdgeId::new(1)]))
                .unwrap(),
            (NodeId::new(0), NodeId::new(3))
        );
        assert!(
            network
                .validate_trajectory(&Trajectory::new(vec![EdgeId::new(0)]))
                .is_err()
        );
        assert!(
            network
                .validate_trajectory(&Trajectory::new(vec![EdgeId::new(0), EdgeId::new(3)]))
                .is_err()
        );
    }
}
