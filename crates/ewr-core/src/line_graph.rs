use crate::network::validate_trajectory_arrays;
use crate::oracle::{QueryEndpoint, ROUTING_INFINITY};
use crate::{EdgeId, NetworkError, NodeId, RoadNetwork, TopologyId, Trajectory, Transition};
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

const MAX_TRANSITIONS: usize = 12_000_000;

/// Backend-independent directed line-graph arrays.
///
/// Each original road edge is one routing node. Consequently, routing-node
/// IDs are represented by [`EdgeId`], and routing arcs are in exactly the same
/// order as the learned transition coordinates returned by
/// [`LineGraph::transitions`].
#[derive(Clone, Debug, PartialEq)]
pub struct RoutingTopology {
    node_count: usize,
    tails: Box<[EdgeId]>,
    heads: Box<[EdgeId]>,
    x: Box<[f32]>,
    y: Box<[f32]>,
    fingerprint: u64,
}

impl RoutingTopology {
    fn new(
        node_count: usize,
        tails: Vec<EdgeId>,
        heads: Vec<EdgeId>,
        x: Vec<f32>,
        y: Vec<f32>,
    ) -> Self {
        let fingerprint = routing_topology_fingerprint(node_count, &tails, &heads, &x, &y);
        Self {
            node_count,
            tails: tails.into_boxed_slice(),
            heads: heads.into_boxed_slice(),
            x: x.into_boxed_slice(),
            y: y.into_boxed_slice(),
            fingerprint,
        }
    }

    /// Number of line-graph nodes (and therefore original road edges).
    pub const fn node_count(&self) -> usize {
        self.node_count
    }

    /// Number of line-graph arcs and learned transition coordinates.
    pub fn arc_count(&self) -> usize {
        self.tails.len()
    }

    /// Tail line-graph node of every transition coordinate.
    pub fn tails(&self) -> &[EdgeId] {
        &self.tails
    }

    /// Head line-graph node of every transition coordinate.
    pub fn heads(&self) -> &[EdgeId] {
        &self.heads
    }

    /// Midpoint X coordinate of every line-graph node.
    pub fn x(&self) -> &[f32] {
        &self.x
    }

    /// Midpoint Y coordinate of every line-graph node.
    pub fn y(&self) -> &[f32] {
        &self.y
    }

    /// Stable backend-preprocessing identity of all routing arrays.
    ///
    /// Unlike [`TopologyId`], this fingerprint includes geometry because an
    /// inertial CCH order depends on it. It excludes weights and fit options.
    pub const fn fingerprint(&self) -> u64 {
        self.fingerprint
    }
}

/// One accepted observation mapped onto transition coordinates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct MappedTrajectory {
    /// Original source node.
    pub source: NodeId,
    /// Original target node.
    pub target: NodeId,
    /// Learned coordinates, one for every consecutive original-edge pair.
    pub coordinates: Box<[usize]>,
}

/// One unique original-node OD query and its observation multiplicity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct QueryGroup {
    pub source: NodeId,
    pub target: NodeId,
    pub sample_count: u64,
}

/// Directed line graph and the complete v1 coordinate mapping.
///
/// This type deliberately contains no shortest-path backend. It establishes
/// coordinate meaning, weights, bounds, observations, and query endpoints;
/// an oracle consumes [`RoutingTopology`] and the returned endpoint lists
/// separately.
#[derive(Clone, Debug, PartialEq)]
pub struct LineGraph {
    original_node_count: usize,
    original_tails: Box<[NodeId]>,
    original_heads: Box<[NodeId]>,
    routing_topology: RoutingTopology,
    transitions: Box<[Transition]>,
    transition_offsets: Box<[usize]>,
    outgoing_offsets: Box<[usize]>,
    outgoing_edges: Box<[EdgeId]>,
    incoming_offsets: Box<[usize]>,
    incoming_edges: Box<[EdgeId]>,
    initial_weights: Box<[f64]>,
    lower_bounds: Box<[f64]>,
    upper_bounds: Box<[f64]>,
    topology_id: TopologyId,
}

impl LineGraph {
    /// Construct the active v1 directed line graph and multiplicative box.
    ///
    /// Transition coordinate order is stable: first by `previous` edge ID and
    /// then by `next` edge ID. A coordinate's initial value is the baseline
    /// weight of the edge being entered; there is no first-edge coordinate.
    pub fn build(
        network: &RoadNetwork,
        lower_factor: f64,
        upper_factor: f64,
    ) -> Result<Self, LineGraphError> {
        validate_bound_factors(lower_factor, upper_factor)?;

        let node_count = network.node_count();
        let edge_count = network.edge_count();
        let (outgoing_offsets, outgoing_edges) = build_incidence(node_count, network.tails())?;
        let (incoming_offsets, incoming_edges) = build_incidence(node_count, network.heads())?;

        let mut transition_count = 0usize;
        for previous in 0..edge_count {
            let junction = network.heads()[previous].index();
            transition_count = transition_count
                .checked_add(incidence_slice(&outgoing_offsets, &outgoing_edges, junction).len())
                .ok_or(LineGraphError::TransitionCountOverflow)?;
        }
        if transition_count == 0 {
            return Err(LineGraphError::NoTransitions);
        }
        if transition_count > MAX_TRANSITIONS {
            return Err(LineGraphError::TooManyTransitions {
                count: transition_count,
                maximum: MAX_TRANSITIONS,
            });
        }

        let mut transitions = Vec::with_capacity(transition_count);
        let mut transition_offsets = Vec::with_capacity(edge_count + 1);
        let mut routing_tails = Vec::with_capacity(transition_count);
        let mut routing_heads = Vec::with_capacity(transition_count);
        let mut initial_weights = Vec::with_capacity(transition_count);
        transition_offsets.push(0);

        for previous_index in 0..edge_count {
            let previous = EdgeId::new(previous_index as u32);
            let junction = network.heads()[previous_index].index();
            for &next in incidence_slice(&outgoing_offsets, &outgoing_edges, junction) {
                transitions.push(Transition { previous, next });
                routing_tails.push(previous);
                routing_heads.push(next);
                initial_weights.push(network.baseline_weights()[next.index()]);
            }
            transition_offsets.push(transitions.len());
        }

        debug_assert_eq!(transitions.len(), transition_count);
        debug_assert!(transition_offsets.windows(2).all(|range| {
            transitions[range[0]..range[1]].is_sorted_by_key(|transition| transition.next)
        }));

        let lower_bounds = scaled_bounds(&initial_weights, lower_factor, BoundKind::Lower)?;
        let upper_bounds = scaled_bounds(&initial_weights, upper_factor, BoundKind::Upper)?;
        validate_routing_upper_bounds(&upper_bounds)?;
        let topology_id = topology_identity(network, &transitions);
        let routing_x = network
            .tails()
            .iter()
            .zip(network.heads())
            .map(|(&tail, &head)| 0.5 * network.x()[tail.index()] + 0.5 * network.x()[head.index()])
            .collect::<Vec<_>>();
        let routing_y = network
            .tails()
            .iter()
            .zip(network.heads())
            .map(|(&tail, &head)| 0.5 * network.y()[tail.index()] + 0.5 * network.y()[head.index()])
            .collect::<Vec<_>>();

        Ok(Self {
            original_node_count: node_count,
            original_tails: network.tails().into(),
            original_heads: network.heads().into(),
            routing_topology: RoutingTopology::new(
                edge_count,
                routing_tails,
                routing_heads,
                routing_x,
                routing_y,
            ),
            transitions: transitions.into_boxed_slice(),
            transition_offsets: transition_offsets.into_boxed_slice(),
            outgoing_offsets: outgoing_offsets.into_boxed_slice(),
            outgoing_edges: outgoing_edges.into_boxed_slice(),
            incoming_offsets: incoming_offsets.into_boxed_slice(),
            incoming_edges: incoming_edges.into_boxed_slice(),
            initial_weights: initial_weights.into_boxed_slice(),
            lower_bounds: lower_bounds.into_boxed_slice(),
            upper_bounds: upper_bounds.into_boxed_slice(),
            topology_id,
        })
    }

    /// Number of learned transition coordinates.
    pub fn coordinate_count(&self) -> usize {
        self.transitions.len()
    }

    /// Stable meaning of each learned coordinate.
    pub fn transitions(&self) -> &[Transition] {
        &self.transitions
    }

    /// Initial direct coordinate weights under frozen v1 semantics.
    pub fn initial_weights(&self) -> &[f64] {
        &self.initial_weights
    }

    /// Coordinate-wise lower projection bounds.
    pub fn lower_bounds(&self) -> &[f64] {
        &self.lower_bounds
    }

    /// Coordinate-wise upper projection bounds.
    pub fn upper_bounds(&self) -> &[f64] {
        &self.upper_bounds
    }

    /// Identity of the original topology and transition coordinate order.
    ///
    /// It intentionally excludes baseline weights, geometry, optimizer bounds,
    /// and every shortest-path-backend artifact (including a CCH order).
    pub fn topology_id(&self) -> &TopologyId {
        &self.topology_id
    }

    /// Backend-independent routing arrays for an oracle implementation.
    pub fn routing_topology(&self) -> &RoutingTopology {
        &self.routing_topology
    }

    /// Find the stable coordinate of one legal consecutive-edge transition.
    pub fn transition_id(&self, previous: EdgeId, next: EdgeId) -> Option<usize> {
        let start = *self.transition_offsets.get(previous.index())?;
        let end = *self.transition_offsets.get(previous.index() + 1)?;
        let relative = self.transitions[start..end]
            .binary_search_by_key(&next, |transition| transition.next)
            .ok()?;
        Some(start + relative)
    }

    /// Validate and map one complete original-edge observation.
    pub(crate) fn map_trajectory(
        &self,
        trajectory: &Trajectory,
    ) -> Result<MappedTrajectory, LineGraphError> {
        let (source, target) = self.validate_trajectory(trajectory)?;
        let coordinates = trajectory
            .edges()
            .windows(2)
            .map(|pair| {
                self.transition_id(pair[0], pair[1])
                    .ok_or(LineGraphError::MissingTransition {
                        previous: pair[0],
                        next: pair[1],
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        debug_assert_eq!(coordinates.len() + 1, trajectory.edges().len());
        Ok(MappedTrajectory {
            source,
            target,
            coordinates: coordinates.into_boxed_slice(),
        })
    }

    /// Validate and map all accepted observations in input order.
    pub(crate) fn map_trajectories(
        &self,
        trajectories: &[Trajectory],
    ) -> Result<Vec<MappedTrajectory>, LineGraphError> {
        trajectories
            .iter()
            .map(|trajectory| self.map_trajectory(trajectory))
            .collect()
    }

    /// Decode a nonempty connected coordinate sequence to original road edges.
    pub fn decode_coordinates(&self, coordinates: &[usize]) -> Result<Vec<EdgeId>, LineGraphError> {
        let Some((&first_coordinate, remaining)) = coordinates.split_first() else {
            return Err(LineGraphError::EmptyCoordinatePath);
        };
        let first = self
            .transitions
            .get(first_coordinate)
            .copied()
            .ok_or(LineGraphError::CoordinateOutOfBounds(first_coordinate))?;
        let mut decoded = vec![first.previous, first.next];
        let mut expected_previous = first.next;

        for &coordinate in remaining {
            let transition = self
                .transitions
                .get(coordinate)
                .copied()
                .ok_or(LineGraphError::CoordinateOutOfBounds(coordinate))?;
            if transition.previous != expected_previous {
                return Err(LineGraphError::DisconnectedCoordinates {
                    expected_previous,
                    actual_previous: transition.previous,
                });
            }
            decoded.push(transition.next);
            expected_previous = transition.next;
        }
        Ok(decoded)
    }

    /// Count every observed transition coordinate, including repetitions.
    pub(crate) fn observed_counts(
        &self,
        trajectories: &[MappedTrajectory],
    ) -> Result<Vec<u64>, LineGraphError> {
        let mut counts = vec![0u64; self.coordinate_count()];
        for trajectory in trajectories {
            for &coordinate in &trajectory.coordinates {
                let count = counts
                    .get_mut(coordinate)
                    .ok_or(LineGraphError::CoordinateOutOfBounds(coordinate))?;
                *count = count
                    .checked_add(1)
                    .ok_or(LineGraphError::ObservedCountOverflow { coordinate })?;
            }
        }
        Ok(counts)
    }

    /// Aggregate accepted observations by original-node OD in stable order.
    pub(crate) fn group_queries(
        &self,
        trajectories: &[MappedTrajectory],
    ) -> Result<Vec<QueryGroup>, LineGraphError> {
        let mut groups = BTreeMap::<(NodeId, NodeId), u64>::new();
        for trajectory in trajectories {
            let key = (trajectory.source, trajectory.target);
            let count = groups.entry(key).or_default();
            *count = count
                .checked_add(1)
                .ok_or(LineGraphError::QueryGroupCountOverflow {
                    source: key.0,
                    target: key.1,
                })?;
        }
        Ok(groups
            .into_iter()
            .map(|((source, target), sample_count)| QueryGroup {
                source,
                target,
                sample_count,
            })
            .collect())
    }

    /// Map an original-node query to zero-offset line-graph edge states.
    ///
    /// Every road edge leaving `source` is a source state and every road edge
    /// entering `target` is a target state. Thus the first edge carries no
    /// learned cost, and a one-edge route may have zero transition cost.
    pub fn node_query_endpoints(
        &self,
        source: NodeId,
        target: NodeId,
    ) -> Result<(Vec<QueryEndpoint>, Vec<QueryEndpoint>), LineGraphError> {
        if source.index() >= self.original_node_count {
            return Err(LineGraphError::QueryNodeOutOfBounds {
                node: source,
                node_count: self.original_node_count,
            });
        }
        if target.index() >= self.original_node_count {
            return Err(LineGraphError::QueryNodeOutOfBounds {
                node: target,
                node_count: self.original_node_count,
            });
        }

        let sources = incidence_slice(&self.outgoing_offsets, &self.outgoing_edges, source.index())
            .iter()
            .copied()
            .map(QueryEndpoint::zero)
            .collect::<Vec<_>>();
        let targets = incidence_slice(&self.incoming_offsets, &self.incoming_edges, target.index())
            .iter()
            .copied()
            .map(QueryEndpoint::zero)
            .collect::<Vec<_>>();
        if sources.is_empty() || targets.is_empty() {
            return Err(LineGraphError::EmptyQueryEndpoints {
                source,
                target,
                source_states: sources.len(),
                target_states: targets.len(),
            });
        }
        Ok((sources, targets))
    }

    fn validate_trajectory(
        &self,
        trajectory: &Trajectory,
    ) -> Result<(NodeId, NodeId), LineGraphError> {
        validate_trajectory_arrays(&self.original_tails, &self.original_heads, trajectory)
            .map_err(Into::into)
    }
}

/// Invalid line-graph construction, mapping, or aggregation input.
#[derive(Clone, Debug, PartialEq)]
pub enum LineGraphError {
    InvalidBoundFactors {
        lower: f64,
        upper: f64,
    },
    InvalidBound {
        coordinate: usize,
        kind: BoundKind,
        value: f64,
    },
    UpperBoundReachesRoutingInfinity {
        coordinate: usize,
        value: f64,
        rounded: f64,
    },
    TransitionCountOverflow,
    NoTransitions,
    TooManyTransitions {
        count: usize,
        maximum: usize,
    },
    InvalidTrajectory(NetworkError),
    MissingTransition {
        previous: EdgeId,
        next: EdgeId,
    },
    EmptyCoordinatePath,
    CoordinateOutOfBounds(usize),
    DisconnectedCoordinates {
        expected_previous: EdgeId,
        actual_previous: EdgeId,
    },
    ObservedCountOverflow {
        coordinate: usize,
    },
    QueryGroupCountOverflow {
        source: NodeId,
        target: NodeId,
    },
    QueryNodeOutOfBounds {
        node: NodeId,
        node_count: usize,
    },
    EmptyQueryEndpoints {
        source: NodeId,
        target: NodeId,
        source_states: usize,
        target_states: usize,
    },
    IncidenceDegreeOverflow {
        node: NodeId,
    },
    IncidenceOffsetOverflow {
        node: NodeId,
    },
}

impl From<NetworkError> for LineGraphError {
    fn from(error: NetworkError) -> Self {
        Self::InvalidTrajectory(error)
    }
}

impl Display for LineGraphError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for LineGraphError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BoundKind {
    Lower,
    Upper,
}

fn validate_bound_factors(lower: f64, upper: f64) -> Result<(), LineGraphError> {
    if !lower.is_finite()
        || !upper.is_finite()
        || lower <= 0.0
        || lower > 1.0
        || upper < 1.0
        || upper < lower
    {
        return Err(LineGraphError::InvalidBoundFactors { lower, upper });
    }
    Ok(())
}

fn scaled_bounds(
    weights: &[f64],
    factor: f64,
    kind: BoundKind,
) -> Result<Vec<f64>, LineGraphError> {
    weights
        .iter()
        .enumerate()
        .map(|(coordinate, &weight)| {
            let value = factor * weight;
            if !value.is_finite() || value <= 0.0 {
                return Err(LineGraphError::InvalidBound {
                    coordinate,
                    kind,
                    value,
                });
            }
            Ok(value)
        })
        .collect()
}

fn validate_routing_upper_bounds(upper_bounds: &[f64]) -> Result<(), LineGraphError> {
    for (coordinate, &value) in upper_bounds.iter().enumerate() {
        let rounded = value.round().max(1.0);
        if rounded >= f64::from(ROUTING_INFINITY) {
            return Err(LineGraphError::UpperBoundReachesRoutingInfinity {
                coordinate,
                value,
                rounded,
            });
        }
    }
    Ok(())
}

fn build_incidence(
    node_count: usize,
    endpoints: &[NodeId],
) -> Result<(Vec<usize>, Vec<EdgeId>), LineGraphError> {
    let mut offsets = vec![0usize; node_count + 1];
    for &node in endpoints {
        offsets[node.index() + 1] = offsets[node.index() + 1]
            .checked_add(1)
            .ok_or(LineGraphError::IncidenceDegreeOverflow { node })?;
    }
    for node in 1..offsets.len() {
        offsets[node] = offsets[node].checked_add(offsets[node - 1]).ok_or(
            LineGraphError::IncidenceOffsetOverflow {
                node: NodeId::new((node - 1) as u32),
            },
        )?;
    }
    let mut cursor = offsets[..node_count].to_vec();
    let mut values = vec![EdgeId::new(0); endpoints.len()];
    for (edge_index, &node) in endpoints.iter().enumerate() {
        let position = &mut cursor[node.index()];
        values[*position] = EdgeId::new(edge_index as u32);
        *position += 1;
    }
    Ok((offsets, values))
}

fn incidence_slice<'a>(offsets: &[usize], values: &'a [EdgeId], node: usize) -> &'a [EdgeId] {
    &values[offsets[node]..offsets[node + 1]]
}

fn topology_identity(network: &RoadNetwork, transitions: &[Transition]) -> TopologyId {
    let mut hash = 0xcbf29ce484222325u64;
    hash_bytes(&mut hash, b"ewr-directed-line-graph-v1");
    hash_u64(&mut hash, network.node_count() as u64);
    hash_edge_endpoints(&mut hash, network.tails());
    hash_edge_endpoints(&mut hash, network.heads());
    hash_u64(&mut hash, transitions.len() as u64);
    for transition in transitions {
        hash_u64(&mut hash, transition.previous.index() as u64);
        hash_u64(&mut hash, transition.next.index() as u64);
    }
    TopologyId::new(format!("line-graph-v1:fnv1a64:{hash:016x}"))
        .expect("the fixed line-graph topology ID prefix is nonempty")
}

fn routing_topology_fingerprint(
    node_count: usize,
    tails: &[EdgeId],
    heads: &[EdgeId],
    x: &[f32],
    y: &[f32],
) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    hash_bytes(&mut hash, b"ewr-routing-topology-v1");
    hash_u64(&mut hash, node_count as u64);
    hash_edge_ids(&mut hash, tails);
    hash_edge_ids(&mut hash, heads);
    hash_f32s(&mut hash, x);
    hash_f32s(&mut hash, y);
    hash
}

fn hash_edge_ids(hash: &mut u64, edges: &[EdgeId]) {
    hash_u64(hash, edges.len() as u64);
    for edge in edges {
        hash_u64(hash, edge.index() as u64);
    }
}

fn hash_f32s(hash: &mut u64, values: &[f32]) {
    hash_u64(hash, values.len() as u64);
    for value in values {
        hash_bytes(hash, &value.to_bits().to_le_bytes());
    }
}

fn hash_edge_endpoints(hash: &mut u64, endpoints: &[NodeId]) {
    hash_u64(hash, endpoints.len() as u64);
    for endpoint in endpoints {
        hash_u64(hash, endpoint.index() as u64);
    }
}

fn hash_u64(hash: &mut u64, value: u64) {
    hash_bytes(hash, &value.to_le_bytes());
}

fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(0x100000001b3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn network() -> RoadNetwork {
        // Two complete 0->3 routes: [0,1,2] and [0,3,4].
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

    #[test]
    fn v1_coordinates_are_stable_and_entered_edge_weighted() {
        let graph = LineGraph::build(&network(), 0.5, 2.0).unwrap();
        let expected = [
            Transition {
                previous: EdgeId::new(0),
                next: EdgeId::new(1),
            },
            Transition {
                previous: EdgeId::new(0),
                next: EdgeId::new(3),
            },
            Transition {
                previous: EdgeId::new(1),
                next: EdgeId::new(2),
            },
            Transition {
                previous: EdgeId::new(3),
                next: EdgeId::new(4),
            },
        ];

        assert_eq!(graph.transitions(), expected);
        assert_eq!(
            graph.routing_topology().tails(),
            &[
                EdgeId::new(0),
                EdgeId::new(0),
                EdgeId::new(1),
                EdgeId::new(3),
            ]
        );
        assert_eq!(
            graph.routing_topology().heads(),
            &[
                EdgeId::new(1),
                EdgeId::new(3),
                EdgeId::new(2),
                EdgeId::new(4),
            ]
        );
        assert_eq!(graph.initial_weights(), &[3.0, 1.0, 5.0, 1.0]);
        assert_eq!(graph.lower_bounds(), &[1.5, 0.5, 2.5, 0.5]);
        assert_eq!(graph.upper_bounds(), &[6.0, 2.0, 10.0, 2.0]);
    }

    #[test]
    fn mapping_has_l_minus_one_coordinates_and_round_trips() {
        let graph = LineGraph::build(&network(), 0.5, 2.0).unwrap();
        let observed = Trajectory::new(vec![EdgeId::new(0), EdgeId::new(1), EdgeId::new(2)]);
        let alternative = Trajectory::new(vec![EdgeId::new(0), EdgeId::new(3), EdgeId::new(4)]);
        let mapped = graph
            .map_trajectories(&[observed.clone(), observed, alternative])
            .unwrap();

        assert_eq!(mapped[0].coordinates.as_ref(), &[0, 2]);
        assert_eq!(mapped[0].coordinates.len(), 3 - 1);
        assert_eq!(
            graph.decode_coordinates(&mapped[0].coordinates).unwrap(),
            &[EdgeId::new(0), EdgeId::new(1), EdgeId::new(2)]
        );
        assert_eq!(graph.observed_counts(&mapped).unwrap(), vec![2, 1, 2, 1]);
        assert_eq!(
            graph.group_queries(&mapped).unwrap(),
            vec![QueryGroup {
                source: NodeId::new(0),
                target: NodeId::new(3),
                sample_count: 3,
            }]
        );
    }

    #[test]
    fn node_endpoints_freeze_zero_cost_first_edge_semantics() {
        let graph = LineGraph::build(&network(), 0.5, 2.0).unwrap();
        assert_eq!(
            graph
                .node_query_endpoints(NodeId::new(0), NodeId::new(3))
                .unwrap(),
            (
                vec![QueryEndpoint::zero(EdgeId::new(0))],
                vec![
                    QueryEndpoint::zero(EdgeId::new(2)),
                    QueryEndpoint::zero(EdgeId::new(4)),
                ],
            )
        );

        let (single_edge_sources, single_edge_targets) = graph
            .node_query_endpoints(NodeId::new(0), NodeId::new(1))
            .unwrap();
        assert_eq!(single_edge_sources, single_edge_targets);
        assert_eq!(single_edge_sources[0].node(), EdgeId::new(0));
        assert_eq!(single_edge_sources[0].offset(), 0);
    }

    #[test]
    fn topology_identity_excludes_weights_geometry_bounds_and_backends() {
        let first = LineGraph::build(&network(), 0.5, 2.0).unwrap();
        let same_topology = RoadNetwork::new(
            network().tails().to_vec(),
            network().heads().to_vec(),
            vec![20.0, 30.0, 50.0, 10.0, 10.0],
            vec![10.0, 11.0, 12.0, 13.0, 12.0],
            vec![7.0, 7.0, 7.0, 7.0, 8.0],
        )
        .unwrap();
        let second = LineGraph::build(&same_topology, 0.25, 4.0).unwrap();

        assert_eq!(first.topology_id(), second.topology_id());
        assert_ne!(
            first.routing_topology().fingerprint(),
            second.routing_topology().fingerprint(),
            "routing preprocessing identity must include geometry"
        );
        assert!(first.topology_id().as_str().starts_with("line-graph-v1:"));
    }

    #[test]
    fn rejects_an_upper_box_that_can_reach_the_routing_sentinel() {
        let network = RoadNetwork::new(
            vec![NodeId::new(0), NodeId::new(1)],
            vec![NodeId::new(1), NodeId::new(2)],
            vec![1.0, f64::from(ROUTING_INFINITY)],
            vec![0.0, 1.0, 2.0],
            vec![0.0, 0.0, 0.0],
        )
        .unwrap();

        assert!(matches!(
            LineGraph::build(&network, 1.0, 1.0),
            Err(LineGraphError::UpperBoundReachesRoutingInfinity { coordinate: 0, .. })
        ));
    }
}
