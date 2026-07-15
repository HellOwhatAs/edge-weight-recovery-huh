//! Route-local diagnostics for the first point where an observed path and the
//! learned shortest path diverge.
//!
//! These statistics are descriptive. In particular, concentration at complex
//! junctions or a turn-category imbalance can motivate a turn-aware model, but
//! does not by itself establish that turn costs cause the remaining error.

use crate::graph::{GraphData, TripPath};
use crate::turn::turn_geometry;
pub use crate::turn::{TurnCategory, TurnGeometry};
use rayon::prelude::*;
use routingkit_cch::{CCHMetric, CCHQuery};
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone, Debug, PartialEq)]
pub struct RejoinEvent {
    pub node: u32,
    pub observed_node_index: usize,
    pub predicted_node_index: usize,
}

impl RejoinEvent {
    fn to_json(&self) -> Value {
        json!({
            "node": self.node,
            "observed_node_index": self.observed_node_index,
            "predicted_node_index": self.predicted_node_index,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FirstDivergence {
    /// Zero-based edge position. This is also the number of common prefix
    /// edges before the first mismatch.
    pub edge_index: usize,
    /// `edge_index / observed_path_length`.
    pub observed_path_fraction: f64,
    pub node: u32,
    pub node_indegree: usize,
    pub node_outdegree: usize,
    pub is_complex_junction: bool,
    pub incoming_edge: Option<usize>,
    pub observed_next_edge: Option<usize>,
    pub predicted_next_edge: Option<usize>,
    pub observed_turn: Option<TurnGeometry>,
    pub predicted_turn: Option<TurnGeometry>,
    pub observed_suffix_cost: u128,
    pub predicted_suffix_cost: u128,
    /// Non-negative for an exact shortest-path oracle because the two suffixes
    /// start at the same divergence node and end at the same destination.
    pub observed_minus_predicted_suffix_cost: u128,
    pub relative_suffix_cost_gap: f64,
    /// First shared downstream node before the common target. The destination
    /// is excluded so that every erroneous route is not trivially a rejoin.
    pub rejoin_before_target: Option<RejoinEvent>,
}

impl FirstDivergence {
    fn to_json(&self) -> Value {
        json!({
            "edge_index": self.edge_index,
            "observed_path_fraction": self.observed_path_fraction,
            "node": self.node,
            "node_indegree": self.node_indegree,
            "node_outdegree": self.node_outdegree,
            "is_complex_junction": self.is_complex_junction,
            "incoming_edge": self.incoming_edge,
            "observed_next_edge": self.observed_next_edge,
            "predicted_next_edge": self.predicted_next_edge,
            "observed_turn": self.observed_turn.map(TurnGeometry::to_json),
            "predicted_turn": self.predicted_turn.map(TurnGeometry::to_json),
            "observed_suffix_cost": self.observed_suffix_cost,
            "predicted_suffix_cost": self.predicted_suffix_cost,
            "observed_minus_predicted_suffix_cost":
                self.observed_minus_predicted_suffix_cost,
            "relative_suffix_cost_gap": self.relative_suffix_cost_gap,
            "rejoin_before_target": self.rejoin_before_target.as_ref().map(RejoinEvent::to_json),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RouteDivergence {
    pub route_index: usize,
    pub source: u32,
    pub target: u32,
    pub observed_path_length: usize,
    pub predicted_path_length: usize,
    pub exact_match: bool,
    pub edge_f1: f64,
    pub contains_unseen_train_edge: bool,
    /// Number of internal observed-path nodes with indegree and outdegree at
    /// least two. Source and destination are not counted.
    pub complex_junction_count: usize,
    pub observed_left_turn_count: usize,
    pub observed_classifiable_turn_count: usize,
    pub common_prefix_edges: usize,
    /// Common prefix edges divided by observed path length. Exact matches are
    /// one; an immediate source-edge mismatch is zero.
    pub prefix_match_ratio: f64,
    pub first_choice_correct: bool,
    pub first_divergence: Option<FirstDivergence>,
}

impl RouteDivergence {
    pub fn to_json(&self) -> Value {
        json!({
            "route_index": self.route_index,
            "source": self.source,
            "target": self.target,
            "observed_path_length": self.observed_path_length,
            "predicted_path_length": self.predicted_path_length,
            "exact_match": self.exact_match,
            "edge_f1": self.edge_f1,
            "contains_unseen_train_edge": self.contains_unseen_train_edge,
            "complex_junction_count": self.complex_junction_count,
            "observed_left_turn_count": self.observed_left_turn_count,
            "observed_classifiable_turn_count": self.observed_classifiable_turn_count,
            "common_prefix_edges": self.common_prefix_edges,
            "prefix_match_ratio": self.prefix_match_ratio,
            "first_choice_correct": self.first_choice_correct,
            "first_divergence": self.first_divergence.as_ref().map(FirstDivergence::to_json),
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DistributionSummary {
    pub mean: f64,
    pub median: f64,
    pub p90: f64,
}

impl DistributionSummary {
    fn to_json(self) -> Value {
        json!({
            "mean": self.mean,
            "median": self.median,
            "p90": self.p90,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConditionalDivergenceSummary {
    pub sample_count: usize,
    pub first_divergence_index: DistributionSummary,
    pub first_divergence_observed_path_fraction: DistributionSummary,
    pub first_divergence_index_histogram: BTreeMap<usize, usize>,
    pub rejoin_before_target_rate: f64,
    pub complex_junction_rate: f64,
    pub divergence_node_indegree: DistributionSummary,
    pub divergence_node_outdegree: DistributionSummary,
    pub observed_minus_predicted_suffix_cost: DistributionSummary,
    pub relative_suffix_cost_gap: DistributionSummary,
    pub observed_turn_available_rate: f64,
    pub predicted_turn_available_rate: f64,
    pub turn_category_match_rate_when_both_available: Option<f64>,
    pub observed_turn_category_counts: BTreeMap<TurnCategory, usize>,
    pub predicted_turn_category_counts: BTreeMap<TurnCategory, usize>,
    pub turn_category_confusion_counts: BTreeMap<String, usize>,
}

impl ConditionalDivergenceSummary {
    fn to_json(&self) -> Value {
        json!({
            "sample_count": self.sample_count,
            "first_divergence_index": self.first_divergence_index.to_json(),
            "first_divergence_observed_path_fraction":
                self.first_divergence_observed_path_fraction.to_json(),
            "first_divergence_index_histogram": usize_map_json(
                &self.first_divergence_index_histogram
            ),
            "rejoin_before_target_rate": self.rejoin_before_target_rate,
            "complex_junction_rate": self.complex_junction_rate,
            "divergence_node_indegree": self.divergence_node_indegree.to_json(),
            "divergence_node_outdegree": self.divergence_node_outdegree.to_json(),
            "observed_minus_predicted_suffix_cost":
                self.observed_minus_predicted_suffix_cost.to_json(),
            "relative_suffix_cost_gap": self.relative_suffix_cost_gap.to_json(),
            "observed_turn_available_rate": self.observed_turn_available_rate,
            "predicted_turn_available_rate": self.predicted_turn_available_rate,
            "turn_category_match_rate_when_both_available":
                self.turn_category_match_rate_when_both_available,
            "observed_turn_category_counts": turn_count_json(
                &self.observed_turn_category_counts
            ),
            "predicted_turn_category_counts": turn_count_json(
                &self.predicted_turn_category_counts
            ),
            "turn_category_confusion_counts": self.turn_category_confusion_counts,
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DivergenceSummary {
    pub sample_count: usize,
    pub exact_match_rate: f64,
    pub divergence_rate: f64,
    pub first_choice_accuracy: f64,
    pub mean_prefix_match_ratio: f64,
    pub mean_edge_f1: f64,
    pub mean_observed_path_length: f64,
    pub unseen_train_edge_route_rate: f64,
    pub mean_complex_junction_count: f64,
    pub mean_observed_left_turn_count: f64,
    pub divergent_routes: Option<ConditionalDivergenceSummary>,
}

impl DivergenceSummary {
    pub fn to_json(&self) -> Value {
        json!({
            "sample_count": self.sample_count,
            "exact_match_rate": self.exact_match_rate,
            "divergence_rate": self.divergence_rate,
            "first_choice_accuracy": self.first_choice_accuracy,
            "mean_prefix_match_ratio": self.mean_prefix_match_ratio,
            "mean_edge_f1": self.mean_edge_f1,
            "mean_observed_path_length": self.mean_observed_path_length,
            "unseen_train_edge_route_rate": self.unseen_train_edge_route_rate,
            "mean_complex_junction_count": self.mean_complex_junction_count,
            "mean_observed_left_turn_count": self.mean_observed_left_turn_count,
            "divergent_routes": self.divergent_routes.as_ref().map(
                ConditionalDivergenceSummary::to_json
            ),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NamedStratum {
    pub label: String,
    pub criteria: Value,
    pub summary: DivergenceSummary,
}

impl NamedStratum {
    fn to_json(&self) -> Value {
        json!({
            "label": self.label,
            "criteria": self.criteria,
            "summary": self.summary.to_json(),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DivergenceAnalysis {
    pub routes: Vec<RouteDivergence>,
    pub overall: DivergenceSummary,
    pub route_length_quartile_cutoffs: [f64; 3],
    pub route_length_strata: Vec<NamedStratum>,
    pub train_edge_visibility_strata: Vec<NamedStratum>,
    pub complex_junction_count_strata: Vec<NamedStratum>,
    /// Joint table to inspect junction-complexity effects without relying only
    /// on the strong marginal association between route length and complexity.
    pub route_length_by_complex_junction_strata: Vec<NamedStratum>,
}

impl DivergenceAnalysis {
    /// Convert to a stable machine-readable representation. Large per-route
    /// records can be omitted from a compact summary file and written to a
    /// second artifact with `include_routes = true`.
    pub fn to_json(&self, include_routes: bool) -> Value {
        let mut object = Map::from_iter([
            (
                "definitions".to_string(),
                json!({
                    "first_divergence_index":
                        "zero-based edge index and number of common prefix edges",
                    "divergence_rate": "fraction of routes whose complete edge sequence differs",
                    "first_choice_accuracy":
                        "fraction whose first predicted edge equals the first observed edge",
                    "prefix_match_ratio":
                        "common prefix edge count divided by observed path edge count",
                    "complex_junction": "directed edge indegree >= 2 and outdegree >= 2",
                    "rejoin_before_target":
                        "a shared downstream node after divergence, excluding the common target",
                    "suffix_cost_gap":
                        "observed suffix cost minus predicted suffix cost under checkpoint weights",
                    "turn_angle": {
                        "method": "planar atan2(cross, dot) using graph x/y coordinates",
                        "sign": "positive is counter-clockwise (left)",
                        "straight_absolute_degrees_at_most": 30.0,
                        "uturn_absolute_degrees_at_least": 150.0,
                        "unavailable_when":
                            "incoming/outgoing edges are discontinuous or either coordinate vector is degenerate",
                    },
                    "interpretation_warning":
                        "These are descriptive associations and do not establish that turn costs cause route errors.",
                }),
            ),
            ("overall".to_string(), self.overall.to_json()),
            (
                "route_length_quartile_cutoffs".to_string(),
                json!(self.route_length_quartile_cutoffs),
            ),
            (
                "strata".to_string(),
                json!({
                    "route_length": strata_json(&self.route_length_strata),
                    "train_edge_visibility": strata_json(&self.train_edge_visibility_strata),
                    "complex_junction_count": strata_json(&self.complex_junction_count_strata),
                    "route_length_by_complex_junction_count":
                        strata_json(&self.route_length_by_complex_junction_strata),
                }),
            ),
        ]);
        if include_routes {
            object.insert(
                "routes".to_string(),
                Value::Array(self.routes.iter().map(RouteDivergence::to_json).collect()),
            );
        }
        Value::Object(object)
    }
}

/// Compare each observed path with the exact learned shortest path and build
/// first-divergence diagnostics. This function performs no training and does
/// not load any split itself.
pub fn analyze_first_divergence(
    metric: &CCHMetric<'_>,
    graph: &GraphData,
    paths: &[TripPath],
    train_observed_edge_counts: &[u64],
    num_chunks: usize,
) -> Result<DivergenceAnalysis, String> {
    validate_inputs(metric, graph, train_observed_edge_counts)?;
    let (indegree, outdegree) = directed_degrees(graph)?;
    let chunk_size = paths.len().div_ceil(num_chunks.max(1)).max(1);
    let chunks: Vec<Result<Vec<RouteDivergence>, String>> = paths
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, chunk)| {
            let mut query = CCHQuery::new(metric);
            chunk
                .iter()
                .enumerate()
                .map(|(index_in_chunk, trip)| {
                    analyze_route(
                        &mut query,
                        metric,
                        graph,
                        &indegree,
                        &outdegree,
                        train_observed_edge_counts,
                        trip,
                        chunk_index * chunk_size + index_in_chunk,
                    )
                })
                .collect()
        })
        .collect();

    let mut routes = Vec::with_capacity(paths.len());
    for chunk in chunks {
        routes.extend(chunk?);
    }
    build_analysis(routes)
}

fn validate_inputs(
    metric: &CCHMetric<'_>,
    graph: &GraphData,
    train_counts: &[u64],
) -> Result<(), String> {
    let edge_count = graph.tail.len();
    if graph.head.len() != edge_count
        || graph.baseline_weights.len() != edge_count
        || metric.weights().len() != edge_count
        || train_counts.len() != edge_count
    {
        return Err(format!(
            "edge-array length mismatch: tail={edge_count}, head={}, baseline={}, metric={}, train_counts={}",
            graph.head.len(),
            graph.baseline_weights.len(),
            metric.weights().len(),
            train_counts.len(),
        ));
    }
    if graph.x.len() != graph.y.len() {
        return Err(format!(
            "coordinate-array length mismatch: x={}, y={}",
            graph.x.len(),
            graph.y.len()
        ));
    }
    Ok(())
}

fn directed_degrees(graph: &GraphData) -> Result<(Vec<usize>, Vec<usize>), String> {
    let node_count = graph.x.len();
    let mut indegree = vec![0usize; node_count];
    let mut outdegree = vec![0usize; node_count];
    for (edge, (&tail, &head)) in graph.tail.iter().zip(&graph.head).enumerate() {
        let tail = tail as usize;
        let head = head as usize;
        if tail >= node_count || head >= node_count {
            return Err(format!(
                "edge {edge} endpoint out of bounds for {node_count} nodes: {tail}->{head}"
            ));
        }
        outdegree[tail] = outdegree[tail]
            .checked_add(1)
            .ok_or_else(|| "node outdegree overflow".to_string())?;
        indegree[head] = indegree[head]
            .checked_add(1)
            .ok_or_else(|| "node indegree overflow".to_string())?;
    }
    Ok((indegree, outdegree))
}

#[allow(clippy::too_many_arguments)]
fn analyze_route<'a>(
    query: &mut CCHQuery<'a>,
    metric: &'a CCHMetric<'a>,
    graph: &GraphData,
    indegree: &[usize],
    outdegree: &[usize],
    train_counts: &[u64],
    trip: &TripPath,
    route_index: usize,
) -> Result<RouteDivergence, String> {
    let ((source, target), observed_path) = trip;
    let observed_nodes = path_nodes(
        graph,
        observed_path,
        *source,
        *target,
        route_index,
        "observed",
    )?;
    query.add_source(*source, 0);
    query.add_target(*target, 0);
    let result = query.run();
    let shortest_distance = result
        .distance()
        .ok_or_else(|| format!("route {route_index} OD ({source}, {target}) is unreachable"))?
        as u128;
    let predicted_path: Vec<usize> = result
        .arc_path()
        .into_iter()
        .map(|edge| edge as usize)
        .collect();
    let predicted_nodes = path_nodes(
        graph,
        &predicted_path,
        *source,
        *target,
        route_index,
        "predicted",
    )?;
    let reconstructed_distance = path_cost(metric.weights(), &predicted_path)?;
    if reconstructed_distance != shortest_distance {
        return Err(format!(
            "route {route_index} CCH distance/path mismatch: {shortest_distance} != {reconstructed_distance}"
        ));
    }

    let common_prefix_edges = observed_path
        .iter()
        .zip(&predicted_path)
        .take_while(|(observed, predicted)| observed == predicted)
        .count();
    let exact_match = observed_path == &predicted_path;
    let first_choice_correct = observed_path.first() == predicted_path.first();
    let prefix_match_ratio = common_prefix_edges as f64 / observed_path.len() as f64;
    let contains_unseen_train_edge = observed_path
        .iter()
        .any(|&edge| train_counts.get(edge).copied().unwrap_or(0) == 0);
    let (complex_junction_count, left_turn_count, classifiable_turn_count) =
        observed_route_structure(graph, observed_path, indegree, outdegree);

    let first_divergence = if exact_match {
        None
    } else {
        let edge_index = common_prefix_edges;
        let node = observed_nodes[edge_index];
        if predicted_nodes.get(edge_index).copied() != Some(node) {
            return Err(format!(
                "route {route_index} paths do not share the divergence node at edge index {edge_index}"
            ));
        }
        let incoming_edge = edge_index
            .checked_sub(1)
            .and_then(|index| observed_path.get(index).copied());
        let observed_next_edge = observed_path.get(edge_index).copied();
        let predicted_next_edge = predicted_path.get(edge_index).copied();
        let observed_turn = incoming_edge
            .zip(observed_next_edge)
            .and_then(|(incoming, outgoing)| turn_geometry(graph, incoming, outgoing));
        let predicted_turn = incoming_edge
            .zip(predicted_next_edge)
            .and_then(|(incoming, outgoing)| turn_geometry(graph, incoming, outgoing));
        let observed_suffix_cost = path_cost(metric.weights(), &observed_path[edge_index..])?;
        let predicted_suffix_cost = path_cost(metric.weights(), &predicted_path[edge_index..])?;
        let observed_minus_predicted_suffix_cost = observed_suffix_cost
            .checked_sub(predicted_suffix_cost)
            .ok_or_else(|| {
                format!(
                    "route {route_index} has negative suffix regret at divergence: observed={observed_suffix_cost}, predicted={predicted_suffix_cost}"
                )
            })?;
        let node_index = node as usize;
        FirstDivergence {
            edge_index,
            observed_path_fraction: edge_index as f64 / observed_path.len() as f64,
            node,
            node_indegree: indegree[node_index],
            node_outdegree: outdegree[node_index],
            is_complex_junction: is_complex(node_index, indegree, outdegree),
            incoming_edge,
            observed_next_edge,
            predicted_next_edge,
            observed_turn,
            predicted_turn,
            observed_suffix_cost,
            predicted_suffix_cost,
            observed_minus_predicted_suffix_cost,
            relative_suffix_cost_gap: if observed_suffix_cost == 0 {
                0.0
            } else {
                observed_minus_predicted_suffix_cost as f64 / observed_suffix_cost as f64
            },
            rejoin_before_target: first_rejoin_before_target(
                &observed_nodes,
                &predicted_nodes,
                edge_index,
            ),
        }
        .into()
    };

    Ok(RouteDivergence {
        route_index,
        source: *source,
        target: *target,
        observed_path_length: observed_path.len(),
        predicted_path_length: predicted_path.len(),
        exact_match,
        edge_f1: edge_f1(observed_path, &predicted_path),
        contains_unseen_train_edge,
        complex_junction_count,
        observed_left_turn_count: left_turn_count,
        observed_classifiable_turn_count: classifiable_turn_count,
        common_prefix_edges,
        prefix_match_ratio,
        first_choice_correct,
        first_divergence,
    })
}

fn path_nodes(
    graph: &GraphData,
    path: &[usize],
    source: u32,
    target: u32,
    route_index: usize,
    label: &str,
) -> Result<Vec<u32>, String> {
    if path.is_empty() {
        return Err(format!("route {route_index} {label} path is empty"));
    }
    let first = *path.first().expect("checked nonempty path");
    if graph.tail.get(first).copied() != Some(source) {
        return Err(format!(
            "route {route_index} {label} path does not start at source {source}"
        ));
    }
    let mut nodes = Vec::with_capacity(path.len() + 1);
    nodes.push(source);
    for (position, &edge) in path.iter().enumerate() {
        let edge_tail =
            graph.tail.get(edge).copied().ok_or_else(|| {
                format!("route {route_index} {label} edge {edge} is out of bounds")
            })?;
        let edge_head = graph.head[edge];
        if nodes.last().copied() != Some(edge_tail) {
            return Err(format!(
                "route {route_index} {label} path is discontinuous at position {position}"
            ));
        }
        nodes.push(edge_head);
    }
    if nodes.last().copied() != Some(target) {
        return Err(format!(
            "route {route_index} {label} path does not end at target {target}"
        ));
    }
    Ok(nodes)
}

fn path_cost(weights: &[u32], path: &[usize]) -> Result<u128, String> {
    path.iter().try_fold(0u128, |sum, &edge| {
        let weight = weights
            .get(edge)
            .ok_or_else(|| format!("path edge {edge} is out of bounds"))?;
        sum.checked_add(*weight as u128)
            .ok_or_else(|| "path cost overflow".to_string())
    })
}

fn observed_route_structure(
    graph: &GraphData,
    path: &[usize],
    indegree: &[usize],
    outdegree: &[usize],
) -> (usize, usize, usize) {
    let mut complex = 0;
    let mut left = 0;
    let mut classifiable = 0;
    for pair in path.windows(2) {
        let junction = graph.head[pair[0]] as usize;
        complex += usize::from(is_complex(junction, indegree, outdegree));
        if let Some(turn) = turn_geometry(graph, pair[0], pair[1]) {
            classifiable += 1;
            left += usize::from(turn.category == TurnCategory::Left);
        }
    }
    (complex, left, classifiable)
}

fn is_complex(node: usize, indegree: &[usize], outdegree: &[usize]) -> bool {
    indegree.get(node).copied().unwrap_or(0) >= 2 && outdegree.get(node).copied().unwrap_or(0) >= 2
}

fn first_rejoin_before_target(
    observed_nodes: &[u32],
    predicted_nodes: &[u32],
    divergence_edge_index: usize,
) -> Option<RejoinEvent> {
    let predicted_positions: HashMap<u32, usize> = predicted_nodes
        .iter()
        .copied()
        .enumerate()
        .skip(divergence_edge_index + 1)
        .take(
            predicted_nodes
                .len()
                .saturating_sub(divergence_edge_index + 2),
        )
        .map(|(index, node)| (node, index))
        .collect();
    observed_nodes
        .iter()
        .copied()
        .enumerate()
        .skip(divergence_edge_index + 1)
        .take(
            observed_nodes
                .len()
                .saturating_sub(divergence_edge_index + 2),
        )
        .find_map(|(observed_node_index, node)| {
            predicted_positions
                .get(&node)
                .copied()
                .map(|predicted_node_index| RejoinEvent {
                    node,
                    observed_node_index,
                    predicted_node_index,
                })
        })
}

fn edge_f1(observed: &[usize], predicted: &[usize]) -> f64 {
    let observed: HashSet<usize> = observed.iter().copied().collect();
    let predicted: HashSet<usize> = predicted.iter().copied().collect();
    let intersection = observed.intersection(&predicted).count() as f64;
    let precision = intersection / predicted.len().max(1) as f64;
    let recall = intersection / observed.len().max(1) as f64;
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

fn build_analysis(routes: Vec<RouteDivergence>) -> Result<DivergenceAnalysis, String> {
    let all: Vec<&RouteDivergence> = routes.iter().collect();
    let overall = summarize(&all);
    let mut lengths: Vec<f64> = routes
        .iter()
        .map(|route| route.observed_path_length as f64)
        .collect();
    lengths.sort_by(f64::total_cmp);
    let cutoffs = [
        percentile(&lengths, 0.25),
        percentile(&lengths, 0.50),
        percentile(&lengths, 0.75),
    ];

    let mut length_members: [Vec<&RouteDivergence>; 4] = std::array::from_fn(|_| Vec::new());
    let mut complexity_members: [Vec<&RouteDivergence>; 4] = std::array::from_fn(|_| Vec::new());
    let mut joint_members: [[Vec<&RouteDivergence>; 4]; 4] =
        std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
    let mut seen = Vec::new();
    let mut unseen = Vec::new();
    for route in &routes {
        let length_bin = length_bin(route.observed_path_length, cutoffs);
        let complexity_bin = complexity_bin(route.complex_junction_count);
        length_members[length_bin].push(route);
        complexity_members[complexity_bin].push(route);
        joint_members[length_bin][complexity_bin].push(route);
        if route.contains_unseen_train_edge {
            unseen.push(route);
        } else {
            seen.push(route);
        }
    }

    let route_length_strata = (0..4)
        .map(|bin| NamedStratum {
            label: format!("q{}", bin + 1),
            criteria: json!({
                "lower_length_exclusive": (bin > 0).then(|| cutoffs[bin - 1]),
                "upper_length_inclusive": (bin < 3).then(|| cutoffs[bin]),
            }),
            summary: summarize(&length_members[bin]),
        })
        .collect();
    let train_edge_visibility_strata = vec![
        NamedStratum {
            label: "all_observed_edges_seen_in_train".to_string(),
            criteria: json!({"contains_unseen_train_edge": false}),
            summary: summarize(&seen),
        },
        NamedStratum {
            label: "contains_unseen_train_edge".to_string(),
            criteria: json!({"contains_unseen_train_edge": true}),
            summary: summarize(&unseen),
        },
    ];
    let complexity_labels = ["0", "1_to_2", "3_to_5", "6_or_more"];
    let complexity_criteria = [
        json!({"minimum_inclusive": 0, "maximum_inclusive": 0}),
        json!({"minimum_inclusive": 1, "maximum_inclusive": 2}),
        json!({"minimum_inclusive": 3, "maximum_inclusive": 5}),
        json!({"minimum_inclusive": 6, "maximum_inclusive": null}),
    ];
    let complex_junction_count_strata = (0..4)
        .map(|bin| NamedStratum {
            label: complexity_labels[bin].to_string(),
            criteria: complexity_criteria[bin].clone(),
            summary: summarize(&complexity_members[bin]),
        })
        .collect();
    let route_length_by_complex_junction_strata = (0..4)
        .flat_map(|length_bin| {
            let joint_members = &joint_members;
            let complexity_criteria = &complexity_criteria;
            (0..4).map(move |complexity_bin| NamedStratum {
                label: format!(
                    "q{}_complex_{}",
                    length_bin + 1,
                    complexity_labels[complexity_bin]
                ),
                criteria: json!({
                    "route_length_quartile": length_bin + 1,
                    "complex_junction_count": complexity_criteria[complexity_bin],
                }),
                summary: summarize(&joint_members[length_bin][complexity_bin]),
            })
        })
        .collect();

    Ok(DivergenceAnalysis {
        routes,
        overall,
        route_length_quartile_cutoffs: cutoffs,
        route_length_strata,
        train_edge_visibility_strata,
        complex_junction_count_strata,
        route_length_by_complex_junction_strata,
    })
}

fn length_bin(length: usize, cutoffs: [f64; 3]) -> usize {
    if length as f64 <= cutoffs[0] {
        0
    } else if length as f64 <= cutoffs[1] {
        1
    } else if length as f64 <= cutoffs[2] {
        2
    } else {
        3
    }
}

fn complexity_bin(count: usize) -> usize {
    match count {
        0 => 0,
        1..=2 => 1,
        3..=5 => 2,
        _ => 3,
    }
}

fn summarize(routes: &[&RouteDivergence]) -> DivergenceSummary {
    if routes.is_empty() {
        return DivergenceSummary::default();
    }
    let denominator = routes.len() as f64;
    let divergent: Vec<&FirstDivergence> = routes
        .iter()
        .filter_map(|route| route.first_divergence.as_ref())
        .collect();
    DivergenceSummary {
        sample_count: routes.len(),
        exact_match_rate: routes.iter().filter(|route| route.exact_match).count() as f64
            / denominator,
        divergence_rate: divergent.len() as f64 / denominator,
        first_choice_accuracy: routes
            .iter()
            .filter(|route| route.first_choice_correct)
            .count() as f64
            / denominator,
        mean_prefix_match_ratio: routes
            .iter()
            .map(|route| route.prefix_match_ratio)
            .sum::<f64>()
            / denominator,
        mean_edge_f1: routes.iter().map(|route| route.edge_f1).sum::<f64>() / denominator,
        mean_observed_path_length: routes
            .iter()
            .map(|route| route.observed_path_length as f64)
            .sum::<f64>()
            / denominator,
        unseen_train_edge_route_rate: routes
            .iter()
            .filter(|route| route.contains_unseen_train_edge)
            .count() as f64
            / denominator,
        mean_complex_junction_count: routes
            .iter()
            .map(|route| route.complex_junction_count as f64)
            .sum::<f64>()
            / denominator,
        mean_observed_left_turn_count: routes
            .iter()
            .map(|route| route.observed_left_turn_count as f64)
            .sum::<f64>()
            / denominator,
        divergent_routes: (!divergent.is_empty()).then(|| summarize_divergent(&divergent)),
    }
}

fn summarize_divergent(routes: &[&FirstDivergence]) -> ConditionalDivergenceSummary {
    let denominator = routes.len() as f64;
    let mut index_histogram = BTreeMap::new();
    let mut indices = Vec::with_capacity(routes.len());
    let mut fractions = Vec::with_capacity(routes.len());
    let mut indegrees = Vec::with_capacity(routes.len());
    let mut outdegrees = Vec::with_capacity(routes.len());
    let mut gaps = Vec::with_capacity(routes.len());
    let mut relative_gaps = Vec::with_capacity(routes.len());
    let mut observed_counts = BTreeMap::new();
    let mut predicted_counts = BTreeMap::new();
    let mut confusion = BTreeMap::new();
    let mut turn_pairs = 0usize;
    let mut turn_matches = 0usize;
    for route in routes {
        *index_histogram.entry(route.edge_index).or_default() += 1;
        indices.push(route.edge_index as f64);
        fractions.push(route.observed_path_fraction);
        indegrees.push(route.node_indegree as f64);
        outdegrees.push(route.node_outdegree as f64);
        gaps.push(route.observed_minus_predicted_suffix_cost as f64);
        relative_gaps.push(route.relative_suffix_cost_gap);
        if let Some(turn) = route.observed_turn {
            *observed_counts.entry(turn.category).or_default() += 1;
        }
        if let Some(turn) = route.predicted_turn {
            *predicted_counts.entry(turn.category).or_default() += 1;
        }
        if let (Some(observed), Some(predicted)) = (route.observed_turn, route.predicted_turn) {
            turn_pairs += 1;
            turn_matches += usize::from(observed.category == predicted.category);
            *confusion
                .entry(format!(
                    "{}->{}",
                    observed.category.as_str(),
                    predicted.category.as_str()
                ))
                .or_default() += 1;
        }
    }
    ConditionalDivergenceSummary {
        sample_count: routes.len(),
        first_divergence_index: distribution(indices),
        first_divergence_observed_path_fraction: distribution(fractions),
        first_divergence_index_histogram: index_histogram,
        rejoin_before_target_rate: routes
            .iter()
            .filter(|route| route.rejoin_before_target.is_some())
            .count() as f64
            / denominator,
        complex_junction_rate: routes
            .iter()
            .filter(|route| route.is_complex_junction)
            .count() as f64
            / denominator,
        divergence_node_indegree: distribution(indegrees),
        divergence_node_outdegree: distribution(outdegrees),
        observed_minus_predicted_suffix_cost: distribution(gaps),
        relative_suffix_cost_gap: distribution(relative_gaps),
        observed_turn_available_rate: routes
            .iter()
            .filter(|route| route.observed_turn.is_some())
            .count() as f64
            / denominator,
        predicted_turn_available_rate: routes
            .iter()
            .filter(|route| route.predicted_turn.is_some())
            .count() as f64
            / denominator,
        turn_category_match_rate_when_both_available: (turn_pairs > 0)
            .then(|| turn_matches as f64 / turn_pairs as f64),
        observed_turn_category_counts: observed_counts,
        predicted_turn_category_counts: predicted_counts,
        turn_category_confusion_counts: confusion,
    }
}

fn distribution(mut values: Vec<f64>) -> DistributionSummary {
    if values.is_empty() {
        return DistributionSummary::default();
    }
    values.sort_by(f64::total_cmp);
    DistributionSummary {
        mean: values.iter().sum::<f64>() / values.len() as f64,
        median: percentile(&values, 0.50),
        p90: percentile(&values, 0.90),
    }
}

/// Type-7/linear empirical quantile, matching the detailed route evaluator.
fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = quantile * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let fraction = rank - lower as f64;
    sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
}

fn turn_count_json(counts: &BTreeMap<TurnCategory, usize>) -> Value {
    Value::Object(
        counts
            .iter()
            .map(|(category, count)| (category.as_str().to_string(), json!(count)))
            .collect(),
    )
}

fn usize_map_json(counts: &BTreeMap<usize, usize>) -> Value {
    Value::Object(
        counts
            .iter()
            .map(|(key, count)| (key.to_string(), json!(count)))
            .collect(),
    )
}

fn strata_json(strata: &[NamedStratum]) -> Value {
    Value::Array(strata.iter().map(NamedStratum::to_json).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use routingkit_cch::{CCH, CCHMetric, compute_order_degree};

    fn small_graph() -> GraphData {
        // Observed: 0->1->2->4. Predicted: 0->1->3->2->4. The paths first
        // diverge at complex node 1 and rejoin at node 2 before the target.
        GraphData {
            tail: vec![0, 1, 2, 1, 3, 5],
            head: vec![1, 2, 4, 3, 2, 1],
            baseline_weights: vec![1, 3, 3, 1, 1, 100],
            x: vec![-1.0, 0.0, 1.0, 1.0, 2.0, -1.0],
            y: vec![0.0, 0.0, 1.0, 0.0, 1.0, -1.0],
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn reports_first_divergence_turn_rejoin_and_suffix_gap() {
        let graph = small_graph();
        let order = compute_order_degree(graph.x.len() as u32, &graph.tail, &graph.head);
        let cch = CCH::new(&order, &graph.tail, &graph.head, |_| {}, false);
        let metric = CCHMetric::new(&cch, graph.baseline_weights.clone());
        let paths = vec![((0, 4), vec![0, 1, 2])];
        let analysis =
            analyze_first_divergence(&metric, &graph, &paths, &[1, 1, 1, 0, 0, 0], 8).unwrap();
        let route = &analysis.routes[0];
        assert_eq!(route.common_prefix_edges, 1);
        assert_close(route.prefix_match_ratio, 1.0 / 3.0);
        assert!(route.first_choice_correct);
        assert_eq!(route.complex_junction_count, 1);
        assert_eq!(route.observed_left_turn_count, 1);

        let divergence = route.first_divergence.as_ref().unwrap();
        assert_eq!(divergence.node, 1);
        assert_eq!(divergence.node_indegree, 2);
        assert_eq!(divergence.node_outdegree, 2);
        assert!(divergence.is_complex_junction);
        assert_eq!(divergence.incoming_edge, Some(0));
        assert_eq!(divergence.observed_next_edge, Some(1));
        assert_eq!(divergence.predicted_next_edge, Some(3));
        assert_eq!(
            divergence.observed_turn.unwrap().category,
            TurnCategory::Left
        );
        assert_eq!(
            divergence.predicted_turn.unwrap().category,
            TurnCategory::Straight
        );
        assert_eq!(divergence.observed_suffix_cost, 6);
        assert_eq!(divergence.predicted_suffix_cost, 5);
        assert_eq!(divergence.observed_minus_predicted_suffix_cost, 1);
        assert_eq!(
            divergence
                .rejoin_before_target
                .as_ref()
                .map(|event| event.node),
            Some(2)
        );
        assert_close(analysis.overall.divergence_rate, 1.0);
        assert_close(
            analysis
                .overall
                .divergent_routes
                .as_ref()
                .unwrap()
                .rejoin_before_target_rate,
            1.0,
        );
    }

    #[test]
    fn classifies_planar_turns_and_rejects_degenerate_geometry() {
        let graph = small_graph();
        let left = turn_geometry(&graph, 0, 1).unwrap();
        let straight = turn_geometry(&graph, 0, 3).unwrap();
        assert_eq!(left.category, TurnCategory::Left);
        assert_close(left.signed_angle_degrees, 45.0);
        assert_eq!(straight.category, TurnCategory::Straight);
        assert_close(straight.signed_angle_degrees, 0.0);

        let mut degenerate = small_graph();
        degenerate.x[2] = degenerate.x[1];
        degenerate.y[2] = degenerate.y[1];
        assert_eq!(turn_geometry(&degenerate, 0, 1), None);
    }

    #[test]
    fn exact_path_has_no_divergence_and_full_prefix() {
        let graph = small_graph();
        let order = compute_order_degree(graph.x.len() as u32, &graph.tail, &graph.head);
        let cch = CCH::new(&order, &graph.tail, &graph.head, |_| {}, false);
        let metric = CCHMetric::new(&cch, graph.baseline_weights.clone());
        let paths = vec![((0, 4), vec![0, 3, 4, 2])];
        let analysis =
            analyze_first_divergence(&metric, &graph, &paths, &[1, 1, 1, 1, 1, 0], 1).unwrap();
        let route = &analysis.routes[0];
        assert!(route.exact_match);
        assert_eq!(route.first_divergence, None);
        assert_close(route.prefix_match_ratio, 1.0);
        assert_close(analysis.overall.first_choice_accuracy, 1.0);
        assert!(analysis.overall.divergent_routes.is_none());
    }
}
