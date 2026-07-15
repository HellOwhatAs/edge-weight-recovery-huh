//! Graph representation for inverse shortest-path learning.
//!
//! The optimizer-facing coordinate system is deliberately independent of the
//! CCH arc system. First-order coordinates are original directed roads.
//! Second-order coordinates are legal adjacent-road pairs `(e, f)`; overlap
//! arcs connect `(e, f)` to `(f, g)`. The CCH-specific conversion of a learned
//! node weight to overlap-arc weights and source offsets stays in this module.

use crate::data::{GraphData, TripPath};
use crate::oracle::cch::{CCH_INFINITY, CchMetric, CchReusableQuery, CchTopology};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

const MAX_SECOND_ORDER_NODES: usize = 12_000_000;
const MAX_SECOND_ORDER_ARCS: usize = 12_000_000;

type QueryEndpointsU32 = (Vec<(u32, u32)>, Vec<(u32, u32)>);
#[cfg(test)]
type QueryEndpointsF64 = (Vec<(u32, f64)>, Vec<(u32, f64)>);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GraphOrder {
    First,
    Second,
}

impl GraphOrder {
    /// Parse the stable configuration/checkpoint spelling of a graph order.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "first" => Ok(Self::First),
            "second" => Ok(Self::Second),
            _ => Err(format!(
                "unsupported graph order {value:?}; expected \"first\" or \"second\""
            )),
        }
    }

    /// Stable spelling used by configuration, logs, identities, and checkpoints.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::First => "first",
            Self::Second => "second",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MappedPath {
    pub source: u32,
    pub target: u32,
    pub coordinates: Vec<usize>,
    pub original_edges: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueryGroup {
    pub source: u32,
    pub target: u32,
    pub sample_count: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShortestPath {
    /// Quantized CCH distance.
    pub distance: u32,
    /// Cost of the returned coordinate path under the unquantized direct weights.
    pub direct_cost: f64,
    pub coordinates: Vec<usize>,
    pub original_edges: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OracleStats {
    pub predicted_counts: Vec<u64>,
    pub weighted_shortest_distance_sum: u128,
    pub weighted_direct_path_cost_sum: f64,
    pub sample_count: u64,
    pub num_queries: usize,
    pub oracle_duration: Duration,
}

pub struct GraphProblem {
    order: GraphOrder,
    original: OriginalTopology,
    routing: RoutingTopology,
    representation: Representation,
    cch: CchTopology,
    initial_weights: Vec<f64>,
    lower_bounds: Vec<f64>,
    upper_bounds: Vec<f64>,
    topology_identity: String,
}

pub struct GraphMetric<'a> {
    problem: &'a GraphProblem,
    cch: CchMetric<'a>,
    direct_weights: Vec<f64>,
    quantized_weights: Vec<u32>,
}

/// Reusable representation-bound query state. This avoids reallocating CCH
/// search buffers while still hiding representation-specific endpoints and
/// path decoding from callers.
pub struct GraphQuery<'metric, 'problem> {
    problem: &'problem GraphProblem,
    cch: CchReusableQuery<'metric>,
    direct_weights: &'metric [f64],
    quantized_weights: &'metric [u32],
}

#[derive(Clone, Debug)]
struct OriginalTopology {
    node_count: usize,
    tail: Vec<u32>,
    head: Vec<u32>,
}

#[derive(Clone, Debug)]
struct RoutingTopology {
    node_count: usize,
    tail: Vec<u32>,
    head: Vec<u32>,
    x: Vec<f32>,
    y: Vec<f32>,
}

enum Representation {
    First,
    Second(SecondOrderGraph),
}

#[derive(Debug)]
struct SecondOrderGraph {
    /// Stable lexicographic coordinate IDs: previous edge, then next edge.
    pairs: Vec<(usize, usize)>,
    pair_offsets: Vec<usize>,
    source_state_offsets: Vec<usize>,
    source_states: Vec<u32>,
    target_state_offsets: Vec<usize>,
    target_states: Vec<u32>,
}

impl GraphProblem {
    /// Build one graph problem and coordinate-wise box from multiplicative
    /// factors around its deterministic initial direct weights.
    pub fn build(
        graph: &GraphData,
        order: GraphOrder,
        lower_factor: f64,
        upper_factor: f64,
    ) -> Result<Self, String> {
        validate_original_graph(graph)?;
        validate_bound_factors(lower_factor, upper_factor)?;

        let original = OriginalTopology {
            node_count: graph.x.len(),
            tail: graph.tail.clone(),
            head: graph.head.clone(),
        };
        let (routing, representation, initial_weights) = match order {
            GraphOrder::First => (
                RoutingTopology {
                    node_count: graph.x.len(),
                    tail: graph.tail.clone(),
                    head: graph.head.clone(),
                    x: graph.x.clone(),
                    y: graph.y.clone(),
                },
                Representation::First,
                graph
                    .baseline_weights
                    .iter()
                    .map(|&weight| weight as f64)
                    .collect(),
            ),
            GraphOrder::Second => build_second_order(graph)?,
        };

        let cch = CchTopology::build(
            routing.node_count,
            &routing.tail,
            &routing.head,
            &routing.x,
            &routing.y,
        )?;
        let lower_bounds = scaled_bounds(&initial_weights, lower_factor, "lower")?;
        let upper_bounds = scaled_bounds(&initial_weights, upper_factor, "upper")?;
        for (coordinate, &upper) in upper_bounds.iter().enumerate() {
            quantize_weight(upper).map_err(|error| {
                format!("invalid upper bound for coordinate {coordinate}: {error}")
            })?;
        }
        let topology_identity =
            topology_identity(order, &original, &routing, &representation, cch.order());

        Ok(Self {
            order,
            original,
            routing,
            representation,
            cch,
            initial_weights,
            lower_bounds,
            upper_bounds,
            topology_identity,
        })
    }

    pub const fn order(&self) -> GraphOrder {
        self.order
    }

    pub fn coordinate_count(&self) -> usize {
        self.initial_weights.len()
    }

    pub fn initial_weights(&self) -> &[f64] {
        &self.initial_weights
    }

    pub fn lower_bounds(&self) -> &[f64] {
        &self.lower_bounds
    }

    pub fn upper_bounds(&self) -> &[f64] {
        &self.upper_bounds
    }

    pub fn topology_identity(&self) -> &str {
        &self.topology_identity
    }

    pub fn routing_node_count(&self) -> usize {
        self.routing.node_count
    }

    pub fn routing_arc_count(&self) -> usize {
        self.routing.tail.len()
    }

    /// Stable second-order coordinate lookup. First-order problems return
    /// `None` because their coordinates are already original edge IDs.
    pub fn second_order_pair(&self, coordinate: usize) -> Option<(usize, usize)> {
        match &self.representation {
            Representation::First => None,
            Representation::Second(second) => second.pairs.get(coordinate).copied(),
        }
    }

    pub fn second_order_pair_id(&self, previous: usize, next: usize) -> Option<usize> {
        let Representation::Second(second) = &self.representation else {
            return None;
        };
        second.pair_id(previous, next)
    }

    /// Validate and map one complete original-edge trajectory.
    pub fn map_path(&self, original_edges: &[usize]) -> Result<MappedPath, String> {
        let (source, target) = self.validate_original_path(original_edges)?;
        let coordinates = match &self.representation {
            Representation::First => original_edges.to_vec(),
            Representation::Second(second) => {
                if original_edges.len() < 2 {
                    return Err(
                        "a second-order trajectory requires at least two original edges"
                            .to_string(),
                    );
                }
                original_edges
                    .windows(2)
                    .map(|pair| {
                        second.pair_id(pair[0], pair[1]).ok_or_else(|| {
                            format!(
                                "missing second-order coordinate for legal transition {} -> {}",
                                pair[0], pair[1]
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?
            }
        };
        Ok(MappedPath {
            source,
            target,
            coordinates,
            original_edges: original_edges.to_vec(),
        })
    }

    pub fn map_trip(&self, trip: &TripPath) -> Result<MappedPath, String> {
        let mapped = self.map_path(&trip.1)?;
        if (mapped.source, mapped.target) != trip.0 {
            return Err(format!(
                "declared OD {:?} does not match mapped path OD ({}, {})",
                trip.0, mapped.source, mapped.target
            ));
        }
        Ok(mapped)
    }

    pub fn map_paths(&self, paths: &[TripPath]) -> Result<Vec<MappedPath>, String> {
        paths.iter().map(|path| self.map_trip(path)).collect()
    }

    /// Decode optimizer coordinates back to a complete original-edge path.
    pub fn decode_path(&self, coordinates: &[usize]) -> Result<Vec<usize>, String> {
        match &self.representation {
            Representation::First => {
                if coordinates.is_empty() {
                    return Err("a first-order coordinate path cannot be empty".to_string());
                }
                let decoded = coordinates.to_vec();
                self.validate_original_path(&decoded)?;
                Ok(decoded)
            }
            Representation::Second(second) => second.decode(coordinates),
        }
    }

    pub fn observed_counts(&self, paths: &[MappedPath]) -> Result<Vec<u64>, String> {
        let mut counts = vec![0u64; self.coordinate_count()];
        for path in paths {
            for &coordinate in &path.coordinates {
                let count = counts.get_mut(coordinate).ok_or_else(|| {
                    format!("mapped coordinate {coordinate} is outside this graph problem")
                })?;
                *count = count
                    .checked_add(1)
                    .ok_or_else(|| "observed coordinate count overflow".to_string())?;
            }
        }
        Ok(counts)
    }

    pub fn group_paths(paths: &[MappedPath]) -> Result<Vec<QueryGroup>, String> {
        let mut groups = BTreeMap::<(u32, u32), u64>::new();
        for path in paths {
            let count = groups.entry((path.source, path.target)).or_default();
            *count = count
                .checked_add(1)
                .ok_or_else(|| "query-group sample count overflow".to_string())?;
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

    /// Quantize direct coordinate weights and fully customize the internal CCH.
    /// The returned metric remains inseparably bound to this representation.
    pub fn customize<'a>(&'a self, weights: &[f64]) -> Result<GraphMetric<'a>, String> {
        self.validate_direct_weights(weights)?;
        let quantized_weights = weights
            .iter()
            .enumerate()
            .map(|(coordinate, &weight)| {
                quantize_weight(weight)
                    .map_err(|error| format!("invalid coordinate {coordinate}: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let arc_weights = self.coordinate_to_arc_weights(&quantized_weights)?;
        let cch = self.cch.customize(&arc_weights)?;
        Ok(GraphMetric {
            problem: self,
            cch,
            direct_weights: weights.to_vec(),
            quantized_weights,
        })
    }

    fn validate_original_path(&self, path: &[usize]) -> Result<(u32, u32), String> {
        let Some(&first) = path.first() else {
            return Err("an original-edge path cannot be empty".to_string());
        };
        if path.iter().any(|&edge| edge >= self.original.tail.len()) {
            return Err("original-edge path contains an out-of-bounds edge".to_string());
        }
        for pair in path.windows(2) {
            if self.original.head[pair[0]] != self.original.tail[pair[1]] {
                return Err(format!(
                    "original-edge path is discontinuous at {} -> {}",
                    pair[0], pair[1]
                ));
            }
        }
        let last = *path.last().expect("nonempty path checked above");
        Ok((self.original.tail[first], self.original.head[last]))
    }

    fn validate_decoded_od(
        &self,
        original_edges: &[usize],
        source: u32,
        target: u32,
    ) -> Result<(), String> {
        let decoded_od = self.validate_original_path(original_edges)?;
        if decoded_od != (source, target) {
            return Err(format!(
                "decoded path OD {:?} does not match query ({source}, {target})",
                decoded_od
            ));
        }
        Ok(())
    }

    fn validate_direct_weights(&self, weights: &[f64]) -> Result<(), String> {
        if weights.len() != self.coordinate_count() {
            return Err(format!(
                "direct weight count {} does not match coordinate count {}",
                weights.len(),
                self.coordinate_count()
            ));
        }
        for (coordinate, ((&weight, &lower), &upper)) in weights
            .iter()
            .zip(&self.lower_bounds)
            .zip(&self.upper_bounds)
            .enumerate()
        {
            if !weight.is_finite() || weight < lower || weight > upper {
                return Err(format!(
                    "weight[{coordinate}]={weight} must be finite and inside [{lower}, {upper}]"
                ));
            }
        }
        Ok(())
    }

    fn coordinate_to_arc_weights(&self, weights: &[u32]) -> Result<Vec<u32>, String> {
        if weights.len() != self.coordinate_count() {
            return Err("coordinate-to-arc conversion received the wrong weight count".into());
        }
        match &self.representation {
            Representation::First => Ok(weights.to_vec()),
            Representation::Second(_) => self
                .routing
                .head
                .iter()
                .enumerate()
                .map(|(arc, &head)| {
                    weights.get(head as usize).copied().ok_or_else(|| {
                        format!("second-order arc {arc} has invalid head state {head}")
                    })
                })
                .collect(),
        }
    }

    fn query_endpoints_u32(
        &self,
        source: u32,
        target: u32,
        weights: &[u32],
    ) -> Result<QueryEndpointsU32, String> {
        if source as usize >= self.original.node_count
            || target as usize >= self.original.node_count
        {
            return Err(format!(
                "query OD ({source}, {target}) is outside {} original nodes",
                self.original.node_count
            ));
        }
        match &self.representation {
            Representation::First => Ok((vec![(source, 0)], vec![(target, 0)])),
            Representation::Second(second) => {
                let sources = second
                    .source_states(source)
                    .iter()
                    .map(|&state| (state, weights[state as usize]))
                    .collect::<Vec<_>>();
                let targets = second
                    .target_states(target)
                    .iter()
                    .map(|&state| (state, 0))
                    .collect::<Vec<_>>();
                if sources.is_empty() || targets.is_empty() {
                    return Err(format!(
                        "second-order OD ({source}, {target}) has {} source states and {} target states",
                        sources.len(),
                        targets.len()
                    ));
                }
                Ok((sources, targets))
            }
        }
    }

    #[cfg(test)]
    fn query_endpoints_f64(
        &self,
        source: u32,
        target: u32,
        weights: &[f64],
    ) -> Result<QueryEndpointsF64, String> {
        let quantized = weights
            .iter()
            .map(|&weight| quantize_weight(weight).map(|value| value as f64))
            .collect::<Result<Vec<_>, _>>()?;
        if source as usize >= self.original.node_count
            || target as usize >= self.original.node_count
        {
            return Err(format!("query OD ({source}, {target}) is out of bounds"));
        }
        match &self.representation {
            Representation::First => Ok((vec![(source, 0.0)], vec![(target, 0.0)])),
            Representation::Second(second) => Ok((
                second
                    .source_states(source)
                    .iter()
                    .map(|&state| (state, quantized[state as usize]))
                    .collect(),
                second
                    .target_states(target)
                    .iter()
                    .map(|&state| (state, 0.0))
                    .collect(),
            )),
        }
    }

    #[cfg(test)]
    fn routing_arc_weights_f64(&self, weights: &[f64]) -> Result<Vec<f64>, String> {
        let quantized = weights
            .iter()
            .map(|&weight| quantize_weight(weight))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self
            .coordinate_to_arc_weights(&quantized)?
            .into_iter()
            .map(|weight| weight as f64)
            .collect())
    }
}

impl<'problem> GraphMetric<'problem> {
    pub fn direct_weights(&self) -> &[f64] {
        &self.direct_weights
    }

    pub fn quantized_weights(&self) -> &[u32] {
        &self.quantized_weights
    }

    pub fn topology_identity(&self) -> &str {
        self.problem.topology_identity()
    }

    pub fn shortest_path(&self, source: u32, target: u32) -> Result<ShortestPath, String> {
        self.new_query().shortest_path(source, target)
    }

    pub fn batch_stats(
        &self,
        groups: &[QueryGroup],
        num_chunks: usize,
    ) -> Result<OracleStats, String> {
        let started = Instant::now();
        if groups.is_empty() {
            return Ok(OracleStats {
                predicted_counts: vec![0; self.problem.coordinate_count()],
                weighted_shortest_distance_sum: 0,
                weighted_direct_path_cost_sum: 0.0,
                sample_count: 0,
                num_queries: 0,
                oracle_duration: started.elapsed(),
            });
        }

        type LocalStats = (Vec<u64>, u128, f64, u64, usize);
        let locals: Vec<Result<LocalStats, String>> = groups
            .par_chunks(chunk_size(groups.len(), num_chunks))
            .map(|chunk| {
                let mut query = self.new_query();
                let mut counts = vec![0u64; self.problem.coordinate_count()];
                let mut quantized_distance_sum = 0u128;
                let mut direct_cost_sum = 0.0;
                let mut sample_count = 0u64;
                for group in chunk {
                    let path =
                        query
                            .shortest_path(group.source, group.target)
                            .map_err(|error| {
                                format!(
                                    "OD ({}, {}) is unreachable or invalid: {error}",
                                    group.source, group.target
                                )
                            })?;
                    for coordinate in path.coordinates {
                        counts[coordinate] = counts[coordinate]
                            .checked_add(group.sample_count)
                            .ok_or_else(|| "predicted coordinate count overflow".to_string())?;
                    }
                    quantized_distance_sum = quantized_distance_sum
                        .checked_add(path.distance as u128 * group.sample_count as u128)
                        .ok_or_else(|| "shortest-distance sum overflow".to_string())?;
                    direct_cost_sum += path.direct_cost * group.sample_count as f64;
                    if !direct_cost_sum.is_finite() {
                        return Err("direct shortest-path cost sum is not finite".to_string());
                    }
                    sample_count = sample_count
                        .checked_add(group.sample_count)
                        .ok_or_else(|| "oracle sample count overflow".to_string())?;
                }
                Ok((
                    counts,
                    quantized_distance_sum,
                    direct_cost_sum,
                    sample_count,
                    chunk.len(),
                ))
            })
            .collect();

        let mut stats = OracleStats {
            predicted_counts: vec![0; self.problem.coordinate_count()],
            weighted_shortest_distance_sum: 0,
            weighted_direct_path_cost_sum: 0.0,
            sample_count: 0,
            num_queries: 0,
            oracle_duration: Duration::ZERO,
        };
        for local in locals {
            let (counts, quantized_distance, direct_cost, samples, queries) = local?;
            for (total, addend) in stats.predicted_counts.iter_mut().zip(counts) {
                *total = total
                    .checked_add(addend)
                    .ok_or_else(|| "predicted coordinate count overflow".to_string())?;
            }
            stats.weighted_shortest_distance_sum = stats
                .weighted_shortest_distance_sum
                .checked_add(quantized_distance)
                .ok_or_else(|| "shortest-distance sum overflow".to_string())?;
            stats.weighted_direct_path_cost_sum += direct_cost;
            if !stats.weighted_direct_path_cost_sum.is_finite() {
                return Err("direct shortest-path cost sum is not finite".to_string());
            }
            stats.sample_count = stats
                .sample_count
                .checked_add(samples)
                .ok_or_else(|| "oracle sample count overflow".to_string())?;
            stats.num_queries = stats
                .num_queries
                .checked_add(queries)
                .ok_or_else(|| "oracle query count overflow".to_string())?;
        }
        stats.oracle_duration = started.elapsed();
        Ok(stats)
    }

    pub fn new_query(&self) -> GraphQuery<'_, 'problem> {
        GraphQuery {
            problem: self.problem,
            cch: self.cch.new_query(),
            direct_weights: &self.direct_weights,
            quantized_weights: &self.quantized_weights,
        }
    }
}

impl GraphQuery<'_, '_> {
    pub fn shortest_path(&mut self, source: u32, target: u32) -> Result<ShortestPath, String> {
        let (sources, targets) =
            self.problem
                .query_endpoints_u32(source, target, self.quantized_weights)?;
        let raw = self.cch.shortest_path(&sources, &targets)?;
        let coordinates = match &self.problem.representation {
            Representation::First => raw.arc_path,
            Representation::Second(_) => raw.node_path,
        };
        let original_edges = self.problem.decode_path(&coordinates)?;
        self.problem
            .validate_decoded_od(&original_edges, source, target)?;

        let reconstructed = coordinates.iter().try_fold(0u128, |sum, &coordinate| {
            let weight = self
                .quantized_weights
                .get(coordinate)
                .ok_or_else(|| format!("CCH returned invalid coordinate {coordinate}"))?;
            sum.checked_add(*weight as u128)
                .ok_or_else(|| "reconstructed path cost overflow".to_string())
        })?;
        if reconstructed != raw.distance as u128 {
            return Err(format!(
                "CCH coordinate path costs {reconstructed} but reported distance {} for OD ({source}, {target})",
                raw.distance
            ));
        }
        let direct_cost = coordinates.iter().try_fold(0.0, |sum, &coordinate| {
            let weight = self
                .direct_weights
                .get(coordinate)
                .ok_or_else(|| format!("CCH returned invalid coordinate {coordinate}"))?;
            let next = sum + weight;
            if next.is_finite() {
                Ok(next)
            } else {
                Err("direct path cost is not finite".to_string())
            }
        })?;
        Ok(ShortestPath {
            distance: raw.distance,
            direct_cost,
            coordinates,
            original_edges,
        })
    }
}

impl SecondOrderGraph {
    fn pair_id(&self, previous: usize, next: usize) -> Option<usize> {
        let start = *self.pair_offsets.get(previous)?;
        let end = *self.pair_offsets.get(previous + 1)?;
        let relative = self.pairs[start..end]
            .binary_search_by_key(&next, |&(_, candidate)| candidate)
            .ok()?;
        Some(start + relative)
    }

    fn decode(&self, coordinates: &[usize]) -> Result<Vec<usize>, String> {
        let Some((&first_coordinate, remaining)) = coordinates.split_first() else {
            return Err("a second-order coordinate path cannot be empty".to_string());
        };
        let &(first, second) = self.pairs.get(first_coordinate).ok_or_else(|| {
            format!("second-order coordinate {first_coordinate} is out of bounds")
        })?;
        let mut decoded = vec![first, second];
        let mut previous_second = second;
        for &coordinate in remaining {
            let &(next_first, next_second) = self
                .pairs
                .get(coordinate)
                .ok_or_else(|| format!("second-order coordinate {coordinate} is out of bounds"))?;
            if next_first != previous_second {
                return Err(format!(
                    "second-order coordinate path does not overlap: expected first edge {previous_second}, got {next_first}"
                ));
            }
            decoded.push(next_second);
            previous_second = next_second;
        }
        Ok(decoded)
    }

    fn source_states(&self, source: u32) -> &[u32] {
        incidence_slice(
            &self.source_state_offsets,
            &self.source_states,
            source as usize,
        )
    }

    fn target_states(&self, target: u32) -> &[u32] {
        incidence_slice(
            &self.target_state_offsets,
            &self.target_states,
            target as usize,
        )
    }
}

fn build_second_order(
    graph: &GraphData,
) -> Result<(RoutingTopology, Representation, Vec<f64>), String> {
    let node_count = graph.x.len();
    let edge_count = graph.tail.len();
    let (outgoing_offsets, outgoing_edges) = build_incidence(node_count, &graph.tail)?;

    let mut pair_count = 0usize;
    for previous in 0..edge_count {
        let junction = graph.head[previous] as usize;
        pair_count = pair_count
            .checked_add(incidence_slice(&outgoing_offsets, &outgoing_edges, junction).len())
            .ok_or_else(|| "second-order node count overflow".to_string())?;
    }
    if pair_count == 0 || pair_count > MAX_SECOND_ORDER_NODES {
        return Err(format!(
            "second-order node count {pair_count} must be in 1..={MAX_SECOND_ORDER_NODES}"
        ));
    }
    if pair_count > u32::MAX as usize {
        return Err(format!(
            "second-order node count {pair_count} does not fit u32"
        ));
    }

    let mut pairs = Vec::with_capacity(pair_count);
    let mut pair_offsets = Vec::with_capacity(edge_count + 1);
    let mut initial_weights = Vec::with_capacity(pair_count);
    let mut state_x = Vec::with_capacity(pair_count);
    let mut state_y = Vec::with_capacity(pair_count);
    let mut source_keys = Vec::with_capacity(pair_count);
    let mut target_keys = Vec::with_capacity(pair_count);
    pair_offsets.push(0);
    for previous in 0..edge_count {
        let junction = graph.head[previous] as usize;
        for &next in incidence_slice(&outgoing_offsets, &outgoing_edges, junction) {
            let next = next as usize;
            pairs.push((previous, next));
            // A second-order coordinate is the direct weight of the transition
            // itself. Its deterministic anchor is the baseline of the second
            // original edge; no separate first-edge cost is introduced.
            initial_weights.push(graph.baseline_weights[next] as f64);
            state_x.push(graph.x[junction]);
            state_y.push(graph.y[junction]);
            source_keys.push(graph.tail[previous]);
            target_keys.push(graph.head[next]);
        }
        pair_offsets.push(pairs.len());
    }
    debug_assert_eq!(pairs.len(), pair_count);
    debug_assert!(
        pair_offsets
            .windows(2)
            .all(|range| pairs[range[0]..range[1]].is_sorted_by_key(|&(_, next)| next))
    );

    let mut arc_count = 0usize;
    for &(_, next) in &pairs {
        arc_count = arc_count
            .checked_add(pair_offsets[next + 1] - pair_offsets[next])
            .ok_or_else(|| "second-order overlap-arc count overflow".to_string())?;
    }
    if arc_count > MAX_SECOND_ORDER_ARCS {
        return Err(format!(
            "second-order graph would contain {arc_count} overlap arcs, exceeding {MAX_SECOND_ORDER_ARCS}"
        ));
    }
    let mut routing_tail = Vec::with_capacity(arc_count);
    let mut routing_head = Vec::with_capacity(arc_count);
    for (state, &(_, next)) in pairs.iter().enumerate() {
        let tail = u32::try_from(state)
            .map_err(|_| format!("second-order state {state} does not fit u32"))?;
        for successor in pair_offsets[next]..pair_offsets[next + 1] {
            routing_tail.push(tail);
            routing_head.push(u32::try_from(successor).map_err(|_| {
                format!("second-order successor state {successor} does not fit u32")
            })?);
        }
    }

    let (source_state_offsets, source_states) = build_state_incidence(node_count, &source_keys)?;
    let (target_state_offsets, target_states) = build_state_incidence(node_count, &target_keys)?;
    let second = SecondOrderGraph {
        pairs,
        pair_offsets,
        source_state_offsets,
        source_states,
        target_state_offsets,
        target_states,
    };
    let routing = RoutingTopology {
        node_count: pair_count,
        tail: routing_tail,
        head: routing_head,
        x: state_x,
        y: state_y,
    };
    Ok((routing, Representation::Second(second), initial_weights))
}

fn build_incidence(node_count: usize, endpoints: &[u32]) -> Result<(Vec<usize>, Vec<u32>), String> {
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
    let mut values = vec![0u32; endpoints.len()];
    for (value, &node) in endpoints.iter().enumerate() {
        let position = &mut cursor[node as usize];
        values[*position] = u32::try_from(value)
            .map_err(|_| format!("incidence value {value} does not fit u32"))?;
        *position += 1;
    }
    Ok((offsets, values))
}

fn build_state_incidence(
    node_count: usize,
    state_keys: &[u32],
) -> Result<(Vec<usize>, Vec<u32>), String> {
    build_incidence(node_count, state_keys)
}

fn incidence_slice<'a>(offsets: &[usize], values: &'a [u32], node: usize) -> &'a [u32] {
    &values[offsets[node]..offsets[node + 1]]
}

fn validate_original_graph(graph: &GraphData) -> Result<(), String> {
    let edge_count = graph.tail.len();
    if edge_count == 0
        || graph.head.len() != edge_count
        || graph.baseline_weights.len() != edge_count
    {
        return Err(format!(
            "invalid original edge arrays: tail={edge_count}, head={}, baseline={}",
            graph.head.len(),
            graph.baseline_weights.len()
        ));
    }
    let node_count = graph.x.len();
    if node_count == 0 || node_count > u32::MAX as usize || graph.y.len() != node_count {
        return Err(format!(
            "invalid original coordinate arrays: x={node_count}, y={}",
            graph.y.len()
        ));
    }
    for (node, (&x, &y)) in graph.x.iter().zip(&graph.y).enumerate() {
        if !x.is_finite() || !y.is_finite() {
            return Err(format!(
                "original node {node} has non-finite coordinates ({x}, {y})"
            ));
        }
    }
    for (edge, ((&tail, &head), &weight)) in graph
        .tail
        .iter()
        .zip(&graph.head)
        .zip(&graph.baseline_weights)
        .enumerate()
    {
        if tail as usize >= node_count || head as usize >= node_count {
            return Err(format!(
                "original edge {edge} endpoint out of bounds: {tail}->{head}"
            ));
        }
        if weight == 0 || weight >= CCH_INFINITY {
            return Err(format!(
                "original edge {edge} has invalid baseline weight {weight}"
            ));
        }
    }
    Ok(())
}

fn validate_bound_factors(lower: f64, upper: f64) -> Result<(), String> {
    if !lower.is_finite() || lower <= 0.0 || lower > 1.0 {
        return Err("lower weight factor must be finite and in (0, 1]".to_string());
    }
    if !upper.is_finite() || upper < 1.0 || upper < lower {
        return Err(
            "upper weight factor must be finite, at least one, and no smaller than lower"
                .to_string(),
        );
    }
    Ok(())
}

fn scaled_bounds(weights: &[f64], factor: f64, kind: &str) -> Result<Vec<f64>, String> {
    weights
        .iter()
        .enumerate()
        .map(|(coordinate, &weight)| {
            let bound = factor * weight;
            if !bound.is_finite() || bound <= 0.0 {
                return Err(format!(
                    "{kind} bound for coordinate {coordinate} is invalid: {bound}"
                ));
            }
            Ok(bound)
        })
        .collect()
}

fn quantize_weight(weight: f64) -> Result<u32, String> {
    if !weight.is_finite() || weight <= 0.0 {
        return Err(format!(
            "direct weight must be finite and positive, got {weight}"
        ));
    }
    let rounded = weight.round().max(1.0);
    if rounded >= CCH_INFINITY as f64 {
        return Err(format!(
            "quantized weight {rounded} reaches the CCH infinity sentinel"
        ));
    }
    Ok(rounded as u32)
}

fn topology_identity(
    order: GraphOrder,
    original: &OriginalTopology,
    routing: &RoutingTopology,
    representation: &Representation,
    cch_order: &[u32],
) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    hash_bytes(&mut hash, order.as_str().as_bytes());
    hash_u64(&mut hash, original.node_count as u64);
    hash_u32_slice(&mut hash, &original.tail);
    hash_u32_slice(&mut hash, &original.head);
    hash_u64(&mut hash, routing.node_count as u64);
    hash_u32_slice(&mut hash, &routing.tail);
    hash_u32_slice(&mut hash, &routing.head);
    hash_f32_slice(&mut hash, &routing.x);
    hash_f32_slice(&mut hash, &routing.y);
    if let Representation::Second(second) = representation {
        hash_u64(&mut hash, second.pairs.len() as u64);
        for &(previous, next) in &second.pairs {
            hash_u64(&mut hash, previous as u64);
            hash_u64(&mut hash, next as u64);
        }
    }
    hash_u32_slice(&mut hash, cch_order);
    format!("fnv1a64:{hash:016x}")
}

fn hash_u32_slice(hash: &mut u64, values: &[u32]) {
    hash_u64(hash, values.len() as u64);
    for value in values {
        hash_bytes(hash, &value.to_le_bytes());
    }
}

fn hash_f32_slice(hash: &mut u64, values: &[f32]) {
    hash_u64(hash, values.len() as u64);
    for value in values {
        hash_bytes(hash, &value.to_bits().to_le_bytes());
    }
}

fn hash_u64(hash: &mut u64, value: u64) {
    hash_bytes(hash, &value.to_le_bytes());
}

fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= *byte as u64;
        *hash = hash.wrapping_mul(0x100000001b3);
    }
}

fn chunk_size(len: usize, chunks: usize) -> usize {
    len.div_ceil(chunks.max(1)).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::shortest_path_multi_source_f64;

    fn graph() -> GraphData {
        // Two complete 0->3 routes:
        // [0,1,2] and [0,3,4].
        GraphData {
            tail: vec![0, 1, 2, 1, 4],
            head: vec![1, 2, 3, 4, 3],
            baseline_weights: vec![2, 3, 5, 1, 1],
            x: vec![0.0, 1.0, 2.0, 3.0, 2.0],
            y: vec![0.0, 0.0, 0.0, 0.0, 1.0],
        }
    }

    #[test]
    fn graph_order_has_stable_strings() {
        assert_eq!(GraphOrder::First.as_str(), "first");
        assert_eq!(GraphOrder::Second.as_str(), "second");
        assert_eq!(GraphOrder::parse("first").unwrap(), GraphOrder::First);
        assert_eq!(GraphOrder::parse("second").unwrap(), GraphOrder::Second);
        assert!(GraphOrder::parse("third").is_err());
    }

    #[test]
    fn first_order_mapping_is_the_original_edge_sequence() {
        let problem = GraphProblem::build(&graph(), GraphOrder::First, 0.5, 2.0).unwrap();
        let mapped = problem.map_path(&[0, 1, 2]).unwrap();
        assert_eq!(mapped.source, 0);
        assert_eq!(mapped.target, 3);
        assert_eq!(mapped.coordinates, vec![0, 1, 2]);
        assert_eq!(
            problem.decode_path(&mapped.coordinates).unwrap(),
            vec![0, 1, 2]
        );
        assert_eq!(problem.initial_weights(), &[2.0, 3.0, 5.0, 1.0, 1.0]);
        assert_eq!(problem.lower_bounds(), &[1.0, 1.5, 2.5, 0.5, 0.5]);
        assert_eq!(problem.upper_bounds(), &[4.0, 6.0, 10.0, 2.0, 2.0]);
    }

    #[test]
    fn second_order_mapping_uses_stable_pairs_and_overlap_decode() {
        let problem = GraphProblem::build(&graph(), GraphOrder::Second, 0.5, 2.0).unwrap();
        let expected_pairs = vec![(0, 1), (0, 3), (1, 2), (3, 4)];
        assert_eq!(problem.coordinate_count(), expected_pairs.len());
        assert_eq!(
            (0..problem.coordinate_count())
                .map(|coordinate| problem.second_order_pair(coordinate).unwrap())
                .collect::<Vec<_>>(),
            expected_pairs
        );
        assert_eq!(problem.initial_weights(), &[3.0, 1.0, 5.0, 1.0]);

        let mapped = problem.map_path(&[0, 1, 2]).unwrap();
        assert_eq!(mapped.coordinates, vec![0, 2]);
        assert_eq!(
            problem.decode_path(&mapped.coordinates).unwrap(),
            vec![0, 1, 2]
        );
        let alternative = problem.map_path(&[0, 3, 4]).unwrap();
        assert_eq!(alternative.coordinates, vec![1, 3]);
        assert_eq!(
            problem.decode_path(&alternative.coordinates).unwrap(),
            vec![0, 3, 4]
        );
        assert!(problem.decode_path(&[0, 3]).is_err());
        assert!(problem.map_path(&[0]).is_err());
    }

    #[test]
    fn both_orders_match_reference_dijkstra_costs() {
        for order in [GraphOrder::First, GraphOrder::Second] {
            let problem = GraphProblem::build(&graph(), order, 0.5, 2.0).unwrap();
            let metric = problem.customize(problem.initial_weights()).unwrap();
            let cch = metric.shortest_path(0, 3).unwrap();
            let arc_weights = problem
                .routing_arc_weights_f64(problem.initial_weights())
                .unwrap();
            let (sources, targets) = problem
                .query_endpoints_f64(0, 3, problem.initial_weights())
                .unwrap();
            let reference = shortest_path_multi_source_f64(
                problem.routing.node_count,
                &problem.routing.tail,
                &problem.routing.head,
                &arc_weights,
                &sources,
                &targets,
            )
            .unwrap()
            .unwrap();
            assert_eq!(cch.distance as f64, reference.distance, "order={order:?}");
        }
    }

    #[test]
    fn second_order_source_offset_pays_the_first_coordinate_once() {
        let problem = GraphProblem::build(&graph(), GraphOrder::Second, 0.5, 2.0).unwrap();
        let metric = problem.customize(problem.initial_weights()).unwrap();
        let shortest = metric.shortest_path(0, 3).unwrap();
        assert_eq!(shortest.coordinates, vec![1, 3]);
        assert_eq!(shortest.original_edges, vec![0, 3, 4]);
        assert_eq!(shortest.distance, 2);
        assert_eq!(shortest.direct_cost, 2.0);

        let groups = vec![QueryGroup {
            source: 0,
            target: 3,
            sample_count: 2,
        }];
        let stats = metric.batch_stats(&groups, 8).unwrap();
        assert_eq!(stats.predicted_counts, vec![0, 2, 0, 2]);
        assert_eq!(stats.weighted_shortest_distance_sum, 4);
        assert_eq!(stats.weighted_direct_path_cost_sum, 4.0);
        assert_eq!(stats.sample_count, 2);
        assert_eq!(stats.num_queries, 1);
    }

    #[test]
    fn a_two_edge_path_is_one_paid_second_order_state() {
        let problem = GraphProblem::build(&graph(), GraphOrder::Second, 0.5, 2.0).unwrap();
        let mapped = problem.map_path(&[0, 1]).unwrap();
        assert_eq!(mapped.coordinates, vec![0]);
        assert_eq!(
            problem.decode_path(&mapped.coordinates).unwrap(),
            vec![0, 1]
        );

        let metric = problem.customize(problem.initial_weights()).unwrap();
        let shortest = metric.shortest_path(0, 2).unwrap();
        assert_eq!(shortest.coordinates, vec![0]);
        assert_eq!(shortest.original_edges, vec![0, 1]);
        assert_eq!(shortest.distance, 3);
    }

    #[test]
    fn topology_identity_distinguishes_orders_and_is_stable() {
        let first_a = GraphProblem::build(&graph(), GraphOrder::First, 0.5, 2.0).unwrap();
        let first_b = GraphProblem::build(&graph(), GraphOrder::First, 0.5, 2.0).unwrap();
        let second = GraphProblem::build(&graph(), GraphOrder::Second, 0.5, 2.0).unwrap();
        assert_eq!(first_a.topology_identity(), first_b.topology_identity());
        assert_ne!(first_a.topology_identity(), second.topology_identity());
    }
}
