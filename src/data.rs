use rayon::prelude::*;
use routingkit_cch::shp_utils;
use std::collections::{BTreeMap, HashSet};
use std::fs::File;

/// One complete observed path expressed in original directed edge IDs.
pub type TripPath = ((u32, u32), Vec<usize>);

#[derive(Debug)]
pub struct GraphData {
    pub tail: Vec<u32>,
    pub head: Vec<u32>,
    pub baseline_weights: Vec<u32>,
    pub x: Vec<f32>,
    pub y: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PathValidationError {
    Empty,
    OutOfBounds,
    Discontinuous,
    Cyclic,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PathValidationReport {
    pub available_samples: usize,
    pub inspected_samples: usize,
    pub accepted_samples: usize,
    pub empty: usize,
    pub out_of_bounds: usize,
    pub discontinuous: usize,
    pub cyclic: usize,
}

impl PathValidationReport {
    pub fn dropped_samples(&self) -> usize {
        self.inspected_samples - self.accepted_samples
    }
}

#[derive(Debug)]
pub struct LoadedTrips {
    pub paths: Vec<TripPath>,
    pub report: PathValidationReport,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OdGroup {
    pub source: u32,
    pub target: u32,
    pub sample_count: u64,
}

/// Load the fixed directed road topology and its positive baseline costs.
pub fn load_graph(city: &str) -> Result<GraphData, String> {
    let edges_path = format!("data/{city}_data/map/edges.shp");
    let nodes_path = format!("data/{city}_data/map/nodes.shp");
    let edges = shp_utils::load_edges(&edges_path)
        .map_err(|error| format!("failed to load {edges_path}: {error}"))?;
    let nodes = shp_utils::load_nodes(&nodes_path)
        .map_err(|error| format!("failed to load {nodes_path}: {error}"))?;
    let arrays = shp_utils::build_graph_arrays(&nodes, &edges)
        .map_err(|error| format!("failed to construct graph arrays: {error}"))?;

    let mut baseline_weights = Vec::with_capacity(arrays.weight.len());
    for (edge_id, length) in arrays.weight.into_iter().enumerate() {
        let scaled = length * 1_000.0;
        if !scaled.is_finite() || scaled <= 0.0 || scaled >= i32::MAX as f64 {
            return Err(format!(
                "edge {edge_id} has invalid scaled baseline cost {scaled}"
            ));
        }
        baseline_weights.push(scaled.round().max(1.0) as u32);
    }

    Ok(GraphData {
        tail: arrays.tail.into_iter().map(|node| node as u32).collect(),
        head: arrays.head.into_iter().map(|node| node as u32).collect(),
        baseline_weights,
        x: arrays.xs.into_iter().map(|value| value as f32).collect(),
        y: arrays.ys.into_iter().map(|value| value as f32).collect(),
    })
}

/// Load one split using the sole training data policy.
///
/// The pickle schema is
/// `(trip_key, Vec<original_edge_id>, (start_time, end_time))`. The edge vector
/// is already the complete path: no first or last edge is removed. Structurally
/// invalid paths and cyclic observations are dropped.
pub fn load_trips(
    city: &str,
    split: &str,
    variant: &str,
    graph: &GraphData,
    max_samples: Option<usize>,
) -> Result<LoadedTrips, String> {
    let path = format!("data/{city}_data/preprocessed_{split}_trips_{variant}.pkl");
    let raw: Vec<(serde_pickle::Value, Vec<usize>, (usize, usize))> = serde_pickle::from_reader(
        File::open(&path).map_err(|error| format!("failed to open {path}: {error}"))?,
        Default::default(),
    )
    .map_err(|error| format!("failed to decode {path}: {error}"))?;

    let mut report = PathValidationReport {
        available_samples: raw.len(),
        ..PathValidationReport::default()
    };
    let inspect_count = max_samples.unwrap_or(raw.len()).min(raw.len());
    let mut paths = Vec::with_capacity(inspect_count);

    for (_, edge_path, _) in raw.into_iter().take(inspect_count) {
        report.inspected_samples += 1;
        match validate_edge_path(&edge_path, &graph.tail, &graph.head) {
            Ok(od) => {
                paths.push((od, edge_path));
                report.accepted_samples += 1;
            }
            Err(PathValidationError::Empty) => report.empty += 1,
            Err(PathValidationError::OutOfBounds) => report.out_of_bounds += 1,
            Err(PathValidationError::Discontinuous) => report.discontinuous += 1,
            Err(PathValidationError::Cyclic) => report.cyclic += 1,
        }
    }

    Ok(LoadedTrips { paths, report })
}

/// Validate a complete original-edge path and return its OD pair.
pub fn validate_edge_path(
    path: &[usize],
    tail: &[u32],
    head: &[u32],
) -> Result<(u32, u32), PathValidationError> {
    let Some(&first_edge) = path.first() else {
        return Err(PathValidationError::Empty);
    };
    if tail.len() != head.len()
        || path
            .iter()
            .any(|&edge_id| edge_id >= tail.len() || edge_id >= head.len())
    {
        return Err(PathValidationError::OutOfBounds);
    }
    if path.windows(2).any(|pair| head[pair[0]] != tail[pair[1]]) {
        return Err(PathValidationError::Discontinuous);
    }

    let mut visited_nodes = HashSet::with_capacity(path.len() + 1);
    visited_nodes.insert(tail[first_edge]);
    for &edge_id in path {
        if !visited_nodes.insert(head[edge_id]) {
            return Err(PathValidationError::Cyclic);
        }
    }

    let last_edge = *path.last().expect("nonempty path checked above");
    Ok((tail[first_edge], head[last_edge]))
}

pub fn compute_observed_edge_counts(
    paths: &[TripPath],
    edge_count: usize,
    num_chunks: usize,
) -> Vec<u64> {
    if paths.is_empty() {
        return vec![0; edge_count];
    }
    paths
        .par_chunks(chunk_size(paths.len(), num_chunks))
        .map(|chunk| {
            let mut local = vec![0u64; edge_count];
            for (_, path) in chunk {
                for &edge_id in path {
                    local[edge_id] = local[edge_id]
                        .checked_add(1)
                        .expect("observed edge count overflow");
                }
            }
            local
        })
        .reduce(
            || vec![0; edge_count],
            |mut left, right| {
                left.iter_mut().zip(right).for_each(|(value, addend)| {
                    *value = value
                        .checked_add(addend)
                        .expect("observed edge count overflow");
                });
                left
            },
        )
}

/// Group observations by OD so each unique pair needs one oracle query.
pub fn group_paths_by_od(paths: &[TripPath]) -> Vec<OdGroup> {
    let mut counts = BTreeMap::<(u32, u32), u64>::new();
    for &((source, target), _) in paths {
        let count = counts.entry((source, target)).or_default();
        *count = count.checked_add(1).expect("OD sample count overflow");
    }
    counts
        .into_iter()
        .map(|((source, target), sample_count)| OdGroup {
            source,
            target,
            sample_count,
        })
        .collect()
}

fn chunk_size(len: usize, num_chunks: usize) -> usize {
    len.div_ceil(num_chunks.max(1)).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_complete_paths_and_rejects_cycles() {
        let tail = [0, 1, 0, 2, 1];
        let head = [1, 3, 2, 3, 0];
        assert_eq!(
            validate_edge_path(&[], &tail, &head),
            Err(PathValidationError::Empty)
        );
        assert_eq!(
            validate_edge_path(&[99], &tail, &head),
            Err(PathValidationError::OutOfBounds)
        );
        assert_eq!(
            validate_edge_path(&[0, 3], &tail, &head),
            Err(PathValidationError::Discontinuous)
        );
        assert_eq!(
            validate_edge_path(&[0, 4], &tail, &head),
            Err(PathValidationError::Cyclic)
        );
        assert_eq!(validate_edge_path(&[0, 1], &tail, &head), Ok((0, 3)));
    }

    #[test]
    fn counts_edges_and_groups_od_deterministically() {
        let paths = vec![
            ((4, 8), vec![0, 1]),
            ((1, 3), vec![2]),
            ((4, 8), vec![0, 1]),
        ];
        assert_eq!(compute_observed_edge_counts(&paths, 3, 16), vec![2, 2, 1]);
        assert_eq!(
            group_paths_by_od(&paths),
            vec![
                OdGroup {
                    source: 1,
                    target: 3,
                    sample_count: 1,
                },
                OdGroup {
                    source: 4,
                    target: 8,
                    sample_count: 2,
                },
            ]
        );
    }

    #[test]
    fn empty_edge_count_workload_is_well_defined() {
        assert_eq!(compute_observed_edge_counts(&[], 3, 64), vec![0; 3]);
    }
}
