use rayon::prelude::*;
use routingkit_cch::shp_utils;
use std::collections::{BTreeMap, HashSet};
use std::fs::File;

/// One complete observed path expressed in original directed edge IDs.
pub type TripPath = ((u32, u32), Vec<usize>);

/// Whole-trajectory timestamps retained from the source pickle.
///
/// The source only supplies one start and end timestamp for the complete path;
/// these values must never be interpreted as per-edge observations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TripTime {
    pub start_time: u64,
    pub end_time: u64,
}

impl TripTime {
    pub fn duration_seconds(self) -> Option<u64> {
        self.end_time
            .checked_sub(self.start_time)
            .filter(|&value| value > 0)
    }
}

/// Raw timestamp evidence collected while decoding a split. The two date-key
/// match counts provide an explicit UTC versus UTC+8 check without making the
/// route loader depend on a timezone database.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TimestampEvidence {
    pub timestamp_samples: usize,
    pub invalid_intervals: usize,
    pub minimum_start_time: Option<u64>,
    pub maximum_end_time: Option<u64>,
    pub mmdd_keys: usize,
    pub mmdd_matches_utc: usize,
    pub mmdd_matches_utc_plus_8: usize,
}

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
    TooShort,
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
    pub too_short: usize,
    pub out_of_bounds: usize,
    pub discontinuous: usize,
    pub cyclic: usize,
}

impl PathValidationReport {
    pub fn dropped_samples(&self) -> usize {
        self.inspected_samples - self.accepted_samples
    }

    fn record_rejection(&mut self, error: PathValidationError) {
        match error {
            PathValidationError::Empty => self.empty += 1,
            PathValidationError::TooShort => self.too_short += 1,
            PathValidationError::OutOfBounds => self.out_of_bounds += 1,
            PathValidationError::Discontinuous => self.discontinuous += 1,
            PathValidationError::Cyclic => self.cyclic += 1,
        }
    }
}

#[derive(Debug)]
pub struct LoadedTrips {
    pub paths: Vec<TripPath>,
    /// Aligned one-to-one with `paths` after applying the common route filter.
    pub times: Vec<TripTime>,
    pub report: PathValidationReport,
    pub timestamp_evidence: TimestampEvidence,
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
/// invalid paths, paths with fewer than two edges, and cyclic observations are
/// dropped. `available_samples` remains the raw split size, independently of
/// the inspection limit and validation outcome.
pub fn load_trips(
    city: &str,
    split: &str,
    variant: &str,
    graph: &GraphData,
    max_samples: Option<usize>,
) -> Result<LoadedTrips, String> {
    let path = format!("data/{city}_data/preprocessed_{split}_trips_{variant}.pkl");
    let raw: Vec<(serde_pickle::Value, Vec<usize>, (u64, u64))> = serde_pickle::from_reader(
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
    let mut times = Vec::with_capacity(inspect_count);
    let mut timestamp_evidence = TimestampEvidence::default();

    for (trip_key, edge_path, (start_time, end_time)) in raw.into_iter().take(inspect_count) {
        report.inspected_samples += 1;
        record_timestamp_evidence(
            &mut timestamp_evidence,
            &trip_key,
            TripTime {
                start_time,
                end_time,
            },
        );
        match validate_edge_path(&edge_path, &graph.tail, &graph.head) {
            Ok(od) => {
                paths.push((od, edge_path));
                times.push(TripTime {
                    start_time,
                    end_time,
                });
                report.accepted_samples += 1;
            }
            Err(error) => report.record_rejection(error),
        }
    }

    debug_assert_eq!(paths.len(), times.len());
    Ok(LoadedTrips {
        paths,
        times,
        report,
        timestamp_evidence,
    })
}

fn record_timestamp_evidence(
    evidence: &mut TimestampEvidence,
    trip_key: &serde_pickle::Value,
    time: TripTime,
) {
    evidence.timestamp_samples += 1;
    evidence.minimum_start_time = Some(
        evidence
            .minimum_start_time
            .map_or(time.start_time, |current| current.min(time.start_time)),
    );
    evidence.maximum_end_time = Some(
        evidence
            .maximum_end_time
            .map_or(time.end_time, |current| current.max(time.end_time)),
    );
    if time.duration_seconds().is_none() {
        evidence.invalid_intervals += 1;
    }

    let Some(key) = mmdd_key(trip_key) else {
        return;
    };
    evidence.mmdd_keys += 1;
    evidence.mmdd_matches_utc += usize::from(epoch_mmdd(time.start_time, 0) == key);
    evidence.mmdd_matches_utc_plus_8 +=
        usize::from(epoch_mmdd(time.start_time, 8 * 60 * 60) == key);
}

fn mmdd_key(value: &serde_pickle::Value) -> Option<String> {
    let values = match value {
        serde_pickle::Value::Tuple(values) | serde_pickle::Value::List(values) => values,
        _ => return None,
    };
    let candidate = match values.first()? {
        serde_pickle::Value::String(candidate) => candidate.clone(),
        serde_pickle::Value::Bytes(candidate) => String::from_utf8(candidate.clone()).ok()?,
        _ => return None,
    };
    if candidate.len() == 4 && candidate.bytes().all(|byte| byte.is_ascii_digit()) {
        Some(candidate)
    } else {
        None
    }
}

fn epoch_mmdd(timestamp: u64, utc_offset_seconds: u64) -> String {
    let local_days = timestamp.saturating_add(utc_offset_seconds) / 86_400;
    let (_, month, day) = civil_from_unix_days(local_days as i64);
    format!("{month:02}{day:02}")
}

/// Convert days since 1970-01-01 to a proleptic-Gregorian civil date.
/// Adapted from Howard Hinnant's public-domain civil-calendar algorithm.
fn civil_from_unix_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (year, month as u32, day as u32)
}

/// Validate a complete original-edge path with at least one transition and
/// return its OD pair.
pub fn validate_edge_path(
    path: &[usize],
    tail: &[u32],
    head: &[u32],
) -> Result<(u32, u32), PathValidationError> {
    let Some(&first_edge) = path.first() else {
        return Err(PathValidationError::Empty);
    };
    if path.len() < 2 {
        return Err(PathValidationError::TooShort);
    }
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
    fn validates_transition_paths_and_rejects_invalid_inputs() {
        let tail = [0, 1, 0, 2, 1];
        let head = [1, 3, 2, 3, 0];
        assert_eq!(
            validate_edge_path(&[], &tail, &head),
            Err(PathValidationError::Empty)
        );
        assert_eq!(
            validate_edge_path(&[0], &tail, &head),
            Err(PathValidationError::TooShort)
        );
        assert_eq!(
            validate_edge_path(&[99, 1], &tail, &head),
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
    fn validation_report_counts_too_short_paths_separately() {
        let mut report = PathValidationReport {
            available_samples: 5,
            inspected_samples: 3,
            accepted_samples: 1,
            ..PathValidationReport::default()
        };
        report.record_rejection(PathValidationError::Empty);
        report.record_rejection(PathValidationError::TooShort);

        assert_eq!(report.empty, 1);
        assert_eq!(report.too_short, 1);
        assert_eq!(report.dropped_samples(), 2);
        assert_eq!(report.available_samples, 5);
    }

    #[test]
    fn counts_edges_and_groups_od_deterministically() {
        let paths = vec![
            ((4, 8), vec![0, 1]),
            ((1, 3), vec![2, 1]),
            ((4, 8), vec![0, 1]),
        ];
        assert_eq!(compute_observed_edge_counts(&paths, 3, 16), vec![2, 3, 1]);
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

    #[test]
    fn timestamp_evidence_distinguishes_utc_from_beijing_date_keys() {
        let mut evidence = TimestampEvidence::default();
        record_timestamp_evidence(
            &mut evidence,
            &serde_pickle::Value::Tuple(vec![
                serde_pickle::Value::Bytes(b"0504".to_vec()),
                serde_pickle::Value::I64(1),
            ]),
            TripTime {
                // 2009-05-03 23:52 UTC / 2009-05-04 07:52 UTC+8.
                start_time: 1_241_394_720,
                end_time: 1_241_394_985,
            },
        );
        assert_eq!(evidence.mmdd_keys, 1);
        assert_eq!(evidence.mmdd_matches_utc, 0);
        assert_eq!(evidence.mmdd_matches_utc_plus_8, 1);
        assert_eq!(evidence.invalid_intervals, 0);
    }
}
