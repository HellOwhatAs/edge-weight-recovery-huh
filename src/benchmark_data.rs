//! Frozen trajectory alignment for cross-method benchmarks.
//!
//! The common protocol intentionally operates on complete, unmodified
//! original-road ID sequences. It never applies NeuroMLR's upstream `(u,v)`
//! parallel-edge condensation and never repairs loops differently per method.

use crate::config::atomic_write;
use crate::data::GraphData;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::{File, create_dir_all};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommonTrip {
    pub manifest_id: String,
    pub source_index: usize,
    pub original_trip_id: String,
    pub edges: Vec<usize>,
    pub start_time: u64,
    pub end_time: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CommonManifestPolicy {
    pub minimum_edges: usize,
    pub maximum_selected: Option<usize>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CommonFilterCounts {
    pub empty: usize,
    pub too_short: usize,
    pub out_of_bounds: usize,
    pub discontinuous: usize,
    pub cyclic: usize,
}

impl CommonFilterCounts {
    pub fn dropped(&self) -> usize {
        self.empty + self.too_short + self.out_of_bounds + self.discontinuous + self.cyclic
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommonManifestAudit {
    pub source_path: PathBuf,
    pub source_bytes: u64,
    pub source_sha256: String,
    pub source_records: usize,
    pub eligible_records: usize,
    pub selected_records: usize,
    pub filters: CommonFilterCounts,
    pub duplicate_original_trip_ids: usize,
    pub selected_unique_edges: usize,
    pub selected_minimum_edges: usize,
    pub selected_maximum_edges: usize,
    pub selected_edge_occurrences: u64,
    pub graph_parallel_edge_ids: usize,
    pub selected_paths_changed_by_upstream_parallel_edge_condensation: usize,
}

impl CommonManifestAudit {
    pub fn as_json(
        &self,
        city: &str,
        split: &str,
        variant: &str,
        policy: CommonManifestPolicy,
    ) -> Value {
        json!({
            "schema_version": 1,
            "city": city,
            "split": split,
            "variant": variant,
            "source": {
                "path": self.source_path,
                "bytes": self.source_bytes,
                "sha256": self.source_sha256,
                "records": self.source_records,
            },
            "policy": {
                "road_id_space": "unaltered_shapefile_record_index",
                "minimum_edges": policy.minimum_edges,
                "continuity": "head(previous)==tail(next)",
                "cycle_policy": "drop_if_any_original_node_repeats",
                "selection": "first_eligible_in_source_order",
                "maximum_selected": policy.maximum_selected,
                "neuromlr_traffic_features": false,
            },
            "filtering": {
                "eligible": self.eligible_records,
                "selected": self.selected_records,
                "dropped": self.filters.dropped(),
                "empty": self.filters.empty,
                "too_short": self.filters.too_short,
                "out_of_bounds_or_unrepresentable": self.filters.out_of_bounds,
                "discontinuous": self.filters.discontinuous,
                "cyclic": self.filters.cyclic,
                "eligibility_coverage": ratio(self.eligible_records, self.source_records),
                "selected_source_coverage": ratio(self.selected_records, self.source_records),
            },
            "identity_audit": {
                "duplicate_original_trip_ids": self.duplicate_original_trip_ids,
                "selected_unique_edges": self.selected_unique_edges,
                "selected_minimum_edges": self.selected_minimum_edges,
                "selected_maximum_edges": self.selected_maximum_edges,
                "selected_edge_occurrences": self.selected_edge_occurrences,
            },
            "upstream_compatibility": {
                "graph_parallel_edge_ids": self.graph_parallel_edge_ids,
                "selected_paths_that_upstream_condense_edges_would_change": self.selected_paths_changed_by_upstream_parallel_edge_condensation,
                "adapter_policy": "preserve_raw_edge_ids_one_to_one",
            },
            "test_read": split == "test",
        })
    }
}

pub fn build_common_manifest(
    city: &str,
    split: &str,
    variant: &str,
    graph: &GraphData,
    policy: CommonManifestPolicy,
) -> Result<(Vec<CommonTrip>, CommonManifestAudit), String> {
    if policy.minimum_edges < 2 {
        return Err("common manifest minimum_edges must be at least two".to_string());
    }
    if policy.maximum_selected == Some(0) {
        return Err("common manifest maximum_selected must be positive".to_string());
    }
    validate_component(city, "city")?;
    validate_component(split, "split")?;
    validate_component(variant, "variant")?;
    let source_path = PathBuf::from(format!(
        "data/{city}_data/preprocessed_{split}_trips_{variant}.pkl"
    ));
    let source_bytes = std::fs::metadata(&source_path)
        .map_err(|error| format!("failed to stat {}: {error}", source_path.display()))?
        .len();
    let source_sha256 = sha256_file(&source_path)?;
    let raw: Vec<(serde_pickle::Value, Vec<usize>, (u64, u64))> = serde_pickle::from_reader(
        File::open(&source_path)
            .map_err(|error| format!("failed to open {}: {error}", source_path.display()))?,
        Default::default(),
    )
    .map_err(|error| format!("failed to decode {}: {error}", source_path.display()))?;

    let canonical_edges = canonical_parallel_edges(graph);
    let graph_parallel_edge_ids = canonical_edges
        .iter()
        .enumerate()
        .filter(|(edge, canonical)| *edge != **canonical)
        .count();
    let mut filters = CommonFilterCounts::default();
    let mut eligible_records = 0usize;
    let mut selected = Vec::new();
    let mut original_ids = HashSet::new();
    let mut duplicate_original_trip_ids = 0usize;
    let mut selected_edges = HashSet::new();
    let mut selected_minimum_edges = usize::MAX;
    let mut selected_maximum_edges = 0usize;
    let mut selected_edge_occurrences = 0u64;
    let mut changed_by_condensation = 0usize;

    for (source_index, (trip_id, edges, (start_time, end_time))) in raw.into_iter().enumerate() {
        let original_trip_id = trip_id.to_string();
        if !original_ids.insert(original_trip_id.clone()) {
            duplicate_original_trip_ids += 1;
        }
        if let Err(reason) = validate_common_path(&edges, graph, policy.minimum_edges) {
            match reason {
                CommonPathError::Empty => filters.empty += 1,
                CommonPathError::TooShort => filters.too_short += 1,
                CommonPathError::OutOfBounds => filters.out_of_bounds += 1,
                CommonPathError::Discontinuous => filters.discontinuous += 1,
                CommonPathError::Cyclic => filters.cyclic += 1,
            }
            continue;
        }
        eligible_records += 1;
        if policy
            .maximum_selected
            .is_some_and(|maximum| selected.len() >= maximum)
        {
            continue;
        }

        if edges
            .iter()
            .any(|&edge| canonical_edges.get(edge).copied() != Some(edge))
        {
            changed_by_condensation += 1;
        }
        selected_minimum_edges = selected_minimum_edges.min(edges.len());
        selected_maximum_edges = selected_maximum_edges.max(edges.len());
        selected_edge_occurrences = selected_edge_occurrences
            .checked_add(edges.len() as u64)
            .ok_or_else(|| "selected edge-occurrence count overflow".to_string())?;
        selected_edges.extend(edges.iter().copied());
        selected.push(CommonTrip {
            manifest_id: format!("{split}:{source_index:09}"),
            source_index,
            original_trip_id,
            edges,
            start_time,
            end_time,
        });
    }
    if selected.is_empty() {
        return Err("common manifest selected no trajectories".to_string());
    }
    if filters.dropped() + eligible_records != original_ids.len() + duplicate_original_trip_ids {
        return Err("common manifest filtering audit does not balance".to_string());
    }

    let audit = CommonManifestAudit {
        source_path,
        source_bytes,
        source_sha256,
        source_records: filters.dropped() + eligible_records,
        eligible_records,
        selected_records: selected.len(),
        filters,
        duplicate_original_trip_ids,
        selected_unique_edges: selected_edges.len(),
        selected_minimum_edges,
        selected_maximum_edges,
        selected_edge_occurrences,
        graph_parallel_edge_ids,
        selected_paths_changed_by_upstream_parallel_edge_condensation: changed_by_condensation,
    };
    Ok((selected, audit))
}

pub fn write_common_manifest(
    trips: &[CommonTrip],
    manifest_path: &Path,
    pickle_path: &Path,
) -> Result<Value, String> {
    ensure_parent(manifest_path)?;
    ensure_parent(pickle_path)?;
    let manifest_tmp = temporary_sibling(manifest_path);
    let manifest_file = File::create(&manifest_tmp)
        .map_err(|error| format!("failed to create {}: {error}", manifest_tmp.display()))?;
    let mut writer = BufWriter::new(manifest_file);
    let mut manifest_hash = Sha256::new();
    for trip in trips {
        let mut encoded = serde_json::to_vec(&json!({
            "manifest_id": trip.manifest_id,
            "source_index": trip.source_index,
            "original_trip_id": trip.original_trip_id,
            "edges": trip.edges,
            "start_time": trip.start_time,
            "end_time": trip.end_time,
        }))
        .map_err(|error| format!("failed to encode common manifest row: {error}"))?;
        encoded.push(b'\n');
        writer
            .write_all(&encoded)
            .map_err(|error| format!("failed to write {}: {error}", manifest_tmp.display()))?;
        manifest_hash.update(&encoded);
    }
    writer
        .flush()
        .map_err(|error| format!("failed to flush {}: {error}", manifest_tmp.display()))?;
    drop(writer);
    std::fs::rename(&manifest_tmp, manifest_path).map_err(|error| {
        format!(
            "failed to move {} to {}: {error}",
            manifest_tmp.display(),
            manifest_path.display()
        )
    })?;

    let pickle_rows = trips
        .iter()
        .map(|trip| {
            (
                trip.manifest_id.clone(),
                trip.edges.clone(),
                (trip.start_time, trip.end_time),
            )
        })
        .collect::<Vec<_>>();
    let pickle_tmp = temporary_sibling(pickle_path);
    let mut pickle_file = File::create(&pickle_tmp)
        .map_err(|error| format!("failed to create {}: {error}", pickle_tmp.display()))?;
    serde_pickle::to_writer(&mut pickle_file, &pickle_rows, Default::default())
        .map_err(|error| format!("failed to encode {}: {error}", pickle_tmp.display()))?;
    pickle_file
        .flush()
        .map_err(|error| format!("failed to flush {}: {error}", pickle_tmp.display()))?;
    drop(pickle_file);
    std::fs::rename(&pickle_tmp, pickle_path).map_err(|error| {
        format!(
            "failed to move {} to {}: {error}",
            pickle_tmp.display(),
            pickle_path.display()
        )
    })?;

    Ok(json!({
        "manifest": {
            "path": manifest_path,
            "bytes": std::fs::metadata(manifest_path).map_err(|error| error.to_string())?.len(),
            "sha256": format!("{:x}", manifest_hash.finalize()),
            "records": trips.len(),
        },
        "training_pickle": {
            "path": pickle_path,
            "bytes": std::fs::metadata(pickle_path).map_err(|error| error.to_string())?.len(),
            "sha256": sha256_file(pickle_path)?,
            "records": trips.len(),
            "trip_key": "manifest_id",
        }
    }))
}

pub fn write_common_audit(path: &Path, audit: &Value, outputs: &Value) -> Result<(), String> {
    let output = json!({
        "audit": audit,
        "outputs": outputs,
    });
    let encoded = serde_json::to_vec_pretty(&output)
        .map_err(|error| format!("failed to encode common audit: {error}"))?;
    atomic_write(path, &encoded)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CommonPathError {
    Empty,
    TooShort,
    OutOfBounds,
    Discontinuous,
    Cyclic,
}

fn validate_common_path(
    edges: &[usize],
    graph: &GraphData,
    minimum_edges: usize,
) -> Result<(), CommonPathError> {
    let Some(&first) = edges.first() else {
        return Err(CommonPathError::Empty);
    };
    if edges.len() < minimum_edges {
        return Err(CommonPathError::TooShort);
    }
    if graph.tail.len() != graph.head.len()
        || edges
            .iter()
            .any(|&edge| edge >= graph.tail.len() || edge >= graph.head.len())
    {
        return Err(CommonPathError::OutOfBounds);
    }
    if edges
        .windows(2)
        .any(|pair| graph.head[pair[0]] != graph.tail[pair[1]])
    {
        return Err(CommonPathError::Discontinuous);
    }
    let mut nodes = HashSet::with_capacity(edges.len() + 1);
    nodes.insert(graph.tail[first]);
    for &edge in edges {
        if !nodes.insert(graph.head[edge]) {
            return Err(CommonPathError::Cyclic);
        }
    }
    Ok(())
}

fn canonical_parallel_edges(graph: &GraphData) -> Vec<usize> {
    let mut last = HashMap::<(u32, u32), usize>::new();
    for (edge, (&tail, &head)) in graph.tail.iter().zip(&graph.head).enumerate() {
        last.insert((tail, head), edge);
    }
    graph
        .tail
        .iter()
        .zip(&graph.head)
        .map(|(&tail, &head)| last[&(tail, head)])
        .collect()
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file =
        File::open(path).map_err(|error| format!("failed to open {}: {error}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn validate_component(value: &str, label: &str) -> Result<(), String> {
    if value.is_empty() || value.contains('/') || value.contains("..") {
        return Err(format!("{label} contains an unsafe path component"));
    }
    Ok(())
}

fn ensure_parent(path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    Ok(())
}

fn temporary_sibling(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    name.push_str(&format!(".tmp-{}", std::process::id()));
    path.with_file_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};

    fn graph() -> GraphData {
        GraphData {
            tail: vec![0, 1, 2, 3, 4, 0],
            head: vec![1, 2, 3, 4, 0, 1],
            baseline_weights: vec![1; 6],
            x: vec![0.0; 5],
            y: vec![0.0; 5],
        }
    }

    #[test]
    fn common_filter_is_minimum_length_continuity_and_node_cycle_exactly() {
        let graph = graph();
        assert_eq!(
            validate_common_path(&[0, 1, 2], &graph, 5),
            Err(CommonPathError::TooShort)
        );
        assert_eq!(
            validate_common_path(&[0, 3, 4, 0, 1], &graph, 5),
            Err(CommonPathError::Discontinuous)
        );
        assert_eq!(
            validate_common_path(&[0, 1, 2, 99, 1], &graph, 5),
            Err(CommonPathError::OutOfBounds)
        );
        assert_eq!(
            validate_common_path(&[5, 1, 2, 3, 4], &graph, 5),
            Err(CommonPathError::Cyclic)
        );
    }

    #[test]
    fn upstream_parallel_edge_condensation_is_detectable_not_applied() {
        let canonical = canonical_parallel_edges(&graph());
        assert_eq!(canonical[0], 5);
        assert_eq!(canonical[5], 5);
        assert_eq!(canonical[1], 1);
    }

    #[test]
    fn jsonl_and_training_pickle_preserve_identical_ids_order_and_edges() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "edge-weight-common-manifest-{}-{nonce}",
            std::process::id()
        ));
        let manifest = root.join("manifest.jsonl");
        let pickle = root.join("trips.pkl");
        let trips = vec![
            CommonTrip {
                manifest_id: "validation:000000001".to_string(),
                source_index: 1,
                original_trip_id: "original-a".to_string(),
                edges: vec![1, 2, 3, 4, 5],
                start_time: 10,
                end_time: 20,
            },
            CommonTrip {
                manifest_id: "validation:000000009".to_string(),
                source_index: 9,
                original_trip_id: "original-b".to_string(),
                edges: vec![8, 7, 6, 5, 4],
                start_time: 30,
                end_time: 40,
            },
        ];
        write_common_manifest(&trips, &manifest, &pickle).unwrap();

        let rows = BufReader::new(File::open(&manifest).unwrap())
            .lines()
            .map(|line| serde_json::from_str::<Value>(&line.unwrap()).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(rows[0]["manifest_id"], "validation:000000001");
        assert_eq!(rows[1]["edges"], json!([8, 7, 6, 5, 4]));
        let decoded: Vec<(String, Vec<usize>, (u64, u64))> =
            serde_pickle::from_reader(File::open(&pickle).unwrap(), Default::default()).unwrap();
        assert_eq!(
            decoded[0],
            (
                trips[0].manifest_id.clone(),
                trips[0].edges.clone(),
                (10, 20)
            )
        );
        assert_eq!(
            decoded[1],
            (
                trips[1].manifest_id.clone(),
                trips[1].edges.clone(),
                (30, 40)
            )
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
