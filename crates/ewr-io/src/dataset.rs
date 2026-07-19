use crate::DatasetPaths;
use ewr_core::{EdgeId, NetworkError, NodeId, RoadNetwork, Trajectory};
use shapefile::dbase::{FieldValue, Record};
use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Production inputs after all file formats have been eliminated.
#[derive(Clone, Debug, PartialEq)]
pub struct LoadedDataset {
    /// Typed directed road network.
    pub network: RoadNetwork,
    /// Structurally valid complete original-edge trajectories.
    pub trajectories: Vec<Trajectory>,
    /// Deterministic accounting of accepted and dropped raw observations.
    pub report: LoadReport,
}

/// Structural trajectory-filtering outcome.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LoadReport {
    /// Number of rows present in the pickle.
    pub available: usize,
    /// Number of valid trajectories returned to the core.
    pub accepted: usize,
    /// Empty edge sequences.
    pub empty: usize,
    /// Single-edge sequences, which have no learned transition under v1.
    pub too_short: usize,
    /// Sequences containing an unknown or unrepresentable edge ID.
    pub out_of_bounds: usize,
    /// Sequences whose consecutive original edges do not connect.
    pub discontinuous: usize,
    /// Sequences that revisit an original node.
    pub cyclic: usize,
}

impl LoadReport {
    /// Number of raw observations omitted from the typed result.
    pub fn dropped(self) -> usize {
        self.available - self.accepted
    }
}

/// Load a road network and historical trajectories from explicit paths.
pub fn load_dataset(paths: &DatasetPaths) -> Result<LoadedDataset, DatasetError> {
    let network = load_network(&paths.nodes, &paths.edges)?;
    let file =
        File::open(&paths.trajectories).map_err(|source| DatasetError::OpenTrajectories {
            path: paths.trajectories.clone(),
            source,
        })?;
    let (trajectories, report) =
        decode_trajectories(file, &network).map_err(|source| DatasetError::DecodeTrajectories {
            path: paths.trajectories.clone(),
            source,
        })?;
    Ok(LoadedDataset {
        network,
        trajectories,
        report,
    })
}

/// Load only the directed road network from explicit node and edge Shapefiles.
///
/// This adapter deliberately does not open or decode any trajectory file. It is
/// intended for inference and research baselines whose network topology is the
/// only dataset input they require.
pub fn load_network(nodes: &Path, edges: &Path) -> Result<RoadNetwork, DatasetError> {
    let edges = load_edges(edges)?;
    let nodes = load_nodes(nodes)?;
    let arrays = build_graph_arrays(&nodes, &edges)?;

    let mut baseline_weights = Vec::with_capacity(arrays.weight.len());
    for (edge, length) in arrays.weight.into_iter().enumerate() {
        baseline_weights.push(scale_baseline(edge, length)?);
    }

    let tail = arrays
        .tail
        .into_iter()
        .enumerate()
        .map(|(edge, node)| {
            u32::try_from(node)
                .map(NodeId::new)
                .map_err(|_| DatasetError::EndpointOutOfRange {
                    edge,
                    endpoint: "tail",
                    node,
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let head = arrays
        .head
        .into_iter()
        .enumerate()
        .map(|(edge, node)| {
            u32::try_from(node)
                .map(NodeId::new)
                .map_err(|_| DatasetError::EndpointOutOfRange {
                    edge,
                    endpoint: "head",
                    node,
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let x = arrays.xs.into_iter().map(|value| value as f32).collect();
    let y = arrays.ys.into_iter().map(|value| value as f32).collect();
    RoadNetwork::new(tail, head, baseline_weights, x, y).map_err(DatasetError::InvalidNetwork)
}

#[derive(Clone, Debug, PartialEq)]
struct EdgeAttributes {
    fid: u64,
    tail_osmid: u64,
    head_osmid: u64,
    length: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct NodeAttributes {
    osmid: u64,
    x: f64,
    y: f64,
}

#[derive(Debug, PartialEq)]
struct GraphArrays {
    xs: Vec<f64>,
    ys: Vec<f64>,
    tail: Vec<usize>,
    head: Vec<usize>,
    weight: Vec<f64>,
}

fn load_edges(path: &Path) -> Result<Vec<EdgeAttributes>, DatasetError> {
    let mut reader =
        shapefile::Reader::from_path(path).map_err(|source| DatasetError::LoadEdges {
            path: path.to_path_buf(),
            source,
        })?;
    let mut edges = Vec::new();
    for (record_index, item) in reader.iter_shapes_and_records().enumerate() {
        let (_shape, record) = item.map_err(|source| DatasetError::LoadEdges {
            path: path.to_path_buf(),
            source,
        })?;
        edges.push(EdgeAttributes {
            fid: required_unsigned(&record, "fid", "edge", record_index)?,
            tail_osmid: required_unsigned(&record, "u", "edge", record_index)?,
            head_osmid: required_unsigned(&record, "v", "edge", record_index)?,
            length: required_f64(&record, "length", "edge", record_index)?,
        });
    }
    Ok(edges)
}

fn load_nodes(path: &Path) -> Result<Vec<NodeAttributes>, DatasetError> {
    let mut reader =
        shapefile::Reader::from_path(path).map_err(|source| DatasetError::LoadNodes {
            path: path.to_path_buf(),
            source,
        })?;
    let mut nodes = Vec::new();
    for (record_index, item) in reader.iter_shapes_and_records().enumerate() {
        let (shape, record) = item.map_err(|source| DatasetError::LoadNodes {
            path: path.to_path_buf(),
            source,
        })?;
        let (x, y) = match shape {
            shapefile::Shape::Point(point) => (point.x, point.y),
            shapefile::Shape::PointZ(point) => (point.x, point.y),
            _ => continue,
        };
        nodes.push(NodeAttributes {
            osmid: required_unsigned(&record, "osmid", "node", record_index)?,
            x,
            y,
        });
    }
    Ok(nodes)
}

fn unsigned(record: &Record, field: &str) -> Option<u64> {
    match record.get(field)? {
        FieldValue::Numeric(Some(number))
            if number.is_finite()
                && *number >= 0.0
                && number.fract() == 0.0
                && *number < u64::MAX as f64 =>
        {
            Some(*number as u64)
        }
        FieldValue::Character(Some(value)) => value.parse().ok(),
        _ => None,
    }
}

fn numeric(record: &Record, field: &str) -> Option<f64> {
    match record.get(field)? {
        FieldValue::Numeric(value) => *value,
        FieldValue::Character(Some(value)) => value.parse().ok(),
        _ => None,
    }
}

fn required_unsigned(
    record: &Record,
    field: &'static str,
    record_kind: &'static str,
    record_index: usize,
) -> Result<u64, DatasetError> {
    unsigned(record, field).ok_or(DatasetError::MissingRequiredField {
        record_kind,
        record_index,
        field,
    })
}

fn required_f64(
    record: &Record,
    field: &'static str,
    record_kind: &'static str,
    record_index: usize,
) -> Result<f64, DatasetError> {
    numeric(record, field).ok_or(DatasetError::MissingRequiredField {
        record_kind,
        record_index,
        field,
    })
}

fn build_graph_arrays(
    nodes: &[NodeAttributes],
    edges: &[EdgeAttributes],
) -> Result<GraphArrays, DatasetError> {
    validate_edge_fids(edges)?;

    let mut node_index_by_osmid = HashMap::with_capacity(nodes.len());
    let mut xs = Vec::with_capacity(nodes.len());
    let mut ys = Vec::with_capacity(nodes.len());
    for (node_index, node) in nodes.iter().enumerate() {
        if let Some(first_node) = node_index_by_osmid.insert(node.osmid, node_index) {
            return Err(DatasetError::DuplicateNodeOsmid {
                osmid: node.osmid,
                first_node,
                duplicate_node: node_index,
            });
        }
        xs.push(node.x);
        ys.push(node.y);
    }

    let mut tail = Vec::with_capacity(edges.len());
    let mut head = Vec::with_capacity(edges.len());
    let mut weight = Vec::with_capacity(edges.len());
    for (edge_index, edge) in edges.iter().enumerate() {
        let &tail_index =
            node_index_by_osmid
                .get(&edge.tail_osmid)
                .ok_or(DatasetError::MissingEdgeEndpoint {
                    edge: edge_index,
                    endpoint: "tail",
                    osmid: edge.tail_osmid,
                })?;
        let &head_index =
            node_index_by_osmid
                .get(&edge.head_osmid)
                .ok_or(DatasetError::MissingEdgeEndpoint {
                    edge: edge_index,
                    endpoint: "head",
                    osmid: edge.head_osmid,
                })?;
        tail.push(tail_index);
        head.push(head_index);
        weight.push(edge.length);
    }

    Ok(GraphArrays {
        xs,
        ys,
        tail,
        head,
        weight,
    })
}

fn validate_edge_fids(edges: &[EdgeAttributes]) -> Result<(), DatasetError> {
    let mut first_record_by_fid = HashMap::with_capacity(edges.len());
    for (record_index, edge) in edges.iter().enumerate() {
        if let Some(first_record) = first_record_by_fid.insert(edge.fid, record_index) {
            return Err(DatasetError::DuplicateEdgeFid {
                fid: edge.fid,
                first_record,
                duplicate_record: record_index,
            });
        }
    }
    for (record_index, edge) in edges.iter().enumerate() {
        if usize::try_from(edge.fid) != Ok(record_index) {
            return Err(DatasetError::EdgeFidMismatch {
                record_index,
                fid: edge.fid,
            });
        }
    }
    Ok(())
}

fn scale_baseline(edge: usize, length: f64) -> Result<f64, DatasetError> {
    let scaled = length * 1_000.0;
    if !scaled.is_finite() || scaled <= 0.0 || scaled >= f64::from(i32::MAX) {
        return Err(DatasetError::InvalidScaledBaseline { edge, scaled });
    }
    Ok(scaled.round().max(1.0))
}

type RawTrajectory = (serde_pickle::Value, Vec<usize>, (u64, u64));

fn decode_trajectories(
    reader: impl Read,
    network: &RoadNetwork,
) -> Result<(Vec<Trajectory>, LoadReport), serde_pickle::Error> {
    let raw: Vec<RawTrajectory> = serde_pickle::from_reader(reader, Default::default())?;
    let mut report = LoadReport {
        available: raw.len(),
        ..LoadReport::default()
    };
    let mut trajectories = Vec::with_capacity(raw.len());

    for (_, raw_edges, _) in raw {
        if raw_edges.is_empty() {
            report.empty += 1;
            continue;
        }
        if raw_edges.len() < 2 {
            report.too_short += 1;
            continue;
        }
        let Some(edges) = raw_edges
            .into_iter()
            .map(|edge| u32::try_from(edge).ok().map(EdgeId::new))
            .collect::<Option<Vec<_>>>()
        else {
            report.out_of_bounds += 1;
            continue;
        };
        let trajectory = Trajectory::new(edges);
        match network.validate_trajectory(&trajectory) {
            Ok(_) => {
                trajectories.push(trajectory);
                report.accepted += 1;
            }
            Err(NetworkError::TrajectoryTooShort(_)) => report.too_short += 1,
            Err(NetworkError::TrajectoryEdgeOutOfBounds(_)) => report.out_of_bounds += 1,
            Err(NetworkError::DiscontinuousTrajectory { .. }) => report.discontinuous += 1,
            Err(NetworkError::CyclicTrajectory(_)) => report.cyclic += 1,
            Err(
                NetworkError::InvalidNodeArrays { .. }
                | NetworkError::InvalidEdgeArrays { .. }
                | NetworkError::TooManyNodes(_)
                | NetworkError::TooManyEdges(_)
                | NetworkError::InvalidCoordinates { .. }
                | NetworkError::EndpointOutOfBounds { .. }
                | NetworkError::InvalidBaselineWeight { .. },
            ) => unreachable!("the network was validated before trajectory filtering"),
        }
    }

    debug_assert_eq!(
        report.accepted
            + report.empty
            + report.too_short
            + report.out_of_bounds
            + report.discontinuous
            + report.cyclic,
        report.available
    );
    Ok((trajectories, report))
}

/// Failure while adapting production dataset files into core values.
#[derive(Debug)]
pub enum DatasetError {
    /// Edge Shapefile loading failed.
    LoadEdges {
        /// Input path.
        path: PathBuf,
        /// Underlying Shapefile/DBF error.
        source: shapefile::Error,
    },
    /// Node Shapefile loading failed.
    LoadNodes {
        /// Input path.
        path: PathBuf,
        /// Underlying Shapefile/DBF error.
        source: shapefile::Error,
    },
    /// A required DBF field is missing or has an unsupported value type.
    MissingRequiredField {
        /// Either `node` or `edge`.
        record_kind: &'static str,
        /// Zero-based Shapefile/DBF record index.
        record_index: usize,
        /// Required field name.
        field: &'static str,
    },
    /// Two edge records declare the same stable feature ID.
    DuplicateEdgeFid {
        /// Repeated DBF `fid` value.
        fid: u64,
        /// First record containing the value.
        first_record: usize,
        /// Later record containing the value.
        duplicate_record: usize,
    },
    /// An edge DBF `fid` differs from its zero-based Shapefile record index.
    EdgeFidMismatch {
        /// Zero-based Shapefile/DBF record index.
        record_index: usize,
        /// DBF `fid` value found at that record.
        fid: u64,
    },
    /// Two node records declare the same OSM identity.
    DuplicateNodeOsmid {
        /// Repeated DBF `osmid` value.
        osmid: u64,
        /// First node-array index containing the value.
        first_node: usize,
        /// Later node-array index containing the value.
        duplicate_node: usize,
    },
    /// An edge references an OSM node absent from the node Shapefile.
    MissingEdgeEndpoint {
        /// Stable original edge index.
        edge: usize,
        /// Either `tail` or `head`.
        endpoint: &'static str,
        /// Missing node DBF `osmid` value.
        osmid: u64,
    },
    /// A scaled baseline violates the frozen positive-u32-compatible policy.
    InvalidScaledBaseline {
        /// Stable original edge index.
        edge: usize,
        /// Length after multiplying by 1000.
        scaled: f64,
    },
    /// A Shapefile endpoint cannot be represented by the core ID type.
    EndpointOutOfRange {
        /// Stable original edge index.
        edge: usize,
        /// Either `tail` or `head`.
        endpoint: &'static str,
        /// Unrepresentable node index.
        node: usize,
    },
    /// Core road-network validation failed.
    InvalidNetwork(NetworkError),
    /// The trajectory pickle could not be opened.
    OpenTrajectories {
        /// Input path.
        path: PathBuf,
        /// Underlying filesystem error.
        source: std::io::Error,
    },
    /// The trajectory pickle does not implement the current raw tuple schema.
    DecodeTrajectories {
        /// Input path.
        path: PathBuf,
        /// Underlying pickle error.
        source: serde_pickle::Error,
    },
}

impl Display for DatasetError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LoadEdges { path, source } => {
                write!(
                    formatter,
                    "failed to load edges from {}: {source}",
                    path.display()
                )
            }
            Self::LoadNodes { path, source } => {
                write!(
                    formatter,
                    "failed to load nodes from {}: {source}",
                    path.display()
                )
            }
            Self::MissingRequiredField {
                record_kind,
                record_index,
                field,
            } => write!(
                formatter,
                "{record_kind} record {record_index} is missing required DBF field '{field}' or has an unsupported value"
            ),
            Self::DuplicateEdgeFid {
                fid,
                first_record,
                duplicate_record,
            } => write!(
                formatter,
                "edge DBF fid {fid} is duplicated at records {first_record} and {duplicate_record}"
            ),
            Self::EdgeFidMismatch { record_index, fid } => write!(
                formatter,
                "edge record {record_index} has DBF fid {fid}; fid must equal the zero-based record index"
            ),
            Self::DuplicateNodeOsmid {
                osmid,
                first_node,
                duplicate_node,
            } => write!(
                formatter,
                "node osmid {osmid} is duplicated at node indices {first_node} and {duplicate_node}"
            ),
            Self::MissingEdgeEndpoint {
                edge,
                endpoint,
                osmid,
            } => write!(
                formatter,
                "edge {edge} {endpoint} references missing node osmid {osmid}"
            ),
            Self::InvalidScaledBaseline { edge, scaled } => {
                write!(
                    formatter,
                    "edge {edge} has invalid scaled baseline {scaled}"
                )
            }
            Self::EndpointOutOfRange {
                edge,
                endpoint,
                node,
            } => write!(
                formatter,
                "edge {edge} {endpoint} node index {node} does not fit the core ID type"
            ),
            Self::InvalidNetwork(source) => write!(formatter, "invalid road network: {source}"),
            Self::OpenTrajectories { path, source } => write!(
                formatter,
                "failed to open trajectories from {}: {source}",
                path.display()
            ),
            Self::DecodeTrajectories { path, source } => write!(
                formatter,
                "failed to decode trajectories from {}: {source}",
                path.display()
            ),
        }
    }
}

impl Error for DatasetError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::LoadEdges { source, .. } | Self::LoadNodes { source, .. } => Some(source),
            Self::InvalidNetwork(source) => Some(source),
            Self::OpenTrajectories { source, .. } => Some(source),
            Self::DecodeTrajectories { source, .. } => Some(source),
            Self::MissingRequiredField { .. }
            | Self::DuplicateEdgeFid { .. }
            | Self::EdgeFidMismatch { .. }
            | Self::DuplicateNodeOsmid { .. }
            | Self::MissingEdgeEndpoint { .. }
            | Self::InvalidScaledBaseline { .. }
            | Self::EndpointOutOfRange { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn graph_nodes() -> Vec<NodeAttributes> {
        vec![
            NodeAttributes {
                osmid: 40,
                x: 1.0,
                y: 2.0,
            },
            NodeAttributes {
                osmid: 20,
                x: 3.0,
                y: 4.0,
            },
            NodeAttributes {
                osmid: 80,
                x: 5.0,
                y: 6.0,
            },
        ]
    }

    fn graph_edge(fid: u64, tail_osmid: u64, head_osmid: u64) -> EdgeAttributes {
        EdgeAttributes {
            fid,
            tail_osmid,
            head_osmid,
            length: fid as f64 + 1.25,
        }
    }

    fn network() -> RoadNetwork {
        RoadNetwork::new(
            vec![
                NodeId::new(0),
                NodeId::new(1),
                NodeId::new(0),
                NodeId::new(2),
                NodeId::new(1),
            ],
            vec![
                NodeId::new(1),
                NodeId::new(3),
                NodeId::new(2),
                NodeId::new(3),
                NodeId::new(0),
            ],
            vec![5.0, 5.0, 2.0, 2.0, 1.0],
            vec![0.0, 1.0, 1.0, 2.0],
            vec![0.0, 0.0, 1.0, 0.0],
        )
        .unwrap()
    }

    #[test]
    fn filters_synthetic_raw_trajectories_without_leaking_pickle_values() {
        let raw = vec![
            ("accepted", vec![0, 1], (10_u64, 20_u64)),
            ("empty", vec![], (10, 20)),
            ("short", vec![0], (10, 20)),
            ("bounds", vec![0, 99], (10, 20)),
            ("disconnected", vec![0, 3], (10, 20)),
            ("cycle", vec![0, 4], (10, 20)),
        ];
        let bytes = serde_pickle::to_vec(&raw, Default::default()).unwrap();
        let (trajectories, report) = decode_trajectories(Cursor::new(bytes), &network()).unwrap();

        assert_eq!(
            trajectories,
            vec![Trajectory::new(vec![EdgeId::new(0), EdgeId::new(1)])]
        );
        assert_eq!(
            report,
            LoadReport {
                available: 6,
                accepted: 1,
                empty: 1,
                too_short: 1,
                out_of_bounds: 1,
                discontinuous: 1,
                cyclic: 1,
            }
        );
        assert_eq!(report.dropped(), 5);
    }

    #[test]
    fn pickle_decode_failures_remain_typed() {
        assert!(decode_trajectories(Cursor::new(b"not a pickle"), &network()).is_err());
    }

    #[test]
    fn freezes_legacy_baseline_scaling() {
        assert_eq!(scale_baseline(0, 0.0001).unwrap(), 1.0);
        assert_eq!(scale_baseline(0, 0.0015).unwrap(), 2.0);
        assert_eq!(scale_baseline(0, 1.2344).unwrap(), 1_234.0);
        assert!(scale_baseline(7, 0.0).is_err());
        assert!(scale_baseline(7, f64::NAN).is_err());
    }

    #[test]
    fn dbf_identity_fields_must_be_exact_unsigned_integers() {
        let mut record = Record::default();
        record.insert("fid".to_owned(), FieldValue::Numeric(Some(0.5)));
        assert!(matches!(
            required_unsigned(&record, "fid", "edge", 0),
            Err(DatasetError::MissingRequiredField {
                record_kind: "edge",
                record_index: 0,
                field: "fid"
            })
        ));

        record.insert(
            "fid".to_owned(),
            FieldValue::Character(Some("0".to_owned())),
        );
        assert_eq!(required_unsigned(&record, "fid", "edge", 0).unwrap(), 0);
    }

    #[test]
    fn graph_arrays_preserve_record_order_and_map_osm_endpoints() {
        let arrays = build_graph_arrays(
            &graph_nodes(),
            &[graph_edge(0, 40, 20), graph_edge(1, 20, 80)],
        )
        .unwrap();

        assert_eq!(
            arrays,
            GraphArrays {
                xs: vec![1.0, 3.0, 5.0],
                ys: vec![2.0, 4.0, 6.0],
                tail: vec![0, 1],
                head: vec![1, 2],
                weight: vec![1.25, 2.25],
            }
        );
    }

    #[test]
    fn rejects_edge_fid_that_is_not_its_record_index() {
        let error = build_graph_arrays(&graph_nodes(), &[graph_edge(7, 40, 20)]).unwrap_err();

        assert!(matches!(
            error,
            DatasetError::EdgeFidMismatch {
                record_index: 0,
                fid: 7
            }
        ));
    }

    #[test]
    fn reports_duplicate_edge_fid_before_alignment_mismatch() {
        let error = build_graph_arrays(
            &graph_nodes(),
            &[graph_edge(0, 40, 20), graph_edge(0, 20, 80)],
        )
        .unwrap_err();

        assert!(matches!(
            error,
            DatasetError::DuplicateEdgeFid {
                fid: 0,
                first_record: 0,
                duplicate_record: 1
            }
        ));
    }

    #[test]
    fn rejects_duplicate_node_osmid_with_typed_indices() {
        let mut nodes = graph_nodes();
        nodes[2].osmid = 40;
        let error = build_graph_arrays(&nodes, &[graph_edge(0, 40, 20)]).unwrap_err();

        assert!(matches!(
            error,
            DatasetError::DuplicateNodeOsmid {
                osmid: 40,
                first_node: 0,
                duplicate_node: 2
            }
        ));
    }

    #[test]
    fn rejects_missing_edge_endpoint_with_typed_identity() {
        let error = build_graph_arrays(&graph_nodes(), &[graph_edge(0, 40, 999)]).unwrap_err();

        assert!(matches!(
            error,
            DatasetError::MissingEdgeEndpoint {
                edge: 0,
                endpoint: "head",
                osmid: 999
            }
        ));
    }
}
