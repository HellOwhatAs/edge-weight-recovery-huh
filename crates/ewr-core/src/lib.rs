//! Stable domain and inverse-shortest-path algorithm boundary.
//!
//! This crate accepts typed in-memory values. Dataset paths, serialization,
//! experiment protocols, and concrete baseline implementations belong outside
//! this dependency boundary.
//!
//! Applications should normally use the high-level `ewr-cch::fit` facade and
//! the domain/model types re-exported here. `LineGraph`, `RoutingTopology`, and
//! the oracle values form a narrow backend SPI for routing adapters; objective
//! and optimizer implementations remain crate-private.

mod line_graph;
mod model;
mod network;
mod objective;
mod optimizer;
mod oracle;
mod trainer;

pub use line_graph::{BoundKind, LineGraph, LineGraphError, RoutingTopology};
pub use model::{
    FitDiagnostics, FitOptions, FitResult, ModelError, TopologyId, Transition,
    TransitionWeightModel,
};
pub use network::{EdgeId, NetworkError, NodeId, RoadNetwork, Trajectory};
pub use oracle::{
    OracleError, OraclePath, OracleQuery, QueryEndpoint, ROUTING_INFINITY, RoutingOracle,
};
pub use trainer::{Trainer, TrainerError, TrainingOutcome, TrainingState, fit};
