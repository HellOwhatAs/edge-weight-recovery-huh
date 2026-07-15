pub(crate) mod cch;
mod dijkstra;

pub use cch::CCH_INFINITY;
pub use dijkstra::{DijkstraPath, shortest_path_f64, shortest_path_multi_source_f64};
