//! Shared planar turn geometry used by diagnostics and turn-aware probes.

use crate::graph::GraphData;
use serde_json::{Value, json};

/// Coarse planar turn class under the graph's `(x, y)` coordinate system.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TurnCategory {
    Left,
    Straight,
    Right,
    UTurn,
}

impl TurnCategory {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Left => "left",
            Self::Straight => "straight",
            Self::Right => "right",
            Self::UTurn => "uturn",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TurnGeometry {
    /// Signed angle in degrees in `[-180, 180]`; positive is counter-clockwise.
    pub signed_angle_degrees: f64,
    pub category: TurnCategory,
}

impl TurnGeometry {
    pub(crate) fn to_json(self) -> Value {
        json!({
            "signed_angle_degrees": self.signed_angle_degrees,
            "category": self.category.as_str(),
        })
    }
}

/// Classify the planar turn from `incoming_edge` to `outgoing_edge`.
///
/// Returns `None` for a discontinuous pair, missing coordinates, non-finite
/// coordinates, or a zero-length edge. Straight includes 30 degrees and U-turn
/// includes 150 degrees, so a left turn is exactly `(30, 150)` degrees.
pub fn turn_geometry(
    graph: &GraphData,
    incoming_edge: usize,
    outgoing_edge: usize,
) -> Option<TurnGeometry> {
    let incoming_tail = *graph.tail.get(incoming_edge)? as usize;
    let junction = *graph.head.get(incoming_edge)? as usize;
    if graph.tail.get(outgoing_edge).copied()? as usize != junction {
        return None;
    }
    let outgoing_head = *graph.head.get(outgoing_edge)? as usize;
    let incoming = (
        *graph.x.get(junction)? as f64 - *graph.x.get(incoming_tail)? as f64,
        *graph.y.get(junction)? as f64 - *graph.y.get(incoming_tail)? as f64,
    );
    let outgoing = (
        *graph.x.get(outgoing_head)? as f64 - *graph.x.get(junction)? as f64,
        *graph.y.get(outgoing_head)? as f64 - *graph.y.get(junction)? as f64,
    );
    if !incoming.0.is_finite()
        || !incoming.1.is_finite()
        || !outgoing.0.is_finite()
        || !outgoing.1.is_finite()
    {
        return None;
    }
    let incoming_norm_squared = incoming.0 * incoming.0 + incoming.1 * incoming.1;
    let outgoing_norm_squared = outgoing.0 * outgoing.0 + outgoing.1 * outgoing.1;
    if incoming_norm_squared <= f64::EPSILON || outgoing_norm_squared <= f64::EPSILON {
        return None;
    }
    let cross = incoming.0 * outgoing.1 - incoming.1 * outgoing.0;
    let dot = incoming.0 * outgoing.0 + incoming.1 * outgoing.1;
    let signed_angle_degrees = cross.atan2(dot).to_degrees();
    let absolute_angle = signed_angle_degrees.abs();
    let category = if absolute_angle <= 30.0 {
        TurnCategory::Straight
    } else if absolute_angle >= 150.0 {
        TurnCategory::UTurn
    } else if signed_angle_degrees > 0.0 {
        TurnCategory::Left
    } else {
        TurnCategory::Right
    };
    Some(TurnGeometry {
        signed_angle_degrees,
        category,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph() -> GraphData {
        GraphData {
            tail: vec![0, 1, 1, 1, 1, 5],
            head: vec![1, 2, 3, 4, 0, 4],
            baseline_weights: vec![1; 6],
            x: vec![0.0, 1.0, 2.0, 1.0, 0.0, 9.0],
            y: vec![0.0, 0.0, 0.0, 1.0, 0.0, 9.0],
        }
    }

    #[test]
    fn classifies_turns_with_frozen_boundaries() {
        let graph = graph();
        assert_eq!(
            turn_geometry(&graph, 0, 1).unwrap().category,
            TurnCategory::Straight
        );
        assert_eq!(
            turn_geometry(&graph, 0, 2).unwrap().category,
            TurnCategory::Left
        );
        assert_eq!(
            turn_geometry(&graph, 0, 3).unwrap().category,
            TurnCategory::UTurn
        );
        assert_eq!(
            turn_geometry(&graph, 0, 4).unwrap().category,
            TurnCategory::UTurn
        );
        assert_eq!(turn_geometry(&graph, 0, 5), None);
    }

    #[test]
    fn rejects_degenerate_geometry() {
        let mut graph = graph();
        graph.x[2] = graph.x[1];
        graph.y[2] = graph.y[1];
        assert_eq!(turn_geometry(&graph, 0, 1), None);
    }
}
