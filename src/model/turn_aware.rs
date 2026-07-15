use super::EdgeOnlyModel;
use crate::turn_graph::ExpandedTurnGraph;

/// Structural placeholder for the per-transition residual model.
///
/// Residuals are nonnegative and initialized to zero, so the initial expanded
/// metric is exactly the edge-only model. This type intentionally provides no
/// residual optimizer or training update in the cleanup phase.
#[derive(Clone, Debug)]
pub struct TurnAwareModel {
    edge_only: EdgeOnlyModel,
    transition_residuals: Vec<f64>,
    residual_scale: f64,
}

impl TurnAwareModel {
    pub fn new(
        edge_only: EdgeOnlyModel,
        expanded: &ExpandedTurnGraph,
        residual_scale: f64,
    ) -> Result<Self, String> {
        if !residual_scale.is_finite() || residual_scale < 0.0 {
            return Err("residual scale must be finite and nonnegative".to_string());
        }
        if edge_only.metric_baseline().len() != expanded.stats.original_edges {
            return Err(format!(
                "edge model has {} edges but expanded graph has {} states",
                edge_only.metric_baseline().len(),
                expanded.stats.original_edges
            ));
        }
        Ok(Self {
            edge_only,
            transition_residuals: vec![0.0; expanded.transition_count()],
            residual_scale,
        })
    }

    pub fn edge_only(&self) -> &EdgeOnlyModel {
        &self.edge_only
    }

    pub fn transition_residuals(&self) -> &[f64] {
        &self.transition_residuals
    }

    pub fn residual_scale(&self) -> f64 {
        self.residual_scale
    }

    pub fn quantized_transition_weights(
        &self,
        expanded: &ExpandedTurnGraph,
    ) -> Result<Vec<u32>, String> {
        let edge_weights = self.edge_only.quantized_weights()?;
        expanded.transition_metric_weights(
            &edge_weights,
            &self.transition_residuals,
            self.residual_scale,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::GraphData;

    fn graph() -> GraphData {
        GraphData {
            tail: vec![0, 1, 0, 2],
            head: vec![1, 3, 2, 3],
            baseline_weights: vec![5, 5, 2, 2],
            x: vec![0.0, 1.0, 1.0, 2.0],
            y: vec![0.0, 0.0, 1.0, 0.0],
        }
    }

    #[test]
    fn zero_residuals_reproduce_next_edge_weights() {
        let graph = graph();
        let expanded = ExpandedTurnGraph::build(&graph).unwrap();
        let edge_only = EdgeOnlyModel::new(&graph.baseline_weights, 1.0).unwrap();
        let model = TurnAwareModel::new(edge_only, &expanded, 1_000.0).unwrap();

        assert_eq!(
            model.transition_residuals(),
            vec![0.0; expanded.transition_count()]
        );
        let transition_weights = model.quantized_transition_weights(&expanded).unwrap();
        for (transition, _, next_edge) in expanded.transitions() {
            assert_eq!(
                transition_weights[transition.index()],
                graph.baseline_weights[next_edge]
            );
        }
    }

    #[test]
    fn validates_scale_and_edge_state_count() {
        let graph = graph();
        let expanded = ExpandedTurnGraph::build(&graph).unwrap();
        let edge_only = EdgeOnlyModel::new(&graph.baseline_weights, 1.0).unwrap();
        assert!(TurnAwareModel::new(edge_only.clone(), &expanded, f64::NAN).is_err());
        assert!(TurnAwareModel::new(edge_only, &expanded, -1.0).is_err());

        let wrong_edges = EdgeOnlyModel::new(&[1, 2], 1.0).unwrap();
        assert!(TurnAwareModel::new(wrong_edges, &expanded, 1.0).is_err());
    }
}
