use super::EdgeOnlyModel;
use crate::optimizer::{
    ProjectedStepStats, ProjectedSubgradientOptimizer, TurnResidualOptimizer,
    TurnResidualStepStats, TurnResidualSummary,
};
use crate::turn_graph::ExpandedTurnGraph;

/// Per-transition residual model
/// `kappa_(e,f) = edge_weight[f] + residual_scale * r_(e,f)`.
///
/// Residuals are continuous, nonnegative, and initialized to zero, so the
/// initial expanded metric is exactly the edge-only model.
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
        let transition_residuals = vec![0.0; expanded.transition_count()];
        Self::from_residuals(edge_only, expanded, residual_scale, &transition_residuals)
    }

    /// Strictly restore residuals without projection or implicit repair.
    pub fn from_residuals(
        edge_only: EdgeOnlyModel,
        expanded: &ExpandedTurnGraph,
        residual_scale: f64,
        transition_residuals: &[f64],
    ) -> Result<Self, String> {
        if !residual_scale.is_finite() || residual_scale <= 0.0 {
            return Err("residual scale must be finite and greater than zero".to_string());
        }
        if edge_only.metric_baseline().len() != expanded.stats.original_edges {
            return Err(format!(
                "edge model has {} edges but expanded graph has {} states",
                edge_only.metric_baseline().len(),
                expanded.stats.original_edges
            ));
        }
        if transition_residuals.len() != expanded.transition_count() {
            return Err(format!(
                "residual count {} does not match expanded transition count {}",
                transition_residuals.len(),
                expanded.transition_count()
            ));
        }
        for (transition, &residual) in transition_residuals.iter().enumerate() {
            if !residual.is_finite() || residual < 0.0 {
                return Err(format!(
                    "r[{transition}] must be finite and nonnegative, got {residual}"
                ));
            }
        }

        let model = Self {
            edge_only,
            transition_residuals: transition_residuals.to_vec(),
            residual_scale,
        };
        model.quantized_transition_weights(expanded)?;
        Ok(model)
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

    pub fn quantized_edge_weights(&self) -> Result<Vec<u32>, String> {
        self.edge_only.quantized_weights()
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

    /// Apply one edge block update while retaining the turn state.
    pub fn projected_edge_step(
        &mut self,
        optimizer: &mut ProjectedSubgradientOptimizer,
        observed: &[u64],
        predicted: &[u64],
        sample_count: u64,
    ) -> ProjectedStepStats {
        self.edge_only
            .projected_step(optimizer, observed, predicted, sample_count)
    }

    /// Apply one transition block update while retaining the edge state.
    pub fn projected_residual_step(
        &mut self,
        optimizer: &mut TurnResidualOptimizer,
        observed: &[u64],
        predicted: &[u64],
        sample_count: u64,
    ) -> TurnResidualStepStats {
        optimizer.step(
            &mut self.transition_residuals,
            observed,
            predicted,
            sample_count,
            self.residual_scale,
        )
    }

    /// `lambda_turn / (2|T|) * ||r||^2`.
    pub fn residual_regularization(&self, lambda_turn: f64) -> f64 {
        if self.transition_residuals.is_empty() {
            return 0.0;
        }
        let squared_norm = self
            .transition_residuals
            .iter()
            .map(|residual| residual * residual)
            .sum::<f64>();
        lambda_turn * squared_norm / (2.0 * self.transition_residuals.len() as f64)
    }

    pub fn residual_summary(&self) -> TurnResidualSummary {
        let transitions = self.transition_residuals.len();
        if transitions == 0 {
            return TurnResidualSummary {
                transitions: 0,
                zero_transitions: 0,
                positive_transitions: 0,
                mean: 0.0,
                max: 0.0,
                l2_norm: 0.0,
            };
        }

        let zero_transitions = self
            .transition_residuals
            .iter()
            .filter(|&&residual| residual == 0.0)
            .count();
        let sum = self.transition_residuals.iter().sum::<f64>();
        let squared_norm = self
            .transition_residuals
            .iter()
            .map(|residual| residual * residual)
            .sum::<f64>();
        TurnResidualSummary {
            transitions,
            zero_transitions,
            positive_transitions: transitions - zero_transitions,
            mean: sum / transitions as f64,
            max: self
                .transition_residuals
                .iter()
                .copied()
                .fold(0.0, f64::max),
            l2_norm: squared_norm.sqrt(),
        }
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
        assert!(TurnAwareModel::new(edge_only.clone(), &expanded, 0.0).is_err());
        assert!(TurnAwareModel::new(edge_only, &expanded, -1.0).is_err());

        let wrong_edges = EdgeOnlyModel::new(&[1, 2], 1.0).unwrap();
        assert!(TurnAwareModel::new(wrong_edges, &expanded, 1.0).is_err());
    }

    #[test]
    fn strictly_restores_nonnegative_residuals() {
        let graph = graph();
        let expanded = ExpandedTurnGraph::build(&graph).unwrap();
        let edge_only =
            EdgeOnlyModel::from_q(&graph.baseline_weights, 1.0, &[0.8, 1.2, 1.0, 1.0]).unwrap();
        let residuals = vec![0.25, 0.0];
        let model = TurnAwareModel::from_residuals(edge_only, &expanded, 4.0, &residuals).unwrap();

        assert_eq!(model.edge_only().q(), &[0.8, 1.2, 1.0, 1.0]);
        assert_eq!(model.transition_residuals(), residuals);
        assert_eq!(model.quantized_edge_weights().unwrap(), vec![4, 6, 2, 2]);
        assert_eq!(
            model.quantized_transition_weights(&expanded).unwrap(),
            vec![7, 2]
        );
        assert!(
            TurnAwareModel::from_residuals(model.edge_only().clone(), &expanded, 4.0, &[0.0])
                .is_err()
        );
        assert!(
            TurnAwareModel::from_residuals(
                model.edge_only().clone(),
                &expanded,
                4.0,
                &[f64::NAN, 0.0]
            )
            .is_err()
        );
        assert!(
            TurnAwareModel::from_residuals(model.edge_only().clone(), &expanded, 4.0, &[-0.1, 0.0])
                .is_err()
        );
    }

    #[test]
    fn updates_edge_and_turn_blocks_independently() {
        let graph = graph();
        let expanded = ExpandedTurnGraph::build(&graph).unwrap();
        let edge_only = EdgeOnlyModel::new(&graph.baseline_weights, 1.0).unwrap();
        let mut model = TurnAwareModel::new(edge_only, &expanded, 10.0).unwrap();
        let mut edge_optimizer = ProjectedSubgradientOptimizer::new(0.01, 0.0, 0.1, 10.0).unwrap();
        let mut turn_optimizer = TurnResidualOptimizer::new(0.1, 0.0, 10.0).unwrap();

        model.projected_edge_step(&mut edge_optimizer, &[1, 0, 0, 0], &[0, 1, 0, 0], 1);
        model.projected_residual_step(&mut turn_optimizer, &[0, 1], &[1, 0], 1);

        assert!(model.edge_only().q()[0] < 1.0);
        assert!(model.edge_only().q()[1] > 1.0);
        assert_eq!(model.transition_residuals(), &[1.0, 0.0]);
        assert_eq!(edge_optimizer.completed_updates(), 1);
        assert_eq!(turn_optimizer.completed_updates(), 1);
    }

    #[test]
    fn computes_residual_regularization_and_summary() {
        let graph = graph();
        let expanded = ExpandedTurnGraph::build(&graph).unwrap();
        let edge_only = EdgeOnlyModel::new(&graph.baseline_weights, 1.0).unwrap();
        let model = TurnAwareModel::from_residuals(edge_only, &expanded, 1.0, &[3.0, 4.0]).unwrap();

        assert_eq!(model.residual_regularization(2.0), 12.5);
        let summary = model.residual_summary();
        assert_eq!(summary.transitions, 2);
        assert_eq!(summary.zero_transitions, 0);
        assert_eq!(summary.positive_transitions, 2);
        assert_eq!(summary.mean, 3.5);
        assert_eq!(summary.max, 4.0);
        assert_eq!(summary.l2_norm, 5.0);
    }
}
