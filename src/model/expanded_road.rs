use super::EdgeOnlyModel;
use crate::optimizer::{ExpandedProjectedStepStats, ExpandedProjectedSubgradientOptimizer};
use crate::oracle::ExpandedOracleStats;
use crate::turn_graph::ExpandedTurnGraph;

/// Integer weights reconstructed from one continuous expanded-road state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpandedMetricWeights {
    pub edge_weights: Vec<u32>,
    pub transition_weights: Vec<u32>,
}

impl ExpandedMetricWeights {
    pub fn edge_weights(&self) -> &[u32] {
        &self.edge_weights
    }

    pub fn transition_weights(&self) -> &[u32] {
        &self.transition_weights
    }
}

/// Compact diagnostics for the continuous transition-residual state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransitionResidualSummary {
    pub transitions: usize,
    pub zero_transitions: usize,
    pub positive_transitions: usize,
    pub mean: f64,
    pub max: f64,
    pub l2_norm: f64,
}

/// Expanded road model
/// `kappa_(e,f) = metric_baseline[f] * q[f] + residual_scale * r[(e,f)]`.
///
/// Edge multipliers and nonnegative transition residuals are parameters of one
/// model and are jointly optimized. At `r=0`, the reconstructed metric is
/// exactly the nested edge-only metric.
#[derive(Clone, Debug)]
pub struct ExpandedRoadModel {
    edge_only: EdgeOnlyModel,
    transition_residuals: Vec<f64>,
    residual_scale: f64,
}

impl ExpandedRoadModel {
    pub fn new(
        edge_only: EdgeOnlyModel,
        expanded: &ExpandedTurnGraph,
        residual_scale: f64,
    ) -> Result<Self, String> {
        let transition_residuals = vec![0.0; expanded.transition_count()];
        Self::from_parameters(edge_only, expanded, residual_scale, &transition_residuals)
    }

    /// Strictly restore all expanded parameters without projection, clamping,
    /// or implicit repair.
    pub fn from_parameters(
        edge_only: EdgeOnlyModel,
        expanded: &ExpandedTurnGraph,
        residual_scale: f64,
        transition_residuals: &[f64],
    ) -> Result<Self, String> {
        if !residual_scale.is_finite() || residual_scale <= 0.0 {
            return Err("residual_scale must be finite and greater than zero".to_string());
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
        model.metric(expanded)?;
        Ok(model)
    }

    pub fn q(&self) -> &[f64] {
        self.edge_only.q()
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

    /// Reconstruct both integer metric blocks from the same continuous state.
    pub fn metric(&self, expanded: &ExpandedTurnGraph) -> Result<ExpandedMetricWeights, String> {
        let edge_weights = self.edge_only.quantized_weights()?;
        let transition_weights = expanded.transition_metric_weights(
            &edge_weights,
            &self.transition_residuals,
            self.residual_scale,
        )?;
        Ok(ExpandedMetricWeights {
            edge_weights,
            transition_weights,
        })
    }

    pub fn quantized_edge_weights(&self) -> Result<Vec<u32>, String> {
        self.edge_only.quantized_weights()
    }

    pub fn quantized_transition_weights(
        &self,
        expanded: &ExpandedTurnGraph,
    ) -> Result<Vec<u32>, String> {
        Ok(self.metric(expanded)?.transition_weights)
    }

    /// Apply one joint update to `q` and `r` using edge and transition counts
    /// from the same pre-update expanded shortest-path batch.
    pub fn projected_step(
        &mut self,
        optimizer: &mut ExpandedProjectedSubgradientOptimizer,
        observed_edge_counts: &[u64],
        observed_transition_counts: &[u64],
        oracle: &ExpandedOracleStats,
    ) -> Result<ExpandedProjectedStepStats, String> {
        let Self {
            edge_only,
            transition_residuals,
            residual_scale,
        } = self;
        let (metric_baseline, q) = edge_only.optimization_state_mut();
        optimizer.step(
            q,
            transition_residuals,
            metric_baseline,
            *residual_scale,
            observed_edge_counts,
            &oracle.predicted_edge_counts,
            observed_transition_counts,
            &oracle.predicted_transition_counts,
            oracle.sample_count,
        )
    }

    /// `lambda_edge / (2|E|) * ||q - 1||^2`.
    pub fn edge_regularization(&self, lambda_edge: f64) -> f64 {
        self.edge_only.regularization(lambda_edge)
    }

    /// `lambda_transition / (2|T|) * ||r||^2`.
    pub fn transition_regularization(&self, lambda_transition: f64) -> f64 {
        if self.transition_residuals.is_empty() {
            return 0.0;
        }
        let squared_norm = self
            .transition_residuals
            .iter()
            .map(|residual| residual * residual)
            .sum::<f64>();
        lambda_transition * squared_norm / (2.0 * self.transition_residuals.len() as f64)
    }

    pub fn regularization(&self, lambda_edge: f64, lambda_transition: f64) -> f64 {
        self.edge_regularization(lambda_edge) + self.transition_regularization(lambda_transition)
    }

    pub fn transition_summary(&self) -> TransitionResidualSummary {
        let transitions = self.transition_residuals.len();
        if transitions == 0 {
            return TransitionResidualSummary {
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
        TransitionResidualSummary {
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
    use std::time::Duration;

    fn graph() -> GraphData {
        GraphData {
            tail: vec![0, 1, 0, 2],
            head: vec![1, 3, 2, 3],
            baseline_weights: vec![5, 5, 2, 2],
            x: vec![0.0, 1.0, 1.0, 2.0],
            y: vec![0.0, 0.0, 1.0, 0.0],
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn zero_residuals_reproduce_the_nested_edge_metric() {
        let graph = graph();
        let expanded = ExpandedTurnGraph::build(&graph).unwrap();
        let edge_only = EdgeOnlyModel::new(&graph.baseline_weights, 1.0).unwrap();
        let model = ExpandedRoadModel::new(edge_only, &expanded, 1_000.0).unwrap();
        let metric = model.metric(&expanded).unwrap();

        assert_eq!(model.q(), &[1.0; 4]);
        assert_eq!(
            model.transition_residuals(),
            vec![0.0; expanded.transition_count()]
        );
        for (transition, _, next_edge) in expanded.transitions() {
            assert_eq!(
                metric.transition_weights[transition.index()],
                metric.edge_weights[next_edge]
            );
        }
    }

    #[test]
    fn validates_scale_and_edge_state_count() {
        let graph = graph();
        let expanded = ExpandedTurnGraph::build(&graph).unwrap();
        let edge_only = EdgeOnlyModel::new(&graph.baseline_weights, 1.0).unwrap();
        assert!(ExpandedRoadModel::new(edge_only.clone(), &expanded, f64::NAN).is_err());
        assert!(ExpandedRoadModel::new(edge_only.clone(), &expanded, 0.0).is_err());
        assert!(ExpandedRoadModel::new(edge_only, &expanded, -1.0).is_err());

        let wrong_edges = EdgeOnlyModel::new(&[1, 2], 1.0).unwrap();
        assert!(ExpandedRoadModel::new(wrong_edges, &expanded, 1.0).is_err());
    }

    #[test]
    fn strictly_restores_continuous_parameters_and_integer_metric() {
        let graph = graph();
        let expanded = ExpandedTurnGraph::build(&graph).unwrap();
        let edge_only =
            EdgeOnlyModel::from_q(&graph.baseline_weights, 1.0, &[0.8, 1.2, 1.0, 1.0]).unwrap();
        let residuals = vec![0.25, 0.0];
        let model =
            ExpandedRoadModel::from_parameters(edge_only, &expanded, 4.0, &residuals).unwrap();
        let metric = model.metric(&expanded).unwrap();

        assert_eq!(model.q(), &[0.8, 1.2, 1.0, 1.0]);
        assert_eq!(model.transition_residuals(), residuals);
        assert_eq!(metric.edge_weights, vec![4, 6, 2, 2]);
        assert_eq!(metric.transition_weights, vec![7, 2]);
        assert!(
            ExpandedRoadModel::from_parameters(model.edge_only().clone(), &expanded, 4.0, &[0.0])
                .is_err()
        );
        assert!(
            ExpandedRoadModel::from_parameters(
                model.edge_only().clone(),
                &expanded,
                4.0,
                &[f64::NAN, 0.0]
            )
            .is_err()
        );
        assert!(
            ExpandedRoadModel::from_parameters(
                model.edge_only().clone(),
                &expanded,
                4.0,
                &[-0.1, 0.0]
            )
            .is_err()
        );
    }

    #[test]
    fn one_projected_step_changes_both_blocks_on_one_clock() {
        let graph = graph();
        let expanded = ExpandedTurnGraph::build(&graph).unwrap();
        let edge_only = EdgeOnlyModel::new(&graph.baseline_weights, 1.0).unwrap();
        let mut model = ExpandedRoadModel::new(edge_only, &expanded, 4.0).unwrap();
        let mut optimizer =
            ExpandedProjectedSubgradientOptimizer::new(0.2, 0.0, 0.0, 0.1, 2.0, 1.0).unwrap();

        let oracle = ExpandedOracleStats {
            predicted_edge_counts: vec![1, 0, 0, 0],
            predicted_transition_counts: vec![1, 0],
            weighted_shortest_distance_sum: 0,
            sample_count: 1,
            num_queries: 1,
            oracle_duration: Duration::ZERO,
        };
        let step = model
            .projected_step(&mut optimizer, &[0, 0, 0, 0], &[0, 0], &oracle)
            .unwrap();

        assert_close(model.q()[0], 1.04);
        assert_close(model.transition_residuals()[0], 0.05);
        assert_close(step.max_abs_edge_cost_delta, 0.2);
        assert_close(step.max_abs_transition_cost_delta, 0.2);
        assert_eq!(optimizer.completed_updates(), 1);
    }

    #[test]
    fn regularization_terms_keep_distinct_mathematical_anchors() {
        let graph = graph();
        let expanded = ExpandedTurnGraph::build(&graph).unwrap();
        let edge_only =
            EdgeOnlyModel::from_q(&graph.baseline_weights, 1.0, &[0.8, 1.2, 1.0, 1.0]).unwrap();
        let model =
            ExpandedRoadModel::from_parameters(edge_only, &expanded, 1.0, &[3.0, 4.0]).unwrap();

        assert_close(model.edge_regularization(2.0), 0.02);
        assert_close(model.transition_regularization(2.0), 12.5);
        assert_close(model.regularization(2.0, 2.0), 12.52);
        let summary = model.transition_summary();
        assert_eq!(summary.transitions, 2);
        assert_eq!(summary.zero_transitions, 0);
        assert_eq!(summary.positive_transitions, 2);
        assert_eq!(summary.mean, 3.5);
        assert_eq!(summary.max, 4.0);
        assert_eq!(summary.l2_norm, 5.0);
    }
}
