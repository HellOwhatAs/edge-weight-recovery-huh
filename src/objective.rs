use crate::oracle::{ExpandedOracleStats, OracleStats};
use crate::turn_graph::ExpandedTurnGraph;

#[derive(Clone, Debug, PartialEq)]
pub struct RegretStats {
    pub observed_cost_sum: u128,
    pub shortest_distance_sum: u128,
    pub data_loss_sum: u128,
    pub mean_data_loss: f64,
    /// Model-relative regret: aggregate regret divided by the observed cost
    /// under this same metric. Because that denominator changes with the
    /// learned cost scale, this is not a fair sole comparison metric between
    /// independently optimized models.
    pub relative_data_loss: f64,
}

pub fn observed_cost(weights: &[u32], observed_counts: &[u64]) -> Result<u128, String> {
    if weights.len() != observed_counts.len() {
        return Err("weight and observed-count lengths differ".to_string());
    }
    weights
        .iter()
        .zip(observed_counts)
        .try_fold(0u128, |sum, (&weight, &count)| {
            sum.checked_add(weight as u128 * count as u128)
                .ok_or_else(|| "observed path-cost sum overflow".to_string())
        })
}

/// Compute the inverse-shortest-path data objective for one metric.
///
/// This is exactly the total observed path cost minus the sample-weighted
/// shortest-path distance. The count residual is deliberately not part of this
/// quantity.
pub fn compute_regret(
    weights: &[u32],
    observed_counts: &[u64],
    oracle: &OracleStats,
) -> Result<RegretStats, String> {
    let observed_cost_sum = observed_cost(weights, observed_counts)?;
    finish_regret(
        observed_cost_sum,
        oracle.weighted_shortest_distance_sum,
        oracle.sample_count,
    )
}

/// Compute the observed cost under the exact expanded integer metric.
///
/// Each original edge is paid once. A transition contributes only its
/// residual increment because its next-edge cost is already included in the
/// edge-count term. This is algebraically identical to paying the first edge
/// as a source offset and every later edge through its transition weight.
pub fn observed_expanded_cost(
    expanded: &ExpandedTurnGraph,
    edge_weights: &[u32],
    transition_weights: &[u32],
    observed_edge_counts: &[u64],
    observed_transition_counts: &[u64],
) -> Result<u128, String> {
    if edge_weights.len() != expanded.stats.original_edges
        || observed_edge_counts.len() != expanded.stats.original_edges
    {
        return Err("expanded edge weight/count lengths do not match the original graph".into());
    }
    if transition_weights.len() != expanded.transition_count()
        || observed_transition_counts.len() != expanded.transition_count()
    {
        return Err("expanded transition weight/count lengths do not match the topology".into());
    }

    let edge_cost = observed_cost(edge_weights, observed_edge_counts)?;
    expanded
        .transitions()
        .try_fold(edge_cost, |sum, (transition, _, next_edge)| {
            let transition_weight = transition_weights[transition.index()];
            let next_edge_weight = edge_weights[next_edge];
            let residual_weight = transition_weight
                .checked_sub(next_edge_weight)
                .ok_or_else(|| {
                    format!(
                        "transition {} weight {transition_weight} is below next-edge weight {next_edge_weight}",
                        transition.index()
                    )
                })?;
            sum.checked_add(
                residual_weight as u128
                    * observed_transition_counts[transition.index()] as u128,
            )
            .ok_or_else(|| "expanded observed path-cost sum overflow".to_string())
        })
}

/// Compute true observed-minus-shortest-path regret for an expanded metric.
pub fn compute_expanded_regret(
    expanded: &ExpandedTurnGraph,
    edge_weights: &[u32],
    transition_weights: &[u32],
    observed_edge_counts: &[u64],
    observed_transition_counts: &[u64],
    oracle: &ExpandedOracleStats,
) -> Result<RegretStats, String> {
    let observed_cost_sum = observed_expanded_cost(
        expanded,
        edge_weights,
        transition_weights,
        observed_edge_counts,
        observed_transition_counts,
    )?;
    finish_regret(
        observed_cost_sum,
        oracle.weighted_shortest_distance_sum,
        oracle.sample_count,
    )
}

fn finish_regret(
    observed_cost_sum: u128,
    weighted_shortest_distance_sum: u128,
    sample_count: u64,
) -> Result<RegretStats, String> {
    let data_loss_sum = observed_cost_sum
        .checked_sub(weighted_shortest_distance_sum)
        .ok_or_else(|| {
            format!(
                "negative regret: observed cost {observed_cost_sum} < shortest-distance sum {}; \
                 verify path endpoints and original-edge mapping",
                weighted_shortest_distance_sum
            )
        })?;
    let mean_data_loss = if sample_count == 0 {
        0.0
    } else {
        data_loss_sum as f64 / sample_count as f64
    };
    // Preserve the historical model-relative diagnostic. Its denominator is
    // the current metric's observed cost, so values from models with different
    // learned cost scales are not directly comparable on their own.
    let relative_data_loss = if observed_cost_sum == 0 {
        0.0
    } else {
        data_loss_sum as f64 / observed_cost_sum as f64
    };
    Ok(RegretStats {
        observed_cost_sum,
        shortest_distance_sum: weighted_shortest_distance_sum,
        data_loss_sum,
        mean_data_loss,
        relative_data_loss,
    })
}

/// L1 norm of one selected aggregate count subgradient.
///
/// This is a diagnostic, not the regret objective. It is tie-breaking
/// dependent and does not include regularization.
pub fn count_residual_l1(predicted: &[u64], observed: &[u64]) -> Result<u128, String> {
    if predicted.len() != observed.len() {
        return Err("predicted and observed count lengths differ".to_string());
    }
    predicted
        .iter()
        .zip(observed)
        .try_fold(0u128, |sum, (&left, &right)| {
            sum.checked_add(left.abs_diff(right) as u128)
                .ok_or_else(|| "count residual overflow".to_string())
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::GraphData;
    use std::time::Duration;

    #[test]
    fn computes_observed_minus_shortest_path_cost() {
        let oracle = OracleStats {
            predicted_edge_counts: vec![0, 0, 2, 2],
            weighted_shortest_distance_sum: 8,
            sample_count: 2,
            num_queries: 1,
            oracle_duration: Duration::ZERO,
        };
        let regret = compute_regret(&[5, 5, 2, 2], &[2, 2, 0, 0], &oracle).unwrap();
        assert_eq!(regret.observed_cost_sum, 20);
        assert_eq!(regret.shortest_distance_sum, 8);
        assert_eq!(regret.data_loss_sum, 12);
        assert_eq!(regret.mean_data_loss, 6.0);
        assert_eq!(regret.relative_data_loss, 0.6);
    }

    #[test]
    fn zero_regret_can_have_nonzero_count_residual_under_a_tie() {
        let predicted = [1, 1, 0, 0];
        let observed = [0, 0, 1, 1];
        assert_eq!(count_residual_l1(&predicted, &observed).unwrap(), 4);
        let oracle = OracleStats {
            predicted_edge_counts: predicted.to_vec(),
            weighted_shortest_distance_sum: 4,
            sample_count: 1,
            num_queries: 1,
            oracle_duration: Duration::ZERO,
        };
        let regret = compute_regret(&[2, 2, 2, 2], &observed, &oracle).unwrap();
        assert_eq!(regret.data_loss_sum, 0);
        assert_eq!(regret.relative_data_loss, 0.0);
    }

    #[test]
    fn expanded_observed_cost_counts_edges_once_and_residuals_once() {
        let graph = GraphData {
            tail: vec![0, 1, 0, 2],
            head: vec![1, 3, 2, 3],
            baseline_weights: vec![5, 5, 2, 2],
            x: vec![0.0, 1.0, 1.0, 2.0],
            y: vec![0.0, 0.0, 1.0, 0.0],
        };
        let expanded = ExpandedTurnGraph::build(&graph).unwrap();
        let transition_weights = vec![7, 2];
        let observed_edges = vec![1, 1, 1, 1];
        let observed_transitions = vec![1, 1];
        assert_eq!(
            observed_expanded_cost(
                &expanded,
                &graph.baseline_weights,
                &transition_weights,
                &observed_edges,
                &observed_transitions,
            )
            .unwrap(),
            16
        );

        let oracle = ExpandedOracleStats {
            predicted_edge_counts: vec![0; 4],
            predicted_transition_counts: vec![0; 2],
            weighted_shortest_distance_sum: 10,
            sample_count: 2,
            num_queries: 1,
            oracle_duration: Duration::ZERO,
        };
        let regret = compute_expanded_regret(
            &expanded,
            &graph.baseline_weights,
            &transition_weights,
            &observed_edges,
            &observed_transitions,
            &oracle,
        )
        .unwrap();
        assert_eq!(regret.data_loss_sum, 6);
        assert_eq!(regret.mean_data_loss, 3.0);
        assert_eq!(regret.relative_data_loss, 0.375);
    }
}
