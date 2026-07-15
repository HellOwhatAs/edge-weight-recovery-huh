use crate::oracle::OracleStats;

#[derive(Clone, Debug, PartialEq)]
pub struct RegretStats {
    pub observed_cost_sum: u128,
    pub shortest_distance_sum: u128,
    pub data_loss_sum: u128,
    pub mean_data_loss: f64,
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
    let data_loss_sum = observed_cost_sum
        .checked_sub(oracle.weighted_shortest_distance_sum)
        .ok_or_else(|| {
            format!(
                "negative regret: observed cost {observed_cost_sum} < shortest-distance sum {}; \
                 verify path endpoints and original-edge mapping",
                oracle.weighted_shortest_distance_sum
            )
        })?;
    let mean_data_loss = if oracle.sample_count == 0 {
        0.0
    } else {
        data_loss_sum as f64 / oracle.sample_count as f64
    };
    let relative_data_loss = if observed_cost_sum == 0 {
        0.0
    } else {
        data_loss_sum as f64 / observed_cost_sum as f64
    };
    Ok(RegretStats {
        observed_cost_sum,
        shortest_distance_sum: oracle.weighted_shortest_distance_sum,
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
}
