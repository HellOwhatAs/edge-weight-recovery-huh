#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegretStats {
    pub observed_cost_sum: f64,
    pub predicted_cost_sum: f64,
    pub data_loss_sum: f64,
    pub mean_data_loss: f64,
    pub relative_data_loss: f64,
}

/// Aggregate the mapped observed-path costs in direct graph coordinates.
pub fn observed_cost(weights: &[f64], observed_counts: &[u64]) -> Result<f64, String> {
    if weights.len() != observed_counts.len() {
        return Err(format!(
            "weight count {} does not match observed-count length {}",
            weights.len(),
            observed_counts.len()
        ));
    }
    let mut sum = 0.0;
    for (coordinate, (&weight, &count)) in weights.iter().zip(observed_counts).enumerate() {
        if !weight.is_finite() || weight < 0.0 {
            return Err(format!(
                "coordinate {coordinate} has invalid nonnegative direct weight {weight}"
            ));
        }
        sum += weight * count as f64;
        if !sum.is_finite() {
            return Err("observed direct path-cost sum is not finite".to_string());
        }
    }
    Ok(sum)
}

/// Compute the common observed-minus-shortest-path data term for either graph
/// representation. The oracle selects paths through its internal CCH metric,
/// then evaluates the returned coordinate paths under this same direct vector.
pub fn compute_regret(
    weights: &[f64],
    observed_counts: &[u64],
    predicted_direct_cost_sum: f64,
    sample_count: u64,
) -> Result<RegretStats, String> {
    let observed_cost_sum = observed_cost(weights, observed_counts)?;
    if !predicted_direct_cost_sum.is_finite() || predicted_direct_cost_sum < 0.0 {
        return Err(format!(
            "predicted direct path-cost sum is invalid: {predicted_direct_cost_sum}"
        ));
    }
    let data_loss_sum = observed_cost_sum - predicted_direct_cost_sum;
    if !data_loss_sum.is_finite() {
        return Err("direct data loss is not finite".to_string());
    }
    let mean_data_loss = if sample_count == 0 {
        0.0
    } else {
        data_loss_sum / sample_count as f64
    };
    let relative_data_loss = if observed_cost_sum == 0.0 {
        0.0
    } else {
        data_loss_sum / observed_cost_sum
    };
    Ok(RegretStats {
        observed_cost_sum,
        predicted_cost_sum: predicted_direct_cost_sum,
        data_loss_sum,
        mean_data_loss,
        relative_data_loss,
    })
}

/// `lambda / (2m) * ||w - w0||^2` in direct learned coordinates.
pub fn direct_regularization(weights: &[f64], initial: &[f64], lambda: f64) -> Result<f64, String> {
    regularization(weights, initial, lambda, false)
}

/// `lambda / (2m) * ||w / w0 - 1||^2` in dimensionless relative coordinates.
pub fn relative_regularization(
    weights: &[f64],
    initial: &[f64],
    lambda: f64,
) -> Result<f64, String> {
    regularization(weights, initial, lambda, true)
}

fn regularization(
    weights: &[f64],
    initial: &[f64],
    lambda: f64,
    relative: bool,
) -> Result<f64, String> {
    if weights.len() != initial.len() {
        return Err("weight and initial-weight lengths differ".to_string());
    }
    if !lambda.is_finite() || lambda < 0.0 {
        return Err("lambda must be finite and nonnegative".to_string());
    }
    if weights.is_empty() {
        return Ok(0.0);
    }
    let mut squared_norm = 0.0;
    for (coordinate, (&weight, &initial_weight)) in weights.iter().zip(initial).enumerate() {
        if !weight.is_finite() || !initial_weight.is_finite() {
            return Err(format!(
                "non-finite regularization state at coordinate {coordinate}"
            ));
        }
        if relative && initial_weight <= 0.0 {
            return Err(format!(
                "relative regularization requires positive initial[{coordinate}], got {initial_weight}"
            ));
        }
        let difference = if relative {
            weight / initial_weight - 1.0
        } else {
            weight - initial_weight
        };
        squared_norm += difference.powi(2);
    }
    let value = lambda * squared_norm / (2.0 * weights.len() as f64);
    if !value.is_finite() {
        return Err("regularization is not finite".to_string());
    }
    Ok(value)
}

/// L1 norm of the aggregate count subgradient, used only as a diagnostic.
pub fn count_difference_l1(predicted: &[u64], observed: &[u64]) -> Result<u128, String> {
    if predicted.len() != observed.len() {
        return Err("predicted and observed count lengths differ".to_string());
    }
    predicted
        .iter()
        .zip(observed)
        .try_fold(0u128, |sum, (&left, &right)| {
            sum.checked_add(left.abs_diff(right) as u128)
                .ok_or_else(|| "count-difference diagnostic overflow".to_string())
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_common_regret_and_both_regularizers() {
        let regret = compute_regret(&[5.0, 5.0, 2.0, 2.0], &[2, 2, 0, 0], 8.0, 2).unwrap();
        assert_eq!(regret.observed_cost_sum, 20.0);
        assert_eq!(regret.predicted_cost_sum, 8.0);
        assert_eq!(regret.data_loss_sum, 12.0);
        assert_eq!(regret.mean_data_loss, 6.0);
        assert_eq!(regret.relative_data_loss, 0.6);

        let direct = direct_regularization(&[2.0, 5.0], &[1.0, 3.0], 4.0).unwrap();
        assert_eq!(direct, 5.0);
        let relative = relative_regularization(&[2.0, 5.0], &[1.0, 3.0], 4.0).unwrap();
        assert!((relative - 13.0 / 9.0).abs() < 1e-12);
        assert!(relative_regularization(&[1.0], &[0.0], 1.0).is_err());
    }

    #[test]
    fn tie_can_have_zero_regret_and_nonzero_count_difference() {
        assert_eq!(
            count_difference_l1(&[1, 1, 0, 0], &[0, 0, 1, 1]).unwrap(),
            4
        );
        let regret = compute_regret(&[2.0, 2.0, 2.0, 2.0], &[0, 0, 1, 1], 4.0, 1).unwrap();
        assert_eq!(regret.data_loss_sum, 0.0);
    }
}
