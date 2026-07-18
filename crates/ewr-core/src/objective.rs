use std::error::Error;
use std::fmt::{Display, Formatter};

/// Full-batch inverse-shortest-path data-loss diagnostics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegretStats {
    /// Cost of all observed trajectories under the current direct weights.
    pub observed_cost_sum: f64,
    /// Cost of all oracle trajectories under the same direct weights.
    pub predicted_cost_sum: f64,
    /// `observed_cost_sum - predicted_cost_sum`.
    pub data_loss_sum: f64,
    /// Data loss divided by the number of observations.
    pub mean_data_loss: f64,
    /// Data loss divided by the observed cost, or zero when that cost is zero.
    pub relative_data_loss: f64,
}

/// Invalid state supplied to an objective calculation.
#[derive(Clone, Debug, PartialEq)]
pub enum ObjectiveError {
    /// Two coordinate vectors must have equal lengths.
    LengthMismatch {
        left: &'static str,
        left_len: usize,
        right: &'static str,
        right_len: usize,
    },
    /// Direct learned weights must be finite and nonnegative when evaluated.
    InvalidWeight { coordinate: usize, weight: f64 },
    /// The oracle path cost must be finite and nonnegative.
    InvalidPredictedCost(f64),
    /// An aggregate direct cost overflowed or otherwise became nonfinite.
    NonFiniteObservedCost,
    /// The observed-minus-predicted loss became nonfinite.
    NonFiniteDataLoss,
    /// The relative regularization coefficient must be finite and nonnegative.
    InvalidLambda(f64),
    /// Relative coordinates require a finite, positive baseline.
    InvalidInitialWeight { coordinate: usize, weight: f64 },
    /// A current weight used by the regularizer must be finite.
    NonFiniteRegularizationWeight { coordinate: usize, weight: f64 },
    /// The regularization calculation overflowed or otherwise became nonfinite.
    NonFiniteRegularization,
}

impl Display for ObjectiveError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LengthMismatch {
                left,
                left_len,
                right,
                right_len,
            } => write!(
                formatter,
                "{left} length {left_len} does not match {right} length {right_len}"
            ),
            Self::InvalidWeight { coordinate, weight } => write!(
                formatter,
                "coordinate {coordinate} has invalid nonnegative direct weight {weight}"
            ),
            Self::InvalidPredictedCost(cost) => {
                write!(
                    formatter,
                    "predicted direct path-cost sum is invalid: {cost}"
                )
            }
            Self::NonFiniteObservedCost => {
                formatter.write_str("observed direct path-cost sum is not finite")
            }
            Self::NonFiniteDataLoss => formatter.write_str("direct data loss is not finite"),
            Self::InvalidLambda(lambda) => write!(
                formatter,
                "regularization coefficient must be finite and nonnegative, got {lambda}"
            ),
            Self::InvalidInitialWeight { coordinate, weight } => write!(
                formatter,
                "relative regularization requires positive initial[{coordinate}], got {weight}"
            ),
            Self::NonFiniteRegularizationWeight { coordinate, weight } => write!(
                formatter,
                "regularization weight[{coordinate}] must be finite, got {weight}"
            ),
            Self::NonFiniteRegularization => {
                formatter.write_str("relative regularization is not finite")
            }
        }
    }
}

impl Error for ObjectiveError {}

/// Aggregate observed-path costs in learned transition coordinates.
pub fn observed_cost(weights: &[f64], observed_counts: &[u64]) -> Result<f64, ObjectiveError> {
    if weights.len() != observed_counts.len() {
        return Err(ObjectiveError::LengthMismatch {
            left: "weights",
            left_len: weights.len(),
            right: "observed counts",
            right_len: observed_counts.len(),
        });
    }

    let mut sum = 0.0;
    for (coordinate, (&weight, &count)) in weights.iter().zip(observed_counts).enumerate() {
        if !weight.is_finite() || weight < 0.0 {
            return Err(ObjectiveError::InvalidWeight { coordinate, weight });
        }
        sum += weight * count as f64;
        if !sum.is_finite() {
            return Err(ObjectiveError::NonFiniteObservedCost);
        }
    }
    Ok(sum)
}

/// Compute the active v1 observed-minus-shortest-path data term.
///
/// The routing backend selects its paths under the quantized metric. The core
/// evaluates those returned coordinate paths under `weights`, so callers must
/// supply the resulting direct cost sum rather than the quantized distance.
pub fn compute_regret(
    weights: &[f64],
    observed_counts: &[u64],
    predicted_direct_cost_sum: f64,
    sample_count: u64,
) -> Result<RegretStats, ObjectiveError> {
    let observed_cost_sum = observed_cost(weights, observed_counts)?;
    if !predicted_direct_cost_sum.is_finite() || predicted_direct_cost_sum < 0.0 {
        return Err(ObjectiveError::InvalidPredictedCost(
            predicted_direct_cost_sum,
        ));
    }

    let data_loss_sum = observed_cost_sum - predicted_direct_cost_sum;
    if !data_loss_sum.is_finite() {
        return Err(ObjectiveError::NonFiniteDataLoss);
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

/// Evaluate `lambda / (2m) * ||weights / initial - 1||^2`.
pub fn relative_regularization(
    weights: &[f64],
    initial: &[f64],
    lambda: f64,
) -> Result<f64, ObjectiveError> {
    if weights.len() != initial.len() {
        return Err(ObjectiveError::LengthMismatch {
            left: "weights",
            left_len: weights.len(),
            right: "initial weights",
            right_len: initial.len(),
        });
    }
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(ObjectiveError::InvalidLambda(lambda));
    }
    if weights.is_empty() {
        return Ok(0.0);
    }

    let mut squared_norm = 0.0;
    for (coordinate, (&weight, &initial_weight)) in weights.iter().zip(initial).enumerate() {
        if !weight.is_finite() {
            return Err(ObjectiveError::NonFiniteRegularizationWeight { coordinate, weight });
        }
        if !initial_weight.is_finite() || initial_weight <= 0.0 {
            return Err(ObjectiveError::InvalidInitialWeight {
                coordinate,
                weight: initial_weight,
            });
        }
        let difference = weight / initial_weight - 1.0;
        squared_norm += difference * difference;
        if !squared_norm.is_finite() {
            return Err(ObjectiveError::NonFiniteRegularization);
        }
    }

    let value = lambda * squared_norm / (2.0 * weights.len() as f64);
    if !value.is_finite() {
        return Err(ObjectiveError::NonFiniteRegularization);
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_v1_regret_and_relative_regularizer() {
        let regret = compute_regret(&[5.0, 5.0, 2.0, 2.0], &[2, 2, 0, 0], 8.0, 2).unwrap();
        assert_eq!(regret.observed_cost_sum, 20.0);
        assert_eq!(regret.predicted_cost_sum, 8.0);
        assert_eq!(regret.data_loss_sum, 12.0);
        assert_eq!(regret.mean_data_loss, 6.0);
        assert_eq!(regret.relative_data_loss, 0.6);

        let penalty = relative_regularization(&[2.0, 5.0], &[1.0, 3.0], 4.0).unwrap();
        assert!((penalty - 13.0 / 9.0).abs() < 1e-12);
    }

    #[test]
    fn zero_observations_keep_v1_zero_mean_convention() {
        let regret = compute_regret(&[], &[], 0.0, 0).unwrap();
        assert_eq!(regret.mean_data_loss, 0.0);
        assert_eq!(regret.relative_data_loss, 0.0);
    }

    #[test]
    fn rejects_invalid_relative_state_with_a_typed_error() {
        assert_eq!(
            relative_regularization(&[1.0], &[0.0], 1.0),
            Err(ObjectiveError::InvalidInitialWeight {
                coordinate: 0,
                weight: 0.0,
            })
        );
    }

    #[test]
    fn tied_costs_can_have_zero_regret() {
        let regret = compute_regret(&[2.0, 2.0, 2.0, 2.0], &[0, 0, 1, 1], 4.0, 1).unwrap();
        assert_eq!(regret.data_loss_sum, 0.0);
    }
}
