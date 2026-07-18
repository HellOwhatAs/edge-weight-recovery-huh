use crate::objective::{ObjectiveError, relative_regularization};
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Diagnostics for one active v1 projected-subgradient update.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectedStepStats {
    /// Learning rate selected from the global square-root clock.
    pub eta: f64,
    /// Largest mutation of the stored direct-weight vector.
    pub max_abs_delta: f64,
    /// Largest mutation of a relative parameter `q = weight / initial`.
    pub max_abs_parameter_delta: f64,
    /// Number of coordinates whose unprojected parameter crossed a bound.
    pub projected_coordinates: usize,
}

/// Invalid state supplied to a relative projected-subgradient update.
#[derive(Clone, Debug, PartialEq)]
pub enum OptimizerError {
    /// The initial learning rate must be finite and positive.
    InvalidEta0(f64),
    /// The regularization coefficient must be finite and nonnegative.
    InvalidLambda(f64),
    /// Every coordinate vector must have the same length.
    LengthMismatch {
        vector: &'static str,
        actual: usize,
        expected: usize,
    },
    /// A training update cannot operate on an empty vector.
    EmptyWeights,
    /// A full-batch update requires at least one sample.
    ZeroSampleCount,
    /// No update exists after the largest representable clock value.
    UpdateClockOverflow,
    /// Current direct weights must be finite.
    InvalidWeight { coordinate: usize, weight: f64 },
    /// Relative parameters require finite, positive initial weights.
    InvalidInitialWeight { coordinate: usize, weight: f64 },
    /// Direct projection bounds must be finite and ordered.
    InvalidBounds {
        coordinate: usize,
        lower: f64,
        upper: f64,
    },
    /// Restored direct weights must already satisfy their projection bounds.
    WeightOutsideBounds {
        coordinate: usize,
        weight: f64,
        lower: f64,
        upper: f64,
    },
    /// A valid finite input produced a nonfinite gradient.
    NonFiniteGradient { coordinate: usize },
    /// A valid finite input produced a nonfinite candidate relative parameter.
    NonFiniteCandidate { coordinate: usize },
    /// Mapping a relative parameter back to a direct weight was nonfinite.
    NonFiniteMappedWeight { coordinate: usize },
    /// The projection diagnostic counter overflowed.
    ProjectionCountOverflow,
}

impl Display for OptimizerError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidEta0(eta0) => {
                write!(
                    formatter,
                    "eta0 must be finite and greater than zero, got {eta0}"
                )
            }
            Self::InvalidLambda(lambda) => write!(
                formatter,
                "lambda must be finite and nonnegative, got {lambda}"
            ),
            Self::LengthMismatch {
                vector,
                actual,
                expected,
            } => write!(
                formatter,
                "{vector} length {actual} does not match weight length {expected}"
            ),
            Self::EmptyWeights => formatter.write_str("direct weight vector must not be empty"),
            Self::ZeroSampleCount => {
                formatter.write_str("sample_count must be positive for an optimizer update")
            }
            Self::UpdateClockOverflow => formatter.write_str("optimizer update clock overflow"),
            Self::InvalidWeight { coordinate, weight } => write!(
                formatter,
                "weight[{coordinate}] must be finite, got {weight}"
            ),
            Self::InvalidInitialWeight { coordinate, weight } => write!(
                formatter,
                "relative optimizer requires positive initial[{coordinate}], got {weight}"
            ),
            Self::InvalidBounds {
                coordinate,
                lower,
                upper,
            } => write!(
                formatter,
                "coordinate {coordinate} has invalid projection bounds [{lower}, {upper}]"
            ),
            Self::WeightOutsideBounds {
                coordinate,
                weight,
                lower,
                upper,
            } => write!(
                formatter,
                "weight[{coordinate}]={weight} is outside [{lower}, {upper}]"
            ),
            Self::NonFiniteGradient { coordinate } => {
                write!(
                    formatter,
                    "gradient at coordinate {coordinate} is not finite"
                )
            }
            Self::NonFiniteCandidate { coordinate } => write!(
                formatter,
                "unprojected parameter at coordinate {coordinate} is not finite"
            ),
            Self::NonFiniteMappedWeight { coordinate } => write!(
                formatter,
                "mapped candidate weight[{coordinate}] is not finite"
            ),
            Self::ProjectionCountOverflow => {
                formatter.write_str("projected-coordinate count overflow")
            }
        }
    }
}

impl Error for OptimizerError {}

/// Full-batch relative projected subgradient descent for one weight vector.
///
/// Direct weights remain the sole stored/checkpointed state. Each update is
/// performed in dimensionless coordinates `q[i] = weights[i] / initial[i]`:
///
/// `g_q[i] = initial[i] * (observed[i] - predicted[i]) / N
///           + lambda / m * (q[i] - 1)`.
///
/// The caller owns the initial vector and direct lower/upper bound vectors.
#[derive(Clone, Debug, PartialEq)]
pub struct RelativeProjectedSubgradient {
    eta0: f64,
    lambda: f64,
    completed_updates: u64,
}

impl RelativeProjectedSubgradient {
    /// Restore the optimizer while continuing the one global learning-rate clock.
    pub fn with_completed_updates(
        eta0: f64,
        lambda: f64,
        completed_updates: u64,
    ) -> Result<Self, OptimizerError> {
        if !eta0.is_finite() || eta0 <= 0.0 {
            return Err(OptimizerError::InvalidEta0(eta0));
        }
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(OptimizerError::InvalidLambda(lambda));
        }
        Ok(Self {
            eta0,
            lambda,
            completed_updates,
        })
    }

    /// Number of successful updates represented by this state.
    pub const fn completed_updates(&self) -> u64 {
        self.completed_updates
    }

    /// Evaluate the regularizer paired with this optimizer.
    pub fn regularization(&self, weights: &[f64], initial: &[f64]) -> Result<f64, ObjectiveError> {
        relative_regularization(weights, initial, self.lambda)
    }

    /// Apply one validated full-batch update.
    ///
    /// Every coordinate and the next clock value are validated before mutation.
    /// Candidate weights are staged separately, so any error leaves both the
    /// supplied weights and this optimizer's clock unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        weights: &mut [f64],
        initial: &[f64],
        lower_bounds: &[f64],
        upper_bounds: &[f64],
        observed: &[u64],
        predicted: &[u64],
        sample_count: u64,
    ) -> Result<ProjectedStepStats, OptimizerError> {
        validate_step_input(
            weights,
            initial,
            lower_bounds,
            upper_bounds,
            observed,
            predicted,
            sample_count,
        )?;
        let next_clock = self
            .completed_updates
            .checked_add(1)
            .ok_or(OptimizerError::UpdateClockOverflow)?;

        let eta = self.eta0 / (self.completed_updates as f64 + 1.0).sqrt();
        let regularization_scale = self.lambda / weights.len() as f64;
        let inverse_sample_count = 1.0 / sample_count as f64;
        let mut candidates = Vec::with_capacity(weights.len());
        let mut max_abs_delta = 0.0_f64;
        let mut max_abs_parameter_delta = 0.0_f64;
        let mut projected_coordinates = 0usize;

        for coordinate in 0..weights.len() {
            let weight = weights[coordinate];
            let initial_weight = initial[coordinate];
            let lower = lower_bounds[coordinate];
            let upper = upper_bounds[coordinate];
            let parameter = weight / initial_weight;
            let lower_parameter = lower / initial_weight;
            let upper_parameter = upper / initial_weight;
            let data_gradient = initial_weight
                * signed_difference(observed[coordinate], predicted[coordinate])
                * inverse_sample_count;
            let regularization_gradient = regularization_scale * (parameter - 1.0);
            let gradient = data_gradient + regularization_gradient;
            if !gradient.is_finite() {
                return Err(OptimizerError::NonFiniteGradient { coordinate });
            }

            let unprojected_parameter = parameter - eta * gradient;
            if !unprojected_parameter.is_finite() {
                return Err(OptimizerError::NonFiniteCandidate { coordinate });
            }
            if unprojected_parameter < lower_parameter || unprojected_parameter > upper_parameter {
                projected_coordinates = projected_coordinates
                    .checked_add(1)
                    .ok_or(OptimizerError::ProjectionCountOverflow)?;
            }
            let candidate_parameter = unprojected_parameter.clamp(lower_parameter, upper_parameter);
            let mapped_candidate = initial_weight * candidate_parameter;
            if !mapped_candidate.is_finite() {
                return Err(OptimizerError::NonFiniteMappedWeight { coordinate });
            }
            // Division followed by multiplication may move an exact projected
            // bound by one floating ULP, so enforce the direct box as well.
            let candidate = mapped_candidate.clamp(lower, upper);
            max_abs_delta = max_abs_delta.max((candidate - weight).abs());
            max_abs_parameter_delta =
                max_abs_parameter_delta.max((candidate_parameter - parameter).abs());
            candidates.push(candidate);
        }

        weights.copy_from_slice(&candidates);
        self.completed_updates = next_clock;
        Ok(ProjectedStepStats {
            eta,
            max_abs_delta,
            max_abs_parameter_delta,
            projected_coordinates,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_step_input(
    weights: &[f64],
    initial: &[f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    observed: &[u64],
    predicted: &[u64],
    sample_count: u64,
) -> Result<(), OptimizerError> {
    let coordinate_count = weights.len();
    for (vector, actual) in [
        ("initial", initial.len()),
        ("lower bounds", lower_bounds.len()),
        ("upper bounds", upper_bounds.len()),
        ("observed counts", observed.len()),
        ("predicted counts", predicted.len()),
    ] {
        if actual != coordinate_count {
            return Err(OptimizerError::LengthMismatch {
                vector,
                actual,
                expected: coordinate_count,
            });
        }
    }
    if coordinate_count == 0 {
        return Err(OptimizerError::EmptyWeights);
    }
    if sample_count == 0 {
        return Err(OptimizerError::ZeroSampleCount);
    }

    for coordinate in 0..coordinate_count {
        let weight = weights[coordinate];
        let initial_weight = initial[coordinate];
        let lower = lower_bounds[coordinate];
        let upper = upper_bounds[coordinate];
        if !weight.is_finite() {
            return Err(OptimizerError::InvalidWeight { coordinate, weight });
        }
        if !initial_weight.is_finite() || initial_weight <= 0.0 {
            return Err(OptimizerError::InvalidInitialWeight {
                coordinate,
                weight: initial_weight,
            });
        }
        if !lower.is_finite() || !upper.is_finite() || lower > upper {
            return Err(OptimizerError::InvalidBounds {
                coordinate,
                lower,
                upper,
            });
        }
        if weight < lower || weight > upper {
            return Err(OptimizerError::WeightOutsideBounds {
                coordinate,
                weight,
                lower,
                upper,
            });
        }
    }
    Ok(())
}

fn signed_difference(left: u64, right: u64) -> f64 {
    if left >= right {
        (left - right) as f64
    } else {
        -((right - left) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn one_step_matches_the_frozen_relative_v1_values() {
        let initial = [5.0, 2.0];
        let lower = [0.5, 0.2];
        let upper = [50.0, 20.0];
        let observed = [2, 0];
        let predicted = [0, 2];
        let mut weights = initial;
        let mut optimizer =
            RelativeProjectedSubgradient::with_completed_updates(0.5, 0.1, 0).unwrap();

        let step = optimizer
            .step(
                &mut weights,
                &initial,
                &lower,
                &upper,
                &observed,
                &predicted,
                2,
            )
            .unwrap();

        // q starts at [1, 1], so its regularization gradient is zero.
        // g_q=[5,-2], q'=[-1.5,2], and the first coordinate projects to 0.1.
        assert_eq!(weights, [0.5, 4.0]);
        assert_eq!(step.eta, 0.5);
        assert_eq!(step.projected_coordinates, 1);
        assert_close(step.max_abs_delta, 4.5);
        assert_close(step.max_abs_parameter_delta, 1.0);
        assert_eq!(optimizer.completed_updates(), 1);
    }

    #[test]
    fn restored_clock_matches_uninterrupted_updates() {
        let initial = [5.0, 2.0];
        let lower = [0.1, 0.1];
        let upper = [100.0, 100.0];
        let observed = [1, 0];
        let predicted = [0, 1];

        let mut uninterrupted_weights = initial;
        let mut uninterrupted =
            RelativeProjectedSubgradient::with_completed_updates(0.01, 0.5, 0).unwrap();
        for _ in 0..4 {
            uninterrupted
                .step(
                    &mut uninterrupted_weights,
                    &initial,
                    &lower,
                    &upper,
                    &observed,
                    &predicted,
                    1,
                )
                .unwrap();
        }

        let mut resumed_weights = initial;
        let mut first_half =
            RelativeProjectedSubgradient::with_completed_updates(0.01, 0.5, 0).unwrap();
        for _ in 0..2 {
            first_half
                .step(
                    &mut resumed_weights,
                    &initial,
                    &lower,
                    &upper,
                    &observed,
                    &predicted,
                    1,
                )
                .unwrap();
        }
        let mut resumed = RelativeProjectedSubgradient::with_completed_updates(
            0.01,
            0.5,
            first_half.completed_updates(),
        )
        .unwrap();
        for _ in 0..2 {
            resumed
                .step(
                    &mut resumed_weights,
                    &initial,
                    &lower,
                    &upper,
                    &observed,
                    &predicted,
                    1,
                )
                .unwrap();
        }

        assert_eq!(resumed_weights, uninterrupted_weights);
        assert_eq!(
            resumed.completed_updates(),
            uninterrupted.completed_updates()
        );
        assert_close(0.01 / (resumed.completed_updates() as f64).sqrt(), 0.005);
    }

    #[test]
    fn invalid_late_coordinate_mutates_neither_weights_nor_clock() {
        let initial = [5.0, 0.0];
        let lower = [0.5, 0.0];
        let upper = [50.0, 20.0];
        let mut weights = [5.0, 2.0];
        let before = weights;
        let mut optimizer =
            RelativeProjectedSubgradient::with_completed_updates(0.5, 0.1, 0).unwrap();

        let error = optimizer
            .step(&mut weights, &initial, &lower, &upper, &[2, 0], &[0, 2], 2)
            .unwrap_err();

        assert_eq!(
            error,
            OptimizerError::InvalidInitialWeight {
                coordinate: 1,
                weight: 0.0,
            }
        );
        assert_eq!(weights, before);
        assert_eq!(optimizer.completed_updates(), 0);
    }

    #[test]
    fn nonfinite_candidate_mutates_neither_weights_nor_clock() {
        let initial = [f64::MAX];
        let lower = [1.0];
        let upper = [f64::MAX];
        let mut weights = initial;
        let before = weights;
        let mut optimizer =
            RelativeProjectedSubgradient::with_completed_updates(f64::MAX, 0.0, 0).unwrap();

        let error = optimizer
            .step(&mut weights, &initial, &lower, &upper, &[u64::MAX], &[0], 1)
            .unwrap_err();

        assert_eq!(error, OptimizerError::NonFiniteGradient { coordinate: 0 });
        assert_eq!(weights, before);
        assert_eq!(optimizer.completed_updates(), 0);
    }
}
