use crate::objective::{direct_regularization, relative_regularization};

/// Representation-neutral optimizer geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OptimizerGeometry {
    /// Optimize the stored direct graph weights with one Euclidean step size.
    DirectWeights,
    /// Optimize dimensionless multipliers `q[i] = w[i] / w0[i]` while keeping
    /// direct weights as the sole stored/checkpointed vector.
    RelativeWeights,
}

impl OptimizerGeometry {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "projected_subgradient" => Ok(Self::DirectWeights),
            "relative_projected_subgradient" => Ok(Self::RelativeWeights),
            _ => Err(format!(
                "unsupported optimizer kind {value:?}; expected \"projected_subgradient\" or \"relative_projected_subgradient\""
            )),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DirectWeights => "projected_subgradient",
            Self::RelativeWeights => "relative_projected_subgradient",
        }
    }

    pub const fn parameterization(self) -> &'static str {
        match self {
            Self::DirectWeights => "direct_weights",
            Self::RelativeWeights => "relative_to_initial",
        }
    }
}

/// Diagnostics for one projected-subgradient update.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectedStepStats {
    pub eta: f64,
    /// Largest mutation of the stored direct-weight vector.
    pub max_abs_delta: f64,
    /// Largest mutation in the optimizer's configured coordinate system.
    pub max_abs_parameter_delta: f64,
    pub projected_coordinates: usize,
}

/// Full-batch projected subgradient descent for one graph-weight vector.
///
/// The optimizer deliberately has no graph-specific state. The graph problem
/// supplies the initial point and element-wise projection bounds, while the
/// trainer supplies observed and predicted coordinate counts. Both geometries
/// keep direct weights as the sole stored and checkpointed state.
#[derive(Clone, Debug)]
pub struct ProjectedSubgradientOptimizer {
    geometry: OptimizerGeometry,
    eta0: f64,
    lambda: f64,
    completed_updates: u64,
}

impl ProjectedSubgradientOptimizer {
    pub fn new(geometry: OptimizerGeometry, eta0: f64, lambda: f64) -> Result<Self, String> {
        Self::with_completed_updates(geometry, eta0, lambda, 0)
    }

    /// Restore the optimizer while continuing the single global square-root
    /// learning-rate clock.
    pub fn with_completed_updates(
        geometry: OptimizerGeometry,
        eta0: f64,
        lambda: f64,
        completed_updates: u64,
    ) -> Result<Self, String> {
        if !eta0.is_finite() || eta0 <= 0.0 {
            return Err("eta0 must be finite and greater than zero".to_string());
        }
        if !lambda.is_finite() || lambda < 0.0 {
            return Err("lambda must be finite and nonnegative".to_string());
        }
        Ok(Self {
            geometry,
            eta0,
            lambda,
            completed_updates,
        })
    }

    pub const fn completed_updates(&self) -> u64 {
        self.completed_updates
    }

    pub const fn geometry(&self) -> OptimizerGeometry {
        self.geometry
    }

    pub const fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Evaluate the regularizer paired with this optimizer geometry.
    pub fn regularization(&self, weights: &[f64], initial: &[f64]) -> Result<f64, String> {
        match self.geometry {
            OptimizerGeometry::DirectWeights => {
                direct_regularization(weights, initial, self.lambda)
            }
            OptimizerGeometry::RelativeWeights => {
                relative_regularization(weights, initial, self.lambda)
            }
        }
    }

    /// In direct geometry, apply
    ///
    /// `g[i] = (observed[i] - predicted[i]) / N
    ///         + lambda / m * (weights[i] - initial[i])`
    ///
    /// In relative geometry, let `q[i] = weights[i] / initial[i]` and apply
    ///
    /// `g_q[i] = initial[i] * (observed[i] - predicted[i]) / N
    ///           + lambda / m * (q[i] - 1)`.
    ///
    /// The latter is equivalent in direct-weight space to the diagonal
    /// preconditioner `diag(initial[i]^2)` applied to the gradient of
    /// `data_loss + lambda/(2m) * ||weights/initial - 1||^2`.
    ///
    /// Both updates are followed by projection onto the graph problem's
    /// element-wise box. Relative geometry maps that box into multiplier
    /// coordinates before projection. No representation-specific state or
    /// update rule is introduced.
    ///
    /// Every candidate is validated before the supplied weight slice is
    /// mutated, so a malformed restored state cannot be partially updated.
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
    ) -> Result<ProjectedStepStats, String> {
        let coordinate_count = weights.len();
        for (label, actual) in [
            ("initial", initial.len()),
            ("lower_bounds", lower_bounds.len()),
            ("upper_bounds", upper_bounds.len()),
            ("observed", observed.len()),
            ("predicted", predicted.len()),
        ] {
            if actual != coordinate_count {
                return Err(format!(
                    "{label} length {actual} does not match weight length {coordinate_count}"
                ));
            }
        }
        if coordinate_count == 0 {
            return Err("direct weight vector must not be empty".to_string());
        }
        if sample_count == 0 {
            return Err("sample_count must be positive for an optimizer update".to_string());
        }

        let eta = self.eta0 / (self.completed_updates as f64 + 1.0).sqrt();
        let regularization_scale = self.lambda / coordinate_count as f64;
        let inverse_sample_count = 1.0 / sample_count as f64;
        let mut candidates = Vec::with_capacity(coordinate_count);
        let mut max_abs_delta = 0.0_f64;
        let mut max_abs_parameter_delta = 0.0_f64;
        let mut projected_coordinates = 0usize;

        for coordinate in 0..coordinate_count {
            let weight = weights[coordinate];
            let initial_weight = initial[coordinate];
            let lower = lower_bounds[coordinate];
            let upper = upper_bounds[coordinate];
            if !weight.is_finite() || !initial_weight.is_finite() {
                return Err(format!(
                    "weight state at coordinate {coordinate} must be finite"
                ));
            }
            if !lower.is_finite() || !upper.is_finite() || lower > upper {
                return Err(format!(
                    "coordinate {coordinate} has invalid projection bounds [{lower}, {upper}]"
                ));
            }
            if weight < lower || weight > upper {
                return Err(format!(
                    "weight[{coordinate}]={weight} is outside [{lower}, {upper}]"
                ));
            }

            let count_difference = signed_difference(observed[coordinate], predicted[coordinate]);
            let (parameter, lower_parameter, upper_parameter, data_gradient, anchor_delta) =
                match self.geometry {
                    OptimizerGeometry::DirectWeights => (
                        weight,
                        lower,
                        upper,
                        count_difference * inverse_sample_count,
                        weight - initial_weight,
                    ),
                    OptimizerGeometry::RelativeWeights => {
                        if initial_weight <= 0.0 {
                            return Err(format!(
                                "relative optimizer requires positive initial[{coordinate}], got {initial_weight}"
                            ));
                        }
                        let parameter = weight / initial_weight;
                        (
                            parameter,
                            lower / initial_weight,
                            upper / initial_weight,
                            initial_weight * count_difference * inverse_sample_count,
                            parameter - 1.0,
                        )
                    }
                };
            let regularization_gradient = regularization_scale * anchor_delta;
            let gradient = data_gradient + regularization_gradient;
            if !gradient.is_finite() {
                return Err(format!("gradient at coordinate {coordinate} is not finite"));
            }
            let unprojected_parameter = parameter - eta * gradient;
            if !unprojected_parameter.is_finite() {
                return Err(format!(
                    "unprojected parameter at coordinate {coordinate} is not finite"
                ));
            }
            if unprojected_parameter < lower_parameter || unprojected_parameter > upper_parameter {
                projected_coordinates = projected_coordinates
                    .checked_add(1)
                    .ok_or_else(|| "projected-coordinate count overflow".to_string())?;
            }
            let candidate_parameter = unprojected_parameter.clamp(lower_parameter, upper_parameter);
            let mapped_candidate = match self.geometry {
                OptimizerGeometry::DirectWeights => candidate_parameter,
                OptimizerGeometry::RelativeWeights => initial_weight * candidate_parameter,
            };
            if !mapped_candidate.is_finite() {
                return Err(format!(
                    "mapped candidate weight[{coordinate}] is not finite"
                ));
            }
            // Mapping an exactly projected relative bound back through a
            // division/multiplication pair may differ by one floating ULP.
            let candidate = mapped_candidate.clamp(lower, upper);
            max_abs_delta = max_abs_delta.max((candidate - weight).abs());
            max_abs_parameter_delta =
                max_abs_parameter_delta.max((candidate_parameter - parameter).abs());
            candidates.push(candidate);
        }

        let next_clock = self
            .completed_updates
            .checked_add(1)
            .ok_or_else(|| "optimizer update clock overflow".to_string())?;
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
    fn direct_formula_regularization_projection_and_clock_are_exact() {
        let mut optimizer =
            ProjectedSubgradientOptimizer::new(OptimizerGeometry::DirectWeights, 2.0, 4.0).unwrap();
        let initial = [10.0, 20.0, 30.0];
        let lower = [9.5, 10.0, 20.0];
        let upper = [20.0, 21.0, 40.0];
        let observed = [4, 1, 3];
        let predicted = [1, 5, 3];
        let mut weights = [11.0, 19.0, 33.0];

        let first = optimizer
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
        assert_close(first.eta, 2.0);
        // coordinate 0: g = 3/2 + 4/3 * 1, unprojected below 9.5
        assert_close(weights[0], 9.5);
        // coordinate 1: g = -4/2 + 4/3 * -1 = -10/3, projected to 21
        assert_close(weights[1], 21.0);
        // coordinate 2: g = 0 + 4/3 * 3 = 4
        assert_close(weights[2], 25.0);
        assert_eq!(first.projected_coordinates, 2);
        assert_close(first.max_abs_delta, 8.0);
        assert_close(first.max_abs_parameter_delta, 8.0);
        assert_eq!(optimizer.completed_updates(), 1);

        let second = optimizer
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
        assert_close(second.eta, 2.0 / 2.0_f64.sqrt());
        assert_eq!(optimizer.completed_updates(), 2);
    }

    #[test]
    fn relative_formula_matches_multiplier_update_and_direct_preconditioner() {
        let initial = [8.0, 18.0];
        let lower = [0.8, 1.8];
        let upper = [80.0, 180.0];
        let observed = [1, 0];
        let predicted = [0, 1];
        let mut weights = initial;
        let mut optimizer =
            ProjectedSubgradientOptimizer::new(OptimizerGeometry::RelativeWeights, 0.01, 0.0)
                .unwrap();

        let step = optimizer
            .step(
                &mut weights,
                &initial,
                &lower,
                &upper,
                &observed,
                &predicted,
                1,
            )
            .unwrap();

        // q = [1, 1], g_q = [8, -18], so q' = [0.92, 1.18].
        assert_close(weights[0], 8.0 * 0.92);
        assert_close(weights[1], 18.0 * 1.18);
        assert_close(step.max_abs_parameter_delta, 0.18);
        assert_close(step.max_abs_delta, 3.24);
        assert_eq!(step.projected_coordinates, 0);

        // In direct coordinates this is w' = w - eta * diag(w0^2) * d.
        assert_close(weights[0], initial[0] - 0.01 * initial[0].powi(2));
        assert_close(weights[1], initial[1] + 0.01 * initial[1].powi(2));
    }

    #[test]
    fn geometry_selects_the_matching_regularizer() {
        let weights = [8.0, 30.0];
        let initial = [10.0, 20.0];
        let direct =
            ProjectedSubgradientOptimizer::new(OptimizerGeometry::DirectWeights, 1.0, 4.0).unwrap();
        let relative =
            ProjectedSubgradientOptimizer::new(OptimizerGeometry::RelativeWeights, 1.0, 4.0)
                .unwrap();
        assert_close(direct.regularization(&weights, &initial).unwrap(), 104.0);
        assert_close(relative.regularization(&weights, &initial).unwrap(), 0.29);
    }

    #[test]
    fn restored_clock_matches_an_uninterrupted_optimizer() {
        let initial = [10.0, 20.0];
        let lower = [1.0, 1.0];
        let upper = [100.0, 100.0];
        let observed = [2, 0];
        let predicted = [0, 2];
        let mut uninterrupted_weights = initial;
        let mut uninterrupted =
            ProjectedSubgradientOptimizer::new(OptimizerGeometry::DirectWeights, 1.0, 0.5).unwrap();
        for _ in 0..4 {
            uninterrupted
                .step(
                    &mut uninterrupted_weights,
                    &initial,
                    &lower,
                    &upper,
                    &observed,
                    &predicted,
                    2,
                )
                .unwrap();
        }

        let mut resumed_weights = initial;
        let mut first_half =
            ProjectedSubgradientOptimizer::new(OptimizerGeometry::DirectWeights, 1.0, 0.5).unwrap();
        for _ in 0..2 {
            first_half
                .step(
                    &mut resumed_weights,
                    &initial,
                    &lower,
                    &upper,
                    &observed,
                    &predicted,
                    2,
                )
                .unwrap();
        }
        let mut resumed = ProjectedSubgradientOptimizer::with_completed_updates(
            OptimizerGeometry::DirectWeights,
            1.0,
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
                    2,
                )
                .unwrap();
        }

        assert_eq!(resumed_weights, uninterrupted_weights);
        assert_eq!(
            resumed.completed_updates(),
            uninterrupted.completed_updates()
        );
    }

    #[test]
    fn invalid_input_does_not_mutate_weights_or_clock() {
        let mut optimizer =
            ProjectedSubgradientOptimizer::new(OptimizerGeometry::DirectWeights, 1.0, 0.0).unwrap();
        let mut weights = [2.0];
        let error = optimizer
            .step(&mut weights, &[2.0], &[1.0], &[3.0], &[1], &[], 1)
            .unwrap_err();
        assert!(error.contains("predicted length"));
        assert_eq!(weights, [2.0]);
        assert_eq!(optimizer.completed_updates(), 0);
    }
}
