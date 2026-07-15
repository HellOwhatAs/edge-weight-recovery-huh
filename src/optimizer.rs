/// Diagnostics for one projected-subgradient update.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectedStepStats {
    pub eta: f64,
    pub max_abs_delta: f64,
    pub projected_edges: usize,
}

/// Full-batch projected subgradient descent for the regularized inverse
/// shortest-path objective.
///
/// Continuous latent multipliers are updated here. Integer quantization stays
/// separate so sub-integer changes can accumulate across epochs.
#[derive(Clone, Debug)]
pub struct ProjectedSubgradientOptimizer {
    eta0: f64,
    lambda: f64,
    q_min: f64,
    q_max: f64,
    completed_updates: u64,
}

impl ProjectedSubgradientOptimizer {
    pub fn new(eta0: f64, lambda: f64, q_min: f64, q_max: f64) -> Result<Self, String> {
        Self::with_completed_updates(eta0, lambda, q_min, q_max, 0)
    }

    /// Restore an optimizer while continuing its original square-root clock.
    pub fn with_completed_updates(
        eta0: f64,
        lambda: f64,
        q_min: f64,
        q_max: f64,
        completed_updates: u64,
    ) -> Result<Self, String> {
        if !eta0.is_finite() || eta0 <= 0.0 {
            return Err("eta0 must be finite and greater than zero".to_string());
        }
        if !lambda.is_finite() || lambda < 0.0 {
            return Err("lambda must be finite and non-negative".to_string());
        }
        if !q_min.is_finite() || q_min <= 0.0 {
            return Err("q_min must be finite and greater than zero".to_string());
        }
        if !q_max.is_finite() || q_max < q_min {
            return Err("q_max must be finite and no smaller than q_min".to_string());
        }

        Ok(Self {
            eta0,
            lambda,
            q_min,
            q_max,
            completed_updates,
        })
    }

    pub const fn completed_updates(&self) -> u64 {
        self.completed_updates
    }

    /// Apply
    /// `g[e] = baseline[e] * (observed[e] - predicted[e]) / N
    ///         + lambda / m * (q[e] - 1)`
    /// followed by projection onto `[q_min, q_max]`.
    pub fn step(
        &mut self,
        q: &mut [f64],
        baseline: &[u32],
        observed: &[u64],
        predicted: &[u64],
        sample_count: u64,
    ) -> ProjectedStepStats {
        assert_eq!(q.len(), baseline.len(), "q and baseline length mismatch");
        assert_eq!(q.len(), observed.len(), "q and observed length mismatch");
        assert_eq!(q.len(), predicted.len(), "q and predicted length mismatch");

        let eta = self.eta0 / (self.completed_updates as f64 + 1.0).sqrt();
        let regularization_scale = if q.is_empty() {
            0.0
        } else {
            self.lambda / q.len() as f64
        };
        let inverse_sample_count = if sample_count == 0 {
            0.0
        } else {
            1.0 / sample_count as f64
        };
        let mut max_abs_delta = 0.0_f64;
        let mut projected_edges = 0;

        for (edge, q_value) in q.iter_mut().enumerate() {
            let old_q = *q_value;
            assert!(old_q.is_finite(), "q[{edge}] must be finite");
            let count_difference = if observed[edge] >= predicted[edge] {
                (observed[edge] - predicted[edge]) as f64
            } else {
                -((predicted[edge] - observed[edge]) as f64)
            };
            let data_gradient = baseline[edge] as f64 * count_difference * inverse_sample_count;
            let regularization_gradient = regularization_scale * (old_q - 1.0);
            let gradient = data_gradient + regularization_gradient;
            assert!(
                gradient.is_finite(),
                "gradient for edge {edge} is not finite"
            );

            let unprojected = old_q - eta * gradient;
            assert!(
                unprojected.is_finite(),
                "unprojected q for edge {edge} is not finite"
            );
            if unprojected < self.q_min || unprojected > self.q_max {
                projected_edges += 1;
            }
            let new_q = unprojected.clamp(self.q_min, self.q_max);
            max_abs_delta = max_abs_delta.max((new_q - old_q).abs());
            *q_value = new_q;
        }

        self.completed_updates = self.completed_updates.saturating_add(1);
        ProjectedStepStats {
            eta,
            max_abs_delta,
            projected_edges,
        }
    }
}

/// Diagnostics for one nonnegative transition-residual update.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TurnResidualStepStats {
    pub eta: f64,
    pub max_abs_delta: f64,
    pub projected_transitions: usize,
}

/// Compact diagnostics for the continuous transition-residual state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TurnResidualSummary {
    pub transitions: usize,
    pub zero_transitions: usize,
    pub positive_transitions: usize,
    pub mean: f64,
    pub max: f64,
    pub l2_norm: f64,
}

/// Full-batch projected subgradient descent for transition residuals.
///
/// For `C(q,r) = C_edge(q) + residual_scale * sum_t r[t]`, the data
/// subgradient is `residual_scale * (observed-predicted) / N`. Residuals are
/// anchored at zero and projected onto `[0, r_max]`.
#[derive(Clone, Debug)]
pub struct TurnResidualOptimizer {
    eta0: f64,
    lambda: f64,
    r_max: f64,
    completed_updates: u64,
}

impl TurnResidualOptimizer {
    pub fn new(eta0: f64, lambda: f64, r_max: f64) -> Result<Self, String> {
        Self::with_completed_updates(eta0, lambda, r_max, 0)
    }

    /// Restore a residual optimizer while continuing its own square-root
    /// clock. The turn clock is independent of the edge optimizer clock.
    pub fn with_completed_updates(
        eta0: f64,
        lambda: f64,
        r_max: f64,
        completed_updates: u64,
    ) -> Result<Self, String> {
        if !eta0.is_finite() || eta0 <= 0.0 {
            return Err("turn eta0 must be finite and greater than zero".to_string());
        }
        if !lambda.is_finite() || lambda < 0.0 {
            return Err("turn lambda must be finite and non-negative".to_string());
        }
        if !r_max.is_finite() || r_max <= 0.0 {
            return Err("r_max must be finite and greater than zero".to_string());
        }
        Ok(Self {
            eta0,
            lambda,
            r_max,
            completed_updates,
        })
    }

    pub const fn completed_updates(&self) -> u64 {
        self.completed_updates
    }

    pub const fn r_max(&self) -> f64 {
        self.r_max
    }

    pub fn step(
        &mut self,
        residuals: &mut [f64],
        observed: &[u64],
        predicted: &[u64],
        sample_count: u64,
        residual_scale: f64,
    ) -> TurnResidualStepStats {
        assert_eq!(
            residuals.len(),
            observed.len(),
            "residual and observed-transition length mismatch"
        );
        assert_eq!(
            residuals.len(),
            predicted.len(),
            "residual and predicted-transition length mismatch"
        );
        assert!(
            residual_scale.is_finite() && residual_scale > 0.0,
            "residual scale must be finite and greater than zero"
        );

        let eta = self.eta0 / (self.completed_updates as f64 + 1.0).sqrt();
        let regularization_scale = if residuals.is_empty() {
            0.0
        } else {
            self.lambda / residuals.len() as f64
        };
        let inverse_sample_count = if sample_count == 0 {
            0.0
        } else {
            1.0 / sample_count as f64
        };
        let mut max_abs_delta = 0.0_f64;
        let mut projected_transitions = 0;

        for (transition, residual) in residuals.iter_mut().enumerate() {
            let old_residual = *residual;
            assert!(
                old_residual.is_finite() && old_residual >= 0.0 && old_residual <= self.r_max,
                "r[{transition}] must be finite and inside [0, r_max]"
            );
            let count_difference = if observed[transition] >= predicted[transition] {
                (observed[transition] - predicted[transition]) as f64
            } else {
                -((predicted[transition] - observed[transition]) as f64)
            };
            let data_gradient = residual_scale * count_difference * inverse_sample_count;
            let regularization_gradient = regularization_scale * old_residual;
            let gradient = data_gradient + regularization_gradient;
            assert!(
                gradient.is_finite(),
                "gradient for transition {transition} is not finite"
            );

            let unprojected = old_residual - eta * gradient;
            assert!(
                unprojected.is_finite(),
                "unprojected residual for transition {transition} is not finite"
            );
            if unprojected < 0.0 || unprojected > self.r_max {
                projected_transitions += 1;
            }
            let new_residual = unprojected.clamp(0.0, self.r_max);
            max_abs_delta = max_abs_delta.max((new_residual - old_residual).abs());
            *residual = new_residual;
        }

        self.completed_updates = self.completed_updates.saturating_add(1);
        TurnResidualStepStats {
            eta,
            max_abs_delta,
            projected_transitions,
        }
    }
}

/// Quantize `baseline .* q * scale` to positive CCH weights.
pub fn quantize_weights(baseline: &[u32], q: &[f64], scale: f64) -> Result<Vec<u32>, String> {
    if baseline.len() != q.len() {
        return Err(format!(
            "baseline and q length mismatch: {} != {}",
            baseline.len(),
            q.len()
        ));
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err("scale must be finite and greater than zero".to_string());
    }

    baseline
        .iter()
        .zip(q)
        .enumerate()
        .map(|(edge, (&base, &multiplier))| {
            if !multiplier.is_finite() {
                return Err(format!("q[{edge}] is not finite"));
            }
            let scaled = base as f64 * multiplier * scale;
            if !scaled.is_finite() {
                return Err(format!("quantized weight for edge {edge} is not finite"));
            }
            let rounded = scaled.round().max(1.0);
            if rounded >= i32::MAX as f64 {
                return Err(format!(
                    "quantized weight for edge {edge} reaches the CCH infinity sentinel"
                ));
            }
            Ok(rounded as u32)
        })
        .collect()
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
    fn update_direction_is_observed_minus_predicted() {
        let baseline = [10, 10, 10];
        let observed = [1, 1, 0];
        let predicted = [1, 0, 1];
        let mut q = [1.0, 1.0, 1.0];
        let mut optimizer = ProjectedSubgradientOptimizer::new(0.01, 0.0, 0.1, 10.0).unwrap();

        let stats = optimizer.step(&mut q, &baseline, &observed, &predicted, 1);

        assert_close(q[0], 1.0);
        assert!(q[1] < 1.0, "observed-only edge should become cheaper");
        assert!(q[2] > 1.0, "predicted-only edge should become dearer");
        assert_eq!(stats.projected_edges, 0);
    }

    #[test]
    fn uses_sqrt_schedule_and_projects_to_the_box() {
        let mut q = [1.0, 1.0];
        let mut optimizer = ProjectedSubgradientOptimizer::new(1.0, 0.0, 0.5, 1.5).unwrap();
        let first = optimizer.step(&mut q, &[1, 1], &[100, 0], &[0, 100], 1);
        assert_eq!(q, [0.5, 1.5]);
        assert_eq!(first.projected_edges, 2);
        assert_close(first.eta, 1.0);
        let second = optimizer.step(&mut q, &[1, 1], &[0, 0], &[0, 0], 1);
        assert_close(second.eta, 1.0 / 2.0_f64.sqrt());
        assert_eq!(optimizer.completed_updates(), 2);
    }

    #[test]
    fn edge_optimizer_continues_the_original_clock() {
        let mut q = [1.0];
        let mut optimizer =
            ProjectedSubgradientOptimizer::with_completed_updates(3.0, 0.0, 0.1, 10.0, 8).unwrap();
        let step = optimizer.step(&mut q, &[1], &[0], &[0], 1);
        assert_close(step.eta, 1.0);
        assert_eq!(optimizer.completed_updates(), 9);
    }

    #[test]
    fn latent_updates_accumulate_across_quantization_boundaries() {
        let baseline = [1];
        let mut q = [1.0];
        let mut optimizer = ProjectedSubgradientOptimizer::new(0.1, 0.0, 0.1, 10.0).unwrap();
        let initial = quantize_weights(&baseline, &q, 1.0).unwrap();
        let mut changed = false;
        for _ in 0..21 {
            optimizer.step(&mut q, &baseline, &[0], &[1], 1);
            if quantize_weights(&baseline, &q, 1.0).unwrap() != initial {
                changed = true;
                break;
            }
        }
        assert!(changed);
    }

    #[test]
    fn rejects_invalid_optimizer_and_quantization_inputs() {
        assert!(ProjectedSubgradientOptimizer::new(0.0, 0.0, 0.1, 1.0).is_err());
        assert!(ProjectedSubgradientOptimizer::new(0.1, -1.0, 0.1, 1.0).is_err());
        assert!(ProjectedSubgradientOptimizer::new(0.1, 0.0, 2.0, 1.0).is_err());
        assert!(quantize_weights(&[1], &[], 1.0).is_err());
        assert!(quantize_weights(&[1], &[f64::NAN], 1.0).is_err());
        assert!(quantize_weights(&[u32::MAX], &[2.0], 1.0).is_err());
        assert_eq!(quantize_weights(&[0], &[0.25], 1.0).unwrap(), vec![1]);
    }

    #[test]
    fn predicted_overuse_increases_a_turn_residual() {
        let mut residuals = [0.0, 0.0];
        let mut optimizer = TurnResidualOptimizer::new(0.1, 0.0, 10.0).unwrap();
        let step = optimizer.step(&mut residuals, &[0, 1], &[1, 0], 1, 10.0);

        assert_close(residuals[0], 1.0);
        assert_close(residuals[1], 0.0);
        assert_eq!(step.projected_transitions, 1);
        assert_eq!(optimizer.completed_updates(), 1);
    }

    #[test]
    fn turn_regularization_pulls_positive_residuals_toward_zero() {
        let mut residuals = [1.0, 0.0];
        let mut optimizer = TurnResidualOptimizer::new(0.5, 2.0, 10.0).unwrap();
        optimizer.step(&mut residuals, &[0, 0], &[0, 0], 1, 1.0);

        assert_close(residuals[0], 0.5);
        assert_close(residuals[1], 0.0);
    }

    #[test]
    fn turn_optimizer_has_an_independent_continuation_clock() {
        let mut residuals = [0.0];
        let mut optimizer =
            TurnResidualOptimizer::with_completed_updates(2.0, 0.0, 5.0, 3).unwrap();
        let step = optimizer.step(&mut residuals, &[0], &[0], 1, 1.0);

        assert_close(step.eta, 1.0);
        assert_eq!(optimizer.completed_updates(), 4);
        assert_close(optimizer.r_max(), 5.0);
    }

    #[test]
    fn rejects_invalid_turn_optimizer_inputs() {
        assert!(TurnResidualOptimizer::new(0.0, 0.0, 1.0).is_err());
        assert!(TurnResidualOptimizer::new(0.1, -1.0, 1.0).is_err());
        assert!(TurnResidualOptimizer::new(0.1, 0.0, 0.0).is_err());
        assert!(TurnResidualOptimizer::new(0.1, 0.0, f64::INFINITY).is_err());
    }
}
