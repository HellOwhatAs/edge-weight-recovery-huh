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

/// Diagnostics for one joint expanded-model projected-subgradient update.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpandedProjectedStepStats {
    pub eta: f64,
    pub max_abs_q_delta: f64,
    pub max_abs_r_delta: f64,
    pub max_abs_edge_cost_delta: f64,
    pub max_abs_transition_cost_delta: f64,
    pub projected_edges: usize,
    pub projected_transitions: usize,
}

/// One projected-subgradient optimizer for the complete expanded road model.
///
/// The continuous metric is
/// `kappa_(e,f) = b_f * q_f + residual_scale * r_(e,f)`. Direct gradients in
/// `q` and `r` have incompatible coordinate scales because they contain
/// `b_e` and `residual_scale`, respectively. The optimizer therefore works in
/// the additive-cost coordinates
///
/// `u_e = b_e * (q_e - 1)` and `v_t = residual_scale * r_t`.
///
/// With `d = (observed - predicted) / sample_count` and
/// `eta_k = eta0 / sqrt(k + 1)`, mapping the additive-cost update back to the
/// stored parameters gives
///
/// `q_e <- project(q_e - eta_k *
///     (d_e / b_e + lambda_edge * (q_e - 1) / (|E| * b_e^2)))`
///
/// `r_t <- project(r_t - eta_k *
///     (d_t / residual_scale
///      + lambda_transition * r_t / (|T| * residual_scale^2)))`.
///
/// Equivalently, this is the fixed diagonal preconditioner
/// `diag(b_e^-2, residual_scale^-2)` applied to the full gradient, including
/// regularization. It changes only the optimization geometry: the continuous
/// objective, its two regularization terms, and both projection sets are
/// unchanged.
#[derive(Clone, Debug)]
pub struct ExpandedProjectedSubgradientOptimizer {
    eta0: f64,
    lambda_edge: f64,
    lambda_transition: f64,
    q_min: f64,
    q_max: f64,
    r_max: f64,
    completed_updates: u64,
}

impl ExpandedProjectedSubgradientOptimizer {
    pub fn new(
        eta0: f64,
        lambda_edge: f64,
        lambda_transition: f64,
        q_min: f64,
        q_max: f64,
        r_max: f64,
    ) -> Result<Self, String> {
        Self::with_completed_updates(eta0, lambda_edge, lambda_transition, q_min, q_max, r_max, 0)
    }

    /// Restore the one global square-root clock used by both parameter blocks.
    pub fn with_completed_updates(
        eta0: f64,
        lambda_edge: f64,
        lambda_transition: f64,
        q_min: f64,
        q_max: f64,
        r_max: f64,
        completed_updates: u64,
    ) -> Result<Self, String> {
        if !eta0.is_finite() || eta0 <= 0.0 {
            return Err("expanded eta0 must be finite and greater than zero".to_string());
        }
        if !lambda_edge.is_finite() || lambda_edge < 0.0 {
            return Err("lambda_edge must be finite and non-negative".to_string());
        }
        if !lambda_transition.is_finite() || lambda_transition < 0.0 {
            return Err("lambda_transition must be finite and non-negative".to_string());
        }
        if !q_min.is_finite() || q_min <= 0.0 {
            return Err("q_min must be finite and greater than zero".to_string());
        }
        if !q_max.is_finite() || q_max < q_min {
            return Err("q_max must be finite and no smaller than q_min".to_string());
        }
        if !r_max.is_finite() || r_max <= 0.0 {
            return Err("r_max must be finite and greater than zero".to_string());
        }

        Ok(Self {
            eta0,
            lambda_edge,
            lambda_transition,
            q_min,
            q_max,
            r_max,
            completed_updates,
        })
    }

    pub const fn completed_updates(&self) -> u64 {
        self.completed_updates
    }

    /// Jointly update all edge multipliers and transition residuals from one
    /// pre-update expanded shortest-path batch.
    ///
    /// Every input and every candidate value is validated before either
    /// parameter slice is mutated. Thus an invalid restored state is rejected
    /// rather than silently repaired by projection.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn step(
        &mut self,
        q: &mut [f64],
        transition_residuals: &mut [f64],
        metric_baseline: &[u32],
        residual_scale: f64,
        observed_edge_counts: &[u64],
        predicted_edge_counts: &[u64],
        observed_transition_counts: &[u64],
        predicted_transition_counts: &[u64],
        sample_count: u64,
    ) -> Result<ExpandedProjectedStepStats, String> {
        validate_lengths(
            q.len(),
            metric_baseline.len(),
            observed_edge_counts.len(),
            predicted_edge_counts.len(),
            "edge",
        )?;
        validate_lengths(
            transition_residuals.len(),
            transition_residuals.len(),
            observed_transition_counts.len(),
            predicted_transition_counts.len(),
            "transition",
        )?;
        if !residual_scale.is_finite() || residual_scale <= 0.0 {
            return Err("residual_scale must be finite and greater than zero".to_string());
        }
        for (edge, (&multiplier, &baseline)) in q.iter().zip(metric_baseline).enumerate() {
            if baseline == 0 {
                return Err(format!("metric_baseline[{edge}] must be positive"));
            }
            if !multiplier.is_finite() || multiplier < self.q_min || multiplier > self.q_max {
                return Err(format!(
                    "q[{edge}]={multiplier} must be finite and inside [{}, {}]",
                    self.q_min, self.q_max
                ));
            }
        }
        for (transition, &residual) in transition_residuals.iter().enumerate() {
            if !residual.is_finite() || residual < 0.0 || residual > self.r_max {
                return Err(format!(
                    "r[{transition}]={residual} must be finite and inside [0, {}]",
                    self.r_max
                ));
            }
        }

        let next_completed_updates = self
            .completed_updates
            .checked_add(1)
            .ok_or_else(|| "expanded optimizer update clock overflow".to_string())?;
        let eta = self.eta0 / (self.completed_updates as f64 + 1.0).sqrt();
        let inverse_sample_count = if sample_count == 0 {
            0.0
        } else {
            1.0 / sample_count as f64
        };
        let edge_regularization_scale = if q.is_empty() {
            0.0
        } else {
            self.lambda_edge / q.len() as f64
        };
        let transition_regularization_scale = if transition_residuals.is_empty() {
            0.0
        } else {
            self.lambda_transition / transition_residuals.len() as f64
        };

        let mut next_q = Vec::with_capacity(q.len());
        let mut max_abs_q_delta = 0.0_f64;
        let mut max_abs_edge_cost_delta = 0.0_f64;
        let mut projected_edges = 0usize;
        for edge in 0..q.len() {
            let old_q = q[edge];
            let baseline = metric_baseline[edge] as f64;
            let inverse_baseline = 1.0 / baseline;
            let count_difference =
                signed_count_difference(observed_edge_counts[edge], predicted_edge_counts[edge]);
            let normalized_gradient = count_difference * inverse_sample_count * inverse_baseline
                + edge_regularization_scale * (old_q - 1.0) * inverse_baseline * inverse_baseline;
            if !normalized_gradient.is_finite() {
                return Err(format!("normalized gradient for edge {edge} is not finite"));
            }
            let unprojected = old_q - eta * normalized_gradient;
            if !unprojected.is_finite() {
                return Err(format!("unprojected q for edge {edge} is not finite"));
            }
            if unprojected < self.q_min || unprojected > self.q_max {
                projected_edges += 1;
            }
            let new_q = unprojected.clamp(self.q_min, self.q_max);
            let delta = (new_q - old_q).abs();
            max_abs_q_delta = max_abs_q_delta.max(delta);
            max_abs_edge_cost_delta = max_abs_edge_cost_delta.max(baseline * delta);
            next_q.push(new_q);
        }

        let inverse_residual_scale = 1.0 / residual_scale;
        let mut next_residuals = Vec::with_capacity(transition_residuals.len());
        let mut max_abs_r_delta = 0.0_f64;
        let mut max_abs_transition_cost_delta = 0.0_f64;
        let mut projected_transitions = 0usize;
        for transition in 0..transition_residuals.len() {
            let old_residual = transition_residuals[transition];
            let count_difference = signed_count_difference(
                observed_transition_counts[transition],
                predicted_transition_counts[transition],
            );
            let normalized_gradient =
                count_difference * inverse_sample_count * inverse_residual_scale
                    + transition_regularization_scale
                        * old_residual
                        * inverse_residual_scale
                        * inverse_residual_scale;
            if !normalized_gradient.is_finite() {
                return Err(format!(
                    "normalized gradient for transition {transition} is not finite"
                ));
            }
            let unprojected = old_residual - eta * normalized_gradient;
            if !unprojected.is_finite() {
                return Err(format!(
                    "unprojected residual for transition {transition} is not finite"
                ));
            }
            if unprojected < 0.0 || unprojected > self.r_max {
                projected_transitions += 1;
            }
            let new_residual = unprojected.clamp(0.0, self.r_max);
            let delta = (new_residual - old_residual).abs();
            max_abs_r_delta = max_abs_r_delta.max(delta);
            max_abs_transition_cost_delta =
                max_abs_transition_cost_delta.max(residual_scale * delta);
            next_residuals.push(new_residual);
        }

        q.copy_from_slice(&next_q);
        transition_residuals.copy_from_slice(&next_residuals);
        self.completed_updates = next_completed_updates;

        Ok(ExpandedProjectedStepStats {
            eta,
            max_abs_q_delta,
            max_abs_r_delta,
            max_abs_edge_cost_delta,
            max_abs_transition_cost_delta,
            projected_edges,
            projected_transitions,
        })
    }
}

fn validate_lengths(
    parameters: usize,
    scales: usize,
    observed: usize,
    predicted: usize,
    kind: &str,
) -> Result<(), String> {
    if parameters != scales || parameters != observed || parameters != predicted {
        return Err(format!(
            "expanded {kind} length mismatch: parameters={parameters}, scales={scales}, observed={observed}, predicted={predicted}"
        ));
    }
    Ok(())
}

fn signed_count_difference(observed: u64, predicted: u64) -> f64 {
    if observed >= predicted {
        (observed - predicted) as f64
    } else {
        -((predicted - observed) as f64)
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
    fn expanded_step_updates_both_blocks_on_one_clock() {
        let mut q = [1.0, 1.0];
        let mut residuals = [0.0];
        let mut optimizer =
            ExpandedProjectedSubgradientOptimizer::new(0.2, 0.0, 0.0, 0.1, 2.0, 1.0).unwrap();

        let step = optimizer
            .step(
                &mut q,
                &mut residuals,
                &[2, 8],
                4.0,
                &[0, 0],
                &[1, 1],
                &[0],
                &[1],
                1,
            )
            .unwrap();

        assert_close(q[0], 1.1);
        assert_close(q[1], 1.025);
        assert_close(residuals[0], 0.05);
        assert_close(step.eta, 0.2);
        assert_eq!(optimizer.completed_updates(), 1);
    }

    #[test]
    fn additive_cost_preconditioning_equalizes_coordinate_scales() {
        let mut q = [1.0, 1.0];
        let mut residuals = [0.0];
        let mut optimizer =
            ExpandedProjectedSubgradientOptimizer::new(0.2, 0.0, 0.0, 0.1, 2.0, 1.0).unwrap();

        let step = optimizer
            .step(
                &mut q,
                &mut residuals,
                &[2, 8],
                4.0,
                &[0, 0],
                &[1, 1],
                &[0],
                &[1],
                1,
            )
            .unwrap();

        assert_close(2.0 * (q[0] - 1.0), 0.2);
        assert_close(8.0 * (q[1] - 1.0), 0.2);
        assert_close(4.0 * residuals[0], 0.2);
        assert_close(step.max_abs_edge_cost_delta, 0.2);
        assert_close(step.max_abs_transition_cost_delta, 0.2);
    }

    #[test]
    fn expanded_regularization_keeps_the_original_anchors() {
        let mut q = [1.5];
        let mut residuals = [2.0];
        let mut optimizer =
            ExpandedProjectedSubgradientOptimizer::new(0.5, 4.0, 8.0, 0.1, 3.0, 3.0).unwrap();

        optimizer
            .step(&mut q, &mut residuals, &[2], 4.0, &[0], &[0], &[0], &[0], 1)
            .unwrap();

        // q delta = -eta * lambda_edge * (q-1) / b^2.
        assert_close(q[0], 1.25);
        // r delta = -eta * lambda_transition * r / residual_scale^2.
        assert_close(residuals[0], 1.5);
    }

    #[test]
    fn expanded_step_projects_both_blocks() {
        let mut q = [1.0, 1.0];
        let mut residuals = [0.0, 1.0];
        let mut optimizer =
            ExpandedProjectedSubgradientOptimizer::new(10.0, 0.0, 0.0, 0.5, 1.5, 2.0).unwrap();

        let step = optimizer
            .step(
                &mut q,
                &mut residuals,
                &[1, 1],
                1.0,
                &[1, 0],
                &[0, 1],
                &[0, 1],
                &[1, 0],
                1,
            )
            .unwrap();

        assert_eq!(q, [0.5, 1.5]);
        assert_eq!(residuals, [2.0, 0.0]);
        assert_eq!(step.projected_edges, 2);
        assert_eq!(step.projected_transitions, 2);
        assert_eq!(optimizer.completed_updates(), 1);
    }

    #[test]
    fn expanded_optimizer_restores_one_square_root_clock() {
        let mut q = [1.0];
        let mut residuals = [0.0];
        let mut optimizer = ExpandedProjectedSubgradientOptimizer::with_completed_updates(
            3.0, 0.0, 0.0, 0.1, 2.0, 1.0, 8,
        )
        .unwrap();

        let step = optimizer
            .step(&mut q, &mut residuals, &[1], 1.0, &[0], &[0], &[0], &[0], 1)
            .unwrap();

        assert_close(step.eta, 1.0);
        assert_eq!(optimizer.completed_updates(), 9);
    }

    #[test]
    fn expanded_step_rejects_invalid_state_before_mutation() {
        let mut q = [3.0];
        let mut residuals = [0.5];
        let original_q = q;
        let original_residuals = residuals;
        let mut optimizer =
            ExpandedProjectedSubgradientOptimizer::new(1.0, 0.0, 0.0, 0.5, 2.0, 1.0).unwrap();

        assert!(
            optimizer
                .step(&mut q, &mut residuals, &[1], 1.0, &[0], &[1], &[0], &[1], 1,)
                .is_err()
        );
        assert_eq!(q, original_q);
        assert_eq!(residuals, original_residuals);
        assert_eq!(optimizer.completed_updates(), 0);
    }

    #[test]
    fn rejects_invalid_expanded_optimizer_inputs() {
        assert!(ExpandedProjectedSubgradientOptimizer::new(0.0, 0.0, 0.0, 0.1, 1.0, 1.0).is_err());
        assert!(ExpandedProjectedSubgradientOptimizer::new(0.1, -1.0, 0.0, 0.1, 1.0, 1.0).is_err());
        assert!(ExpandedProjectedSubgradientOptimizer::new(0.1, 0.0, -1.0, 0.1, 1.0, 1.0).is_err());
        assert!(ExpandedProjectedSubgradientOptimizer::new(0.1, 0.0, 0.0, 2.0, 1.0, 1.0).is_err());
        assert!(ExpandedProjectedSubgradientOptimizer::new(0.1, 0.0, 0.0, 0.1, 1.0, 0.0).is_err());
    }
}
