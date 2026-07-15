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
    epoch: u64,
}

impl ProjectedSubgradientOptimizer {
    pub fn new(eta0: f64, lambda: f64, q_min: f64, q_max: f64) -> Result<Self, String> {
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
            epoch: 0,
        })
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

        let eta = self.eta0 / (self.epoch as f64 + 1.0).sqrt();
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

        self.epoch = self.epoch.saturating_add(1);
        ProjectedStepStats {
            eta,
            max_abs_delta,
            projected_edges,
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
}
