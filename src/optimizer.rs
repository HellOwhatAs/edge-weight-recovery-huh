use std::collections::BTreeMap;

/// Diagnostics for one projected-subgradient update.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectedStepStats {
    /// Learning rate used for this step.
    pub eta: f64,
    /// Largest absolute change to any latent multiplier after projection.
    pub max_abs_delta: f64,
    /// Number of coordinates whose unprojected update was outside the box.
    pub projected_edges: usize,
}

/// Full-batch projected subgradient descent for regularized inverse shortest paths.
///
/// The optimizer operates on continuous edge-cost multipliers. Quantization for
/// the integer shortest-path oracle is deliberately kept separate so that small
/// latent updates can accumulate across epochs.
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

    /// Apply one full-batch projected-subgradient step.
    ///
    /// For edge `e`, this uses exactly
    ///
    /// `g[e] = baseline[e] * (observed[e] - predicted[e]) / N
    ///         + lambda / m * (q[e] - 1)`.
    ///
    /// The data term is defined as zero when `sample_count == 0`.
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

            let data_gradient = if sample_count == 0 {
                0.0
            } else {
                let count_difference = if observed[edge] >= predicted[edge] {
                    (observed[edge] - predicted[edge]) as f64
                } else {
                    -((predicted[edge] - observed[edge]) as f64)
                };
                baseline[edge] as f64 * count_difference * inverse_sample_count
            };
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

/// L2 anchoring penalty around the baseline multiplier `q = 1`.
pub fn regularization_loss(q: &[f64], lambda: f64) -> f64 {
    if q.is_empty() {
        return 0.0;
    }

    let squared_distance = q
        .iter()
        .map(|&multiplier| {
            let difference = multiplier - 1.0;
            difference * difference
        })
        .sum::<f64>();
    lambda * squared_distance / (2.0 * q.len() as f64)
}

/// Convert latent cost multipliers into positive integer oracle weights.
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
        .zip(q.iter())
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
            // routingkit-cch uses i32::MAX (stored as u32) as its unreachable
            // sentinel, so finite edge costs must stay strictly below it.
            if rounded >= i32::MAX as f64 {
                return Err(format!(
                    "quantized weight for edge {edge} reaches the CCH infinity sentinel"
                ));
            }

            Ok(rounded as u32)
        })
        .collect()
}

#[derive(Clone)]
pub struct AdamOptimizer {
    m: Vec<f32>,
    v: Vec<f32>,
    t: u64,
    alpha: f32, // Learning rate
    beta1: f32,
    beta2: f32,
    epsilon: f32,
}

impl AdamOptimizer {
    const CONSISTENCY_THRESHOLD: f32 = 0.1;
    const VARIANCE_THRESHOLD: f32 = 1e-5;

    pub fn new(size: usize, alpha: f32) -> Self {
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
            t: 0,
            alpha,
            beta1: 0.9,
            beta2: 0.95, // Lower beta2 from 0.999 to 0.95 to make it more reactive
            epsilon: 1e-8,
        }
    }

    pub fn step(&mut self, current_weights: &[u32], grads: &[i64]) -> BTreeMap<u32, u32> {
        self.t += 1;
        let mut updates = BTreeMap::new();

        // Bias correction factors
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, (&w, &g)) in current_weights.iter().zip(grads.iter()).enumerate() {
            let g = g as f32;

            // Update raw moments
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;

            // Bias correction
            let m_hat = self.m[i] / bias_correction1;
            let v_hat = self.v[i] / bias_correction2;

            // Compute update
            // We want to INCREASE weight if sim > obs (grad > 0) to discourage usage
            let delta = self.alpha * m_hat / (v_hat.sqrt() + self.epsilon);

            // Apply update only if significant enough to change integer weight
            let new_w = (w as f64 + delta as f64)
                .round()
                .clamp(1.0, (i32::MAX - 1) as f64) as u32;

            if new_w != w {
                updates.insert(i as u32, new_w);
            }
        }
        updates
    }

    // Reset internal state for perturbation/restart
    pub fn reset(&mut self) {
        self.m.fill(0.0);
        self.v.fill(0.0);
        self.t = 0;
    }

    // Decay momentum to allow exploration without full reset
    pub fn decay_momentum(&mut self, factor: f32) {
        for x in self.m.iter_mut() {
            *x *= factor;
        }
        for x in self.v.iter_mut() {
            *x *= factor;
        }
        // We do NOT reset t, because we want to keep bias correction low
    }

    /// Diagnose optimizer state for oscillation and stagnation
    /// Returns: (high_oscillation_count, low_gradient_count, avg_consistency)
    pub fn diagnose(&self) -> (usize, usize, f32) {
        let mut osc_count = 0;
        let mut quiet_count = 0;
        let mut total_consistency = 0.0;
        let epsilon = 1e-10;

        for (m, v) in self.m.iter().zip(self.v.iter()) {
            if *v < epsilon {
                quiet_count += 1;
                continue;
            }

            // Consistency = m^2 / v. Range [0, 1].
            // Near 1: consistently moving in one direction.
            // Near 0: high variance relative to mean (oscillating).
            let consistency = (m * m) / (v + epsilon);
            total_consistency += consistency;

            if consistency < Self::CONSISTENCY_THRESHOLD && *v > Self::VARIANCE_THRESHOLD {
                // High energy (v is not small) but low consistency -> oscillating
                osc_count += 1;
            }
        }

        let active_count = self.m.len() - quiet_count;
        let avg_consistency = if active_count > 0 {
            total_consistency / active_count as f32
        } else {
            1.0
        };

        (osc_count, quiet_count, avg_consistency)
    }

    /// Get indices of edges that are oscillating (high variance, low consistency)
    pub fn get_oscillating_indices(&self) -> Vec<usize> {
        let epsilon = 1e-10;
        self.m
            .iter()
            .zip(self.v.iter())
            .enumerate()
            .filter_map(|(i, (m, v))| {
                if *v < epsilon {
                    return None;
                }
                let consistency = (m * m) / (v + epsilon);
                if consistency < Self::CONSISTENCY_THRESHOLD && *v > Self::VARIANCE_THRESHOLD {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AdamOptimizer, ProjectedSubgradientOptimizer, quantize_weights, regularization_loss,
    };

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn parallel_path_update_has_the_correct_direction_and_cancels_shared_edges() {
        // Edges are [shared, observed-only, predicted-only].
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
    fn empty_data_leaves_q_one_at_the_regularization_anchor() {
        let baseline = [5, 10];
        // Counts must be ignored when N is zero.
        let observed = [100, 0];
        let predicted = [0, 200];
        let mut q = [1.0, 1.0];
        let mut optimizer = ProjectedSubgradientOptimizer::new(0.5, 2.0, 0.1, 10.0).unwrap();

        let stats = optimizer.step(&mut q, &baseline, &observed, &predicted, 0);

        assert_eq!(q, [1.0, 1.0]);
        assert_close(stats.max_abs_delta, 0.0);
        assert_close(regularization_loss(&q, 2.0), 0.0);
        assert_close(regularization_loss(&[], 2.0), 0.0);
    }

    #[test]
    fn update_is_projected_onto_the_multiplier_box() {
        let baseline = [1, 1];
        let observed = [100, 0];
        let predicted = [0, 100];
        let mut q = [1.0, 1.0];
        let mut optimizer = ProjectedSubgradientOptimizer::new(1.0, 0.0, 0.5, 1.5).unwrap();

        let stats = optimizer.step(&mut q, &baseline, &observed, &predicted, 1);

        assert_eq!(q, [0.5, 1.5]);
        assert_eq!(stats.projected_edges, 2);
        assert_close(stats.max_abs_delta, 0.5);
    }

    #[test]
    fn latent_updates_accumulate_until_the_quantized_weight_changes() {
        let baseline = [1];
        let observed = [0];
        let predicted = [1];
        let mut q = [1.0];
        let mut optimizer = ProjectedSubgradientOptimizer::new(0.1, 0.0, 0.1, 10.0).unwrap();
        let initial_weights = quantize_weights(&baseline, &q, 1.0).unwrap();

        optimizer.step(&mut q, &baseline, &observed, &predicted, 1);
        assert_eq!(
            quantize_weights(&baseline, &q, 1.0).unwrap(),
            initial_weights
        );

        let mut changed = false;
        for _ in 0..20 {
            optimizer.step(&mut q, &baseline, &observed, &predicted, 1);
            if quantize_weights(&baseline, &q, 1.0).unwrap() != initial_weights {
                changed = true;
                break;
            }
        }

        assert!(
            changed,
            "small latent steps should eventually cross a quantization boundary"
        );
    }

    #[test]
    fn constructor_and_quantization_reject_invalid_inputs() {
        assert!(ProjectedSubgradientOptimizer::new(0.0, 0.0, 0.1, 1.0).is_err());
        assert!(ProjectedSubgradientOptimizer::new(0.1, -1.0, 0.1, 1.0).is_err());
        assert!(ProjectedSubgradientOptimizer::new(0.1, 0.0, 2.0, 1.0).is_err());

        assert!(quantize_weights(&[1], &[], 1.0).is_err());
        assert!(quantize_weights(&[1], &[f64::NAN], 1.0).is_err());
        assert!(quantize_weights(&[u32::MAX], &[2.0], 1.0).is_err());
        assert_eq!(quantize_weights(&[0], &[0.25], 1.0).unwrap(), vec![1]);
    }

    #[test]
    fn exact_f64_oracle_satisfies_the_subgradient_inequality() {
        // Two directed alternatives: 0-1-3 is observed; 0-2-3 is the
        // strictly cheaper f64 shortest path at q = 1.
        let tail = [0usize, 1, 0, 2];
        let head = [1usize, 3, 2, 3];
        let baseline = [5.0, 5.0, 2.0, 2.0];
        let q = [1.0, 1.0, 1.0, 1.0];
        let lambda = 2.0;

        let objective = |multipliers: &[f64; 4]| {
            let weights = std::array::from_fn::<_, 4, _>(|edge| baseline[edge] * multipliers[edge]);
            let observed_cost = weights[0] + weights[1];
            let shortest = dense_dijkstra(4, &tail, &head, &weights, 0, 3);
            let regularization = multipliers
                .iter()
                .map(|value| (value - 1.0).powi(2))
                .sum::<f64>()
                * lambda
                / 8.0;
            observed_cost - shortest + regularization
        };

        // g = b * (x_obs - x_shortest) + lambda/m * (q - 1).
        let subgradient = [5.0, 5.0, -2.0, -2.0];
        let at_q = objective(&q);
        for candidate in [
            [0.8, 1.2, 1.1, 0.9],
            [1.4, 1.3, 0.7, 0.8],
            [0.5, 0.5, 2.0, 2.0],
            [1.0, 1.0, 1.0, 1.0],
        ] {
            let affine_lower_bound = at_q
                + subgradient
                    .iter()
                    .zip(candidate.iter().zip(q))
                    .map(|(gradient, (candidate, current))| gradient * (candidate - current))
                    .sum::<f64>();
            assert!(
                objective(&candidate) + 1e-12 >= affine_lower_bound,
                "subgradient inequality failed for {candidate:?}"
            );
        }
    }

    #[test]
    fn integer_quantization_can_change_the_continuous_shortest_path() {
        // Three 0.51-cost edges beat one 1.60-cost edge continuously, but each
        // positive integer edge costs at least one after quantization. This test
        // records why the CCH update is a quantized approximate oracle rather
        // than an exact oracle for the continuous convex objective.
        let q = [0.51, 0.51, 0.51, 1.60];
        let continuous_long_path = q[0] + q[1] + q[2];
        let continuous_short_path = q[3];
        assert!(continuous_long_path < continuous_short_path);

        let quantized = quantize_weights(&[1, 1, 1, 1], &q, 1.0).unwrap();
        let quantized_long_path = quantized[0] + quantized[1] + quantized[2];
        let quantized_short_path = quantized[3];
        assert!(quantized_long_path > quantized_short_path);
    }

    #[test]
    fn legacy_adam_never_reaches_the_cch_infinity_sentinel() {
        let mut optimizer = AdamOptimizer::new(1, f32::MAX);
        let updates = optimizer.step(&[1], &[1]);
        assert_eq!(updates[&0], (i32::MAX - 1) as u32);
    }

    fn dense_dijkstra(
        node_count: usize,
        tail: &[usize],
        head: &[usize],
        weights: &[f64],
        source: usize,
        target: usize,
    ) -> f64 {
        let mut distance = vec![f64::INFINITY; node_count];
        let mut settled = vec![false; node_count];
        distance[source] = 0.0;
        for _ in 0..node_count {
            let next = (0..node_count)
                .filter(|&node| !settled[node])
                .min_by(|&left, &right| distance[left].total_cmp(&distance[right]));
            let Some(node) = next else { break };
            if !distance[node].is_finite() {
                break;
            }
            settled[node] = true;
            for edge in 0..tail.len() {
                if tail[edge] == node {
                    distance[head[edge]] = distance[head[edge]].min(distance[node] + weights[edge]);
                }
            }
        }
        distance[target]
    }
}
