use std::collections::BTreeMap;

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

    pub fn step(&mut self, current_weights: &[u32], grads: &[i32]) -> BTreeMap<u32, u32> {
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
            let new_w_f32 = w as f32 + delta;
            let new_w = new_w_f32.max(1.0).round() as u32;

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
