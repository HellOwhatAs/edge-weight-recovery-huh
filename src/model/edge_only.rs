use crate::optimizer::{ProjectedStepStats, ProjectedSubgradientOptimizer, quantize_weights};

/// Shared edge-cost model `w_e = b_e * q_e`.
///
/// `metric_baseline` is the first-stage integer baseline
/// `round(original_baseline * quantization_scale)`. Latent multipliers remain
/// continuous; current CCH weights are produced by a second explicit
/// quantization `round(metric_baseline * q)`.
#[derive(Clone, Debug)]
pub struct EdgeOnlyModel {
    metric_baseline: Vec<u32>,
    q: Vec<f64>,
}

impl EdgeOnlyModel {
    pub fn new(original_baseline: &[u32], quantization_scale: f64) -> Result<Self, String> {
        let q = vec![1.0; original_baseline.len()];
        Self::from_q(original_baseline, quantization_scale, &q)
    }

    /// Restore a latent edge model without clamping or repairing checkpoint
    /// values. Every multiplier must already be finite, positive, and
    /// representable as a valid CCH weight.
    pub fn from_q(
        original_baseline: &[u32],
        quantization_scale: f64,
        q: &[f64],
    ) -> Result<Self, String> {
        if original_baseline.is_empty() {
            return Err("edge model requires at least one baseline edge".to_string());
        }
        if original_baseline.len() != q.len() {
            return Err(format!(
                "original baseline and q length mismatch: {} != {}",
                original_baseline.len(),
                q.len()
            ));
        }
        if let Some((edge, _)) = original_baseline
            .iter()
            .enumerate()
            .find(|(_, baseline)| **baseline == 0)
        {
            return Err(format!("edge {edge} has zero original baseline cost"));
        }
        for (edge, &multiplier) in q.iter().enumerate() {
            if !multiplier.is_finite() || multiplier <= 0.0 {
                return Err(format!(
                    "q[{edge}] must be finite and greater than zero, got {multiplier}"
                ));
            }
        }

        let unit = vec![1.0; original_baseline.len()];
        let metric_baseline = quantize_weights(original_baseline, &unit, quantization_scale)?;
        let model = Self {
            metric_baseline,
            q: q.to_vec(),
        };
        model.quantized_weights()?;
        Ok(model)
    }

    pub fn metric_baseline(&self) -> &[u32] {
        &self.metric_baseline
    }

    pub fn q(&self) -> &[f64] {
        &self.q
    }

    /// Borrow the fixed baseline and continuous multipliers for a model-level
    /// optimizer update. The baseline remains immutable because it defines the
    /// coordinate normalization throughout training.
    pub(crate) fn optimization_state_mut(&mut self) -> (&[u32], &mut [f64]) {
        (&self.metric_baseline, &mut self.q)
    }

    pub fn quantized_weights(&self) -> Result<Vec<u32>, String> {
        quantize_weights(&self.metric_baseline, &self.q, 1.0)
    }

    /// `lambda / (2m) * ||q - 1||^2`.
    pub fn regularization(&self, lambda: f64) -> f64 {
        if self.q.is_empty() {
            return 0.0;
        }
        let squared_distance = self
            .q
            .iter()
            .map(|&multiplier| {
                let difference = multiplier - 1.0;
                difference * difference
            })
            .sum::<f64>();
        lambda * squared_distance / (2.0 * self.q.len() as f64)
    }

    pub fn projected_step(
        &mut self,
        optimizer: &mut ProjectedSubgradientOptimizer,
        observed: &[u64],
        predicted: &[u64],
        sample_count: u64,
    ) -> ProjectedStepStats {
        optimizer.step(
            &mut self.q,
            &self.metric_baseline,
            observed,
            predicted,
            sample_count,
        )
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
    fn keeps_continuous_q_separate_from_two_stage_integer_weights() {
        let mut model = EdgeOnlyModel::new(&[3, 7], 2.5).unwrap();
        assert_eq!(model.metric_baseline(), &[8, 18]);
        assert_eq!(model.q(), &[1.0, 1.0]);
        assert_eq!(model.quantized_weights().unwrap(), vec![8, 18]);

        let mut optimizer = ProjectedSubgradientOptimizer::new(0.01, 0.0, 0.1, 10.0).unwrap();
        model.projected_step(&mut optimizer, &[1, 0], &[0, 1], 1);
        assert_close(model.q()[0], 0.92);
        assert_close(model.q()[1], 1.18);
        assert_eq!(model.quantized_weights().unwrap(), vec![7, 21]);
    }

    #[test]
    fn computes_the_normalized_l2_anchor() {
        let mut model = EdgeOnlyModel::new(&[10, 10, 10], 1.0).unwrap();
        let mut optimizer = ProjectedSubgradientOptimizer::new(0.01, 6.0, 0.1, 10.0).unwrap();
        model.projected_step(&mut optimizer, &[1, 1, 0], &[1, 0, 1], 1);
        assert_eq!(model.q(), &[1.0, 0.9, 1.1]);
        assert_close(model.regularization(6.0), 0.02);
    }

    #[test]
    fn strictly_restores_valid_latent_multipliers() {
        let model = EdgeOnlyModel::from_q(&[3, 7], 2.5, &[0.92, 1.18]).unwrap();
        assert_eq!(model.metric_baseline(), &[8, 18]);
        assert_eq!(model.q(), &[0.92, 1.18]);
        assert_eq!(model.quantized_weights().unwrap(), vec![7, 21]);
    }

    #[test]
    fn strict_restore_rejects_invalid_or_implicitly_repaired_state() {
        assert!(EdgeOnlyModel::from_q(&[], 1.0, &[]).is_err());
        assert!(EdgeOnlyModel::from_q(&[1, 2], 1.0, &[1.0]).is_err());
        assert!(EdgeOnlyModel::from_q(&[0], 1.0, &[1.0]).is_err());
        assert!(EdgeOnlyModel::from_q(&[1], 1.0, &[0.0]).is_err());
        assert!(EdgeOnlyModel::from_q(&[1], 1.0, &[f64::NAN]).is_err());
        assert!(EdgeOnlyModel::from_q(&[u32::MAX], 1.0, &[2.0]).is_err());
    }
}
