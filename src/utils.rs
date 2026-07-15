use rand::prelude::*;
use rand::rngs::StdRng;
use std::collections::HashSet;

/// Reproducible perturbation retained only for the legacy Adam+shock ablation.
/// Positive metric weights are preserved even when a factor is below one.
pub fn perturb_weights(
    weights: &[u32],
    target_indices: Option<&[usize]>,
    ratio: f32,
    factor: std::ops::Range<f32>,
    seed: u64,
) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let edge_count = weights.len();

    let shock_set: HashSet<usize> = if let Some(indices) = target_indices {
        indices.iter().copied().collect()
    } else {
        let shock_count = (edge_count as f32 * ratio).round() as usize;
        let mut shock_indices: Vec<usize> = (0..edge_count).collect();
        shock_indices.shuffle(&mut rng);
        shock_indices.into_iter().take(shock_count).collect()
    };

    weights
        .iter()
        .enumerate()
        .map(|(edge, &weight)| {
            if shock_set.contains(&edge) {
                let multiplier = rng.gen_range(factor.clone());
                (weight as f64 * multiplier as f64)
                    .round()
                    .clamp(1.0, (i32::MAX - 1) as f64) as u32
            } else {
                weight
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perturbation_is_positive_and_reproducible() {
        let left = perturb_weights(&[1, 10, 20], None, 1.0, 0.1..0.5, 7);
        let right = perturb_weights(&[1, 10, 20], None, 1.0, 0.1..0.5, 7);
        assert_eq!(left, right);
        assert!(left.iter().all(|&weight| weight >= 1));

        let capped = perturb_weights(&[(i32::MAX - 1) as u32], None, 1.0, 2.0..3.0, 9);
        assert_eq!(capped, vec![(i32::MAX - 1) as u32]);
    }
}
