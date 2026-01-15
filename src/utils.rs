use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rand::prelude::*;
use std::borrow::Cow;
use std::collections::HashSet;

/// Apply "Ruin & Recreate" style perturbation
pub fn perturb_weights(
    weights: &[u32],
    target_indices: Option<&[usize]>,
    ratio: f32,
    factor: std::ops::Range<f32>,
) -> Vec<u32> {
    let mut rng = rand::thread_rng();
    let edge_count = weights.len();

    let shock_set: HashSet<usize> = if let Some(indices) = target_indices {
        indices.iter().cloned().collect()
    } else {
        let shock_count = (edge_count as f32 * ratio) as usize;
        let mut shock_indices: Vec<usize> = (0..edge_count).collect();
        shock_indices.shuffle(&mut rng);
        shock_indices.into_iter().take(shock_count).collect()
    };

    weights
        .iter()
        .enumerate()
        .map(|(i, &w)| {
            if shock_set.contains(&i) {
                let factor = rng.gen_range(factor.clone());
                (w as f32 * factor) as u32
            } else {
                w
            }
        })
        .collect()
}

pub fn build_pb(
    num_epochs: u64,
    color: &str,
    prefix: impl Into<Cow<'static, str>>,
    m: &MultiProgress,
) -> ProgressBar {
    let pb = m.add(ProgressBar::new(num_epochs));
    pb.set_style(
        ProgressStyle::with_template(
            &("{prefix} [{elapsed_precise}] [{bar:40.".to_string()
                + color
                + "}] {pos}/{len} ({eta})"),
        )
        .unwrap(),
    );
    pb.set_prefix(prefix);
    pb
}
