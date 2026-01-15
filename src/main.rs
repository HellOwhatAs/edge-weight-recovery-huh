mod config;
mod graph;
mod optimizer;
mod utils;

use config::{TrainingConfig, TrainingState};
use graph::{
    compute_current_counts, compute_loss, compute_precision, compute_trip_cnt, load_graph,
};
use optimizer::AdamOptimizer;
use utils::{build_pb, perturb_weights};

use indicatif::ProgressIterator;
use routingkit_cch::{CCH, CCHMetric, CCHMetricPartialUpdater, compute_order_inertial};

fn perturb<'a>(
    state: &mut TrainingState,
    config: &TrainingConfig,
    optimizer: &mut AdamOptimizer,
    cch: &'a CCH,
    metric: &mut CCHMetric<'a>,
) -> bool {
    if state.stale_epochs <= config.patience {
        return false;
    }

    let old_era_loss = state.era_best_loss;
    state.stale_epochs = 0;
    state.restarts += 1;
    state.era_best_loss = usize::MAX;

    // Identify oscillating edges
    let oscillating_indices = optimizer.get_oscillating_indices();
    let (perturbed, strategy) = if !oscillating_indices.is_empty() {
        (
            perturb_weights(
                &state.global_best_weights,
                Some(&oscillating_indices),
                0.0,
                0.5..2.5,
            ),
            format!(
                "Targeted Oscillation Shock ({} edges)",
                oscillating_indices.len()
            ),
        )
    } else {
        (
            perturb_weights(&state.global_best_weights, None, 0.05, 0.5..2.0),
            "Random Global Shock (5% edges)".to_string(),
        )
    };

    *metric = CCHMetric::new(cch, perturbed);

    if let Some(opt) = &state.global_best_optimizer {
        *optimizer = opt.clone();
        optimizer.decay_momentum(0.5); // Decay momentum to allow exploration
    } else {
        optimizer.reset();
    }

    config.log(&format!(
        "> Stagnated (Era Loss: {old_era_loss}). {strategy}. Restarting Era."
    ));
    true
}

fn run_training(config: &TrainingConfig) {
    let city = config.city.as_str();
    let (tail, head, weights, lat, lon, paths_train, paths_test) = load_graph(city);
    let (node_count, edge_count) = (lat.len(), tail.len());

    let order = compute_order_inertial(node_count as u32, &tail, &head, &lat, &lon);
    let cch = CCH::new(&order, &tail, &head, |_| {}, false);
    let mut metric = CCHMetric::new(&cch, weights.clone());
    let mut updater = CCHMetricPartialUpdater::new(&cch);

    let num_chunks = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1);
    let trip_cnt = compute_trip_cnt(&paths_train, edge_count, num_chunks);
    let mut optimizer = AdamOptimizer::new(edge_count, config.learning_rate);
    let mut state = TrainingState::new(metric.weights());

    let pb = build_pb(config.num_epochs, "green/blue", city.to_string(), &config.m);
    for epoch in (0..config.num_epochs).progress_with(pb) {
        let cur_cnt = compute_current_counts(
            &metric,
            &paths_train,
            edge_count,
            num_chunks,
            epoch,
            city,
            &config.m,
        );
        let loss = compute_loss(&cur_cnt, &trip_cnt);

        let (is_global_best, is_era_best) =
            state.update(loss, metric.weights(), &optimizer, &config);

        if perturb(&mut state, &config, &mut optimizer, &cch, &mut metric) {
            continue;
        }

        let grads: Vec<i32> = cur_cnt
            .iter()
            .zip(trip_cnt.iter())
            .map(|(&sim, &obs)| sim as i32 - obs as i32)
            .collect();

        let updates = optimizer.step(metric.weights(), &grads);

        if updates.is_empty() {
            config.log(&format!(
                "> Converged at epoch {epoch} (global best: {})",
                state.global_best_loss
            ));
            break;
        }

        let (osc, quiet, consistency) = optimizer.diagnose();
        let diag_log = format!(" [Osc:{} Quiet:{} Cons:{:.2}]", osc, quiet, consistency);

        let (precision_log, best_marker) = match (is_global_best, is_era_best) {
            (true, _) => (
                format!(
                    "train precision {:.4}, test precision {:.4}, ",
                    compute_precision(&metric, &paths_train, &weights, num_chunks),
                    compute_precision(&metric, &paths_test, &weights, num_chunks)
                ),
                "(BEST)",
            ),
            (_, true) => (String::new(), "(ERA_BEST)"),
            _ => (String::new(), ""),
        };

        config.log(&format!(
            "Epoch {epoch}: train loss {loss}, {precision_log}{} edges changed {best_marker}{diag_log}",
            updates.len(),
        ));
        updater.apply(&mut metric, &updates);
    }

    let weights = metric.weights().to_vec();
    if let Ok(json) = serde_json::to_string(&weights) {
        std::fs::write(&config.best_weights_path, json).expect("Failed to write weights to file");
    }
    if !config.save_best_immediately {
        if let Ok(json) = serde_json::to_string(&state.global_best_weights) {
            let _ = std::fs::write(&config.best_weights_path, json);
        }
    }
    config.log(&format!(
        "> Finished. Global Best Loss={}",
        state.global_best_loss
    ));
}

fn main() {
    let config = TrainingConfig::new("beijing");
    run_training(&config);
}
