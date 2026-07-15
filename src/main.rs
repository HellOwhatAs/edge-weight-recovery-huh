use edge_weight_recovery::config::{
    MetricUpdateMode, SelectionMetric, SolverKind, TrainingConfig, TrainingState,
};
use edge_weight_recovery::graph::{
    PathMetrics, PathValidationReport, compute_observed_edge_counts, compute_oracle_stats,
    compute_regret, count_residual_l1, evaluate_paths, group_paths_by_od, load_graph, load_trips,
};
use edge_weight_recovery::optimizer::{
    AdamOptimizer, ProjectedSubgradientOptimizer, quantize_weights, regularization_loss,
};
use edge_weight_recovery::utils::perturb_weights;
use routingkit_cch::{CCH, CCHMetric, CCHMetricPartialUpdater, compute_order_inertial};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::{Duration, Instant};

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let Some(config) = TrainingConfig::from_args()? else {
        return Ok(());
    };
    prepare_output(&config)?;
    run_training(&config)
}

fn prepare_output(config: &TrainingConfig) -> Result<(), String> {
    for output in [
        &config.log_path,
        &config.best_weights_path,
        &config.best_multipliers_path,
        &config.checkpoint_path,
    ] {
        if let Some(parent) = Path::new(output).parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
        }
    }
    std::fs::write(&config.log_path, "")
        .map_err(|error| format!("failed to initialize {}: {error}", config.log_path))
}

fn run_training(config: &TrainingConfig) -> Result<(), String> {
    config.log(&format!(
        "CONFIG city={} epochs={} train={} validation={} test={} solver={:?} metric_update={:?} selection_metric={:?} \
         eta0={} lambda={} q_box=[{},{}] scale={} cycle_policy={:?} trim_boundary_edges={} \
         run_test={} seed={}",
        config.city,
        config.num_epochs,
        config.train_variant,
        config.validation_variant,
        config.test_variant,
        config.solver,
        config.metric_update_mode,
        config.selection_metric,
        config.eta0,
        config.lambda,
        config.q_min,
        config.q_max,
        config.quantization_scale,
        config.cycle_policy,
        config.trim_boundary_edges,
        config.run_test,
        config.random_seed,
    ))?;

    let load_started = Instant::now();
    let graph = load_graph(&config.city)?;
    let train = load_trips(
        &config.city,
        "train",
        &config.train_variant,
        &graph,
        config.max_train_samples,
        config.trim_boundary_edges,
        config.cycle_policy,
    )?;
    if train.paths.is_empty() {
        return Err("no valid training paths remain after validation".to_string());
    }
    log_split_report(config, "train", &train.report)?;

    let validation = if config.eval_every == 0 {
        None
    } else {
        let loaded = load_trips(
            &config.city,
            "validation",
            &config.validation_variant,
            &graph,
            config.max_validation_samples,
            config.trim_boundary_edges,
            config.cycle_policy,
        )?;
        if loaded.paths.is_empty() {
            return Err("no valid validation paths remain after validation".to_string());
        }
        log_split_report(config, "validation", &loaded.report)?;
        Some(loaded)
    };
    config.log(&format!(
        "LOAD nodes={} edges={} valid_train={} wall_ms={:.3}",
        graph.x.len(),
        graph.tail.len(),
        train.paths.len(),
        milliseconds(load_started.elapsed())
    ))?;

    let thread_count = rayon::current_num_threads().max(1);
    let edge_count = graph.tail.len();
    let train_observed = compute_observed_edge_counts(&train.paths, edge_count, thread_count);
    let train_groups = group_paths_by_od(&train.paths);
    let train_reduction = 100.0 * (1.0 - train_groups.len() as f64 / train.paths.len() as f64);
    config.log(&format!(
        "OD_GROUP train_samples={} unique_od={} query_reduction_pct={train_reduction:.3}",
        train.paths.len(),
        train_groups.len()
    ))?;

    let validation_data = validation.as_ref().map(|loaded| {
        (
            compute_observed_edge_counts(&loaded.paths, edge_count, thread_count),
            group_paths_by_od(&loaded.paths),
        )
    });

    let cch_started = Instant::now();
    let order = compute_order_inertial(
        graph.x.len() as u32,
        &graph.tail,
        &graph.head,
        &graph.x,
        &graph.y,
    );
    let cch = CCH::new(&order, &graph.tail, &graph.head, |_| {}, false);
    let cch_build_duration = cch_started.elapsed();

    let mut latent_q = vec![1.0; edge_count];
    // Incorporate the optional fixed scale into the baseline once. q remains a
    // dimensionless multiplier around this integer oracle baseline.
    let metric_baseline = quantize_weights(
        &graph.baseline_weights,
        &latent_q,
        config.quantization_scale,
    )?;
    let customization_started = Instant::now();
    let mut metric = CCHMetric::new(&cch, metric_baseline.clone());
    let initial_customization_duration = customization_started.elapsed();
    let mut partial_updater = CCHMetricPartialUpdater::new(&cch);

    config.log(&format!(
        "CCH build_ms={:.3} initial_full_customization_ms={:.3} threads={thread_count}",
        milliseconds(cch_build_duration),
        milliseconds(initial_customization_duration)
    ))?;

    let mut projected_optimizer = match config.solver {
        SolverKind::ProjectedSubgradient => Some(ProjectedSubgradientOptimizer::new(
            config.eta0,
            config.lambda,
            config.q_min,
            config.q_max,
        )?),
        SolverKind::LegacyAdamShock => None,
    };
    let mut adam_optimizer = match config.solver {
        SolverKind::ProjectedSubgradient => None,
        SolverKind::LegacyAdamShock => {
            Some(AdamOptimizer::new(edge_count, config.adam_learning_rate))
        }
    };
    let mut best_adam_optimizer: Option<AdamOptimizer> = None;
    let mut restarts = 0u64;
    let mut state = TrainingState::new(metric.weights(), &latent_q);
    let mut initial_train_regret = None;

    for epoch in 0..config.num_epochs {
        let epoch_started = Instant::now();
        let oracle = compute_oracle_stats(&metric, &train_groups, edge_count, thread_count)?;
        let regret = compute_regret(metric.weights(), &train_observed, &oracle)?;
        initial_train_regret.get_or_insert(regret.mean_data_loss);
        let residual = count_residual_l1(&oracle.predicted_edge_counts, &train_observed)?;
        let regularization = match config.solver {
            SolverKind::ProjectedSubgradient => regularization_loss(&latent_q, config.lambda),
            SolverKind::LegacyAdamShock => 0.0,
        };
        let train_objective = regret.mean_data_loss + regularization;
        let (current_q_min, current_q_max, current_q_at_min, current_q_at_max) =
            multiplier_summary(&latent_q, config.q_min, config.q_max);
        let current_quantization_error =
            max_quantization_error(metric.weights(), &metric_baseline, &latent_q);

        let should_evaluate = validation_data.is_some()
            && (epoch == 0
                || (epoch + 1) % config.eval_every == 0
                || epoch + 1 == config.num_epochs);
        let mut validation_log = String::new();
        let (selection_loss, evaluated) = if should_evaluate {
            let (validation_observed, validation_groups) =
                validation_data.as_ref().expect("checked above");
            let validation_oracle =
                compute_oracle_stats(&metric, validation_groups, edge_count, thread_count)?;
            let validation_regret =
                compute_regret(metric.weights(), validation_observed, &validation_oracle)?;
            validation_log = format!(
                " validation_regret={:.6} validation_relative_regret={:.8} validation_oracle_ms={:.3}",
                validation_regret.mean_data_loss,
                validation_regret.relative_data_loss,
                milliseconds(validation_oracle.oracle_duration)
            );
            // Regularization shapes the learned parameters through the training
            // update. Held-out checkpoint selection uses only the validation
            // task metric so lambda is not counted a second time.
            let selection_loss = match config.selection_metric {
                SelectionMetric::MeanRegret => validation_regret.mean_data_loss,
                SelectionMetric::RelativeRegret => validation_regret.relative_data_loss,
            };
            (selection_loss, true)
        } else if validation_data.is_none() {
            (train_objective, true)
        } else {
            (f64::INFINITY, false)
        };

        let is_best = if evaluated {
            state.update(
                epoch,
                selection_loss,
                regret.mean_data_loss,
                metric.weights(),
                &latent_q,
            )
        } else {
            false
        };
        let mut checkpoint_duration = Duration::ZERO;
        if is_best {
            let checkpoint_started = Instant::now();
            state.save(config)?;
            checkpoint_duration = checkpoint_started.elapsed();
            if let Some(optimizer) = &adam_optimizer {
                best_adam_optimizer = Some(optimizer.clone());
            }
        }
        let stop_for_patience = config.solver == SolverKind::ProjectedSubgradient
            && evaluated
            && state.stale_evaluations >= config.patience;

        let mut optimizer_duration = Duration::ZERO;
        let mut customization_duration = Duration::ZERO;
        let mut changed_edges = 0usize;
        let mut step_log = String::new();
        let mut update_status = if stop_for_patience {
            "early_stop_skipped"
        } else if epoch + 1 == config.num_epochs {
            "final_skipped"
        } else {
            "applied"
        };

        // Do not make an unmeasured final update: every saved/final weight vector
        // has now been evaluated on the declared objective.
        if epoch + 1 < config.num_epochs && !stop_for_patience {
            if config.solver == SolverKind::LegacyAdamShock
                && state.stale_evaluations > config.patience
            {
                let optimizer = adam_optimizer.as_mut().expect("legacy solver");
                let oscillating = optimizer.get_oscillating_indices();
                let (perturbed, strategy) = if oscillating.is_empty() {
                    (
                        perturb_weights(
                            &state.best_weights,
                            None,
                            0.05,
                            0.5..2.0,
                            config.random_seed.wrapping_add(restarts),
                        ),
                        "global_5pct",
                    )
                } else {
                    (
                        perturb_weights(
                            &state.best_weights,
                            Some(&oscillating),
                            0.0,
                            0.5..2.5,
                            config.random_seed.wrapping_add(restarts),
                        ),
                        "oscillation_targeted",
                    )
                };
                let updates = changed_weight_map(metric.weights(), &perturbed);
                changed_edges = updates.len();
                customization_duration = apply_metric_weights(
                    &cch,
                    &mut partial_updater,
                    &mut metric,
                    perturbed,
                    &updates,
                    config.metric_update_mode,
                );
                latent_q = multipliers_from_weights(metric.weights(), &metric_baseline);
                if let Some(best_optimizer) = &best_adam_optimizer {
                    *optimizer = best_optimizer.clone();
                    optimizer.decay_momentum(0.5);
                } else {
                    optimizer.reset();
                }
                state.stale_evaluations = 0;
                restarts += 1;
                step_log = format!(" shock={strategy} restart={restarts}");
                update_status = "shock_applied";
            } else {
                let optimizer_started = Instant::now();
                let new_weights = match config.solver {
                    SolverKind::ProjectedSubgradient => {
                        let step = projected_optimizer
                            .as_mut()
                            .expect("projected solver")
                            .step(
                                &mut latent_q,
                                &metric_baseline,
                                &train_observed,
                                &oracle.predicted_edge_counts,
                                oracle.sample_count,
                            );
                        step_log = format!(
                            " eta={:.8} latent_max_delta={:.8} projected_edges={}",
                            step.eta, step.max_abs_delta, step.projected_edges
                        );
                        quantize_weights(&metric_baseline, &latent_q, 1.0)?
                    }
                    SolverKind::LegacyAdamShock => {
                        let gradients =
                            count_difference_i64(&oracle.predicted_edge_counts, &train_observed)?;
                        let updates = adam_optimizer
                            .as_mut()
                            .expect("legacy solver")
                            .step(metric.weights(), &gradients);
                        let mut weights = metric.weights().to_vec();
                        for (&edge, &weight) in &updates {
                            weights[edge as usize] = weight;
                        }
                        let (oscillating, quiet, consistency) =
                            adam_optimizer.as_ref().expect("legacy solver").diagnose();
                        step_log = format!(
                            " oscillating_edges={oscillating} quiet_edges={quiet} consistency={consistency:.4}"
                        );
                        weights
                    }
                };
                optimizer_duration = optimizer_started.elapsed();
                let updates = changed_weight_map(metric.weights(), &new_weights);
                changed_edges = updates.len();
                customization_duration = apply_metric_weights(
                    &cch,
                    &mut partial_updater,
                    &mut metric,
                    new_weights,
                    &updates,
                    config.metric_update_mode,
                );
                if config.solver == SolverKind::LegacyAdamShock {
                    latent_q = multipliers_from_weights(metric.weights(), &metric_baseline);
                }
                if changed_edges == 0 {
                    update_status = "latent_only_no_integer_change";
                }
            }
        }

        let (q_min, q_max, q_at_min, q_at_max) =
            multiplier_summary(&latent_q, config.q_min, config.q_max);
        let quantization_error =
            max_quantization_error(metric.weights(), &metric_baseline, &latent_q);
        let best_marker = if is_best { " BEST" } else { "" };
        let selection_log = if evaluated {
            format!(" selection_loss={selection_loss:.12}")
        } else {
            " selection_loss=NA".to_string()
        };
        config.log(&format!(
            "EPOCH epoch={epoch} train_regret={:.6} train_relative_regret={:.8} \
             regularization={regularization:.6} \
             train_objective={train_objective:.6} count_residual_l1={residual} \
             train_queries={} train_oracle_ms={:.3} changed_edges={} changed_pct={:.4} \
             optimizer_ms={:.3} customization_ms={:.3} checkpoint_ms={:.3} update_mode={:?} \
             update_status={update_status} current_q_min={current_q_min:.6} \
             current_q_max={current_q_max:.6} current_q_at_min={current_q_at_min} \
             current_q_at_max={current_q_at_max} current_max_quantization_error={current_quantization_error:.6} \
             next_q_min={q_min:.6} next_q_max={q_max:.6} next_q_at_min={q_at_min} \
             next_q_at_max={q_at_max} next_max_quantization_error={quantization_error:.6} \
             epoch_ms={:.3}{}{}{}{}",
            regret.mean_data_loss,
            regret.relative_data_loss,
            oracle.num_queries,
            milliseconds(oracle.oracle_duration),
            changed_edges,
            100.0 * changed_edges as f64 / edge_count.max(1) as f64,
            milliseconds(optimizer_duration),
            milliseconds(customization_duration),
            milliseconds(checkpoint_duration),
            config.metric_update_mode,
            milliseconds(epoch_started.elapsed()),
            validation_log,
            selection_log,
            step_log,
            best_marker,
        ))?;
        if stop_for_patience {
            config.log(&format!(
                "EARLY_STOP epoch={epoch} stale_evaluations={} patience={}",
                state.stale_evaluations, config.patience
            ))?;
            break;
        }
    }

    // Reconstruct the metric from the selected checkpoint. The final files and
    // held-out results can therefore never refer to an unselected last update.
    state.save(config)?;
    let restore_started = Instant::now();
    metric = CCHMetric::new(&cch, state.best_weights.clone());
    let restore_duration = restore_started.elapsed();

    if let Some(validation_loaded) = &validation {
        let metrics = evaluate_paths(&metric, &validation_loaded.paths, thread_count)?;
        log_path_metrics(config, "validation_best", &metrics)?;
    }

    // Release potentially large training/validation path vectors before loading
    // the held-out full test split.
    drop(train_groups);
    drop(train_observed);
    drop(train);
    drop(validation_data);
    drop(validation);

    if config.run_test {
        let test = load_trips(
            &config.city,
            "test",
            &config.test_variant,
            &graph,
            config.max_test_samples,
            config.trim_boundary_edges,
            config.cycle_policy,
        )?;
        if test.paths.is_empty() {
            return Err("no valid test paths remain after validation".to_string());
        }
        log_split_report(config, "test", &test.report)?;
        let test_metrics = evaluate_paths(&metric, &test.paths, thread_count)?;
        log_path_metrics(config, "test_final", &test_metrics)?;
    } else {
        config.log("TEST_SKIPPED use --run-test only after freezing the experiment protocol")?;
    }

    let initial = initial_train_regret.unwrap_or(0.0);
    let improvement = if initial > 0.0 {
        100.0 * (initial - state.best_train_data_loss) / initial
    } else {
        0.0
    };
    let (best_q_min, best_q_max, best_q_at_min, best_q_at_max) =
        multiplier_summary(&state.best_multipliers, config.q_min, config.q_max);
    let best_regularization = match config.solver {
        SolverKind::ProjectedSubgradient => {
            regularization_loss(&state.best_multipliers, config.lambda)
        }
        SolverKind::LegacyAdamShock => 0.0,
    };
    let peak_rss_kib = process_peak_rss_kib().unwrap_or(0);
    config.log(&format!(
        "FINISHED best_epoch={} selection_loss={:.12} best_train_regret={:.6} \
         best_regularization={best_regularization:.6} best_q_min={best_q_min:.6} \
         best_q_max={best_q_max:.6} best_q_at_min={best_q_at_min} best_q_at_max={best_q_at_max} \
         train_regret_improvement_pct={improvement:.3} restore_full_customization_ms={:.3} \
         peak_rss_kib={peak_rss_kib} checkpoint_path={} weights_path={} multipliers_path={}",
        state.best_epoch,
        state.best_selection_loss,
        state.best_train_data_loss,
        milliseconds(restore_duration),
        config.checkpoint_path,
        config.best_weights_path,
        config.best_multipliers_path,
    ))
}

fn process_peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

fn log_split_report(
    config: &TrainingConfig,
    split: &str,
    report: &PathValidationReport,
) -> Result<(), String> {
    config.log(&format!(
        "DATA split={split} available={} inspected={} accepted={} dropped={} cyclic={} \
         empty_or_short={} out_of_bounds={} discontinuous={} trimmed_edges={}",
        report.available_samples,
        report.inspected_samples,
        report.accepted_samples,
        report.dropped_samples(),
        report.cyclic,
        report.empty_or_too_short,
        report.out_of_bounds,
        report.discontinuous,
        report.trimmed_boundary_edges,
    ))
}

fn log_path_metrics(
    config: &TrainingConfig,
    label: &str,
    metrics: &PathMetrics,
) -> Result<(), String> {
    config.log(&format!(
        "EVAL split={label} samples={} mean_regret={:.6} relative_regret={:.8} exact_match={:.6} \
         edge_precision={:.6} edge_recall={:.6} edge_f1={:.6} edge_jaccard={:.6}",
        metrics.sample_count,
        metrics.mean_regret,
        metrics.relative_regret,
        metrics.exact_match,
        metrics.edge_precision,
        metrics.edge_recall,
        metrics.edge_f1,
        metrics.edge_jaccard,
    ))
}

fn changed_weight_map(current: &[u32], next: &[u32]) -> BTreeMap<u32, u32> {
    assert_eq!(current.len(), next.len(), "metric weight length mismatch");
    current
        .iter()
        .zip(next)
        .enumerate()
        .filter_map(|(edge, (&old, &new))| (old != new).then_some((edge as u32, new)))
        .collect()
}

fn apply_metric_weights<'a>(
    cch: &'a CCH,
    updater: &mut CCHMetricPartialUpdater<'a>,
    metric: &mut CCHMetric<'a>,
    new_weights: Vec<u32>,
    updates: &BTreeMap<u32, u32>,
    mode: MetricUpdateMode,
) -> Duration {
    if updates.is_empty() {
        return Duration::ZERO;
    }
    let started = Instant::now();
    match mode {
        MetricUpdateMode::Partial => updater.apply(metric, updates),
        MetricUpdateMode::Full => *metric = CCHMetric::new(cch, new_weights),
    }
    started.elapsed()
}

fn count_difference_i64(predicted: &[u64], observed: &[u64]) -> Result<Vec<i64>, String> {
    if predicted.len() != observed.len() {
        return Err("predicted and observed count lengths differ".to_string());
    }
    predicted
        .iter()
        .zip(observed)
        .enumerate()
        .map(|(edge, (&predicted, &observed))| {
            if predicted >= observed {
                i64::try_from(predicted - observed)
                    .map_err(|_| format!("positive count difference overflows i64 at edge {edge}"))
            } else {
                i64::try_from(observed - predicted)
                    .map(|difference| -difference)
                    .map_err(|_| format!("negative count difference overflows i64 at edge {edge}"))
            }
        })
        .collect()
}

fn multipliers_from_weights(weights: &[u32], baseline: &[u32]) -> Vec<f64> {
    assert_eq!(weights.len(), baseline.len(), "baseline length mismatch");
    weights
        .iter()
        .zip(baseline)
        .map(|(&weight, &base)| weight as f64 / base as f64)
        .collect()
}

fn multiplier_summary(q: &[f64], lower: f64, upper: f64) -> (f64, f64, usize, usize) {
    let min = q.iter().copied().fold(f64::INFINITY, f64::min);
    let max = q.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let at_min = q.iter().filter(|&&value| value <= lower + 1e-12).count();
    let at_max = q.iter().filter(|&&value| value >= upper - 1e-12).count();
    (min, max, at_min, at_max)
}

fn max_quantization_error(weights: &[u32], baseline: &[u32], q: &[f64]) -> f64 {
    assert_eq!(weights.len(), baseline.len(), "baseline length mismatch");
    assert_eq!(weights.len(), q.len(), "multiplier length mismatch");
    weights
        .iter()
        .zip(baseline)
        .zip(q)
        .map(|((&weight, &base), &multiplier)| (weight as f64 - base as f64 * multiplier).abs())
        .fold(0.0, f64::max)
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_difference_uses_a_wide_signed_type() {
        assert_eq!(
            count_difference_i64(&[u32::MAX as u64 + 5, 1], &[1, 4]).unwrap(),
            vec![u32::MAX as i64 + 4, -3]
        );
    }

    #[test]
    fn changed_weights_only_contains_integer_metric_changes() {
        let updates = changed_weight_map(&[1, 2, 3], &[1, 4, 3]);
        assert_eq!(updates, BTreeMap::from([(1, 4)]));
    }
}
