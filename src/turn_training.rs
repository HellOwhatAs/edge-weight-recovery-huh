use crate::config::{TurnExperimentArm, TurnTrainingConfig, atomic_write, load_checkpoint};
use crate::data::{
    GraphData, PathValidationReport, compute_observed_edge_counts,
    compute_observed_transition_counts, group_paths_by_od,
};
use crate::evaluation::{PathMetrics, evaluate_expanded_paths};
use crate::model::{EdgeOnlyModel, TurnAwareModel};
use crate::objective::{compute_turn_aware_regret, count_residual_l1};
use crate::optimizer::{ProjectedSubgradientOptimizer, TurnResidualOptimizer};
use crate::oracle::ExpandedCchOracle;
use crate::training::{
    JsonlLogger, baseline_fingerprint, log_data_report, metrics_json, milliseconds,
    process_peak_rss_kib, q_summary,
};
use crate::turn_graph::ExpandedTurnGraph;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct TurnTrainingOutcome {
    pub checkpoint_path: PathBuf,
    pub log_path: PathBuf,
    pub best_step: u64,
    pub selection_value: f64,
    pub validation_metrics: PathMetrics,
}

struct EdgeInitialization {
    q: Vec<f64>,
    weights: Vec<u32>,
    completed_updates: u64,
    identity: Value,
}

struct TurnTrainingState {
    best_selection_value: f64,
    best_train_mean_regret: f64,
    best_step: u64,
    best_q: Vec<f64>,
    best_residuals: Vec<f64>,
    best_edge_weights: Vec<u32>,
    best_transition_weights: Vec<u32>,
    best_q_completed_updates: u64,
    best_r_completed_updates: u64,
}

struct ConsideredState<'a> {
    step: u64,
    selection_value: f64,
    train_mean_regret: f64,
    q: &'a [f64],
    residuals: &'a [f64],
    edge_weights: &'a [u32],
    transition_weights: &'a [u32],
    q_completed_updates: u64,
    r_completed_updates: u64,
}

struct RestoredTurnCheckpoint {
    step: u64,
    selection_value: f64,
    train_mean_regret: f64,
    q: Vec<f64>,
    residuals: Vec<f64>,
    edge_weights: Vec<u32>,
    transition_weights: Vec<u32>,
    q_completed_updates: u64,
    r_completed_updates: u64,
}

struct RestoreContext<'a> {
    config: &'a TurnTrainingConfig,
    graph: &'a GraphData,
    expanded: &'a ExpandedTurnGraph,
    runtime_identity: &'a Value,
    topology_identity: &'a str,
    initialization_identity: &'a Value,
    initial_q: &'a [f64],
    initial_q_completed_updates: u64,
}

/// Run one configured expanded-graph turn-residual update mode.
///
/// All modes share the same topology, OD grouping, integer metric, validation
/// selection, and logging. The arm enum controls only which latent blocks are
/// updated after both gradients have been computed from the same query state;
/// it does not encode a scientific ranking among those modes.
pub fn run_turn_training(
    config: &TurnTrainingConfig,
    output_dir: &Path,
) -> Result<TurnTrainingOutcome, String> {
    let actual_threads = rayon::current_num_threads().max(1);
    if actual_threads != config.rayon_threads {
        return Err(format!(
            "protocol requires {} Rayon threads, process has {actual_threads}; set RAYON_NUM_THREADS before launch",
            config.rayon_threads
        ));
    }

    let mut logger = JsonlLogger::new(output_dir)?;
    logger.log(json!({
        "event": "configuration",
        "run_id": config.run_id,
        "protocol_id": config.protocol_id,
        "stage": config.stage,
        "arm": config.arm.as_str(),
        "configuration": config.as_json(),
        "test_read": false,
    }))?;

    let load_started = Instant::now();
    let graph = crate::data::load_graph(&config.city)?;
    verify_declared_data(config, "train", &config.train_variant)?;
    verify_declared_data(config, "validation", &config.validation_variant)?;
    let train =
        crate::data::load_trips(&config.city, "train", &config.train_variant, &graph, None)?;
    let validation = crate::data::load_trips(
        &config.city,
        "validation",
        &config.validation_variant,
        &graph,
        None,
    )?;
    if train.paths.is_empty() || validation.paths.is_empty() {
        return Err("turn-aware train and validation must both contain valid paths".to_string());
    }
    verify_declared_sample_count(config, "train", &train.report)?;
    verify_declared_sample_count(config, "validation", &validation.report)?;
    log_data_report(&mut logger, "train", &config.train_variant, &train.report)?;
    log_data_report(
        &mut logger,
        "validation",
        &config.validation_variant,
        &validation.report,
    )?;

    let initialization = load_edge_initialization(config, &graph)?;
    let expanded = ExpandedTurnGraph::build(&graph)?;
    let runtime_identity = runtime_identity(
        config,
        &graph,
        &expanded,
        &train.report,
        &validation.report,
        &initialization.identity,
    );
    logger.log(json!({
        "event": "loaded",
        "nodes": graph.x.len(),
        "edges": graph.tail.len(),
        "expanded_states": expanded.stats.expanded_nodes,
        "transitions": expanded.transition_count(),
        "state_self_transitions": expanded.stats.state_self_transitions,
        "valid_train": train.paths.len(),
        "valid_validation": validation.paths.len(),
        "wall_ms": milliseconds(load_started),
        "runtime_identity": &runtime_identity,
    }))?;

    let train_observed_edges =
        compute_observed_edge_counts(&train.paths, graph.tail.len(), actual_threads);
    let train_observed_transitions =
        compute_observed_transition_counts(&train.paths, &expanded, actual_threads)?;
    let validation_observed_edges =
        compute_observed_edge_counts(&validation.paths, graph.tail.len(), actual_threads);
    let validation_observed_transitions =
        compute_observed_transition_counts(&validation.paths, &expanded, actual_threads)?;
    let train_groups = group_paths_by_od(&train.paths);
    let validation_groups = group_paths_by_od(&validation.paths);
    logger.log(json!({
        "event": "od_grouping",
        "train_samples": train.paths.len(),
        "train_unique_od": train_groups.len(),
        "validation_samples": validation.paths.len(),
        "validation_unique_od": validation_groups.len(),
        "observed_train_transition_uses": train_observed_transitions.iter().sum::<u64>(),
    }))?;

    let build_started = Instant::now();
    let oracle = ExpandedCchOracle::build(&graph, &expanded)?;
    let edge_model = EdgeOnlyModel::from_q(
        &graph.baseline_weights,
        config.quantization_scale,
        &initialization.q,
    )?;
    let mut model = TurnAwareModel::new(edge_model, &expanded, config.residual_scale)?;
    let initial_edge_weights = model.quantized_edge_weights()?;
    if initial_edge_weights != initialization.weights {
        return Err("restored q does not reproduce the imported edge weights".to_string());
    }
    let resolved_scale = median_u32(&initial_edge_weights)?;
    if resolved_scale != config.residual_scale {
        return Err(format!(
            "configured residual scale {} does not equal frozen edge-metric median {resolved_scale}",
            config.residual_scale
        ));
    }
    let initial_transition_weights = model.quantized_transition_weights(&expanded)?;
    let customization_started = Instant::now();
    let mut metric = oracle.customize(&initial_edge_weights, &initial_transition_weights)?;
    logger.log(json!({
        "event": "oracle_built",
        "kind": "expanded_cch",
        "customization": "full",
        "build_ms": milliseconds(build_started),
        "initial_customization_ms": milliseconds(customization_started),
        "topology_identity": oracle.topology_identity(),
        "threads": actual_threads,
    }))?;

    let mut edge_optimizer = ProjectedSubgradientOptimizer::with_completed_updates(
        config.eta_q0,
        config.lambda_edge,
        config.q_min,
        config.q_max,
        initialization.completed_updates,
    )?;
    let mut residual_optimizer = if config.arm.updates_residuals() {
        Some(TurnResidualOptimizer::new(
            config.eta_r0.expect("validated residual eta"),
            config.lambda_turn.expect("validated turn lambda"),
            config.r_max,
        )?)
    } else {
        None
    };
    let mut state = TurnTrainingState::new();

    for step in 0..=config.updates {
        let step_started = Instant::now();
        let train_oracle = metric.batch_stats(&train_groups, actual_threads)?;
        let train_regret = compute_turn_aware_regret(
            &expanded,
            metric.edge_weights(),
            metric.transition_weights(),
            &train_observed_edges,
            &train_observed_transitions,
            &train_oracle,
        )?;
        let edge_count_residual =
            count_residual_l1(&train_oracle.predicted_edge_counts, &train_observed_edges)?;
        let transition_count_residual = count_residual_l1(
            &train_oracle.predicted_transition_counts,
            &train_observed_transitions,
        )?;
        let edge_regularization = model.edge_only().regularization(config.lambda_edge);
        let turn_regularization = model.residual_regularization(config.lambda_turn.unwrap_or(0.0));
        let train_objective =
            train_regret.mean_data_loss + edge_regularization + turn_regularization;
        let pre_q_completed_updates = edge_optimizer.completed_updates();
        let pre_r_completed_updates = residual_optimizer
            .as_ref()
            .map_or(0, TurnResidualOptimizer::completed_updates);
        validate_clock_contract(
            config.arm,
            initialization.completed_updates,
            step,
            pre_q_completed_updates,
            pre_r_completed_updates,
        )?;
        let current_edge_weights = metric.edge_weights().to_vec();
        let current_transition_weights = metric.transition_weights().to_vec();
        let current_q_summary = q_summary(model.edge_only().q(), config.q_min, config.q_max);
        let current_q_l2_drift = l2_distance(model.edge_only().q(), &initialization.q);
        let current_residual_summary = residual_summary_json(
            &model,
            &expanded,
            &current_edge_weights,
            &current_transition_weights,
            config.r_max,
        );

        let should_validate = step % config.validation_every == 0 || step == config.updates;
        let mut validation_event = Value::Null;
        let mut is_best = false;
        if should_validate {
            let validation_oracle = metric.batch_stats(&validation_groups, actual_threads)?;
            let validation_regret = compute_turn_aware_regret(
                &expanded,
                metric.edge_weights(),
                metric.transition_weights(),
                &validation_observed_edges,
                &validation_observed_transitions,
                &validation_oracle,
            )?;
            // Preserve the archived protocol's checkpoint behavior exactly.
            // This ratio uses the current model's observed cost as its
            // denominator, so selecting it here must not be interpreted as a
            // fair cross-model ranking of edge-only, turn-only, and joint.
            is_best = state.consider(ConsideredState {
                step,
                selection_value: validation_regret.relative_data_loss,
                train_mean_regret: train_regret.mean_data_loss,
                q: model.edge_only().q(),
                residuals: model.transition_residuals(),
                edge_weights: metric.edge_weights(),
                transition_weights: metric.transition_weights(),
                q_completed_updates: pre_q_completed_updates,
                r_completed_updates: pre_r_completed_updates,
            });
            if is_best {
                state.save_checkpoint(
                    output_dir,
                    config,
                    &runtime_identity,
                    oracle.topology_identity(),
                    &initialization.identity,
                )?;
            }
            validation_event = json!({
                "mean_regret": validation_regret.mean_data_loss,
                "relative_regret": validation_regret.relative_data_loss,
                "selection_value": validation_regret.relative_data_loss,
                "queries": validation_oracle.num_queries,
                "oracle_ms": validation_oracle.oracle_duration.as_secs_f64() * 1_000.0,
                "is_best": is_best,
            });
        }

        let mut update = json!({
            "status": "final_skipped",
            "q_completed_updates_before": pre_q_completed_updates,
            "q_completed_updates_after": pre_q_completed_updates,
            "r_completed_updates_before": pre_r_completed_updates,
            "r_completed_updates_after": pre_r_completed_updates,
        });
        if step < config.updates {
            let optimizer_started = Instant::now();
            // Both block gradients consume counts from the one oracle query
            // above. Neither block is allowed to re-query after the other has
            // changed its latent state.
            let edge_step = if config.arm.updates_q() {
                Some(model.projected_edge_step(
                    &mut edge_optimizer,
                    &train_observed_edges,
                    &train_oracle.predicted_edge_counts,
                    train_oracle.sample_count,
                ))
            } else {
                None
            };
            let residual_step = if let Some(optimizer) = residual_optimizer.as_mut() {
                Some(model.projected_residual_step(
                    optimizer,
                    &train_observed_transitions,
                    &train_oracle.predicted_transition_counts,
                    train_oracle.sample_count,
                ))
            } else {
                None
            };
            validate_frozen_blocks(
                config.arm,
                model.edge_only().q(),
                model.transition_residuals(),
                &initialization.q,
            )?;
            let post_q_completed_updates = edge_optimizer.completed_updates();
            let post_r_completed_updates = residual_optimizer
                .as_ref()
                .map_or(0, TurnResidualOptimizer::completed_updates);
            validate_clock_contract(
                config.arm,
                initialization.completed_updates,
                step.checked_add(1)
                    .ok_or_else(|| "local update step overflow".to_string())?,
                post_q_completed_updates,
                post_r_completed_updates,
            )?;
            let optimizer_ms = milliseconds(optimizer_started);
            let next_edge_weights = model.quantized_edge_weights()?;
            let next_transition_weights = model.quantized_transition_weights(&expanded)?;
            let changed_edges = changed_count(&current_edge_weights, &next_edge_weights);
            let changed_transitions =
                changed_count(&current_transition_weights, &next_transition_weights);
            let customization_started = Instant::now();
            let customization_ms = if changed_edges == 0 && changed_transitions == 0 {
                0.0
            } else {
                metric = oracle.customize(&next_edge_weights, &next_transition_weights)?;
                milliseconds(customization_started)
            };
            update = json!({
                "status": if changed_edges == 0 && changed_transitions == 0 {
                    "latent_only_no_integer_change"
                } else {
                    "applied"
                },
                "edge": edge_step.map(|value| json!({
                    "eta": value.eta,
                    "latent_max_delta": value.max_abs_delta,
                    "projected": value.projected_edges,
                })),
                "turn": residual_step.map(|value| json!({
                    "eta": value.eta,
                    "latent_max_delta": value.max_abs_delta,
                    "projected": value.projected_transitions,
                })),
                "changed_edges": changed_edges,
                "changed_transitions": changed_transitions,
                "q_completed_updates_before": pre_q_completed_updates,
                "q_completed_updates_after": post_q_completed_updates,
                "r_completed_updates_before": pre_r_completed_updates,
                "r_completed_updates_after": post_r_completed_updates,
                "optimizer_ms": optimizer_ms,
                "customization_ms": customization_ms,
            });
        }

        logger.log(json!({
            "event": "step",
            "step": step,
            "arm": config.arm.as_str(),
            "q_completed_updates": pre_q_completed_updates,
            "r_completed_updates": pre_r_completed_updates,
            "train_mean_regret": train_regret.mean_data_loss,
            "train_relative_regret": train_regret.relative_data_loss,
            "edge_regularization": edge_regularization,
            "turn_regularization": turn_regularization,
            "train_objective": train_objective,
            "edge_count_residual_l1_diagnostic": edge_count_residual,
            "transition_count_residual_l1_diagnostic": transition_count_residual,
            "train_queries": train_oracle.num_queries,
            "train_oracle_ms": train_oracle.oracle_duration.as_secs_f64() * 1_000.0,
            "q": current_q_summary,
            "q_l2_drift_from_initialization": current_q_l2_drift,
            "residual": current_residual_summary,
            "validation": validation_event,
            "update": update,
            "step_ms": milliseconds(step_started),
            "is_best": is_best,
        }))?;
    }

    validate_frozen_blocks(
        config.arm,
        &state.best_q,
        &state.best_residuals,
        &initialization.q,
    )?;
    validate_clock_contract(
        config.arm,
        initialization.completed_updates,
        state.best_step,
        state.best_q_completed_updates,
        state.best_r_completed_updates,
    )?;
    let checkpoint_path = state.save_checkpoint(
        output_dir,
        config,
        &runtime_identity,
        oracle.topology_identity(),
        &initialization.identity,
    )?;
    let restored = restore_turn_checkpoint(
        &checkpoint_path,
        &RestoreContext {
            config,
            graph: &graph,
            expanded: &expanded,
            runtime_identity: &runtime_identity,
            topology_identity: oracle.topology_identity(),
            initialization_identity: &initialization.identity,
            initial_q: &initialization.q,
            initial_q_completed_updates: initialization.completed_updates,
        },
    )?;
    ensure_restored_matches_selected(&restored, &state)?;
    let best_edge = EdgeOnlyModel::from_q(
        &graph.baseline_weights,
        config.quantization_scale,
        &restored.q,
    )?;
    let best_model = TurnAwareModel::from_residuals(
        best_edge,
        &expanded,
        config.residual_scale,
        &restored.residuals,
    )?;
    if best_model.quantized_edge_weights()? != restored.edge_weights
        || best_model.quantized_transition_weights(&expanded)? != restored.transition_weights
    {
        return Err("selected latent state does not reproduce selected integer metrics".into());
    }
    let restore_started = Instant::now();
    metric = oracle.customize(&restored.edge_weights, &restored.transition_weights)?;
    let validation_metrics = evaluate_expanded_paths(&metric, &validation.paths, actual_threads)?;
    logger.log(json!({
        "event": "evaluation",
        "split": "validation_best",
        "metrics": metrics_json(&validation_metrics),
    }))?;
    logger.log(json!({
        "event": "finished",
        "best_step": restored.step,
        "best_epoch": restored.step,
        "arm": config.arm.as_str(),
        "selection_metric": "aggregate_validation_relative_regret",
        "selection_value": restored.selection_value,
        "best_train_mean_regret": restored.train_mean_regret,
        "q_completed_updates": restored.q_completed_updates,
        "r_completed_updates": restored.r_completed_updates,
        "checkpoint_restore_verified": true,
        "restore_full_customization_ms": milliseconds(restore_started),
        "peak_rss_kib": process_peak_rss_kib().unwrap_or(0),
        "checkpoint_path": checkpoint_path,
        "topology_identity": oracle.topology_identity(),
        "test_read": false,
    }))?;

    Ok(TurnTrainingOutcome {
        checkpoint_path,
        log_path: logger.path,
        best_step: restored.step,
        selection_value: restored.selection_value,
        validation_metrics,
    })
}

impl TurnTrainingState {
    fn new() -> Self {
        Self {
            best_selection_value: f64::INFINITY,
            best_train_mean_regret: f64::INFINITY,
            best_step: 0,
            best_q: Vec::new(),
            best_residuals: Vec::new(),
            best_edge_weights: Vec::new(),
            best_transition_weights: Vec::new(),
            best_q_completed_updates: 0,
            best_r_completed_updates: 0,
        }
    }

    fn consider(&mut self, candidate: ConsideredState<'_>) -> bool {
        if candidate.selection_value >= self.best_selection_value {
            return false;
        }
        self.best_selection_value = candidate.selection_value;
        self.best_train_mean_regret = candidate.train_mean_regret;
        self.best_step = candidate.step;
        self.best_q = candidate.q.to_vec();
        self.best_residuals = candidate.residuals.to_vec();
        self.best_edge_weights = candidate.edge_weights.to_vec();
        self.best_transition_weights = candidate.transition_weights.to_vec();
        self.best_q_completed_updates = candidate.q_completed_updates;
        self.best_r_completed_updates = candidate.r_completed_updates;
        true
    }

    fn save_checkpoint(
        &self,
        output_dir: &Path,
        config: &TurnTrainingConfig,
        runtime_identity: &Value,
        topology_identity: &str,
        initialization_identity: &Value,
    ) -> Result<PathBuf, String> {
        if !self.best_selection_value.is_finite() {
            return Err("cannot save a turn-aware checkpoint before validation".into());
        }
        let checkpoint = json!({
            "schema_version": 3,
            "model": "turn_aware",
            "arm": config.arm.as_str(),
            "epoch": self.best_step,
            "local_step": self.best_step,
            "configuration": config.as_json(),
            "selection": {
                "split": "validation",
                "metric": "aggregate_relative_regret",
                "value": self.best_selection_value,
            },
            "train_mean_regret": self.best_train_mean_regret,
            "runtime_identity": runtime_identity,
            "source_initialization": initialization_identity,
            "expanded_topology_identity": topology_identity,
            "residual_scale": config.residual_scale,
            "q_completed_updates": self.best_q_completed_updates,
            "r_completed_updates": self.best_r_completed_updates,
            "q": &self.best_q,
            "r": &self.best_residuals,
            "quantized_edge_weights": &self.best_edge_weights,
            "quantized_transition_weights": &self.best_transition_weights,
        });
        let bytes = serde_json::to_vec(&checkpoint)
            .map_err(|error| format!("failed to serialize turn-aware checkpoint: {error}"))?;
        let path = output_dir.join("checkpoint.json");
        atomic_write(&path, &bytes)?;
        Ok(path)
    }
}

fn restore_turn_checkpoint(
    path: &Path,
    context: &RestoreContext<'_>,
) -> Result<RestoredTurnCheckpoint, String> {
    let checkpoint = load_checkpoint(path)?;
    if required_str(&checkpoint, "/arm")? != context.config.arm.as_str() {
        return Err("turn checkpoint arm does not match the requested run".into());
    }
    if checkpoint.pointer("/configuration") != Some(context.config.as_json()) {
        return Err("turn checkpoint configuration does not match the requested run".into());
    }
    if checkpoint.pointer("/runtime_identity") != Some(context.runtime_identity) {
        return Err("turn checkpoint runtime identity does not match the loaded data".into());
    }
    if checkpoint.pointer("/source_initialization") != Some(context.initialization_identity) {
        return Err("turn checkpoint source initialization identity mismatch".into());
    }
    if required_str(&checkpoint, "/expanded_topology_identity")? != context.topology_identity {
        return Err("turn checkpoint expanded topology identity mismatch".into());
    }
    if required_f64(&checkpoint, "/residual_scale")? != context.config.residual_scale {
        return Err("turn checkpoint residual scale mismatch".into());
    }
    if required_str(&checkpoint, "/selection/split")? != "validation"
        || required_str(&checkpoint, "/selection/metric")? != "aggregate_relative_regret"
    {
        return Err("turn checkpoint has an invalid selection contract".into());
    }

    let step = required_u64(&checkpoint, "/local_step")?;
    if required_u64(&checkpoint, "/epoch")? != step
        || step > context.config.updates
        || (step % context.config.validation_every != 0 && step != context.config.updates)
    {
        return Err("turn checkpoint local step is not a scheduled validation state".into());
    }
    let selection_value = required_f64(&checkpoint, "/selection/value")?;
    let train_mean_regret = required_f64(&checkpoint, "/train_mean_regret")?;
    if !selection_value.is_finite() || !train_mean_regret.is_finite() {
        return Err("turn checkpoint contains non-finite objective values".into());
    }

    let q_completed_updates = required_u64(&checkpoint, "/q_completed_updates")?;
    let r_completed_updates = required_u64(&checkpoint, "/r_completed_updates")?;
    validate_clock_contract(
        context.config.arm,
        context.initial_q_completed_updates,
        step,
        q_completed_updates,
        r_completed_updates,
    )?;

    let q = parse_f64_array(&checkpoint, "/q")?;
    let residuals = parse_f64_array(&checkpoint, "/r")?;
    if q.len() != context.graph.tail.len()
        || q.iter().any(|&value| {
            !value.is_finite() || value < context.config.q_min || value > context.config.q_max
        })
    {
        return Err("turn checkpoint q does not match the configured edge box".into());
    }
    if residuals.len() != context.expanded.transition_count()
        || residuals
            .iter()
            .any(|&value| !value.is_finite() || value < 0.0 || value > context.config.r_max)
    {
        return Err("turn checkpoint residuals do not match the configured turn box".into());
    }
    validate_frozen_blocks(context.config.arm, &q, &residuals, context.initial_q)?;

    let edge_weights = parse_u32_array(&checkpoint, "/quantized_edge_weights")?;
    let transition_weights = parse_u32_array(&checkpoint, "/quantized_transition_weights")?;
    let edge_model = EdgeOnlyModel::from_q(
        &context.graph.baseline_weights,
        context.config.quantization_scale,
        &q,
    )?;
    let model = TurnAwareModel::from_residuals(
        edge_model,
        context.expanded,
        context.config.residual_scale,
        &residuals,
    )?;
    if model.quantized_edge_weights()? != edge_weights
        || model.quantized_transition_weights(context.expanded)? != transition_weights
    {
        return Err("turn checkpoint latent state does not reproduce its integer metric".into());
    }

    Ok(RestoredTurnCheckpoint {
        step,
        selection_value,
        train_mean_regret,
        q,
        residuals,
        edge_weights,
        transition_weights,
        q_completed_updates,
        r_completed_updates,
    })
}

fn ensure_restored_matches_selected(
    restored: &RestoredTurnCheckpoint,
    selected: &TurnTrainingState,
) -> Result<(), String> {
    for (field, restored_value, selected_value) in [
        ("step", restored.step, selected.best_step),
        (
            "q_completed_updates",
            restored.q_completed_updates,
            selected.best_q_completed_updates,
        ),
        (
            "r_completed_updates",
            restored.r_completed_updates,
            selected.best_r_completed_updates,
        ),
    ] {
        if restored_value != selected_value {
            return Err(format!(
                "atomically restored checkpoint field {field} differs: restored={restored_value}, selected={selected_value}"
            ));
        }
    }
    for (field, restored_value, selected_value) in [
        (
            "selection_value",
            restored.selection_value,
            selected.best_selection_value,
        ),
        (
            "train_mean_regret",
            restored.train_mean_regret,
            selected.best_train_mean_regret,
        ),
    ] {
        if restored_value.to_bits() != selected_value.to_bits() {
            return Err(format!(
                "atomically restored checkpoint field {field} differs: restored={restored_value} ({:#018x}), selected={selected_value} ({:#018x})",
                restored_value.to_bits(),
                selected_value.to_bits()
            ));
        }
    }
    if let Some(difference) = first_f64_slice_difference("q", &restored.q, &selected.best_q) {
        return Err(difference);
    }
    if let Some(difference) =
        first_f64_slice_difference("r", &restored.residuals, &selected.best_residuals)
    {
        return Err(difference);
    }
    if let Some(difference) = first_slice_difference(
        "quantized_edge_weights",
        &restored.edge_weights,
        &selected.best_edge_weights,
    ) {
        return Err(difference);
    }
    if let Some(difference) = first_slice_difference(
        "quantized_transition_weights",
        &restored.transition_weights,
        &selected.best_transition_weights,
    ) {
        return Err(difference);
    }
    Ok(())
}

fn first_f64_slice_difference(field: &str, restored: &[f64], selected: &[f64]) -> Option<String> {
    if restored.len() != selected.len() {
        return Some(format!(
            "atomically restored checkpoint field {field} length differs: restored={}, selected={}",
            restored.len(),
            selected.len()
        ));
    }
    restored
        .iter()
        .zip(selected)
        .position(|(&left, &right)| left.to_bits() != right.to_bits())
        .map(|index| {
            format!(
                "atomically restored checkpoint field {field} first differs at index {index}: restored={} ({:#018x}), selected={} ({:#018x})",
                restored[index],
                restored[index].to_bits(),
                selected[index],
                selected[index].to_bits()
            )
        })
}

fn first_slice_difference<T: std::fmt::Debug + PartialEq>(
    field: &str,
    restored: &[T],
    selected: &[T],
) -> Option<String> {
    if restored.len() != selected.len() {
        return Some(format!(
            "atomically restored checkpoint field {field} length differs: restored={}, selected={}",
            restored.len(),
            selected.len()
        ));
    }
    restored
        .iter()
        .zip(selected)
        .position(|(left, right)| left != right)
        .map(|index| {
            format!(
                "atomically restored checkpoint field {field} first differs at index {index}: restored={:?}, selected={:?}",
                restored[index], selected[index]
            )
        })
}

fn load_edge_initialization(
    config: &TurnTrainingConfig,
    graph: &GraphData,
) -> Result<EdgeInitialization, String> {
    let bytes = std::fs::read(&config.initialization_path).map_err(|error| {
        format!(
            "failed to read {}: {error}",
            config.initialization_path.display()
        )
    })?;
    let digest = sha256_bytes(&bytes);
    if digest != config.initialization_sha256.to_ascii_lowercase() {
        return Err(format!(
            "edge initialization SHA-256 mismatch: expected {}, got {digest}",
            config.initialization_sha256
        ));
    }
    let value: Value = serde_json::from_slice(&bytes).map_err(|error| {
        format!(
            "failed to decode {}: {error}",
            config.initialization_path.display()
        )
    })?;
    if value.pointer("/schema").and_then(Value::as_str) != Some("edge_initialization")
        || value.pointer("/schema_version").and_then(Value::as_u64) != Some(1)
        || value.pointer("/model").and_then(Value::as_str) != Some("edge_only")
        || value.pointer("/status").and_then(Value::as_str) != Some("frozen_validated")
    {
        return Err("initialization is not a frozen edge_initialization schema-1 artifact".into());
    }
    let completed_updates = required_u64(&value, "/completed_q_updates")?;
    if completed_updates != 99 {
        return Err(format!(
            "frozen initialization must contain 99 completed q updates, got {completed_updates}"
        ));
    }
    let expected_fingerprint = required_str(&value, "/baseline_identity/fnv1a64")?;
    let actual_fingerprint = baseline_fingerprint(graph);
    if expected_fingerprint != actual_fingerprint
        || required_u64(&value, "/baseline_identity/nodes")? as usize != graph.x.len()
        || required_u64(&value, "/baseline_identity/edges")? as usize != graph.tail.len()
    {
        return Err("edge initialization baseline identity does not match the loaded graph".into());
    }
    for (pointer, expected) in [
        ("/source_optimizer/eta0", config.eta_q0),
        ("/source_optimizer/lambda_edge", config.lambda_edge),
        ("/source_optimizer/q_min", config.q_min),
        ("/source_optimizer/q_max", config.q_max),
        (
            "/source_optimizer/quantization_scale",
            config.quantization_scale,
        ),
    ] {
        let actual = required_f64(&value, pointer)?;
        if actual != expected {
            return Err(format!(
                "initialization {pointer}={actual} does not match config {expected}"
            ));
        }
    }
    let q = parse_f64_array(&value, "/q")?;
    if q.len() != graph.tail.len()
        || q.iter()
            .any(|item| !item.is_finite() || *item < config.q_min || *item > config.q_max)
    {
        return Err(
            "initialization q does not match the finite configured box and graph size".into(),
        );
    }
    let weights = parse_u32_array(&value, "/quantized_metric_weights")?;
    let model = EdgeOnlyModel::from_q(&graph.baseline_weights, config.quantization_scale, &q)?;
    if weights != model.quantized_weights()? {
        return Err("initialization q does not reproduce its integer edge weights".into());
    }
    let identity = json!({
        "artifact_path": config.initialization_path,
        "artifact_sha256": digest,
        "completed_q_updates": completed_updates,
        "source": value.pointer("/source").cloned().unwrap_or(Value::Null),
        "baseline_identity": value.pointer("/baseline_identity").cloned().unwrap_or(Value::Null),
        "selection": value.pointer("/selection").cloned().unwrap_or(Value::Null),
    });
    Ok(EdgeInitialization {
        q,
        weights,
        completed_updates,
        identity,
    })
}

fn verify_declared_data(
    config: &TurnTrainingConfig,
    split: &str,
    variant: &str,
) -> Result<(), String> {
    let pointer = format!("/data/{split}_identity");
    let identity = config
        .as_json()
        .pointer(&pointer)
        .ok_or_else(|| format!("turn config requires {pointer}"))?;
    let expected_path = PathBuf::from(format!(
        "data/{}_data/preprocessed_{split}_trips_{variant}.pkl",
        config.city
    ));
    if required_str(identity, "/path")? != expected_path.to_string_lossy() {
        return Err(format!("{pointer}/path does not match {expected_path:?}"));
    }
    let metadata = std::fs::metadata(&expected_path)
        .map_err(|error| format!("failed to inspect {}: {error}", expected_path.display()))?;
    let expected_bytes = required_u64(identity, "/bytes")?;
    if metadata.len() != expected_bytes {
        return Err(format!(
            "{} byte identity mismatch: expected {expected_bytes}, got {}",
            expected_path.display(),
            metadata.len()
        ));
    }
    let expected_sha = required_str(identity, "/sha256")?;
    let actual_sha = sha256_file(&expected_path)?;
    if actual_sha != expected_sha.to_ascii_lowercase() {
        return Err(format!(
            "{} SHA-256 mismatch: expected {expected_sha}, got {actual_sha}",
            expected_path.display()
        ));
    }
    Ok(())
}

fn verify_declared_sample_count(
    config: &TurnTrainingConfig,
    split: &str,
    report: &PathValidationReport,
) -> Result<(), String> {
    let pointer = format!("/data/{split}_identity/sample_count");
    if let Some(expected) = config.as_json().pointer(&pointer).and_then(Value::as_u64)
        && expected as usize != report.available_samples
    {
        return Err(format!(
            "{pointer}={expected} but loader found {} records",
            report.available_samples
        ));
    }
    Ok(())
}

fn runtime_identity(
    config: &TurnTrainingConfig,
    graph: &GraphData,
    expanded: &ExpandedTurnGraph,
    train: &PathValidationReport,
    validation: &PathValidationReport,
    initialization: &Value,
) -> Value {
    json!({
        "baseline": {
            "city": config.city,
            "nodes": graph.x.len(),
            "edges": graph.tail.len(),
            "fnv1a64": baseline_fingerprint(graph),
        },
        "expanded": {
            "states": expanded.stats.expanded_nodes,
            "transitions": expanded.transition_count(),
            "state_self_transitions": expanded.stats.state_self_transitions,
        },
        "train": {
            "variant": config.train_variant,
            "declared": config.as_json().pointer("/data/train_identity").cloned().unwrap_or(Value::Null),
            "available": train.available_samples,
            "inspected": train.inspected_samples,
            "accepted": train.accepted_samples,
        },
        "validation": {
            "variant": config.validation_variant,
            "declared": config.as_json().pointer("/data/validation_identity").cloned().unwrap_or(Value::Null),
            "available": validation.available_samples,
            "inspected": validation.inspected_samples,
            "accepted": validation.accepted_samples,
        },
        "initialization": initialization,
    })
}

fn residual_summary_json(
    model: &TurnAwareModel,
    expanded: &ExpandedTurnGraph,
    edge_weights: &[u32],
    transition_weights: &[u32],
    r_max: f64,
) -> Value {
    let summary = model.residual_summary();
    let at_max = model
        .transition_residuals()
        .iter()
        .filter(|&&value| value >= r_max - 1e-12)
        .count();
    let quantized_nonzero = expanded
        .transitions()
        .filter(|(transition, _, next)| {
            transition_weights[transition.index()] > edge_weights[*next]
        })
        .count();
    json!({
        "transitions": summary.transitions,
        "zero": summary.zero_transitions,
        "positive": summary.positive_transitions,
        "at_max": at_max,
        "mean": summary.mean,
        "max": summary.max,
        "l2_norm": summary.l2_norm,
        "quantized_nonzero": quantized_nonzero,
    })
}

fn validate_frozen_blocks(
    arm: TurnExperimentArm,
    q: &[f64],
    residuals: &[f64],
    initial_q: &[f64],
) -> Result<(), String> {
    if arm == TurnExperimentArm::ExpandedEdgeContinuation
        && residuals.iter().any(|&value| value != 0.0)
    {
        return Err("expanded-edge control changed a transition residual".into());
    }
    if arm == TurnExperimentArm::TurnOnly && q != initial_q {
        return Err("turn-only arm changed the frozen q state".into());
    }
    Ok(())
}

fn validate_clock_contract(
    arm: TurnExperimentArm,
    initial_q_completed_updates: u64,
    local_step: u64,
    q_completed_updates: u64,
    r_completed_updates: u64,
) -> Result<(), String> {
    let expected_q = initial_q_completed_updates
        .checked_add(if arm.updates_q() { local_step } else { 0 })
        .ok_or_else(|| "q update clock overflow".to_string())?;
    let expected_r = if arm.updates_residuals() {
        local_step
    } else {
        0
    };
    if q_completed_updates != expected_q || r_completed_updates != expected_r {
        return Err(format!(
            "{} clock mismatch at local step {local_step}: expected q/r={expected_q}/{expected_r}, got {q_completed_updates}/{r_completed_updates}",
            arm.as_str()
        ));
    }
    Ok(())
}

fn median_u32(values: &[u32]) -> Result<f64, String> {
    if values.is_empty() {
        return Err("cannot compute residual scale from empty weights".into());
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        Ok((sorted[middle - 1] as f64 + sorted[middle] as f64) / 2.0)
    } else {
        Ok(sorted[middle] as f64)
    }
}

fn changed_count<T: PartialEq>(left: &[T], right: &[T]) -> usize {
    left.iter().zip(right).filter(|(a, b)| a != b).count()
}

fn l2_distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file =
        File::open(path).map_err(|error| format!("failed to open {}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("failed to hash {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn required_str<'a>(value: &'a Value, pointer: &str) -> Result<&'a str, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string {pointer}"))
}

fn required_u64(value: &Value, pointer: &str) -> Result<u64, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("missing nonnegative integer {pointer}"))
}

fn required_f64(value: &Value, pointer: &str) -> Result<f64, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("missing number {pointer}"))
}

fn parse_f64_array(value: &Value, pointer: &str) -> Result<Vec<f64>, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing array {pointer}"))?
        .iter()
        .enumerate()
        .map(|(index, item)| {
            item.as_f64()
                .ok_or_else(|| format!("{pointer}/{index} is not a number"))
        })
        .collect()
}

fn parse_u32_array(value: &Value, pointer: &str) -> Result<Vec<u32>, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing array {pointer}"))?
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let raw = item
                .as_u64()
                .ok_or_else(|| format!("{pointer}/{index} is not an unsigned integer"))?;
            u32::try_from(raw).map_err(|_| format!("{pointer}/{index} does not fit u32"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_uses_the_mean_of_the_two_middle_integer_weights() {
        assert_eq!(median_u32(&[9, 1, 7, 3]).unwrap(), 5.0);
        assert_eq!(median_u32(&[9, 1, 3]).unwrap(), 3.0);
        assert!(median_u32(&[]).is_err());
    }

    #[test]
    fn arm_contract_protects_frozen_blocks() {
        assert!(
            validate_frozen_blocks(
                TurnExperimentArm::ExpandedEdgeContinuation,
                &[0.9],
                &[0.0],
                &[1.0]
            )
            .is_ok()
        );
        assert!(
            validate_frozen_blocks(
                TurnExperimentArm::ExpandedEdgeContinuation,
                &[0.9],
                &[0.1],
                &[1.0]
            )
            .is_err()
        );
        assert!(
            validate_frozen_blocks(TurnExperimentArm::TurnOnly, &[0.9], &[0.1], &[1.0]).is_err()
        );
        assert!(
            validate_frozen_blocks(TurnExperimentArm::JointEdgeTurn, &[0.9], &[0.1], &[1.0])
                .is_ok()
        );
    }

    #[test]
    fn arm_clocks_are_independent_and_match_pre_update_steps() {
        assert!(
            validate_clock_contract(TurnExperimentArm::ExpandedEdgeContinuation, 99, 10, 109, 0)
                .is_ok()
        );
        assert!(validate_clock_contract(TurnExperimentArm::TurnOnly, 99, 10, 99, 10).is_ok());
        assert!(validate_clock_contract(TurnExperimentArm::JointEdgeTurn, 99, 10, 109, 10).is_ok());
        assert!(
            validate_clock_contract(TurnExperimentArm::JointEdgeTurn, 99, 10, 110, 11).is_err()
        );
    }

    #[test]
    fn checkpoint_json_round_trips_latent_f64_bits_exactly() {
        // These are representative latent values from the first screening
        // checkpoint. Serde JSON's legacy fast float parser shifts the first
        // value by one ULP unless the `float_roundtrip` feature is enabled.
        let original: Vec<f64> = vec![
            0.9702229782477475,
            0.06348409082193338,
            0.00025297242600556537,
        ];
        let encoded = serde_json::to_vec(&original).unwrap();
        let restored: Vec<f64> = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(
            restored
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            original
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
}
