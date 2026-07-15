use crate::config::{ExpandedTrainingConfig, atomic_write, load_checkpoint};
use crate::data::{
    GraphData, PathValidationReport, compute_observed_edge_counts,
    compute_observed_transition_counts, group_paths_by_od,
};
use crate::evaluation::{PathMetrics, evaluate_expanded_paths};
use crate::model::{EdgeOnlyModel, ExpandedMetricWeights, ExpandedRoadModel};
use crate::objective::{compute_expanded_regret, count_residual_l1};
use crate::optimizer::ExpandedProjectedSubgradientOptimizer;
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
pub struct ExpandedTrainingOutcome {
    pub checkpoint_path: PathBuf,
    pub log_path: PathBuf,
    pub best_update: u64,
    pub selection_value: f64,
    pub validation_metrics: PathMetrics,
}

struct ExpandedTrainingState {
    best_selection_value: f64,
    best_train_mean_regret: f64,
    best_completed_updates: u64,
    best_q: Vec<f64>,
    best_residuals: Vec<f64>,
    best_edge_weights: Vec<u32>,
    best_transition_weights: Vec<u32>,
}

struct ConsideredState<'a> {
    completed_updates: u64,
    selection_value: f64,
    train_mean_regret: f64,
    q: &'a [f64],
    residuals: &'a [f64],
    edge_weights: &'a [u32],
    transition_weights: &'a [u32],
}

struct RestoredExpandedCheckpoint {
    completed_updates: u64,
    selection_value: f64,
    train_mean_regret: f64,
    q: Vec<f64>,
    residuals: Vec<f64>,
    edge_weights: Vec<u32>,
    transition_weights: Vec<u32>,
}

struct RestoreContext<'a> {
    config: &'a ExpandedTrainingConfig,
    graph: &'a GraphData,
    expanded: &'a ExpandedTurnGraph,
    runtime_identity: &'a Value,
    topology_identity: &'a str,
}

/// Train the fully optimized expanded road model from `q=1, r=0`.
///
/// Each update performs one expanded training batch query. Edge and transition
/// subgradients come from that same pre-update metric, then one optimizer call
/// updates both continuous blocks and advances one global clock.
pub fn run_expanded_training(
    config: &ExpandedTrainingConfig,
    output_dir: &Path,
) -> Result<ExpandedTrainingOutcome, String> {
    let actual_threads = rayon::current_num_threads().max(1);
    if actual_threads != config.rayon_threads {
        return Err(format!(
            "configuration requires {} Rayon threads, process has {actual_threads}; set RAYON_NUM_THREADS before launch",
            config.rayon_threads
        ));
    }

    let mut logger = JsonlLogger::new(output_dir)?;
    logger.log(json!({
        "event": "configuration",
        "run_id": config.run_id,
        "model": "expanded",
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
        return Err("expanded train and validation must both contain valid paths".to_string());
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

    let expanded = ExpandedTurnGraph::build(&graph)?;
    let runtime_identity =
        runtime_identity(config, &graph, &expanded, &train.report, &validation.report);
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
    let edge_model = EdgeOnlyModel::new(&graph.baseline_weights, config.quantization_scale)?;
    let mut model = ExpandedRoadModel::new(edge_model, &expanded, config.residual_scale)?;
    let initial_metric = model.metric(&expanded)?;
    let customization_started = Instant::now();
    let mut metric = oracle.customize(
        initial_metric.edge_weights(),
        initial_metric.transition_weights(),
    )?;
    logger.log(json!({
        "event": "oracle_built",
        "kind": "expanded_cch",
        "customization": "full",
        "build_ms": milliseconds(build_started),
        "initial_customization_ms": milliseconds(customization_started),
        "topology_identity": oracle.topology_identity(),
        "threads": actual_threads,
    }))?;

    let mut optimizer = ExpandedProjectedSubgradientOptimizer::new(
        config.eta0,
        config.lambda_edge,
        config.lambda_transition,
        config.q_min,
        config.q_max,
        config.r_max,
    )?;
    let mut state = ExpandedTrainingState::new();

    for completed_updates in 0..=config.updates {
        validate_clock(completed_updates, optimizer.completed_updates())?;
        let update_started = Instant::now();

        // This is the sole expanded shortest-path training batch for this
        // pre-update state. Both parameter blocks consume these predictions.
        let train_oracle = metric.batch_stats(&train_groups, actual_threads)?;
        let train_regret = compute_expanded_regret(
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
        let edge_regularization = model.edge_regularization(config.lambda_edge);
        let transition_regularization = model.transition_regularization(config.lambda_transition);
        let train_objective =
            train_regret.mean_data_loss + edge_regularization + transition_regularization;

        let current_edge_weights = metric.edge_weights().to_vec();
        let current_transition_weights = metric.transition_weights().to_vec();
        let current_q_summary = q_summary(model.q(), config.q_min, config.q_max);
        let current_q_l2_drift = l2_distance_from_one(model.q());
        let current_residual_summary = residual_summary_json(
            &model,
            &expanded,
            &current_edge_weights,
            &current_transition_weights,
            config.r_max,
        );

        let should_validate =
            completed_updates % config.validation_every == 0 || completed_updates == config.updates;
        let mut validation_event = Value::Null;
        let mut is_best = false;
        if should_validate {
            let validation_oracle = metric.batch_stats(&validation_groups, actual_threads)?;
            let validation_regret = compute_expanded_regret(
                &expanded,
                metric.edge_weights(),
                metric.transition_weights(),
                &validation_observed_edges,
                &validation_observed_transitions,
                &validation_oracle,
            )?;
            let validation_objective =
                validation_regret.mean_data_loss + edge_regularization + transition_regularization;
            is_best = state.consider(ConsideredState {
                completed_updates,
                selection_value: validation_objective,
                train_mean_regret: train_regret.mean_data_loss,
                q: model.q(),
                residuals: model.transition_residuals(),
                edge_weights: metric.edge_weights(),
                transition_weights: metric.transition_weights(),
            });
            if is_best {
                state.save_checkpoint(
                    output_dir,
                    config,
                    &runtime_identity,
                    oracle.topology_identity(),
                )?;
            }
            validation_event = json!({
                "mean_regret": validation_regret.mean_data_loss,
                "relative_regret": validation_regret.relative_data_loss,
                "objective": validation_objective,
                "selection_value": validation_objective,
                "queries": validation_oracle.num_queries,
                "oracle_ms": validation_oracle.oracle_duration.as_secs_f64() * 1_000.0,
                "is_best": is_best,
            });
        }

        let mut update = json!({
            "status": "final_skipped",
            "completed_updates_before": completed_updates,
            "completed_updates_after": completed_updates,
        });
        if completed_updates < config.updates {
            let optimizer_started = Instant::now();
            let step = model.projected_step(
                &mut optimizer,
                &train_observed_edges,
                &train_observed_transitions,
                &train_oracle,
            )?;
            let next_completed_updates = completed_updates
                .checked_add(1)
                .ok_or_else(|| "expanded update clock overflow".to_string())?;
            validate_clock(next_completed_updates, optimizer.completed_updates())?;

            // Rebuild both integer metric components only after the joint
            // continuous update has finished.
            let next_metric = model.metric(&expanded)?;
            let changed_edges = changed_count(&current_edge_weights, next_metric.edge_weights());
            let changed_transitions = changed_count(
                &current_transition_weights,
                next_metric.transition_weights(),
            );
            let customization_started = Instant::now();
            let customization_ms = if changed_edges == 0 && changed_transitions == 0 {
                0.0
            } else {
                metric = oracle
                    .customize(next_metric.edge_weights(), next_metric.transition_weights())?;
                milliseconds(customization_started)
            };
            update = json!({
                "status": if changed_edges == 0 && changed_transitions == 0 {
                    "latent_only_no_integer_change"
                } else {
                    "applied"
                },
                "eta_cost": step.eta,
                "max_abs_q_delta": step.max_abs_q_delta,
                "max_abs_r_delta": step.max_abs_r_delta,
                "max_abs_edge_cost_delta": step.max_abs_edge_cost_delta,
                "max_abs_transition_cost_delta": step.max_abs_transition_cost_delta,
                "projected_edges": step.projected_edges,
                "projected_transitions": step.projected_transitions,
                "changed_edges": changed_edges,
                "changed_transitions": changed_transitions,
                "completed_updates_before": completed_updates,
                "completed_updates_after": next_completed_updates,
                "optimizer_ms": milliseconds(optimizer_started),
                "customization_ms": customization_ms,
            });
        }

        logger.log(json!({
            "event": "update_state",
            "completed_updates": completed_updates,
            "train_mean_regret": train_regret.mean_data_loss,
            "train_relative_regret": train_regret.relative_data_loss,
            "edge_regularization": edge_regularization,
            "transition_regularization": transition_regularization,
            "train_objective": train_objective,
            "edge_count_residual_l1_diagnostic": edge_count_residual,
            "transition_count_residual_l1_diagnostic": transition_count_residual,
            "train_queries": train_oracle.num_queries,
            "train_oracle_ms": train_oracle.oracle_duration.as_secs_f64() * 1_000.0,
            "q": current_q_summary,
            "q_l2_drift_from_one": current_q_l2_drift,
            "residual": current_residual_summary,
            "validation": validation_event,
            "update": update,
            "state_ms": milliseconds(update_started),
            "is_best": is_best,
        }))?;
    }

    let checkpoint_path = state.save_checkpoint(
        output_dir,
        config,
        &runtime_identity,
        oracle.topology_identity(),
    )?;
    let restored = restore_expanded_checkpoint(
        &checkpoint_path,
        &RestoreContext {
            config,
            graph: &graph,
            expanded: &expanded,
            runtime_identity: &runtime_identity,
            topology_identity: oracle.topology_identity(),
        },
    )?;
    ensure_restored_matches_selected(&restored, &state)?;
    let best_edge = EdgeOnlyModel::from_q(
        &graph.baseline_weights,
        config.quantization_scale,
        &restored.q,
    )?;
    let best_model = ExpandedRoadModel::from_parameters(
        best_edge,
        &expanded,
        config.residual_scale,
        &restored.residuals,
    )?;
    let restored_metric = best_model.metric(&expanded)?;
    if restored_metric.edge_weights != restored.edge_weights
        || restored_metric.transition_weights != restored.transition_weights
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
        "best_update": restored.completed_updates,
        "selection_metric": "validation_mean_regret_plus_regularization",
        "selection_value": restored.selection_value,
        "best_train_mean_regret": restored.train_mean_regret,
        "completed_updates": restored.completed_updates,
        "checkpoint_restore_verified": true,
        "restore_full_customization_ms": milliseconds(restore_started),
        "peak_rss_kib": process_peak_rss_kib().unwrap_or(0),
        "checkpoint_path": checkpoint_path,
        "topology_identity": oracle.topology_identity(),
        "test_read": false,
    }))?;

    Ok(ExpandedTrainingOutcome {
        checkpoint_path,
        log_path: logger.path,
        best_update: restored.completed_updates,
        selection_value: restored.selection_value,
        validation_metrics,
    })
}

impl ExpandedTrainingState {
    fn new() -> Self {
        Self {
            best_selection_value: f64::INFINITY,
            best_train_mean_regret: f64::INFINITY,
            best_completed_updates: 0,
            best_q: Vec::new(),
            best_residuals: Vec::new(),
            best_edge_weights: Vec::new(),
            best_transition_weights: Vec::new(),
        }
    }

    fn consider(&mut self, candidate: ConsideredState<'_>) -> bool {
        if candidate.selection_value >= self.best_selection_value {
            return false;
        }
        self.best_selection_value = candidate.selection_value;
        self.best_train_mean_regret = candidate.train_mean_regret;
        self.best_completed_updates = candidate.completed_updates;
        self.best_q = candidate.q.to_vec();
        self.best_residuals = candidate.residuals.to_vec();
        self.best_edge_weights = candidate.edge_weights.to_vec();
        self.best_transition_weights = candidate.transition_weights.to_vec();
        true
    }

    fn save_checkpoint(
        &self,
        output_dir: &Path,
        config: &ExpandedTrainingConfig,
        runtime_identity: &Value,
        topology_identity: &str,
    ) -> Result<PathBuf, String> {
        if !self.best_selection_value.is_finite() {
            return Err("cannot save an expanded checkpoint before validation".into());
        }
        let checkpoint = json!({
            "schema_version": 4,
            "model": "expanded",
            "configuration": config.as_json(),
            "initialization_identity": initialization_identity(),
            "selection": {
                "split": "validation",
                "metric": "mean_regret_plus_regularization",
                "value": self.best_selection_value,
            },
            "train_mean_regret": self.best_train_mean_regret,
            "runtime_identity": runtime_identity,
            "expanded_topology_identity": topology_identity,
            "residual_scale": config.residual_scale,
            "completed_updates": self.best_completed_updates,
            "q": &self.best_q,
            "r": &self.best_residuals,
            "quantized_edge_weights": &self.best_edge_weights,
            "quantized_transition_weights": &self.best_transition_weights,
        });
        let bytes = serde_json::to_vec(&checkpoint)
            .map_err(|error| format!("failed to serialize expanded checkpoint: {error}"))?;
        let path = output_dir.join("checkpoint.json");
        atomic_write(&path, &bytes)?;
        Ok(path)
    }
}

fn restore_expanded_checkpoint(
    path: &Path,
    context: &RestoreContext<'_>,
) -> Result<RestoredExpandedCheckpoint, String> {
    let checkpoint = load_checkpoint(path)?;
    restore_expanded_checkpoint_value(&checkpoint, context)
}

fn restore_expanded_checkpoint_value(
    checkpoint: &Value,
    context: &RestoreContext<'_>,
) -> Result<RestoredExpandedCheckpoint, String> {
    if checkpoint.pointer("/configuration") != Some(context.config.as_json()) {
        return Err("expanded checkpoint configuration does not match the requested run".into());
    }
    if checkpoint.pointer("/runtime_identity") != Some(context.runtime_identity) {
        return Err("expanded checkpoint runtime identity does not match the loaded data".into());
    }
    if checkpoint.pointer("/initialization_identity") != Some(&initialization_identity()) {
        return Err("expanded checkpoint initialization identity mismatch".into());
    }
    if required_str(checkpoint, "/expanded_topology_identity")? != context.topology_identity {
        return Err("expanded checkpoint topology identity mismatch".into());
    }
    if required_f64(checkpoint, "/residual_scale")? != context.config.residual_scale {
        return Err("expanded checkpoint residual scale mismatch".into());
    }
    if required_str(checkpoint, "/selection/split")? != "validation"
        || required_str(checkpoint, "/selection/metric")? != "mean_regret_plus_regularization"
    {
        return Err("expanded checkpoint has an invalid selection contract".into());
    }

    let completed_updates = required_u64(checkpoint, "/completed_updates")?;
    if completed_updates > context.config.updates
        || (completed_updates % context.config.validation_every != 0
            && completed_updates != context.config.updates)
    {
        return Err("expanded checkpoint clock is not a scheduled validation state".into());
    }
    let restored_optimizer = ExpandedProjectedSubgradientOptimizer::with_completed_updates(
        context.config.eta0,
        context.config.lambda_edge,
        context.config.lambda_transition,
        context.config.q_min,
        context.config.q_max,
        context.config.r_max,
        completed_updates,
    )?;
    validate_clock(completed_updates, restored_optimizer.completed_updates())?;

    let selection_value = required_f64(checkpoint, "/selection/value")?;
    let train_mean_regret = required_f64(checkpoint, "/train_mean_regret")?;
    if !selection_value.is_finite() || !train_mean_regret.is_finite() {
        return Err("expanded checkpoint contains non-finite objective values".into());
    }

    let q = parse_f64_array(checkpoint, "/q")?;
    let residuals = parse_f64_array(checkpoint, "/r")?;
    if q.len() != context.graph.tail.len()
        || q.iter().any(|&value| {
            !value.is_finite() || value < context.config.q_min || value > context.config.q_max
        })
    {
        return Err("expanded checkpoint q does not match the configured edge box".into());
    }
    if residuals.len() != context.expanded.transition_count()
        || residuals
            .iter()
            .any(|&value| !value.is_finite() || value < 0.0 || value > context.config.r_max)
    {
        return Err(
            "expanded checkpoint residuals do not match the configured transition box".into(),
        );
    }

    let edge_weights = parse_u32_array(checkpoint, "/quantized_edge_weights")?;
    let transition_weights = parse_u32_array(checkpoint, "/quantized_transition_weights")?;
    let edge_model = EdgeOnlyModel::from_q(
        &context.graph.baseline_weights,
        context.config.quantization_scale,
        &q,
    )?;
    let model = ExpandedRoadModel::from_parameters(
        edge_model,
        context.expanded,
        context.config.residual_scale,
        &residuals,
    )?;
    let reconstructed_metric = model.metric(context.expanded)?;
    if reconstructed_metric.edge_weights != edge_weights
        || reconstructed_metric.transition_weights != transition_weights
    {
        return Err(
            "expanded checkpoint latent state does not reproduce its integer metric".into(),
        );
    }

    Ok(RestoredExpandedCheckpoint {
        completed_updates,
        selection_value,
        train_mean_regret,
        q,
        residuals,
        edge_weights,
        transition_weights,
    })
}

/// Strictly reconstruct the integer metric carried by a schema-4 expanded
/// checkpoint before an external consumer evaluates it.
///
/// This performs the same latent-state, configuration, initialization,
/// topology, selection, and clock checks as training restore. It additionally
/// checks that the checkpoint's self-contained runtime data identity matches
/// the loaded graph and its own declared data identities.
pub fn restore_expanded_metric(
    checkpoint: &Value,
    graph: &GraphData,
    expanded: &ExpandedTurnGraph,
    topology_identity: &str,
) -> Result<ExpandedMetricWeights, String> {
    if checkpoint
        .pointer("/schema_version")
        .and_then(Value::as_u64)
        != Some(4)
        || checkpoint.pointer("/model").and_then(Value::as_str) != Some("expanded")
    {
        return Err("checkpoint is not an expanded schema-4 checkpoint".into());
    }
    let configuration = checkpoint
        .pointer("/configuration")
        .cloned()
        .ok_or_else(|| "expanded checkpoint is missing configuration".to_string())?;
    let config = ExpandedTrainingConfig::from_value(configuration)
        .map_err(|error| format!("invalid expanded checkpoint configuration: {error}"))?;
    let runtime_identity = checkpoint
        .pointer("/runtime_identity")
        .ok_or_else(|| "expanded checkpoint is missing runtime identity".to_string())?;
    validate_self_contained_runtime_identity(&config, graph, expanded, runtime_identity)?;
    let restored = restore_expanded_checkpoint_value(
        checkpoint,
        &RestoreContext {
            config: &config,
            graph,
            expanded,
            runtime_identity,
            topology_identity,
        },
    )?;
    Ok(ExpandedMetricWeights {
        edge_weights: restored.edge_weights,
        transition_weights: restored.transition_weights,
    })
}

fn validate_self_contained_runtime_identity(
    config: &ExpandedTrainingConfig,
    graph: &GraphData,
    expanded: &ExpandedTurnGraph,
    identity: &Value,
) -> Result<(), String> {
    if required_str(identity, "/baseline/city")? != config.city
        || required_u64(identity, "/baseline/nodes")? as usize != graph.x.len()
        || required_u64(identity, "/baseline/edges")? as usize != graph.tail.len()
        || required_str(identity, "/baseline/fnv1a64")? != baseline_fingerprint(graph)
    {
        return Err("expanded checkpoint baseline/data identity mismatch".into());
    }
    if required_u64(identity, "/expanded/states")? as usize != expanded.stats.expanded_nodes
        || required_u64(identity, "/expanded/transitions")? as usize != expanded.transition_count()
        || required_u64(identity, "/expanded/state_self_transitions")? as usize
            != expanded.stats.state_self_transitions
    {
        return Err("expanded checkpoint runtime topology shape mismatch".into());
    }
    if required_str(identity, "/train/variant")? != config.train_variant
        || required_str(identity, "/validation/variant")? != config.validation_variant
        || identity.pointer("/train/declared") != config.as_json().pointer("/data/train_identity")
        || identity.pointer("/validation/declared")
            != config.as_json().pointer("/data/validation_identity")
    {
        return Err("expanded checkpoint declared data identity mismatch".into());
    }
    for split in ["train", "validation"] {
        for field in ["available", "inspected", "accepted"] {
            required_u64(identity, &format!("/{split}/{field}"))?;
        }
    }
    if identity.pointer("/initialization") != Some(&initialization_identity()) {
        return Err("expanded checkpoint runtime initialization identity mismatch".into());
    }
    Ok(())
}

fn ensure_restored_matches_selected(
    restored: &RestoredExpandedCheckpoint,
    selected: &ExpandedTrainingState,
) -> Result<(), String> {
    if restored.completed_updates != selected.best_completed_updates {
        return Err(format!(
            "restored completed_updates differs: restored={}, selected={}",
            restored.completed_updates, selected.best_completed_updates
        ));
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
                "restored checkpoint field {field} differs: restored={restored_value}, selected={selected_value}"
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
            "restored checkpoint field {field} length differs: restored={}, selected={}",
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
                "restored checkpoint field {field} first differs at index {index}: restored={}, selected={}",
                restored[index], selected[index]
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
            "restored checkpoint field {field} length differs: restored={}, selected={}",
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
                "restored checkpoint field {field} first differs at index {index}: restored={:?}, selected={:?}",
                restored[index], selected[index]
            )
        })
}

fn verify_declared_data(
    config: &ExpandedTrainingConfig,
    split: &str,
    variant: &str,
) -> Result<(), String> {
    let pointer = format!("/data/{split}_identity");
    let identity = config
        .as_json()
        .pointer(&pointer)
        .ok_or_else(|| format!("expanded config requires {pointer}"))?;
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
    config: &ExpandedTrainingConfig,
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

fn initialization_identity() -> Value {
    json!({
        "kind": "deterministic",
        "q": "all_one",
        "r": "all_zero",
    })
}

fn runtime_identity(
    config: &ExpandedTrainingConfig,
    graph: &GraphData,
    expanded: &ExpandedTurnGraph,
    train: &PathValidationReport,
    validation: &PathValidationReport,
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
        "initialization": initialization_identity(),
    })
}

fn residual_summary_json(
    model: &ExpandedRoadModel,
    expanded: &ExpandedTurnGraph,
    edge_weights: &[u32],
    transition_weights: &[u32],
    r_max: f64,
) -> Value {
    let summary = model.transition_summary();
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

fn validate_clock(expected: u64, actual: u64) -> Result<(), String> {
    if expected == actual {
        Ok(())
    } else {
        Err(format!(
            "expanded optimizer clock mismatch: expected {expected}, got {actual}"
        ))
    }
}

fn changed_count<T: PartialEq>(left: &[T], right: &[T]) -> usize {
    left.iter().zip(right).filter(|(a, b)| a != b).count()
}

fn l2_distance_from_one(values: &[f64]) -> f64 {
    values
        .iter()
        .map(|&value| (value - 1.0).powi(2))
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

    fn graph() -> GraphData {
        GraphData {
            tail: vec![0, 1, 0, 2],
            head: vec![1, 3, 2, 3],
            baseline_weights: vec![5, 5, 2, 2],
            x: vec![0.0, 1.0, 1.0, 2.0],
            y: vec![0.0, 0.0, 1.0, 0.0],
        }
    }

    fn config_json() -> Value {
        json!({
            "schema_version": 1,
            "run_id": "expanded_checkpoint_fixture",
            "data": {
                "city": "fixture",
                "train_variant": "train",
                "validation_variant": "validation",
                "path_contract": "complete_original_edge_id_sequence",
                "cycle_policy": "drop",
                "train_identity": {
                    "path": "data/fixture_data/preprocessed_train_trips_train.pkl",
                    "bytes": 1,
                    "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                },
                "validation_identity": {
                    "path": "data/fixture_data/preprocessed_validation_trips_validation.pkl",
                    "bytes": 2,
                    "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                }
            },
            "model": {
                "kind": "expanded",
                "eta0": 2.0,
                "lambda_edge": 3.0,
                "lambda_transition": 4.0,
                "q_min": 0.1,
                "q_max": 10.0,
                "quantization_scale": 1.0,
                "residual_scale": 4.0,
                "r_max": 10.0
            },
            "oracle": {
                "kind": "expanded_cch",
                "customization": "full",
                "group_unique_od": true
            },
            "training": {"updates": 10, "validation_every": 5},
            "runtime": {"rayon_threads": 1},
            "selection": {"split": "validation", "metric": "mean_regret_plus_regularization"},
            "test_policy": "never_read"
        })
    }

    fn load_fixture_config(root: &Path) -> ExpandedTrainingConfig {
        let path = root.join("config.json");
        std::fs::write(&path, serde_json::to_vec(&config_json()).unwrap()).unwrap();
        ExpandedTrainingConfig::load(&path).unwrap()
    }

    #[test]
    fn checkpoint_round_trip_restores_all_expanded_state_exactly() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "expanded-checkpoint-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let config = load_fixture_config(&root);
        let graph = graph();
        let expanded = ExpandedTurnGraph::build(&graph).unwrap();
        let oracle = ExpandedCchOracle::build(&graph, &expanded).unwrap();
        let q: Vec<f64> = vec![0.8, 1.2, 1.0, 1.0];
        let residuals: Vec<f64> = vec![0.25, 0.0];
        let model = ExpandedRoadModel::from_parameters(
            EdgeOnlyModel::from_q(&graph.baseline_weights, 1.0, &q).unwrap(),
            &expanded,
            config.residual_scale,
            &residuals,
        )
        .unwrap();
        let edge_weights = model.quantized_edge_weights().unwrap();
        let transition_weights = model.quantized_transition_weights(&expanded).unwrap();
        let runtime_identity = json!({
            "baseline": {
                "city": config.city,
                "nodes": graph.x.len(),
                "edges": graph.tail.len(),
                "fnv1a64": baseline_fingerprint(&graph),
            },
            "expanded": {
                "states": expanded.stats.expanded_nodes,
                "transitions": expanded.transition_count(),
                "state_self_transitions": expanded.stats.state_self_transitions,
            },
            "train": {
                "variant": config.train_variant,
                "declared": config.as_json().pointer("/data/train_identity").unwrap(),
                "available": 10,
                "inspected": 10,
                "accepted": 9,
            },
            "validation": {
                "variant": config.validation_variant,
                "declared": config.as_json().pointer("/data/validation_identity").unwrap(),
                "available": 8,
                "inspected": 8,
                "accepted": 7,
            },
            "initialization": initialization_identity(),
        });
        let mut state = ExpandedTrainingState::new();
        assert!(state.consider(ConsideredState {
            completed_updates: 5,
            selection_value: 0.25,
            train_mean_regret: 4.0,
            q: &q,
            residuals: &residuals,
            edge_weights: &edge_weights,
            transition_weights: &transition_weights,
        }));
        let path = state
            .save_checkpoint(
                &root,
                &config,
                &runtime_identity,
                oracle.topology_identity(),
            )
            .unwrap();
        let context = RestoreContext {
            config: &config,
            graph: &graph,
            expanded: &expanded,
            runtime_identity: &runtime_identity,
            topology_identity: oracle.topology_identity(),
        };
        let restored = restore_expanded_checkpoint(&path, &context).unwrap();
        ensure_restored_matches_selected(&restored, &state).unwrap();
        assert_eq!(restored.completed_updates, 5);
        assert_eq!(
            restored
                .q
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            q.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            restored
                .residuals
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            residuals
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(restored.edge_weights, edge_weights);
        assert_eq!(restored.transition_weights, transition_weights);

        let original_checkpoint = load_checkpoint(&path).unwrap();
        let externally_restored = restore_expanded_metric(
            &original_checkpoint,
            &graph,
            &expanded,
            oracle.topology_identity(),
        )
        .unwrap();
        assert_eq!(externally_restored.edge_weights, edge_weights);
        assert_eq!(externally_restored.transition_weights, transition_weights);
        assert_eq!(
            original_checkpoint["initialization_identity"],
            initialization_identity()
        );
        assert_eq!(
            original_checkpoint["expanded_topology_identity"],
            oracle.topology_identity()
        );
        assert_eq!(original_checkpoint["selection"]["value"], 0.25);
        let reject = |candidate: Value| {
            atomic_write(&path, &serde_json::to_vec(&candidate).unwrap()).unwrap();
            assert!(restore_expanded_checkpoint(&path, &context).is_err());
        };

        let mut corrupted = original_checkpoint.clone();
        corrupted["q"][0] = json!(0.0);
        reject(corrupted);
        let mut corrupted = original_checkpoint.clone();
        corrupted["runtime_identity"]["train"]["variant"] = json!("different-data");
        reject(corrupted);
        let mut corrupted = original_checkpoint.clone();
        corrupted["expanded_topology_identity"] = json!("different-topology");
        reject(corrupted);
        let mut corrupted = original_checkpoint.clone();
        corrupted["completed_updates"] = json!(6);
        reject(corrupted);
        let mut corrupted = original_checkpoint;
        corrupted["quantized_edge_weights"][0] = json!(99);
        reject(corrupted);

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn clock_requires_one_global_update_count() {
        assert!(validate_clock(9, 9).is_ok());
        assert!(validate_clock(9, 8).is_err());
    }

    #[test]
    fn checkpoint_json_round_trips_latent_f64_bits_exactly() {
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
