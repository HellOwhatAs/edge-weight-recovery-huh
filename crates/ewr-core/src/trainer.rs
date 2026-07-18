use crate::line_graph::{LineGraph, LineGraphError, QueryGroup};
use crate::model::{
    FitDiagnostics, FitOptions, FitResult, ModelError, TopologyId, TransitionWeightModel,
};
use crate::network::{RoadNetwork, Trajectory};
use crate::objective::{ObjectiveError, compute_regret};
use crate::optimizer::{OptimizerError, RelativeProjectedSubgradient};
use crate::oracle::{OracleError, OraclePath, OracleQuery, ROUTING_INFINITY, RoutingOracle};
use std::error::Error;
use std::fmt::{Display, Formatter};

type SnapshotObserver<'a> = dyn FnMut(&TrainingOutcome) -> Result<(), String> + 'a;

/// Backend- and serialization-independent state required to resume v1 fitting.
///
/// The direct vector and global optimizer clock are the only evolving values.
/// The remaining fields are immutable identity: they prevent a checkpoint from
/// being silently resumed against a different baseline or optimizer geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingState {
    topology_id: TopologyId,
    training_problem_id: String,
    oracle_identity: String,
    initial_weights: Box<[f64]>,
    weights: Box<[f64]>,
    completed_updates: u64,
    eta0: f64,
    lambda: f64,
    lower_factor: f64,
    upper_factor: f64,
}

impl TrainingState {
    /// Reconstruct state from a storage adapter without depending on its schema.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        topology_id: TopologyId,
        training_problem_id: String,
        oracle_identity: String,
        initial_weights: Vec<f64>,
        weights: Vec<f64>,
        completed_updates: u64,
        eta0: f64,
        lambda: f64,
        lower_factor: f64,
        upper_factor: f64,
    ) -> Result<Self, TrainerError> {
        validate_optimizer_identity(eta0, lambda, lower_factor, upper_factor)?;
        validate_checkpoint_identity("training problem", &training_problem_id)?;
        validate_checkpoint_identity("routing oracle", &oracle_identity)?;
        if initial_weights.is_empty() || initial_weights.len() != weights.len() {
            return Err(TrainerError::InvalidState(
                "initial and direct weight vectors must have the same nonzero length".to_string(),
            ));
        }
        for (coordinate, (&initial, &weight)) in initial_weights.iter().zip(&weights).enumerate() {
            if !initial.is_finite() || initial <= 0.0 {
                return Err(TrainerError::InvalidState(format!(
                    "initial weight[{coordinate}] must be finite and positive, got {initial}"
                )));
            }
            quantize_weight(initial, coordinate)?;
            let lower = initial * lower_factor;
            let upper = initial * upper_factor;
            if !lower.is_finite() || !upper.is_finite() || lower <= 0.0 || upper <= 0.0 {
                return Err(TrainerError::InvalidState(format!(
                    "bounds derived for weight[{coordinate}] must be finite and positive"
                )));
            }
            quantize_weight(upper, coordinate)?;
            if !weight.is_finite() || weight <= 0.0 {
                return Err(TrainerError::InvalidState(format!(
                    "direct weight[{coordinate}] must be finite and positive, got {weight}"
                )));
            }
            if weight < lower || weight > upper {
                return Err(TrainerError::StateWeightOutsideBounds {
                    coordinate,
                    weight,
                    lower,
                    upper,
                });
            }
            quantize_weight(weight, coordinate)?;
        }

        Ok(Self {
            topology_id,
            training_problem_id,
            oracle_identity,
            initial_weights: initial_weights.into_boxed_slice(),
            weights: weights.into_boxed_slice(),
            completed_updates,
            eta0,
            lambda,
            lower_factor,
            upper_factor,
        })
    }

    /// Stable transition-coordinate identity.
    pub fn topology_id(&self) -> &TopologyId {
        &self.topology_id
    }

    /// Versioned identity of the topology, geometry, and training statistics.
    pub fn training_problem_id(&self) -> &str {
        &self.training_problem_id
    }

    /// Versioned identity of the routing semantics used to produce this state.
    pub fn oracle_identity(&self) -> &str {
        &self.oracle_identity
    }

    /// Exact optimizer anchor, also serving as the baseline identity.
    pub fn initial_weights(&self) -> &[f64] {
        &self.initial_weights
    }

    /// Current direct transition weights.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Global number of successful optimizer updates.
    pub const fn completed_updates(&self) -> u64 {
        self.completed_updates
    }

    /// Initial learning rate represented by this state.
    pub const fn eta0(&self) -> f64 {
        self.eta0
    }

    /// Relative regularization coefficient represented by this state.
    pub const fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Multiplicative lower bound represented by this state.
    pub const fn lower_factor(&self) -> f64 {
        self.lower_factor
    }

    /// Multiplicative upper bound represented by this state.
    pub const fn upper_factor(&self) -> f64 {
        self.upper_factor
    }
}

/// Complete fitting output plus the exact state that can be serialized.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingOutcome {
    /// Stable high-level model and diagnostics.
    pub result: FitResult,
    /// Resume state at the same optimizer clock as `result`.
    pub state: TrainingState,
}

/// Sole reusable full-batch trainer for the active v1 algorithm.
pub struct Trainer<'network, 'oracle, O: RoutingOracle + ?Sized> {
    network: &'network RoadNetwork,
    line_graph: LineGraph,
    observed_counts: Vec<u64>,
    groups: Vec<QueryGroup>,
    queries: Vec<OracleQuery>,
    training_problem_id: String,
    oracle_identity: String,
    options: FitOptions,
    oracle: &'oracle mut O,
}

impl<'network, 'oracle, O: RoutingOracle + ?Sized> Trainer<'network, 'oracle, O> {
    /// Validate and prepare one in-memory fitting problem without querying the
    /// routing backend.
    pub fn new(
        network: &'network RoadNetwork,
        trajectories: &[Trajectory],
        options: &FitOptions,
        oracle: &'oracle mut O,
    ) -> Result<Self, TrainerError> {
        let options = options.validate().map_err(TrainerError::Model)?;
        let line_graph = LineGraph::build(network, options.lower_factor, options.upper_factor)?;
        for (coordinate, &upper) in line_graph.upper_bounds().iter().enumerate() {
            quantize_weight(upper, coordinate)?;
        }

        let mapped = line_graph.map_trajectories(trajectories)?;
        if mapped.is_empty() {
            return Err(TrainerError::EmptyTrainingSet);
        }
        let observed_counts = line_graph.observed_counts(&mapped)?;
        let groups = line_graph.group_queries(&mapped)?;
        if groups.is_empty() {
            return Err(TrainerError::EmptyTrainingSet);
        }
        let queries = groups
            .iter()
            .map(|group| {
                let (sources, targets) =
                    line_graph.node_query_endpoints(group.source, group.target)?;
                validate_zero_endpoints(
                    line_graph.routing_topology().node_count(),
                    &sources,
                    &targets,
                )?;
                Ok(OracleQuery::new(sources, targets))
            })
            .collect::<Result<Vec<_>, TrainerError>>()?;
        let training_problem_id = training_problem_identity(
            line_graph.topology_id(),
            line_graph.routing_topology().fingerprint(),
            &observed_counts,
            &groups,
        )?;
        let oracle_identity = oracle.identity().to_string();
        validate_checkpoint_identity("routing oracle", &oracle_identity)?;

        Ok(Self {
            network,
            line_graph,
            observed_counts,
            groups,
            queries,
            training_problem_id,
            oracle_identity,
            options,
            oracle,
        })
    }

    /// Fit from the deterministic baseline and return serializable-independent
    /// state alongside the coordinate-bound model.
    pub fn fit(&mut self) -> Result<TrainingOutcome, TrainerError> {
        self.run(None, None, None)
    }

    /// Fit while exposing self-consistent snapshots at a fixed
    /// update cadence.
    ///
    /// The observer is called at update zero, at each cadence boundary, and at
    /// the final target. Core still owns the only training loop; adapters may
    /// serialize a snapshot but cannot alter optimizer or routing semantics.
    pub fn fit_with_snapshots<F, E>(
        &mut self,
        checkpoint_every: u64,
        mut observer: F,
    ) -> Result<TrainingOutcome, TrainerError>
    where
        F: FnMut(&TrainingOutcome) -> Result<(), E>,
        E: Display,
    {
        let mut erased =
            |outcome: &TrainingOutcome| observer(outcome).map_err(|error| error.to_string());
        self.run(None, Some(checkpoint_every), Some(&mut erased))
    }

    /// Continue fitting from a validated state to `options.updates`.
    ///
    /// Validation is completed before the state is cloned, before the oracle is
    /// called, and before any optimizer mutation can occur.
    pub fn resume(&mut self, state: &TrainingState) -> Result<TrainingOutcome, TrainerError> {
        self.run(Some(state), None, None)
    }

    /// Continue from a validated state while exposing periodic snapshots.
    ///
    /// The restored clock is always observed once, even when it is not a
    /// cadence multiple, so a new output location immediately receives the
    /// exact accepted resume state.
    pub fn resume_with_snapshots<F, E>(
        &mut self,
        state: &TrainingState,
        checkpoint_every: u64,
        mut observer: F,
    ) -> Result<TrainingOutcome, TrainerError>
    where
        F: FnMut(&TrainingOutcome) -> Result<(), E>,
        E: Display,
    {
        let mut erased =
            |outcome: &TrainingOutcome| observer(outcome).map_err(|error| error.to_string());
        self.run(Some(state), Some(checkpoint_every), Some(&mut erased))
    }

    fn run(
        &mut self,
        resume_state: Option<&TrainingState>,
        checkpoint_every: Option<u64>,
        mut observer: Option<&mut SnapshotObserver<'_>>,
    ) -> Result<TrainingOutcome, TrainerError> {
        if checkpoint_every == Some(0) {
            return Err(TrainerError::InvalidSnapshotCadence(0));
        }
        let (mut weights, completed_updates) = match resume_state {
            Some(state) => {
                self.validate_resume_state(state)?;
                (state.weights.to_vec(), state.completed_updates)
            }
            None => (self.line_graph.initial_weights().to_vec(), 0),
        };
        let mut optimizer = RelativeProjectedSubgradient::with_completed_updates(
            self.options.eta0,
            self.options.lambda,
            completed_updates,
        )?;
        let restored_updates = optimizer.completed_updates();

        loop {
            let batch = self.evaluate_batch(&weights)?;
            let completed_updates = optimizer.completed_updates();
            let final_state = completed_updates == self.options.updates;
            let snapshot_due = checkpoint_every.is_some_and(|cadence| {
                completed_updates == restored_updates
                    || completed_updates % cadence == 0
                    || final_state
            });
            if final_state || snapshot_due {
                let regret = compute_regret(
                    &weights,
                    &self.observed_counts,
                    batch.weighted_direct_cost_sum,
                    batch.sample_count,
                )?;
                let objective = regret.mean_data_loss
                    + optimizer.regularization(&weights, self.line_graph.initial_weights())?;
                if !objective.is_finite() {
                    return Err(TrainerError::NonFiniteObjective(objective));
                }
                let outcome = self.make_outcome(&weights, completed_updates, objective)?;
                if snapshot_due {
                    observer
                        .as_mut()
                        .expect("a snapshot cadence always has an observer")(
                        &outcome
                    )
                    .map_err(TrainerError::SnapshotObserver)?;
                }
                if final_state {
                    return Ok(outcome);
                }
            }

            optimizer.step(
                &mut weights,
                self.line_graph.initial_weights(),
                self.line_graph.lower_bounds(),
                self.line_graph.upper_bounds(),
                &self.observed_counts,
                &batch.predicted_counts,
                batch.sample_count,
            )?;
        }
    }

    fn make_outcome(
        &self,
        weights: &[f64],
        completed_updates: u64,
        objective: f64,
    ) -> Result<TrainingOutcome, TrainerError> {
        let state = TrainingState::from_parts(
            self.line_graph.topology_id().clone(),
            self.training_problem_id.clone(),
            self.oracle_identity.clone(),
            self.line_graph.initial_weights().to_vec(),
            weights.to_vec(),
            completed_updates,
            self.options.eta0,
            self.options.lambda,
            self.options.lower_factor,
            self.options.upper_factor,
        )?;
        let model = TransitionWeightModel::new(
            self.line_graph.topology_id().clone(),
            self.line_graph.transitions().to_vec(),
            weights.to_vec(),
        )?;
        Ok(TrainingOutcome {
            result: FitResult {
                model,
                diagnostics: FitDiagnostics {
                    completed_updates,
                    objective,
                },
            },
            state,
        })
    }

    fn validate_resume_state(&self, state: &TrainingState) -> Result<(), TrainerError> {
        if state.topology_id != *self.line_graph.topology_id() {
            return Err(TrainerError::StateTopologyMismatch {
                expected: self.line_graph.topology_id().clone(),
                actual: state.topology_id.clone(),
            });
        }
        if state.training_problem_id != self.training_problem_id {
            return Err(TrainerError::StateTrainingProblemMismatch {
                expected: self.training_problem_id.clone(),
                actual: state.training_problem_id.clone(),
            });
        }
        if state.oracle_identity != self.oracle_identity {
            return Err(TrainerError::StateOracleMismatch {
                expected: self.oracle_identity.clone(),
                actual: state.oracle_identity.clone(),
            });
        }
        for (name, state_value, option_value) in [
            ("eta0", state.eta0, self.options.eta0),
            ("lambda", state.lambda, self.options.lambda),
            (
                "lower_factor",
                state.lower_factor,
                self.options.lower_factor,
            ),
            (
                "upper_factor",
                state.upper_factor,
                self.options.upper_factor,
            ),
        ] {
            if state_value.to_bits() != option_value.to_bits() {
                return Err(TrainerError::StateOptionMismatch {
                    option: name,
                    expected: option_value,
                    actual: state_value,
                });
            }
        }
        if !same_f64_bits(&state.initial_weights, self.line_graph.initial_weights()) {
            return Err(TrainerError::StateInitialWeightsMismatch);
        }
        if state.weights.len() != self.line_graph.coordinate_count() {
            return Err(TrainerError::StateWeightLengthMismatch {
                expected: self.line_graph.coordinate_count(),
                actual: state.weights.len(),
            });
        }
        if state.completed_updates > self.options.updates {
            return Err(TrainerError::StateClockAhead {
                completed: state.completed_updates,
                target: self.options.updates,
            });
        }
        for (coordinate, ((&weight, &lower), &upper)) in state
            .weights
            .iter()
            .zip(self.line_graph.lower_bounds())
            .zip(self.line_graph.upper_bounds())
            .enumerate()
        {
            if !weight.is_finite() || weight < lower || weight > upper {
                return Err(TrainerError::StateWeightOutsideBounds {
                    coordinate,
                    weight,
                    lower,
                    upper,
                });
            }
            quantize_weight(weight, coordinate)?;
        }
        Ok(())
    }

    fn evaluate_batch(&mut self, weights: &[f64]) -> Result<BatchEvaluation, TrainerError> {
        let quantized = weights
            .iter()
            .copied()
            .enumerate()
            .map(|(coordinate, weight)| quantize_weight(weight, coordinate))
            .collect::<Result<Vec<_>, _>>()?;
        let paths = self.oracle.shortest_paths(
            self.line_graph.routing_topology(),
            &quantized,
            &self.queries,
        )?;
        if paths.len() != self.groups.len() {
            return Err(TrainerError::OraclePathCountMismatch {
                expected: self.groups.len(),
                actual: paths.len(),
            });
        }

        let mut predicted_counts = vec![0u64; self.line_graph.coordinate_count()];
        let mut weighted_direct_cost_sum = 0.0;
        let mut sample_count = 0u64;
        for (query_index, ((group, query), path)) in self
            .groups
            .iter()
            .zip(&self.queries)
            .zip(&paths)
            .enumerate()
        {
            self.validate_oracle_path(query_index, group, query, path, &quantized)?;
            let mut direct_cost = 0.0;
            for &coordinate in path.coordinates() {
                predicted_counts[coordinate] = predicted_counts[coordinate]
                    .checked_add(group.sample_count)
                    .ok_or(TrainerError::AggregationOverflow(
                        "predicted coordinate count",
                    ))?;
                direct_cost += weights[coordinate];
                if !direct_cost.is_finite() {
                    return Err(TrainerError::NonFiniteDirectPathCost { query: query_index });
                }
            }
            weighted_direct_cost_sum += direct_cost * group.sample_count as f64;
            if !weighted_direct_cost_sum.is_finite() {
                return Err(TrainerError::NonFiniteDirectPathCost { query: query_index });
            }
            sample_count = sample_count
                .checked_add(group.sample_count)
                .ok_or(TrainerError::AggregationOverflow("oracle sample count"))?;
        }

        Ok(BatchEvaluation {
            predicted_counts,
            weighted_direct_cost_sum,
            sample_count,
        })
    }

    fn validate_oracle_path(
        &self,
        query_index: usize,
        group: &QueryGroup,
        query: &OracleQuery,
        path: &OraclePath,
        quantized: &[u32],
    ) -> Result<(), TrainerError> {
        if path.distance() >= ROUTING_INFINITY {
            return invalid_oracle_path(query_index, "distance reaches the infinity sentinel");
        }
        if path.nodes().len() != path.coordinates().len().saturating_add(1)
            || path.nodes().is_empty()
        {
            return invalid_oracle_path(
                query_index,
                "node path must contain exactly one more item than the coordinate path",
            );
        }
        let first = path.nodes()[0];
        let last = path.nodes()[path.nodes().len() - 1];
        if !query
            .sources()
            .iter()
            .any(|endpoint| endpoint.node() == first)
            || !query
                .targets()
                .iter()
                .any(|endpoint| endpoint.node() == last)
        {
            return invalid_oracle_path(
                query_index,
                "path endpoints are not members of the requested endpoint sets",
            );
        }
        if self.network.tail(first) != Some(group.source)
            || self.network.head(last) != Some(group.target)
        {
            return invalid_oracle_path(query_index, "decoded path does not match the query OD");
        }

        let topology = self.line_graph.routing_topology();
        let mut reconstructed = 0u128;
        for (&coordinate, nodes) in path.coordinates().iter().zip(path.nodes().windows(2)) {
            let Some((&tail, &head)) = topology
                .tails()
                .get(coordinate)
                .zip(topology.heads().get(coordinate))
            else {
                return invalid_oracle_path(
                    query_index,
                    "coordinate is outside the routing topology",
                );
            };
            if (tail, head) != (nodes[0], nodes[1]) {
                return invalid_oracle_path(
                    query_index,
                    "coordinate endpoints disagree with the routing-node path",
                );
            }
            reconstructed = reconstructed
                .checked_add(u128::from(quantized[coordinate]))
                .ok_or(TrainerError::AggregationOverflow(
                    "reconstructed quantized path cost",
                ))?;
        }
        if reconstructed != u128::from(path.distance()) {
            return invalid_oracle_path(
                query_index,
                "reported distance differs from the coordinate-path metric sum",
            );
        }

        if !path.coordinates().is_empty() {
            let decoded = self
                .line_graph
                .decode_coordinates(path.coordinates())
                .map_err(|error| TrainerError::InvalidOraclePath {
                    query: query_index,
                    reason: format!("invalid coordinate path: {error}"),
                })?;
            if decoded != path.nodes() {
                return invalid_oracle_path(
                    query_index,
                    "decoded coordinates disagree with the routing-node path",
                );
            }
        }
        Ok(())
    }
}

/// Fit once from the deterministic baseline using the supplied backend.
///
/// Concrete production adapters such as `ewr-cch` can wrap this function to
/// expose the three-argument high-level boundary documented for applications.
pub fn fit<O: RoutingOracle + ?Sized>(
    network: &RoadNetwork,
    trajectories: &[Trajectory],
    options: &FitOptions,
    oracle: &mut O,
) -> Result<FitResult, TrainerError> {
    Ok(Trainer::new(network, trajectories, options, oracle)?
        .fit()?
        .result)
}

#[derive(Debug)]
struct BatchEvaluation {
    predicted_counts: Vec<u64>,
    weighted_direct_cost_sum: f64,
    sample_count: u64,
}

/// Invalid core training input, state, backend response, or arithmetic result.
#[derive(Debug)]
pub enum TrainerError {
    Model(ModelError),
    LineGraph(LineGraphError),
    Oracle(OracleError),
    Objective(ObjectiveError),
    Optimizer(OptimizerError),
    EmptyTrainingSet,
    InvalidState(String),
    InvalidSnapshotCadence(u64),
    SnapshotObserver(String),
    InvalidQuantizedWeight {
        coordinate: usize,
        weight: f64,
    },
    StateTopologyMismatch {
        expected: TopologyId,
        actual: TopologyId,
    },
    StateTrainingProblemMismatch {
        expected: String,
        actual: String,
    },
    StateOracleMismatch {
        expected: String,
        actual: String,
    },
    StateOptionMismatch {
        option: &'static str,
        expected: f64,
        actual: f64,
    },
    StateInitialWeightsMismatch,
    StateWeightLengthMismatch {
        expected: usize,
        actual: usize,
    },
    StateClockAhead {
        completed: u64,
        target: u64,
    },
    StateWeightOutsideBounds {
        coordinate: usize,
        weight: f64,
        lower: f64,
        upper: f64,
    },
    InvalidQueryEndpoints(String),
    OraclePathCountMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidOraclePath {
        query: usize,
        reason: String,
    },
    AggregationOverflow(&'static str),
    NonFiniteDirectPathCost {
        query: usize,
    },
    NonFiniteObjective(f64),
}

impl Display for TrainerError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Model(error) => Display::fmt(error, formatter),
            Self::LineGraph(error) => Display::fmt(error, formatter),
            Self::Oracle(error) => write!(formatter, "routing oracle failed: {error}"),
            Self::Objective(error) => Display::fmt(error, formatter),
            Self::Optimizer(error) => Display::fmt(error, formatter),
            Self::EmptyTrainingSet => {
                formatter.write_str("training trajectories must not be empty")
            }
            Self::InvalidState(reason) => write!(formatter, "invalid training state: {reason}"),
            Self::InvalidSnapshotCadence(cadence) => write!(
                formatter,
                "snapshot cadence must be positive, got {cadence}"
            ),
            Self::SnapshotObserver(reason) => {
                write!(formatter, "training snapshot observer failed: {reason}")
            }
            Self::InvalidQuantizedWeight { coordinate, weight } => write!(
                formatter,
                "weight[{coordinate}]={weight} cannot be represented by the v1 routing metric"
            ),
            Self::StateTopologyMismatch { expected, actual } => write!(
                formatter,
                "training-state topology {:?} does not match {:?}",
                actual.as_str(),
                expected.as_str()
            ),
            Self::StateTrainingProblemMismatch { expected, actual } => write!(
                formatter,
                "training-state problem identity {actual:?} does not match {expected:?}"
            ),
            Self::StateOracleMismatch { expected, actual } => write!(
                formatter,
                "training-state oracle identity {actual:?} does not match {expected:?}"
            ),
            Self::StateOptionMismatch {
                option,
                expected,
                actual,
            } => write!(
                formatter,
                "training-state {option}={actual} does not match requested {expected}"
            ),
            Self::StateInitialWeightsMismatch => formatter.write_str(
                "training-state initial weights do not match the current network baseline",
            ),
            Self::StateWeightLengthMismatch { expected, actual } => write!(
                formatter,
                "training-state weight length {actual} does not match {expected} coordinates"
            ),
            Self::StateClockAhead { completed, target } => write!(
                formatter,
                "training-state clock {completed} is ahead of target {target}"
            ),
            Self::StateWeightOutsideBounds {
                coordinate,
                weight,
                lower,
                upper,
            } => write!(
                formatter,
                "training-state weight[{coordinate}]={weight} is outside [{lower}, {upper}]"
            ),
            Self::InvalidQueryEndpoints(reason) => {
                write!(formatter, "invalid line-graph query endpoints: {reason}")
            }
            Self::OraclePathCountMismatch { expected, actual } => write!(
                formatter,
                "routing oracle returned {actual} paths for {expected} queries"
            ),
            Self::InvalidOraclePath { query, reason } => {
                write!(
                    formatter,
                    "routing oracle returned invalid path {query}: {reason}"
                )
            }
            Self::AggregationOverflow(kind) => write!(formatter, "{kind} overflow"),
            Self::NonFiniteDirectPathCost { query } => {
                write!(
                    formatter,
                    "direct path-cost sum is not finite at query {query}"
                )
            }
            Self::NonFiniteObjective(value) => {
                write!(formatter, "final training objective is not finite: {value}")
            }
        }
    }
}

impl Error for TrainerError {}

impl From<ModelError> for TrainerError {
    fn from(error: ModelError) -> Self {
        Self::Model(error)
    }
}

impl From<LineGraphError> for TrainerError {
    fn from(error: LineGraphError) -> Self {
        Self::LineGraph(error)
    }
}

impl From<OracleError> for TrainerError {
    fn from(error: OracleError) -> Self {
        Self::Oracle(error)
    }
}

impl From<ObjectiveError> for TrainerError {
    fn from(error: ObjectiveError) -> Self {
        Self::Objective(error)
    }
}

impl From<OptimizerError> for TrainerError {
    fn from(error: OptimizerError) -> Self {
        Self::Optimizer(error)
    }
}

fn quantize_weight(weight: f64, coordinate: usize) -> Result<u32, TrainerError> {
    if !weight.is_finite() || weight <= 0.0 {
        return Err(TrainerError::InvalidQuantizedWeight { coordinate, weight });
    }
    let rounded = weight.round().max(1.0);
    if rounded >= f64::from(ROUTING_INFINITY) {
        return Err(TrainerError::InvalidQuantizedWeight { coordinate, weight });
    }
    Ok(rounded as u32)
}

fn validate_optimizer_identity(
    eta0: f64,
    lambda: f64,
    lower_factor: f64,
    upper_factor: f64,
) -> Result<(), TrainerError> {
    let updates = 1;
    FitOptions {
        eta0,
        lambda,
        lower_factor,
        upper_factor,
        updates,
    }
    .validate()
    .map(|_| ())
    .map_err(TrainerError::Model)
}

fn validate_checkpoint_identity(kind: &'static str, identity: &str) -> Result<(), TrainerError> {
    if identity.trim().is_empty() || identity.chars().any(char::is_control) {
        return Err(TrainerError::InvalidState(format!(
            "{kind} identity must be nonblank and contain no control characters"
        )));
    }
    Ok(())
}

/// Hash only the deterministic sufficient statistics consumed by v1 fitting.
///
/// Raw trajectory order is deliberately absent: the optimizer sees aggregate
/// transition counts and stable OD multiplicities, so two inputs with those
/// same statistics are the same resumable training problem.
fn training_problem_identity(
    topology_id: &TopologyId,
    routing_fingerprint: u64,
    observed_counts: &[u64],
    groups: &[QueryGroup],
) -> Result<String, TrainerError> {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    hash_identity_bytes(&mut hash, b"ewr-training-problem-v1");
    hash_identity_u64(&mut hash, topology_id.as_str().len() as u64);
    hash_identity_bytes(&mut hash, topology_id.as_str().as_bytes());
    hash_identity_u64(&mut hash, routing_fingerprint);

    hash_identity_u64(&mut hash, observed_counts.len() as u64);
    for &count in observed_counts {
        hash_identity_u64(&mut hash, count);
    }

    hash_identity_u64(&mut hash, groups.len() as u64);
    let mut sample_count = 0u64;
    for group in groups {
        hash_identity_u64(&mut hash, group.source.index() as u64);
        hash_identity_u64(&mut hash, group.target.index() as u64);
        hash_identity_u64(&mut hash, group.sample_count);
        sample_count = sample_count.checked_add(group.sample_count).ok_or(
            TrainerError::AggregationOverflow("training-problem sample count"),
        )?;
    }
    hash_identity_u64(&mut hash, sample_count);

    Ok(format!("training-problem-v1:fnv1a64:{hash:016x}"))
}

fn hash_identity_u64(hash: &mut u64, value: u64) {
    hash_identity_bytes(hash, &value.to_le_bytes());
}

fn hash_identity_bytes(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn validate_zero_endpoints(
    node_count: usize,
    sources: &[crate::oracle::QueryEndpoint],
    targets: &[crate::oracle::QueryEndpoint],
) -> Result<(), TrainerError> {
    if sources.is_empty() || targets.is_empty() {
        return Err(TrainerError::InvalidQueryEndpoints(
            "source and target lists must both be nonempty".to_string(),
        ));
    }
    for endpoint in sources.iter().chain(targets) {
        if endpoint.node().index() >= node_count {
            return Err(TrainerError::InvalidQueryEndpoints(format!(
                "edge state {} is outside {node_count} routing nodes",
                endpoint.node().index()
            )));
        }
        if endpoint.offset() != 0 {
            return Err(TrainerError::InvalidQueryEndpoints(format!(
                "edge state {} has nonzero v1 offset {}",
                endpoint.node().index(),
                endpoint.offset()
            )));
        }
    }
    Ok(())
}

fn invalid_oracle_path<T>(query: usize, reason: &str) -> Result<T, TrainerError> {
    Err(TrainerError::InvalidOraclePath {
        query,
        reason: reason.to_string(),
    })
}

fn same_f64_bits(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(&left, &right)| left.to_bits() == right.to_bits())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RoutingTopology;
    use crate::network::{EdgeId, NodeId};
    use crate::oracle::{OraclePath, RoutingOracle};

    struct OneArcOracle {
        calls: usize,
        identity: &'static str,
    }

    impl Default for OneArcOracle {
        fn default() -> Self {
            Self {
                calls: 0,
                identity: "ewr-core-test:one-arc-oracle:v1",
            }
        }
    }

    impl RoutingOracle for OneArcOracle {
        fn identity(&self) -> &'static str {
            self.identity
        }

        fn shortest_paths(
            &mut self,
            topology: &RoutingTopology,
            quantized_weights: &[u32],
            queries: &[OracleQuery],
        ) -> Result<Vec<OraclePath>, OracleError> {
            self.calls += 1;
            queries
                .iter()
                .map(|query| {
                    let coordinate = topology
                        .tails()
                        .iter()
                        .zip(topology.heads())
                        .enumerate()
                        .filter(|(_, (tail, head))| {
                            query
                                .sources()
                                .iter()
                                .any(|endpoint| endpoint.node() == **tail)
                                && query
                                    .targets()
                                    .iter()
                                    .any(|endpoint| endpoint.node() == **head)
                        })
                        .min_by_key(|(coordinate, _)| (quantized_weights[*coordinate], *coordinate))
                        .map(|(coordinate, _)| coordinate)
                        .ok_or_else(|| OracleError::new("fixture query has no one-arc route"))?;
                    Ok(OraclePath::new(
                        quantized_weights[coordinate],
                        vec![topology.tails()[coordinate], topology.heads()[coordinate]],
                        vec![coordinate],
                    ))
                })
                .collect()
        }
    }

    fn network() -> RoadNetwork {
        RoadNetwork::new(
            vec![
                NodeId::new(0),
                NodeId::new(1),
                NodeId::new(0),
                NodeId::new(2),
            ],
            vec![
                NodeId::new(1),
                NodeId::new(3),
                NodeId::new(2),
                NodeId::new(3),
            ],
            vec![5.0, 5.0, 2.0, 2.0],
            vec![0.0, 1.0, 1.0, 2.0],
            vec![0.0, 0.0, 1.0, 0.0],
        )
        .unwrap()
    }

    fn observations() -> Vec<Trajectory> {
        vec![
            Trajectory::new(vec![EdgeId::new(0), EdgeId::new(1)]),
            Trajectory::new(vec![EdgeId::new(0), EdgeId::new(1)]),
        ]
    }

    fn options(updates: u64) -> FitOptions {
        FitOptions {
            eta0: 0.5,
            lambda: 0.1,
            lower_factor: 0.1,
            upper_factor: 10.0,
            updates,
        }
    }

    #[test]
    fn public_one_update_freezes_active_relative_training_semantics() {
        let mut oracle = OneArcOracle::default();
        let outcome = Trainer::new(&network(), &observations(), &options(1), &mut oracle)
            .unwrap()
            .fit()
            .unwrap();

        assert_eq!(outcome.result.model.weights(), &[0.5, 4.0]);
        assert_eq!(outcome.result.diagnostics.completed_updates, 1);
        assert_eq!(outcome.state.weights(), &[0.5, 4.0]);
        // One query before the update and one final-objective query.
        assert_eq!(oracle.calls, 2);
    }

    #[test]
    fn snapshot_observer_sees_initial_cadence_and_final_states() {
        let network = network();
        let observations = observations();
        let mut oracle = OneArcOracle::default();
        let mut clocks = Vec::new();
        let outcome = Trainer::new(&network, &observations, &options(5), &mut oracle)
            .unwrap()
            .fit_with_snapshots(2, |snapshot| {
                clocks.push(snapshot.state.completed_updates());
                assert_eq!(
                    snapshot.result.diagnostics.completed_updates,
                    snapshot.state.completed_updates()
                );
                assert_eq!(snapshot.result.model.weights(), snapshot.state.weights());
                Ok::<(), String>(())
            })
            .unwrap();

        assert_eq!(clocks, vec![0, 2, 4, 5]);
        assert_eq!(outcome.state.completed_updates(), 5);
        assert_eq!(oracle.calls, 6);
    }

    #[test]
    fn invalid_snapshot_cadence_is_rejected_before_oracle_call() {
        let network = network();
        let observations = observations();
        let mut oracle = OneArcOracle::default();
        let error = Trainer::new(&network, &observations, &options(1), &mut oracle)
            .unwrap()
            .fit_with_snapshots(0, |_| Ok::<(), String>(()))
            .unwrap_err();

        assert!(matches!(error, TrainerError::InvalidSnapshotCadence(0)));
        assert_eq!(oracle.calls, 0);
    }

    #[test]
    fn resumed_training_is_bitwise_equal_to_uninterrupted_training() {
        let network = network();
        let observations = observations();
        let mut first_oracle = OneArcOracle::default();
        let first = Trainer::new(&network, &observations, &options(2), &mut first_oracle)
            .unwrap()
            .fit()
            .unwrap();

        let mut resumed_oracle = OneArcOracle::default();
        let resumed = Trainer::new(&network, &observations, &options(4), &mut resumed_oracle)
            .unwrap()
            .resume(&first.state)
            .unwrap();
        let mut uninterrupted_oracle = OneArcOracle::default();
        let uninterrupted = Trainer::new(
            &network,
            &observations,
            &options(4),
            &mut uninterrupted_oracle,
        )
        .unwrap()
        .fit()
        .unwrap();

        assert_eq!(resumed.state, uninterrupted.state);
        assert_eq!(resumed.result, uninterrupted.result);
    }

    #[test]
    fn mismatched_state_is_rejected_before_the_oracle_is_called() {
        let network = network();
        let observations = observations();
        let mut seed_oracle = OneArcOracle::default();
        let seed = Trainer::new(&network, &observations, &options(1), &mut seed_oracle)
            .unwrap()
            .fit()
            .unwrap();
        let original_state = seed.state.clone();
        let wrong = TrainingState::from_parts(
            TopologyId::new("line-graph-v1:wrong").unwrap(),
            seed.state.training_problem_id().to_string(),
            seed.state.oracle_identity().to_string(),
            seed.state.initial_weights().to_vec(),
            seed.state.weights().to_vec(),
            seed.state.completed_updates(),
            seed.state.eta0(),
            seed.state.lambda(),
            seed.state.lower_factor(),
            seed.state.upper_factor(),
        )
        .unwrap();

        let mut oracle = OneArcOracle::default();
        let error = Trainer::new(&network, &observations, &options(4), &mut oracle)
            .unwrap()
            .resume(&wrong)
            .unwrap_err();
        assert!(matches!(error, TrainerError::StateTopologyMismatch { .. }));
        assert_eq!(oracle.calls, 0);
        assert_eq!(seed.state, original_state);
    }

    #[test]
    fn changed_baseline_is_rejected_even_when_topology_is_identical() {
        let network = network();
        let observations = observations();
        let mut seed_oracle = OneArcOracle::default();
        let seed = Trainer::new(&network, &observations, &options(1), &mut seed_oracle)
            .unwrap()
            .fit()
            .unwrap();
        let changed_baseline = RoadNetwork::new(
            network.tails().to_vec(),
            network.heads().to_vec(),
            vec![5.0, 6.0, 2.0, 3.0],
            network.x().to_vec(),
            network.y().to_vec(),
        )
        .unwrap();
        let changed_graph = LineGraph::build(&changed_baseline, 0.1, 10.0).unwrap();
        assert_eq!(changed_graph.topology_id(), seed.state.topology_id());

        let mut oracle = OneArcOracle::default();
        let error = Trainer::new(&changed_baseline, &observations, &options(4), &mut oracle)
            .unwrap()
            .resume(&seed.state)
            .unwrap_err();
        assert!(matches!(error, TrainerError::StateInitialWeightsMismatch));
        assert_eq!(oracle.calls, 0);
    }

    #[test]
    fn changed_trajectory_statistics_are_rejected_before_oracle_call() {
        let network = network();
        let observations = observations();
        let mut seed_oracle = OneArcOracle::default();
        let seed = Trainer::new(&network, &observations, &options(1), &mut seed_oracle)
            .unwrap()
            .fit()
            .unwrap();
        let changed_observations = vec![
            Trajectory::new(vec![EdgeId::new(0), EdgeId::new(1)]),
            Trajectory::new(vec![EdgeId::new(2), EdgeId::new(3)]),
        ];

        let mut oracle = OneArcOracle::default();
        let error = Trainer::new(&network, &changed_observations, &options(4), &mut oracle)
            .unwrap()
            .resume(&seed.state)
            .unwrap_err();

        assert!(matches!(
            error,
            TrainerError::StateTrainingProblemMismatch { .. }
        ));
        assert_eq!(oracle.calls, 0);
    }

    #[test]
    fn problem_identity_binds_od_groups_even_when_observed_counts_match() {
        let network = RoadNetwork::new(
            vec![NodeId::new(0), NodeId::new(1), NodeId::new(2)],
            vec![NodeId::new(1), NodeId::new(2), NodeId::new(3)],
            vec![1.0, 1.0, 1.0],
            vec![0.0, 1.0, 2.0, 3.0],
            vec![0.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        let graph = LineGraph::build(&network, 0.1, 10.0).unwrap();
        let one_complete_path = graph
            .map_trajectories(&[Trajectory::new(vec![
                EdgeId::new(0),
                EdgeId::new(1),
                EdgeId::new(2),
            ])])
            .unwrap();
        let two_path_fragments = graph
            .map_trajectories(&[
                Trajectory::new(vec![EdgeId::new(0), EdgeId::new(1)]),
                Trajectory::new(vec![EdgeId::new(1), EdgeId::new(2)]),
            ])
            .unwrap();
        let complete_counts = graph.observed_counts(&one_complete_path).unwrap();
        let fragment_counts = graph.observed_counts(&two_path_fragments).unwrap();
        assert_eq!(complete_counts, fragment_counts);

        let complete_groups = graph.group_queries(&one_complete_path).unwrap();
        let fragment_groups = graph.group_queries(&two_path_fragments).unwrap();
        let complete_id = training_problem_identity(
            graph.topology_id(),
            graph.routing_topology().fingerprint(),
            &complete_counts,
            &complete_groups,
        )
        .unwrap();
        let fragment_id = training_problem_identity(
            graph.topology_id(),
            graph.routing_topology().fingerprint(),
            &fragment_counts,
            &fragment_groups,
        )
        .unwrap();

        assert_ne!(complete_groups, fragment_groups);
        assert_ne!(complete_id, fragment_id);
    }

    #[test]
    fn changed_routing_geometry_is_rejected_before_oracle_call() {
        let network = network();
        let observations = observations();
        let mut seed_oracle = OneArcOracle::default();
        let seed = Trainer::new(&network, &observations, &options(1), &mut seed_oracle)
            .unwrap()
            .fit()
            .unwrap();
        let mut changed_x = network.x().to_vec();
        changed_x[0] = 10.0;
        let changed_geometry = RoadNetwork::new(
            network.tails().to_vec(),
            network.heads().to_vec(),
            network.baseline_weights().to_vec(),
            changed_x,
            network.y().to_vec(),
        )
        .unwrap();
        let changed_graph = LineGraph::build(&changed_geometry, 0.1, 10.0).unwrap();
        assert_eq!(changed_graph.topology_id(), seed.state.topology_id());

        let mut oracle = OneArcOracle::default();
        let error = Trainer::new(&changed_geometry, &observations, &options(4), &mut oracle)
            .unwrap()
            .resume(&seed.state)
            .unwrap_err();

        assert!(matches!(
            error,
            TrainerError::StateTrainingProblemMismatch { .. }
        ));
        assert_eq!(oracle.calls, 0);
    }

    #[test]
    fn changed_oracle_identity_is_rejected_before_oracle_call() {
        let network = network();
        let observations = observations();
        let mut seed_oracle = OneArcOracle::default();
        let seed = Trainer::new(&network, &observations, &options(1), &mut seed_oracle)
            .unwrap()
            .fit()
            .unwrap();

        let mut oracle = OneArcOracle {
            calls: 0,
            identity: "ewr-core-test:one-arc-oracle:v2",
        };
        let error = Trainer::new(&network, &observations, &options(4), &mut oracle)
            .unwrap()
            .resume(&seed.state)
            .unwrap_err();

        assert!(matches!(error, TrainerError::StateOracleMismatch { .. }));
        assert_eq!(oracle.calls, 0);
    }

    #[test]
    fn trajectory_order_with_same_sufficient_statistics_can_resume() {
        let network = network();
        let first_order = vec![
            Trajectory::new(vec![EdgeId::new(0), EdgeId::new(1)]),
            Trajectory::new(vec![EdgeId::new(2), EdgeId::new(3)]),
        ];
        let second_order = vec![first_order[1].clone(), first_order[0].clone()];
        let mut seed_oracle = OneArcOracle::default();
        let seed = Trainer::new(&network, &first_order, &options(1), &mut seed_oracle)
            .unwrap()
            .fit()
            .unwrap();

        let mut oracle = OneArcOracle::default();
        let resumed = Trainer::new(&network, &second_order, &options(2), &mut oracle)
            .unwrap()
            .resume(&seed.state)
            .unwrap();

        assert_eq!(
            resumed.state.training_problem_id(),
            seed.state.training_problem_id()
        );
        assert_eq!(oracle.calls, 2);
    }
}
