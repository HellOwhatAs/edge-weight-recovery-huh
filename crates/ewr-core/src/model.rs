use crate::EdgeId;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Stable identity of the topology and transition coordinate order.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct TopologyId(String);

impl TopologyId {
    /// Construct a nonempty topology identity.
    pub fn new(value: impl Into<String>) -> Result<Self, ModelError> {
        let value = value.into();
        if value.is_empty() {
            return Err(ModelError::EmptyTopologyId);
        }
        Ok(Self(value))
    }

    /// Borrow the stable textual identity.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// One learned directed-line-graph coordinate.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Transition {
    /// Original edge left by this transition.
    pub previous: EdgeId,
    /// Original edge entered by this transition.
    pub next: EdgeId,
}

/// Learned transition weights bound to their coordinate identity.
#[derive(Clone, Debug, PartialEq)]
pub struct TransitionWeightModel {
    topology_id: TopologyId,
    transitions: Box<[Transition]>,
    weights: Box<[f64]>,
}

impl TransitionWeightModel {
    /// Construct a validated model without losing coordinate meaning.
    pub fn new(
        topology_id: TopologyId,
        transitions: Vec<Transition>,
        weights: Vec<f64>,
    ) -> Result<Self, ModelError> {
        if transitions.is_empty() || transitions.len() != weights.len() {
            return Err(ModelError::CoordinateLengthMismatch {
                transitions: transitions.len(),
                weights: weights.len(),
            });
        }
        if let Some((coordinate, _)) = transitions
            .windows(2)
            .enumerate()
            .find(|(_, pair)| (pair[0].previous, pair[0].next) >= (pair[1].previous, pair[1].next))
        {
            return Err(ModelError::InvalidTransitionOrder {
                coordinate: coordinate + 1,
            });
        }
        if let Some((coordinate, weight)) = weights
            .iter()
            .copied()
            .enumerate()
            .find(|(_, weight)| !weight.is_finite() || *weight <= 0.0)
        {
            return Err(ModelError::InvalidWeight { coordinate, weight });
        }
        Ok(Self {
            topology_id,
            transitions: transitions.into_boxed_slice(),
            weights: weights.into_boxed_slice(),
        })
    }

    /// Topology and coordinate-order identity.
    pub fn topology_id(&self) -> &TopologyId {
        &self.topology_id
    }

    /// Transition meaning of every weight coordinate.
    pub fn transitions(&self) -> &[Transition] {
        &self.transitions
    }

    /// Positive direct transition weights.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
}

/// Stable options of the active v1 optimizer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitOptions {
    /// Initial learning rate before the global square-root schedule.
    pub eta0: f64,
    /// Relative-coordinate regularization coefficient.
    pub lambda: f64,
    /// Multiplicative lower projection bound around the baseline.
    pub lower_factor: f64,
    /// Multiplicative upper projection bound around the baseline.
    pub upper_factor: f64,
    /// Target number of completed updates.
    pub updates: u64,
}

impl FitOptions {
    /// Validate active v1 fit options.
    pub fn validate(self) -> Result<Self, ModelError> {
        if !self.eta0.is_finite() || self.eta0 <= 0.0 {
            return Err(ModelError::InvalidOption("eta0"));
        }
        if !self.lambda.is_finite() || self.lambda < 0.0 {
            return Err(ModelError::InvalidOption("lambda"));
        }
        if !self.lower_factor.is_finite()
            || !self.upper_factor.is_finite()
            || self.lower_factor <= 0.0
            || self.lower_factor > 1.0
            || self.upper_factor < 1.0
        {
            return Err(ModelError::InvalidOption("weight factors"));
        }
        if self.updates == 0 {
            return Err(ModelError::InvalidOption("updates"));
        }
        Ok(self)
    }
}

/// Minimal algorithm diagnostics independent from any experiment log schema.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitDiagnostics {
    /// Number of optimizer updates represented by the returned state.
    pub completed_updates: u64,
    /// Final full-batch training objective.
    pub objective: f64,
}

/// High-level result returned by the production fitting boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct FitResult {
    /// Learned and coordinate-bound transition model.
    pub model: TransitionWeightModel,
    /// Backend- and experiment-independent fit diagnostics.
    pub diagnostics: FitDiagnostics,
}

/// Invalid model or fit configuration.
#[derive(Clone, Debug, PartialEq)]
pub enum ModelError {
    /// Topology identities must be explicit.
    EmptyTopologyId,
    /// Coordinate metadata and weights differ in length or are empty.
    CoordinateLengthMismatch { transitions: usize, weights: usize },
    /// A learned direct weight is nonpositive or nonfinite.
    InvalidWeight { coordinate: usize, weight: f64 },
    /// Transition coordinates must be unique and in stable lexicographic order.
    InvalidTransitionOrder { coordinate: usize },
    /// One fit option violates the active v1 contract.
    InvalidOption(&'static str),
}

impl Display for ModelError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for ModelError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_keeps_weights_bound_to_transition_coordinates() {
        let model = TransitionWeightModel::new(
            TopologyId::new("sha256:fixture").unwrap(),
            vec![Transition {
                previous: EdgeId::new(4),
                next: EdgeId::new(9),
            }],
            vec![12.5],
        )
        .unwrap();
        assert_eq!(model.transitions()[0].previous, EdgeId::new(4));
        assert_eq!(model.weights(), &[12.5]);
    }

    #[test]
    fn model_rejects_duplicate_or_unsorted_coordinates() {
        for transitions in [
            vec![
                Transition {
                    previous: EdgeId::new(1),
                    next: EdgeId::new(2),
                },
                Transition {
                    previous: EdgeId::new(1),
                    next: EdgeId::new(2),
                },
            ],
            vec![
                Transition {
                    previous: EdgeId::new(2),
                    next: EdgeId::new(0),
                },
                Transition {
                    previous: EdgeId::new(1),
                    next: EdgeId::new(9),
                },
            ],
        ] {
            assert!(matches!(
                TransitionWeightModel::new(
                    TopologyId::new("line-graph-v1:fixture").unwrap(),
                    transitions,
                    vec![1.0, 2.0],
                ),
                Err(ModelError::InvalidTransitionOrder { coordinate: 1 })
            ));
        }
    }
}
