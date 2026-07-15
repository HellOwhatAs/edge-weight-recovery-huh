use edge_weight_recovery::config::TurnExperimentArm;
use edge_weight_recovery::data::{
    GraphData, compute_observed_edge_counts, compute_observed_transition_counts, group_paths_by_od,
};
use edge_weight_recovery::model::{EdgeOnlyModel, TurnAwareModel};
use edge_weight_recovery::objective::compute_turn_aware_regret;
use edge_weight_recovery::oracle::{CchOracle, ExpandedCchOracle};
use edge_weight_recovery::turn_graph::ExpandedTurnGraph;

const IN_A: usize = 0;
const IN_B: usize = 1;
const UPPER: usize = 2;
const UPPER_TO_TARGET: usize = 3;
const LOWER: usize = 4;
const LOWER_TO_TARGET: usize = 5;

const OBSERVED_A: &[usize] = &[IN_A, UPPER, UPPER_TO_TARGET];
const ALTERNATIVE_A: &[usize] = &[IN_A, LOWER, LOWER_TO_TARGET];
const OBSERVED_B: &[usize] = &[IN_B, LOWER, LOWER_TO_TARGET];
const ALTERNATIVE_B: &[usize] = &[IN_B, UPPER, UPPER_TO_TARGET];

/// Two sources enter the same junction through different incoming edges. From
/// there, both ODs have the same upper/lower alternatives to one target, but
/// their observed choices are opposed. Edge-only costs therefore impose one
/// common branch ordering, whereas transition residuals may condition that
/// ordering on the incoming edge.
fn conflict_graph() -> GraphData {
    GraphData {
        tail: vec![0, 1, 2, 3, 2, 4],
        head: vec![2, 2, 3, 5, 4, 5],
        baseline_weights: vec![1; 6],
        x: vec![-1.0, -1.0, 0.0, 1.0, 1.0, 2.0],
        y: vec![1.0, -1.0, 0.0, 1.0, -1.0, 0.0],
    }
}

fn path_cost(weights: &[u32], path: &[usize]) -> u64 {
    path.iter().map(|&edge| weights[edge] as u64).sum()
}

fn assert_continuous_od(
    graph: &GraphData,
    path: &[usize],
    expected_source: u32,
    expected_target: u32,
) {
    let first = *path.first().expect("correctness path must be nonempty");
    let last = *path.last().expect("correctness path must be nonempty");
    assert_eq!(graph.tail[first], expected_source);
    assert_eq!(graph.head[last], expected_target);
    assert!(
        path.windows(2)
            .all(|pair| graph.head[pair[0]] == graph.tail[pair[1]])
    );
}

#[test]
fn edge_only_cannot_make_both_opposed_observations_strictly_shortest() {
    // The incoming-edge cost cancels within each OD. If U and L are the two
    // branch costs, OD A requires U < L while OD B requires L < U. The loop
    // exercises all positive component costs in a small box and also checks
    // the exact algebraic identity: the two observed-minus-alternative margins
    // are negatives of one another.
    for upper in 1..=3 {
        for upper_to_target in 1..=3 {
            for lower in 1..=3 {
                for lower_to_target in 1..=3 {
                    let weights = [7, 11, upper, upper_to_target, lower, lower_to_target];
                    let margin_a = path_cost(&weights, OBSERVED_A) as i64
                        - path_cost(&weights, ALTERNATIVE_A) as i64;
                    let margin_b = path_cost(&weights, OBSERVED_B) as i64
                        - path_cost(&weights, ALTERNATIVE_B) as i64;

                    assert_eq!(margin_a, -margin_b);
                    assert!(
                        !(margin_a < 0 && margin_b < 0),
                        "edge-only costs cannot satisfy both strict inequalities"
                    );
                }
            }
        }
    }
}

#[test]
fn stable_transition_mapping_and_zero_residual_contract_hold() {
    let graph = conflict_graph();
    let expanded = ExpandedTurnGraph::build(&graph).unwrap();
    let expected_transitions = [
        (0, IN_A, UPPER),
        (1, IN_A, LOWER),
        (2, IN_B, UPPER),
        (3, IN_B, LOWER),
        (4, UPPER, UPPER_TO_TARGET),
        (5, LOWER, LOWER_TO_TARGET),
    ];
    assert_eq!(expanded.transition_count(), expected_transitions.len());
    assert_eq!(expanded.stats.state_self_transitions, 0);
    assert_eq!(
        expanded
            .transitions()
            .map(|(id, previous, next)| (id.index(), previous, next))
            .collect::<Vec<_>>(),
        expected_transitions
    );
    for &(index, previous, next) in &expected_transitions {
        let transition = expanded.transition_id(previous, next).unwrap();
        assert_eq!(transition.index(), index);
        assert_eq!(
            expanded.transition_edges(transition),
            Some((previous, next))
        );
    }

    let edge_model = EdgeOnlyModel::new(&graph.baseline_weights, 1.0).unwrap();
    let turn_model = TurnAwareModel::new(edge_model, &expanded, 1.0).unwrap();
    let edge_weights = turn_model.quantized_edge_weights().unwrap();
    let transition_weights = turn_model.quantized_transition_weights(&expanded).unwrap();
    for (transition, _, next) in expanded.transitions() {
        assert_eq!(transition_weights[transition.index()], edge_weights[next]);
    }

    let original_oracle = CchOracle::build(&graph).unwrap();
    let original_metric = original_oracle.customize(&edge_weights).unwrap();
    let expanded_oracle = ExpandedCchOracle::build(&graph, &expanded).unwrap();
    let expanded_metric = expanded_oracle
        .customize(&edge_weights, &transition_weights)
        .unwrap();
    assert_eq!(
        expanded_metric.topology_identity(),
        expanded_oracle.topology_identity()
    );

    for (source, target, observed) in [(0, 5, OBSERVED_A), (1, 5, OBSERVED_B)] {
        let original = original_oracle
            .shortest_path(&original_metric, source, target)
            .unwrap();
        let expanded_path = expanded_metric.query(source, target).unwrap();
        assert_eq!(original.distance, 3);
        assert_eq!(expanded_path.distance, original.distance);
        assert_eq!(
            expanded_metric.observed_path_cost(observed).unwrap(),
            path_cost(&edge_weights, observed)
        );
        assert_continuous_od(&graph, observed, source, target);
        assert_continuous_od(&graph, &expanded_path.original_edges, source, target);
    }
}

#[test]
fn nonnegative_transition_residuals_make_both_observations_uniquely_shortest() {
    let graph = conflict_graph();
    let expanded = ExpandedTurnGraph::build(&graph).unwrap();
    let mut residuals = vec![0.0; expanded.transition_count()];

    // Penalize only the two cross choices. Both observed turns remain at the
    // nested edge-only value r=0; every residual is nonnegative.
    let a_cross = expanded.transition_id(IN_A, LOWER).unwrap();
    let b_cross = expanded.transition_id(IN_B, UPPER).unwrap();
    residuals[a_cross.index()] = 2.0;
    residuals[b_cross.index()] = 2.0;
    assert!(residuals.iter().all(|&residual| residual >= 0.0));
    assert_eq!(
        residuals[expanded.transition_id(IN_A, UPPER).unwrap().index()],
        0.0
    );
    assert_eq!(
        residuals[expanded.transition_id(IN_B, LOWER).unwrap().index()],
        0.0
    );

    let edge_model = EdgeOnlyModel::new(&graph.baseline_weights, 1.0).unwrap();
    let turn_model =
        TurnAwareModel::from_residuals(edge_model, &expanded, 1.0, &residuals).unwrap();
    let edge_weights = turn_model.quantized_edge_weights().unwrap();
    let transition_weights = turn_model.quantized_transition_weights(&expanded).unwrap();
    assert_eq!(transition_weights[a_cross.index()], edge_weights[LOWER] + 2);
    assert_eq!(transition_weights[b_cross.index()], edge_weights[UPPER] + 2);

    let oracle = ExpandedCchOracle::build(&graph, &expanded).unwrap();
    let metric = oracle
        .customize(&edge_weights, &transition_weights)
        .unwrap();
    for (source, target, observed, alternative) in [
        (0, 5, OBSERVED_A, ALTERNATIVE_A),
        (1, 5, OBSERVED_B, ALTERNATIVE_B),
    ] {
        let observed_cost = metric.observed_path_cost(observed).unwrap();
        let alternative_cost = metric.observed_path_cost(alternative).unwrap();
        assert_eq!(observed_cost, 3);
        assert_eq!(alternative_cost, 5);
        assert!(observed_cost < alternative_cost);

        // The acyclic conflict graph has exactly these two source-target
        // routes, so the strict cost inequality establishes uniqueness.
        let predicted = metric.query(source, target).unwrap();
        assert_eq!(predicted.distance as u64, observed_cost);
        assert_eq!(predicted.original_edges, observed);
        assert_continuous_od(&graph, &predicted.original_edges, source, target);
    }
}

#[test]
fn turn_only_state_is_a_joint_feasible_candidate_with_the_same_objective() {
    let graph = GraphData {
        tail: vec![0, 1, 0, 2],
        head: vec![1, 3, 2, 3],
        baseline_weights: vec![5, 5, 2, 2],
        x: vec![0.0, 1.0, 1.0, 2.0],
        y: vec![0.0, 0.0, 1.0, 0.0],
    };
    let expanded = ExpandedTurnGraph::build(&graph).unwrap();
    let q_star = [0.8, 1.2, 1.0, 1.0];
    let residuals = [0.25, 0.0];
    let q_min = 0.1;
    let q_max = 10.0;
    let r_max = 10.0;
    let residual_scale = 4.0;

    assert!(q_star.iter().all(|&q| q >= q_min && q <= q_max));
    assert!(residuals.iter().all(|&r| r >= 0.0 && r <= r_max));
    assert!(!TurnExperimentArm::TurnOnly.updates_q());
    assert!(TurnExperimentArm::TurnOnly.updates_residuals());
    assert!(TurnExperimentArm::JointEdgeTurn.updates_q());
    assert!(TurnExperimentArm::JointEdgeTurn.updates_residuals());

    // Joint learning permits q to move but does not require it to move. Thus a
    // turn-only state with q=q* is also a joint-feasible candidate. Construct
    // the same candidate twice to check that arm labels do not alter its model
    // state, integer metric, or objective.
    let turn_only = TurnAwareModel::from_residuals(
        EdgeOnlyModel::from_q(&graph.baseline_weights, 1.0, &q_star).unwrap(),
        &expanded,
        residual_scale,
        &residuals,
    )
    .unwrap();
    let joint_candidate = TurnAwareModel::from_residuals(
        EdgeOnlyModel::from_q(&graph.baseline_weights, 1.0, &q_star).unwrap(),
        &expanded,
        residual_scale,
        &residuals,
    )
    .unwrap();

    assert_eq!(turn_only.edge_only().q(), joint_candidate.edge_only().q());
    assert_eq!(
        turn_only.transition_residuals(),
        joint_candidate.transition_residuals()
    );
    let turn_only_edges = turn_only.quantized_edge_weights().unwrap();
    let joint_edges = joint_candidate.quantized_edge_weights().unwrap();
    let turn_only_transitions = turn_only.quantized_transition_weights(&expanded).unwrap();
    let joint_transitions = joint_candidate
        .quantized_transition_weights(&expanded)
        .unwrap();
    assert_eq!(turn_only_edges, vec![4, 6, 2, 2]);
    assert_eq!(turn_only_transitions, vec![7, 2]);
    assert_eq!(turn_only_edges, joint_edges);
    assert_eq!(turn_only_transitions, joint_transitions);

    let paths = vec![((0, 3), vec![0, 1]), ((0, 3), vec![2, 3])];
    let observed_edges = compute_observed_edge_counts(&paths, graph.tail.len(), 1);
    let observed_transitions = compute_observed_transition_counts(&paths, &expanded, 1).unwrap();
    let groups = group_paths_by_od(&paths);
    let oracle = ExpandedCchOracle::build(&graph, &expanded).unwrap();
    let turn_only_metric = oracle
        .customize(&turn_only_edges, &turn_only_transitions)
        .unwrap();
    let joint_metric = oracle.customize(&joint_edges, &joint_transitions).unwrap();
    let turn_only_oracle = turn_only_metric.batch_stats(&groups, 1).unwrap();
    let joint_oracle = joint_metric.batch_stats(&groups, 1).unwrap();
    assert_eq!(
        turn_only_oracle.predicted_edge_counts,
        joint_oracle.predicted_edge_counts
    );
    assert_eq!(
        turn_only_oracle.predicted_transition_counts,
        joint_oracle.predicted_transition_counts
    );
    assert_eq!(
        turn_only_oracle.weighted_shortest_distance_sum,
        joint_oracle.weighted_shortest_distance_sum
    );
    assert_eq!(turn_only_oracle.sample_count, joint_oracle.sample_count);

    let turn_only_regret = compute_turn_aware_regret(
        &expanded,
        turn_only_metric.edge_weights(),
        turn_only_metric.transition_weights(),
        &observed_edges,
        &observed_transitions,
        &turn_only_oracle,
    )
    .unwrap();
    let joint_regret = compute_turn_aware_regret(
        &expanded,
        joint_metric.edge_weights(),
        joint_metric.transition_weights(),
        &observed_edges,
        &observed_transitions,
        &joint_oracle,
    )
    .unwrap();
    assert_eq!(turn_only_regret.data_loss_sum, 7);
    assert_eq!(turn_only_regret.mean_data_loss, 3.5);
    assert_eq!(turn_only_regret, joint_regret);

    let lambda_edge = 2.0;
    let lambda_turn = 3.0;
    let turn_only_edge_regularization = turn_only.edge_only().regularization(lambda_edge);
    let joint_edge_regularization = joint_candidate.edge_only().regularization(lambda_edge);
    let turn_only_residual_regularization = turn_only.residual_regularization(lambda_turn);
    let joint_residual_regularization = joint_candidate.residual_regularization(lambda_turn);
    assert_eq!(
        turn_only_edge_regularization.to_bits(),
        joint_edge_regularization.to_bits()
    );
    assert_eq!(
        turn_only_residual_regularization.to_bits(),
        joint_residual_regularization.to_bits()
    );
    let turn_only_objective = turn_only_regret.mean_data_loss
        + turn_only_edge_regularization
        + turn_only_residual_regularization;
    let joint_objective =
        joint_regret.mean_data_loss + joint_edge_regularization + joint_residual_regularization;
    assert_eq!(turn_only_objective.to_bits(), joint_objective.to_bits());

    // This is a feasible-set and objective identity contract only. It makes no
    // claim that a finite number of simultaneous joint updates must dominate a
    // separately optimized turn-only run.
}
