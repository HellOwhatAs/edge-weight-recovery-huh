use edge_weight_recovery::data::{
    GraphData, compute_observed_edge_counts, compute_observed_transition_counts, group_paths_by_od,
};
use edge_weight_recovery::model::{EdgeOnlyModel, ExpandedRoadModel};
use edge_weight_recovery::objective::{compute_expanded_regret, compute_regret};
use edge_weight_recovery::optimizer::ExpandedProjectedSubgradientOptimizer;
use edge_weight_recovery::oracle::{CchOracle, ExpandedCchOracle, ExpandedOracleStats};
use edge_weight_recovery::turn_graph::ExpandedTurnGraph;
use std::time::Duration;

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

fn nested_graph() -> GraphData {
    GraphData {
        tail: vec![0, 1, 0, 2],
        head: vec![1, 3, 2, 3],
        baseline_weights: vec![5, 5, 2, 2],
        x: vec![0.0, 1.0, 1.0, 2.0],
        y: vec![0.0, 0.0, 1.0, 0.0],
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
    let model = ExpandedRoadModel::new(edge_model, &expanded, 1.0).unwrap();
    let weights = model.metric(&expanded).unwrap();
    for (transition, _, next) in expanded.transitions() {
        assert_eq!(
            weights.transition_weights()[transition.index()],
            weights.edge_weights()[next]
        );
    }

    let original_oracle = CchOracle::build(&graph).unwrap();
    let original_metric = original_oracle.customize(weights.edge_weights()).unwrap();
    let expanded_oracle = ExpandedCchOracle::build(&graph, &expanded).unwrap();
    let expanded_metric = expanded_oracle
        .customize(weights.edge_weights(), weights.transition_weights())
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
            path_cost(weights.edge_weights(), observed)
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

    // Penalize only the two cross choices. Both observed transitions remain at
    // the nested edge-only value r=0; every residual is nonnegative.
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
    let model = ExpandedRoadModel::from_parameters(edge_model, &expanded, 1.0, &residuals).unwrap();
    let weights = model.metric(&expanded).unwrap();
    assert_eq!(
        weights.transition_weights()[a_cross.index()],
        weights.edge_weights()[LOWER] + 2
    );
    assert_eq!(
        weights.transition_weights()[b_cross.index()],
        weights.edge_weights()[UPPER] + 2
    );

    let oracle = ExpandedCchOracle::build(&graph, &expanded).unwrap();
    let metric = oracle
        .customize(weights.edge_weights(), weights.transition_weights())
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
fn an_edge_only_state_is_exactly_nested_in_the_expanded_model() {
    let graph = nested_graph();
    let expanded = ExpandedTurnGraph::build(&graph).unwrap();
    let q = [0.8, 1.2, 1.0, 1.0];
    let residuals = vec![0.0; expanded.transition_count()];
    let lambda_edge = 2.0;
    let lambda_transition = 3.0;

    let edge_model = EdgeOnlyModel::from_q(&graph.baseline_weights, 1.0, &q).unwrap();
    let edge_weights = edge_model.quantized_weights().unwrap();
    let edge_regularization = edge_model.regularization(lambda_edge);
    let model =
        ExpandedRoadModel::from_parameters(edge_model.clone(), &expanded, 4.0, &residuals).unwrap();
    assert_eq!(model.q(), q);
    assert_eq!(model.transition_residuals(), residuals);
    let weights = model.metric(&expanded).unwrap();
    assert_eq!(weights.edge_weights(), edge_weights);
    for (transition, _, next) in expanded.transitions() {
        assert_eq!(
            weights.transition_weights()[transition.index()],
            edge_weights[next]
        );
    }

    let paths = vec![((0, 3), vec![0, 1]), ((0, 3), vec![2, 3])];
    let observed_edges = compute_observed_edge_counts(&paths, graph.tail.len(), 1);
    let observed_transitions = compute_observed_transition_counts(&paths, &expanded, 1).unwrap();
    let groups = group_paths_by_od(&paths);

    let original_oracle = CchOracle::build(&graph).unwrap();
    let original_metric = original_oracle.customize(&edge_weights).unwrap();
    let original_stats = original_oracle
        .batch_stats(&original_metric, &groups, 1)
        .unwrap();
    let edge_regret = compute_regret(&edge_weights, &observed_edges, &original_stats).unwrap();

    let expanded_oracle = ExpandedCchOracle::build(&graph, &expanded).unwrap();
    let expanded_metric = expanded_oracle
        .customize(weights.edge_weights(), weights.transition_weights())
        .unwrap();
    let expanded_stats = expanded_metric.batch_stats(&groups, 1).unwrap();
    let expanded_regret = compute_expanded_regret(
        &expanded,
        expanded_metric.edge_weights(),
        expanded_metric.transition_weights(),
        &observed_edges,
        &observed_transitions,
        &expanded_stats,
    )
    .unwrap();

    assert_eq!(edge_regret, expanded_regret);
    assert_eq!(
        edge_regularization.to_bits(),
        model.edge_regularization(lambda_edge).to_bits()
    );
    assert_eq!(model.transition_regularization(lambda_transition), 0.0);
    let edge_objective = edge_regret.mean_data_loss + edge_regularization;
    let expanded_objective =
        expanded_regret.mean_data_loss + model.regularization(lambda_edge, lambda_transition);
    assert_eq!(edge_objective.to_bits(), expanded_objective.to_bits());

    for path in paths.iter().map(|(_, path)| path) {
        assert_eq!(
            expanded_metric.observed_path_cost(path).unwrap(),
            path_cost(&edge_weights, path)
        );
    }
    assert_eq!(
        original_stats.weighted_shortest_distance_sum,
        expanded_stats.weighted_shortest_distance_sum
    );
}

#[test]
fn one_model_step_updates_both_parameter_blocks_on_one_clock() {
    let graph = conflict_graph();
    let expanded = ExpandedTurnGraph::build(&graph).unwrap();
    let edge_model = EdgeOnlyModel::new(&graph.baseline_weights, 1.0).unwrap();
    let mut model = ExpandedRoadModel::new(edge_model, &expanded, 1.0).unwrap();
    let mut optimizer =
        ExpandedProjectedSubgradientOptimizer::new(0.5, 0.0, 0.0, 0.1, 10.0, 10.0).unwrap();

    let mut observed_edges = vec![0; graph.tail.len()];
    let mut predicted_edges = vec![0; graph.tail.len()];
    observed_edges[IN_A] = 1;
    predicted_edges[IN_B] = 1;
    let mut observed_transitions = vec![0; expanded.transition_count()];
    let mut predicted_transitions = vec![0; expanded.transition_count()];
    observed_transitions[1] = 1;
    predicted_transitions[0] = 1;

    let oracle = ExpandedOracleStats {
        predicted_edge_counts: predicted_edges,
        predicted_transition_counts: predicted_transitions,
        weighted_shortest_distance_sum: 0,
        sample_count: 1,
        num_queries: 1,
        oracle_duration: Duration::ZERO,
    };
    let stats = model
        .projected_step(
            &mut optimizer,
            &observed_edges,
            &observed_transitions,
            &oracle,
        )
        .unwrap();

    assert_eq!(stats.eta, 0.5);
    assert_eq!(model.q()[IN_A], 0.5);
    assert_eq!(model.q()[IN_B], 1.5);
    assert_eq!(model.transition_residuals()[0], 0.5);
    assert_eq!(model.transition_residuals()[1], 0.0);
    assert_eq!(optimizer.completed_updates(), 1);
}
