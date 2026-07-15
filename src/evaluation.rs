use crate::graph::TripPath;
use rayon::prelude::*;
use routingkit_cch::{CCHMetric, CCHQuery};
use serde_json::{Value, json};
use std::collections::HashSet;

const EPSILON_THRESHOLDS: [f64; 4] = [0.0, 0.01, 0.05, 0.10];

/// Route-level held-out diagnostics for one observed trip.
///
/// Costs and regret stay as `u128` until summary statistics are converted to
/// `f64`. This keeps long-path accumulation exact even though the CCH metric
/// itself uses positive integer edge weights.
#[derive(Clone, Debug, PartialEq)]
pub struct RouteEvaluation {
    pub route_index: usize,
    pub source: u32,
    pub target: u32,
    pub observed_path_cost: u128,
    pub shortest_path_cost: u128,
    pub raw_regret: u128,
    pub relative_regret: f64,
    pub exact_match: bool,
    pub edge_precision: f64,
    pub edge_recall: f64,
    pub edge_f1: f64,
    pub edge_jaccard: f64,
    pub route_length: usize,
    pub contains_unseen_train_edge: bool,
}

impl RouteEvaluation {
    pub fn to_json(&self) -> Value {
        json!({
            "route_index": self.route_index,
            "source": self.source,
            "target": self.target,
            "observed_path_cost": self.observed_path_cost,
            "shortest_path_cost": self.shortest_path_cost,
            "raw_regret": self.raw_regret,
            "relative_regret": self.relative_regret,
            "exact_match": self.exact_match,
            "edge_precision": self.edge_precision,
            "edge_recall": self.edge_recall,
            "edge_f1": self.edge_f1,
            "edge_jaccard": self.edge_jaccard,
            "route_length": self.route_length,
            "contains_unseen_train_edge": self.contains_unseen_train_edge,
        })
    }
}

/// Fractions of observed routes whose regret is no more than the stated
/// fraction of their observed cost.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EpsilonOptimalRates {
    pub epsilon_0: f64,
    pub epsilon_1_percent: f64,
    pub epsilon_5_percent: f64,
    pub epsilon_10_percent: f64,
}

impl EpsilonOptimalRates {
    pub fn to_json(self) -> Value {
        json!({
            "0_percent": self.epsilon_0,
            "1_percent": self.epsilon_1_percent,
            "5_percent": self.epsilon_5_percent,
            "10_percent": self.epsilon_10_percent,
        })
    }
}

/// Aggregate route diagnostics. Empty groups contain zero-valued rates and an
/// undefined (`None`) Pearson correlation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EvaluationSummary {
    pub sample_count: usize,
    pub mean_raw_regret: f64,
    pub median_raw_regret: f64,
    pub p75_raw_regret: f64,
    pub p90_raw_regret: f64,
    pub p95_raw_regret: f64,
    pub mean_relative_regret: f64,
    /// Ratio of total regret to total observed cost. Unlike
    /// `mean_relative_regret`, this weights routes by their observed cost.
    pub aggregate_relative_regret: f64,
    pub median_relative_regret: f64,
    pub p90_relative_regret: f64,
    pub exact_match_rate: f64,
    pub mean_edge_precision: f64,
    pub mean_edge_recall: f64,
    pub mean_edge_f1: f64,
    pub mean_edge_jaccard: f64,
    pub epsilon_optimal_rates: EpsilonOptimalRates,
    pub zero_regret_but_nonexact_rate: f64,
    pub pearson_relative_regret_vs_one_minus_f1: Option<f64>,
}

impl EvaluationSummary {
    pub fn to_json(&self) -> Value {
        json!({
            "sample_count": self.sample_count,
            "raw_regret": {
                "mean": self.mean_raw_regret,
                "median": self.median_raw_regret,
                "p75": self.p75_raw_regret,
                "p90": self.p90_raw_regret,
                "p95": self.p95_raw_regret,
            },
            "relative_regret": {
                "mean": self.mean_relative_regret,
                "aggregate": self.aggregate_relative_regret,
                "median": self.median_relative_regret,
                "p90": self.p90_relative_regret,
            },
            "exact_match_rate": self.exact_match_rate,
            "mean_edge_precision": self.mean_edge_precision,
            "mean_edge_recall": self.mean_edge_recall,
            "mean_edge_f1": self.mean_edge_f1,
            "mean_edge_jaccard": self.mean_edge_jaccard,
            "epsilon_optimal_rates": self.epsilon_optimal_rates.to_json(),
            "zero_regret_but_nonexact_rate": self.zero_regret_but_nonexact_rate,
            "pearson_relative_regret_vs_one_minus_f1":
                self.pearson_relative_regret_vs_one_minus_f1,
        })
    }
}

/// Metrics for one route-length quartile. Empirical length cutoffs are used
/// instead of splitting tied lengths arbitrarily, so groups need not contain
/// exactly the same number of routes.
#[derive(Clone, Debug, PartialEq)]
pub struct LengthQuartileSummary {
    /// One-based quartile number in `1..=4`.
    pub quartile: u8,
    /// The empirical lower cutoff. It is exclusive for quartiles 2--4.
    pub lower_length_exclusive: Option<f64>,
    /// The empirical upper cutoff. It is inclusive for quartiles 1--3.
    pub upper_length_inclusive: Option<f64>,
    pub observed_min_length: Option<usize>,
    pub observed_max_length: Option<usize>,
    pub summary: EvaluationSummary,
}

impl LengthQuartileSummary {
    pub fn to_json(&self) -> Value {
        json!({
            "quartile": self.quartile,
            "lower_length_exclusive": self.lower_length_exclusive,
            "upper_length_inclusive": self.upper_length_inclusive,
            "observed_min_length": self.observed_min_length,
            "observed_max_length": self.observed_max_length,
            "summary": self.summary.to_json(),
        })
    }
}

/// Complete held-out result: individual routes, the overall distribution, and
/// the two requested diagnostic stratifications.
#[derive(Clone, Debug, PartialEq)]
pub struct DetailedEvaluation {
    pub routes: Vec<RouteEvaluation>,
    pub overall: EvaluationSummary,
    pub seen_only: EvaluationSummary,
    pub contains_unseen_train_edge: EvaluationSummary,
    pub route_length_quartile_cutoffs: [f64; 3],
    pub length_quartiles: Vec<LengthQuartileSummary>,
}

impl DetailedEvaluation {
    pub fn to_json(&self) -> Value {
        Value::Object(
            [
                (
                    "routes".to_string(),
                    Value::Array(RouteEvaluation::to_json_values(&self.routes)),
                ),
                ("overall".to_string(), self.overall.to_json()),
                ("seen_only".to_string(), self.seen_only.to_json()),
                (
                    "contains_unseen_train_edge".to_string(),
                    self.contains_unseen_train_edge.to_json(),
                ),
                (
                    "route_length_quartile_cutoffs".to_string(),
                    json!(self.route_length_quartile_cutoffs),
                ),
                (
                    "length_quartiles".to_string(),
                    Value::Array(
                        self.length_quartiles
                            .iter()
                            .map(LengthQuartileSummary::to_json)
                            .collect(),
                    ),
                ),
            ]
            .into_iter()
            .collect(),
        )
    }
}

impl RouteEvaluation {
    fn to_json_values(routes: &[Self]) -> Vec<Value> {
        routes.iter().map(Self::to_json).collect()
    }
}

/// Evaluate every held-out route against one customized CCH metric.
///
/// `train_observed_edge_counts` must use original edge IDs and normally comes
/// directly from `graph::compute_observed_edge_counts`. A held-out route is in
/// the unseen group if any of its observed edges has a zero training count.
pub fn evaluate_detailed_paths(
    metric: &CCHMetric<'_>,
    paths: &[TripPath],
    train_observed_edge_counts: &[u64],
    num_chunks: usize,
) -> Result<DetailedEvaluation, String> {
    if metric.weights().len() != train_observed_edge_counts.len() {
        return Err(format!(
            "metric weight and training edge-count lengths differ: {} != {}",
            metric.weights().len(),
            train_observed_edge_counts.len()
        ));
    }

    let chunk_size = paths.len().div_ceil(num_chunks.max(1)).max(1);
    let chunks: Vec<Result<Vec<RouteEvaluation>, String>> = paths
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, chunk)| {
            let mut query = CCHQuery::new(metric);
            chunk
                .iter()
                .enumerate()
                .map(|(index_in_chunk, trip)| {
                    evaluate_route(
                        &mut query,
                        metric,
                        trip,
                        train_observed_edge_counts,
                        chunk_index * chunk_size + index_in_chunk,
                    )
                })
                .collect()
        })
        .collect();

    let mut routes = Vec::with_capacity(paths.len());
    for chunk in chunks {
        routes.extend(chunk?);
    }

    build_detailed_evaluation(routes)
}

fn evaluate_route<'a>(
    query: &mut CCHQuery<'a>,
    metric: &'a CCHMetric<'a>,
    trip: &TripPath,
    train_observed_edge_counts: &[u64],
    route_index: usize,
) -> Result<RouteEvaluation, String> {
    let ((source, target), observed_path) = trip;
    query.add_source(*source, 0);
    query.add_target(*target, 0);
    let result = query.run();
    let distance = result.distance().ok_or_else(|| {
        format!("held-out route {route_index} OD ({source}, {target}) is unreachable")
    })?;
    let predicted_path: Vec<usize> = result
        .arc_path()
        .into_iter()
        .map(|edge| edge as usize)
        .collect();

    let reconstructed_shortest_cost = path_cost(
        &predicted_path,
        metric.weights(),
        route_index,
        "CCH-predicted",
    )?;
    if reconstructed_shortest_cost != distance as u128 {
        return Err(format!(
            "CCH path/distance mismatch for held-out route {route_index} OD ({source}, {target}): \
             path={reconstructed_shortest_cost}, distance={distance}"
        ));
    }

    let observed_path_cost = path_cost(observed_path, metric.weights(), route_index, "observed")?;
    let raw_regret = observed_path_cost
        .checked_sub(distance as u128)
        .ok_or_else(|| {
            format!(
                "negative held-out regret for route {route_index} OD ({source}, {target}): \
                 observed={observed_path_cost}, shortest={distance}"
            )
        })?;
    let relative_regret = if observed_path_cost == 0 {
        0.0
    } else {
        raw_regret as f64 / observed_path_cost as f64
    };

    let predicted_edges: HashSet<usize> = predicted_path.iter().copied().collect();
    let observed_edges: HashSet<usize> = observed_path.iter().copied().collect();
    let intersection = predicted_edges.intersection(&observed_edges).count() as f64;
    let edge_precision = intersection / predicted_edges.len().max(1) as f64;
    let edge_recall = intersection / observed_edges.len().max(1) as f64;
    let edge_f1 = if edge_precision + edge_recall == 0.0 {
        0.0
    } else {
        2.0 * edge_precision * edge_recall / (edge_precision + edge_recall)
    };
    let edge_jaccard = intersection / predicted_edges.union(&observed_edges).count().max(1) as f64;

    // Bounds were checked by `path_cost`, so indexing here is safe.
    let contains_unseen_train_edge = observed_path
        .iter()
        .any(|&edge| train_observed_edge_counts[edge] == 0);

    Ok(RouteEvaluation {
        route_index,
        source: *source,
        target: *target,
        observed_path_cost,
        shortest_path_cost: distance as u128,
        raw_regret,
        relative_regret,
        exact_match: predicted_path == *observed_path,
        edge_precision,
        edge_recall,
        edge_f1,
        edge_jaccard,
        route_length: observed_path.len(),
        contains_unseen_train_edge,
    })
}

fn path_cost(
    path: &[usize],
    weights: &[u32],
    route_index: usize,
    path_kind: &str,
) -> Result<u128, String> {
    path.iter().try_fold(0u128, |sum, &edge| {
        let weight = weights.get(edge).ok_or_else(|| {
            format!("held-out route {route_index} {path_kind} path edge {edge} is out of bounds")
        })?;
        sum.checked_add(*weight as u128)
            .ok_or_else(|| format!("held-out route {route_index} {path_kind} path cost overflow"))
    })
}

fn build_detailed_evaluation(routes: Vec<RouteEvaluation>) -> Result<DetailedEvaluation, String> {
    let overall_refs: Vec<&RouteEvaluation> = routes.iter().collect();
    let seen_refs: Vec<&RouteEvaluation> = routes
        .iter()
        .filter(|route| !route.contains_unseen_train_edge)
        .collect();
    let unseen_refs: Vec<&RouteEvaluation> = routes
        .iter()
        .filter(|route| route.contains_unseen_train_edge)
        .collect();

    let mut lengths: Vec<f64> = routes
        .iter()
        .map(|route| route.route_length as f64)
        .collect();
    lengths.sort_by(f64::total_cmp);
    let cutoffs = [
        percentile_f64(&lengths, 0.25),
        percentile_f64(&lengths, 0.50),
        percentile_f64(&lengths, 0.75),
    ];

    let mut quartile_routes: [Vec<&RouteEvaluation>; 4] = std::array::from_fn(|_| Vec::new());
    for route in &routes {
        let length = route.route_length as f64;
        let quartile = if length <= cutoffs[0] {
            0
        } else if length <= cutoffs[1] {
            1
        } else if length <= cutoffs[2] {
            2
        } else {
            3
        };
        quartile_routes[quartile].push(route);
    }

    let mut length_quartiles = Vec::with_capacity(4);
    for (quartile, members) in quartile_routes.iter().enumerate() {
        let observed_min_length = members.iter().map(|route| route.route_length).min();
        let observed_max_length = members.iter().map(|route| route.route_length).max();
        length_quartiles.push(LengthQuartileSummary {
            quartile: quartile as u8 + 1,
            lower_length_exclusive: (quartile > 0).then(|| cutoffs[quartile - 1]),
            upper_length_inclusive: (quartile < 3).then(|| cutoffs[quartile]),
            observed_min_length,
            observed_max_length,
            summary: summarize(members)?,
        });
    }

    Ok(DetailedEvaluation {
        overall: summarize(&overall_refs)?,
        seen_only: summarize(&seen_refs)?,
        contains_unseen_train_edge: summarize(&unseen_refs)?,
        routes,
        route_length_quartile_cutoffs: cutoffs,
        length_quartiles,
    })
}

fn summarize(routes: &[&RouteEvaluation]) -> Result<EvaluationSummary, String> {
    if routes.is_empty() {
        return Ok(EvaluationSummary::default());
    }

    let mut raw_regrets: Vec<u128> = routes.iter().map(|route| route.raw_regret).collect();
    raw_regrets.sort_unstable();
    let mut relative_regrets: Vec<f64> = routes.iter().map(|route| route.relative_regret).collect();
    relative_regrets.sort_by(f64::total_cmp);

    let raw_regret_sum = raw_regrets.iter().try_fold(0u128, |sum, &regret| {
        sum.checked_add(regret)
            .ok_or_else(|| "held-out raw-regret summary overflow".to_string())
    })?;
    let observed_cost_sum = routes.iter().try_fold(0u128, |sum, route| {
        sum.checked_add(route.observed_path_cost)
            .ok_or_else(|| "held-out observed-cost summary overflow".to_string())
    })?;
    let denominator = routes.len() as f64;
    let mean_relative_regret = relative_regrets.iter().sum::<f64>() / denominator;
    let exact_match_rate =
        routes.iter().filter(|route| route.exact_match).count() as f64 / denominator;
    let mean_edge_precision =
        routes.iter().map(|route| route.edge_precision).sum::<f64>() / denominator;
    let mean_edge_recall = routes.iter().map(|route| route.edge_recall).sum::<f64>() / denominator;
    let mean_edge_f1 = routes.iter().map(|route| route.edge_f1).sum::<f64>() / denominator;
    let mean_edge_jaccard =
        routes.iter().map(|route| route.edge_jaccard).sum::<f64>() / denominator;

    let mut epsilon_counts = [0usize; EPSILON_THRESHOLDS.len()];
    for route in routes {
        for (count, &epsilon) in epsilon_counts.iter_mut().zip(&EPSILON_THRESHOLDS) {
            if route.relative_regret <= epsilon {
                *count += 1;
            }
        }
    }
    let epsilon_optimal_rates = EpsilonOptimalRates {
        epsilon_0: epsilon_counts[0] as f64 / denominator,
        epsilon_1_percent: epsilon_counts[1] as f64 / denominator,
        epsilon_5_percent: epsilon_counts[2] as f64 / denominator,
        epsilon_10_percent: epsilon_counts[3] as f64 / denominator,
    };
    let zero_regret_but_nonexact_rate = routes
        .iter()
        .filter(|route| route.raw_regret == 0 && !route.exact_match)
        .count() as f64
        / denominator;

    Ok(EvaluationSummary {
        sample_count: routes.len(),
        mean_raw_regret: raw_regret_sum as f64 / denominator,
        median_raw_regret: percentile_u128(&raw_regrets, 0.50),
        p75_raw_regret: percentile_u128(&raw_regrets, 0.75),
        p90_raw_regret: percentile_u128(&raw_regrets, 0.90),
        p95_raw_regret: percentile_u128(&raw_regrets, 0.95),
        mean_relative_regret,
        aggregate_relative_regret: if observed_cost_sum == 0 {
            0.0
        } else {
            raw_regret_sum as f64 / observed_cost_sum as f64
        },
        median_relative_regret: percentile_f64(&relative_regrets, 0.50),
        p90_relative_regret: percentile_f64(&relative_regrets, 0.90),
        exact_match_rate,
        mean_edge_precision,
        mean_edge_recall,
        mean_edge_f1,
        mean_edge_jaccard,
        epsilon_optimal_rates,
        zero_regret_but_nonexact_rate,
        pearson_relative_regret_vs_one_minus_f1: pearson_relative_regret_vs_one_minus_f1(routes),
    })
}

/// Type-7/linear empirical quantile, matching the common default used by R,
/// NumPy, and many dataframe libraries.
fn percentile_u128(sorted: &[u128], quantile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = quantile * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let fraction = rank - lower as f64;
    sorted[lower] as f64 * (1.0 - fraction) + sorted[upper] as f64 * fraction
}

fn percentile_f64(sorted: &[f64], quantile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = quantile * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let fraction = rank - lower as f64;
    sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
}

fn pearson_relative_regret_vs_one_minus_f1(routes: &[&RouteEvaluation]) -> Option<f64> {
    if routes.len() < 2 {
        return None;
    }
    let denominator = routes.len() as f64;
    let mean_x = routes
        .iter()
        .map(|route| route.relative_regret)
        .sum::<f64>()
        / denominator;
    let mean_y = routes.iter().map(|route| 1.0 - route.edge_f1).sum::<f64>() / denominator;

    let (cross_product, squared_x, squared_y) = routes.iter().fold(
        (0.0, 0.0, 0.0),
        |(cross_product, squared_x, squared_y), route| {
            let centered_x = route.relative_regret - mean_x;
            let centered_y = (1.0 - route.edge_f1) - mean_y;
            (
                cross_product + centered_x * centered_y,
                squared_x + centered_x * centered_x,
                squared_y + centered_y * centered_y,
            )
        },
    );
    let scale = (squared_x * squared_y).sqrt();
    if scale == 0.0 {
        None
    } else {
        Some((cross_product / scale).clamp(-1.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use routingkit_cch::{CCH, compute_order_degree};

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn evaluates_route_distributions_and_seen_edge_groups() {
        // The observed upper path costs 10; the lower shortest path costs 4.
        let tail = vec![0, 1, 0, 2];
        let head = vec![1, 3, 2, 3];
        let weights = vec![5, 5, 2, 2];
        let order = compute_order_degree(4, &tail, &head);
        let cch = CCH::new(&order, &tail, &head, |_| {}, false);
        let metric = CCHMetric::new(&cch, weights);
        let paths = vec![((0, 3), vec![0, 1]), ((0, 3), vec![2, 3])];
        let train_counts = vec![1, 1, 0, 0];

        let evaluation = evaluate_detailed_paths(&metric, &paths, &train_counts, 16).unwrap();

        assert_eq!(evaluation.routes.len(), 2);
        assert_eq!(evaluation.routes[0].raw_regret, 6);
        assert_close(evaluation.routes[0].relative_regret, 0.6);
        assert!(!evaluation.routes[0].exact_match);
        assert_close(evaluation.routes[0].edge_f1, 0.0);
        assert!(!evaluation.routes[0].contains_unseen_train_edge);
        assert_eq!(evaluation.routes[1].raw_regret, 0);
        assert!(evaluation.routes[1].exact_match);
        assert_close(evaluation.routes[1].edge_f1, 1.0);
        assert!(evaluation.routes[1].contains_unseen_train_edge);

        assert_close(evaluation.overall.mean_raw_regret, 3.0);
        assert_close(evaluation.overall.median_raw_regret, 3.0);
        assert_close(evaluation.overall.p75_raw_regret, 4.5);
        assert_close(evaluation.overall.mean_relative_regret, 0.3);
        assert_close(evaluation.overall.aggregate_relative_regret, 6.0 / 14.0);
        assert_close(evaluation.overall.epsilon_optimal_rates.epsilon_0, 0.5);
        assert_close(
            evaluation
                .overall
                .pearson_relative_regret_vs_one_minus_f1
                .unwrap(),
            1.0,
        );
        assert_eq!(evaluation.seen_only.sample_count, 1);
        assert_close(evaluation.seen_only.mean_raw_regret, 6.0);
        assert_eq!(evaluation.contains_unseen_train_edge.sample_count, 1);
        assert_close(evaluation.contains_unseen_train_edge.mean_raw_regret, 0.0);
        assert_eq!(evaluation.length_quartiles[0].summary.sample_count, 2);
        assert_eq!(
            evaluation
                .length_quartiles
                .iter()
                .skip(1)
                .map(|quartile| quartile.summary.sample_count)
                .sum::<usize>(),
            0
        );

        let serialized = evaluation.to_json();
        assert_eq!(serialized["overall"]["sample_count"], 2);
        assert_eq!(serialized["routes"][0]["raw_regret"], 6);
    }

    #[test]
    fn counts_optimal_alternative_as_zero_regret_but_nonexact() {
        let tail = vec![0, 1, 0, 2];
        let head = vec![1, 3, 2, 3];
        let weights = vec![1, 1, 1, 1];
        let order = compute_order_degree(4, &tail, &head);
        let cch = CCH::new(&order, &tail, &head, |_| {}, false);
        let metric = CCHMetric::new(&cch, weights);

        let predicted = {
            let mut query = CCHQuery::new(&metric);
            query.add_source(0, 0);
            query.add_target(3, 0);
            query
                .run()
                .arc_path()
                .into_iter()
                .map(|edge| edge as usize)
                .collect::<Vec<_>>()
        };
        let alternative = if predicted == [0, 1] {
            vec![2, 3]
        } else {
            vec![0, 1]
        };
        let evaluation =
            evaluate_detailed_paths(&metric, &[((0, 3), alternative)], &[1, 1, 1, 1], 1).unwrap();

        assert_eq!(evaluation.routes[0].raw_regret, 0);
        assert!(!evaluation.routes[0].exact_match);
        assert_close(evaluation.overall.zero_regret_but_nonexact_rate, 1.0);
        assert_close(evaluation.overall.epsilon_optimal_rates.epsilon_0, 1.0);
    }

    #[test]
    fn reports_unreachable_held_out_od() {
        let tail = vec![0, 1];
        let head = vec![1, 2];
        let order = compute_order_degree(4, &tail, &head);
        let cch = CCH::new(&order, &tail, &head, |_| {}, false);
        let metric = CCHMetric::new(&cch, vec![1, 1]);

        let error =
            evaluate_detailed_paths(&metric, &[((0, 3), vec![0, 1])], &[1, 1], 1).unwrap_err();
        assert!(error.contains("unreachable"));
    }
}
