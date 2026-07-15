use edge_weight_recovery::graph::load_graph;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use routingkit_cch::{CCH, CCHMetric, CCHMetricPartialUpdater, compute_order_inertial};
use serde_json::json;
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
    let (city, output, repeats) = parse_args()?;
    let graph = load_graph(&city)?;
    let order = compute_order_inertial(
        graph.x.len() as u32,
        &graph.tail,
        &graph.head,
        &graph.x,
        &graph.y,
    );
    let cch = CCH::new(&order, &graph.tail, &graph.head, |_| {}, false);
    let mut indices: Vec<usize> = (0..graph.tail.len()).collect();
    indices.shuffle(&mut StdRng::seed_from_u64(42));

    let mut results = Vec::new();
    for requested_ratio in [0.01, 0.05, 0.10] {
        let changed_count = ((graph.tail.len() as f64 * requested_ratio).round() as usize).max(1);
        let changed = &indices[..changed_count];
        let mut modified = graph.baseline_weights.clone();
        let mut updates = BTreeMap::new();
        for &edge in changed {
            let increment = (modified[edge] / 100).max(1);
            modified[edge] = modified[edge].saturating_add(increment);
            updates.insert(edge as u32, modified[edge]);
        }

        let mut full_times = Vec::with_capacity(repeats);
        let mut partial_times = Vec::with_capacity(repeats);
        let mut updater = CCHMetricPartialUpdater::new(&cch);
        for repeat in 0..repeats {
            if repeat % 2 == 0 {
                full_times.push(time_full(&cch, &modified));
                partial_times.push(time_partial(
                    &cch,
                    &graph.baseline_weights,
                    &updates,
                    &mut updater,
                    &modified,
                ));
            } else {
                partial_times.push(time_partial(
                    &cch,
                    &graph.baseline_weights,
                    &updates,
                    &mut updater,
                    &modified,
                ));
                full_times.push(time_full(&cch, &modified));
            }
        }
        let full_median = median_ms(&full_times);
        let partial_median = median_ms(&partial_times);
        println!(
            "CUSTOMIZATION changed_pct={:.3} full_median_ms={full_median:.3} partial_median_ms={partial_median:.3} partial_over_full={:.3}",
            100.0 * changed_count as f64 / graph.tail.len() as f64,
            partial_median / full_median,
        );
        results.push(json!({
            "requested_changed_ratio": requested_ratio,
            "changed_edges": changed_count,
            "changed_ratio": changed_count as f64 / graph.tail.len() as f64,
            "full_ms": durations_ms(&full_times),
            "partial_ms": durations_ms(&partial_times),
            "full_median_ms": full_median,
            "partial_median_ms": partial_median,
            "partial_over_full": partial_median / full_median,
        }));
    }
    let result = json!({
        "schema_version": 1,
        "city": city,
        "graph_edges": graph.tail.len(),
        "rayon_threads": rayon::current_num_threads(),
        "repeats": repeats,
        "seed": 42,
        "edge_selection": "one fixed random permutation; nested prefixes at 1%, 5%, and 10%",
        "weight_change": "new weight = baseline + max(1, floor(baseline / 100))",
        "timing_policy": "median; fresh baseline metric prepared outside partial timer; alternating measurement order",
        "results": results,
    });
    if let Some(parent) = Path::new(&output).parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    std::fs::write(
        &output,
        serde_json::to_vec_pretty(&result)
            .map_err(|error| format!("failed to encode benchmark: {error}"))?,
    )
    .map_err(|error| format!("failed to write {output}: {error}"))?;
    println!("WROTE {output}");
    Ok(())
}

fn time_full(cch: &CCH, weights: &[u32]) -> Duration {
    let started = Instant::now();
    let metric = CCHMetric::new(cch, weights.to_vec());
    std::hint::black_box(metric.weights());
    started.elapsed()
}

fn time_partial<'a>(
    cch: &'a CCH,
    baseline: &[u32],
    updates: &BTreeMap<u32, u32>,
    updater: &mut CCHMetricPartialUpdater<'a>,
    expected: &[u32],
) -> Duration {
    let mut metric = CCHMetric::new(cch, baseline.to_vec());
    let started = Instant::now();
    updater.apply(&mut metric, updates);
    let elapsed = started.elapsed();
    assert_eq!(metric.weights(), expected);
    elapsed
}

fn durations_ms(values: &[Duration]) -> Vec<f64> {
    values
        .iter()
        .map(|duration| duration.as_secs_f64() * 1_000.0)
        .collect()
}

fn median_ms(values: &[Duration]) -> f64 {
    let mut values = durations_ms(values);
    values.sort_by(f64::total_cmp);
    if values.len().is_multiple_of(2) {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
    } else {
        values[values.len() / 2]
    }
}

fn parse_args() -> Result<(String, String, usize), String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut city = None;
    let mut output = None;
    let mut repeats = 9usize;
    let mut index = 0;
    while index < raw.len() {
        let flag = &raw[index];
        let value = raw
            .get(index + 1)
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--city" => city = Some(value.clone()),
            "--output" => output = Some(value.clone()),
            "--repeats" => {
                repeats = value
                    .parse()
                    .map_err(|error| format!("invalid --repeats: {error}"))?
            }
            _ => return Err(format!("unknown argument {flag}")),
        }
        index += 2;
    }
    if repeats == 0 {
        return Err("--repeats must be positive".to_string());
    }
    Ok((
        city.ok_or_else(|| "missing --city".to_string())?,
        output.ok_or_else(|| "missing --output".to_string())?,
        repeats,
    ))
}
