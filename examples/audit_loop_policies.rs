use edge_weight_recovery::graph::{GraphData, chronological_loop_erasure, load_graph};
use serde_json::{Value, json};
use serde_pickle::Value as PickleValue;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

type RawTrip = (PickleValue, Vec<usize>, (usize, usize));

const LOCAL_UTC_OFFSET_SECONDS: i64 = 8 * 60 * 60;

#[derive(Debug)]
struct Args {
    city: String,
    train_variant: String,
    output: PathBuf,
    max_samples: Option<usize>,
    progress_every: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InvalidPath {
    Empty,
    OutOfBounds,
    Discontinuous,
}

#[derive(Debug)]
struct ValidatedPath {
    nodes: Vec<u32>,
    loop_info: LoopInfo,
}

#[derive(Debug, Default)]
struct StructuralReport {
    available_records: usize,
    inspected_records: usize,
    valid_continuous_records: usize,
    empty_records: usize,
    out_of_bounds_records: usize,
    discontinuous_records: usize,
}

#[derive(Debug, Default)]
struct LoopInfo {
    closure_spans: Vec<usize>,
    distinct_repeated_nodes: usize,
    self_loop_edges: usize,
    repeats_start_node: bool,
    closed_trip: bool,
}

impl LoopInfo {
    fn is_cyclic(&self) -> bool {
        !self.closure_spans.is_empty()
    }
}

#[derive(Debug, Default)]
struct LoopAggregate {
    cyclic_records: usize,
    closed_trip_records: usize,
    repeats_start_records: usize,
    self_loop_records: usize,
    self_loop_edge_occurrences: usize,
    multiple_closure_records: usize,
    closure_events: usize,
    closure_span_edges: Vec<usize>,
    distinct_repeated_nodes_per_record: Vec<usize>,
    loop_erasure_removed_edges_per_record: Vec<usize>,
    loop_erasure_removed_fraction_per_record: Vec<f64>,
}

impl LoopAggregate {
    fn observe(&mut self, info: &LoopInfo, original_len: usize, erased_len: usize) {
        if !info.is_cyclic() {
            return;
        }
        self.cyclic_records += 1;
        self.closed_trip_records += usize::from(info.closed_trip);
        self.repeats_start_records += usize::from(info.repeats_start_node);
        self.self_loop_records += usize::from(info.self_loop_edges > 0);
        self.self_loop_edge_occurrences += info.self_loop_edges;
        self.multiple_closure_records += usize::from(info.closure_spans.len() > 1);
        self.closure_events += info.closure_spans.len();
        self.closure_span_edges.extend(&info.closure_spans);
        self.distinct_repeated_nodes_per_record
            .push(info.distinct_repeated_nodes);
        let removed = original_len - erased_len;
        self.loop_erasure_removed_edges_per_record.push(removed);
        self.loop_erasure_removed_fraction_per_record
            .push(ratio(removed, original_len));
    }

    fn to_json(&self, valid_records: usize, valid_edge_occurrences: usize) -> Value {
        let mut spans = self.closure_span_edges.clone();
        let mut repeated = self.distinct_repeated_nodes_per_record.clone();
        let mut removed = self.loop_erasure_removed_edges_per_record.clone();
        let mut removed_fraction = self.loop_erasure_removed_fraction_per_record.clone();
        spans.sort_unstable();
        repeated.sort_unstable();
        removed.sort_unstable();
        removed_fraction.sort_by(f64::total_cmp);
        let removed_total: usize = removed.iter().sum();
        let span_1 = spans.iter().filter(|&&span| span == 1).count();
        let span_2 = spans.iter().filter(|&&span| span == 2).count();
        let span_3_5 = spans
            .iter()
            .filter(|&&span| (3..=5).contains(&span))
            .count();
        let span_6_10 = spans
            .iter()
            .filter(|&&span| (6..=10).contains(&span))
            .count();
        let span_gt_10 = spans.iter().filter(|&&span| span > 10).count();

        json!({
            "cyclic_records": self.cyclic_records,
            "cyclic_record_rate_of_structurally_valid": ratio(self.cyclic_records, valid_records),
            "closed_trip_records": self.closed_trip_records,
            "repeats_start_node_records": self.repeats_start_records,
            "self_loop_records": self.self_loop_records,
            "self_loop_edge_occurrences": self.self_loop_edge_occurrences,
            "multiple_closure_records": self.multiple_closure_records,
            "closure_events": self.closure_events,
            "closure_span_definition": "edge distance since the preceding occurrence of the repeated node, using the unmodified original walk",
            "closure_span_buckets": {
                "self_loop_1": span_1,
                "backtrack_2": span_2,
                "short_3_to_5": span_3_5,
                "medium_6_to_10": span_6_10,
                "long_gt_10": span_gt_10,
            },
            "closure_span_edges": numeric_summary_usize(&spans),
            "distinct_repeated_nodes_per_cyclic_record": numeric_summary_usize(&repeated),
            "chronological_loop_erasure": {
                "removed_edge_occurrences": removed_total,
                "removed_rate_of_all_structurally_valid_edge_occurrences": ratio(removed_total, valid_edge_occurrences),
                "removed_edges_per_cyclic_record": numeric_summary_usize(&removed),
                "removed_fraction_per_cyclic_record": numeric_summary_f64(&removed_fraction),
            },
        })
    }
}

#[derive(Debug, Default)]
struct TemporalStats {
    records: usize,
    valid_duration_records: usize,
    end_before_start_records: usize,
    durations_seconds: Vec<usize>,
    local_hour_counts: [usize; 24],
    local_weekday_counts: [usize; 7],
    local_date_counts: BTreeMap<String, usize>,
}

impl TemporalStats {
    fn observe(&mut self, times: (usize, usize)) {
        self.records += 1;
        let (start, end) = times;
        if end >= start {
            self.valid_duration_records += 1;
            self.durations_seconds.push(end - start);
        } else {
            self.end_before_start_records += 1;
        }
        if let Some(local) = local_time_parts(start) {
            self.local_hour_counts[local.hour] += 1;
            self.local_weekday_counts[local.weekday] += 1;
            *self.local_date_counts.entry(local.date).or_default() += 1;
        }
    }

    fn to_json(&self) -> Value {
        let mut durations = self.durations_seconds.clone();
        durations.sort_unstable();
        json!({
            "source_records": self.records,
            "timestamp_interpretation": "Unix seconds, summarized in fixed UTC+08:00 (Asia/Shanghai has no DST during the observed period)",
            "valid_duration_records": self.valid_duration_records,
            "end_before_start_records": self.end_before_start_records,
            "duration_seconds": numeric_summary_usize(&durations),
            "start_local_hour_counts_00_to_23": self.local_hour_counts,
            "start_local_weekday_counts_monday_to_sunday": self.local_weekday_counts,
            "start_local_date_counts": self.local_date_counts,
        })
    }
}

#[derive(Debug)]
struct SourceStats {
    records: usize,
    edge_occurrences: usize,
    lengths: Vec<usize>,
    edge_seen: Vec<bool>,
    ods: HashSet<u64>,
    temporal: TemporalStats,
}

impl SourceStats {
    fn new(edge_count: usize) -> Self {
        Self {
            records: 0,
            edge_occurrences: 0,
            lengths: Vec::new(),
            edge_seen: vec![false; edge_count],
            ods: HashSet::new(),
            temporal: TemporalStats::default(),
        }
    }

    fn observe(&mut self, path: &[usize], nodes: &[u32], times: (usize, usize), graph: &GraphData) {
        self.records += 1;
        self.edge_occurrences += path.len();
        self.lengths.push(path.len());
        for &edge in path {
            self.edge_seen[edge] = true;
        }
        self.ods.insert(pack_od(nodes[0], *nodes.last().unwrap()));
        self.temporal.observe(times);
        debug_assert!(
            path.windows(2)
                .all(|pair| graph.head[pair[0]] == graph.tail[pair[1]])
        );
    }

    fn to_json(&self, graph_edge_count: usize) -> Value {
        let mut lengths = self.lengths.clone();
        lengths.sort_unstable();
        let observed_edges = self.edge_seen.iter().filter(|&&seen| seen).count();
        json!({
            "source_records": self.records,
            "edge_occurrences": self.edge_occurrences,
            "path_length_edges": numeric_summary_usize(&lengths),
            "unique_od": self.ods.len(),
            "unique_edges": observed_edges,
            "graph_edge_coverage": ratio(observed_edges, graph_edge_count),
            "time": self.temporal.to_json(),
        })
    }
}

#[derive(Debug)]
struct PolicyStats {
    name: &'static str,
    source_records_considered: usize,
    source_records_retained: usize,
    output_paths: usize,
    output_edge_occurrences: usize,
    source_edge_occurrences_considered: usize,
    policy_removed_edge_occurrences: usize,
    empty_after_transform_records: usize,
    lengths: Vec<usize>,
    edge_seen: Vec<bool>,
    ods: HashSet<u64>,
    temporal: TemporalStats,
}

impl PolicyStats {
    fn new(name: &'static str, edge_count: usize) -> Self {
        Self {
            name,
            source_records_considered: 0,
            source_records_retained: 0,
            output_paths: 0,
            output_edge_occurrences: 0,
            source_edge_occurrences_considered: 0,
            policy_removed_edge_occurrences: 0,
            empty_after_transform_records: 0,
            lengths: Vec::new(),
            edge_seen: vec![false; edge_count],
            ods: HashSet::new(),
            temporal: TemporalStats::default(),
        }
    }

    fn observe_record<'a, I>(
        &mut self,
        output_paths: I,
        source_len: usize,
        times: (usize, usize),
        graph: &GraphData,
    ) where
        I: IntoIterator<Item = &'a [usize]>,
    {
        self.source_records_considered += 1;
        self.source_edge_occurrences_considered += source_len;
        let mut retained_edges = 0;
        let mut retained_any = false;
        for path in output_paths.into_iter().filter(|path| !path.is_empty()) {
            retained_any = true;
            debug_assert!(is_simple_continuous_path(path, graph));
            self.output_paths += 1;
            self.output_edge_occurrences += path.len();
            retained_edges += path.len();
            self.lengths.push(path.len());
            for &edge in path {
                self.edge_seen[edge] = true;
            }
            self.ods.insert(pack_od(
                graph.tail[path[0]],
                graph.head[*path.last().unwrap()],
            ));
        }
        self.policy_removed_edge_occurrences += source_len - retained_edges;
        if retained_any {
            self.source_records_retained += 1;
            self.temporal.observe(times);
        } else {
            self.empty_after_transform_records += 1;
        }
    }

    fn to_json(&self, graph_edge_count: usize) -> Value {
        let mut lengths = self.lengths.clone();
        lengths.sort_unstable();
        let unique_edges = self.edge_seen.iter().filter(|&&seen| seen).count();
        json!({
            "policy": self.name,
            "source_records_considered": self.source_records_considered,
            "source_records_retained": self.source_records_retained,
            "source_record_retention_rate": ratio(self.source_records_retained, self.source_records_considered),
            "output_paths": self.output_paths,
            "output_paths_per_retained_source_record": ratio(self.output_paths, self.source_records_retained),
            "source_edge_occurrences_considered": self.source_edge_occurrences_considered,
            "output_edge_occurrences": self.output_edge_occurrences,
            "policy_removed_edge_occurrences": self.policy_removed_edge_occurrences,
            "policy_removed_rate_of_considered_source_edges": ratio(self.policy_removed_edge_occurrences, self.source_edge_occurrences_considered),
            "records_with_no_output_path": self.empty_after_transform_records,
            "output_path_length_edges": numeric_summary_usize(&lengths),
            "unique_od_after_policy": self.ods.len(),
            "unique_edges_after_policy": unique_edges,
            "graph_edge_coverage_after_policy": ratio(unique_edges, graph_edge_count),
            "retained_source_record_time": self.temporal.to_json(),
        })
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let started = Instant::now();
    let graph = load_graph(&args.city)?;
    let source_path = format!(
        "data/{}_data/preprocessed_train_trips_{}.pkl",
        args.city, args.train_variant
    );
    let load_started = Instant::now();
    let raw: Vec<RawTrip> = serde_pickle::from_reader(
        File::open(&source_path)
            .map_err(|error| format!("failed to open {source_path}: {error}"))?,
        Default::default(),
    )
    .map_err(|error| format!("failed to decode {source_path}: {error}"))?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    let inspect_count = args.max_samples.unwrap_or(raw.len()).min(raw.len());
    eprintln!(
        "loaded {} train records in {:.3}s; inspecting {}",
        raw.len(),
        load_seconds,
        inspect_count
    );

    let edge_count = graph.tail.len();
    let mut structural = StructuralReport {
        available_records: raw.len(),
        ..StructuralReport::default()
    };
    let mut all_valid = SourceStats::new(edge_count);
    let mut acyclic = SourceStats::new(edge_count);
    let mut cyclic = SourceStats::new(edge_count);
    let mut loops = LoopAggregate::default();
    let mut drop_policy = PolicyStats::new("drop_cyclic_original_records", edge_count);
    let mut erase_policy = PolicyStats::new("chronological_loop_erasure", edge_count);
    let mut split_policy = PolicyStats::new("greedy_split_at_repeated_nodes", edge_count);

    let audit_started = Instant::now();
    for (index, (_, path, times)) in raw.into_iter().take(inspect_count).enumerate() {
        structural.inspected_records += 1;
        let validated = match validate_path(&path, &graph) {
            Ok(validated) => validated,
            Err(InvalidPath::Empty) => {
                structural.empty_records += 1;
                continue;
            }
            Err(InvalidPath::OutOfBounds) => {
                structural.out_of_bounds_records += 1;
                continue;
            }
            Err(InvalidPath::Discontinuous) => {
                structural.discontinuous_records += 1;
                continue;
            }
        };
        structural.valid_continuous_records += 1;
        all_valid.observe(&path, &validated.nodes, times, &graph);

        let erased = chronological_loop_erasure(&path, &graph.tail, &graph.head)
            .expect("structurally validated path must support loop erasure");
        loops.observe(&validated.loop_info, path.len(), erased.len());
        if validated.loop_info.is_cyclic() {
            cyclic.observe(&path, &validated.nodes, times, &graph);
            drop_policy.observe_record(std::iter::empty::<&[usize]>(), path.len(), times, &graph);
        } else {
            acyclic.observe(&path, &validated.nodes, times, &graph);
            drop_policy.observe_record(std::iter::once(path.as_slice()), path.len(), times, &graph);
        }
        erase_policy.observe_record(
            std::iter::once(erased.as_slice()),
            path.len(),
            times,
            &graph,
        );
        let split = split_at_repeated_nodes(&path, &validated.nodes);
        split_policy.observe_record(split.iter().map(Vec::as_slice), path.len(), times, &graph);

        if args.progress_every > 0 && (index + 1) % args.progress_every == 0 {
            eprintln!(
                "processed {}/{} records in {:.3}s",
                index + 1,
                inspect_count,
                audit_started.elapsed().as_secs_f64()
            );
        }
    }
    let audit_seconds = audit_started.elapsed().as_secs_f64();

    let selection_bias = selection_bias_json(&acyclic, &cyclic);
    let policies = [
        drop_policy.to_json(edge_count),
        erase_policy.to_json(edge_count),
        split_policy.to_json(edge_count),
    ];
    let result = json!({
        "schema_version": 1,
        "audit": "train_loop_policy_comparison",
        "city": args.city,
        "train_variant": args.train_variant,
        "source_path": source_path,
        "test_data_read": false,
        "training_performed": false,
        "max_samples": args.max_samples,
        "graph": {
            "nodes": graph.x.len(),
            "directed_edges": edge_count,
        },
        "methodology": {
            "path_boundary_trimming": false,
            "drop": "retain an original record iff its continuous node walk has no repeated node",
            "chronological_loop_erasure": "scan in time order; when a node repeats, erase every edge after its previous retained occurrence through the edge that closes the loop",
            "greedy_split_at_repeated_nodes": "maintain a maximal simple contiguous segment; when the next node repeats in that segment, close the segment before the loop-closing edge and start a new segment with that edge; self-loop edges cannot belong to a nonempty simple path and are removed",
            "important_comparability_note": "split output-path counts and ODs are not source-record counts or original ODs; both are reported separately",
        },
        "structural_validation": {
            "available_records": structural.available_records,
            "inspected_records": structural.inspected_records,
            "valid_continuous_records": structural.valid_continuous_records,
            "empty_records": structural.empty_records,
            "out_of_bounds_records": structural.out_of_bounds_records,
            "discontinuous_records": structural.discontinuous_records,
        },
        "all_structurally_valid_originals": all_valid.to_json(edge_count),
        "loop_types": loops.to_json(structural.valid_continuous_records, all_valid.edge_occurrences),
        "selection_bias_if_cycles_are_dropped": selection_bias,
        "policies": policies,
        "runtime": {
            "pickle_load_seconds": load_seconds,
            "audit_seconds": audit_seconds,
            "total_seconds_before_write": started.elapsed().as_secs_f64(),
        },
    });
    atomic_write_json(&args.output, &result)?;
    println!(
        "LOOP_AUDIT inspected={} valid={} cyclic={} cycle_rate={:.6} elapsed_seconds={:.3}",
        structural.inspected_records,
        structural.valid_continuous_records,
        loops.cyclic_records,
        ratio(loops.cyclic_records, structural.valid_continuous_records),
        started.elapsed().as_secs_f64()
    );
    println!("WROTE {}", args.output.display());
    Ok(())
}

fn validate_path(path: &[usize], graph: &GraphData) -> Result<ValidatedPath, InvalidPath> {
    let Some(&first_edge) = path.first() else {
        return Err(InvalidPath::Empty);
    };
    if path
        .iter()
        .any(|&edge| edge >= graph.tail.len() || edge >= graph.head.len())
    {
        return Err(InvalidPath::OutOfBounds);
    }
    if path
        .windows(2)
        .any(|pair| graph.head[pair[0]] != graph.tail[pair[1]])
    {
        return Err(InvalidPath::Discontinuous);
    }

    let mut nodes = Vec::with_capacity(path.len() + 1);
    nodes.push(graph.tail[first_edge]);
    nodes.extend(path.iter().map(|&edge| graph.head[edge]));
    let loop_info = analyze_loops(&nodes);
    Ok(ValidatedPath { nodes, loop_info })
}

fn analyze_loops(nodes: &[u32]) -> LoopInfo {
    let mut last_position = HashMap::with_capacity(nodes.len());
    let mut repeated_nodes = HashSet::new();
    let mut closure_spans = Vec::new();
    let mut self_loop_edges = 0;
    for (position, &node) in nodes.iter().enumerate() {
        if let Some(previous) = last_position.insert(node, position) {
            let span = position - previous;
            closure_spans.push(span);
            repeated_nodes.insert(node);
            self_loop_edges += usize::from(span == 1);
        }
    }
    let first = nodes.first().copied();
    let last = nodes.last().copied();
    LoopInfo {
        closure_spans,
        distinct_repeated_nodes: repeated_nodes.len(),
        self_loop_edges,
        repeats_start_node: first
            .is_some_and(|start| nodes.iter().skip(1).any(|&node| node == start)),
        closed_trip: nodes.len() > 1 && first == last,
    }
}

fn split_at_repeated_nodes(path: &[usize], nodes: &[u32]) -> Vec<Vec<usize>> {
    debug_assert_eq!(nodes.len(), path.len() + 1);
    if path.is_empty() {
        return Vec::new();
    }
    let mut output = Vec::new();
    let mut current = Vec::new();
    let mut visited = HashSet::new();
    visited.insert(nodes[0]);

    for (index, &edge) in path.iter().enumerate() {
        let tail = nodes[index];
        let head = nodes[index + 1];
        if visited.contains(&head) {
            if !current.is_empty() {
                output.push(std::mem::take(&mut current));
            }
            visited.clear();
            visited.insert(tail);
            if head != tail {
                current.push(edge);
                visited.insert(head);
            }
        } else {
            current.push(edge);
            visited.insert(head);
        }
    }
    if !current.is_empty() {
        output.push(current);
    }
    output
}

fn is_simple_continuous_path(path: &[usize], graph: &GraphData) -> bool {
    if path.is_empty()
        || path
            .windows(2)
            .any(|pair| graph.head[pair[0]] != graph.tail[pair[1]])
    {
        return false;
    }
    let mut seen = HashSet::with_capacity(path.len() + 1);
    seen.insert(graph.tail[path[0]]);
    path.iter().all(|&edge| seen.insert(graph.head[edge]))
}

fn selection_bias_json(acyclic: &SourceStats, cyclic: &SourceStats) -> Value {
    let acyclic_edges = acyclic.edge_seen.iter().filter(|&&seen| seen).count();
    let cyclic_edges = cyclic.edge_seen.iter().filter(|&&seen| seen).count();
    let shared_edges = acyclic
        .edge_seen
        .iter()
        .zip(&cyclic.edge_seen)
        .filter(|(left, right)| **left && **right)
        .count();
    let acyclic_only_edges = acyclic
        .edge_seen
        .iter()
        .zip(&cyclic.edge_seen)
        .filter(|(left, right)| **left && !**right)
        .count();
    let cyclic_only_edges = acyclic
        .edge_seen
        .iter()
        .zip(&cyclic.edge_seen)
        .filter(|(left, right)| !**left && **right)
        .count();
    let shared_od = acyclic
        .ods
        .iter()
        .filter(|od| cyclic.ods.contains(od))
        .count();
    let mut hourly = Vec::with_capacity(24);
    for hour in 0..24 {
        let acyclic_count = acyclic.temporal.local_hour_counts[hour];
        let cyclic_count = cyclic.temporal.local_hour_counts[hour];
        hourly.push(json!({
            "local_hour": hour,
            "acyclic_records": acyclic_count,
            "cyclic_records": cyclic_count,
            "cyclic_rate": ratio(cyclic_count, acyclic_count + cyclic_count),
        }));
    }
    let all_dates: HashSet<&String> = acyclic
        .temporal
        .local_date_counts
        .keys()
        .chain(cyclic.temporal.local_date_counts.keys())
        .collect();
    let mut dates: Vec<&String> = all_dates.into_iter().collect();
    dates.sort_unstable();
    let daily: Vec<Value> = dates
        .into_iter()
        .map(|date| {
            let acyclic_count = acyclic
                .temporal
                .local_date_counts
                .get(date)
                .copied()
                .unwrap_or(0);
            let cyclic_count = cyclic
                .temporal
                .local_date_counts
                .get(date)
                .copied()
                .unwrap_or(0);
            json!({
                "local_date": date,
                "acyclic_records": acyclic_count,
                "cyclic_records": cyclic_count,
                "cyclic_rate": ratio(cyclic_count, acyclic_count + cyclic_count),
            })
        })
        .collect();

    let mut acyclic_lengths = acyclic.lengths.clone();
    let mut cyclic_lengths = cyclic.lengths.clone();
    let mut acyclic_durations = acyclic.temporal.durations_seconds.clone();
    let mut cyclic_durations = cyclic.temporal.durations_seconds.clone();
    acyclic_lengths.sort_unstable();
    cyclic_lengths.sort_unstable();
    acyclic_durations.sort_unstable();
    cyclic_durations.sort_unstable();
    let union_edges = shared_edges + acyclic_only_edges + cyclic_only_edges;
    let od_union = acyclic.ods.len() + cyclic.ods.len() - shared_od;
    json!({
        "interpretation": "differences below quantify the population selection induced by dropping every cyclic source record; they do not establish a causal effect",
        "acyclic_originals": {
            "records": acyclic.records,
            "edge_occurrences": acyclic.edge_occurrences,
            "path_length_edges": numeric_summary_usize(&acyclic_lengths),
            "duration_seconds": numeric_summary_usize(&acyclic_durations),
            "unique_edges": acyclic_edges,
            "unique_od": acyclic.ods.len(),
        },
        "cyclic_originals": {
            "records": cyclic.records,
            "edge_occurrences": cyclic.edge_occurrences,
            "path_length_edges": numeric_summary_usize(&cyclic_lengths),
            "duration_seconds": numeric_summary_usize(&cyclic_durations),
            "unique_edges": cyclic_edges,
            "unique_od": cyclic.ods.len(),
        },
        "mean_differences_cyclic_minus_acyclic": {
            "path_length_edges": mean_usize(&cyclic_lengths) - mean_usize(&acyclic_lengths),
            "duration_seconds": mean_usize(&cyclic_durations) - mean_usize(&acyclic_durations),
        },
        "edge_set_overlap": {
            "shared": shared_edges,
            "acyclic_only": acyclic_only_edges,
            "cyclic_only_lost_if_dropped": cyclic_only_edges,
            "union": union_edges,
            "jaccard": ratio(shared_edges, union_edges),
        },
        "original_od_set_overlap": {
            "shared": shared_od,
            "acyclic_only": acyclic.ods.len() - shared_od,
            "cyclic_only_lost_if_dropped": cyclic.ods.len() - shared_od,
            "union": od_union,
            "jaccard": ratio(shared_od, od_union),
        },
        "cyclic_rate_by_local_hour": hourly,
        "cyclic_rate_by_local_date": daily,
        "acyclic_time_distribution": acyclic.temporal.to_json(),
        "cyclic_time_distribution": cyclic.temporal.to_json(),
    })
}

fn numeric_summary_usize(sorted: &[usize]) -> Value {
    json!({
        "count": sorted.len(),
        "mean": mean_usize(sorted),
        "min": quantile_usize(sorted, 0.0),
        "p25": quantile_usize(sorted, 0.25),
        "median": quantile_usize(sorted, 0.5),
        "p75": quantile_usize(sorted, 0.75),
        "p90": quantile_usize(sorted, 0.90),
        "p95": quantile_usize(sorted, 0.95),
        "p99": quantile_usize(sorted, 0.99),
        "max": quantile_usize(sorted, 1.0),
    })
}

fn numeric_summary_f64(sorted: &[f64]) -> Value {
    let mean = if sorted.is_empty() {
        0.0
    } else {
        sorted.iter().sum::<f64>() / sorted.len() as f64
    };
    json!({
        "count": sorted.len(),
        "mean": mean,
        "min": quantile_f64(sorted, 0.0),
        "p25": quantile_f64(sorted, 0.25),
        "median": quantile_f64(sorted, 0.5),
        "p75": quantile_f64(sorted, 0.75),
        "p90": quantile_f64(sorted, 0.90),
        "p95": quantile_f64(sorted, 0.95),
        "p99": quantile_f64(sorted, 0.99),
        "max": quantile_f64(sorted, 1.0),
    })
}

fn mean_usize(values: &[usize]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().map(|&value| value as f64).sum::<f64>() / values.len() as f64
    }
}

fn quantile_usize(sorted: &[usize], probability: f64) -> usize {
    sorted
        .get(quantile_index(sorted.len(), probability))
        .copied()
        .unwrap_or(0)
}

fn quantile_f64(sorted: &[f64], probability: f64) -> f64 {
    sorted
        .get(quantile_index(sorted.len(), probability))
        .copied()
        .unwrap_or(0.0)
}

fn quantile_index(len: usize, probability: f64) -> usize {
    if len <= 1 {
        0
    } else {
        ((len - 1) as f64 * probability).round() as usize
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn pack_od(source: u32, target: u32) -> u64 {
    (u64::from(source) << 32) | u64::from(target)
}

#[derive(Debug)]
struct LocalTimeParts {
    date: String,
    hour: usize,
    weekday: usize,
}

fn local_time_parts(timestamp: usize) -> Option<LocalTimeParts> {
    let timestamp = i64::try_from(timestamp).ok()?;
    let local = timestamp.checked_add(LOCAL_UTC_OFFSET_SECONDS)?;
    let days = local.div_euclid(86_400);
    let seconds_in_day = local.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    Some(LocalTimeParts {
        date: format!("{year:04}-{month:02}-{day:02}"),
        hour: (seconds_in_day / 3_600) as usize,
        weekday: (days + 3).rem_euclid(7) as usize,
    })
}

// Howard Hinnant's proleptic-Gregorian civil-from-days transform. `days == 0`
// denotes 1970-01-01.
fn civil_from_days(days_since_epoch: i64) -> (i64, i64, i64) {
    let z = days_since_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (year, month, day)
}

fn atomic_write_json(path: &Path, value: &Value) -> Result<(), String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("loop_policy_audit.json");
    let temporary = path.with_file_name(format!(".{file_name}.{}.tmp", std::process::id()));
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("failed to encode audit JSON: {error}"))?;
    let write_result = (|| -> Result<(), String> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
            .map_err(|error| format!("failed to create {}: {error}", temporary.display()))?;
        file.write_all(&bytes)
            .map_err(|error| format!("failed to write {}: {error}", temporary.display()))?;
        file.write_all(b"\n")
            .map_err(|error| format!("failed to finish {}: {error}", temporary.display()))?;
        file.sync_all()
            .map_err(|error| format!("failed to sync {}: {error}", temporary.display()))?;
        std::fs::rename(&temporary, path).map_err(|error| {
            format!(
                "failed to atomically rename {} to {}: {error}",
                temporary.display(),
                path.display()
            )
        })?;
        Ok(())
    })();
    if write_result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    write_result
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!(
            "Usage: cargo run --release --example audit_loop_policies -- \\
             \n  [--city beijing] [--train-variant all] \\
             \n  [--output experiments/loop_policy_audit.json] \\
             \n  [--max-samples N] [--progress-every N]\n\n\
             This program reads only the requested train pickle. It never reads test data and \
             never trains a model. Omit --max-samples for the required full-train audit."
        );
        std::process::exit(0);
    }
    let mut args = Args {
        city: "beijing".to_string(),
        train_variant: "all".to_string(),
        output: PathBuf::from("experiments/loop_policy_audit.json"),
        max_samples: None,
        progress_every: 100_000,
    };
    let mut index = 0;
    while index < raw.len() {
        let flag = &raw[index];
        let value = raw
            .get(index + 1)
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--city" => args.city = value.clone(),
            "--train-variant" => args.train_variant = value.clone(),
            "--output" => args.output = PathBuf::from(value),
            "--max-samples" => {
                args.max_samples = Some(parse_positive_usize(flag, value)?);
            }
            "--progress-every" => {
                args.progress_every = value
                    .parse::<usize>()
                    .map_err(|error| format!("invalid {flag} value {value:?}: {error}"))?;
            }
            _ => return Err(format!("unknown argument {flag:?}; use --help")),
        }
        index += 2;
    }
    Ok(args)
}

fn parse_positive_usize(flag: &str, value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("invalid {flag} value {value:?}: {error}"))?;
    if parsed == 0 {
        Err(format!("{flag} must be positive"))
    } else {
        Ok(parsed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_graph() -> GraphData {
        // Edges: 0->1, 1->2, 2->1, 1->3, 3->3, 3->4.
        GraphData {
            tail: vec![0, 1, 2, 1, 3, 3],
            head: vec![1, 2, 1, 3, 3, 4],
            baseline_weights: vec![1; 6],
            x: vec![0.0; 5],
            y: vec![0.0; 5],
        }
    }

    #[test]
    fn loop_analysis_classifies_backtrack_and_closed_walk() {
        let info = analyze_loops(&[0, 1, 2, 1, 0]);
        assert_eq!(info.closure_spans, vec![2, 4]);
        assert_eq!(info.distinct_repeated_nodes, 2);
        assert!(info.repeats_start_node);
        assert!(info.closed_trip);
        assert_eq!(info.self_loop_edges, 0);
    }

    #[test]
    fn chronological_erasure_removes_complete_loops() {
        let path = vec![0, 1, 2, 3, 4, 5];
        let graph = test_graph();
        assert_eq!(
            chronological_loop_erasure(&path, &graph.tail, &graph.head).unwrap(),
            vec![0, 3, 5]
        );
    }

    #[test]
    fn split_preserves_non_self_loop_edges_as_simple_paths() {
        let graph = test_graph();
        let path = vec![0, 1, 2, 3, 4, 5];
        let nodes = vec![0, 1, 2, 1, 3, 3, 4];
        let split = split_at_repeated_nodes(&path, &nodes);
        assert_eq!(split, vec![vec![0, 1], vec![2, 3], vec![5]]);
        assert!(
            split
                .iter()
                .all(|part| is_simple_continuous_path(part, &graph))
        );
        assert_eq!(split.iter().map(Vec::len).sum::<usize>(), path.len() - 1);
    }

    #[test]
    fn validation_separates_structural_errors_from_cycles() {
        let graph = test_graph();
        assert!(matches!(
            validate_path(&[], &graph),
            Err(InvalidPath::Empty)
        ));
        assert!(matches!(
            validate_path(&[99], &graph),
            Err(InvalidPath::OutOfBounds)
        ));
        assert!(matches!(
            validate_path(&[0, 5], &graph),
            Err(InvalidPath::Discontinuous)
        ));
        let cyclic = validate_path(&[0, 1, 2, 3], &graph).unwrap();
        assert!(cyclic.loop_info.is_cyclic());
    }

    #[test]
    fn local_time_conversion_uses_utc_plus_eight() {
        let local = local_time_parts(0).unwrap();
        assert_eq!(local.date, "1970-01-01");
        assert_eq!(local.hour, 8);
        assert_eq!(local.weekday, 3); // Thursday with Monday == 0.
    }
}
