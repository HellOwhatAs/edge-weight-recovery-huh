use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde_json::{Value as JsonValue, json};
use serde_pickle::Value;
use std::collections::HashSet;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process;

type RawTrip = (Value, Vec<usize>, (usize, usize));

const USAGE: &str = "Usage:
  cargo run --release --example generate_validation_blocks -- \\
    --city CITY \\
    --source-variant SOURCE_VARIANT \\
    --exclude-manifest PRIOR_VALIDATION_MANIFEST.json \\
    --development-variant VARIANT \\
    --development-label LABEL \\
    --development-start START_UNIX_SECONDS \\
    --development-end-exclusive END_UNIX_SECONDS \\
    --confirmation-a-variant VARIANT \\
    --confirmation-a-label LABEL \\
    --confirmation-a-start START_UNIX_SECONDS \\
    --confirmation-a-end-exclusive END_UNIX_SECONDS \\
    --confirmation-a-count N \\
    --confirmation-a-seed SEED \\
    --confirmation-b-variant VARIANT \\
    --confirmation-b-label LABEL \\
    --confirmation-b-start START_UNIX_SECONDS \\
    --confirmation-b-end-exclusive END_UNIX_SECONDS \\
    --confirmation-b-count N \\
    --confirmation-b-seed SEED \\
    --manifest OUTPUT_MANIFEST.json

Reads only data/{city}_data/preprocessed_validation_trips_{source_variant}.pkl.
It creates one complete early-time development block and two fixed-size,
later-time confirmation samples. The three time windows must be ordered and
non-overlapping. Every source index listed in --exclude-manifest is excluded
before sampling, making the new blocks disjoint from a previously used
validation sample.

A route is eligible only when block_start <= start < block_end_exclusive and
start < end < block_end_exclusive. Confirmation sampling is uniform without
replacement using independent StdRng seeds. Selected records are sorted by
source index before output, so every pickle preserves source-file order.";

#[derive(Clone, Debug, Eq, PartialEq)]
struct TimeWindow {
    start: usize,
    end_exclusive: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct BlockSpec {
    role: &'static str,
    variant: String,
    label: String,
    window: TimeWindow,
    sample: Option<SampleSpec>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SampleSpec {
    count: usize,
    seed: u64,
}

#[derive(Debug)]
struct Args {
    city: String,
    source_variant: String,
    exclusion_manifest: PathBuf,
    development: BlockSpec,
    confirmation_a: BlockSpec,
    confirmation_b: BlockSpec,
    manifest: PathBuf,
}

enum ParseOutcome {
    Run(Box<Args>),
    Help,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Membership {
    Outside,
    NonpositiveDuration,
    CrossesEndBoundary,
    Eligible,
}

#[derive(Debug, Default)]
struct TimestampRange {
    minimum_start: Option<usize>,
    maximum_start: Option<usize>,
    minimum_end: Option<usize>,
    maximum_end: Option<usize>,
}

impl TimestampRange {
    fn observe(&mut self, start: usize, end: usize) {
        self.minimum_start = Some(self.minimum_start.map_or(start, |value| value.min(start)));
        self.maximum_start = Some(self.maximum_start.map_or(start, |value| value.max(start)));
        self.minimum_end = Some(self.minimum_end.map_or(end, |value| value.min(end)));
        self.maximum_end = Some(self.maximum_end.map_or(end, |value| value.max(end)));
    }

    fn to_json(&self) -> JsonValue {
        json!({
            "minimum_start": self.minimum_start,
            "maximum_start": self.maximum_start,
            "minimum_end": self.minimum_end,
            "maximum_end": self.maximum_end,
        })
    }
}

#[derive(Debug, Default)]
struct BlockStats {
    starts_in_window: usize,
    excluded_nonpositive_duration: usize,
    excluded_crossing_end_boundary: usize,
    excluded_prior_validation: usize,
    eligible_before_sampling: usize,
    excluded_by_sampling: usize,
    selected: usize,
    selected_timestamps: TimestampRange,
}

#[derive(Debug, Default)]
struct SourceStats {
    raw_count: usize,
    nonpositive_duration: usize,
    starts_outside_all_blocks: usize,
    timestamps: TimestampRange,
}

#[derive(Debug)]
struct Candidate {
    source_index: usize,
    trip: RawTrip,
}

#[derive(Debug)]
struct CandidateBlock {
    candidates: Vec<Candidate>,
    stats: BlockStats,
}

impl CandidateBlock {
    fn new(capacity: usize) -> Self {
        Self {
            candidates: Vec::with_capacity(capacity),
            stats: BlockStats::default(),
        }
    }
}

#[derive(Debug)]
struct BlockSelection {
    records: Vec<Value>,
    source_indices: Vec<usize>,
    stats: BlockStats,
}

#[derive(Debug)]
struct ExclusionSet {
    path: PathBuf,
    indices: HashSet<usize>,
    declared_available_count: usize,
    declared_sample_count: usize,
    declared_output_variant: Option<String>,
    declared_seed: Option<u64>,
}

fn main() {
    match parse_args().and_then(|outcome| match outcome {
        ParseOutcome::Run(args) => run(*args),
        ParseOutcome::Help => {
            println!("{USAGE}");
            Ok(())
        }
    }) {
        Ok(()) => {}
        Err(error) => {
            eprintln!("error: {error}\n\n{USAGE}");
            process::exit(2);
        }
    }
}

fn parse_args() -> Result<ParseOutcome, String> {
    let mut city = None;
    let mut source_variant = None;
    let mut exclusion_manifest = None;
    let mut development_variant = None;
    let mut development_label = None;
    let mut development_start = None;
    let mut development_end_exclusive = None;
    let mut confirmation_a_variant = None;
    let mut confirmation_a_label = None;
    let mut confirmation_a_start = None;
    let mut confirmation_a_end_exclusive = None;
    let mut confirmation_a_count = None;
    let mut confirmation_a_seed = None;
    let mut confirmation_b_variant = None;
    let mut confirmation_b_label = None;
    let mut confirmation_b_start = None;
    let mut confirmation_b_end_exclusive = None;
    let mut confirmation_b_count = None;
    let mut confirmation_b_seed = None;
    let mut manifest = None;

    let mut arguments = env::args().skip(1);
    while let Some(flag) = arguments.next() {
        if flag == "--help" || flag == "-h" {
            return Ok(ParseOutcome::Help);
        }
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value after {flag}"))?;
        match flag.as_str() {
            "--city" => set_once(&mut city, value, &flag)?,
            "--source-variant" => set_once(&mut source_variant, value, &flag)?,
            "--exclude-manifest" => {
                set_once(&mut exclusion_manifest, PathBuf::from(value), &flag)?;
            }
            "--development-variant" => set_once(&mut development_variant, value, &flag)?,
            "--development-label" => set_once(&mut development_label, value, &flag)?,
            "--development-start" => {
                set_once(&mut development_start, parse_usize(&value, &flag)?, &flag)?;
            }
            "--development-end-exclusive" => set_once(
                &mut development_end_exclusive,
                parse_usize(&value, &flag)?,
                &flag,
            )?,
            "--confirmation-a-variant" => set_once(&mut confirmation_a_variant, value, &flag)?,
            "--confirmation-a-label" => set_once(&mut confirmation_a_label, value, &flag)?,
            "--confirmation-a-start" => set_once(
                &mut confirmation_a_start,
                parse_usize(&value, &flag)?,
                &flag,
            )?,
            "--confirmation-a-end-exclusive" => set_once(
                &mut confirmation_a_end_exclusive,
                parse_usize(&value, &flag)?,
                &flag,
            )?,
            "--confirmation-a-count" => set_once(
                &mut confirmation_a_count,
                parse_usize(&value, &flag)?,
                &flag,
            )?,
            "--confirmation-a-seed" => {
                set_once(&mut confirmation_a_seed, parse_u64(&value, &flag)?, &flag)?
            }
            "--confirmation-b-variant" => set_once(&mut confirmation_b_variant, value, &flag)?,
            "--confirmation-b-label" => set_once(&mut confirmation_b_label, value, &flag)?,
            "--confirmation-b-start" => set_once(
                &mut confirmation_b_start,
                parse_usize(&value, &flag)?,
                &flag,
            )?,
            "--confirmation-b-end-exclusive" => set_once(
                &mut confirmation_b_end_exclusive,
                parse_usize(&value, &flag)?,
                &flag,
            )?,
            "--confirmation-b-count" => set_once(
                &mut confirmation_b_count,
                parse_usize(&value, &flag)?,
                &flag,
            )?,
            "--confirmation-b-seed" => {
                set_once(&mut confirmation_b_seed, parse_u64(&value, &flag)?, &flag)?
            }
            "--manifest" => set_once(&mut manifest, PathBuf::from(value), &flag)?,
            _ => return Err(format!("unknown argument {flag:?}")),
        }
    }

    let args = Args {
        city: required(city, "--city")?,
        source_variant: required(source_variant, "--source-variant")?,
        exclusion_manifest: required(exclusion_manifest, "--exclude-manifest")?,
        development: BlockSpec {
            role: "development",
            variant: required(development_variant, "--development-variant")?,
            label: required(development_label, "--development-label")?,
            window: TimeWindow {
                start: required(development_start, "--development-start")?,
                end_exclusive: required(development_end_exclusive, "--development-end-exclusive")?,
            },
            sample: None,
        },
        confirmation_a: BlockSpec {
            role: "confirmation_a",
            variant: required(confirmation_a_variant, "--confirmation-a-variant")?,
            label: required(confirmation_a_label, "--confirmation-a-label")?,
            window: TimeWindow {
                start: required(confirmation_a_start, "--confirmation-a-start")?,
                end_exclusive: required(
                    confirmation_a_end_exclusive,
                    "--confirmation-a-end-exclusive",
                )?,
            },
            sample: Some(SampleSpec {
                count: required(confirmation_a_count, "--confirmation-a-count")?,
                seed: required(confirmation_a_seed, "--confirmation-a-seed")?,
            }),
        },
        confirmation_b: BlockSpec {
            role: "confirmation_b",
            variant: required(confirmation_b_variant, "--confirmation-b-variant")?,
            label: required(confirmation_b_label, "--confirmation-b-label")?,
            window: TimeWindow {
                start: required(confirmation_b_start, "--confirmation-b-start")?,
                end_exclusive: required(
                    confirmation_b_end_exclusive,
                    "--confirmation-b-end-exclusive",
                )?,
            },
            sample: Some(SampleSpec {
                count: required(confirmation_b_count, "--confirmation-b-count")?,
                seed: required(confirmation_b_seed, "--confirmation-b-seed")?,
            }),
        },
        manifest: required(manifest, "--manifest")?,
    };
    validate_args(&args)?;
    Ok(ParseOutcome::Run(Box::new(args)))
}

fn parse_usize(value: &str, flag: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid {flag} value {value:?}: {error}"))
}

fn parse_u64(value: &str, flag: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|error| format!("invalid {flag} value {value:?}: {error}"))
}

fn set_once<T>(slot: &mut Option<T>, value: T, flag: &str) -> Result<(), String> {
    if slot.replace(value).is_some() {
        return Err(format!("argument {flag} was provided more than once"));
    }
    Ok(())
}

fn required<T>(value: Option<T>, flag: &str) -> Result<T, String> {
    value.ok_or_else(|| format!("required argument {flag} is missing"))
}

fn validate_args(args: &Args) -> Result<(), String> {
    validate_path_component(&args.city, "--city")?;
    validate_path_component(&args.source_variant, "--source-variant")?;
    let blocks = [
        &args.development,
        &args.confirmation_a,
        &args.confirmation_b,
    ];
    let mut variants = HashSet::new();
    for block in blocks {
        validate_path_component(&block.variant, "output variant")?;
        if block.variant == args.source_variant {
            return Err("output variants must differ from --source-variant".to_owned());
        }
        if !variants.insert(block.variant.as_str()) {
            return Err("all output variants must differ".to_owned());
        }
        if block.label.is_empty() {
            return Err(format!("{} label must not be empty", block.role));
        }
        if block.window.start >= block.window.end_exclusive {
            return Err(format!(
                "{} start must precede its exclusive end",
                block.role
            ));
        }
        if block.sample.is_some_and(|sample| sample.count == 0) {
            return Err(format!("{} sample count must be positive", block.role));
        }
    }
    if args.development.window.end_exclusive > args.confirmation_a.window.start
        || args.confirmation_a.window.end_exclusive > args.confirmation_b.window.start
    {
        return Err(
            "blocks must be chronological and non-overlapping: development, confirmation_a, confirmation_b"
                .to_owned(),
        );
    }
    if args.confirmation_a.sample.map(|sample| sample.seed)
        == args.confirmation_b.sample.map(|sample| sample.seed)
    {
        return Err("confirmation blocks must use independent (different) seeds".to_owned());
    }
    for path in [&args.exclusion_manifest, &args.manifest] {
        if path.as_os_str().is_empty() {
            return Err("manifest paths must not be empty".to_owned());
        }
    }
    if args.exclusion_manifest == args.manifest {
        return Err("input exclusion manifest and output manifest must differ".to_owned());
    }
    Ok(())
}

fn validate_path_component(value: &str, flag: &str) -> Result<(), String> {
    let safe = !value.is_empty()
        && value != "."
        && value != ".."
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'));
    if safe {
        Ok(())
    } else {
        Err(format!(
            "{flag} must contain only ASCII letters, digits, '.', '_' or '-'"
        ))
    }
}

fn run(args: Args) -> Result<(), String> {
    let source_path = validation_path(&args.city, &args.source_variant);
    let specs = [
        &args.development,
        &args.confirmation_a,
        &args.confirmation_b,
    ];
    let output_paths = specs
        .iter()
        .map(|spec| validation_path(&args.city, &spec.variant))
        .collect::<Vec<_>>();
    for path in &output_paths {
        if args.manifest == *path || args.exclusion_manifest == *path {
            return Err("manifest paths must differ from output pickle paths".to_owned());
        }
    }
    if args.manifest == source_path || args.exclusion_manifest == source_path {
        return Err("manifest paths must differ from the source pickle path".to_owned());
    }

    let exclusion = load_exclusion_set(&args.exclusion_manifest)?;
    let source_bytes = file_size(&source_path)?;
    let source_sha256 = sha256_file(&source_path)?;
    let source = File::open(&source_path)
        .map_err(|error| format!("failed to open {}: {error}", source_path.display()))?;
    let raw: Vec<RawTrip> = serde_pickle::from_reader(BufReader::new(source), Default::default())
        .map_err(|error| {
        format!(
            "failed to decode {} as Vec<(trip_key, Vec<edge_id>, (start_time, end_time))>: {error}",
            source_path.display()
        )
    })?;
    if exclusion.declared_available_count != raw.len() {
        return Err(format!(
            "exclusion manifest declares available_count={} but source contains {} records",
            exclusion.declared_available_count,
            raw.len()
        ));
    }
    if exclusion.indices.len() != exclusion.declared_sample_count {
        return Err(format!(
            "exclusion manifest declares sample_count={} but has {} unique selected_source_indices",
            exclusion.declared_sample_count,
            exclusion.indices.len()
        ));
    }
    if let Some(out_of_bounds) = exclusion.indices.iter().find(|&&index| index >= raw.len()) {
        return Err(format!(
            "exclusion manifest contains out-of-bounds source index {out_of_bounds}"
        ));
    }

    let (selections, source_stats) = partition_records(raw, &specs, &exclusion.indices)?;
    if selections
        .iter()
        .any(|selection| selection.records.is_empty())
    {
        return Err("at least one block selected zero records".to_owned());
    }

    let mut block_json = Vec::with_capacity(specs.len());
    let mut selected_sets = Vec::with_capacity(specs.len());
    for ((spec, output_path), selection) in specs.iter().zip(output_paths.iter()).zip(selections) {
        let BlockSelection {
            records,
            source_indices,
            stats,
        } = selection;
        write_pickle_atomic(output_path, &Value::List(records))?;
        let output_bytes = file_size(output_path)?;
        let output_sha256 = sha256_file(output_path)?;
        block_json.push(block_manifest(
            spec,
            output_path,
            output_bytes,
            &output_sha256,
            &stats,
            &source_indices,
        ));
        println!(
            "BLOCK role={} label={} interval=[{},{}) starts={} prior_excluded={} eligible={} sampled_out={} selected={} output={}",
            spec.role,
            spec.label,
            spec.window.start,
            spec.window.end_exclusive,
            stats.starts_in_window,
            stats.excluded_prior_validation,
            stats.eligible_before_sampling,
            stats.excluded_by_sampling,
            stats.selected,
            output_path.display()
        );
        selected_sets.push(source_indices);
    }
    if !sets_are_pairwise_disjoint(&selected_sets)
        || selected_sets
            .iter()
            .flatten()
            .any(|index| exclusion.indices.contains(index))
    {
        return Err("internal error: selected source-index sets are not disjoint".to_owned());
    }

    let excluded_matched = block_json
        .iter()
        .map(|block| block["excluded_prior_validation"].as_u64().unwrap_or(0) as usize)
        .sum::<usize>();
    let manifest = json!({
        "schema_version": 1,
        "generator": "examples/generate_validation_blocks.rs",
        "city": args.city,
        "split": "validation",
        "timestamp_semantics": {
            "unit": "Unix seconds",
            "display_timezone": "Asia/Shanghai (UTC+08:00)",
            "selection_fields": ["trip start timestamp", "trip end timestamp"]
        },
        "source": {
            "variant": args.source_variant,
            "path": source_path.to_string_lossy(),
            "file_bytes": source_bytes,
            "sha256": source_sha256,
            "raw_count": source_stats.raw_count,
            "timestamp_range": source_stats.timestamps.to_json(),
            "nonpositive_duration_count": source_stats.nonpositive_duration
        },
        "prior_validation_exclusion": {
            "manifest_path": exclusion.path.to_string_lossy(),
            "manifest_sha256": sha256_file(&exclusion.path)?,
            "declared_available_count": exclusion.declared_available_count,
            "declared_sample_count": exclusion.declared_sample_count,
            "declared_output_variant": exclusion.declared_output_variant,
            "declared_seed": exclusion.declared_seed,
            "unique_source_indices": exclusion.indices.len(),
            "excluded_eligible_records_across_blocks": excluded_matched,
            "rule": "exclude by source index before any block-specific sampling"
        },
        "selection": {
            "eligibility_rule": "block_start <= start < block_end_exclusive AND start < end < block_end_exclusive",
            "development_sampling": "none; retain every eligible non-excluded record",
            "confirmation_sampling": "uniform without replacement within each time block using independent rand 0.8.5 StdRng seeds",
            "emission_order": "selected indices ascending; output retains source-file order",
            "starts_outside_all_blocks": source_stats.starts_outside_all_blocks,
            "blocks_chronological_and_non_overlapping": true,
            "selected_source_index_sets_pairwise_disjoint": true,
            "all_blocks_disjoint_from_prior_validation": true
        },
        "blocks": block_json
    });
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("failed to encode manifest JSON: {error}"))?;
    write_bytes_atomic(&args.manifest, &manifest_bytes)?;

    println!(
        "VALIDATION_BLOCKS source_raw={} prior_excluded={} blocks={} outside={} manifest={}",
        source_stats.raw_count,
        exclusion.indices.len(),
        selected_sets.len(),
        source_stats.starts_outside_all_blocks,
        args.manifest.display()
    );
    Ok(())
}

fn load_exclusion_set(path: &Path) -> Result<ExclusionSet, String> {
    let file = File::open(path).map_err(|error| {
        format!(
            "failed to open exclusion manifest {}: {error}",
            path.display()
        )
    })?;
    let value: JsonValue = serde_json::from_reader(BufReader::new(file)).map_err(|error| {
        format!(
            "failed to decode exclusion manifest {}: {error}",
            path.display()
        )
    })?;
    let available_count = json_usize(&value, "available_count", path)?;
    let sample_count = json_usize(&value, "sample_count", path)?;
    let raw_indices = value["selected_source_indices"]
        .as_array()
        .ok_or_else(|| format!("{} has no selected_source_indices array", path.display()))?;
    let mut indices = HashSet::with_capacity(raw_indices.len());
    for (position, value) in raw_indices.iter().enumerate() {
        let index = value.as_u64().ok_or_else(|| {
            format!(
                "{} selected_source_indices[{position}] is not a nonnegative integer",
                path.display()
            )
        })?;
        let index = usize::try_from(index).map_err(|_| {
            format!(
                "{} selected_source_indices[{position}] does not fit usize",
                path.display()
            )
        })?;
        if !indices.insert(index) {
            return Err(format!(
                "{} repeats selected source index {index}",
                path.display()
            ));
        }
    }
    Ok(ExclusionSet {
        path: path.to_path_buf(),
        indices,
        declared_available_count: available_count,
        declared_sample_count: sample_count,
        declared_output_variant: value["output_variant"].as_str().map(str::to_owned),
        declared_seed: value["seed"].as_u64(),
    })
}

fn json_usize(value: &JsonValue, field: &str, path: &Path) -> Result<usize, String> {
    let raw = value[field].as_u64().ok_or_else(|| {
        format!(
            "{} field {field:?} is not a nonnegative integer",
            path.display()
        )
    })?;
    usize::try_from(raw)
        .map_err(|_| format!("{} field {field:?} does not fit usize", path.display()))
}

fn validation_path(city: &str, variant: &str) -> PathBuf {
    PathBuf::from(format!(
        "data/{city}_data/preprocessed_validation_trips_{variant}.pkl"
    ))
}

fn partition_records(
    raw: Vec<RawTrip>,
    specs: &[&BlockSpec],
    exclusions: &HashSet<usize>,
) -> Result<(Vec<BlockSelection>, SourceStats), String> {
    let mut candidates = specs
        .iter()
        .map(|_| CandidateBlock::new(raw.len() / specs.len()))
        .collect::<Vec<_>>();
    let mut source = SourceStats {
        raw_count: raw.len(),
        ..SourceStats::default()
    };

    for (source_index, trip) in raw.into_iter().enumerate() {
        let start = trip.2.0;
        let end = trip.2.1;
        source.timestamps.observe(start, end);
        if end <= start {
            source.nonpositive_duration += 1;
        }
        let Some(block_index) = specs
            .iter()
            .position(|spec| start >= spec.window.start && start < spec.window.end_exclusive)
        else {
            source.starts_outside_all_blocks += 1;
            continue;
        };

        let membership = classify(start, end, &specs[block_index].window);
        let candidate_block = &mut candidates[block_index];
        candidate_block.stats.starts_in_window += 1;
        match membership {
            Membership::Outside => unreachable!("window lookup and classification disagree"),
            Membership::NonpositiveDuration => {
                candidate_block.stats.excluded_nonpositive_duration += 1;
            }
            Membership::CrossesEndBoundary => {
                candidate_block.stats.excluded_crossing_end_boundary += 1;
            }
            Membership::Eligible if exclusions.contains(&source_index) => {
                candidate_block.stats.excluded_prior_validation += 1;
            }
            Membership::Eligible => candidate_block
                .candidates
                .push(Candidate { source_index, trip }),
        }
    }

    let selections = candidates
        .into_iter()
        .zip(specs.iter())
        .map(|(candidates, spec)| finalize_block(candidates, spec))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((selections, source))
}

fn classify(start: usize, end: usize, window: &TimeWindow) -> Membership {
    if start < window.start || start >= window.end_exclusive {
        Membership::Outside
    } else if end <= start {
        Membership::NonpositiveDuration
    } else if end >= window.end_exclusive {
        Membership::CrossesEndBoundary
    } else {
        Membership::Eligible
    }
}

fn finalize_block(
    mut candidate_block: CandidateBlock,
    spec: &BlockSpec,
) -> Result<BlockSelection, String> {
    candidate_block.stats.eligible_before_sampling = candidate_block.candidates.len();
    let selected_positions = match spec.sample {
        Some(sample) => {
            if sample.count > candidate_block.candidates.len() {
                return Err(format!(
                    "{} requests {} records but only {} eligible records remain after exclusions",
                    spec.role,
                    sample.count,
                    candidate_block.candidates.len()
                ));
            }
            candidate_block.stats.excluded_by_sampling =
                candidate_block.candidates.len() - sample.count;
            sample_positions(candidate_block.candidates.len(), sample.count, sample.seed)
        }
        None => (0..candidate_block.candidates.len()).collect(),
    };
    let mut selected_positions = selected_positions.into_iter().peekable();
    let mut records = Vec::with_capacity(
        spec.sample
            .map_or(candidate_block.candidates.len(), |sample| sample.count),
    );
    let mut source_indices = Vec::with_capacity(records.capacity());
    for (position, candidate) in candidate_block.candidates.into_iter().enumerate() {
        if selected_positions.peek().copied() != Some(position) {
            continue;
        }
        selected_positions.next();
        let start = candidate.trip.2.0;
        let end = candidate.trip.2.1;
        candidate_block
            .stats
            .selected_timestamps
            .observe(start, end);
        source_indices.push(candidate.source_index);
        records.push(trip_to_value(candidate.trip)?);
    }
    if selected_positions.next().is_some() {
        return Err("internal error while selecting block records".to_owned());
    }
    candidate_block.stats.selected = records.len();
    Ok(BlockSelection {
        records,
        source_indices,
        stats: candidate_block.stats,
    })
}

fn sample_positions(available: usize, count: usize, seed: u64) -> Vec<usize> {
    let mut positions = (0..available).collect::<Vec<_>>();
    let mut rng = StdRng::seed_from_u64(seed);
    positions.shuffle(&mut rng);
    positions.truncate(count);
    positions.sort_unstable();
    positions
}

fn sets_are_pairwise_disjoint(sets: &[Vec<usize>]) -> bool {
    let mut seen = HashSet::new();
    sets.iter()
        .flatten()
        .all(|source_index| seen.insert(*source_index))
}

fn block_manifest(
    spec: &BlockSpec,
    path: &Path,
    file_bytes: u64,
    sha256: &str,
    stats: &BlockStats,
    source_indices: &[usize],
) -> JsonValue {
    let sampling = match spec.sample {
        Some(sample) => json!({
            "kind": "uniform_without_replacement",
            "requested_count": sample.count,
            "seed": sample.seed,
            "rng": "rand 0.8.5 rand::rngs::StdRng seeded with SeedableRng::seed_from_u64",
            "algorithm": "Fisher-Yates shuffle of eligible candidate positions, truncate, then sort positions"
        }),
        None => json!({
            "kind": "all_eligible_records",
            "requested_count": null,
            "seed": null
        }),
    };
    json!({
        "role": spec.role,
        "label": spec.label,
        "variant": spec.variant,
        "path": path.to_string_lossy(),
        "file_bytes": file_bytes,
        "sha256": sha256,
        "window": {
            "start_inclusive": spec.window.start,
            "end_exclusive": spec.window.end_exclusive
        },
        "eligibility_raw_count": stats.eligible_before_sampling,
        "raw_count": stats.selected,
        "starts_in_window": stats.starts_in_window,
        "excluded_nonpositive_duration": stats.excluded_nonpositive_duration,
        "excluded_crossing_end_boundary": stats.excluded_crossing_end_boundary,
        "excluded_prior_validation": stats.excluded_prior_validation,
        "excluded_by_sampling": stats.excluded_by_sampling,
        "sampling": sampling,
        "actual_selected_timestamp_range": stats.selected_timestamps.to_json(),
        "selected_source_indices": source_indices
    })
}

fn trip_to_value((key, edges, (start_time, end_time)): RawTrip) -> Result<Value, String> {
    let edge_values = edges
        .into_iter()
        .map(|edge| usize_value(edge, "edge id"))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Value::Tuple(vec![
        key,
        Value::List(edge_values),
        Value::Tuple(vec![
            usize_value(start_time, "start time")?,
            usize_value(end_time, "end time")?,
        ]),
    ]))
}

fn usize_value(value: usize, label: &str) -> Result<Value, String> {
    i64::try_from(value)
        .map(Value::I64)
        .map_err(|_| format!("{label} {value} cannot be represented as a pickle i64"))
}

fn file_size(path: &Path) -> Result<u64, String> {
    fs::metadata(path)
        .map(|metadata| metadata.len())
        .map_err(|error| format!("failed to inspect {}: {error}", path.display()))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("failed to read {} for SHA-256: {error}", path.display()))?;
    Ok(sha256(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn sha256(input: &[u8]) -> [u8; 32] {
    const INITIAL: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const ROUND_CONSTANTS: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let bit_length = u64::try_from(input.len())
        .expect("usize always fits u64 on supported targets")
        .checked_mul(8)
        .expect("input length cannot overflow SHA-256 bit length");
    let mut padded = Vec::with_capacity(input.len() + 72);
    padded.extend_from_slice(input);
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_length.to_be_bytes());

    let mut state = INITIAL;
    let mut schedule = [0_u32; 64];
    for chunk in padded.chunks_exact(64) {
        for (index, word) in chunk.chunks_exact(4).enumerate() {
            schedule[index] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for index in 16..64 {
            let s0 = schedule[index - 15].rotate_right(7)
                ^ schedule[index - 15].rotate_right(18)
                ^ (schedule[index - 15] >> 3);
            let s1 = schedule[index - 2].rotate_right(17)
                ^ schedule[index - 2].rotate_right(19)
                ^ (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(s0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let big_s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ ((!e) & g);
            let temporary_1 = h
                .wrapping_add(big_s1)
                .wrapping_add(choice)
                .wrapping_add(ROUND_CONSTANTS[index])
                .wrapping_add(schedule[index]);
            let big_s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temporary_2 = big_s0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temporary_1);
            d = c;
            c = b;
            b = a;
            a = temporary_1.wrapping_add(temporary_2);
        }
        for (value, compressed) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *value = value.wrapping_add(compressed);
        }
    }

    let mut digest = [0_u8; 32];
    for (chunk, value) in digest.chunks_exact_mut(4).zip(state) {
        chunk.copy_from_slice(&value.to_be_bytes());
    }
    digest
}

fn write_pickle_atomic(path: &Path, value: &Value) -> Result<(), String> {
    ensure_parent(path)?;
    let (temporary_path, file) = create_temporary_file(path)?;
    let result = (|| {
        let mut writer = BufWriter::new(file);
        serde_pickle::value_to_writer(&mut writer, value, Default::default())
            .map_err(|error| format!("failed to encode {}: {error}", path.display()))?;
        writer
            .flush()
            .map_err(|error| format!("failed to flush {}: {error}", temporary_path.display()))?;
        writer
            .get_ref()
            .sync_all()
            .map_err(|error| format!("failed to sync {}: {error}", temporary_path.display()))?;
        Ok(())
    })();
    finish_atomic_write(path, &temporary_path, result)
}

fn write_bytes_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    ensure_parent(path)?;
    let (temporary_path, mut file) = create_temporary_file(path)?;
    let result = (|| {
        file.write_all(bytes)
            .map_err(|error| format!("failed to write {}: {error}", temporary_path.display()))?;
        file.write_all(b"\n")
            .map_err(|error| format!("failed to write {}: {error}", temporary_path.display()))?;
        file.sync_all()
            .map_err(|error| format!("failed to sync {}: {error}", temporary_path.display()))?;
        Ok(())
    })();
    finish_atomic_write(path, &temporary_path, result)
}

fn ensure_parent(path: &Path) -> Result<(), String> {
    let parent = usable_parent(path);
    fs::create_dir_all(parent)
        .map_err(|error| format!("failed to create {}: {error}", parent.display()))
}

fn usable_parent(path: &Path) -> &Path {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    }
}

fn create_temporary_file(path: &Path) -> Result<(PathBuf, File), String> {
    let parent = usable_parent(path);
    let name = path
        .file_name()
        .ok_or_else(|| format!("output path {} has no file name", path.display()))?
        .to_string_lossy();
    for counter in 0..1_000_u32 {
        let temporary_path = parent.join(format!(".{name}.tmp-{}-{counter}", process::id()));
        match OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary_path)
        {
            Ok(file) => return Ok((temporary_path, file)),
            Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(format!(
                    "failed to create {}: {error}",
                    temporary_path.display()
                ));
            }
        }
    }
    Err(format!(
        "could not allocate a temporary file next to {}",
        path.display()
    ))
}

fn finish_atomic_write(
    path: &Path,
    temporary_path: &Path,
    write_result: Result<(), String>,
) -> Result<(), String> {
    if let Err(error) = write_result {
        let _ = fs::remove_file(temporary_path);
        return Err(error);
    }
    if let Err(error) = fs::rename(temporary_path, path) {
        let _ = fs::remove_file(temporary_path);
        return Err(format!(
            "failed to replace {} atomically: {error}",
            path.display()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(start: usize, end_exclusive: usize) -> TimeWindow {
        TimeWindow {
            start,
            end_exclusive,
        }
    }

    #[test]
    fn classification_respects_half_open_start_and_strict_end() {
        let window = window(100, 200);
        assert_eq!(classify(100, 101, &window), Membership::Eligible);
        assert_eq!(classify(199, 199, &window), Membership::NonpositiveDuration);
        assert_eq!(classify(199, 200, &window), Membership::CrossesEndBoundary);
        assert_eq!(classify(99, 150, &window), Membership::Outside);
        assert_eq!(classify(200, 201, &window), Membership::Outside);
    }

    #[test]
    fn sampling_is_reproducible_sorted_and_without_replacement() {
        let first = sample_positions(100, 20, 42);
        let second = sample_positions(100, 20, 42);
        assert_eq!(first, second);
        assert_eq!(first.len(), 20);
        assert!(first.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(first.iter().all(|&position| position < 100));
    }

    #[test]
    fn pairwise_disjoint_check_detects_overlap() {
        assert!(sets_are_pairwise_disjoint(&[
            vec![1, 3],
            vec![2, 4],
            vec![5]
        ]));
        assert!(!sets_are_pairwise_disjoint(&[vec![1, 3], vec![2, 3]]));
    }

    #[test]
    fn sha256_matches_standard_vectors() {
        let hex = |bytes: &[u8]| {
            sha256(bytes)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        };
        assert_eq!(
            hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
