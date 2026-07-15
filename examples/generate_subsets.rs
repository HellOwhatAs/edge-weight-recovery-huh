use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde_json::json;
use serde_pickle::Value;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process;

type RawTrip = (Value, Vec<usize>, (usize, usize));

const USAGE: &str = "Usage:
  cargo run --release --example generate_subsets -- \\
    --city CITY \\
    --split SPLIT \\
    --source-variant SOURCE_VARIANT \\
    --output-variant OUTPUT_VARIANT \\
    --sample-count N \\
    --seed SEED \\
    --manifest PATH

Reads data/{city}_data/preprocessed_{split}_trips_{source_variant}.pkl and writes
data/{city}_data/preprocessed_{split}_trips_{output_variant}.pkl.

Sampling is uniform without replacement. StdRng shuffles all source indices,
the first N indices are selected, and those indices are sorted before records
are emitted, so the sampled records retain their original source order.";

#[derive(Debug)]
struct Args {
    city: String,
    split: String,
    source_variant: String,
    output_variant: String,
    sample_count: usize,
    seed: u64,
    manifest: PathBuf,
}

enum ParseOutcome {
    Run(Args),
    Help,
}

fn main() {
    match parse_args().and_then(|outcome| match outcome {
        ParseOutcome::Run(args) => run(args),
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
    let mut split = None;
    let mut source_variant = None;
    let mut output_variant = None;
    let mut sample_count = None;
    let mut seed = None;
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
            "--split" => set_once(&mut split, value, &flag)?,
            "--source-variant" => set_once(&mut source_variant, value, &flag)?,
            "--output-variant" => set_once(&mut output_variant, value, &flag)?,
            "--sample-count" => {
                let parsed = value
                    .parse::<usize>()
                    .map_err(|error| format!("invalid {flag} value {value:?}: {error}"))?;
                set_once(&mut sample_count, parsed, &flag)?;
            }
            "--seed" => {
                let parsed = value
                    .parse::<u64>()
                    .map_err(|error| format!("invalid {flag} value {value:?}: {error}"))?;
                set_once(&mut seed, parsed, &flag)?;
            }
            "--manifest" => set_once(&mut manifest, PathBuf::from(value), &flag)?,
            _ => return Err(format!("unknown argument {flag:?}")),
        }
    }

    let args = Args {
        city: required(city, "--city")?,
        split: required(split, "--split")?,
        source_variant: required(source_variant, "--source-variant")?,
        output_variant: required(output_variant, "--output-variant")?,
        sample_count: required(sample_count, "--sample-count")?,
        seed: required(seed, "--seed")?,
        manifest: required(manifest, "--manifest")?,
    };
    validate_args(&args)?;
    Ok(ParseOutcome::Run(args))
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
    validate_path_component(&args.split, "--split")?;
    validate_path_component(&args.source_variant, "--source-variant")?;
    validate_path_component(&args.output_variant, "--output-variant")?;

    if args.source_variant == args.output_variant {
        return Err("--source-variant and --output-variant must differ".to_owned());
    }
    if args.sample_count == 0 {
        return Err("--sample-count must be greater than zero".to_owned());
    }
    if args.manifest.as_os_str().is_empty() {
        return Err("--manifest must not be empty".to_owned());
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
            "{flag} must be a nonempty path component containing only ASCII letters, digits, '.', '_' or '-'"
        ))
    }
}

fn run(args: Args) -> Result<(), String> {
    let source_path = trip_path(&args.city, &args.split, &args.source_variant);
    let output_path = trip_path(&args.city, &args.split, &args.output_variant);
    if args.manifest == source_path || args.manifest == output_path {
        return Err("--manifest must differ from both pickle paths".to_owned());
    }

    let source_bytes = fs::metadata(&source_path)
        .map_err(|error| format!("failed to inspect {}: {error}", source_path.display()))?
        .len();
    let source = File::open(&source_path)
        .map_err(|error| format!("failed to open {}: {error}", source_path.display()))?;
    let raw: Vec<RawTrip> = serde_pickle::from_reader(BufReader::new(source), Default::default())
        .map_err(|error| {
        format!(
            "failed to decode {} as Vec<(trip_key, Vec<edge_id>, (start_time, end_time))>: {error}",
            source_path.display()
        )
    })?;

    let available_count = raw.len();
    if args.sample_count > available_count {
        return Err(format!(
            "--sample-count {} exceeds the {} records available in {}",
            args.sample_count,
            available_count,
            source_path.display()
        ));
    }

    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut selected_indices: Vec<usize> = (0..available_count).collect();
    selected_indices.shuffle(&mut rng);
    selected_indices.truncate(args.sample_count);
    selected_indices.sort_unstable();

    let sampled = select_records(raw, &selected_indices)?;
    write_pickle_atomic(&output_path, &Value::List(sampled))?;
    let output_bytes = fs::metadata(&output_path)
        .map_err(|error| format!("failed to inspect {}: {error}", output_path.display()))?
        .len();

    let manifest = json!({
        "schema_version": 1,
        "city": args.city,
        "split": args.split,
        "source_variant": args.source_variant,
        "output_variant": args.output_variant,
        "source": {
            "path": source_path.to_string_lossy(),
            "file_bytes": source_bytes,
            "sha256": null
        },
        "output": {
            "path": output_path.to_string_lossy(),
            "file_bytes": output_bytes,
            "sha256": null
        },
        "seed": args.seed,
        "available_count": available_count,
        "sample_count": args.sample_count,
        "selected_source_indices": selected_indices,
        "sampling": {
            "replacement": false,
            "rng": "rand 0.8.5 rand::rngs::StdRng seeded with SeedableRng::seed_from_u64",
            "selection": "Fisher-Yates shuffle of indices 0..available_count, then take sample_count",
            "emission_order": "selected indices sorted ascending; records emitted in source order",
            "nesting": "for the same source and seed, a smaller sample is a strict prefix-set of a larger sample"
        }
    });
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("failed to encode manifest JSON: {error}"))?;
    write_bytes_atomic(&args.manifest, &manifest_bytes)?;

    println!(
        "sampled {} of {} records from {} into {} ({} bytes); manifest: {}",
        args.sample_count,
        available_count,
        source_path.display(),
        output_path.display(),
        output_bytes,
        args.manifest.display()
    );
    Ok(())
}

fn trip_path(city: &str, split: &str, variant: &str) -> PathBuf {
    PathBuf::from(format!(
        "data/{city}_data/preprocessed_{split}_trips_{variant}.pkl"
    ))
}

fn select_records(raw: Vec<RawTrip>, selected_indices: &[usize]) -> Result<Vec<Value>, String> {
    let mut selected = selected_indices.iter().copied().peekable();
    let mut records = Vec::with_capacity(selected_indices.len());

    for (index, trip) in raw.into_iter().enumerate() {
        if selected.peek().copied() == Some(index) {
            records.push(trip_to_value(trip)?);
            selected.next();
        }
    }

    if selected.next().is_some() || records.len() != selected_indices.len() {
        return Err("internal error while selecting sampled records".to_owned());
    }
    Ok(records)
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
