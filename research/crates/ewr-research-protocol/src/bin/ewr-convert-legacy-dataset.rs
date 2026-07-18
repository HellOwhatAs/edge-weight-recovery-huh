use ewr_research_protocol::DatasetRecordV1;
use serde::Deserialize;
use serde::de::IgnoredAny;
use std::collections::HashSet;
use std::ffi::{OsStr, OsString};
use std::fmt::{self, Display, Formatter};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const USAGE: &str = "Usage: ewr-convert-legacy-dataset --input PATH --output PATH";

static TEMPORARY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyDatasetRecord {
    manifest_id: String,
    edges: Vec<u32>,
    #[serde(rename = "end_time")]
    _end_time: Option<IgnoredAny>,
    #[serde(rename = "original_trip_id")]
    _original_trip_id: Option<IgnoredAny>,
    #[serde(rename = "source_index")]
    _source_index: Option<IgnoredAny>,
    #[serde(rename = "start_time")]
    _start_time: Option<IgnoredAny>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        eprintln!("{USAGE}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let action = parse_cli(std::env::args_os().skip(1))?;
    let CliAction::Run(arguments) = action else {
        println!("{USAGE}");
        return Ok(());
    };
    let converted = convert_file(&arguments.input, &arguments.output)?;
    println!("converted {converted} records");
    Ok(())
}

fn convert_file(input: &Path, output: &Path) -> Result<usize, ConversionError> {
    let input_file = File::open(input).map_err(|source| ConversionError::OpenInput {
        path: input.to_path_buf(),
        source,
    })?;
    let parent = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty());
    if let Some(parent) = parent {
        std::fs::create_dir_all(parent).map_err(|source| {
            ConversionError::CreateOutputDirectory {
                path: parent.to_path_buf(),
                source,
            }
        })?;
    }

    let mut temporary = TemporaryOutput::create_for(output)?;
    let converted = {
        let mut writer = BufWriter::new(&mut temporary.file);
        let converted = convert_jsonl(BufReader::new(input_file), &mut writer)?;
        writer
            .flush()
            .map_err(|source| ConversionError::WriteTemporary {
                path: temporary.path.clone(),
                source,
            })?;
        converted
    };
    temporary
        .file
        .sync_all()
        .map_err(|source| ConversionError::SyncTemporary {
            path: temporary.path.clone(),
            source,
        })?;
    std::fs::rename(&temporary.path, output).map_err(|source| ConversionError::ReplaceOutput {
        temporary: temporary.path.clone(),
        destination: output.to_path_buf(),
        source,
    })?;
    temporary.committed = true;
    Ok(converted)
}

fn convert_jsonl(reader: impl BufRead, mut writer: impl Write) -> Result<usize, ConversionError> {
    let mut sample_ids = HashSet::new();
    let mut converted = 0;
    for (index, line) in reader.lines().enumerate() {
        let line_number = index + 1;
        let line = line.map_err(|source| ConversionError::ReadInputLine {
            line: line_number,
            source,
        })?;
        if line.trim().is_empty() {
            return Err(ConversionError::BlankLine(line_number));
        }
        let legacy: LegacyDatasetRecord =
            serde_json::from_str(&line).map_err(|source| ConversionError::InvalidRecord {
                line: line_number,
                source,
            })?;
        if legacy.manifest_id.trim().is_empty() {
            return Err(ConversionError::EmptySampleId(line_number));
        }
        if legacy.manifest_id.chars().any(char::is_control) {
            return Err(ConversionError::ControlCharacterInSampleId {
                line: line_number,
                sample_id: legacy.manifest_id,
            });
        }
        if legacy.edges.len() < 2 {
            return Err(ConversionError::TooFewEdges {
                line: line_number,
                sample_id: legacy.manifest_id,
                count: legacy.edges.len(),
            });
        }
        if !sample_ids.insert(legacy.manifest_id.clone()) {
            return Err(ConversionError::DuplicateSampleId {
                line: line_number,
                sample_id: legacy.manifest_id,
            });
        }

        let record = DatasetRecordV1 {
            sample_id: legacy.manifest_id,
            original_edge_ids: legacy.edges,
        };
        serde_json::to_writer(&mut writer, &record).map_err(|source| {
            ConversionError::EncodeRecord {
                line: line_number,
                source,
            }
        })?;
        writer
            .write_all(b"\n")
            .map_err(|source| ConversionError::WriteRecord {
                line: line_number,
                source,
            })?;
        converted += 1;
    }
    if converted == 0 {
        return Err(ConversionError::EmptyInput);
    }
    Ok(converted)
}

fn temporary_path(destination: &Path) -> Result<PathBuf, ConversionError> {
    let filename = destination
        .file_name()
        .ok_or_else(|| ConversionError::InvalidOutputPath(destination.to_path_buf()))?;
    let mut temporary_name = OsString::from(".");
    temporary_name.push(filename);
    temporary_name.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    Ok(destination.with_file_name(temporary_name))
}

struct TemporaryOutput {
    path: PathBuf,
    file: File,
    committed: bool,
}

impl TemporaryOutput {
    fn create_for(destination: &Path) -> Result<Self, ConversionError> {
        for _ in 0..100 {
            let path = temporary_path(destination)?;
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => {
                    return Ok(Self {
                        path,
                        file,
                        committed: false,
                    });
                }
                Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(source) => return Err(ConversionError::CreateTemporary { path, source }),
            }
        }
        Err(ConversionError::TemporaryNameExhausted(
            destination.to_path_buf(),
        ))
    }
}

impl Drop for TemporaryOutput {
    fn drop(&mut self) {
        if !self.committed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[derive(Debug)]
enum ConversionError {
    OpenInput {
        path: PathBuf,
        source: std::io::Error,
    },
    ReadInputLine {
        line: usize,
        source: std::io::Error,
    },
    BlankLine(usize),
    InvalidRecord {
        line: usize,
        source: serde_json::Error,
    },
    EmptySampleId(usize),
    ControlCharacterInSampleId {
        line: usize,
        sample_id: String,
    },
    TooFewEdges {
        line: usize,
        sample_id: String,
        count: usize,
    },
    DuplicateSampleId {
        line: usize,
        sample_id: String,
    },
    EmptyInput,
    InvalidOutputPath(PathBuf),
    CreateOutputDirectory {
        path: PathBuf,
        source: std::io::Error,
    },
    CreateTemporary {
        path: PathBuf,
        source: std::io::Error,
    },
    TemporaryNameExhausted(PathBuf),
    EncodeRecord {
        line: usize,
        source: serde_json::Error,
    },
    WriteRecord {
        line: usize,
        source: std::io::Error,
    },
    WriteTemporary {
        path: PathBuf,
        source: std::io::Error,
    },
    SyncTemporary {
        path: PathBuf,
        source: std::io::Error,
    },
    ReplaceOutput {
        temporary: PathBuf,
        destination: PathBuf,
        source: std::io::Error,
    },
}

impl Display for ConversionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::OpenInput { path, source } => {
                write!(
                    formatter,
                    "failed to open input {}: {source}",
                    path.display()
                )
            }
            Self::ReadInputLine { line, source } => {
                write!(formatter, "failed to read input line {line}: {source}")
            }
            Self::BlankLine(line) => write!(formatter, "blank input JSONL line {line}"),
            Self::InvalidRecord { line, source } => {
                write!(formatter, "invalid legacy JSONL line {line}: {source}")
            }
            Self::EmptySampleId(line) => {
                write!(
                    formatter,
                    "legacy JSONL line {line} has an empty manifest_id"
                )
            }
            Self::ControlCharacterInSampleId { line, sample_id } => write!(
                formatter,
                "legacy JSONL line {line} manifest_id {sample_id:?} contains a control character"
            ),
            Self::TooFewEdges {
                line,
                sample_id,
                count,
            } => write!(
                formatter,
                "legacy JSONL line {line} sample {sample_id:?} has {count} edges; at least 2 required"
            ),
            Self::DuplicateSampleId { line, sample_id } => write!(
                formatter,
                "legacy JSONL line {line} repeats manifest_id {sample_id:?}"
            ),
            Self::EmptyInput => formatter.write_str("input JSONL contains no records"),
            Self::InvalidOutputPath(path) => {
                write!(
                    formatter,
                    "output path has no file name: {}",
                    path.display()
                )
            }
            Self::CreateOutputDirectory { path, source } => write!(
                formatter,
                "failed to create output directory {}: {source}",
                path.display()
            ),
            Self::CreateTemporary { path, source } => write!(
                formatter,
                "failed to create temporary output {}: {source}",
                path.display()
            ),
            Self::TemporaryNameExhausted(path) => write!(
                formatter,
                "failed to allocate a temporary name for {}",
                path.display()
            ),
            Self::EncodeRecord { line, source } => {
                write!(
                    formatter,
                    "failed to encode output record for line {line}: {source}"
                )
            }
            Self::WriteRecord { line, source } => {
                write!(
                    formatter,
                    "failed to write output record for line {line}: {source}"
                )
            }
            Self::WriteTemporary { path, source } => write!(
                formatter,
                "failed to flush temporary output {}: {source}",
                path.display()
            ),
            Self::SyncTemporary { path, source } => write!(
                formatter,
                "failed to sync temporary output {}: {source}",
                path.display()
            ),
            Self::ReplaceOutput {
                temporary,
                destination,
                source,
            } => write!(
                formatter,
                "failed to replace {} with {}: {source}",
                destination.display(),
                temporary.display()
            ),
        }
    }
}

impl std::error::Error for ConversionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::OpenInput { source, .. }
            | Self::ReadInputLine { source, .. }
            | Self::CreateOutputDirectory { source, .. }
            | Self::CreateTemporary { source, .. }
            | Self::WriteRecord { source, .. }
            | Self::WriteTemporary { source, .. }
            | Self::SyncTemporary { source, .. }
            | Self::ReplaceOutput { source, .. } => Some(source),
            Self::InvalidRecord { source, .. } | Self::EncodeRecord { source, .. } => Some(source),
            Self::BlankLine(_)
            | Self::EmptySampleId(_)
            | Self::ControlCharacterInSampleId { .. }
            | Self::TooFewEdges { .. }
            | Self::DuplicateSampleId { .. }
            | Self::EmptyInput
            | Self::InvalidOutputPath(_)
            | Self::TemporaryNameExhausted(_) => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum CliAction {
    Run(CliArguments),
    Help,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CliArguments {
    input: PathBuf,
    output: PathBuf,
}

fn parse_cli<I, S>(arguments: I) -> Result<CliAction, CliError>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let arguments = arguments.into_iter().map(Into::into).collect::<Vec<_>>();
    if arguments
        .iter()
        .any(|argument| argument == OsStr::new("--help") || argument == OsStr::new("-h"))
    {
        return Ok(CliAction::Help);
    }

    let mut input = None;
    let mut output = None;
    let mut index = 0;
    while index < arguments.len() {
        let flag = arguments[index]
            .to_str()
            .ok_or_else(|| CliError::NonUnicodeFlag(arguments[index].clone()))?;
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| CliError::MissingValue(flag.into()))?;
        let slot = match flag {
            "--input" => &mut input,
            "--output" => &mut output,
            _ => return Err(CliError::UnknownArgument(flag.into())),
        };
        if slot.replace(PathBuf::from(value)).is_some() {
            return Err(CliError::DuplicateArgument(flag.into()));
        }
        index += 2;
    }

    Ok(CliAction::Run(CliArguments {
        input: input.ok_or(CliError::MissingRequired("--input"))?,
        output: output.ok_or(CliError::MissingRequired("--output"))?,
    }))
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum CliError {
    NonUnicodeFlag(OsString),
    UnknownArgument(String),
    MissingValue(String),
    DuplicateArgument(String),
    MissingRequired(&'static str),
}

impl Display for CliError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonUnicodeFlag(flag) => write!(formatter, "argument {flag:?} is not Unicode"),
            Self::UnknownArgument(argument) => write!(formatter, "unknown argument {argument}"),
            Self::MissingValue(argument) => write!(formatter, "missing value for {argument}"),
            Self::DuplicateArgument(argument) => {
                write!(formatter, "{argument} was provided more than once")
            }
            Self::MissingRequired(argument) => write!(formatter, "missing required {argument}"),
        }
    }
}

impl std::error::Error for CliError {}

#[cfg(test)]
mod tests {
    use super::*;
    use ewr_research_protocol::read_dataset_jsonl;
    use std::io::Cursor;

    static TEST_DIRECTORY_ID: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn conversion_is_deterministic_and_ignores_only_known_legacy_fields() {
        let input = concat!(
            "{\"edges\":[3,5,8],\"end_time\":17,\"manifest_id\":\"train:2\",",
            "\"original_trip_id\":\"trip-2\",\"source_index\":9,\"start_time\":11}\n",
            "{\"manifest_id\":\"train:1\",\"edges\":[13,21],\"start_time\":null}\n"
        );
        let mut output = Vec::new();
        assert_eq!(convert_jsonl(Cursor::new(input), &mut output).unwrap(), 2);
        assert_eq!(
            String::from_utf8(output.clone()).unwrap(),
            concat!(
                "{\"sample_id\":\"train:2\",\"original_edge_ids\":[3,5,8]}\n",
                "{\"sample_id\":\"train:1\",\"original_edge_ids\":[13,21]}\n"
            )
        );
        let records = read_dataset_jsonl(Cursor::new(output)).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].sample_id, "train:2");
    }

    #[test]
    fn conversion_rejects_missing_wrong_unknown_empty_short_and_duplicate_rows() {
        for input in [
            "",
            "\n",
            "{\"edges\":[1,2]}\n",
            "{\"manifest_id\":\"x\"}\n",
            "{\"manifest_id\":1,\"edges\":[1,2]}\n",
            "{\"manifest_id\":\"x\",\"edges\":[1,\"2\"]}\n",
            "{\"manifest_id\":\"x\",\"edges\":[1,2],\"method\":\"old\"}\n",
            "{\"manifest_id\":\"  \",\"edges\":[1,2]}\n",
            "{\"manifest_id\":\"x\",\"edges\":[]}\n",
            "{\"manifest_id\":\"x\",\"edges\":[1]}\n",
            concat!(
                "{\"manifest_id\":\"x\",\"edges\":[1,2]}\n",
                "{\"manifest_id\":\"x\",\"edges\":[2,3]}\n"
            ),
        ] {
            assert!(
                convert_jsonl(Cursor::new(input), Vec::new()).is_err(),
                "{input:?}"
            );
        }
    }

    #[test]
    fn file_conversion_replaces_atomically_and_preserves_output_on_failure() {
        let directory = test_directory();
        std::fs::create_dir_all(&directory).unwrap();
        let input = directory.join("legacy.jsonl");
        let output = directory.join("nested/dataset.jsonl");
        std::fs::write(&input, "{\"manifest_id\":\"x\",\"edges\":[4,8,15]}\n").unwrap();
        assert_eq!(convert_file(&input, &output).unwrap(), 1);
        let expected = b"{\"sample_id\":\"x\",\"original_edge_ids\":[4,8,15]}\n";
        assert_eq!(std::fs::read(&output).unwrap(), expected);

        std::fs::write(&input, "{\"manifest_id\":\"broken\",\"edges\":[1]}\n").unwrap();
        assert!(convert_file(&input, &output).is_err());
        assert_eq!(std::fs::read(&output).unwrap(), expected);
        assert_eq!(
            std::fs::read_dir(output.parent().unwrap()).unwrap().count(),
            1
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn cli_accepts_exact_arguments_and_rejects_missing_duplicate_and_unknown() {
        assert_eq!(
            parse_cli(["--input", "legacy.jsonl", "--output", "dataset.jsonl"]).unwrap(),
            CliAction::Run(CliArguments {
                input: "legacy.jsonl".into(),
                output: "dataset.jsonl".into(),
            })
        );
        assert_eq!(parse_cli(["--help"]).unwrap(), CliAction::Help);
        assert!(matches!(
            parse_cli(["--input", "legacy.jsonl"]),
            Err(CliError::MissingRequired("--output"))
        ));
        assert!(matches!(
            parse_cli([
                "--input",
                "one.jsonl",
                "--input",
                "two.jsonl",
                "--output",
                "dataset.jsonl"
            ]),
            Err(CliError::DuplicateArgument(_))
        ));
        assert!(matches!(
            parse_cli([
                "--input",
                "legacy.jsonl",
                "--output",
                "dataset.jsonl",
                "--extra",
                "forbidden"
            ]),
            Err(CliError::UnknownArgument(_))
        ));
    }

    fn test_directory() -> PathBuf {
        let id = TEST_DIRECTORY_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "ewr-convert-legacy-dataset-test-{}-{id}",
            std::process::id()
        ))
    }
}
