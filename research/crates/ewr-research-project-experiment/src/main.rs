use ewr_research_project_experiment::{CliAction, USAGE, parse_cli, run_predict, run_train};
use std::error::Error;
use std::io::{self, Write};

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let action = parse_cli(std::env::args_os().skip(1))?;
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    match action {
        CliAction::Help => stdout.write_all(USAGE.as_bytes())?,
        CliAction::Train(arguments) => {
            serde_json::to_writer_pretty(&mut stdout, &run_train(&arguments)?)?;
            stdout.write_all(b"\n")?;
        }
        CliAction::Predict(arguments) => {
            serde_json::to_writer_pretty(&mut stdout, &run_predict(&arguments)?.diagnostics)?;
            stdout.write_all(b"\n")?;
        }
    }
    stdout.flush()?;
    Ok(())
}
