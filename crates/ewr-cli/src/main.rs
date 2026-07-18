use ewr_cli::{CliAction, USAGE, parse_args, run_train};
use std::process::ExitCode;

fn main() -> ExitCode {
    match parse_args(std::env::args_os().skip(1)) {
        Ok(CliAction::Help) => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(CliAction::Train(args)) => match run_train(&args) {
            Ok(summary) => {
                println!("{summary}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("error: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("error: {error}");
            eprint!("{USAGE}");
            ExitCode::from(2)
        }
    }
}
