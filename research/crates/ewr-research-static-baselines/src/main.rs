use ewr_research_static_baselines::{CliAction, USAGE, parse_cli, run_predict, run_train};
use std::error::Error;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    match parse_cli(std::env::args_os().skip(1))? {
        CliAction::Help => println!("{USAGE}"),
        CliAction::Train(arguments) => {
            println!("{}", serde_json::to_string_pretty(&run_train(&arguments)?)?);
        }
        CliAction::Predict(arguments) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&run_predict(&arguments)?)?
            );
        }
    }
    Ok(())
}
