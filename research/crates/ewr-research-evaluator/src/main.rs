use ewr_research_evaluator::{
    CliAction, USAGE, evaluate_files, parse_cli, write_summary, write_summary_atomic,
};

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
    let summary = evaluate_files(&arguments.dataset_jsonl, &arguments.predictions_jsonl)?;
    if let Some(output) = &arguments.output {
        write_summary_atomic(output, &summary)?;
    }
    write_summary(std::io::stdout().lock(), &summary)?;
    Ok(())
}
