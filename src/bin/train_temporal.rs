use edge_weight_recovery::config::RunOptions;
use edge_weight_recovery::temporal::TemporalTrainingConfig;
use edge_weight_recovery::temporal_training::run_temporal_training;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    if std::env::args()
        .skip(1)
        .any(|argument| argument == "--help" || argument == "-h")
    {
        println!(
            "Usage: train_temporal --config PATH --output-dir PATH [--resume CHECKPOINT]\n\n\
             Train a convex shared-global plus coarse time-bucket residual model.\n\
             Only configured train and validation data are read; test is never read."
        );
        return Ok(());
    }
    let Some(options) = RunOptions::from_args()? else {
        return Ok(());
    };
    let config = TemporalTrainingConfig::load(&options.config_path)?;
    let outcome = run_temporal_training(&config, &options.output_dir, options.resume.as_deref())?;
    println!(
        "completed {} time-conditioned updates; checkpoint {}",
        outcome.completed_updates,
        outcome.checkpoint_path.display()
    );
    Ok(())
}
