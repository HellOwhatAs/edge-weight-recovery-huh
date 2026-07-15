use edge_weight_recovery::config::{RunOptions, TrainingConfig};
use edge_weight_recovery::training::run_training;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let Some(options) = RunOptions::from_args()? else {
        return Ok(());
    };
    let config = TrainingConfig::load(&options.config_path)?;
    let outcome = run_training(&config, &options.output_dir)?;
    println!(
        "selected epoch {} with validation relative regret {:.8}; checkpoint {}",
        outcome.best_epoch,
        outcome.selection_value,
        outcome.checkpoint_path.display()
    );
    Ok(())
}
