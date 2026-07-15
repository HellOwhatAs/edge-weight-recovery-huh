use edge_weight_recovery::config::{ExperimentConfig, RunOptions};
use edge_weight_recovery::expanded_training::run_expanded_training;
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
    match ExperimentConfig::load(&options.config_path)? {
        ExperimentConfig::EdgeOnly(config) => {
            let outcome = run_training(&config, &options.output_dir)?;
            println!(
                "selected epoch {} with validation relative regret {:.8}; checkpoint {}",
                outcome.best_epoch,
                outcome.selection_value,
                outcome.checkpoint_path.display()
            );
        }
        ExperimentConfig::Expanded(config) => {
            let outcome = run_expanded_training(&config, &options.output_dir)?;
            println!(
                "selected update {} with validation objective {:.8}; checkpoint {}",
                outcome.best_update,
                outcome.selection_value,
                outcome.checkpoint_path.display()
            );
        }
    }
    Ok(())
}
