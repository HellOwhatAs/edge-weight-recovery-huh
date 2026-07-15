use edge_weight_recovery::config::{ExperimentConfig, RunOptions};
use edge_weight_recovery::training::run_training;
use edge_weight_recovery::turn_training::run_turn_training;

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
        ExperimentConfig::TurnAware(config) => {
            let outcome = run_turn_training(&config, &options.output_dir)?;
            println!(
                "selected step {} with validation relative regret {:.8}; checkpoint {}",
                outcome.best_step,
                outcome.selection_value,
                outcome.checkpoint_path.display()
            );
        }
    }
    Ok(())
}
