use crate::optimizer::AdamOptimizer;
use indicatif::MultiProgress;
use std::fs::OpenOptions;
use std::io::Write;

pub struct TrainingConfig {
    pub city: String,
    pub num_epochs: u64,
    pub patience: usize,
    pub log_path: String,
    pub best_weights_path: String,
    pub save_best_immediately: bool,
    pub learning_rate: f32,
    pub m: MultiProgress,
}

impl TrainingConfig {
    pub fn new(city: &str) -> Self {
        let city_s = city.to_string();
        Self {
            city: city_s.clone(),
            num_epochs: 4000,
            patience: 100, // Increase patience slightly for Adam
            log_path: "training.log".to_string(),
            best_weights_path: format!("{city}_best_weights.json"),
            save_best_immediately: true,
            learning_rate: 3000.0, // Aggressive learning rate for fast convergence
            m: MultiProgress::new(),
        }
    }

    // pub fn logg(&self, )
    pub fn log(&self, message: &str) {
        self.m.println(message).unwrap();
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)
            .unwrap();
        writeln!(file, "{}", message).unwrap();
    }
}

pub struct TrainingState {
    pub global_best_loss: usize,                      // All-time best loss
    pub global_best_weights: Vec<u32>,                // All-time best weights
    pub global_best_optimizer: Option<AdamOptimizer>, // Optimizer state at global best

    pub era_best_loss: usize, // Best loss in the current "era" (since last restart)
    pub era_best_weights: Vec<u32>, // Best weights in the current "era"

    pub stale_epochs: usize,
    pub restarts: usize,
}

impl TrainingState {
    pub fn new(initial_weights: &[u32]) -> Self {
        Self {
            global_best_loss: usize::MAX,
            global_best_weights: initial_weights.to_vec(),
            global_best_optimizer: None,

            era_best_loss: usize::MAX,
            era_best_weights: initial_weights.to_vec(),

            stale_epochs: 0,
            restarts: 0,
        }
    }

    pub fn update(
        &mut self,
        loss: usize,
        weights: &[u32],
        optimizer: &AdamOptimizer,
        config: &TrainingConfig,
    ) -> (bool, bool) {
        let is_global_best = if loss < self.global_best_loss {
            self.global_best_loss = loss;
            self.global_best_weights = weights.to_vec();
            self.global_best_optimizer = Some(optimizer.clone());
            if config.save_best_immediately {
                if let Ok(json) = serde_json::to_string(&self.global_best_weights) {
                    let _ = std::fs::write(&config.best_weights_path, json);
                }
            }
            true
        } else {
            false
        };

        let is_era_best = if loss < self.era_best_loss {
            self.era_best_loss = loss;
            self.era_best_weights = weights.to_vec();
            self.stale_epochs = 0;
            true
        } else {
            self.stale_epochs += 1;
            false
        };

        (is_global_best, is_era_best)
    }
}
