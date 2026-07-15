use crate::graph::CyclePolicy;
use std::fs::OpenOptions;
use std::io::Write;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SolverKind {
    ProjectedSubgradient,
    LegacyAdamShock,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetricUpdateMode {
    Partial,
    Full,
}

#[derive(Debug)]
pub struct TrainingConfig {
    pub city: String,
    pub num_epochs: u64,
    pub patience: usize,
    pub train_variant: String,
    pub validation_variant: String,
    pub test_variant: String,
    pub max_train_samples: Option<usize>,
    pub max_validation_samples: Option<usize>,
    pub max_test_samples: Option<usize>,
    pub run_test: bool,
    pub trim_boundary_edges: bool,
    pub cycle_policy: CyclePolicy,
    pub solver: SolverKind,
    pub metric_update_mode: MetricUpdateMode,
    pub eta0: f64,
    pub lambda: f64,
    pub q_min: f64,
    pub q_max: f64,
    pub quantization_scale: f64,
    pub adam_learning_rate: f32,
    pub random_seed: u64,
    pub eval_every: u64,
    pub output_prefix: String,
    pub log_path: String,
    pub best_weights_path: String,
    pub best_multipliers_path: String,
    pub checkpoint_path: String,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        let city = "beijing".to_string();
        let output_prefix = format!("{city}_projected");
        let mut config = Self {
            city,
            num_epochs: 20,
            patience: 20,
            train_variant: "small".to_string(),
            validation_variant: "small".to_string(),
            test_variant: "small".to_string(),
            max_train_samples: None,
            max_validation_samples: None,
            max_test_samples: None,
            run_test: false,
            trim_boundary_edges: false,
            cycle_policy: CyclePolicy::Drop,
            solver: SolverKind::ProjectedSubgradient,
            metric_update_mode: MetricUpdateMode::Full,
            eta0: 1e-5,
            lambda: 10_000_000.0,
            q_min: 0.1,
            q_max: 10.0,
            quantization_scale: 1.0,
            adam_learning_rate: 3_000.0,
            random_seed: 42,
            eval_every: 5,
            output_prefix,
            log_path: String::new(),
            best_weights_path: String::new(),
            best_multipliers_path: String::new(),
            checkpoint_path: String::new(),
        };
        config.refresh_output_paths();
        config
    }
}

impl TrainingConfig {
    pub fn from_args() -> Result<Option<Self>, String> {
        Self::from_iter(std::env::args().skip(1))
    }

    fn from_iter<I, S>(args: I) -> Result<Option<Self>, String>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut config = Self::default();
        let mut output_prefix_was_set = false;
        let args: Vec<String> = args.into_iter().map(Into::into).collect();
        let mut index = 0;
        while index < args.len() {
            let flag = &args[index];
            if flag == "--help" || flag == "-h" {
                print_help();
                return Ok(None);
            }
            if flag == "--trim-boundary-edges" {
                config.trim_boundary_edges = true;
                index += 1;
                continue;
            }
            if flag == "--keep-cycles" {
                config.cycle_policy = CyclePolicy::Keep;
                index += 1;
                continue;
            }
            if flag == "--run-test" {
                config.run_test = true;
                index += 1;
                continue;
            }
            let value = args
                .get(index + 1)
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--city" => config.city = value.clone(),
                "--epochs" => config.num_epochs = parse(value, flag)?,
                "--patience" => config.patience = parse(value, flag)?,
                "--train-variant" => config.train_variant = value.clone(),
                "--validation-variant" => config.validation_variant = value.clone(),
                "--test-variant" => config.test_variant = value.clone(),
                "--max-train-samples" => config.max_train_samples = Some(parse(value, flag)?),
                "--max-validation-samples" => {
                    config.max_validation_samples = Some(parse(value, flag)?)
                }
                "--max-test-samples" => config.max_test_samples = Some(parse(value, flag)?),
                "--solver" => {
                    config.solver = match value.as_str() {
                        "projected" => SolverKind::ProjectedSubgradient,
                        "adam-shock" => SolverKind::LegacyAdamShock,
                        _ => return Err(format!("unknown solver {value:?}")),
                    }
                }
                "--metric-update" => {
                    config.metric_update_mode = match value.as_str() {
                        "partial" => MetricUpdateMode::Partial,
                        "full" => MetricUpdateMode::Full,
                        _ => return Err(format!("unknown metric update mode {value:?}")),
                    }
                }
                "--eta0" => config.eta0 = parse(value, flag)?,
                "--lambda" => config.lambda = parse(value, flag)?,
                "--q-min" => config.q_min = parse(value, flag)?,
                "--q-max" => config.q_max = parse(value, flag)?,
                "--quantization-scale" => config.quantization_scale = parse(value, flag)?,
                "--adam-learning-rate" => config.adam_learning_rate = parse(value, flag)?,
                "--seed" => config.random_seed = parse(value, flag)?,
                "--eval-every" => config.eval_every = parse(value, flag)?,
                "--output-prefix" => {
                    config.output_prefix = value.clone();
                    output_prefix_was_set = true;
                }
                _ => return Err(format!("unknown argument {flag:?}; use --help")),
            }
            index += 2;
        }
        if !output_prefix_was_set {
            let solver = match config.solver {
                SolverKind::ProjectedSubgradient => "projected",
                SolverKind::LegacyAdamShock => "adam_shock",
            };
            config.output_prefix = format!("{}_{}", config.city, solver);
        }
        config.refresh_output_paths();
        config.validate()?;
        Ok(Some(config))
    }

    fn validate(&self) -> Result<(), String> {
        if self.num_epochs == 0 {
            return Err("--epochs must be at least 1".to_string());
        }
        if !self.quantization_scale.is_finite() || self.quantization_scale <= 0.0 {
            return Err("--quantization-scale must be finite and positive".to_string());
        }
        if !self.eta0.is_finite() || self.eta0 <= 0.0 {
            return Err("--eta0 must be finite and positive".to_string());
        }
        if !self.lambda.is_finite() || self.lambda < 0.0 {
            return Err("--lambda must be finite and non-negative".to_string());
        }
        if !self.q_min.is_finite()
            || !self.q_max.is_finite()
            || self.q_min <= 0.0
            || self.q_max < self.q_min
        {
            return Err("--q-min/--q-max must define a finite positive box".to_string());
        }
        if !self.adam_learning_rate.is_finite() || self.adam_learning_rate <= 0.0 {
            return Err("--adam-learning-rate must be finite and positive".to_string());
        }
        if self.patience == 0 {
            return Err("--patience must be at least 1".to_string());
        }
        for (flag, limit) in [
            ("--max-train-samples", self.max_train_samples),
            ("--max-validation-samples", self.max_validation_samples),
            ("--max-test-samples", self.max_test_samples),
        ] {
            if limit == Some(0) {
                return Err(format!("{flag} must be at least 1"));
            }
        }
        if self.output_prefix.is_empty() {
            return Err("--output-prefix must not be empty".to_string());
        }
        Ok(())
    }

    fn refresh_output_paths(&mut self) {
        self.log_path = format!("{}_training.log", self.output_prefix);
        self.best_weights_path = format!("{}_best_weights.json", self.output_prefix);
        self.best_multipliers_path = format!("{}_best_multipliers.json", self.output_prefix);
        self.checkpoint_path = format!("{}_checkpoint.json", self.output_prefix);
    }

    pub fn log(&self, message: &str) -> Result<(), String> {
        println!("{message}");
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)
            .map_err(|error| format!("failed to open {}: {error}", self.log_path))?;
        writeln!(file, "{message}")
            .map_err(|error| format!("failed to write {}: {error}", self.log_path))
    }
}

fn parse<T>(value: &str, flag: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse::<T>()
        .map_err(|error| format!("invalid value for {flag}: {error}"))
}

fn print_help() {
    println!(
        "edge-weight-recovery\n\
         Safe defaults use Beijing small, 20 epochs, projected subgradient, and drop cycles.\n\n\
         Core options:\n\
           --city CITY\n\
           --epochs N\n\
           --train-variant small|all|all_partial_1.0|...\n\
           --validation-variant small|all\n\
           --test-variant small|all\n\
           --max-train-samples N (likewise validation/test)\n\
           --solver projected|adam-shock\n\
           --metric-update partial|full\n\
           --eta0 FLOAT --lambda FLOAT --q-min FLOAT --q-max FLOAT\n\
           --quantization-scale FLOAT (also rescales data loss/gradient)\n\
           --seed N (legacy shock reproducibility)\n\
           --eval-every N (0 disables validation during training)\n\
           --run-test (evaluate test once after model selection)\n\
           --trim-boundary-edges (ablation; full paths are default)\n\
           --keep-cycles (ablation; cycles are dropped by default)\n\
           --output-prefix PATH"
    );
}

#[derive(Debug)]
pub struct TrainingState {
    pub best_selection_loss: f64,
    pub best_train_data_loss: f64,
    pub best_weights: Vec<u32>,
    pub best_multipliers: Vec<f64>,
    pub best_epoch: u64,
    pub stale_evaluations: usize,
}

impl TrainingState {
    pub fn new(initial_weights: &[u32], initial_multipliers: &[f64]) -> Self {
        Self {
            best_selection_loss: f64::INFINITY,
            best_train_data_loss: f64::INFINITY,
            best_weights: initial_weights.to_vec(),
            best_multipliers: initial_multipliers.to_vec(),
            best_epoch: 0,
            stale_evaluations: 0,
        }
    }

    pub fn update(
        &mut self,
        epoch: u64,
        selection_loss: f64,
        train_data_loss: f64,
        weights: &[u32],
        multipliers: &[f64],
    ) -> bool {
        if selection_loss < self.best_selection_loss {
            self.best_selection_loss = selection_loss;
            self.best_train_data_loss = train_data_loss;
            self.best_weights = weights.to_vec();
            self.best_multipliers = multipliers.to_vec();
            self.best_epoch = epoch;
            self.stale_evaluations = 0;
            true
        } else {
            self.stale_evaluations += 1;
            false
        }
    }

    pub fn save(&self, config: &TrainingConfig) -> Result<(), String> {
        let weights = serde_json::to_string(&self.best_weights)
            .map_err(|error| format!("failed to serialize best weights: {error}"))?;
        let multipliers = serde_json::to_string(&self.best_multipliers)
            .map_err(|error| format!("failed to serialize best multipliers: {error}"))?;
        let checkpoint = serde_json::json!({
            "schema_version": 1,
            "city": &config.city,
            "train_variant": &config.train_variant,
            "validation_variant": &config.validation_variant,
            "test_variant": &config.test_variant,
            "num_epochs": config.num_epochs,
            "eval_every": config.eval_every,
            "max_train_samples": config.max_train_samples,
            "max_validation_samples": config.max_validation_samples,
            "max_test_samples": config.max_test_samples,
            "run_test": config.run_test,
            "random_seed": config.random_seed,
            "solver": format!("{:?}", config.solver),
            "metric_update_mode": format!("{:?}", config.metric_update_mode),
            "cycle_policy": format!("{:?}", config.cycle_policy),
            "trim_boundary_edges": config.trim_boundary_edges,
            "eta0": config.eta0,
            "lambda": config.lambda,
            "q_min": config.q_min,
            "q_max": config.q_max,
            "quantization_scale": config.quantization_scale,
            "best_epoch": self.best_epoch,
            "selection_loss": self.best_selection_loss,
            "train_data_loss": self.best_train_data_loss,
            "weights": &self.best_weights,
            "multipliers": &self.best_multipliers,
        });
        let checkpoint = serde_json::to_vec(&checkpoint)
            .map_err(|error| format!("failed to serialize checkpoint: {error}"))?;

        // Compatibility arrays remain available, while the structured file is
        // the authoritative paired checkpoint with its experiment metadata.
        atomic_write(&config.best_weights_path, weights.as_bytes())?;
        atomic_write(&config.best_multipliers_path, multipliers.as_bytes())?;
        atomic_write(&config.checkpoint_path, &checkpoint)
    }
}

fn atomic_write(path: &str, contents: &[u8]) -> Result<(), String> {
    let temporary = format!("{path}.{}.tmp", std::process::id());
    std::fs::write(&temporary, contents)
        .map_err(|error| format!("failed to write {temporary}: {error}"))?;
    std::fs::rename(&temporary, path)
        .map_err(|error| format!("failed to atomically replace {path} with {temporary}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_short_experiment_configuration() {
        let config = TrainingConfig::from_iter([
            "--city",
            "porto",
            "--epochs",
            "3",
            "--max-train-samples",
            "128",
            "--solver",
            "adam-shock",
            "--metric-update",
            "full",
            "--run-test",
        ])
        .unwrap()
        .unwrap();
        assert_eq!(config.city, "porto");
        assert_eq!(config.num_epochs, 3);
        assert_eq!(config.max_train_samples, Some(128));
        assert_eq!(config.solver, SolverKind::LegacyAdamShock);
        assert_eq!(config.metric_update_mode, MetricUpdateMode::Full);
        assert!(config.run_test);
        assert_eq!(config.output_prefix, "porto_adam_shock");
    }

    #[test]
    fn state_keeps_the_true_best_checkpoint() {
        let mut state = TrainingState::new(&[10, 20], &[1.0, 1.0]);
        assert!(state.update(0, 5.0, 4.0, &[9, 21], &[0.9, 1.05]));
        assert!(!state.update(1, 8.0, 3.0, &[1, 99], &[0.1, 4.95]));
        assert_eq!(state.best_weights, vec![9, 21]);
        assert_eq!(state.best_epoch, 0);
    }

    #[test]
    fn rejects_unsafe_cli_boundaries() {
        assert!(TrainingConfig::from_iter(["--patience", "0"]).is_err());
        assert!(TrainingConfig::from_iter(["--adam-learning-rate", "NaN"]).is_err());
        assert!(TrainingConfig::from_iter(["--max-test-samples", "0"]).is_err());
    }
}
