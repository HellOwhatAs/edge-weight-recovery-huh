use crate::config::atomic_write;
use crate::data::{GraphData, TripPath, TripTime};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};

pub const TEMPORAL_CONFIG_SCHEMA_VERSION: u64 = 1;
pub const TIME_BUCKET_SCHEMA_VERSION: u64 = 1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TimeBucket {
    pub id: String,
    pub start_hour: u8,
    pub end_hour: u8,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TimeBucketSpec {
    raw: Value,
    pub timestamp_unit: String,
    pub timezone: String,
    pub utc_offset_seconds: i32,
    pub derived_from: String,
    pub buckets: Vec<TimeBucket>,
}

impl TimeBucketSpec {
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let value = serde_json::from_slice(&bytes)
            .map_err(|error| format!("failed to decode {}: {error}", path.display()))?;
        Self::from_value(value).map_err(|error| format!("{}: {error}", path.display()))
    }

    pub fn from_value(raw: Value) -> Result<Self, String> {
        reject_unknown_keys(
            &raw,
            "",
            &[
                "schema_version",
                "timestamp_unit",
                "timezone",
                "utc_offset_seconds",
                "derived_from",
                "buckets",
            ],
        )?;
        if require_u64(&raw, "/schema_version")? != TIME_BUCKET_SCHEMA_VERSION {
            return Err(format!(
                "schema_version must be {TIME_BUCKET_SCHEMA_VERSION}"
            ));
        }
        let timestamp_unit = require_str(&raw, "/timestamp_unit")?.to_string();
        if timestamp_unit != "unix_seconds" {
            return Err("timestamp_unit must be \"unix_seconds\"".to_string());
        }
        let timezone = require_str(&raw, "/timezone")?.to_string();
        if timezone != "Asia/Shanghai" {
            return Err("timezone must be \"Asia/Shanghai\"".to_string());
        }
        let utc_offset_seconds = require_i64(&raw, "/utc_offset_seconds")?;
        if utc_offset_seconds != 28_800 {
            return Err("utc_offset_seconds must be 28800 for Asia/Shanghai".to_string());
        }
        let derived_from = require_str(&raw, "/derived_from")?.to_string();
        if derived_from != "beijing_full_train_timestamp_audit" {
            return Err("derived_from must be \"beijing_full_train_timestamp_audit\"".to_string());
        }
        let bucket_values = raw
            .pointer("/buckets")
            .and_then(Value::as_array)
            .ok_or_else(|| "missing array /buckets".to_string())?;
        if !(2..=8).contains(&bucket_values.len()) {
            return Err("time bucket count must be between 2 and 8".to_string());
        }
        let mut buckets = Vec::with_capacity(bucket_values.len());
        for (index, bucket) in bucket_values.iter().enumerate() {
            reject_unknown_keys(bucket, "", &["id", "start_hour", "end_hour", "label"])
                .map_err(|error| format!("bucket {index}: {error}"))?;
            let id = require_safe_string(bucket, "/id", "bucket id")?;
            let start_hour = u8::try_from(require_u64(bucket, "/start_hour")?)
                .map_err(|_| format!("bucket {index} start_hour does not fit u8"))?;
            let end_hour = u8::try_from(require_u64(bucket, "/end_hour")?)
                .map_err(|_| format!("bucket {index} end_hour does not fit u8"))?;
            if start_hour >= end_hour || end_hour > 24 {
                return Err(format!(
                    "bucket {index} must satisfy 0 <= start_hour < end_hour <= 24"
                ));
            }
            if buckets
                .iter()
                .any(|existing: &TimeBucket| existing.id == id)
            {
                return Err(format!("duplicate time bucket id {id:?}"));
            }
            buckets.push(TimeBucket {
                id,
                start_hour,
                end_hour,
            });
        }
        if buckets.first().map(|bucket| bucket.start_hour) != Some(0)
            || buckets.last().map(|bucket| bucket.end_hour) != Some(24)
            || buckets
                .windows(2)
                .any(|pair| pair[0].end_hour != pair[1].start_hour)
        {
            return Err("time buckets must be ordered, contiguous, and cover [0,24)".to_string());
        }
        Ok(Self {
            raw,
            timestamp_unit,
            timezone,
            utc_offset_seconds: utc_offset_seconds as i32,
            derived_from,
            buckets,
        })
    }

    pub fn as_json(&self) -> &Value {
        &self.raw
    }

    pub fn bucket_index(&self, timestamp: u64) -> usize {
        let hour = local_hour(timestamp, self.utc_offset_seconds);
        self.buckets
            .iter()
            .position(|bucket| hour >= bucket.start_hour && hour < bucket.end_hour)
            .expect("validated buckets cover every local hour")
    }

    pub fn bucket_id(&self, timestamp: u64) -> &str {
        &self.buckets[self.bucket_index(timestamp)].id
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TemporalBaselineKind {
    Length,
    TripAverageTravelTime,
}

impl TemporalBaselineKind {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "length" => Ok(Self::Length),
            "trip_average_travel_time" => Ok(Self::TripAverageTravelTime),
            _ => Err(format!(
                "baseline kind must be \"length\" or \"trip_average_travel_time\", got {value:?}"
            )),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Length => "length",
            Self::TripAverageTravelTime => "trip_average_travel_time",
        }
    }
}

#[derive(Clone, Debug)]
pub struct TemporalTrainingConfig {
    raw: Value,
    pub run_id: String,
    pub city: String,
    pub train_variant: String,
    pub validation_variant: String,
    pub graph_representation: String,
    pub weight_lower_factor: f64,
    pub weight_upper_factor: f64,
    pub eta0: f64,
    pub global_lambda: f64,
    pub residual_lambda: f64,
    pub updates: u64,
    pub validation_every: u64,
    pub rayon_threads: usize,
    pub bucket_spec_path: PathBuf,
    pub bucket_spec_sha256: String,
    pub global_lower_factor: f64,
    pub global_upper_factor: f64,
    pub residual_lower: f64,
    pub residual_upper: f64,
    pub residual_eta_multiplier: f64,
    pub baseline_kind: TemporalBaselineKind,
    pub minimum_trip_speed_mps: Option<f64>,
    pub maximum_trip_speed_mps: Option<f64>,
    pub global_support_quantile: Option<f64>,
    pub bucket_support_quantile: Option<f64>,
}

impl TemporalTrainingConfig {
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let raw: Value = serde_json::from_slice(&bytes)
            .map_err(|error| format!("failed to decode {}: {error}", path.display()))?;
        Self::from_value(raw).map_err(|error| format!("{}: {error}", path.display()))
    }

    pub fn from_value(raw: Value) -> Result<Self, String> {
        reject_unknown_keys(
            &raw,
            "",
            &[
                "schema_version",
                "run_id",
                "description",
                "data",
                "graph",
                "optimizer",
                "oracle",
                "training",
                "runtime",
                "time_conditioning",
                "baseline",
                "test_policy",
            ],
        )?;
        reject_unknown_keys(
            &raw,
            "/data",
            &[
                "city",
                "train_variant",
                "validation_variant",
                "path_contract",
                "cycle_policy",
                "train_identity",
                "validation_identity",
            ],
        )?;
        for pointer in ["/data/train_identity", "/data/validation_identity"] {
            reject_unknown_keys(
                &raw,
                pointer,
                &[
                    "path",
                    "bytes",
                    "sha256",
                    "source_sha256",
                    "sample_count",
                    "seed",
                ],
            )?;
        }
        reject_unknown_keys(
            &raw,
            "/graph",
            &[
                "representation",
                "weight_lower_factor",
                "weight_upper_factor",
            ],
        )?;
        reject_unknown_keys(
            &raw,
            "/optimizer",
            &["kind", "eta0", "global_lambda", "residual_lambda"],
        )?;
        reject_unknown_keys(
            &raw,
            "/oracle",
            &["kind", "customization", "group_unique_od"],
        )?;
        reject_unknown_keys(&raw, "/training", &["updates", "validation_every"])?;
        reject_unknown_keys(&raw, "/runtime", &["rayon_threads"])?;
        reject_unknown_keys(
            &raw,
            "/time_conditioning",
            &[
                "kind",
                "bucket_spec_path",
                "bucket_spec_sha256",
                "global_lower_factor",
                "global_upper_factor",
                "residual_lower",
                "residual_upper",
                "residual_eta_multiplier",
            ],
        )?;
        reject_unknown_keys(
            &raw,
            "/baseline",
            &[
                "kind",
                "minimum_trip_speed_mps",
                "maximum_trip_speed_mps",
                "global_support_quantile",
                "bucket_support_quantile",
                "fixed_point_scale",
            ],
        )?;

        if require_u64(&raw, "/schema_version")? != TEMPORAL_CONFIG_SCHEMA_VERSION {
            return Err(format!(
                "schema_version must be {TEMPORAL_CONFIG_SCHEMA_VERSION}"
            ));
        }
        for (pointer, expected) in [
            (
                "/data/path_contract",
                "complete_original_edge_id_sequence_min_2_edges_with_trip_timestamps",
            ),
            ("/data/cycle_policy", "drop"),
            ("/graph/representation", "edge_transition_arcs"),
            ("/optimizer/kind", "relative_projected_subgradient"),
            ("/oracle/kind", "cch"),
            ("/oracle/customization", "full"),
            ("/time_conditioning/kind", "global_plus_bucket_residual"),
            ("/test_policy", "never_read"),
        ] {
            let actual = require_str(&raw, pointer)?;
            if actual != expected {
                return Err(format!("{pointer} must be {expected:?}, got {actual:?}"));
            }
        }
        if raw
            .pointer("/oracle/group_unique_od")
            .and_then(Value::as_bool)
            != Some(true)
        {
            return Err("/oracle/group_unique_od must be true".to_string());
        }

        let run_id = require_safe_string(&raw, "/run_id", "run_id")?;
        let city = require_safe_string(&raw, "/data/city", "city")?;
        let train_variant = require_safe_string(&raw, "/data/train_variant", "train_variant")?;
        let validation_variant =
            require_safe_string(&raw, "/data/validation_variant", "validation_variant")?;
        let graph_representation = require_str(&raw, "/graph/representation")?.to_string();
        let weight_lower_factor = finite_f64(&raw, "/graph/weight_lower_factor")?;
        let weight_upper_factor = finite_f64(&raw, "/graph/weight_upper_factor")?;
        if weight_lower_factor <= 0.0 || weight_lower_factor > 1.0 || weight_upper_factor < 1.0 {
            return Err("graph weight factors must satisfy 0 < lower <= 1 <= upper".to_string());
        }

        let eta0 = finite_f64(&raw, "/optimizer/eta0")?;
        let global_lambda = finite_f64(&raw, "/optimizer/global_lambda")?;
        let residual_lambda = finite_f64(&raw, "/optimizer/residual_lambda")?;
        if eta0 <= 0.0 || global_lambda < 0.0 || residual_lambda < 0.0 {
            return Err("eta0 must be positive and lambdas must be nonnegative".to_string());
        }
        let updates = require_u64(&raw, "/training/updates")?;
        let validation_every = require_u64(&raw, "/training/validation_every")?;
        if updates == 0 || validation_every == 0 || validation_every > updates {
            return Err(
                "updates and validation_every must be positive, with cadence <= updates"
                    .to_string(),
            );
        }
        let rayon_threads = usize::try_from(require_u64(&raw, "/runtime/rayon_threads")?)
            .map_err(|_| "rayon_threads does not fit usize".to_string())?;
        if rayon_threads == 0 {
            return Err("rayon_threads must be positive".to_string());
        }

        let bucket_spec_path =
            PathBuf::from(require_str(&raw, "/time_conditioning/bucket_spec_path")?);
        if bucket_spec_path.is_absolute()
            || bucket_spec_path
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            return Err("bucket_spec_path must be a safe repository-relative path".to_string());
        }
        let bucket_spec_sha256 = require_sha256(&raw, "/time_conditioning/bucket_spec_sha256")?;
        let global_lower_factor = finite_f64(&raw, "/time_conditioning/global_lower_factor")?;
        let global_upper_factor = finite_f64(&raw, "/time_conditioning/global_upper_factor")?;
        let residual_lower = finite_f64(&raw, "/time_conditioning/residual_lower")?;
        let residual_upper = finite_f64(&raw, "/time_conditioning/residual_upper")?;
        let residual_eta_multiplier =
            finite_f64(&raw, "/time_conditioning/residual_eta_multiplier")?;
        if global_lower_factor > 1.0
            || global_upper_factor < 1.0
            || residual_lower > 0.0
            || residual_upper < 0.0
            || residual_eta_multiplier <= 0.0
            || global_lower_factor + residual_lower + 1e-12 < weight_lower_factor
            || global_upper_factor + residual_upper - 1e-12 > weight_upper_factor
        {
            return Err(
                "global/residual bounds must contain their initial values and keep every effective multiplier inside graph bounds"
                    .to_string(),
            );
        }

        let baseline_kind = TemporalBaselineKind::parse(require_str(&raw, "/baseline/kind")?)?;
        let (
            minimum_trip_speed_mps,
            maximum_trip_speed_mps,
            global_support_quantile,
            bucket_support_quantile,
        ) = match baseline_kind {
            TemporalBaselineKind::Length => {
                for pointer in [
                    "/baseline/minimum_trip_speed_mps",
                    "/baseline/maximum_trip_speed_mps",
                    "/baseline/global_support_quantile",
                    "/baseline/bucket_support_quantile",
                    "/baseline/fixed_point_scale",
                ] {
                    if raw.pointer(pointer).is_some() {
                        return Err(format!(
                            "{pointer} is only valid for trip_average_travel_time"
                        ));
                    }
                }
                (None, None, None, None)
            }
            TemporalBaselineKind::TripAverageTravelTime => {
                if raw
                    .pointer("/baseline/fixed_point_scale")
                    .and_then(Value::as_str)
                    .is_some_and(|value| value != "train_network_mean_speed")
                {
                    return Err(
                        "/baseline/fixed_point_scale must be \"train_network_mean_speed\""
                            .to_string(),
                    );
                }
                let minimum = finite_f64(&raw, "/baseline/minimum_trip_speed_mps")?;
                let maximum = finite_f64(&raw, "/baseline/maximum_trip_speed_mps")?;
                let global_quantile = finite_f64(&raw, "/baseline/global_support_quantile")?;
                let bucket_quantile = finite_f64(&raw, "/baseline/bucket_support_quantile")?;
                if minimum <= 0.0
                    || minimum >= maximum
                    || maximum > 100.0
                    || !(0.0..=1.0).contains(&global_quantile)
                    || !(0.0..=1.0).contains(&bucket_quantile)
                {
                    return Err("invalid trip-speed or support-quantile settings".to_string());
                }
                (
                    Some(minimum),
                    Some(maximum),
                    Some(global_quantile),
                    Some(bucket_quantile),
                )
            }
        };

        Ok(Self {
            raw,
            run_id,
            city,
            train_variant,
            validation_variant,
            graph_representation,
            weight_lower_factor,
            weight_upper_factor,
            eta0,
            global_lambda,
            residual_lambda,
            updates,
            validation_every,
            rayon_threads,
            bucket_spec_path,
            bucket_spec_sha256,
            global_lower_factor,
            global_upper_factor,
            residual_lower,
            residual_upper,
            residual_eta_multiplier,
            baseline_kind,
            minimum_trip_speed_mps,
            maximum_trip_speed_mps,
            global_support_quantile,
            bucket_support_quantile,
        })
    }

    pub fn as_json(&self) -> &Value {
        &self.raw
    }

    pub fn load_bucket_spec(&self) -> Result<TimeBucketSpec, String> {
        let actual = sha256_file(&self.bucket_spec_path)?;
        if actual != self.bucket_spec_sha256 {
            return Err(format!(
                "{} SHA-256 mismatch: expected {}, got {actual}",
                self.bucket_spec_path.display(),
                self.bucket_spec_sha256
            ));
        }
        TimeBucketSpec::load(&self.bucket_spec_path)
    }
}

#[derive(Clone, Debug)]
pub struct BaselineModel {
    pub kind: TemporalBaselineKind,
    /// One original-edge baseline vector per time bucket. Values use the
    /// model's direct CCH units (length millimetres or a train-scaled multiple
    /// of travel milliseconds, with the recovery divisor logged explicitly).
    pub edge_weights_by_bucket: Vec<Vec<f64>>,
    pub diagnostics: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TemporalParameters {
    pub global_relative: Vec<f64>,
    pub bucket_residuals: Vec<Vec<f64>>,
}

impl TemporalParameters {
    pub fn initial(coordinate_count: usize, bucket_count: usize) -> Result<Self, String> {
        if coordinate_count == 0 || bucket_count == 0 {
            return Err("temporal parameters require coordinates and buckets".to_string());
        }
        Ok(Self {
            global_relative: vec![1.0; coordinate_count],
            bucket_residuals: vec![vec![0.0; coordinate_count]; bucket_count],
        })
    }

    pub fn effective_weights(
        &self,
        bucket: usize,
        initial_weights: &[f64],
    ) -> Result<Vec<f64>, String> {
        let residual = self
            .bucket_residuals
            .get(bucket)
            .ok_or_else(|| format!("time bucket {bucket} is out of bounds"))?;
        if initial_weights.len() != self.global_relative.len()
            || residual.len() != self.global_relative.len()
        {
            return Err("temporal parameter and baseline lengths differ".to_string());
        }
        self.global_relative
            .iter()
            .zip(residual)
            .zip(initial_weights)
            .enumerate()
            .map(|(coordinate, ((&global, &residual), &initial))| {
                let effective = global + residual;
                let weight = initial * effective;
                if initial <= 0.0
                    || !initial.is_finite()
                    || !effective.is_finite()
                    || effective <= 0.0
                    || !weight.is_finite()
                {
                    Err(format!(
                        "invalid temporal weight at bucket {bucket}, coordinate {coordinate}: initial={initial}, global={global}, residual={residual}"
                    ))
                } else {
                    Ok(weight)
                }
            })
            .collect()
    }

    pub fn validate_for_config(
        &self,
        config: &TemporalTrainingConfig,
        coordinate_count: usize,
        bucket_count: usize,
    ) -> Result<(), String> {
        if self.global_relative.len() != coordinate_count
            || self.bucket_residuals.len() != bucket_count
            || self
                .bucket_residuals
                .iter()
                .any(|residual| residual.len() != coordinate_count)
        {
            return Err("temporal parameter shape does not match graph and buckets".to_string());
        }
        for (coordinate, &global) in self.global_relative.iter().enumerate() {
            if !global.is_finite()
                || global < config.global_lower_factor
                || global > config.global_upper_factor
            {
                return Err(format!(
                    "global_relative[{coordinate}]={global} is outside [{}, {}]",
                    config.global_lower_factor, config.global_upper_factor
                ));
            }
        }
        for (bucket, residuals) in self.bucket_residuals.iter().enumerate() {
            for (coordinate, &residual) in residuals.iter().enumerate() {
                if !residual.is_finite()
                    || residual < config.residual_lower
                    || residual > config.residual_upper
                {
                    return Err(format!(
                        "bucket_residuals[{bucket}][{coordinate}]={residual} is outside [{}, {}]",
                        config.residual_lower, config.residual_upper
                    ));
                }
                let effective = self.global_relative[coordinate] + residual;
                if effective + 1e-12 < config.weight_lower_factor
                    || effective - 1e-12 > config.weight_upper_factor
                {
                    return Err(format!(
                        "effective multiplier at bucket {bucket}, coordinate {coordinate} is {effective}, outside [{}, {}]",
                        config.weight_lower_factor, config.weight_upper_factor
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalStepStats {
    pub global_eta: f64,
    pub residual_eta: f64,
    pub max_abs_global_delta: f64,
    pub max_abs_residual_delta: f64,
    pub projected_global_coordinates: usize,
    pub projected_residual_coordinates: usize,
}

/// Convex projected-subgradient optimizer for
/// `q_bucket = q_global + residual_bucket`.
#[derive(Clone, Debug)]
pub struct TemporalProjectedSubgradientOptimizer {
    eta0: f64,
    global_lambda: f64,
    residual_lambda: f64,
    residual_eta_multiplier: f64,
    completed_updates: u64,
}

impl TemporalProjectedSubgradientOptimizer {
    pub fn with_completed_updates(config: &TemporalTrainingConfig, completed_updates: u64) -> Self {
        Self {
            eta0: config.eta0,
            global_lambda: config.global_lambda,
            residual_lambda: config.residual_lambda,
            residual_eta_multiplier: config.residual_eta_multiplier,
            completed_updates,
        }
    }

    pub const fn completed_updates(&self) -> u64 {
        self.completed_updates
    }

    pub fn regularization(&self, parameters: &TemporalParameters) -> Result<f64, String> {
        let coordinate_count = parameters.global_relative.len();
        let bucket_count = parameters.bucket_residuals.len();
        if coordinate_count == 0
            || bucket_count == 0
            || parameters
                .bucket_residuals
                .iter()
                .any(|residuals| residuals.len() != coordinate_count)
        {
            return Err("invalid temporal parameter shape for regularization".to_string());
        }
        let global_squared = parameters
            .global_relative
            .iter()
            .map(|&value| (value - 1.0).powi(2))
            .sum::<f64>();
        let residual_squared = parameters
            .bucket_residuals
            .iter()
            .flat_map(|residuals| residuals.iter())
            .map(|&value| value.powi(2))
            .sum::<f64>();
        let penalty = self.global_lambda * global_squared / (2.0 * coordinate_count as f64)
            + self.residual_lambda * residual_squared
                / (2.0 * coordinate_count as f64 * bucket_count as f64);
        if penalty.is_finite() {
            Ok(penalty)
        } else {
            Err("temporal regularization is not finite".to_string())
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        parameters: &mut TemporalParameters,
        initial_weights_by_bucket: &[Vec<f64>],
        observed_by_bucket: &[Vec<u64>],
        predicted_by_bucket: &[Vec<u64>],
        sample_count: u64,
        config: &TemporalTrainingConfig,
    ) -> Result<TemporalStepStats, String> {
        let coordinate_count = parameters.global_relative.len();
        let bucket_count = parameters.bucket_residuals.len();
        parameters.validate_for_config(config, coordinate_count, bucket_count)?;
        if sample_count == 0
            || initial_weights_by_bucket.len() != bucket_count
            || observed_by_bucket.len() != bucket_count
            || predicted_by_bucket.len() != bucket_count
        {
            return Err("invalid temporal optimizer batch shape or sample count".to_string());
        }
        for bucket in 0..bucket_count {
            for (label, len) in [
                ("initial", initial_weights_by_bucket[bucket].len()),
                ("observed", observed_by_bucket[bucket].len()),
                ("predicted", predicted_by_bucket[bucket].len()),
            ] {
                if len != coordinate_count {
                    return Err(format!(
                        "bucket {bucket} {label} length {len} differs from {coordinate_count}"
                    ));
                }
            }
        }

        let global_eta = self.eta0 / (self.completed_updates as f64 + 1.0).sqrt();
        let residual_eta = global_eta * self.residual_eta_multiplier;
        let inverse_samples = 1.0 / sample_count as f64;
        let global_regularization_scale = self.global_lambda / coordinate_count as f64;
        let residual_regularization_scale =
            self.residual_lambda / (coordinate_count as f64 * bucket_count as f64);
        let mut next_global = Vec::with_capacity(coordinate_count);
        let mut next_residuals = vec![Vec::with_capacity(coordinate_count); bucket_count];
        let mut max_abs_global_delta = 0.0_f64;
        let mut max_abs_residual_delta = 0.0_f64;
        let mut projected_global_coordinates = 0usize;
        let mut projected_residual_coordinates = 0usize;

        for coordinate in 0..coordinate_count {
            let mut global_data_gradient = 0.0;
            for bucket in 0..bucket_count {
                let difference = signed_count_difference(
                    observed_by_bucket[bucket][coordinate],
                    predicted_by_bucket[bucket][coordinate],
                );
                let data_gradient =
                    initial_weights_by_bucket[bucket][coordinate] * difference * inverse_samples;
                global_data_gradient += data_gradient;
                let residual = parameters.bucket_residuals[bucket][coordinate];
                let residual_gradient = data_gradient + residual_regularization_scale * residual;
                if !residual_gradient.is_finite() {
                    return Err(format!(
                        "residual gradient at bucket {bucket}, coordinate {coordinate} is not finite"
                    ));
                }
                let unprojected = residual - residual_eta * residual_gradient;
                projected_residual_coordinates += usize::from(
                    unprojected < config.residual_lower || unprojected > config.residual_upper,
                );
                let candidate = unprojected.clamp(config.residual_lower, config.residual_upper);
                max_abs_residual_delta = max_abs_residual_delta.max((candidate - residual).abs());
                next_residuals[bucket].push(candidate);
            }
            let global = parameters.global_relative[coordinate];
            let global_gradient =
                global_data_gradient + global_regularization_scale * (global - 1.0);
            if !global_gradient.is_finite() {
                return Err(format!(
                    "global gradient at coordinate {coordinate} is not finite"
                ));
            }
            let unprojected = global - global_eta * global_gradient;
            projected_global_coordinates += usize::from(
                unprojected < config.global_lower_factor
                    || unprojected > config.global_upper_factor,
            );
            let candidate =
                unprojected.clamp(config.global_lower_factor, config.global_upper_factor);
            max_abs_global_delta = max_abs_global_delta.max((candidate - global).abs());
            next_global.push(candidate);
        }
        let next_clock = self
            .completed_updates
            .checked_add(1)
            .ok_or_else(|| "temporal optimizer update clock overflow".to_string())?;
        let candidate_parameters = TemporalParameters {
            global_relative: next_global,
            bucket_residuals: next_residuals,
        };
        candidate_parameters.validate_for_config(config, coordinate_count, bucket_count)?;
        *parameters = candidate_parameters;
        self.completed_updates = next_clock;
        Ok(TemporalStepStats {
            global_eta,
            residual_eta,
            max_abs_global_delta,
            max_abs_residual_delta,
            projected_global_coordinates,
            projected_residual_coordinates,
        })
    }
}

fn signed_count_difference(left: u64, right: u64) -> f64 {
    if left >= right {
        (left - right) as f64
    } else {
        -((right - left) as f64)
    }
}

pub const TEMPORAL_CHECKPOINT_SCHEMA_VERSION: u64 = 1;

#[derive(Clone, Debug, PartialEq)]
pub struct TemporalCheckpoint {
    pub graph_representation: String,
    pub completed_updates: u64,
    pub parameters: TemporalParameters,
    pub bucket_edge_baselines: Vec<Vec<f64>>,
    pub bucket_specification: Value,
    pub baseline_diagnostics: Value,
    pub configuration: Value,
    pub runtime_identity: Value,
    pub topology_identity: String,
}

impl TemporalCheckpoint {
    pub fn save(&self, output_dir: &Path) -> Result<PathBuf, String> {
        let path = output_dir.join("checkpoint.json");
        self.save_to(&path)?;
        Ok(path)
    }

    pub fn save_to(&self, path: &Path) -> Result<(), String> {
        self.validate()?;
        let value = serde_json::json!({
            "schema_version": TEMPORAL_CHECKPOINT_SCHEMA_VERSION,
            "model_kind": "global_plus_bucket_residual",
            "graph_representation": self.graph_representation,
            "completed_updates": self.completed_updates,
            "parameters": {
                "parameterization": "q_bucket = q_global + residual_bucket",
                "global_relative": self.parameters.global_relative,
                "bucket_residuals": self.parameters.bucket_residuals,
            },
            "bucket_edge_baselines": self.bucket_edge_baselines,
            "bucket_specification": self.bucket_specification,
            "baseline_diagnostics": self.baseline_diagnostics,
            "configuration": self.configuration,
            "runtime_identity": self.runtime_identity,
            "topology_identity": self.topology_identity,
        });
        let bytes = serde_json::to_vec(&value)
            .map_err(|error| format!("failed to encode temporal checkpoint: {error}"))?;
        atomic_write(path, &bytes)
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|error| format!("failed to decode {}: {error}", path.display()))?;
        if value.pointer("/schema_version").and_then(Value::as_u64)
            != Some(TEMPORAL_CHECKPOINT_SCHEMA_VERSION)
            || value.pointer("/model_kind").and_then(Value::as_str)
                != Some("global_plus_bucket_residual")
        {
            return Err(format!(
                "{} is not a schema-{TEMPORAL_CHECKPOINT_SCHEMA_VERSION} temporal checkpoint",
                path.display()
            ));
        }
        let global_relative = f64_vector(&value, "/parameters/global_relative")?;
        let bucket_residuals = f64_matrix(&value, "/parameters/bucket_residuals")?;
        let checkpoint = Self {
            graph_representation: require_str(&value, "/graph_representation")?.to_string(),
            completed_updates: require_u64(&value, "/completed_updates")?,
            parameters: TemporalParameters {
                global_relative,
                bucket_residuals,
            },
            bucket_edge_baselines: f64_matrix(&value, "/bucket_edge_baselines")?,
            bucket_specification: value
                .pointer("/bucket_specification")
                .cloned()
                .ok_or_else(|| "checkpoint lacks /bucket_specification".to_string())?,
            baseline_diagnostics: value
                .pointer("/baseline_diagnostics")
                .cloned()
                .ok_or_else(|| "checkpoint lacks /baseline_diagnostics".to_string())?,
            configuration: value
                .pointer("/configuration")
                .cloned()
                .ok_or_else(|| "checkpoint lacks /configuration".to_string())?,
            runtime_identity: value
                .pointer("/runtime_identity")
                .cloned()
                .ok_or_else(|| "checkpoint lacks /runtime_identity".to_string())?,
            topology_identity: require_str(&value, "/topology_identity")?.to_string(),
        };
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    fn validate(&self) -> Result<(), String> {
        if self.graph_representation != "edge_transition_arcs" {
            return Err("temporal checkpoint must use edge_transition_arcs".to_string());
        }
        let coordinates = self.parameters.global_relative.len();
        let buckets = self.parameters.bucket_residuals.len();
        let baseline_edges = self.bucket_edge_baselines.first().map_or(0, Vec::len);
        if coordinates == 0
            || buckets == 0
            || self
                .parameters
                .global_relative
                .iter()
                .any(|value| !value.is_finite())
            || self.parameters.bucket_residuals.iter().any(|residuals| {
                residuals.len() != coordinates || residuals.iter().any(|value| !value.is_finite())
            })
            || self.bucket_edge_baselines.len() != buckets
            || self.bucket_edge_baselines.iter().any(|weights| {
                weights.len() != baseline_edges
                    || weights.is_empty()
                    || weights
                        .iter()
                        .any(|value| !value.is_finite() || *value <= 0.0)
            })
        {
            return Err("temporal checkpoint has invalid parameter or baseline shape".to_string());
        }
        if self.topology_identity.is_empty() {
            return Err("temporal checkpoint topology identity is empty".to_string());
        }
        Ok(())
    }
}

fn f64_vector(value: &Value, pointer: &str) -> Result<Vec<f64>, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing array {pointer}"))?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_f64()
                .filter(|number| number.is_finite())
                .ok_or_else(|| format!("{pointer}/{index} is not finite"))
        })
        .collect()
}

fn f64_matrix(value: &Value, pointer: &str) -> Result<Vec<Vec<f64>>, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing array {pointer}"))?
        .iter()
        .enumerate()
        .map(|(index, _)| f64_vector(value, &format!("{pointer}/{index}")))
        .collect()
}

/// Estimate all temporal baselines from training trajectories only.
pub fn estimate_baseline_model(
    graph: &GraphData,
    train_paths: &[TripPath],
    train_times: &[TripTime],
    bucket_spec: &TimeBucketSpec,
    config: &TemporalTrainingConfig,
) -> Result<BaselineModel, String> {
    if train_paths.len() != train_times.len() || train_paths.is_empty() {
        return Err("training paths and timestamps must be nonempty and aligned".to_string());
    }
    match config.baseline_kind {
        TemporalBaselineKind::Length => {
            let edge_weights = graph
                .baseline_weights
                .iter()
                .map(|&weight| weight as f64)
                .collect::<Vec<_>>();
            Ok(BaselineModel {
                kind: TemporalBaselineKind::Length,
                edge_weights_by_bucket: vec![edge_weights; bucket_spec.buckets.len()],
                diagnostics: serde_json::json!({
                    "kind": "length",
                    "units": "millimetres",
                    "estimated_from": "fixed_map_edge_lengths",
                    "validation_used": false,
                    "test_used": false,
                }),
            })
        }
        TemporalBaselineKind::TripAverageTravelTime => {
            estimate_travel_time_baseline(graph, train_paths, train_times, bucket_spec, config)
        }
    }
}

fn estimate_travel_time_baseline(
    graph: &GraphData,
    train_paths: &[TripPath],
    train_times: &[TripTime],
    bucket_spec: &TimeBucketSpec,
    config: &TemporalTrainingConfig,
) -> Result<BaselineModel, String> {
    let minimum_speed = config
        .minimum_trip_speed_mps
        .expect("travel-time config validated");
    let maximum_speed = config
        .maximum_trip_speed_mps
        .expect("travel-time config validated");
    let edge_count = graph.baseline_weights.len();
    let bucket_count = bucket_spec.buckets.len();
    let mut global_speed_sums = vec![0.0; edge_count];
    let mut global_support = vec![0u64; edge_count];
    let mut bucket_speed_sums = vec![vec![0.0; edge_count]; bucket_count];
    let mut bucket_support = vec![vec![0u64; edge_count]; bucket_count];
    let mut raw_speeds = Vec::with_capacity(train_paths.len());
    let mut clipped_speeds = Vec::with_capacity(train_paths.len());
    let mut clipped_low = 0usize;
    let mut clipped_high = 0usize;

    for (trip, &time) in train_paths.iter().zip(train_times) {
        let duration = time.duration_seconds().ok_or_else(|| {
            format!(
                "accepted training trip has invalid interval {}..{}",
                time.start_time, time.end_time
            )
        })? as f64;
        let length_metres = trip.1.iter().try_fold(0.0, |sum, &edge| {
            graph
                .baseline_weights
                .get(edge)
                .map(|&millimetres| sum + millimetres as f64 / 1_000.0)
                .ok_or_else(|| format!("training path edge {edge} is out of bounds"))
        })?;
        let raw_speed = length_metres / duration;
        if !raw_speed.is_finite() || raw_speed <= 0.0 {
            return Err(format!(
                "training trip has invalid average speed {raw_speed}"
            ));
        }
        raw_speeds.push(raw_speed);
        clipped_low += usize::from(raw_speed < minimum_speed);
        clipped_high += usize::from(raw_speed > maximum_speed);
        let speed = raw_speed.clamp(minimum_speed, maximum_speed);
        clipped_speeds.push(speed);
        let bucket = bucket_spec.bucket_index(time.start_time);
        for &edge in &trip.1 {
            global_speed_sums[edge] += speed;
            global_support[edge] = global_support[edge]
                .checked_add(1)
                .ok_or_else(|| "global road support overflow".to_string())?;
            bucket_speed_sums[bucket][edge] += speed;
            bucket_support[bucket][edge] = bucket_support[bucket][edge]
                .checked_add(1)
                .ok_or_else(|| "bucket road support overflow".to_string())?;
        }
    }

    let network_speed = clipped_speeds.iter().sum::<f64>() / clipped_speeds.len() as f64;
    let positive_global_support = global_support
        .iter()
        .copied()
        .filter(|&count| count > 0)
        .collect::<Vec<_>>();
    let positive_bucket_support = bucket_support
        .iter()
        .flat_map(|counts| counts.iter().copied())
        .filter(|&count| count > 0)
        .collect::<Vec<_>>();
    let global_prior_count = support_quantile(
        positive_global_support.clone(),
        config
            .global_support_quantile
            .expect("travel-time config validated"),
    )? as f64;
    let bucket_quantile_count = support_quantile(
        positive_bucket_support.clone(),
        config
            .bucket_support_quantile
            .expect("travel-time config validated"),
    )? as f64;
    // The bucket prior is never weaker than the train-derived road-global
    // prior, which makes low-support road-time cells conservatively collapse
    // to their road-global estimate.
    let bucket_prior_count = bucket_quantile_count.max(global_prior_count);

    let road_global_speeds = global_speed_sums
        .iter()
        .zip(&global_support)
        .map(|(&sum, &support)| {
            (sum + global_prior_count * network_speed) / (support as f64 + global_prior_count)
        })
        .collect::<Vec<_>>();
    let mut edge_weights_by_bucket = Vec::with_capacity(bucket_count);
    for bucket in 0..bucket_count {
        let mut weights = Vec::with_capacity(edge_count);
        for edge in 0..edge_count {
            let speed = (bucket_speed_sums[bucket][edge]
                + bucket_prior_count * road_global_speeds[edge])
                / (bucket_support[bucket][edge] as f64 + bucket_prior_count);
            let length_metres = graph.baseline_weights[edge] as f64 / 1_000.0;
            let travel_milliseconds = length_metres / speed * 1_000.0;
            // One train-derived positive global scale leaves every shortest
            // path unchanged, aligns the numerical magnitude with the length
            // baseline, and reduces u32 fixed-point quantization error. Divide
            // a stored weight by `network_speed` to recover milliseconds.
            let scaled_travel_milliseconds = travel_milliseconds * network_speed;
            if !scaled_travel_milliseconds.is_finite() || scaled_travel_milliseconds <= 0.0 {
                return Err(format!(
                    "road {edge}, bucket {bucket} has invalid travel-time baseline {scaled_travel_milliseconds}"
                ));
            }
            weights.push(scaled_travel_milliseconds.max(1.0));
        }
        edge_weights_by_bucket.push(weights);
    }

    let bucket_support_diagnostics = bucket_spec
        .buckets
        .iter()
        .enumerate()
        .map(|(bucket, definition)| {
            serde_json::json!({
                "id": definition.id,
                "nonzero_road_cells": bucket_support[bucket].iter().filter(|&&count| count > 0).count(),
                "zero_road_cells": bucket_support[bucket].iter().filter(|&&count| count == 0).count(),
                "support_quantiles": support_summary(&bucket_support[bucket]),
            })
        })
        .collect::<Vec<_>>();
    Ok(BaselineModel {
        kind: TemporalBaselineKind::TripAverageTravelTime,
        edge_weights_by_bucket,
        diagnostics: serde_json::json!({
            "kind": "trip_average_travel_time",
            "direct_weight_units": "train_network_mean_speed_scaled_milliseconds",
            "millisecond_recovery_divisor": network_speed,
            "fixed_point_scale": network_speed,
            "fixed_point_scale_effect": "positive global scaling only; route ordering and physical travel-time ratios are unchanged",
            "source": "complete training road sequence plus whole-trip start/end timestamps",
            "interpretation_warning": "whole-trip average speeds are proxies, not observed per-edge speeds",
            "trip_count": train_paths.len(),
            "raw_trip_average_speed_mps": f64_summary(&raw_speeds),
            "clipped_trip_average_speed_mps": f64_summary(&clipped_speeds),
            "minimum_trip_speed_mps": minimum_speed,
            "maximum_trip_speed_mps": maximum_speed,
            "clipped_low": clipped_low,
            "clipped_high": clipped_high,
            "network_mean_clipped_speed_mps": network_speed,
            "road_global_support": support_summary(&global_support),
            "global_support_quantile": config.global_support_quantile,
            "global_prior_count": global_prior_count,
            "bucket_support_quantile": config.bucket_support_quantile,
            "bucket_quantile_count": bucket_quantile_count,
            "bucket_prior_count": bucket_prior_count,
            "bucket_support": bucket_support_diagnostics,
            "smoothing": "road-bucket speed -> road-global speed -> network trip-average speed",
            "estimated_from_split": "train",
            "validation_used": false,
            "test_used": false,
        }),
    })
}

pub fn local_hour(timestamp: u64, utc_offset_seconds: i32) -> u8 {
    let seconds = timestamp as i128 + utc_offset_seconds as i128;
    seconds.div_euclid(3_600).rem_euclid(24) as u8
}

pub fn local_unix_day(timestamp: u64, utc_offset_seconds: i32) -> i64 {
    let seconds = timestamp as i128 + utc_offset_seconds as i128;
    seconds.div_euclid(86_400) as i64
}

pub fn civil_date_from_unix_day(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (year, month as u32, day as u32)
}

fn support_quantile(mut values: Vec<u64>, quantile: f64) -> Result<u64, String> {
    if values.is_empty() {
        return Err("cannot derive a smoothing prior from empty support".to_string());
    }
    values.sort_unstable();
    let index = ((values.len() - 1) as f64 * quantile).round() as usize;
    Ok(values[index].max(1))
}

fn support_summary(values: &[u64]) -> Value {
    let positive = values
        .iter()
        .copied()
        .filter(|&value| value > 0)
        .collect::<Vec<_>>();
    if positive.is_empty() {
        return serde_json::json!({"positive": 0, "zero": values.len()});
    }
    serde_json::json!({
        "positive": positive.len(),
        "zero": values.len() - positive.len(),
        "minimum_positive": support_quantile(positive.clone(), 0.0).expect("nonempty"),
        "p25_positive": support_quantile(positive.clone(), 0.25).expect("nonempty"),
        "p50_positive": support_quantile(positive.clone(), 0.5).expect("nonempty"),
        "p75_positive": support_quantile(positive.clone(), 0.75).expect("nonempty"),
        "p90_positive": support_quantile(positive.clone(), 0.9).expect("nonempty"),
        "maximum_positive": support_quantile(positive, 1.0).expect("nonempty"),
    })
}

fn f64_summary(values: &[f64]) -> Value {
    if values.is_empty() {
        return Value::Null;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let quantile =
        |probability: f64| sorted[((sorted.len() - 1) as f64 * probability).round() as usize];
    serde_json::json!({
        "minimum": quantile(0.0),
        "p01": quantile(0.01),
        "p10": quantile(0.1),
        "p50": quantile(0.5),
        "p90": quantile(0.9),
        "p99": quantile(0.99),
        "maximum": quantile(1.0),
        "mean": values.iter().sum::<f64>() / values.len() as f64,
    })
}

pub fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = std::fs::File::open(path)
        .map_err(|error| format!("failed to open {}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("failed to hash {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn reject_unknown_keys(value: &Value, pointer: &str, allowed: &[&str]) -> Result<(), String> {
    let object = value
        .pointer(pointer)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("missing object {pointer:?}"))?;
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(format!(
            "unknown field {}/{key}",
            pointer.trim_end_matches('/')
        ));
    }
    Ok(())
}

fn require_str<'a>(value: &'a Value, pointer: &str) -> Result<&'a str, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing string {pointer}"))
}

fn require_safe_string(value: &Value, pointer: &str, label: &str) -> Result<String, String> {
    let component = require_str(value, pointer)?;
    if component.is_empty()
        || component.contains('/')
        || component.contains("..")
        || !component
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_.-".contains(&byte))
    {
        return Err(format!("{label} is not a safe path component"));
    }
    Ok(component.to_string())
}

fn require_u64(value: &Value, pointer: &str) -> Result<u64, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("missing nonnegative integer {pointer}"))
}

fn require_i64(value: &Value, pointer: &str) -> Result<i64, String> {
    value
        .pointer(pointer)
        .and_then(Value::as_i64)
        .ok_or_else(|| format!("missing integer {pointer}"))
}

fn finite_f64(value: &Value, pointer: &str) -> Result<f64, String> {
    let number = value
        .pointer(pointer)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("missing number {pointer}"))?;
    if !number.is_finite() {
        return Err(format!("{pointer} must be finite"));
    }
    Ok(number)
}

fn require_sha256(value: &Value, pointer: &str) -> Result<String, String> {
    let digest = require_str(value, pointer)?;
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{pointer} must be a 64-digit SHA-256 hex digest"));
    }
    Ok(digest.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn bucket_value() -> Value {
        json!({
            "schema_version": 1,
            "timestamp_unit": "unix_seconds",
            "timezone": "Asia/Shanghai",
            "utc_offset_seconds": 28800,
            "derived_from": "beijing_full_train_timestamp_audit",
            "buckets": [
                {"id": "night", "label": "00:00-06:00", "start_hour": 0, "end_hour": 6},
                {"id": "day", "label": "06:00-24:00", "start_hour": 6, "end_hour": 24}
            ]
        })
    }

    fn temporal_config_value(baseline: &str) -> Value {
        let mut value = json!({
            "schema_version": 1,
            "run_id": "fixture",
            "data": {
                "city": "beijing",
                "train_variant": "all",
                "validation_variant": "fixed",
                "path_contract": "complete_original_edge_id_sequence_min_2_edges_with_trip_timestamps",
                "cycle_policy": "drop",
                "train_identity": {},
                "validation_identity": {}
            },
            "graph": {
                "representation": "edge_transition_arcs",
                "weight_lower_factor": 0.1,
                "weight_upper_factor": 10.0
            },
            "time_conditioning": {
                "kind": "global_plus_bucket_residual",
                "bucket_spec_path": "experiments/time_buckets.json",
                "bucket_spec_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "global_lower_factor": 0.6,
                "global_upper_factor": 9.5,
                "residual_lower": -0.5,
                "residual_upper": 0.5,
                "residual_eta_multiplier": 2.0
            },
            "baseline": {"kind": baseline},
            "optimizer": {
                "kind": "relative_projected_subgradient",
                "eta0": 0.1,
                "global_lambda": 0.0,
                "residual_lambda": 0.0
            },
            "oracle": {
                "kind": "cch",
                "customization": "full",
                "group_unique_od": true
            },
            "training": {"updates": 2, "validation_every": 1},
            "runtime": {"rayon_threads": 1},
            "test_policy": "never_read"
        });
        if baseline == "trip_average_travel_time" {
            value["baseline"]["minimum_trip_speed_mps"] = json!(1.0);
            value["baseline"]["maximum_trip_speed_mps"] = json!(30.0);
            value["baseline"]["global_support_quantile"] = json!(0.5);
            value["baseline"]["bucket_support_quantile"] = json!(0.75);
            value["baseline"]["fixed_point_scale"] = json!("train_network_mean_speed");
        }
        value
    }

    #[test]
    fn bucket_spec_uses_explicit_utc_plus_8_departure_time() {
        let spec = TimeBucketSpec::from_value(bucket_value()).unwrap();
        // 2009-05-03 23:52 UTC is 2009-05-04 07:52 in Beijing.
        assert_eq!(local_hour(1_241_394_720, 28_800), 7);
        assert_eq!(spec.bucket_id(1_241_394_720), "day");
        assert_eq!(
            civil_date_from_unix_day(local_unix_day(1_241_394_720, 28_800)),
            (2009, 5, 4)
        );
    }

    #[test]
    fn bucket_spec_rejects_gaps_and_hourly_explosion() {
        let mut gap = bucket_value();
        gap["buckets"][1]["start_hour"] = json!(7);
        assert!(TimeBucketSpec::from_value(gap).is_err());

        let mut too_many = bucket_value();
        too_many["buckets"] = Value::Array(
            (0..9)
                .map(|hour| {
                    json!({"id": format!("b{hour}"), "start_hour": hour, "end_hour": hour + 1})
                })
                .collect(),
        );
        assert!(TimeBucketSpec::from_value(too_many).is_err());
    }

    #[test]
    fn global_plus_bucket_residual_step_matches_the_convex_formula() {
        let config = TemporalTrainingConfig::from_value(temporal_config_value("length")).unwrap();
        let mut parameters = TemporalParameters::initial(1, 2).unwrap();
        let mut optimizer =
            TemporalProjectedSubgradientOptimizer::with_completed_updates(&config, 0);
        let step = optimizer
            .step(
                &mut parameters,
                &[vec![10.0], vec![20.0]],
                &[vec![2], vec![0]],
                &[vec![0], vec![2]],
                4,
                &config,
            )
            .unwrap();

        // Global data gradient is (10*2 + 20*(-2))/4 = -5.
        assert!((parameters.global_relative[0] - 1.5).abs() < 1e-12);
        // Residual eta is 0.2; raw candidates -1 and +2 hit the shared box.
        assert_eq!(parameters.bucket_residuals, vec![vec![-0.5], vec![0.5]]);
        assert_eq!(step.global_eta, 0.1);
        assert_eq!(step.residual_eta, 0.2);
        assert_eq!(step.projected_residual_coordinates, 2);
        assert_eq!(optimizer.completed_updates(), 1);
        assert_eq!(
            parameters.effective_weights(0, &[10.0]).unwrap(),
            vec![10.0]
        );
        assert_eq!(
            parameters.effective_weights(1, &[20.0]).unwrap(),
            vec![40.0]
        );
    }

    #[test]
    fn trip_average_baseline_shrinks_unseen_bucket_roads_without_validation() {
        let config =
            TemporalTrainingConfig::from_value(temporal_config_value("trip_average_travel_time"))
                .unwrap();
        let spec = TimeBucketSpec::from_value(bucket_value()).unwrap();
        let graph = GraphData {
            tail: vec![0, 1, 0, 2],
            head: vec![1, 3, 2, 3],
            baseline_weights: vec![1_000, 2_000, 1_000, 2_000],
            x: vec![0.0, 1.0, 1.0, 2.0],
            y: vec![0.0, 0.0, 1.0, 0.0],
        };
        // 22:00 UTC is 06:00 in Beijing (day bucket); 12 m / 3 s = 4 m/s.
        let paths = vec![((0, 3), vec![0, 1]), ((0, 3), vec![2, 3])];
        let times = vec![
            TripTime {
                start_time: 79_200,
                end_time: 79_203,
            },
            TripTime {
                start_time: 79_200,
                end_time: 79_203,
            },
        ];
        let baseline = estimate_baseline_model(&graph, &paths, &times, &spec, &config).unwrap();
        assert_eq!(baseline.edge_weights_by_bucket.len(), 2);
        assert!(
            baseline
                .edge_weights_by_bucket
                .iter()
                .flatten()
                .all(|weight| weight.is_finite() && *weight > 0.0)
        );
        assert_eq!(
            baseline.diagnostics.pointer("/validation_used"),
            Some(&Value::Bool(false))
        );
        // The unobserved night cell remains well-defined through both priors.
        assert!(baseline.edge_weights_by_bucket[0][0] > 0.0);
    }

    #[test]
    fn temporal_checkpoint_round_trips_parameters_and_bucket_baselines() {
        let checkpoint = TemporalCheckpoint {
            graph_representation: "edge_transition_arcs".to_string(),
            completed_updates: 3,
            parameters: TemporalParameters {
                global_relative: vec![1.1, 0.9],
                bucket_residuals: vec![vec![0.1, -0.1], vec![0.0, 0.2]],
            },
            bucket_edge_baselines: vec![vec![5.0, 6.0], vec![7.0, 8.0]],
            bucket_specification: bucket_value(),
            baseline_diagnostics: json!({"validation_used": false}),
            configuration: temporal_config_value("length"),
            runtime_identity: json!({"fixture": true}),
            topology_identity: "fnv1a64:fixture".to_string(),
        };
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("temporal-checkpoint-{nonce}.json"));
        checkpoint.save_to(&path).unwrap();
        assert_eq!(TemporalCheckpoint::load(&path).unwrap(), checkpoint);
        std::fs::remove_file(path).unwrap();
    }
}
