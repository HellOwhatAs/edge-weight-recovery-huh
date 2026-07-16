use crate::data::LoadedTrips;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;

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
            return Err("derived_from must name the full-train timestamp audit".to_string());
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
            let id = require_safe_component(bucket, "/id", "bucket id")?;
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

    pub fn bucket(&self, id: &str) -> Result<(usize, &TimeBucket), String> {
        self.buckets
            .iter()
            .enumerate()
            .find(|(_, bucket)| bucket.id == id)
            .ok_or_else(|| format!("time bucket {id:?} is not registered"))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BucketSelectionReport {
    pub bucket_id: String,
    pub source_accepted: usize,
    pub selected: usize,
}

/// Retain one departure-time partition after the common structural path
/// filter. This changes only the dataset presented to the static trainer; path
/// coordinates, objective, optimizer, and checkpoint shape remain untouched.
pub fn retain_departure_bucket(
    loaded: &mut LoadedTrips,
    spec: &TimeBucketSpec,
    bucket_id: &str,
) -> Result<BucketSelectionReport, String> {
    if loaded.paths.len() != loaded.times.len() {
        return Err("loaded paths and timestamps are not aligned".to_string());
    }
    let (target_bucket, _) = spec.bucket(bucket_id)?;
    let source_accepted = loaded.paths.len();
    let paths = std::mem::take(&mut loaded.paths);
    let times = std::mem::take(&mut loaded.times);
    let mut selected_paths = Vec::new();
    let mut selected_times = Vec::new();
    for (path, time) in paths.into_iter().zip(times) {
        if spec.bucket_index(time.start_time) == target_bucket {
            selected_paths.push(path);
            selected_times.push(time);
        }
    }
    loaded.paths = selected_paths;
    loaded.times = selected_times;
    Ok(BucketSelectionReport {
        bucket_id: bucket_id.to_string(),
        source_accepted,
        selected: loaded.paths.len(),
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

fn require_safe_component(value: &Value, pointer: &str, label: &str) -> Result<String, String> {
    let component = require_str(value, pointer)?;
    if component.is_empty()
        || component.contains('/')
        || component.contains("..")
        || !component
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_.-".contains(&byte))
    {
        return Err(format!("{label} is not a safe component"));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{PathValidationReport, TimestampEvidence, TripTime};
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

    #[test]
    fn specification_uses_explicit_utc_plus_8_departure_time() {
        let spec = TimeBucketSpec::from_value(bucket_value()).unwrap();
        assert_eq!(local_hour(1_241_394_720, 28_800), 7);
        assert_eq!(spec.bucket_id(1_241_394_720), "day");
        assert_eq!(
            civil_date_from_unix_day(local_unix_day(1_241_394_720, 28_800)),
            (2009, 5, 4)
        );
    }

    #[test]
    fn specification_rejects_gaps_and_too_many_buckets() {
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
    fn selecting_a_bucket_preserves_path_time_alignment_and_order() {
        let spec = TimeBucketSpec::from_value(bucket_value()).unwrap();
        let mut loaded = LoadedTrips {
            paths: vec![
                ((0, 2), vec![0, 1]),
                ((0, 2), vec![2, 3]),
                ((0, 2), vec![4, 5]),
            ],
            times: vec![
                TripTime {
                    start_time: 0,
                    end_time: 1,
                },
                TripTime {
                    start_time: 80_000,
                    end_time: 80_001,
                },
                TripTime {
                    start_time: 60_000,
                    end_time: 60_001,
                },
            ],
            report: PathValidationReport {
                accepted_samples: 3,
                ..PathValidationReport::default()
            },
            timestamp_evidence: TimestampEvidence::default(),
        };
        let report = retain_departure_bucket(&mut loaded, &spec, "day").unwrap();
        assert_eq!(report.source_accepted, 3);
        assert_eq!(report.selected, 2);
        assert_eq!(loaded.paths[0].1, vec![0, 1]);
        assert_eq!(loaded.paths[1].1, vec![2, 3]);
        assert_eq!(loaded.paths.len(), loaded.times.len());
    }
}
