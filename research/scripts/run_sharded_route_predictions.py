#!/usr/bin/env python3
"""Crash-safe, memory-bounded quality prediction for route baselines.

The model adapters predate the full-test experiment and materialize an entire
dataset plus every generated route in one process.  This runner gives each
adapter a small protocol-v1 manifest, commits one shard at a time, and joins
only validated prediction rows.  The pipeline measures the complete full-test
task with one outer GNU-time boundary.  Adapter and shard timings retained here
are decomposition evidence, not a separate latency/throughput result.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, BinaryIO, Iterable, Mapping, Sequence


DATASET_MANIFEST_SCHEMA = "ewr.dataset-manifest/v1"
DATASET_RECORD_SCHEMA = "ewr.dataset-record/v1"
PREDICTION_RECORD_SCHEMA = "ewr.prediction-record/v1"
BINDING_SCHEMA = "ewr.sharded-prediction-binding/v1"
PLAN_SCHEMA = "ewr.sharded-prediction-plan/v1"
INPUT_COMMIT_SCHEMA = "ewr.sharded-prediction-input-commit/v1"
SHARD_COMMIT_SCHEMA = "ewr.sharded-prediction-shard-commit/v1"
PROGRESS_SCHEMA = "ewr.sharded-prediction-progress/v1"
DIAGNOSTICS_SCHEMA = "ewr.sharded-quality-prediction-diagnostics/v1"
RUN_RECEIPT_SCHEMA = "ewr.sharded-prediction-run-receipt/v1"
COMPLETE_SCHEMA = "ewr.sharded-prediction-complete/v1"
RUNNER_VERSION = "1"
METHODS = {"drncs_lg", "drpk_static", "drp_tp"}
U32_MAX = 0xFFFF_FFFF
MAX_SHARD_SIZE = 8192

_STOP_SIGNAL: int | None = None


class ShardedPredictionError(RuntimeError):
    """Expected user-facing validation or execution error."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                if _STOP_SIGNAL is not None:
                    raise KeyboardInterrupt
                digest.update(chunk)
    except OSError as error:
        raise ShardedPredictionError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


def read_json(path: Path, context: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ShardedPredictionError(f"missing {context}: {path}") from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ShardedPredictionError(f"cannot read {context} {path}: {error}") from error


def require_exact_object(value: Any, keys: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ShardedPredictionError(
            f"{context} fields differ: expected {sorted(keys)}, got {actual}"
        )
    return value


def safe_records_path(manifest_path: Path, relative_value: Any) -> Path:
    if not isinstance(relative_value, str) or not relative_value:
        raise ShardedPredictionError("dataset records_file must be a nonempty string")
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ShardedPredictionError("dataset records_file must be a safe relative path")
    path = (manifest_path.parent / relative).resolve()
    if not path.is_relative_to(manifest_path.parent.resolve()) or not path.is_file():
        raise ShardedPredictionError(f"dataset records file is missing or unsafe: {path}")
    return path


def split_role(dataset_id: str) -> str:
    tokens = {
        token
        for token in dataset_id.lower().replace("/", "-").replace("_", "-").split("-")
        if token
    }
    roles: set[str] = set()
    if tokens & {"train", "training"}:
        roles.add("train")
    if tokens & {"validation", "valid", "val", "dev"}:
        roles.add("validation")
    if tokens & {"test", "testing"}:
        roles.add("test")
    if len(roles) != 1:
        raise ShardedPredictionError(
            f"dataset_id {dataset_id!r} must encode exactly one split role"
        )
    return next(iter(roles))


def load_dataset_descriptor(path: Path) -> dict[str, Any]:
    path = path.resolve()
    descriptor = require_exact_object(
        read_json(path, "dataset manifest"),
        {"schema", "dataset_id", "network_id", "records_schema", "records_file"},
        "dataset manifest",
    )
    if descriptor["schema"] != DATASET_MANIFEST_SCHEMA:
        raise ShardedPredictionError("unsupported dataset manifest schema")
    if descriptor["records_schema"] != DATASET_RECORD_SCHEMA:
        raise ShardedPredictionError("unsupported dataset records schema")
    for field in ("dataset_id", "network_id"):
        if not isinstance(descriptor[field], str) or not descriptor[field].strip():
            raise ShardedPredictionError(f"dataset {field} must be a nonempty string")
    if split_role(descriptor["dataset_id"]) != "test":
        raise ShardedPredictionError("full quality prediction accepts only a test split")
    records_path = safe_records_path(path, descriptor["records_file"])
    return {
        **descriptor,
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
        "records_path": str(records_path),
    }


def validate_dataset_row(raw: bytes, line_number: int) -> str:
    if not raw or not raw.strip():
        raise ShardedPredictionError(f"blank dataset row at line {line_number}")
    try:
        row = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ShardedPredictionError(
            f"invalid dataset JSON at line {line_number}: {error}"
        ) from error
    row = require_exact_object(
        row, {"sample_id", "original_edge_ids"}, f"dataset row {line_number}"
    )
    sample_id = row["sample_id"]
    edges = row["original_edge_ids"]
    if not isinstance(sample_id, str) or not sample_id:
        raise ShardedPredictionError(f"dataset row {line_number} has invalid sample_id")
    if not isinstance(edges, list) or len(edges) < 2:
        raise ShardedPredictionError(
            f"dataset row {line_number} must contain at least two raw edges"
        )
    if any(
        isinstance(edge, bool)
        or not isinstance(edge, int)
        or not 0 <= edge <= U32_MAX
        for edge in edges
    ):
        raise ShardedPredictionError(f"dataset row {line_number} has a non-u32 edge")
    return sample_id


def scan_dataset(records_path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    sample_ids: set[str] = set()
    first_sample_id: str | None = None
    last_sample_id: str | None = None
    samples = 0
    try:
        with records_path.open("rb") as source:
            for line_number, raw in enumerate(source, 1):
                digest.update(raw)
                sample_id = validate_dataset_row(raw, line_number)
                if sample_id in sample_ids:
                    raise ShardedPredictionError(f"duplicate sample_id {sample_id!r}")
                sample_ids.add(sample_id)
                first_sample_id = first_sample_id or sample_id
                last_sample_id = sample_id
                samples += 1
    except OSError as error:
        raise ShardedPredictionError(f"cannot scan dataset records: {error}") from error
    if samples == 0:
        raise ShardedPredictionError("dataset records are empty")
    return {
        "records_sha256": digest.hexdigest(),
        "samples": samples,
        "first_sample_id": first_sample_id,
        "last_sample_id": last_sample_id,
    }


def file_identity(path: Path, label: str) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ShardedPredictionError(f"missing {label}: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def map_identity(path: Path) -> dict[str, Any]:
    root = path.resolve()
    if not root.is_dir():
        raise ShardedPredictionError(f"map directory is missing: {root}")
    files: list[dict[str, Any]] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if child.is_symlink():
            raise ShardedPredictionError(f"map directory contains a symlink: {child}")
        if child.is_file() and child.name.startswith(("edges.", "nodes.")):
            identity = file_identity(child, "map file")
            files.append({"relative_path": child.name, **identity})
    if not files:
        raise ShardedPredictionError(f"map directory has no edge/node files: {root}")
    return {"path": str(root), "files": files, "identity_sha256": canonical_sha256(files)}


def preprocess_identity(path: Path, method: str) -> dict[str, Any]:
    root = path.resolve()
    if not root.is_dir():
        raise ShardedPredictionError(f"preprocess directory is missing: {root}")
    names = (
        ("configuration.json", "core-artifacts.json")
        if method == "drpk_static"
        else ("routing-configuration.json", "routing-artifacts.json")
    )
    files = [file_identity(root / name, "preprocess identity file") for name in names]
    configuration = read_json(root / names[0], "preprocess configuration")
    reference_name = "core_artifacts" if method == "drpk_static" else "routing_artifacts"
    if not isinstance(configuration, Mapping) or not isinstance(
        configuration.get(reference_name), Mapping
    ):
        raise ShardedPredictionError(
            f"preprocess configuration does not bind {reference_name}"
        )
    reference = configuration[reference_name]
    if (
        reference.get("path") != names[1]
        or reference.get("sha256") != files[1]["sha256"]
    ):
        raise ShardedPredictionError(
            "preprocess configuration and artifact-manifest hashes differ"
        )
    artifact_manifest = read_json(root / names[1], "preprocess artifact manifest")
    artifacts = artifact_manifest.get("artifacts") if isinstance(artifact_manifest, Mapping) else None
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ShardedPredictionError("preprocess artifact manifest has no artifacts")
    verified_artifacts: list[dict[str, Any]] = []
    for relative_value, expected in sorted(artifacts.items()):
        if not isinstance(relative_value, str) or not isinstance(expected, Mapping):
            raise ShardedPredictionError("preprocess artifact descriptor is malformed")
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ShardedPredictionError("preprocess artifact path is unsafe")
        observed = file_identity(root / relative, "preprocess artifact")
        if (
            observed["sha256"] != expected.get("sha256")
            or observed["bytes"] != expected.get("bytes")
        ):
            raise ShardedPredictionError(
                f"preprocess artifact differs from its manifest: {relative_value}"
            )
        verified_artifacts.append({"relative_path": relative_value, **observed})
    identity_material = {"files": files, "artifacts": verified_artifacts}
    return {
        "path": str(root),
        **identity_material,
        "identity_sha256": canonical_sha256(identity_material),
    }


def resolved_settings(args: argparse.Namespace) -> dict[str, Any]:
    seed = args.seed
    if seed is None:
        seed = 20260716 if args.method == "drncs_lg" else 20260718
    device = args.device
    if device is None:
        device = "cuda" if args.method == "drncs_lg" else (
            "cuda:0" if args.method == "drpk_static" else "cpu"
        )
    inference_batch_size = args.inference_batch_size
    if inference_batch_size is None:
        inference_batch_size = 1000 if args.method == "drncs_lg" else 32
    return {
        "seed": seed,
        "workers": args.workers,
        "device": device,
        "cuda_visible_devices": (
            args.cuda_visible_devices if device.startswith("cuda") else None
        ),
        "inference_batch_size": inference_batch_size,
        "max_steps": args.max_steps if args.method == "drncs_lg" else None,
        "warmup_repetitions": 0,
        "measured_repetitions": 1,
        "latency_samples": 0 if args.method == "drncs_lg" else None,
    }


def make_binding(
    args: argparse.Namespace, descriptor: Mapping[str, Any], scan: Mapping[str, Any]
) -> dict[str, Any]:
    runner_path = Path(__file__).resolve()
    artifacts: dict[str, Any]
    if args.method == "drncs_lg":
        artifacts = {
            "checkpoint": file_identity(args.checkpoint, "DRNCS-LG checkpoint"),
            "map": map_identity(args.map_dir),
        }
    else:
        artifacts = {"preprocess": preprocess_identity(args.preprocess_dir, args.method)}
        if args.method == "drpk_static":
            artifacts["checkpoint"] = file_identity(
                args.checkpoint, "DRPK-static checkpoint"
            )
    value = {
        "schema": BINDING_SCHEMA,
        "runner": {
            "version": RUNNER_VERSION,
            "path": str(runner_path),
            "sha256": sha256_file(runner_path),
        },
        "method": args.method,
        "adapter_executable": file_identity(
            args.adapter_executable, "adapter executable"
        ),
        "adapter_source": file_identity(args.adapter_source, "adapter source"),
        "source_revision": args.source_revision,
        "dataset": {
            "dataset_id": descriptor["dataset_id"],
            "network_id": descriptor["network_id"],
            "manifest_path": descriptor["manifest_path"],
            "manifest_sha256": descriptor["manifest_sha256"],
            "records_path": descriptor["records_path"],
            "records_sha256": scan["records_sha256"],
            "samples": scan["samples"],
            "first_sample_id": scan["first_sample_id"],
            "last_sample_id": scan["last_sample_id"],
        },
        "artifacts": artifacts,
        "configuration": {
            "shard_size": args.shard_size,
            **resolved_settings(args),
            "purpose": "full_test_quality_prediction_only",
            "efficiency_table_source": "current_full_test_single_pass_outer_wall",
        },
    }
    return {**value, "binding_sha256": canonical_sha256(value)}


def install_or_validate_binding(output_dir: Path, binding: Mapping[str, Any]) -> None:
    path = output_dir / "binding.json"
    if path.is_file():
        existing = read_json(path, "existing run binding")
        if existing != binding:
            raise ShardedPredictionError(
                "output directory belongs to different code, data, artifacts, or settings"
            )
        return
    unexpected = [child for child in output_dir.iterdir() if child.name != ".runner.lock"]
    if unexpected:
        raise ShardedPredictionError(
            f"refusing an unbound nonempty output directory: {output_dir}"
        )
    atomic_write_json(path, binding)


def chunks(source: BinaryIO, size: int) -> Iterable[list[bytes]]:
    current: list[bytes] = []
    for raw in source:
        current.append(raw)
        if len(current) == size:
            yield current
            current = []
    if current:
        yield current


def input_commit_valid(shard_dir: Path, expected: Mapping[str, Any]) -> bool:
    marker_path = shard_dir / "input.complete.json"
    if not marker_path.is_file():
        return False
    try:
        marker = read_json(marker_path, "shard input marker")
        if marker != expected:
            return False
        return all(
            sha256_file(shard_dir / name) == marker[field]
            for name, field in (
                ("records.jsonl", "records_sha256"),
                ("manifest.json", "manifest_sha256"),
            )
        )
    except ShardedPredictionError:
        return False


def materialize_shards(
    output_dir: Path,
    descriptor: Mapping[str, Any],
    scan: Mapping[str, Any],
    binding: Mapping[str, Any],
    shard_size: int,
) -> list[dict[str, Any]]:
    records_path = Path(str(descriptor["records_path"]))
    plan: list[dict[str, Any]] = []
    observed_samples = 0
    try:
        with records_path.open("rb") as source:
            for index, rows in enumerate(chunks(source, shard_size)):
                start = observed_samples
                end = start + len(rows)
                shard_dir = output_dir / "shards" / f"{index:06d}"
                records_bytes = b"".join(rows)
                records_sha256 = hashlib.sha256(records_bytes).hexdigest()
                first_id = validate_dataset_row(rows[0], start + 1)
                last_id = validate_dataset_row(rows[-1], end)
                shard_manifest = {
                    "schema": DATASET_MANIFEST_SCHEMA,
                    "dataset_id": f"{descriptor['dataset_id']}/shard-{index:06d}",
                    "network_id": descriptor["network_id"],
                    "records_schema": DATASET_RECORD_SCHEMA,
                    "records_file": "records.jsonl",
                }
                manifest_bytes = (
                    json.dumps(
                        shard_manifest,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                marker = {
                    "schema": INPUT_COMMIT_SCHEMA,
                    "binding_sha256": binding["binding_sha256"],
                    "shard_index": index,
                    "sample_start": start,
                    "sample_end_exclusive": end,
                    "samples": len(rows),
                    "first_sample_id": first_id,
                    "last_sample_id": last_id,
                    "records_sha256": records_sha256,
                    "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                }
                if not input_commit_valid(shard_dir, marker):
                    atomic_write_bytes(shard_dir / "records.jsonl", records_bytes)
                    atomic_write_bytes(shard_dir / "manifest.json", manifest_bytes)
                    atomic_write_json(shard_dir / "input.complete.json", marker)
                plan.append(
                    {
                        "index": index,
                        "directory": str(shard_dir),
                        "sample_start": start,
                        "sample_end_exclusive": end,
                        "samples": len(rows),
                        "first_sample_id": first_id,
                        "last_sample_id": last_id,
                        "records_sha256": records_sha256,
                        "manifest_sha256": marker["manifest_sha256"],
                    }
                )
                observed_samples = end
    except OSError as error:
        raise ShardedPredictionError(f"cannot materialize dataset shards: {error}") from error
    if observed_samples != scan["samples"]:
        raise ShardedPredictionError("dataset changed while its shards were materialized")
    if sha256_file(records_path) != scan["records_sha256"]:
        raise ShardedPredictionError("dataset records changed during materialization")
    plan_value = {
        "schema": PLAN_SCHEMA,
        "binding_sha256": binding["binding_sha256"],
        "dataset_records_sha256": scan["records_sha256"],
        "samples": scan["samples"],
        "shard_size": shard_size,
        "shards": plan,
    }
    atomic_write_json(output_dir / "plan.json", plan_value)
    return plan


def validate_prediction_rows(records: Path, predictions: Path) -> int:
    count = 0
    try:
        with records.open("rb") as truth, predictions.open("rb") as generated:
            while True:
                truth_raw = truth.readline()
                predicted_raw = generated.readline()
                if not truth_raw and not predicted_raw:
                    break
                if not truth_raw or not predicted_raw:
                    raise ShardedPredictionError("prediction row count differs from shard")
                sample_id = validate_dataset_row(truth_raw, count + 1)
                try:
                    row = json.loads(predicted_raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ShardedPredictionError(
                        f"invalid prediction JSON at row {count + 1}: {error}"
                    ) from error
                row = require_exact_object(
                    row,
                    {"sample_id", "predicted_edge_ids"},
                    f"prediction row {count + 1}",
                )
                edges = row["predicted_edge_ids"]
                if row["sample_id"] != sample_id:
                    raise ShardedPredictionError(
                        f"prediction sample order differs at row {count + 1}"
                    )
                if not isinstance(edges, list) or not edges or any(
                    isinstance(edge, bool)
                    or not isinstance(edge, int)
                    or not 0 <= edge <= U32_MAX
                    for edge in edges
                ):
                    raise ShardedPredictionError(
                        f"prediction row {count + 1} has invalid raw edges"
                    )
                count += 1
    except OSError as error:
        raise ShardedPredictionError(f"cannot validate predictions: {error}") from error
    return count


def nested(value: Any, dotted: str) -> Any:
    current = value
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ShardedPredictionError(f"adapter output is missing field {dotted!r}")
        current = current[part]
    return current


def validate_adapter_outputs(
    args: argparse.Namespace,
    binding: Mapping[str, Any],
    shard: Mapping[str, Any],
) -> dict[str, Any]:
    shard_dir = Path(str(shard["directory"]))
    predictions = shard_dir / "predictions.jsonl"
    diagnostics_path = shard_dir / "diagnostics.json"
    receipt_path = shard_dir / "run.json"
    samples = validate_prediction_rows(shard_dir / "records.jsonl", predictions)
    if samples != shard["samples"]:
        raise ShardedPredictionError("adapter prediction count differs from shard plan")
    diagnostics = read_json(diagnostics_path, "adapter diagnostics")
    receipt = read_json(receipt_path, "adapter run receipt")
    expected_method = args.method
    if nested(diagnostics, "method") != expected_method:
        raise ShardedPredictionError("adapter diagnostics use the wrong method")
    if nested(receipt, "method.name") != expected_method:
        raise ShardedPredictionError("adapter receipt uses the wrong method")
    if nested(receipt, "dataset_manifest_sha256") != shard["manifest_sha256"]:
        raise ShardedPredictionError("adapter receipt does not bind its shard manifest")
    if nested(receipt, "source_revision") != args.source_revision:
        raise ShardedPredictionError("adapter receipt uses the wrong source revision")
    if int(nested(diagnostics, "samples")) != samples:
        raise ShardedPredictionError("adapter diagnostics report the wrong sample count")
    settings = binding["configuration"]
    if int(nested(diagnostics, "warmup_repetitions")) != 0:
        raise ShardedPredictionError("quality shard unexpectedly ran warm-up repetitions")
    if int(nested(diagnostics, "measured_repetitions")) != 1:
        raise ShardedPredictionError("quality shard must run exactly one repetition")
    environment = receipt.get("environment")
    if not isinstance(environment, Mapping) or not environment:
        raise ShardedPredictionError("adapter receipt does not record its environment")
    if args.method == "drncs_lg":
        if not bool(nested(diagnostics, "dataset_hash_pins_enforced")):
            raise ShardedPredictionError("DRNCS-LG shard did not enforce dataset hashes")
        expected_checkpoint = binding["artifacts"]["checkpoint"]["sha256"]
        if nested(diagnostics, "checkpoint_sha256") != expected_checkpoint:
            raise ShardedPredictionError("DRNCS-LG shard used the wrong checkpoint")
        if int(nested(diagnostics, "max_steps")) != settings["max_steps"]:
            raise ShardedPredictionError("DRNCS-LG shard used the wrong max_steps")
    elif args.method == "drpk_static":
        expected_checkpoint = binding["artifacts"]["checkpoint"]["sha256"]
        if nested(diagnostics, "checkpoint.sha256") != expected_checkpoint:
            raise ShardedPredictionError("DRPK-static shard used the wrong checkpoint")
    return {
        "samples": samples,
        "predictions_sha256": sha256_file(predictions),
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "run_receipt_sha256": sha256_file(receipt_path),
        "adapter_environment_sha256": canonical_sha256(environment),
        "diagnostics": diagnostics,
        "receipt": receipt,
    }


def shard_commit_valid(
    args: argparse.Namespace,
    binding: Mapping[str, Any],
    shard: Mapping[str, Any],
) -> dict[str, Any] | None:
    marker_path = Path(str(shard["directory"])) / "complete.json"
    if not marker_path.is_file():
        return None
    try:
        marker = read_json(marker_path, "shard completion marker")
        if (
            marker.get("schema") != SHARD_COMMIT_SCHEMA
            or marker.get("binding_sha256") != binding["binding_sha256"]
            or marker.get("shard_index") != shard["index"]
            or marker.get("input_manifest_sha256") != shard["manifest_sha256"]
            or marker.get("input_records_sha256") != shard["records_sha256"]
        ):
            return None
        observed = validate_adapter_outputs(args, binding, shard)
        for field in (
            "samples",
            "predictions_sha256",
            "diagnostics_sha256",
            "run_receipt_sha256",
            "adapter_environment_sha256",
        ):
            if marker.get(field) != observed[field]:
                return None
        return marker
    except (ShardedPredictionError, AttributeError, TypeError, ValueError):
        return None


def adapter_command(
    args: argparse.Namespace,
    binding: Mapping[str, Any],
    shard: Mapping[str, Any],
) -> list[str]:
    shard_dir = Path(str(shard["directory"]))
    settings = binding["configuration"]
    command = [str(args.adapter_executable), "predict"]
    if args.method == "drncs_lg":
        command.extend(
            [
                "--checkpoint", str(args.checkpoint),
                "--map-dir", str(args.map_dir),
                "--dataset-manifest", str(shard_dir / "manifest.json"),
                "--predictions", str(shard_dir / "predictions.jsonl"),
                "--run-receipt", str(shard_dir / "run.json"),
                "--diagnostics", str(shard_dir / "diagnostics.json"),
                "--source-revision", args.source_revision,
                "--expected-dataset-manifest-sha256", shard["manifest_sha256"],
                "--expected-dataset-records-sha256", shard["records_sha256"],
                "--device", settings["device"],
                "--workers", str(settings["workers"]),
                "--inference-batch-size", str(settings["inference_batch_size"]),
                "--max-steps", str(settings["max_steps"]),
                "--warmup-repetitions", "0",
                "--measured-repetitions", "1",
                "--latency-samples", "0",
                "--seed", str(settings["seed"]),
            ]
        )
    else:
        command.extend(
            [
                "--preprocess-dir", str(args.preprocess_dir),
                "--dataset-manifest", str(shard_dir / "manifest.json"),
                "--method", args.method,
                "--predictions", str(shard_dir / "predictions.jsonl"),
                "--run-receipt", str(shard_dir / "run.json"),
                "--diagnostics", str(shard_dir / "diagnostics.json"),
                "--source-revision", args.source_revision,
                "--device", settings["device"],
                "--workers", str(settings["workers"]),
                "--inference-batch-size", str(settings["inference_batch_size"]),
                "--warmup-repetitions", "0",
                "--measured-repetitions", "1",
                "--seed", str(settings["seed"]),
            ]
        )
        if args.method == "drpk_static":
            command.extend(["--checkpoint", str(args.checkpoint)])
    return command


def progress_value(
    binding: Mapping[str, Any],
    plan: Sequence[Mapping[str, Any]],
    commits: Mapping[int, Mapping[str, Any]],
    *,
    status: str,
    current_shard: int | None = None,
    current_shard_seconds: float | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    completed_samples = sum(int(item["samples"]) for item in commits.values())
    completed_seconds = sum(float(item["adapter_wall_seconds"]) for item in commits.values())
    total_samples = int(binding["dataset"]["samples"])
    remaining = total_samples - completed_samples
    eta = (
        completed_seconds * remaining / completed_samples
        if completed_samples and remaining
        else (0.0 if not remaining else None)
    )
    value: dict[str, Any] = {
        "schema": PROGRESS_SCHEMA,
        "status": status,
        "method": binding["method"],
        "binding_sha256": binding["binding_sha256"],
        "completed_shards": len(commits),
        "total_shards": len(plan),
        "completed_samples": completed_samples,
        "total_samples": total_samples,
        "percent": 100.0 * completed_samples / total_samples,
        "estimated_remaining_adapter_seconds": eta,
        "eta_scope": "quality_shard_adapter_processes_only",
        "current_shard": current_shard,
        "current_shard_elapsed_seconds": current_shard_seconds,
        "updated_at": utc_now(),
    }
    if message is not None:
        value["message"] = message
    return value


def write_progress(
    output_dir: Path,
    binding: Mapping[str, Any],
    plan: Sequence[Mapping[str, Any]],
    commits: Mapping[int, Mapping[str, Any]],
    **kwargs: Any,
) -> None:
    atomic_write_json(
        output_dir / "progress.json", progress_value(binding, plan, commits, **kwargs)
    )


def child_environment(binding: Mapping[str, Any]) -> dict[str, str]:
    environment = dict(os.environ)
    settings = binding["configuration"]
    threads = str(settings["workers"])
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = threads
    environment["PYTHONHASHSEED"] = str(settings["seed"])
    if settings["cuda_visible_devices"] is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(settings["cuda_visible_devices"])
    return environment


def stop_handler(signum: int, _frame: Any) -> None:
    global _STOP_SIGNAL
    _STOP_SIGNAL = signum


def run_one_shard(
    args: argparse.Namespace,
    output_dir: Path,
    binding: Mapping[str, Any],
    plan: Sequence[Mapping[str, Any]],
    commits: dict[int, Mapping[str, Any]],
    shard: Mapping[str, Any],
) -> dict[str, Any]:
    command = adapter_command(args, binding, shard)
    shard_dir = Path(str(shard["directory"]))
    log_path = shard_dir / "adapter.log"
    started = time.perf_counter()
    with log_path.open("ab", buffering=0) as log:
        log.write(
            (json.dumps({"event": "start", "at": utc_now(), "command": command}) + "\n").encode()
        )
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=child_environment(binding),
        )
        last_heartbeat = 0.0
        while process.poll() is None:
            elapsed = time.perf_counter() - started
            if elapsed - last_heartbeat >= 10.0:
                write_progress(
                    output_dir,
                    binding,
                    plan,
                    commits,
                    status="running",
                    current_shard=int(shard["index"]),
                    current_shard_seconds=elapsed,
                )
                last_heartbeat = elapsed
            if _STOP_SIGNAL is not None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
            time.sleep(0.25)
        return_code = process.wait()
        elapsed = time.perf_counter() - started
        log.write(
            (json.dumps({"event": "finish", "at": utc_now(), "return_code": return_code}) + "\n").encode()
        )
    if _STOP_SIGNAL is not None:
        raise KeyboardInterrupt
    if return_code != 0:
        raise ShardedPredictionError(
            f"adapter failed for shard {shard['index']} with exit code {return_code}; "
            f"see {log_path}"
        )
    observed = validate_adapter_outputs(args, binding, shard)
    prior_environments = {
        str(marker["adapter_environment_sha256"]) for marker in commits.values()
    }
    if prior_environments and observed["adapter_environment_sha256"] not in prior_environments:
        raise ShardedPredictionError(
            "adapter environment changed between shards; use a new output directory"
        )
    marker = {
        "schema": SHARD_COMMIT_SCHEMA,
        "binding_sha256": binding["binding_sha256"],
        "shard_index": shard["index"],
        "sample_start": shard["sample_start"],
        "sample_end_exclusive": shard["sample_end_exclusive"],
        "samples": observed["samples"],
        "input_manifest_sha256": shard["manifest_sha256"],
        "input_records_sha256": shard["records_sha256"],
        "predictions_sha256": observed["predictions_sha256"],
        "diagnostics_sha256": observed["diagnostics_sha256"],
        "run_receipt_sha256": observed["run_receipt_sha256"],
        "adapter_environment_sha256": observed["adapter_environment_sha256"],
        "adapter_command": command,
        "adapter_wall_seconds": elapsed,
        "completed_at": utc_now(),
    }
    atomic_write_json(shard_dir / "complete.json", marker)
    return marker


def sum_int_dict(target: dict[str, int], value: Any) -> None:
    if not isinstance(value, Mapping):
        return
    for key, item in value.items():
        if isinstance(item, int) and not isinstance(item, bool):
            target[str(key)] = target.get(str(key), 0) + item


def assemble_outputs(
    args: argparse.Namespace,
    output_dir: Path,
    binding: Mapping[str, Any],
    plan: Sequence[Mapping[str, Any]],
    commits: Mapping[int, Mapping[str, Any]],
) -> None:
    predictions_path = output_dir / "predictions.jsonl"
    temporary = predictions_path.with_name(
        f".{predictions_path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    digest = hashlib.sha256()
    samples = 0
    try:
        with temporary.open("wb") as output:
            for shard in plan:
                shard_path = Path(str(shard["directory"])) / "predictions.jsonl"
                with shard_path.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(chunk)
                        digest.update(chunk)
                samples += int(shard["samples"])
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, predictions_path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if samples != binding["dataset"]["samples"]:
        raise ShardedPredictionError("assembled prediction sample count is wrong")

    endpoint_failures = 0
    peak_rss_kib = 0
    peak_cuda_bytes = 0
    adapter_process_seconds = 0.0
    adapter_prediction_seconds = 0.0
    route_validity: dict[str, int] = {}
    shard_outputs: list[dict[str, Any]] = []
    for shard in plan:
        index = int(shard["index"])
        shard_dir = Path(str(shard["directory"]))
        diagnostics = read_json(shard_dir / "diagnostics.json", "shard diagnostics")
        marker = commits[index]
        endpoint_failures += int(diagnostics.get("endpoint_failures", 0))
        peak_rss_kib = max(peak_rss_kib, int(diagnostics.get("peak_rss_kib", 0)))
        peak_cuda_bytes = max(
            peak_cuda_bytes, int(diagnostics.get("peak_cuda_memory_bytes", 0))
        )
        adapter_process_seconds += float(marker["adapter_wall_seconds"])
        timing = diagnostics.get("timing", {})
        if isinstance(timing, Mapping):
            prediction_value = timing.get(
                "mean_prediction_seconds", timing.get("prediction_seconds", 0.0)
            )
            adapter_prediction_seconds += float(prediction_value)
        sum_int_dict(route_validity, diagnostics.get("generated_route_validity"))
        shard_outputs.append(
            {
                "index": index,
                "samples": shard["samples"],
                "predictions_sha256": marker["predictions_sha256"],
                "diagnostics_sha256": marker["diagnostics_sha256"],
                "run_receipt_sha256": marker["run_receipt_sha256"],
                "adapter_environment_sha256": marker[
                    "adapter_environment_sha256"
                ],
                "adapter_wall_seconds": marker["adapter_wall_seconds"],
            }
        )
    predictions_sha256 = digest.hexdigest()
    diagnostics_value = {
        "schema": DIAGNOSTICS_SCHEMA,
        "method": args.method,
        "purpose": "full_test_quality_prediction_only",
        "efficiency_comparable": False,
        "efficiency_exclusion_reason": (
            "internal shard timing is decomposition only; use the full-test "
            "pipeline's cumulative outer GNU-time boundary for the primary "
            "latency, throughput, and peak-RSS result"
        ),
        "binding_sha256": binding["binding_sha256"],
        "dataset": binding["dataset"],
        "configuration": binding["configuration"],
        "shards": len(plan),
        "samples": samples,
        "endpoint_failures": endpoint_failures,
        "generated_route_validity": route_validity or None,
        "operational_timing": {
            "sum_adapter_process_seconds": adapter_process_seconds,
            "sum_adapter_prediction_seconds": adapter_prediction_seconds,
        },
        "maximum_shard_peak_rss_kib": peak_rss_kib,
        "maximum_shard_peak_cuda_memory_bytes": peak_cuda_bytes,
        "predictions": {
            "path": str(predictions_path),
            "sha256": predictions_sha256,
            "records_schema": PREDICTION_RECORD_SCHEMA,
        },
        "shard_outputs": shard_outputs,
        "completed_at": utc_now(),
    }
    atomic_write_json(output_dir / "diagnostics.json", diagnostics_value)
    receipt_value = {
        "schema": RUN_RECEIPT_SCHEMA,
        "method": args.method,
        "binding_sha256": binding["binding_sha256"],
        "source_revision": args.source_revision,
        "dataset": binding["dataset"],
        "artifacts": binding["artifacts"],
        "configuration": binding["configuration"],
        "prediction_records_schema": PREDICTION_RECORD_SCHEMA,
        "predictions_sha256": predictions_sha256,
        "diagnostics_sha256": sha256_file(output_dir / "diagnostics.json"),
        "shard_commit_sha256": {
            f"{index:06d}": sha256_file(
                Path(str(plan[index]["directory"])) / "complete.json"
            )
            for index in range(len(plan))
        },
        "completed_at": utc_now(),
    }
    atomic_write_json(output_dir / "run-receipt.json", receipt_value)
    complete_value = {
        "schema": COMPLETE_SCHEMA,
        "binding_sha256": binding["binding_sha256"],
        "samples": samples,
        "shards": len(plan),
        "predictions_sha256": predictions_sha256,
        "diagnostics_sha256": sha256_file(output_dir / "diagnostics.json"),
        "run_receipt_sha256": sha256_file(output_dir / "run-receipt.json"),
        "completed_at": utc_now(),
    }
    atomic_write_json(output_dir / "complete.json", complete_value)


def final_outputs_valid(output_dir: Path, binding: Mapping[str, Any]) -> bool:
    marker_path = output_dir / "complete.json"
    if not marker_path.is_file():
        return False
    try:
        marker = read_json(marker_path, "completion marker")
        if (
            marker.get("schema") != COMPLETE_SCHEMA
            or marker.get("binding_sha256") != binding["binding_sha256"]
            or marker.get("samples") != binding["dataset"]["samples"]
        ):
            return False
        for name, field in (
            ("predictions.jsonl", "predictions_sha256"),
            ("diagnostics.json", "diagnostics_sha256"),
            ("run-receipt.json", "run_receipt_sha256"),
        ):
            expected = marker.get(field)
            if not isinstance(expected, str):
                return False
            if sha256_file(output_dir / name) != expected:
                return False
        return True
    except (ShardedPredictionError, AttributeError, TypeError):
        return False


def run(args: argparse.Namespace) -> str:
    if args.method not in METHODS:
        raise ShardedPredictionError(f"unsupported method {args.method!r}")
    if (
        args.shard_size <= 0
        or args.shard_size > MAX_SHARD_SIZE
        or args.workers <= 0
        or (args.inference_batch_size is not None and args.inference_batch_size <= 0)
    ):
        raise ShardedPredictionError(
            f"shard size must be 1..{MAX_SHARD_SIZE}; workers and batch size must be positive"
        )
    if args.shard_limit < 0:
        raise ShardedPredictionError("shard limit must be nonnegative")
    if not args.source_revision.strip():
        raise ShardedPredictionError("source revision must not be empty")
    if args.method == "drncs_lg" and (args.checkpoint is None or args.map_dir is None):
        raise ShardedPredictionError("DRNCS-LG requires --checkpoint and --map-dir")
    if args.method == "drpk_static" and (
        args.checkpoint is None or args.preprocess_dir is None
    ):
        raise ShardedPredictionError("DRPK-static requires --checkpoint and --preprocess-dir")
    if args.method == "drp_tp" and args.preprocess_dir is None:
        raise ShardedPredictionError("DRP-TP requires --preprocess-dir")
    if args.method == "drp_tp" and args.checkpoint is not None:
        raise ShardedPredictionError("DRP-TP must not receive a checkpoint")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (output_dir / ".runner.lock").open("a+")
    try:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ShardedPredictionError(
                f"another sharded prediction process holds {output_dir}"
            ) from error
        descriptor = load_dataset_descriptor(args.dataset_manifest)
        scan = scan_dataset(Path(str(descriptor["records_path"])))
        binding = make_binding(args, descriptor, scan)
        install_or_validate_binding(output_dir, binding)
        plan = materialize_shards(
            output_dir, descriptor, scan, binding, args.shard_size
        )
        commits: dict[int, Mapping[str, Any]] = {}
        for shard in plan:
            marker = shard_commit_valid(args, binding, shard)
            if marker is not None:
                commits[int(shard["index"])] = marker
        committed_environments = {
            str(marker["adapter_environment_sha256"]) for marker in commits.values()
        }
        if len(committed_environments) > 1:
            raise ShardedPredictionError(
                "committed shards contain mixed adapter environments"
            )
        write_progress(output_dir, binding, plan, commits, status="running")
        executed = 0
        for shard in plan:
            index = int(shard["index"])
            if index in commits:
                continue
            if _STOP_SIGNAL is not None:
                raise KeyboardInterrupt
            if args.shard_limit and executed >= args.shard_limit:
                write_progress(
                    output_dir,
                    binding,
                    plan,
                    commits,
                    status="paused",
                    message="clean shard-limit boundary; resume without --shard-limit",
                )
                return "paused"
            marker = run_one_shard(args, output_dir, binding, plan, commits, shard)
            commits[index] = marker
            executed += 1
            write_progress(output_dir, binding, plan, commits, status="running")
        if len(commits) != len(plan):
            raise ShardedPredictionError("not every shard has a valid completion marker")
        if not final_outputs_valid(output_dir, binding):
            assemble_outputs(args, output_dir, binding, plan, commits)
        write_progress(output_dir, binding, plan, commits, status="completed")
        return "completed"
    except KeyboardInterrupt:
        try:
            if "binding" in locals() and "plan" in locals() and "commits" in locals():
                write_progress(
                    output_dir,
                    binding,
                    plan,
                    commits,
                    status="stopped",
                    message=(
                        f"stopped by {signal.Signals(_STOP_SIGNAL).name} at a shard boundary"
                        if _STOP_SIGNAL is not None
                        else "stopped by keyboard interrupt at a shard boundary"
                    ),
                )
        finally:
            raise
    finally:
        lock_handle.close()


def default_adapter_source(method: str) -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "baselines" / (
        "drncs_lg/drncs_lg_adapter.py"
        if method == "drncs_lg"
        else "drpk_static/drpk_static_adapter.py"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Memory-bounded, resumable full-test route prediction"
    )
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-executable", type=Path, required=True)
    parser.add_argument("--adapter-source", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--map-dir", type=Path)
    parser.add_argument("--preprocess-dir", type=Path)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument(
        "--shard-limit",
        type=int,
        default=0,
        help="execute at most N pending shards, then pause cleanly; 0 means all",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--inference-batch-size", type=int)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--cuda-visible-devices", default="0")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.adapter_source is None:
        args.adapter_source = default_adapter_source(args.method)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    global _STOP_SIGNAL
    _STOP_SIGNAL = None
    for handled in (signal.SIGINT, signal.SIGTERM):
        signal.signal(handled, stop_handler)
    try:
        status = run(parse_args(argv))
    except KeyboardInterrupt:
        return 130
    except ShardedPredictionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": status}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
