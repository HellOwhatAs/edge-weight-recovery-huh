#!/usr/bin/env python3
"""Run a bounded, content-addressed validation-only experiment matrix.

The matrix is a CSV with these required columns:
run_id, phase, scale, seed, eta0, lambda, train_variant.
Optional columns include metric_update, epochs, patience, eval_every,
eval_path_metrics, early_stop_min_delta, and train_cycle_policy.

Cache reuse requires an exact matrix row, command, Rayon thread count, binary
SHA-256, and graph/train/validation input SHA-256 set. The test split is never
opened or fingerprinted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


REQUIRED_COLUMNS = {
    "run_id",
    "phase",
    "scale",
    "seed",
    "eta0",
    "lambda",
    "train_variant",
}
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
TEST_PATH_TOKEN = re.compile(r"(^|[_.-])test([_.-]|$)", re.IGNORECASE)
CACHE_SCHEMA_VERSION = 1


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--binary", default=Path("target/release/edge-weight-recovery"), type=Path
    )
    parser.add_argument("--city", default="beijing")
    parser.add_argument("--validation-variant", required=True)
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--rayon-threads", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--default-epochs", type=int, default=20)
    parser.add_argument("--default-patience", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_matrix(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"matrix is missing columns: {sorted(missing)}")
        rows = list(reader)
    seen: set[str] = set()
    for row in rows:
        run_id = row["run_id"]
        if not SAFE_RUN_ID.fullmatch(run_id):
            raise ValueError(f"unsafe run_id {run_id!r}")
        if run_id in seen:
            raise ValueError(f"duplicate run_id {run_id!r}")
        seen.add(run_id)
        if row.get("metric_update", "full") not in {"full", "partial"}:
            raise ValueError(f"invalid metric_update in {run_id}")
        if (row.get("train_cycle_policy") or "drop") not in {"drop", "keep", "erase"}:
            raise ValueError(f"invalid train_cycle_policy in {run_id}")
    return rows


def tokens(line: str) -> tuple[str, dict[str, str]]:
    pieces = shlex.split(line.strip())
    if not pieces:
        return "", {}
    values: dict[str, str] = {}
    for token in pieces[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            values[key] = value
    return pieces[0], values


def as_number(value: str | None) -> int | float | str | None:
    if value is None or value == "NA":
        return None
    try:
        integer = int(value)
        if str(integer) == value:
            return integer
    except ValueError:
        pass
    try:
        number = float(value)
        return number if math.isfinite(number) else value
    except ValueError:
        return value


def mean(rows: list[dict[str, str]], key: str) -> float | None:
    values = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return sum(values) / len(values) if values else None


def parse_log(path: Path) -> dict[str, Any]:
    data: dict[str, dict[str, str]] = {}
    epochs: list[dict[str, str]] = []
    evaluation: dict[str, str] = {}
    finished: dict[str, str] = {}
    config: dict[str, str] = {}
    if not path.exists():
        return {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        kind, values = tokens(line)
        if kind == "CONFIG":
            config = values
        elif kind == "DATA" and "split" in values:
            data[values["split"]] = values
        elif kind == "EPOCH":
            epochs.append(values)
        elif kind == "EVAL" and values.get("split") == "validation_best":
            evaluation = values
        elif kind == "FINISHED":
            finished = values
    if not finished or not evaluation:
        return {}
    best_epoch = str(finished.get("best_epoch", ""))
    best_epoch_row = next(
        (epoch for epoch in epochs if epoch.get("epoch") == best_epoch), {}
    )
    update_epochs = [
        epoch
        for epoch in epochs
        if epoch.get("update_status")
        in {"applied", "latent_only_no_integer_change", "shock_applied"}
    ]
    result: dict[str, Any] = {
        "status": "ok",
        "epochs_executed": len(epochs),
        "avg_epoch_ms": mean(epochs, "epoch_ms"),
        "avg_train_oracle_ms": mean(epochs, "train_oracle_ms"),
        "avg_customization_ms": mean(epochs, "customization_ms"),
        "avg_changed_pct": mean(epochs, "changed_pct"),
        "avg_update_customization_ms": mean(update_epochs, "customization_ms"),
        "avg_update_changed_pct": mean(update_epochs, "changed_pct"),
    }
    for prefix, values, keys in [
        (
            "config_",
            config,
            [
                "solver",
                "selection_metric",
                "metric_update",
                "run_test",
                "eval_path_metrics",
                "early_stop_min_delta",
                "train_cycle_policy",
                "evaluation_cycle_policy",
            ],
        ),
        (
            "train_",
            data.get("train", {}),
            [
                "available",
                "inspected",
                "accepted",
                "cyclic",
                "cycle_erased_records",
                "empty_after_cycle_transform",
                "cycle_edges_removed",
            ],
        ),
        (
            "validation_",
            data.get("validation", {}),
            [
                "available",
                "inspected",
                "accepted",
                "cyclic",
                "cycle_erased_records",
                "empty_after_cycle_transform",
                "cycle_edges_removed",
            ],
        ),
        (
            "best_",
            finished,
            [
                "best_epoch",
                "selection_loss",
                "best_train_regret",
                "best_regularization",
                "best_q_min",
                "best_q_max",
                "best_q_at_min",
                "best_q_at_max",
                "peak_rss_kib",
                "train_regret_improvement_pct",
            ],
        ),
        (
            "best_epoch_",
            best_epoch_row,
            [
                "train_relative_regret",
                "count_residual_l1",
                "changed_pct",
                "train_oracle_ms",
                "customization_ms",
                "current_max_quantization_error",
                "validation_relative_regret",
                "validation_exact_match",
                "validation_edge_f1",
                "validation_edge_jaccard",
            ],
        ),
        (
            "validation_",
            evaluation,
            [
                "samples",
                "mean_regret",
                "relative_regret",
                "exact_match",
                "edge_precision",
                "edge_recall",
                "edge_f1",
                "edge_jaccard",
            ],
        ),
    ]:
        for key in keys:
            result[prefix + key] = as_number(values.get(key))
    # Preserve the original parser keys for old result files, but expose
    # non-duplicated aliases for new analyses (for example best_epoch rather
    # than best_best_epoch).
    for key in [
        "best_epoch",
        "best_train_regret",
        "best_regularization",
        "best_q_min",
        "best_q_max",
        "best_q_at_min",
        "best_q_at_max",
    ]:
        legacy_key = "best_" + key
        if legacy_key in result:
            result[key] = result[legacy_key]
    train_relative = result.get("best_epoch_train_relative_regret")
    validation_relative = result.get("validation_relative_regret")
    if isinstance(train_relative, (int, float)) and isinstance(
        validation_relative, (int, float)
    ):
        result["relative_regret_gap"] = validation_relative - train_relative
    train_mean = result.get("best_train_regret")
    validation_mean = result.get("validation_mean_regret")
    if isinstance(train_mean, (int, float)) and isinstance(validation_mean, (int, float)):
        result["mean_regret_gap"] = validation_mean - train_mean
    return result


def checkpoint_quantiles(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    values = sorted(float(value) for value in checkpoint.get("multipliers", []))
    if not values:
        return {}

    def quantile(probability: float) -> float:
        rank = probability * (len(values) - 1)
        lower = int(math.floor(rank))
        upper = int(math.ceil(rank))
        fraction = rank - lower
        return values[lower] * (1.0 - fraction) + values[upper] * fraction

    return {
        "best_q_p05": quantile(0.05),
        "best_q_p25": quantile(0.25),
        "best_q_median": quantile(0.50),
        "best_q_p75": quantile(0.75),
        "best_q_p95": quantile(0.95),
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def build_command(
    row: dict[str, str], args: argparse.Namespace, prefix: Path
) -> list[str]:
    epochs = int(row.get("epochs") or args.default_epochs)
    patience = int(row.get("patience") or args.default_patience)
    metric_update = row.get("metric_update") or "full"
    solver = row.get("solver") or "projected"
    if solver not in {"projected", "adam-shock"}:
        raise ValueError(f"invalid solver {solver!r} in {row['run_id']}")
    command = [
        str(args.binary.resolve()),
        "--city",
        args.city,
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--train-variant",
        row["train_variant"],
        "--validation-variant",
        args.validation_variant,
        "--test-variant",
        "all",
        "--solver",
        solver,
        "--metric-update",
        metric_update,
        "--selection-metric",
        "relative-regret",
        "--eta0",
        row["eta0"],
        "--lambda",
        row["lambda"],
        "--eval-every",
        row.get("eval_every") or "1",
        "--seed",
        row["seed"],
        "--output-prefix",
        str(prefix.resolve()),
    ]
    if solver == "adam-shock":
        command.extend(
            ["--adam-learning-rate", row.get("adam_learning_rate") or "3000"]
        )
    if (row.get("eval_path_metrics") or "").lower() in {"1", "true", "yes"}:
        command.append("--eval-path-metrics")
    if row.get("early_stop_min_delta"):
        command.extend(
            ["--early-stop-min-delta", row["early_stop_min_delta"]]
        )
    if row.get("train_cycle_policy"):
        command.extend(["--train-cycle-policy", row["train_cycle_policy"]])
    return command


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_mentions_test(path: Path) -> bool:
    return any(TEST_PATH_TOKEN.search(part) for part in path.parts)


def fingerprint_scientific_inputs(
    rows: list[dict[str, str]], args: argparse.Namespace
) -> dict[str, dict[str, str]]:
    """Hash graph/train/validation inputs once, without opening the test split."""
    repository = Path(__file__).resolve().parents[1]
    city_data = repository / "data" / f"{args.city}_data"
    resolved_city_data = city_data.resolve()
    shared_paths = [
        city_data / "map" / f"{stem}.{suffix}"
        for stem in ("edges", "nodes")
        for suffix in ("shp", "shx", "dbf")
    ]
    shared_paths.append(
        city_data
        / f"preprocessed_validation_trips_{args.validation_variant}.pkl"
    )
    unique_paths = set(shared_paths)
    train_path_by_run: dict[str, Path] = {}
    for row in rows:
        train_path = (
            city_data / f"preprocessed_train_trips_{row['train_variant']}.pkl"
        )
        train_path_by_run[row["run_id"]] = train_path
        unique_paths.add(train_path)
    digests: dict[Path, str] = {}
    for path in sorted(unique_paths, key=lambda value: str(value)):
        relative = path.relative_to(city_data)
        if path_mentions_test(relative):
            raise ValueError(f"refusing to fingerprint test input: {path}")
        resolved = path.resolve()
        try:
            resolved_relative = resolved.relative_to(resolved_city_data)
        except ValueError as error:
            raise ValueError(
                f"scientific input resolves outside the city data root: {path} -> {resolved}"
            ) from error
        if path_mentions_test(resolved_relative):
            raise ValueError(f"refusing to fingerprint test input: {resolved}")
        if not resolved.is_file():
            raise FileNotFoundError(
                f"missing scientific input for cache identity: {path}"
            )
        digests[path] = sha256_file(resolved)
    return {
        row["run_id"]: {
            str(path.resolve()): digests[path]
            for path in [*shared_paths, train_path_by_run[row["run_id"]]]
        }
        for row in rows
    }


def cache_signature(
    command: list[str],
    binary_sha256: str,
    rayon_threads: int,
    jobs: int,
    timeout_seconds: int,
    scientific_input_sha256: dict[str, str],
    matrix_row: dict[str, str],
) -> str:
    identity = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "command": command,
        "binary_sha256": binary_sha256,
        "runner": {
            "rayon_threads": rayon_threads,
            "jobs": jobs,
            "timeout_seconds": timeout_seconds,
        },
        "scientific_input_sha256": scientific_input_sha256,
        "matrix_row": matrix_row,
    }
    canonical = json.dumps(
        identity, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def run_one(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    run_id = row["run_id"]
    output_dir = args.run_root / row["phase"] / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "model"
    log_path = Path(f"{prefix}_training.log")
    checkpoint_path = Path(f"{prefix}_checkpoint.json")
    result_path = output_dir / "runner_result.json"
    command = build_command(row, args, prefix)
    binary_sha256 = sha256_file(Path(command[0]))
    scientific_input_sha256 = args.scientific_input_sha256_by_run[run_id]
    expected_cache_signature = cache_signature(
        command,
        binary_sha256,
        args.rayon_threads,
        args.jobs,
        args.timeout_seconds,
        scientific_input_sha256,
        row,
    )

    cached = parse_log(log_path)
    if (
        cached
        and not args.force
        and checkpoint_path.is_file()
        and result_path.exists()
    ):
        try:
            prior = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prior = {}
        if not isinstance(prior, dict):
            prior = {}
        if (
            prior.get("status") == "ok"
            and prior.get("cache_signature") == expected_cache_signature
            and prior.get("checkpoint_sha256") == sha256_file(checkpoint_path)
            and prior.get("log_sha256") == sha256_file(log_path)
        ):
            prior.update(cached)
            prior.update(checkpoint_quantiles(checkpoint_path))
            prior.update(row)
            prior.update(
                {
                    "cached": True,
                    "cache_hit_last_invocation": True,
                    "checkpoint_path": str(checkpoint_path),
                    "log_path": str(log_path),
                    "command": command,
                    "binary_sha256": binary_sha256,
                    "scientific_input_sha256": scientific_input_sha256,
                    "cache_signature": expected_cache_signature,
                    "checkpoint_sha256": prior["checkpoint_sha256"],
                    "log_sha256": prior["log_sha256"],
                    "runner_context": {
                        "jobs": args.jobs,
                        "rayon_threads": args.rayon_threads,
                        "timeout_seconds": args.timeout_seconds,
                    },
                }
            )
            return prior

    environment = os.environ.copy()
    environment["RAYON_NUM_THREADS"] = str(args.rayon_threads)
    started = time.monotonic()
    status = "error"
    error = ""
    returncode: int | None = None
    try:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout_seconds,
            check=False,
        )
        returncode = completed.returncode
        (output_dir / "process_output.txt").write_text(
            completed.stdout, encoding="utf-8"
        )
        status = "ok" if returncode == 0 else "error"
        if returncode != 0:
            error = "\n".join(completed.stdout.splitlines()[-20:])
    except subprocess.TimeoutExpired as timeout:
        status = "timeout"
        output = timeout.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        (output_dir / "process_output.txt").write_text(output, encoding="utf-8")
        error = f"exceeded {args.timeout_seconds} seconds"
    wall_seconds = time.monotonic() - started

    parsed = parse_log(log_path) if status == "ok" else {}
    if status == "ok" and not parsed:
        status = "incomplete"
        error = "process exited successfully without complete FINISHED/EVAL records"
    elif status == "ok":
        parsed.update(checkpoint_quantiles(checkpoint_path))
    result: dict[str, Any] = dict(row)
    result.update(parsed)
    checkpoint_sha256 = (
        sha256_file(checkpoint_path)
        if status == "ok" and checkpoint_path.is_file()
        else None
    )
    log_sha256 = (
        sha256_file(log_path) if status == "ok" and log_path.is_file() else None
    )
    result.update(
        {
            "status": status,
            "cached": False,
            "cache_hit_last_invocation": False,
            "wall_seconds": wall_seconds,
            "returncode": returncode,
            "error": error,
            "checkpoint_path": str(checkpoint_path),
            "log_path": str(log_path),
            "command": command,
            "binary_sha256": binary_sha256,
            "scientific_input_sha256": scientific_input_sha256,
            "cache_signature": expected_cache_signature,
            "checkpoint_sha256": checkpoint_sha256,
            "log_sha256": log_sha256,
            "runner_context": {
                "jobs": args.jobs,
                "rayon_threads": args.rayon_threads,
                "timeout_seconds": args.timeout_seconds,
            },
        }
    )
    atomic_json(result_path, result)
    return result


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("runs", []) if isinstance(value, dict) else value
    return {str(row["run_id"]): row for row in rows if "run_id" in row}


def write_summaries(
    rows: list[dict[str, Any]], csv_path: Path, json_path: Path
) -> None:
    existing = load_existing(json_path)
    existing.update({str(row["run_id"]): row for row in rows})
    combined = sorted(existing.values(), key=lambda row: (row.get("phase", ""), row["run_id"]))
    atomic_json(
        json_path,
        {
            "schema_version": 1,
            "selection_policy": "minimum validation relative regret; edge F1 is secondary",
            "test_policy": "test is never loaded by this runner",
            "runs": combined,
        },
    )
    columns: list[str] = []
    for preferred in [
        "run_id",
        "phase",
        "scale",
        "seed",
        "eta0",
        "lambda",
        "metric_update",
        "train_variant",
        "train_cycle_policy",
        "status",
        "cached",
        "wall_seconds",
        "train_accepted",
        "train_cyclic",
        "train_cycle_erased_records",
        "train_empty_after_cycle_transform",
        "train_cycle_edges_removed",
        "validation_accepted",
        "best_best_epoch",
        "best_selection_loss",
        "validation_relative_regret",
        "validation_mean_regret",
        "validation_exact_match",
        "validation_edge_f1",
        "validation_edge_jaccard",
        "best_epoch_train_relative_regret",
        "best_train_regret",
        "best_q_min",
        "best_q_max",
        "avg_epoch_ms",
        "avg_train_oracle_ms",
        "avg_customization_ms",
        "avg_changed_pct",
        "avg_update_customization_ms",
        "avg_update_changed_pct",
        "relative_regret_gap",
        "mean_regret_gap",
        "best_peak_rss_kib",
        "best_q_p05",
        "best_q_median",
        "best_q_p95",
    ]:
        if any(preferred in row for row in combined):
            columns.append(preferred)
    columns.extend(
        sorted({key for row in combined for key in row} - set(columns))
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(csv_path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined)
    temporary.replace(csv_path)


def main() -> int:
    args = arguments()
    if args.jobs < 1 or args.rayon_threads < 1:
        raise ValueError("--jobs and --rayon-threads must be positive")
    if not 1 <= args.timeout_seconds <= 900:
        raise ValueError("per-run timeout must be in 1..=900 seconds")
    rows = read_matrix(args.matrix)
    args.scientific_input_sha256_by_run = fingerprint_scientific_inputs(rows, args)
    args.run_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_to_id = {executor.submit(run_one, row, args): row["run_id"] for row in rows}
        for future in as_completed(future_to_id):
            run_id = future_to_id[future]
            try:
                result = future.result()
            except Exception as error:  # keep the rest of a matrix alive
                result = {"run_id": run_id, "status": "runner_error", "error": repr(error)}
            results.append(result)
            print(
                f"RUN run_id={run_id} status={result.get('status')} "
                f"wall_seconds={result.get('wall_seconds')} cached={result.get('cached')}",
                flush=True,
            )
    write_summaries(results, args.summary_csv, args.summary_json)
    failures = [row for row in results if row.get("status") != "ok"]
    print(
        f"SUMMARY completed={len(results) - len(failures)} failed={len(failures)} "
        f"csv={args.summary_csv} json={args.summary_json}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
