#!/usr/bin/env python3
"""Run bounded first- and second-order configurations through one training CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from typing import Any


SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--binary", default=Path("target/release/train"), type=Path)
    parser.add_argument("--timeout-seconds", default=900, type=int)
    parser.add_argument("--rayon-threads", default=4, type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 2:
        raise ValueError(f"{path}: expected a schema_version 2 object")
    run_id = config.get("run_id")
    if not isinstance(run_id, str) or not SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError(f"{path}: unsafe run_id {run_id!r}")
    if config.get("test_policy") != "never_read":
        raise ValueError(f"{path}: test_policy must be 'never_read'")
    return config


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def validate_training_log(
    path: Path, config: dict[str, Any], checkpoint: Path
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(event, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        events.append(event)

    states = [event for event in events if event.get("event") == "state"]
    finished_events = [event for event in events if event.get("event") == "finished"]
    if not states or len(finished_events) != 1:
        raise ValueError(f"{path}: expected state events and exactly one finished event")
    for state in states:
        if not finite_number(state.get("train_objective")):
            raise ValueError(f"{path}: state has a non-finite train_objective")
        validation = state.get("validation")
        if validation is not None and (
            not isinstance(validation, dict)
            or not finite_number(validation.get("objective"))
        ):
            raise ValueError(f"{path}: state has a non-finite validation objective")

    finished = finished_events[0]
    required_updates = config.get("training", {}).get("updates")
    checks = {
        "completed_updates": finished.get("completed_updates") == required_updates,
        "train_objective_finite": finite_number(finished.get("train_objective")),
        "validation_objective_finite": finite_number(
            finished.get("validation_objective")
        ),
        "weights_changed": isinstance(finished.get("changed_coordinates"), int)
        and finished["changed_coordinates"] > 0,
        "shortest_path_queries_ok": finished.get("shortest_path_queries_ok") is True,
        "checkpoint_restore_verified": finished.get("checkpoint_restore_verified") is True,
        "test_not_read": finished.get("test_read") is False,
        "checkpoint_exists": checkpoint.is_file(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{path}: failed smoke health checks: {', '.join(failed)}")
    return checks


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_one(
    args: argparse.Namespace,
    config_path: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> int:
    run_id = config["run_id"]
    command = [
        str(args.binary.resolve()),
        "--config",
        str(config_path.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
    ]
    if args.dry_run:
        print(shlex.join(command))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    console_path = output_dir / "console.log"
    environment = os.environ.copy()
    environment["RAYON_NUM_THREADS"] = str(args.rayon_threads)
    started = time.monotonic()
    status = "ok"
    returncode: int | None = None
    try:
        with console_path.open("w", encoding="utf-8") as console:
            completed = subprocess.run(
                command,
                stdout=console,
                stderr=subprocess.STDOUT,
                env=environment,
                timeout=args.timeout_seconds,
                check=False,
                text=True,
            )
        returncode = completed.returncode
        if returncode != 0:
            status = "failed"
    except subprocess.TimeoutExpired:
        status = "timeout"

    training_log = output_dir / "training.jsonl"
    checkpoint = output_dir / "checkpoint.json"
    if status == "ok" and (not training_log.is_file() or not checkpoint.is_file()):
        status = "missing_outputs"
    health_checks: dict[str, Any] | None = None
    health_error: str | None = None
    if status == "ok":
        try:
            health_checks = validate_training_log(training_log, config, checkpoint)
        except ValueError as error:
            status = "unhealthy_output"
            health_error = str(error)
    result = {
        "schema_version": 2,
        "run_id": run_id,
        "status": status,
        "returncode": returncode,
        "wall_seconds": time.monotonic() - started,
        "config_path": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "command": command,
        "rayon_threads": args.rayon_threads,
        "training_log": str(training_log.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "health_checks": health_checks,
        "health_error": health_error,
    }
    atomic_json(output_dir / "runner_result.json", result)
    print(f"{run_id}: {status} ({result['wall_seconds']:.2f}s)")
    return 0 if status == "ok" else 1


def main() -> int:
    args = arguments()
    if args.timeout_seconds <= 0 or args.rayon_threads <= 0:
        raise ValueError("timeout and Rayon thread count must be positive")
    runs = [(path, load_config(path)) for path in args.config]
    run_ids = [config["run_id"] for _, config in runs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("duplicate run_id across configurations")
    if not args.dry_run and not args.binary.is_file():
        raise FileNotFoundError(f"training binary not found: {args.binary}")
    statuses = [
        run_one(args, path, config, args.output_root / config["run_id"])
        for path, config in runs
    ]
    return max(statuses)


if __name__ == "__main__":
    raise SystemExit(main())
