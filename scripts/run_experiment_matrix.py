#!/usr/bin/env python3
"""Run one or more bounded edge-only validation configurations sequentially.

This runner deliberately exposes only the frozen paper mainline: projected
subgradient updates, full CCH customization, dropped cyclic observations, and
aggregate validation relative-regret selection.  It never names or reads a
test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    parser.add_argument(
        "--binary", default=Path("target/release/edge-weight-recovery"), type=Path
    )
    parser.add_argument("--timeout-seconds", default=900, type=int)
    parser.add_argument("--rayon-threads", default=4, type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"{path}: expected schema_version 1 object")
    run_id = value.get("run_id")
    if not isinstance(run_id, str) or not SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError(f"{path}: unsafe run_id {run_id!r}")

    expected = {
        "model.kind": value.get("model", {}).get("kind"),
        "model.solver": value.get("model", {}).get("solver"),
        "oracle.customization": value.get("oracle", {}).get("customization"),
        "data.cycle_policy": value.get("data", {}).get("cycle_policy"),
        "selection.split": value.get("selection", {}).get("split"),
        "selection.metric": value.get("selection", {}).get("metric"),
        "test_policy": value.get("test_policy"),
    }
    required = {
        "model.kind": "edge_only",
        "model.solver": "projected_subgradient",
        "oracle.customization": "full",
        "data.cycle_policy": "drop",
        "selection.split": "validation",
        "selection.metric": "aggregate_relative_regret",
        "test_policy": "never_read",
    }
    for field, required_value in required.items():
        if expected[field] != required_value:
            raise ValueError(
                f"{path}: {field} must be {required_value!r}, got {expected[field]!r}"
            )

    training = value.get("training", {})
    for field in ("epochs", "validation_every", "early_stop_patience"):
        if not isinstance(training.get(field), int) or training[field] <= 0:
            raise ValueError(f"{path}: training.{field} must be a positive integer")
    return value


def build_command(binary: Path, config: dict[str, Any], prefix: Path) -> list[str]:
    data = config["data"]
    model = config["model"]
    training = config["training"]
    command = [
        str(binary.resolve()),
        "--city",
        str(data["city"]),
        "--train-variant",
        str(data["train_variant"]),
        "--validation-variant",
        str(data["validation_variant"]),
        "--train-cycle-policy",
        "drop",
        "--solver",
        "projected",
        "--metric-update",
        "full",
        "--selection-metric",
        "relative-regret",
        "--epochs",
        str(training["epochs"]),
        "--patience",
        str(training["early_stop_patience"]),
        "--eval-every",
        str(training["validation_every"]),
        "--early-stop-min-delta",
        str(training["early_stop_min_delta"]),
        "--eta0",
        str(model["eta0"]),
        "--lambda",
        str(model["lambda_edge"]),
        "--q-min",
        str(model["q_min"]),
        "--q-max",
        str(model["q_max"]),
        "--quantization-scale",
        str(model["quantization_scale"]),
        "--output-prefix",
        str(prefix.resolve()),
    ]
    if training.get("evaluate_path_metrics", True):
        command.append("--eval-path-metrics")
    return command


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_one(args: argparse.Namespace, path: Path, config: dict[str, Any]) -> int:
    run_dir = args.output_root / config["run_id"]
    prefix = run_dir / "model"
    command = build_command(args.binary, config, prefix)
    if args.dry_run:
        print(shlex.join(command))
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "runner.log"
    started = time.monotonic()
    environment = os.environ.copy()
    environment["RAYON_NUM_THREADS"] = str(args.rayon_threads)
    status = "ok"
    returncode: int | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                stdout=log,
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

    result = {
        "schema_version": 1,
        "run_id": config["run_id"],
        "status": status,
        "returncode": returncode,
        "wall_seconds": time.monotonic() - started,
        "config_path": str(path),
        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "command": command,
        "rayon_threads": args.rayon_threads,
        "training_log": str(prefix.with_name(prefix.name + "_training.log")),
    }
    atomic_json(run_dir / "runner_result.json", result)
    print(f"{config['run_id']}: {status} ({result['wall_seconds']:.2f}s)")
    return 0 if status == "ok" else 1


def main() -> int:
    args = arguments()
    if args.timeout_seconds <= 0 or args.rayon_threads <= 0:
        raise ValueError("timeout and Rayon thread count must be positive")
    configs = [(path, load_config(path)) for path in args.config]
    run_ids = [config["run_id"] for _, config in configs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("duplicate run_id across configs")
    if not args.dry_run and not args.binary.is_file():
        raise FileNotFoundError(f"training binary not found: {args.binary}")
    return max(run_one(args, path, config) for path, config in configs)


if __name__ == "__main__":
    raise SystemExit(main())
