#!/usr/bin/env python3
"""Run bounded edge-only configurations through the final training CLI."""

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
    parser.add_argument("--binary", default=Path("target/release/train"), type=Path)
    parser.add_argument("--timeout-seconds", default=900, type=int)
    parser.add_argument("--rayon-threads", default=4, type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_run_id(path: Path) -> str:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError(f"{path}: expected a schema_version 1 object")
    run_id = config.get("run_id")
    if not isinstance(run_id, str) or not SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError(f"{path}: unsafe run_id {run_id!r}")
    if config.get("test_policy") != "never_read":
        raise ValueError(f"{path}: test_policy must be 'never_read'")
    return run_id


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_one(
    args: argparse.Namespace, config_path: Path, run_id: str, output_dir: Path
) -> int:
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
    result = {
        "schema_version": 1,
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
    }
    atomic_json(output_dir / "runner_result.json", result)
    print(f"{run_id}: {status} ({result['wall_seconds']:.2f}s)")
    return 0 if status == "ok" else 1


def main() -> int:
    args = arguments()
    if args.timeout_seconds <= 0 or args.rayon_threads <= 0:
        raise ValueError("timeout and Rayon thread count must be positive")
    runs = [(path, load_run_id(path)) for path in args.config]
    run_ids = [run_id for _, run_id in runs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("duplicate run_id across configurations")
    if not args.dry_run and not args.binary.is_file():
        raise FileNotFoundError(f"training binary not found: {args.binary}")
    statuses = [
        run_one(args, path, run_id, args.output_root / run_id)
        for path, run_id in runs
    ]
    return max(statuses)


if __name__ == "__main__":
    raise SystemExit(main())
