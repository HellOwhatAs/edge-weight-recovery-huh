#!/usr/bin/env python3
"""Run one bounded time-conditioned configuration and route-based selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any


SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--binary", default=Path("target/release/train_temporal"), type=Path)
    parser.add_argument(
        "--evaluate-binary",
        default=Path("target/release/evaluate_temporal"),
        type=Path,
    )
    parser.add_argument(
        "--selector", default=Path("scripts/select_route_checkpoint.py"), type=Path
    )
    parser.add_argument("--timeout-seconds", default=7200, type=int)
    parser.add_argument("--rayon-threads", default=4, type=int)
    return parser.parse_args()


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def read_events(path: Path) -> list[dict[str, Any]]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}: non-object event")
            events.append(value)
    return events


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = arguments()
    if args.timeout_seconds <= 0 or args.rayon_threads <= 0:
        raise ValueError("timeout and threads must be positive")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_id = config.get("run_id")
    if config.get("schema_version") != 1 or not isinstance(run_id, str):
        raise ValueError("expected a temporal schema-version 1 configuration")
    if not SAFE_RUN_ID.fullmatch(run_id) or config.get("test_policy") != "never_read":
        raise ValueError("unsafe run_id or test policy")
    if config.get("time_conditioning", {}).get("kind") != "global_plus_bucket_residual":
        raise ValueError("configuration is not the supported shared temporal model")
    if config.get("graph", {}).get("representation") != "edge_transition_arcs":
        raise ValueError("temporal experiment must use edge_transition_arcs")
    for binary in (args.binary, args.evaluate_binary):
        if not binary.is_file():
            raise FileNotFoundError(binary)

    output_dir = (args.output_root / run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    console = output_dir / "console.log"
    command = [
        str(args.binary.resolve()),
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]
    environment = os.environ.copy()
    environment["RAYON_NUM_THREADS"] = str(args.rayon_threads)
    started = time.monotonic()
    with console.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
            timeout=args.timeout_seconds,
            check=False,
            text=True,
        )
    training_wall_seconds = time.monotonic() - started
    if completed.returncode != 0:
        raise ValueError(f"temporal training failed; see {console}")
    training_log = output_dir / "training.jsonl"
    checkpoint = output_dir / "checkpoint.json"
    if not training_log.is_file() or not checkpoint.is_file():
        raise ValueError("temporal training did not produce required outputs")
    events = read_events(training_log)
    finished = [event for event in events if event.get("event") == "finished"]
    if len(finished) != 1:
        raise ValueError("temporal training log lacks one finished event")
    finished = finished[0]
    checks = {
        "graph_representation": finished.get("graph_representation")
        == "edge_transition_arcs",
        "model_kind": finished.get("model_kind") == "global_plus_bucket_residual",
        "optimizer_kind": finished.get("optimizer_kind")
        == "relative_projected_subgradient",
        "completed_updates": finished.get("completed_updates")
        == config.get("training", {}).get("updates"),
        "train_objective_finite": finite(finished.get("train_objective")),
        "validation_objective_finite": finite(finished.get("validation_objective")),
        "global_weights_changed": isinstance(
            finished.get("changed_global_coordinates"), int
        )
        and finished["changed_global_coordinates"] > 0,
        "residuals_changed": isinstance(
            finished.get("changed_residual_coordinates"), int
        )
        and finished["changed_residual_coordinates"] > 0,
        "checkpoint_restore_verified": finished.get("checkpoint_restore_verified")
        is True,
        "shortest_path_queries_ok": finished.get("shortest_path_queries_ok") is True,
        "baseline_train_only": finished.get("baseline_train_only") is True,
        "test_not_read": finished.get("test_read") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"unhealthy temporal output: {', '.join(failed)}")

    selector_command = [
        "python3",
        str(args.selector.resolve()),
        "--run-dir",
        str(output_dir),
        "--kind",
        "temporal",
        "--evaluate-binary",
        str(args.evaluate_binary.resolve()),
        "--rayon-threads",
        str(args.rayon_threads),
        "--timeout-seconds",
        str(min(args.timeout_seconds, 900)),
    ]
    selection_console = output_dir / "route_selection.log"
    with selection_console.open("w", encoding="utf-8") as stream:
        selection_completed = subprocess.run(
            selector_command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
            timeout=args.timeout_seconds,
            check=False,
            text=True,
        )
    if selection_completed.returncode != 0:
        raise ValueError(f"route checkpoint selection failed; see {selection_console}")
    selection_path = output_dir / "route_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))

    result = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "ok",
        "graph_representation": "edge_transition_arcs",
        "model_kind": "global_plus_bucket_residual",
        "baseline_kind": config.get("baseline", {}).get("kind"),
        "eta0": config.get("optimizer", {}).get("eta0"),
        "global_lambda": config.get("optimizer", {}).get("global_lambda"),
        "residual_lambda": config.get("optimizer", {}).get("residual_lambda"),
        "updates": config.get("training", {}).get("updates"),
        "validation_every": config.get("training", {}).get("validation_every"),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "training_command": command,
        "training_wall_seconds": training_wall_seconds,
        "total_wall_seconds": time.monotonic() - started,
        "rayon_threads": args.rayon_threads,
        "training_log": str(training_log),
        "checkpoint": str(checkpoint),
        "health_checks": checks,
        "finished": finished,
        "route_selection_command": selector_command,
        "route_selection": selection,
        "test_read": False,
    }
    atomic_json(output_dir / "runner_result.json", result)
    selected = selection["selected"]
    print(
        f"{run_id}: ok; selected update {selected['update']} "
        f"F1={selected['metrics']['edge_f1']:.9f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
