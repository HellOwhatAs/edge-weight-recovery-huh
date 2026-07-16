#!/usr/bin/env python3
"""Run bounded static configurations through the common train/evaluate CLIs."""

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
STANDARD_METRICS = (
    "samples",
    "mean_regret",
    "relative_regret",
    "exact_match",
    "edge_precision",
    "edge_recall",
    "edge_f1",
    "edge_jaccard",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--binary", default=Path("target/release/train"), type=Path)
    parser.add_argument(
        "--evaluate-binary", default=Path("target/release/evaluate"), type=Path
    )
    parser.add_argument("--timeout-seconds", default=900, type=int)
    parser.add_argument("--rayon-threads", default=4, type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 3:
        raise ValueError(f"{path}: expected a schema_version 3 object")
    run_id = config.get("run_id")
    if not isinstance(run_id, str) or not SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError(f"{path}: unsafe run_id {run_id!r}")
    if config.get("test_policy") != "never_read":
        raise ValueError(f"{path}: test_policy must be 'never_read'")
    graph = config.get("graph")
    if not isinstance(graph, dict) or graph.get("representation") not in {
        "original_edges",
        "edge_transition_arcs",
    }:
        raise ValueError(f"{path}: invalid graph.representation")
    optimizer = config.get("optimizer")
    if not isinstance(optimizer, dict) or optimizer.get("kind") not in {
        "projected_subgradient",
        "relative_projected_subgradient",
    }:
        raise ValueError(f"{path}: invalid optimizer.kind")
    return config


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def read_events(path: Path) -> list[dict[str, Any]]:
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
    return events


def latest(events: list[dict[str, Any]], event_name: str) -> dict[str, Any]:
    matches = [event for event in events if event.get("event") == event_name]
    if not matches:
        raise ValueError(f"missing {event_name!r} event")
    return matches[-1]


def validate_training_log(
    path: Path,
    config: dict[str, Any],
    checkpoint: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events = read_events(path)

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
        "graph_representation": finished.get("graph_representation")
        == config["graph"]["representation"],
        "optimizer_kind": finished.get("optimizer_kind")
        == config["optimizer"]["kind"],
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
    return checks, events


def validation_candidates(
    events: list[dict[str, Any]], output_dir: Path
) -> list[dict[str, Any]]:
    candidates = []
    for state in events:
        validation = state.get("validation")
        if state.get("event") != "state" or not isinstance(validation, dict):
            continue
        update = state.get("completed_updates")
        objective = validation.get("objective")
        if not isinstance(update, int) or update < 0 or not finite_number(objective):
            raise ValueError("validation state has an invalid update or objective")
        weights = state.get("weights")
        if not isinstance(weights, dict) or not isinstance(
            weights.get("changed_from_initial"), int
        ):
            raise ValueError(f"validation state {update} lacks a weight-change summary")
        checkpoint = output_dir / f"checkpoint-{update}.json"
        if not checkpoint.is_file():
            raise ValueError(f"validation checkpoint does not exist: {checkpoint}")
        candidates.append(
            {
                "update": update,
                "validation_objective": objective,
                "validation_mean_regret": validation.get("mean_regret"),
                "validation_relative_regret": validation.get("relative_regret"),
                "train_objective": state.get("train_objective"),
                "regularization": state.get("regularization"),
                "changed_coordinates": weights["changed_from_initial"],
                "max_quantization_error": weights.get("max_quantization_error"),
                "checkpoint": str(checkpoint.resolve()),
            }
        )
    if not candidates:
        raise ValueError("training log has no validation checkpoint candidates")
    candidates.sort(key=lambda candidate: candidate["update"])
    return candidates


def validate_evaluation(
    path: Path,
    config: dict[str, Any],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    evaluation = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(evaluation, dict):
        raise ValueError(f"{path}: evaluation output must be an object")
    checks = {
        "validation_split": evaluation.get("split") == "validation",
        "validation_variant": evaluation.get("variant")
        == config["data"]["validation_variant"],
        "graph_representation": evaluation.get("graph_representation")
        == config["graph"]["representation"],
        "checkpoint_update": evaluation.get("checkpoint_completed_updates")
        == checkpoint["update"],
        "test_not_read": evaluation.get("test_read") is False,
    }
    metrics = evaluation.get("metrics")
    checks["metrics_object"] = isinstance(metrics, dict)
    if isinstance(metrics, dict):
        checks["metrics_complete"] = all(name in metrics for name in STANDARD_METRICS)
        checks["metrics_finite"] = all(
            finite_number(metrics.get(name)) for name in STANDARD_METRICS[1:]
        )
        checks["sample_count"] = isinstance(metrics.get("samples"), int) and metrics[
            "samples"
        ] > 0
        departure_filter = config.get("data", {}).get("departure_time_filter")
        if isinstance(departure_filter, dict):
            checks["filtered_sample_count"] = metrics.get("samples") == departure_filter.get(
                "expected_validation_samples"
            )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"{path}: failed evaluation checks: {', '.join(failed)}")
    return evaluation


def evaluate_checkpoint(
    args: argparse.Namespace,
    config: dict[str, Any],
    output_dir: Path,
    checkpoint: dict[str, Any],
    label: str,
    environment: dict[str, str],
) -> dict[str, Any]:
    output = output_dir / f"evaluation-{label}.json"
    console_path = output_dir / f"evaluation-{label}.log"
    command = [
        str(args.evaluate_binary.resolve()),
        "--checkpoint",
        checkpoint["checkpoint"],
        "--split",
        "validation",
        "--variant",
        config["data"]["validation_variant"],
        "--output",
        str(output.resolve()),
    ]
    started = time.monotonic()
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
    except subprocess.TimeoutExpired as error:
        raise ValueError(
            f"evaluation {label} timed out after {args.timeout_seconds}s"
        ) from error
    wall_seconds = time.monotonic() - started
    if completed.returncode != 0:
        raise ValueError(
            f"evaluation {label} failed with return code {completed.returncode}; "
            f"see {console_path}"
        )
    if not output.is_file():
        raise ValueError(f"evaluation {label} did not create {output}")
    evaluation = validate_evaluation(output, config, checkpoint)
    return {
        "label": label,
        "command": command,
        "wall_seconds": wall_seconds,
        "console_log": str(console_path.resolve()),
        "output": str(output.resolve()),
        "checkpoint_update": checkpoint["update"],
        "metrics": evaluation["metrics"],
        "path_report": evaluation.get("path_report"),
    }


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
    training_wall_seconds = time.monotonic() - started

    training_log = output_dir / "training.jsonl"
    checkpoint = output_dir / "checkpoint.json"
    if status == "ok" and (not training_log.is_file() or not checkpoint.is_file()):
        status = "missing_outputs"
    health_checks: dict[str, Any] | None = None
    health_error: str | None = None
    events: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_evaluation: dict[str, Any] | None = None
    baseline_evaluation: dict[str, Any] | None = None
    finished_summary: dict[str, Any] | None = None
    if status == "ok":
        try:
            health_checks, events = validate_training_log(
                training_log, config, checkpoint
            )
            candidates = validation_candidates(events, output_dir)
            selected = min(
                candidates,
                key=lambda candidate: (
                    candidate["validation_objective"],
                    candidate["update"],
                ),
            )
            selected_evaluation = evaluate_checkpoint(
                args,
                config,
                output_dir,
                selected,
                "selected",
                environment,
            )
            baseline = next(
                (candidate for candidate in candidates if candidate["update"] == 0),
                None,
            )
            if baseline is None:
                raise ValueError("training log has no update-0 validation checkpoint")
            if selected["update"] == 0:
                baseline_evaluation = {
                    **selected_evaluation,
                    "label": "baseline",
                    "reused_selected_evaluation": True,
                }
            else:
                baseline_evaluation = evaluate_checkpoint(
                    args,
                    config,
                    output_dir,
                    baseline,
                    "baseline",
                    environment,
                )
                baseline_evaluation["reused_selected_evaluation"] = False
            finished = latest(events, "finished")
            finished_summary = {
                "completed_updates": finished.get("completed_updates"),
                "optimizer_kind": finished.get("optimizer_kind"),
                "optimizer_parameterization": finished.get(
                    "optimizer_parameterization"
                ),
                "changed_coordinates": finished.get("changed_coordinates"),
                "changed_quantized_coordinates": finished.get(
                    "changed_quantized_coordinates"
                ),
                "peak_rss_kib": finished.get("peak_rss_kib"),
                "test_read": finished.get("test_read"),
            }
        except ValueError as error:
            status = "unhealthy_output"
            health_error = str(error)
    result = {
        "schema_version": 4,
        "run_id": run_id,
        "graph_representation": config["graph"]["representation"],
        "optimizer_kind": config["optimizer"]["kind"],
        "eta0": config["optimizer"]["eta0"],
        "lambda": config["optimizer"]["lambda"],
        "updates": config["training"]["updates"],
        "validation_every": config["training"]["validation_every"],
        "departure_time_filter": config.get("data", {}).get("departure_time_filter"),
        "status": status,
        "returncode": returncode,
        "training_wall_seconds": training_wall_seconds,
        "total_wall_seconds": time.monotonic() - started,
        "config_path": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "command": command,
        "rayon_threads": args.rayon_threads,
        "training_log": str(training_log.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "health_checks": health_checks,
        "health_error": health_error,
        "validation_checkpoint_candidates": candidates,
        "selection_rule": (
            "minimum validation objective; earliest update breaks exact ties"
        ),
        "selected_checkpoint": selected,
        "selected_evaluation": selected_evaluation,
        "baseline_evaluation": baseline_evaluation,
        "finished": finished_summary,
    }
    atomic_json(output_dir / "runner_result.json", result)
    print(f"{run_id}: {status} ({result['total_wall_seconds']:.2f}s)")
    return 0 if status == "ok" else 1


def main() -> int:
    args = arguments()
    if args.timeout_seconds <= 0 or args.rayon_threads <= 0:
        raise ValueError("timeout and Rayon thread count must be positive")
    runs = [(path, load_config(path)) for path in args.config]
    run_ids = [config["run_id"] for _, config in runs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("duplicate run_id across configurations")
    if not args.dry_run:
        for label, binary in (
            ("training", args.binary),
            ("evaluation", args.evaluate_binary),
        ):
            if not binary.is_file():
                raise FileNotFoundError(f"{label} binary not found: {binary}")
    statuses = [
        run_one(args, path, config, args.output_root / config["run_id"])
        for path, config in runs
    ]
    return max(statuses)


if __name__ == "__main__":
    raise SystemExit(main())
