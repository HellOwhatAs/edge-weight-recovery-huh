#!/usr/bin/env python3
"""Extract final validation metrics from unified direct-weight training logs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


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
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_events(path: Path) -> list[dict[str, Any]]:
    events = []
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


def finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def summarize(path: Path) -> dict[str, Any]:
    events = read_events(path)
    configuration = latest(events, "configuration")
    evaluations = [
        event
        for event in events
        if event.get("event") == "evaluation"
        and event.get("split") == "validation_final"
    ]
    if not evaluations:
        raise ValueError(f"{path}: missing validation_final evaluation")
    evaluation = evaluations[-1]
    finished = latest(events, "finished")
    if configuration.get("test_read") is not False or finished.get("test_read") is not False:
        raise ValueError(f"{path}: test_read must remain false")

    metrics = evaluation.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{path}: validation metrics must be an object")
    missing = [name for name in STANDARD_METRICS if name not in metrics]
    if missing:
        raise ValueError(f"{path}: missing validation metrics: {missing}")
    if not isinstance(finished.get("completed_updates"), int):
        raise ValueError(f"{path}: finished.completed_updates must be an integer")
    for field in ("train_objective", "validation_objective"):
        if not finite_number(finished.get(field)):
            raise ValueError(f"{path}: finished.{field} must be finite")
    if (
        not isinstance(finished.get("changed_coordinates"), int)
        or finished["changed_coordinates"] <= 0
    ):
        raise ValueError(f"{path}: no direct coordinate changed")
    for field in ("shortest_path_queries_ok", "checkpoint_restore_verified"):
        if finished.get(field) is not True:
            raise ValueError(f"{path}: finished.{field} must be true")

    return {
        "run_id": configuration.get("run_id"),
        "graph_representation": configuration.get("graph_representation"),
        "training_log": str(path),
        "completed_updates": finished["completed_updates"],
        "train_objective": finished.get("train_objective"),
        "validation_objective": finished.get("validation_objective"),
        "validation": {name: metrics[name] for name in STANDARD_METRICS},
        "checkpoint": finished.get("checkpoint_path"),
        "test_read": False,
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = arguments()
    atomic_json(
        args.output,
        {
            "schema_version": 3,
            "training": "unified_direct_weight",
            "runs": [summarize(path) for path in args.logs],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
