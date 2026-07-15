#!/usr/bin/env python3
"""Extract final validation metrics from edge-only JSONL training logs."""

from __future__ import annotations

import argparse
import json
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


def summarize(path: Path) -> dict[str, Any]:
    events = read_events(path)
    configuration = latest(events, "configuration")
    evaluations = [
        event
        for event in events
        if event.get("event") == "evaluation"
        and event.get("split") == "validation_best"
    ]
    if not evaluations:
        raise ValueError(f"{path}: missing validation_best evaluation")
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
    if not isinstance(finished.get("best_epoch"), int):
        raise ValueError(f"{path}: finished.best_epoch must be an integer")

    return {
        "run_id": configuration.get("run_id"),
        "training_log": str(path),
        "selected_epoch": finished["best_epoch"],
        "selection_metric": finished.get("selection_metric"),
        "selection_value": finished.get("selection_value"),
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
            "schema_version": 1,
            "selection_metric": "aggregate_validation_relative_regret",
            "runs": [summarize(path) for path in args.logs],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
