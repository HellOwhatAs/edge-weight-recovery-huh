#!/usr/bin/env python3
"""Evaluate registered validation checkpoints and select decoded Edge F1."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any


METRICS = (
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
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--evaluate-binary", required=True, type=Path)
    parser.add_argument("--time-buckets", type=Path)
    parser.add_argument("--rayon-threads", default=4, type=int)
    parser.add_argument("--timeout-seconds", default=900, type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_events(path: Path) -> list[dict[str, Any]]:
    events = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        events.append(value)
    return events


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def validate_metrics(metrics: Any, path: Path) -> dict[str, Any]:
    if not isinstance(metrics, dict) or any(name not in metrics for name in METRICS):
        raise ValueError(f"{path}: incomplete metrics")
    if not isinstance(metrics["samples"], int) or metrics["samples"] <= 0:
        raise ValueError(f"{path}: invalid sample count")
    if any(not finite(metrics[name]) for name in METRICS[1:]):
        raise ValueError(f"{path}: non-finite metrics")
    return metrics


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = arguments()
    if args.rayon_threads <= 0 or args.timeout_seconds <= 0:
        raise ValueError("threads and timeout must be positive")
    run_dir = args.run_dir.resolve()
    events = read_events(run_dir / "training.jsonl")
    configurations = [event for event in events if event.get("event") == "configuration"]
    if len(configurations) != 1 or not isinstance(
        configurations[0].get("configuration"), dict
    ):
        raise ValueError("training log must contain exactly one configuration event")
    config = configurations[0]["configuration"]
    if config.get("test_policy") != "never_read":
        raise ValueError("configuration test_policy must be never_read")
    validation_variant = config.get("data", {}).get("validation_variant")
    if not isinstance(validation_variant, str):
        raise ValueError("configuration lacks validation variant")
    departure_filter = config.get("data", {}).get("departure_time_filter")
    if departure_filter is None and args.time_buckets is None:
        raise ValueError(
            "--time-buckets is required when selecting an unfiltered static run"
        )
    if departure_filter is not None and args.time_buckets is not None:
        raise ValueError(
            "--time-buckets must be omitted for an already-filtered static run"
        )

    candidates = []
    for event in events:
        validation = event.get("validation")
        if event.get("event") != "state" or not isinstance(validation, dict):
            continue
        update = event.get("completed_updates")
        objective = validation.get("objective")
        if not isinstance(update, int) or update < 0 or not finite(objective):
            raise ValueError("invalid validation checkpoint event")
        checkpoint = run_dir / f"checkpoint-{update}.json"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        candidates.append(
            {
                "update": update,
                "validation_objective": objective,
                "validation_mean_regret_logged": validation.get("mean_regret"),
                "checkpoint": str(checkpoint),
            }
        )
    if not candidates:
        raise ValueError("no validation checkpoints found")
    candidates.sort(key=lambda candidate: candidate["update"])

    environment = os.environ.copy()
    environment["RAYON_NUM_THREADS"] = str(args.rayon_threads)
    evaluations = []
    for candidate in candidates:
        update = candidate["update"]
        output = run_dir / f"evaluation-route-{update}.json"
        console = run_dir / f"evaluation-route-{update}.log"
        command = [
            str(args.evaluate_binary.resolve()),
            "--checkpoint",
            candidate["checkpoint"],
            "--split",
            "validation",
            "--variant",
            validation_variant,
        ]
        if args.time_buckets is not None:
            command.extend(
                [
                    "--time-buckets",
                    str(args.time_buckets.resolve()),
                ]
            )
        command.extend(["--output", str(output)])
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
        if completed.returncode != 0:
            raise ValueError(f"evaluation update {update} failed; see {console}")
        result = json.loads(output.read_text(encoding="utf-8"))
        if result.get("split") != "validation" or result.get("variant") != validation_variant:
            raise ValueError(f"{output}: evaluator did not use fixed validation")
        if result.get("test_read") is not False:
            raise ValueError(f"{output}: evaluator did not certify test_read=false")
        if result.get("checkpoint_completed_updates") != update:
            raise ValueError(f"{output}: checkpoint update mismatch")
        metrics = validate_metrics(result.get("metrics"), output)
        bucket_evaluation = result.get("time_bucket_evaluation")
        if not isinstance(bucket_evaluation, dict) or not isinstance(
            bucket_evaluation.get("buckets"), list
        ):
            raise ValueError(f"{output}: missing time-bucket metrics")
        bucket_rows = bucket_evaluation["buckets"]
        bucket_sample_count = 0
        for row in bucket_rows:
            if not isinstance(row, dict):
                raise ValueError(f"{output}: invalid time-bucket row")
            bucket_metrics = validate_metrics(row.get("metrics"), output)
            bucket_sample_count += bucket_metrics["samples"]
        if bucket_sample_count != metrics["samples"]:
            raise ValueError(
                f"{output}: bucket samples {bucket_sample_count} differ from "
                f"overall samples {metrics['samples']}"
            )
        metric_totals = result.get("metric_totals")
        if (
            not isinstance(metric_totals, dict)
            or not finite(metric_totals.get("regret_sum"))
            or not finite(metric_totals.get("observed_cost_sum"))
        ):
            raise ValueError(f"{output}: missing additive metric totals")
        evaluations.append(
            {
                **candidate,
                "evaluation_command": command,
                "evaluation_wall_seconds": time.monotonic() - started,
                "evaluation_output": str(output),
                "metrics": metrics,
                "metric_totals": metric_totals,
                "time_buckets": bucket_rows,
                "quantization": result.get("quantization"),
            }
        )

    selected = min(
        evaluations,
        key=lambda candidate: (
            -candidate["metrics"]["edge_f1"],
            -candidate["metrics"]["exact_match"],
            candidate["update"],
        ),
    )
    maximum_update = max(candidate["update"] for candidate in evaluations)
    output = args.output or run_dir / "route_selection.json"
    value = {
        "schema_version": 1,
        "kind": "independent_static_bucket" if departure_filter else "static",
        "run_dir": str(run_dir),
        "selection_rule": (
            "maximum validation Edge F1; maximum Exact Match then earliest update "
            "break exact ties"
        ),
        "validation_variant": validation_variant,
        "departure_time_filter": departure_filter,
        "candidates": evaluations,
        "selected": selected,
        "selected_at_budget_boundary": selected["update"] == maximum_update,
        "test_read": False,
    }
    atomic_json(output.resolve(), value)
    print(
        f"selected update {selected['update']} with Edge F1 "
        f"{selected['metrics']['edge_f1']:.9f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
