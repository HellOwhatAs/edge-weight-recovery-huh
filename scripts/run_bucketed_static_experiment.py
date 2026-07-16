#!/usr/bin/env python3
"""Run independent static models on pre-registered departure-time partitions."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any


QUALITY_METRICS = (
    "edge_precision",
    "edge_recall",
    "edge_f1",
    "exact_match",
    "edge_jaccard",
    "mean_regret",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--matrix-runner",
        default=Path("scripts/run_experiment_matrix.py"),
        type=Path,
    )
    parser.add_argument(
        "--selector", default=Path("scripts/select_route_checkpoint.py"), type=Path
    )
    parser.add_argument("--binary", default=Path("target/release/train"), type=Path)
    parser.add_argument(
        "--evaluate-binary", default=Path("target/release/evaluate"), type=Path
    )
    parser.add_argument("--timeout-seconds", default=10800, type=int)
    parser.add_argument("--rayon-threads", default=4, type=int)
    return parser.parse_args()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def shared_static_configuration(config: dict[str, Any]) -> dict[str, Any]:
    shared = copy.deepcopy(config)
    shared.pop("run_id", None)
    shared.pop("description", None)
    data = shared.get("data")
    if not isinstance(data, dict):
        raise ValueError("bucket configuration lacks a data object")
    data.pop("departure_time_filter", None)
    return shared


def validate_configs(
    paths: list[Path],
) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
    configs = [load_object(path) for path in paths]
    if not configs:
        raise ValueError("at least one bucket config is required")
    reference = configs[0]
    shared_reference = shared_static_configuration(reference)
    filters = []
    for path, config in zip(paths, configs, strict=True):
        if config.get("schema_version") != 3 or config.get("test_policy") != "never_read":
            raise ValueError(f"{path}: expected schema 3 and never_read")
        if config.get("graph", {}).get("representation") != "edge_transition_arcs":
            raise ValueError(f"{path}: expected edge_transition_arcs")
        if (
            config.get("optimizer", {}).get("kind")
            != "relative_projected_subgradient"
        ):
            raise ValueError(f"{path}: expected relative_projected_subgradient")
        if shared_static_configuration(config) != shared_reference:
            raise ValueError(
                f"{path}: bucket configs may differ only by run_id, description, "
                "and departure_time_filter"
            )
        departure_filter = config.get("data", {}).get("departure_time_filter")
        if not isinstance(departure_filter, dict):
            raise ValueError(f"{path}: missing departure_time_filter")
        if departure_filter.get("selection_timestamp") != "start_time":
            raise ValueError(f"{path}: bucket selection must use start_time")
        filters.append(departure_filter)

    spec_paths = {entry.get("spec_path") for entry in filters}
    spec_hashes = {entry.get("spec_sha256") for entry in filters}
    if len(spec_paths) != 1 or len(spec_hashes) != 1:
        raise ValueError("bucket configs do not share one specification")
    spec_path = Path(next(iter(spec_paths)))
    expected_hash = next(iter(spec_hashes))
    actual_hash = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(
            f"{spec_path}: expected SHA-256 {expected_hash}, got {actual_hash}"
        )
    spec = load_object(spec_path)
    registered_ids = [bucket["id"] for bucket in spec["buckets"]]
    configured_ids = [entry.get("bucket_id") for entry in filters]
    if len(configured_ids) != len(set(configured_ids)):
        raise ValueError("duplicate configured time bucket")
    if set(configured_ids) != set(registered_ids):
        raise ValueError("configs must cover every registered bucket exactly once")
    return configs, spec_path, spec


def weighted_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample_count = sum(row["metrics"]["samples"] for row in rows)
    if sample_count <= 0:
        raise ValueError("bucket aggregation has no validation samples")
    metrics: dict[str, Any] = {"samples": sample_count}
    for name in QUALITY_METRICS:
        metrics[name] = (
            sum(row["metrics"][name] * row["metrics"]["samples"] for row in rows)
            / sample_count
        )
    regret_sum = sum(row["metric_totals"]["regret_sum"] for row in rows)
    observed_cost_sum = sum(
        row["metric_totals"]["observed_cost_sum"] for row in rows
    )
    metrics["relative_regret"] = (
        0.0 if observed_cost_sum == 0.0 else regret_sum / observed_cost_sum
    )
    if any(not finite(metrics[name]) for name in metrics if name != "samples"):
        raise ValueError("bucket aggregation produced a non-finite metric")
    return metrics


def main() -> int:
    args = arguments()
    if args.timeout_seconds <= 0 or args.rayon_threads <= 0:
        raise ValueError("timeout and thread count must be positive")
    config_paths = [path.resolve() for path in args.config]
    configs, spec_path, spec = validate_configs(config_paths)
    for path in (
        args.matrix_runner,
        args.selector,
        args.binary,
        args.evaluate_binary,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    matrix_command = ["python3", str(args.matrix_runner.resolve())]
    for config_path in config_paths:
        matrix_command.extend(["--config", str(config_path)])
    matrix_command.extend(
        [
            "--output-root",
            str(output_root),
            "--binary",
            str(args.binary.resolve()),
            "--evaluate-binary",
            str(args.evaluate_binary.resolve()),
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--rayon-threads",
            str(args.rayon_threads),
        ]
    )
    environment = os.environ.copy()
    environment["RAYON_NUM_THREADS"] = str(args.rayon_threads)
    started = time.monotonic()
    matrix_log = output_root / "matrix.log"
    with matrix_log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            matrix_command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
            timeout=args.timeout_seconds * len(configs) + 1800,
            check=False,
            text=True,
        )
    if completed.returncode != 0:
        raise ValueError(f"static matrix failed; see {matrix_log}")

    selection_commands = []
    selections: dict[str, dict[str, Any]] = {}
    runners: dict[str, dict[str, Any]] = {}
    for config in configs:
        run_id = config["run_id"]
        run_dir = output_root / run_id
        command = [
            "python3",
            str(args.selector.resolve()),
            "--run-dir",
            str(run_dir),
            "--evaluate-binary",
            str(args.evaluate_binary.resolve()),
            "--rayon-threads",
            str(args.rayon_threads),
            "--timeout-seconds",
            "900",
        ]
        selection_commands.append(command)
        log = run_dir / "route_selection.log"
        with log.open("w", encoding="utf-8") as stream:
            selected = subprocess.run(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=environment,
                timeout=1800,
                check=False,
                text=True,
            )
        if selected.returncode != 0:
            raise ValueError(f"route selection failed for {run_id}; see {log}")
        selection = load_object(run_dir / "route_selection.json")
        runner = load_object(run_dir / "runner_result.json")
        if selection.get("test_read") is not False or runner.get("status") != "ok":
            raise ValueError(f"{run_id}: unhealthy or test-read output")
        bucket_id = config["data"]["departure_time_filter"]["bucket_id"]
        selected_metrics = selection["selected"]["metrics"]
        expected = config["data"]["departure_time_filter"][
            "expected_validation_samples"
        ]
        if selected_metrics["samples"] != expected:
            raise ValueError(f"{run_id}: selected validation count mismatch")
        selections[bucket_id] = selection
        runners[bucket_id] = runner

    rows = []
    config_by_bucket = {
        config["data"]["departure_time_filter"]["bucket_id"]: config
        for config in configs
    }
    for bucket in spec["buckets"]:
        bucket_id = bucket["id"]
        config = config_by_bucket[bucket_id]
        selection = selections[bucket_id]
        selected = selection["selected"]
        rows.append(
            {
                "id": bucket_id,
                "label": bucket["label"],
                "start_hour": bucket["start_hour"],
                "end_hour": bucket["end_hour"],
                "train_samples": config["data"]["departure_time_filter"][
                    "expected_train_samples"
                ],
                "validation_samples": config["data"]["departure_time_filter"][
                    "expected_validation_samples"
                ],
                "run_id": config["run_id"],
                "selected_update": selected["update"],
                "selected_at_budget_boundary": selection[
                    "selected_at_budget_boundary"
                ],
                "metrics": selected["metrics"],
                "metric_totals": selected["metric_totals"],
                "quantization": selected.get("quantization"),
                "checkpoint": selected["checkpoint"],
                "route_selection": str(
                    (output_root / config["run_id"] / "route_selection.json").resolve()
                ),
                "training_wall_seconds": runners[bucket_id][
                    "training_wall_seconds"
                ],
                "peak_rss_kib": runners[bucket_id]["finished"]["peak_rss_kib"],
            }
        )

    result = {
        "schema_version": 1,
        "study": "independent_departure_bucket_static_models",
        "model_boundary": {
            "time_role": "data selection and checkpoint dispatch only",
            "graph_representation": "edge_transition_arcs",
            "optimizer": "relative_projected_subgradient",
            "baseline": "length",
            "checkpoint": "ordinary static direct-weight checkpoint",
        },
        "time_bucket_specification_path": str(spec_path),
        "time_bucket_specification": spec,
        "selection_rule": (
            "maximum validation Edge F1 within each bucket; maximum Exact Match "
            "then earliest update break exact ties"
        ),
        "metrics": weighted_metrics(rows),
        "buckets": rows,
        "configurations": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in config_paths
        ],
        "matrix_command": matrix_command,
        "selection_commands": selection_commands,
        "total_wall_seconds": time.monotonic() - started,
        "rayon_threads": args.rayon_threads,
        "test_read": False,
    }
    atomic_json(output_root / "bucketed_runner_result.json", result)
    print(
        "independent bucketed static models: "
        f"F1={result['metrics']['edge_f1']:.9f}, "
        f"Exact={result['metrics']['exact_match']:.9f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
