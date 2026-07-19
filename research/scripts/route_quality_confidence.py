#!/usr/bin/env python3
"""Compute route-level confidence intervals beside the strict Rust evaluator.

The Rust evaluator remains the source of point estimates.  This script reads
the same aligned rows, verifies those point estimates, and adds deterministic
large-sample normal intervals plus paired differences to a registered
reference method.  It streams every JSONL file in lockstep and therefore does
not retain the full test corpus or prediction matrices in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO


SCHEMA = "ewr.route-quality-confidence/v1"
EVALUATION_SCHEMA = "ewr.evaluation-summary/v1"
METRICS = (
    "edge_precision",
    "edge_recall",
    "edge_f1",
    "edge_jaccard",
    "exact_match",
    "endpoint_failure_rate",
)


class ConfidenceError(ValueError):
    """An input is malformed, misaligned, or inconsistent with evaluation."""


def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfidenceError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_object(text: str, context: str) -> dict[str, Any]:
    try:
        value = json.loads(text, object_pairs_hook=reject_duplicate_pairs)
    except json.JSONDecodeError as error:
        raise ConfidenceError(f"{context}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ConfidenceError(f"{context}: expected a JSON object")
    return value


def exact_object(
    value: dict[str, Any], expected: set[str], context: str
) -> dict[str, Any]:
    missing = sorted(expected - value.keys())
    extra = sorted(value.keys() - expected)
    if missing or extra:
        raise ConfidenceError(
            f"{context}: missing={missing or None}, unexpected={extra or None}"
        )
    return value


def identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value or any(ord(ch) < 32 for ch in value):
        raise ConfidenceError(f"{context}: invalid identifier")
    return value


def edge_list(value: Any, context: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            isinstance(edge, bool)
            or not isinstance(edge, int)
            or edge < 0
            or edge > 0xFFFF_FFFF
            for edge in value
        )
    ):
        raise ConfidenceError(f"{context}: expected nonempty uint32 edge array")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class RunningStats:
    count: int = 0
    total: float = 0.0
    total_squares: float = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.total_squares += value * value

    def summary(self, z_value: float, *, bounded: bool) -> dict[str, float]:
        if self.count < 2:
            raise ConfidenceError("at least two samples are required for an interval")
        mean = self.total / self.count
        centered = self.total_squares - self.total * self.total / self.count
        variance = max(0.0, centered / (self.count - 1))
        standard_error = math.sqrt(variance / self.count)
        lower = mean - z_value * standard_error
        upper = mean + z_value * standard_error
        if bounded:
            lower, upper = max(0.0, lower), min(1.0, upper)
        return {
            "mean": mean,
            "standard_error": standard_error,
            "lower": lower,
            "upper": upper,
        }


def route_metrics(truth: list[int], prediction: list[int]) -> dict[str, float]:
    truth_set, prediction_set = set(truth), set(prediction)
    intersection = len(truth_set & prediction_set)
    precision = intersection / len(prediction_set)
    recall = intersection / len(truth_set)
    f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
    return {
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "edge_jaccard": intersection / len(truth_set | prediction_set),
        "exact_match": float(truth == prediction),
        "endpoint_failure_rate": float(
            prediction[0] != truth[0] or prediction[-1] != truth[-1]
        ),
    }


def parse_named_path(raw: str, option: str) -> tuple[str, Path]:
    name, separator, path_text = raw.partition("=")
    if not separator or not name or not path_text:
        raise ConfidenceError(f"{option} must be METHOD=PATH")
    if any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_" for ch in name):
        raise ConfidenceError(f"{option}: invalid method ID {name!r}")
    path = Path(path_text).resolve()
    if not path.is_file():
        raise ConfidenceError(f"{option}: file does not exist: {path}")
    return name, path


def named_paths(values: list[str], option: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        name, path = parse_named_path(raw, option)
        if name in result:
            raise ConfidenceError(f"{option}: duplicate method {name!r}")
        result[name] = path
    if not result:
        raise ConfidenceError(f"at least one {option} is required")
    return result


def load_evaluation(path: Path) -> dict[str, Any]:
    obj = load_object(path.read_text(encoding="utf-8"), str(path))
    exact_object(obj, {"schema", "sample_count", "metrics"}, str(path))
    if obj["schema"] != EVALUATION_SCHEMA:
        raise ConfidenceError(f"{path}: unsupported evaluation schema")
    if isinstance(obj["sample_count"], bool) or not isinstance(obj["sample_count"], int):
        raise ConfidenceError(f"{path}: invalid sample_count")
    metrics = obj["metrics"]
    if not isinstance(metrics, dict):
        raise ConfidenceError(f"{path}: metrics must be an object")
    exact_object(metrics, set(METRICS) - {"endpoint_failure_rate"}, f"{path}.metrics")
    for name, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ConfidenceError(f"{path}.metrics.{name}: invalid value")
    return obj


def next_prediction(
    source: TextIO, method: str, row_number: int
) -> tuple[str, list[int]]:
    line = source.readline()
    if not line:
        raise ConfidenceError(f"{method}: predictions ended before row {row_number}")
    obj = exact_object(
        load_object(line, f"{method} prediction row {row_number}"),
        {"sample_id", "predicted_edge_ids"},
        f"{method} prediction row {row_number}",
    )
    return (
        identifier(obj["sample_id"], f"{method} row {row_number}.sample_id"),
        edge_list(
            obj["predicted_edge_ids"],
            f"{method} row {row_number}.predicted_edge_ids",
        ),
    )


def evaluate(
    dataset: Path,
    predictions: dict[str, Path],
    evaluations: dict[str, Path],
    reference: str,
    confidence_level: float,
    progress_every: int,
) -> dict[str, Any]:
    if predictions.keys() != evaluations.keys():
        raise ConfidenceError("prediction and evaluation method sets differ")
    if reference not in predictions:
        raise ConfidenceError(f"reference method {reference!r} is absent")
    if not 0.0 < confidence_level < 1.0:
        raise ConfidenceError("confidence level must be in (0, 1)")
    if progress_every <= 0:
        raise ConfidenceError("progress interval must be positive")
    methods = list(predictions)
    method_stats = {
        method: {metric: RunningStats() for metric in METRICS} for method in methods
    }
    difference_stats = {
        method: {metric: RunningStats() for metric in METRICS}
        for method in methods
        if method != reference
    }
    seen: set[str] = set()
    started = time.monotonic()
    with ExitStack() as stack:
        dataset_source = stack.enter_context(dataset.open("r", encoding="utf-8"))
        prediction_sources = {
            method: stack.enter_context(path.open("r", encoding="utf-8"))
            for method, path in predictions.items()
        }
        count = 0
        for row_number, line in enumerate(dataset_source, 1):
            obj = exact_object(
                load_object(line, f"dataset row {row_number}"),
                {"sample_id", "original_edge_ids"},
                f"dataset row {row_number}",
            )
            sample_id = identifier(obj["sample_id"], f"dataset row {row_number}.sample_id")
            if sample_id in seen:
                raise ConfidenceError(f"dataset row {row_number}: duplicate sample_id {sample_id!r}")
            seen.add(sample_id)
            truth = edge_list(obj["original_edge_ids"], f"dataset row {row_number}.original_edge_ids")
            row_metrics: dict[str, dict[str, float]] = {}
            for method in methods:
                predicted_id, predicted = next_prediction(
                    prediction_sources[method], method, row_number
                )
                if predicted_id != sample_id:
                    raise ConfidenceError(
                        f"{method} row {row_number}: sample_id {predicted_id!r} "
                        f"does not match {sample_id!r}"
                    )
                metrics = route_metrics(truth, predicted)
                row_metrics[method] = metrics
                for metric, value in metrics.items():
                    method_stats[method][metric].add(value)
            reference_metrics = row_metrics[reference]
            for method, stats in difference_stats.items():
                for metric in METRICS:
                    stats[metric].add(
                        row_metrics[method][metric] - reference_metrics[metric]
                    )
            count += 1
            if count % progress_every == 0:
                elapsed = time.monotonic() - started
                print(
                    f"confidence rows={count} rate={count / max(elapsed, 1e-9):.1f}/s",
                    file=sys.stderr,
                    flush=True,
                )
        if count < 2:
            raise ConfidenceError("dataset must contain at least two rows")
        for method, source in prediction_sources.items():
            if source.readline():
                raise ConfidenceError(f"{method}: predictions contain extra rows")

    z_value = statistics.NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    evaluation_objects = {
        method: load_evaluation(path) for method, path in evaluations.items()
    }
    method_output: dict[str, Any] = {}
    for method in methods:
        evaluation = evaluation_objects[method]
        if evaluation["sample_count"] != count:
            raise ConfidenceError(f"{method}: evaluation sample_count differs")
        intervals = {
            metric: method_stats[method][metric].summary(z_value, bounded=True)
            for metric in METRICS
        }
        for metric, expected in evaluation["metrics"].items():
            if not math.isclose(
                intervals[metric]["mean"], float(expected), rel_tol=0.0, abs_tol=1e-12
            ):
                raise ConfidenceError(
                    f"{method}: {metric} mean differs from strict evaluator: "
                    f"{intervals[metric]['mean']} != {expected}"
                )
        method_output[method] = {
            "prediction_path": str(predictions[method]),
            "prediction_sha256": sha256(predictions[method]),
            "evaluation_path": str(evaluations[method]),
            "evaluation_sha256": sha256(evaluations[method]),
            "endpoint_failures": round(
                intervals["endpoint_failure_rate"]["mean"] * count
            ),
            "intervals": intervals,
        }

    return {
        "schema": SCHEMA,
        "sample_count": count,
        "dataset_path": str(dataset),
        "dataset_sha256": sha256(dataset),
        "reference_method": reference,
        "confidence": {
            "level": confidence_level,
            "method": "route_level_normal_mean_interval",
            "z_value": z_value,
            "aggregation": "unweighted_routes",
        },
        "methods": method_output,
        "paired_differences_vs_reference": {
            method: {
                metric: stats[metric].summary(z_value, bounded=False)
                for metric in METRICS
            }
            for method, stats in difference_stats.items()
        },
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as writer:
            json.dump(value, writer, indent=2, sort_keys=True)
            writer.write("\n")
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prediction", action="append", default=[])
    parser.add_argument("--evaluation", action="append", default=[])
    parser.add_argument("--reference", default="project")
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = args.dataset.resolve()
    if not dataset.is_file():
        raise ConfidenceError(f"dataset does not exist: {dataset}")
    result = evaluate(
        dataset,
        named_paths(args.prediction, "--prediction"),
        named_paths(args.evaluation, "--evaluation"),
        args.reference,
        args.confidence_level,
        args.progress_every,
    )
    atomic_json(args.output.resolve(), result)
    print(json.dumps({"output": str(args.output.resolve()), "samples": result["sample_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
