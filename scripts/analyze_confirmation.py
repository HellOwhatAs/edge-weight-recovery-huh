#!/usr/bin/env python3
"""Validate and analyze the preregistered one-shot confirmation evaluation.

The script deliberately reads only the confirmation plan, the six compact
evaluation summaries, the six route-level JSON exports, and the three frozen
JSON checkpoints named by the plan.  It never discovers or opens trip pickle
files or a test split.

For each AM/PM block, all models must have exactly aligned
``(route_index, source, target)`` rows.  Paired bootstrap samples reuse the
same route indices for every model.  The pooled result is stratified: each
replicate resamples within AM and PM independently while preserving each
block's accepted-route count.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from typing import Any, Iterable, Mapping

try:
    import numpy as np
except ImportError as error:  # pragma: no cover - exercised only without numpy
    raise SystemExit(
        "analyze_confirmation.py requires NumPy for the preregistered paired "
        "bootstrap"
    ) from error


DEFAULT_PLAN = Path("experiments/convergence_study/confirmation_plan.json")
DEFAULT_SUMMARY_DIR = Path("experiments/convergence_study/confirmation")
DEFAULT_ROUTES_DIR = Path("/tmp/edge-weight-convergence-study/confirmation")
DEFAULT_OUTPUT_JSON = Path(
    "experiments/convergence_study/confirmation_summary.json"
)
DEFAULT_OUTPUT_CSV = Path("experiments/convergence_study/confirmation_summary.csv")

EXPECTED_MODELS = (
    "baseline_q1",
    "edge_t20_eta1e4",
    "edge_t100_eta3e4",
)
EXPECTED_BLOCKS = ("am", "pm")
COMPARISONS = (
    ("baseline_q1", "edge_t20_eta1e4", "baseline_to_t20"),
    ("baseline_q1", "edge_t100_eta3e4", "baseline_to_t100"),
    ("edge_t20_eta1e4", "edge_t100_eta3e4", "t20_to_t100"),
)
METRICS = (
    "aggregate_relative_regret",
    "mean_edge_f1",
    "exact_match_rate",
)
LOWER_IS_BETTER = {"aggregate_relative_regret"}
PREREGISTERED_SEED = 20260718
PREREGISTERED_REPLICATES = 2000
BOOTSTRAP_BATCH_SIZE = 32
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
TEST_PATH_TOKEN = re.compile(r"(?:^|[_.-])tests?(?:$|[_.-])", re.IGNORECASE)


class AnalysisError(RuntimeError):
    """Raised when an input violates the frozen confirmation contract."""


@dataclass(frozen=True)
class RouteData:
    identities: np.ndarray
    raw_regret: np.ndarray
    observed_cost: np.ndarray
    edge_f1: np.ndarray
    exact_match: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.raw_regret.shape[0])


@dataclass(frozen=True)
class MetricSums:
    raw_regret: np.ndarray
    observed_cost: np.ndarray
    edge_f1: np.ndarray
    exact_match: np.ndarray
    sample_count: int


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--routes-dir", type=Path, default=DEFAULT_ROUTES_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run synthetic integration and invariant tests, then exit",
    )
    return parser.parse_args(argv)


def safe_input_path(path: Path, label: str, *, json_only: bool = True) -> Path:
    """Resolve an input and reject pickle or explicit test-split paths."""
    resolved = path.expanduser().resolve()
    if json_only and resolved.suffix.lower() != ".json":
        raise AnalysisError(f"{label} must be JSON, got {resolved}")
    if resolved.suffix.lower() in {".pickle", ".pkl", ".pck"}:
        raise AnalysisError(f"refusing to read pickle input for {label}: {resolved}")
    if any(TEST_PATH_TOKEN.search(part) for part in resolved.parts):
        raise AnalysisError(f"refusing to read a test-path input for {label}: {resolved}")
    if not resolved.is_file():
        raise AnalysisError(f"missing {label}: {resolved}")
    return resolved


def load_json(path: Path, label: str) -> tuple[dict[str, Any], Path]:
    resolved = safe_input_path(path, label)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalysisError(f"cannot read {label} {resolved}: {error}") from error
    if not isinstance(value, dict):
        raise AnalysisError(f"{label} must contain a top-level JSON object")
    return value, resolved


def sha256_file(path: Path, label: str) -> str:
    resolved = safe_input_path(path, label)
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise AnalysisError(f"cannot hash {label} {resolved}: {error}") from error
    return digest.hexdigest()


def require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AnalysisError(f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise AnalysisError(f"{label} must be an array")
    return value


def require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnalysisError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise AnalysisError(f"{label} must be >= {minimum}, got {value}")
    return value


def require_number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AnalysisError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise AnalysisError(f"{label} must be >= {minimum}, got {result}")
    if maximum is not None and result > maximum:
        raise AnalysisError(f"{label} must be <= {maximum}, got {result}")
    return result


def validate_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise AnalysisError(f"{label} is not a safe identifier: {value!r}")
    return value


def same_path(left: Any, right: Path, label: str) -> None:
    if not isinstance(left, str):
        raise AnalysisError(f"{label} must be a path string")
    if Path(left).expanduser().resolve() != right.resolve():
        raise AnalysisError(f"{label} points to {left!r}, expected {str(right)!r}")


def close_enough(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-11, abs_tol=1e-13)


def validate_plan(plan: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    require_int(plan.get("schema_version"), "plan.schema_version", minimum=1)
    if plan.get("status") != "frozen_before_confirmation_evaluation":
        raise AnalysisError("confirmation plan is not in its frozen preregistration state")
    test_policy = plan.get("test_policy")
    if not isinstance(test_policy, str) or "no test" not in test_policy.lower():
        raise AnalysisError("plan.test_policy must explicitly prohibit test access")

    models = [
        require_mapping(value, f"plan.models[{index}]")
        for index, value in enumerate(require_list(plan.get("models"), "plan.models"))
    ]
    model_ids = [
        validate_identifier(model.get("model_id"), f"plan.models[{index}].model_id")
        for index, model in enumerate(models)
    ]
    if tuple(model_ids) != EXPECTED_MODELS:
        raise AnalysisError(
            f"expected models {EXPECTED_MODELS}, found {tuple(model_ids)}"
        )
    for model in models:
        digest = model.get("checkpoint_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise AnalysisError(
                f"invalid checkpoint SHA for model {model.get('model_id')!r}"
            )

    blocks = [
        require_mapping(value, f"plan.confirmation_blocks[{index}]")
        for index, value in enumerate(
            require_list(plan.get("confirmation_blocks"), "plan.confirmation_blocks")
        )
    ]
    block_ids = [
        validate_identifier(block.get("block_id"), f"confirmation_blocks[{index}].block_id")
        for index, block in enumerate(blocks)
    ]
    if tuple(block_ids) != EXPECTED_BLOCKS:
        raise AnalysisError(
            f"expected confirmation blocks {EXPECTED_BLOCKS}, found {tuple(block_ids)}"
        )
    for block in blocks:
        if not isinstance(block.get("validation_variant"), str):
            raise AnalysisError(
                f"validation variant missing for block {block.get('block_id')!r}"
            )
        require_int(
            block.get("raw_records"),
            f"block {block.get('block_id')}.raw_records",
            minimum=1,
        )

    evaluation = require_mapping(plan.get("evaluation"), "plan.evaluation")
    seed = require_int(evaluation.get("bootstrap_seed"), "bootstrap_seed", minimum=0)
    replicates = require_int(
        evaluation.get("bootstrap_replicates"), "bootstrap_replicates", minimum=1
    )
    if seed != PREREGISTERED_SEED or replicates != PREREGISTERED_REPLICATES:
        raise AnalysisError(
            "bootstrap settings differ from the preregistered "
            f"seed={PREREGISTERED_SEED}, replicates={PREREGISTERED_REPLICATES}"
        )
    return models, blocks


def expected_input_files(
    summary_dir: Path,
    routes_dir: Path,
    models: Iterable[Mapping[str, Any]],
    blocks: Iterable[Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], Path], dict[tuple[str, str], Path]]:
    summary_paths: dict[tuple[str, str], Path] = {}
    route_paths: dict[tuple[str, str], Path] = {}
    for model in models:
        model_id = str(model["model_id"])
        for block in blocks:
            block_id = str(block["block_id"])
            key = (model_id, block_id)
            summary_paths[key] = summary_dir / f"{model_id}_{block_id}.json"
            route_paths[key] = routes_dir / f"{model_id}_{block_id}_routes.json"

    if not summary_dir.resolve().is_dir():
        raise AnalysisError(f"missing confirmation summary directory: {summary_dir}")
    if not routes_dir.resolve().is_dir():
        raise AnalysisError(f"missing route export directory: {routes_dir}")
    actual_summaries = {path.name for path in summary_dir.resolve().glob("*.json")}
    actual_routes = {
        path.name for path in routes_dir.resolve().glob("*_routes.json")
    }
    expected_summaries = {path.name for path in summary_paths.values()}
    expected_routes = {path.name for path in route_paths.values()}
    if actual_summaries != expected_summaries:
        raise AnalysisError(
            "confirmation summary set is incomplete or unexpected: "
            f"missing={sorted(expected_summaries - actual_summaries)}, "
            f"unexpected={sorted(actual_summaries - expected_summaries)}"
        )
    if actual_routes != expected_routes:
        raise AnalysisError(
            "route export set is incomplete or unexpected: "
            f"missing={sorted(expected_routes - actual_routes)}, "
            f"unexpected={sorted(actual_routes - expected_routes)}"
        )
    return summary_paths, route_paths


def extract_route_data(document: dict[str, Any], label: str) -> RouteData:
    evaluation = require_mapping(document.get("evaluation"), f"{label}.evaluation")
    routes = require_list(evaluation.get("routes"), f"{label}.evaluation.routes")
    if not routes:
        raise AnalysisError(f"{label} contains no accepted routes")

    identities = np.empty((len(routes), 3), dtype=np.int64)
    raw_regret = np.empty(len(routes), dtype=np.float64)
    observed_cost = np.empty(len(routes), dtype=np.float64)
    edge_f1 = np.empty(len(routes), dtype=np.float64)
    exact_match = np.empty(len(routes), dtype=np.float64)
    for index, value in enumerate(routes):
        route = require_mapping(value, f"{label}.routes[{index}]")
        route_index = require_int(route.get("route_index"), f"{label}[{index}].route_index")
        if route_index != index:
            raise AnalysisError(
                f"{label} route_index is not sequential at row {index}: {route_index}"
            )
        identities[index, 0] = route_index
        identities[index, 1] = require_int(
            route.get("source"), f"{label}[{index}].source", minimum=0
        )
        identities[index, 2] = require_int(
            route.get("target"), f"{label}[{index}].target", minimum=0
        )
        raw_regret[index] = require_number(
            route.get("raw_regret"), f"{label}[{index}].raw_regret", minimum=0.0
        )
        observed_cost[index] = require_number(
            route.get("observed_path_cost"),
            f"{label}[{index}].observed_path_cost",
            minimum=0.0,
        )
        if observed_cost[index] <= 0.0:
            raise AnalysisError(f"{label}[{index}].observed_path_cost must be positive")
        edge_f1[index] = require_number(
            route.get("edge_f1"), f"{label}[{index}].edge_f1", minimum=0.0, maximum=1.0
        )
        exact = route.get("exact_match")
        if not isinstance(exact, bool):
            raise AnalysisError(f"{label}[{index}].exact_match must be boolean")
        exact_match[index] = float(exact)

    return RouteData(
        identities=identities,
        raw_regret=raw_regret,
        observed_cost=observed_cost,
        edge_f1=edge_f1,
        exact_match=exact_match,
    )


def point_sums(data: RouteData) -> MetricSums:
    return MetricSums(
        raw_regret=np.asarray([data.raw_regret.sum(dtype=np.float64)]),
        observed_cost=np.asarray([data.observed_cost.sum(dtype=np.float64)]),
        edge_f1=np.asarray([data.edge_f1.sum(dtype=np.float64)]),
        exact_match=np.asarray([data.exact_match.sum(dtype=np.float64)]),
        sample_count=data.sample_count,
    )


def add_sums(left: MetricSums, right: MetricSums) -> MetricSums:
    if left.raw_regret.shape != right.raw_regret.shape:
        raise AnalysisError("cannot pool bootstrap arrays with different replicate counts")
    return MetricSums(
        raw_regret=left.raw_regret + right.raw_regret,
        observed_cost=left.observed_cost + right.observed_cost,
        edge_f1=left.edge_f1 + right.edge_f1,
        exact_match=left.exact_match + right.exact_match,
        sample_count=left.sample_count + right.sample_count,
    )


def metrics_from_sums(sums: MetricSums) -> dict[str, np.ndarray]:
    return {
        "aggregate_relative_regret": sums.raw_regret / sums.observed_cost,
        "mean_edge_f1": sums.edge_f1 / float(sums.sample_count),
        "exact_match_rate": sums.exact_match / float(sums.sample_count),
    }


def bootstrap_block(
    data_by_model: Mapping[str, RouteData],
    rng: np.random.Generator,
    replicates: int,
) -> dict[str, MetricSums]:
    counts = {data.sample_count for data in data_by_model.values()}
    if len(counts) != 1:
        raise AnalysisError(f"model route counts differ within a block: {sorted(counts)}")
    sample_count = counts.pop()
    result = {
        model_id: MetricSums(
            raw_regret=np.empty(replicates, dtype=np.float64),
            observed_cost=np.empty(replicates, dtype=np.float64),
            edge_f1=np.empty(replicates, dtype=np.float64),
            exact_match=np.empty(replicates, dtype=np.float64),
            sample_count=sample_count,
        )
        for model_id in data_by_model
    }

    for start in range(0, replicates, BOOTSTRAP_BATCH_SIZE):
        stop = min(start + BOOTSTRAP_BATCH_SIZE, replicates)
        indices = rng.integers(
            0,
            sample_count,
            size=(stop - start, sample_count),
            dtype=np.int32,
        )
        # Exactly the same `indices` matrix is used for every model: this is
        # the paired component of the preregistered route bootstrap.
        for model_id, data in data_by_model.items():
            target = result[model_id]
            target.raw_regret[start:stop] = data.raw_regret[indices].sum(
                axis=1, dtype=np.float64
            )
            target.observed_cost[start:stop] = data.observed_cost[indices].sum(
                axis=1, dtype=np.float64
            )
            target.edge_f1[start:stop] = data.edge_f1[indices].sum(
                axis=1, dtype=np.float64
            )
            target.exact_match[start:stop] = data.exact_match[indices].sum(
                axis=1, dtype=np.float64
            )
    return result


def scalar_metrics(sums: MetricSums) -> dict[str, float]:
    values = metrics_from_sums(sums)
    return {metric: float(array[0]) for metric, array in values.items()}


def validate_overall(
    summary: Mapping[str, Any], expected: Mapping[str, float], label: str
) -> None:
    overall = require_mapping(
        require_mapping(summary.get("evaluation"), f"{label}.evaluation").get("overall"),
        f"{label}.evaluation.overall",
    )
    reported = {
        "aggregate_relative_regret": require_number(
            require_mapping(overall.get("relative_regret"), f"{label}.relative_regret").get(
                "aggregate"
            ),
            f"{label}.relative_regret.aggregate",
        ),
        "mean_edge_f1": require_number(
            overall.get("mean_edge_f1"), f"{label}.mean_edge_f1"
        ),
        "exact_match_rate": require_number(
            overall.get("exact_match_rate"), f"{label}.exact_match_rate"
        ),
    }
    for metric in METRICS:
        if not close_enough(reported[metric], expected[metric]):
            raise AnalysisError(
                f"{label} {metric} disagrees with its route rows: "
                f"reported={reported[metric]}, recomputed={expected[metric]}"
            )


def validate_document_pair(
    summary: dict[str, Any],
    route_document: dict[str, Any],
    summary_path: Path,
    route_path: Path,
    model: Mapping[str, Any],
    block: Mapping[str, Any],
    plan: Mapping[str, Any],
    route_data: RouteData,
) -> None:
    model_id = str(model["model_id"])
    block_id = str(block["block_id"])
    label = f"{model_id}/{block_id}"
    checkpoint_path = safe_input_path(
        Path(str(model["checkpoint"])), f"checkpoint for {model_id}"
    )
    expected_variant = block["validation_variant"]
    expected_train_variant = plan.get("train_variant_for_seen_edge_strata")
    for kind, document in (("summary", summary), ("routes", route_document)):
        same_path(
            document.get("checkpoint_path"),
            checkpoint_path,
            f"{label} {kind}.checkpoint_path",
        )
        if document.get("validation_variant") != expected_variant:
            raise AnalysisError(
                f"{label} {kind} has wrong validation variant: "
                f"{document.get('validation_variant')!r}"
            )
        if document.get("train_variant") != expected_train_variant:
            raise AnalysisError(
                f"{label} {kind} has wrong train variant: "
                f"{document.get('train_variant')!r}"
            )
        if document.get("city") != plan.get("city"):
            raise AnalysisError(f"{label} {kind} has wrong city")
        report = require_mapping(
            document.get("validation_validation_report"),
            f"{label} {kind}.validation_validation_report",
        )
        accepted = require_int(report.get("accepted"), f"{label} {kind}.accepted")
        available = require_int(report.get("available"), f"{label} {kind}.available")
        if accepted != route_data.sample_count:
            raise AnalysisError(
                f"{label} {kind} accepted={accepted}, routes={route_data.sample_count}"
            )
        if available != block["raw_records"]:
            raise AnalysisError(
                f"{label} {kind} available={available}, preregistered raw_records="
                f"{block['raw_records']}"
            )

    same_path(summary.get("routes_output"), route_path, f"{label} routes_output")
    for key in (
        "checkpoint_metadata",
        "train_validation_report",
        "validation_validation_report",
    ):
        if summary.get(key) != route_document.get(key):
            raise AnalysisError(f"{label} {key} differs between summary and route export")

    expected_metrics = scalar_metrics(point_sums(route_data))
    validate_overall(summary, expected_metrics, f"{label} summary")
    validate_overall(route_document, expected_metrics, f"{label} routes")
    overall_count = require_int(
        require_mapping(
            require_mapping(summary.get("evaluation"), f"{label}.evaluation").get(
                "overall"
            ),
            f"{label}.overall",
        ).get("sample_count"),
        f"{label}.overall.sample_count",
    )
    if overall_count != route_data.sample_count:
        raise AnalysisError(
            f"{label} overall sample_count={overall_count}, routes={route_data.sample_count}"
        )


def comparison_record(
    scope: str,
    block_ids: list[str],
    sample_count: int,
    from_model: str,
    to_model: str,
    comparison_id: str,
    point_metrics: Mapping[str, Mapping[str, float]],
    bootstrap_metrics: Mapping[str, Mapping[str, np.ndarray]],
    replicates: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for metric in METRICS:
        direction = "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better"
        point_from = float(point_metrics[from_model][metric])
        point_to = float(point_metrics[to_model][metric])
        bootstrap_from = bootstrap_metrics[from_model][metric]
        bootstrap_to = bootstrap_metrics[to_model][metric]
        if metric in LOWER_IS_BETTER:
            point_improvement = point_from - point_to
            improvements = bootstrap_from - bootstrap_to
        else:
            point_improvement = point_to - point_from
            improvements = bootstrap_to - bootstrap_from
        ci_lower, ci_upper = np.quantile(
            improvements, [0.025, 0.975], method="linear"
        )
        probability = float(np.mean(improvements > 0.0))
        ties = int(np.count_nonzero(improvements == 0.0))
        metric_record = {
            "direction": direction,
            "from": point_from,
            "to": point_to,
            "improvement": float(point_improvement),
            "bootstrap_95_percentile_ci": [float(ci_lower), float(ci_upper)],
            "bootstrap_probability_improvement": probability,
            "bootstrap_positive_replicates": int(np.count_nonzero(improvements > 0.0)),
            "bootstrap_tied_replicates": ties,
        }
        metrics[metric] = metric_record
        csv_rows.append(
            {
                "scope": scope,
                "block_ids": "+".join(block_ids),
                "sample_count": sample_count,
                "comparison_id": comparison_id,
                "from_model": from_model,
                "to_model": to_model,
                "metric": metric,
                "direction": direction,
                "from_value": point_from,
                "to_value": point_to,
                "improvement": float(point_improvement),
                "ci_2_5_percent": float(ci_lower),
                "ci_97_5_percent": float(ci_upper),
                "probability_improvement": probability,
                "positive_replicates": int(np.count_nonzero(improvements > 0.0)),
                "tied_replicates": ties,
                "bootstrap_replicates": replicates,
            }
        )
    return (
        {
            "scope": scope,
            "block_ids": block_ids,
            "sample_count": sample_count,
            "comparison_id": comparison_id,
            "from_model": from_model,
            "to_model": to_model,
            "metrics": metrics,
        },
        csv_rows,
    )


def analyze(
    plan_path: Path,
    summary_dir: Path,
    routes_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan, resolved_plan = load_json(plan_path, "confirmation plan")
    models, blocks = validate_plan(plan)
    summary_dir = summary_dir.expanduser().resolve()
    routes_dir = routes_dir.expanduser().resolve()
    summary_paths, route_paths = expected_input_files(
        summary_dir, routes_dir, models, blocks
    )

    model_by_id = {str(model["model_id"]): model for model in models}
    block_by_id = {str(block["block_id"]): block for block in blocks}
    checkpoint_provenance: dict[str, Any] = {}
    for model_id in EXPECTED_MODELS:
        model = model_by_id[model_id]
        checkpoint_path = safe_input_path(
            Path(str(model["checkpoint"])), f"checkpoint for {model_id}"
        )
        actual_sha = sha256_file(checkpoint_path, f"checkpoint for {model_id}")
        if actual_sha != model["checkpoint_sha256"]:
            raise AnalysisError(
                f"checkpoint SHA mismatch for {model_id}: expected "
                f"{model['checkpoint_sha256']}, found {actual_sha}"
            )
        checkpoint_provenance[model_id] = {
            "path": str(checkpoint_path),
            "sha256": actual_sha,
            "sha256_validated": True,
            "role": model.get("role"),
        }

    point_by_scope: dict[str, dict[str, MetricSums]] = {}
    bootstrap_by_scope: dict[str, dict[str, MetricSums]] = {}
    input_provenance: dict[str, dict[str, Any]] = {}
    rng = np.random.Generator(np.random.PCG64(PREREGISTERED_SEED))

    for block_id in EXPECTED_BLOCKS:
        block = block_by_id[block_id]
        data_by_model: dict[str, RouteData] = {}
        block_inputs: dict[str, Any] = {}
        reference_identities: np.ndarray | None = None
        for model_id in EXPECTED_MODELS:
            key = (model_id, block_id)
            summary, resolved_summary = load_json(
                summary_paths[key], f"summary for {model_id}/{block_id}"
            )
            route_document, resolved_routes = load_json(
                route_paths[key], f"route export for {model_id}/{block_id}"
            )
            route_data = extract_route_data(
                route_document, f"route export {model_id}/{block_id}"
            )
            validate_document_pair(
                summary,
                route_document,
                resolved_summary,
                resolved_routes,
                model_by_id[model_id],
                block,
                plan,
                route_data,
            )
            if reference_identities is None:
                reference_identities = route_data.identities
            elif not np.array_equal(reference_identities, route_data.identities):
                mismatches = np.flatnonzero(
                    np.any(reference_identities != route_data.identities, axis=1)
                )
                row = int(mismatches[0]) if mismatches.size else -1
                expected_identity = (
                    reference_identities[row].tolist() if row >= 0 else "different shape"
                )
                actual_identity = (
                    route_data.identities[row].tolist() if row >= 0 else "different shape"
                )
                raise AnalysisError(
                    f"route alignment mismatch in block {block_id}, model {model_id}, "
                    f"row {row}: expected {expected_identity}, found {actual_identity}"
                )
            data_by_model[model_id] = route_data
            block_inputs[model_id] = {
                "summary_path": str(resolved_summary),
                "summary_sha256": sha256_file(
                    resolved_summary, f"summary for {model_id}/{block_id}"
                ),
                "routes_path": str(resolved_routes),
                "routes_sha256": sha256_file(
                    resolved_routes, f"route export for {model_id}/{block_id}"
                ),
                "sample_count": route_data.sample_count,
            }

        sample_counts = {data.sample_count for data in data_by_model.values()}
        if len(sample_counts) != 1:
            raise AnalysisError(
                f"route counts differ among models in block {block_id}: {sample_counts}"
            )
        point_by_scope[block_id] = {
            model_id: point_sums(data) for model_id, data in data_by_model.items()
        }
        bootstrap_by_scope[block_id] = bootstrap_block(
            data_by_model, rng, PREREGISTERED_REPLICATES
        )
        input_provenance[block_id] = block_inputs

    point_by_scope["pooled"] = {
        model_id: add_sums(
            point_by_scope[EXPECTED_BLOCKS[0]][model_id],
            point_by_scope[EXPECTED_BLOCKS[1]][model_id],
        )
        for model_id in EXPECTED_MODELS
    }
    bootstrap_by_scope["pooled"] = {
        model_id: add_sums(
            bootstrap_by_scope[EXPECTED_BLOCKS[0]][model_id],
            bootstrap_by_scope[EXPECTED_BLOCKS[1]][model_id],
        )
        for model_id in EXPECTED_MODELS
    }

    scopes: dict[str, Any] = {}
    comparisons: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for scope in (*EXPECTED_BLOCKS, "pooled"):
        scope_blocks = [scope] if scope != "pooled" else list(EXPECTED_BLOCKS)
        points = {
            model_id: scalar_metrics(point_by_scope[scope][model_id])
            for model_id in EXPECTED_MODELS
        }
        bootstraps = {
            model_id: metrics_from_sums(bootstrap_by_scope[scope][model_id])
            for model_id in EXPECTED_MODELS
        }
        sample_count = point_by_scope[scope][EXPECTED_MODELS[0]].sample_count
        scopes[scope] = {
            "block_ids": scope_blocks,
            "sample_count": sample_count,
            "model_point_estimates": points,
            "resampling": (
                "within_block_paired"
                if scope != "pooled"
                else "stratified_within_block_paired_preserving_block_sizes"
            ),
        }
        for from_model, to_model, comparison_id in COMPARISONS:
            record, rows = comparison_record(
                scope,
                scope_blocks,
                sample_count,
                from_model,
                to_model,
                comparison_id,
                points,
                bootstraps,
                PREREGISTERED_REPLICATES,
            )
            comparisons.append(record)
            csv_rows.extend(rows)

    report = {
        "schema_version": 1,
        "status": "confirmation_analysis_complete",
        "test_policy": "No pickle or test-split path was read by this analyzer.",
        "plan": {
            "path": str(resolved_plan),
            "sha256": sha256_file(resolved_plan, "confirmation plan"),
            "frozen_status_validated": True,
        },
        "validation": {
            "expected_model_block_pairs": len(EXPECTED_MODELS) * len(EXPECTED_BLOCKS),
            "all_expected_pairs_present": True,
            "route_index_source_target_aligned_within_each_block": True,
            "checkpoint_sha256_validated": True,
        },
        "bootstrap": {
            "seed": PREREGISTERED_SEED,
            "replicates": PREREGISTERED_REPLICATES,
            "rng": "NumPy PCG64",
            "interval": "95% percentile interval (2.5%, 97.5%; linear quantiles)",
            "probability_improvement": "fraction of paired replicates with strict positive improvement",
            "pairing": "same resampled route indices for all three models within a block",
            "pooled": "AM and PM resampled independently within block; original accepted block sizes preserved",
        },
        "checkpoints": checkpoint_provenance,
        "inputs": input_provenance,
        "scopes": scopes,
        "comparisons": comparisons,
    }
    return report, csv_rows


CSV_FIELDS = (
    "scope",
    "block_ids",
    "sample_count",
    "comparison_id",
    "from_model",
    "to_model",
    "metric",
    "direction",
    "from_value",
    "to_value",
    "improvement",
    "ci_2_5_percent",
    "ci_97_5_percent",
    "probability_improvement",
    "positive_replicates",
    "tied_replicates",
    "bootstrap_replicates",
)


def atomic_write_text(path: Path, text: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_outputs(
    report: Mapping[str, Any],
    csv_rows: Iterable[Mapping[str, Any]],
    output_json: Path,
    output_csv: Path,
) -> None:
    json_text = json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(csv_rows)
    atomic_write_text(output_json, json_text)
    atomic_write_text(output_csv, csv_buffer.getvalue())


def synthetic_overall(routes: list[dict[str, Any]]) -> dict[str, Any]:
    raw = sum(float(route["raw_regret"]) for route in routes)
    cost = sum(float(route["observed_path_cost"]) for route in routes)
    return {
        "sample_count": len(routes),
        "relative_regret": {"aggregate": raw / cost},
        "mean_edge_f1": sum(float(route["edge_f1"]) for route in routes)
        / len(routes),
        "exact_match_rate": sum(bool(route["exact_match"]) for route in routes)
        / len(routes),
    }


def build_synthetic_fixture(root: Path) -> tuple[Path, Path, Path]:
    summaries = root / "confirmation"
    routes_dir = root / "routes"
    checkpoints = root / "runs"
    summaries.mkdir()
    routes_dir.mkdir()
    checkpoints.mkdir()
    model_specs: list[dict[str, Any]] = []
    for model_id in EXPECTED_MODELS:
        checkpoint = checkpoints / f"{model_id}.json"
        checkpoint.write_text(
            json.dumps({"model_id": model_id, "run_test": False}), encoding="utf-8"
        )
        model_specs.append(
            {
                "model_id": model_id,
                "role": "synthetic",
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
        )

    blocks = [
        {"block_id": block_id, "validation_variant": f"confirm_{block_id}", "raw_records": 3}
        for block_id in EXPECTED_BLOCKS
    ]
    plan = {
        "schema_version": 1,
        "status": "frozen_before_confirmation_evaluation",
        "city": "synthetic",
        "train_variant_for_seen_edge_strata": "all",
        "test_policy": "No test path may be opened by this plan.",
        "models": model_specs,
        "confirmation_blocks": blocks,
        "evaluation": {
            "bootstrap_seed": PREREGISTERED_SEED,
            "bootstrap_replicates": PREREGISTERED_REPLICATES,
        },
    }
    plan_path = root / "confirmation_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    # Each successive model improves regret, F1, and exact match on every row.
    raw_by_model = {
        "baseline_q1": [30.0, 20.0, 10.0],
        "edge_t20_eta1e4": [20.0, 12.0, 5.0],
        "edge_t100_eta3e4": [10.0, 4.0, 0.0],
    }
    f1_by_model = {
        "baseline_q1": [0.2, 0.4, 0.6],
        "edge_t20_eta1e4": [0.4, 0.6, 0.8],
        "edge_t100_eta3e4": [0.7, 0.8, 1.0],
    }
    exact_by_model = {
        "baseline_q1": [False, False, False],
        "edge_t20_eta1e4": [False, True, False],
        "edge_t100_eta3e4": [True, True, True],
    }
    for block in blocks:
        block_id = block["block_id"]
        for model in model_specs:
            model_id = model["model_id"]
            route_rows = []
            for index in range(3):
                route_rows.append(
                    {
                        "route_index": index,
                        "source": 10 + index,
                        "target": 20 + index,
                        "raw_regret": raw_by_model[model_id][index],
                        "observed_path_cost": 100.0 + 10.0 * index,
                        "edge_f1": f1_by_model[model_id][index],
                        "exact_match": exact_by_model[model_id][index],
                    }
                )
            overall = synthetic_overall(route_rows)
            common = {
                "schema_version": 1,
                "checkpoint_path": model["checkpoint"],
                "checkpoint_metadata": {"model_id": model_id},
                "city": "synthetic",
                "train_variant": "all",
                "validation_variant": block["validation_variant"],
                "train_validation_report": {"accepted": 3, "available": 3},
                "validation_validation_report": {"accepted": 3, "available": 3},
            }
            route_path = routes_dir / f"{model_id}_{block_id}_routes.json"
            route_document = {
                **common,
                "evaluation": {"overall": overall, "routes": route_rows},
            }
            route_path.write_text(json.dumps(route_document), encoding="utf-8")
            summary = {
                **common,
                "evaluation": {"overall": overall},
                "routes_output": str(route_path),
            }
            (summaries / f"{model_id}_{block_id}.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
    return plan_path, summaries, routes_dir


class SyntheticConfirmationTests(unittest.TestCase):
    def test_end_to_end_is_deterministic_and_positive(self) -> None:
        with tempfile.TemporaryDirectory(prefix="confirmation-synthetic-") as directory:
            root = Path(directory)
            plan, summaries, routes = build_synthetic_fixture(root)
            first, rows = analyze(plan, summaries, routes)
            second, second_rows = analyze(plan, summaries, routes)
            self.assertEqual(first["comparisons"], second["comparisons"])
            self.assertEqual(rows, second_rows)
            self.assertEqual(len(rows), 3 * 3 * 3)
            self.assertTrue(first["validation"]["all_expected_pairs_present"])
            for record in first["comparisons"]:
                for metric in record["metrics"].values():
                    self.assertGreater(metric["improvement"], 0.0)
                    self.assertGreater(
                        metric["bootstrap_probability_improvement"], 0.5
                    )

            output_json = root / "result.json"
            output_csv = root / "result.csv"
            write_outputs(first, rows, output_json, output_csv)
            self.assertEqual(json.loads(output_json.read_text()), first)
            self.assertEqual(
                len(list(csv.DictReader(output_csv.read_text().splitlines()))), 27
            )

    def test_alignment_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="confirmation-synthetic-") as directory:
            root = Path(directory)
            plan, summaries, routes = build_synthetic_fixture(root)
            route_path = routes / "edge_t20_eta1e4_am_routes.json"
            document = json.loads(route_path.read_text(encoding="utf-8"))
            document["evaluation"]["routes"][1]["target"] = 999
            route_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(AnalysisError, "route alignment mismatch"):
                analyze(plan, summaries, routes)


def run_self_tests() -> bool:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        SyntheticConfirmationTests
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    if args.self_test:
        return 0 if run_self_tests() else 1
    try:
        report, csv_rows = analyze(args.plan, args.summary_dir, args.routes_dir)
        write_outputs(
            report, csv_rows, args.output_json, args.output_csv
        )
    except (AnalysisError, OSError) as error:
        print(f"confirmation analysis failed: {error}", file=sys.stderr)
        return 2
    pooled = report["scopes"]["pooled"]
    print(
        "confirmation analysis complete: "
        f"pooled_routes={pooled['sample_count']} "
        f"comparisons={len(report['comparisons'])} "
        f"json={args.output_json} csv={args.output_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
