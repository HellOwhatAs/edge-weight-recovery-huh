#!/usr/bin/env python3
"""Build tracked evidence for independent departure-bucket static models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


METRICS = (
    "edge_precision",
    "edge_recall",
    "edge_f1",
    "exact_match",
    "edge_jaccard",
    "mean_regret",
    "relative_regret",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--static-selection",
        default=Path(
            "artifacts/full_data_time_conditioning/static_final/"
            "static_full_eta0002_u500/route_selection.json"
        ),
        type=Path,
    )
    parser.add_argument(
        "--bucketed-result",
        default=Path(
            "artifacts/independent_time_buckets/formal/bucketed_runner_result.json"
        ),
        type=Path,
    )
    parser.add_argument(
        "--static-config",
        default=Path(
            "experiments/independent_time_buckets/configs/"
            "static_full_reference_u500.json"
        ),
        type=Path,
    )
    parser.add_argument(
        "--audit",
        default=Path("experiments/independent_time_buckets/time_audit.json"),
        type=Path,
    )
    parser.add_argument(
        "--ten-percent-summary",
        default=Path("experiments/optimizer_recovery/summary.json"),
        type=Path,
    )
    parser.add_argument(
        "--output",
        default=Path("experiments/independent_time_buckets/summary.json"),
        type=Path,
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def repository_relative(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        return candidate.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return candidate.as_posix()


def delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, float]:
    return {name: after[name] - before[name] for name in METRICS}


def main() -> int:
    args = arguments()
    static_selection = load(args.static_selection)
    bucketed = load(args.bucketed_result)
    audit = load(args.audit)
    recovery = load(args.ten_percent_summary)
    ten_percent = recovery["runs"]["edge_transition_arcs"]
    if (
        static_selection.get("test_read") is not False
        or bucketed.get("test_read") is not False
        or audit.get("test_read") is not False
        or recovery.get("scope", {}).get("test_read") is not False
        or ten_percent.get("test_read") is not False
    ):
        raise ValueError("one input does not certify test_read=false")

    static = static_selection["selected"]
    static_config = load(args.static_config)
    static_checkpoint = load(Path(static["checkpoint"]))
    if static_checkpoint.get("configuration") != static_config:
        raise ValueError("active full-static config differs from the selected checkpoint")
    if (
        ten_percent.get("graph_representation") != "edge_transition_arcs"
        or ten_percent.get("optimizer_kind") != "relative_projected_subgradient"
        or ten_percent["selected_validation_metrics"]["samples"]
        != static["metrics"]["samples"]
        or recovery["data"]["validation_identity"]["sha256"]
        != audit["data_identity"]["validation"]["sha256"]
    ):
        raise ValueError("10% reference is not the matched static line-graph result")
    static_rows = {row["id"]: row for row in static["time_buckets"]}
    bucketed_rows = {row["id"]: row for row in bucketed["buckets"]}
    registered_ids = [
        bucket["id"] for bucket in audit["time_bucket_specification"]["buckets"]
    ]
    if set(static_rows) != set(registered_ids) or set(bucketed_rows) != set(
        registered_ids
    ):
        raise ValueError("static and independent results do not cover the same buckets")

    bucket_deltas = []
    for bucket_id in registered_ids:
        before = static_rows[bucket_id]
        after = bucketed_rows[bucket_id]
        if before["metrics"]["samples"] != after["metrics"]["samples"]:
            raise ValueError(f"{bucket_id}: validation sample count differs")
        bucket_deltas.append(
            {
                "id": bucket_id,
                "train_samples": after["train_samples"],
                "validation_samples": after["validation_samples"],
                "delta": delta(after["metrics"], before["metrics"]),
            }
        )
    overall_delta = delta(bucketed["metrics"], static["metrics"])
    full_data_delta = delta(
        static["metrics"], ten_percent["selected_validation_metrics"]
    )
    positive_f1_buckets = sum(row["delta"]["edge_f1"] > 0 for row in bucket_deltas)
    positive_exact_buckets = sum(
        row["delta"]["exact_match"] > 0 for row in bucket_deltas
    )
    portable_bucket_rows = []
    for row in bucketed["buckets"]:
        portable_row = dict(row)
        portable_row["checkpoint"] = repository_relative(row["checkpoint"])
        portable_row["route_selection"] = repository_relative(
            row["route_selection"]
        )
        portable_bucket_rows.append(portable_row)

    summary = {
        "schema_version": 1,
        "study": "independent_departure_bucket_static_models",
        "protocol": {
            "model_difference": "data partition only",
            "full_static_models": 1,
            "independent_bucket_static_models": len(bucketed_rows),
            "graph_representation": "edge_transition_arcs",
            "optimizer": "relative_projected_subgradient",
            "baseline": "length",
            "checkpoint_schema": "ordinary static direct weights",
            "full_static_reference_config": (
                "experiments/independent_time_buckets/configs/"
                "static_full_reference_u500.json"
            ),
            "full_static_reference_config_sha256": hashlib.sha256(
                args.static_config.read_bytes()
            ).hexdigest(),
            "full_static_checkpoint_configuration_exact_match": True,
            "time_bucket_specification": audit["time_bucket_specification"],
            "data_identity": audit["data_identity"],
            "data_audit": audit["splits"],
            "validation_selection_metric": "decoded Edge F1 within each run",
            "test_read": False,
        },
        "stages": {
            "existing_10pct_static": {
                "role": "historical matched-validation reference",
                "train_samples": ten_percent["graph_problem"]["train_mapped_paths"],
                "selected_update": ten_percent["checkpoint_selection"][
                    "selected_update"
                ],
                "selected_at_budget_boundary": ten_percent["checkpoint_selection"][
                    "at_registered_budget_boundary"
                ],
                "metrics": ten_percent["selected_validation_metrics"],
                "configuration": ten_percent["config_path"],
                "configuration_sha256": ten_percent["config_sha256"],
                "source_summary": repository_relative(args.ten_percent_summary),
            },
            "full_static": {
                "selected_update": static["update"],
                "selected_at_budget_boundary": static_selection[
                    "selected_at_budget_boundary"
                ],
                "metrics": static["metrics"],
                "metric_totals": static["metric_totals"],
                "time_buckets": static["time_buckets"],
                "quantization": static.get("quantization"),
                "checkpoint": repository_relative(static["checkpoint"]),
                "selection": repository_relative(args.static_selection),
            },
            "independent_bucket_static": {
                "metrics": bucketed["metrics"],
                "buckets": portable_bucket_rows,
                "selection_rule": bucketed["selection_rule"],
                "runner_result": repository_relative(args.bucketed_result),
            },
        },
        "deltas": {
            "full_static_minus_existing_10pct_static": full_data_delta,
            "independent_bucket_static_minus_full_static": {
                "overall": overall_delta,
                "time_buckets": bucket_deltas,
            }
        },
        "conclusions": {
            "full_data_edge_f1_improved": full_data_delta["edge_f1"] > 0,
            "full_data_exact_match_improved": full_data_delta["exact_match"] > 0,
            "overall_edge_f1_improved": overall_delta["edge_f1"] > 0,
            "overall_exact_match_improved": overall_delta["exact_match"] > 0,
            "positive_edge_f1_buckets": positive_f1_buckets,
            "positive_exact_match_buckets": positive_exact_buckets,
            "total_buckets": len(bucket_deltas),
            "cross_bucket_consistent_f1_improvement": positive_f1_buckets
            == len(bucket_deltas),
            "all_bucket_checkpoints_inside_budget": all(
                not row["selected_at_budget_boundary"]
                for row in bucketed["buckets"]
            ),
            "formal_significance_test_performed": False,
            "scope": "single fixed validation split; test never read",
        },
        "implementation": {
            "time_role": "load-time data filtering and inference checkpoint dispatch",
            "special_temporal_optimizer": False,
            "special_temporal_checkpoint": False,
            "travel_time_baseline": False,
            "historical_shared_residual_archive": (
                "experiments/archive/full_data_shared_temporal_residual"
            ),
        },
        "test_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
