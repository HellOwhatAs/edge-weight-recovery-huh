#!/usr/bin/env python3
"""Build tracked evidence for the full-data time-conditioning study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROUTE_QUALITY_METRICS = (
    "edge_precision",
    "edge_recall",
    "edge_f1",
    "exact_match",
    "edge_jaccard",
)
REGRET_METRICS = ("mean_regret", "relative_regret")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        default=Path("artifacts/full_data_time_conditioning"),
        type=Path,
    )
    parser.add_argument(
        "--output",
        default=Path("experiments/full_data_time_conditioning/summary.json"),
        type=Path,
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def config_provenance(runner: dict[str, Any]) -> dict[str, Any]:
    path = Path(runner["config_path"])
    tracked_bytes = path.read_bytes()
    tracked_config = json.loads(tracked_bytes)
    configuration_events = [
        event
        for event in map(
            json.loads,
            Path(runner["training_log"])
            .read_text(encoding="utf-8")
            .splitlines(),
        )
        if event.get("event") == "configuration"
    ]
    if len(configuration_events) != 1:
        raise ValueError(f"{runner['training_log']}: expected one configuration event")
    tracked_sha256 = hashlib.sha256(tracked_bytes).hexdigest()
    executed_sha256 = runner["config_sha256"]
    return {
        "executed_sha256": executed_sha256,
        "tracked_sha256": tracked_sha256,
        "byte_identical": tracked_sha256 == executed_sha256,
        "semantic_json_identical": configuration_events[0]["configuration"]
        == tracked_config,
    }


def stage_from_selection(selection: dict[str, Any]) -> dict[str, Any]:
    selected = selection["selected"]
    return {
        "selected_update": selected["update"],
        "selected_at_budget_boundary": selection["selected_at_budget_boundary"],
        "selection_rule": selection["selection_rule"],
        "metrics": selected["metrics"],
        "time_buckets": selected["time_buckets"],
        "validation_objective": selected["validation_objective"],
        "checkpoint": selected["checkpoint"],
        "evaluation_command": selected["evaluation_command"],
        "evaluation_output": selected["evaluation_output"],
        "quantization": selected.get("quantization"),
        "candidate_trace": [
            {
                "update": candidate["update"],
                "validation_objective": candidate["validation_objective"],
                "edge_f1": candidate["metrics"]["edge_f1"],
                "exact_match": candidate["metrics"]["exact_match"],
                "mean_regret": candidate["metrics"]["mean_regret"],
            }
            for candidate in selection["candidates"]
        ],
    }


def metric_delta(
    after: dict[str, Any],
    before: dict[str, Any],
    *,
    comparable_cost_units: bool,
) -> dict[str, float]:
    names = ROUTE_QUALITY_METRICS + (
        REGRET_METRICS if comparable_cost_units else ()
    )
    return {name: after[name] - before[name] for name in names}


def bucket_deltas(
    after_rows: list[dict[str, Any]],
    before_rows: list[dict[str, Any]],
    *,
    comparable_cost_units: bool,
) -> list[dict[str, Any]]:
    before = {row["id"]: row for row in before_rows}
    rows = []
    for after in after_rows:
        prior = before[after["id"]]
        if after["metrics"]["samples"] != prior["metrics"]["samples"]:
            raise ValueError(f"bucket sample mismatch for {after['id']}")
        rows.append(
            {
                "id": after["id"],
                "samples": after["metrics"]["samples"],
                "delta": metric_delta(
                    after["metrics"],
                    prior["metrics"],
                    comparable_cost_units=comparable_cost_units,
                ),
            }
        )
    return rows


def main() -> int:
    args = arguments()
    root = args.artifact_root
    audit = load(Path("experiments/full_data_time_conditioning/time_audit.json"))
    existing = load(root / "existing_10pct_evaluation_time.json")
    static_runner = load(
        root / "static_final/static_full_eta0002_u500/runner_result.json"
    )
    static_selection = load(
        root / "static_final/static_full_eta0002_u500/route_selection.json"
    )
    length_runner = load(
        root
        / "temporal_final/temporal_length_full_eta0002_u500/runner_result.json"
    )
    travel_runner = load(
        root
        / "temporal_final/temporal_travel_time_full_eta0002_u500/runner_result.json"
    )

    static = stage_from_selection(static_selection)
    length = stage_from_selection(length_runner["route_selection"])
    travel = stage_from_selection(travel_runner["route_selection"])
    existing_stage = {
        "selected_update": existing["checkpoint_completed_updates"],
        "metrics": existing["metrics"],
        "time_buckets": existing["time_bucket_evaluation"]["buckets"],
        "checkpoint": str(
            Path(
                "artifacts/optimizer_recovery/"
                "edge_transition_arcs_relative_10pct_u299/checkpoint-299.json"
            )
        ),
    }
    fixed_point_scale = audit["travel_time_baseline"]["diagnostics"][
        "fixed_point_scale"
    ]
    travel["metrics_physical_time"] = {
        "mean_regret_milliseconds": travel["metrics"]["mean_regret"]
        / fixed_point_scale,
        "fixed_point_scale": fixed_point_scale,
    }
    travel["time_buckets_physical_time"] = [
        {
            "id": row["id"],
            "samples": row["metrics"]["samples"],
            "mean_regret_milliseconds": row["metrics"]["mean_regret"]
            / fixed_point_scale,
            "relative_regret": row["metrics"]["relative_regret"],
        }
        for row in travel["time_buckets"]
    ]

    static_screen = []
    for run_id in ("static_full_eta0002_u60", "static_full_eta0004_u60"):
        runner = load(root / f"static_screen/{run_id}/runner_result.json")
        static_screen.append(
            {
                "run_id": run_id,
                "eta0": runner["eta0"],
                "status": runner["status"],
                "selected_update": runner["selected_checkpoint"]["update"],
                "selected_metrics": runner["selected_evaluation"]["metrics"],
                "training_wall_seconds": runner["training_wall_seconds"],
                "command": runner["command"],
            }
        )
    unstable_events = [
        event
        for event in map(
            json.loads,
            (
                root
                / "static_screen/static_full_eta0008_u60/training.jsonl"
            ).read_text(encoding="utf-8").splitlines(),
        )
        if event.get("event") == "state"
    ]
    unstable_validations = [
        {
            "update": event["completed_updates"],
            "validation_objective": event["validation"]["objective"],
        }
        for event in unstable_events
        if isinstance(event.get("validation"), dict)
    ]
    static_screen.append(
        {
            "run_id": "static_full_eta0008_u60",
            "eta0": 0.0008,
            "status": "terminated_after_registered_update_10_divergence",
            "last_completed_state": unstable_events[-1]["completed_updates"],
            "validation_trace": unstable_validations,
        }
    )

    residual_screen = []
    for run_id, multiplier in (
        ("temporal_length_residual_eta1_u40", 1.0),
        ("temporal_length_residual_eta2_u40", 2.0),
    ):
        runner = load(root / f"temporal_screen/{run_id}/runner_result.json")
        residual_screen.append(
            {
                "run_id": run_id,
                "residual_eta_multiplier": multiplier,
                "selected": runner["route_selection"]["selected"],
                "training_wall_seconds": runner["training_wall_seconds"],
                "training_command": runner["training_command"],
            }
        )
    partial_five = load(
        root
        / "temporal_final/temporal_length_full_eta0002_u500/evaluation-screen-25.json"
    )
    residual_screen.append(
        {
            "run_id": "early_temporal_length_multiplier5_diagnostic",
            "residual_eta_multiplier": 5.0,
            "status": "terminated_after_update_25_instability_check",
            "selected_update": 25,
            "metrics": partial_five["metrics"],
        }
    )
    travel_screen = load(
        root / "temporal_screen/temporal_travel_time_scaled_u40/runner_result.json"
    )

    summary = {
        "schema_version": 1,
        "study": "full_data_time_conditioned_route_learning",
        "protocol": {
            "repository_start": "f08d5364a4c4763fd985858ee1d398778b1f50fe",
            "graph_representation": "edge_transition_arcs",
            "optimizer": "relative_projected_subgradient",
            "validation_variant": "scale_fixed_seed20260715",
            "time_bucket_specification": audit["time_bucket_specification"],
            "data_audit": audit["splits"],
            "data_identity": audit["data_identity"],
            "travel_time_baseline": audit["travel_time_baseline"],
            "test_read": False,
        },
        "screening": {
            "static_learning_rate": static_screen,
            "temporal_residual_step": residual_screen,
            "travel_time_scaled_smoke": {
                "selected": travel_screen["route_selection"]["selected"],
                "training_command": travel_screen["training_command"],
            },
        },
        "stages": {
            "existing_10pct_static_line_graph": existing_stage,
            "full_static_line_graph": static,
            "full_time_conditioned_length_baseline": length,
            "full_time_conditioned_travel_time_baseline": travel,
        },
        "deltas": {
            "full_data_minus_existing_10pct": {
                "overall": metric_delta(
                    static["metrics"],
                    existing_stage["metrics"],
                    comparable_cost_units=True,
                ),
                "time_buckets": bucket_deltas(
                    static["time_buckets"],
                    existing_stage["time_buckets"],
                    comparable_cost_units=True,
                ),
            },
            "time_conditioning_minus_full_static": {
                "overall": metric_delta(
                    length["metrics"],
                    static["metrics"],
                    comparable_cost_units=True,
                ),
                "time_buckets": bucket_deltas(
                    length["time_buckets"],
                    static["time_buckets"],
                    comparable_cost_units=True,
                ),
            },
            "travel_time_baseline_minus_length_temporal": {
                "overall": metric_delta(
                    travel["metrics"],
                    length["metrics"],
                    comparable_cost_units=False,
                ),
                "time_buckets": bucket_deltas(
                    travel["time_buckets"],
                    length["time_buckets"],
                    comparable_cost_units=False,
                ),
                "mean_regret_not_subtracted": (
                    "direct costs use different length and scaled-time units"
                ),
            },
        },
        "runs": {
            "static": {
                "config_path": static_runner["config_path"],
                "config_sha256": static_runner["config_sha256"],
                "config_provenance": config_provenance(static_runner),
                "command": static_runner["command"],
                "training_wall_seconds": static_runner["training_wall_seconds"],
                "peak_rss_kib": static_runner["finished"]["peak_rss_kib"],
            },
            "temporal_length": {
                "config_path": length_runner["config_path"],
                "config_sha256": length_runner["config_sha256"],
                "config_provenance": config_provenance(length_runner),
                "command": length_runner["training_command"],
                "training_wall_seconds": length_runner["training_wall_seconds"],
                "peak_rss_kib": length_runner["finished"]["peak_rss_kib"],
            },
            "temporal_travel_time": {
                "config_path": travel_runner["config_path"],
                "config_sha256": travel_runner["config_sha256"],
                "config_provenance": config_provenance(travel_runner),
                "command": travel_runner["training_command"],
                "training_wall_seconds": travel_runner["training_wall_seconds"],
                "peak_rss_kib": travel_runner["finished"]["peak_rss_kib"],
            },
        },
        "test_read": False,
    }
    travel_vs_static = {
        "overall": metric_delta(
            travel["metrics"],
            static["metrics"],
            comparable_cost_units=False,
        ),
        "time_buckets": bucket_deltas(
            travel["time_buckets"],
            static["time_buckets"],
            comparable_cost_units=False,
        ),
        "mean_regret_not_subtracted": (
            "direct costs use different length and scaled-time units"
        ),
    }
    summary["deltas"]["travel_time_model_minus_full_static"] = travel_vs_static

    def positive_f1_buckets(comparison: dict[str, Any]) -> int:
        return sum(
            row["delta"]["edge_f1"] > 0.0 for row in comparison["time_buckets"]
        )

    full_delta = summary["deltas"]["full_data_minus_existing_10pct"]
    temporal_delta = summary["deltas"]["time_conditioning_minus_full_static"]
    travel_increment = summary["deltas"][
        "travel_time_baseline_minus_length_temporal"
    ]
    formal_stages = (static, length, travel)
    summary["conclusions"] = {
        "full_data_itself": {
            "clear_cross_bucket_improvement": positive_f1_buckets(full_delta)
            == len(full_delta["time_buckets"]),
            "positive_f1_buckets": positive_f1_buckets(full_delta),
            "total_buckets": len(full_delta["time_buckets"]),
            "edge_f1_delta": full_delta["overall"]["edge_f1"],
            "exact_match_delta": full_delta["overall"]["exact_match"],
        },
        "time_conditioning_over_full_static": {
            "small_overall_improvement": temporal_delta["overall"]["edge_f1"] > 0.0,
            "positive_f1_buckets": positive_f1_buckets(temporal_delta),
            "total_buckets": len(temporal_delta["time_buckets"]),
            "edge_f1_delta": temporal_delta["overall"]["edge_f1"],
            "exact_match_delta": temporal_delta["overall"]["exact_match"],
            "significant_and_stable_improvement_supported": False,
        },
        "travel_time_baseline_increment": {
            "small_overall_improvement": travel_increment["overall"]["edge_f1"]
            > 0.0,
            "positive_f1_buckets": positive_f1_buckets(travel_increment),
            "total_buckets": len(travel_increment["time_buckets"]),
            "edge_f1_delta": travel_increment["overall"]["edge_f1"],
            "exact_match_delta": travel_increment["overall"]["exact_match"],
            "significant_and_stable_improvement_supported": False,
        },
        "best_temporal_model_over_full_static": {
            "positive_f1_buckets": positive_f1_buckets(travel_vs_static),
            "total_buckets": len(travel_vs_static["time_buckets"]),
            "edge_f1_delta": travel_vs_static["overall"]["edge_f1"],
            "exact_match_delta": travel_vs_static["overall"]["exact_match"],
            "significant_and_stable_improvement_supported": False,
        },
        "convergence": {
            "all_selected_checkpoints_inside_budget": all(
                not stage["selected_at_budget_boundary"] for stage in formal_stages
            ),
            "decoded_route_quality_plateau_supported": all(
                not stage["selected_at_budget_boundary"]
                and stage["candidate_trace"][-1]["edge_f1"]
                < stage["metrics"]["edge_f1"]
                for stage in formal_stages
            ),
            "numerical_objective_convergence_confirmed": False,
            "objective_still_lower_at_final_update": all(
                stage["candidate_trace"][-1]["validation_objective"]
                < stage["validation_objective"]
                for stage in formal_stages
            ),
        },
        "scope": (
            "single fixed validation split; no independent test or formal "
            "statistical significance test"
        ),
        "test_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
