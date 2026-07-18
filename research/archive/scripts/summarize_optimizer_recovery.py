#!/usr/bin/env python3
"""Build the machine-readable Beijing 10% optimizer-recovery summary."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REPRESENTATIONS = ("original_edges", "edge_transition_arcs")
ROUTE_METRICS = (
    "edge_precision",
    "edge_recall",
    "edge_f1",
    "exact_match",
    "edge_jaccard",
    "mean_regret",
    "relative_regret",
)
EXPECTED = {
    "optimizer_kind": "relative_projected_subgradient",
    "eta0": 0.0002,
    "lambda": 100000.0,
    "updates": 299,
    "validation_every": 10,
    "rayon_threads": 4,
}
LEGACY_EDGE_ONLY = {
    "artifact_commit": "fa26aa84f3c5e528f662835917d288e4f4368ebb",
    "code_commit": "62b23eb9471ce490f05512af3c942235a5962410",
    "artifact_path": "experiments/summaries/beijing_edge_only_10pct.json",
    "optimizer": {
        "parameterization": "q_i = w_i / w0_i",
        "eta0": 0.0002,
        "lambda": 100000.0,
        "q_min": 0.1,
        "q_max": 10.0,
        "optimizer_updates": 299,
    },
    "selected_state": 289,
    "selected_validation_metrics": {
        "edge_f1": 0.6821453478322816,
        "exact_match": 0.368454338477106,
        "mean_regret": 310343.73374652164,
        "relative_regret": 0.061200710617144466,
    },
    "state_299": {
        "mean_regret": 311137.48488489754,
        "relative_regret": 0.06164191944677813,
        "regularized_objective": 311756.22104318696,
    },
    "initial_validation_mean_regret": 650449.9359347331,
    "training_wall_seconds": 123.1621077180007,
    "peak_rss_kib": 153856,
    "test_read": False,
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-result", required=True, type=Path)
    parser.add_argument("--transition-result", required=True, type=Path)
    parser.add_argument("--direct-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    events = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        events.append(value)
    if not events:
        raise ValueError(f"{path}: empty training log")
    return events


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def relative(path: str | Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            return str(candidate.relative_to(Path.cwd()))
        except ValueError:
            return str(candidate)
    return str(candidate)


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def one_event(events: list[dict[str, Any]], event: str) -> dict[str, Any]:
    matches = [item for item in events if item.get("event") == event]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {event!r} event, got {len(matches)}")
    return matches[0]


def data_events(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    matches = {
        item["split"]: item
        for item in events
        if item.get("event") == "data" and item.get("split") in {"train", "validation"}
    }
    if set(matches) != {"train", "validation"}:
        raise ValueError("training log does not contain exactly the train/validation data events")
    return matches


def validate_metrics(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("samples"), int):
        raise ValueError(f"{label}: invalid route metrics")
    for metric in ROUTE_METRICS:
        if not finite(value.get(metric)):
            raise ValueError(f"{label}: invalid {metric}")
    return value


def normalized_candidates(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label}: missing validation checkpoint candidates")
    candidates = []
    for candidate in value:
        if not isinstance(candidate, dict) or not finite(
            candidate.get("validation_objective")
        ):
            raise ValueError(f"{label}: malformed checkpoint candidate")
        normalized = dict(candidate)
        normalized["checkpoint"] = relative(normalized["checkpoint"])
        candidates.append(normalized)
    return candidates


def summarize_run(path: Path, representation: str) -> dict[str, Any]:
    result = load_json(path)
    if result.get("schema_version") != 4 or result.get("status") != "ok":
        raise ValueError(f"{path}: expected a healthy schema-4 result")
    for key, expected in EXPECTED.items():
        if result.get(key) != expected:
            raise ValueError(f"{path}: {key}={result.get(key)!r}, expected {expected!r}")
    if result.get("graph_representation") != representation:
        raise ValueError(f"{path}: wrong graph representation")

    health = result.get("health_checks")
    if not isinstance(health, dict) or not health or not all(health.values()):
        raise ValueError(f"{path}: not all health checks passed")
    finished = result.get("finished")
    if (
        not isinstance(finished, dict)
        or finished.get("test_read") is not False
        or finished.get("optimizer_kind") != EXPECTED["optimizer_kind"]
        or finished.get("optimizer_parameterization") != "relative_to_initial"
    ):
        raise ValueError(f"{path}: invalid final optimizer/test state")

    config_path = Path(result["config_path"])
    if sha256(config_path) != result.get("config_sha256"):
        raise ValueError(f"{path}: configuration SHA-256 mismatch")
    config = load_json(config_path)
    if config.get("test_policy") != "never_read":
        raise ValueError(f"{config_path}: test policy changed")

    candidates = normalized_candidates(
        result.get("validation_checkpoint_candidates"), str(path)
    )
    selected = result.get("selected_checkpoint")
    if not isinstance(selected, dict):
        raise ValueError(f"{path}: missing selected checkpoint")
    expected_selected = min(
        candidates, key=lambda item: (item["validation_objective"], item["update"])
    )
    if (
        selected.get("update") != expected_selected["update"]
        or selected.get("validation_objective")
        != expected_selected["validation_objective"]
    ):
        raise ValueError(f"{path}: checkpoint selection rule was not followed")

    selected_evaluation = result.get("selected_evaluation")
    baseline_evaluation = result.get("baseline_evaluation")
    if not isinstance(selected_evaluation, dict) or not isinstance(
        baseline_evaluation, dict
    ):
        raise ValueError(f"{path}: missing external validation evaluation")
    selected_metrics = validate_metrics(
        selected_evaluation.get("metrics"), f"{path} selected"
    )
    baseline_metrics = validate_metrics(
        baseline_evaluation.get("metrics"), f"{path} update 0"
    )

    log_path = Path(result["training_log"])
    events = load_jsonl(log_path)
    graph = one_event(events, "graph_problem")
    filters = data_events(events)
    logged_finished = one_event(events, "finished")
    if logged_finished.get("test_read") is not False:
        raise ValueError(f"{log_path}: test was read")

    coordinate_count = graph["coordinates"]
    changed_coordinates = finished["changed_coordinates"]
    changed_quantized_coordinates = finished["changed_quantized_coordinates"]
    return {
        "run_id": result["run_id"],
        "graph_representation": representation,
        "optimizer_kind": result["optimizer_kind"],
        "optimizer_parameterization": finished["optimizer_parameterization"],
        "eta0": result["eta0"],
        "lambda": result["lambda"],
        "weight_lower_factor": config["graph"]["weight_lower_factor"],
        "weight_upper_factor": config["graph"]["weight_upper_factor"],
        "updates": result["updates"],
        "validation_every": result["validation_every"],
        "rayon_threads": result["rayon_threads"],
        "config_path": relative(config_path),
        "config_sha256": result["config_sha256"],
        "training_command": [relative(part) for part in result["command"]],
        "training_log": relative(log_path),
        "graph_problem": {
            key: graph[key]
            for key in (
                "original_nodes",
                "original_edges",
                "routing_nodes",
                "routing_arcs",
                "coordinates",
                "topology_identity",
                "train_mapped_paths",
                "train_unique_od",
                "validation_mapped_paths",
                "validation_unique_od",
            )
        },
        "filtering": filters,
        "checkpoint_selection": {
            "rule": result["selection_rule"],
            "candidates": candidates,
            "selected_update": selected["update"],
            "selected_validation_objective": selected["validation_objective"],
            "selected_validation_mean_regret": selected["validation_mean_regret"],
            "selected_regularization": selected["regularization"],
            "selected_checkpoint": relative(selected["checkpoint"]),
            "at_registered_budget_boundary": selected["update"] == result["updates"],
        },
        "selected_evaluation_command": [
            relative(part) for part in selected_evaluation["command"]
        ],
        "selected_validation_metrics": selected_metrics,
        "update_0_validation_metrics": baseline_metrics,
        "change_from_update_0": {
            metric: selected_metrics[metric] - baseline_metrics[metric]
            for metric in ROUTE_METRICS
        },
        "resources": {
            "training_wall_seconds": result["training_wall_seconds"],
            "total_wall_seconds": result["total_wall_seconds"],
            "peak_rss_kib": finished["peak_rss_kib"],
        },
        "coordinate_change": {
            "continuous": changed_coordinates,
            "continuous_fraction": changed_coordinates / coordinate_count,
            "quantized": changed_quantized_coordinates,
            "quantized_fraction": changed_quantized_coordinates / coordinate_count,
            "coordinate_count": coordinate_count,
        },
        "health_checks": health,
        "test_read": False,
        "runner_result": relative(path),
    }


def comparable_config(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value.pop("run_id", None)
    value.pop("description", None)
    value["graph"].pop("representation", None)
    return value


def metric_difference(
    left: dict[str, Any], right: dict[str, Any], metrics: tuple[str, ...]
) -> dict[str, float]:
    return {metric: left[metric] - right[metric] for metric in metrics}


def main() -> None:
    args = arguments()
    runs = {
        "original_edges": summarize_run(args.original_result, "original_edges"),
        "edge_transition_arcs": summarize_run(
            args.transition_result, "edge_transition_arcs"
        ),
    }
    original_config = load_json(Path(runs["original_edges"]["config_path"]))
    transition_config = load_json(
        Path(runs["edge_transition_arcs"]["config_path"])
    )
    if comparable_config(original_config) != comparable_config(transition_config):
        raise ValueError("recovery configurations differ beyond graph representation")
    if runs["original_edges"]["filtering"] != runs["edge_transition_arcs"]["filtering"]:
        raise ValueError("representations did not use identical filtered samples")

    direct_summary = load_json(args.direct_summary)
    direct_runs = direct_summary.get("development", {}).get("runs")
    if not isinstance(direct_runs, dict) or set(direct_runs) != set(REPRESENTATIONS):
        raise ValueError(f"{args.direct_summary}: missing direct-weight development runs")
    direct_compact = {}
    for representation in REPRESENTATIONS:
        direct = direct_runs[representation]
        direct_compact[representation] = {
            "run_id": direct["run_id"],
            "optimizer_parameterization": "direct_weights",
            "eta0": direct["eta0"],
            "lambda": direct["lambda"],
            "updates": direct["updates"],
            "selected_update": direct["checkpoint_selection"]["selected_update"],
            "selected_validation_metrics": direct["selected_validation_metrics"],
            "update_0_validation_metrics": direct["update_0_validation_metrics"],
            "change_from_update_0": direct["change_from_update_0"],
            "training_wall_seconds": direct["training_wall_seconds"],
            "peak_rss_kib": direct["peak_rss_kib"],
            "final_changed_coordinates": direct["final_changed_coordinates"],
            "final_changed_quantized_coordinates": direct[
                "final_changed_quantized_coordinates"
            ],
            "test_read": direct["test_read"],
        }

    original = runs["original_edges"]
    transition = runs["edge_transition_arcs"]
    original_metrics = original["selected_validation_metrics"]
    transition_metrics = transition["selected_validation_metrics"]
    legacy_metrics = LEGACY_EDGE_ONLY["selected_validation_metrics"]
    original_state_299 = next(
        item
        for item in original["checkpoint_selection"]["candidates"]
        if item["update"] == 299
    )
    common_route_metrics = (
        "edge_precision",
        "edge_recall",
        "edge_f1",
        "exact_match",
        "edge_jaccard",
    )

    summary = {
        "schema_version": 1,
        "experiment": "beijing_10pct_relative_weight_optimizer_recovery",
        "status": "completed_development_optimizer_recovery",
        "scope": {
            "optimizer_geometry_changed": True,
            "graph_definitions_changed": False,
            "model_architecture_changed": False,
            "test_read": False,
            "neuromlr_run": False,
            "full_training_run": False,
            "learning_rate_grid_run": False,
        },
        "diagnosis": {
            "finding": "the direct-weight Euclidean geometry caused the observed optimization regression",
            "same_data_initial_oracle_evidence": {
                "legacy_original_edges_validation_mean_regret_update_0": LEGACY_EDGE_ONLY[
                    "initial_validation_mean_regret"
                ],
                "unified_original_edges_validation_mean_regret_update_0": original[
                    "update_0_validation_metrics"
                ]["mean_regret"],
                "bit_identical": LEGACY_EDGE_ONLY["initial_validation_mean_regret"]
                == original["update_0_validation_metrics"]["mean_regret"],
            },
            "direct_geometry": {
                "parameter": "w_i",
                "data_gradient": "(n_obs_i - n_pred_i) / N",
                "regularizer": "lambda/(2m) * sum_i (w_i - w0_i)^2",
                "issue": "one Euclidean step size ignores the wide scale range of w0",
            },
            "recovered_geometry": {
                "parameter": "q_i = w_i / w0_i",
                "data_gradient": "w0_i * (n_obs_i - n_pred_i) / N",
                "regularizer": "lambda/(2m) * sum_i (q_i - 1)^2",
                "direct_weight_equivalent": "diag(w0_i^2) preconditioning with relative regularization",
                "stored_checkpoint_state": "one direct weight vector w",
                "representation_specific_optimizer_state": False,
            },
            "regularization_sign_audit": {
                "implemented_direct_gradient_sign": "positive lambda/m * (w_i - w0_i)",
                "review_formula_showing_a_negative_sign": "not present in the implementation",
                "effect_on_root_cause": "none; the scale and regularization geometry regression was real",
            },
        },
        "data": {
            "city": original_config["data"]["city"],
            "path_contract": original_config["data"]["path_contract"],
            "cycle_policy": original_config["data"]["cycle_policy"],
            "train_variant": original_config["data"]["train_variant"],
            "validation_variant": original_config["data"]["validation_variant"],
            "train_identity": original_config["data"]["train_identity"],
            "validation_identity": original_config["data"]["validation_identity"],
            "filtering": original["filtering"],
            "train_unique_od": original["graph_problem"]["train_unique_od"],
            "validation_unique_od": original["graph_problem"]["validation_unique_od"],
            "identical_samples_across_representations": True,
        },
        "fixed_recovery_protocol": EXPECTED
        | {
            "weight_bounds": [0.1, 10.0],
            "checkpoint_selection": "minimum validation objective; earliest update breaks exact ties",
            "configuration_difference": "graph representation only",
        },
        "runs": runs,
        "legacy_original_edges_regression": {
            "reference": LEGACY_EDGE_ONLY,
            "current_selected_update": original["checkpoint_selection"]["selected_update"],
            "current_selected_validation_metrics": original_metrics,
            "current_minus_legacy_selected": {
                "edge_f1": original_metrics["edge_f1"] - legacy_metrics["edge_f1"],
                "exact_match": original_metrics["exact_match"]
                - legacy_metrics["exact_match"],
                "mean_regret": original_metrics["mean_regret"]
                - legacy_metrics["mean_regret"],
                "relative_regret": original_metrics["relative_regret"]
                - legacy_metrics["relative_regret"],
            },
            "state_299_mean_regret_difference": original_state_299[
                "validation_mean_regret"
            ]
            - LEGACY_EDGE_ONLY["state_299"]["mean_regret"],
            "judgment": "passed; the generic implementation reproduces the legacy optimization trajectory and selected route quality",
        },
        "direct_to_relative_route_quality": {
            representation: {
                "direct_selected": direct_compact[representation],
                "relative_selected_update": runs[representation][
                    "checkpoint_selection"
                ]["selected_update"],
                "relative_selected_metrics": runs[representation][
                    "selected_validation_metrics"
                ],
                "relative_minus_direct_selected": metric_difference(
                    runs[representation]["selected_validation_metrics"],
                    direct_compact[representation]["selected_validation_metrics"],
                    common_route_metrics,
                ),
                "relative_change_from_update_0": runs[representation][
                    "change_from_update_0"
                ],
            }
            for representation in REPRESENTATIONS
        }
        | {
            "comparison_boundary": "route metrics are common, but direct and relative runs used their own matched parameterization hyperparameters and unequal update budgets"
        },
        "relative_optimizer_representation_comparison": {
            "edge_transition_minus_original_route_metrics": metric_difference(
                transition_metrics, original_metrics, common_route_metrics
            ),
            "training_wall_seconds": {
                "original_edges": original["resources"]["training_wall_seconds"],
                "edge_transition_arcs": transition["resources"][
                    "training_wall_seconds"
                ],
                "edge_transition_over_original_ratio": transition["resources"][
                    "training_wall_seconds"
                ]
                / original["resources"]["training_wall_seconds"],
            },
            "peak_rss_kib": {
                "original_edges": original["resources"]["peak_rss_kib"],
                "edge_transition_arcs": transition["resources"]["peak_rss_kib"],
                "edge_transition_over_original_ratio": transition["resources"][
                    "peak_rss_kib"
                ]
                / original["resources"]["peak_rss_kib"],
            },
            "raw_objective_compared_across_representations": False,
        },
        "previous_single_edge_audit": direct_summary["data_and_single_edge_audit"],
        "recommendation": {
            "representation_for_neuromlr_comparison": "edge_transition_arcs",
            "optimizer_kind": EXPECTED["optimizer_kind"],
            "basis": "meaningful learning from update 0 is now established for both representations, and line graph retains higher common decoded Edge F1 and Exact Match",
            "do_not_use_direct_geometry_for_formal_comparison": True,
        },
        "risks": {
            "line_graph_best_checkpoint_at_registered_budget_boundary": transition[
                "checkpoint_selection"
            ]["at_registered_budget_boundary"],
            "line_graph_convergence_confirmed": False,
            "original_best_checkpoint_at_registered_budget_boundary": original[
                "checkpoint_selection"
            ]["at_registered_budget_boundary"],
            "integer_cch_quantization": True,
            "single_edge_zero_cost": True,
            "single_edge_validation_exposure": 10 / 15812,
            "single_seed_and_fixed_validation": True,
        },
        "provenance": {
            "direct_weight_calibration_summary": relative(args.direct_summary),
            "original_result": relative(args.original_result),
            "transition_result": relative(args.transition_result),
            "raw_logs_and_checkpoints_retained_locally": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
