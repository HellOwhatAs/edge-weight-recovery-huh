#!/usr/bin/env python3
"""Build the machine-readable Beijing 10% representation calibration summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


REPRESENTATIONS = ("original_edges", "edge_transition_arcs")
INITIAL_ETAS = {300.0, 1000.0, 3000.0}
ROUTE_METRICS = (
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
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--screening-root", required=True, type=Path)
    parser.add_argument("--development-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def relative(path: str | Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            return str(candidate.relative_to(Path.cwd()))
        except ValueError:
            return str(candidate)
    return str(candidate)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_results(root: Path) -> list[dict[str, Any]]:
    paths = sorted(root.glob("*/runner_result.json"))
    if not paths:
        raise ValueError(f"{root}: no runner results")
    results = []
    for path in paths:
        result = load_json(path)
        if result.get("schema_version") != 4 or result.get("status") != "ok":
            raise ValueError(f"{path}: expected a healthy schema-4 runner result")
        finished = result.get("finished")
        health = result.get("health_checks")
        if (
            not isinstance(finished, dict)
            or finished.get("test_read") is not False
            or not isinstance(health, dict)
            or health.get("test_not_read") is not True
        ):
            raise ValueError(f"{path}: test-read invariant was not preserved")
        result["_result_path"] = str(path)
        results.append(result)
    return results


def validate_metrics(metrics: Any, label: str) -> dict[str, Any]:
    if not isinstance(metrics, dict) or not isinstance(metrics.get("samples"), int):
        raise ValueError(f"{label}: invalid metrics object")
    for name in ROUTE_METRICS:
        if not finite(metrics.get(name)):
            raise ValueError(f"{label}: invalid metric {name}")
    return metrics


def summarize_run(result: dict[str, Any]) -> dict[str, Any]:
    selected = result.get("selected_checkpoint")
    selected_evaluation = result.get("selected_evaluation")
    baseline_evaluation = result.get("baseline_evaluation")
    if not all(
        isinstance(value, dict)
        for value in (selected, selected_evaluation, baseline_evaluation)
    ):
        raise ValueError(f"{result.get('run_id')}: incomplete checkpoint selection")
    selected_metrics = validate_metrics(
        selected_evaluation["metrics"], f"{result['run_id']} selected"
    )
    baseline_metrics = validate_metrics(
        baseline_evaluation["metrics"], f"{result['run_id']} baseline"
    )
    candidates = result.get("validation_checkpoint_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"{result['run_id']}: no checkpoint selection basis")
    candidate_basis = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or not finite(
            candidate.get("validation_objective")
        ):
            raise ValueError(f"{result['run_id']}: invalid checkpoint candidate")
        candidate_basis.append(
            {
                **candidate,
                "checkpoint": relative(candidate["checkpoint"]),
            }
        )

    config_path = Path(result["config_path"])
    if sha256(config_path) != result["config_sha256"]:
        raise ValueError(f"{config_path}: config hash changed after the run")
    output = {
        "run_id": result["run_id"],
        "graph_representation": result["graph_representation"],
        "eta0": result["eta0"],
        "lambda": result["lambda"],
        "updates": result["updates"],
        "validation_every": result["validation_every"],
        "rayon_threads": result["rayon_threads"],
        "config_path": relative(config_path),
        "config_sha256": result["config_sha256"],
        "training_command": [relative(part) for part in result["command"]],
        "training_log": relative(result["training_log"]),
        "training_wall_seconds": result["training_wall_seconds"],
        "total_wall_seconds": result["total_wall_seconds"],
        "peak_rss_kib": result["finished"]["peak_rss_kib"],
        "final_changed_coordinates": result["finished"]["changed_coordinates"],
        "final_changed_quantized_coordinates": result["finished"][
            "changed_quantized_coordinates"
        ],
        "checkpoint_selection": {
            "rule": result["selection_rule"],
            "candidates": candidate_basis,
            "selected_update": selected["update"],
            "selected_validation_objective": selected["validation_objective"],
            "selected_changed_coordinates": selected["changed_coordinates"],
            "selected_checkpoint": relative(selected["checkpoint"]),
        },
        "selected_evaluation_command": [
            relative(part) for part in selected_evaluation["command"]
        ],
        "selected_validation_metrics": selected_metrics,
        "update_0_validation_metrics": baseline_metrics,
        "change_from_update_0": {
            name: selected_metrics[name] - baseline_metrics[name]
            for name in ROUTE_METRICS
        },
        "health_checks": result["health_checks"],
        "test_read": False,
        "runner_result": relative(result["_result_path"]),
    }
    return output


def screen_key(run: dict[str, Any]) -> tuple[float, float, float]:
    metrics = run["selected_validation_metrics"]
    return (
        metrics["edge_f1"],
        metrics["exact_match"],
        -run["checkpoint_selection"]["selected_validation_objective"],
    )


def select_screening_runs(
    runs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    selections = {}
    for representation in REPRESENTATIONS:
        representation_runs = [
            run for run in runs if run["graph_representation"] == representation
        ]
        initial = [run for run in representation_runs if run["eta0"] in INITIAL_ETAS]
        if {run["eta0"] for run in initial} != INITIAL_ETAS:
            raise ValueError(f"{representation}: incomplete initial eta grid")
        initial_best = max(initial, key=screen_key)
        expected_supplement = {
            300.0: 100.0,
            3000.0: 10000.0,
        }.get(initial_best["eta0"])
        supplements = [
            run for run in representation_runs if run["eta0"] not in INITIAL_ETAS
        ]
        actual_supplements = {run["eta0"] for run in supplements}
        expected_supplements = (
            {expected_supplement} if expected_supplement is not None else set()
        )
        if actual_supplements != expected_supplements:
            raise ValueError(
                f"{representation}: supplements {actual_supplements} do not match "
                f"boundary rule {expected_supplements}"
            )
        best = max(representation_runs, key=screen_key)
        selections[representation] = {
            "initial_best_eta0": initial_best["eta0"],
            "boundary_supplement_eta0": expected_supplement,
            "selected_eta0": best["eta0"],
            "selected_run_id": best["run_id"],
            "selection_metrics": {
                "edge_f1": best["selected_validation_metrics"]["edge_f1"],
                "exact_match": best["selected_validation_metrics"]["exact_match"],
                "validation_objective": best["checkpoint_selection"][
                    "selected_validation_objective"
                ],
            },
        }
    return selections


def development_comparison(
    development: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    original = development["original_edges"]
    transition = development["edge_transition_arcs"]
    original_metrics = original["selected_validation_metrics"]
    transition_metrics = transition["selected_validation_metrics"]
    return {
        "route_metric_difference_edge_transition_minus_original": {
            name: transition_metrics[name] - original_metrics[name]
            for name in (
                "edge_precision",
                "edge_recall",
                "edge_f1",
                "exact_match",
                "edge_jaccard",
            )
        },
        "training_wall_seconds": {
            "original_edges": original["training_wall_seconds"],
            "edge_transition_arcs": transition["training_wall_seconds"],
            "edge_transition_over_original_ratio": transition[
                "training_wall_seconds"
            ]
            / original["training_wall_seconds"],
        },
        "peak_rss_kib": {
            "original_edges": original["peak_rss_kib"],
            "edge_transition_arcs": transition["peak_rss_kib"],
            "edge_transition_over_original_ratio": transition["peak_rss_kib"]
            / original["peak_rss_kib"],
        },
        "raw_objective_compared_across_representations": False,
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = arguments()
    audit = load_json(args.audit)
    if audit.get("test_read") is not False:
        raise ValueError("single-edge audit did not preserve the test-read invariant")
    screening = [summarize_run(run) for run in load_results(args.screening_root)]
    development_runs = [
        summarize_run(run) for run in load_results(args.development_root)
    ]
    selections = select_screening_runs(screening)
    development = {
        run["graph_representation"]: run for run in development_runs
    }
    if set(development) != set(REPRESENTATIONS):
        raise ValueError("development results must contain exactly both representations")
    for representation in REPRESENTATIONS:
        run = development[representation]
        if run["eta0"] != selections[representation]["selected_eta0"]:
            raise ValueError(f"{representation}: development eta does not match screening")
        if run["updates"] != 200 or run["validation_every"] != 10:
            raise ValueError(f"{representation}: invalid development cadence")

    comparison = development_comparison(development)
    transition_wins_common_metrics = all(
        comparison["route_metric_difference_edge_transition_minus_original"][name] > 0
        for name in ("edge_f1", "exact_match")
    )
    summary = {
        "schema_version": 1,
        "experiment": "beijing_10pct_graph_representation_calibration",
        "source_branch": "line-graph-10pct-calibration",
        "source_commit": "a587e9ed2b239a448f4eeaaebf274b313a596359",
        "scope": {
            "model_architecture_changed": False,
            "objective_changed": False,
            "optimizer_changed": False,
            "graph_definition_changed": False,
            "lambda_tuned": False,
            "test_read": False,
            "neuromlr_run": False,
            "full_training_run": False,
        },
        "fixed_configuration": {
            "train_variant": "scale_10pct_seed42",
            "validation_variant": "scale_fixed_seed20260715",
            "lambda": 0.001,
            "weight_lower_factor": 0.1,
            "weight_upper_factor": 10.0,
            "validation_every": 10,
            "rayon_threads": 4,
            "checkpoint_selection": "minimum validation objective among saved checkpoints",
            "eta_selection": "maximum Edge F1, then Exact Match, then lower validation objective",
        },
        "data_and_single_edge_audit": audit,
        "learning_rate_screening": {
            "initial_eta0_grid": sorted(INITIAL_ETAS),
            "runs": sorted(
                screening,
                key=lambda run: (run["graph_representation"], run["eta0"]),
            ),
            "selection": selections,
        },
        "development": {
            "runs": development,
            "comparison": comparison,
            "unconfirmed_convergence": {
                representation: development[representation][
                    "checkpoint_selection"
                ]["selected_update"]
                == development[representation]["updates"]
                for representation in REPRESENTATIONS
            },
        },
        "recommendation": {
            "representation_for_neuromlr_comparison": (
                "edge_transition_arcs"
                if transition_wins_common_metrics
                else "original_edges"
            ),
            "basis": (
                "decoded common-route Edge F1 and Exact Match; raw objectives were not "
                "compared across representations"
            ),
        },
        "risks": {
            "single_edge_zero_cost": True,
            "single_edge_exposure_is_small": True,
            "integer_cch_quantization": True,
            "best_checkpoint_at_update_200": any(
                development[representation]["checkpoint_selection"][
                    "selected_update"
                ]
                == 200
                for representation in REPRESENTATIONS
            ),
            "single_seed_and_fixed_validation": True,
        },
        "provenance": {
            "audit": relative(args.audit),
            "screening_root": relative(args.screening_root),
            "development_root": relative(args.development_root),
            "all_raw_logs_and_checkpoints_retained_locally": True,
        },
    }
    atomic_json(args.output, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
