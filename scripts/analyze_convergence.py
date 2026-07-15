#!/usr/bin/env python3
"""Analyze bounded convergence trajectories without loading any trip dataset.

The analyzer recursively discovers ``model_training.log`` files below a run
root.  It emits one row per validation event, one row per run, and a JSON
report containing horizon rankings, the predeclared primary candidate, and an
exact deterministic-prefix check for the 20-epoch control.

Only text logs, adjacent runner-result metadata, and the declared protocol JSON
are read.  In particular this script never opens a pickle or any
train/validation/test data file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile
import unittest
from typing import Any, Iterable


HORIZONS = (20, 50, 100)
CONTROL_RUN_ID = "control_full_eta1e4_t20"
LONG_RUN_ID = "conv_full_eta1e4"

# These are the quantities covered by the protocol's "all reported losses are
# finite" criterion.  NA is expected at epochs where validation is skipped.
LOSS_KEYS = {
    "train_regret",
    "train_relative_regret",
    "regularization",
    "train_objective",
    "validation_regret",
    "validation_relative_regret",
    "selection_loss",
    "mean_regret",
    "relative_regret",
    "best_train_regret",
    "best_regularization",
}

# Exact strings are compared here: this is deliberately stronger than a
# tolerance test.  Timing fields are excluded because they are not part of the
# mathematical trajectory.
PREFIX_STATE_FIELDS = (
    "train_regret",
    "train_relative_regret",
    "regularization",
    "train_objective",
    "count_residual_l1",
    "train_queries",
    "current_q_min",
    "current_q_max",
    "current_q_at_min",
    "current_q_at_max",
    "current_max_quantization_error",
    "validation_regret",
    "validation_relative_regret",
    "validation_exact_match",
    "validation_edge_f1",
    "validation_edge_jaccard",
    "selection_loss",
    "stale_evaluations",
)
PREFIX_UPDATE_FIELDS = (
    "changed_edges",
    "changed_pct",
    "update_status",
    "next_q_min",
    "next_q_max",
    "next_q_at_min",
    "next_q_at_max",
    "next_max_quantization_error",
    "eta",
    "latent_max_delta",
    "projected_edges",
)


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--trajectory-csv", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run isolated synthetic-log unit tests and exit",
    )
    args = parser.parse_args(argv)
    if not args.self_test:
        missing = [
            flag
            for flag, value in (
                ("--run-root", args.run_root),
                ("--protocol", args.protocol),
                ("--trajectory-csv", args.trajectory_csv),
                ("--summary-csv", args.summary_csv),
                ("--summary-json", args.summary_json),
            )
            if value is None
        ]
        if missing:
            parser.error(f"the following arguments are required: {', '.join(missing)}")
    return args


def tokens(line: str) -> tuple[str, dict[str, str], set[str]]:
    """Return record kind, key/value tokens, and bare marker tokens."""
    pieces = shlex.split(line.strip())
    if not pieces:
        return "", {}, set()
    values: dict[str, str] = {}
    markers: set[str] = set()
    for piece in pieces[1:]:
        if "=" in piece:
            key, value = piece.split("=", 1)
            values[key] = value
        else:
            markers.add(piece)
    return pieces[0], values, markers


def finite_float(value: Any) -> float | None:
    if value is None or value == "NA":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def integer(value: Any) -> int | None:
    if value is None or value == "NA":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def json_number(value: str | None) -> int | float | str | None:
    if value is None or value == "NA":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        result = float(value)
    except ValueError:
        return value
    return result if math.isfinite(result) else value


def nonfinite_numeric(value: str) -> bool:
    try:
        return not math.isfinite(float(value))
    except ValueError:
        return True


def run_identity(log_path: Path, run_root: Path) -> tuple[str, str]:
    run_dir = log_path.parent
    run_id = run_dir.name
    try:
        relative = run_dir.relative_to(run_root)
    except ValueError:
        return run_id, ""
    phase = relative.parts[-2] if len(relative.parts) >= 2 else ""
    return run_id, phase


def parse_log(log_path: Path, run_root: Path) -> dict[str, Any]:
    """Parse a single log into raw records and scientific diagnostics."""
    run_id, phase = run_identity(log_path, run_root)
    config: dict[str, str] = {}
    load: dict[str, str] = {}
    cch: dict[str, str] = {}
    epochs: list[dict[str, Any]] = []
    final_evaluation: dict[str, str] = {}
    finished: dict[str, str] = {}
    early_stop: dict[str, str] = {}
    loss_nonfinite: list[str] = []
    loss_unparseable: list[str] = []
    runner_result: dict[str, Any] = {}

    runner_result_path = log_path.parent / "runner_result.json"
    if runner_result_path.is_file():
        try:
            loaded_runner_result = json.loads(
                runner_result_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            runner_result = {"_read_error": repr(error)}
        else:
            if isinstance(loaded_runner_result, dict):
                runner_result = loaded_runner_result
            else:
                runner_result = {"_read_error": "top-level value is not an object"}

    for line_number, line in enumerate(
        log_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        kind, values, markers = tokens(line)
        for key in LOSS_KEYS & values.keys():
            value = values[key]
            if value == "NA":
                continue
            try:
                parsed = float(value)
            except ValueError:
                loss_unparseable.append(f"line {line_number}:{kind}.{key}={value}")
            else:
                if not math.isfinite(parsed):
                    loss_nonfinite.append(f"line {line_number}:{kind}.{key}={value}")
        if kind == "CONFIG":
            config = values
        elif kind == "LOAD":
            load = values
        elif kind == "CCH":
            cch = values
        elif kind == "EPOCH":
            epoch = dict(values)
            epoch["_markers"] = markers
            epoch["_line_number"] = line_number
            epochs.append(epoch)
        elif kind == "EVAL" and values.get("split") == "validation_best":
            final_evaluation = values
        elif kind == "EARLY_STOP":
            early_stop = values
        elif kind == "FINISHED":
            finished = values

    edges = integer(load.get("edges"))
    validation_epochs = [
        epoch
        for epoch in epochs
        if epoch.get("validation_relative_regret") not in (None, "NA")
    ]
    trajectory: list[dict[str, Any]] = []
    prefix_best: dict[str, Any] | None = None
    for epoch in validation_epochs:
        epoch_number = integer(epoch.get("epoch"))
        regret = finite_float(epoch.get("validation_relative_regret"))
        f1 = finite_float(epoch.get("validation_edge_f1"))
        exact = finite_float(epoch.get("validation_exact_match"))
        q_at_min = integer(epoch.get("current_q_at_min"))
        q_at_max = integer(epoch.get("current_q_at_max"))
        boundary_count = (
            q_at_min + q_at_max
            if q_at_min is not None and q_at_max is not None
            else None
        )
        boundary_pct = (
            100.0 * boundary_count / edges
            if boundary_count is not None and edges not in (None, 0)
            else None
        )
        is_new_best = regret is not None and (
            prefix_best is None or regret < prefix_best["relative_regret"]
        )
        if is_new_best:
            prefix_best = {
                "epoch": epoch_number,
                "relative_regret": regret,
                "edge_f1": f1,
                "exact_match": exact,
            }
        event_nonfinite = []
        for key in (
            "validation_regret",
            "validation_relative_regret",
            "selection_loss",
            "validation_exact_match",
            "validation_edge_f1",
            "validation_edge_jaccard",
            "current_q_min",
            "current_q_max",
        ):
            value = epoch.get(key)
            if value not in (None, "NA") and nonfinite_numeric(str(value)):
                event_nonfinite.append(key)
        trajectory.append(
            {
                "run_id": run_id,
                "phase": phase,
                "train_variant": config.get("train"),
                "eta0": json_number(config.get("eta0")),
                "lambda": json_number(config.get("lambda")),
                "epoch": epoch_number,
                "validation_relative_regret": regret,
                "validation_edge_f1": f1,
                "validation_exact_match": exact,
                "validation_edge_jaccard": finite_float(
                    epoch.get("validation_edge_jaccard")
                ),
                "selection_loss": finite_float(epoch.get("selection_loss")),
                "logged_best_marker": "BEST" in epoch["_markers"],
                "is_new_prefix_best": is_new_best,
                "prefix_best_epoch": prefix_best["epoch"] if prefix_best else None,
                "prefix_best_relative_regret": (
                    prefix_best["relative_regret"] if prefix_best else None
                ),
                "prefix_best_edge_f1": prefix_best["edge_f1"] if prefix_best else None,
                "prefix_best_exact_match": (
                    prefix_best["exact_match"] if prefix_best else None
                ),
                "stale_evaluations": integer(epoch.get("stale_evaluations")),
                "update_status": epoch.get("update_status"),
                "projected_edges": integer(epoch.get("projected_edges")),
                "current_q_min": finite_float(epoch.get("current_q_min")),
                "current_q_max": finite_float(epoch.get("current_q_max")),
                "current_q_at_min": q_at_min,
                "current_q_at_max": q_at_max,
                "q_boundary_count": boundary_count,
                "q_boundary_pct": boundary_pct,
                "event_has_nonfinite": bool(event_nonfinite),
                "event_nonfinite_fields": ";".join(event_nonfinite),
                "early_stop_here": (
                    bool(early_stop)
                    and epoch_number == integer(early_stop.get("epoch"))
                ),
                "epoch_ms": finite_float(epoch.get("epoch_ms")),
            }
        )

    # The sum omits untimed final path evaluation, so its provenance is made
    # explicit in every output rather than presented as exact wall time.
    logged_runtime_ms = sum(
        value
        for value in (
            finite_float(load.get("wall_ms")),
            finite_float(cch.get("build_ms")),
            finite_float(cch.get("initial_full_customization_ms")),
            *(finite_float(epoch.get("epoch_ms")) for epoch in epochs),
            finite_float(finished.get("restore_full_customization_ms")),
        )
        if value is not None
    )
    runner_wall_seconds = finite_float(runner_result.get("wall_seconds"))
    runtime_seconds = (
        runner_wall_seconds
        if runner_wall_seconds is not None
        else logged_runtime_ms / 1000.0
    )
    runtime_source = (
        "runner_result.wall_seconds"
        if runner_wall_seconds is not None
        else "sum_of_logged_compute_times_lower_bound"
    )
    return {
        "run_id": run_id,
        "phase": phase,
        "log_path": str(log_path),
        "config": config,
        "load": load,
        "epochs": epochs,
        "trajectory": trajectory,
        "final_evaluation": final_evaluation,
        "finished": finished,
        "early_stop": early_stop,
        "runner_result": runner_result,
        "runner_result_path": (
            str(runner_result_path) if runner_result_path.is_file() else None
        ),
        "loss_nonfinite": loss_nonfinite,
        "loss_unparseable": loss_unparseable,
        "logged_runtime_seconds": logged_runtime_ms / 1000.0,
        "runtime_seconds": runtime_seconds,
        "runtime_source": runtime_source,
    }


def protocol_parameters(protocol: dict[str, Any]) -> dict[str, Any]:
    grid = protocol.get("convergence_grid", {})
    model = protocol.get("model", {})
    rules = protocol.get("predeclared_decision_rules", {})
    success_text = str(rules.get("successful_run", ""))
    primary_text = str(rules.get("primary_configuration", ""))

    boundary_match = re.search(
        r"fewer\s+than\s+([0-9.eE+-]+)\s+percent", success_text, re.IGNORECASE
    )
    tie_match = re.search(
        r"(?:at\s+most|within)\s+([0-9.eE+-]+)", primary_text, re.IGNORECASE
    )
    q_box = model.get("q_box", [0.1, 10.0])
    return {
        "timeout_seconds": float(grid.get("per_run_timeout_seconds", 900)),
        "boundary_threshold_pct": (
            float(boundary_match.group(1)) if boundary_match else 1.0
        ),
        "tie_tolerance": float(tie_match.group(1)) if tie_match else 1e-5,
        "q_min": float(q_box[0]),
        "q_max": float(q_box[1]),
    }


def best_event_before(
    trajectory: list[dict[str, Any]], horizon: int
) -> dict[str, Any] | None:
    eligible = [
        row
        for row in trajectory
        if row["epoch"] is not None
        and row["epoch"] < horizon
        and row["validation_relative_regret"] is not None
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda row: row["validation_relative_regret"])


def summarize_run(run: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    config = run["config"]
    finished = run["finished"]
    evaluation = run["final_evaluation"]
    epochs = run["epochs"]
    edges = integer(run["load"].get("edges"))
    q_at_min = integer(finished.get("best_q_at_min"))
    q_at_max = integer(finished.get("best_q_at_max"))
    boundary_count = (
        q_at_min + q_at_max
        if q_at_min is not None and q_at_max is not None
        else None
    )
    boundary_pct = (
        100.0 * boundary_count / edges
        if boundary_count is not None and edges not in (None, 0)
        else None
    )
    runner_read_error = run["runner_result"].get("_read_error")
    runner_status = run["runner_result"].get("status")
    runner_completed = runner_status in (None, "ok")
    complete = (
        bool(finished)
        and bool(evaluation)
        and runner_completed
        and runner_read_error is None
    )
    runtime_within_limit = (
        run["runtime_seconds"] <= parameters["timeout_seconds"]
    )
    loss_values_finite = not run["loss_nonfinite"] and not run["loss_unparseable"]
    boundary_below_threshold = (
        boundary_pct is not None
        and boundary_pct < parameters["boundary_threshold_pct"]
    )
    reasons = []
    if not finished or not evaluation:
        reasons.append("missing_FINISHED_or_validation_best")
    if runner_read_error is not None:
        reasons.append("runner_result_read_error")
    elif not runner_completed:
        reasons.append(f"runner_status_{runner_status}")
    if not runtime_within_limit:
        reasons.append("runtime_exceeds_timeout")
    if not loss_values_finite:
        reasons.append("nonfinite_or_unparseable_reported_loss")
    if not boundary_below_threshold:
        reasons.append("projection_boundary_fraction_not_below_threshold")

    last_epoch = max(
        (integer(epoch.get("epoch")) for epoch in epochs),
        default=None,
        key=lambda value: -1 if value is None else value,
    )
    summary: dict[str, Any] = {
        "run_id": run["run_id"],
        "phase": run["phase"],
        "log_path": run["log_path"],
        "train_variant": config.get("train"),
        "validation_variant": config.get("validation"),
        "eta0": json_number(config.get("eta0")),
        "lambda": json_number(config.get("lambda")),
        "configured_epochs": integer(config.get("epochs")),
        "epochs_executed": len(epochs),
        "last_epoch": last_epoch,
        "validation_events": len(run["trajectory"]),
        "complete": complete,
        "runner_status": runner_status,
        "runner_result_read_error": runner_read_error,
        "runner_result_path": run["runner_result_path"],
        "early_stopped": bool(run["early_stop"]),
        "early_stop_epoch": integer(run["early_stop"].get("epoch")),
        "early_stop_stale_evaluations": integer(
            run["early_stop"].get("stale_evaluations")
        ),
        "early_stop_patience": integer(run["early_stop"].get("patience")),
        "early_stop_min_delta": json_number(run["early_stop"].get("min_delta")),
        "best_epoch": integer(finished.get("best_epoch")),
        "selection_loss": finite_float(finished.get("selection_loss")),
        "validation_relative_regret": finite_float(evaluation.get("relative_regret")),
        "validation_mean_regret": finite_float(evaluation.get("mean_regret")),
        "validation_edge_f1": finite_float(evaluation.get("edge_f1")),
        "validation_exact_match": finite_float(evaluation.get("exact_match")),
        "validation_edge_jaccard": finite_float(evaluation.get("edge_jaccard")),
        "best_q_min": finite_float(finished.get("best_q_min")),
        "best_q_max": finite_float(finished.get("best_q_max")),
        "best_q_at_min": q_at_min,
        "best_q_at_max": q_at_max,
        "q_boundary_count": boundary_count,
        "q_boundary_pct": boundary_pct,
        "loss_values_finite": loss_values_finite,
        "nonfinite_loss_count": len(run["loss_nonfinite"]),
        "nonfinite_loss_fields": ";".join(run["loss_nonfinite"]),
        "unparseable_loss_count": len(run["loss_unparseable"]),
        "unparseable_loss_fields": ";".join(run["loss_unparseable"]),
        "logged_runtime_seconds": run["logged_runtime_seconds"],
        "runtime_seconds": run["runtime_seconds"],
        "runtime_source": run["runtime_source"],
        "runtime_within_limit": runtime_within_limit,
        "boundary_below_threshold": boundary_below_threshold,
        "checkpoint_path": finished.get("checkpoint_path"),
        "weights_path": finished.get("weights_path"),
        "multipliers_path": finished.get("multipliers_path"),
        "successful": not reasons,
        "failure_reasons": ";".join(reasons),
    }
    terminal_horizon = last_epoch + 1 if last_epoch is not None else None
    last20_start_epoch = (
        max(0, terminal_horizon - 20) if terminal_horizon is not None else None
    )
    best_before_last20 = (
        best_event_before(run["trajectory"], last20_start_epoch)
        if last20_start_epoch not in (None, 0)
        else None
    )
    best_at_terminal = (
        best_event_before(run["trajectory"], terminal_horizon)
        if terminal_horizon is not None
        else None
    )
    before_loss = (
        best_before_last20["validation_relative_regret"]
        if best_before_last20
        else None
    )
    terminal_loss = (
        best_at_terminal["validation_relative_regret"] if best_at_terminal else None
    )
    summary.update(
        {
            "convergence_evidence": (
                "strong_early_stop_after_declared_patience"
                if run["early_stop"]
                else "no_early_stop_retain_optimization_budget_caveat"
            ),
            "last20_start_epoch": last20_start_epoch,
            "best_relative_regret_before_last20": before_loss,
            "terminal_prefix_best_relative_regret": terminal_loss,
            "last20_absolute_improvement": (
                before_loss - terminal_loss
                if before_loss is not None and terminal_loss is not None
                else None
            ),
            "last20_relative_improvement": (
                1.0 - terminal_loss / before_loss
                if before_loss not in (None, 0) and terminal_loss is not None
                else None
            ),
        }
    )
    for horizon in HORIZONS:
        best = best_event_before(run["trajectory"], horizon)
        reached = last_epoch is not None and last_epoch >= horizon - 1
        status = (
            "reached"
            if reached
            else "early_stopped_before_horizon"
            if run["early_stop"]
            else "trajectory_ended_before_horizon"
        )
        summary.update(
            {
                f"h{horizon}_status": status,
                f"h{horizon}_reached": reached,
                f"h{horizon}_observed_through_epoch": (
                    min(last_epoch, horizon - 1) if last_epoch is not None else None
                ),
                f"h{horizon}_best_epoch": best["epoch"] if best else None,
                f"h{horizon}_best_relative_regret": (
                    best["validation_relative_regret"] if best else None
                ),
                f"h{horizon}_best_edge_f1": (
                    best["validation_edge_f1"] if best else None
                ),
                f"h{horizon}_best_exact_match": (
                    best["validation_exact_match"] if best else None
                ),
            }
        )
    h20 = summary.get("h20_best_relative_regret")
    h100 = summary.get("h100_best_relative_regret")
    summary["h20_to_h100_absolute_improvement"] = (
        h20 - h100
        if isinstance(h20, (int, float)) and isinstance(h100, (int, float))
        else None
    )
    summary["h20_to_h100_relative_improvement"] = (
        1.0 - h100 / h20
        if isinstance(h20, (int, float))
        and isinstance(h100, (int, float))
        and h20 != 0
        else None
    )
    return summary


def compare_prefix(
    runs_by_id: dict[str, dict[str, Any]], horizon: int = 20
) -> dict[str, Any]:
    control = runs_by_id.get(CONTROL_RUN_ID)
    long_run = runs_by_id.get(LONG_RUN_ID)
    if control is None or long_run is None:
        missing = [
            run_id
            for run_id, run in (
                (CONTROL_RUN_ID, control),
                (LONG_RUN_ID, long_run),
            )
            if run is None
        ]
        return {
            "status": "missing_run",
            "exact_match": False,
            "missing_run_ids": missing,
            "required_epochs": horizon,
        }

    control_epochs = {
        integer(row.get("epoch")): row
        for row in control["epochs"]
        if integer(row.get("epoch")) is not None
    }
    long_epochs = {
        integer(row.get("epoch")): row
        for row in long_run["epochs"]
        if integer(row.get("epoch")) is not None
    }
    differences: list[dict[str, Any]] = []
    missing_epochs: list[dict[str, Any]] = []
    for epoch in range(horizon):
        left = control_epochs.get(epoch)
        right = long_epochs.get(epoch)
        if left is None or right is None:
            missing_epochs.append(
                {
                    "epoch": epoch,
                    "control_present": left is not None,
                    "long_present": right is not None,
                }
            )
            continue
        fields: Iterable[str] = PREFIX_STATE_FIELDS
        if epoch < horizon - 1:
            fields = (*PREFIX_STATE_FIELDS, *PREFIX_UPDATE_FIELDS)
        for field in fields:
            if left.get(field) != right.get(field):
                differences.append(
                    {
                        "epoch": epoch,
                        "field": field,
                        "control": left.get(field),
                        "long": right.get(field),
                    }
                )

    exact = not differences and not missing_epochs
    return {
        "status": "passed" if exact else "failed",
        "exact_match": exact,
        "control_run_id": CONTROL_RUN_ID,
        "long_run_id": LONG_RUN_ID,
        "required_epochs": horizon,
        "terminal_epoch_update_excluded": horizon - 1,
        "comparison": "exact token strings; timing excluded; terminal post-evaluation update excluded",
        "state_fields": list(PREFIX_STATE_FIELDS),
        "update_fields_for_epochs_before_terminal": list(PREFIX_UPDATE_FIELDS),
        "missing_epochs": missing_epochs,
        "difference_count": len(differences),
        "differences": differences[:100],
        "differences_truncated": len(differences) > 100,
    }


def is_full_convergence(summary: dict[str, Any]) -> bool:
    return summary.get("phase") == "convergence_full" or str(
        summary.get("run_id", "")
    ).startswith("conv_full_")


def choose_candidate(
    summaries: list[dict[str, Any]], tie_tolerance: float
) -> dict[str, Any]:
    candidates = [
        row
        for row in summaries
        if is_full_convergence(row)
        and row.get("successful")
        and isinstance(row.get("selection_loss"), (int, float))
    ]
    if not candidates:
        return {
            "status": "no_successful_full_convergence_run",
            "run_id": None,
            "eligible_run_ids": [],
        }
    minimum = min(row["selection_loss"] for row in candidates)
    tied = [
        row
        for row in candidates
        if row["selection_loss"] <= minimum + tie_tolerance
    ]

    def secondary_key(row: dict[str, Any]) -> tuple[float, float, str]:
        f1 = row.get("validation_edge_f1")
        eta = row.get("eta0")
        return (
            -(float(f1) if isinstance(f1, (int, float)) else -math.inf),
            float(eta) if isinstance(eta, (int, float)) else math.inf,
            str(row["run_id"]),
        )

    chosen = min(tied, key=secondary_key)
    ranking = sorted(
        candidates,
        key=lambda row: (
            row["selection_loss"],
            -(row.get("validation_edge_f1") or -math.inf),
            row.get("eta0") if isinstance(row.get("eta0"), (int, float)) else math.inf,
        ),
    )
    return {
        "status": "selected",
        "run_id": chosen["run_id"],
        "selection_loss": chosen["selection_loss"],
        "validation_relative_regret": chosen["validation_relative_regret"],
        "validation_edge_f1": chosen.get("validation_edge_f1"),
        "validation_exact_match": chosen.get("validation_exact_match"),
        "best_epoch": chosen.get("best_epoch"),
        "eta0": chosen.get("eta0"),
        "checkpoint_path": chosen.get("checkpoint_path"),
        "weights_path": chosen.get("weights_path"),
        "multipliers_path": chosen.get("multipliers_path"),
        "tie_tolerance": tie_tolerance,
        "minimum_selection_loss": minimum,
        "minimum_relative_regret": minimum,
        "tie_set_run_ids": [row["run_id"] for row in tied],
        "eligible_run_ids": [row["run_id"] for row in ranking],
        "selection_rule": "minimum relative regret; within tolerance use higher edge F1, then smaller eta0",
    }


def horizon_rankings(
    summaries: list[dict[str, Any]], tie_tolerance: float
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in HORIZONS:
        metric = f"h{horizon}_best_relative_regret"
        rows = [
            row
            for row in summaries
            if is_full_convergence(row)
            and row.get("successful")
            and row.get(f"h{horizon}_reached")
            and isinstance(row.get(metric), (int, float))
        ]
        ordered = sorted(
            rows,
            key=lambda row: (
                row[metric],
                -(row.get(f"h{horizon}_best_edge_f1") or -math.inf),
                row.get("eta0") if isinstance(row.get("eta0"), (int, float)) else math.inf,
            ),
        )
        best = None
        if ordered:
            minimum = ordered[0][metric]
            tied = [row for row in ordered if row[metric] <= minimum + tie_tolerance]
            best = min(
                tied,
                key=lambda row: (
                    -(row.get(f"h{horizon}_best_edge_f1") or -math.inf),
                    row.get("eta0")
                    if isinstance(row.get("eta0"), (int, float))
                    else math.inf,
                ),
            )["run_id"]
        result[str(horizon)] = {
            "best_successful_full_run_id": best,
            "ranking": [
                {
                    "run_id": row["run_id"],
                    "best_epoch": row[f"h{horizon}_best_epoch"],
                    "relative_regret": row[metric],
                    "edge_f1": row[f"h{horizon}_best_edge_f1"],
                    "exact_match": row[f"h{horizon}_best_exact_match"],
                }
                for row in ordered
            ],
        }
    return result


def write_csv(path: Path, rows: list[dict[str, Any]], preferred: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred_present = [key for key in preferred if any(key in row for row in rows)]
    remaining = sorted({key for row in rows for key in row} - set(preferred_present))
    columns = [*preferred_present, *remaining]
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def analyze(
    run_root: Path,
    protocol_path: Path,
    trajectory_csv: Path,
    summary_csv: Path,
    summary_json: Path,
) -> dict[str, Any]:
    if not run_root.is_dir():
        raise ValueError(f"run root is not a directory: {run_root}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    parameters = protocol_parameters(protocol)
    log_paths = sorted(run_root.rglob("model_training.log"))
    if not log_paths:
        raise ValueError(f"no model_training.log found below {run_root}")
    runs = [parse_log(path, run_root) for path in log_paths]
    duplicate_ids = sorted(
        run_id
        for run_id in {run["run_id"] for run in runs}
        if sum(candidate["run_id"] == run_id for candidate in runs) > 1
    )
    if duplicate_ids:
        raise ValueError(f"duplicate run IDs below run root: {duplicate_ids}")
    runs_by_id = {run["run_id"]: run for run in runs}
    trajectories = [row for run in runs for row in run["trajectory"]]
    trajectories.sort(key=lambda row: (row["phase"], row["run_id"], row["epoch"]))
    summaries = [summarize_run(run, parameters) for run in runs]
    summaries.sort(key=lambda row: (row["phase"], row["run_id"]))

    candidate = choose_candidate(summaries, parameters["tie_tolerance"])
    prefix_check = compare_prefix(runs_by_id)
    report = {
        "schema_version": 1,
        "study": protocol.get("study"),
        "protocol_path": str(protocol_path),
        "run_root": str(run_root),
        "source_policy": (
            "only model_training.log, adjacent runner_result.json when present, and the "
            "protocol JSON were read; no dataset split was opened"
        ),
        "test_policy": protocol.get("data_policy", {}).get("test_policy"),
        "runtime_caveat": (
            "runner_result.wall_seconds is used when present; otherwise runtime is a lower "
            "bound from timed log fields and FINISHED is treated as completion under the "
            "protocol's bounded runner"
        ),
        "decision_parameters": parameters,
        "run_count": len(summaries),
        "successful_run_count": sum(bool(row["successful"]) for row in summaries),
        "trajectory_event_count": len(trajectories),
        "primary_candidate": candidate,
        "control_prefix_check": prefix_check,
        "horizon_rankings": horizon_rankings(
            summaries, parameters["tie_tolerance"]
        ),
        "runs": summaries,
    }
    write_csv(
        trajectory_csv,
        trajectories,
        (
            "run_id",
            "phase",
            "train_variant",
            "eta0",
            "lambda",
            "epoch",
            "validation_relative_regret",
            "validation_edge_f1",
            "validation_exact_match",
            "is_new_prefix_best",
            "prefix_best_epoch",
            "prefix_best_relative_regret",
            "stale_evaluations",
            "early_stop_here",
            "current_q_min",
            "current_q_max",
            "q_boundary_pct",
            "event_has_nonfinite",
        ),
    )
    write_csv(
        summary_csv,
        summaries,
        (
            "run_id",
            "phase",
            "train_variant",
            "eta0",
            "lambda",
            "successful",
            "failure_reasons",
            "complete",
            "epochs_executed",
            "early_stopped",
            "early_stop_epoch",
            "convergence_evidence",
            "last20_absolute_improvement",
            "last20_relative_improvement",
            "best_epoch",
            "selection_loss",
            "validation_relative_regret",
            "validation_edge_f1",
            "validation_exact_match",
            "q_boundary_pct",
            "loss_values_finite",
            "logged_runtime_seconds",
            "h20_best_epoch",
            "h20_best_relative_regret",
            "h50_best_epoch",
            "h50_best_relative_regret",
            "h100_best_epoch",
            "h100_best_relative_regret",
            "h20_to_h100_absolute_improvement",
        ),
    )
    atomic_json(summary_json, report)
    return report


def synthetic_epoch(epoch: int, terminal: bool = False) -> str:
    validation = epoch == 0 or (epoch + 1) % 5 == 0
    regret = 0.2 - 0.001 * epoch
    update_status = "final_skipped" if terminal else "applied"
    suffix = ""
    if validation:
        suffix = (
            f" validation_regret={1000 * regret:.6f}"
            f" validation_relative_regret={regret:.8f}"
            f" validation_exact_match={0.3 + epoch / 1000:.6f}"
            f" validation_edge_f1={0.5 + epoch / 1000:.6f}"
            " validation_edge_jaccard=0.400000"
            f" selection_loss={regret:.12f} stale_evaluations=0 BEST"
        )
    else:
        suffix = " selection_loss=NA"
    return (
        f"EPOCH epoch={epoch} train_regret={2000 - epoch:.6f}"
        f" train_relative_regret={0.3 - epoch / 1000:.8f} regularization=1.000000"
        f" train_objective={2001 - epoch:.6f} count_residual_l1={500 - epoch}"
        " train_queries=100 train_oracle_ms=1.000 changed_edges=10 changed_pct=1.0000"
        f" optimizer_ms=0.100 customization_ms=0.100 checkpoint_ms=0.100"
        f" update_mode=Full update_status={update_status} current_q_min=0.500000"
        " current_q_max=1.500000 current_q_at_min=0 current_q_at_max=0"
        " current_max_quantization_error=0.100000 next_q_min=0.500000"
        " next_q_max=1.500000 next_q_at_min=0 next_q_at_max=0"
        " next_max_quantization_error=0.100000 epoch_ms=10.000"
        + suffix
        + ("" if terminal else " eta=0.00010000 latent_max_delta=0.01000000 projected_edges=0")
    )


def write_synthetic_log(
    path: Path,
    run_id: str,
    epochs: int,
    eta0: float,
    final_regret: float | None = None,
    final_f1: float = 0.7,
    nonfinite: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"CONFIG city=beijing epochs={epochs} train=all validation=development"
        f" solver=ProjectedSubgradient eta0={eta0} lambda=100000 patience=4",
        "LOAD nodes=100 edges=1000 valid_train=100 wall_ms=5.000",
        "CCH build_ms=2.000 initial_full_customization_ms=1.000 threads=1",
    ]
    for epoch in range(epochs):
        lines.append(synthetic_epoch(epoch, terminal=epoch + 1 == epochs))
    best_epoch = epochs - 1 if (epochs - 1 == 0 or epochs % 5 == 0) else epochs - 2
    if final_regret is None:
        final_regret = 0.2 - 0.001 * best_epoch
    loss = "NaN" if nonfinite else f"{final_regret:.8f}"
    lines.extend(
        [
            f"EVAL split=validation_best samples=100 mean_regret=100.000000"
            f" relative_regret={loss} exact_match=0.400000 edge_precision=0.700000"
            f" edge_recall=0.700000 edge_f1={final_f1:.6f} edge_jaccard=0.600000",
            "TEST_SKIPPED synthetic",
            f"FINISHED best_epoch={best_epoch} selection_loss={loss}"
            " best_train_regret=100.000000 best_regularization=1.000000"
            " best_q_min=0.500000 best_q_max=1.500000 best_q_at_min=0"
            " best_q_at_max=0 train_regret_improvement_pct=1.0"
            " restore_full_customization_ms=1.000 peak_rss_kib=100",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class SyntheticLogTests(unittest.TestCase):
    def test_end_to_end_and_tie_break(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "runs"
            write_synthetic_log(
                run_root / "controls" / CONTROL_RUN_ID / "model_training.log",
                CONTROL_RUN_ID,
                20,
                1e-4,
            )
            write_synthetic_log(
                run_root / "convergence_full" / LONG_RUN_ID / "model_training.log",
                LONG_RUN_ID,
                100,
                1e-4,
                final_regret=0.100000,
                final_f1=0.8,
            )
            (
                run_root / "convergence_full" / LONG_RUN_ID / "runner_result.json"
            ).write_text(
                json.dumps({"status": "ok", "wall_seconds": 123.456}),
                encoding="utf-8",
            )
            write_synthetic_log(
                run_root
                / "convergence_full"
                / "conv_full_eta3e4"
                / "model_training.log",
                "conv_full_eta3e4",
                100,
                3e-4,
                final_regret=0.099995,
                final_f1=0.7,
            )
            write_synthetic_log(
                run_root
                / "convergence_full"
                / "conv_full_bad"
                / "model_training.log",
                "conv_full_bad",
                20,
                1e-3,
                nonfinite=True,
            )
            protocol_path = root / "protocol.json"
            protocol_path.write_text(
                json.dumps(
                    {
                        "study": "synthetic",
                        "model": {"q_box": [0.1, 10.0]},
                        "convergence_grid": {"per_run_timeout_seconds": 900},
                        "predeclared_decision_rules": {
                            "successful_run": "all losses finite and fewer than 1 percent at boundary",
                            "primary_configuration": "if losses differ by at most 1e-5 choose higher edge F1, then smaller eta0",
                        },
                        "data_policy": {"test_policy": "never read test"},
                    }
                ),
                encoding="utf-8",
            )
            report = analyze(
                run_root,
                protocol_path,
                root / "trajectory.csv",
                root / "summary.csv",
                root / "summary.json",
            )
            self.assertTrue(report["control_prefix_check"]["exact_match"])
            self.assertEqual(report["primary_candidate"]["run_id"], LONG_RUN_ID)
            self.assertEqual(
                report["horizon_rankings"]["100"]["ranking"][0]["best_epoch"],
                99,
            )
            selected = next(
                row for row in report["runs"] if row["run_id"] == LONG_RUN_ID
            )
            self.assertEqual(selected["runtime_source"], "runner_result.wall_seconds")
            self.assertEqual(selected["runtime_seconds"], 123.456)
            bad = next(row for row in report["runs"] if row["run_id"] == "conv_full_bad")
            self.assertFalse(bad["successful"])
            self.assertGreater(bad["nonfinite_loss_count"], 0)
            self.assertTrue((root / "trajectory.csv").is_file())
            self.assertTrue((root / "summary.csv").is_file())
            self.assertTrue((root / "summary.json").is_file())


def self_test() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticLogTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    if args.self_test:
        return self_test()
    report = analyze(
        args.run_root,
        args.protocol,
        args.trajectory_csv,
        args.summary_csv,
        args.summary_json,
    )
    candidate = report["primary_candidate"]
    prefix = report["control_prefix_check"]
    print(
        f"ANALYZED runs={report['run_count']} successful={report['successful_run_count']} "
        f"events={report['trajectory_event_count']} primary={candidate.get('run_id')} "
        f"prefix_check={prefix['status']}"
    )
    print(
        f"OUTPUT trajectory_csv={args.trajectory_csv} summary_csv={args.summary_csv} "
        f"summary_json={args.summary_json}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
