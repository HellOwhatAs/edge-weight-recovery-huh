#!/usr/bin/env python3
"""Run a bounded, cacheable validation-only experiment matrix.

The matrix is a CSV with these required columns:
run_id, phase, scale, seed, eta0, lambda, train_variant.
Optional columns are metric_update, epochs, and patience.
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
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


REQUIRED_COLUMNS = {
    "run_id",
    "phase",
    "scale",
    "seed",
    "eta0",
    "lambda",
    "train_variant",
}
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--binary", default=Path("target/release/edge-weight-recovery"), type=Path
    )
    parser.add_argument("--city", default="beijing")
    parser.add_argument("--validation-variant", required=True)
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--rayon-threads", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--default-epochs", type=int, default=20)
    parser.add_argument("--default-patience", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_matrix(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"matrix is missing columns: {sorted(missing)}")
        rows = list(reader)
    seen: set[str] = set()
    for row in rows:
        run_id = row["run_id"]
        if not SAFE_RUN_ID.fullmatch(run_id):
            raise ValueError(f"unsafe run_id {run_id!r}")
        if run_id in seen:
            raise ValueError(f"duplicate run_id {run_id!r}")
        seen.add(run_id)
        if row.get("metric_update", "full") not in {"full", "partial"}:
            raise ValueError(f"invalid metric_update in {run_id}")
    return rows


def tokens(line: str) -> tuple[str, dict[str, str]]:
    pieces = shlex.split(line.strip())
    if not pieces:
        return "", {}
    values: dict[str, str] = {}
    for token in pieces[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            values[key] = value
    return pieces[0], values


def as_number(value: str | None) -> int | float | str | None:
    if value is None or value == "NA":
        return None
    try:
        integer = int(value)
        if str(integer) == value:
            return integer
    except ValueError:
        pass
    try:
        number = float(value)
        return number if math.isfinite(number) else value
    except ValueError:
        return value


def mean(rows: list[dict[str, str]], key: str) -> float | None:
    values = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return sum(values) / len(values) if values else None


def parse_log(path: Path) -> dict[str, Any]:
    data: dict[str, dict[str, str]] = {}
    epochs: list[dict[str, str]] = []
    evaluation: dict[str, str] = {}
    finished: dict[str, str] = {}
    config: dict[str, str] = {}
    if not path.exists():
        return {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        kind, values = tokens(line)
        if kind == "CONFIG":
            config = values
        elif kind == "DATA" and "split" in values:
            data[values["split"]] = values
        elif kind == "EPOCH":
            epochs.append(values)
        elif kind == "EVAL" and values.get("split") == "validation_best":
            evaluation = values
        elif kind == "FINISHED":
            finished = values
    if not finished or not evaluation:
        return {}
    best_epoch = str(finished.get("best_epoch", ""))
    best_epoch_row = next(
        (epoch for epoch in epochs if epoch.get("epoch") == best_epoch), {}
    )
    update_epochs = [
        epoch
        for epoch in epochs
        if epoch.get("update_status")
        in {"applied", "latent_only_no_integer_change", "shock_applied"}
    ]
    result: dict[str, Any] = {
        "status": "ok",
        "epochs_executed": len(epochs),
        "avg_epoch_ms": mean(epochs, "epoch_ms"),
        "avg_train_oracle_ms": mean(epochs, "train_oracle_ms"),
        "avg_customization_ms": mean(epochs, "customization_ms"),
        "avg_changed_pct": mean(epochs, "changed_pct"),
        "avg_update_customization_ms": mean(update_epochs, "customization_ms"),
        "avg_update_changed_pct": mean(update_epochs, "changed_pct"),
    }
    for prefix, values, keys in [
        (
            "config_",
            config,
            ["solver", "selection_metric", "metric_update", "run_test"],
        ),
        ("train_", data.get("train", {}), ["available", "inspected", "accepted", "cyclic"]),
        (
            "validation_",
            data.get("validation", {}),
            ["available", "inspected", "accepted", "cyclic"],
        ),
        (
            "best_",
            finished,
            [
                "best_epoch",
                "selection_loss",
                "best_train_regret",
                "best_regularization",
                "best_q_min",
                "best_q_max",
                "best_q_at_min",
                "best_q_at_max",
                "peak_rss_kib",
                "train_regret_improvement_pct",
            ],
        ),
        (
            "best_epoch_",
            best_epoch_row,
            [
                "train_relative_regret",
                "count_residual_l1",
                "changed_pct",
                "train_oracle_ms",
                "customization_ms",
                "current_max_quantization_error",
            ],
        ),
        (
            "validation_",
            evaluation,
            [
                "samples",
                "mean_regret",
                "relative_regret",
                "exact_match",
                "edge_precision",
                "edge_recall",
                "edge_f1",
                "edge_jaccard",
            ],
        ),
    ]:
        for key in keys:
            result[prefix + key] = as_number(values.get(key))
    train_relative = result.get("best_epoch_train_relative_regret")
    validation_relative = result.get("validation_relative_regret")
    if isinstance(train_relative, (int, float)) and isinstance(
        validation_relative, (int, float)
    ):
        result["relative_regret_gap"] = validation_relative - train_relative
    train_mean = result.get("best_best_train_regret")
    validation_mean = result.get("validation_mean_regret")
    if isinstance(train_mean, (int, float)) and isinstance(validation_mean, (int, float)):
        result["mean_regret_gap"] = validation_mean - train_mean
    return result


def checkpoint_quantiles(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    values = sorted(float(value) for value in checkpoint.get("multipliers", []))
    if not values:
        return {}

    def quantile(probability: float) -> float:
        rank = probability * (len(values) - 1)
        lower = int(math.floor(rank))
        upper = int(math.ceil(rank))
        fraction = rank - lower
        return values[lower] * (1.0 - fraction) + values[upper] * fraction

    return {
        "best_q_p05": quantile(0.05),
        "best_q_p25": quantile(0.25),
        "best_q_median": quantile(0.50),
        "best_q_p75": quantile(0.75),
        "best_q_p95": quantile(0.95),
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run_one(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    run_id = row["run_id"]
    output_dir = args.run_root / row["phase"] / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "model"
    log_path = Path(f"{prefix}_training.log")
    result_path = output_dir / "runner_result.json"
    cached = parse_log(log_path)
    if cached and not args.force:
        cached.update(checkpoint_quantiles(Path(f"{prefix}_checkpoint.json")))
        originally_cached = True
        if result_path.exists():
            prior = json.loads(result_path.read_text(encoding="utf-8"))
            originally_cached = bool(prior.get("cached", False))
            prior.update(cached)
            cached = prior
        cached.update(row)
        cached.update(
            {
                "cached": originally_cached,
                "cache_hit_last_invocation": True,
                "checkpoint_path": str(Path(f"{prefix}_checkpoint.json")),
                "log_path": str(log_path),
            }
        )
        return cached

    epochs = int(row.get("epochs") or args.default_epochs)
    patience = int(row.get("patience") or args.default_patience)
    metric_update = row.get("metric_update") or "full"
    solver = row.get("solver") or "projected"
    if solver not in {"projected", "adam-shock"}:
        raise ValueError(f"invalid solver {solver!r} in {run_id}")
    command = [
        str(args.binary.resolve()),
        "--city",
        args.city,
        "--epochs",
        str(epochs),
        "--patience",
        str(patience),
        "--train-variant",
        row["train_variant"],
        "--validation-variant",
        args.validation_variant,
        "--test-variant",
        "all",
        "--solver",
        solver,
        "--metric-update",
        metric_update,
        "--selection-metric",
        "relative-regret",
        "--eta0",
        row["eta0"],
        "--lambda",
        row["lambda"],
        "--eval-every",
        "1",
        "--seed",
        row["seed"],
        "--output-prefix",
        str(prefix.resolve()),
    ]
    if solver == "adam-shock":
        command.extend(
            ["--adam-learning-rate", row.get("adam_learning_rate") or "3000"]
        )
    environment = os.environ.copy()
    environment["RAYON_NUM_THREADS"] = str(args.rayon_threads)
    started = time.monotonic()
    status = "error"
    error = ""
    returncode: int | None = None
    try:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout_seconds,
            check=False,
        )
        returncode = completed.returncode
        (output_dir / "process_output.txt").write_text(
            completed.stdout, encoding="utf-8"
        )
        status = "ok" if returncode == 0 else "error"
        if returncode != 0:
            error = "\n".join(completed.stdout.splitlines()[-20:])
    except subprocess.TimeoutExpired as timeout:
        status = "timeout"
        output = timeout.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        (output_dir / "process_output.txt").write_text(output, encoding="utf-8")
        error = f"exceeded {args.timeout_seconds} seconds"
    wall_seconds = time.monotonic() - started

    parsed = parse_log(log_path) if status == "ok" else {}
    parsed.update(checkpoint_quantiles(Path(f"{prefix}_checkpoint.json")))
    if status == "ok" and not parsed:
        status = "incomplete"
        error = "process exited successfully without complete FINISHED/EVAL records"
    result: dict[str, Any] = dict(row)
    result.update(parsed)
    result.update(
        {
            "status": status,
            "cached": False,
            "cache_hit_last_invocation": False,
            "wall_seconds": wall_seconds,
            "returncode": returncode,
            "error": error,
            "checkpoint_path": str(Path(f"{prefix}_checkpoint.json")),
            "log_path": str(log_path),
        }
    )
    atomic_json(result_path, result)
    return result


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("runs", []) if isinstance(value, dict) else value
    return {str(row["run_id"]): row for row in rows if "run_id" in row}


def write_summaries(
    rows: list[dict[str, Any]], csv_path: Path, json_path: Path
) -> None:
    existing = load_existing(json_path)
    existing.update({str(row["run_id"]): row for row in rows})
    combined = sorted(existing.values(), key=lambda row: (row.get("phase", ""), row["run_id"]))
    atomic_json(
        json_path,
        {
            "schema_version": 1,
            "selection_policy": "minimum validation relative regret; edge F1 is secondary",
            "test_policy": "test is never loaded by this runner",
            "runs": combined,
        },
    )
    columns: list[str] = []
    for preferred in [
        "run_id",
        "phase",
        "scale",
        "seed",
        "eta0",
        "lambda",
        "metric_update",
        "train_variant",
        "status",
        "cached",
        "wall_seconds",
        "train_accepted",
        "validation_accepted",
        "best_best_epoch",
        "best_selection_loss",
        "validation_relative_regret",
        "validation_mean_regret",
        "validation_exact_match",
        "validation_edge_f1",
        "validation_edge_jaccard",
        "best_epoch_train_relative_regret",
        "best_train_regret",
        "best_q_min",
        "best_q_max",
        "avg_epoch_ms",
        "avg_train_oracle_ms",
        "avg_customization_ms",
        "avg_changed_pct",
        "avg_update_customization_ms",
        "avg_update_changed_pct",
        "relative_regret_gap",
        "mean_regret_gap",
        "best_peak_rss_kib",
        "best_q_p05",
        "best_q_median",
        "best_q_p95",
    ]:
        if any(preferred in row for row in combined):
            columns.append(preferred)
    columns.extend(
        sorted({key for row in combined for key in row} - set(columns))
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(csv_path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined)
    temporary.replace(csv_path)


def main() -> int:
    args = arguments()
    if args.jobs < 1 or args.rayon_threads < 1:
        raise ValueError("--jobs and --rayon-threads must be positive")
    if args.timeout_seconds > 900:
        raise ValueError("per-run timeout must not exceed 900 seconds")
    rows = read_matrix(args.matrix)
    args.run_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_to_id = {executor.submit(run_one, row, args): row["run_id"] for row in rows}
        for future in as_completed(future_to_id):
            run_id = future_to_id[future]
            try:
                result = future.result()
            except Exception as error:  # keep the rest of a matrix alive
                result = {"run_id": run_id, "status": "runner_error", "error": repr(error)}
            results.append(result)
            print(
                f"RUN run_id={run_id} status={result.get('status')} "
                f"wall_seconds={result.get('wall_seconds')} cached={result.get('cached')}",
                flush=True,
            )
    write_summaries(results, args.summary_csv, args.summary_json)
    failures = [row for row in results if row.get("status") != "ok"]
    print(
        f"SUMMARY completed={len(results) - len(failures)} failed={len(failures)} "
        f"csv={args.summary_csv} json={args.summary_json}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
