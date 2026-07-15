#!/usr/bin/env python3
"""Summarize standard edge-only metrics from structured training logs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
from typing import Any


STANDARD_METRICS = (
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
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def fields(line: str) -> tuple[str, dict[str, str]]:
    pieces = shlex.split(line.strip())
    if not pieces:
        return "", {}
    values: dict[str, str] = {}
    for piece in pieces[1:]:
        if "=" in piece:
            key, value = piece.split("=", 1)
            values[key] = value
    return pieces[0], values


def number(value: str | None) -> float | int | None:
    if value is None:
        return None
    try:
        integer = int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return None
    return integer


def summarize(path: Path) -> dict[str, Any]:
    config: dict[str, str] = {}
    data: dict[str, dict[str, int | float | None]] = {}
    epochs: list[dict[str, int | float | None]] = []
    validation: dict[str, int | float | None] | None = None
    finished: dict[str, int | float | None] | None = None
    test_skipped = False

    for line in path.read_text(encoding="utf-8").splitlines():
        kind, values = fields(line)
        if kind == "CONFIG":
            config = values
        elif kind == "DATA" and "split" in values:
            data[values["split"]] = {
                key: number(values.get(key))
                for key in ("available", "inspected", "accepted", "dropped", "cyclic")
            }
        elif kind == "EPOCH":
            epochs.append(
                {
                    key: number(values.get(key))
                    for key in (
                        "epoch",
                        "train_regret",
                        "train_relative_regret",
                        "validation_relative_regret",
                        "selection_loss",
                    )
                }
            )
        elif kind == "EVAL" and values.get("split") == "validation_best":
            validation = {key: number(values.get(key)) for key in STANDARD_METRICS}
            validation["samples"] = number(values.get("samples"))
        elif kind == "FINISHED":
            finished = {
                key: number(values.get(key))
                for key in (
                    "best_epoch",
                    "selection_loss",
                    "best_train_regret",
                    "best_regularization",
                    "best_q_min",
                    "best_q_max",
                )
            }
        elif kind == "TEST_SKIPPED":
            test_skipped = True

    if not config or not epochs or validation is None or finished is None:
        raise ValueError(f"{path}: incomplete structured training log")
    if not test_skipped or config.get("run_test") != "false":
        raise ValueError(f"{path}: test was not demonstrably skipped")
    return {
        "log": str(path),
        "city": config.get("city"),
        "train_variant": config.get("train"),
        "validation_variant": config.get("validation"),
        "data": data,
        "epochs": epochs,
        "selected": finished,
        "validation": validation,
        "test_read": False,
    }


def main() -> int:
    args = arguments()
    result = {
        "schema_version": 1,
        "selection_metric": "aggregate_validation_relative_regret",
        "runs": [summarize(path) for path in args.logs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
