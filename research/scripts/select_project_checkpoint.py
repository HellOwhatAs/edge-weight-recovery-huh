#!/usr/bin/env python3
"""Select the Project checkpoint using the frozen validation split only.

Every ``checkpoint-N.json`` in the training directory must have a matching
``checkpoint-N.evaluation.json`` in the validation directory.  The ranking is
the study protocol's macro edge F1, exact sequence match, then earliest update.
The resulting receipt binds both inputs by SHA-256 so the later test run can be
audited without reopening the validation predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "ewr.project-checkpoint-selection/v1"
EVALUATION_SCHEMA = "ewr.evaluation-summary/v1"
CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)\.json$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_evaluation(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read validation evaluation {path}: {error}") from error
    if not isinstance(value, dict) or value.get("schema") != EVALUATION_SCHEMA:
        raise RuntimeError(f"{path} is not an {EVALUATION_SCHEMA} document")
    if value.get("sample_count") != 500:
        raise RuntimeError(f"{path} does not contain the frozen 500-route split")
    metrics = value.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError(f"{path}.metrics must be an object")
    result: dict[str, float] = {}
    for key in ("edge_precision", "edge_recall", "edge_f1", "edge_jaccard", "exact_match"):
        observed = metrics.get(key)
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            raise RuntimeError(f"{path}.metrics.{key} must be numeric")
        converted = float(observed)
        if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
            raise RuntimeError(f"{path}.metrics.{key} must be finite in [0, 1]")
        result[key] = converted
    return result


def select(training_dir: Path, validation_dir: Path) -> dict[str, Any]:
    checkpoints: list[tuple[int, Path]] = []
    for path in training_dir.iterdir():
        match = CHECKPOINT_RE.fullmatch(path.name)
        if match and path.is_file():
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort()
    if not checkpoints:
        raise RuntimeError(f"no checkpoint-N.json files found in {training_dir}")
    if len({update for update, _ in checkpoints}) != len(checkpoints):
        raise RuntimeError("duplicate checkpoint update numbers")

    candidates: list[dict[str, Any]] = []
    for update, checkpoint in checkpoints:
        evaluation = validation_dir / f"checkpoint-{update}.evaluation.json"
        if not evaluation.is_file():
            raise RuntimeError(f"missing validation evaluation for update {update}: {evaluation}")
        metrics = load_evaluation(evaluation)
        candidates.append(
            {
                "update": update,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "checkpoint_bytes": checkpoint.stat().st_size,
                "validation_evaluation": str(evaluation),
                "validation_evaluation_sha256": sha256_file(evaluation),
                "metrics": metrics,
            }
        )

    winner = max(
        candidates,
        key=lambda item: (
            item["metrics"]["edge_f1"],
            item["metrics"]["exact_match"],
            -item["update"],
        ),
    )
    return {
        "schema": SCHEMA,
        "selection_split": "validation",
        "selection_rule": "maximum macro edge F1, then exact match, then earliest update",
        "candidate_count": len(candidates),
        "selected_update": winner["update"],
        "selected_checkpoint": winner["checkpoint"],
        "selected_checkpoint_sha256": winner["checkpoint_sha256"],
        "selected_checkpoint_bytes": winner["checkpoint_bytes"],
        "selected_validation_metrics": winner["metrics"],
        "candidates": candidates,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = select(args.training_dir.resolve(), args.validation_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
