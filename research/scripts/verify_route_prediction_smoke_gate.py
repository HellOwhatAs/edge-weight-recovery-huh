#!/usr/bin/env python3
"""Authorize full-test prediction after seven exact reference-set reruns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


SCHEMA = "ewr.full-test-route-smoke-gate/v3"
EXPECTED_METHODS = (
    "project",
    "sp_length",
    "markov_sp",
    "neuromlr_greedy",
    "drncs_lg",
    "drpk_static",
    "drp_tp",
)


class GateError(ValueError):
    """A smoke output or frozen reference differs from the protocol."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_json(raw: bytes, fields: set[str], context: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateError(f"{context}: invalid JSON: {error}") from error
    if not isinstance(value, dict) or set(value) != fields:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise GateError(
            f"{context}: expected fields {sorted(fields)}, got {actual}"
        )
    return value


def valid_edge_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            not isinstance(edge, bool)
            and isinstance(edge, int)
            and 0 <= edge <= 0xFFFF_FFFF
            for edge in value
        )
    )


def dataset_id(raw: bytes, row: int) -> str:
    value = exact_json(
        raw,
        {"sample_id", "original_edge_ids"},
        f"reference dataset row {row}",
    )
    sample_id = value["sample_id"]
    if not isinstance(sample_id, str) or not sample_id:
        raise GateError(f"reference dataset row {row}: invalid sample_id")
    if not valid_edge_list(value["original_edge_ids"]):
        raise GateError(f"reference dataset row {row}: invalid edge list")
    return sample_id


def prediction_id(raw: bytes, row: int, context: str) -> str:
    value = exact_json(
        raw, {"sample_id", "predicted_edge_ids"}, f"{context} row {row}"
    )
    sample_id = value["sample_id"]
    if not isinstance(sample_id, str) or not sample_id:
        raise GateError(f"{context} row {row}: invalid sample_id")
    if not valid_edge_list(value["predicted_edge_ids"]):
        raise GateError(f"{context} row {row}: invalid predicted edge list")
    return sample_id


def require_file(path: Path, context: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise GateError(f"missing {context}: {resolved}")
    return resolved


def read_reference_ids(path: Path, expected_rows: int) -> list[str]:
    sample_ids: list[str] = []
    seen: set[str] = set()
    with path.open("rb") as source:
        for row, line in enumerate(source, 1):
            if row > expected_rows:
                raise GateError(
                    f"reference dataset has more than {expected_rows} rows"
                )
            sample_id = dataset_id(line, row)
            if sample_id in seen:
                raise GateError(
                    f"reference dataset repeats sample_id {sample_id!r}"
                )
            seen.add(sample_id)
            sample_ids.append(sample_id)
    if len(sample_ids) != expected_rows:
        raise GateError(
            f"reference dataset has {len(sample_ids)} rows, expected {expected_rows}"
        )
    return sample_ids


def verify_prediction(
    method: str, reference: Path, candidate: Path, sample_ids: list[str]
) -> dict[str, Any]:
    reference_bytes = reference.read_bytes()
    candidate_bytes = candidate.read_bytes()
    if candidate_bytes != reference_bytes:
        raise GateError(
            f"{method}: candidate prediction bytes differ from frozen reference"
        )
    lines = candidate_bytes.splitlines(keepends=True)
    if len(lines) != len(sample_ids):
        raise GateError(
            f"{method}: prediction has {len(lines)} rows, expected {len(sample_ids)}"
        )
    seen: set[str] = set()
    for row, (line, expected_id) in enumerate(zip(lines, sample_ids), 1):
        if not line.endswith(b"\n"):
            raise GateError(f"{method}: row {row} has no terminating newline")
        observed_id = prediction_id(line, row, method)
        if observed_id != expected_id:
            raise GateError(
                f"{method}: row {row} sample_id differs from reference dataset"
            )
        if observed_id in seen:
            raise GateError(f"{method}: duplicate sample_id {observed_id!r}")
        seen.add(observed_id)
    return {
        "reference_path": str(reference),
        "candidate_path": str(candidate),
        "bytes": len(candidate_bytes),
        "sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "rows": len(lines),
        "byte_exact": True,
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_rows <= 0:
        raise GateError("--expected-rows must be positive")
    reference_dataset = require_file(
        args.reference_dataset, "reference dataset"
    )
    sample_ids = read_reference_ids(reference_dataset, args.expected_rows)

    predictions: dict[str, tuple[Path, Path]] = {}
    for method, raw_reference, raw_candidate in args.prediction:
        if method in predictions:
            raise GateError(f"duplicate prediction method {method!r}")
        predictions[method] = (
            require_file(Path(raw_reference), f"{method} frozen prediction"),
            require_file(Path(raw_candidate), f"{method} smoke prediction"),
        )
    if tuple(predictions) != EXPECTED_METHODS:
        raise GateError(
            "prediction methods/order differ: expected "
            f"{list(EXPECTED_METHODS)}, got {list(predictions)}"
        )

    methods = {
        method: verify_prediction(method, reference, candidate, sample_ids)
        for method, (reference, candidate) in predictions.items()
    }
    result = {
        "schema": SCHEMA,
        "expected_rows": args.expected_rows,
        "reference_dataset": {
            "path": str(reference_dataset),
            "bytes": reference_dataset.stat().st_size,
            "sha256": sha256(reference_dataset),
            "records": len(sample_ids),
        },
        "methods": methods,
        "formal_full_test_prediction_authorized": True,
    }
    atomic_json(args.output.resolve(), result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dataset", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=500)
    parser.add_argument(
        "--prediction",
        action="append",
        nargs=3,
        metavar=("METHOD", "REFERENCE", "CANDIDATE"),
        default=[],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        result = run(parse_args(argv))
    except (GateError, OSError) as error:
        print(f"smoke gate error: {error}", file=os.sys.stderr)
        return 2
    print(
        f"authorized full-test prediction: {len(result['methods'])} methods, "
        f"{result['expected_rows']} exact reference rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
