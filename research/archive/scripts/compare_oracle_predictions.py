#!/usr/bin/env python3
"""Audit aligned CCH and Dijkstra raw-road predictions and tie differences."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cch-predictions", type=Path, required=True)
    parser.add_argument("--dijkstra-predictions", type=Path, required=True)
    parser.add_argument("--cch-summary", type=Path, required=True)
    parser.add_argument("--dijkstra-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open() as source:
        rows = [json.loads(line) for line in source]
    if not rows:
        raise RuntimeError(f"empty prediction file {path}")
    return rows


def main() -> None:
    args = parse_args()
    cch = load_rows(args.cch_predictions)
    dijkstra = load_rows(args.dijkstra_predictions)
    if len(cch) != len(dijkstra):
        raise RuntimeError("oracle prediction counts differ")
    cch_summary = json.loads(args.cch_summary.read_text())
    dijkstra_summary = json.loads(args.dijkstra_summary.read_text())
    if cch_summary["quantized_weight_sha256"] != dijkstra_summary["quantized_weight_sha256"]:
        raise RuntimeError("oracles did not receive the same quantized weight vector")

    distance_mismatches = []
    equal_distance_path_differences = []
    exact_matches = 0
    for left, right in zip(cch, dijkstra):
        if left["manifest_id"] != right["manifest_id"]:
            raise RuntimeError("oracle prediction rows are not identically aligned")
        if left["observed_edges"] != right["observed_edges"]:
            raise RuntimeError("oracle prediction truths differ")
        identifier = left["manifest_id"]
        if left["distance_u32"] != right["distance_u32"]:
            distance_mismatches.append(identifier)
        elif left["predicted_edges"] != right["predicted_edges"]:
            equal_distance_path_differences.append(identifier)
        else:
            exact_matches += 1
    output = {
        "schema_version": 1,
        "cch_predictions": str(args.cch_predictions),
        "dijkstra_predictions": str(args.dijkstra_predictions),
        "records": len(cch),
        "quantized_weight_sha256": cch_summary["quantized_weight_sha256"],
        "distance_mismatches": len(distance_mismatches),
        "distance_mismatch_manifest_ids": distance_mismatches,
        "identical_routes": exact_matches,
        "equal_distance_path_differences": len(equal_distance_path_differences),
        "tie_path_difference_manifest_ids": equal_distance_path_differences,
        "cch_route_checksum": cch_summary["route_checksum_sha256"],
        "dijkstra_route_checksum": dijkstra_summary["route_checksum_sha256"],
        "input_sha256": {
            "cch_predictions": sha256(args.cch_predictions),
            "dijkstra_predictions": sha256(args.dijkstra_predictions),
            "cch_summary": sha256(args.cch_summary),
            "dijkstra_summary": sha256(args.dijkstra_summary),
        },
        "test_read": bool(cch_summary["test_read"] or dijkstra_summary["test_read"]),
    }
    atomic_json(args.output, output)
    print(json.dumps(output, indent=2))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
