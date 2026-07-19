#!/usr/bin/env python3
"""Export an aligned NeuroMLR pickle into the method-neutral JSONL protocol."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any


DATASET_MANIFEST_SCHEMA = "ewr.dataset-manifest/v1"
DATASET_RECORD_SCHEMA = "ewr.dataset-record/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pickle", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--network-id", required=True)
    return parser.parse_args()


def validate_trip(value: Any, index: int) -> tuple[str, list[int]]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"trip {index} is not a three-field tuple")
    sample_id, edges, timestamps = value
    if not isinstance(sample_id, str) or not sample_id or any(
        ord(character) < 32 or ord(character) == 127 for character in sample_id
    ):
        raise ValueError(f"trip {index} has an invalid sample ID")
    if not isinstance(edges, list) or len(edges) < 2:
        raise ValueError(f"trip {index} has fewer than two edges")
    if any(
        isinstance(edge, bool)
        or not isinstance(edge, int)
        or edge < 0
        or edge > 0xFFFF_FFFF
        for edge in edges
    ):
        raise ValueError(f"trip {index} has an invalid raw edge ID")
    if (
        not isinstance(timestamps, tuple)
        or len(timestamps) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in timestamps)
    ):
        raise ValueError(f"trip {index} has invalid whole-trip timestamps")
    return sample_id, edges


def atomic_text_writer(destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    return os.fdopen(descriptor, "w", encoding="utf-8"), Path(temporary_name)


def export(args: argparse.Namespace) -> int:
    with args.input_pickle.open("rb") as source:
        trips = pickle.load(source)
    if not isinstance(trips, list) or not trips:
        raise ValueError("input pickle must contain a nonempty trip list")

    writer, temporary = atomic_text_writer(args.output_jsonl)
    sample_ids: set[str] = set()
    try:
        with writer:
            for index, value in enumerate(trips):
                sample_id, edges = validate_trip(value, index)
                if sample_id in sample_ids:
                    raise ValueError(f"duplicate sample ID {sample_id!r}")
                sample_ids.add(sample_id)
                row = {"sample_id": sample_id, "original_edge_ids": edges}
                writer.write(json.dumps(row, separators=(",", ":")) + "\n")
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, args.output_jsonl)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    manifest = {
        "schema": DATASET_MANIFEST_SCHEMA,
        "dataset_id": args.dataset_id,
        "network_id": args.network_id,
        "records_schema": DATASET_RECORD_SCHEMA,
        "records_file": os.path.relpath(args.output_jsonl, args.manifest.parent),
    }
    if Path(manifest["records_file"]).is_absolute() or ".." in Path(
        manifest["records_file"]
    ).parts:
        raise ValueError("output JSONL must be at or below the manifest directory")

    writer, temporary = atomic_text_writer(args.manifest)
    try:
        with writer:
            json.dump(manifest, writer, indent=2)
            writer.write("\n")
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, args.manifest)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return len(trips)


def main() -> None:
    args = parse_args()
    count = export(args)
    print(f"exported {count} records to {args.output_jsonl}")


if __name__ == "__main__":
    main()
