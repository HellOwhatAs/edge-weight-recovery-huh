#!/usr/bin/env python3
"""Create the hash-bound one-time test unlock after validation is frozen."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--validation-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text())
    evidence = json.loads(args.validation_evidence.read_text())
    if protocol.get("status") != "frozen_after_validation":
        raise SystemExit("protocol is not frozen_after_validation")
    true_test_read_paths = find_true_test_read(evidence)
    if true_test_read_paths:
        raise SystemExit(f"validation evidence contains test_read=true at {true_test_read_paths}")
    output = {
        "schema_version": 1,
        "status": "test_unlocked",
        "protocol": str(args.protocol),
        "protocol_sha256": sha256(args.protocol),
        "validation_evidence": {
            "path": str(args.validation_evidence),
            "sha256": sha256(args.validation_evidence),
        },
        "authorized_split": "test",
        "maximum_manifest_decodes": 1,
        "receipt_required": True,
    }
    atomic_json(args.output, output)
    print(json.dumps(output, indent=2))


def find_true_test_read(value, path="") -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}/{key}"
            if key == "test_read" and child is True:
                found.append(child_path)
            found.extend(find_true_test_read(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(find_true_test_read(child, f"{path}/{index}"))
    return found


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
