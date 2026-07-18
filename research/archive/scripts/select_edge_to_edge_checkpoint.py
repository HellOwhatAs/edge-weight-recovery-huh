#!/usr/bin/env python3
"""Select a project checkpoint only on frozen edge-to-edge validation routes."""

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


CHECKPOINT = re.compile(r"checkpoint-(\d+)\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads <= 0:
        raise SystemExit("--threads must be positive")
    first = json.loads(args.manifest.open().readline())
    if first["manifest_id"].startswith("test:"):
        raise SystemExit("checkpoint selection refuses a test manifest")
    checkpoints = []
    for path in args.run_dir.glob("checkpoint-*.json"):
        match = CHECKPOINT.match(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort()
    if not checkpoints:
        raise SystemExit(f"no numbered checkpoints in {args.run_dir}")

    evaluation_dir = args.run_dir / "edge_to_edge_validation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["RAYON_NUM_THREADS"] = str(args.threads)
    rows = []
    for update, checkpoint in checkpoints:
        predictions = evaluation_dir / f"predictions-{update}.jsonl"
        summary = evaluation_dir / f"evaluation-{update}.json"
        command = [
            str(args.benchmark_binary),
            "--checkpoint",
            str(checkpoint),
            "--manifest",
            str(args.manifest),
            "--predictions",
            str(predictions),
            "--summary",
            str(summary),
            "--oracle",
            "cch",
            "--query-protocol",
            "edge_to_edge",
            "--threads",
            str(args.threads),
            "--warmup-repetitions",
            "0",
            "--measured-repetitions",
            "1",
        ]
        subprocess.run(command, check=True, env=environment)
        result = json.loads(summary.read_text())
        if result["test_read"]:
            raise RuntimeError("validation evaluator unexpectedly marked test_read")
        rows.append(
            {
                "update": update,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "evaluation": str(summary),
                "predictions": str(predictions),
                "metrics": result["metrics"],
            }
        )
    selected = max(
        rows,
        key=lambda row: (
            row["metrics"]["edge_f1"],
            row["metrics"]["exact_match"],
            -row["update"],
        ),
    )
    output = {
        "schema_version": 1,
        "query_protocol": "true_first_edge_to_true_last_edge_complete_sequence",
        "selection_rule": ["maximum_edge_f1", "maximum_exact_match", "earliest_update"],
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "threads": args.threads,
        "selected": selected,
        "evaluations": rows,
        "test_read": False,
    }
    atomic_json(args.output, output)
    print(json.dumps(selected, indent=2))


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
