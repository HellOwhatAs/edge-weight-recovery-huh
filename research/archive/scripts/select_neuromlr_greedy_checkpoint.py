#!/usr/bin/env python3
"""Select a NeuroMLR checkpoint using Greedy raw roads and the common evaluator."""

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


CHECKPOINT = re.compile(r"checkpoint-epoch-(\d+)\.pt$")
METRIC_KEYS = ["edge_precision", "edge_recall", "edge_f1", "exact_match", "edge_jaccard"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--common-evaluator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.manifest.open() as source:
        first = json.loads(source.readline())
    if first["manifest_id"].startswith("test:"):
        raise SystemExit("checkpoint selection refuses a test manifest")

    checkpoints = []
    for path in args.run_dir.glob("checkpoint-epoch-*.pt"):
        match = CHECKPOINT.match(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort()
    if not checkpoints:
        raise SystemExit(f"no NeuroMLR checkpoints in {args.run_dir}")

    evaluation_dir = args.run_dir / "greedy_common_validation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for epoch, checkpoint in checkpoints:
        predictions = evaluation_dir / f"predictions-epoch-{epoch}.jsonl"
        native_summary = evaluation_dir / f"native-summary-epoch-{epoch}.json"
        common_summary = evaluation_dir / f"common-summary-epoch-{epoch}.json"
        subprocess.run(
            [
                str(args.python),
                str(args.driver),
                "predict",
                "--upstream-dir",
                str(args.upstream_dir),
                "--map-dir",
                str(args.map_dir),
                "--checkpoint",
                str(checkpoint),
                "--manifest",
                str(args.manifest),
                "--method",
                "greedy",
                "--predictions",
                str(predictions),
                "--summary",
                str(native_summary),
                "--seed",
                str(args.seed),
                "--device",
                args.device,
            ],
            check=True,
        )
        subprocess.run(
            [
                str(args.common_evaluator),
                "--predictions",
                str(predictions),
                "--output",
                str(common_summary),
            ],
            check=True,
        )
        native = json.loads(native_summary.read_text())
        common = json.loads(common_summary.read_text())
        if native["test_read"] or common["test_read"]:
            raise RuntimeError("validation checkpoint selection unexpectedly read test")
        for key in METRIC_KEYS:
            difference = abs(float(native["metrics"][key]) - float(common["metrics"][key]))
            if difference > 1e-12:
                raise RuntimeError(f"Python and common {key} differ by {difference}")
        rows.append(
            {
                "epoch": epoch,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "predictions": str(predictions),
                "predictions_sha256": sha256(predictions),
                "native_summary": str(native_summary),
                "common_evaluation": str(common_summary),
                "metrics": common["metrics"],
                "endpoint_mismatches": common["endpoint_mismatches"],
                "manifest_id_order_sha256": common["manifest_id_order_sha256"],
            }
        )

    selected = max(
        rows,
        key=lambda row: (
            row["metrics"]["edge_f1"],
            row["metrics"]["exact_match"],
            -row["epoch"],
        ),
    )
    output = {
        "schema_version": 1,
        "method": "neuromlr_greedy",
        "query_protocol": "true_first_edge_to_true_last_edge_complete_sequence",
        "selection_rule": ["maximum_edge_f1", "maximum_exact_match", "earliest_epoch"],
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "selected": selected,
        "evaluations": rows,
        "seed": args.seed,
        "device": args.device,
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
