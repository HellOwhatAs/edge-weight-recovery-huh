#!/usr/bin/env python3
"""Generate every scale-study subset sequentially and attach SHA-256 provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--aggregate-manifest", required=True, type=Path)
    parser.add_argument(
        "--binary", default=Path("target/release/examples/generate_subsets"), type=Path
    )
    parser.add_argument("--city", default="beijing")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path, cache: dict[Path, str]) -> str:
    resolved = path.resolve()
    if resolved in cache:
        return cache[resolved]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    cache[resolved] = digest.hexdigest()
    return cache[resolved]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def manifest_is_valid(
    manifest_path: Path, root: Path, hash_cache: dict[Path, str]
) -> bool:
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = root / manifest["source"]["path"]
        output = root / manifest["output"]["path"]
        return (
            source.exists()
            and output.exists()
            and source.stat().st_size == manifest["source"]["file_bytes"]
            and output.stat().st_size == manifest["output"]["file_bytes"]
            and sha256(source, hash_cache) == manifest["source"].get("sha256")
            and sha256(output, hash_cache) == manifest["output"].get("sha256")
        )
    except (KeyError, OSError, ValueError, TypeError):
        return False


def main() -> int:
    args = arguments()
    if args.timeout_seconds > 900:
        raise ValueError("per-subset timeout must not exceed 900 seconds")
    root = Path(__file__).resolve().parents[1]
    with args.plan.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    hash_cache: dict[Path, str] = {}
    manifest_index: list[dict[str, Any]] = []

    for row in rows:
        variant = row["output_variant"]
        manifest_path = args.manifest_dir / f"{row['split']}_{variant}.json"
        if not args.force and manifest_is_valid(manifest_path, root, hash_cache):
            print(f"SUBSET variant={variant} status=cached", flush=True)
        else:
            command = [
                str(args.binary.resolve()),
                "--city",
                args.city,
                "--split",
                row["split"],
                "--source-variant",
                row["source_variant"],
                "--output-variant",
                variant,
                "--sample-count",
                row["sample_count"],
                "--seed",
                row["seed"],
                "--manifest",
                str(manifest_path.resolve()),
            ]
            started = time.monotonic()
            completed = subprocess.run(
                command,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                print(completed.stdout, file=sys.stderr)
                return completed.returncode
            print(
                f"SUBSET variant={variant} status=generated "
                f"wall_seconds={time.monotonic() - started:.3f}",
                flush=True,
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_path = root / manifest["source"]["path"]
        output_path = root / manifest["output"]["path"]
        manifest["source"]["sha256"] = sha256(source_path, hash_cache)
        manifest["output"]["sha256"] = sha256(output_path, hash_cache)
        manifest["purpose"] = row["purpose"]
        atomic_json(manifest_path, manifest)
        manifest_index.append(
            {
                "manifest_path": str(manifest_path),
                "split": manifest["split"],
                "output_variant": manifest["output_variant"],
                "sample_count": manifest["sample_count"],
                "seed": manifest["seed"],
                "source": manifest["source"],
                "output": manifest["output"],
            }
        )

    aggregate = {
        "schema_version": 1,
        "city": args.city,
        "plan": str(args.plan),
        "generator": "examples/generate_subsets.rs",
        "subsets": manifest_index,
    }
    atomic_json(args.aggregate_manifest, aggregate)
    print(
        f"SUMMARY subsets={len(manifest_index)} manifest={args.aggregate_manifest}", flush=True
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
