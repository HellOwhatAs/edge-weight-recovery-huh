#!/usr/bin/env python3
"""Aggregate frozen CCH/Dijkstra training and inference benchmark records."""

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-run", type=Path, action="append", required=True)
    parser.add_argument("--training-memory-cch", type=Path, required=True)
    parser.add_argument("--training-memory-dijkstra", type=Path, required=True)
    parser.add_argument("--inference-cch", type=Path, required=True)
    parser.add_argument("--inference-dijkstra", type=Path, required=True)
    parser.add_argument("--inference-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.training_run) != 4:
        raise SystemExit("the frozen protocol requires exactly four training repetitions")
    training = [load(path) for path in args.training_run]
    expected_orders = [
        ["cch", "dijkstra"],
        ["dijkstra", "cch"],
        ["cch", "dijkstra"],
        ["dijkstra", "cch"],
    ]
    for index, (record, order) in enumerate(zip(training, expected_orders), 1):
        if record["oracle_run_order"] != order:
            raise RuntimeError(f"training repetition {index} has wrong oracle order")
        if record["test_read"]:
            raise RuntimeError(f"training repetition {index} read test")
        if {run["oracle"] for run in record["runs"]} != {"cch", "dijkstra"}:
            raise RuntimeError(f"training repetition {index} lacks both oracle runs")
        consistency = record["consistency"]
        for key in [
            "all_distance_sums_equal",
            "all_predicted_counts_equal",
            "final_weights_bitwise_equal",
        ]:
            if consistency[key] is not True:
                raise RuntimeError(f"training repetition {index} failed {key}")
        if consistency["different_final_weights"] != 0:
            raise RuntimeError(f"training repetition {index} has different final weights")
        if any(
            state["objective_abs_difference"] != 0.0
            for state in consistency["states"]
        ):
            raise RuntimeError(f"training repetition {index} has different objectives")
    require_identical(training, "invariants")
    require_identical(training, "workload")
    require_identical(training, "candidate_samples")
    require_identical(training, "candidate_unique_od")

    per_oracle = {"cch": [], "dijkstra": []}
    for record in training:
        for run in record["runs"]:
            per_oracle[run["oracle"]].append(run)
    training_summary = {}
    for oracle, runs in per_oracle.items():
        setup_key = (
            "cch_topology_preprocessing_seconds"
            if oracle == "cch"
            else "dijkstra_adjacency_setup_seconds"
        )
        setup = [float(row["oracle_setup"][setup_key]) for row in training]
        core = [float(run["timing_totals"]["core_end_to_end_seconds"]) for run in runs]
        end_to_end = [left + right for left, right in zip(setup, core)]
        training_summary[oracle] = {
            "setup_seconds": samples(setup),
            "core_seconds": samples(core),
            "setup_plus_core_seconds": samples(end_to_end),
            "customization_seconds": samples(
                [float(run["timing_totals"]["customization_seconds"]) for run in runs]
            ),
            "query_seconds": samples(
                [float(run["timing_totals"]["query_seconds"]) for run in runs]
            ),
            "optimizer_seconds": samples(
                [float(run["timing_totals"]["optimizer_seconds"]) for run in runs]
            ),
            "peak_rss_kib_at_run_end": [int(run["peak_rss_kib"]) for run in runs],
            "final_weight_sha256": [run["final_weight_sha256"] for run in runs],
        }
    training_speedup = (
        training_summary["dijkstra"]["setup_plus_core_seconds"]["mean"]
        / training_summary["cch"]["setup_plus_core_seconds"]["mean"]
    )
    memory_records = {
        "cch": load(args.training_memory_cch),
        "dijkstra": load(args.training_memory_dijkstra),
    }
    for oracle, record in memory_records.items():
        if record["test_read"] or len(record["runs"]) != 1:
            raise RuntimeError(f"isolated {oracle} memory record is invalid")
        if record["runs"][0]["oracle"] != oracle:
            raise RuntimeError(f"isolated {oracle} memory record has the wrong oracle")
        if record["workload"] != training[0]["workload"]:
            raise RuntimeError(f"isolated {oracle} memory workload differs")
        training_summary[oracle]["isolated_process_peak_rss_kib"] = int(
            record["peak_rss_kib"]
        )

    cch = load(args.inference_cch)
    dijkstra = load(args.inference_dijkstra)
    audit = load(args.inference_audit)
    for key in ["checkpoint", "manifest", "manifest_records", "query_protocol", "threads"]:
        if cch[key] != dijkstra[key]:
            raise RuntimeError(f"inference summaries differ on {key}")
    if cch["query_protocol"] != "node_to_node" or cch["threads"] != 1:
        raise RuntimeError("inference workload is not frozen one-thread node-to-node")
    if cch["warmup_repetitions"] != 1 or dijkstra["warmup_repetitions"] != 1:
        raise RuntimeError("inference summaries lack the frozen warm-up")
    if cch["measured_repetitions"] != 5 or dijkstra["measured_repetitions"] != 5:
        raise RuntimeError("inference summaries lack five measured repetitions")
    if cch["quantized_weight_sha256"] != dijkstra["quantized_weight_sha256"]:
        raise RuntimeError("inference oracles used different integer metrics")
    if not cch["test_read"] or not dijkstra["test_read"] or not audit["test_read"]:
        raise RuntimeError("final inference records are not bound to the common test manifest")

    cch_query = float(cch["timing"]["mean_total_query_seconds"])
    dijkstra_query = float(dijkstra["timing"]["mean_total_query_seconds"])
    inference_summary = {
        "workload": {
            "checkpoint": cch["checkpoint"],
            "manifest": cch["manifest"],
            "queries": cch["manifest_records"],
            "threads": cch["threads"],
            "warmup_repetitions": cch["warmup_repetitions"],
            "measured_repetitions": cch["measured_repetitions"],
            "quantized_weight_sha256": cch["quantized_weight_sha256"],
        },
        "cch": inference_method(cch),
        "dijkstra": inference_method(dijkstra),
        "query_only_speedup_dijkstra_over_cch": dijkstra_query / cch_query,
        "consistency": audit,
    }

    output = {
        "schema_version": 1,
        "training": {
            "repetitions": len(training),
            "workload": training[0]["workload"],
            "invariants": training[0]["invariants"],
            "cch": training_summary["cch"],
            "dijkstra": training_summary["dijkstra"],
            "speedup_dijkstra_over_cch_setup_plus_core": training_speedup,
            "consistency_by_repetition": [row["consistency"] for row in training],
            "process_peak_rss_kib": [int(row["peak_rss_kib"]) for row in training],
        },
        "inference": inference_summary,
        "inputs": {
            "training_runs": [file_identity(path) for path in args.training_run],
            "training_memory_cch": file_identity(args.training_memory_cch),
            "training_memory_dijkstra": file_identity(args.training_memory_dijkstra),
            "inference_cch": file_identity(args.inference_cch),
            "inference_dijkstra": file_identity(args.inference_dijkstra),
            "inference_audit": file_identity(args.inference_audit),
        },
        "test_read": True,
    }
    atomic_json(args.output, output)
    print(json.dumps({"training_speedup": training_speedup, "inference_speedup": dijkstra_query / cch_query}, indent=2))


def inference_method(record: dict) -> dict:
    timing = record["timing"]
    return {
        "query_repetition_seconds": timing["query_repetition_seconds"],
        "mean_total_query_seconds": timing["mean_total_query_seconds"],
        "mean_query_latency_seconds": timing["mean_query_latency_seconds"],
        "mean_throughput_queries_per_second": timing["mean_throughput_queries_per_second"],
        "topology_preprocessing_seconds": timing["cch_topology_preprocessing_seconds"],
        "dijkstra_adjacency_setup_seconds": timing["dijkstra_adjacency_setup_seconds"],
        "customization_seconds": timing["customization_seconds"],
        "peak_rss_kib": record["peak_rss_kib"],
        "route_checksum_sha256": record["route_checksum_sha256"],
    }


def samples(values: list[float]) -> dict:
    return {
        "values": values,
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "population_stdev": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def require_identical(records: list[dict], key: str) -> None:
    encoded = [json.dumps(record[key], sort_keys=True, separators=(",", ":")) for record in records]
    if len(set(encoded)) != 1:
        raise RuntimeError(f"training repetitions differ on {key}")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def file_identity(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


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
