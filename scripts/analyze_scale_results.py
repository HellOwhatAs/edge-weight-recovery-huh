#!/usr/bin/env python3
"""Aggregate the fixed scale-study protocol into machine-readable tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


METRICS = [
    "train_accepted",
    "validation_relative_regret",
    "validation_mean_regret",
    "validation_edge_f1",
    "validation_exact_match",
    "relative_regret_gap",
    "avg_epoch_ms",
    "avg_train_oracle_ms",
    "avg_update_customization_ms",
    "avg_update_changed_pct",
    "best_peak_rss_kib",
    "best_best_q_min",
    "best_q_p05",
    "best_q_median",
    "best_q_p95",
    "best_best_q_max",
    "best_best_q_at_min",
    "best_best_q_at_max",
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--coverage", required=True, type=Path)
    parser.add_argument("--coverage-audit", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--scale-csv", required=True, type=Path)
    parser.add_argument("--grid-csv", required=True, type=Path)
    return parser.parse_args()


def numeric(values: list[dict[str, Any]], key: str) -> list[float]:
    result = []
    for value in values:
        candidate = value.get(key)
        if isinstance(candidate, (int, float)) and math.isfinite(candidate):
            result.append(float(candidate))
    return result


def describe(values: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    samples = numeric(values, key)
    if not samples:
        return None
    return {
        "mean": statistics.mean(samples),
        "sample_std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "values": samples,
    }


def config_summary(
    label: str, scale: str, eta0: str, lambda_value: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "config": label,
        "scale": scale,
        "eta0": float(eta0),
        "lambda": float(lambda_value),
        "seeds": [int(row["seed"]) for row in rows],
        "run_ids": [row["run_id"] for row in rows],
        "metrics": {
            key: summary
            for key in METRICS
            if (summary := describe(rows, key)) is not None
        },
    }


def flatten_scale(summary: dict[str, Any], coverage: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "config": summary["config"],
        "scale": summary["scale"],
        "eta0": summary["eta0"],
        "lambda": summary["lambda"],
        "seeds": ";".join(map(str, summary["seeds"])),
    }
    for key, value in summary["metrics"].items():
        row[f"{key}_mean"] = value["mean"]
        row[f"{key}_std"] = value["sample_std"]
    for key, value in coverage.items():
        row[f"coverage_{key}_mean"] = value["mean"]
        row[f"coverage_{key}_std"] = value["sample_std"]
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = arguments()
    runs = json.loads(args.results.read_text(encoding="utf-8"))["runs"]
    by_id = {row["run_id"]: row for row in runs}
    failed = [row["run_id"] for row in runs if row.get("status") != "ok"]
    if failed:
        raise ValueError(f"cannot aggregate failed runs: {failed}")

    grid = sorted(
        (row for row in runs if row.get("phase") == "grid_1pct"),
        key=lambda row: (
            row["validation_relative_regret"],
            -row["validation_edge_f1"],
        ),
    )
    grid_rows = []
    baseline_relative = by_id["baseline_q1"]["validation_relative_regret"]
    for rank, row in enumerate(grid, 1):
        grid_rows.append(
            {
                "rank": rank,
                "run_id": row["run_id"],
                "eta0": row["eta0"],
                "lambda": row["lambda"],
                "validation_relative_regret": row["validation_relative_regret"],
                "relative_improvement_vs_baseline": 1.0
                - row["validation_relative_regret"] / baseline_relative,
                "validation_edge_f1": row["validation_edge_f1"],
                "validation_exact_match": row["validation_exact_match"],
                "best_epoch": row["best_best_epoch"],
                "q_min": row["best_best_q_min"],
                "q_max": row["best_best_q_max"],
                "q_at_min": row["best_best_q_at_min"],
                "q_at_max": row["best_best_q_at_max"],
                "peak_rss_kib": row["best_peak_rss_kib"],
            }
        )
    write_csv(args.grid_csv, grid_rows)

    ids = {
        ("1pct", "lambda_1e5"): [
            "grid_e1m4_l1e5",
            "top_e1m4_l1e5_s43",
            "top_e1m4_l1e5_s44",
        ],
        ("1pct", "lambda_1e7"): [
            "grid_e1m4_l1e7",
            "top_e1m4_l1e7_s43",
            "top_e1m4_l1e7_s44",
        ],
    }
    for scale in ("5pct", "10pct"):
        for label in ("lambda_1e5", "lambda_1e7"):
            suffix = label.removeprefix("lambda_")
            ids[(scale, label)] = [
                f"scale_{scale}_e1m4_l{suffix}_s{seed}" for seed in (42, 43, 44)
            ]

    summaries = []
    for (scale, label), run_ids in ids.items():
        lambda_value = label.removeprefix("lambda_")
        summaries.append(
            config_summary(
                f"eta_1e-4_{label}",
                scale,
                "1e-4",
                lambda_value,
                [by_id[run_id] for run_id in run_ids],
            )
        )

    coverage_raw = json.loads(args.coverage.read_text(encoding="utf-8"))["scales"]
    pattern = re.compile(r"scale_(1pct|5pct|10pct)_seed(42|43|44)$")
    grouped_coverage: dict[str, list[dict[str, Any]]] = {}
    for row in coverage_raw:
        match = pattern.fullmatch(row["train_variant"])
        if match:
            grouped_coverage.setdefault(match.group(1), []).append(row)
    coverage_summary: dict[str, dict[str, Any]] = {}
    for scale, rows in grouped_coverage.items():
        remapped = [
            {
                "valid_routes": row["valid_routes"],
                "graph_edge_coverage": row["graph_edge_coverage"],
                "validation_routes_with_unseen_edge_rate": row["validation"][
                    "routes_with_unseen_edge_rate"
                ],
                "validation_unseen_edge_occurrence_rate": row["validation"][
                    "unseen_edge_occurrence_rate"
                ],
            }
            for row in rows
        ]
        coverage_summary[scale] = {
            key: description
            for key in remapped[0]
            if (description := describe(remapped, key)) is not None
        }

    full_coverage_row = next(
        row
        for row in json.loads(args.coverage_audit.read_text(encoding="utf-8"))["scales"]
        if row["train_variant"] == "all"
    )
    full_coverage_values = [
        {
            "valid_routes": full_coverage_row["valid_routes"],
            "graph_edge_coverage": full_coverage_row["graph_edge_coverage"],
            "validation_routes_with_unseen_edge_rate": full_coverage_row["validation"][
                "routes_with_unseen_edge_rate"
            ],
            "validation_unseen_edge_occurrence_rate": full_coverage_row["validation"][
                "unseen_edge_occurrence_rate"
            ],
        }
    ]
    coverage_summary["full"] = {
        key: description
        for key in full_coverage_values[0]
        if (description := describe(full_coverage_values, key)) is not None
    }

    full_summary = config_summary(
        "eta_1e-4_lambda_1e5",
        "full",
        "1e-4",
        "1e5",
        [by_id["scale_full_e1m4_l1e5"]],
    )

    scale_rows = [
        flatten_scale(summary, coverage_summary[summary["scale"]])
        for summary in [*summaries, full_summary]
    ]
    write_csv(args.scale_csv, scale_rows)

    top3_ids = {
        "eta_1e-4_lambda_1e5": [
            "grid_e1m4_l1e5",
            "top_e1m4_l1e5_s43",
            "top_e1m4_l1e5_s44",
        ],
        "eta_1e-4_lambda_1e7": [
            "grid_e1m4_l1e7",
            "top_e1m4_l1e7_s43",
            "top_e1m4_l1e7_s44",
        ],
        "eta_3e-5_lambda_1e5": [
            "grid_e3m5_l1e5",
            "top_e3m5_l1e5_s43",
            "top_e3m5_l1e5_s44",
        ],
    }
    top3 = [
        config_summary(label, "1pct", label.split("_")[1], label.split("_")[3], [by_id[item] for item in run_ids])
        for label, run_ids in top3_ids.items()
    ]
    best_run = min(
        (
            row
            for row in runs
            if row.get("phase") in {"scale_curve", "full_exploratory"}
        ),
        key=lambda row: row["validation_relative_regret"],
    )
    aggregate = {
        "schema_version": 1,
        "selection_policy": "validation relative regret primary, validation edge F1 secondary; reject severe saturation/nonfinite runs",
        "test_policy": "no model test evaluation was run",
        "baseline": by_id["baseline_q1"],
        "grid_ranking": grid_rows,
        "top3_one_percent_three_seed": top3,
        "scale_curve_top2_three_seed": summaries,
        "full_exploratory": full_summary,
        "coverage_three_seed": coverage_summary,
        "legacy_adam_ablation": by_id["legacy_adam_s42"],
        "best_validation_run": best_run,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
