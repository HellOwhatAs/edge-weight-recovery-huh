#!/usr/bin/env python3
"""Build one uniform full-test operational inference-efficiency artifact.

The primary wall time, throughput, and peak RSS come from GNU time around the
entire prediction task.  Adapter diagnostics are retained only as decomposition
evidence (and as the source of peak CUDA allocation), so direct and sharded
methods share one end-to-end outer boundary without hiding shard overhead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


SCHEMA = "ewr.full-test-operational-efficiency/v1"
EXPECTED_METHODS = (
    "project",
    "sp_length",
    "markov_sp",
    "neuromlr_greedy",
    "drncs_lg",
    "drpk_static",
    "drp_tp",
)
SHARDED_SCHEMA = "ewr.sharded-quality-prediction-diagnostics/v1"
TASK_TIME_EVIDENCE_SCHEMA = "ewr.route-baseline-task-time-evidence/v1"


class OperationalError(ValueError):
    """A time report or prediction diagnostic violates the full-test contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, context: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise OperationalError(f"missing {context}: {resolved}")
    return resolved


def load_object(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OperationalError(f"cannot read {context} {path}: {error}") from error
    if not isinstance(value, dict):
        raise OperationalError(f"{context} must be a JSON object")
    return value


def number(value: Any, context: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OperationalError(f"{context} must be numeric")
    result = float(value)
    if result < 0 or (positive and result <= 0):
        raise OperationalError(f"{context} has an invalid value")
    return result


def integer(value: Any, context: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OperationalError(f"{context} must be an integer")
    if value < 0 or (positive and value <= 0):
        raise OperationalError(f"{context} has an invalid value")
    return value


def parse_elapsed(raw: str) -> float:
    value = raw.strip()
    if not re.fullmatch(r"(?:\d+:){1,2}\d+(?:\.\d+)?", value):
        raise OperationalError(f"invalid GNU time elapsed value {value!r}")
    parts = [float(component) for component in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        result = 60 * minutes + seconds
    else:
        hours, minutes, seconds = parts
        result = 3600 * hours + 60 * minutes + seconds
    if result <= 0:
        raise OperationalError("GNU time elapsed wall time must be positive")
    return result


def parse_time_report_text(text: str, context: str) -> dict[str, Any]:
    fields: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        for label, key in (
            ("User time (seconds):", "user_seconds"),
            ("System time (seconds):", "system_seconds"),
            ("Maximum resident set size (kbytes):", "peak_rss_kib"),
            ("Exit status:", "exit_status"),
        ):
            if line.startswith(label):
                fields[key] = line[len(label) :].strip()
        if line.startswith("Elapsed (wall clock) time"):
            match = re.search(r":\s*([0-9:.]+)\s*$", line)
            if match:
                fields["elapsed"] = match.group(1)
    missing = {
        "user_seconds", "system_seconds", "peak_rss_kib", "exit_status", "elapsed"
    } - fields.keys()
    if missing:
        raise OperationalError(f"GNU time report is missing {sorted(missing)}: {context}")
    try:
        user = float(fields["user_seconds"])
        system = float(fields["system_seconds"])
        peak_rss = int(fields["peak_rss_kib"])
        exit_status = int(fields["exit_status"])
    except ValueError as error:
        raise OperationalError(f"GNU time report has an invalid number: {context}") from error
    if user < 0 or system < 0 or peak_rss <= 0:
        raise OperationalError(f"GNU time report contains negative/empty resources: {context}")
    return {
        "wall_seconds": parse_elapsed(fields["elapsed"]),
        "user_seconds": user,
        "system_seconds": system,
        "peak_rss_kib": peak_rss,
        "exit_status": exit_status,
    }


def parse_time_evidence(path: Path) -> dict[str, Any]:
    value = load_object(path, "task time evidence")
    expected = {
        "schema", "task_id", "attempt_count", "attempts", "aggregation_rule",
        "created_at", "timing_complete", "lost_attempts",
    }
    if set(value) != expected or value.get("schema") != TASK_TIME_EVIDENCE_SCHEMA:
        raise OperationalError(f"unsupported task time evidence schema/shape: {path}")
    attempts = value["attempts"]
    count = integer(value["attempt_count"], "time evidence attempt_count", positive=True)
    if not isinstance(attempts, list) or len(attempts) != count:
        raise OperationalError("time evidence attempts length differs")
    parsed: list[dict[str, Any]] = []
    for index, attempt in enumerate(attempts, 1):
        fields = {
            "attempt", "recorded_status", "status", "timing_status",
            "return_code", "started_at", "ended_at",
            "time_report_path", "time_report_bytes", "time_report_sha256",
            "time_report_text",
        }
        if not isinstance(attempt, dict) or set(attempt) != fields:
            raise OperationalError(f"time evidence attempt {index} shape differs")
        if attempt["attempt"] != index:
            raise OperationalError(f"time evidence attempt {index} index differs")
        if attempt["timing_status"] == "lost":
            if attempt["status"] != "lost" or any(
                attempt[key] is not None
                for key in (
                    "time_report_bytes", "time_report_sha256", "time_report_text"
                )
            ):
                raise OperationalError(f"lost attempt {index} evidence is inconsistent")
            parsed.append({"lost": True})
            continue
        if attempt["timing_status"] != "complete":
            raise OperationalError(f"attempt {index} timing status is unsupported")
        text = attempt["time_report_text"]
        if not isinstance(text, str) or not text.strip():
            raise OperationalError(f"time evidence attempt {index} text is empty")
        encoded = text.encode("utf-8")
        if attempt["time_report_bytes"] != len(encoded) or attempt[
            "time_report_sha256"
        ] != hashlib.sha256(encoded).hexdigest():
            raise OperationalError(f"time evidence attempt {index} hash/size differs")
        report = parse_time_report_text(text, f"{path} attempt {index}")
        status = attempt["status"]
        return_code = attempt["return_code"]
        if status == "succeeded":
            if return_code != 0 or report["exit_status"] != 0:
                raise OperationalError(f"successful attempt {index} has nonzero exit status")
        elif status == "failed":
            if return_code == 0 or report["exit_status"] == 0:
                raise OperationalError(f"failed attempt {index} has zero exit status")
        else:
            raise OperationalError(f"time evidence attempt {index} status is not final")
        parsed.append(report)
    if attempts[-1]["status"] not in {"succeeded", "lost"}:
        raise OperationalError("final prediction attempt did not succeed")
    observed_lost = [
        index for index, report in enumerate(parsed, 1) if report.get("lost")
    ]
    if value["lost_attempts"] != observed_lost or value["timing_complete"] != (
        not observed_lost
    ):
        raise OperationalError("time evidence lost-attempt summary differs")
    complete_reports = [report for report in parsed if not report.get("lost")]
    known_wall = sum(report["wall_seconds"] for report in complete_reports)
    timing_complete = not observed_lost
    total_wall = known_wall if timing_complete else None
    final_wall = (
        parsed[-1]["wall_seconds"] if not parsed[-1].get("lost") else None
    )
    known_prior_wall = sum(
        report["wall_seconds"] for report in parsed[:-1] if not report.get("lost")
    )
    return {
        "wall_seconds": total_wall,
        "known_active_wall_lower_bound_seconds": known_wall,
        "successful_final_attempt_wall_seconds": final_wall,
        "wasted_interrupted_wall_seconds": (
            total_wall - final_wall
            if timing_complete and final_wall is not None
            else known_prior_wall
        ),
        "attempt_count": count,
        "lost_attempt_count": len(observed_lost),
        "timing_complete": timing_complete,
        "user_seconds": sum(report["user_seconds"] for report in complete_reports),
        "system_seconds": sum(report["system_seconds"] for report in complete_reports),
        "peak_rss_kib": (
            max(report["peak_rss_kib"] for report in complete_reports)
            if complete_reports
            else None
        ),
        "exit_status": (
            parsed[-1]["exit_status"] if not parsed[-1].get("lost") else None
        ),
    }


def diagnostic_decomposition(
    method: str, diagnostic: dict[str, Any], expected_samples: int
) -> dict[str, Any]:
    if diagnostic.get("samples") != expected_samples:
        raise OperationalError(f"{method}: diagnostic samples differ")
    schema = diagnostic.get("schema")
    observed_method = diagnostic.get("method")
    if method == "project":
        if schema != "ewr.project-prediction-diagnostics/v1" or observed_method not in {
            "project", "project_cch"
        }:
            raise OperationalError("project diagnostic method/schema differs")
        internal = number(
            diagnostic.get("timing", {}).get("mean_metric_and_query_seconds"),
            "project internal prediction time",
            positive=True,
        )
        device, peak_cuda = "cpu", 0
        shard_process = None
    elif method in {"sp_length", "markov_sp"}:
        if (
            schema != "ewr.static-route-baseline-prediction-diagnostics/v1"
            or observed_method != method
        ):
            raise OperationalError(f"{method}: diagnostic method/schema differs")
        internal = number(
            diagnostic.get("timing", {}).get("mean_metric_and_query_seconds"),
            f"{method} internal prediction time",
            positive=True,
        )
        device, peak_cuda = "cpu", 0
        shard_process = None
    elif method == "neuromlr_greedy":
        if schema != "ewr.neuromlr-diagnostics/v1" or observed_method != method:
            raise OperationalError("NeuroMLR-G diagnostic method/schema differs")
        execution = diagnostic.get("execution")
        if not isinstance(execution, dict) or execution.get("mode") != (
            "chunked_resumable_quality_prediction"
        ):
            raise OperationalError("NeuroMLR-G full prediction was not chunked/resumable")
        if diagnostic.get("warmup_repetitions") != 0 or diagnostic.get(
            "measured_repetitions"
        ) != 1:
            raise OperationalError("NeuroMLR-G full prediction is not a 0/1 pass")
        internal = number(
            diagnostic.get("timing", {}).get("prediction_seconds"),
            "NeuroMLR-G chunk prediction time",
            positive=True,
        )
        peak_cuda = integer(
            diagnostic.get("peak_cuda_memory_bytes"),
            "NeuroMLR-G CUDA peak",
            positive=True,
        )
        device, shard_process = "cuda:0", None
    else:
        if schema != SHARDED_SCHEMA or observed_method != method:
            raise OperationalError(f"{method}: sharded diagnostic method/schema differs")
        configuration = diagnostic.get("configuration")
        operational = diagnostic.get("operational_timing")
        if not isinstance(configuration, dict) or not isinstance(operational, dict):
            raise OperationalError(f"{method}: sharded diagnostic is incomplete")
        if configuration.get("warmup_repetitions") != 0 or configuration.get(
            "measured_repetitions"
        ) != 1:
            raise OperationalError(f"{method}: sharded prediction is not a 0/1 pass")
        internal = number(
            operational.get("sum_adapter_prediction_seconds"),
            f"{method} adapter prediction sum",
            positive=True,
        )
        shard_process = number(
            operational.get("sum_adapter_process_seconds"),
            f"{method} adapter process sum",
            positive=True,
        )
        if internal > shard_process + 1e-6:
            raise OperationalError(f"{method}: prediction sum exceeds adapter process sum")
        peak_cuda = integer(
            diagnostic.get("maximum_shard_peak_cuda_memory_bytes"),
            f"{method} CUDA peak",
        )
        device = str(configuration.get("device"))
        if method in {"drncs_lg", "drpk_static"}:
            if not device.startswith("cuda") or peak_cuda <= 0:
                raise OperationalError(f"{method}: formal prediction did not use CUDA")
        elif device != "cpu" or peak_cuda != 0:
            raise OperationalError("DRP-TP formal prediction must be CPU-only")
    return {
        "diagnostic_schema": schema,
        "device": device,
        "internal_prediction_seconds": internal,
        "shard_adapter_process_seconds": shard_process,
        "peak_cuda_memory_bytes": peak_cuda,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.samples <= 0:
        raise OperationalError("--samples must be positive")
    entries: dict[str, tuple[Path, Path]] = {}
    for method, raw_time, raw_diagnostic in args.entry:
        if method in entries:
            raise OperationalError(f"duplicate method {method!r}")
        entries[method] = (
            require_file(Path(raw_time), f"{method} GNU time report"),
            require_file(Path(raw_diagnostic), f"{method} diagnostic"),
        )
    if tuple(entries) != EXPECTED_METHODS:
        raise OperationalError(
            f"method set/order differs: {list(entries)} != {list(EXPECTED_METHODS)}"
        )
    methods: dict[str, Any] = {}
    for method, (time_path, diagnostic_path) in entries.items():
        outer = parse_time_evidence(time_path)
        diagnostic = load_object(diagnostic_path, f"{method} diagnostic")
        decomposition = diagnostic_decomposition(method, diagnostic, args.samples)
        if outer["timing_complete"] and decomposition["internal_prediction_seconds"] > outer["wall_seconds"] + 0.25:
            raise OperationalError(f"{method}: internal prediction exceeds outer wall")
        shard_process = decomposition["shard_adapter_process_seconds"]
        if (
            outer["timing_complete"]
            and shard_process is not None
            and shard_process > outer["wall_seconds"] + 0.25
        ):
            raise OperationalError(f"{method}: shard process sum exceeds outer wall")
        wall = outer["wall_seconds"]
        methods[method] = {
            "samples": args.samples,
            "outer_boundary": (
                "GNU time around the complete prediction task, including input/model "
                "loading, shard materialization/reloads where applicable, prediction, "
                "validation, and final output assembly"
            ),
            "timing_complete": outer["timing_complete"],
            "known_active_wall_lower_bound_seconds": outer[
                "known_active_wall_lower_bound_seconds"
            ],
            "wall_seconds": wall,
            "successful_final_attempt_wall_seconds": outer[
                "successful_final_attempt_wall_seconds"
            ],
            "wasted_interrupted_wall_seconds": outer[
                "wasted_interrupted_wall_seconds"
            ],
            "attempt_count": outer["attempt_count"],
            "lost_attempt_count": outer["lost_attempt_count"],
            "mean_ms_per_query": (
                1000.0 * wall / args.samples if wall is not None else None
            ),
            "queries_per_second": args.samples / wall if wall is not None else None,
            "user_seconds": outer["user_seconds"],
            "system_seconds": outer["system_seconds"],
            "peak_rss_kib": outer["peak_rss_kib"],
            "exit_status": outer["exit_status"],
            "device": decomposition["device"],
            "peak_cuda_memory_bytes": decomposition["peak_cuda_memory_bytes"],
            "internal_prediction_seconds": decomposition[
                "internal_prediction_seconds"
            ],
            "shard_adapter_process_seconds": shard_process,
            "time_evidence": {
                "path": str(time_path),
                "sha256": sha256(time_path),
            },
            "diagnostic": {
                "path": str(diagnostic_path),
                "sha256": sha256(diagnostic_path),
                "schema": decomposition["diagnostic_schema"],
            },
        }
    return {
        "schema": SCHEMA,
        "samples": args.samples,
        "methods": methods,
        "comparability_note": (
            "The outer boundary is uniform. Internal decomposition is not directly "
            "comparable because sharded adapters reload artifacts per shard."
        ),
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument(
        "--entry", action="append", nargs=3,
        metavar=("METHOD", "TASK_TIME_EVIDENCE", "DIAGNOSTIC"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        result = build(args)
        atomic_json(args.output.resolve(), result)
    except (OperationalError, OSError) as error:
        print(f"operational summary error: {error}", file=os.sys.stderr)
        return 2
    print(f"wrote uniform operational efficiency for {len(result['methods'])} methods")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
