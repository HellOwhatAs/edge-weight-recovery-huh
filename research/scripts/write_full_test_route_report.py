#!/usr/bin/env python3
"""Render the registered full-test route-baseline result as a paper-style report."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


PROTOCOL_SCHEMA = "ewr.route-baseline-full-test-protocol/v1"
SUMMARY_SCHEMA = "ewr.route-baseline-summary/v1"
CONFIDENCE_SCHEMA = "ewr.route-quality-confidence/v1"
EXPECTED_METHODS = (
    "project",
    "sp_length",
    "markov_sp",
    "neuromlr_greedy",
    "drncs_lg",
    "drpk_static",
    "drp_tp",
)


class ReportError(ValueError):
    """A report input violates the frozen full-test contract."""


def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReportError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ReportError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReportError(f"{path}: top-level value must be an object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportError(message)


def number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ReportError(f"{context} must be finite")
    return result


def integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReportError(f"{context} must be an integer")
    return value


def fmt_percent(value: Any) -> str:
    return f"{100.0 * number(value, 'metric'):.2f}"


def fmt_interval(interval: dict[str, Any]) -> str:
    return (
        f"{fmt_percent(interval['mean'])} "
        f"[{fmt_percent(interval['lower'])}, {fmt_percent(interval['upper'])}]"
    )


def fmt_seconds(value: Any) -> str:
    return "—" if value is None else f"{number(value, 'seconds'):.3f}"


def fmt_memory(kib: Any) -> str:
    return "—" if kib is None else f"{number(kib, 'memory') / 1024.0:.1f}"


def validate(
    protocol: dict[str, Any], summary: dict[str, Any], confidence: dict[str, Any]
) -> tuple[int, dict[str, dict[str, Any]], dict[str, Any]]:
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "wrong protocol schema")
    require(
        protocol.get("status") == "frozen_before_full_test",
        "protocol was not frozen before full-test execution",
    )
    included = protocol.get("methods", {}).get("included")
    require(included == list(EXPECTED_METHODS), "protocol method order differs")
    test_count = integer(
        protocol.get("data", {}).get("test", {}).get("eligible_records"),
        "protocol test eligible_records",
    )
    require(test_count == 248_233, "full test count must be 248233")
    selection = protocol.get("selection", {})
    require(
        selection.get("test_routes_or_metrics_used_for_selection") is False,
        "test routes or metrics must not participate in model or configuration selection",
    )
    require(
        selection.get("checkpoints_and_configurations_frozen_before_full_test")
        is True,
        "checkpoints and configurations were not frozen before full test",
    )

    require(summary.get("schema") == SUMMARY_SCHEMA, "wrong summary schema")
    rows_raw = summary.get("methods")
    require(isinstance(rows_raw, list), "summary methods must be an array")
    rows = {
        row.get("id"): row
        for row in rows_raw
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    require(len(rows) == len(rows_raw), "summary contains invalid or duplicate method rows")
    require(
        tuple(rows) == EXPECTED_METHODS,
        "summary must contain exactly the seven included full-test methods",
    )
    for method in EXPECTED_METHODS:
        require(method in rows, f"summary is missing {method}")
        row = rows[method]
        quality = row.get("quality", {})
        require(quality.get("status") == "complete", f"{method} quality is incomplete")
        require(quality.get("sample_count") == test_count, f"{method} sample count differs")
        sources = row.get("sources", {})
        require(sources.get("prediction"), f"{method} full prediction source is missing")
        require(
            sources.get("operational_prediction"),
            f"{method} full-test operational efficiency source is missing",
        )
        efficiency = row.get("efficiency", {})
        for key in (
            "prediction_device", "operational_timing_complete",
            "operational_known_active_wall_lower_bound_seconds",
            "operational_wasted_interrupted_wall_seconds", "operational_attempt_count",
            "operational_lost_attempt_count",
            "internal_prediction_seconds", "operational_time_report_sha256",
            "operational_comparability_note",
        ):
            require(efficiency.get(key) is not None, f"{method} efficiency {key} is missing")
        require(
            efficiency.get("operational_full_test_samples") == test_count,
            f"{method} operational sample count differs",
        )
        if efficiency["operational_timing_complete"]:
            for key in (
                "mean_ms_per_query", "queries_per_second", "operational_wall_seconds",
                "operational_successful_final_attempt_wall_seconds",
            ):
                require(
                    efficiency.get(key) is not None,
                    f"{method} complete operational timing {key} is missing",
                )
            require(
                efficiency["operational_lost_attempt_count"] == 0,
                f"{method} complete timing declares lost attempts",
            )
        else:
            require(
                efficiency["operational_lost_attempt_count"] > 0,
                f"{method} incomplete timing does not record a lost attempt",
            )
            require(
                efficiency.get("operational_wall_seconds") is None
                and efficiency.get("mean_ms_per_query") is None
                and efficiency.get("queries_per_second") is None,
                f"{method} incomplete timing claims an exact rate",
            )
        shard_seconds = efficiency.get("shard_adapter_process_seconds")
        if method in {"drncs_lg", "drpk_static", "drp_tp"}:
            require(shard_seconds is not None, f"{method} shard process sum is missing")
        else:
            require(shard_seconds is None, f"{method} unexpectedly has shard process time")
        peak_cuda = efficiency.get("prediction_peak_gpu_memory_bytes")
        if method in {"neuromlr_greedy", "drncs_lg", "drpk_static"}:
            require(
                isinstance(peak_cuda, (int, float)) and peak_cuda > 0,
                f"{method} CUDA peak memory is missing",
            )
        else:
            require(
                peak_cuda in {None, 0}, f"{method} CPU run reports CUDA allocation"
            )
    require(confidence.get("schema") == CONFIDENCE_SCHEMA, "wrong confidence schema")
    require(confidence.get("sample_count") == test_count, "confidence sample count differs")
    require(confidence.get("reference_method") == "project", "paired reference differs")
    confidence_methods = confidence.get("methods")
    require(isinstance(confidence_methods, dict), "confidence methods must be an object")
    require(
        list(confidence_methods.keys()) == sorted(EXPECTED_METHODS),
        "confidence method set/order differs",
    )
    for method in EXPECTED_METHODS:
        intervals = confidence_methods[method].get("intervals", {})
        quality = rows[method]["quality"]
        for metric in (
            "edge_precision",
            "edge_recall",
            "edge_f1",
            "edge_jaccard",
            "exact_match",
        ):
            interval = intervals.get(metric)
            require(isinstance(interval, dict), f"{method} {metric} interval missing")
            require(
                math.isclose(
                    number(interval.get("mean"), f"{method}.{metric}.mean"),
                    number(quality.get(metric), f"{method}.quality.{metric}"),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                f"{method} {metric} mean differs between summary and confidence",
            )
        require(
            confidence_methods[method].get("endpoint_failures")
            == quality.get("endpoint_failures"),
            f"{method} endpoint failures differ",
        )
    return test_count, rows, confidence_methods


def render(
    protocol: dict[str, Any], summary: dict[str, Any], confidence: dict[str, Any]
) -> str:
    test_count, rows, confidence_methods = validate(protocol, summary, confidence)
    lines = [
        "# 北京全量测试路线推荐基线结果",
        "",
        "## 实验设置",
        "",
        "本实验比较 Project、SP-Length、Markov-SP、NeuroMLR-G、DRNCS-LG、DRPK-static 和 DRP-TP 共七种方法。所有方法共享相同的有向 raw-edge 身份、首尾 edge 查询和严格无真值修补协议。",
        "",
        f"NeuroMLR 提供的北京数据经统一结构过滤后包含 605,935 条训练路线、500 条验证路线和 **{test_count:,} 条测试路线**。训练、预处理、checkpoint 与超参数选择仅使用训练集和验证集；所有配置在读取全量测试质量之前冻结。",
        "",
        f"质量和主推理开销来自同一次 {test_count:,}-query 正式任务：0 次预热、1 次完整生产运行。",
        "",
        "## 全量路线质量",
        "",
        "Precision–Exact 单元格为路线级宏平均百分比及 95% 正态均值区间；端点失败为本测试集上的观测计数，0 表示 248,233 条路线中未观察到失败。",
        "",
        "| 方法 | N | Precision [95% CI] | Recall [95% CI] | F1 [95% CI] | Jaccard [95% CI] | Exact [95% CI] | 端点失败 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in EXPECTED_METHODS:
        row = rows[method]
        intervals = confidence_methods[method]["intervals"]
        lines.append(
            "| {label} | {count} | {precision} | {recall} | {f1} | {jaccard} | {exact} | {failures} |".format(
                label=row["label"],
                count=test_count,
                precision=fmt_interval(intervals["edge_precision"]),
                recall=fmt_interval(intervals["edge_recall"]),
                f1=fmt_interval(intervals["edge_f1"]),
                jaccard=fmt_interval(intervals["edge_jaccard"]),
                exact=fmt_interval(intervals["exact_match"]),
                failures=row["quality"]["endpoint_failures"],
            )
        )

    lines.extend(
        [
            "",
            "## 相对 Project 的配对差异",
            "",
            "差异按同一路线成对计算；正值表示该方法高于 Project。",
            "",
            "| 方法 | F1 差值百分点 [95% CI] | Exact 差值百分点 [95% CI] |",
            "|---|---:|---:|",
            "| Project | +0.00 [0.00, 0.00] | +0.00 [0.00, 0.00] |",
        ]
    )
    paired = confidence["paired_differences_vs_reference"]
    for method in EXPECTED_METHODS[1:]:
        row = rows[method]
        f1, exact = paired[method]["edge_f1"], paired[method]["exact_match"]
        lines.append(
            f"| {row['label']} | {100*f1['mean']:+.2f} [{100*f1['lower']:+.2f}, {100*f1['upper']:+.2f}] | "
            f"{100*exact['mean']:+.2f} [{100*exact['lower']:+.2f}, {100*exact['upper']:+.2f}] |"
        )

    neuro_f1_delta = paired["neuromlr_greedy"]["edge_f1"]
    lines.extend(
        [
            "",
            "## 质量结果解读",
            "",
            f"- NeuroMLR-G 的 F1、Jaccard 和 Exact 最高，分别为 {fmt_percent(rows['neuromlr_greedy']['quality']['edge_f1'])}%、{fmt_percent(rows['neuromlr_greedy']['quality']['edge_jaccard'])}% 和 {fmt_percent(rows['neuromlr_greedy']['quality']['exact_match'])}%。相对 Project 的配对 F1 差为 {100*neuro_f1_delta['mean']:+.2f} 个百分点，95% CI [{100*neuro_f1_delta['lower']:+.2f}, {100*neuro_f1_delta['upper']:+.2f}]。",
            f"- Project 的 Precision 最高（{fmt_percent(rows['project']['quality']['edge_precision'])}%），F1 为 {fmt_percent(rows['project']['quality']['edge_f1'])}%，且在 {test_count:,} 条测试路线中没有观察到端点失败。",
            f"- Markov-SP 和 DRPK-static 的 F1 分别为 {fmt_percent(rows['markov_sp']['quality']['edge_f1'])}% 和 {fmt_percent(rows['drpk_static']['quality']['edge_f1'])}%；两者 Exact 均高于 Project，但 F1 显著低于 Project。",
            f"- DRNCS-LG 的 F1 为 {fmt_percent(rows['drncs_lg']['quality']['edge_f1'])}%，端点失败 {rows['drncs_lg']['quality']['endpoint_failures']:,} 条。该结果表明当前 line-graph edge-state 适配不成功，不能外推为对原始 node-state DRNCS 的结论。",
        ]
    )

    lines.extend(
        [
            "",
            "## 训练、预处理与全量推理开销",
            "",
            "训练与预处理时间来自冻结运行记录。推理主值统一使用 `/usr/bin/time -v` 包围完整 production task 的 wall time，包含输入与模型加载、分片物化和重载（若适用）、预测、校验及最终拼接。ms/query 是该完整任务的摊销开销，不是纯模型 kernel latency。",
            "",
            "| 方法 | Offline (s) | Training (s) | 预测设备 | Full wall (s) | ms/query | QPS | Prediction peak RSS (MiB) | Prediction peak CUDA (MiB) |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in EXPECTED_METHODS:
        row, efficiency = rows[method], rows[method]["efficiency"]
        timing_complete = efficiency["operational_timing_complete"]
        wall = (
            f"{efficiency['operational_wall_seconds']:.3f}"
            if timing_complete
            else f"不完整 (已知 ≥{efficiency['operational_known_active_wall_lower_bound_seconds']:.3f})"
        )
        ms_per_query = (
            f"{efficiency['mean_ms_per_query']:.4f}" if timing_complete else "—"
        )
        qps = f"{efficiency['queries_per_second']:.2f}" if timing_complete else "—"
        lines.append(
            f"| {row['label']} | {fmt_seconds(efficiency['offline_seconds'])} | "
            f"{fmt_seconds(efficiency['training_total_seconds'])} | "
            f"{efficiency['prediction_device']} | "
            f"{wall} | {ms_per_query} | {qps} | "
            f"{fmt_memory(efficiency['prediction_peak_rss_kib'])} | "
            f"{fmt_memory(efficiency['prediction_peak_gpu_memory_bytes'] / 1024.0 if efficiency['prediction_peak_gpu_memory_bytes'] is not None else None)} |"
        )

    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- DRNCS-LG 是 clean-room line-graph edge-state 适配，不等同于论文 node-state DRNCS。",
            "- DRPK-static 是缺少时间信息时的等信息 time-collapsed 适配，不是完整 time-aware DRPK。",
            "- DRP-TP 是非学习规划组件，用作 sanity baseline，不是独立训练的 DRPK 复现。",
            "- 本实验只有北京一个城市和单个冻结 checkpoint/seed；路线级置信区间不覆盖训练随机性或跨城市泛化。",
            "- 推理开销来自一次生产运行，没有运行间方差；分片方法的 outer wall 包含分片进程启动和 artifact 重载。",
            "- CPU 与 CUDA 数值按设备和统一 outer wall 边界并列，不解释为硬件无关的算法 speedup。Prediction peak RSS/CUDA 只描述预测阶段；CUDA 值是框架记录的 allocated memory，不是 `nvidia-smi` 总显存。",
            "- NeuroMLR-G 的历史训练记录没有保存训练阶段 peak CUDA 和线程数；训练耗时与本次完整推理计时仍可复核。",
            "- 端点失败为测试集观测计数；0 不表示总体失败概率严格为零。",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as writer:
            writer.write(text)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--confidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = render(
        load_json(args.protocol), load_json(args.summary), load_json(args.confidence)
    )
    atomic_text(args.output.resolve(), report)
    print(f"wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
