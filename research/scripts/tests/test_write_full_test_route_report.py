import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "write_full_test_route_report.py"
SPEC = importlib.util.spec_from_file_location("write_full_test_route_report", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def protocol():
    return {
        "schema": MODULE.PROTOCOL_SCHEMA,
        "status": "frozen_before_full_test",
        "selection": {
            "test_routes_or_metrics_used_for_selection": False,
            "checkpoints_and_configurations_frozen_before_full_test": True,
        },
        "data": {"test": {"eligible_records": 248233}},
        "methods": {
            "included": list(MODULE.EXPECTED_METHODS),
        },
    }


def quality():
    return {
        "status": "complete",
        "sample_count": 248233,
        "edge_precision": 0.8,
        "edge_recall": 0.7,
        "edge_f1": 0.74,
        "edge_jaccard": 0.6,
        "exact_match": 0.5,
        "endpoint_failures": 3,
    }


def efficiency():
    return {
        "offline_seconds": 1.0,
        "training_total_seconds": 2.0,
        "prediction_device": "cpu",
        "mean_ms_per_query": 0.2,
        "queries_per_second": 5000.0,
        "prediction_peak_rss_kib": 1024,
        "prediction_peak_gpu_memory_bytes": 0,
        "operational_full_test_samples": 248233,
        "operational_timing_complete": True,
        "operational_known_active_wall_lower_bound_seconds": 49.6466,
        "operational_wall_seconds": 49.6466,
        "operational_successful_final_attempt_wall_seconds": 49.6466,
        "operational_wasted_interrupted_wall_seconds": 0.0,
        "operational_attempt_count": 1,
        "operational_lost_attempt_count": 0,
        "internal_prediction_seconds": 40.0,
        "shard_adapter_process_seconds": None,
        "operational_time_report_sha256": "a" * 64,
        "operational_comparability_note": "uniform outer boundary",
    }


def summary():
    rows = []
    for method in MODULE.EXPECTED_METHODS:
        rows.append(
            {
                "id": method,
                "label": method,
                "quality": quality(),
                "efficiency": efficiency(),
                "sources": {
                    "prediction": "full.json",
                    "operational_prediction": "operational.json",
                },
            }
        )
        if method in {"drncs_lg", "drpk_static", "drp_tp"}:
            rows[-1]["efficiency"]["shard_adapter_process_seconds"] = 45.0
        if method in {"neuromlr_greedy", "drncs_lg", "drpk_static"}:
            rows[-1]["efficiency"]["prediction_device"] = "cuda:0"
            rows[-1]["efficiency"]["prediction_peak_gpu_memory_bytes"] = 1024 * 1024
    return {"schema": MODULE.SUMMARY_SCHEMA, "methods": rows}


def confidence():
    methods = {}
    paired = {}
    metric_means = {
        "edge_precision": 0.8,
        "edge_recall": 0.7,
        "edge_f1": 0.74,
        "edge_jaccard": 0.6,
        "exact_match": 0.5,
        "endpoint_failure_rate": 3 / 248233,
    }
    for method in sorted(MODULE.EXPECTED_METHODS):
        methods[method] = {
            "endpoint_failures": 3,
            "intervals": {
                metric: {
                    "mean": mean,
                    "standard_error": 0.001,
                    "lower": max(0.0, mean - 0.002),
                    "upper": min(1.0, mean + 0.002),
                }
                for metric, mean in metric_means.items()
            },
        }
        if method != "project":
            paired[method] = {
                metric: {
                    "mean": 0.0,
                    "standard_error": 0.001,
                    "lower": -0.002,
                    "upper": 0.002,
                }
                for metric in metric_means
            }
    return {
        "schema": MODULE.CONFIDENCE_SCHEMA,
        "sample_count": 248233,
        "reference_method": "project",
        "methods": methods,
        "paired_differences_vs_reference": paired,
    }


class FullTestReportTests(unittest.TestCase):
    def test_report_uses_full_operational_efficiency_boundary(self):
        rendered = MODULE.render(protocol(), summary(), confidence())
        self.assertIn("248,233", rendered)
        self.assertIn("共七种方法", rendered)
        self.assertIn("质量结果解读", rendered)
        self.assertIn("Full wall", rendered)
        self.assertIn("Prediction peak RSS", rendered)
        self.assertIn("Prediction peak CUDA", rendered)
        self.assertIn("74.00 [73.80, 74.20]", rendered)
        self.assertIn("端点失败为本测试集上的观测计数", rendered)
        self.assertNotIn("pilot", rendered)
        self.assertNotIn("NeuroMLR-D", rendered)
        self.assertNotIn("Internal prediction", rendered)

    def test_mixed_dijkstra_or_incomplete_protocol_is_rejected(self):
        bad_summary = summary()
        bad_summary["methods"].append(
            {
                "id": "neuromlr_dijkstra",
                "label": "NeuroMLR-D",
                "quality": {"sample_count": 248233},
            }
        )
        with self.assertRaisesRegex(MODULE.ReportError, "exactly the seven"):
            MODULE.render(protocol(), bad_summary, confidence())
        bad_protocol = protocol()
        bad_protocol["status"] = "implementation_in_progress"
        with self.assertRaisesRegex(MODULE.ReportError, "not frozen"):
            MODULE.render(bad_protocol, summary(), confidence())
        bad_protocol = protocol()
        bad_protocol["selection"]["test_routes_or_metrics_used_for_selection"] = True
        with self.assertRaisesRegex(MODULE.ReportError, "must not participate"):
            MODULE.render(bad_protocol, summary(), confidence())


if __name__ == "__main__":
    unittest.main()
