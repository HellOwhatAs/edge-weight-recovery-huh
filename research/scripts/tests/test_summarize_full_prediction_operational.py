import argparse
import json
from pathlib import Path
import tempfile
import unittest

from research.scripts import summarize_full_prediction_operational as operational


def time_report(*, elapsed: str = "0:10.00", exit_status: int = 0) -> str:
    return f"""Command being timed: "predict"
User time (seconds): 8.00
System time (seconds): 1.00
Elapsed (wall clock) time (h:mm:ss or m:ss): {elapsed}
Maximum resident set size (kbytes): 123456
Exit status: {exit_status}
"""


def time_evidence(reports: list[tuple[str, int, str]] | None = None) -> dict:
    if reports is None:
        reports = [("succeeded", 0, time_report())]
    attempts = []
    import hashlib

    for index, (status, return_code, text) in enumerate(reports, 1):
        encoded = text.encode()
        attempts.append(
            {
                "attempt": index,
                "recorded_status": status,
                "status": status,
                "timing_status": "complete",
                "return_code": return_code,
                "started_at": "2026-07-19T00:00:00+00:00",
                "ended_at": "2026-07-19T00:00:10+00:00",
                "time_report_path": f"/fixture/attempt-{index}.txt",
                "time_report_bytes": len(encoded),
                "time_report_sha256": hashlib.sha256(encoded).hexdigest(),
                "time_report_text": text,
            }
        )
    return {
        "schema": operational.TASK_TIME_EVIDENCE_SCHEMA,
        "task_id": "fixture-predict",
        "attempt_count": len(attempts),
        "attempts": attempts,
        "timing_complete": True,
        "lost_attempts": [],
        "aggregation_rule": "sum elapsed wall seconds across every active attempt; exclude downtime between attempts",
        "created_at": "2026-07-19T00:00:11+00:00",
    }


def diagnostic(method: str, samples: int = 100) -> dict:
    if method == "project":
        return {
            "schema": "ewr.project-prediction-diagnostics/v1",
            "method": "project_cch",
            "samples": samples,
            "timing": {"mean_metric_and_query_seconds": 4.0},
        }
    if method in {"sp_length", "markov_sp"}:
        return {
            "schema": "ewr.static-route-baseline-prediction-diagnostics/v1",
            "method": method,
            "samples": samples,
            "timing": {"mean_metric_and_query_seconds": 4.0},
        }
    if method == "neuromlr_greedy":
        return {
            "schema": "ewr.neuromlr-diagnostics/v1",
            "method": method,
            "samples": samples,
            "execution": {"mode": "chunked_resumable_quality_prediction"},
            "warmup_repetitions": 0,
            "measured_repetitions": 1,
            "timing": {"prediction_seconds": 4.0},
            "peak_cuda_memory_bytes": 4096,
        }
    cuda = method != "drp_tp"
    return {
        "schema": operational.SHARDED_SCHEMA,
        "method": method,
        "samples": samples,
        "configuration": {
            "warmup_repetitions": 0,
            "measured_repetitions": 1,
            "device": "cuda:0" if cuda else "cpu",
        },
        "operational_timing": {
            "sum_adapter_prediction_seconds": 3.0,
            "sum_adapter_process_seconds": 6.0,
        },
        "maximum_shard_peak_cuda_memory_bytes": 4096 if cuda else 0,
    }


class OperationalSummaryTests(unittest.TestCase):
    def fixture(self, root: Path, samples: int = 100) -> argparse.Namespace:
        entries = []
        for method in operational.EXPECTED_METHODS:
            time_path = root / f"{method}.time.txt"
            diagnostics_path = root / f"{method}.json"
            time_path.write_text(json.dumps(time_evidence()), encoding="utf-8")
            diagnostics_path.write_text(
                json.dumps(diagnostic(method, samples)), encoding="utf-8"
            )
            entries.append([method, str(time_path), str(diagnostics_path)])
        return argparse.Namespace(samples=samples, entry=entries)

    def test_uniform_outer_wall_is_primary_and_sharded_timing_is_decomposed(self):
        with tempfile.TemporaryDirectory() as raw:
            result = operational.build(self.fixture(Path(raw)))
            self.assertEqual(result["schema"], operational.SCHEMA)
            project = result["methods"]["project"]
            self.assertEqual(project["wall_seconds"], 10.0)
            self.assertEqual(project["mean_ms_per_query"], 100.0)
            self.assertEqual(project["queries_per_second"], 10.0)
            self.assertEqual(project["peak_rss_kib"], 123456)
            self.assertEqual(project["internal_prediction_seconds"], 4.0)
            self.assertIsNone(project["shard_adapter_process_seconds"])
            drpk = result["methods"]["drpk_static"]
            self.assertEqual(drpk["internal_prediction_seconds"], 3.0)
            self.assertEqual(drpk["shard_adapter_process_seconds"], 6.0)
            self.assertEqual(drpk["device"], "cuda:0")

    def test_nonzero_exit_sample_mismatch_and_cuda_fallback_are_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            args = self.fixture(root)
            Path(args.entry[0][1]).write_text(
                json.dumps(
                    time_evidence(
                        [("succeeded", 0, time_report(exit_status=9))]
                    )
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(operational.OperationalError, "exit status"):
                operational.build(args)

    def test_interrupted_attempt_wall_is_preserved_and_reported_separately(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            args = self.fixture(root)
            evidence = time_evidence(
                [
                    ("failed", 143, time_report(elapsed="0:03.00", exit_status=143)),
                    ("succeeded", 0, time_report(elapsed="0:10.00")),
                ]
            )
            Path(args.entry[0][1]).write_text(json.dumps(evidence), encoding="utf-8")
            result = operational.build(args)
            project = result["methods"]["project"]
            self.assertEqual(project["wall_seconds"], 13.0)
            self.assertEqual(project["successful_final_attempt_wall_seconds"], 10.0)
            self.assertEqual(project["wasted_interrupted_wall_seconds"], 3.0)
            self.assertEqual(project["attempt_count"], 2)

    def test_hard_crash_lost_attempt_keeps_quality_pipeline_usable_but_no_exact_rate(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            args = self.fixture(root)
            evidence = time_evidence(
                [
                    ("failed", 143, time_report(elapsed="0:03.00", exit_status=143)),
                    ("succeeded", 0, time_report(elapsed="0:10.00")),
                ]
            )
            lost = evidence["attempts"][0]
            lost.update(
                recorded_status="running",
                status="lost",
                timing_status="lost",
                return_code=None,
                time_report_bytes=None,
                time_report_sha256=None,
                time_report_text=None,
            )
            evidence["timing_complete"] = False
            evidence["lost_attempts"] = [1]
            Path(args.entry[0][1]).write_text(json.dumps(evidence), encoding="utf-8")
            result = operational.build(args)
            project = result["methods"]["project"]
            self.assertFalse(project["timing_complete"])
            self.assertIsNone(project["wall_seconds"])
            self.assertIsNone(project["mean_ms_per_query"])
            self.assertIsNone(project["queries_per_second"])
            self.assertEqual(project["known_active_wall_lower_bound_seconds"], 10.0)
            self.assertEqual(project["lost_attempt_count"], 1)

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            args = self.fixture(root)
            value = diagnostic("drpk_static", 99)
            Path(args.entry[5][2]).write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(operational.OperationalError, "samples differ"):
                operational.build(args)

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            args = self.fixture(root)
            value = diagnostic("drncs_lg")
            value["configuration"]["device"] = "cpu"
            value["maximum_shard_peak_cuda_memory_bytes"] = 0
            Path(args.entry[4][2]).write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(operational.OperationalError, "did not use CUDA"):
                operational.build(args)


if __name__ == "__main__":
    unittest.main()
