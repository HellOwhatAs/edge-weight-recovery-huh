from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "summarize_route_baselines.py"
SPEC = importlib.util.spec_from_file_location("summarize_route_baselines", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
summary_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary_module)

FIXTURE_SHA256 = "0" * 64


def write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def evaluation(method_scale: float = 1.0) -> dict[str, object]:
    return {
        "schema": "ewr.evaluation-summary/v1",
        "sample_count": 2,
        "metrics": {
            "edge_precision": 0.8 * method_scale,
            "edge_recall": 0.7 * method_scale,
            "edge_f1": 0.74 * method_scale,
            "edge_jaccard": 0.62 * method_scale,
            "exact_match": 0.5 * method_scale,
        },
    }


def static_training(method: str = "sp_length") -> dict[str, object]:
    return {
        "schema": "ewr.static-route-baseline-training-diagnostics/v1",
        "method": method,
        "query_protocol": "fixed_true_first_edge_to_true_last_edge",
        "threads": 16,
        "training_samples": 0,
        "validation_samples": 2,
        "coordinate_count": 4,
        "transition_observations": 0,
        "observed_coordinates": 0,
        "selected_alpha": None,
        "validation_candidates": [],
        "timing": {
            "network_and_topology_seconds": 0.1,
            "training_records_load_seconds": 0.0,
            "transition_counting_seconds": 0.0,
            "validation_selection_seconds": 0.2,
            "total_before_artifact_write_seconds": 0.3,
        },
        "peak_rss_kib": 100,
    }


def static_prediction(method: str = "sp_length") -> dict[str, object]:
    return {
        "schema": "ewr.static-route-baseline-prediction-diagnostics/v1",
        "method": method,
        "query_protocol": "fixed_true_first_edge_to_true_last_edge",
        "samples": 2,
        "threads": 16,
        "warmup_repetitions": 1,
        "measured_repetitions": 2,
        "selected_alpha": None,
        "topology_id": "fixture",
        "deterministic_repetitions": True,
        "endpoint_mismatches": 0,
        "timing": {
            "input_and_network_load_seconds": 0.1,
            "topology_and_query_preparation_seconds": 0.1,
            "warmup_metric_and_query_seconds": [0.02],
            "measured_metric_and_query_seconds": [0.01, 0.01],
            "mean_metric_and_query_seconds": 0.01,
            "mean_seconds_per_query": 0.005,
            "queries_per_second": 200.0,
            "total_before_diagnostics_write_seconds": 0.3,
            "timing_boundary": "fixture prediction batch",
        },
        "peak_rss_kib": 120,
    }


def project_prediction() -> dict[str, object]:
    return {
        "schema": "ewr.project-prediction-diagnostics/v1",
        "method": "project",
        "query_protocol": "fixed_true_first_edge_to_true_last_edge",
        "samples": 2,
        "threads": 16,
        "warmup_repetitions": 1,
        "measured_repetitions": 2,
        "completed_updates": 500,
        "objective": 1.0,
        "topology_id": "fixture",
        "oracle_identity": "fixture-cch",
        "deterministic_repetitions": True,
        "timing": {
            "input_and_network_adapter_load_seconds": 0.1,
            "line_graph_and_query_preparation_seconds": 0.1,
            "warmup_metric_and_query_seconds": [0.02],
            "measured_metric_and_query_seconds": [0.01, 0.01],
            "mean_metric_and_query_seconds": 0.01,
            "mean_seconds_per_query": 0.005,
            "queries_per_second": 200.0,
            "total_before_diagnostics_write_seconds": 0.3,
            "timing_boundary": "fixture prediction batch",
        },
        "peak_rss_kib": 120,
    }


def neuromlr_prediction(
    endpoint_failures: int | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "schema": "ewr.neuromlr-diagnostics/v1",
        "method": "neuromlr_greedy",
        "upstream_commit": "fixture",
        "checkpoint": "checkpoint.pt",
        "checkpoint_epoch": 45,
        "dataset_manifest": "test.manifest.json",
        "dataset_manifest_sha256": "fixture",
        "dataset_id": "fixture/test",
        "network_id": "fixture-map",
        "samples": 2,
        "query_protocol": "fixed_true_first_edge_to_true_last_edge",
        "timing": {
            "data_and_graph_seconds": 0.1,
            "model_load_seconds": 0.1,
            "prediction_seconds": 0.02,
            "warmup_repetition_seconds": [0.02],
            "prediction_repetition_seconds": [0.02, 0.02],
            "mean_seconds_per_query": 0.01,
            "queries_per_second": 100.0,
            "component_totals": {},
            "component_totals_per_repetition": [{}, {}],
            "total_process_seconds": 0.3,
        },
        "peak_rss_kib": 200,
        "peak_cuda_memory_bytes": 4096,
        "warmup_repetitions": 1,
        "measured_repetitions": 2,
        "seed": 7,
        "traffic": False,
    }
    if endpoint_failures is not None:
        result["endpoint_failures"] = endpoint_failures
    return result


def drncs_prediction(endpoint_failures: int) -> dict[str, object]:
    return {
        "schema": "ewr.drncs-lg-prediction-diagnostics/v1",
        "method": "drncs_lg",
        "adapter_version": "fixture",
        "audited_upstream_commit": "fixture",
        "checkpoint": "checkpoint.pt",
        "checkpoint_sha256": FIXTURE_SHA256,
        "dataset_manifest": "test.manifest.json",
        "dataset_manifest_sha256": "fixture",
        "dataset_id": "fixture/test",
        "network_id": "fixture-map",
        "samples": 2,
        "query_protocol": "fixed_true_first_raw_edge_to_true_last_raw_edge",
        "endpoint_repair": False,
        "truth_interior_read_during_prediction": False,
        "endpoint_failures": endpoint_failures,
        "timing": {
            "data_and_graph_seconds": 0.1,
            "model_load_seconds": 0.1,
            "warmup_repetition_seconds": [0.02],
            "prediction_repetition_seconds": [0.02, 0.02],
            "mean_prediction_seconds": 0.02,
            "mean_seconds_per_query": 0.01,
            "queries_per_second": 100.0,
            "single_query_latency_samples": 0,
            "single_query_latency_p50_seconds": None,
            "single_query_latency_p95_seconds": None,
            "single_query_latency_max_seconds": None,
            "component_stats_per_repetition": [{}, {}],
            "total_process_seconds": 0.3,
        },
        "warmup_repetitions": 1,
        "measured_repetitions": 2,
        "seed": 7,
        "workers": 16,
        "inference_batch_size": 2,
        "max_steps": 300,
        "peak_rss_kib": 200,
        "peak_cuda_memory_bytes": 0,
        "environment": {"device": "cpu", "workers": "16"},
    }


def sharded_quality_prediction(
    method: str = "drncs_lg", endpoint_failures: int = 1
) -> dict[str, object]:
    cuda = method != "drp_tp"
    return {
        "schema": "ewr.sharded-quality-prediction-diagnostics/v1",
        "method": method,
        "purpose": "full_test_quality_prediction_only",
        "efficiency_comparable": False,
        "efficiency_exclusion_reason": "shards reload immutable artifacts",
        "binding_sha256": "1" * 64,
        "dataset": {
            "dataset_id": "fixture/full-test",
            "network_id": "fixture-roads-v1",
            "manifest_path": "/fixture/test.manifest.json",
            "manifest_sha256": "2" * 64,
            "records_path": "/fixture/test.jsonl",
            "records_sha256": "3" * 64,
            "samples": 2,
            "first_sample_id": "test:0",
            "last_sample_id": "test:1",
        },
        "configuration": {
            "shard_size": 1,
            "seed": 7,
            "workers": 16,
            "device": "cuda" if cuda else "cpu",
            "cuda_visible_devices": "0" if cuda else None,
            "inference_batch_size": 2,
            "max_steps": 300 if method == "drncs_lg" else None,
            "warmup_repetitions": 0,
            "measured_repetitions": 1,
            "latency_samples": 0 if method == "drncs_lg" else None,
            "purpose": "full_test_quality_prediction_only",
            "efficiency_table_source": "current_full_test_single_pass_outer_wall",
        },
        "shards": 2,
        "samples": 2,
        "endpoint_failures": endpoint_failures,
        "generated_route_validity": {"queries": 2},
        "operational_timing": {
            "sum_adapter_process_seconds": 99.0,
            "sum_adapter_prediction_seconds": 88.0,
        },
        "maximum_shard_peak_rss_kib": 321,
        "maximum_shard_peak_cuda_memory_bytes": 4096 if cuda else 0,
        "predictions": {
            "path": "/fixture/predictions.jsonl",
            "sha256": "4" * 64,
            "records_schema": "ewr.prediction-record/v1",
        },
        "shard_outputs": [
            {
                "index": index,
                "samples": 1,
                "predictions_sha256": "5" * 64,
                "diagnostics_sha256": "6" * 64,
                "run_receipt_sha256": "7" * 64,
                "adapter_environment_sha256": "8" * 64,
                "adapter_wall_seconds": 1.0,
            }
            for index in range(2)
        ],
        "completed_at": "2026-07-19T00:00:00+00:00",
    }


def chunked_neuromlr_quality_prediction() -> dict[str, object]:
    result = neuromlr_prediction(endpoint_failures=1)
    result.update(
        checkpoint_sha256="1" * 64,
        dataset_manifest_sha256="2" * 64,
        dataset_records="/fixture/test.jsonl",
        dataset_records_sha256="3" * 64,
        graph_identity="graph-sha256",
        coordinate_identity="coordinate-sha256",
        predictions_sha256="4" * 64,
        execution={
            "mode": "chunked_resumable_quality_prediction",
            "route_chunk_size": 1,
            "completed_chunks": 2,
            "resumed_chunks": 0,
            "resume_dir": "/fixture/resume",
            "progress": "/fixture/progress.json",
            "prediction_chunk_seconds": [0.01, 0.01],
            "timing_scope": "sum_of_atomic_chunk_measurements_across_sessions",
            "resource_scope": "maximum_observed_across_committed_sessions",
        },
        warmup_repetitions=0,
        measured_repetitions=1,
        peak_cuda_memory_bytes=4096,
    )
    result["timing"].update(
        warmup_repetition_seconds=[],
        prediction_repetition_seconds=[0.02],
        component_totals_per_repetition=[],
    )
    return result


def operational_prediction(samples: int = 2) -> dict[str, object]:
    methods = {}
    for method in (
        "project", "sp_length", "markov_sp", "neuromlr_greedy", "drncs_lg",
        "drpk_static", "drp_tp",
    ):
        cuda = method in {"neuromlr_greedy", "drncs_lg", "drpk_static"}
        sharded = method in {"drncs_lg", "drpk_static", "drp_tp"}
        methods[method] = {
            "samples": samples,
            "outer_boundary": "complete prediction task",
            "timing_complete": True,
            "known_active_wall_lower_bound_seconds": 2.0,
            "wall_seconds": 2.0,
            "successful_final_attempt_wall_seconds": 2.0,
            "wasted_interrupted_wall_seconds": 0.0,
            "attempt_count": 1,
            "lost_attempt_count": 0,
            "mean_ms_per_query": 2000.0 / samples,
            "queries_per_second": samples / 2.0,
            "user_seconds": 1.5,
            "system_seconds": 0.2,
            "peak_rss_kib": 400,
            "exit_status": 0,
            "device": "cuda:0" if cuda else "cpu",
            "peak_cuda_memory_bytes": 4096 if cuda else 0,
            "internal_prediction_seconds": 1.0,
            "shard_adapter_process_seconds": 1.5 if sharded else None,
            "time_evidence": {"path": "/fixture/time.json", "sha256": "8" * 64},
            "diagnostic": {
                "path": "/fixture/diagnostic.json",
                "sha256": "9" * 64,
                "schema": "fixture/v1",
            },
        }
    return {
        "schema": "ewr.full-test-operational-efficiency/v1",
        "samples": samples,
        "methods": methods,
        "comparability_note": "uniform outer wall; shard internals differ",
    }


def drpk_selection_v2() -> dict[str, object]:
    selected = {
        "epoch": 10,
        "metrics": {"top1_key_accuracy": 0.5, "mean_weighted_bce": 0.7},
        "seconds": 0.1,
        "learning_rate": 0.001,
        "checkpoint": "checkpoint-best-accuracy.pt",
        "checkpoint_sha256": "3" * 64,
    }
    return {
        "schema": "ewr.drpk-static-selection/v2",
        "selection_rule": ["fixture"],
        "selected": selected,
        "best_loss": dict(selected),
        "evaluations": [],
        "epochs_completed": 10,
        "training_seconds": 2.0,
        "total_seconds": 2.5,
        "peak_rss_kib": 300,
        "peak_cuda_memory_bytes": 8192,
        "resolved_device": "cuda:0",
        "workers": 16,
        "environment": {"device": "cuda:0", "python": "fixture"},
        "checkpoint_last": {
            "path": "checkpoint-last.pt",
            "sha256": "4" * 64,
            "epoch": 10,
        },
    }


def drpk_prediction_v2() -> dict[str, object]:
    return {
        "schema": "ewr.drpk-static-diagnostics/v2",
        "method": "drpk_static",
        "provenance": {},
        "adaptation": "time_collapsed",
        "dataset_id": "fixture/test",
        "dataset_manifest_sha256": "5" * 64,
        "preprocess_configuration_sha256": "6" * 64,
        "checkpoint": {
            "path": "checkpoint-best-accuracy.pt",
            "sha256": "3" * 64,
            "epoch": 10,
        },
        "samples": 2,
        "query_protocol": "fixed_true_first_edge_to_true_last_edge",
        "truth_repair": False,
        "endpoint_failures": 1,
        "timing": {
            "artifact_and_model_load_seconds": 0.1,
            "warmup_repetition_seconds": [0.02],
            "prediction_repetition_seconds": [0.02, 0.02],
            "warmup_component_totals_per_repetition": [{}],
            "component_totals_per_repetition": [{}, {}],
            "component_totals": {},
            "prediction_seconds": 0.02,
            "mean_seconds_per_query": 0.01,
            "queries_per_second": 100.0,
            "total_process_seconds": 0.3,
        },
        "peak_rss_kib": 220,
        "peak_cuda_memory_bytes": 4096,
        "resolved_device": "cuda:0",
        "environment": {"device": "cuda:0", "python": "fixture"},
        "seed": 7,
        "workers": 16,
        "inference_batch_size": 2,
        "warmup_repetitions": 1,
        "measured_repetitions": 2,
    }


def drpk_preprocess_v2() -> dict[str, object]:
    return {
        "schema": "ewr.drpk-static-preprocess-diagnostics/v2",
        "configuration": {"workers": 16, "efficiency_accounting": {}},
        "counts": {},
        "timing": {
            "graph_seconds": 0.1,
            "da_seconds": 0.3,
            "popularity_seconds": 0.05,
            "candidate_label_seconds": 1.0,
            "node2vec": {},
            "drp_tp_ready_seconds": 0.4,
            "drp_tp_ready_peak_rss_kib": 111,
            "total_seconds": 5.0,
        },
        "peak_rss_kib": 333,
    }


def drp_tp_prediction_v2() -> dict[str, object]:
    result = drpk_prediction_v2()
    result["method"] = "drp_tp"
    result["checkpoint"] = None
    result["resolved_device"] = "cpu"
    result["environment"] = {"device": "cpu", "python": "fixture"}
    result["peak_cuda_memory_bytes"] = 0
    return result


def archived_summary() -> dict[str, object]:
    metrics = {
        "edge_precision": 0.75,
        "edge_recall": 0.74,
        "edge_f1": 0.745,
        "edge_jaccard": 0.65,
        "exact_match": 0.45,
        "samples": 2,
    }
    return {
        "schema_version": 1,
        "status": "complete",
        "study": "fixture",
        "repository": {},
        "scope": {},
        "protocol": {},
        "random_seed": 7,
        "data": {},
        "quality": {
            "validation_selection": {
                "project": {"checkpoint_sha256": "1" * 64},
                "neuromlr_greedy": {"checkpoint_sha256": "2" * 64},
            },
            "test": {
                "project_edge_to_edge": {
                    "metrics": dict(metrics),
                    "endpoint_mismatches": 0,
                },
                "neuromlr_greedy": {
                    "metrics": dict(metrics),
                    "endpoint_mismatches": 1,
                },
            }
        },
        "training_runtime_supplementary": {
            "project_full_common_train_500_updates": {
                "wall_seconds": 12.0,
                "peak_rss_kib": 200,
                "threads": 16,
            },
            "neuromlr_50_epochs": {
                "training_seconds": 20.0,
                "total_seconds": 21.0,
                "peak_rss_kib": 300,
                "device": "cuda:0",
            },
        },
        "oracle_efficiency": {},
        "quality_checks": {},
        "resume_candidates": [],
    }


def empty_paths() -> dict[str, dict[str, Path | None]]:
    return {
        method: {key: None for key in summary_module.METHOD_INPUT_KEYS}
        for method in summary_module.METHODS
    }


class RouteBaselineSummaryTests(unittest.TestCase):
    def test_cli_auto_discovers_canonical_static_and_marks_missing_pending(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runtime = root / "runtime"
            write_json(runtime / "static/sp_length.evaluation.json", evaluation())
            write_json(
                runtime / "static/sp_length.train.diagnostics.json", static_training()
            )
            write_json(
                runtime / "static/sp_length.test.diagnostics.json", static_prediction()
            )
            model_artifact = runtime / "static/sp_length.artifact.json"
            model_artifact.write_bytes(b"abc")
            archive = write_json(root / "archive.json", archived_summary())
            manifest = write_json(
                root / "inputs.json",
                {
                    "schema": "ewr.route-baseline-summary-input/v1",
                    "runtime_root": "runtime",
                    "archived_summary": "archive.json",
                    "methods": {
                        "sp_length": {
                            "artifacts": [
                                {
                                    "kind": "model",
                                    "role": "selected_checkpoint",
                                    "path": "static/sp_length.artifact.json",
                                }
                            ]
                        },
                        "drncs_lg": {
                            "evaluation": None,
                            "offline": None,
                            "training": None,
                            "prediction": None,
                        },
                    },
                },
            )
            output = root / "summary.json"
            markdown = root / "results.md"
            csv_output = root / "results.csv"
            self.assertEqual(
                summary_module.main(
                    [
                        "--input-manifest", str(manifest),
                        "--summary-output", str(output),
                        "--markdown-output", str(markdown),
                        "--csv-output", str(csv_output),
                    ]
                ),
                0,
            )
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["schema"], "ewr.route-baseline-summary/v1")
            self.assertEqual([row["id"] for row in result["methods"]], list(summary_module.METHODS))
            rows = {row["id"]: row for row in result["methods"]}
            self.assertEqual(rows["sp_length"]["quality"]["status"], "complete")
            self.assertEqual(rows["sp_length"]["quality"]["endpoint_mismatches"], 0)
            self.assertEqual(rows["sp_length"]["efficiency"]["mean_ms_per_query"], 5.0)
            self.assertEqual(rows["sp_length"]["efficiency"]["threads"], 16)
            self.assertEqual(rows["sp_length"]["artifacts"]["status"], "complete")
            self.assertEqual(rows["sp_length"]["artifacts"]["model_bytes"], 3)
            self.assertEqual(
                len(rows["sp_length"]["artifacts"]["items"][0]["sha256"]), 64
            )
            self.assertEqual(rows["project"]["quality"]["edge_f1"], 0.745)
            self.assertIsNone(rows["project"]["efficiency"]["mean_ms_per_query"])
            self.assertEqual(rows["neuromlr_greedy"]["efficiency"]["device"], "cuda:0")
            self.assertEqual(
                rows["neuromlr_greedy"]["efficiency"]["training_total_seconds"],
                21.0,
            )
            self.assertIsNone(
                rows["neuromlr_greedy"]["efficiency"]["mean_ms_per_query"]
            )
            for method in ("markov_sp", "drncs_lg", "drpk_static", "drp_tp"):
                self.assertEqual(rows[method]["status"], "pending")
            rendered = markdown.read_text(encoding="utf-8")
            self.assertIn("| SP-Length | complete |", rendered)
            self.assertIn("| DRPK-static | pending |", rendered)
            self.assertIn("DRNCS-LG is the registered directed-line-graph", rendered)
            csv_text = csv_output.read_text(encoding="utf-8")
            self.assertIn("endpoint_failures", csv_text.splitlines()[0])
            self.assertIn("sp_length,SP-Length,complete", csv_text)
            self.assertTrue(archive.is_file())

    def test_unknown_evaluator_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            bad = evaluation()
            bad["invented"] = 1
            path = write_json(root / "bad.json", bad)
            paths = empty_paths()
            paths["sp_length"]["evaluation"] = path
            with self.assertRaisesRegex(summary_module.SchemaError, "unexpected"):
                summary_module.summarize(paths, None)

    def test_registered_methods_match_the_formal_publication_set(self) -> None:
        expected = (
            "project",
            "sp_length",
            "markov_sp",
            "neuromlr_greedy",
            "drncs_lg",
            "drpk_static",
            "drp_tp",
        )
        self.assertEqual(tuple(summary_module.METHODS), expected)
        result = summary_module.summarize(empty_paths(), None, expected)
        self.assertEqual(
            tuple(row["id"] for row in result["methods"]), expected
        )

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runtime = root / "runtime"
            runtime.mkdir()
            manifest = write_json(
                root / "inputs.json",
                {
                    "schema": "ewr.route-baseline-summary-input/v1",
                    "runtime_root": "runtime",
                    "methods": {"neuromlr_dijkstra": {}},
                },
            )
            with self.assertRaisesRegex(summary_module.SchemaError, "unknown methods"):
                summary_module._load_manifest(manifest, None)

    def test_explicit_checkpoint_identity_takes_precedence_over_archive(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            checkpoint = root / "formal-checkpoint.json"
            checkpoint.write_bytes(b"formal rerun")
            paths = empty_paths()
            paths["project"]["artifacts"] = [
                {
                    "kind": "model",
                    "role": "selected_checkpoint",
                    "path": checkpoint,
                    "sha256": None,
                }
            ]
            archive = write_json(root / "archive.json", archived_summary())
            result = summary_module.summarize(paths, archive)
            row = next(
                item for item in result["methods"] if item["id"] == "project"
            )
            selected = next(
                item
                for item in row["artifacts"]["items"]
                if item["role"] == "selected_checkpoint"
            )
            self.assertEqual(selected["path"], str(checkpoint.resolve()))
            self.assertNotEqual(selected["sha256"], "1" * 64)
            self.assertEqual(selected["bytes"], len(b"formal rerun"))

    def test_duplicate_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "duplicate.json"
            path.write_text('{"schema":"one","schema":"two"}\n', encoding="utf-8")
            with self.assertRaisesRegex(summary_module.SchemaError, "duplicate JSON key"):
                summary_module.load_json(path)

    def test_prediction_method_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path = write_json(root / "prediction.json", static_prediction("markov_sp"))
            paths = empty_paths()
            paths["sp_length"]["prediction"] = path
            with self.assertRaisesRegex(summary_module.SchemaError, "does not match"):
                summary_module.summarize(paths, None)

    def test_project_prediction_accepts_only_explicit_historical_alias(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paths = empty_paths()
            historical = project_prediction()
            historical["method"] = "project_cch"
            paths["project"]["prediction"] = write_json(
                root / "historical-project.json", historical
            )
            result = summary_module.summarize(paths, None)
            project = next(
                row for row in result["methods"] if row["id"] == "project"
            )
            self.assertEqual(project["efficiency"]["threads"], 16)

            invalid = project_prediction()
            invalid["method"] = "project_cch_unregistered"
            paths["project"]["prediction"] = write_json(
                root / "invalid-project.json", invalid
            )
            with self.assertRaisesRegex(summary_module.SchemaError, "does not match"):
                summary_module.summarize(paths, None)

    def test_evaluator_prediction_sample_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            observed = evaluation()
            observed["sample_count"] = 3
            evaluation_path = write_json(root / "evaluation.json", observed)
            prediction_path = write_json(root / "prediction.json", static_prediction())
            paths = empty_paths()
            paths["sp_length"]["evaluation"] = evaluation_path
            paths["sp_length"]["prediction"] = prediction_path
            with self.assertRaisesRegex(summary_module.SchemaError, "samples differ"):
                summary_module.summarize(paths, None)

    def test_sharded_full_quality_never_contributes_efficiency(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paths = empty_paths()
            paths["drncs_lg"]["evaluation"] = write_json(
                root / "evaluation.json", evaluation()
            )
            paths["drncs_lg"]["prediction"] = write_json(
                root / "sharded.json", sharded_quality_prediction()
            )
            result = summary_module.summarize(paths, None)
            row = next(item for item in result["methods"] if item["id"] == "drncs_lg")
            self.assertEqual(row["quality"]["sample_count"], 2)
            self.assertEqual(row["quality"]["endpoint_failures"], 1)
            self.assertIsNone(row["efficiency"]["mean_ms_per_query"])
            self.assertIsNone(row["efficiency"]["queries_per_second"])

    def test_sharded_quality_integrity_and_device_contracts_are_strict(self) -> None:
        valid = sharded_quality_prediction()
        summary_module.validate_prediction(valid, "drncs_lg", "fixture")
        mutations = (
            ("comparable", lambda value: value.update(efficiency_comparable=True)),
            (
                "cpu fallback",
                lambda value: value["configuration"].update(
                    device="cpu", cuda_visible_devices=None
                ),
            ),
            (
                "sample sum",
                lambda value: value["shard_outputs"][0].update(samples=2),
            ),
            (
                "prediction schema",
                lambda value: value["predictions"].update(records_schema="wrong"),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                candidate = copy.deepcopy(sharded_quality_prediction())
                mutate(candidate)
                with self.assertRaises(summary_module.SchemaError):
                    summary_module.validate_prediction(candidate, "drncs_lg", "fixture")

    def test_full_operational_outer_wall_replaces_internal_shard_rates(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paths = empty_paths()
            paths["drncs_lg"]["evaluation"] = write_json(
                root / "evaluation.json", evaluation()
            )
            paths["drncs_lg"]["prediction"] = write_json(
                root / "sharded.json", sharded_quality_prediction()
            )
            paths["drncs_lg"]["operational_prediction"] = write_json(
                root / "operational.json", operational_prediction()
            )
            result = summary_module.summarize(paths, None)
            row = next(item for item in result["methods"] if item["id"] == "drncs_lg")
            efficiency = row["efficiency"]
            self.assertEqual(efficiency["operational_full_test_samples"], 2)
            self.assertEqual(efficiency["operational_wall_seconds"], 2.0)
            self.assertEqual(efficiency["mean_batch_seconds"], 2.0)
            self.assertEqual(efficiency["mean_ms_per_query"], 1000.0)
            self.assertEqual(efficiency["queries_per_second"], 1.0)
            self.assertEqual(efficiency["internal_prediction_seconds"], 1.0)
            self.assertEqual(efficiency["shard_adapter_process_seconds"], 1.5)
            self.assertEqual(efficiency["prediction_peak_rss_kib"], 400)
            self.assertEqual(efficiency["prediction_peak_gpu_memory_bytes"], 4096)
            self.assertTrue(row["sources"]["operational_prediction"].endswith("operational.json"))

            bad = operational_prediction()
            bad["methods"]["drncs_lg"]["mean_ms_per_query"] = 0.1
            with self.assertRaisesRegex(summary_module.SchemaError, "rate differs"):
                summary_module.validate_operational_prediction(
                    bad, "drncs_lg", "fixture"
                )

    def test_chunked_neuromlr_quality_uses_operational_efficiency(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paths = empty_paths()
            paths["neuromlr_greedy"]["evaluation"] = write_json(
                root / "evaluation.json", evaluation()
            )
            paths["neuromlr_greedy"]["prediction"] = write_json(
                root / "chunked.json", chunked_neuromlr_quality_prediction()
            )
            paths["neuromlr_greedy"]["operational_prediction"] = write_json(
                root / "operational.json",
                operational_prediction(),
            )
            result = summary_module.summarize(paths, None)
            row = next(
                item for item in result["methods"]
                if item["id"] == "neuromlr_greedy"
            )
            self.assertEqual(row["quality"]["endpoint_failures"], 1)
            self.assertEqual(row["efficiency"]["mean_ms_per_query"], 1000.0)
            self.assertEqual(row["efficiency"]["queries_per_second"], 1.0)

            bad = chunked_neuromlr_quality_prediction()
            bad["warmup_repetitions"] = 1
            with self.assertRaisesRegex(
                summary_module.SchemaError, "requires 0 warm-ups"
            ):
                summary_module.validate_prediction(
                    bad, "neuromlr_greedy", "fixture"
                )

            bad = chunked_neuromlr_quality_prediction()
            bad["execution"]["resource_scope"] = "current_process_only"
            with self.assertRaisesRegex(
                summary_module.SchemaError, "wrong chunk resource scope"
            ):
                summary_module.validate_prediction(
                    bad, "neuromlr_greedy", "fixture"
                )

    def test_endpoint_mismatches_use_diagnostics_and_explicit_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paths = empty_paths()
            paths["project"]["prediction"] = write_json(
                root / "project.json", project_prediction()
            )
            paths["neuromlr_greedy"]["prediction"] = write_json(
                root / "greedy.json",
                neuromlr_prediction(endpoint_failures=1),
            )
            paths["drncs_lg"]["prediction"] = write_json(
                root / "drncs.json", drncs_prediction(endpoint_failures=1)
            )
            result = summary_module.summarize(paths, None)
            rows = {row["id"]: row for row in result["methods"]}
            self.assertEqual(rows["project"]["quality"]["endpoint_mismatches"], 0)
            self.assertEqual(rows["project"]["quality"]["endpoint_failures"], 0)
            self.assertEqual(
                rows["neuromlr_greedy"]["quality"]["endpoint_mismatches"], 1
            )
            self.assertEqual(rows["drncs_lg"]["quality"]["endpoint_mismatches"], 1)
            self.assertEqual(rows["drncs_lg"]["quality"]["endpoint_failures"], 1)

    def test_efficiency_completeness_is_method_and_device_aware(self) -> None:
        sp_length = summary_module._empty_efficiency()
        sp_length.update(
            offline_seconds=1.0,
            offline_peak_rss_kib=10,
            offline_device="cpu",
            threads=16,
            prediction_device="cpu",
            batch_boundary="fixture prediction batch",
            mean_batch_seconds=0.1,
            mean_ms_per_query=1.0,
            queries_per_second=1000.0,
            prediction_peak_rss_kib=20,
        )
        self.assertEqual(
            summary_module._efficiency_status("sp_length", sp_length), "complete"
        )
        self.assertIsNone(sp_length["training_total_seconds"])
        self.assertIsNone(sp_length["prediction_peak_gpu_memory_bytes"])

        project = dict(sp_length)
        project["offline_seconds"] = None
        project["offline_peak_rss_kib"] = None
        project["offline_device"] = None
        project["training_total_seconds"] = 2.0
        project["training_peak_rss_kib"] = 30
        project["training_device"] = "cpu"
        self.assertEqual(
            summary_module._efficiency_status("project", project), "complete"
        )
        project["prediction_device"] = "cuda:0"
        self.assertEqual(
            summary_module._efficiency_status("project", project), "partial"
        )
        project["prediction_peak_gpu_memory_bytes"] = 1024
        self.assertEqual(
            summary_module._efficiency_status("project", project), "complete"
        )

    def test_drpk_v2_device_checkpoint_and_endpoint_evidence_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paths = empty_paths()
            paths["drpk_static"]["training"] = write_json(
                root / "selection.json", drpk_selection_v2()
            )
            paths["drpk_static"]["prediction"] = write_json(
                root / "prediction.json", drpk_prediction_v2()
            )
            result = summary_module.summarize(paths, None)
            row = next(
                item for item in result["methods"] if item["id"] == "drpk_static"
            )
            self.assertEqual(row["quality"]["endpoint_failures"], 1)
            self.assertEqual(row["efficiency"]["training_device"], "cuda:0")
            self.assertEqual(row["efficiency"]["prediction_device"], "cuda:0")
            self.assertEqual(
                row["efficiency"]["training_peak_gpu_memory_bytes"], 8192
            )
            checkpoint = next(
                item for item in row["artifacts"]["items"]
                if item["role"] == "selected_checkpoint"
            )
            self.assertEqual(checkpoint["sha256"], "3" * 64)
            self.assertIsNone(checkpoint["bytes"])

    def test_drp_tp_uses_ready_boundary_and_needs_no_training_or_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paths = empty_paths()
            paths["drp_tp"]["offline"] = write_json(
                root / "preprocess.json", drpk_preprocess_v2()
            )
            paths["drp_tp"]["prediction"] = write_json(
                root / "prediction.json", drp_tp_prediction_v2()
            )
            result = summary_module.summarize(paths, None)
            row = next(
                item for item in result["methods"] if item["id"] == "drp_tp"
            )
            efficiency = row["efficiency"]
            self.assertEqual(efficiency["status"], "complete")
            self.assertEqual(efficiency["offline_seconds"], 0.4)
            self.assertEqual(efficiency["offline_peak_rss_kib"], 111)
            self.assertIsNone(efficiency["training_total_seconds"])
            self.assertEqual(efficiency["prediction_peak_gpu_memory_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
