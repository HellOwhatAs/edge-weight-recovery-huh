import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "run_route_baseline_pipeline.py"
SPEC = importlib.util.spec_from_file_location("route_pipeline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
pipeline_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline_module)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PipelineTests(unittest.TestCase):
    def write_config(self, root: Path, tasks: list[dict]) -> Path:
        config = {
            "schema": pipeline_module.CONFIG_SCHEMA,
            "workspace_root": str(root),
            "runtime_root": "runtime",
            "state_dir": ".pipeline",
            "cpu_threads": 16,
            "resources": {
                "MemoryHigh": "8G",
                "MemoryMax": "10G",
                "MemorySwapMax": "4G",
                "OOMPolicy": "stop",
            },
            "tasks": tasks,
        }
        path = root / "pipeline.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def test_success_receipt_binds_inputs_and_skips_verified_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = root / "worker.py"
            worker.write_text(
                """import json
from pathlib import Path
counter = Path('counter')
counter.write_text(str(int(counter.read_text()) + 1) if counter.exists() else '1')
Path('result.json').write_text(json.dumps({'schema': 'result/v1'}))
""",
                encoding="utf-8",
            )
            config = self.write_config(
                root,
                [
                    {
                        "id": "smoke-one",
                        "profiles": ["smoke"],
                        "action": "command",
                        "device": "cpu",
                        "command": [sys.executable, "worker.py"],
                        "requires": [{"path": "worker.py"}],
                        "outputs": [
                            {
                                "path": "result.json",
                                "json_contains": {"schema": "result/v1"},
                            }
                        ],
                    }
                ],
            )
            pipeline = pipeline_module.Pipeline(config)
            pipeline.run("smoke", direct=True, poll_seconds=0.01)
            self.assertEqual((root / "counter").read_text(), "1")
            pipeline.run("smoke", direct=True, poll_seconds=0.01)
            self.assertEqual((root / "counter").read_text(), "1")
            state = pipeline.read_state()
            self.assertEqual(state["status"], "succeeded")
            self.assertEqual(state["tasks"]["smoke-one"]["status"], "skipped_verified")
            receipt = json.loads(pipeline.receipt_path("smoke-one").read_text())
            self.assertEqual(receipt["requires"][0]["sha256"], digest(worker))
            self.assertEqual(receipt["outputs"][0]["sha256"], digest(root / "result.json"))
            self.assertTrue(Path(receipt["time_report"]).is_file())

    def test_receipt_binds_runner_hash_global_context_and_isolation_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.write_config(
                root,
                [
                    {
                        "id": "identity",
                        "profiles": ["smoke"],
                        "action": "command",
                        "device": "cpu",
                        "command": [sys.executable, "-c", "pass"],
                        "outputs": [],
                    }
                ],
            )
            pipeline = pipeline_module.Pipeline(config)
            pipeline.run("smoke", direct=True, poll_seconds=0.01)
            task = pipeline.task_by_id["identity"]
            valid, _ = pipeline.validate_receipt(task, expected_direct=True)
            self.assertTrue(valid)
            valid, reason = pipeline.validate_receipt(task, expected_direct=False)
            self.assertFalse(valid)
            self.assertIn("execution mode", reason)

            changed_runner = pipeline_module.Pipeline(config)
            changed_runner.runner_sha256 = "0" * 64
            valid, reason = changed_runner.validate_receipt(
                changed_runner.task_by_id["identity"], expected_direct=True
            )
            self.assertFalse(valid)
            self.assertIn("configuration changed", reason)

            raw = json.loads(config.read_text(encoding="utf-8"))
            raw["cpu_threads"] = 8
            raw["environment"] = {"PIPELINE_TEST_MODE": "changed"}
            raw["resources"]["MemoryMax"] = "9G"
            config.write_text(json.dumps(raw), encoding="utf-8")
            changed_defaults = pipeline_module.Pipeline(config)
            valid, reason = changed_defaults.validate_receipt(
                changed_defaults.task_by_id["identity"], expected_direct=True
            )
            self.assertFalse(valid)
            self.assertIn("configuration changed", reason)

    def test_full_includes_smoke_and_is_fail_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.write_config(
                root,
                [
                    {
                        "id": "smoke-fails",
                        "profiles": ["smoke"],
                        "action": "command",
                        "device": "cpu",
                        "command": [sys.executable, "-c", "raise SystemExit(7)"],
                        "outputs": [],
                    },
                    {
                        "id": "full-must-not-run",
                        "profiles": ["full"],
                        "depends_on": ["smoke-fails"],
                        "action": "command",
                        "device": "cpu",
                        "command": [
                            sys.executable,
                            "-c",
                            "from pathlib import Path; Path('forbidden').touch()",
                        ],
                        "outputs": [{"path": "forbidden"}],
                    },
                ],
            )
            pipeline = pipeline_module.Pipeline(config)
            with self.assertRaisesRegex(pipeline_module.PipelineError, "exit code 7"):
                pipeline.run("full", direct=True, poll_seconds=0.01)
            self.assertFalse((root / "forbidden").exists())
            state = pipeline.read_state()
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["tasks"]["smoke-fails"]["status"], "failed")
            self.assertEqual(state["tasks"]["full-must-not-run"]["status"], "pending")

    def test_valid_outputs_can_be_adopted_after_runner_interruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "done.json").write_text(
                json.dumps({"schema": "done/v1"}), encoding="utf-8"
            )
            config = self.write_config(
                root,
                [
                    {
                        "id": "adopt",
                        "profiles": ["full"],
                        "action": "command",
                        "device": "cpu",
                        "adopt_outputs": True,
                        "command": [
                            sys.executable,
                            "-c",
                            "from pathlib import Path; Path('must-not-run').touch()",
                        ],
                        "outputs": [
                            {
                                "path": "done.json",
                                "json_contains": {"schema": "done/v1"},
                            }
                        ],
                    }
                ],
            )
            pipeline = pipeline_module.Pipeline(config)
            pipeline.run("full", direct=True, poll_seconds=0.01)
            self.assertFalse((root / "must-not-run").exists())
            receipt = json.loads(pipeline.receipt_path("adopt").read_text())
            self.assertEqual(receipt["disposition"], "adopted_validated_outputs")

    def test_stable_time_output_is_promoted_before_and_bound_by_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.write_config(
                root,
                [
                    {
                        "id": "timed",
                        "profiles": ["full"],
                        "action": "command",
                        "device": "cpu",
                        "command": [
                            sys.executable,
                            "-c",
                            "from pathlib import Path; Path('done.json').write_text('{}')",
                        ],
                        "time_output": "formal-time.txt",
                        "time_aggregate_output": "formal-time.aggregate.json",
                        "outputs": [
                            {"path": "done.json"},
                            {"path": "formal-time.txt", "min_bytes": 10},
                            {
                                "path": "formal-time.aggregate.json",
                                "json_contains": {
                                    "schema": pipeline_module.TASK_TIME_EVIDENCE_SCHEMA,
                                    "attempt_count": 1,
                                },
                            },
                        ],
                    }
                ],
            )
            pipeline = pipeline_module.Pipeline(config)
            pipeline.run("full", direct=True, poll_seconds=0.01)
            formal = root / "formal-time.txt"
            self.assertTrue(formal.is_file())
            receipt = json.loads(pipeline.receipt_path("timed").read_text())
            bound = next(
                item for item in receipt["outputs"] if item["path"] == str(formal)
            )
            self.assertEqual(bound["sha256"], digest(formal))
            formal.write_text("tampered", encoding="utf-8")
            valid, reason = pipeline.validate_receipt(
                pipeline.task_by_id["timed"], expected_direct=True
            )
            self.assertFalse(valid)
            self.assertIn("unexpectedly small", reason)

    def test_adoption_recovers_exact_successful_attempt_time_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "done.json").write_text("{}", encoding="utf-8")
            prior_time = root / "prior-attempt.time.txt"
            prior_time.write_text("exact prior GNU time evidence\n", encoding="utf-8")
            config = self.write_config(
                root,
                [
                    {
                        "id": "recover-time",
                        "profiles": ["full"],
                        "action": "command",
                        "device": "cpu",
                        "adopt_outputs": True,
                        "command": [
                            sys.executable,
                            "-c",
                            "from pathlib import Path; Path('forbidden').touch()",
                        ],
                        "time_output": "formal-time.txt",
                        "time_aggregate_output": "formal-time.aggregate.json",
                        "outputs": [
                            {"path": "done.json"},
                            {"path": "formal-time.txt", "min_bytes": 10},
                            {
                                "path": "formal-time.aggregate.json",
                                "json_contains": {
                                    "schema": pipeline_module.TASK_TIME_EVIDENCE_SCHEMA,
                                    "attempt_count": 1,
                                },
                            },
                        ],
                    }
                ],
            )
            pipeline = pipeline_module.Pipeline(config)
            state = pipeline._new_state("full")
            state["tasks"]["recover-time"]["attempts"] = [
                {
                    "attempt": 1,
                    "status": "succeeded",
                    "time_report": str(prior_time),
                }
            ]
            pipeline.state_dir.mkdir(parents=True, exist_ok=True)
            pipeline.write_state(state)
            pipeline.run("full", direct=True, poll_seconds=0.01)
            self.assertFalse((root / "forbidden").exists())
            self.assertEqual(
                (root / "formal-time.txt").read_bytes(), prior_time.read_bytes()
            )
            receipt = json.loads(pipeline.receipt_path("recover-time").read_text())
            self.assertEqual(receipt["disposition"], "adopted_validated_outputs")
            self.assertEqual(receipt["attempt"], 1)

    def test_time_aggregate_preserves_failed_and_successful_attempt_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prior_time = root / "failed.time.txt"
            prior_time.write_text(
                "User time (seconds): 1.0\n"
                "System time (seconds): 0.1\n"
                "Elapsed (wall clock) time (h:mm:ss or m:ss): 0:01.2\n"
                "Maximum resident set size (kbytes): 100\n"
                "Exit status: 143\n",
                encoding="utf-8",
            )
            config = self.write_config(
                root,
                [
                    {
                        "id": "resumed-prediction",
                        "profiles": ["full"],
                        "action": "command",
                        "device": "cpu",
                        "command": [
                            sys.executable,
                            "-c",
                            "from pathlib import Path; Path('done.json').write_text('{}')",
                        ],
                        "time_output": "final.time.txt",
                        "time_aggregate_output": "all-attempts.json",
                        "outputs": [
                            {"path": "done.json"},
                            {"path": "final.time.txt"},
                            {
                                "path": "all-attempts.json",
                                "json_contains": {
                                    "schema": pipeline_module.TASK_TIME_EVIDENCE_SCHEMA,
                                    "attempt_count": 2,
                                },
                            },
                        ],
                    }
                ],
            )
            pipeline = pipeline_module.Pipeline(config)
            state = pipeline._new_state("full")
            state["tasks"]["resumed-prediction"]["attempts"] = [
                {
                    "attempt": 1,
                    "status": "failed",
                    "return_code": 143,
                    "started_at": pipeline_module.utc_now(),
                    "ended_at": pipeline_module.utc_now(),
                    "time_report": str(prior_time),
                }
            ]
            pipeline.state_dir.mkdir(parents=True, exist_ok=True)
            pipeline.write_state(state)
            pipeline.run("full", direct=True, poll_seconds=0.01)
            aggregate = json.loads((root / "all-attempts.json").read_text())
            self.assertEqual(aggregate["attempt_count"], 2)
            self.assertEqual(
                [item["status"] for item in aggregate["attempts"]],
                ["failed", "succeeded"],
            )
            self.assertEqual(
                aggregate["attempts"][0]["time_report_text"],
                prior_time.read_text(),
            )

    def test_hard_crash_lost_time_does_not_block_resumed_quality_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing_time = root / "hard-crash.time.txt"
            missing_time.write_text("truncated\nExit status: 143\n", encoding="utf-8")
            config = self.write_config(
                root,
                [
                    {
                        "id": "crash-resume",
                        "profiles": ["full"],
                        "action": "command",
                        "device": "cpu",
                        "command": [
                            sys.executable,
                            "-c",
                            "from pathlib import Path; Path('quality.json').write_text('{}')",
                        ],
                        "time_output": "final.time.txt",
                        "time_aggregate_output": "all-attempts.json",
                        "outputs": [
                            {"path": "quality.json"},
                            {"path": "final.time.txt"},
                            {
                                "path": "all-attempts.json",
                                "json_contains": {
                                    "schema": pipeline_module.TASK_TIME_EVIDENCE_SCHEMA,
                                    "attempt_count": 2,
                                    "timing_complete": False,
                                    "lost_attempts": [1],
                                },
                            },
                        ],
                    }
                ],
            )
            pipeline = pipeline_module.Pipeline(config)
            state = pipeline._new_state("full")
            state["tasks"]["crash-resume"]["attempts"] = [
                {
                    "attempt": 1,
                    "status": "running",
                    "return_code": None,
                    "started_at": pipeline_module.utc_now(),
                    "time_report": str(missing_time),
                }
            ]
            pipeline.state_dir.mkdir(parents=True, exist_ok=True)
            pipeline.write_state(state)
            pipeline.run("full", direct=True, poll_seconds=0.01)
            self.assertTrue((root / "quality.json").is_file())
            aggregate = json.loads((root / "all-attempts.json").read_text())
            self.assertFalse(aggregate["timing_complete"])
            self.assertEqual(aggregate["lost_attempts"], [1])
            self.assertEqual(aggregate["attempts"][0]["status"], "lost")
            self.assertEqual(aggregate["attempts"][1]["status"], "succeeded")

    def test_stale_running_complete_exit_zero_is_adopted_as_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "quality.json").write_text("{}", encoding="utf-8")
            stale_time = root / "stale.time.txt"
            stale_time.write_text(
                "User time (seconds): 1.0\n"
                "System time (seconds): 0.1\n"
                "Elapsed (wall clock) time (h:mm:ss or m:ss): 0:01.2\n"
                "Maximum resident set size (kbytes): 100\n"
                "Exit status: 0\n",
                encoding="utf-8",
            )
            config = self.write_config(
                root,
                [
                    {
                        "id": "adopt-crash",
                        "profiles": ["full"],
                        "action": "command",
                        "device": "cpu",
                        "adopt_outputs": True,
                        "command": [
                            sys.executable,
                            "-c",
                            "from pathlib import Path; Path('forbidden').touch()",
                        ],
                        "time_output": "final.time.txt",
                        "time_aggregate_output": "all-attempts.json",
                        "outputs": [
                            {"path": "quality.json"},
                            {"path": "final.time.txt"},
                            {
                                "path": "all-attempts.json",
                                "json_contains": {
                                    "timing_complete": True,
                                    "attempt_count": 1,
                                },
                            },
                        ],
                    }
                ],
            )
            pipeline = pipeline_module.Pipeline(config)
            state = pipeline._new_state("full")
            state["tasks"]["adopt-crash"]["attempts"] = [
                {
                    "attempt": 1,
                    "status": "running",
                    "return_code": None,
                    "started_at": pipeline_module.utc_now(),
                    "time_report": str(stale_time),
                }
            ]
            pipeline.state_dir.mkdir(parents=True, exist_ok=True)
            pipeline.write_state(state)
            pipeline.run("full", direct=True, poll_seconds=0.01)
            self.assertFalse((root / "forbidden").exists())
            aggregate = json.loads((root / "all-attempts.json").read_text())
            self.assertEqual(aggregate["attempts"][0]["recorded_status"], "running")
            self.assertEqual(aggregate["attempts"][0]["status"], "succeeded")
            self.assertEqual(aggregate["attempts"][0]["return_code"], 0)

    def test_adoption_requires_diagnostic_to_bind_exact_checkpoint_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"model-v1")
            diagnostics = root / "diagnostics.json"
            diagnostics.write_text(
                json.dumps(
                    {"schema": "train/v1", "checkpoint_sha256": digest(checkpoint)}
                ),
                encoding="utf-8",
            )
            task = {
                "id": "strict-adopt",
                "profiles": ["full"],
                "action": "command",
                "device": "cpu",
                "adopt_outputs": True,
                "command": [
                    sys.executable,
                    "-c",
                    "from pathlib import Path; Path('must-not-run').touch()",
                ],
                "outputs": [
                    {"path": "checkpoint.pt"},
                    {
                        "path": "diagnostics.json",
                        "json_contains": {"schema": "train/v1"},
                    },
                ],
                "hash_bindings": [
                    {
                        "json_path": "diagnostics.json",
                        "field": "checkpoint_sha256",
                        "file_path": "checkpoint.pt",
                    }
                ],
            }
            config = self.write_config(root, [task])
            pipeline = pipeline_module.Pipeline(config)
            pipeline.run("full", direct=True, poll_seconds=0.01)
            self.assertFalse((root / "must-not-run").exists())

            pipeline.receipt_path("strict-adopt").unlink()
            checkpoint.write_bytes(b"corrupted")
            with self.assertRaisesRegex(
                pipeline_module.PipelineError, "hash binding mismatch"
            ):
                pipeline.run("full", direct=True, poll_seconds=0.01)
            self.assertFalse((root / "must-not-run").exists())

    def test_immutable_invalid_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "formal.json").write_text("{}", encoding="utf-8")
            config = self.write_config(
                root,
                [
                    {
                        "id": "formal",
                        "profiles": ["full"],
                        "action": "command",
                        "device": "cpu",
                        "immutable": True,
                        "command": [sys.executable, "-c", "raise SystemExit(0)"],
                        "outputs": [
                            {
                                "path": "formal.json",
                                "sha256": "0" * 64,
                            }
                        ],
                    }
                ],
            )
            pipeline = pipeline_module.Pipeline(config)
            with self.assertRaisesRegex(
                pipeline_module.PipelineError, "refusing to overwrite"
            ):
                pipeline.run("full", direct=True, poll_seconds=0.01)
            self.assertEqual((root / "formal.json").read_text(), "{}")

    def test_status_defers_formal_test_access_until_training_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.write_config(
                root,
                [
                    {
                        "id": "training",
                        "profiles": ["full"],
                        "action": "verify",
                        "outputs": [{"path": "missing-training.json"}],
                    },
                    {
                        "id": "formal-test",
                        "profiles": ["full"],
                        "depends_on": ["training"],
                        "action": "verify",
                        "requires": [{"path": "must-not-be-opened-test.jsonl"}],
                        "outputs": [{"path": "missing-evaluation.json"}],
                    },
                ],
            )
            pipeline = pipeline_module.Pipeline(config)
            state = pipeline._new_state("full")
            pipeline.write_state(state)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                pipeline.status(verify=True, as_json=True)
            snapshot = json.loads(output.getvalue())
            self.assertEqual(
                snapshot["tasks"]["formal-test"]["verification"],
                "deferred_until_dependencies_complete",
            )

    def test_cuda_task_must_explicitly_resolve_to_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.write_config(
                root,
                [
                    {
                        "id": "bad-cuda",
                        "profiles": ["full"],
                        "action": "command",
                        "device": "cuda:0",
                        "command": ["trainer", "--device", "cpu"],
                        "outputs": [],
                    }
                ],
            )
            with self.assertRaisesRegex(
                pipeline_module.PipelineError, "may not use device"
            ):
                pipeline_module.Pipeline(config)

    def test_status_exposes_device_recoverable_progress_eta_and_heartbeat(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "progress.json").write_text(
                json.dumps(
                    {
                        "stage": "sparse_training",
                        "completed_units": 7,
                        "recoverable_completed_units": 6,
                        "total_units": 20,
                        "unit": "epochs",
                        "pipeline_percent": 78.5,
                        "phase_eta_seconds": 123,
                        "updated_at": pipeline_module.utc_now(),
                    }
                ),
                encoding="utf-8",
            )
            config = self.write_config(
                root,
                [
                    {
                        "id": "training",
                        "profiles": ["full"],
                        "action": "command",
                        "device": "mixed:cuda:0+cpu",
                        "execution_note": "GPU model plus CPU planner",
                        "command": ["trainer", "--device", "cuda"],
                        "progress_path": "progress.json",
                        "outputs": [],
                    }
                ],
            )
            pipeline = pipeline_module.Pipeline(config)
            state = pipeline._new_state("full")
            state["current_task"] = "training"
            state["tasks"]["training"].update(
                {"status": "running", "started_at": pipeline_module.utc_now()}
            )
            pipeline.write_state(state)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                pipeline.status(verify=False, as_json=False)
            rendered = output.getvalue()
            self.assertIn("mixed:cuda:0+cpu", rendered)
            self.assertIn("sparse_training", rendered)
            self.assertIn("7/20 epochs", rendered)
            self.assertIn("6/20 epochs recoverable", rendered)
            self.assertIn("2m 03s", rendered)
            self.assertIn("heartbeat", rendered)

    def test_status_understands_sharded_and_neuromlr_sample_progress(self) -> None:
        cases = (
            (
                {
                    "schema": "ewr.sharded-prediction-progress/v1",
                    "status": "running",
                    "method": "drpk_static",
                    "completed_samples": 4096,
                    "total_samples": 248233,
                    "percent": 1.65,
                    "estimated_remaining_adapter_seconds": 123.0,
                    "updated_at": pipeline_module.utc_now(),
                },
                ("4096/248233 samples", "2m 03s", "running"),
            ),
            (
                {
                    "schema": "ewr.neuromlr-prediction-progress/v1",
                    "status": "running",
                    "completed_samples": 50,
                    "total_samples": 100,
                    "prediction_seconds": 10.0,
                },
                ("50/100 samples", "10s", "heartbeat"),
            ),
        )
        for index, (progress, expected) in enumerate(cases):
            with self.subTest(schema=progress["schema"]), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                progress_path = root / f"progress-{index}.json"
                progress_path.write_text(json.dumps(progress), encoding="utf-8")
                config = self.write_config(
                    root,
                    [
                        {
                            "id": "prediction",
                            "profiles": ["full"],
                            "action": "command",
                            "device": "cuda:0",
                            "command": ["predictor", "--device", "cuda:0"],
                            "progress_path": str(progress_path),
                            "outputs": [],
                        }
                    ],
                )
                pipeline = pipeline_module.Pipeline(config)
                state = pipeline._new_state("full")
                state["current_task"] = "prediction"
                state["tasks"]["prediction"].update(
                    {"status": "running", "started_at": pipeline_module.utc_now()}
                )
                pipeline.state_dir.mkdir(parents=True, exist_ok=True)
                pipeline.write_state(state)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    pipeline.status(verify=False, as_json=False)
                rendered = output.getvalue()
                for fragment in expected:
                    self.assertIn(fragment, rendered)


if __name__ == "__main__":
    unittest.main()
