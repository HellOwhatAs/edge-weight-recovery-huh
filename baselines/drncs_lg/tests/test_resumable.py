from __future__ import annotations

import importlib.util
import json
import runpy
import signal
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch


BASELINE_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = BASELINE_DIR.parents[1]
sys.path.insert(0, str(BASELINE_DIR))
import drncs_lg_adapter as adapter  # noqa: E402
import drncs_lg_resumable as resumable  # noqa: E402


HAS_ARRAY_STACK = all(
    importlib.util.find_spec(name) is not None for name in ("numpy", "shapefile", "torch")
)


@unittest.skipUnless(HAS_ARRAY_STACK, "pinned NumPy/pyshp/PyTorch stack is unavailable")
class ResumableTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        adapter.load_array_dependencies()

    def tearDown(self) -> None:
        resumable.reset_stop_request_for_tests()

    def write_dataset(self, root: Path, dataset_id: str, rows: list[dict]) -> Path:
        root.mkdir()
        records = root / "records.jsonl"
        records.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": adapter.DATASET_MANIFEST_SCHEMA,
                    "dataset_id": dataset_id,
                    "network_id": "resume-tiny-network",
                    "records_schema": adapter.DATASET_RECORD_SCHEMA,
                    "records_file": "records.jsonl",
                },
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        return manifest

    def prepare_inputs(self, root: Path) -> tuple[Path, Path, Path]:
        map_dir = root / "map"
        map_dir.mkdir()
        writer = adapter.shapefile.Writer(
            str(map_dir / "edges"), shapeType=adapter.shapefile.POLYLINE
        )
        writer.field("fid", "N", decimal=0)
        writer.field("u", "N", decimal=0)
        writer.field("v", "N", decimal=0)
        # Contracting states 0 and 1 creates 2->3 and 4->5.  Training never
        # observes either segment, so the tiny run exercises two independently
        # recoverable SC2 destination units rather than satisfying shortcuts
        # from SC1.
        endpoints = [
            (20, 30),
            (60, 70),
            (10, 20),
            (30, 40),
            (50, 60),
            (70, 80),
            (90, 100),
            (100, 110),
        ]
        for edge, (tail, head) in enumerate(endpoints):
            writer.line([[(float(tail), 0.0), (float(head), 0.0)]])
            writer.record(edge, tail, head)
        writer.close()
        train_manifest = self.write_dataset(
            root / "train",
            "resume-tiny-train",
            [
                {"sample_id": "train-a", "original_edge_ids": [6, 7]},
                {"sample_id": "train-b", "original_edge_ids": [6, 7]},
            ],
        )
        validation_manifest = self.write_dataset(
            root / "validation",
            "resume-tiny-validation",
            [
                {"sample_id": "validation-a", "original_edge_ids": [2, 0, 3]},
                {"sample_id": "validation-b", "original_edge_ids": [4, 1, 5]},
                {"sample_id": "validation-c", "original_edge_ids": [6, 7]},
            ],
        )
        preprocess_dir = root / "preprocess"
        preprocess_args = adapter.parse_args(
            [
                "preprocess",
                "--train-manifest",
                str(train_manifest),
                "--map-dir",
                str(map_dir),
                "--output-dir",
                str(preprocess_dir),
                "--source-revision",
                "resume-test-revision",
                "--contraction-ratio",
                "0.25",
                "--workers",
                "2",
            ]
        )

        def tiny_embeddings(outgoing, **_):
            return adapter.np.arange(
                len(outgoing) * 2, dtype=adapter.np.float32
            ).reshape(len(outgoing), 2)

        with patch.object(adapter, "train_node2vec_embeddings", tiny_embeddings):
            with redirect_stdout(StringIO()):
                adapter.preprocess_command(preprocess_args)
        return preprocess_dir, train_manifest, validation_manifest

    def training_args(
        self,
        preprocess: Path,
        train: Path,
        validation: Path,
        output: Path,
        *,
        device: str,
        epochs: int,
        resume: str = "auto",
    ):
        return resumable.parse_args(
            [
                "--preprocess-dir",
                str(preprocess),
                "--train-manifest",
                str(train),
                "--validation-manifest",
                str(validation),
                "--output-dir",
                str(output),
                "--source-revision",
                "resume-test-revision",
                "--device",
                device,
                "--workers",
                "2",
                "--epochs",
                str(epochs),
                "--validation-every",
                "1",
                "--batch-size",
                "8",
                "--transition-chunk-size",
                "8",
                "--hidden-dimension",
                "4",
                "--max-steps",
                "10",
                "--sc2-score-batch-size",
                "8",
                "--sc2-checkpoint-every",
                "1",
                "--resume",
                resume,
            ]
        )

    def assert_model_states_equal(self, left: dict, right: dict, key: str) -> None:
        self.assertEqual(set(left[key]), set(right[key]))
        for name in left[key]:
            self.assertTrue(
                adapter.torch.equal(left[key][name], right[key][name]),
                f"state tensor differs: {key}.{name}",
            )

    def exercise_stage_resume(self, root: Path, stage: str) -> None:
        preprocess, train, validation = self.prepare_inputs(root)
        resumed_output = root / "resumed"
        fresh_output = root / "fresh"
        args = self.training_args(
            preprocess, train, validation, resumed_output, device="cpu", epochs=2
        )
        original_write = resumable.ProgressReporter.write
        triggered = False

        def request_stop(reporter, **kwargs):
            nonlocal triggered
            value = original_write(reporter, **kwargs)
            if (
                not triggered
                and kwargs["status"] == "running"
                and kwargs["stage"] == stage
                and kwargs["completed_units"] == 1
            ):
                triggered = True
                resumable._STOP_SIGNAL = signal.SIGTERM
            return value

        with patch.object(resumable.ProgressReporter, "write", request_stop):
            with redirect_stdout(StringIO()), self.assertRaises(resumable.StopRequested):
                resumable.train_command(args)
        progress = json.loads(
            (resumed_output / "progress.json").read_text(encoding="utf-8")
        )
        self.assertEqual(progress["status"], "interrupted")
        self.assertEqual(progress["stage"], stage)
        self.assertEqual(progress["completed_units"], 1)
        self.assertEqual(progress["recoverable_completed_units"], 1)
        self.assertEqual(progress["maximum_redo_units"], 0)

        resumable.reset_stop_request_for_tests()
        with redirect_stdout(StringIO()):
            resumable.train_command(args)
        fresh_args = self.training_args(
            preprocess,
            train,
            validation,
            fresh_output,
            device="cpu",
            epochs=2,
            resume="never",
        )
        with redirect_stdout(StringIO()):
            resumable.train_command(fresh_args)
        resumed_checkpoint = adapter.load_torch_artifact(resumed_output / "checkpoint.pt")
        fresh_checkpoint = adapter.load_torch_artifact(fresh_output / "checkpoint.pt")
        self.assert_model_states_equal(
            resumed_checkpoint, fresh_checkpoint, "original_model_state"
        )
        self.assert_model_states_equal(
            resumed_checkpoint, fresh_checkpoint, "sparse_model_state"
        )
        self.assertEqual(resumed_checkpoint["shortcuts"], fresh_checkpoint["shortcuts"])

    def test_sigterm_resume_matches_uninterrupted_and_sc2_is_checkpointed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preprocess, train, validation = self.prepare_inputs(root)
            resumed_output = root / "resumed"
            fresh_output = root / "fresh"
            args = self.training_args(
                preprocess, train, validation, resumed_output, device="cpu", epochs=2
            )
            original_write = resumable.ProgressReporter.write
            triggered = False

            def request_stop(reporter, **kwargs):
                nonlocal triggered
                value = original_write(reporter, **kwargs)
                if (
                    not triggered
                    and kwargs["status"] == "running"
                    and kwargs["stage"] == "original_training"
                    and kwargs["completed_units"] == 1
                ):
                    triggered = True
                    resumable._STOP_SIGNAL = signal.SIGTERM
                return value

            with patch.object(resumable.ProgressReporter, "write", request_stop):
                with redirect_stdout(StringIO()), self.assertRaises(resumable.StopRequested):
                    resumable.train_command(args)
            interrupted = json.loads(
                (resumed_output / "progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(interrupted["status"], "interrupted")
            self.assertEqual(interrupted["completed_units"], 1)
            self.assertTrue((resumed_output / "resume" / "original.pt").is_file())

            resumable.reset_stop_request_for_tests()
            with redirect_stdout(StringIO()):
                resumable.train_command(args)
            completed = json.loads(
                (resumed_output / "progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(completed["status"], "complete")
            self.assertEqual(completed["device"], "cpu")
            sc2_state = adapter.load_torch_artifact(resumed_output / "resume" / "sc2.pt")
            self.assertTrue(sc2_state["complete"])
            self.assertEqual(sc2_state["completed_destinations"], 2)
            self.assertEqual(len(sc2_state["database"]), 2)

            fresh_args = self.training_args(
                preprocess,
                train,
                validation,
                fresh_output,
                device="cpu",
                epochs=2,
                resume="never",
            )
            with redirect_stdout(StringIO()):
                resumable.train_command(fresh_args)
            resumed_checkpoint = adapter.load_torch_artifact(
                resumed_output / "checkpoint.pt"
            )
            fresh_checkpoint = adapter.load_torch_artifact(fresh_output / "checkpoint.pt")
            self.assert_model_states_equal(
                resumed_checkpoint, fresh_checkpoint, "original_model_state"
            )
            self.assert_model_states_equal(
                resumed_checkpoint, fresh_checkpoint, "sparse_model_state"
            )
            self.assertEqual(resumed_checkpoint["shortcuts"], fresh_checkpoint["shortcuts"])
            self.assertEqual(
                resumed_checkpoint["original_selected_epoch"],
                fresh_checkpoint["original_selected_epoch"],
            )
            self.assertEqual(
                resumed_checkpoint["sparse_selected_epoch"],
                fresh_checkpoint["sparse_selected_epoch"],
            )
            diagnostics = json.loads(
                (resumed_output / "training_diagnostics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(diagnostics),
                {
                    "schema",
                    "method",
                    "adapter_version",
                    "source",
                    "checkpoint",
                    "checkpoint_sha256",
                    "preprocess_dir",
                    "preprocess_artifact_sha256",
                    "train_manifest",
                    "train_dataset",
                    "validation_manifest",
                    "validation_dataset",
                    "split_roles_read",
                    "test_data_read",
                    "graph_identity",
                    "configuration",
                    "original_model",
                    "sc2",
                    "sparse_model",
                    "shortcut_storage",
                    "sparse_training_routes",
                    "sparse_training_routes_dropped_below_two_states",
                    "sparse_training_index_storage_bytes",
                    "total_process_seconds",
                    "peak_rss_kib",
                    "peak_cuda_memory_bytes",
                    "environment",
                },
            )
            validate_training = runpy.run_path(
                str(REPOSITORY_ROOT / "research" / "scripts" / "summarize_route_baselines.py")
            )["validate_training"]
            validate_training(diagnostics, "drncs_lg", "tiny resumable diagnostics")
            self.assertEqual(diagnostics["configuration"]["device"], "cpu")
            provenance = json.loads(
                (resumed_output / "resume_provenance.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                provenance["runner"]["sha256"],
                adapter.sha256_file(BASELINE_DIR / "drncs_lg_resumable.py"),
            )
            self.assertGreaterEqual(
                diagnostics["total_process_seconds"],
                diagnostics["original_model"]["wall_seconds"]
                + diagnostics["sc2"]["wall_seconds"]
                + diagnostics["sparse_model"]["wall_seconds"],
            )
            # The adapter loader accepts the final v2 checkpoint unchanged.
            adapter.load_checkpoint_models(
                resumed_output / "checkpoint.pt", adapter.torch.device("cpu")
            )

    def test_sigterm_after_intermediate_sc2_checkpoint_resumes_identically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.exercise_stage_resume(Path(temporary), "sc2")

    def test_sigterm_after_sparse_epoch_resumes_identically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.exercise_stage_resume(Path(temporary), "sparse_training")

    def test_tiny_cuda_run_covers_original_sc2_and_sparse_stages(self) -> None:
        if not adapter.torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preprocess, train, validation = self.prepare_inputs(root)
            output = root / "cuda"
            fresh_output = root / "cuda-fresh"
            args = self.training_args(
                preprocess, train, validation, output, device="cuda", epochs=1
            )
            original_write = resumable.ProgressReporter.write
            triggered = False

            def stop_inside_sc2(reporter, **kwargs):
                nonlocal triggered
                value = original_write(reporter, **kwargs)
                if (
                    not triggered
                    and kwargs["stage"] == "sc2"
                    and kwargs["completed_units"] == 1
                ):
                    triggered = True
                    resumable._STOP_SIGNAL = signal.SIGTERM
                return value

            with patch.object(resumable.ProgressReporter, "write", stop_inside_sc2):
                with redirect_stdout(StringIO()), self.assertRaises(resumable.StopRequested):
                    resumable.train_command(args)
            resumable.reset_stop_request_for_tests()
            with redirect_stdout(StringIO()):
                resumable.train_command(args)
            fresh_args = self.training_args(
                preprocess,
                train,
                validation,
                fresh_output,
                device="cuda",
                epochs=1,
                resume="never",
            )
            with redirect_stdout(StringIO()):
                resumable.train_command(fresh_args)
            diagnostics = json.loads(
                (output / "training_diagnostics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(diagnostics["configuration"]["device"], "cuda")
            self.assertGreater(diagnostics["peak_cuda_memory_bytes"], 0)
            self.assertEqual(diagnostics["sc2"]["destinations"], 2)
            self.assertEqual(len(diagnostics["original_model"]["history"]), 1)
            self.assertEqual(len(diagnostics["sparse_model"]["history"]), 1)
            resumed_checkpoint = adapter.load_torch_artifact(output / "checkpoint.pt")
            fresh_checkpoint = adapter.load_torch_artifact(fresh_output / "checkpoint.pt")
            self.assert_model_states_equal(
                resumed_checkpoint, fresh_checkpoint, "original_model_state"
            )
            self.assert_model_states_equal(
                resumed_checkpoint, fresh_checkpoint, "sparse_model_state"
            )
            self.assertEqual(resumed_checkpoint["shortcuts"], fresh_checkpoint["shortcuts"])
            prediction_dir = root / "cuda-prediction"
            predict_args = adapter.parse_args(
                [
                    "predict",
                    "--checkpoint",
                    str(output / "checkpoint.pt"),
                    "--map-dir",
                    str(root / "map"),
                    "--dataset-manifest",
                    str(validation),
                    "--predictions",
                    str(prediction_dir / "predictions.jsonl"),
                    "--run-receipt",
                    str(prediction_dir / "run.json"),
                    "--diagnostics",
                    str(prediction_dir / "diagnostics.json"),
                    "--source-revision",
                    "resume-test-revision",
                    "--device",
                    "cuda",
                    "--workers",
                    "2",
                    "--max-steps",
                    "10",
                    "--warmup-repetitions",
                    "0",
                    "--measured-repetitions",
                    "1",
                    "--latency-samples",
                    "1",
                ]
            )
            with redirect_stdout(StringIO()):
                adapter.predict_command(predict_args)
            prediction_diagnostics = json.loads(
                (prediction_dir / "diagnostics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(prediction_diagnostics["environment"]["device"], "cuda")
            self.assertEqual(prediction_diagnostics["samples"], 3)
            self.assertTrue((prediction_dir / "predictions.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
