import hashlib
import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SPEC = importlib.util.spec_from_file_location(
    "neuromlr_adapter", Path(__file__).parents[1] / "neuromlr_adapter.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

try:
    MODULE.load_model_dependencies()
except RuntimeError as error:
    MODEL_DEPENDENCIES_AVAILABLE = False
    MODEL_DEPENDENCY_REASON = str(error)
else:
    MODEL_DEPENDENCIES_AVAILABLE = True
    MODEL_DEPENDENCY_REASON = "model dependencies are available"


def graph_for_paths(paths):
    edge_count = max(edge for path in paths for edge in path) + 1
    neighbors = [[] for _ in range(edge_count)]
    for path in paths:
        for left, right in zip(path, path[1:]):
            if right not in neighbors[left]:
                neighbors[left].append(right)
    return SimpleNamespace(tail=[0] * edge_count, neighbors=neighbors)


class NeuroMLRAdapterTests(unittest.TestCase):
    def test_upstream_manifest_matches_adapter_source_lock(self):
        manifest = json.loads(
            (Path(__file__).parents[1] / "upstream.json").read_text()
        )
        self.assertEqual(manifest["commit"], MODULE.UPSTREAM_COMMIT)
        self.assertEqual(manifest["tree"], MODULE.UPSTREAM_TREE)
        self.assertEqual(manifest["files"], MODULE.UPSTREAM_FILE_SHA256)

    def test_common_macro_metrics_score_complete_raw_edge_sequences(self):
        metrics = MODULE.route_metrics(
            [[1, 2, 3], [7, 8]],
            [[1, 2, 4, 5], [7, 8]],
        )
        self.assertEqual(metrics["samples"], 2)
        self.assertAlmostEqual(metrics["edge_precision"], 0.75)
        self.assertAlmostEqual(metrics["edge_recall"], 5 / 6)
        self.assertAlmostEqual(metrics["edge_f1"], 11 / 14)
        self.assertAlmostEqual(metrics["exact_match"], 0.5)
        self.assertAlmostEqual(metrics["edge_jaccard"], 0.7)

    @unittest.skipUnless(MODEL_DEPENDENCIES_AVAILABLE, MODEL_DEPENDENCY_REASON)
    def test_greedy_rollout_starts_at_true_first_and_reaches_true_last_edge(self):
        class FirstValidCandidate:
            def eval(self):
                return self

            def __call__(self, current, destinations, candidates, traffic):
                del current, destinations, traffic
                return MODULE.torch.tensor(
                    [[1.0] if candidate >= 0 else [-100.0] for candidate in candidates]
                )

        trip = MODULE.Trip("validation:0", [0, 1, 2, 3, 4])
        graph = SimpleNamespace(
            padded_neighbors=[[1, -1], [2, -1], [3, -1], [4, -1], [-1, -1]],
            max_neighbors=2,
        )
        generated = MODULE.greedy_paths(
            FirstValidCandidate(), [trip], graph, MODULE.torch.device("cpu")
        )
        self.assertEqual(generated, [[0, 1, 2, 3, 4]])

    @unittest.skipUnless(MODEL_DEPENDENCIES_AVAILABLE, MODEL_DEPENDENCY_REASON)
    def test_greedy_route_partition_preserves_complete_routes(self):
        class HighestValidCandidate:
            def eval(self):
                return self

            def __call__(self, current, destinations, candidates, traffic):
                del current, destinations, traffic
                return MODULE.torch.tensor(
                    [
                        [float(candidate)] if candidate >= 0 else [-100.0]
                        for candidate in candidates
                    ]
                )

        graph = SimpleNamespace(
            padded_neighbors=[
                [1, 2],
                [3, -1],
                [3, -1],
                [4, 5],
                [6, -1],
                [6, -1],
                [7, -1],
                [-1, -1],
            ],
            max_neighbors=2,
        )
        trips = [
            MODULE.Trip("test:0", [0, 2, 3, 5, 6, 7]),
            MODULE.Trip("test:1", [1, 3, 5, 6, 7]),
            MODULE.Trip("test:2", [2, 3, 5, 6, 7]),
            MODULE.Trip("test:3", [3, 5, 6, 7]),
            MODULE.Trip("test:4", [5, 6, 7]),
        ]
        model = HighestValidCandidate()
        device = MODULE.torch.device("cpu")
        full_batch = MODULE.greedy_paths(model, trips, graph, device)
        partitioned = []
        for start in range(0, len(trips), 2):
            partitioned.extend(
                MODULE.greedy_paths(model, trips[start : start + 2], graph, device)
            )
        self.assertEqual(partitioned, full_batch)
        self.assertEqual(full_batch, [trip.edges for trip in trips])

    @unittest.skipUnless(MODEL_DEPENDENCIES_AVAILABLE, MODEL_DEPENDENCY_REASON)
    def test_length_l_training_path_has_l_minus_one_targets(self):
        trip = MODULE.Trip("train:0", [0, 1, 2, 3, 4])
        graph = SimpleNamespace(
            padded_neighbors=[[1, -1], [2, -1], [3, -1], [4, -1], [-1, -1]],
            max_neighbors=2,
        )
        random.seed(1)
        current, destinations, candidates, classes, predictions = MODULE.training_batch(
            [trip], graph, 1
        )
        self.assertEqual(predictions, 4)
        self.assertEqual(len(classes), 4)
        self.assertEqual(len(current), 8)
        self.assertEqual(len(destinations), 8)
        self.assertEqual(len(candidates), 8)

    def test_dijkstra_repetitions_run_every_full_batch_and_retain_component_totals(self):
        calls = []

        def fake_dijkstra_paths(model, trips, graph, chunk_edges):
            del model, trips, graph
            repetition = len(calls)
            calls.append(chunk_edges)
            return (
                [[10, 11, 12]],
                [
                    {
                        "sample_id": "test:0",
                        "embedding_seconds": repetition + 0.1,
                        "transition_scoring_seconds": repetition + 0.2,
                        "dijkstra_seconds": repetition + 0.3,
                    }
                ],
            )

        with mock.patch.object(
            MODULE, "dijkstra_paths", side_effect=fake_dijkstra_paths
        ):
            result = MODULE.repeated_dijkstra_paths(
                model=object(),
                trips=[MODULE.Trip("test:0", [10, 11, 12])],
                graph=object(),
                chunk_edges=2048,
                device=SimpleNamespace(type="cpu"),
                warmup_repetitions=2,
                measured_repetitions=3,
            )

        self.assertEqual(calls, [2048] * 5)
        self.assertEqual(result.generated, [[10, 11, 12]])
        self.assertEqual(len(result.warmup_seconds), 2)
        self.assertEqual(len(result.measured_seconds), 3)
        self.assertEqual(
            result.component_totals_per_repetition,
            [
                {
                    "embedding_seconds": 2.1,
                    "transition_scoring_seconds": 2.2,
                    "dijkstra_seconds": 2.3,
                },
                {
                    "embedding_seconds": 3.1,
                    "transition_scoring_seconds": 3.2,
                    "dijkstra_seconds": 3.3,
                },
                {
                    "embedding_seconds": 4.1,
                    "transition_scoring_seconds": 4.2,
                    "dijkstra_seconds": 4.3,
                },
            ],
        )
        mean_components = MODULE.mean_timing(
            result.component_totals_per_repetition
        )
        self.assertAlmostEqual(mean_components["embedding_seconds"], 3.1)
        self.assertAlmostEqual(mean_components["transition_scoring_seconds"], 3.2)
        self.assertAlmostEqual(mean_components["dijkstra_seconds"], 3.3)

    def test_dijkstra_repetitions_reject_different_paths(self):
        batches = iter(
            [
                ([[1, 2, 3]], []),
                ([[1, 4, 3]], []),
            ]
        )
        with mock.patch.object(MODULE, "dijkstra_paths", side_effect=batches):
            with self.assertRaisesRegex(
                RuntimeError,
                "Dijkstra measured repetition 1 produced different routes",
            ):
                MODULE.repeated_dijkstra_paths(
                    model=object(),
                    trips=[MODULE.Trip("test:0", [1, 2, 3])],
                    graph=object(),
                    chunk_edges=128,
                    device=SimpleNamespace(type="cpu"),
                    warmup_repetitions=0,
                    measured_repetitions=2,
                )

    def test_dijkstra_prediction_defaults_to_one_measured_repetition(self):
        arguments = [
            "ewr-neuromlr",
            "predict",
            "--upstream-dir",
            "upstream",
            "--map-dir",
            "map",
            "--checkpoint",
            "checkpoint.pt",
            "--dataset-manifest",
            "test.manifest.json",
            "--method",
            "dijkstra",
            "--predictions",
            "predictions.jsonl",
            "--run-receipt",
            "run.json",
            "--diagnostics",
            "diagnostics.json",
            "--source-revision",
            "revision",
        ]
        with mock.patch.object(sys, "argv", arguments):
            parsed = MODULE.parse_args()

        self.assertEqual(parsed.warmup_repetitions, 0)
        self.assertEqual(parsed.measured_repetitions, 1)
        self.assertEqual(parsed.route_chunk_size, 0)
        self.assertEqual(parsed.resume, "auto")
        self.assertIsNone(parsed.resume_dir)
        self.assertIsNone(parsed.progress)

    def test_two_edge_path_is_supported_and_one_edge_path_is_rejected(self):
        graph = SimpleNamespace(
            tail=[0, 1],
            head=[1, 2],
        )

        MODULE.validate_trips([MODULE.Trip("two-edges", [0, 1])], graph)

        with self.assertRaisesRegex(RuntimeError, "fewer than 2 roads"):
            MODULE.validate_trips([MODULE.Trip("one-edge", [0])], graph)

    @unittest.skipUnless(MODEL_DEPENDENCIES_AVAILABLE, MODEL_DEPENDENCY_REASON)
    def test_lipschitz_sparse_graph_takes_minimum_parallel_edge(self):
        np = MODULE.np
        graph = SimpleNamespace(
            tail=np.asarray([0, 0, 1]),
            head=np.asarray([1, 1, 2]),
            x=np.zeros(3),
        )
        reverse = MODULE.reverse_sparse_graph_with_minimum_parallel_edges(
            graph, np.asarray([5.0, 2.0, 3.0])
        )
        self.assertEqual(float(reverse[1, 0]), 2.0)
        self.assertEqual(float(reverse[2, 1]), 3.0)

    def test_manifest_descriptor_and_jsonl_are_strict_and_relative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = (
                '{"sample_id":"train:0","original_edge_ids":[1,2,3]}\n'
                '{"sample_id":"train:1","original_edge_ids":[4,5]}\n'
            )
            (root / "records.jsonl").write_text(records, encoding="utf-8")
            encoded_manifest = json.dumps(
                {
                    "schema": MODULE.DATASET_MANIFEST_SCHEMA,
                    "dataset_id": "fixture/train",
                    "network_id": "fixture-roads-v1",
                    "records_schema": MODULE.DATASET_RECORD_SCHEMA,
                    "records_file": "records.jsonl",
                },
                separators=(",", ":"),
            ).encode()
            manifest_path = root / "manifest.json"
            manifest_path.write_bytes(encoded_manifest)

            artifact = MODULE.load_dataset_manifest(manifest_path)

            self.assertEqual(artifact.manifest.dataset_id, "fixture/train")
            self.assertEqual(artifact.manifest.network_id, "fixture-roads-v1")
            self.assertEqual(
                artifact.manifest_sha256,
                hashlib.sha256(encoded_manifest).hexdigest(),
            )
            self.assertEqual(artifact.records_path, (root / "records.jsonl").resolve())
            self.assertEqual(
                artifact.records_sha256,
                hashlib.sha256(records.encode()).hexdigest(),
            )
            self.assertEqual(
                [(trip.sample_id, trip.edges) for trip in artifact.trips],
                [("train:0", [1, 2, 3]), ("train:1", [4, 5])],
            )

            for records_file in [str(root / "records.jsonl"), "../records.jsonl"]:
                descriptor = json.loads(encoded_manifest)
                descriptor["records_file"] = records_file
                manifest_path.write_text(json.dumps(descriptor), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "safe path relative"):
                    MODULE.load_dataset_manifest(manifest_path)

    def test_protocol_reader_rejects_unknown_blank_empty_duplicate_and_non_u32_rows(self):
        invalid_inputs = [
            '{"sample_id":"x","original_edge_ids":[1],"method":"x"}\n',
            '{"sample_id":"x","original_edge_ids":[1,2]}\n\n',
            "",
            '{"sample_id":"x","original_edge_ids":[]}\n',
            '{"sample_id":"x","original_edge_ids":[1]}\n',
            (
                '{"sample_id":"x","original_edge_ids":[1,2]}\n'
                '{"sample_id":"x","original_edge_ids":[2,3]}\n'
            ),
            '{"sample_id":"x","original_edge_ids":[true,2]}\n',
            '{"sample_id":"x","original_edge_ids":[4294967296,2]}\n',
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            for value in invalid_inputs:
                with self.subTest(value=value):
                    path.write_text(value, encoding="utf-8")
                    with self.assertRaises(RuntimeError):
                        MODULE.load_dataset_records(path)

    def test_manifest_descriptor_rejects_unknown_fields_and_wrong_schemas(self):
        base = {
            "schema": MODULE.DATASET_MANIFEST_SCHEMA,
            "dataset_id": "fixture/test",
            "network_id": "fixture-roads-v1",
            "records_schema": MODULE.DATASET_RECORD_SCHEMA,
            "records_file": "records.jsonl",
        }
        invalid_descriptors = [
            {**base, "method": "neuromlr"},
            {**base, "schema": "ewr.dataset-manifest/v2"},
            {**base, "records_schema": "ewr.dataset-record/v2"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            for descriptor in invalid_descriptors:
                with self.subTest(descriptor=descriptor):
                    path.write_text(json.dumps(descriptor), encoding="utf-8")
                    with self.assertRaises(RuntimeError):
                        MODULE.load_dataset_manifest(path)

    def test_predictions_contain_only_protocol_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory) / "predictions.jsonl"
            trips = [MODULE.Trip("test:7", [1, 2, 3])]

            MODULE.write_prediction_rows(predictions, trips, [[1, 4, 3]])

            self.assertEqual(
                predictions.read_text(encoding="utf-8"),
                '{"sample_id":"test:7","predicted_edge_ids":[1,4,3]}\n',
            )
            self.assertEqual(
                set(json.loads(predictions.read_text()).keys()),
                {"sample_id", "predicted_edge_ids"},
            )

    def test_chunked_greedy_prediction_is_byte_identical_to_legacy_output(self):
        trips = [
            MODULE.Trip(f"test:{index}", [index, index + 10])
            for index in range(5)
        ]

        def deterministic_paths(model, batch, graph, device):
            del model, graph, device
            return [
                [
                    trip.edges[0],
                    100 + int(trip.sample_id.split(":")[1]),
                    trip.edges[-1],
                ]
                for trip in batch
            ]

        expected_paths = deterministic_paths(None, trips, None, None)
        graph = graph_for_paths(expected_paths)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "legacy.jsonl"
            chunked = root / "chunked.jsonl"
            MODULE.write_prediction_rows(
                legacy,
                trips,
                expected_paths,
            )
            with mock.patch.object(
                MODULE, "greedy_paths", side_effect=deterministic_paths
            ), mock.patch.object(MODULE, "synchronize_device"):
                result = MODULE.run_chunked_greedy_prediction(
                    model=object(),
                    trips=trips,
                    graph=graph,
                    device=SimpleNamespace(type="cpu"),
                    predictions=chunked,
                    chunk_size=2,
                    resume_dir=root / "resume",
                    progress_path=root / "resume" / "progress.json",
                    resume_mode="auto",
                    binding={
                        "fixture": "byte-equivalence-v1",
                        "predictions_path": str(chunked.resolve()),
                    },
                )

            self.assertEqual(chunked.read_bytes(), legacy.read_bytes())
            self.assertEqual(result.completed_chunks, 3)
            self.assertEqual(result.resumed_chunks, 0)
            self.assertEqual(result.endpoint_failures, 0)
            self.assertEqual(
                result.output_sha256,
                hashlib.sha256(legacy.read_bytes()).hexdigest(),
            )
            progress = json.loads(result.progress_path.read_text())
            self.assertEqual(progress["status"], "complete")
            self.assertEqual(progress["completed_samples"], 5)
            self.assertEqual(
                progress["estimated_remaining_prediction_seconds"], 0.0
            )
            self.assertGreater(progress["peak_rss_kib"], 0)
            self.assertEqual(progress["peak_cuda_memory_bytes"], 0)
            self.assertEqual(
                [row["samples"] for row in progress["completed_chunks"]],
                [2, 2, 1],
            )
            self.assertFalse(list((root / "resume").rglob("*.tmp")))

    def test_chunked_greedy_prediction_resumes_only_uncommitted_chunks(self):
        trips = [
            MODULE.Trip(f"test:{index}", [index, index + 20])
            for index in range(5)
        ]
        calls = []

        def interrupted_paths(model, batch, graph, device):
            del model, graph, device
            calls.append([trip.sample_id for trip in batch])
            if len(calls) == 2:
                raise RuntimeError("simulated interruption")
            return [trip.edges.copy() for trip in batch]

        def completed_paths(model, batch, graph, device):
            del model, graph, device
            calls.append([trip.sample_id for trip in batch])
            return [trip.edges.copy() for trip in batch]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = dict(
                model=object(),
                trips=trips,
                graph=graph_for_paths([trip.edges for trip in trips]),
                device=SimpleNamespace(type="cpu"),
                predictions=root / "predictions.jsonl",
                chunk_size=2,
                resume_dir=root / "resume",
                progress_path=root / "resume" / "progress.json",
                resume_mode="auto",
                binding={
                    "fixture": "resume-v1",
                    "checkpoint_sha256": "a" * 64,
                    "predictions_path": str((root / "predictions.jsonl").resolve()),
                },
            )
            with mock.patch.object(
                MODULE, "greedy_paths", side_effect=interrupted_paths
            ), mock.patch.object(MODULE, "synchronize_device"):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    MODULE.run_chunked_greedy_prediction(**arguments)

            progress = json.loads(arguments["progress_path"].read_text())
            self.assertEqual(progress["status"], "running")
            self.assertEqual(progress["completed_samples"], 2)
            self.assertEqual(len(progress["completed_chunks"]), 1)

            calls.clear()
            with mock.patch.object(
                MODULE, "greedy_paths", side_effect=completed_paths
            ), mock.patch.object(MODULE, "synchronize_device"):
                result = MODULE.run_chunked_greedy_prediction(**arguments)
            self.assertEqual(calls, [["test:2", "test:3"], ["test:4"]])
            self.assertEqual(result.resumed_chunks, 1)
            progress = json.loads(arguments["progress_path"].read_text())
            self.assertEqual(progress["sessions"], 2)
            self.assertEqual(progress["status"], "complete")

            arguments["predictions"].unlink()
            calls.clear()
            with mock.patch.object(
                MODULE, "greedy_paths", side_effect=completed_paths
            ), mock.patch.object(MODULE, "synchronize_device"):
                recovered = MODULE.run_chunked_greedy_prediction(**arguments)
            self.assertEqual(calls, [])
            self.assertTrue(arguments["predictions"].is_file())
            self.assertEqual(recovered.resumed_chunks, 3)

            changed_binding = {
                **arguments["binding"],
                "checkpoint_sha256": "b" * 64,
            }
            changed = {**arguments, "binding": changed_binding}
            with self.assertRaisesRegex(RuntimeError, "resume binding differs"):
                MODULE.run_chunked_greedy_prediction(**changed)

    def test_chunked_greedy_prediction_rejects_corrupt_committed_shard(self):
        trips = [
            MODULE.Trip("test:0", [1, 2]),
            MODULE.Trip("test:1", [3, 4]),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = dict(
                model=object(),
                trips=trips,
                graph=graph_for_paths([trip.edges for trip in trips]),
                device=SimpleNamespace(type="cpu"),
                predictions=root / "predictions.jsonl",
                chunk_size=1,
                resume_dir=root / "resume",
                progress_path=root / "resume" / "progress.json",
                resume_mode="auto",
                binding={
                    "fixture": "corruption-v1",
                    "predictions_path": str((root / "predictions.jsonl").resolve()),
                },
            )
            with mock.patch.object(
                MODULE,
                "greedy_paths",
                side_effect=lambda model, batch, graph, device: [
                    trip.edges.copy() for trip in batch
                ],
            ), mock.patch.object(MODULE, "synchronize_device"):
                MODULE.run_chunked_greedy_prediction(**arguments)
            shard = root / "resume" / "parts" / "part-000000.jsonl"
            shard.write_text(
                '{"sample_id":"test:0","predicted_edge_ids":[1,4]}\n',
                encoding="utf-8",
            )
            progress = json.loads(arguments["progress_path"].read_text())
            progress["completed_chunks"][0]["sha256"] = hashlib.sha256(
                shard.read_bytes()
            ).hexdigest()
            arguments["progress_path"].write_text(json.dumps(progress))
            with self.assertRaisesRegex(RuntimeError, "illegal directed transition"):
                MODULE.run_chunked_greedy_prediction(**arguments)

    def test_prediction_shard_rejects_wrong_start_and_out_of_graph_edge(self):
        trip = MODULE.Trip("test:0", [1, 2])
        graph = graph_for_paths([trip.edges])
        invalid_rows = [
            (
                '{"sample_id":"test:0","predicted_edge_ids":[2]}\n',
                "changes the fixed first edge",
            ),
            (
                '{"sample_id":"test:0","predicted_edge_ids":[1,99]}\n',
                "outside the raw graph",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "part.jsonl"
            for row, message in invalid_rows:
                with self.subTest(message=message):
                    shard.write_text(row, encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, message):
                        MODULE.validate_prediction_part(
                            shard,
                            [trip],
                            graph,
                            hashlib.sha256(shard.read_bytes()).hexdigest(),
                        )

    def test_run_receipt_has_exact_v1_shape_and_no_diagnostics(self):
        args = SimpleNamespace(
            method="greedy",
            checkpoint=Path("checkpoint.pt"),
            seed=11,
            embedding_size=8,
            hidden_size=16,
            mlp_hidden_layers=2,
            gnn_layers=1,
            score_edge_chunk=32,
            warmup_repetitions=0,
            measured_repetitions=1,
            route_chunk_size=0,
            source_revision="adapter-source-revision",
        )
        dataset = MODULE.DatasetArtifact(
            manifest=MODULE.DatasetManifest(
                MODULE.DATASET_MANIFEST_SCHEMA,
                "fixture/test",
                "fixture-roads-v1",
                MODULE.DATASET_RECORD_SCHEMA,
                "test.jsonl",
            ),
            manifest_path=Path("manifest.json"),
            manifest_sha256="a" * 64,
            records_path=Path("test.jsonl"),
            records_sha256="b" * 64,
            trips=[MODULE.Trip("test:0", [1])],
        )
        receipt = MODULE.build_run_receipt(
            args,
            dataset,
            SimpleNamespace(
                identity="graph-sha256", coordinate_identity="coordinate-sha256"
            ),
            "cpu",
            {"epoch": 5},
            "c" * 64,
            environment={
                "device": "cpu",
                "numpy": "test-numpy",
                "python": "test-python",
                "torch": "test-torch",
            },
        )

        self.assertEqual(
            set(receipt),
            {
                "schema",
                "method",
                "dataset_id",
                "dataset_manifest_sha256",
                "prediction_records_schema",
                "configuration",
                "source_revision",
                "environment",
            },
        )
        self.assertEqual(receipt["schema"], MODULE.RUN_RECEIPT_SCHEMA)
        self.assertIsInstance(receipt["configuration"], dict)
        self.assertIsInstance(receipt["environment"], dict)
        self.assertTrue(receipt["environment"])
        self.assertTrue(
            all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in receipt["environment"].items()
            )
        )
        self.assertEqual(len(receipt["dataset_manifest_sha256"]), 64)
        self.assertEqual(
            receipt["prediction_records_schema"], MODULE.PREDICTION_RECORD_SCHEMA
        )
        self.assertEqual(
            receipt["method"],
            {"name": "neuromlr_greedy", "version": "0.2.0"},
        )
        self.assertEqual(receipt["configuration"]["checkpoint_sha256"], "c" * 64)
        self.assertEqual(
            receipt["configuration"]["dataset_records_sha256"], "b" * 64
        )
        self.assertEqual(receipt["configuration"]["route_chunk_size"], 0)
        self.assertNotIn("timing", receipt)
        self.assertNotIn("metrics", receipt)


if __name__ == "__main__":
    unittest.main()
