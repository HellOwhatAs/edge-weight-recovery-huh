import hashlib
import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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


class NeuroMLRAdapterTests(unittest.TestCase):
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
            trips=[MODULE.Trip("test:0", [1])],
        )
        receipt = MODULE.build_run_receipt(
            args,
            dataset,
            SimpleNamespace(identity="graph-sha256"),
            "cpu",
            {"epoch": 5},
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
            {"name": "neuromlr_greedy", "version": "0.1.0"},
        )
        self.assertNotIn("timing", receipt)
        self.assertNotIn("metrics", receipt)


if __name__ == "__main__":
    unittest.main()
