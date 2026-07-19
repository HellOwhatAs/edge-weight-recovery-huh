import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SPEC = importlib.util.spec_from_file_location(
    "drpk_static_adapter", Path(__file__).parents[1] / "drpk_static_adapter.py"
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


def branching_graph():
    # 0 branches through 1/3 or 2/4 before reaching destination road 5.
    return MODULE.StaticGraph(
        tail=[0, 1, 1, 2, 3, 4],
        head=[1, 2, 3, 4, 4, 5],
        node_x=[0.0, 1.0, 2.0, 1.0, 3.0, 4.0],
        node_y=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        neighbors=[(1, 2), (3,), (4,), (5,), (5,), ()],
        identity="synthetic-branching-v1",
    )


def small_ksd_configuration():
    return {
        "schema": MODULE.KSD_MODEL_SCHEMA,
        "embedding_size": 4,
        "query_hidden_size": 8,
        "representation_size": 6,
        "candidate_embedding_size": 3,
        "candidate_hidden_size": 7,
        "dropout": 0.0,
        "fixed_source_offset": MODULE.FIXED_SOURCE_OFFSET,
        "fixed_destination_offset": MODULE.FIXED_DESTINATION_OFFSET,
    }


class DRPKStaticAdapterTests(unittest.TestCase):
    def test_default_cpu_parallelism_is_sixteen(self):
        args = MODULE.parse_args(
            [
                "predict",
                "--preprocess-dir",
                "preprocess",
                "--dataset-manifest",
                "test.manifest.json",
                "--method",
                "drp_tp",
                "--predictions",
                "predictions.jsonl",
                "--run-receipt",
                "run.json",
                "--diagnostics",
                "diagnostics.json",
                "--source-revision",
                "revision",
            ]
        )
        self.assertEqual(args.workers, 16)
        self.assertEqual(args.inference_batch_size, 32)
        self.assertEqual(args.measured_repetitions, 1)

        train_args = MODULE.parse_args(
            [
                "train",
                "--preprocess-dir",
                "preprocess",
                "--validation-manifest",
                "validation.manifest.json",
                "--output-dir",
                "train",
                "--source-revision",
                "revision",
            ]
        )
        self.assertEqual(train_args.epochs, 300)
        self.assertEqual(train_args.batch_size, 8192)
        self.assertEqual(train_args.microbatch_size, 512)
        self.assertEqual(train_args.query_hidden_size, 2048)
        self.assertEqual(train_args.representation_size, 256)
        self.assertEqual(train_args.candidate_embedding_size, 64)
        self.assertEqual(train_args.candidate_hidden_size, 512)
        self.assertEqual(train_args.early_stop_learning_rate, 0.0)

    def test_sparse_da_counts_every_ordered_precedence_pair(self):
        trips = [
            MODULE.Trip("train:0", [0, 1, 2, 3]),
            MODULE.Trip("train:1", [0, 2, 3]),
        ]
        da, popularity, transitions, pair_events = MODULE.build_training_statistics(
            trips, edge_count=4, workers=2
        )
        expected = {
            (0, 1): 1,
            (0, 2): 2,
            (0, 3): 2,
            (1, 2): 1,
            (1, 3): 1,
            (2, 3): 2,
        }
        self.assertEqual(pair_events, 9)
        self.assertEqual(popularity, [2, 1, 2, 2])
        self.assertEqual(transitions, {(0, 1): 1, (1, 2): 1, (2, 3): 2, (0, 2): 1})
        for pair, count in expected.items():
            self.assertEqual(da.value(*pair), count)
        self.assertEqual(da.candidate_pool(0, 3, 100), [(2, 2), (1, 1)])

    @unittest.skipUnless(MODEL_DEPENDENCIES_AVAILABLE, MODEL_DEPENDENCY_REASON)
    def test_candidate_rows_are_compact_memmaps_with_paper_exponential_weights(self):
        da = MODULE.SparseDA.from_counts(
            4,
            {
                (0, 0): 100,
                (0, 3): 100,
                (3, 3): 99,
                (0, 1): 5,
                (1, 3): 5,
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "candidates"
            rows = MODULE.build_candidate_rows(
                [MODULE.Trip("train:0", [0, 1, 2, 3])],
                da,
                pool_size=3,
                positive_fraction=0.2,
                storage_dir=artifact,
            )
            self.assertIsInstance(rows["candidates"], MODULE.np.memmap)
            self.assertEqual(rows["candidates"].dtype, MODULE.np.dtype("int32"))
            self.assertEqual(rows["labels"].dtype, MODULE.np.dtype("uint8"))
            self.assertEqual(rows["weights"].dtype, MODULE.np.dtype("float32"))
            self.assertEqual(rows["candidates"].tolist(), [[0, 3, 1]])
            self.assertEqual(rows["labels"].tolist(), [[0, 0, 1]])
            self.assertAlmostEqual(float(rows["weights"][0, 2]), math.e, places=6)
            MODULE.save_candidate_rows(artifact, rows)
            loaded = MODULE.load_candidate_rows(artifact, expected_pool_size=3)
            self.assertIsInstance(loaded["candidates"], MODULE.np.memmap)
            self.assertEqual(loaded["candidates"].tolist(), [[0, 3, 1]])
            metadata = json.loads((artifact / "metadata.json").read_text())
            self.assertEqual(metadata["storage"], "npy_memmap_no_full_training_copy")

    def test_static_popularity_replicates_all_48_slots(self):
        counts = [0, 2, 4]
        slots = MODULE.replicated_popularity(counts)
        self.assertEqual(len(slots), 48)
        self.assertTrue(all(slot == counts for slot in slots))
        self.assertEqual(MODULE.normalize_global_popularity(counts), [0.0, 0.5, 1.0])

    def test_drp_uses_popularity_only_for_positive_association_tie(self):
        graph = branching_graph()
        da = MODULE.SparseDA.from_counts(
            graph.edge_count,
            {(1, 5): 4, (2, 5): 4, (3, 5): 4, (4, 5): 4},
        )
        route = MODULE.plan_drp_leg(
            0,
            5,
            graph,
            da,
            [0.0, 0.1, 0.9, 0.0, 0.0, 0.0],
            300,
            use_popularity=True,
        )
        self.assertEqual(route, [0, 2, 4, 5])

        no_support = MODULE.SparseDA.from_counts(graph.edge_count, {})
        angular_route = MODULE.plan_drp_leg(
            0,
            5,
            graph,
            no_support,
            [0.0, 0.1, 0.9, 0.0, 0.0, 0.0],
            300,
            use_popularity=True,
        )
        self.assertEqual(angular_route, [0, 1, 3, 5])

    def test_drp_tp_uses_first_raw_edge_order_for_tie(self):
        graph = branching_graph()
        da = MODULE.SparseDA.from_counts(
            graph.edge_count,
            {(1, 5): 4, (2, 5): 4, (3, 5): 4, (4, 5): 4},
        )
        routes = MODULE.drp_tp_routes(
            [MODULE.Trip("test:0", [0, 2, 4, 5])],
            graph,
            da,
            300,
        )
        self.assertEqual(routes, [[0, 1, 3, 5]])

    def test_endpoint_only_query_is_independent_of_truth_interior(self):
        graph = branching_graph()
        da = MODULE.SparseDA.from_counts(
            graph.edge_count,
            {(1, 5): 4, (2, 5): 4, (3, 5): 4, (4, 5): 4},
        )
        trips = [
            MODULE.Trip("test:a", [0, 1, 3, 5]),
            MODULE.Trip("test:b", [0, 2, 4, 5]),
        ]
        routes = MODULE.drp_tp_routes(
            trips, graph, da, 300
        )
        self.assertEqual(routes[0], routes[1])

    def test_failure_returns_partial_route_without_destination_repair(self):
        graph = MODULE.StaticGraph(
            tail=[0, 1, 2],
            head=[1, 2, 3],
            node_x=[0.0, 1.0, 2.0, 3.0],
            node_y=[0.0, 0.0, 0.0, 0.0],
            neighbors=[(), (2,), ()],
            identity="dead-end",
        )
        da = MODULE.SparseDA.from_counts(3, {})
        self.assertEqual(
            MODULE.plan_drp_leg(
                0, 2, graph, da, [0.0, 0.0, 0.0], 300, use_popularity=True
            ),
            [0],
        )
        self.assertEqual(
            MODULE.plan_via_key(0, 1, 2, graph, da, [0.0, 0.0, 0.0], 300),
            [0],
        )

    def test_drp_tp_rejects_a_ksd_checkpoint_before_artifact_loading(self):
        args = SimpleNamespace(
            warmup_repetitions=0,
            measured_repetitions=1,
            inference_batch_size=32,
            source_revision="revision",
            method="drp_tp",
            checkpoint=Path("must-not-be-read.pt"),
        )
        with self.assertRaisesRegex(RuntimeError, "must not receive"):
            MODULE.predict_command(args)

    @unittest.skipUnless(MODEL_DEPENDENCIES_AVAILABLE, MODEL_DEPENDENCY_REASON)
    def test_drp_tp_environment_reports_only_loaded_base_dependencies(self):
        self.assertEqual(
            set(MODULE.base_environment()), {"device", "numpy", "python"}
        )

    def test_each_via_key_leg_has_its_own_length_cap(self):
        graph = MODULE.StaticGraph(
            tail=[0, 1, 2],
            head=[1, 2, 3],
            node_x=[0.0, 1.0, 2.0, 3.0],
            node_y=[0.0, 0.0, 0.0, 0.0],
            neighbors=[(1,), (2,), ()],
            identity="chain",
        )
        da = MODULE.SparseDA.from_counts(3, {(1, 2): 1})
        route = MODULE.plan_via_key(
            0, 1, 2, graph, da, [0.0, 0.0, 0.0], max_length=2
        )
        self.assertEqual(route, [0, 1, 2])

    def test_manifest_descriptor_and_records_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = '{"sample_id":"train:0","original_edge_ids":[1,2]}\n'
            (root / "records.jsonl").write_text(records, encoding="utf-8")
            manifest = {
                "schema": MODULE.DATASET_MANIFEST_SCHEMA,
                "dataset_id": "fixture/train",
                "network_id": "fixture-roads-v1",
                "records_schema": MODULE.DATASET_RECORD_SCHEMA,
                "records_file": "records.jsonl",
            }
            encoded = json.dumps(manifest, separators=(",", ":")).encode()
            path = root / "manifest.json"
            path.write_bytes(encoded)
            artifact = MODULE.load_dataset_manifest(path)
            self.assertEqual(artifact.manifest_sha256, hashlib.sha256(encoded).hexdigest())
            self.assertEqual(artifact.trips, [MODULE.Trip("train:0", [1, 2])])

            manifest["records_file"] = "../records.jsonl"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "safe path relative"):
                MODULE.load_dataset_manifest(path)

    def test_predictions_have_only_common_protocol_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            trips = [MODULE.Trip("test:0", [1, 2])]
            MODULE.write_prediction_rows(path, trips, [[1, 3, 2]])
            row = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(row), {"sample_id", "predicted_edge_ids"})
            self.assertEqual(row["predicted_edge_ids"], [1, 3, 2])

    def test_run_receipt_has_exact_version_one_top_level_shape(self):
        manifest = MODULE.DatasetManifest(
            MODULE.DATASET_MANIFEST_SCHEMA,
            "fixture/test",
            "fixture-roads-v1",
            MODULE.DATASET_RECORD_SCHEMA,
            "test.jsonl",
        )
        dataset = MODULE.DatasetArtifact(
            manifest, Path("test.manifest.json"), "a" * 64, "b" * 64, []
        )
        graph = branching_graph()
        args = SimpleNamespace(
            method="drp_tp",
            checkpoint=None,
            source_revision="revision",
            seed=MODULE.DEFAULT_SEED,
            workers=16,
            inference_batch_size=32,
            warmup_repetitions=0,
            measured_repetitions=1,
            device="cpu",
        )
        receipt = MODULE.build_run_receipt(
            args,
            dataset,
            graph,
            SimpleNamespace(type="cpu"),
            checkpoint=None,
            preprocess_configuration={
                "adaptation": "time_collapsed_global_popularity_replicated_48_slots",
                "source": MODULE.adapter_source_identity("revision"),
            },
            environment={"python": "fixture", "device": "cpu"},
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
        self.assertFalse(receipt["configuration"]["truth_repair"])
        self.assertEqual(receipt["configuration"]["inference_batch_size"], 32)

    def test_common_route_metrics_score_complete_raw_edge_sequences(self):
        metrics = MODULE.route_metrics(
            [[1, 2, 3], [7, 8]],
            [[1, 2, 4, 5], [7, 8]],
        )
        self.assertEqual(metrics["samples"], 2)
        self.assertAlmostEqual(metrics["edge_precision"], 0.75)
        self.assertAlmostEqual(metrics["edge_recall"], 5 / 6)
        self.assertAlmostEqual(metrics["edge_f1"], 11 / 14)
        self.assertAlmostEqual(metrics["edge_jaccard"], 0.7)
        self.assertAlmostEqual(metrics["exact_match"], 0.5)

    def test_mean_timing_requires_matching_component_boundaries(self):
        mean = MODULE.mean_timing(
            [
                {"pool": 1.0, "planner": 3.0},
                {"pool": 3.0, "planner": 5.0},
            ]
        )
        self.assertEqual(mean, {"planner": 4.0, "pool": 2.0})
        with self.assertRaisesRegex(RuntimeError, "different components"):
            MODULE.mean_timing([{"pool": 1.0}, {"planner": 1.0}])

    @unittest.skipUnless(MODEL_DEPENDENCIES_AVAILABLE, MODEL_DEPENDENCY_REASON)
    def test_node2vec_walks_are_deterministic_across_worker_counts(self):
        graph = branching_graph()
        arguments = dict(
            graph=graph,
            walk_length=5,
            walks_per_edge=3,
            p=1.0,
            q=1.0,
            seed=123,
        )
        one = MODULE.generate_node2vec_walks(workers=1, **arguments)
        four = MODULE.generate_node2vec_walks(workers=4, **arguments)
        self.assertEqual(one, four)

    @unittest.skipUnless(MODEL_DEPENDENCIES_AVAILABLE, MODEL_DEPENDENCY_REASON)
    def test_external_da_merge_matches_in_memory_counts_without_dense_matrix(self):
        trips = [
            MODULE.Trip("train:0", [0, 1, 2, 3]),
            MODULE.Trip("train:1", [0, 2, 3]),
        ]
        expected, expected_popularity, expected_transitions, expected_events = (
            MODULE.build_training_statistics(trips, edge_count=4, workers=2)
        )
        with tempfile.TemporaryDirectory() as directory:
            actual, popularity, transitions, events, diagnostics = (
                MODULE.build_training_statistics_external(
                    trips,
                    edge_count=4,
                    output_dir=Path(directory),
                    pair_chunk_size=3,
                )
            )
            self.assertEqual(events, expected_events)
            self.assertEqual(popularity, expected_popularity)
            self.assertEqual(
                MODULE.build_global_popularity(trips, edge_count=4),
                expected_popularity,
            )
            self.assertEqual(transitions, expected_transitions)
            self.assertGreater(diagnostics["runs"], 1)
            self.assertEqual(actual.nonzero, expected.nonzero)
            self.assertEqual(
                actual.candidate_pool(0, 3, 100),
                expected.candidate_pool(0, 3, 100),
            )
            for source in range(4):
                for destination in range(4):
                    self.assertEqual(
                        actual.value(source, destination),
                        expected.value(source, destination),
                    )
            metadata = json.loads(
                (Path(directory) / "da" / "metadata.json").read_text()
            )
            self.assertEqual(metadata["storage"], "memory_mapped_csr_and_csc")

            fixture_tail = [0, 1, 2, 3]
            fixture_head = [1, 2, 3, 4]
            fixture_x = [0.0, 1.0, 2.0, 3.0, 4.0]
            fixture_y = [0.0, 0.0, 0.0, 0.0, 0.0]
            graph = MODULE.StaticGraph(
                tail=fixture_tail,
                head=fixture_head,
                node_x=fixture_x,
                node_y=fixture_y,
                neighbors=[(1,), (2,), (3,), ()],
                identity=MODULE.graph_identity(
                    fixture_tail, fixture_head, fixture_x, fixture_y
                ),
            )
            root = Path(directory)
            MODULE.save_graph(root / "graph.npz", graph)
            manifest_path, manifest_sha256 = (
                MODULE.write_routing_artifact_manifest(
                    root, "fixture-roads-v1", graph.identity
                )
            )
            routing_configuration = {
                "schema": MODULE.ROUTING_PREPROCESS_SCHEMA,
                "source": MODULE.adapter_source_identity("fixture"),
                "provenance": {"official_commit": MODULE.OFFICIAL_COMMIT},
                "network_id": "fixture-roads-v1",
                "graph_identity": graph.identity,
                "routing_artifacts": {
                    "schema": MODULE.ROUTING_ARTIFACT_MANIFEST_SCHEMA,
                    "path": manifest_path.name,
                    "sha256": manifest_sha256,
                },
            }
            MODULE.write_json(
                root / "routing-configuration.json", routing_configuration
            )
            loaded_configuration, loaded_graph, loaded_da = (
                MODULE._load_routing_preprocess(root)
            )
            self.assertEqual(loaded_configuration, routing_configuration)
            self.assertEqual(loaded_graph.identity, graph.identity)
            self.assertEqual(loaded_da.nonzero, expected.nonzero)

            raw_path = root / "da" / "row_values.u32"
            raw = bytearray(raw_path.read_bytes())
            raw[0] ^= 1
            raw_path.write_bytes(raw)
            with self.assertRaisesRegex(RuntimeError, "differs from manifest"):
                MODULE._load_routing_preprocess(root)

    @unittest.skipUnless(MODEL_DEPENDENCIES_AVAILABLE, MODEL_DEPENDENCY_REASON)
    def test_ksd_implements_paper_equations_six_through_nine(self):
        np = MODULE.np
        torch = MODULE.torch
        graph = branching_graph()
        configuration = small_ksd_configuration()
        model = MODULE.build_ksd_model(
            graph,
            np.arange(graph.edge_count * 4, dtype=np.float32).reshape(
                graph.edge_count, 4
            ),
            np.linspace(0, 1, graph.edge_count, dtype=np.float32),
            configuration,
            torch.device("cpu"),
        )
        model.eval()
        state_fields = set(model.state_dict())
        self.assertIn("candidate_embedding.weight", state_fields)
        self.assertIn("candidate_encoder.0.weight", state_fields)
        self.assertIn("candidate_encoder.2.weight", state_fields)
        self.assertNotIn("segment_classifier.weight", state_fields)
        self.assertFalse(any("edge_source_xy" in name for name in state_fields))
        self.assertEqual(
            [type(layer).__name__ for layer in model.query_encoder],
            ["Linear", "ReLU", "Linear"],
        )
        self.assertEqual(
            [type(layer).__name__ for layer in model.candidate_encoder],
            ["Linear", "ReLU", "Linear"],
        )
        source = torch.tensor([0, 0])
        destination = torch.tensor([5, 5])
        candidates = torch.tensor([[1, 2], [1, 2]])
        with torch.no_grad():
            logits = model(source, destination, candidates)
        self.assertTrue(torch.equal(logits[0], logits[1]))

        # A single paper-objective step must reach both endpoint embeddings,
        # the candidate embedding, and every MLP affine layer.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.1)
        model.zero_grad(set_to_none=True)
        logits = model(source, destination, candidates)
        logits.sum().backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0, name)

    @unittest.skipUnless(MODEL_DEPENDENCIES_AVAILABLE, MODEL_DEPENDENCY_REASON)
    def test_small_graph_ksd_optimization_and_checkpoint_binding(self):
        np = MODULE.np
        torch = MODULE.torch
        graph = branching_graph()
        configuration = small_ksd_configuration()
        model = MODULE.build_ksd_model(
            graph,
            np.zeros((graph.edge_count, 4), dtype=np.float32),
            np.zeros(graph.edge_count, dtype=np.float32),
            configuration,
            torch.device("cpu"),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        arrays = {
            "source": np.asarray([0, 0, 0, 0], dtype=np.int64),
            "destination": np.asarray([5, 5, 5, 5], dtype=np.int64),
            "candidates": np.asarray([[1, 2]] * 4, dtype=np.int64),
            "labels": np.asarray([[1, 0]] * 4, dtype=np.uint8),
            "weights": np.asarray([[math.e, 1.0]] * 4, dtype=np.float32),
        }
        model.train()
        logits = model(
            torch.tensor(arrays["source"]),
            torch.tensor(arrays["destination"]),
            torch.tensor(arrays["candidates"]),
        )
        loss = MODULE.F.binary_cross_entropy_with_logits(
            logits, torch.tensor(arrays["labels"]).float()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        metrics = MODULE.evaluate_ksd_candidates(
            model, arrays, batch_size=4, device=torch.device("cpu")
        )
        self.assertEqual(metrics["rows"], 4)
        self.assertTrue(math.isfinite(metrics["mean_weighted_bce"]))
        self.assertEqual(
            metrics["loss_normalization"],
            "sum_candidates_then_mean_routes_eq3_4",
        )
        with torch.no_grad():
            logits = model(
                torch.tensor(arrays["source"]),
                torch.tensor(arrays["destination"]),
                torch.tensor(arrays["candidates"]),
            )
            losses = MODULE.F.binary_cross_entropy_with_logits(
                logits, torch.tensor(arrays["labels"]).float(), reduction="none"
            )
            expected = float(
                (losses * torch.tensor(arrays["weights"])).sum() / 4
            )
        self.assertAlmostEqual(metrics["mean_weighted_bce"], expected, places=6)

        training_configuration = {
            "graph_identity": graph.identity,
            "network_id": "fixture-roads-v1",
            "preprocess_configuration_sha256": "a" * 64,
            "source": MODULE.adapter_source_identity("fixture"),
            "model": configuration,
        }
        preprocess_configuration = {
            "network_id": "fixture-roads-v1",
            "source": MODULE.adapter_source_identity("fixture"),
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            MODULE.save_checkpoint(
                checkpoint, model, optimizer, 1, training_configuration
            )
            loaded, payload = MODULE._load_checkpoint(
                checkpoint,
                preprocess_configuration,
                "a" * 64,
                graph,
                np.zeros((graph.edge_count, 4), dtype=np.float32),
                np.zeros(graph.edge_count, dtype=np.float32),
                torch.device("cpu"),
            )
            self.assertEqual(payload["epoch"], 1)
            self.assertFalse(loaded.training)
            with self.assertRaisesRegex(RuntimeError, "preprocess configuration"):
                MODULE._load_checkpoint(
                    checkpoint,
                    preprocess_configuration,
                    "b" * 64,
                    graph,
                    np.zeros((graph.edge_count, 4), dtype=np.float32),
                    np.zeros(graph.edge_count, dtype=np.float32),
                    torch.device("cpu"),
                )


if __name__ == "__main__":
    unittest.main()
