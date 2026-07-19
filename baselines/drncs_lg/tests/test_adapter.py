from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch


BASELINE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE_DIR))
import drncs_lg_adapter as adapter  # noqa: E402


HAS_ARRAY_STACK = all(
    importlib.util.find_spec(name) is not None for name in ("numpy", "shapefile", "torch")
)


class ProtocolTests(unittest.TestCase):
    def write_dataset(self, root: Path, rows: list[dict], dataset_id: str = "tiny") -> Path:
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
                    "network_id": "tiny-network",
                    "records_schema": adapter.DATASET_RECORD_SCHEMA,
                    "records_file": "records.jsonl",
                },
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        return manifest

    def test_strict_manifest_and_prediction_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.write_dataset(
                root,
                [
                    {"sample_id": "a", "original_edge_ids": [0, 2]},
                    {"sample_id": "b", "original_edge_ids": [1, 2]},
                ],
            )
            dataset = adapter.load_dataset_manifest(manifest)
            self.assertIsInstance(dataset.trips, adapter.CompactTrips)
            self.assertEqual(dataset.trips.edge_occurrences, 4)
            self.assertEqual(dataset.trips.storage_bytes, 4 * 4 + 3 * 8)
            output = root / "predictions.jsonl"
            adapter.write_prediction_rows(output, dataset.trips, [[0], [1, 2]])
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(
                rows,
                [
                    {"sample_id": "a", "predicted_edge_ids": [0]},
                    {"sample_id": "b", "predicted_edge_ids": [1, 2]},
                ],
            )
            self.assertTrue(output.read_bytes().endswith(b"\n"))

    def test_rejects_unknown_dataset_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.write_dataset(
                root,
                [{"sample_id": "a", "original_edge_ids": [0, 1], "time": 7}],
            )
            with self.assertRaisesRegex(RuntimeError, "fields differ"):
                adapter.load_dataset_manifest(manifest)

    def test_split_role_and_optional_dataset_hashes_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.write_dataset(
                root,
                [{"sample_id": "a", "original_edge_ids": [0, 1]}],
                "tiny-test",
            )
            with self.assertRaisesRegex(RuntimeError, "split role"):
                adapter.load_dataset_manifest(manifest, expected_role="validation")
            with self.assertRaisesRegex(RuntimeError, "pinned hash"):
                adapter.load_dataset_manifest(
                    manifest,
                    expected_role="test",
                    expected_manifest_sha256="0" * 64,
                )
            with self.assertRaisesRegex(RuntimeError, "records hash"):
                adapter.load_dataset_manifest(
                    manifest,
                    expected_role="test",
                    expected_manifest_sha256=adapter.sha256_file(manifest),
                    expected_records_sha256="0" * 64,
                )
            dataset = adapter.load_dataset_manifest(
                manifest,
                expected_role="test",
                expected_manifest_sha256=adapter.sha256_file(manifest),
                expected_records_sha256=adapter.sha256_file(root / "records.jsonl"),
            )
            self.assertEqual(dataset.manifest.split_role, "test")

    def test_preprocess_and_train_interfaces_cannot_accept_test(self) -> None:
        preprocess = adapter.parse_args(
            [
                "preprocess",
                "--train-manifest",
                "train.json",
                "--map-dir",
                "map",
                "--output-dir",
                "out",
            ]
        )
        self.assertFalse(hasattr(preprocess, "validation_manifest"))
        self.assertFalse(hasattr(preprocess, "test_manifest"))
        train = adapter.parse_args(
            [
                "train",
                "--preprocess-dir",
                "prep",
                "--train-manifest",
                "train.json",
                "--validation-manifest",
                "validation.json",
                "--output-dir",
                "out",
            ]
        )
        self.assertFalse(hasattr(train, "test_manifest"))
        self.assertEqual(train.epochs, 200)
        self.assertEqual(train.batch_size, 512)
        self.assertEqual(train.workers, 16)
        preprocess = adapter.parse_args(
            [
                "preprocess",
                "--train-manifest",
                "train.json",
                "--map-dir",
                "map",
                "--output-dir",
                "out",
            ]
        )
        self.assertEqual(preprocess.node2vec_batch_words, 4)
        self.assertEqual(preprocess.sc1_aggregation_buffer_entries, 8192)
        self.assertIsNone(preprocess.expected_train_records_sha256)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            adapter.parse_args(
                [
                    "train",
                    "--preprocess-dir",
                    "prep",
                    "--train-manifest",
                    "train.json",
                    "--validation-manifest",
                    "validation.json",
                    "--output-dir",
                    "out",
                    "--test-manifest",
                    "test.json",
                ]
            )


class GraphAndShortcutTests(unittest.TestCase):
    def test_line_graph_preserves_parallel_raw_edges(self) -> None:
        graph = adapter.build_line_graph([10, 10, 20], [20, 20, 30])
        self.assertEqual(graph.state_count, 3)
        self.assertEqual(graph.outgoing, [[2], [2], []])
        self.assertEqual(graph.incoming, [[], [], [0, 1]])
        self.assertNotEqual(graph.tail[0:1], graph.tail[0:0])

    def test_route_continuity_is_checked_on_raw_edge_identity(self) -> None:
        graph = adapter.build_line_graph([10, 10, 20], [20, 20, 30])
        adapter.validate_trips([adapter.Trip("ok", [1, 2])], graph)
        with self.assertRaisesRegex(RuntimeError, "discontinuous"):
            adapter.validate_trips([adapter.Trip("bad", [0, 1])], graph)

    def test_contraction_is_deterministic_and_creates_shortcut(self) -> None:
        # State 0 is the middle of 1->0->2 and wins the all-minus-one tie by ID.
        result = adapter.contract_graph([[2], [0], []], 0.34)
        self.assertEqual(result.order, [0])
        self.assertEqual(result.active, [False, True, True])
        self.assertEqual(result.sparse_outgoing, [[], [2], []])
        self.assertEqual(result.shortcut_pairs, [(1, 2)])

    def test_sc1_uses_only_supplied_training_segments(self) -> None:
        contraction = adapter.contract_graph([[2], [0], []], 0.34)
        database, stats = adapter.build_sc1_database(
            [adapter.Trip("train", [1, 0, 2])], contraction
        )
        self.assertEqual(database, {(1, 2): [1, 0, 2]})
        self.assertEqual(stats["candidate_segments"], 1)

    def test_sc1_medoid_tie_break_is_lexicographic(self) -> None:
        selected = adapter.select_sc1_path([[1, 4, 9], [1, 5, 9]])
        self.assertEqual(selected, [1, 4, 9])

    def test_sc1_medoid_preserves_historical_multiplicity(self) -> None:
        frequent = [1, 5, 9]
        rare = [1, 4, 9]
        selected = adapter.select_sc1_path([frequent, frequent, frequent, rare])
        self.assertEqual(selected, frequent)

    def test_shortcut_expansion_never_uses_truth(self) -> None:
        graph = adapter.build_line_graph([10, 20, 30], [20, 30, 40])
        expanded = adapter.expand_sparse_path(
            [0, 2], graph, {(0, 2): [0, 1, 2]}
        )
        self.assertEqual(expanded, [0, 1, 2])
        with self.assertRaisesRegex(RuntimeError, "missing SC1/SC2"):
            adapter.expand_sparse_path([0, 2], graph, {})

    def test_directed_walks_are_seeded_and_reiterable(self) -> None:
        corpus = adapter.Node2VecCorpus(
            [[1, 2], [2], [0]], walk_length=5, walks_per_state=3, seed=17
        )
        first = list(corpus)
        second = list(corpus)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 9)
        self.assertTrue(all(1 <= len(walk) <= 5 for walk in first))

    def test_macro_route_metrics(self) -> None:
        metrics = adapter.route_metrics([[0, 1], [2, 3]], [[0, 1], [2]])
        self.assertEqual(metrics["samples"], 2)
        self.assertAlmostEqual(metrics["edge_precision"], 1.0)
        self.assertAlmostEqual(metrics["edge_recall"], 0.75)
        self.assertAlmostEqual(metrics["edge_f1"], 5 / 6)
        self.assertAlmostEqual(metrics["exact_match"], 0.5)

    def test_transition_minibatch_reports_route_count_for_paper_loss(self) -> None:
        batches = list(
            adapter.transition_minibatches(
                [adapter.Trip("a", [0, 1]), adapter.Trip("b", [0, 1, 2])],
                [[1], [2], []],
                512,
                adapter.random.Random(7),
            )
        )
        self.assertEqual(len(batches), 1)
        current, destinations, targets, route_count = batches[0]
        self.assertEqual(route_count, 2)
        self.assertEqual(len(current), 3)
        self.assertEqual(len(destinations), 3)
        self.assertEqual(len(targets), 3)


@unittest.skipUnless(HAS_ARRAY_STACK, "pinned NumPy/pyshp/PyTorch stack is unavailable")
class ModelPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        adapter.load_array_dependencies()
        adapter.configure_runtime(11, 2)

    def test_greedy_prediction_uses_only_query_endpoints(self) -> None:
        graph = adapter.build_line_graph([10, 20, 20, 30], [20, 30, 30, 40])
        # Raw states: 0->{1,2}; both 1 and 2 ->3. Candidate embedding makes 2 win.
        embeddings = adapter.np.arange(4, dtype=adapter.np.float32).reshape(-1, 1)

        class CandidateScore(adapter.torch.nn.Module):
            def forward(self, values):
                return values[:, 2:3]

        predicted, stats = adapter.greedy_paths(
            CandidateScore(),
            embeddings,
            [adapter.Trip("q", [0, 1, 3])],
            graph.outgoing,
            adapter.torch.device("cpu"),
            inference_batch_size=8,
            max_steps=10,
        )
        self.assertEqual(predicted, [[0, 2, 3]])
        self.assertEqual(stats["reached_destination"], 1)

    def test_cycle_returns_generated_prefix_without_endpoint_append(self) -> None:
        embeddings = adapter.np.arange(3, dtype=adapter.np.float32).reshape(-1, 1)

        class ZeroScore(adapter.torch.nn.Module):
            def forward(self, values):
                return values[:, :1] * 0

        predicted, stats = adapter.greedy_paths(
            ZeroScore(),
            embeddings,
            [adapter.Trip("q", [0, 2])],
            [[1], [0], []],
            adapter.torch.device("cpu"),
            inference_batch_size=8,
            max_steps=10,
        )
        self.assertEqual(predicted, [[0, 1, 0]])
        self.assertNotEqual(predicted[0][-1], 2)
        self.assertEqual(stats["cycle"], 1)

    def test_one_epoch_training_and_validation_selection(self) -> None:
        embeddings = adapter.np.asarray([[0.0], [1.0], [2.0]], dtype=adapter.np.float32)
        outgoing = [[1], [2], []]
        trips = [adapter.Trip("train", [0, 1, 2])]
        model = adapter.make_transition_model(1, 4)

        def validate(candidate):
            return adapter.greedy_paths(
                candidate,
                embeddings,
                [adapter.Trip("validation", [0, 1, 2])],
                outgoing,
                adapter.torch.device("cpu"),
                inference_batch_size=4,
                max_steps=5,
            )

        _, diagnostics = adapter.fit_transition_model(
            model,
            embeddings,
            outgoing,
            trips,
            validate,
            [[0, 1, 2]],
            epochs=1,
            validation_every=1,
            batch_size=4,
            transition_chunk_size=4,
            learning_rate=0.001,
            seed=3,
            device=adapter.torch.device("cpu"),
        )
        self.assertEqual(diagnostics["selected_epoch"], 1)
        self.assertEqual(diagnostics["history"][0]["validation"]["exact_match"], 1.0)

    def test_sc2_groups_sources_and_uses_model_nll(self) -> None:
        graph = adapter.build_line_graph([10, 20, 20, 30], [20, 30, 30, 40])
        embeddings = adapter.np.arange(4, dtype=adapter.np.float32).reshape(-1, 1)

        class CandidateScore(adapter.torch.nn.Module):
            def forward(self, values):
                return values[:, 2:3]

        paths = adapter.shortest_model_paths(
            CandidateScore(),
            embeddings,
            graph,
            [0, 1, 2],
            3,
            adapter.torch.device("cpu"),
            score_batch_size=8,
        )
        self.assertEqual(paths[0], [0, 2, 3])
        self.assertEqual(paths[1], [1, 3])
        self.assertEqual(paths[2], [2, 3])

    def test_small_graph_three_stage_cli_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            map_dir = root / "map"
            map_dir.mkdir()
            writer = adapter.shapefile.Writer(
                str(map_dir / "edges"), shapeType=adapter.shapefile.POLYLINE
            )
            writer.field("fid", "N", decimal=0)
            writer.field("u", "N", decimal=0)
            writer.field("v", "N", decimal=0)
            # Raw state 0 is the middle edge in 1->0->2, so a one-state
            # contraction creates final shortcut 1->2 and exercises SC1.
            endpoints = [(20, 30), (10, 20), (30, 40)]
            for edge, (tail, head) in enumerate(endpoints):
                writer.line([[(float(tail), 0.0), (float(head), 0.0)]])
                writer.record(edge, tail, head)
            writer.close()

            train_dir, validation_dir, test_dir = (
                root / "train",
                root / "validation",
                root / "test",
            )
            train_dir.mkdir()
            validation_dir.mkdir()
            test_dir.mkdir()
            train_manifest = ProtocolTests().write_dataset(
                train_dir,
                [
                    {"sample_id": "train-a", "original_edge_ids": [1, 0, 2]},
                    {"sample_id": "train-b", "original_edge_ids": [1, 0, 2]},
                ],
                "tiny-train",
            )
            validation_manifest = ProtocolTests().write_dataset(
                validation_dir,
                [{"sample_id": "validation-a", "original_edge_ids": [1, 0, 2]}],
                "tiny-validation",
            )
            test_manifest = ProtocolTests().write_dataset(
                test_dir,
                [{"sample_id": "test-a", "original_edge_ids": [1, 0, 2]}],
                "tiny-test",
            )
            preprocess_dir = root / "preprocess"
            train_output = root / "trained"
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
                    "test-revision",
                    "--contraction-ratio",
                    "0.34",
                    "--workers",
                    "2",
                ]
            )

            def tiny_embeddings(outgoing, **_):
                return adapter.np.arange(
                    len(outgoing) * 2, dtype=adapter.np.float32
                ).reshape(len(outgoing), 2)

            with patch.object(adapter, "train_node2vec_embeddings", tiny_embeddings):
                with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
                    adapter.preprocess_command(preprocess_args)
            train_args = adapter.parse_args(
                [
                    "train",
                    "--preprocess-dir",
                    str(preprocess_dir),
                    "--train-manifest",
                    str(train_manifest),
                    "--validation-manifest",
                    str(validation_manifest),
                    "--output-dir",
                    str(train_output),
                    "--source-revision",
                    "test-revision",
                    "--device",
                    "cpu",
                    "--workers",
                    "2",
                    "--epochs",
                    "1",
                    "--validation-every",
                    "1",
                    "--batch-size",
                    "8",
                    "--hidden-dimension",
                    "4",
                    "--max-steps",
                    "10",
                ]
            )
            with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
                adapter.train_command(train_args)
            prediction_dir = root / "prediction"
            predictions = prediction_dir / "predictions.jsonl"
            receipt = prediction_dir / "run.json"
            diagnostics = prediction_dir / "diagnostics.json"
            predict_args = adapter.parse_args(
                [
                    "predict",
                    "--checkpoint",
                    str(train_output / "checkpoint.pt"),
                    "--map-dir",
                    str(map_dir),
                    "--dataset-manifest",
                    str(test_manifest),
                    "--predictions",
                    str(predictions),
                    "--run-receipt",
                    str(receipt),
                    "--diagnostics",
                    str(diagnostics),
                    "--source-revision",
                    "test-revision",
                    "--device",
                    "cpu",
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
            with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
                adapter.predict_command(predict_args)
            row = json.loads(predictions.read_text(encoding="utf-8"))
            self.assertEqual(row["sample_id"], "test-a")
            self.assertEqual(row["predicted_edge_ids"], [1, 0, 2])
            prediction_diagnostics = json.loads(diagnostics.read_text(encoding="utf-8"))
            self.assertFalse(prediction_diagnostics["endpoint_repair"])
            self.assertFalse(
                prediction_diagnostics["truth_interior_used_for_route_generation"]
            )
            self.assertEqual(prediction_diagnostics["endpoint_failures"], 0)
            self.assertEqual(
                prediction_diagnostics["generated_route_validity"][
                    "routes_with_illegal_transitions"
                ],
                0,
            )
            training_diagnostics = json.loads(
                (train_output / "training_diagnostics.json").read_text(encoding="utf-8")
            )
            self.assertFalse(training_diagnostics["test_data_read"])
            self.assertEqual(
                training_diagnostics["configuration"]["loss_normalization"],
                "paper_eq10_mean_route_summed_transition_nll",
            )
            self.assertEqual(training_diagnostics["shortcut_storage"]["final_shortcuts"], 1)
            self.assertEqual(training_diagnostics["shortcut_storage"]["sc1_train_historical"], 1)


if __name__ == "__main__":
    unittest.main()
