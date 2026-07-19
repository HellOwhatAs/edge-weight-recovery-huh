import importlib.util
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).parents[1] / "export_protocol_dataset.py"
SPEC = importlib.util.spec_from_file_location("export_protocol_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ExportProtocolDatasetTests(unittest.TestCase):
    def test_export_preserves_ids_and_edges_and_discards_only_timestamps(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "common.pkl"
            output = directory / "protocol" / "records.jsonl"
            manifest = directory / "protocol" / "manifest.json"
            with source.open("wb") as writer:
                pickle.dump(
                    [
                        ("train:0", [3, 5, 8], (100, 110)),
                        ("train:1", [13, 21], (120, 130)),
                    ],
                    writer,
                )
            count = MODULE.export(
                SimpleNamespace(
                    input_pickle=source,
                    output_jsonl=output,
                    manifest=manifest,
                    dataset_id="beijing/train-fixture",
                    network_id="beijing-roads-v1",
                )
            )
            self.assertEqual(count, 2)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(
                rows,
                [
                    {"sample_id": "train:0", "original_edge_ids": [3, 5, 8]},
                    {"sample_id": "train:1", "original_edge_ids": [13, 21]},
                ],
            )
            descriptor = json.loads(manifest.read_text())
            self.assertEqual(descriptor["records_file"], "records.jsonl")
            self.assertEqual(descriptor["dataset_id"], "beijing/train-fixture")

    def test_invalid_and_duplicate_rows_are_rejected(self):
        for trips in [
            [("x", [1], (1, 2))],
            [("x", [1, 2], (1, 2)), ("x", [2, 3], (2, 3))],
            [("x", [True, 2], (1, 2))],
            [("x", [1, 2], (1, None))],
        ]:
            with self.subTest(trips=trips), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                source = directory / "bad.pkl"
                with source.open("wb") as writer:
                    pickle.dump(trips, writer)
                with self.assertRaises(ValueError):
                    MODULE.export(
                        SimpleNamespace(
                            input_pickle=source,
                            output_jsonl=directory / "out" / "records.jsonl",
                            manifest=directory / "out" / "manifest.json",
                            dataset_id="bad",
                            network_id="network",
                        )
                    )


if __name__ == "__main__":
    unittest.main()
