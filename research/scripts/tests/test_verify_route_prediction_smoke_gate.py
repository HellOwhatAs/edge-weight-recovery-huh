import argparse
import json
from pathlib import Path
import tempfile
import unittest

from research.scripts import verify_route_prediction_smoke_gate as gate


def dataset_row(sample_id: str, edges: list[int]) -> bytes:
    return (
        json.dumps(
            {"sample_id": sample_id, "original_edge_ids": edges},
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def prediction_row(index: int) -> bytes:
    return (
        json.dumps(
            {
                "sample_id": f"reference:{index}",
                "predicted_edge_ids": [index, index + 1],
            },
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


class SmokeGateTests(unittest.TestCase):
    def fixture(self, root: Path) -> argparse.Namespace:
        dataset = root / "reference.jsonl"
        dataset.write_bytes(
            b"".join(
                dataset_row(f"reference:{index}", [index, index + 1])
                for index in range(3)
            )
        )
        predictions = []
        for method in gate.EXPECTED_METHODS:
            reference = root / f"{method}.reference.jsonl"
            candidate = root / f"{method}.candidate.jsonl"
            content = b"".join(prediction_row(index) for index in range(3))
            reference.write_bytes(content)
            candidate.write_bytes(content)
            predictions.append([method, str(reference), str(candidate)])
        return argparse.Namespace(
            reference_dataset=dataset,
            expected_rows=3,
            prediction=predictions,
            output=root / "gate.json",
        )

    def test_exact_seven_method_gate_is_atomic_and_authorizes(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            args = self.fixture(Path(raw))
            result = gate.run(args)
            self.assertTrue(result["formal_full_test_prediction_authorized"])
            self.assertEqual(list(result["methods"]), list(gate.EXPECTED_METHODS))
            self.assertEqual(result["reference_dataset"]["records"], 3)
            on_disk = json.loads(args.output.read_text())
            self.assertEqual(on_disk["schema"], gate.SCHEMA)
            self.assertFalse(list(Path(raw).glob(".gate.json.*.tmp")))

    def test_prediction_byte_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            args = self.fixture(Path(raw))
            candidate = Path(args.prediction[3][2])
            candidate.write_bytes(candidate.read_bytes() + b" ")
            with self.assertRaisesRegex(gate.GateError, "bytes differ"):
                gate.run(args)

    def test_reference_count_and_identity_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            args = self.fixture(Path(raw))
            args.reference_dataset.write_bytes(
                dataset_row("reference:0", [0, 1])
                + dataset_row("reference:0", [1, 2])
                + dataset_row("reference:2", [2, 3])
            )
            with self.assertRaisesRegex(gate.GateError, "repeats sample_id"):
                gate.run(args)

    def test_method_set_and_order_are_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            args = self.fixture(Path(raw))
            args.prediction = list(reversed(args.prediction))
            with self.assertRaisesRegex(gate.GateError, "methods/order differ"):
                gate.run(args)


if __name__ == "__main__":
    unittest.main()
