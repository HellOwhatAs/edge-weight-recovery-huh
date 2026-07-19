from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research.scripts.select_project_checkpoint import select


def write_checkpoint(root: Path, update: int) -> None:
    (root / f"checkpoint-{update}.json").write_text(
        json.dumps({"update": update}), encoding="utf-8"
    )


def write_evaluation(root: Path, update: int, f1: float, exact: float) -> None:
    (root / f"checkpoint-{update}.evaluation.json").write_text(
        json.dumps(
            {
                "schema": "ewr.evaluation-summary/v1",
                "sample_count": 500,
                "metrics": {
                    "edge_precision": f1,
                    "edge_recall": f1,
                    "edge_f1": f1,
                    "edge_jaccard": f1,
                    "exact_match": exact,
                },
            }
        ),
        encoding="utf-8",
    )


class SelectProjectCheckpointTests(unittest.TestCase):
    def test_selects_f1_then_exact_then_earliest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            training = tmp_path / "training"
            validation = tmp_path / "validation"
            training.mkdir()
            validation.mkdir()
            for update, f1, exact in (
                (0, 0.8, 0.7),
                (25, 0.9, 0.6),
                (50, 0.9, 0.7),
                (75, 0.9, 0.7),
            ):
                write_checkpoint(training, update)
                write_evaluation(validation, update, f1, exact)

            result = select(training, validation)

            self.assertEqual(result["selected_update"], 50)
            self.assertEqual(result["candidate_count"], 4)
            self.assertEqual(len(result["selected_checkpoint_sha256"]), 64)

    def test_rejects_incomplete_validation_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            training = tmp_path / "training"
            validation = tmp_path / "validation"
            training.mkdir()
            validation.mkdir()
            write_checkpoint(training, 0)

            with self.assertRaisesRegex(
                RuntimeError, "missing validation evaluation"
            ):
                select(training, validation)


if __name__ == "__main__":
    unittest.main()
