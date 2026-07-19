import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "route_quality_confidence.py"
SPEC = importlib.util.spec_from_file_location("route_quality_confidence", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def write_evaluation(path: Path, truth: list[list[int]], predictions: list[list[int]]) -> Path:
    totals = {name: 0.0 for name in MODULE.METRICS if name != "endpoint_failure_rate"}
    for expected, predicted in zip(truth, predictions):
        metrics = MODULE.route_metrics(expected, predicted)
        for name in totals:
            totals[name] += metrics[name]
    path.write_text(
        json.dumps(
            {
                "schema": MODULE.EVALUATION_SCHEMA,
                "sample_count": len(truth),
                "metrics": {name: value / len(truth) for name, value in totals.items()},
            }
        ),
        encoding="utf-8",
    )
    return path


class ConfidenceTests(unittest.TestCase):
    def test_streaming_intervals_match_point_estimates_and_paired_rows(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            truth = [[1, 2], [3, 4], [5, 6]]
            project = [[1, 2], [3], [5, 7]]
            other = [[1, 2], [3, 4], [5, 6]]
            dataset = write_jsonl(
                root / "dataset.jsonl",
                [
                    {"sample_id": f"s{i}", "original_edge_ids": edges}
                    for i, edges in enumerate(truth)
                ],
            )
            prediction_paths = {
                "project": write_jsonl(
                    root / "project.jsonl",
                    [
                        {"sample_id": f"s{i}", "predicted_edge_ids": edges}
                        for i, edges in enumerate(project)
                    ],
                ),
                "other": write_jsonl(
                    root / "other.jsonl",
                    [
                        {"sample_id": f"s{i}", "predicted_edge_ids": edges}
                        for i, edges in enumerate(other)
                    ],
                ),
            }
            evaluations = {
                "project": write_evaluation(root / "project.eval.json", truth, project),
                "other": write_evaluation(root / "other.eval.json", truth, other),
            }
            result = MODULE.evaluate(
                dataset, prediction_paths, evaluations, "project", 0.95, 2
            )
            self.assertEqual(result["sample_count"], 3)
            self.assertEqual(result["methods"]["project"]["endpoint_failures"], 2)
            self.assertEqual(result["methods"]["other"]["endpoint_failures"], 0)
            self.assertGreater(
                result["paired_differences_vs_reference"]["other"]["edge_f1"]["mean"],
                0.0,
            )
            self.assertEqual(
                result["methods"]["other"]["intervals"]["exact_match"]["mean"],
                1.0,
            )

    def test_misalignment_and_extra_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            dataset = write_jsonl(
                root / "dataset.jsonl",
                [
                    {"sample_id": "a", "original_edge_ids": [1]},
                    {"sample_id": "b", "original_edge_ids": [2]},
                ],
            )
            prediction = write_jsonl(
                root / "prediction.jsonl",
                [
                    {"sample_id": "wrong", "predicted_edge_ids": [1]},
                    {"sample_id": "b", "predicted_edge_ids": [2]},
                ],
            )
            evaluation = write_evaluation(root / "evaluation.json", [[1], [2]], [[1], [2]])
            with self.assertRaisesRegex(MODULE.ConfidenceError, "does not match"):
                MODULE.evaluate(
                    dataset,
                    {"project": prediction},
                    {"project": evaluation},
                    "project",
                    0.95,
                    1,
                )


if __name__ == "__main__":
    unittest.main()
