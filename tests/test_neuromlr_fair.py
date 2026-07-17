import importlib.util
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "neuromlr_fair", Path(__file__).parents[1] / "scripts" / "neuromlr_fair.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class NeuroMLRFairTests(unittest.TestCase):
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

    def test_greedy_rollout_starts_at_true_first_and_reaches_true_last_edge(self):
        class FirstValidCandidate:
            def eval(self):
                return self

            def __call__(self, current, destinations, candidates, traffic):
                del current, destinations, traffic
                return MODULE.torch.tensor(
                    [[1.0] if candidate >= 0 else [-100.0] for candidate in candidates]
                )

        trip = MODULE.Trip("validation:0", "source", 0, [0, 1, 2, 3, 4], 10, 20)
        graph = SimpleNamespace(
            padded_neighbors=[[1, -1], [2, -1], [3, -1], [4, -1], [-1, -1]],
            max_neighbors=2,
        )
        generated = MODULE.greedy_paths(
            FirstValidCandidate(), [trip], graph, MODULE.torch.device("cpu")
        )
        self.assertEqual(generated, [[0, 1, 2, 3, 4]])

    def test_length_l_training_path_has_l_minus_one_targets(self):
        trip = MODULE.Trip("train:0", "source", 0, [0, 1, 2, 3, 4], 10, 20)
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

    def test_lipschitz_sparse_graph_takes_minimum_parallel_edge(self):
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


if __name__ == "__main__":
    unittest.main()
