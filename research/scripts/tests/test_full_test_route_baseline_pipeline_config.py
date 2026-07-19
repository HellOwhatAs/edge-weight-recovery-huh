import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
CONFIG = (
    ROOT
    / "research"
    / "experiments"
    / "route_baselines_full_test_20260719"
    / "pipeline.json"
)
PIPELINE_SCRIPT = ROOT / "research" / "scripts" / "run_route_baseline_pipeline.py"
SHARDED_SCRIPT = ROOT / "research" / "scripts" / "run_sharded_route_predictions.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline_module = load_module("full_test_route_pipeline", PIPELINE_SCRIPT)
sharded_module = load_module("full_test_sharded_predictions", SHARDED_SCRIPT)


METHOD_PATHS = {
    "project": "project/full/predictions.jsonl",
    "sp_length": "sp-length/full/predictions.jsonl",
    "markov_sp": "markov-sp/full/predictions.jsonl",
    "neuromlr_greedy": "neuromlr-g/full/predictions.jsonl",
    "drncs_lg": "drncs-lg/full/predictions.jsonl",
    "drpk_static": "drpk-static/full/predictions.jsonl",
    "drp_tp": "drp-tp/full/predictions.jsonl",
}
METHOD_ORDER = tuple(METHOD_PATHS)
SMOKE_PREDICTION_SHA256 = {
    "project": "a0b0d3492dc75b3b6bcb14ce95703b81022ecd3d72569e690204d483212e87a8",
    "sp_length": "28a5a58007ec865f13158fdbcad6eae683fd2408f69880d502a3c8bb4c68e7ca",
    "markov_sp": "8f2060829e22a9882f957f9f53ed22c48544b1884d233f632a03a7ac7fbc6ba1",
    "neuromlr_greedy": "22baa50d15d60746a912f01b747f179ee168e8f2bc20124c5d2a8418492a716b",
    "drncs_lg": "168b704a9d5596fed9073d4b6db5237a44fb305e67306943e7b47d32c30fd943",
    "drpk_static": "20bfbc8d636abb052bc3e682f82cc4bdb47ee2777bcc37ae57d467efa2f8a080",
    "drp_tp": "d8542027d8f13ded4236df057220ef0d9aaf70e74620e79d0c02f2ab74f0272d",
}
EXPECTED_DEVICES = {
    "project": "cpu",
    "sp_length": "cpu",
    "markov_sp": "cpu",
    "neuromlr_greedy": "cuda:0",
    "drncs_lg": "mixed:cuda:0+cpu",
    "drpk_static": "cuda:0",
    "drp_tp": "cpu",
}


def option(command: list[str], name: str) -> str:
    try:
        index = command.index(name)
        return command[index + 1]
    except (ValueError, IndexError) as error:
        raise AssertionError(f"command is missing {name}: {command}") from error


class FullTestPipelineConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = pipeline_module.Pipeline(CONFIG)
        cls.config = cls.pipeline.config
        cls.task_index = {
            task["id"]: index for index, task in enumerate(cls.pipeline.tasks)
        }
        cls.formal_predictions = cls._find_formal_predictions()

    @classmethod
    def _find_formal_predictions(cls) -> dict[str, dict]:
        by_relative_output: dict[str, dict] = {}
        runtime = cls.pipeline.runtime_root
        for task in cls.pipeline.tasks:
            if "full" not in task.get("profiles", ["full"]):
                continue
            for output in task.get("outputs", []):
                path = cls.pipeline.path(output["path"])
                try:
                    relative = path.relative_to(runtime).as_posix()
                except ValueError:
                    continue
                if relative in METHOD_PATHS.values():
                    if relative in by_relative_output:
                        raise AssertionError(
                            f"multiple formal tasks produce {relative}: "
                            f"{by_relative_output[relative]['id']} and {task['id']}"
                        )
                    by_relative_output[relative] = task
        return {
            method: by_relative_output[path]
            for method, path in METHOD_PATHS.items()
            if path in by_relative_output
        }

    def ancestors(self, task_id: str) -> set[str]:
        result: set[str] = set()
        pending = list(self.pipeline.task_by_id[task_id].get("depends_on", []))
        while pending:
            dependency = pending.pop()
            if dependency in result:
                continue
            result.add(dependency)
            pending.extend(
                self.pipeline.task_by_id[dependency].get("depends_on", [])
            )
        return result

    def output_spec(self, task: dict, resolved_path: Path) -> dict:
        for spec in task.get("outputs", []):
            if self.pipeline.path(spec["path"]) == resolved_path:
                return spec
        self.fail(f"{task['id']} does not declare output {resolved_path}")

    def test_fixed_runtime_resources_and_environment(self) -> None:
        self.assertEqual(
            self.pipeline.runtime_root,
            ROOT / "research" / "generated" / "full_test_comparison_20260719",
        )
        self.assertEqual(self.config.get("cpu_threads"), 16)
        self.assertEqual(
            self.config.get("resources"),
            {
                "MemoryHigh": "8G",
                "MemoryMax": "10G",
                "MemorySwapMax": "4G",
                "OOMPolicy": "stop",
            },
        )
        environment = self.config.get("environment", {})
        self.assertEqual(environment.get("LC_ALL"), "C")
        self.assertEqual(environment.get("PYTHONUNBUFFERED"), "1")

    def test_exactly_seven_formal_methods_and_no_neuromlr_dijkstra(self) -> None:
        self.assertEqual(tuple(self.formal_predictions), METHOD_ORDER)
        self.assertEqual(len(self.formal_predictions), 7)

        for task in self.pipeline.tasks:
            command = task.get("command", [])
            lowered = [part.lower() for part in command]
            self.assertFalse(
                any("neuromlr_dijkstra" in part for part in lowered), task["id"]
            )
            self.assertNotIn(("--method", "dijkstra"), zip(lowered, lowered[1:]))
            self.assertNotIn("dijkstra", task["id"].lower())

        neuromlr = self.formal_predictions["neuromlr_greedy"]
        self.assertEqual(option(neuromlr["command"], "--method"), "greedy")

    def test_all_formal_predictions_transitively_depend_on_last_smoke_gate(self) -> None:
        gate = self.pipeline.task_by_id.get("smoke-equivalence-gate")
        self.assertIsNotNone(gate)
        assert gate is not None
        self.assertEqual(gate.get("profiles"), ["smoke"])

        smoke_indices = [
            self.task_index[task["id"]]
            for task in self.pipeline.tasks
            if "smoke" in task.get("profiles", ["full"])
        ]
        self.assertEqual(self.task_index[gate["id"]], max(smoke_indices))
        formal_tasks = [
            task
            for task in self.pipeline.tasks
            if "full" in task.get("profiles", ["full"])
        ]
        for task in formal_tasks:
            self.assertIn(
                gate["id"],
                self.ancestors(task["id"]),
                f"formal gate: {task['id']}",
            )

        protocol_guard = self.pipeline.task_by_id["verify-frozen-protocol"]
        self.assertEqual(protocol_guard.get("action"), "verify")
        protocol_specs = [
            spec
            for spec in protocol_guard["outputs"]
            if self.pipeline.path(spec["path"]).name == "protocol.json"
        ]
        self.assertEqual(len(protocol_specs), 1)
        self.assertEqual(
            protocol_specs[0]["json_contains"]["status"],
            "frozen_before_full_test",
        )
        self.assertEqual(
            self.formal_predictions["project"].get("depends_on"),
            ["verify-frozen-protocol"],
        )

    def test_device_thread_and_memory_scope_policy(self) -> None:
        for method, task in self.formal_predictions.items():
            self.assertEqual(task.get("device", "cpu"), EXPECTED_DEVICES[method])
            self.assertTrue(task.get("systemd_scope", True), method)
            command = task["command"]
            if method in {"project", "sp_length", "markov_sp"}:
                self.assertEqual(option(command, "--threads"), "16")
            elif method in {"drncs_lg", "drpk_static", "drp_tp"}:
                self.assertEqual(option(command, "--workers"), "16")

            if EXPECTED_DEVICES[method] in {"cuda:0", "mixed:cuda:0+cpu"}:
                self.assertIn(option(command, "--device"), {"cuda", "cuda:0"})
                if method in {"drncs_lg", "drpk_static"}:
                    self.assertEqual(option(command, "--cuda-visible-devices"), "0")
            elif "--device" in command:
                self.assertEqual(option(command, "--device"), "cpu")

    def test_formal_predictions_are_zero_warmup_one_measured_pass(self) -> None:
        direct = ("project", "sp_length", "markov_sp", "neuromlr_greedy")
        for method in direct:
            command = self.formal_predictions[method]["command"]
            self.assertEqual(option(command, "--warmup-repetitions"), "0")
            self.assertEqual(option(command, "--measured-repetitions"), "1")

        for method in ("drncs_lg", "drpk_static", "drp_tp"):
            command = self.formal_predictions[method]["command"]
            script_index = command.index("research/scripts/run_sharded_route_predictions.py")
            args = sharded_module.parse_args(command[script_index + 1 :])
            settings = sharded_module.resolved_settings(args)
            self.assertEqual(settings["warmup_repetitions"], 0)
            self.assertEqual(settings["measured_repetitions"], 1)
            self.assertEqual(args.shard_limit, 0)

    def test_each_formal_prediction_records_stable_outer_time_evidence(self) -> None:
        for method, task in self.formal_predictions.items():
            with self.subTest(method=method):
                time_output = task.get("time_output")
                aggregate_output = task.get("time_aggregate_output")
                self.assertIsInstance(time_output, str)
                self.assertIsInstance(aggregate_output, str)
                self.assertNotEqual(time_output, aggregate_output)
                resolved_time = self.pipeline.path(time_output)
                resolved_aggregate = self.pipeline.path(aggregate_output)
                self.assertTrue(resolved_time.is_relative_to(self.pipeline.runtime_root))
                self.assertTrue(
                    resolved_aggregate.is_relative_to(self.pipeline.runtime_root)
                )
                self.assertEqual(resolved_time.name, "external-time.txt")
                self.assertEqual(resolved_aggregate.name, "time-evidence.json")
                self.assertGreaterEqual(
                    self.output_spec(task, resolved_time).get("min_bytes", 0), 10
                )
                aggregate_spec = self.output_spec(task, resolved_aggregate)
                self.assertEqual(
                    aggregate_spec.get("json_contains", {}).get("schema"),
                    pipeline_module.TASK_TIME_EVIDENCE_SCHEMA,
                )

    def test_formal_receipts_bind_gate_and_adopted_output_hashes(self) -> None:
        gate_path = self.pipeline.runtime_root / "smoke" / "equivalence-gate.json"
        for method, task in self.formal_predictions.items():
            with self.subTest(method=method):
                gate_specs = [
                    spec
                    for spec in task.get("requires", [])
                    if self.pipeline.path(spec["path"]) == gate_path
                ]
                self.assertEqual(len(gate_specs), 1)
                self.assertTrue(
                    gate_specs[0]["json_contains"][
                        "formal_full_test_prediction_authorized"
                    ]
                )
                self.assertEqual(
                    gate_specs[0]["json_contains"]["schema"],
                    "ewr.full-test-route-smoke-gate/v3",
                )

        neuromlr = self.formal_predictions["neuromlr_greedy"]
        self.assertEqual(
            neuromlr.get("hash_bindings"),
            [
                {
                    "json_path": "${runtime}/neuromlr-g/full/diagnostics.json",
                    "field": "predictions_sha256",
                    "file_path": "${runtime}/neuromlr-g/full/predictions.jsonl",
                }
            ],
        )
        for method in ("drncs_lg", "drpk_static", "drp_tp"):
            bindings = self.formal_predictions[method].get("hash_bindings", [])
            self.assertEqual(
                {binding["field"] for binding in bindings},
                {
                    "predictions_sha256",
                    "diagnostics_sha256",
                    "run_receipt_sha256",
                },
            )

    def test_neuromlr_chunk_and_sharded_full_sizes_are_frozen(self) -> None:
        neuromlr = self.formal_predictions["neuromlr_greedy"]["command"]
        self.assertEqual(option(neuromlr, "--route-chunk-size"), "500")
        self.assertEqual(option(neuromlr, "--resume"), "auto")

        for method in ("drncs_lg", "drpk_static", "drp_tp"):
            command = self.formal_predictions[method]["command"]
            self.assertEqual(option(command, "--shard-size"), "4096")

    def test_drpk_and_drp_receipts_bind_manifested_routing_files(self) -> None:
        preprocess = (
            ROOT
            / "research"
            / "generated"
            / "full_data_comparison_20260718"
            / "drpk-static"
            / "preprocess"
        )
        expected_common = {
            preprocess / "graph.npz",
            preprocess / "da" / "metadata.json",
            preprocess / "da" / "row_offsets.npy",
            preprocess / "da" / "col_offsets.npy",
            preprocess / "da" / "row_indices.u32",
            preprocess / "da" / "row_values.u32",
            preprocess / "da" / "col_indices.u32",
            preprocess / "da" / "col_values.u32",
        }
        drpk_requires = {
            self.pipeline.path(spec["path"])
            for spec in self.formal_predictions["drpk_static"]["requires"]
        }
        drp_requires = {
            self.pipeline.path(spec["path"])
            for spec in self.formal_predictions["drp_tp"]["requires"]
        }
        self.assertTrue(expected_common <= drpk_requires)
        self.assertTrue(expected_common <= drp_requires)
        self.assertTrue(
            {
                preprocess / "configuration.json",
                preprocess / "core-artifacts.json",
                preprocess / "node2vec.npy",
                preprocess / "static_features.npz",
            }
            <= drpk_requires
        )
        self.assertIn(preprocess / "routing-configuration.json", drp_requires)

    def test_smoke_gate_binds_all_seven_exact_500_prediction_hashes(self) -> None:
        gate = self.pipeline.task_by_id["smoke-equivalence-gate"]
        command = gate["command"]
        self.assertEqual(option(command, "--expected-rows"), "500")
        self.assertIsNotNone(option(command, "--reference-dataset"))
        self.assertNotIn("--full-dataset", command)
        prediction_indices = [
            index for index, part in enumerate(command) if part == "--prediction"
        ]
        triples = [tuple(command[index + 1 : index + 4]) for index in prediction_indices]
        self.assertEqual(tuple(triple[0] for triple in triples), METHOD_ORDER)

        for method, _reference, candidate in triples:
            candidate_path = self.pipeline.path(candidate)
            producers = [
                task
                for task in self.pipeline.tasks
                if any(
                    self.pipeline.path(spec["path"]) == candidate_path
                    for spec in task.get("outputs", [])
                )
            ]
            self.assertEqual(
                len(producers), 1, f"{method} smoke output producer: {candidate_path}"
            )
            producer = producers[0]
            spec = self.output_spec(producer, candidate_path)
            self.assertEqual(spec.get("sha256"), SMOKE_PREDICTION_SHA256[method])
            self.assertIn(producer["id"], self.ancestors(gate["id"]))

        gate_output = self.pipeline.path("${runtime}/smoke/equivalence-gate.json")
        contains = self.output_spec(gate, gate_output)["json_contains"]
        self.assertEqual(contains["schema"], "ewr.full-test-route-smoke-gate/v3")
        self.assertEqual(contains["reference_dataset.records"], 500)
        self.assertTrue(contains["formal_full_test_prediction_authorized"])

    def test_formal_prediction_order_is_frozen_and_sequential(self) -> None:
        observed = tuple(
            method
            for method, _task in sorted(
                self.formal_predictions.items(),
                key=lambda item: self.task_index[item[1]["id"]],
            )
        )
        self.assertEqual(observed, METHOD_ORDER)
        indices = [
            self.task_index[self.formal_predictions[method]["id"]]
            for method in METHOD_ORDER
        ]
        self.assertEqual(indices, sorted(indices))

        full_inputs = {
            self.pipeline.runtime_root / "manifests" / "test.jsonl",
            self.pipeline.runtime_root / "manifests" / "test.manifest.json",
        }
        for method, task in self.formal_predictions.items():
            resolved_arguments = {
                self.pipeline.path(part)
                for part in task["command"]
                if part.endswith(("/test.jsonl", "/test.manifest.json"))
            }
            self.assertEqual(len(resolved_arguments & full_inputs), 1, method)

    def test_summary_receipt_binds_every_manifest_transitive_input(self) -> None:
        task = self.pipeline.task_by_id["summarize-paper-tables"]
        required = {
            self.pipeline.path(spec["path"]): spec for spec in task["requires"]
        }
        manifest_path = (
            ROOT
            / "research"
            / "experiments"
            / "route_baselines_full_test_20260719"
            / "summary-input.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        runtime = (manifest_path.parent / manifest["runtime_root"]).resolve()
        expected = {
            (manifest_path.parent / manifest["archived_summary"]).resolve()
        }
        for entry in manifest["methods"].values():
            for key, value in entry.items():
                if key == "artifacts":
                    expected.update((runtime / item["path"]).resolve() for item in value)
                elif key in {"training", "offline"}:
                    expected.add((runtime / value).resolve())
        missing = expected - required.keys()
        self.assertFalse(missing, f"unbound summary inputs: {sorted(missing)}")
        for path in expected:
            self.assertEqual(len(required[path].get("sha256", "")), 64)


if __name__ == "__main__":
    unittest.main()
