from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "run_sharded_route_predictions.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


FAKE_ADAPTER = r'''#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("command")
parser.add_argument("--dataset-manifest", type=Path, required=True)
parser.add_argument("--predictions", type=Path, required=True)
parser.add_argument("--run-receipt", type=Path, required=True)
parser.add_argument("--diagnostics", type=Path, required=True)
parser.add_argument("--source-revision", required=True)
parser.add_argument("--method", default="drncs_lg")
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--warmup-repetitions", type=int, required=True)
parser.add_argument("--measured-repetitions", type=int, required=True)
parser.add_argument("--max-steps", type=int, default=300)
args, _ = parser.parse_known_args()

started_path = os.environ.get("FAKE_ADAPTER_STARTED")
if started_path:
    Path(started_path).write_text("started", encoding="utf-8")
sleep_seconds = float(os.environ.get("FAKE_ADAPTER_SLEEP_SECONDS", "0"))
if sleep_seconds:
    import time
    time.sleep(sleep_seconds)

manifest_bytes = args.dataset_manifest.read_bytes()
manifest = json.loads(manifest_bytes)
records = args.dataset_manifest.parent / manifest["records_file"]
rows = [json.loads(line) for line in records.read_text().splitlines()]
with args.predictions.open("w", encoding="utf-8", newline="\n") as output:
    for row in rows:
        output.write(json.dumps({
            "sample_id": row["sample_id"],
            "predicted_edge_ids": row["original_edge_ids"],
        }, separators=(",", ":")) + "\n")

method = args.method
checkpoint_hash = (
    hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if args.checkpoint is not None else None
)
diagnostics = {
    "method": method,
    "samples": len(rows),
    "warmup_repetitions": args.warmup_repetitions,
    "measured_repetitions": args.measured_repetitions,
    "endpoint_failures": 0,
    "peak_rss_kib": 10,
    "peak_cuda_memory_bytes": 0,
    "timing": {"prediction_seconds": 0.01},
}
if method == "drncs_lg":
    diagnostics.update({
        "dataset_hash_pins_enforced": True,
        "checkpoint_sha256": checkpoint_hash,
        "max_steps": args.max_steps,
    })
elif method == "drpk_static":
    diagnostics["checkpoint"] = {"sha256": checkpoint_hash}
args.diagnostics.write_text(json.dumps(diagnostics), encoding="utf-8")
receipt = {
    "method": {"name": method},
    "dataset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    "source_revision": args.source_revision,
    "environment": {"python": "fixture", "device": "cpu"},
}
args.run_receipt.write_text(json.dumps(receipt), encoding="utf-8")
counter = Path(os.environ["FAKE_ADAPTER_COUNTER"])
with counter.open("a", encoding="utf-8") as output:
    output.write(manifest["dataset_id"] + "\n")
'''


class ShardedPredictionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fake = self.root / "fake-adapter"
        self.fake.write_text(FAKE_ADAPTER, encoding="utf-8")
        self.fake.chmod(0o755)
        self.counter = self.root / "counter.txt"
        self.preprocess = self.root / "preprocess"
        self.preprocess.mkdir()
        artifact = self.preprocess / "artifact.bin"
        artifact.write_bytes(b"immutable-routing-artifact")
        artifact_manifest = {
            "schema": "fixture-routing-artifacts/v1",
            "artifacts": {
                "artifact.bin": {
                    "bytes": artifact.stat().st_size,
                    "sha256": digest(artifact),
                }
            },
        }
        artifact_manifest_path = self.preprocess / "routing-artifacts.json"
        artifact_manifest_path.write_text(
            json.dumps(artifact_manifest), encoding="utf-8"
        )
        configuration = {
            "schema": "fixture-routing/v1",
            "routing_artifacts": {
                "path": "routing-artifacts.json",
                "sha256": digest(artifact_manifest_path),
            },
        }
        (self.preprocess / "routing-configuration.json").write_text(
            json.dumps(configuration), encoding="utf-8"
        )
        rows = [
            {"sample_id": f"test:{index}", "original_edge_ids": [index, index + 1]}
            for index in range(5)
        ]
        self.rows = rows
        records = self.root / "test.jsonl"
        records.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        manifest = {
            "schema": "ewr.dataset-manifest/v1",
            "dataset_id": "tiny/full-test-5",
            "network_id": "tiny-network",
            "records_schema": "ewr.dataset-record/v1",
            "records_file": "test.jsonl",
        }
        self.manifest = self.root / "test.manifest.json"
        self.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        self.output = self.root / "output"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def command(self, *extra: str) -> list[str]:
        return [
            sys.executable,
            str(SCRIPT),
            "--method",
            "drp_tp",
            "--dataset-manifest",
            str(self.manifest),
            "--output-dir",
            str(self.output),
            "--adapter-executable",
            str(self.fake),
            "--adapter-source",
            str(self.fake),
            "--preprocess-dir",
            str(self.preprocess),
            "--source-revision",
            "fixture-revision",
            "--shard-size",
            "2",
            *extra,
        ]

    def invoke(self, *extra: str) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ)
        environment["FAKE_ADAPTER_COUNTER"] = str(self.counter)
        return subprocess.run(
            self.command(*extra),
            text=True,
            capture_output=True,
            env=environment,
            check=False,
        )

    def invocation_count(self) -> int:
        return len(self.counter.read_text().splitlines()) if self.counter.exists() else 0

    def test_pause_resume_exact_join_and_verified_noop(self) -> None:
        paused = self.invoke("--shard-limit", "1")
        self.assertEqual(paused.returncode, 0, paused.stderr)
        self.assertIn('"status":"paused"', paused.stdout)
        self.assertEqual(self.invocation_count(), 1)
        progress = json.loads((self.output / "progress.json").read_text())
        self.assertEqual(progress["status"], "paused")
        self.assertEqual(progress["completed_samples"], 2)

        completed = self.invoke()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(self.invocation_count(), 3)
        expected = "".join(
            json.dumps(
                {
                    "sample_id": row["sample_id"],
                    "predicted_edge_ids": row["original_edge_ids"],
                },
                separators=(",", ":"),
            )
            + "\n"
            for row in self.rows
        )
        self.assertEqual((self.output / "predictions.jsonl").read_text(), expected)
        progress = json.loads((self.output / "progress.json").read_text())
        self.assertEqual(progress["status"], "completed")
        self.assertEqual(progress["completed_samples"], 5)
        diagnostics = json.loads((self.output / "diagnostics.json").read_text())
        self.assertFalse(diagnostics["efficiency_comparable"])
        self.assertEqual(diagnostics["samples"], 5)

        no_op = self.invoke()
        self.assertEqual(no_op.returncode, 0, no_op.stderr)
        self.assertEqual(self.invocation_count(), 3)

    def test_corrupt_shard_is_rerun_from_last_valid_commit(self) -> None:
        completed = self.invoke()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        corrupt = self.output / "shards" / "000001" / "predictions.jsonl"
        corrupt.write_text('{"sample_id":"wrong","predicted_edge_ids":[0]}\n')
        recovered = self.invoke()
        self.assertEqual(recovered.returncode, 0, recovered.stderr)
        self.assertEqual(self.invocation_count(), 4)
        marker = json.loads(
            (self.output / "shards" / "000001" / "complete.json").read_text()
        )
        self.assertEqual(marker["predictions_sha256"], digest(corrupt))

    def test_missing_final_hash_reassembles_without_rerunning_shards(self) -> None:
        completed = self.invoke()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        invocation_count = self.invocation_count()
        marker_path = self.output / "complete.json"

        for field, output_name in (
            ("predictions_sha256", "predictions.jsonl"),
            ("diagnostics_sha256", "diagnostics.json"),
            ("run_receipt_sha256", "run-receipt.json"),
        ):
            with self.subTest(field=field):
                marker = json.loads(marker_path.read_text())
                marker.pop(field)
                marker_path.write_text(json.dumps(marker), encoding="utf-8")

                recovered = self.invoke()
                self.assertEqual(recovered.returncode, 0, recovered.stderr)
                self.assertEqual(self.invocation_count(), invocation_count)
                repaired = json.loads(marker_path.read_text())
                self.assertEqual(repaired[field], digest(self.output / output_name))

    def test_changed_configuration_cannot_reuse_output_directory(self) -> None:
        completed = self.invoke()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        configuration_path = self.preprocess / "routing-configuration.json"
        configuration = json.loads(configuration_path.read_text())
        configuration["audit_note"] = "changed"
        configuration_path.write_text(json.dumps(configuration), encoding="utf-8")
        refused = self.invoke()
        self.assertEqual(refused.returncode, 2)
        self.assertIn("different code, data, artifacts, or settings", refused.stderr)
        self.assertEqual(self.invocation_count(), 3)

    def test_unsafe_unbounded_shard_size_is_rejected(self) -> None:
        refused = self.invoke("--shard-size", "8193")
        self.assertEqual(refused.returncode, 2)
        self.assertIn("shard size must be 1..8192", refused.stderr)
        self.assertEqual(self.invocation_count(), 0)

    def test_sigterm_discards_only_current_uncommitted_shard(self) -> None:
        started = self.root / "started"
        environment = dict(os.environ)
        environment.update(
            {
                "FAKE_ADAPTER_COUNTER": str(self.counter),
                "FAKE_ADAPTER_STARTED": str(started),
                "FAKE_ADAPTER_SLEEP_SECONDS": "30",
            }
        )
        process = subprocess.Popen(
            self.command(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
        deadline = time.monotonic() + 10
        while not started.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(started.exists(), "fake adapter did not start")
        process.terminate()
        stdout, stderr = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 130, (stdout, stderr))
        progress = json.loads((self.output / "progress.json").read_text())
        self.assertEqual(progress["status"], "stopped")
        self.assertEqual(progress["completed_shards"], 0)
        self.assertFalse(
            (self.output / "shards" / "000000" / "complete.json").exists()
        )

        resumed = self.invoke()
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        self.assertEqual(self.invocation_count(), 3)
        self.assertEqual(
            json.loads((self.output / "progress.json").read_text())["status"],
            "completed",
        )


if __name__ == "__main__":
    unittest.main()
