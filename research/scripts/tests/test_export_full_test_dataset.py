import hashlib
import importlib.util
import io
import json
import pickle
import struct
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "export_full_test_dataset.py"
SPEC = importlib.util.spec_from_file_location("export_full_test_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_dbf(path: Path, fields: list[str], rows: list[dict[str, int]]) -> None:
    width = 12
    header_length = 32 + 32 * len(fields) + 1
    record_length = 1 + width * len(fields)
    header = bytearray(32)
    header[0] = 3
    struct.pack_into("<I", header, 4, len(rows))
    struct.pack_into("<HH", header, 8, header_length, record_length)
    encoded = bytearray(header)
    for name in fields:
        descriptor = bytearray(32)
        descriptor[: len(name)] = name.encode("ascii")
        descriptor[11] = ord("N")
        descriptor[16] = width
        encoded.extend(descriptor)
    encoded.append(0x0D)
    for row in rows:
        encoded.extend(b" ")
        for name in fields:
            value = str(row[name]).encode("ascii")
            if len(value) > width:
                raise AssertionError("fixture value does not fit DBF field")
            encoded.extend(value.rjust(width, b" "))
    encoded.append(0x1A)
    path.write_bytes(encoded)


def create_map(root: Path, *, bad_fid: bool = False) -> Path:
    map_dir = root / "map"
    map_dir.mkdir()
    for name in MODULE.MAP_COMPONENTS:
        (map_dir / name).write_bytes(f"fixture:{name}".encode())
    write_dbf(
        map_dir / "nodes.dbf",
        ["osmid"],
        [{"osmid": node} for node in range(6)],
    )
    endpoints = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 1), (3, 0)]
    edge_rows = [
        {"fid": edge + (1 if bad_fid and edge == 3 else 0), "u": tail, "v": head}
        for edge, (tail, head) in enumerate(endpoints)
    ]
    write_dbf(map_dir / "edges.dbf", ["fid", "u", "v"], edge_rows)
    return map_dir


def write_pickle(path: Path) -> list[tuple[str, list[int], tuple[int, int]]]:
    trips = [
        ("duplicate", [0, 1, 2, 3, 4], (10, 20)),
        ("duplicate", [5, 1, 2, 3, 4], (20, 30)),
        ("empty", [], (30, 40)),
        ("short", [0, 1, 2, 3], (40, 50)),
        ("outside", [0, 1, 2, 3, 99], (50, 60)),
        ("broken", [0, 2, 3, 4, 4], (60, 70)),
        ("cycle", [0, 1, 2, 6, 0], (70, 80)),
    ]
    with path.open("wb") as writer:
        pickle.dump(trips, writer)
    return trips


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExportFullTestDatasetTests(unittest.TestCase):
    def configuration(
        self,
        source: Path,
        map_dir: Path,
        output: Path,
        *,
        eligible: int = 2,
    ) -> MODULE.ExportConfiguration:
        return MODULE.ExportConfiguration(
            source_pickle=source,
            map_dir=map_dir,
            output_root=output,
            dataset_id="beijing/full-common-test-fixture",
            network_id="beijing-roads-fixture",
            expected_source_sha256=digest(source),
            expected=MODULE.ExpectedCounts(
                source_records=7,
                eligible_records=eligible,
                nodes=6,
                edges=7,
            ),
            progress_every=2,
        )

    def test_exports_all_eligible_source_order_rows_with_a_balanced_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            map_dir = create_map(root)
            source = root / "test.pkl"
            write_pickle(source)
            output = root / "output"

            progress = io.StringIO()
            returned = MODULE.export_full_test(
                self.configuration(source, map_dir, output), progress=progress
            )
            manifests = output / "manifests"
            rows = [
                json.loads(line)
                for line in (manifests / "test.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                rows,
                [
                    {
                        "sample_id": "test:000000000",
                        "original_edge_ids": [0, 1, 2, 3, 4],
                    },
                    {
                        "sample_id": "test:000000001",
                        "original_edge_ids": [5, 1, 2, 3, 4],
                    },
                ],
            )

            manifest = json.loads((manifests / "test.manifest.json").read_text())
            self.assertEqual(manifest["records_file"], "test.jsonl")
            self.assertEqual(manifest["dataset_id"], "beijing/full-common-test-fixture")
            audit = json.loads((manifests / "test.audit.json").read_text())
            self.assertEqual(audit, returned)
            self.assertEqual(
                audit["filtering"],
                {
                    "eligible": 2,
                    "selected": 2,
                    "dropped": 5,
                    "empty": 1,
                    "too_short": 1,
                    "out_of_bounds_or_unrepresentable": 1,
                    "discontinuous": 1,
                    "cyclic": 1,
                    "eligibility_coverage": 2 / 7,
                    "selected_source_coverage": 2 / 7,
                },
            )
            self.assertEqual(audit["identity_audit"]["duplicate_original_trip_ids"], 1)
            self.assertEqual(audit["upstream_compatibility"]["graph_parallel_edge_ids"], 1)
            self.assertEqual(
                audit["upstream_compatibility"][
                    "selected_paths_that_upstream_condense_edges_would_change"
                ],
                1,
            )
            self.assertEqual(
                audit["outputs"]["records"]["sha256"],
                digest(manifests / "test.jsonl"),
            )
            self.assertEqual(audit["source"]["sha256"], digest(source))
            self.assertEqual(len(audit["network"]["components"]), 10)
            self.assertEqual(
                audit["outputs"]["manifest"]["sha256"],
                digest(manifests / "test.manifest.json"),
            )
            self.assertIn("7/7 source rows", progress.getvalue())
            self.assertFalse(any(output.glob(".manifests.export-*")))

            with self.assertRaises(FileExistsError):
                MODULE.export_full_test(
                    self.configuration(source, map_dir, output), progress=io.StringIO()
                )

    def test_count_mismatch_never_publishes_or_leaves_staging_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            map_dir = create_map(root)
            source = root / "test.pkl"
            write_pickle(source)
            output = root / "output"
            with self.assertRaisesRegex(ValueError, "eligible record count mismatch"):
                MODULE.export_full_test(
                    self.configuration(source, map_dir, output, eligible=3),
                    progress=io.StringIO(),
                )
            self.assertFalse((output / "manifests").exists())
            self.assertFalse(any(output.glob(".manifests.export-*")))

    def test_filter_precedence_matches_the_frozen_common_policy(self) -> None:
        topology = MODULE.RoadTopology(
            node_count=6,
            tail=(0, 1, 2, 3, 4, 0, 3),
            head=(1, 2, 3, 4, 5, 1, 0),
        )
        cases = [
            ([], "empty"),
            ([True, 99], "too_short"),
            ([0, 1, 2, 3, 99], "out_of_bounds_or_unrepresentable"),
            ([0, 2, 3, 4, 4], "discontinuous"),
            ([0, 1, 2, 6, 0], "cyclic"),
            ([0, 1, 2, 3, 4], None),
        ]
        for edges, expected in cases:
            with self.subTest(edges=edges):
                reason, accepted = MODULE.classify_path(edges, topology)
                self.assertEqual(reason, expected)
                self.assertEqual(accepted is not None, expected is None)

    def test_map_loader_rejects_noncanonical_raw_edge_fids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            map_dir = create_map(Path(temporary), bad_fid=True)
            with self.assertRaisesRegex(ValueError, "raw edge IDs require fid==record"):
                MODULE.load_road_topology(map_dir)

    def test_source_hash_and_shape_are_pinned_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            map_dir = create_map(root)
            source = root / "test.pkl"
            write_pickle(source)
            output = root / "output"
            configuration = self.configuration(source, map_dir, output)
            configuration = MODULE.ExportConfiguration(
                **{
                    **configuration.__dict__,
                    "expected_source_sha256": "0" * 64,
                }
            )
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                MODULE.export_full_test(configuration, progress=io.StringIO())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
