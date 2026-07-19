#!/usr/bin/env python3
"""Build the full common-filter test dataset without opening any model artifact.

The exporter intentionally fixes the historical cross-method structural policy:
complete paths in raw Shapefile-record edge IDs, at least five edges, continuous
adjacent edges, and no repeated original road-network node.  The source-list
index, rather than the upstream trip ID, is the stable sample identity.

All three output files are written below a private staging directory and the
whole ``manifests`` directory is renamed into place only after validation and
fsync.  An interrupted or rejected export therefore cannot publish a manifest
that names a partial JSONL file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import secrets
import shutil
import struct
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, TextIO


AUDIT_SCHEMA = "ewr.common-test-export-audit/v1"
DATASET_MANIFEST_SCHEMA = "ewr.dataset-manifest/v1"
DATASET_RECORD_SCHEMA = "ewr.dataset-record/v1"
MINIMUM_EDGES = 5
SPLIT = "test"
MANIFEST_DIRECTORY = "manifests"
RECORDS_FILENAME = "test.jsonl"
MANIFEST_FILENAME = "test.manifest.json"
AUDIT_FILENAME = "test.audit.json"
MAP_COMPONENTS = (
    "edges.cpg",
    "edges.dbf",
    "edges.prj",
    "edges.shp",
    "edges.shx",
    "nodes.cpg",
    "nodes.dbf",
    "nodes.prj",
    "nodes.shp",
    "nodes.shx",
)


@dataclass(frozen=True)
class RoadTopology:
    node_count: int
    tail: tuple[int, ...]
    head: tuple[int, ...]

    @property
    def edge_count(self) -> int:
        return len(self.tail)


@dataclass
class FilterCounts:
    empty: int = 0
    too_short: int = 0
    out_of_bounds_or_unrepresentable: int = 0
    discontinuous: int = 0
    cyclic: int = 0

    @property
    def dropped(self) -> int:
        return sum(asdict(self).values())


@dataclass(frozen=True)
class ExpectedCounts:
    source_records: int
    eligible_records: int
    nodes: int
    edges: int


@dataclass(frozen=True)
class ExportConfiguration:
    source_pickle: Path
    map_dir: Path
    output_root: Path
    dataset_id: str
    network_id: str
    expected_source_sha256: str
    expected: ExpectedCounts
    progress_every: int = 25_000


@dataclass(frozen=True)
class DbfField:
    name: str
    kind: str
    offset: int
    length: int
    decimals: int


class HashingWriter:
    def __init__(self, writer: BinaryIO):
        self.writer = writer
        self.digest = hashlib.sha256()
        self.bytes_written = 0

    def write(self, value: bytes) -> None:
        self.writer.write(value)
        self.digest.update(value)
        self.bytes_written += len(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export every eligible row of the released test pickle under the "
            "frozen common structural filter. This command never runs a model."
        )
    )
    parser.add_argument("--source-pickle", type=Path, required=True)
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--network-id", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-source-records", type=positive_int, required=True)
    parser.add_argument("--expected-eligible-records", type=positive_int, required=True)
    parser.add_argument("--expected-nodes", type=positive_int, required=True)
    parser.add_argument("--expected-edges", type=positive_int, required=True)
    parser.add_argument("--progress-every", type=positive_int, default=25_000)
    return parser.parse_args(argv)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def validate_label(value: str, label: str) -> None:
    if not value.strip() or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise ValueError(f"{label} must be nonempty and contain no control characters")


def validate_sha256(value: str, label: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be 64 hexadecimal characters")
    return normalized


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        while block := reader.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, relative_path: str) -> dict[str, Any]:
    return {
        "path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_dbf_fields(path: Path, required: Iterable[str]) -> list[dict[str, int]]:
    """Read selected integer DBF columns without a third-party GIS package."""

    required_names = tuple(required)
    with path.open("rb") as reader:
        header = reader.read(32)
        if len(header) != 32:
            raise ValueError(f"{path} has a truncated DBF header")
        record_count = struct.unpack_from("<I", header, 4)[0]
        header_length, record_length = struct.unpack_from("<HH", header, 8)
        if header_length < 33 or record_length < 2:
            raise ValueError(f"{path} has invalid DBF dimensions")

        descriptor_bytes = reader.read(header_length - 32)
        if len(descriptor_bytes) != header_length - 32 or not descriptor_bytes.endswith(b"\r"):
            raise ValueError(f"{path} has a malformed DBF field header")
        descriptors = descriptor_bytes[:-1]
        if len(descriptors) % 32:
            raise ValueError(f"{path} has a misaligned DBF field header")

        fields: dict[str, DbfField] = {}
        offset = 1
        for start in range(0, len(descriptors), 32):
            descriptor = descriptors[start : start + 32]
            name = descriptor[:11].split(b"\0", 1)[0].decode("ascii")
            length = descriptor[16]
            field = DbfField(
                name=name,
                kind=chr(descriptor[11]),
                offset=offset,
                length=length,
                decimals=descriptor[17],
            )
            if not name or length == 0 or name in fields:
                raise ValueError(f"{path} has an invalid or duplicate DBF field {name!r}")
            fields[name] = field
            offset += length
        if offset != record_length:
            raise ValueError(
                f"{path} DBF field widths total {offset}, expected record length {record_length}"
            )

        missing = [name for name in required_names if name not in fields]
        if missing:
            raise ValueError(f"{path} lacks required DBF fields {missing}")
        for name in required_names:
            field = fields[name]
            if field.kind not in {"N", "F"} or field.decimals != 0:
                raise ValueError(f"{path} field {name!r} is not an integer numeric field")

        rows: list[dict[str, int]] = []
        for index in range(record_count):
            record = reader.read(record_length)
            if len(record) != record_length:
                raise ValueError(f"{path} DBF record {index} is truncated")
            if record[0:1] != b" ":
                raise ValueError(f"{path} DBF record {index} is deleted or invalid")
            row: dict[str, int] = {}
            for name in required_names:
                field = fields[name]
                encoded = record[field.offset : field.offset + field.length].strip()
                if not encoded:
                    raise ValueError(f"{path} DBF record {index} has empty {name!r}")
                try:
                    value = int(encoded)
                except ValueError as error:
                    raise ValueError(
                        f"{path} DBF record {index} has non-integer {name!r}"
                    ) from error
                if value < 0:
                    raise ValueError(f"{path} DBF record {index} has negative {name!r}")
                row[name] = value
            rows.append(row)

        trailer = reader.read(1)
        if trailer not in {b"", b"\x1a"}:
            raise ValueError(f"{path} has trailing data after its DBF records")
        if reader.read(1):
            raise ValueError(f"{path} has trailing data after its DBF terminator")
        return rows


def load_road_topology(map_dir: Path) -> tuple[RoadTopology, dict[str, Any]]:
    map_dir = map_dir.resolve()
    components: list[dict[str, Any]] = []
    combined = hashlib.sha256()
    for name in MAP_COMPONENTS:
        path = map_dir / name
        if not path.is_file():
            raise ValueError(f"map component is missing: {path}")
        identity = file_identity(path, name)
        components.append(identity)
        combined.update(name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(bytes.fromhex(identity["sha256"]))

    node_rows = read_dbf_fields(map_dir / "nodes.dbf", ("osmid",))
    node_ids = {row["osmid"] for row in node_rows}
    if len(node_ids) != len(node_rows):
        raise ValueError("nodes.dbf contains duplicate osmid values")

    edge_rows = read_dbf_fields(map_dir / "edges.dbf", ("fid", "u", "v"))
    tail: list[int] = []
    head: list[int] = []
    for index, row in enumerate(edge_rows):
        if row["fid"] != index:
            raise ValueError(
                f"edges.dbf record {index} has fid {row['fid']}; raw edge IDs require fid==record"
            )
        if row["u"] not in node_ids or row["v"] not in node_ids:
            raise ValueError(f"edges.dbf record {index} references an unknown endpoint")
        tail.append(row["u"])
        head.append(row["v"])

    topology = RoadTopology(len(node_rows), tuple(tail), tuple(head))
    identity = {
        "map_dir": str(map_dir),
        "map_sha256": combined.hexdigest(),
        "hash_definition": "sha256(component_utf8 + NUL + component_sha256_bytes in listed order)",
        "components": components,
        "nodes": topology.node_count,
        "edges": topology.edge_count,
    }
    return topology, identity


def classify_path(edges: Any, topology: RoadTopology) -> tuple[str | None, list[int] | None]:
    if not isinstance(edges, list):
        raise ValueError("trajectory edges must be a list")
    if not edges:
        return "empty", None
    if len(edges) < MINIMUM_EDGES:
        return "too_short", None
    if any(
        isinstance(edge, bool)
        or not isinstance(edge, int)
        or edge < 0
        or edge > 0xFFFF_FFFF
        or edge >= topology.edge_count
        for edge in edges
    ):
        return "out_of_bounds_or_unrepresentable", None
    if any(
        topology.head[previous] != topology.tail[next_edge]
        for previous, next_edge in zip(edges, edges[1:])
    ):
        return "discontinuous", None

    visited = {topology.tail[edges[0]]}
    for edge in edges:
        if topology.head[edge] in visited:
            return "cyclic", None
        visited.add(topology.head[edge])
    return None, edges


def validate_raw_trip(value: Any, source_index: int) -> tuple[str, list[int]]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"source record {source_index} is not a three-field tuple")
    original_trip_id, edges, timestamps = value
    if (
        not isinstance(timestamps, tuple)
        or len(timestamps) != 2
        or any(
            isinstance(timestamp, bool)
            or not isinstance(timestamp, int)
            or timestamp < 0
            or timestamp > 0xFFFF_FFFF_FFFF_FFFF
            for timestamp in timestamps
        )
    ):
        raise ValueError(f"source record {source_index} has invalid timestamps")
    return str(original_trip_id), edges


def canonical_parallel_edges(topology: RoadTopology) -> tuple[list[int], int]:
    last: dict[tuple[int, int], int] = {}
    for edge, endpoints in enumerate(zip(topology.tail, topology.head)):
        last[endpoints] = edge
    canonical = [last[endpoints] for endpoints in zip(topology.tail, topology.head)]
    return canonical, sum(edge != representative for edge, representative in enumerate(canonical))


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def encode_json(value: Any, *, pretty: bool) -> bytes:
    if pretty:
        return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def write_synced(path: Path, value: bytes) -> dict[str, Any]:
    with path.open("xb") as writer:
        writer.write(value)
        writer.flush()
        os.fsync(writer.fileno())
    return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}


def validate_configuration(configuration: ExportConfiguration) -> str:
    validate_label(configuration.dataset_id, "dataset_id")
    validate_label(configuration.network_id, "network_id")
    if configuration.progress_every <= 0:
        raise ValueError("progress_every must be positive")
    expected_sha256 = validate_sha256(
        configuration.expected_source_sha256, "expected_source_sha256"
    )
    for label, value in asdict(configuration.expected).items():
        if value <= 0:
            raise ValueError(f"expected {label} must be positive")
    return expected_sha256


def export_full_test(
    configuration: ExportConfiguration, *, progress: TextIO = sys.stderr
) -> dict[str, Any]:
    expected_sha256 = validate_configuration(configuration)
    source = configuration.source_pickle.resolve()
    if not source.is_file():
        raise ValueError(f"source pickle is missing: {source}")

    output_root = configuration.output_root.resolve()
    destination = output_root / MANIFEST_DIRECTORY
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to replace existing export directory: {destination}")

    source_identity = file_identity(source, str(configuration.source_pickle))
    if source_identity["sha256"] != expected_sha256:
        raise ValueError(
            "source pickle SHA-256 mismatch: "
            f"expected {expected_sha256}, got {source_identity['sha256']}"
        )

    print(f"[export] validating map {configuration.map_dir}", file=progress, flush=True)
    topology, map_identity = load_road_topology(configuration.map_dir)
    if topology.node_count != configuration.expected.nodes:
        raise ValueError(
            f"network node count mismatch: expected {configuration.expected.nodes}, "
            f"got {topology.node_count}"
        )
    if topology.edge_count != configuration.expected.edges:
        raise ValueError(
            f"network edge count mismatch: expected {configuration.expected.edges}, "
            f"got {topology.edge_count}"
        )

    print(f"[export] decoding {source}", file=progress, flush=True)
    with source.open("rb") as reader:
        raw = pickle.load(reader)
        if reader.read(1):
            raise ValueError("source pickle contains trailing data")
    if not isinstance(raw, list) or not raw:
        raise ValueError("source pickle must contain a nonempty list")
    if len(raw) != configuration.expected.source_records:
        raise ValueError(
            f"source record count mismatch: expected {configuration.expected.source_records}, "
            f"got {len(raw)}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    staging = output_root / (
        f".{MANIFEST_DIRECTORY}.export-{os.getpid()}-{secrets.token_hex(6)}"
    )
    staging.mkdir()
    committed = False
    started = time.monotonic()
    try:
        filters = FilterCounts()
        eligible = 0
        selected_unique_edges: set[int] = set()
        selected_minimum_edges: int | None = None
        selected_maximum_edges = 0
        selected_edge_occurrences = 0
        original_ids: set[str] = set()
        duplicate_original_trip_ids = 0
        canonical, graph_parallel_edge_ids = canonical_parallel_edges(topology)
        changed_by_condensation = 0

        records_path = staging / RECORDS_FILENAME
        with records_path.open("xb") as raw_writer:
            writer = HashingWriter(raw_writer)
            for source_index, value in enumerate(raw):
                original_trip_id, raw_edges = validate_raw_trip(value, source_index)
                if original_trip_id in original_ids:
                    duplicate_original_trip_ids += 1
                else:
                    original_ids.add(original_trip_id)

                reason, edges = classify_path(raw_edges, topology)
                if reason is not None:
                    setattr(filters, reason, getattr(filters, reason) + 1)
                else:
                    assert edges is not None
                    eligible += 1
                    selected_unique_edges.update(edges)
                    selected_minimum_edges = (
                        len(edges)
                        if selected_minimum_edges is None
                        else min(selected_minimum_edges, len(edges))
                    )
                    selected_maximum_edges = max(selected_maximum_edges, len(edges))
                    selected_edge_occurrences += len(edges)
                    if any(edge != canonical[edge] for edge in edges):
                        changed_by_condensation += 1
                    writer.write(
                        encode_json(
                            {
                                "sample_id": f"{SPLIT}:{source_index:09d}",
                                "original_edge_ids": edges,
                            },
                            pretty=False,
                        )
                    )

                processed = source_index + 1
                if processed % configuration.progress_every == 0 or processed == len(raw):
                    elapsed = time.monotonic() - started
                    rate = processed / elapsed if elapsed else 0.0
                    print(
                        f"[export] {processed}/{len(raw)} source rows "
                        f"({processed / len(raw):.1%}), eligible={eligible}, "
                        f"{rate:.0f} rows/s",
                        file=progress,
                        flush=True,
                    )
            raw_writer.flush()
            os.fsync(raw_writer.fileno())

        if filters.dropped + eligible != len(raw):
            raise RuntimeError("filter audit does not balance")
        if eligible != configuration.expected.eligible_records:
            raise ValueError(
                f"eligible record count mismatch: expected "
                f"{configuration.expected.eligible_records}, got {eligible}"
            )
        if eligible == 0 or selected_minimum_edges is None:
            raise ValueError("the common filter selected no trajectories")

        manifest = {
            "schema": DATASET_MANIFEST_SCHEMA,
            "dataset_id": configuration.dataset_id,
            "network_id": configuration.network_id,
            "records_schema": DATASET_RECORD_SCHEMA,
            "records_file": RECORDS_FILENAME,
        }
        manifest_identity = write_synced(
            staging / MANIFEST_FILENAME, encode_json(manifest, pretty=True)
        )

        audit = {
            "schema": AUDIT_SCHEMA,
            "status": "complete",
            "split": SPLIT,
            "source": {**source_identity, "records": len(raw)},
            "network": map_identity,
            "policy": {
                "road_id_space": "unaltered_shapefile_record_index",
                "minimum_edges": MINIMUM_EDGES,
                "continuity": "head(previous)==tail(next)",
                "cycle_policy": "drop_if_any_original_node_repeats",
                "selection": "all_eligible_in_source_order",
                "sample_id": "test:{source_pickle_zero_based_index:09d}",
                "neuromlr_traffic_features": False,
            },
            "filtering": {
                "eligible": eligible,
                "selected": eligible,
                "dropped": filters.dropped,
                **asdict(filters),
                "eligibility_coverage": ratio(eligible, len(raw)),
                "selected_source_coverage": ratio(eligible, len(raw)),
            },
            "identity_audit": {
                "duplicate_original_trip_ids": duplicate_original_trip_ids,
                "selected_unique_edges": len(selected_unique_edges),
                "selected_minimum_edges": selected_minimum_edges,
                "selected_maximum_edges": selected_maximum_edges,
                "selected_edge_occurrences": selected_edge_occurrences,
            },
            "upstream_compatibility": {
                "graph_parallel_edge_ids": graph_parallel_edge_ids,
                "selected_paths_that_upstream_condense_edges_would_change": (
                    changed_by_condensation
                ),
                "adapter_policy": "preserve_raw_edge_ids_one_to_one",
            },
            "pins": {
                "source_sha256_enforced": True,
                "source_records_enforced": True,
                "eligible_records_enforced": True,
                "network_node_count_enforced": True,
                "network_edge_count_enforced": True,
            },
            "outputs": {
                "records": {
                    "path": RECORDS_FILENAME,
                    "records": eligible,
                    "bytes": writer.bytes_written,
                    "sha256": writer.digest.hexdigest(),
                },
                "manifest": {
                    "path": MANIFEST_FILENAME,
                    **manifest_identity,
                },
                "audit": {"path": AUDIT_FILENAME},
            },
            "publication": {
                "boundary": "staged_files_fsync_then_atomic_directory_rename",
                "destination": str(destination),
                "existing_destination_policy": "fail_without_overwrite",
            },
        }
        write_synced(staging / AUDIT_FILENAME, encode_json(audit, pretty=True))

        directory_descriptor = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        if os.path.lexists(destination):
            raise FileExistsError(
                f"refusing to replace concurrently created export directory: {destination}"
            )
        os.rename(staging, destination)
        committed = True
        root_descriptor = os.open(output_root, os.O_RDONLY)
        try:
            os.fsync(root_descriptor)
        finally:
            os.close(root_descriptor)

        elapsed = time.monotonic() - started
        print(
            f"[export] published {eligible} rows to {destination} in {elapsed:.1f}s",
            file=progress,
            flush=True,
        )
        return audit
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)


def configuration_from_args(args: argparse.Namespace) -> ExportConfiguration:
    return ExportConfiguration(
        source_pickle=args.source_pickle,
        map_dir=args.map_dir,
        output_root=args.output_root,
        dataset_id=args.dataset_id,
        network_id=args.network_id,
        expected_source_sha256=args.expected_source_sha256,
        expected=ExpectedCounts(
            source_records=args.expected_source_records,
            eligible_records=args.expected_eligible_records,
            nodes=args.expected_nodes,
            edges=args.expected_edges,
        ),
        progress_every=args.progress_every,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    export_full_test(configuration_from_args(args))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EOFError, OSError, pickle.PickleError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
