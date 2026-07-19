#!/usr/bin/env python3
"""Clean-room DRPK-static adapter for the EWR version-one protocol.

This module implements the algorithm from the published DRPK description and
the behavioral audit recorded in ``upstream.json``.  It imports no upstream
DRPK source.  The temporal popularity table is deliberately collapsed to one
training-only global edge-frequency vector, min-max normalized to [0, 1], and
replicated across 48 slots; query timestamps therefore cannot influence this
baseline.
"""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import hashlib
import heapq
import importlib.metadata
import json
import math
import os
import platform
import random
import resource
import struct
import tempfile
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Iterator, Sequence

OFFICIAL_REPOSITORY = "https://github.com/derekwtian/DRPK"
OFFICIAL_COMMIT = "2f65eaeb784d7266b591196c795d72cf909294d8"
OFFICIAL_LICENSE = "MIT"
DATASET_MANIFEST_SCHEMA = "ewr.dataset-manifest/v1"
DATASET_RECORD_SCHEMA = "ewr.dataset-record/v1"
PREDICTION_RECORD_SCHEMA = "ewr.prediction-record/v1"
RUN_RECEIPT_SCHEMA = "ewr.run-receipt/v1"
ADAPTER_VERSION = "0.3.0"
PREPROCESS_SCHEMA = "ewr.drpk-static-preprocess/v2"
ROUTING_PREPROCESS_SCHEMA = "ewr.drpk-static-routing-preprocess/v1"
PREPROCESS_DIAGNOSTICS_SCHEMA = "ewr.drpk-static-preprocess-diagnostics/v2"
CANDIDATE_ROWS_SCHEMA = "ewr.drpk-static-candidate-rows/v2"
STATIC_FEATURES_SCHEMA = "ewr.drpk-static-features/v2"
KSD_MODEL_SCHEMA = "ewr.drpk-static-ksd-paper-eq6-9/v1"
TRAINING_SCHEMA = "ewr.drpk-static-training/v2"
SELECTION_SCHEMA = "ewr.drpk-static-selection/v2"
CHECKPOINT_SCHEMA = "ewr.drpk-static-checkpoint/v2"
DIAGNOSTICS_SCHEMA = "ewr.drpk-static-diagnostics/v2"
CORE_ARTIFACT_MANIFEST_SCHEMA = "ewr.drpk-static-core-artifacts/v1"
ROUTING_ARTIFACT_MANIFEST_SCHEMA = "ewr.drpk-static-routing-artifacts/v1"
DEFAULT_SEED = 20260718
DEFAULT_WORKERS = 16
TIME_SLOT_COUNT = 48
FIXED_SOURCE_OFFSET = 0.0
FIXED_DESTINATION_OFFSET = 1.0
U32_MAX = 0xFFFF_FFFF

np: Any = None
shapefile: Any = None
torch: Any = None
F: Any = None
_MODEL_DEPENDENCY_ERROR: ImportError | None = None
_ARRAY_DEPENDENCY_ERROR: ImportError | None = None
_MAP_DEPENDENCY_ERROR: ImportError | None = None


def load_array_dependencies() -> None:
    """Load only NumPy for graph/DA routing artifacts."""

    global _ARRAY_DEPENDENCY_ERROR, np
    if np is not None:
        return
    if _ARRAY_DEPENDENCY_ERROR is not None:
        raise RuntimeError(
            f"DRPK-static array dependency is unavailable: {_ARRAY_DEPENDENCY_ERROR}"
        ) from _ARRAY_DEPENDENCY_ERROR
    try:
        import numpy as numpy_module
    except ImportError as error:
        _ARRAY_DEPENDENCY_ERROR = error
        raise RuntimeError(
            f"DRPK-static array dependency is unavailable: {error}"
        ) from error
    np = numpy_module


def load_model_dependencies() -> None:
    """Load the scientific stack only for model-side commands."""

    global F, _MODEL_DEPENDENCY_ERROR, np, shapefile, torch
    if np is not None and shapefile is not None and torch is not None and F is not None:
        return
    if _MODEL_DEPENDENCY_ERROR is not None:
        raise RuntimeError(
            f"DRPK-static model dependencies are unavailable: {_MODEL_DEPENDENCY_ERROR}"
        ) from _MODEL_DEPENDENCY_ERROR
    load_preprocess_dependencies()
    try:
        import torch as torch_module
        import torch.nn.functional as functional_module
    except ImportError as error:
        _MODEL_DEPENDENCY_ERROR = error
        raise RuntimeError(
            f"DRPK-static model dependencies are unavailable: {error}"
        ) from error
    torch = torch_module
    F = functional_module


def load_preprocess_dependencies() -> None:
    """Load NumPy and PyShp without importing or initializing PyTorch."""

    global _MAP_DEPENDENCY_ERROR, shapefile
    load_array_dependencies()
    if shapefile is not None:
        return
    if _MAP_DEPENDENCY_ERROR is not None:
        raise RuntimeError(
            f"DRPK-static map dependency is unavailable: {_MAP_DEPENDENCY_ERROR}"
        ) from _MAP_DEPENDENCY_ERROR
    try:
        import shapefile as shapefile_module
    except ImportError as error:
        _MAP_DEPENDENCY_ERROR = error
        raise RuntimeError(
            f"DRPK-static map dependency is unavailable: {error}"
        ) from error
    shapefile = shapefile_module


@dataclass(frozen=True)
class Trip:
    sample_id: str
    edges: list[int]


@dataclass(frozen=True)
class DatasetManifest:
    schema: str
    dataset_id: str
    network_id: str
    records_schema: str
    records_file: str


@dataclass(frozen=True)
class DatasetArtifact:
    manifest: DatasetManifest
    manifest_path: Path
    manifest_sha256: str
    records_sha256: str
    trips: list[Trip]


@dataclass(frozen=True)
class StaticGraph:
    """Directed raw-road line graph with original road IDs as indices."""

    tail: list[int]
    head: list[int]
    node_x: list[float]
    node_y: list[float]
    neighbors: list[tuple[int, ...]]
    identity: str

    @property
    def edge_count(self) -> int:
        return len(self.tail)

    def source_xy(self, edge: int) -> tuple[float, float]:
        node = self.tail[edge]
        return self.node_x[node], self.node_y[node]

    def target_xy(self, edge: int) -> tuple[float, float]:
        node = self.head[edge]
        return self.node_x[node], self.node_y[node]


@dataclass(frozen=True)
class SparseDA:
    """Both CSR and CSC views of the directed-association count matrix."""

    size: int
    row_offsets: Sequence[int]
    row_indices: Sequence[int]
    row_values: Sequence[int]
    col_offsets: Sequence[int]
    col_indices: Sequence[int]
    col_values: Sequence[int]

    @classmethod
    def from_counts(cls, size: int, counts: dict[tuple[int, int], int]) -> "SparseDA":
        if size <= 0:
            raise RuntimeError("DA size must be positive")
        rows: list[list[tuple[int, int]]] = [[] for _ in range(size)]
        cols: list[list[tuple[int, int]]] = [[] for _ in range(size)]
        for (source, destination), value in counts.items():
            if not (0 <= source < size and 0 <= destination < size):
                raise RuntimeError("DA entry is outside the road vocabulary")
            if value <= 0:
                raise RuntimeError("DA counts must be positive")
            rows[source].append((destination, int(value)))
            cols[destination].append((source, int(value)))
        for entries in rows:
            entries.sort()
        for entries in cols:
            entries.sort()
        row_offsets, row_indices, row_values = _compress_sparse_rows(rows)
        col_offsets, col_indices, col_values = _compress_sparse_rows(cols)
        return cls(
            size=size,
            row_offsets=tuple(row_offsets),
            row_indices=tuple(row_indices),
            row_values=tuple(row_values),
            col_offsets=tuple(col_offsets),
            col_indices=tuple(col_indices),
            col_values=tuple(col_values),
        )

    @property
    def nonzero(self) -> int:
        return len(self.row_indices)

    def value(self, source: int, destination: int) -> int:
        if not (0 <= source < self.size and 0 <= destination < self.size):
            return 0
        left = self.row_offsets[source]
        right = self.row_offsets[source + 1]
        position = bisect.bisect_left(self.row_indices, destination, left, right)
        if position < right and self.row_indices[position] == destination:
            return int(self.row_values[position])
        return 0

    def candidate_pool(
        self, source: int, destination: int, limit: int
    ) -> list[tuple[int, int]]:
        """Return top candidates by min(sigma(source,k), sigma(k,destination))."""

        if limit <= 0:
            raise RuntimeError("candidate pool limit must be positive")
        if not (0 <= source < self.size and 0 <= destination < self.size):
            raise RuntimeError("candidate query is outside the DA matrix")
        row_left, row_right = self.row_offsets[source : source + 2]
        col_left, col_right = self.col_offsets[destination : destination + 2]
        row_position, col_position = row_left, col_left
        candidates: list[tuple[int, int]] = []
        while row_position < row_right and col_position < col_right:
            row_edge = self.row_indices[row_position]
            col_edge = self.col_indices[col_position]
            if row_edge == col_edge:
                # Compact DA arrays are uint32 memmaps.  Convert before unary
                # negation in the deterministic descending sort; otherwise
                # NumPy wraps ``-uint32`` and reverses rankings for positive
                # association strengths.
                strength = min(
                    int(self.row_values[row_position]),
                    int(self.col_values[col_position]),
                )
                if strength > 0:
                    candidates.append((int(row_edge), strength))
                row_position += 1
                col_position += 1
            elif row_edge < col_edge:
                row_position += 1
            else:
                col_position += 1
        candidates.sort(key=lambda item: (-item[1], item[0]))
        return candidates[:limit]


def _compress_sparse_rows(
    rows: Sequence[Sequence[tuple[int, int]]],
) -> tuple[list[int], list[int], list[int]]:
    offsets = [0]
    indices: list[int] = []
    values: list[int] = []
    for entries in rows:
        indices.extend(index for index, _ in entries)
        values.extend(value for _, value in entries)
        offsets.append(len(indices))
    return offsets, indices, values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess")
    preprocess.add_argument("--map-dir", type=Path, required=True)
    preprocess.add_argument("--train-manifest", type=Path, required=True)
    preprocess.add_argument("--validation-manifest", type=Path, required=True)
    preprocess.add_argument("--output-dir", type=Path, required=True)
    preprocess.add_argument("--source-revision", required=True)
    preprocess.add_argument("--seed", type=int, default=DEFAULT_SEED)
    preprocess.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    preprocess.add_argument("--candidate-pool-size", type=int, default=100)
    preprocess.add_argument("--da-chunk-pair-events", type=int, default=1_000_000)
    preprocess.add_argument("--positive-route-fraction", type=float, default=0.2)
    preprocess.add_argument("--node2vec-dim", type=int, default=64)
    preprocess.add_argument("--walk-length", type=int, default=30)
    preprocess.add_argument("--walks-per-edge", type=int, default=25)
    preprocess.add_argument("--node2vec-p", type=float, default=1.0)
    preprocess.add_argument("--node2vec-q", type=float, default=1.0)
    preprocess.add_argument("--node2vec-window", type=int, default=5)
    preprocess.add_argument("--node2vec-epochs", type=int, default=10)
    preprocess.add_argument("--node2vec-negative-samples", type=int, default=5)
    preprocess.add_argument("--node2vec-batch-size", type=int, default=4096)
    preprocess.add_argument("--node2vec-learning-rate", type=float, default=0.025)
    preprocess.add_argument(
        "--node2vec-engine", choices=["gensim", "torch"], default="gensim"
    )

    train = subparsers.add_parser("train")
    train.add_argument("--preprocess-dir", type=Path, required=True)
    train.add_argument("--validation-manifest", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--source-revision", required=True)
    train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    train.add_argument(
        "--device", default="cuda:0", help="cpu, auto, cuda, or cuda:N"
    )
    train.add_argument("--epochs", type=int, default=300)
    train.add_argument("--validation-every", type=int, default=1)
    train.add_argument(
        "--batch-size", type=int, default=8192, help="optimizer-step route batch"
    )
    train.add_argument(
        "--microbatch-size",
        type=int,
        default=512,
        help="memory-bounded gradient accumulation chunk",
    )
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--query-hidden-size", type=int, default=2048)
    train.add_argument("--representation-size", type=int, default=256)
    train.add_argument("--candidate-embedding-size", type=int, default=64)
    train.add_argument("--candidate-hidden-size", type=int, default=512)
    train.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="non-paper variant; formal Eq. (7)/(9) uses 0",
    )
    train.add_argument("--max-route-length", type=int, default=300)
    train.add_argument("--scheduler-patience", type=int, default=2)
    train.add_argument("--scheduler-factor", type=float, default=0.8)
    train.add_argument("--scheduler-threshold", type=float, default=0.001)
    train.add_argument(
        "--early-stop-learning-rate",
        type=float,
        default=0.0,
        help="optional positive LR threshold; 0 disables early stopping",
    )

    predict = subparsers.add_parser("predict")
    predict.add_argument("--preprocess-dir", type=Path, required=True)
    predict.add_argument("--checkpoint", type=Path)
    predict.add_argument("--dataset-manifest", type=Path, required=True)
    predict.add_argument(
        "--method", choices=["drpk_static", "drp_tp"], required=True
    )
    predict.add_argument("--predictions", type=Path, required=True)
    predict.add_argument("--run-receipt", type=Path, required=True)
    predict.add_argument("--diagnostics", type=Path, required=True)
    predict.add_argument("--source-revision", required=True)
    predict.add_argument("--seed", type=int, default=DEFAULT_SEED)
    predict.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    predict.add_argument(
        "--device", default="cuda:0", help="model device; ignored by drp_tp"
    )
    predict.add_argument("--inference-batch-size", type=int, default=32)
    predict.add_argument("--warmup-repetitions", type=int, default=0)
    predict.add_argument("--measured-repetitions", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "predict" and args.method == "drp_tp":
        load_array_dependencies()
        seed_base_dependencies(args.seed, args.workers)
    elif args.command == "preprocess":
        load_preprocess_dependencies()
        seed_base_dependencies(args.seed, args.workers)
    else:
        load_model_dependencies()
        seed_everything(args.seed, args.workers)
    if args.command == "preprocess":
        preprocess_command(args)
    elif args.command == "train":
        train_command(args)
    else:
        predict_command(args)


def seed_everything(seed: int, workers: int) -> None:
    seed_base_dependencies(seed, workers)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.set_num_threads(workers)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(max(1, min(workers, 4)))
        except RuntimeError:
            pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_base_dependencies(seed: int, workers: int) -> None:
    """Seed DRP-TP without importing or initializing PyTorch."""

    if workers <= 0:
        raise RuntimeError("workers must be positive")
    random.seed(seed)
    np.random.seed(seed)


def resolve_model_device(requested: str) -> Any:
    """Resolve CPU/CUDA strictly; only ``auto`` may choose a fallback."""

    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" or requested.startswith("cuda:"):
        try:
            device = torch.device(requested)
        except (RuntimeError, ValueError) as error:
            raise RuntimeError(f"invalid model device {requested!r}") from error
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"model device {requested!r} was requested but CUDA is unavailable"
            )
        index = torch.cuda.current_device() if device.index is None else device.index
        if not (0 <= index < torch.cuda.device_count()):
            raise RuntimeError(
                f"model device {requested!r} is outside the available CUDA devices"
            )
        return torch.device("cuda", index)
    raise RuntimeError("model device must be 'cpu', 'auto', 'cuda', or 'cuda:N'")


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def decode_json(text: str, kind: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=reject_duplicate_keys)
    except (json.JSONDecodeError, RuntimeError) as error:
        raise RuntimeError(f"invalid {kind}: {error}") from error


def require_exact_fields(value: Any, expected: set[str], kind: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{kind} must be a JSON object")
    actual = set(value)
    if actual != expected:
        raise RuntimeError(
            f"{kind} fields differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    return value


def require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{field} must be a nonempty string")
    return value


def load_dataset_manifest(path: Path) -> DatasetArtifact:
    encoded = path.read_bytes()
    descriptor = require_exact_fields(
        decode_json(encoded.decode("utf-8"), "dataset manifest JSON"),
        {"schema", "dataset_id", "network_id", "records_schema", "records_file"},
        "dataset manifest",
    )
    manifest = DatasetManifest(
        schema=require_nonempty_string(descriptor["schema"], "schema"),
        dataset_id=require_nonempty_string(descriptor["dataset_id"], "dataset_id"),
        network_id=require_nonempty_string(descriptor["network_id"], "network_id"),
        records_schema=require_nonempty_string(
            descriptor["records_schema"], "records_schema"
        ),
        records_file=require_nonempty_string(descriptor["records_file"], "records_file"),
    )
    if manifest.schema != DATASET_MANIFEST_SCHEMA:
        raise RuntimeError(f"unsupported dataset manifest schema {manifest.schema!r}")
    if manifest.records_schema != DATASET_RECORD_SCHEMA:
        raise RuntimeError(f"unsupported dataset record schema {manifest.records_schema!r}")
    relative = Path(manifest.records_file)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("records_file must be a safe path relative to its manifest")
    records_path = path.parent / relative
    records = records_path.read_bytes()
    trips = load_dataset_records(records_path)
    return DatasetArtifact(
        manifest=manifest,
        manifest_path=path.resolve(),
        manifest_sha256=hashlib.sha256(encoded).hexdigest(),
        records_sha256=hashlib.sha256(records).hexdigest(),
        trips=trips,
    )


def load_dataset_records(path: Path) -> list[Trip]:
    trips: list[Trip] = []
    sample_ids: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise RuntimeError(f"blank dataset JSONL line {line_number}")
            row = require_exact_fields(
                decode_json(line, f"dataset JSONL line {line_number}"),
                {"sample_id", "original_edge_ids"},
                f"dataset JSONL line {line_number}",
            )
            sample_id = require_nonempty_string(row["sample_id"], "sample_id")
            if any(unicodedata.category(character) == "Cc" for character in sample_id):
                raise RuntimeError(f"sample_id {sample_id!r} contains a control character")
            if sample_id in sample_ids:
                raise RuntimeError(f"duplicate sample_id {sample_id!r}")
            edges = row["original_edge_ids"]
            if not isinstance(edges, list) or len(edges) < 2:
                raise RuntimeError(
                    f"sample {sample_id!r} must have at least 2 original_edge_ids"
                )
            if any(
                isinstance(edge, bool)
                or not isinstance(edge, int)
                or edge < 0
                or edge > U32_MAX
                for edge in edges
            ):
                raise RuntimeError(f"sample {sample_id!r} has a non-u32 edge ID")
            sample_ids.add(sample_id)
            trips.append(Trip(sample_id, edges.copy()))
    if not trips:
        raise RuntimeError(f"empty dataset records file {path}")
    return trips


def _field_indices(reader: Any) -> dict[str, int]:
    return {field[0]: index for index, field in enumerate(reader.fields[1:])}


def load_road_graph(map_dir: Path) -> StaticGraph:
    node_reader = shapefile.Reader(str(map_dir / "nodes.shp"))
    node_fields = _field_indices(node_reader)
    required_node_fields = {"osmid", "x", "y"}
    if not required_node_fields <= set(node_fields):
        raise RuntimeError(f"nodes.shp lacks {sorted(required_node_fields - set(node_fields))}")
    node_rows = node_reader.records()
    osmids = [int(row[node_fields["osmid"]]) for row in node_rows]
    node_index = {osmid: index for index, osmid in enumerate(osmids)}
    if len(node_index) != len(osmids):
        raise RuntimeError("nodes.shp contains duplicate osmid values")
    node_x = [float(row[node_fields["x"]]) for row in node_rows]
    node_y = [float(row[node_fields["y"]]) for row in node_rows]

    edge_reader = shapefile.Reader(str(map_dir / "edges.shp"))
    edge_fields = _field_indices(edge_reader)
    if not {"u", "v"} <= set(edge_fields):
        raise RuntimeError("edges.shp must contain u and v fields")
    edge_rows = edge_reader.records()
    if "fid" in edge_fields:
        fids = [int(row[edge_fields["fid"]]) for row in edge_rows]
        if fids != list(range(len(fids))):
            raise RuntimeError("edges.shp fid must equal unmodified record order")
    try:
        tail = [node_index[int(row[edge_fields["u"]])] for row in edge_rows]
        head = [node_index[int(row[edge_fields["v"]])] for row in edge_rows]
    except KeyError as error:
        raise RuntimeError(f"edges.shp refers to missing node {error.args[0]}") from error
    outgoing: list[list[int]] = [[] for _ in node_rows]
    for edge, node in enumerate(tail):
        outgoing[node].append(edge)
    for edges in outgoing:
        edges.sort()
    neighbors = [tuple(outgoing[node]) for node in head]
    identity = graph_identity(tail, head, node_x, node_y)
    return StaticGraph(tail, head, node_x, node_y, neighbors, identity)


def graph_identity(
    tail: Sequence[int], head: Sequence[int], x: Sequence[float], y: Sequence[float]
) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<QQ", len(tail), len(x)))
    for source, destination in zip(tail, head):
        digest.update(struct.pack("<QQ", source, destination))
    for longitude, latitude in zip(x, y):
        digest.update(struct.pack("<dd", longitude, latitude))
    return digest.hexdigest()


def validate_trips(trips: Iterable[Trip], graph: StaticGraph) -> None:
    for trip in trips:
        if len(trip.edges) < 2:
            raise RuntimeError(f"{trip.sample_id} has fewer than 2 roads")
        if any(edge < 0 or edge >= graph.edge_count for edge in trip.edges):
            raise RuntimeError(f"{trip.sample_id} has an unrepresentable road")
        if any(
            graph.head[left] != graph.tail[right]
            for left, right in zip(trip.edges, trip.edges[1:])
        ):
            raise RuntimeError(f"{trip.sample_id} is discontinuous")


def _count_trip_chunk(
    trips: Sequence[Trip], edge_count: int
) -> tuple[dict[tuple[int, int], int], list[int], dict[tuple[int, int], int], int]:
    da: dict[tuple[int, int], int] = {}
    popularity = [0] * edge_count
    transitions: dict[tuple[int, int], int] = {}
    pair_events = 0
    for trip in trips:
        edges = trip.edges
        for edge in edges:
            popularity[edge] += 1
        for left, right in zip(edges, edges[1:]):
            transition = (left, right)
            transitions[transition] = transitions.get(transition, 0) + 1
        for left_index, source in enumerate(edges[:-1]):
            for destination in edges[left_index + 1 :]:
                pair = (source, destination)
                da[pair] = da.get(pair, 0) + 1
                pair_events += 1
    return da, popularity, transitions, pair_events


def build_training_statistics(
    trips: Sequence[Trip], edge_count: int, workers: int
) -> tuple[SparseDA, list[int], dict[tuple[int, int], int], int]:
    if not trips:
        raise RuntimeError("training statistics require trajectories")
    worker_count = max(1, min(workers, len(trips)))
    chunk_size = math.ceil(len(trips) / worker_count)
    chunks = [trips[start : start + chunk_size] for start in range(0, len(trips), chunk_size)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(lambda chunk: _count_trip_chunk(chunk, edge_count), chunks))
    combined_da: dict[tuple[int, int], int] = {}
    combined_popularity = [0] * edge_count
    combined_transitions: dict[tuple[int, int], int] = {}
    pair_events = 0
    for local_da, local_popularity, local_transitions, local_events in results:
        for key, value in local_da.items():
            combined_da[key] = combined_da.get(key, 0) + value
        for edge, value in enumerate(local_popularity):
            combined_popularity[edge] += value
        for key, value in local_transitions.items():
            combined_transitions[key] = combined_transitions.get(key, 0) + value
        pair_events += local_events
    return (
        SparseDA.from_counts(edge_count, combined_da),
        combined_popularity,
        combined_transitions,
        pair_events,
    )


def _flush_da_run(
    encoded_pairs: Any,
    used: int,
    scratch: Path,
    ordinal: int,
) -> tuple[tuple[Path, Path], tuple[Path, Path], int]:
    keys, counts = np.unique(encoded_pairs[:used], return_counts=True)
    counts = counts.astype("<u4", copy=False)
    row_keys = scratch / f"row-{ordinal:06d}.keys.npy"
    row_counts = scratch / f"row-{ordinal:06d}.counts.npy"
    col_keys = scratch / f"col-{ordinal:06d}.keys.npy"
    col_counts = scratch / f"col-{ordinal:06d}.counts.npy"
    atomic_save_npy(row_keys, keys.astype("<u8", copy=False))
    atomic_save_npy(row_counts, counts)
    swapped = ((keys & np.uint64(U32_MAX)) << np.uint64(32)) | (keys >> np.uint64(32))
    order = np.argsort(swapped, kind="stable")
    atomic_save_npy(col_keys, swapped[order].astype("<u8", copy=False))
    atomic_save_npy(col_counts, counts[order])
    return (row_keys, row_counts), (col_keys, col_counts), int(len(keys))


def _merge_da_runs(
    run_paths: Sequence[tuple[Path, Path]],
    output_dir: Path,
    prefix: str,
    edge_count: int,
) -> tuple[int, int]:
    """K-way merge sorted sparse runs into one compact CSR-like view."""

    keys = [
        np.load(key_path, allow_pickle=False, mmap_mode="r")
        for key_path, _ in run_paths
    ]
    counts = [
        np.load(count_path, allow_pickle=False, mmap_mode="r")
        for _, count_path in run_paths
    ]
    positions = [0] * len(run_paths)
    heap: list[tuple[int, int]] = [
        (int(run_keys[0]), run_index)
        for run_index, run_keys in enumerate(keys)
        if len(run_keys)
    ]
    heapq.heapify(heap)
    offsets = np.zeros(edge_count + 1, dtype="<u8")
    indices_path = output_dir / f"{prefix}_indices.u32"
    values_path = output_dir / f"{prefix}_values.u32"
    index_buffer = np.empty(262_144, dtype="<u4")
    value_buffer = np.empty(262_144, dtype="<u4")
    buffered = 0
    nonzero = 0
    total_events = 0
    completed_source = 0

    def flush_buffers(index_output: Any, value_output: Any) -> None:
        nonlocal buffered
        if buffered:
            index_buffer[:buffered].tofile(index_output)
            value_buffer[:buffered].tofile(value_output)
            buffered = 0

    with indices_path.open("wb") as index_output, values_path.open("wb") as value_output:
        while heap:
            encoded, run_index = heapq.heappop(heap)
            count = int(counts[run_index][positions[run_index]])
            positions[run_index] += 1
            if positions[run_index] < len(keys[run_index]):
                heapq.heappush(
                    heap,
                    (int(keys[run_index][positions[run_index]]), run_index),
                )
            while heap and heap[0][0] == encoded:
                _, duplicate_run = heapq.heappop(heap)
                count += int(counts[duplicate_run][positions[duplicate_run]])
                positions[duplicate_run] += 1
                if positions[duplicate_run] < len(keys[duplicate_run]):
                    heapq.heappush(
                        heap,
                        (
                            int(keys[duplicate_run][positions[duplicate_run]]),
                            duplicate_run,
                        ),
                    )
            source = encoded >> 32
            destination = encoded & U32_MAX
            if source >= edge_count or destination >= edge_count:
                raise RuntimeError("external DA run contains an out-of-range road ID")
            if count > U32_MAX:
                raise RuntimeError("DA count exceeds compact u32 representation")
            while completed_source < source:
                offsets[completed_source + 1] = nonzero
                completed_source += 1
            index_buffer[buffered] = destination
            value_buffer[buffered] = count
            buffered += 1
            nonzero += 1
            total_events += count
            if buffered == len(index_buffer):
                flush_buffers(index_output, value_output)
        flush_buffers(index_output, value_output)
    while completed_source < edge_count:
        offsets[completed_source + 1] = nonzero
        completed_source += 1
    atomic_save_npy(output_dir / f"{prefix}_offsets.npy", offsets)
    for array in [*keys, *counts]:
        if getattr(array, "_mmap", None) is not None:
            array._mmap.close()
    return nonzero, total_events


def load_compact_da(path: Path) -> SparseDA:
    metadata = load_json(path / "metadata.json")
    if metadata.get("schema") != "ewr.drpk-static-compact-da/v1":
        raise RuntimeError("unsupported compact DA schema")
    size = int(metadata["size"])
    nonzero = int(metadata["nonzero"])
    row_offsets = np.load(path / "row_offsets.npy", allow_pickle=False, mmap_mode="r")
    col_offsets = np.load(path / "col_offsets.npy", allow_pickle=False, mmap_mode="r")
    row_indices = np.memmap(
        path / "row_indices.u32", dtype="<u4", mode="r", shape=(nonzero,)
    )
    row_values = np.memmap(
        path / "row_values.u32", dtype="<u4", mode="r", shape=(nonzero,)
    )
    col_indices = np.memmap(
        path / "col_indices.u32", dtype="<u4", mode="r", shape=(nonzero,)
    )
    col_values = np.memmap(
        path / "col_values.u32", dtype="<u4", mode="r", shape=(nonzero,)
    )
    if tuple(row_offsets.shape) != (size + 1,) or tuple(col_offsets.shape) != (size + 1,):
        raise RuntimeError("compact DA offsets have the wrong shape")
    if int(row_offsets[-1]) != nonzero or int(col_offsets[-1]) != nonzero:
        raise RuntimeError("compact DA offsets disagree with nonzero count")
    return SparseDA(
        size,
        row_offsets,
        row_indices,
        row_values,
        col_offsets,
        col_indices,
        col_values,
    )


def build_training_statistics_external(
    trips: Sequence[Trip],
    edge_count: int,
    output_dir: Path,
    pair_chunk_size: int,
    collect_popularity: bool = True,
    collect_transitions: bool = True,
) -> tuple[SparseDA, list[int], dict[tuple[int, int], int], int, dict[str, int]]:
    """Build DA with bounded RAM using sorted disk runs and a K-way merge."""

    if not trips or pair_chunk_size <= 0:
        raise RuntimeError("external DA construction requires trips and a positive chunk")
    da_dir = output_dir / "da"
    da_dir.mkdir(parents=True, exist_ok=True)
    popularity = (
        np.zeros(edge_count, dtype=np.int64) if collect_popularity else None
    )
    transition_counts: dict[int, int] = {}
    encoded_pairs = np.empty(pair_chunk_size, dtype="<u8")
    used = 0
    pair_events = 0
    run_nonzero_total = 0
    row_runs: list[tuple[Path, Path]] = []
    col_runs: list[tuple[Path, Path]] = []
    with tempfile.TemporaryDirectory(prefix=".da-runs-", dir=output_dir) as temporary:
        scratch = Path(temporary)

        def flush() -> None:
            nonlocal run_nonzero_total, used
            if not used:
                return
            row_path, col_path, run_nonzero = _flush_da_run(
                encoded_pairs, used, scratch, len(row_runs)
            )
            row_runs.append(row_path)
            col_runs.append(col_path)
            run_nonzero_total += run_nonzero
            used = 0

        for trip in trips:
            edges = np.asarray(trip.edges, dtype="<u8")
            if popularity is not None:
                np.add.at(popularity, edges, 1)
            if collect_transitions:
                for left, right in zip(trip.edges, trip.edges[1:]):
                    key = (left << 32) | right
                    transition_counts[key] = transition_counts.get(key, 0) + 1
            for left_index, source in enumerate(trip.edges[:-1]):
                destinations = edges[left_index + 1 :]
                needed = len(destinations)
                if needed > pair_chunk_size:
                    raise RuntimeError(
                        "DA pair chunk must be at least maximum route length minus one"
                    )
                if used + needed > pair_chunk_size:
                    flush()
                encoded_pairs[used : used + needed] = (
                    np.uint64(source) << np.uint64(32)
                ) | destinations
                used += needed
                pair_events += needed
        flush()
        if not row_runs:
            raise RuntimeError("training routes produced no DA pairs")
        scratch_bytes = sum(
            path.stat().st_size
            for run in [*row_runs, *col_runs]
            for path in run
        )
        row_nonzero, row_events = _merge_da_runs(
            row_runs, da_dir, "row", edge_count
        )
        col_nonzero, col_events = _merge_da_runs(
            col_runs, da_dir, "col", edge_count
        )
    if row_nonzero != col_nonzero or row_events != pair_events or col_events != pair_events:
        raise RuntimeError("external DA row/column merge totals differ")
    metadata = {
        "schema": "ewr.drpk-static-compact-da/v1",
        "size": edge_count,
        "nonzero": row_nonzero,
        "pair_events": pair_events,
        "index_dtype": "uint32-le",
        "value_dtype": "uint32-le",
        "offset_dtype": "uint64-le-npy",
        "storage": "memory_mapped_csr_and_csc",
    }
    write_json(da_dir / "metadata.json", metadata)
    compact_bytes = sum(path.stat().st_size for path in da_dir.iterdir())
    transitions = {
        (encoded >> 32, encoded & U32_MAX): count
        for encoded, count in transition_counts.items()
    }
    return (
        load_compact_da(da_dir),
        [] if popularity is None else popularity.tolist(),
        transitions,
        pair_events,
        {
            "runs": len(row_runs),
            "run_nonzero_total": run_nonzero_total,
            "scratch_peak_bytes": scratch_bytes,
            "compact_bytes": compact_bytes,
        },
    )


def build_global_popularity(
    trips: Sequence[Trip], edge_count: int, chunk_edges: int = 1_000_000
) -> list[int]:
    """Count training road occurrences with bounded temporary storage."""

    if not trips or edge_count <= 0 or chunk_edges <= 0:
        raise RuntimeError("popularity counting requires trips and positive sizes")
    counts = np.zeros(edge_count, dtype=np.int64)
    buffer = np.empty(chunk_edges, dtype=np.int32)
    used = 0

    def flush() -> None:
        nonlocal used
        if used:
            counts[:] += np.bincount(buffer[:used], minlength=edge_count)
            used = 0

    for trip in trips:
        if len(trip.edges) > chunk_edges:
            raise RuntimeError("popularity chunk is shorter than one route")
        if used + len(trip.edges) > chunk_edges:
            flush()
        buffer[used : used + len(trip.edges)] = trip.edges
        used += len(trip.edges)
    flush()
    return counts.tolist()


def normalize_global_popularity(counts: Sequence[int]) -> list[float]:
    if not counts:
        raise RuntimeError("popularity vector is empty")
    low, high = min(counts), max(counts)
    if high == low:
        return [0.0] * len(counts)
    return [(value - low) / (high - low) for value in counts]


def replicated_popularity(counts: Sequence[int]) -> list[list[int]]:
    """Materialize the audited 48 identical time slots for diagnostics/tests."""

    return [list(counts) for _ in range(TIME_SLOT_COUNT)]


def build_candidate_rows(
    trips: Sequence[Trip],
    da: SparseDA,
    pool_size: int,
    positive_fraction: float,
    storage_dir: Path | None = None,
) -> dict[str, Any]:
    if pool_size <= 0 or not (0.0 < positive_fraction <= 1.0):
        raise RuntimeError("invalid candidate preprocessing configuration")
    # A Python list-of-lists for the formal 605k x 100 candidate table retains
    # tens of millions of Python objects.  Allocate the exact compact dtypes up
    # front instead; peak storage is bounded by this split's row count.
    row_capacity = len(trips)
    if storage_dir is not None:
        storage_dir.mkdir(parents=True, exist_ok=True)

    def allocate(name: str, shape: tuple[int, ...], dtype: Any) -> Any:
        if storage_dir is None:
            return np.empty(shape, dtype=dtype)
        return np.lib.format.open_memmap(
            storage_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
        )

    sources = allocate("source", (row_capacity,), np.int32)
    destinations = allocate("destination", (row_capacity,), np.int32)
    candidates_rows = allocate(
        "candidates", (row_capacity, pool_size), np.int32
    )
    labels_rows = allocate("labels", (row_capacity, pool_size), np.uint8)
    weights_rows = allocate("weights", (row_capacity, pool_size), np.float32)
    output_row = 0
    omitted_empty_pool = 0
    omitted_no_positive = 0
    for trip in trips:
        pool = da.candidate_pool(trip.edges[0], trip.edges[-1], pool_size)
        if not pool:
            omitted_empty_pool += 1
            continue
        # DRPK labels only interior road segments as possible key segments.
        truth = set(trip.edges[1:-1])
        positive_limit = max(1, math.ceil(len(trip.edges) * positive_fraction))
        positive_edges = [edge for edge, _ in pool if edge in truth][:positive_limit]
        positive = set(positive_edges)
        if not positive:
            omitted_no_positive += 1
            continue
        strength_by_edge = dict(pool)
        positive_total = sum(strength_by_edge[edge] for edge in positive)
        candidates = [edge for edge, _ in pool]
        candidate_count = len(candidates)
        sources[output_row] = trip.edges[0]
        destinations[output_row] = trip.edges[-1]
        candidates_rows[output_row].fill(-1)
        labels_rows[output_row].fill(0)
        weights_rows[output_row].fill(0.0)
        candidates_rows[output_row, :candidate_count] = candidates
        weights_rows[output_row, :candidate_count] = 1.0
        for index, edge in enumerate(candidates):
            if edge in positive:
                labels_rows[output_row, index] = 1
                # Paper Eq. (10): exp of the normalized key importance.  A
                # non-key has normalized importance zero and therefore weight 1.
                normalized_importance = strength_by_edge[edge] / positive_total
                weights_rows[output_row, index] = math.exp(normalized_importance)
        output_row += 1

    if storage_dir is None:
        # ndarray.resize releases unused tail capacity without a second
        # full-table copy.  Every array owns its allocation at this point.
        sources.resize((output_row,), refcheck=False)
        destinations.resize((output_row,), refcheck=False)
        candidates_rows.resize((output_row, pool_size), refcheck=False)
        labels_rows.resize((output_row, pool_size), refcheck=False)
        weights_rows.resize((output_row, pool_size), refcheck=False)
    else:
        for array in (
            sources,
            destinations,
            candidates_rows,
            labels_rows,
            weights_rows,
        ):
            array.flush()
    rows = {
        "source": sources[:output_row],
        "destination": destinations[:output_row],
        "candidates": candidates_rows[:output_row],
        "labels": labels_rows[:output_row],
        "weights": weights_rows[:output_row],
        "omitted_empty_pool": omitted_empty_pool,
        "omitted_no_positive": omitted_no_positive,
        "_capacity": row_capacity if storage_dir is not None else output_row,
    }
    if storage_dir is not None:
        rows["_storage_dir"] = str(storage_dir.resolve())
    return rows


def _weighted_choice(rng: random.Random, values: Sequence[int], weights: Sequence[float]) -> int:
    total = sum(weights)
    if total <= 0:
        return values[0]
    threshold = rng.random() * total
    cumulative = 0.0
    for value, weight in zip(values, weights):
        cumulative += weight
        if cumulative >= threshold:
            return value
    return values[-1]


def _node2vec_walk(
    task: tuple[int, int],
    graph: StaticGraph,
    length: int,
    p: float,
    q: float,
    seed: int,
) -> tuple[int, list[int]]:
    ordinal, start = task
    rng = random.Random(seed + ordinal * 1_000_003)
    walk = [start]
    while len(walk) < length:
        current = walk[-1]
        neighbors = graph.neighbors[current]
        if not neighbors:
            break
        previous = walk[-2] if len(walk) >= 2 else None
        # The conjugate graph has one unweighted arc per legal raw-road
        # transition.  The official RandomWalker takes its uniform
        # ``deepwalk_walk`` fast path at the formal p=q=1 setting.
        weights: list[float] = []
        for candidate in neighbors:
            weight = 1.0
            if previous is not None and not (p == 1.0 and q == 1.0):
                if candidate == previous:
                    weight /= p
                elif previous not in graph.neighbors[candidate]:
                    weight /= q
            weights.append(weight)
        walk.append(_weighted_choice(rng, neighbors, weights))
    return ordinal, walk


def generate_node2vec_walks(
    graph: StaticGraph,
    walk_length: int,
    walks_per_edge: int,
    p: float,
    q: float,
    workers: int,
    seed: int,
) -> list[list[int]]:
    if walk_length <= 1 or walks_per_edge <= 0 or p <= 0 or q <= 0:
        raise RuntimeError("invalid Node2Vec walk configuration")
    task_rng = random.Random(seed)
    tasks: list[tuple[int, int]] = []
    for round_index in range(walks_per_edge):
        starts = list(range(graph.edge_count))
        task_rng.shuffle(starts)
        tasks.extend(
            (round_index * graph.edge_count + position, edge)
            for position, edge in enumerate(starts)
        )
    worker = lambda task: _node2vec_walk(task, graph, walk_length, p, q, seed)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(worker, tasks))
    results.sort(key=lambda item: item[0])
    return [walk for _, walk in results]


def _skipgram_batches(
    walks: Sequence[Sequence[int]], window: int, batch_size: int
) -> Iterator[tuple[list[int], list[int]]]:
    centers: list[int] = []
    contexts: list[int] = []
    for walk in walks:
        for index, center in enumerate(walk):
            left = max(0, index - window)
            right = min(len(walk), index + window + 1)
            for context_index in range(left, right):
                if context_index == index:
                    continue
                centers.append(center)
                contexts.append(walk[context_index])
                if len(centers) == batch_size:
                    yield centers, contexts
                    centers, contexts = [], []
    if centers:
        yield centers, contexts


def train_node2vec(
    graph: StaticGraph,
    *,
    dimensions: int,
    walk_length: int,
    walks_per_edge: int,
    p: float,
    q: float,
    window: int,
    epochs: int,
    negative_samples: int,
    batch_size: int,
    learning_rate: float,
    workers: int,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    estimated_tokens = graph.edge_count * walks_per_edge * walk_length
    if estimated_tokens > 5_000_000:
        raise RuntimeError(
            "the torch Node2Vec engine is restricted to small smoke graphs; "
            "use the streaming gensim engine for a formal run"
        )
    if min(dimensions, window, epochs, negative_samples, batch_size) <= 0:
        raise RuntimeError("Node2Vec dimensions and optimizer values must be positive")
    walks_started = time.perf_counter()
    walks = generate_node2vec_walks(
        graph,
        walk_length,
        walks_per_edge,
        p,
        q,
        workers,
        seed,
    )
    walk_seconds = time.perf_counter() - walks_started
    frequencies = torch.ones(graph.edge_count, dtype=torch.float64)
    for walk in walks:
        for edge in walk:
            frequencies[edge] += 1
    negative_distribution = frequencies.pow(0.75)
    negative_distribution /= negative_distribution.sum()
    input_embedding = torch.nn.Embedding(graph.edge_count, dimensions, sparse=True)
    output_embedding = torch.nn.Embedding(graph.edge_count, dimensions, sparse=True)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        initial = torch.rand(
            input_embedding.weight.shape,
            generator=generator,
            dtype=input_embedding.weight.dtype,
        )
        initial = (initial - 0.5) / dimensions
        input_embedding.weight.copy_(initial)
        output_embedding.weight.zero_()
    optimizer = torch.optim.SparseAdam(
        list(input_embedding.parameters()) + list(output_embedding.parameters()),
        lr=learning_rate,
    )
    training_started = time.perf_counter()
    epoch_losses: list[float] = []
    pair_count = 0
    for _ in range(epochs):
        loss_total = 0.0
        epoch_pairs = 0
        for centers, contexts in _skipgram_batches(walks, window, batch_size):
            center_tensor = torch.tensor(centers, dtype=torch.long)
            context_tensor = torch.tensor(contexts, dtype=torch.long)
            negative = torch.multinomial(
                negative_distribution,
                len(centers) * negative_samples,
                replacement=True,
                generator=generator,
            ).reshape(len(centers), negative_samples)
            center_vectors = input_embedding(center_tensor)
            positive_vectors = output_embedding(context_tensor)
            negative_vectors = output_embedding(negative)
            positive_score = (center_vectors * positive_vectors).sum(dim=1)
            negative_score = torch.bmm(
                negative_vectors, center_vectors.unsqueeze(2)
            ).squeeze(2)
            loss = -(
                F.logsigmoid(positive_score)
                + F.logsigmoid(-negative_score).sum(dim=1)
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(centers)
            epoch_pairs += len(centers)
        if epoch_pairs == 0:
            raise RuntimeError("Node2Vec walks produced no skip-gram pairs")
        pair_count = epoch_pairs
        epoch_losses.append(loss_total / epoch_pairs)
    return (
        input_embedding.weight.detach().cpu().numpy().astype(np.float32),
        {
            "walk_seconds": walk_seconds,
            "training_seconds": time.perf_counter() - training_started,
            "walks": len(walks),
            "walk_tokens": sum(len(walk) for walk in walks),
            "pairs_per_epoch": pair_count,
            "epoch_losses": epoch_losses,
        },
    )


class StreamingWalkCorpus:
    """Re-iterable deterministic walk corpus; never stores all walks in RAM."""

    def __init__(
        self,
        graph: StaticGraph,
        walk_length: int,
        walks_per_edge: int,
        p: float,
        q: float,
        seed: int,
    ) -> None:
        self.graph = graph
        self.walk_length = walk_length
        self.walks_per_edge = walks_per_edge
        self.p = p
        self.q = q
        self.seed = seed

    def __iter__(self) -> Iterator[list[int]]:
        task_rng = random.Random(self.seed)
        ordinal = 0
        for _ in range(self.walks_per_edge):
            starts = list(range(self.graph.edge_count))
            task_rng.shuffle(starts)
            for edge in starts:
                _, walk = _node2vec_walk(
                    (ordinal, edge),
                    self.graph,
                    self.walk_length,
                    self.p,
                    self.q,
                    self.seed,
                )
                ordinal += 1
                yield walk


def train_node2vec_gensim(
    graph: StaticGraph,
    *,
    dimensions: int,
    walk_length: int,
    walks_per_edge: int,
    p: float,
    q: float,
    window: int,
    epochs: int,
    negative_samples: int,
    learning_rate: float,
    workers: int,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    try:
        import gensim
        from gensim.models import Word2Vec
    except ImportError as error:
        raise RuntimeError(
            "formal Node2Vec preprocessing requires gensim==4.4.0"
        ) from error
    corpus = StreamingWalkCorpus(
        graph,
        walk_length,
        walks_per_edge,
        p,
        q,
        seed,
    )
    started = time.perf_counter()
    model = Word2Vec(
        sentences=corpus,
        vector_size=dimensions,
        window=window,
        min_count=0,
        sg=1,
        hs=0,
        negative=negative_samples,
        workers=workers,
        epochs=epochs,
        alpha=learning_rate,
        seed=seed,
    )
    embeddings = np.empty((graph.edge_count, dimensions), dtype=np.float32)
    for edge in range(graph.edge_count):
        embeddings[edge] = model.wv[edge]
    return embeddings, {
        "engine": "gensim_streaming_word2vec",
        "gensim_version": str(gensim.__version__),
        "training_seconds": time.perf_counter() - started,
        "walks_per_corpus_pass": graph.edge_count * walks_per_edge,
        "maximum_tokens_per_corpus_pass": (
            graph.edge_count * walks_per_edge * walk_length
        ),
        "walks_materialized": False,
        "workers": workers,
    }


def train_node2vec_dispatch(
    engine: str,
    graph: StaticGraph,
    **settings: Any,
) -> tuple[Any, dict[str, Any]]:
    if engine == "gensim":
        settings = settings.copy()
        settings.pop("batch_size")
        return train_node2vec_gensim(graph, **settings)
    result, diagnostics = train_node2vec(graph, **settings)
    diagnostics["engine"] = "torch_small_graph_only"
    diagnostics["walks_materialized"] = True
    return result, diagnostics


def preprocess_command(args: argparse.Namespace) -> None:
    numeric_positive = [
        args.workers,
        args.candidate_pool_size,
        args.da_chunk_pair_events,
        args.node2vec_dim,
        args.walk_length,
        args.walks_per_edge,
        args.node2vec_window,
        args.node2vec_epochs,
        args.node2vec_negative_samples,
        args.node2vec_batch_size,
    ]
    if any(value <= 0 for value in numeric_positive):
        raise RuntimeError("preprocess integer settings must be positive")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    total_started = time.perf_counter()
    source = adapter_source_identity(args.source_revision)
    graph_started = time.perf_counter()
    graph = load_road_graph(args.map_dir)
    save_graph(output / "graph.npz", graph)
    persisted_graph = load_saved_graph(output / "graph.npz")
    if persisted_graph.identity != graph.identity:
        raise RuntimeError("persisted graph differs from the loaded road network")
    graph_seconds = time.perf_counter() - graph_started
    train_dataset = load_dataset_manifest(args.train_manifest)
    validation_dataset = load_dataset_manifest(args.validation_manifest)
    if train_dataset.manifest.network_id != validation_dataset.manifest.network_id:
        raise RuntimeError("training and validation manifests use different networks")
    validation_ids = {trip.sample_id for trip in validation_dataset.trips}
    if any(trip.sample_id in validation_ids for trip in train_dataset.trips):
        raise RuntimeError("training and validation sample IDs overlap")
    validate_trips(train_dataset.trips, graph)
    validate_trips(validation_dataset.trips, graph)

    da_started = time.perf_counter()
    da, _, _, pair_events, da_storage_diagnostics = (
        build_training_statistics_external(
            train_dataset.trips,
            graph.edge_count,
            output,
            args.da_chunk_pair_events,
            collect_popularity=False,
            collect_transitions=False,
        )
    )
    da_seconds = time.perf_counter() - da_started
    routing_manifest_path, routing_manifest_sha256 = (
        write_routing_artifact_manifest(
            output, train_dataset.manifest.network_id, graph.identity
        )
    )
    routing_configuration = {
        "schema": ROUTING_PREPROCESS_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "source": source,
        "provenance": upstream_provenance(),
        "adaptation": "time_collapsed_train_minmax_popularity_replicated_48_slots",
        "network_id": train_dataset.manifest.network_id,
        "graph_identity": graph.identity,
        "edge_count": graph.edge_count,
        "routing_artifacts": {
            "schema": ROUTING_ARTIFACT_MANIFEST_SCHEMA,
            "path": routing_manifest_path.name,
            "sha256": routing_manifest_sha256,
        },
        "train_dataset_id": train_dataset.manifest.dataset_id,
        "train_manifest_sha256": train_dataset.manifest_sha256,
        "train_records_sha256": train_dataset.records_sha256,
        "validation_dataset_id": validation_dataset.manifest.dataset_id,
        "validation_manifest_sha256": validation_dataset.manifest_sha256,
        "validation_records_sha256": validation_dataset.records_sha256,
        "train_records": len(train_dataset.trips),
        "validation_records": len(validation_dataset.trips),
        "seed": args.seed,
        "workers": args.workers,
        "da": {
            "storage": "external_merge_compact_csr_csc",
            "chunk_pair_events": args.da_chunk_pair_events,
            "dense_matrix_constructed": False,
        },
        "query_protocol": "fixed_true_first_edge_to_true_last_edge",
        "truth_repair": False,
        "max_route_length": 300,
        "test_manifest_read": False,
    }
    write_json(output / "routing-configuration.json", routing_configuration)
    routing_configuration_sha256 = sha256_file(
        output / "routing-configuration.json"
    )
    # DRP-TP is genuinely loadable at this boundary: graph, DA, their manifest,
    # and the routing configuration are all durable.  Snapshot it before KSD-only candidate
    # popularity, candidate tables, and Node2Vec so a shared preprocess run
    # cannot overcharge the non-learned sanity baseline in the efficiency table.
    drp_tp_ready_seconds = time.perf_counter() - total_started
    drp_tp_ready_peak_rss_kib = peak_rss_kib()
    # Defer PyTorch (and any CUDA probing/initialization) until after the exact
    # DRP-TP-ready snapshot, and never import it for the formal Gensim engine.
    # Popularity, candidate rows, and streaming Gensim Node2Vec need only the
    # already-loaded NumPy/PyShp stack.
    if args.node2vec_engine == "torch":
        load_model_dependencies()
        seed_everything(args.seed, args.workers)
    popularity_started = time.perf_counter()
    popularity_counts = build_global_popularity(
        train_dataset.trips, graph.edge_count
    )
    normalized_popularity = normalize_global_popularity(popularity_counts)
    popularity_seconds = time.perf_counter() - popularity_started
    train_candidates_started = time.perf_counter()
    train_rows = build_candidate_rows(
        train_dataset.trips,
        da,
        args.candidate_pool_size,
        args.positive_route_fraction,
        output / "train_candidates",
    )
    save_candidate_rows(output / "train_candidates", train_rows)
    train_candidate_counts = {
        "samples": len(train_rows["source"]),
        "omitted_empty_pool": train_rows["omitted_empty_pool"],
        "omitted_no_positive": train_rows["omitted_no_positive"],
    }
    del train_rows
    validation_rows = build_candidate_rows(
        validation_dataset.trips,
        da,
        args.candidate_pool_size,
        args.positive_route_fraction,
        output / "validation_candidates",
    )
    save_candidate_rows(output / "validation_candidates", validation_rows)
    validation_candidate_counts = {
        "samples": len(validation_rows["source"]),
        "omitted_empty_pool": validation_rows["omitted_empty_pool"],
        "omitted_no_positive": validation_rows["omitted_no_positive"],
    }
    del validation_rows
    candidate_seconds = time.perf_counter() - train_candidates_started
    embeddings, node2vec_diagnostics = train_node2vec_dispatch(
        args.node2vec_engine,
        graph,
        dimensions=args.node2vec_dim,
        walk_length=args.walk_length,
        walks_per_edge=args.walks_per_edge,
        p=args.node2vec_p,
        q=args.node2vec_q,
        window=args.node2vec_window,
        epochs=args.node2vec_epochs,
        negative_samples=args.node2vec_negative_samples,
        batch_size=args.node2vec_batch_size,
        learning_rate=args.node2vec_learning_rate,
        workers=args.workers,
        seed=args.seed,
    )
    atomic_save_npy(output / "node2vec.npy", embeddings)
    atomic_save_npz(
        output / "static_features.npz",
        popularity_counts=np.asarray(popularity_counts, dtype=np.int64),
        normalized_popularity=np.asarray(normalized_popularity, dtype=np.float32),
        popularity_48=np.repeat(
            np.asarray(normalized_popularity, dtype=np.float32)[:, None],
            TIME_SLOT_COUNT,
            axis=1,
        ),
        schema=np.asarray(STATIC_FEATURES_SCHEMA),
    )
    core_manifest_path, core_manifest_sha256 = write_core_artifact_manifest(
        output, train_dataset.manifest.network_id, graph.identity
    )
    configuration = {
        "schema": PREPROCESS_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "source": source,
        "provenance": upstream_provenance(),
        "adaptation": "time_collapsed_train_minmax_popularity_replicated_48_slots",
        "network_id": train_dataset.manifest.network_id,
        "graph_identity": graph.identity,
        "core_artifacts": {
            "schema": CORE_ARTIFACT_MANIFEST_SCHEMA,
            "path": core_manifest_path.name,
            "sha256": core_manifest_sha256,
        },
        "routing_configuration": {
            "schema": ROUTING_PREPROCESS_SCHEMA,
            "path": "routing-configuration.json",
            "sha256": routing_configuration_sha256,
        },
        "edge_count": graph.edge_count,
        "train_dataset_id": train_dataset.manifest.dataset_id,
        "train_manifest_sha256": train_dataset.manifest_sha256,
        "train_records_sha256": train_dataset.records_sha256,
        "validation_dataset_id": validation_dataset.manifest.dataset_id,
        "validation_manifest_sha256": validation_dataset.manifest_sha256,
        "validation_records_sha256": validation_dataset.records_sha256,
        "train_records": len(train_dataset.trips),
        "validation_records": len(validation_dataset.trips),
        "seed": args.seed,
        "workers": args.workers,
        "candidate_pool_size": args.candidate_pool_size,
        "candidate_rows": {
            "schema": CANDIDATE_ROWS_SCHEMA,
            "storage": "preallocated_compact_npy_memmap",
            "train_metadata_sha256": sha256_file(
                output / "train_candidates" / "metadata.json"
            ),
            "validation_metadata_sha256": sha256_file(
                output / "validation_candidates" / "metadata.json"
            ),
            "source_dtype": "int32",
            "destination_dtype": "int32",
            "candidate_dtype": "int32",
            "label_dtype": "uint8",
            "weight_dtype": "float32",
        },
        "da": {
            "storage": "external_merge_compact_csr_csc",
            "chunk_pair_events": args.da_chunk_pair_events,
            "dense_matrix_constructed": False,
        },
        "positive_route_fraction": args.positive_route_fraction,
        "positive_weighting": "exp(normalized_da_importance)",
        "popularity": {
            "schema": STATIC_FEATURES_SCHEMA,
            "source": "training_only_global_edge_occurrence_count",
            "normalization": "minmax_[0,1]",
            "time_collapse": "same_normalized_vector_in_all_48_slots",
        },
        "node2vec": {
            "dimensions": args.node2vec_dim,
            "walk_length": args.walk_length,
            "walks_per_edge": args.walks_per_edge,
            "p": args.node2vec_p,
            "q": args.node2vec_q,
            "window": args.node2vec_window,
            "epochs": args.node2vec_epochs,
            "negative_samples": args.node2vec_negative_samples,
            "batch_size": args.node2vec_batch_size,
            "learning_rate": args.node2vec_learning_rate,
            "walk_graph": "unweighted_directed_raw_edge_conjugate_topology",
            "transition_weighting": "none",
            "formal_p_q_fast_path": "uniform_deepwalk_at_p_equals_q_equals_1",
            "engine": args.node2vec_engine,
        },
        "query_time": "constant_and_unused",
        "source_offset_ratio": FIXED_SOURCE_OFFSET,
        "destination_offset_ratio": FIXED_DESTINATION_OFFSET,
        "time_slots": TIME_SLOT_COUNT,
        "efficiency_accounting": {
            "drp_tp_boundary": "after_graph_dataset_validation_and_da",
            "drpk_static_boundary": "complete_preprocess",
            "shared_artifact_total_must_not_be_charged_to_drp_tp": True,
        },
        "test_manifest_read": False,
    }
    write_json(output / "configuration.json", configuration)
    diagnostics = {
        "schema": PREPROCESS_DIAGNOSTICS_SCHEMA,
        "configuration": configuration,
        "counts": {
            "da_nonzero": da.nonzero,
            "da_pair_events": pair_events,
            "da_storage": da_storage_diagnostics,
            "adjacent_transition_counts": "not_collected_not_used_by_node2vec",
            "train_ksd_samples": train_candidate_counts["samples"],
            "validation_ksd_samples": validation_candidate_counts["samples"],
            "train_omitted_empty_pool": train_candidate_counts[
                "omitted_empty_pool"
            ],
            "train_omitted_no_positive": train_candidate_counts[
                "omitted_no_positive"
            ],
            "validation_omitted_empty_pool": validation_candidate_counts[
                "omitted_empty_pool"
            ],
            "validation_omitted_no_positive": validation_candidate_counts[
                "omitted_no_positive"
            ],
        },
        "timing": {
            "graph_seconds": graph_seconds,
            "da_seconds": da_seconds,
            "popularity_seconds": popularity_seconds,
            "candidate_label_seconds": candidate_seconds,
            "node2vec": node2vec_diagnostics,
            "drp_tp_ready_seconds": drp_tp_ready_seconds,
            "drp_tp_ready_peak_rss_kib": drp_tp_ready_peak_rss_kib,
            "total_seconds": time.perf_counter() - total_started,
        },
        "peak_rss_kib": peak_rss_kib(),
    }
    write_json(output / "diagnostics.json", diagnostics)
    print(json.dumps(diagnostics["counts"], indent=2))


def upstream_provenance() -> dict[str, str]:
    return {
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "official_license": OFFICIAL_LICENSE,
        "implementation": "clean_room_no_upstream_imports",
    }


def adapter_source_identity(source_revision: str) -> dict[str, str]:
    revision = require_nonempty_string(source_revision, "source_revision")
    path = Path(__file__).resolve()
    return {
        "adapter_path": str(path),
        "adapter_sha256": sha256_file(path),
        "source_revision": revision,
    }


def require_same_adapter_source(expected: Any, context: str) -> None:
    if not isinstance(expected, dict):
        raise RuntimeError(f"{context} does not bind adapter source identity")
    required = {"adapter_path", "adapter_sha256", "source_revision"}
    if set(expected) != required:
        raise RuntimeError(f"{context} adapter source identity is malformed")
    current = adapter_source_identity(expected["source_revision"])
    if expected["adapter_sha256"] != current["adapter_sha256"]:
        raise RuntimeError(
            f"{context} was produced by a different DRPK-static adapter source"
        )


def _core_artifact_specs() -> list[tuple[str, str]]:
    return [
        ("graph.npz", "routing"),
        ("da/metadata.json", "routing"),
        ("da/row_offsets.npy", "routing"),
        ("da/col_offsets.npy", "routing"),
        ("da/row_indices.u32", "routing"),
        ("da/row_values.u32", "routing"),
        ("da/col_indices.u32", "routing"),
        ("da/col_values.u32", "routing"),
        ("node2vec.npy", "model"),
        ("static_features.npz", "model"),
    ]


def _routing_artifact_specs() -> list[tuple[str, str]]:
    return [
        (relative, role)
        for relative, role in _core_artifact_specs()
        if role == "routing"
    ]


def _describe_core_artifact(root: Path, relative: str, role: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        raise RuntimeError(f"missing DRPK-static core artifact {relative}")
    descriptor: dict[str, Any] = {
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            descriptor["format"] = "npz"
            descriptor["arrays"] = {
                name: {
                    "shape": list(archive[name].shape),
                    "dtype": str(archive[name].dtype),
                }
                for name in sorted(archive.files)
            }
    elif path.suffix == ".npy":
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        descriptor.update(
            {
                "format": "npy",
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        )
        del array
    elif path.suffix == ".u32":
        if path.stat().st_size % 4:
            raise RuntimeError(f"unaligned raw uint32 artifact {relative}")
        descriptor.update(
            {
                "format": "raw",
                "shape": [path.stat().st_size // 4],
                "dtype": "uint32-le",
            }
        )
    elif path.suffix == ".json":
        descriptor["format"] = "json"
    else:
        raise RuntimeError(f"unsupported core artifact format {relative}")
    return descriptor


def write_core_artifact_manifest(
    root: Path, network_id: str, graph_identity_value: str
) -> tuple[Path, str]:
    artifacts = {
        relative: _describe_core_artifact(root, relative, role)
        for relative, role in _core_artifact_specs()
    }
    manifest = {
        "schema": CORE_ARTIFACT_MANIFEST_SCHEMA,
        "network_id": network_id,
        "graph_identity": graph_identity_value,
        "artifacts": artifacts,
    }
    path = root / "core-artifacts.json"
    write_json(path, manifest)
    return path, sha256_file(path)


def write_routing_artifact_manifest(
    root: Path, network_id: str, graph_identity_value: str
) -> tuple[Path, str]:
    artifacts = {
        relative: _describe_core_artifact(root, relative, role)
        for relative, role in _routing_artifact_specs()
    }
    manifest = {
        "schema": ROUTING_ARTIFACT_MANIFEST_SCHEMA,
        "network_id": network_id,
        "graph_identity": graph_identity_value,
        "artifacts": artifacts,
    }
    path = root / "routing-artifacts.json"
    write_json(path, manifest)
    return path, sha256_file(path)


def validate_routing_artifact_manifest(
    root: Path, configuration: dict[str, Any]
) -> dict[str, Any]:
    reference = configuration.get("routing_artifacts")
    if not isinstance(reference, dict) or set(reference) != {
        "schema",
        "path",
        "sha256",
    }:
        raise RuntimeError("routing artifact reference is malformed")
    if (
        reference["schema"] != ROUTING_ARTIFACT_MANIFEST_SCHEMA
        or reference["path"] != "routing-artifacts.json"
    ):
        raise RuntimeError("unsupported routing artifact reference")
    path = root / reference["path"]
    if sha256_file(path) != reference["sha256"]:
        raise RuntimeError("routing artifact manifest hash mismatch")
    manifest = load_json(path)
    if set(manifest) != {"schema", "network_id", "graph_identity", "artifacts"}:
        raise RuntimeError("routing artifact manifest has an unsupported shape")
    if manifest["schema"] != ROUTING_ARTIFACT_MANIFEST_SCHEMA:
        raise RuntimeError("unsupported routing artifact manifest schema")
    if (
        manifest["network_id"] != configuration.get("network_id")
        or manifest["graph_identity"] != configuration.get("graph_identity")
    ):
        raise RuntimeError("routing artifact manifest identity mismatch")
    specifications = _routing_artifact_specs()
    if set(manifest["artifacts"]) != {
        relative for relative, _ in specifications
    }:
        raise RuntimeError("routing artifact manifest file set mismatch")
    for relative, role in specifications:
        if manifest["artifacts"][relative] != _describe_core_artifact(
            root, relative, role
        ):
            raise RuntimeError(f"routing artifact {relative} differs from manifest")
    return manifest


def validate_core_artifact_manifest(
    root: Path,
    configuration: dict[str, Any],
    required_roles: set[str],
) -> dict[str, Any]:
    reference = configuration.get("core_artifacts")
    if not isinstance(reference, dict) or set(reference) != {
        "schema",
        "path",
        "sha256",
    }:
        raise RuntimeError("preprocess core-artifact reference is malformed")
    if (
        reference["schema"] != CORE_ARTIFACT_MANIFEST_SCHEMA
        or reference["path"] != "core-artifacts.json"
    ):
        raise RuntimeError("unsupported preprocess core-artifact reference")
    manifest_path = root / reference["path"]
    if sha256_file(manifest_path) != reference["sha256"]:
        raise RuntimeError("preprocess core-artifact manifest hash mismatch")
    manifest = load_json(manifest_path)
    if set(manifest) != {"schema", "network_id", "graph_identity", "artifacts"}:
        raise RuntimeError("core-artifact manifest has an unsupported shape")
    if manifest["schema"] != CORE_ARTIFACT_MANIFEST_SCHEMA:
        raise RuntimeError("unsupported core-artifact manifest schema")
    if (
        manifest["network_id"] != configuration.get("network_id")
        or manifest["graph_identity"] != configuration.get("graph_identity")
    ):
        raise RuntimeError("core-artifact manifest identity mismatch")
    specifications = _core_artifact_specs()
    expected_paths = {relative for relative, _ in specifications}
    if not isinstance(manifest["artifacts"], dict) or set(
        manifest["artifacts"]
    ) != expected_paths:
        raise RuntimeError("core-artifact manifest file set mismatch")
    known_roles = {role for _, role in specifications}
    if not required_roles or not required_roles <= known_roles:
        raise RuntimeError("invalid core-artifact validation role")
    for relative, role in specifications:
        if role in required_roles:
            expected = _describe_core_artifact(root, relative, role)
            if manifest["artifacts"][relative] != expected:
                raise RuntimeError(f"core artifact {relative} differs from manifest")
    return manifest


def save_graph(path: Path, graph: StaticGraph) -> None:
    offsets = [0]
    neighbor_edges: list[int] = []
    for neighbors in graph.neighbors:
        neighbor_edges.extend(neighbors)
        offsets.append(len(neighbor_edges))
    atomic_save_npz(
        path,
        tail=np.asarray(graph.tail, dtype=np.int64),
        head=np.asarray(graph.head, dtype=np.int64),
        node_x=np.asarray(graph.node_x, dtype=np.float64),
        node_y=np.asarray(graph.node_y, dtype=np.float64),
        neighbor_offsets=np.asarray(offsets, dtype=np.int64),
        neighbor_edges=np.asarray(neighbor_edges, dtype=np.int64),
        identity=np.asarray(graph.identity),
    )


def load_saved_graph(path: Path) -> StaticGraph:
    with np.load(path, allow_pickle=False) as data:
        offsets = data["neighbor_offsets"].astype(np.int64).tolist()
        edges = data["neighbor_edges"].astype(np.int64).tolist()
        neighbors = [tuple(edges[offsets[i] : offsets[i + 1]]) for i in range(len(offsets) - 1)]
        graph = StaticGraph(
            tail=data["tail"].astype(np.int64).tolist(),
            head=data["head"].astype(np.int64).tolist(),
            node_x=data["node_x"].astype(np.float64).tolist(),
            node_y=data["node_y"].astype(np.float64).tolist(),
            neighbors=neighbors,
            identity=str(data["identity"].item()),
        )
    expected = graph_identity(graph.tail, graph.head, graph.node_x, graph.node_y)
    if expected != graph.identity:
        raise RuntimeError("saved graph identity mismatch")
    return graph


def save_da(path: Path, da: SparseDA) -> None:
    atomic_save_npz(
        path,
        size=np.asarray(da.size, dtype=np.int64),
        row_offsets=np.asarray(da.row_offsets, dtype=np.int64),
        row_indices=np.asarray(da.row_indices, dtype=np.int64),
        row_values=np.asarray(da.row_values, dtype=np.int64),
        col_offsets=np.asarray(da.col_offsets, dtype=np.int64),
        col_indices=np.asarray(da.col_indices, dtype=np.int64),
        col_values=np.asarray(da.col_values, dtype=np.int64),
    )


def load_da(path: Path) -> SparseDA:
    with np.load(path, allow_pickle=False) as data:
        return SparseDA(
            size=int(data["size"].item()),
            row_offsets=tuple(data["row_offsets"].astype(np.int64).tolist()),
            row_indices=tuple(data["row_indices"].astype(np.int64).tolist()),
            row_values=tuple(data["row_values"].astype(np.int64).tolist()),
            col_offsets=tuple(data["col_offsets"].astype(np.int64).tolist()),
            col_indices=tuple(data["col_indices"].astype(np.int64).tolist()),
            col_values=tuple(data["col_values"].astype(np.int64).tolist()),
        )


def save_candidate_rows(path: Path, rows: dict[str, Any]) -> None:
    arrays = _validate_candidate_rows(rows)
    path.mkdir(parents=True, exist_ok=True)
    already_mapped = rows.get("_storage_dir") == str(path.resolve())
    files: dict[str, dict[str, Any]] = {}
    for name, array in arrays.items():
        target = path / f"{name}.npy"
        source_filename = getattr(array, "filename", None)
        if already_mapped and source_filename is not None and Path(
            source_filename
        ).resolve() == target.resolve():
            array.flush()
        else:
            atomic_save_npy(target, array)
        stored = np.load(target, mmap_mode="r", allow_pickle=False)
        files[name] = {
            "file": target.name,
            "dtype": str(stored.dtype),
            "shape": list(stored.shape),
            "sha256": sha256_file(target),
        }
        del stored
    metadata = {
        "schema": CANDIDATE_ROWS_SCHEMA,
        "row_count": len(arrays["source"]),
        "capacity": int(rows.get("_capacity", len(arrays["source"]))),
        "pool_size": int(arrays["candidates"].shape[1]),
        "storage": "npy_memmap_no_full_training_copy",
        "files": files,
    }
    write_json(path / "metadata.json", metadata)


def load_candidate_rows(path: Path, expected_pool_size: int) -> dict[str, Any]:
    metadata = load_json(path / "metadata.json")
    if set(metadata) != {
        "schema",
        "row_count",
        "capacity",
        "pool_size",
        "storage",
        "files",
    }:
        raise RuntimeError(f"{path} has an unsupported candidate metadata shape")
    if metadata["schema"] != CANDIDATE_ROWS_SCHEMA:
        raise RuntimeError(f"{path} uses an unsupported candidate row schema")
    if metadata["storage"] != "npy_memmap_no_full_training_copy":
        raise RuntimeError(f"{path} uses an unsupported candidate storage mode")
    pool_size = int(metadata["pool_size"])
    row_count = int(metadata["row_count"])
    capacity = int(metadata["capacity"])
    if pool_size != expected_pool_size:
        raise RuntimeError(f"{path} candidate pool size differs from preprocessing")
    if row_count < 0 or capacity < row_count:
        raise RuntimeError(f"{path} has invalid candidate row counts")
    expected_names = {"source", "destination", "candidates", "labels", "weights"}
    if not isinstance(metadata["files"], dict) or set(metadata["files"]) != expected_names:
        raise RuntimeError(f"{path} has an unsupported candidate file table")
    arrays: dict[str, Any] = {}
    for name in sorted(expected_names):
        descriptor = metadata["files"][name]
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "file",
            "dtype",
            "shape",
            "sha256",
        }:
            raise RuntimeError(f"{path} candidate file descriptor is malformed")
        if descriptor["file"] != f"{name}.npy":
            raise RuntimeError(f"{path} candidate filename is not canonical")
        array_path = path / descriptor["file"]
        if sha256_file(array_path) != descriptor["sha256"]:
            raise RuntimeError(f"{path} candidate file hash mismatch")
        stored = np.load(array_path, mmap_mode="c", allow_pickle=False)
        if str(stored.dtype) != descriptor["dtype"] or list(stored.shape) != descriptor[
            "shape"
        ]:
            raise RuntimeError(f"{path} candidate file shape/dtype mismatch")
        if stored.shape[0] != capacity:
            raise RuntimeError(f"{path} candidate file capacity mismatch")
        arrays[name] = stored[:row_count]
    return _validate_candidate_rows(arrays, expected_pool_size)


def _validate_candidate_rows(
    rows: dict[str, Any], expected_pool_size: int | None = None
) -> dict[str, Any]:
    expected_dtypes = {
        "source": np.dtype(np.int32),
        "destination": np.dtype(np.int32),
        "candidates": np.dtype(np.int32),
        "labels": np.dtype(np.uint8),
        "weights": np.dtype(np.float32),
    }
    arrays: dict[str, Any] = {}
    for name, dtype in expected_dtypes.items():
        value = rows.get(name)
        if not isinstance(value, np.ndarray) or value.dtype != dtype:
            raise RuntimeError(f"candidate field {name!r} must have dtype {dtype}")
        if not value.flags.c_contiguous:
            raise RuntimeError(f"candidate field {name!r} must be C-contiguous")
        arrays[name] = value
    row_count = len(arrays["source"])
    if arrays["source"].shape != (row_count,) or arrays["destination"].shape != (
        row_count,
    ):
        raise RuntimeError("candidate source/destination arrays have incompatible shapes")
    candidate_shape = arrays["candidates"].shape
    if len(candidate_shape) != 2 or candidate_shape[0] != row_count:
        raise RuntimeError("candidate ID array must be a two-dimensional row table")
    if expected_pool_size is not None and candidate_shape[1] != expected_pool_size:
        raise RuntimeError("candidate ID array has the wrong pool size")
    if arrays["labels"].shape != candidate_shape or arrays["weights"].shape != candidate_shape:
        raise RuntimeError("candidate label/weight arrays have incompatible shapes")
    # Validate in bounded row blocks so loading a memory-mapped formal table
    # never creates a table-sized boolean mask or fancy-indexed copy.
    for start in range(0, row_count, 16_384):
        stop = min(start + 16_384, row_count)
        candidate_block = arrays["candidates"][start:stop]
        label_block = arrays["labels"][start:stop]
        weight_block = arrays["weights"][start:stop]
        if np.any(candidate_block < -1):
            raise RuntimeError("candidate IDs must use -1 as the only padding value")
        if np.any(label_block > 1):
            raise RuntimeError("candidate labels must be binary")
        padding = candidate_block == -1
        if np.any(label_block[padding] != 0) or np.any(weight_block[padding] != 0):
            raise RuntimeError("candidate padding must have zero label and weight")
        if np.any(weight_block[~padding] < 1.0):
            raise RuntimeError("real candidate weights must be at least one")
    return arrays


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as output:
        np.savez(output, **arrays)
    os.replace(temporary, path)


def atomic_save_npy(path: Path, array: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as output:
        np.save(output, array, allow_pickle=False)
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = decode_json(path.read_text(encoding="utf-8"), str(path))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain an object")
    return value


def _model_configuration(args: argparse.Namespace, embedding_size: int) -> dict[str, Any]:
    return {
        "schema": KSD_MODEL_SCHEMA,
        "embedding_size": embedding_size,
        "query_hidden_size": args.query_hidden_size,
        "representation_size": args.representation_size,
        "candidate_embedding_size": args.candidate_embedding_size,
        "candidate_hidden_size": args.candidate_hidden_size,
        "dropout": args.dropout,
        "fixed_source_offset": FIXED_SOURCE_OFFSET,
        "fixed_destination_offset": FIXED_DESTINATION_OFFSET,
        "query_features": [
            "trainable_node2vec_source_embedding",
            "source_offset_ratio",
            "trainable_node2vec_destination_embedding",
            "destination_offset_ratio",
        ],
        "query_concat_order": ["e_s", "r_s", "e_d", "r_d"],
        "candidate_features": [
            "trainable_candidate_id_embedding",
            "global_popularity_minmax_[0,1]",
        ],
        "candidate_scoring": "query_candidate_representation_dot_product",
        "query_mlp": "linear_relu_linear",
        "candidate_mlp": "linear_relu_linear",
        "normalization": "none",
        "architecture_variant": (
            "paper_eq6_9"
            if args.dropout == 0
            else "paper_eq6_9_plus_explicit_dropout"
        ),
        "paper_equations": [6, 7, 8, 9],
        "source_of_truth": "published_paper_over_conflicting_official_release",
        "key_num": 1,
        "max_route_length": args.max_route_length,
    }


def build_ksd_model(
    graph: StaticGraph,
    node2vec: Any,
    popularity: Any,
    configuration: dict[str, Any],
    device: Any,
) -> Any:
    nn = torch.nn
    embedding_size = int(configuration["embedding_size"])
    representation_size = int(configuration["representation_size"])
    query_hidden = int(configuration["query_hidden_size"])
    candidate_embedding_size = int(configuration["candidate_embedding_size"])
    candidate_hidden = int(configuration["candidate_hidden_size"])
    dropout = float(configuration["dropout"])
    if configuration.get("schema") != KSD_MODEL_SCHEMA:
        raise RuntimeError("unsupported DRPK-static KSD model schema")
    if min(
        embedding_size,
        representation_size,
        query_hidden,
        candidate_embedding_size,
        candidate_hidden,
    ) <= 0:
        raise RuntimeError("KSD representation dimensions must be positive")
    if configuration.get("fixed_source_offset") != FIXED_SOURCE_OFFSET or configuration.get(
        "fixed_destination_offset"
    ) != FIXED_DESTINATION_OFFSET:
        raise RuntimeError("KSD fixed offset ratios differ from DRPK-static protocol")

    class KSD(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            initial = torch.as_tensor(node2vec, dtype=torch.float32)
            if tuple(initial.shape) != (graph.edge_count, embedding_size):
                raise RuntimeError("Node2Vec shape differs from model configuration")
            self.source_embedding = nn.Embedding.from_pretrained(
                initial.clone(), freeze=False
            )
            self.destination_embedding = nn.Embedding.from_pretrained(
                initial.clone(), freeze=False
            )
            self.query_encoder = nn.Sequential(
                nn.Linear(embedding_size * 2 + 2, query_hidden),
                nn.ReLU(),
                *([] if dropout == 0 else [nn.Dropout(dropout)]),
                nn.Linear(query_hidden, representation_size),
            )
            self.candidate_embedding = nn.Embedding(
                graph.edge_count, candidate_embedding_size
            )
            self.candidate_encoder = nn.Sequential(
                nn.Linear(candidate_embedding_size + 1, candidate_hidden),
                nn.ReLU(),
                *([] if dropout == 0 else [nn.Dropout(dropout)]),
                nn.Linear(candidate_hidden, representation_size),
            )
            self.register_buffer(
                "popularity", torch.as_tensor(popularity, dtype=torch.float32)
            )
            if tuple(self.popularity.shape) != (graph.edge_count,):
                raise RuntimeError("popularity shape differs from road vocabulary")

        def forward(self, source: Any, destination: Any, candidates: Any) -> Any:
            candidate_mask = candidates >= 0
            source_offset = torch.full(
                (source.shape[0], 1),
                FIXED_SOURCE_OFFSET,
                dtype=torch.float32,
                device=source.device,
            )
            destination_offset = torch.full(
                (source.shape[0], 1),
                FIXED_DESTINATION_OFFSET,
                dtype=torch.float32,
                device=source.device,
            )
            query_input = torch.cat(
                (
                    self.source_embedding(source),
                    source_offset,
                    self.destination_embedding(destination),
                    destination_offset,
                ),
                dim=1,
            )
            query = self.query_encoder(query_input)
            safe_candidates = candidates.clamp(min=0)
            candidate_input = torch.cat(
                (
                    self.candidate_embedding(safe_candidates),
                    self.popularity[safe_candidates].unsqueeze(2),
                ),
                dim=2,
            )
            flattened_input = candidate_input.reshape(
                -1, candidate_embedding_size + 1
            )
            flattened_mask = candidate_mask.reshape(-1)
            flattened_representation = torch.zeros(
                (flattened_input.shape[0], representation_size),
                dtype=query.dtype,
                device=query.device,
            )
            if bool(flattened_mask.any()):
                flattened_representation[flattened_mask] = self.candidate_encoder(
                    flattened_input[flattened_mask]
                )
            candidate_representation = flattened_representation.reshape(
                candidates.shape[0], candidates.shape[1], representation_size
            )
            logits = (candidate_representation * query.unsqueeze(1)).sum(dim=2)
            return logits.masked_fill(~candidate_mask, -1e9)

    return KSD().to(device)


def _load_routing_preprocess(
    preprocess_dir: Path,
) -> tuple[dict[str, Any], StaticGraph, SparseDA]:
    """Load only the configuration, raw-edge graph, and DA needed by DRP-TP."""

    configuration = load_json(preprocess_dir / "routing-configuration.json")
    if configuration.get("schema") != ROUTING_PREPROCESS_SCHEMA:
        raise RuntimeError("unsupported DRPK-static routing preprocess schema")
    require_same_adapter_source(configuration.get("source"), "routing preprocess")
    if configuration.get("provenance", {}).get("official_commit") != OFFICIAL_COMMIT:
        raise RuntimeError("preprocess provenance commit mismatch")
    validate_routing_artifact_manifest(preprocess_dir, configuration)
    graph = load_saved_graph(preprocess_dir / "graph.npz")
    if graph.identity != configuration.get("graph_identity"):
        raise RuntimeError("preprocess graph identity mismatch")
    da = load_compact_da(preprocess_dir / "da")
    if da.size != graph.edge_count:
        raise RuntimeError("DA and graph size differ")
    return configuration, graph, da


def _load_preprocess(
    preprocess_dir: Path,
) -> tuple[dict[str, Any], StaticGraph, SparseDA, Any, Any]:
    routing_configuration, graph, da = _load_routing_preprocess(preprocess_dir)
    configuration = load_json(preprocess_dir / "configuration.json")
    if configuration.get("schema") != PREPROCESS_SCHEMA:
        raise RuntimeError("unsupported DRPK-static preprocess schema")
    require_same_adapter_source(configuration.get("source"), "full preprocess")
    if configuration.get("provenance", {}).get("official_commit") != OFFICIAL_COMMIT:
        raise RuntimeError("preprocess provenance commit mismatch")
    routing_reference = configuration.get("routing_configuration")
    if not isinstance(routing_reference, dict) or routing_reference != {
        "schema": ROUTING_PREPROCESS_SCHEMA,
        "path": "routing-configuration.json",
        "sha256": sha256_file(preprocess_dir / "routing-configuration.json"),
    }:
        raise RuntimeError("full preprocess does not bind the routing configuration")
    if (
        routing_configuration.get("network_id") != configuration.get("network_id")
        or routing_configuration.get("graph_identity")
        != configuration.get("graph_identity")
    ):
        raise RuntimeError("routing and full preprocess identities differ")
    if routing_configuration.get("source") != configuration.get("source"):
        raise RuntimeError("routing and full preprocess source identities differ")
    validate_core_artifact_manifest(
        preprocess_dir, configuration, required_roles={"routing", "model"}
    )
    node2vec = np.load(preprocess_dir / "node2vec.npy", allow_pickle=False)
    with np.load(preprocess_dir / "static_features.npz", allow_pickle=False) as features:
        expected_fields = {
            "schema",
            "popularity_counts",
            "normalized_popularity",
            "popularity_48",
        }
        if set(features.files) != expected_fields:
            raise RuntimeError("static feature artifact has an unsupported shape")
        if str(features["schema"].item()) != STATIC_FEATURES_SCHEMA:
            raise RuntimeError("unsupported DRPK-static feature schema")
        popularity = features["normalized_popularity"].astype(np.float32)
        popularity_48 = features["popularity_48"]
        if popularity.shape != (graph.edge_count,):
            raise RuntimeError("time-collapsed popularity vector has wrong shape")
        if np.any(popularity < 0.0) or np.any(popularity > 1.0):
            raise RuntimeError("DRPK-static popularity is not min-max normalized")
        if popularity_48.shape != (graph.edge_count, TIME_SLOT_COUNT):
            raise RuntimeError("time-collapsed popularity table has wrong shape")
        if not np.all(popularity_48 == popularity_48[:, :1]):
            raise RuntimeError("DRPK-static popularity slots are not identical")
        if not np.array_equal(popularity_48[:, 0], popularity):
            raise RuntimeError("DRPK-static popularity table differs from its vector")
    return configuration, graph, da, node2vec, popularity


def evaluate_ksd_candidates(
    model: Any,
    arrays: dict[str, Any],
    batch_size: int,
    device: Any,
) -> dict[str, Any]:
    """Evaluate the native DRPK key-segment validation objective."""

    if len(arrays["source"]) == 0:
        raise RuntimeError("KSD validation requires labeled candidate rows")
    model.eval()
    loss_sum = 0.0
    scored_candidates = 0
    correct = 0
    with torch.no_grad():
        for start in range(0, len(arrays["source"]), batch_size):
            stop = min(start + batch_size, len(arrays["source"]))
            source = torch.as_tensor(arrays["source"][start:stop], device=device).long()
            destination = torch.as_tensor(
                arrays["destination"][start:stop], device=device
            ).long()
            candidates = torch.as_tensor(
                arrays["candidates"][start:stop], device=device
            ).long()
            labels = torch.as_tensor(
                arrays["labels"][start:stop], device=device
            ).float()
            weights = torch.as_tensor(arrays["weights"][start:stop], device=device)
            mask = candidates >= 0
            logits = model(source, destination, candidates)
            losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            loss_sum += float((losses * weights * mask).sum())
            scored_candidates += int(mask.sum())
            best = torch.argmax(logits, dim=1)
            selected_labels = labels.gather(1, best.unsqueeze(1)).squeeze(1)
            correct += int((selected_labels > 0).sum())
    rows = int(len(arrays["source"]))
    return {
        "rows": rows,
        "scored_candidates": scored_candidates,
        # DRPK Eq. (3) sums candidate losses inside each route and Eq. (4)
        # averages those route losses.  Dividing by the number of candidates
        # would silently optimize a different objective.
        "mean_weighted_bce": loss_sum / rows,
        "loss_normalization": "sum_candidates_then_mean_routes_eq3_4",
        "top1_key_accuracy": correct / rows,
    }


def train_command(args: argparse.Namespace) -> None:
    if (
        min(
            args.workers,
            args.epochs,
            args.validation_every,
            args.batch_size,
            args.microbatch_size,
            args.query_hidden_size,
            args.representation_size,
            args.candidate_embedding_size,
            args.candidate_hidden_size,
            args.max_route_length,
            args.scheduler_patience,
        )
        <= 0
        or args.learning_rate <= 0
        or not (0 <= args.dropout < 1)
        or not (0 < args.scheduler_factor < 1)
        or args.scheduler_threshold < 0
        or args.early_stop_learning_rate < 0
    ):
        raise RuntimeError("invalid KSD training configuration")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    total_started = time.perf_counter()
    preprocess_configuration, graph, da, node2vec, popularity = _load_preprocess(
        args.preprocess_dir
    )
    if (
        args.source_revision
        != preprocess_configuration.get("source", {}).get("source_revision")
    ):
        raise RuntimeError("training source revision differs from preprocessing")
    validation = load_dataset_manifest(args.validation_manifest)
    if validation.manifest.network_id != preprocess_configuration["network_id"]:
        raise RuntimeError("validation network differs from preprocessing")
    if validation.manifest_sha256 != preprocess_configuration["validation_manifest_sha256"]:
        raise RuntimeError("validation manifest differs from preprocessing")
    if validation.records_sha256 != preprocess_configuration["validation_records_sha256"]:
        raise RuntimeError("validation records differ from preprocessing")
    validate_trips(validation.trips, graph)
    pool_size = int(preprocess_configuration["candidate_pool_size"])
    candidate_configuration = preprocess_configuration.get("candidate_rows", {})
    if sha256_file(args.preprocess_dir / "train_candidates" / "metadata.json") != candidate_configuration.get(
        "train_metadata_sha256"
    ):
        raise RuntimeError("training candidate metadata differs from preprocessing")
    if sha256_file(
        args.preprocess_dir / "validation_candidates" / "metadata.json"
    ) != candidate_configuration.get("validation_metadata_sha256"):
        raise RuntimeError("validation candidate metadata differs from preprocessing")
    train_arrays = load_candidate_rows(
        args.preprocess_dir / "train_candidates", pool_size
    )
    validation_arrays = load_candidate_rows(
        args.preprocess_dir / "validation_candidates", pool_size
    )
    if len(train_arrays["source"]) < 2:
        raise RuntimeError("KSD training requires at least two labeled candidate rows")
    if len(validation_arrays["source"]) == 0:
        raise RuntimeError("KSD validation requires labeled candidate rows")
    device = resolve_model_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model_configuration = _model_configuration(args, int(node2vec.shape[1]))
    model = build_ksd_model(graph, node2vec, popularity, model_configuration, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=args.scheduler_patience,
        factor=args.scheduler_factor,
        threshold=args.scheduler_threshold,
    )
    configuration = {
        "schema": TRAINING_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "source": adapter_source_identity(args.source_revision),
        "provenance": upstream_provenance(),
        "preprocess_configuration_sha256": sha256_file(
            args.preprocess_dir / "configuration.json"
        ),
        "graph_identity": graph.identity,
        "network_id": preprocess_configuration["network_id"],
        "seed": args.seed,
        "workers": args.workers,
        "requested_device": args.device,
        "resolved_device": str(device),
        "environment": model_environment(device),
        "epochs": args.epochs,
        "validation_every": args.validation_every,
        "batch_size": args.batch_size,
        "microbatch_size": args.microbatch_size,
        "optimizer_step_semantics": "one_step_per_macro_route_batch",
        "loss_normalization": "sum_candidates_then_mean_routes_eq3_4",
        "learning_rate": args.learning_rate,
        "scheduler": {
            "metric": "validation_top1_key_accuracy",
            "patience": args.scheduler_patience,
            "factor": args.scheduler_factor,
            "threshold": args.scheduler_threshold,
            "early_stopping_enabled": args.early_stop_learning_rate > 0,
            "early_stop_learning_rate": args.early_stop_learning_rate,
        },
        "model": model_configuration,
        "validation_manifest_sha256": validation.manifest_sha256,
        "test_manifest_read": False,
    }
    write_json(output / "configuration.json", configuration)
    log_path = output / "training.jsonl"
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    log_file.write(json.dumps({"event": "configuration", **configuration}) + "\n")
    rng = np.random.default_rng(args.seed)
    evaluations: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    best_loss: dict[str, Any] | None = None
    epochs_completed = 0
    optimizer_steps_total = 0
    microbatches_total = 0
    training_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epochs_completed = epoch
        epoch_started = time.perf_counter()
        model.train()
        order = rng.permutation(len(train_arrays["source"]))
        loss_sum = 0.0
        effective_rows = 0
        epoch_optimizer_steps = 0
        epoch_microbatches = 0
        for start in range(0, len(order), args.batch_size):
            macro_indices = order[start : start + args.batch_size]
            macro_rows = len(macro_indices)
            if macro_rows == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            macro_loss_sum = 0.0
            # Accumulate gradients over memory-bounded chunks but normalize
            # every chunk by the complete macro batch.  The resulting update
            # is exactly the Eq. (4) route mean for the requested 8192-route
            # optimizer batch, including its smaller final batch.
            for micro_start in range(0, macro_rows, args.microbatch_size):
                indices = macro_indices[
                    micro_start : micro_start + args.microbatch_size
                ]
                source = torch.as_tensor(
                    train_arrays["source"][indices], device=device
                ).long()
                destination = torch.as_tensor(
                    train_arrays["destination"][indices], device=device
                ).long()
                candidates = torch.as_tensor(
                    train_arrays["candidates"][indices], device=device
                ).long()
                labels = torch.as_tensor(
                    train_arrays["labels"][indices], device=device
                ).float()
                weights = torch.as_tensor(
                    train_arrays["weights"][indices], device=device
                ).float()
                mask = candidates >= 0
                logits = model(source, destination, candidates)
                losses = F.binary_cross_entropy_with_logits(
                    logits, labels, reduction="none"
                )
                chunk_loss_sum = (losses * weights * mask).sum()
                (chunk_loss_sum / macro_rows).backward()
                macro_loss_sum += float(chunk_loss_sum.detach())
                epoch_microbatches += 1
            optimizer.step()
            epoch_optimizer_steps += 1
            loss_sum += macro_loss_sum
            effective_rows += macro_rows
        if effective_rows == 0:
            raise RuntimeError("KSD epoch contained no batch with at least two rows")
        epoch_event = {
            "event": "epoch",
            "epoch": epoch,
            "mean_weighted_bce": loss_sum / effective_rows,
            "loss_normalization": "sum_candidates_then_mean_routes_eq3_4",
            "effective_rows": effective_rows,
            "optimizer_steps": epoch_optimizer_steps,
            "microbatches": epoch_microbatches,
            "epoch_seconds": time.perf_counter() - epoch_started,
            "peak_rss_kib": peak_rss_kib(),
        }
        log_file.write(json.dumps(epoch_event) + "\n")
        optimizer_steps_total += epoch_optimizer_steps
        microbatches_total += epoch_microbatches
        if epoch % args.validation_every == 0 or epoch == args.epochs:
            validation_started = time.perf_counter()
            metrics = evaluate_ksd_candidates(
                model, validation_arrays, args.microbatch_size, device
            )
            evaluation = {
                "epoch": epoch,
                "metrics": metrics,
                "seconds": time.perf_counter() - validation_started,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            evaluations.append(evaluation)
            log_file.write(json.dumps({"event": "validation", **evaluation}) + "\n")
            if selected is None or (
                metrics["top1_key_accuracy"],
                -metrics["mean_weighted_bce"],
                -epoch,
            ) > (
                selected["metrics"]["top1_key_accuracy"],
                -selected["metrics"]["mean_weighted_bce"],
                -selected["epoch"],
            ):
                selected = evaluation.copy()
                selected["checkpoint"] = str(output / "checkpoint-best-accuracy.pt")
                save_checkpoint(
                    output / "checkpoint-best-accuracy.pt",
                    model,
                    optimizer,
                    epoch,
                    configuration,
                )
            if best_loss is None or metrics["mean_weighted_bce"] < best_loss["metrics"]["mean_weighted_bce"]:
                best_loss = evaluation.copy()
                best_loss["checkpoint"] = str(output / "checkpoint-best-loss.pt")
                save_checkpoint(
                    output / "checkpoint-best-loss.pt",
                    model,
                    optimizer,
                    epoch,
                    configuration,
                )
            scheduler.step(float(metrics["top1_key_accuracy"]))
            if (
                args.early_stop_learning_rate > 0
                and optimizer.param_groups[0]["lr"]
                <= args.early_stop_learning_rate
            ):
                log_file.write(
                    json.dumps(
                        {
                            "event": "early_stop",
                            "epoch": epoch,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        }
                    )
                    + "\n"
                )
                break
    if not evaluations:
        raise RuntimeError("KSD training performed no validation")
    assert selected is not None and best_loss is not None
    save_checkpoint(
        output / "checkpoint-last.pt",
        model,
        optimizer,
        epochs_completed,
        configuration,
    )
    selected["checkpoint_sha256"] = sha256_file(Path(selected["checkpoint"]))
    best_loss["checkpoint_sha256"] = sha256_file(Path(best_loss["checkpoint"]))
    selection = {
        "schema": SELECTION_SCHEMA,
        "source": adapter_source_identity(args.source_revision),
        "selection_rule": [
            "maximum_validation_top1_key_accuracy",
            "minimum_validation_weighted_bce",
            "earliest_epoch",
        ],
        "selected": selected,
        "best_loss": best_loss,
        "evaluations": evaluations,
        "epochs_completed": epochs_completed,
        "optimizer_steps": optimizer_steps_total,
        "microbatches": microbatches_total,
        "training_seconds": time.perf_counter() - training_started,
        "total_seconds": time.perf_counter() - total_started,
        "peak_rss_kib": peak_rss_kib(),
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "requested_device": args.device,
        "resolved_device": str(device),
        "workers": args.workers,
        "environment": model_environment(device),
        "checkpoint_last": {
            "path": str(output / "checkpoint-last.pt"),
            "sha256": sha256_file(output / "checkpoint-last.pt"),
            "epoch": epochs_completed,
        },
    }
    write_json(output / "selection.json", selection)
    log_file.write(json.dumps({"event": "finished", **selection}) + "\n")
    log_file.close()
    print(json.dumps(selected, indent=2))


def save_checkpoint(
    path: Path, model: Any, optimizer: Any, epoch: int, configuration: dict[str, Any]
) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "adapter_version": ADAPTER_VERSION,
            "official_commit": OFFICIAL_COMMIT,
            "epoch": epoch,
            "configuration": configuration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        temporary,
    )
    os.replace(temporary, path)


def _load_checkpoint(
    checkpoint_path: Path,
    preprocess_configuration: dict[str, Any],
    preprocess_configuration_sha256: str,
    graph: StaticGraph,
    node2vec: Any,
    popularity: Any,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("unsupported DRPK-static checkpoint")
    if checkpoint.get("official_commit") != OFFICIAL_COMMIT:
        raise RuntimeError("checkpoint provenance mismatch")
    configuration = checkpoint["configuration"]
    require_same_adapter_source(configuration.get("source"), "training checkpoint")
    if configuration.get("source") != preprocess_configuration.get("source"):
        raise RuntimeError("checkpoint and preprocess source identities differ")
    if configuration["graph_identity"] != graph.identity:
        raise RuntimeError("checkpoint graph identity mismatch")
    if configuration["network_id"] != preprocess_configuration["network_id"]:
        raise RuntimeError("checkpoint network mismatch")
    if (
        configuration["preprocess_configuration_sha256"]
        != preprocess_configuration_sha256
    ):
        raise RuntimeError("checkpoint preprocess configuration mismatch")
    model = build_ksd_model(
        graph, node2vec, popularity, configuration["model"], device
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _cosine_tie_break(
    current: int, destination: int, candidates: Sequence[int], graph: StaticGraph
) -> int:
    current_target = graph.target_xy(current)
    destination_source = graph.source_xy(destination)
    toward_destination = (
        destination_source[0] - current_target[0],
        destination_source[1] - current_target[1],
    )
    target_norm = math.hypot(*toward_destination)
    ranked: list[tuple[float, int]] = []
    for candidate in candidates:
        source = graph.source_xy(candidate)
        target = graph.target_xy(candidate)
        direction = (target[0] - source[0], target[1] - source[1])
        direction_norm = math.hypot(*direction)
        cosine = (
            1.0
            if direction_norm == 0 or target_norm == 0
            else (
                toward_destination[0] * direction[0]
                + toward_destination[1] * direction[1]
            )
            / (target_norm * direction_norm)
        )
        ranked.append((-cosine, candidate))
    return min(ranked)[1]


def plan_drp_leg(
    source: int,
    destination: int,
    graph: StaticGraph,
    da: SparseDA,
    popularity: Sequence[float],
    max_length: int,
    *,
    use_popularity: bool,
) -> list[int]:
    if max_length <= 0:
        raise RuntimeError("maximum route length must be positive")
    route = [source]
    used = [0] * graph.edge_count
    used[source] = 1
    used[destination] = 1
    while route[-1] != destination and len(route) < max_length:
        current = route[-1]
        neighbors = list(graph.neighbors[current])
        if not neighbors:
            break
        if destination in neighbors:
            route.append(destination)
            break
        eligible = [edge for edge in neighbors if used[edge] < 1]
        if eligible:
            best_da = max(da.value(edge, destination) for edge in eligible)
            tied = [
                edge for edge in eligible if da.value(edge, destination) == best_da
            ]
        else:
            # The upstream planner falls back to every outgoing road only
            # after its one-use filter has exhausted them all.
            best_da = -1
            tied = neighbors
        if len(tied) == 1:
            selected = tied[0]
        elif use_popularity:
            if best_da >= 1:
                best_popularity = max(popularity[edge] for edge in tied)
                tied = [
                    edge for edge in tied if popularity[edge] == best_popularity
                ]
            selected = tied[0] if len(tied) == 1 else _cosine_tie_break(
                current, destination, tied, graph
            )
        else:
            selected = tied[0]
        route.append(selected)
        used[selected] += 1
    return route


def plan_via_key(
    source: int,
    key: int | None,
    destination: int,
    graph: StaticGraph,
    da: SparseDA,
    popularity: Sequence[float],
    max_length: int,
) -> list[int]:
    if key is None or key in {source, destination}:
        return plan_drp_leg(
            source,
            destination,
            graph,
            da,
            popularity,
            max_length,
            use_popularity=True,
        )
    first = plan_drp_leg(
        source, key, graph, da, popularity, max_length, use_popularity=True
    )
    if first[-1] != key:
        return first
    second = plan_drp_leg(
        key,
        destination,
        graph,
        da,
        popularity,
        max_length,
        use_popularity=True,
    )
    return first + second[1:]


def predict_key_segment(
    model: Any,
    source: int,
    destination: int,
    da: SparseDA,
    pool_size: int,
    device: Any,
) -> int | None:
    pool = da.candidate_pool(source, destination, pool_size)
    if not pool:
        return None
    candidates = torch.tensor([[edge for edge, _ in pool]], device=device).long()
    source_tensor = torch.tensor([source], device=device).long()
    destination_tensor = torch.tensor([destination], device=device).long()
    model.eval()
    with torch.no_grad():
        logits = model(source_tensor, destination_tensor, candidates)
    return pool[int(torch.argmax(logits, dim=1).item())][0]


def predict_keys_batched(
    model: Any,
    trips: Sequence[Trip],
    da: SparseDA,
    pool_size: int,
    device: Any,
    batch_size: int,
) -> tuple[list[int | None], dict[str, float]]:
    if batch_size <= 0:
        raise RuntimeError("inference batch size must be positive")
    pool_started = time.perf_counter()
    pools = [
        [edge for edge, _ in da.candidate_pool(trip.edges[0], trip.edges[-1], pool_size)]
        for trip in trips
    ]
    pool_seconds = time.perf_counter() - pool_started
    selected: list[int | None] = []
    scoring_started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(trips), batch_size):
            batch_trips = trips[start : start + batch_size]
            batch_pools = pools[start : start + batch_size]
            candidates = [
                pool + [-1] * (pool_size - len(pool)) for pool in batch_pools
            ]
            source = torch.tensor(
                [trip.edges[0] for trip in batch_trips], device=device
            ).long()
            destination = torch.tensor(
                [trip.edges[-1] for trip in batch_trips], device=device
            ).long()
            candidate_tensor = torch.tensor(candidates, device=device).long()
            logits = model(source, destination, candidate_tensor)
            best_indices = torch.argmax(logits, dim=1).detach().cpu().tolist()
            selected.extend(
                None if not pool else pool[index]
                for pool, index in zip(batch_pools, best_indices)
            )
    synchronize_device(device)
    return selected, {
        "candidate_pool_seconds": pool_seconds,
        "key_model_seconds": time.perf_counter() - scoring_started,
    }


def drpk_routes(
    model: Any,
    trips: Sequence[Trip],
    graph: StaticGraph,
    da: SparseDA,
    popularity: Sequence[float],
    device: Any,
    max_length: int,
    pool_size: int,
    batch_size: int = 32,
) -> list[list[int]]:
    keys, _ = predict_keys_batched(
        model, trips, da, pool_size, device, batch_size
    )
    return [
        plan_via_key(
            trip.edges[0], key, trip.edges[-1], graph, da, popularity, max_length
        )
        for trip, key in zip(trips, keys)
    ]


def drp_tp_routes(
    trips: Sequence[Trip],
    graph: StaticGraph,
    da: SparseDA,
    max_length: int,
) -> list[list[int]]:
    return [
        plan_drp_leg(
            trip.edges[0],
            trip.edges[-1],
            graph,
            da,
            (),
            max_length,
            use_popularity=False,
        )
        for trip in trips
    ]


def synchronize_device(device: Any) -> None:
    if resolved_device_type(device) == "cuda":
        torch.cuda.synchronize(device)


def resolved_device_type(device: Any) -> str:
    return str(getattr(device, "type", device)).split(":", maxsplit=1)[0]


def predict_command(args: argparse.Namespace) -> None:
    if (
        args.warmup_repetitions < 0
        or args.measured_repetitions <= 0
        or args.inference_batch_size <= 0
    ):
        raise RuntimeError("invalid prediction repetition counts")
    if not args.source_revision.strip():
        raise RuntimeError("source revision must not be empty")
    if args.method == "drpk_static" and args.checkpoint is None:
        raise RuntimeError("drpk_static prediction requires --checkpoint")
    if args.method == "drp_tp" and args.checkpoint is not None:
        raise RuntimeError("drp_tp prediction must not receive a KSD checkpoint")
    total_started = time.perf_counter()
    load_started = time.perf_counter()
    popularity: list[float] | None = None
    node2vec = None
    popularity_array = None
    if args.method == "drpk_static":
        (
            preprocess_configuration,
            graph,
            da,
            node2vec,
            popularity_array,
        ) = _load_preprocess(args.preprocess_dir)
        popularity = popularity_array.tolist()
        device: Any = resolve_model_device(args.device)
    else:
        preprocess_configuration, graph, da = _load_routing_preprocess(
            args.preprocess_dir
        )
        # DRP-TP is a NumPy/stdlib CPU planner.  Do not resolve the requested
        # model device through PyTorch or charge GPU initialization to it.
        device = "cpu"
    if (
        args.source_revision
        != preprocess_configuration.get("source", {}).get("source_revision")
    ):
        raise RuntimeError("prediction source revision differs from preprocessing")
    source = adapter_source_identity(args.source_revision)
    dataset = load_dataset_manifest(args.dataset_manifest)
    if dataset.manifest.network_id != preprocess_configuration["network_id"]:
        raise RuntimeError("prediction dataset uses a different network")
    validate_trips(dataset.trips, graph)
    checkpoint: dict[str, Any] | None = None
    model = None
    preprocess_configuration_sha256 = sha256_file(
        args.preprocess_dir
        / (
            "configuration.json"
            if args.method == "drpk_static"
            else "routing-configuration.json"
        )
    )
    if args.method == "drpk_static":
        assert node2vec is not None and popularity_array is not None
        model, checkpoint = _load_checkpoint(
            args.checkpoint,
            preprocess_configuration,
            preprocess_configuration_sha256,
            graph,
            node2vec,
            popularity_array,
            device,
        )
        max_length = int(checkpoint["configuration"]["model"]["max_route_length"])
    else:
        max_length = int(preprocess_configuration["max_route_length"])
    load_seconds = time.perf_counter() - load_started

    def generate() -> tuple[list[list[int]], dict[str, float]]:
        if args.method == "drpk_static":
            assert model is not None and checkpoint is not None and popularity is not None
            keys, components = predict_keys_batched(
                model,
                dataset.trips,
                da,
                int(preprocess_configuration["candidate_pool_size"]),
                device,
                args.inference_batch_size,
            )
            planning_started = time.perf_counter()
            routes = [
                plan_via_key(
                    trip.edges[0],
                    key,
                    trip.edges[-1],
                    graph,
                    da,
                    popularity,
                    max_length,
                )
                for trip, key in zip(dataset.trips, keys)
            ]
            components["route_planning_seconds"] = (
                time.perf_counter() - planning_started
            )
            return routes, components
        planning_started = time.perf_counter()
        routes = drp_tp_routes(dataset.trips, graph, da, max_length)
        return routes, {
            "candidate_pool_seconds": 0.0,
            "key_model_seconds": 0.0,
            "route_planning_seconds": time.perf_counter() - planning_started,
        }

    warmup_seconds: list[float] = []
    warmup_components: list[dict[str, float]] = []
    reference: list[list[int]] | None = None
    for _ in range(args.warmup_repetitions):
        synchronize_device(device)
        started = time.perf_counter()
        candidate, components = generate()
        synchronize_device(device)
        warmup_seconds.append(time.perf_counter() - started)
        warmup_components.append(components)
        reference = candidate if reference is None else _verify_same(reference, candidate)
    if resolved_device_type(device) == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    repetition_seconds: list[float] = []
    repetition_components: list[dict[str, float]] = []
    generated: list[list[int]] | None = None
    for _ in range(args.measured_repetitions):
        synchronize_device(device)
        started = time.perf_counter()
        candidate, components = generate()
        synchronize_device(device)
        repetition_seconds.append(time.perf_counter() - started)
        repetition_components.append(components)
        reference = candidate if reference is None else _verify_same(reference, candidate)
        generated = candidate if generated is None else _verify_same(generated, candidate)
    assert generated is not None
    prediction_seconds = sum(repetition_seconds) / len(repetition_seconds)
    write_prediction_rows(args.predictions, dataset.trips, generated)
    failures = sum(
        prediction[-1] != trip.edges[-1]
        for trip, prediction in zip(dataset.trips, generated)
    )
    environment = (
        model_environment(device)
        if args.method == "drpk_static"
        else base_environment()
    )
    diagnostics = {
        "schema": DIAGNOSTICS_SCHEMA,
        "method": args.method,
        "source": source,
        "provenance": upstream_provenance(),
        "adaptation": preprocess_configuration["adaptation"],
        "dataset_id": dataset.manifest.dataset_id,
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "preprocess_configuration_sha256": preprocess_configuration_sha256,
        "checkpoint": (
            None
            if args.checkpoint is None or checkpoint is None
            else {
                "path": str(args.checkpoint),
                "sha256": sha256_file(args.checkpoint),
                "epoch": int(checkpoint["epoch"]),
            }
        ),
        "samples": len(dataset.trips),
        "query_protocol": "fixed_true_first_edge_to_true_last_edge",
        "truth_repair": False,
        "endpoint_failures": failures,
        "timing": {
            "artifact_and_model_load_seconds": load_seconds,
            "warmup_repetition_seconds": warmup_seconds,
            "prediction_repetition_seconds": repetition_seconds,
            "warmup_component_totals_per_repetition": warmup_components,
            "component_totals_per_repetition": repetition_components,
            "component_totals": mean_timing(repetition_components),
            "prediction_seconds": prediction_seconds,
            "mean_seconds_per_query": prediction_seconds / len(dataset.trips),
            "queries_per_second": len(dataset.trips) / prediction_seconds,
            "total_process_seconds": time.perf_counter() - total_started,
        },
        "peak_rss_kib": peak_rss_kib(),
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device)
            if resolved_device_type(device) == "cuda"
            else 0
        ),
        "requested_device": args.device,
        "resolved_device": str(device),
        "environment": environment,
        "seed": args.seed,
        "workers": args.workers,
        "inference_batch_size": args.inference_batch_size,
        "warmup_repetitions": args.warmup_repetitions,
        "measured_repetitions": args.measured_repetitions,
    }
    receipt = build_run_receipt(
        args,
        dataset,
        graph,
        device,
        checkpoint,
        preprocess_configuration,
        environment,
    )
    write_json(args.diagnostics, diagnostics)
    write_json(args.run_receipt, receipt)
    print(json.dumps(receipt, indent=2))


def _verify_same(reference: list[list[int]], candidate: list[list[int]]) -> list[list[int]]:
    if candidate != reference:
        raise RuntimeError("prediction repetitions produced different routes")
    return reference


def mean_timing(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise RuntimeError("timing mean requires at least one repetition")
    fields = set(rows[0])
    if any(set(row) != fields for row in rows):
        raise RuntimeError("timing repetitions expose different components")
    return {
        field: sum(row[field] for row in rows) / len(rows)
        for field in sorted(fields)
    }


def build_run_receipt(
    args: argparse.Namespace,
    dataset: DatasetArtifact,
    graph: StaticGraph,
    device: Any,
    checkpoint: dict[str, Any] | None,
    preprocess_configuration: dict[str, Any],
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    if environment is None:
        environment = (
            base_environment()
            if torch is None or resolved_device_type(device) == "cpu" and isinstance(device, str)
            else model_environment(device)
        )
    if not environment or any(
        not isinstance(key, str)
        or not key.strip()
        or not isinstance(value, str)
        or not value.strip()
        for key, value in environment.items()
    ):
        raise RuntimeError("run receipt environment must contain string pairs")
    checkpoint_epoch = None if checkpoint is None else int(checkpoint["epoch"])
    checkpoint_sha256 = (
        None if args.checkpoint is None else sha256_file(args.checkpoint)
    )
    return {
        "schema": RUN_RECEIPT_SCHEMA,
        "method": {"name": args.method, "version": ADAPTER_VERSION},
        "dataset_id": dataset.manifest.dataset_id,
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "prediction_records_schema": PREDICTION_RECORD_SCHEMA,
        "configuration": {
            "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_sha256": checkpoint_sha256,
            "network_id": dataset.manifest.network_id,
            "graph_identity": graph.identity,
            "query_protocol": "fixed_true_first_edge_to_true_last_edge",
            "truth_repair": False,
            "seed": args.seed,
            "workers": args.workers,
            "inference_batch_size": args.inference_batch_size,
            "warmup_repetitions": args.warmup_repetitions,
            "measured_repetitions": args.measured_repetitions,
            "requested_device": args.device,
            "resolved_device": str(device),
            "official_commit": OFFICIAL_COMMIT,
            "adapter_sha256": preprocess_configuration["source"][
                "adapter_sha256"
            ],
            "adaptation": preprocess_configuration["adaptation"],
        },
        "source_revision": args.source_revision,
        "environment": environment,
    }


def model_environment(device: Any) -> dict[str, str]:
    def version(distribution: str) -> str:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return "not-installed"

    return {
        **base_environment(str(device)),
        "gensim": version("gensim"),
        "pyshp": version("pyshp"),
        "scipy": version("scipy"),
        "torch": str(torch.__version__),
    }


def base_environment(device: str = "cpu") -> dict[str, str]:
    """Dependencies actually loaded by the DA-only DRP-TP path."""

    return {
        "device": device,
        "numpy": str(np.__version__),
        "python": platform.python_version(),
    }


def route_metrics(
    observed: list[list[int]], predicted: list[list[int]]
) -> dict[str, float | int]:
    if len(observed) != len(predicted) or not observed:
        raise RuntimeError("route metrics require equal nonempty route lists")
    precision = recall = f1 = jaccard = exact = 0.0
    for truth, prediction in zip(observed, predicted):
        truth_set = set(truth)
        predicted_set = set(prediction)
        intersection = len(truth_set & predicted_set)
        sample_precision = intersection / max(1, len(predicted_set))
        sample_recall = intersection / max(1, len(truth_set))
        sample_f1 = (
            0.0
            if sample_precision + sample_recall == 0
            else 2 * sample_precision * sample_recall / (sample_precision + sample_recall)
        )
        precision += sample_precision
        recall += sample_recall
        f1 += sample_f1
        jaccard += intersection / max(1, len(truth_set | predicted_set))
        exact += prediction == truth
    count = len(observed)
    return {
        "samples": count,
        "edge_precision": precision / count,
        "edge_recall": recall / count,
        "edge_f1": f1 / count,
        "edge_jaccard": jaccard / count,
        "exact_match": exact / count,
    }


def write_prediction_rows(
    path: Path, trips: Sequence[Trip], predictions: Sequence[Sequence[int]]
) -> None:
    if not trips or len(trips) != len(predictions):
        raise RuntimeError("predictions require equal nonempty rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for trip, prediction in zip(trips, predictions):
            if not prediction or any(
                isinstance(edge, bool)
                or not isinstance(edge, int)
                or edge < 0
                or edge > U32_MAX
                for edge in prediction
            ):
                raise RuntimeError(f"sample {trip.sample_id!r} has invalid prediction")
            output.write(
                json.dumps(
                    {
                        "sample_id": trip.sample_id,
                        "predicted_edge_ids": list(prediction),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def peak_rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


if __name__ == "__main__":
    main()
