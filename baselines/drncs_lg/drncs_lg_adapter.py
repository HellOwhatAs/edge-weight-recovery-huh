"""Clean-room DRNCS edge-state adapter for the EWR research protocol.

The implementation follows the algorithm described in the DRNCS paper while
using raw road IDs as graph states.  It intentionally does not import either
the production Rust workspace or the upstream DRNCS source tree.
"""

from __future__ import annotations

import argparse
import array
import copy
import dataclasses
import hashlib
import heapq
import importlib.metadata
import json
import math
import os
import pickle
import platform
import random
import resource
import sqlite3
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, Iterable, Iterator, Sequence, overload


ADAPTER_VERSION = "0.2.0"
METHOD_NAME = "drncs_lg"
DATASET_MANIFEST_SCHEMA = "ewr.dataset-manifest/v1"
DATASET_RECORD_SCHEMA = "ewr.dataset-record/v1"
PREDICTION_RECORD_SCHEMA = "ewr.prediction-record/v1"
RUN_RECEIPT_SCHEMA = "ewr.run-receipt/v1"
PREPROCESS_SCHEMA = "ewr.drncs-lg-preprocess/v2"
TRAINING_DIAGNOSTICS_SCHEMA = "ewr.drncs-lg-training-diagnostics/v2"
PREDICTION_DIAGNOSTICS_SCHEMA = "ewr.drncs-lg-prediction-diagnostics/v2"
CHECKPOINT_SCHEMA = "ewr.drncs-lg-checkpoint/v2"
SOURCE_PAPER = (
    "https://ecmlpkdd-storage.s3.eu-central-1.amazonaws.com/preprints/2025/"
    "research/preprint_ecml_pkdd_2025_research_132.pdf"
)
AUDITED_UPSTREAM_COMMIT = "8847482eb507785ee5b4e145f1a5144d1737fbe0"
UINT32_MAX = (1 << 32) - 1
MAX_CPU_WORKERS = 16


np: Any = None
torch: Any = None
shapefile: Any = None
Word2Vec: Any = None
gensim: Any = None


def load_preprocess_dependencies() -> None:
    """Load the NumPy/PyShp stack without importing or probing PyTorch."""

    global np, shapefile
    if np is not None and shapefile is not None:
        return
    try:
        import numpy as imported_numpy
        import shapefile as imported_shapefile
    except ImportError as error:
        raise RuntimeError(
            "DRNCS-LG preprocessing dependencies are unavailable; install this package "
            "in its pinned environment"
        ) from error
    np = imported_numpy
    shapefile = imported_shapefile


def load_array_dependencies() -> None:
    """Load preprocessing dependencies plus PyTorch for model commands."""

    global torch
    load_preprocess_dependencies()
    if torch is not None:
        return
    try:
        import torch as imported_torch
    except ImportError as error:
        raise RuntimeError(
            "DRNCS-LG model dependencies are unavailable; install this package "
            "in its pinned environment"
        ) from error
    torch = imported_torch


def load_node2vec_dependency() -> None:
    """Load the Gensim skip-gram implementation used by Node2Vec."""

    global Word2Vec, gensim
    load_preprocess_dependencies()
    if Word2Vec is not None:
        return
    try:
        import gensim as imported_gensim
        from gensim.models import Word2Vec as ImportedWord2Vec
    except ImportError as error:
        raise RuntimeError(
            "Gensim is required for Node2Vec preprocessing; install the pinned "
            "DRNCS-LG environment"
        ) from error
    gensim = imported_gensim
    Word2Vec = ImportedWord2Vec


def configure_runtime(seed: int, workers: int) -> int:
    """Fix available RNGs and cap all CPU thread pools at sixteen workers."""

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise RuntimeError("seed must be a non-negative integer")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise RuntimeError("workers must be a positive integer")
    effective = min(workers, MAX_CPU_WORKERS)
    os.environ["OMP_NUM_THREADS"] = str(effective)
    os.environ["MKL_NUM_THREADS"] = str(effective)
    os.environ["OPENBLAS_NUM_THREADS"] = str(effective)
    os.environ["NUMEXPR_NUM_THREADS"] = str(effective)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(effective)
    random.seed(seed)
    if np is not None:
        np.random.seed(seed % (1 << 32))
    if torch is not None:
        torch.manual_seed(seed)
        torch.set_num_threads(effective)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch permits setting this value only before parallel work starts.
            pass
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    return effective


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def load_strict_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"failed to read {path}: {error}") from error
    try:
        return json.loads(text, object_pairs_hook=reject_duplicate_keys)
    except (json.JSONDecodeError, RuntimeError) as error:
        raise RuntimeError(f"invalid JSON in {path}: {error}") from error


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_source_identity(source_revision: str | None = None) -> dict[str, str]:
    source_path = Path(__file__).resolve()
    revision = source_revision or os.environ.get("EWR_SOURCE_REVISION") or "unrecorded"
    return {
        "adapter_path": str(source_path),
        "adapter_sha256": sha256_file(source_path),
        "source_revision": require_nonempty_string(revision, "source revision"),
    }


def require_same_adapter_source(expected: Any, context: str) -> None:
    if not isinstance(expected, dict):
        raise RuntimeError(f"{context} does not contain adapter source identity")
    current = adapter_source_identity(expected.get("source_revision"))
    if expected.get("adapter_sha256") != current["adapter_sha256"]:
        raise RuntimeError(
            f"{context} was produced by a different DRNCS-LG adapter source"
        )


def write_bytes_atomic(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json_atomic(path: Path, value: Any) -> None:
    write_bytes_atomic(path, canonical_json_bytes(value))


def write_pickle_atomic(path: Path, value: Any) -> None:
    """Stream a pickle to an atomic temporary file.

    ``pickle.dumps`` briefly requires a second, contiguous in-memory copy of the
    complete preprocessing artifact.  The Beijing shortcut database can be
    large enough for that copy to trigger host swapping, so artifacts are
    serialized directly to disk instead.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            pickle.dump(value, stream, protocol=5)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_pickle(path: Path) -> Any:
    try:
        with path.open("rb") as stream:
            return pickle.load(stream)
    except (OSError, pickle.PickleError, EOFError) as error:
        raise RuntimeError(f"failed to load internal artifact {path}: {error}") from error


def require_exact_keys(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must be a JSON object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise RuntimeError(
            f"{context} fields differ: missing={missing}, unknown={unknown}"
        )
    return value


def require_nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{context} must be a non-empty string")
    return value


def require_u32(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError(f"{context} must be an unsigned 32-bit integer")
    if value < 0 or value > UINT32_MAX:
        raise RuntimeError(f"{context} is outside the unsigned 32-bit range")
    return value


@dataclass(frozen=True)
class Trip:
    sample_id: str
    edges: list[int]


class CompactTrips(Sequence[Trip]):
    """Routes backed by contiguous uint32 edges and uint64 offsets."""

    __slots__ = ("_sample_ids", "_offsets", "_edges")

    def __init__(
        self,
        sample_ids: list[str],
        offsets: array.array[int],
        edges: array.array[int],
    ) -> None:
        if (
            offsets.typecode != "Q"
            or edges.typecode != "I"
            or offsets.itemsize != 8
            or edges.itemsize != 4
        ):
            raise RuntimeError("compact routes require uint64 offsets and uint32 edges")
        if len(offsets) != len(sample_ids) + 1 or not offsets or offsets[0] != 0:
            raise RuntimeError("compact route offsets are inconsistent")
        if offsets[-1] != len(edges):
            raise RuntimeError("compact route edge storage is inconsistent")
        self._sample_ids = sample_ids
        self._offsets = offsets
        self._edges = edges

    def __len__(self) -> int:
        return len(self._sample_ids)

    @overload
    def __getitem__(self, index: int) -> Trip: ...

    @overload
    def __getitem__(self, index: slice) -> list[Trip]: ...

    def __getitem__(self, index: int | slice) -> Trip | list[Trip]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        left = int(self._offsets[index])
        right = int(self._offsets[index + 1])
        return Trip(self._sample_ids[index], self._edges[left:right].tolist())

    def __iter__(self) -> Iterator[Trip]:
        for index, sample_id in enumerate(self._sample_ids):
            left = int(self._offsets[index])
            right = int(self._offsets[index + 1])
            yield Trip(sample_id, self._edges[left:right].tolist())

    @property
    def edge_occurrences(self) -> int:
        return len(self._edges)

    def iter_sample_ids(self) -> Iterator[str]:
        return iter(self._sample_ids)

    @property
    def storage_bytes(self) -> int:
        return (
            self._offsets.buffer_info()[1] * self._offsets.itemsize
            + self._edges.buffer_info()[1] * self._edges.itemsize
        )


@dataclass(frozen=True)
class DatasetManifest:
    schema: str
    dataset_id: str
    network_id: str
    records_schema: str
    records_file: str
    split_role: str


@dataclass(frozen=True)
class DatasetArtifact:
    manifest: DatasetManifest
    manifest_path: Path
    manifest_sha256: str
    records_path: Path
    records_sha256: str
    trips: CompactTrips


def safe_relative_file(value: Any, context: str) -> str:
    path = require_nonempty_string(value, context)
    pure = PurePath(path)
    if pure.is_absolute() or ".." in pure.parts or path != pure.as_posix():
        raise RuntimeError(f"{context} must be a safe path relative to the manifest")
    return path


def _load_dataset_records_with_sha256(path: Path) -> tuple[CompactTrips, str]:
    """Parse strict JSONL once while hashing its exact bytes."""

    digest = hashlib.sha256()
    sample_ids: list[str] = []
    seen_sample_ids: set[str] = set()
    offsets = array.array("Q", [0])
    compact_edges = array.array("I")
    try:
        stream = path.open("rb")
    except OSError as error:
        raise RuntimeError(f"failed to read dataset records {path}: {error}") from error
    try:
        with stream:
            for line_number, raw_line in enumerate(stream, start=1):
                digest.update(raw_line)
                if not raw_line.endswith(b"\n"):
                    raise RuntimeError(
                        f"dataset records {path} must end with a newline"
                    )
                line = raw_line[:-1]
                if line.endswith(b"\r"):
                    line = line[:-1]
                if not line:
                    raise RuntimeError(f"blank dataset row at {path}:{line_number}")
                try:
                    decoded = line.decode("utf-8")
                    row = json.loads(decoded, object_pairs_hook=reject_duplicate_keys)
                except UnicodeDecodeError as error:
                    raise RuntimeError(
                        f"dataset records {path} are not UTF-8"
                    ) from error
                except (json.JSONDecodeError, RuntimeError) as error:
                    raise RuntimeError(
                        f"invalid dataset row at {path}:{line_number}: {error}"
                    ) from error
                row = require_exact_keys(
                    row,
                    {"sample_id", "original_edge_ids"},
                    f"dataset row {line_number}",
                )
                sample_id = require_nonempty_string(row["sample_id"], "sample_id")
                if sample_id in seen_sample_ids:
                    raise RuntimeError(f"duplicate sample_id {sample_id!r}")
                seen_sample_ids.add(sample_id)
                sample_ids.append(sample_id)
                raw_edges = row["original_edge_ids"]
                if not isinstance(raw_edges, list) or len(raw_edges) < 2:
                    raise RuntimeError(
                        f"sample {sample_id!r} must contain at least two original_edge_ids"
                    )
                for edge in raw_edges:
                    compact_edges.append(
                        require_u32(edge, f"sample {sample_id!r} edge")
                    )
                offsets.append(len(compact_edges))
    except OSError as error:
        raise RuntimeError(f"failed to read dataset records {path}: {error}") from error
    if not sample_ids:
        raise RuntimeError(f"dataset records {path} are empty")
    return CompactTrips(sample_ids, offsets, compact_edges), digest.hexdigest()


def load_dataset_records(path: Path) -> CompactTrips:
    """Parse strict JSONL incrementally into compact route arrays."""

    trips, _ = _load_dataset_records_with_sha256(path)
    return trips


def infer_dataset_split_role(dataset_id: str) -> str:
    tokens = {
        token
        for token in dataset_id.lower().replace("/", "-").replace("_", "-").split("-")
        if token
    }
    roles = set()
    if tokens & {"train", "training"}:
        roles.add("train")
    if tokens & {"validation", "valid", "val", "dev"}:
        roles.add("validation")
    if tokens & {"test", "testing"}:
        roles.add("test")
    if not roles:
        return "unspecified"
    if len(roles) != 1:
        raise RuntimeError(
            f"dataset_id {dataset_id!r} must encode exactly one split role"
        )
    return next(iter(roles))


def normalize_sha256(value: str, context: str) -> str:
    normalized = require_nonempty_string(value, context).lower()
    if normalized.startswith("sha256:"):
        normalized = normalized.removeprefix("sha256:")
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise RuntimeError(f"{context} must be a SHA-256 digest")
    return normalized


def load_dataset_manifest(
    path: Path,
    *,
    expected_role: str | None = None,
    expected_manifest_sha256: str | None = None,
    expected_records_sha256: str | None = None,
) -> DatasetArtifact:
    path = path.resolve()
    manifest_sha256 = sha256_file(path)
    if expected_manifest_sha256 is not None and manifest_sha256 != normalize_sha256(
        expected_manifest_sha256, "expected manifest hash"
    ):
        raise RuntimeError(
            f"dataset manifest hash {manifest_sha256} does not match its pinned hash"
        )
    descriptor = require_exact_keys(
        load_strict_json(path),
        {"schema", "dataset_id", "network_id", "records_schema", "records_file"},
        "dataset manifest",
    )
    if descriptor["schema"] != DATASET_MANIFEST_SCHEMA:
        raise RuntimeError(f"unsupported dataset manifest schema {descriptor['schema']!r}")
    if descriptor["records_schema"] != DATASET_RECORD_SCHEMA:
        raise RuntimeError(f"unsupported dataset record schema {descriptor['records_schema']!r}")
    records_file = safe_relative_file(descriptor["records_file"], "records_file")
    records_path = path.parent / records_file
    trips, records_sha256 = _load_dataset_records_with_sha256(records_path)
    if expected_records_sha256 is not None and records_sha256 != normalize_sha256(
        expected_records_sha256, "expected records hash"
    ):
        raise RuntimeError(
            f"dataset records hash {records_sha256} does not match its pinned hash"
        )
    dataset_id = require_nonempty_string(descriptor["dataset_id"], "dataset_id")
    split_role = infer_dataset_split_role(dataset_id)
    if expected_role is not None and split_role != expected_role:
        raise RuntimeError(
            f"dataset {dataset_id!r} has split role {split_role!r}, "
            f"expected {expected_role!r}"
        )
    manifest = DatasetManifest(
        schema=DATASET_MANIFEST_SCHEMA,
        dataset_id=dataset_id,
        network_id=require_nonempty_string(descriptor["network_id"], "network_id"),
        records_schema=DATASET_RECORD_SCHEMA,
        records_file=records_file,
        split_role=split_role,
    )
    return DatasetArtifact(
        manifest=manifest,
        manifest_path=path,
        manifest_sha256=manifest_sha256,
        records_path=records_path,
        records_sha256=records_sha256,
        trips=trips,
    )


@dataclass
class LineGraph:
    """Directed line graph whose states are unmodified raw-edge IDs."""

    tail: list[int]
    head: list[int]
    outgoing: list[list[int]]
    incoming: list[list[int]]
    identity: str

    @property
    def state_count(self) -> int:
        return len(self.tail)

    @property
    def transition_count(self) -> int:
        return sum(len(neighbors) for neighbors in self.outgoing)

    def validate(self) -> None:
        n = len(self.tail)
        if n == 0 or len(self.head) != n or len(self.outgoing) != n or len(self.incoming) != n:
            raise RuntimeError("line graph arrays have inconsistent non-zero lengths")
        for state, neighbors in enumerate(self.outgoing):
            if neighbors != sorted(set(neighbors)):
                raise RuntimeError(f"outgoing neighbors for state {state} are not canonical")
            for neighbor in neighbors:
                if not 0 <= neighbor < n:
                    raise RuntimeError(f"outgoing state {neighbor} is out of bounds")
                if self.head[state] != self.tail[neighbor]:
                    raise RuntimeError(f"illegal raw-edge transition {state}->{neighbor}")
                if state not in self.incoming[neighbor]:
                    raise RuntimeError("incoming and outgoing line-graph arrays disagree")


def graph_identity(tail: Sequence[int], head: Sequence[int]) -> str:
    digest = hashlib.sha256()
    digest.update(b"ewr.drncs-lg-line-graph/v1\0")
    for u, v in zip(tail, head):
        digest.update(int(u).to_bytes(8, "little", signed=True))
        digest.update(int(v).to_bytes(8, "little", signed=True))
    return f"sha256:{digest.hexdigest()}"


def build_line_graph(tail: Sequence[int], head: Sequence[int]) -> LineGraph:
    if len(tail) != len(head) or not tail:
        raise RuntimeError("raw road endpoint arrays must have the same non-zero length")
    tail_values = [int(value) for value in tail]
    head_values = [int(value) for value in head]
    edges_from_node: dict[int, list[int]] = defaultdict(list)
    for edge, node in enumerate(tail_values):
        edges_from_node[node].append(edge)
    for edges in edges_from_node.values():
        edges.sort()
    outgoing = [list(edges_from_node.get(node, ())) for node in head_values]
    incoming: list[list[int]] = [[] for _ in tail_values]
    for previous, neighbors in enumerate(outgoing):
        for following in neighbors:
            incoming[following].append(previous)
    for neighbors in incoming:
        neighbors.sort()
    graph = LineGraph(
        tail_values,
        head_values,
        outgoing,
        incoming,
        graph_identity(tail_values, head_values),
    )
    graph.validate()
    return graph


def read_road_endpoints(map_dir: Path) -> tuple[list[int], list[int], dict[str, Any]]:
    load_preprocess_dependencies()
    edges_path = map_dir / "edges.shp"
    if not edges_path.is_file():
        raise RuntimeError(f"missing road shapefile {edges_path}")
    try:
        reader = shapefile.Reader(str(edges_path))
    except Exception as error:
        raise RuntimeError(f"failed to read {edges_path}: {error}") from error
    fields = [field[0] for field in reader.fields[1:]]
    try:
        tail_index = fields.index("u")
        head_index = fields.index("v")
    except ValueError as error:
        raise RuntimeError("edges.shp must contain u and v fields") from error
    fid_index = fields.index("fid") if "fid" in fields else None
    tail: list[int] = []
    head: list[int] = []
    for record_index, record in enumerate(reader.iterRecords()):
        if fid_index is not None and int(record[fid_index]) != record_index:
            raise RuntimeError(
                f"edges.shp fid {record[fid_index]!r} does not equal record index {record_index}"
            )
        tail.append(int(record[tail_index]))
        head.append(int(record[head_index]))
    components = sorted(map_dir.glob("edges.*")) + sorted(map_dir.glob("nodes.*"))
    map_hash = hashlib.sha256()
    for component in components:
        if not component.is_file():
            continue
        map_hash.update(component.name.encode("utf-8"))
        map_hash.update(bytes.fromhex(sha256_file(component)))
    return tail, head, {
        "map_dir": str(map_dir),
        "map_components": [component.name for component in components if component.is_file()],
        "map_sha256": map_hash.hexdigest(),
        "edges": len(tail),
    }


def validate_trips(trips: Sequence[Trip], graph: LineGraph) -> None:
    for trip in trips:
        if len(trip.edges) < 2:
            raise RuntimeError(f"sample {trip.sample_id!r} has fewer than two roads")
        for index, edge in enumerate(trip.edges):
            if not 0 <= edge < graph.state_count:
                raise RuntimeError(f"sample {trip.sample_id!r} edge {edge} is out of bounds")
            if index and edge not in graph.outgoing[trip.edges[index - 1]]:
                raise RuntimeError(
                    f"sample {trip.sample_id!r} has discontinuous raw-edge transition "
                    f"{trip.edges[index - 1]}->{edge}"
                )


def validate_query_endpoints(trips: Sequence[Trip], graph: LineGraph) -> None:
    """Validate only fields that are query inputs, never truth interiors."""

    for trip in trips:
        for role, edge in (("source", trip.edges[0]), ("destination", trip.edges[-1])):
            if not 0 <= edge < graph.state_count:
                raise RuntimeError(
                    f"sample {trip.sample_id!r} {role} edge {edge} is out of bounds"
                )


def line_graph_to_plain(graph: LineGraph) -> dict[str, Any]:
    return {
        "tail": graph.tail,
        "head": graph.head,
        "outgoing": graph.outgoing,
        "incoming": graph.incoming,
        "identity": graph.identity,
    }


def line_graph_from_plain(value: Any) -> LineGraph:
    if not isinstance(value, dict):
        raise RuntimeError("internal line graph is not an object")
    graph = LineGraph(
        tail=[int(item) for item in value["tail"]],
        head=[int(item) for item in value["head"]],
        outgoing=[[int(item) for item in row] for row in value["outgoing"]],
        incoming=[[int(item) for item in row] for row in value["incoming"]],
        identity=str(value["identity"]),
    )
    graph.validate()
    if graph.identity != graph_identity(graph.tail, graph.head):
        raise RuntimeError("internal line graph identity does not match its endpoint arrays")
    return graph


@dataclass
class ContractionResult:
    active: list[bool]
    sparse_outgoing: list[list[int]]
    sparse_incoming: list[list[int]]
    order: list[int]
    shortcut_pairs: list[tuple[int, int]]
    requested_ratio: float
    contracted_nodes: int


def _shortcut_score(
    node: int, active: Sequence[bool], outgoing: Sequence[set[int]], incoming: Sequence[set[int]]
) -> tuple[int, list[tuple[int, int]]]:
    predecessors = [item for item in incoming[node] if active[item]]
    successors = [item for item in outgoing[node] if active[item]]
    pairs: list[tuple[int, int]] = []
    for previous in predecessors:
        for following in successors:
            if previous == following or following in outgoing[previous]:
                continue
            pairs.append((previous, following))
    # Paper definition: shortcuts added minus the original in/out edge count.
    return len(pairs) - len(predecessors) - len(successors), pairs


def contract_graph(outgoing_rows: Sequence[Sequence[int]], ratio: float) -> ContractionResult:
    """Apply deterministic shortcut-edge differential contraction.

    Ties are resolved by the raw state ID.  Scores are updated for the exact
    local neighborhood whose adjacency or potential shortcuts changed.
    """

    if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) or not 0 <= ratio < 1:
        raise RuntimeError("contraction ratio must be in [0, 1)")
    n = len(outgoing_rows)
    if n == 0:
        raise RuntimeError("cannot contract an empty graph")
    outgoing = [set(int(item) for item in row) for row in outgoing_rows]
    incoming: list[set[int]] = [set() for _ in range(n)]
    for previous, neighbors in enumerate(outgoing):
        for following in neighbors:
            if not 0 <= following < n:
                raise RuntimeError("contraction input neighbor is out of bounds")
            incoming[following].add(previous)
    active = [True] * n
    target = int(math.floor(n * float(ratio)))
    versions = [0] * n
    heap: list[tuple[int, int, int]] = []

    def refresh(node: int) -> None:
        if not active[node]:
            return
        versions[node] += 1
        score, _ = _shortcut_score(node, active, outgoing, incoming)
        heapq.heappush(heap, (score, node, versions[node]))

    for node in range(n):
        refresh(node)

    order: list[int] = []
    added_shortcuts: set[tuple[int, int]] = set()
    while len(order) < target:
        while heap:
            stored_score, node, version = heapq.heappop(heap)
            if active[node] and versions[node] == version:
                current_score, pairs = _shortcut_score(node, active, outgoing, incoming)
                if current_score != stored_score:
                    refresh(node)
                    continue
                break
        else:
            raise RuntimeError("contraction queue became empty before reaching its target")

        predecessors = {item for item in incoming[node] if active[item]}
        successors = {item for item in outgoing[node] if active[item]}
        newly_added: list[tuple[int, int]] = []
        for previous, following in sorted(pairs):
            if following not in outgoing[previous]:
                outgoing[previous].add(following)
                incoming[following].add(previous)
                added_shortcuts.add((previous, following))
                newly_added.append((previous, following))

        for previous in predecessors:
            outgoing[previous].discard(node)
        for following in successors:
            incoming[following].discard(node)
        outgoing[node].clear()
        incoming[node].clear()
        active[node] = False
        order.append(node)

        affected = set(predecessors) | set(successors)
        for previous, following in newly_added:
            affected.add(previous)
            affected.add(following)
            affected.update(outgoing[previous] & incoming[following])
        for affected_node in sorted(affected):
            refresh(affected_node)

    sparse_outgoing = [sorted(item for item in row if active[item]) if active[index] else [] for index, row in enumerate(outgoing)]
    sparse_incoming: list[list[int]] = [[] for _ in range(n)]
    for previous, neighbors in enumerate(sparse_outgoing):
        for following in neighbors:
            sparse_incoming[following].append(previous)
    shortcuts = sorted(
        pair
        for pair in added_shortcuts
        if active[pair[0]] and active[pair[1]] and pair[1] in outgoing[pair[0]]
    )
    return ContractionResult(
        active=active,
        sparse_outgoing=sparse_outgoing,
        sparse_incoming=sparse_incoming,
        order=order,
        shortcut_pairs=shortcuts,
        requested_ratio=float(ratio),
        contracted_nodes=len(order),
    )


def contraction_to_plain(value: ContractionResult) -> dict[str, Any]:
    return {
        "active": value.active,
        "sparse_outgoing": value.sparse_outgoing,
        "sparse_incoming": value.sparse_incoming,
        "order": value.order,
        "shortcut_pairs": [list(pair) for pair in value.shortcut_pairs],
        "requested_ratio": value.requested_ratio,
        "contracted_nodes": value.contracted_nodes,
    }


def contraction_from_plain(value: Any) -> ContractionResult:
    if not isinstance(value, dict):
        raise RuntimeError("internal contraction result is not an object")
    return ContractionResult(
        active=[bool(item) for item in value["active"]],
        sparse_outgoing=[[int(item) for item in row] for row in value["sparse_outgoing"]],
        sparse_incoming=[[int(item) for item in row] for row in value["sparse_incoming"]],
        order=[int(item) for item in value["order"]],
        shortcut_pairs=[(int(pair[0]), int(pair[1])) for pair in value["shortcut_pairs"]],
        requested_ratio=float(value["requested_ratio"]),
        contracted_nodes=int(value["contracted_nodes"]),
    )


def sparse_trajectory(edges: Sequence[int], active: Sequence[bool]) -> list[int]:
    return [edge for edge in edges if active[edge]]


class SparseTripView(Sequence[Trip]):
    """Lazy sparse-route view with only compact retained base indices."""

    __slots__ = ("_base", "_active", "_indices", "dropped_routes")

    def __init__(self, base: Sequence[Trip], active: Sequence[bool]) -> None:
        self._base = base
        self._active = active
        self._indices = array.array("I")
        dropped = 0
        for index, trip in enumerate(base):
            retained = sum(bool(active[edge]) for edge in trip.edges)
            if retained >= 2:
                self._indices.append(index)
            else:
                dropped += 1
        self.dropped_routes = dropped

    def __len__(self) -> int:
        return len(self._indices)

    @overload
    def __getitem__(self, index: int) -> Trip: ...

    @overload
    def __getitem__(self, index: slice) -> list[Trip]: ...

    def __getitem__(self, index: int | slice) -> Trip | list[Trip]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        trip = self._base[int(self._indices[index])]
        return Trip(trip.sample_id, sparse_trajectory(trip.edges, self._active))

    def __iter__(self) -> Iterator[Trip]:
        for base_index in self._indices:
            trip = self._base[int(base_index)]
            yield Trip(trip.sample_id, sparse_trajectory(trip.edges, self._active))

    @property
    def index_storage_bytes(self) -> int:
        return self._indices.buffer_info()[1] * self._indices.itemsize


def collect_historical_shortcuts(
    trips: Sequence[Trip], contraction: ContractionResult
) -> dict[tuple[int, int], dict[tuple[int, ...], int]]:
    """Collect unique train-only shortcut segments and their multiplicities.

    This compatibility helper is intended for small callers and tests.  Formal
    preprocessing uses the SQLite-backed implementation below so observations
    never accumulate as nested Python lists.
    """

    shortcut_pairs = set(contraction.shortcut_pairs)
    candidates: dict[tuple[int, int], dict[tuple[int, ...], int]] = defaultdict(dict)
    for trip in trips:
        retained = [index for index, edge in enumerate(trip.edges) if contraction.active[edge]]
        for left, right in zip(retained, retained[1:]):
            pair = (trip.edges[left], trip.edges[right])
            if pair in shortcut_pairs and right > left + 1:
                path = tuple(trip.edges[left : right + 1])
                candidates[pair][path] = candidates[pair].get(path, 0) + 1
    return {pair: dict(counts) for pair, counts in candidates.items()}


def path_precision(candidate: Sequence[int], reference: Sequence[int]) -> float:
    candidate_set = set(candidate)
    if not candidate_set:
        return 0.0
    return len(candidate_set & set(reference)) / len(candidate_set)


def select_sc1_path(paths: Sequence[Sequence[int]]) -> list[int]:
    """Select the exact paper Eq. (11) medoid, preserving multiplicity."""

    if not paths:
        raise RuntimeError("SC1 selection requires at least one historical path")
    counts: dict[tuple[int, ...], int] = {}
    for raw_path in paths:
        path = tuple(int(item) for item in raw_path)
        counts[path] = counts.get(path, 0) + 1
    return select_weighted_sc1_path(counts.items())


def select_weighted_sc1_path(
    path_counts: Iterable[tuple[Sequence[int], int]],
) -> list[int]:
    """Select the multiplicity-weighted precision medoid without O(U^2).

    For a candidate C, Eq. (11)'s numerator can be rearranged as
    ``sum(e in C, number of historical paths containing e)``.  This is exactly
    the same average pairwise set precision, including self-comparisons and
    duplicate historical paths, but requires two linear passes over the unique
    paths rather than materializing all observations or every path pair.
    """

    canonical_counts: list[tuple[tuple[int, ...], int]] = []
    edge_mass: dict[int, int] = defaultdict(int)
    total_observations = 0
    for raw_path, raw_count in path_counts:
        path = tuple(int(item) for item in raw_path)
        count = int(raw_count)
        if not path or count <= 0:
            raise RuntimeError("SC1 paths and multiplicities must be positive")
        canonical_counts.append((path, count))
        total_observations += count
        for edge in set(path):
            edge_mass[edge] += count
    if total_observations == 0:
        raise RuntimeError("SC1 selection requires at least one historical path")
    selected: tuple[int, ...] | None = None
    best_numerator = -1
    best_denominator = 1
    for path, _ in sorted(canonical_counts):
        candidate_edges = set(path)
        denominator = len(candidate_edges)
        numerator = sum(edge_mass[edge] for edge in candidate_edges)
        if (
            selected is None
            or numerator * best_denominator > best_numerator * denominator
            or (
                numerator * best_denominator == best_numerator * denominator
                and path < selected
            )
        ):
            selected = path
            best_numerator = numerator
            best_denominator = denominator
    if selected is None:
        raise RuntimeError("SC1 selection failed")
    return list(selected)


def encode_u32_path(path: Sequence[int]) -> bytes:
    values = array.array("I", (require_u32(item, "SC1 path edge") for item in path))
    if values.itemsize != 4:
        raise RuntimeError("SC1 encoding requires a 32-bit unsigned-int array type")
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def decode_u32_path(encoded: bytes) -> tuple[int, ...]:
    if not encoded or len(encoded) % 4:
        raise RuntimeError("SC1 path blob has invalid length")
    values = array.array("I")
    if values.itemsize != 4:
        raise RuntimeError("SC1 decoding requires a 32-bit unsigned-int array type")
    values.frombytes(encoded)
    if sys.byteorder != "little":
        values.byteswap()
    return tuple(int(item) for item in values)


def _flush_sc1_counts(
    connection: sqlite3.Connection,
    pending: dict[tuple[int, int, bytes], int],
) -> None:
    if not pending:
        return
    connection.executemany(
        """
        INSERT INTO sc1_paths(source_state, destination_state, path_blob, multiplicity)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(source_state, destination_state, path_blob)
        DO UPDATE SET multiplicity = multiplicity + excluded.multiplicity
        """,
        (
            (source, destination, path_blob, multiplicity)
            for (source, destination, path_blob), multiplicity in pending.items()
        ),
    )
    pending.clear()


def build_sc1_database_external(
    trips: Sequence[Trip],
    contraction: ContractionResult,
    database_path: Path,
    *,
    aggregation_buffer_entries: int = 8192,
) -> tuple[dict[tuple[int, int], list[int]], dict[str, Any]]:
    """Build the exact train-multiset SC1 database with bounded RAM."""

    if aggregation_buffer_entries <= 0:
        raise RuntimeError("SC1 aggregation buffer must be positive")
    if database_path.exists():
        raise RuntimeError(f"SC1 temporary database already exists: {database_path}")
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    shortcut_pairs = set(contraction.shortcut_pairs)
    observations = 0
    pending: dict[tuple[int, int, bytes], int] = {}
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute(
            """
            CREATE TABLE sc1_paths(
                source_state INTEGER NOT NULL,
                destination_state INTEGER NOT NULL,
                path_blob BLOB NOT NULL,
                multiplicity INTEGER NOT NULL,
                PRIMARY KEY(source_state, destination_state, path_blob)
            ) WITHOUT ROWID
            """
        )
        with connection:
            for trip in trips:
                retained = [
                    index
                    for index, edge in enumerate(trip.edges)
                    if contraction.active[edge]
                ]
                for left, right in zip(retained, retained[1:]):
                    pair = (trip.edges[left], trip.edges[right])
                    if pair not in shortcut_pairs or right <= left + 1:
                        continue
                    key = (
                        pair[0],
                        pair[1],
                        encode_u32_path(trip.edges[left : right + 1]),
                    )
                    pending[key] = pending.get(key, 0) + 1
                    observations += 1
                    if len(pending) >= aggregation_buffer_entries:
                        _flush_sc1_counts(connection, pending)
            _flush_sc1_counts(connection, pending)

        selected: dict[tuple[int, int], list[int]] = {}
        unique_segments = 0
        max_unique_segments_per_pair = 0
        pair_cursor = connection.execute(
            """
            SELECT source_state, destination_state, SUM(multiplicity), COUNT(*)
            FROM sc1_paths
            GROUP BY source_state, destination_state
            ORDER BY source_state, destination_state
            """
        )
        for source, destination, pair_observations, pair_unique in pair_cursor:
            edge_mass: dict[int, int] = defaultdict(int)
            rows = connection.execute(
                """
                SELECT path_blob, multiplicity FROM sc1_paths
                WHERE source_state = ? AND destination_state = ?
                ORDER BY path_blob
                """,
                (source, destination),
            )
            for path_blob, multiplicity in rows:
                for edge in set(decode_u32_path(path_blob)):
                    edge_mass[edge] += int(multiplicity)
            best_path: tuple[int, ...] | None = None
            best_numerator = -1
            best_denominator = 1
            rows = connection.execute(
                """
                SELECT path_blob FROM sc1_paths
                WHERE source_state = ? AND destination_state = ?
                ORDER BY path_blob
                """,
                (source, destination),
            )
            for (path_blob,) in rows:
                path = decode_u32_path(path_blob)
                candidate_edges = set(path)
                numerator = sum(edge_mass[edge] for edge in candidate_edges)
                denominator = len(candidate_edges)
                if (
                    best_path is None
                    or numerator * best_denominator > best_numerator * denominator
                    or (
                        numerator * best_denominator == best_numerator * denominator
                        and path < best_path
                    )
                ):
                    best_path = path
                    best_numerator = numerator
                    best_denominator = denominator
            if best_path is None or int(pair_observations) <= 0:
                raise RuntimeError("SC1 SQLite aggregation produced an empty pair")
            selected[(int(source), int(destination))] = list(best_path)
            unique_segments += int(pair_unique)
            max_unique_segments_per_pair = max(
                max_unique_segments_per_pair, int(pair_unique)
            )
        connection.commit()
        database_bytes = database_path.stat().st_size
        return selected, {
            "candidate_pairs": len(selected),
            "candidate_segments": observations,
            "unique_candidate_segments": unique_segments,
            "max_unique_candidate_segments_per_pair": max_unique_segments_per_pair,
            "selected_pairs": len(selected),
            "selection": "paper_eq11_multiplicity_weighted_average_precision_medoid",
            "aggregation": "bounded_sqlite_unique_path_counts",
            "aggregation_buffer_entries": aggregation_buffer_entries,
            "temporary_database_bytes": database_bytes,
        }
    finally:
        connection.close()


def build_sc1_database(
    trips: Sequence[Trip], contraction: ContractionResult
) -> tuple[dict[tuple[int, int], list[int]], dict[str, Any]]:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="drncs-lg-sc1-", suffix=".sqlite3"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        return build_sc1_database_external(trips, contraction, temporary)
    finally:
        for candidate in (
            temporary,
            Path(f"{temporary}-journal"),
            Path(f"{temporary}-shm"),
            Path(f"{temporary}-wal"),
        ):
            if candidate.exists():
                candidate.unlink()


def route_metrics(
    truth: Sequence[Sequence[int] | Trip],
    predicted: Sequence[Sequence[int]],
) -> dict[str, float | int]:
    if len(truth) != len(predicted) or not truth:
        raise RuntimeError("route metrics require equal, non-empty truth and prediction lists")
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    jaccards: list[float] = []
    exact = 0
    for expected_item, actual in zip(truth, predicted):
        expected = (
            expected_item.edges
            if isinstance(expected_item, Trip)
            else expected_item
        )
        expected_set = set(expected)
        actual_set = set(actual)
        intersection = len(expected_set & actual_set)
        precision = intersection / len(actual_set) if actual_set else 0.0
        recall = intersection / len(expected_set) if expected_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        union = len(expected_set | actual_set)
        jaccard = intersection / union if union else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        jaccards.append(jaccard)
        exact += list(expected) == list(actual)
    return {
        "samples": len(truth),
        "edge_precision": statistics.fmean(precisions),
        "edge_recall": statistics.fmean(recalls),
        "edge_f1": statistics.fmean(f1s),
        "exact_match": exact / len(truth),
        "edge_jaccard": statistics.fmean(jaccards),
    }


def write_prediction_rows(path: Path, trips: Sequence[Trip], predictions: Sequence[Sequence[int]]) -> None:
    if len(trips) != len(predictions):
        raise RuntimeError("prediction count does not match dataset manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for trip, predicted in zip(trips, predictions):
                if not predicted:
                    raise RuntimeError(f"prediction for {trip.sample_id!r} is empty")
                edges = [
                    require_u32(edge, f"prediction for {trip.sample_id!r}")
                    for edge in predicted
                ]
                json.dump(
                    {"sample_id": trip.sample_id, "predicted_edge_ids": edges},
                    stream,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def peak_rss_kib() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        value //= 1024
    return int(value)


def environment_info(device: str, workers: int) -> dict[str, str]:
    result = {
        "device": device,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "workers": str(workers),
    }
    if np is not None:
        result["numpy"] = str(np.__version__)
    if torch is not None:
        result["torch"] = str(torch.__version__)
    for distribution, key in (
        ("gensim", "gensim"),
        ("scipy", "scipy"),
        ("pyshp", "pyshp"),
    ):
        try:
            result[key] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[key] = "unavailable"
    return result


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


class Node2VecCorpus:
    """Re-iterable directed Node2Vec walk corpus.

    The official implementation uses the Node2Vec defaults p=q=1.  The
    general second-order transition is retained so the choice is explicit.
    """

    def __init__(
        self,
        outgoing: Sequence[Sequence[int]],
        *,
        walk_length: int,
        walks_per_state: int,
        seed: int,
        p: float = 1.0,
        q: float = 1.0,
    ) -> None:
        if walk_length <= 0 or walks_per_state <= 0 or p <= 0 or q <= 0:
            raise RuntimeError("Node2Vec walk parameters must be positive")
        self.outgoing = [tuple(int(item) for item in row) for row in outgoing]
        self.outgoing_sets = [set(row) for row in self.outgoing]
        self.walk_length = walk_length
        self.walks_per_state = walks_per_state
        self.seed = seed
        self.p = p
        self.q = q

    def __iter__(self) -> Iterator[list[str]]:
        rng = random.Random(self.seed)
        states = list(range(len(self.outgoing)))
        for _ in range(self.walks_per_state):
            rng.shuffle(states)
            for start in states:
                walk = [start]
                while len(walk) < self.walk_length and self.outgoing[walk[-1]]:
                    current = walk[-1]
                    candidates = self.outgoing[current]
                    if len(walk) == 1 or self.p == self.q == 1.0:
                        following = candidates[rng.randrange(len(candidates))]
                    else:
                        previous = walk[-2]
                        weights = []
                        for candidate in candidates:
                            if candidate == previous:
                                weights.append(1.0 / self.p)
                            elif candidate in self.outgoing_sets[previous]:
                                weights.append(1.0)
                            else:
                                weights.append(1.0 / self.q)
                        following = rng.choices(candidates, weights=weights, k=1)[0]
                    walk.append(following)
                yield [str(item) for item in walk]


def train_node2vec_embeddings(
    outgoing: Sequence[Sequence[int]],
    *,
    dimensions: int,
    walk_length: int,
    walks_per_state: int,
    window: int,
    workers: int,
    seed: int,
    epochs: int,
    batch_words: int,
) -> Any:
    load_node2vec_dependency()
    if dimensions <= 0 or window <= 0 or epochs <= 0 or batch_words <= 0:
        raise RuntimeError(
            "Node2Vec dimensions, window, epochs, and batch_words must be positive"
        )
    corpus = Node2VecCorpus(
        outgoing,
        walk_length=walk_length,
        walks_per_state=walks_per_state,
        seed=seed,
    )
    model = Word2Vec(
        sentences=corpus,
        vector_size=dimensions,
        window=window,
        min_count=1,
        sg=1,
        workers=workers,
        seed=seed,
        epochs=epochs,
        sorted_vocab=1,
        batch_words=batch_words,
        negative=5,
        hs=0,
        sample=1e-3,
        ns_exponent=0.75,
        shrink_windows=True,
    )
    values = np.empty((len(outgoing), dimensions), dtype=np.float32)
    for state in range(len(outgoing)):
        values[state] = model.wv[str(state)]
    return values


def make_transition_model(embedding_dimension: int, hidden_dimension: int) -> Any:
    load_array_dependencies()

    class TransitionMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(3 * embedding_dimension, hidden_dimension),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dimension, hidden_dimension),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dimension, 1),
                # The final ReLU is present in the released implementation.
                torch.nn.ReLU(),
            )

        def forward(self, values: Any) -> Any:
            return self.layers(values)

    return TransitionMLP()


def device_from_name(name: str) -> Any:
    load_array_dependencies()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name not in {"cpu", "cuda"}:
        raise RuntimeError("device must be auto, cpu, or cuda")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def candidate_logits(
    model: Any,
    embeddings: Any,
    current: Sequence[int],
    destinations: Sequence[int],
    outgoing: Sequence[Sequence[int]],
    device: Any,
) -> tuple[Any, Any, list[list[int]]]:
    """Return padded logits and mask for one transition per current state."""

    if len(current) != len(destinations):
        raise RuntimeError("current and destination batches differ in length")
    candidates = [list(outgoing[state]) for state in current]
    if any(not row for row in candidates):
        raise RuntimeError("candidate logits requested for a dead-end state")
    width = max(len(row) for row in candidates)
    rows: list[list[int]] = []
    mask_rows: list[list[bool]] = []
    for row in candidates:
        rows.append(row + [row[0]] * (width - len(row)))
        mask_rows.append([True] * len(row) + [False] * (width - len(row)))
    current_tensor = torch.as_tensor(current, dtype=torch.long)
    destination_tensor = torch.as_tensor(destinations, dtype=torch.long)
    candidate_tensor = torch.as_tensor(rows, dtype=torch.long)
    embedding_tensor = embeddings if torch.is_tensor(embeddings) else torch.from_numpy(embeddings)
    embedding_tensor = embedding_tensor.to(device)
    current_values = embedding_tensor[current_tensor.to(device)].unsqueeze(1).expand(-1, width, -1)
    destination_values = embedding_tensor[destination_tensor.to(device)].unsqueeze(1).expand(-1, width, -1)
    candidate_values = embedding_tensor[candidate_tensor.to(device)]
    # Released DRNCS ordering is (current, destination, candidate).
    inputs = torch.cat((current_values, destination_values, candidate_values), dim=-1)
    logits = model(inputs.reshape(-1, inputs.shape[-1])).reshape(len(current), width)
    mask = torch.as_tensor(mask_rows, dtype=torch.bool, device=device)
    logits = logits.masked_fill(~mask, float("-inf"))
    return logits, mask, candidates


def synchronize_device(device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def transition_minibatches(
    trips: Sequence[Trip],
    outgoing: Sequence[Sequence[int]],
    batch_size: int,
    rng: random.Random,
) -> Iterator[tuple[list[int], list[int], list[int], int]]:
    """Yield categorical transitions from shuffled route mini-batches."""

    if batch_size <= 0:
        raise RuntimeError("batch size must be positive")
    order = list(range(len(trips)))
    rng.shuffle(order)
    for offset in range(0, len(order), batch_size):
        route_indices = order[offset : offset + batch_size]
        current: list[int] = []
        destinations: list[int] = []
        targets: list[int] = []
        for trip_index in route_indices:
            edges = trips[trip_index].edges
            destination = edges[-1]
            for previous, following in zip(edges, edges[1:]):
                neighbors = outgoing[previous]
                try:
                    target = neighbors.index(following)
                except ValueError as error:
                    raise RuntimeError(
                        f"training sample {trips[trip_index].sample_id!r} contains "
                        f"a transition absent from its training graph: "
                        f"{previous}->{following}"
                    ) from error
                current.append(previous)
                destinations.append(destination)
                targets.append(target)
        yield current, destinations, targets, len(route_indices)


def greedy_paths(
    model: Any,
    embeddings: Any,
    trips: Sequence[Trip],
    outgoing: Sequence[Sequence[int]],
    device: Any,
    *,
    inference_batch_size: int,
    max_steps: int,
) -> tuple[list[list[int]], dict[str, int]]:
    """Autoregressively predict routes without consulting truth interiors.

    Only the first and last raw-edge IDs of each trip are used.  A dead end,
    cycle, or step limit returns the generated prefix; the endpoint is never
    appended and the prefix is never aligned to the observed route.
    """

    if inference_batch_size <= 0 or max_steps <= 0:
        raise RuntimeError("inference batch size and maximum steps must be positive")
    model.eval()
    routes = [[trip.edges[0]] for trip in trips]
    destinations = [trip.edges[-1] for trip in trips]
    visited = [{trip.edges[0]} for trip in trips]
    pending = {
        index
        for index, route in enumerate(routes)
        if route[-1] != destinations[index]
    }
    stats = {
        "queries": len(trips),
        "reached_destination": len(trips) - len(pending),
        "dead_end": 0,
        "cycle": 0,
        "step_limit": 0,
        "model_steps": 0,
    }
    embedding_tensor = (
        embeddings.to(device)
        if torch.is_tensor(embeddings)
        else torch.from_numpy(embeddings).to(device)
    )
    with torch.no_grad():
        for _ in range(max_steps):
            if not pending:
                break
            dead = [index for index in pending if not outgoing[routes[index][-1]]]
            for index in dead:
                pending.remove(index)
                stats["dead_end"] += 1
            active = sorted(pending)
            for offset in range(0, len(active), inference_batch_size):
                indices = active[offset : offset + inference_batch_size]
                current = [routes[index][-1] for index in indices]
                destination = [destinations[index] for index in indices]
                logits, _, candidates = candidate_logits(
                    model, embedding_tensor, current, destination, outgoing, device
                )
                choices = logits.argmax(dim=1).detach().cpu().tolist()
                for index, row, choice in zip(indices, candidates, choices):
                    following = row[int(choice)]
                    routes[index].append(following)
                    stats["model_steps"] += 1
                    if following == destinations[index]:
                        pending.remove(index)
                        stats["reached_destination"] += 1
                    elif following in visited[index]:
                        pending.remove(index)
                        stats["cycle"] += 1
                    else:
                        visited[index].add(following)
        for _ in pending:
            stats["step_limit"] += 1
    return routes, stats


def fit_transition_model(
    model: Any,
    embeddings: Any,
    outgoing: Sequence[Sequence[int]],
    train_trips: Sequence[Trip],
    validation_predictor: Any,
    validation_truth: Sequence[Sequence[int] | Trip],
    *,
    epochs: int,
    validation_every: int,
    batch_size: int,
    transition_chunk_size: int,
    learning_rate: float,
    seed: int,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    """Fit fixed-embedding DRNCS transition MLP and select on validation F1."""

    if (
        epochs <= 0
        or validation_every <= 0
        or transition_chunk_size <= 0
        or learning_rate <= 0
    ):
        raise RuntimeError("epochs, validation interval, and learning rate must be positive")
    if not train_trips or not validation_truth:
        raise RuntimeError("training and validation routes must be non-empty")
    model.to(device)
    embedding_tensor = torch.from_numpy(embeddings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_rank: tuple[float, float, int] | None = None
    total_updates = 0
    training_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_transitions = 0
        epoch_routes = 0
        for current, destinations, targets, route_count in transition_minibatches(
            train_trips, outgoing, batch_size, random.Random(seed + epoch)
        ):
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            count = len(current)
            for offset in range(0, count, transition_chunk_size):
                chunk = slice(offset, offset + transition_chunk_size)
                logits, _, _ = candidate_logits(
                    model,
                    embedding_tensor,
                    current[chunk],
                    destinations[chunk],
                    outgoing,
                    device,
                )
                target_tensor = torch.as_tensor(
                    targets[chunk], dtype=torch.long, device=device
                )
                loss_sum = torch.nn.functional.cross_entropy(
                    logits, target_tensor, reduction="sum"
                )
                # Paper Eq. (10) averages each route's summed transition NLL
                # over |D| routes.  The released script instead relies on
                # cross_entropy's transition mean.  DRNCS-LG intentionally
                # follows the paper and preserves the 512-route update boundary.
                (loss_sum / route_count).backward()
                batch_loss += float(loss_sum.detach().cpu())
            optimizer.step()
            epoch_loss += batch_loss
            epoch_transitions += count
            epoch_routes += route_count
            total_updates += 1
        if epoch_transitions == 0 or epoch_routes == 0:
            raise RuntimeError("training routes contain no transitions")
        should_validate = epoch % validation_every == 0 or epoch == epochs
        entry: dict[str, Any] = {
            "epoch": epoch,
            "mean_training_loss": epoch_loss / epoch_routes,
            "release_transition_mean_loss": epoch_loss / epoch_transitions,
            "training_routes": epoch_routes,
            "training_transitions": epoch_transitions,
            "loss_normalization": "paper_eq10_mean_route_summed_transition_nll",
        }
        if should_validate:
            synchronize_device(device)
            validation_started = time.perf_counter()
            predicted, prediction_stats = validation_predictor(model)
            synchronize_device(device)
            metrics = route_metrics(validation_truth, predicted)
            entry["validation"] = metrics
            entry["validation_prediction"] = prediction_stats
            entry["validation_seconds"] = time.perf_counter() - validation_started
            rank = (
                float(metrics["edge_f1"]),
                float(metrics["exact_match"]),
                -epoch,
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        history.append(entry)
    if best_state is None:
        raise RuntimeError("no validation checkpoint was selected")
    model.load_state_dict(best_state)
    model.to(device)
    return model, {
        "selected_epoch": best_epoch,
        "selected_rank": list(best_rank) if best_rank is not None else None,
        "optimizer_updates": total_updates,
        "loss_normalization": "paper_eq10_mean_route_summed_transition_nll",
        "release_difference": "release_uses_transition_mean_cross_entropy",
        "wall_seconds": time.perf_counter() - training_started,
        "history": history,
    }


def shortest_model_paths(
    model: Any,
    embeddings: Any,
    graph: LineGraph,
    sources: Sequence[int],
    destination: int,
    device: Any,
    *,
    score_batch_size: int,
) -> dict[int, list[int] | None]:
    """Find minimum model-NLL routes for one SC2 destination."""

    if score_batch_size <= 0:
        raise RuntimeError("SC2 score batch size must be positive")
    requested = sorted(set(int(source) for source in sources))
    if not requested:
        return {}
    model.eval()
    embedding_tensor = (
        embeddings.to(device)
        if torch.is_tensor(embeddings)
        else torch.from_numpy(embeddings).to(device)
    )
    weights: dict[int, dict[int, float]] = {}

    def ensure_weights(states: Sequence[int]) -> None:
        missing = sorted(
            {state for state in states if state not in weights and graph.outgoing[state]}
        )
        with torch.no_grad():
            for offset in range(0, len(missing), score_batch_size):
                current = missing[offset : offset + score_batch_size]
                destinations = [destination] * len(current)
                logits, _, candidates = candidate_logits(
                    model,
                    embedding_tensor,
                    current,
                    destinations,
                    graph.outgoing,
                    device,
                )
                nll = -torch.nn.functional.log_softmax(logits, dim=1)
                nll_rows = nll.detach().cpu().tolist()
                for state, row_candidates, row_weights in zip(
                    current, candidates, nll_rows
                ):
                    weights[state] = {
                        following: float(weight)
                        for following, weight in zip(row_candidates, row_weights)
                    }

    distances = {destination: 0.0}
    next_hop: dict[int, int] = {}
    heap: list[tuple[float, int]] = [(0.0, destination)]
    settled: set[int] = set()
    remaining = set(requested) - {destination}
    while heap:
        distance, state = heapq.heappop(heap)
        if state in settled or distance != distances.get(state):
            continue
        settled.add(state)
        remaining.discard(state)
        if not remaining:
            break
        predecessors = graph.incoming[state]
        ensure_weights(predecessors)
        for previous in predecessors:
            if previous in settled:
                continue
            candidate = distance + weights[previous][state]
            previous_distance = distances.get(previous)
            # Stable tie break: retain the smaller next raw-edge ID.
            if previous_distance is None or candidate < previous_distance or (
                candidate == previous_distance
                and state < next_hop.get(previous, UINT32_MAX)
            ):
                distances[previous] = candidate
                next_hop[previous] = state
                heapq.heappush(heap, (candidate, previous))
    result: dict[int, list[int] | None] = {}
    for source in requested:
        if source not in settled:
            result[source] = None
            continue
        path = [source]
        seen = {source}
        while path[-1] != destination:
            following = next_hop.get(path[-1])
            if following is None or following in seen:
                raise RuntimeError("SC2 shortest-path reconstruction is inconsistent")
            path.append(following)
            seen.add(following)
        result[source] = path
    return result


def shortest_model_path(
    model: Any,
    embeddings: Any,
    graph: LineGraph,
    source: int,
    destination: int,
    device: Any,
    *,
    score_batch_size: int,
) -> list[int] | None:
    """Convenience wrapper for one SC2 source/destination query."""

    return shortest_model_paths(
        model,
        embeddings,
        graph,
        [source],
        destination,
        device,
        score_batch_size=score_batch_size,
    )[source]


def build_sc2_database(
    model: Any,
    embeddings: Any,
    graph: LineGraph,
    contraction: ContractionResult,
    sc1: dict[tuple[int, int], list[int]],
    device: Any,
    *,
    score_batch_size: int,
) -> tuple[dict[tuple[int, int], list[int]], dict[str, Any]]:
    """Complete train-only shortcut storage with model-cost paths (SC2)."""

    missing = [pair for pair in contraction.shortcut_pairs if pair not in sc1]
    grouped: dict[int, list[int]] = defaultdict(list)
    for source, destination in missing:
        grouped[destination].append(source)
    database: dict[tuple[int, int], list[int]] = {}
    started = time.perf_counter()
    for destination in sorted(grouped):
        paths = shortest_model_paths(
            model,
            embeddings,
            graph,
            grouped[destination],
            destination,
            device,
            score_batch_size=score_batch_size,
        )
        for source in sorted(grouped[destination]):
            path = paths[source]
            if path is None:
                raise RuntimeError(
                    f"SC2 cannot connect shortcut {source}->{destination} on the original graph"
                )
            database[(source, destination)] = path
    return database, {
        "shortcut_pairs": len(database),
        "destinations": len(grouped),
        "wall_seconds": time.perf_counter() - started,
    }


def validate_expansion_path(
    path: Sequence[int], pair: tuple[int, int], graph: LineGraph
) -> None:
    if len(path) < 2 or path[0] != pair[0] or path[-1] != pair[1]:
        raise RuntimeError(f"shortcut expansion for {pair} has invalid endpoints")
    for previous, following in zip(path, path[1:]):
        if following not in graph.outgoing[previous]:
            raise RuntimeError(
                f"shortcut expansion for {pair} has illegal transition {previous}->{following}"
            )


def expand_sparse_path(
    skeleton: Sequence[int],
    graph: LineGraph,
    shortcuts: dict[tuple[int, int], list[int]],
) -> list[int]:
    if not skeleton:
        raise RuntimeError("cannot expand an empty sparse route")
    expanded = [int(skeleton[0])]
    for previous, following in zip(skeleton, skeleton[1:]):
        pair = (int(previous), int(following))
        if following in graph.outgoing[previous]:
            expanded.append(int(following))
            continue
        path = shortcuts.get(pair)
        if path is None:
            raise RuntimeError(f"missing SC1/SC2 expansion for shortcut {pair}")
        validate_expansion_path(path, pair, graph)
        expanded.extend(path[1:])
    return expanded


def generated_route_validity(
    trips: Sequence[Trip],
    predictions: Sequence[Sequence[int]],
    graph: LineGraph,
) -> dict[str, int]:
    if len(trips) != len(predictions):
        raise RuntimeError("generated route validation count mismatch")
    stats = {
        "queries": len(trips),
        "empty_routes": 0,
        "source_mismatches": 0,
        "destination_mismatches": 0,
        "routes_with_illegal_transitions": 0,
        "illegal_transition_count": 0,
        "non_simple_routes": 0,
        "repeated_state_events": 0,
    }
    for trip, route in zip(trips, predictions):
        if not route:
            stats["empty_routes"] += 1
            continue
        stats["source_mismatches"] += route[0] != trip.edges[0]
        stats["destination_mismatches"] += route[-1] != trip.edges[-1]
        illegal = 0
        for previous, following in zip(route, route[1:]):
            if (
                not 0 <= previous < graph.state_count
                or not 0 <= following < graph.state_count
                or following not in graph.outgoing[previous]
            ):
                illegal += 1
        if illegal:
            stats["routes_with_illegal_transitions"] += 1
            stats["illegal_transition_count"] += illegal
        unique_states = len(set(route))
        if unique_states != len(route):
            stats["non_simple_routes"] += 1
            stats["repeated_state_events"] += len(route) - unique_states
    if (
        stats["empty_routes"]
        or stats["source_mismatches"]
        or stats["routes_with_illegal_transitions"]
    ):
        raise RuntimeError(f"generated routes violate the raw-edge protocol: {stats}")
    return stats


def predict_dual_level(
    original_model: Any,
    sparse_model: Any,
    embeddings: Any,
    graph: LineGraph,
    contraction: ContractionResult,
    shortcuts: dict[tuple[int, int], list[int]],
    trips: Sequence[Trip],
    device: Any,
    *,
    inference_batch_size: int,
    max_steps: int,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Run sparse DRNCS when endpoints survive, otherwise original DRNCS."""

    sparse_indices = [
        index
        for index, trip in enumerate(trips)
        if contraction.active[trip.edges[0]] and contraction.active[trip.edges[-1]]
    ]
    sparse_set = set(sparse_indices)
    original_indices = [index for index in range(len(trips)) if index not in sparse_set]
    results: list[list[int] | None] = [None] * len(trips)
    original_stats = {
        "queries": 0,
        "reached_destination": 0,
        "dead_end": 0,
        "cycle": 0,
        "step_limit": 0,
        "model_steps": 0,
    }
    sparse_stats = dict(original_stats)
    if original_indices:
        selected = [trips[index] for index in original_indices]
        synchronize_device(device)
        original_started = time.perf_counter()
        generated, original_stats = greedy_paths(
            original_model,
            embeddings,
            selected,
            graph.outgoing,
            device,
            inference_batch_size=inference_batch_size,
            max_steps=max_steps,
        )
        synchronize_device(device)
        original_stats["wall_seconds"] = time.perf_counter() - original_started
        for index, route in zip(original_indices, generated):
            results[index] = route
    if sparse_indices:
        selected = [trips[index] for index in sparse_indices]
        synchronize_device(device)
        sparse_started = time.perf_counter()
        skeletons, sparse_stats = greedy_paths(
            sparse_model,
            embeddings,
            selected,
            contraction.sparse_outgoing,
            device,
            inference_batch_size=inference_batch_size,
            max_steps=max_steps,
        )
        synchronize_device(device)
        sparse_stats["wall_seconds"] = time.perf_counter() - sparse_started
        expansion_started = time.perf_counter()
        for index, skeleton in zip(sparse_indices, skeletons):
            results[index] = expand_sparse_path(skeleton, graph, shortcuts)
        expansion_seconds = time.perf_counter() - expansion_started
    else:
        expansion_seconds = 0.0
    if any(route is None for route in results):
        raise RuntimeError("dual-level prediction failed to fill all output positions")
    return [list(route) for route in results if route is not None], {
        "endpoint_fallback_queries": len(original_indices),
        "sparse_queries": len(sparse_indices),
        "original_rollout": original_stats,
        "sparse_rollout": sparse_stats,
        "shortcut_expansion_seconds": expansion_seconds,
    }


def dataset_binding(dataset: DatasetArtifact) -> dict[str, Any]:
    return {
        "dataset_id": dataset.manifest.dataset_id,
        "network_id": dataset.manifest.network_id,
        "split_role": dataset.manifest.split_role,
        "manifest_sha256": dataset.manifest_sha256,
        "records_sha256": dataset.records_sha256,
        "samples": len(dataset.trips),
        "edge_occurrences": dataset.trips.edge_occurrences,
        "route_storage": "contiguous_uint32_edges_uint64_offsets",
        "compact_numeric_storage_bytes": dataset.trips.storage_bytes,
    }


def require_same_dataset(actual: DatasetArtifact, expected: Any, context: str) -> None:
    if dataset_binding(actual) != expected:
        raise RuntimeError(f"{context} does not match the dataset bound to preprocessing")


def require_disjoint_samples(left: DatasetArtifact, right: DatasetArtifact) -> None:
    left_ids = set(left.trips.iter_sample_ids())
    for sample_id in right.trips.iter_sample_ids():
        if sample_id in left_ids:
            raise RuntimeError(
                "training and validation sample IDs overlap "
                f"(for example {sample_id!r})"
            )


def write_torch_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_torch_artifact(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, pickle.PickleError) as error:
        raise RuntimeError(f"failed to load DRNCS-LG artifact {path}: {error}") from error


def state_dict_on_cpu(model: Any) -> dict[str, Any]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def preprocess_command(args: argparse.Namespace) -> None:
    total_started = time.perf_counter()
    workers = configure_runtime(args.seed, args.workers)
    load_preprocess_dependencies()
    configure_runtime(args.seed, workers)
    source = adapter_source_identity(getattr(args, "source_revision", None))
    if not 0 <= args.contraction_ratio < 1:
        raise RuntimeError("contraction ratio must be in [0, 1)")
    graph_started = time.perf_counter()
    tail, head, map_info = read_road_endpoints(args.map_dir.resolve())
    graph = build_line_graph(tail, head)
    graph_seconds = time.perf_counter() - graph_started

    embedding_started = time.perf_counter()
    embeddings = train_node2vec_embeddings(
        graph.outgoing,
        dimensions=args.embedding_dimension,
        walk_length=args.walk_length,
        walks_per_state=args.walks_per_state,
        window=args.window,
        workers=workers,
        seed=args.seed,
        epochs=args.node2vec_epochs,
        batch_words=args.node2vec_batch_words,
    )
    embedding_seconds = time.perf_counter() - embedding_started

    contraction_started = time.perf_counter()
    contraction = contract_graph(graph.outgoing, args.contraction_ratio)
    contraction_seconds = time.perf_counter() - contraction_started

    # Route observations are not needed for Node2Vec or contraction.  Delaying
    # this compact load avoids retaining the full training split throughout the
    # multi-billion-token embedding stage.
    train_data_started = time.perf_counter()
    train_dataset = load_dataset_manifest(
        args.train_manifest,
        expected_role="train",
        expected_manifest_sha256=getattr(
            args, "expected_train_manifest_sha256", None
        ),
        expected_records_sha256=getattr(
            args, "expected_train_records_sha256", None
        ),
    )
    validate_trips(train_dataset.trips, graph)
    train_data_seconds = time.perf_counter() - train_data_started

    sc1_started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".drncs-lg-sc1-", suffix=".sqlite3", dir=args.output_dir
    )
    os.close(descriptor)
    sc1_database = Path(temporary_name)
    sc1_database.unlink()
    try:
        sc1, sc1_stats = build_sc1_database_external(
            train_dataset.trips,
            contraction,
            sc1_database,
            aggregation_buffer_entries=args.sc1_aggregation_buffer_entries,
        )
    finally:
        for candidate in (
            sc1_database,
            Path(f"{sc1_database}-journal"),
            Path(f"{sc1_database}-shm"),
            Path(f"{sc1_database}-wal"),
        ):
            if candidate.exists():
                candidate.unlink()
    for pair, path in sc1.items():
        validate_expansion_path(path, pair, graph)
    sc1_seconds = time.perf_counter() - sc1_started

    configuration = {
        "seed": args.seed,
        "workers": workers,
        "embedding_dimension": args.embedding_dimension,
        "walk_length": args.walk_length,
        "walks_per_state": args.walks_per_state,
        "window": args.window,
        "node2vec_epochs": args.node2vec_epochs,
        "node2vec_batch_words": args.node2vec_batch_words,
        "node2vec_p": 1.0,
        "node2vec_q": 1.0,
        "node2vec_architecture": "skip_gram_negative_sampling",
        "node2vec_negative_samples": 5,
        "node2vec_hierarchical_softmax": False,
        "node2vec_subsampling_threshold": 0.001,
        "node2vec_negative_sampling_exponent": 0.75,
        "node2vec_shrink_windows": True,
        "node2vec_corpus_storage": "reiterable_stream_not_materialized_walk_list",
        "contraction_ratio": args.contraction_ratio,
        "contraction_score": "shortcuts_added_minus_active_indegree_outdegree",
        "contraction_tie_break": "ascending_raw_edge_id",
        "state_space": "directed_line_graph_raw_edge_id",
        "sc1_selection": "paper_eq11_multiplicity_weighted_average_precision_medoid",
        "sc1_aggregation": "bounded_sqlite_unique_path_counts",
        "sc1_aggregation_buffer_entries": args.sc1_aggregation_buffer_entries,
        "train_split_role_enforced": True,
        "train_manifest_hash_pin_enforced": getattr(
            args, "expected_train_manifest_sha256", None
        )
        is not None,
        "train_records_hash_pin_enforced": getattr(
            args, "expected_train_records_sha256", None
        )
        is not None,
        "train_dataset_hash_pins_enforced": all(
            getattr(args, name, None) is not None
            for name in (
                "expected_train_manifest_sha256",
                "expected_train_records_sha256",
            )
        ),
    }
    artifact = {
        "schema": PREPROCESS_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "audited_upstream_commit": AUDITED_UPSTREAM_COMMIT,
        "source": source,
        "train_dataset": dataset_binding(train_dataset),
        "map": map_info,
        "graph": line_graph_to_plain(graph),
        "contraction": contraction_to_plain(contraction),
        "embeddings": embeddings,
        "sc1": sc1,
        "configuration": configuration,
    }
    artifact_path = args.output_dir / "preprocess.pkl"
    write_pickle_atomic(artifact_path, artifact)
    metadata = {
        "schema": PREPROCESS_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "source": source,
        "artifact": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "train_manifest": str(train_dataset.manifest_path),
        "train_dataset": dataset_binding(train_dataset),
        "map": map_info,
        "graph_identity": graph.identity,
        "states": graph.state_count,
        "transitions": graph.transition_count,
        "contracted_states": contraction.contracted_nodes,
        "surviving_states": sum(contraction.active),
        "final_shortcuts": len(contraction.shortcut_pairs),
        "sc1": sc1_stats,
        "configuration": configuration,
        "timing": {
            "data_and_graph_seconds": graph_seconds + train_data_seconds,
            "map_and_graph_seconds": graph_seconds,
            "train_data_load_and_validation_seconds": train_data_seconds,
            "node2vec_seconds": embedding_seconds,
            "contraction_seconds": contraction_seconds,
            "sc1_seconds": sc1_seconds,
            "total_process_seconds": time.perf_counter() - total_started,
        },
        "peak_rss_kib": peak_rss_kib(),
        "environment": environment_info("cpu", workers),
    }
    metadata_path = args.output_dir / "preprocess.json"
    write_json_atomic(metadata_path, metadata)
    print(json.dumps(metadata, indent=2))


def load_preprocess_directory(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = load_strict_json(path / "preprocess.json")
    if metadata.get("schema") != PREPROCESS_SCHEMA:
        raise RuntimeError("unsupported DRNCS-LG preprocessing metadata")
    artifact_path = path / "preprocess.pkl"
    expected_hash = require_nonempty_string(
        metadata.get("artifact_sha256"), "preprocess artifact hash"
    )
    if sha256_file(artifact_path) != expected_hash:
        raise RuntimeError("preprocess.pkl hash does not match preprocess.json")
    artifact = load_pickle(artifact_path)
    if not isinstance(artifact, dict) or artifact.get("schema") != PREPROCESS_SCHEMA:
        raise RuntimeError("unsupported DRNCS-LG preprocessing artifact")
    if metadata.get("source") != artifact.get("source"):
        raise RuntimeError("preprocessing metadata and artifact source identities differ")
    require_same_adapter_source(metadata.get("source"), "preprocessing metadata")
    return artifact, metadata


def train_command(args: argparse.Namespace) -> None:
    total_started = time.perf_counter()
    workers = configure_runtime(args.seed, args.workers)
    load_array_dependencies()
    configure_runtime(args.seed, workers)
    device = device_from_name(args.device)
    source = adapter_source_identity(getattr(args, "source_revision", None))
    artifact, preprocess_metadata = load_preprocess_directory(args.preprocess_dir)
    require_same_adapter_source(artifact.get("source"), "preprocessing artifact")
    if source != artifact.get("source"):
        raise RuntimeError("training source identity differs from preprocessing")
    train_dataset = load_dataset_manifest(
        args.train_manifest,
        expected_role="train",
        expected_manifest_sha256=getattr(
            args, "expected_train_manifest_sha256", None
        ),
        expected_records_sha256=getattr(
            args, "expected_train_records_sha256", None
        ),
    )
    validation_dataset = load_dataset_manifest(
        args.validation_manifest,
        expected_role="validation",
        expected_manifest_sha256=getattr(
            args, "expected_validation_manifest_sha256", None
        ),
        expected_records_sha256=getattr(
            args, "expected_validation_records_sha256", None
        ),
    )
    require_same_dataset(train_dataset, artifact["train_dataset"], "training manifest")
    if validation_dataset.manifest.network_id != train_dataset.manifest.network_id:
        raise RuntimeError("training and validation manifests use different networks")
    require_disjoint_samples(train_dataset, validation_dataset)
    graph_plain = artifact.pop("graph")
    graph = line_graph_from_plain(graph_plain)
    del graph_plain
    contraction_plain = artifact.pop("contraction")
    contraction = contraction_from_plain(contraction_plain)
    del contraction_plain
    validate_trips(train_dataset.trips, graph)
    validate_trips(validation_dataset.trips, graph)
    embeddings = np.asarray(artifact.pop("embeddings"), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != graph.state_count:
        raise RuntimeError("preprocessed embeddings have an invalid shape")
    raw_sc1 = artifact.pop("sc1")
    if not isinstance(raw_sc1, dict):
        raise RuntimeError("preprocessing SC1 artifact is not a dictionary")
    sc1: dict[tuple[int, int], list[int]] = raw_sc1
    artifact_map = artifact.pop("map")
    artifact.clear()

    original_model = make_transition_model(embeddings.shape[1], args.hidden_dimension)

    def original_validation(model: Any) -> tuple[list[list[int]], dict[str, Any]]:
        return greedy_paths(
            model,
            embeddings,
            validation_dataset.trips,
            graph.outgoing,
            device,
            inference_batch_size=args.inference_batch_size,
            max_steps=args.max_steps,
        )

    original_model, original_training = fit_transition_model(
        original_model,
        embeddings,
        graph.outgoing,
        train_dataset.trips,
        original_validation,
        validation_dataset.trips,
        epochs=args.epochs,
        validation_every=args.validation_every,
        batch_size=args.batch_size,
        transition_chunk_size=args.transition_chunk_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=device,
    )

    sc2, sc2_stats = build_sc2_database(
        original_model,
        embeddings,
        graph,
        contraction,
        sc1,
        device,
        score_batch_size=args.sc2_score_batch_size,
    )
    sc1_count = len(sc1)
    overlap = next((pair for pair in sc2 if pair in sc1), None)
    if overlap is not None:
        raise RuntimeError(
            f"SC1 and SC2 shortcut databases unexpectedly overlap at {overlap}"
        )
    # SC1 can be the largest retained Python object after preprocessing.  It
    # is no longer needed separately, so extend it in place rather than briefly
    # cloning every path and key while constructing the final database.
    shortcuts = sc1
    shortcuts.update(sc2)
    if len(shortcuts) != len(contraction.shortcut_pairs) or any(
        pair not in shortcuts for pair in contraction.shortcut_pairs
    ):
        raise RuntimeError("SC1/SC2 do not cover every final sparse shortcut")
    for pair, path in shortcuts.items():
        validate_expansion_path(path, pair, graph)

    sparse_train_trips = SparseTripView(train_dataset.trips, contraction.active)
    dropped_sparse_routes = sparse_train_trips.dropped_routes
    if not sparse_train_trips:
        raise RuntimeError("contraction removed every sparse training transition")
    torch.manual_seed(args.seed + 1)
    sparse_model = make_transition_model(embeddings.shape[1], args.hidden_dimension)

    def sparse_validation(model: Any) -> tuple[list[list[int]], dict[str, Any]]:
        return predict_dual_level(
            original_model,
            model,
            embeddings,
            graph,
            contraction,
            shortcuts,
            validation_dataset.trips,
            device,
            inference_batch_size=args.inference_batch_size,
            max_steps=args.max_steps,
        )

    sparse_model, sparse_training = fit_transition_model(
        sparse_model,
        embeddings,
        contraction.sparse_outgoing,
        sparse_train_trips,
        sparse_validation,
        validation_dataset.trips,
        epochs=args.epochs,
        validation_every=args.validation_every,
        batch_size=args.batch_size,
        transition_chunk_size=args.transition_chunk_size,
        learning_rate=args.learning_rate,
        seed=args.seed + 1,
        device=device,
    )

    configuration = {
        "seed": args.seed,
        "workers": workers,
        "device": str(device),
        "epochs": args.epochs,
        "validation_every": args.validation_every,
        "batch_size_routes": args.batch_size,
        "transition_chunk_size": args.transition_chunk_size,
        "learning_rate": args.learning_rate,
        "hidden_dimension": args.hidden_dimension,
        "inference_batch_size": args.inference_batch_size,
        "max_steps": args.max_steps,
        "sc2_score_batch_size": args.sc2_score_batch_size,
        "checkpoint_selection": "validation_macro_edge_f1_then_exact_match_then_earliest_epoch",
        "loss_normalization": "paper_eq10_mean_route_summed_transition_nll",
        "release_loss_difference": "release_cross_entropy_uses_transition_mean",
        "train_split_role_enforced": True,
        "validation_split_role_enforced": True,
        "train_manifest_hash_pin_enforced": getattr(
            args, "expected_train_manifest_sha256", None
        )
        is not None,
        "train_records_hash_pin_enforced": getattr(
            args, "expected_train_records_sha256", None
        )
        is not None,
        "train_dataset_hash_pins_enforced": all(
            getattr(args, name, None) is not None
            for name in (
                "expected_train_manifest_sha256",
                "expected_train_records_sha256",
            )
        ),
        "validation_manifest_hash_pin_enforced": getattr(
            args, "expected_validation_manifest_sha256", None
        )
        is not None,
        "validation_records_hash_pin_enforced": getattr(
            args, "expected_validation_records_sha256", None
        )
        is not None,
        "validation_dataset_hash_pins_enforced": all(
            getattr(args, name, None) is not None
            for name in (
                "expected_validation_manifest_sha256",
                "expected_validation_records_sha256",
            )
        ),
        "sparse_training_storage": "lazy_view_with_uint32_base_route_indices",
    }
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "audited_upstream_commit": AUDITED_UPSTREAM_COMMIT,
        "source": source,
        "train_dataset": dataset_binding(train_dataset),
        "validation_dataset": dataset_binding(validation_dataset),
        "map": artifact_map,
        "graph": line_graph_to_plain(graph),
        "contraction": contraction_to_plain(contraction),
        "embeddings": embeddings,
        "shortcuts": shortcuts,
        "embedding_dimension": int(embeddings.shape[1]),
        "hidden_dimension": args.hidden_dimension,
        "original_model_state": state_dict_on_cpu(original_model),
        "sparse_model_state": state_dict_on_cpu(sparse_model),
        "original_selected_epoch": original_training["selected_epoch"],
        "sparse_selected_epoch": sparse_training["selected_epoch"],
        "preprocess_artifact_sha256": preprocess_metadata["artifact_sha256"],
        "configuration": configuration,
    }
    checkpoint_path = args.output_dir / "checkpoint.pt"
    write_torch_atomic(checkpoint_path, checkpoint)
    diagnostics = {
        "schema": TRAINING_DIAGNOSTICS_SCHEMA,
        "method": METHOD_NAME,
        "adapter_version": ADAPTER_VERSION,
        "source": source,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "preprocess_dir": str(args.preprocess_dir),
        "preprocess_artifact_sha256": preprocess_metadata["artifact_sha256"],
        "train_manifest": str(train_dataset.manifest_path),
        "train_dataset": dataset_binding(train_dataset),
        "validation_manifest": str(validation_dataset.manifest_path),
        "validation_dataset": dataset_binding(validation_dataset),
        "split_roles_read": [
            train_dataset.manifest.split_role,
            validation_dataset.manifest.split_role,
        ],
        "test_data_read": any(
            dataset.manifest.split_role == "test"
            for dataset in (train_dataset, validation_dataset)
        ),
        "graph_identity": graph.identity,
        "configuration": configuration,
        "original_model": original_training,
        "sc2": sc2_stats,
        "sparse_model": sparse_training,
        "shortcut_storage": {
            "final_shortcuts": len(contraction.shortcut_pairs),
            "sc1_train_historical": sc1_count,
            "sc2_model_cost": len(sc2),
        },
        "sparse_training_routes": len(sparse_train_trips),
        "sparse_training_routes_dropped_below_two_states": dropped_sparse_routes,
        "sparse_training_index_storage_bytes": sparse_train_trips.index_storage_bytes,
        "total_process_seconds": time.perf_counter() - total_started,
        "peak_rss_kib": peak_rss_kib(),
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "environment": environment_info(str(device), workers),
    }
    diagnostics_path = args.output_dir / "training_diagnostics.json"
    write_json_atomic(diagnostics_path, diagnostics)
    print(json.dumps(diagnostics, indent=2))


def load_checkpoint_models(
    path: Path, device: Any
) -> tuple[
    dict[str, Any],
    LineGraph,
    ContractionResult,
    Any,
    dict[tuple[int, int], list[int]],
    Any,
    Any,
]:
    checkpoint = load_torch_artifact(path)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("unsupported DRNCS-LG checkpoint")
    require_same_adapter_source(checkpoint.get("source"), "training checkpoint")
    graph_plain = checkpoint.pop("graph")
    graph = line_graph_from_plain(graph_plain)
    del graph_plain
    contraction_plain = checkpoint.pop("contraction")
    contraction = contraction_from_plain(contraction_plain)
    del contraction_plain
    embeddings = np.asarray(checkpoint.pop("embeddings"), dtype=np.float32)
    raw_shortcuts = checkpoint.pop("shortcuts")
    if not isinstance(raw_shortcuts, dict):
        raise RuntimeError("checkpoint shortcut artifact is not a dictionary")
    shortcuts: dict[tuple[int, int], list[int]] = raw_shortcuts
    original = make_transition_model(
        int(checkpoint["embedding_dimension"]), int(checkpoint["hidden_dimension"])
    )
    sparse = make_transition_model(
        int(checkpoint["embedding_dimension"]), int(checkpoint["hidden_dimension"])
    )
    original.load_state_dict(checkpoint.pop("original_model_state"))
    sparse.load_state_dict(checkpoint.pop("sparse_model_state"))
    original.to(device).eval()
    sparse.to(device).eval()
    return checkpoint, graph, contraction, embeddings, shortcuts, original, sparse


def predict_command(args: argparse.Namespace) -> None:
    if args.warmup_repetitions < 0 or args.measured_repetitions <= 0:
        raise RuntimeError(
            "warm-up repetitions must be nonnegative and measured repetitions positive"
        )
    if args.latency_samples < 0:
        raise RuntimeError("latency sample count must be nonnegative")
    if not args.source_revision.strip():
        raise RuntimeError("source revision must not be empty")
    total_started = time.perf_counter()
    workers = configure_runtime(args.seed, args.workers)
    load_array_dependencies()
    configure_runtime(args.seed, workers)
    device = device_from_name(args.device)
    source = adapter_source_identity(args.source_revision)
    data_started = time.perf_counter()
    dataset = load_dataset_manifest(
        args.dataset_manifest,
        expected_manifest_sha256=getattr(
            args, "expected_dataset_manifest_sha256", None
        ),
        expected_records_sha256=getattr(
            args, "expected_dataset_records_sha256", None
        ),
    )
    if dataset.manifest.split_role not in {"validation", "test"}:
        raise RuntimeError("prediction accepts only validation or test split manifests")
    tail, head, map_info = read_road_endpoints(args.map_dir.resolve())
    observed_graph = build_line_graph(tail, head)
    validate_query_endpoints(dataset.trips, observed_graph)
    data_seconds = time.perf_counter() - data_started
    model_started = time.perf_counter()
    (
        checkpoint,
        graph,
        contraction,
        embeddings,
        shortcuts,
        original_model,
        sparse_model,
    ) = load_checkpoint_models(args.checkpoint, device)
    if source != checkpoint.get("source"):
        raise RuntimeError("prediction source identity differs from checkpoint")
    if observed_graph.identity != graph.identity:
        raise RuntimeError("prediction map does not match the checkpoint line graph")
    if map_info["map_sha256"] != checkpoint["map"]["map_sha256"]:
        raise RuntimeError("prediction map files do not match the checkpoint map hash")
    if dataset.manifest.network_id != checkpoint["train_dataset"]["network_id"]:
        raise RuntimeError("prediction dataset network does not match the checkpoint")
    model_seconds = time.perf_counter() - model_started

    def run(selected: Sequence[Trip]) -> tuple[list[list[int]], dict[str, Any]]:
        return predict_dual_level(
            original_model,
            sparse_model,
            embeddings,
            graph,
            contraction,
            shortcuts,
            selected,
            device,
            inference_batch_size=args.inference_batch_size,
            max_steps=args.max_steps,
        )

    warmup_seconds: list[float] = []
    for _ in range(args.warmup_repetitions):
        synchronize_device(device)
        started = time.perf_counter()
        run(dataset.trips)
        synchronize_device(device)
        warmup_seconds.append(time.perf_counter() - started)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    measured_seconds: list[float] = []
    component_stats: list[dict[str, Any]] = []
    generated: list[list[int]] | None = None
    for repetition in range(args.measured_repetitions):
        synchronize_device(device)
        started = time.perf_counter()
        candidate, components = run(dataset.trips)
        synchronize_device(device)
        measured_seconds.append(time.perf_counter() - started)
        component_stats.append(components)
        if generated is None:
            generated = candidate
        elif generated != candidate:
            raise RuntimeError(
                f"prediction repetition {repetition} produced different raw-edge routes"
            )
    if generated is None:
        raise RuntimeError("no measured prediction was generated")
    route_validity = generated_route_validity(dataset.trips, generated, graph)
    endpoint_failures = route_validity["destination_mismatches"]
    latency_seconds: list[float] = []
    for trip in dataset.trips[: args.latency_samples]:
        synchronize_device(device)
        started = time.perf_counter()
        run([trip])
        synchronize_device(device)
        latency_seconds.append(time.perf_counter() - started)
    write_prediction_rows(args.predictions, dataset.trips, generated)
    prediction_seconds = statistics.fmean(measured_seconds)
    diagnostics = {
        "schema": PREDICTION_DIAGNOSTICS_SCHEMA,
        "method": METHOD_NAME,
        "adapter_version": ADAPTER_VERSION,
        "audited_upstream_commit": AUDITED_UPSTREAM_COMMIT,
        "source": source,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "dataset_manifest": str(dataset.manifest_path),
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "dataset_id": dataset.manifest.dataset_id,
        "dataset_split_role": dataset.manifest.split_role,
        "network_id": dataset.manifest.network_id,
        "samples": len(dataset.trips),
        "query_protocol": "fixed_true_first_raw_edge_to_true_last_raw_edge",
        "endpoint_repair": False,
        "truth_interior_used_for_route_generation": False,
        "manifest_hash_pin_enforced": getattr(
            args, "expected_dataset_manifest_sha256", None
        )
        is not None,
        "records_hash_pin_enforced": getattr(
            args, "expected_dataset_records_sha256", None
        )
        is not None,
        "dataset_hash_pins_enforced": all(
            getattr(args, name, None) is not None
            for name in (
                "expected_dataset_manifest_sha256",
                "expected_dataset_records_sha256",
            )
        ),
        "endpoint_failures": endpoint_failures,
        "generated_route_validity": route_validity,
        "timing": {
            "data_and_graph_seconds": data_seconds,
            "model_load_seconds": model_seconds,
            "warmup_repetition_seconds": warmup_seconds,
            "prediction_repetition_seconds": measured_seconds,
            "mean_prediction_seconds": prediction_seconds,
            "mean_seconds_per_query": prediction_seconds / len(dataset.trips),
            "queries_per_second": len(dataset.trips) / prediction_seconds,
            "single_query_latency_samples": len(latency_seconds),
            "single_query_latency_p50_seconds": percentile(latency_seconds, 0.50),
            "single_query_latency_p95_seconds": percentile(latency_seconds, 0.95),
            "single_query_latency_max_seconds": max(latency_seconds, default=None),
            "component_stats_per_repetition": component_stats,
            "total_process_seconds": time.perf_counter() - total_started,
        },
        "warmup_repetitions": args.warmup_repetitions,
        "measured_repetitions": args.measured_repetitions,
        "seed": args.seed,
        "workers": workers,
        "inference_batch_size": args.inference_batch_size,
        "max_steps": args.max_steps,
        "peak_rss_kib": peak_rss_kib(),
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "environment": environment_info(str(device), workers),
    }
    receipt = {
        "schema": RUN_RECEIPT_SCHEMA,
        "method": {"name": METHOD_NAME, "version": ADAPTER_VERSION},
        "dataset_id": dataset.manifest.dataset_id,
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "prediction_records_schema": PREDICTION_RECORD_SCHEMA,
        "configuration": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "network_id": dataset.manifest.network_id,
            "graph_identity": graph.identity,
            "query_protocol": "fixed_true_first_raw_edge_to_true_last_raw_edge",
            "state_space": "directed_line_graph_raw_edge_id",
            "endpoint_repair": False,
            "truth_interior_used_for_route_generation": False,
            "dataset_split_role": dataset.manifest.split_role,
            "manifest_hash_pin_enforced": getattr(
                args, "expected_dataset_manifest_sha256", None
            )
            is not None,
            "records_hash_pin_enforced": getattr(
                args, "expected_dataset_records_sha256", None
            )
            is not None,
            "dataset_hash_pins_enforced": all(
                getattr(args, name, None) is not None
                for name in (
                    "expected_dataset_manifest_sha256",
                    "expected_dataset_records_sha256",
                )
            ),
            "seed": args.seed,
            "workers": workers,
            "inference_batch_size": args.inference_batch_size,
            "max_steps": args.max_steps,
            "warmup_repetitions": args.warmup_repetitions,
            "measured_repetitions": args.measured_repetitions,
            "audited_upstream_commit": AUDITED_UPSTREAM_COMMIT,
            "adapter_sha256": source["adapter_sha256"],
        },
        "source_revision": args.source_revision,
        "environment": environment_info(str(device), workers),
    }
    write_json_atomic(args.diagnostics, diagnostics)
    write_json_atomic(args.run_receipt, receipt)
    print(json.dumps(receipt, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean-room DRNCS-LG baseline adapter"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preprocess = subparsers.add_parser("preprocess")
    preprocess.add_argument("--train-manifest", type=Path, required=True)
    preprocess.add_argument("--map-dir", type=Path, required=True)
    preprocess.add_argument("--output-dir", type=Path, required=True)
    preprocess.add_argument("--source-revision")
    preprocess.add_argument("--expected-train-manifest-sha256")
    preprocess.add_argument("--expected-train-records-sha256")
    preprocess.add_argument("--seed", type=int, default=20260716)
    preprocess.add_argument("--workers", type=int, default=16)
    preprocess.add_argument("--embedding-dimension", type=int, default=64)
    preprocess.add_argument("--walk-length", type=int, default=30)
    preprocess.add_argument("--walks-per-state", type=int, default=200)
    preprocess.add_argument("--window", type=int, default=10)
    preprocess.add_argument("--node2vec-epochs", type=int, default=5)
    preprocess.add_argument("--node2vec-batch-words", type=int, default=4)
    preprocess.add_argument("--contraction-ratio", type=float, default=0.5)
    preprocess.add_argument("--sc1-aggregation-buffer-entries", type=int, default=8192)

    train = subparsers.add_parser("train")
    train.add_argument("--preprocess-dir", type=Path, required=True)
    train.add_argument("--train-manifest", type=Path, required=True)
    train.add_argument("--validation-manifest", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--source-revision")
    train.add_argument("--expected-train-manifest-sha256")
    train.add_argument("--expected-train-records-sha256")
    train.add_argument("--expected-validation-manifest-sha256")
    train.add_argument("--expected-validation-records-sha256")
    train.add_argument("--seed", type=int, default=20260716)
    train.add_argument("--workers", type=int, default=16)
    train.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    train.add_argument("--epochs", type=int, default=200)
    train.add_argument("--validation-every", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--transition-chunk-size", type=int, default=8192)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--hidden-dimension", type=int, default=128)
    train.add_argument("--inference-batch-size", type=int, default=1000)
    train.add_argument("--max-steps", type=int, default=1000)
    train.add_argument("--sc2-score-batch-size", type=int, default=4096)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--map-dir", type=Path, required=True)
    predict.add_argument("--dataset-manifest", type=Path, required=True)
    predict.add_argument("--predictions", type=Path, required=True)
    predict.add_argument("--run-receipt", type=Path, required=True)
    predict.add_argument("--diagnostics", type=Path, required=True)
    predict.add_argument("--source-revision", required=True)
    predict.add_argument("--expected-dataset-manifest-sha256")
    predict.add_argument("--expected-dataset-records-sha256")
    predict.add_argument("--seed", type=int, default=20260716)
    predict.add_argument("--workers", type=int, default=16)
    predict.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    predict.add_argument("--inference-batch-size", type=int, default=1000)
    predict.add_argument("--max-steps", type=int, default=1000)
    predict.add_argument("--warmup-repetitions", type=int, default=1)
    predict.add_argument("--measured-repetitions", type=int, default=5)
    predict.add_argument("--latency-samples", type=int, default=0)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "preprocess":
        preprocess_command(args)
    elif args.command == "train":
        train_command(args)
    else:
        predict_command(args)


if __name__ == "__main__":
    main()
