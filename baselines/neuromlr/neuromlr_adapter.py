#!/usr/bin/env python3
"""NeuroMLR adapter for the version-one EWR research protocol.

The model classes are imported from the pinned official checkout. This driver
replaces upstream data plumbing, deterministic seeding, checkpointing, and
artifact emission. Dataset and prediction artifacts use only original road IDs;
method diagnostics never leak into the shared prediction interchange format.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import importlib
import json
import math
import os
import platform
import random
import resource
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

UPSTREAM_COMMIT = "c45e3b5811e5a59b36e4682307d2196c02dac360"
UPSTREAM_TREE = "318d28140208ae3cfa538ed8e0fd3adcb023dae7"
UPSTREAM_FILE_SHA256 = {
    "model_all.py": "091d18e83c618a3dcc3676c4fb63d68737bd3d872f45732547298fa995e6c2af",
    "models_general.py": "180af33516fc21c7572b20db95367246a35d1017fe194d6848de261b59e172d5",
    "train.py": "b031fc272c2c776dd4bcd77539599b8fa80b1d2c0bbc555fcd140f7cb08cb1a0",
    "eval.py": "8e78946ef9b1b0602939c09695e8610604edc944f95a005728d04820266fd19c",
    "utils.py": "c4f54ae923e993340232ccd1a21f6ce5e711dcf127f460f70a843e89dcb04017",
    "requirements.txt": "aa4e4644cb56597c25c4dede0dc3cbe65c1d2c5421da94b678f65be8e2639503",
}
MAX_GENERATION_STEPS = 300
DATASET_MANIFEST_SCHEMA = "ewr.dataset-manifest/v1"
DATASET_RECORD_SCHEMA = "ewr.dataset-record/v1"
PREDICTION_RECORD_SCHEMA = "ewr.prediction-record/v1"
RUN_RECEIPT_SCHEMA = "ewr.run-receipt/v1"
PREDICTION_PROGRESS_SCHEMA = "ewr.neuromlr-prediction-progress/v1"
PREDICTION_RESUME_BINDING_SCHEMA = "ewr.neuromlr-prediction-resume-binding/v1"
ADAPTER_VERSION = "0.2.0"

# Loaded only after CLI parsing, so protocol validation and `--help` do not
# require the heavyweight model environment.
np: Any = None
shapefile: Any = None
torch: Any = None
F: Any = None
sparse: Any = None
sparse_dijkstra: Any = None
Data: Any = None
_MODEL_DEPENDENCY_ERROR: ImportError | None = None


def load_model_dependencies() -> None:
    global Data, F, _MODEL_DEPENDENCY_ERROR, np, shapefile, sparse
    global sparse_dijkstra, torch
    if np is not None:
        return
    if _MODEL_DEPENDENCY_ERROR is not None:
        raise RuntimeError(
            f"NeuroMLR model dependencies are unavailable: {_MODEL_DEPENDENCY_ERROR}"
        ) from _MODEL_DEPENDENCY_ERROR
    try:
        import numpy as numpy_module
        import shapefile as shapefile_module
        import torch as torch_module
        import torch.nn.functional as functional_module
        from scipy import sparse as sparse_module
        from scipy.sparse.csgraph import dijkstra as dijkstra_function
        from torch_geometric.data import Data as data_class
    except ImportError as error:
        _MODEL_DEPENDENCY_ERROR = error
        raise RuntimeError(
            f"NeuroMLR model dependencies are unavailable: {error}"
        ) from error
    np = numpy_module
    shapefile = shapefile_module
    torch = torch_module
    F = functional_module
    sparse = sparse_module
    sparse_dijkstra = dijkstra_function
    Data = data_class


@dataclass
class RoadGraph:
    tail: np.ndarray
    head: np.ndarray
    x: np.ndarray
    y: np.ndarray
    osmids: np.ndarray
    neighbors: list[list[int]]
    padded_neighbors: list[list[int]]
    max_neighbors: int
    edge_mapping: dict[int, tuple[int, int]]
    edge_index: torch.Tensor
    identity: str
    coordinate_identity: str


@dataclass
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
    records_path: Path
    records_sha256: str
    trips: list[Trip]


@dataclass(frozen=True)
class RepeatedDijkstraResult:
    generated: list[list[int]]
    warmup_seconds: list[float]
    measured_seconds: list[float]
    component_totals_per_repetition: list[dict[str, float]]


@dataclass(frozen=True)
class ChunkedGreedyResult:
    prediction_seconds: float
    chunk_seconds: list[float]
    endpoint_failures: int
    resumed_chunks: int
    completed_chunks: int
    output_sha256: str
    progress_path: Path
    resume_dir: Path
    peak_rss_kib: int
    peak_cuda_memory_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    add_common_arguments(train)
    train.add_argument("--train-manifest", type=Path, required=True)
    train.add_argument("--validation-manifest", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--validation-every", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--max-steps-per-epoch", type=int)

    predict = subparsers.add_parser("predict")
    add_common_arguments(predict)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--dataset-manifest", type=Path, required=True)
    predict.add_argument("--method", choices=["dijkstra", "greedy"], required=True)
    predict.add_argument("--predictions", type=Path, required=True)
    predict.add_argument("--run-receipt", type=Path, required=True)
    predict.add_argument("--diagnostics", type=Path, required=True)
    predict.add_argument("--source-revision", required=True)
    predict.add_argument("--score-edge-chunk", type=int, default=4096)
    predict.add_argument("--warmup-repetitions", type=int, default=0)
    predict.add_argument("--measured-repetitions", type=int, default=1)
    predict.add_argument(
        "--route-chunk-size",
        type=int,
        default=0,
        help="bounded Greedy route batch size; zero preserves legacy full-batch prediction",
    )
    predict.add_argument(
        "--resume",
        choices=["auto", "never", "require"],
        default="auto",
        help="recovery policy when --route-chunk-size is positive",
    )
    predict.add_argument(
        "--resume-dir",
        type=Path,
        help="atomic Greedy prediction shards (default: <predictions>.resume)",
    )
    predict.add_argument(
        "--progress",
        type=Path,
        help="atomic progress JSON (default: <resume-dir>/progress.json)",
    )
    return parser.parse_args()


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--mlp-hidden-layers", type=int, default=3)
    parser.add_argument("--gnn-layers", type=int, default=1)


def main() -> None:
    args = parse_args()
    load_model_dependencies()
    verify_upstream(args.upstream_dir)
    seed_everything(args.seed)
    if args.command == "train":
        train(args)
    else:
        predict(args)


def verify_upstream(upstream_dir: Path) -> None:
    actual = subprocess.run(
        ["git", "-C", str(upstream_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != UPSTREAM_COMMIT:
        raise RuntimeError(f"NeuroMLR checkout is {actual}, expected {UPSTREAM_COMMIT}")
    tree = subprocess.run(
        ["git", "-C", str(upstream_dir), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tree != UPSTREAM_TREE:
        raise RuntimeError(f"NeuroMLR tree is {tree}, expected {UPSTREAM_TREE}")
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(upstream_dir),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("pinned NeuroMLR checkout has modified tracked files")
    required = [*UPSTREAM_FILE_SHA256, "README.md"]
    missing = [name for name in required if not (upstream_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"pinned checkout lacks {missing}")
    actual_file_hashes = {
        name: sha256_file(upstream_dir / name) for name in UPSTREAM_FILE_SHA256
    }
    mismatched = {
        name: {"actual": actual_file_hashes[name], "expected": expected}
        for name, expected in UPSTREAM_FILE_SHA256.items()
        if actual_file_hashes[name] != expected
    }
    if mismatched:
        raise RuntimeError(f"pinned NeuroMLR files have unexpected hashes: {mismatched}")
    sys.path.insert(0, str(upstream_dir.resolve()))


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def field_indices(reader: shapefile.Reader) -> dict[str, int]:
    return {field[0]: index for index, field in enumerate(reader.fields[1:])}


def load_road_graph(map_dir: Path) -> RoadGraph:
    nodes_reader = shapefile.Reader(str(map_dir / "nodes.shp"))
    node_fields = field_indices(nodes_reader)
    node_records = nodes_reader.records()
    osmids = np.asarray([int(row[node_fields["osmid"]]) for row in node_records], dtype=np.int64)
    x = np.asarray([float(row[node_fields["x"]]) for row in node_records], dtype=np.float64)
    y = np.asarray([float(row[node_fields["y"]]) for row in node_records], dtype=np.float64)
    node_index = {int(osmid): index for index, osmid in enumerate(osmids)}

    edges_reader = shapefile.Reader(str(map_dir / "edges.shp"))
    edge_fields = field_indices(edges_reader)
    edge_records = edges_reader.records()
    tail = np.asarray(
        [node_index[int(row[edge_fields["u"]])] for row in edge_records], dtype=np.int64
    )
    head = np.asarray(
        [node_index[int(row[edge_fields["v"]])] for row in edge_records], dtype=np.int64
    )
    outgoing: list[list[int]] = [[] for _ in range(len(osmids))]
    for edge, node in enumerate(tail):
        outgoing[int(node)].append(edge)
    for edges in outgoing:
        edges.sort()
    neighbors = [outgoing[int(node)].copy() for node in head]
    max_neighbors = max(len(edges) for edges in neighbors)
    padded = [edges + [-1] * (max_neighbors - len(edges)) for edges in neighbors]
    mapping = {edge: (int(tail[edge]), int(head[edge])) for edge in range(len(tail))}
    mapping[-1] = (-1, -1)
    edge_index = torch.from_numpy(np.stack((tail, head))).long()
    identity_hash = hashlib.sha256()
    for values in (tail, head):
        identity_hash.update(values.astype("<u8", copy=False).tobytes())
    identity = identity_hash.hexdigest()
    coordinate_hash = hashlib.sha256()
    for values in (x, y):
        coordinate_hash.update(values.astype("<f8", copy=False).tobytes())
    coordinate_identity = coordinate_hash.hexdigest()
    return RoadGraph(
        tail=tail,
        head=head,
        x=x,
        y=y,
        osmids=osmids,
        neighbors=neighbors,
        padded_neighbors=padded,
        max_neighbors=max_neighbors,
        edge_mapping=mapping,
        edge_index=edge_index,
        identity=identity,
        coordinate_identity=coordinate_identity,
    )


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RuntimeError(f"duplicate JSON field {key!r}")
        value[key] = item
    return value


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
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise RuntimeError(f"{kind} fields differ: missing={missing}, unknown={unknown}")
    return value


def require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{field} must be a nonempty string")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_dataset_manifest(path: Path) -> DatasetArtifact:
    manifest_bytes = path.read_bytes()
    descriptor = require_exact_fields(
        decode_json(manifest_bytes.decode("utf-8"), "dataset manifest JSON"),
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
        raise RuntimeError(
            f"unsupported dataset manifest schema {manifest.schema!r}; "
            f"expected {DATASET_MANIFEST_SCHEMA!r}"
        )
    if manifest.records_schema != DATASET_RECORD_SCHEMA:
        raise RuntimeError(
            f"unsupported dataset records schema {manifest.records_schema!r}; "
            f"expected {DATASET_RECORD_SCHEMA!r}"
        )
    records_path = Path(manifest.records_file)
    if records_path.is_absolute() or ".." in records_path.parts:
        raise RuntimeError("records_file must be a safe path relative to its manifest")
    records_path = path.parent / records_path
    records_sha256_before = sha256_file(records_path)
    trips = load_dataset_records(records_path)
    records_sha256_after = sha256_file(records_path)
    if records_sha256_before != records_sha256_after:
        raise RuntimeError("dataset records changed while they were being loaded")
    return DatasetArtifact(
        manifest=manifest,
        manifest_path=path.resolve(),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        records_path=records_path.resolve(),
        records_sha256=records_sha256_after,
        trips=trips,
    )


def load_dataset_records(path: Path) -> list[Trip]:
    trips: list[Trip] = []
    ids: set[str] = set()
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
            if sample_id in ids:
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
                or edge > 0xFFFF_FFFF
                for edge in edges
            ):
                raise RuntimeError(
                    f"sample {sample_id!r} has a non-u32 original edge ID"
                )
            ids.add(sample_id)
            trips.append(Trip(sample_id=sample_id, edges=edges.copy()))
    if not trips:
        raise RuntimeError(f"empty dataset records file {path}")
    return trips


def validate_trips(trips: Iterable[Trip], graph: RoadGraph, minimum_edges: int = 2) -> None:
    for trip in trips:
        if len(trip.edges) < minimum_edges:
            raise RuntimeError(f"{trip.sample_id} has fewer than {minimum_edges} roads")
        if any(edge < 0 or edge >= len(graph.tail) for edge in trip.edges):
            raise RuntimeError(f"{trip.sample_id} has an unrepresentable road")
        if any(
            graph.head[left] != graph.tail[right]
            for left, right in zip(trip.edges, trip.edges[1:])
        ):
            raise RuntimeError(f"{trip.sample_id} is discontinuous")
        nodes = [int(graph.tail[trip.edges[0]])] + [int(graph.head[e]) for e in trip.edges]
        if len(nodes) != len(set(nodes)):
            raise RuntimeError(f"{trip.sample_id} violates the common cycle policy")


def model_arguments(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        num_layers=args.mlp_hidden_layers,
        gnn="GCN",
        gnn_layers=args.gnn_layers,
        trainable_embeddings=True,
        traffic=False,
        attention=False,
        num_heads=2,
    )


def haversine_edge_weights(graph: RoadGraph) -> np.ndarray:
    lat1 = np.radians(graph.y[graph.tail])
    lat2 = np.radians(graph.y[graph.head])
    dlat = lat2 - lat1
    dlon = np.radians(graph.x[graph.head] - graph.x[graph.tail])
    value = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.minimum(value, 1.0)))


def reverse_sparse_graph_with_minimum_parallel_edges(
    graph: RoadGraph, weights: np.ndarray
) -> sparse.csr_matrix:
    reverse_minimum: dict[tuple[int, int], float] = {}
    for source, target, weight in zip(graph.head, graph.tail, weights):
        key = (int(source), int(target))
        reverse_minimum[key] = min(reverse_minimum.get(key, math.inf), float(weight))
    rows = np.fromiter((key[0] for key in reverse_minimum), dtype=np.int64)
    columns = np.fromiter((key[1] for key in reverse_minimum), dtype=np.int64)
    values = np.fromiter(reverse_minimum.values(), dtype=np.float64)
    return sparse.csr_matrix(
        (values, (rows, columns)), shape=(len(graph.x), len(graph.x))
    )


def lipschitz_embeddings(
    graph: RoadGraph, dimensions: int, seed: int
) -> tuple[torch.Tensor, list[int]]:
    if dimensions > len(graph.x):
        raise RuntimeError("embedding size exceeds map node count")
    anchors = random.Random(seed).sample(range(len(graph.x)), dimensions)
    weights = haversine_edge_weights(graph)
    # NetworkX Dijkstra on an OSM MultiDiGraph uses the least-cost parallel
    # edge. scipy.sparse would otherwise sum duplicate COO entries.
    reverse = reverse_sparse_graph_with_minimum_parallel_edges(graph, weights)
    distances = sparse_dijkstra(reverse, directed=True, indices=np.asarray(anchors))
    embeddings = np.zeros_like(distances, dtype=np.float64)
    finite = np.isfinite(distances)
    embeddings[finite] = 1.0 / (distances[finite] + 1.0)
    embeddings = embeddings.T
    standard_deviation = embeddings.std(axis=0)
    if np.any(standard_deviation == 0):
        raise RuntimeError("Lipschitz embedding has a zero-variance anchor")
    embeddings = (embeddings - embeddings.mean(axis=0)) / standard_deviation
    return torch.from_numpy(embeddings.astype(np.float32)), anchors


def build_model(
    args: argparse.Namespace,
    graph: RoadGraph,
    device: torch.device,
    embeddings: torch.Tensor | None,
):
    model_module = importlib.import_module("model_all")
    torch_graph = Data(x=embeddings, edge_index=graph.edge_index).to(device)
    model = model_module.Model(
        num_nodes=len(graph.x),
        graph=torch_graph,
        device=device,
        args=model_arguments(args),
        embeddings=embeddings,
        mapping=graph.edge_mapping,
        traffic_matrix=None,
    ).to(device)
    return model


def training_batch(
    trips: list[Trip], graph: RoadGraph, batch_size: int
) -> tuple[list[int], list[int], list[int], torch.Tensor, int]:
    sampled = random.sample(trips, batch_size)
    current_roads: list[int] = []
    destinations: list[int] = []
    candidate_roads: list[int] = []
    true_classes: list[int] = []
    predictions = 0
    for trip in sampled:
        destination = trip.edges[-1]
        for current, next_road in zip(trip.edges, trip.edges[1:]):
            neighbors = graph.padded_neighbors[current]
            current_roads.extend([current] * graph.max_neighbors)
            destinations.extend([destination] * graph.max_neighbors)
            candidate_roads.extend(neighbors)
            true_classes.append(neighbors.index(next_road))
            predictions += 1
    return current_roads, destinations, candidate_roads, torch.tensor(true_classes), predictions


def train(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.validation_every <= 0 or args.batch_size <= 0:
        raise RuntimeError("epochs, validation cadence, and batch size must be positive")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    total_started = time.perf_counter()
    graph = load_road_graph(args.map_dir)
    train_dataset = load_dataset_manifest(args.train_manifest)
    validation_dataset = load_dataset_manifest(args.validation_manifest)
    if train_dataset.manifest.network_id != validation_dataset.manifest.network_id:
        raise RuntimeError("training and validation manifests refer to different networks")
    train_trips = train_dataset.trips
    validation_trips = validation_dataset.trips
    validation_ids = {trip.sample_id for trip in validation_trips}
    if any(trip.sample_id in validation_ids for trip in train_trips):
        raise RuntimeError("training and validation manifests contain overlapping sample IDs")
    validate_trips(train_trips, graph)
    validate_trips(validation_trips, graph)
    if len(train_trips) < args.batch_size:
        raise RuntimeError("training set is smaller than one upstream batch")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    embeddings, anchors = lipschitz_embeddings(graph, args.embedding_size, args.seed)
    model = build_model(args, graph, device, embeddings)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    loss_function = torch.nn.CrossEntropyLoss(reduction="sum")
    steps_per_epoch = math.ceil(len(train_trips) / args.batch_size)
    if args.max_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    configuration = {
        "schema_version": 1,
        "upstream_commit": UPSTREAM_COMMIT,
        "seed": args.seed,
        "device": str(device),
        "network_id": train_dataset.manifest.network_id,
        "train_dataset_id": train_dataset.manifest.dataset_id,
        "train_manifest_sha256": train_dataset.manifest_sha256,
        "validation_dataset_id": validation_dataset.manifest.dataset_id,
        "validation_manifest_sha256": validation_dataset.manifest_sha256,
        "train_records": len(train_trips),
        "validation_records": len(validation_trips),
        "epochs": args.epochs,
        "validation_every": args.validation_every,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "steps_per_epoch": steps_per_epoch,
        "model": vars(model_arguments(args)),
        "lipschitz_anchor_nodes": anchors,
        "lipschitz_parallel_edge_reduction": "minimum_haversine_like_networkx_multigraph_dijkstra",
        "traffic": False,
        "raw_edge_identity_mapping": True,
        "common_cycle_policy_pre_applied": True,
        "graph_identity": graph.identity,
    }
    write_json(output_dir / "configuration.json", configuration)
    log_path = output_dir / "training.jsonl"
    log_file = log_path.open("w", buffering=1)
    log_file.write(json.dumps({"event": "configuration", **configuration}) + "\n")
    training_started = time.perf_counter()
    evaluations: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        loss_sum = 0.0
        target_count = 0
        for _ in range(steps_per_epoch):
            current, destinations, candidates, classes, predictions = training_batch(
                train_trips, graph, args.batch_size
            )
            logits = model(current, destinations, candidates, None).reshape(
                -1, graph.max_neighbors
            )
            loss = loss_function(logits, classes.to(device))
            optimizer.zero_grad(set_to_none=True)
            (loss / args.batch_size).backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            target_count += predictions
        event = {
            "event": "epoch",
            "epoch": epoch,
            "loss_sum": loss_sum,
            "mean_loss_per_transition": loss_sum / target_count,
            "transition_targets": target_count,
            "expected_L_minus_1_targets": True,
            "epoch_seconds": time.perf_counter() - epoch_started,
            "peak_rss_kib": peak_rss_kib(),
        }
        log_file.write(json.dumps(event) + "\n")
        if epoch % args.validation_every == 0 or epoch == args.epochs:
            validation_started = time.perf_counter()
            generated = greedy_paths(model, validation_trips, graph, device)
            metrics = route_metrics([trip.edges for trip in validation_trips], generated)
            checkpoint_path = output_dir / f"checkpoint-epoch-{epoch}.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                args,
                graph,
                train_dataset.manifest.network_id,
            )
            evaluation = {
                "epoch": epoch,
                "method": "neuromlr_greedy",
                "metrics": metrics,
                "seconds": time.perf_counter() - validation_started,
                "checkpoint": str(checkpoint_path),
            }
            evaluations.append(evaluation)
            log_file.write(json.dumps({"event": "validation", **evaluation}) + "\n")
    training_seconds = time.perf_counter() - training_started
    selected = max(
        evaluations,
        key=lambda row: (
            row["metrics"]["edge_f1"],
            row["metrics"]["exact_match"],
            -row["epoch"],
        ),
    )
    selection = {
        "schema_version": 1,
        "selection_rule": ["maximum_edge_f1", "maximum_exact_match", "earliest_epoch"],
        "selected": selected,
        "evaluations": evaluations,
        "training_seconds": training_seconds,
        "total_seconds": time.perf_counter() - total_started,
        "peak_rss_kib": peak_rss_kib(),
    }
    write_json(output_dir / "selection.json", selection)
    log_file.write(json.dumps({"event": "finished", **selection}) + "\n")
    log_file.close()
    print(json.dumps(selection["selected"], indent=2))


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    args: argparse.Namespace,
    graph: RoadGraph,
    network_id: str,
) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(
        {
            "schema_version": 1,
            "upstream_commit": UPSTREAM_COMMIT,
            "epoch": epoch,
            "seed": args.seed,
            "graph_identity": graph.identity,
            "network_id": network_id,
            "model_configuration": vars(model_arguments(args)),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        temporary,
    )
    os.replace(temporary, path)


def load_checkpoint_model(
    args: argparse.Namespace, graph: RoadGraph, device: torch.device, network_id: str
):
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint["upstream_commit"] != UPSTREAM_COMMIT:
        raise RuntimeError("checkpoint upstream commit mismatch")
    if checkpoint["graph_identity"] != graph.identity:
        raise RuntimeError("checkpoint graph identity mismatch")
    if checkpoint["network_id"] != network_id:
        raise RuntimeError("checkpoint network identity mismatch")
    expected = vars(model_arguments(args))
    if checkpoint["model_configuration"] != expected:
        raise RuntimeError("checkpoint model configuration differs from command")
    model = build_model(args, graph, device, torch.zeros((len(graph.x), args.embedding_size)))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def greedy_paths(
    model, trips: list[Trip], graph: RoadGraph, device: torch.device
) -> list[list[int]]:
    generated = [[trip.edges[0]] for trip in trips]
    pending = list(range(len(trips)))
    model.eval()
    with torch.no_grad():
        for _ in range(MAX_GENERATION_STEPS):
            if not pending:
                break
            current_rows = [generated[index][-1] for index in pending]
            current = [road for road in current_rows for _ in range(graph.max_neighbors)]
            candidates = [
                neighbor
                for road in current_rows
                for neighbor in graph.padded_neighbors[road]
            ]
            destinations = [
                trips[index].edges[-1]
                for index in pending
                for _ in range(graph.max_neighbors)
            ]
            logits = model(current, destinations, candidates, None).reshape(
                -1, graph.max_neighbors
            )
            choices = torch.argmax(logits, dim=1).detach().cpu().tolist()
            next_pending: list[int] = []
            for index, choice in zip(pending, choices):
                next_road = graph.padded_neighbors[generated[index][-1]][choice]
                if next_road == -1:
                    continue
                generated[index].append(next_road)
                if next_road != trips[index].edges[-1]:
                    next_pending.append(index)
            pending = next_pending
    return [
        path if path[-1] == trip.edges[-1] else shorten_path(path, trip.edges[-1], graph)
        for path, trip in zip(generated, trips)
    ]


def shorten_path(path: list[int], destination_edge: int, graph: RoadGraph) -> list[int]:
    destination_node = int(graph.tail[destination_edge])
    distances = [
        haversine_points(
            graph.x[int(graph.head[edge])],
            graph.y[int(graph.head[edge])],
            graph.x[destination_node],
            graph.y[destination_node],
        )
        for edge in path
    ]
    return path[: int(np.argmin(distances)) + 1]


def haversine_points(x1: float, y1: float, x2: float, y2: float) -> float:
    lat1, lat2 = math.radians(y1), math.radians(y2)
    dlat = lat2 - lat1
    dlon = math.radians(x2 - x1)
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(min(value, 1.0)))


def static_node_embeddings(model) -> torch.Tensor:
    if model.args.gnn is not None:
        model.GNN.data.x = model.embeddings.weight
        embeddings = model.GNN()
    else:
        embeddings = model.embeddings.weight
    return torch.cat(
        (torch.zeros((1, embeddings.shape[1]), device=model.device), embeddings), dim=0
    )


def transition_costs(
    model,
    destination: int,
    graph: RoadGraph,
    chunk_edges: int,
) -> tuple[list[list[float]], float, float]:
    embedding_started = time.perf_counter()
    embeddings = static_node_embeddings(model)
    embedding_seconds = time.perf_counter() - embedding_started
    costs: list[list[float]] = [[] for _ in graph.neighbors]
    score_started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(graph.tail), chunk_edges):
            stop = min(start + chunk_edges, len(graph.tail))
            current_rows = list(range(start, stop))
            current = [road for road in current_rows for _ in range(graph.max_neighbors)]
            candidates = [
                neighbor
                for road in current_rows
                for neighbor in graph.padded_neighbors[road]
            ]
            destinations = [destination] * len(current)
            source_left, source_right = mapping_tensors(current, graph, model.device)
            nbr_left, nbr_right = mapping_tensors(candidates, graph, model.device)
            dest_left, dest_right = mapping_tensors(destinations, graph, model.device)
            source_vec = torch.cat(
                (embeddings[1 + source_left], embeddings[1 + source_right]), dim=1
            )
            nbr_vec = torch.cat((embeddings[1 + nbr_left], embeddings[1 + nbr_right]), dim=1)
            dest_vec = torch.cat((embeddings[1 + dest_left], embeddings[1 + dest_right]), dim=1)
            logits = model.confidence_model(torch.cat((source_vec, nbr_vec, dest_vec), dim=1))
            logits[nbr_left == -1] = -100
            nll = -F.log_softmax(logits.reshape(-1, graph.max_neighbors), dim=1)
            nll_rows = nll.detach().cpu().tolist()
            for road, row in zip(current_rows, nll_rows):
                costs[road] = row[: len(graph.neighbors[road])]
    return costs, embedding_seconds, time.perf_counter() - score_started


def mapping_tensors(
    roads: list[int], graph: RoadGraph, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    left = torch.tensor([graph.edge_mapping[road][0] for road in roads], device=device)
    right = torch.tensor([graph.edge_mapping[road][1] for road in roads], device=device)
    return left.long(), right.long()


def dijkstra_path(
    source: int, target: int, graph: RoadGraph, costs: list[list[float]]
) -> list[int]:
    distance = {source: 0.0}
    predecessor: dict[int, int] = {}
    queue = [(0.0, source)]
    while queue:
        current_distance, road = heapq.heappop(queue)
        if current_distance != distance.get(road):
            continue
        if road == target:
            break
        for index, neighbor in enumerate(graph.neighbors[road]):
            candidate = current_distance + costs[road][index]
            if candidate < distance.get(neighbor, math.inf):
                distance[neighbor] = candidate
                predecessor[neighbor] = road
                heapq.heappush(queue, (candidate, neighbor))
    if target not in distance:
        raise RuntimeError(f"NeuroMLR transformed graph cannot reach {source}->{target}")
    path = [target]
    while path[-1] != source:
        path.append(predecessor[path[-1]])
    path.reverse()
    return path


def dijkstra_paths(
    model,
    trips: list[Trip],
    graph: RoadGraph,
    chunk_edges: int,
) -> tuple[list[list[int]], list[dict[str, float | str]]]:
    generated = []
    timing_rows = []
    with torch.no_grad():
        for trip in trips:
            costs, embedding_seconds, scoring_seconds = transition_costs(
                model, trip.edges[-1], graph, chunk_edges
            )
            route_started = time.perf_counter()
            path = dijkstra_path(trip.edges[0], trip.edges[-1], graph, costs)
            route_seconds = time.perf_counter() - route_started
            generated.append(path)
            timing_rows.append(
                {
                    "sample_id": trip.sample_id,
                    "embedding_seconds": embedding_seconds,
                    "transition_scoring_seconds": scoring_seconds,
                    "dijkstra_seconds": route_seconds,
                }
            )
    return generated, timing_rows


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def verify_repeated_paths(
    reference: list[list[int]] | None,
    candidate: list[list[int]],
    phase: str,
    repetition: int,
) -> list[list[int]]:
    if reference is not None and candidate != reference:
        raise RuntimeError(
            f"Dijkstra {phase} repetition {repetition} produced different routes"
        )
    return candidate if reference is None else reference


def repeated_dijkstra_paths(
    model,
    trips: list[Trip],
    graph: RoadGraph,
    chunk_edges: int,
    device: torch.device,
    warmup_repetitions: int,
    measured_repetitions: int,
) -> RepeatedDijkstraResult:
    reference = None
    warmup_seconds = []
    for repetition in range(warmup_repetitions):
        synchronize_device(device)
        started = time.perf_counter()
        candidate, _ = dijkstra_paths(model, trips, graph, chunk_edges)
        synchronize_device(device)
        warmup_seconds.append(time.perf_counter() - started)
        reference = verify_repeated_paths(reference, candidate, "warm-up", repetition)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    generated = None
    measured_seconds = []
    component_totals_per_repetition = []
    for repetition in range(measured_repetitions):
        synchronize_device(device)
        started = time.perf_counter()
        candidate, timing_rows = dijkstra_paths(model, trips, graph, chunk_edges)
        synchronize_device(device)
        measured_seconds.append(time.perf_counter() - started)
        reference = verify_repeated_paths(reference, candidate, "measured", repetition)
        if generated is None:
            generated = candidate
        component_totals_per_repetition.append(sum_timing(timing_rows))

    assert generated is not None
    return RepeatedDijkstraResult(
        generated=generated,
        warmup_seconds=warmup_seconds,
        measured_seconds=measured_seconds,
        component_totals_per_repetition=component_totals_per_repetition,
    )


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_prediction_resume_binding(
    args: argparse.Namespace,
    dataset: DatasetArtifact,
    graph: RoadGraph,
    device: torch.device,
    checkpoint: dict[str, Any],
    checkpoint_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": PREDICTION_RESUME_BINDING_SCHEMA,
        "adapter_version": ADAPTER_VERSION,
        "adapter_source_sha256": sha256_file(Path(__file__).resolve()),
        "source_revision": args.source_revision,
        "upstream_commit": UPSTREAM_COMMIT,
        "method": "neuromlr_greedy",
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": checkpoint_sha256,
            "epoch": checkpoint["epoch"],
        },
        "dataset": {
            "manifest_path": str(dataset.manifest_path),
            "manifest_sha256": dataset.manifest_sha256,
            "records_path": str(dataset.records_path),
            "records_sha256": dataset.records_sha256,
            "dataset_id": dataset.manifest.dataset_id,
            "network_id": dataset.manifest.network_id,
            "samples": len(dataset.trips),
        },
        "graph_identity": graph.identity,
        "coordinate_identity": graph.coordinate_identity,
        "model": vars(model_arguments(args)),
        "seed": args.seed,
        "device": str(device),
        "environment": prediction_environment_identity(device),
        "route_chunk_size": args.route_chunk_size,
        "query_protocol": "fixed_true_first_edge_to_true_last_edge",
        "predictions_path": str(args.predictions.resolve()),
    }


def prediction_part_name(index: int) -> str:
    return f"parts/part-{index:06d}.jsonl"


def require_nonnegative_number(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise RuntimeError(f"{field} must be a finite nonnegative number")
    return float(value)


def validate_prediction_part(
    path: Path,
    trips: list[Trip],
    graph: RoadGraph,
    expected_sha256: str,
) -> int:
    if not path.is_file():
        raise RuntimeError(f"completed prediction shard is missing: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"prediction shard hash mismatch for {path}: "
            f"{actual_sha256} != {expected_sha256}"
        )
    endpoint_failures = 0
    row_count = 0
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if row_count >= len(trips):
                raise RuntimeError(f"prediction shard {path} has too many rows")
            if not line.strip():
                raise RuntimeError(
                    f"blank prediction JSONL line {line_number} in {path}"
                )
            row = require_exact_fields(
                decode_json(line, f"prediction JSONL line {line_number}"),
                {"sample_id", "predicted_edge_ids"},
                f"prediction JSONL line {line_number}",
            )
            trip = trips[row_count]
            if row["sample_id"] != trip.sample_id:
                raise RuntimeError(
                    f"prediction shard {path} sample order differs at row {line_number}"
                )
            prediction = row["predicted_edge_ids"]
            if not isinstance(prediction, list) or not prediction:
                raise RuntimeError(
                    f"prediction shard {path} has an empty or non-list path"
                )
            if any(
                isinstance(edge, bool)
                or not isinstance(edge, int)
                or edge < 0
                or edge >= len(graph.tail)
                for edge in prediction
            ):
                raise RuntimeError(
                    f"prediction shard {path} has an edge outside the raw graph"
                )
            if prediction[0] != trip.edges[0]:
                raise RuntimeError(
                    f"prediction shard {path} changes the fixed first edge for "
                    f"sample {trip.sample_id!r}"
                )
            if any(
                right not in graph.neighbors[left]
                for left, right in zip(prediction, prediction[1:])
            ):
                raise RuntimeError(
                    f"prediction shard {path} has an illegal directed transition "
                    f"for sample {trip.sample_id!r}"
                )
            endpoint_failures += prediction[-1] != trip.edges[-1]
            row_count += 1
    if row_count != len(trips):
        raise RuntimeError(
            f"prediction shard {path} has {row_count} rows, expected {len(trips)}"
        )
    return endpoint_failures


def validate_prediction_progress(
    value: Any,
    binding: dict[str, Any],
    trips: list[Trip],
    chunk_size: int,
    resume_dir: Path,
    graph: RoadGraph,
) -> dict[str, Any]:
    progress = require_exact_fields(
        value,
        {
            "schema",
            "binding",
            "binding_sha256",
            "status",
            "total_samples",
            "chunk_size",
            "total_chunks",
            "completed_samples",
            "completed_chunks",
            "prediction_seconds",
            "estimated_remaining_prediction_seconds",
            "endpoint_failures",
            "peak_rss_kib",
            "peak_cuda_memory_bytes",
            "sessions",
            "output",
        },
        "prediction progress",
    )
    if progress["schema"] != PREDICTION_PROGRESS_SCHEMA:
        raise RuntimeError("prediction progress schema mismatch")
    binding_sha256 = canonical_json_sha256(binding)
    if progress["binding"] != binding or progress["binding_sha256"] != binding_sha256:
        raise RuntimeError(
            "prediction resume binding differs; checkpoint, dataset, graph, "
            "configuration, device, source, or output path changed"
        )
    total_chunks = math.ceil(len(trips) / chunk_size)
    expected_scalars = {
        "total_samples": len(trips),
        "chunk_size": chunk_size,
        "total_chunks": total_chunks,
    }
    for field, expected in expected_scalars.items():
        if progress[field] != expected:
            raise RuntimeError(f"prediction progress {field} differs")
    if progress["status"] not in {"running", "complete"}:
        raise RuntimeError("prediction progress has an invalid status")
    if (
        isinstance(progress["sessions"], bool)
        or not isinstance(progress["sessions"], int)
        or progress["sessions"] < 1
    ):
        raise RuntimeError("prediction progress sessions must be a positive integer")
    chunks = progress["completed_chunks"]
    if not isinstance(chunks, list) or len(chunks) > total_chunks:
        raise RuntimeError("prediction progress completed_chunks is invalid")
    completed_samples = 0
    prediction_seconds = 0.0
    endpoint_failures = 0
    for index, chunk_value in enumerate(chunks):
        chunk = require_exact_fields(
            chunk_value,
            {
                "index",
                "start",
                "stop",
                "samples",
                "path",
                "sha256",
                "prediction_seconds",
                "endpoint_failures",
            },
            f"prediction progress chunk {index}",
        )
        start = index * chunk_size
        stop = min(start + chunk_size, len(trips))
        expected_path = prediction_part_name(index)
        if chunk["index"] != index or chunk["start"] != start or chunk["stop"] != stop:
            raise RuntimeError(f"prediction progress chunk {index} range differs")
        if chunk["samples"] != stop - start or chunk["path"] != expected_path:
            raise RuntimeError(f"prediction progress chunk {index} metadata differs")
        sha256 = chunk["sha256"]
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise RuntimeError(f"prediction progress chunk {index} hash is invalid")
        seconds = require_nonnegative_number(
            chunk["prediction_seconds"],
            f"prediction progress chunk {index} seconds",
        )
        failures = chunk["endpoint_failures"]
        if (
            isinstance(failures, bool)
            or not isinstance(failures, int)
            or failures < 0
            or failures > stop - start
        ):
            raise RuntimeError(
                f"prediction progress chunk {index} endpoint failures are invalid"
            )
        actual_failures = validate_prediction_part(
            resume_dir / expected_path, trips[start:stop], graph, sha256
        )
        if actual_failures != failures:
            raise RuntimeError(
                f"prediction progress chunk {index} endpoint failures differ"
            )
        completed_samples += stop - start
        prediction_seconds += seconds
        endpoint_failures += failures
    if progress["completed_samples"] != completed_samples:
        raise RuntimeError("prediction progress completed_samples differs")
    recorded_seconds = require_nonnegative_number(
        progress["prediction_seconds"], "prediction progress prediction_seconds"
    )
    if not math.isclose(recorded_seconds, prediction_seconds, rel_tol=0, abs_tol=1e-9):
        raise RuntimeError("prediction progress prediction_seconds differs")
    estimated_remaining = progress["estimated_remaining_prediction_seconds"]
    if completed_samples == 0:
        if estimated_remaining is not None:
            raise RuntimeError(
                "prediction progress estimate must be null before the first chunk"
            )
    else:
        recorded_estimate = require_nonnegative_number(
            estimated_remaining,
            "prediction progress estimated_remaining_prediction_seconds",
        )
        expected_estimate = prediction_seconds / completed_samples * (
            len(trips) - completed_samples
        )
        if not math.isclose(
            recorded_estimate, expected_estimate, rel_tol=0, abs_tol=1e-9
        ):
            raise RuntimeError("prediction progress remaining-time estimate differs")
    if progress["endpoint_failures"] != endpoint_failures:
        raise RuntimeError("prediction progress endpoint_failures differs")
    for field in ("peak_rss_kib", "peak_cuda_memory_bytes"):
        value = progress[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"prediction progress {field} is invalid")
    if progress["status"] == "complete":
        if len(chunks) != total_chunks or not isinstance(progress["output"], dict):
            raise RuntimeError("complete prediction progress is incomplete")
        output = require_exact_fields(
            progress["output"],
            {"path", "sha256", "bytes", "samples"},
            "prediction progress output",
        )
        if output["path"] != binding["predictions_path"]:
            raise RuntimeError("prediction progress output path differs")
        output_sha256 = output["sha256"]
        if (
            not isinstance(output_sha256, str)
            or len(output_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in output_sha256
            )
        ):
            raise RuntimeError("prediction progress output hash is invalid")
        if (
            isinstance(output["bytes"], bool)
            or not isinstance(output["bytes"], int)
            or output["bytes"] <= 0
            or output["samples"] != len(trips)
        ):
            raise RuntimeError("prediction progress output metadata is invalid")
    elif progress["output"] is not None:
        raise RuntimeError("running prediction progress must not declare final output")
    return progress


def concatenate_prediction_parts(
    destination: Path,
    part_paths: list[Path],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as output:
        for part_path in part_paths:
            with part_path.open("rb") as source:
                while block := source.read(1024 * 1024):
                    output.write(block)
    os.replace(temporary, destination)


def run_chunked_greedy_prediction(
    model,
    trips: list[Trip],
    graph: RoadGraph,
    device: torch.device,
    predictions: Path,
    chunk_size: int,
    resume_dir: Path,
    progress_path: Path,
    resume_mode: str,
    binding: dict[str, Any],
) -> ChunkedGreedyResult:
    if chunk_size <= 0:
        raise RuntimeError("chunked Greedy prediction requires a positive chunk size")
    if resume_mode not in {"auto", "never", "require"}:
        raise RuntimeError("invalid prediction resume mode")
    resume_dir = resume_dir.resolve()
    progress_path = progress_path.resolve()
    predictions = predictions.resolve()
    if binding.get("predictions_path") != str(predictions):
        raise RuntimeError(
            "prediction resume binding must contain the resolved predictions path"
        )
    resume_files_exist = resume_dir.exists() and any(resume_dir.iterdir())
    if resume_mode == "never" and (progress_path.exists() or resume_files_exist):
        raise RuntimeError("prediction resume artifacts exist but --resume=never was used")
    if resume_mode == "require" and not progress_path.is_file():
        raise RuntimeError("prediction resume was required but progress does not exist")
    if not progress_path.exists() and resume_files_exist:
        raise RuntimeError("prediction shards exist without bound progress metadata")

    resume_dir.mkdir(parents=True, exist_ok=True)
    (resume_dir / "parts").mkdir(parents=True, exist_ok=True)
    total_chunks = math.ceil(len(trips) / chunk_size)
    if progress_path.is_file() and resume_mode != "never":
        progress = validate_prediction_progress(
            decode_json(progress_path.read_text(encoding="utf-8"), "prediction progress"),
            binding,
            trips,
            chunk_size,
            resume_dir,
            graph,
        )
        resumed_chunks = len(progress["completed_chunks"])
        if progress["status"] != "complete":
            progress["sessions"] += 1
            write_json(progress_path, progress)
    else:
        progress = {
            "schema": PREDICTION_PROGRESS_SCHEMA,
            "binding": binding,
            "binding_sha256": canonical_json_sha256(binding),
            "status": "running",
            "total_samples": len(trips),
            "chunk_size": chunk_size,
            "total_chunks": total_chunks,
            "completed_samples": 0,
            "completed_chunks": [],
            "prediction_seconds": 0.0,
            "estimated_remaining_prediction_seconds": None,
            "endpoint_failures": 0,
            "peak_rss_kib": peak_rss_kib(),
            "peak_cuda_memory_bytes": (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else 0
            ),
            "sessions": 1,
            "output": None,
        }
        resumed_chunks = 0
        write_json(progress_path, progress)

    for index in range(len(progress["completed_chunks"]), total_chunks):
        start = index * chunk_size
        stop = min(start + chunk_size, len(trips))
        chunk_trips = trips[start:stop]
        synchronize_device(device)
        started = time.perf_counter()
        generated = greedy_paths(model, chunk_trips, graph, device)
        synchronize_device(device)
        seconds = time.perf_counter() - started
        failures = sum(
            prediction[-1] != trip.edges[-1]
            for trip, prediction in zip(chunk_trips, generated)
        )
        relative_path = prediction_part_name(index)
        part_path = resume_dir / relative_path
        write_prediction_rows(part_path, chunk_trips, generated)
        part_sha256 = sha256_file(part_path)
        validated_failures = validate_prediction_part(
            part_path, chunk_trips, graph, part_sha256
        )
        if validated_failures != failures:
            raise RuntimeError(
                f"new prediction shard {index} endpoint failures differ"
            )
        progress["completed_chunks"].append(
            {
                "index": index,
                "start": start,
                "stop": stop,
                "samples": stop - start,
                "path": relative_path,
                "sha256": part_sha256,
                "prediction_seconds": seconds,
                "endpoint_failures": failures,
            }
        )
        progress["completed_samples"] = stop
        progress["prediction_seconds"] += seconds
        progress["estimated_remaining_prediction_seconds"] = (
            progress["prediction_seconds"] / stop * (len(trips) - stop)
        )
        progress["endpoint_failures"] += failures
        progress["peak_rss_kib"] = max(progress["peak_rss_kib"], peak_rss_kib())
        progress["peak_cuda_memory_bytes"] = max(
            progress["peak_cuda_memory_bytes"],
            (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else 0
            ),
        )
        write_json(progress_path, progress)
        print(
            json.dumps(
                {
                    "event": "neuromlr_greedy_progress",
                    "completed_chunks": index + 1,
                    "total_chunks": total_chunks,
                    "completed_samples": stop,
                    "total_samples": len(trips),
                    "fraction": stop / len(trips),
                    "prediction_seconds": progress["prediction_seconds"],
                    "estimated_remaining_prediction_seconds": progress[
                        "estimated_remaining_prediction_seconds"
                    ],
                    "peak_rss_kib": progress["peak_rss_kib"],
                    "peak_cuda_memory_bytes": progress[
                        "peak_cuda_memory_bytes"
                    ],
                    "progress": str(progress_path),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    part_paths = [
        resume_dir / prediction_part_name(index) for index in range(total_chunks)
    ]
    concatenate_prediction_parts(predictions, part_paths)
    output_sha256 = sha256_file(predictions)
    progress["status"] = "complete"
    progress["output"] = {
        "path": str(predictions),
        "sha256": output_sha256,
        "bytes": predictions.stat().st_size,
        "samples": len(trips),
    }
    write_json(progress_path, progress)
    return ChunkedGreedyResult(
        prediction_seconds=float(progress["prediction_seconds"]),
        chunk_seconds=[
            float(chunk["prediction_seconds"])
            for chunk in progress["completed_chunks"]
        ],
        endpoint_failures=int(progress["endpoint_failures"]),
        resumed_chunks=resumed_chunks,
        completed_chunks=total_chunks,
        output_sha256=output_sha256,
        progress_path=progress_path,
        resume_dir=resume_dir,
        peak_rss_kib=int(progress["peak_rss_kib"]),
        peak_cuda_memory_bytes=int(progress["peak_cuda_memory_bytes"]),
    )


def predict(args: argparse.Namespace) -> None:
    if args.warmup_repetitions < 0 or args.measured_repetitions <= 0:
        raise RuntimeError(
            "warm-up repetitions must be nonnegative and measured repetitions positive"
        )
    if not args.source_revision.strip():
        raise RuntimeError("source revision must not be empty")
    if args.route_chunk_size < 0:
        raise RuntimeError("route chunk size must be nonnegative")
    chunked_greedy = args.route_chunk_size > 0
    if chunked_greedy and args.method != "greedy":
        raise RuntimeError("route chunking is supported only for NeuroMLR-Greedy")
    if chunked_greedy and (
        args.warmup_repetitions != 0 or args.measured_repetitions != 1
    ):
        raise RuntimeError(
            "resumable Greedy quality prediction requires zero warm-ups and one "
            "measured repetition; use legacy mode for efficiency measurements"
        )
    if not chunked_greedy and (
        args.resume_dir is not None
        or args.progress is not None
        or args.resume != "auto"
    ):
        raise RuntimeError("prediction resume options require a positive route chunk size")
    total_started = time.perf_counter()
    graph_started = time.perf_counter()
    graph = load_road_graph(args.map_dir)
    dataset = load_dataset_manifest(args.dataset_manifest)
    trips = dataset.trips
    validate_trips(trips, graph)
    data_seconds = time.perf_counter() - graph_started
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "prediction requested CUDA, but CUDA is unavailable; refusing an "
                "accidental CPU fallback"
            )
        if requested_device.index not in {None, 0}:
            raise RuntimeError("prediction CUDA protocol requires logical cuda:0")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise RuntimeError(
                "prediction CUDA protocol requires CUDA_VISIBLE_DEVICES=0"
            )
    device = requested_device
    model_started = time.perf_counter()
    checkpoint_sha256 = sha256_file(args.checkpoint)
    model, checkpoint = load_checkpoint_model(
        args, graph, device, dataset.manifest.network_id
    )
    if sha256_file(args.checkpoint) != checkpoint_sha256:
        raise RuntimeError("checkpoint changed while it was being loaded")
    model_seconds = time.perf_counter() - model_started
    if chunked_greedy:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        resume_dir = (
            args.resume_dir
            or args.predictions.with_name(args.predictions.name + ".resume")
        )
        progress_path = args.progress or resume_dir / "progress.json"
        binding = build_prediction_resume_binding(
            args, dataset, graph, device, checkpoint, checkpoint_sha256
        )
        chunked = run_chunked_greedy_prediction(
            model=model,
            trips=trips,
            graph=graph,
            device=device,
            predictions=args.predictions,
            chunk_size=args.route_chunk_size,
            resume_dir=resume_dir,
            progress_path=progress_path,
            resume_mode=args.resume,
            binding=binding,
        )
        prediction_seconds = chunked.prediction_seconds
        warmup_seconds = []
        repetition_seconds = [prediction_seconds]
        component_totals_per_repetition = []
        endpoint_failures = chunked.endpoint_failures
        output_sha256 = chunked.output_sha256
        execution = {
            "mode": "chunked_resumable_quality_prediction",
            "route_chunk_size": args.route_chunk_size,
            "completed_chunks": chunked.completed_chunks,
            "resumed_chunks": chunked.resumed_chunks,
            "resume_dir": str(chunked.resume_dir),
            "progress": str(chunked.progress_path),
            "prediction_chunk_seconds": chunked.chunk_seconds,
            "timing_scope": "sum_of_atomic_chunk_measurements_across_sessions",
            "resource_scope": "maximum_observed_across_committed_sessions",
        }
    elif args.method == "greedy":
        warmup_seconds = []
        for _ in range(args.warmup_repetitions):
            synchronize_device(device)
            warmup_started = time.perf_counter()
            greedy_paths(model, trips, graph, device)
            synchronize_device(device)
            warmup_seconds.append(time.perf_counter() - warmup_started)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        repetition_seconds = []
        generated = None
        for repetition in range(args.measured_repetitions):
            prediction_started = time.perf_counter()
            candidate = greedy_paths(model, trips, graph, device)
            synchronize_device(device)
            repetition_seconds.append(time.perf_counter() - prediction_started)
            if generated is None:
                generated = candidate
            elif candidate != generated:
                raise RuntimeError(f"Greedy repetition {repetition} produced different routes")
        assert generated is not None
        prediction_seconds = sum(repetition_seconds) / len(repetition_seconds)
        component_totals_per_repetition = []
        endpoint_failures = sum(
            prediction[-1] != trip.edges[-1]
            for trip, prediction in zip(trips, generated)
        )
        write_prediction_rows(args.predictions, trips, generated)
        output_sha256 = sha256_file(args.predictions)
        execution = {
            "mode": "legacy_full_batch",
            "route_chunk_size": 0,
        }
    else:
        repeated = repeated_dijkstra_paths(
            model,
            trips,
            graph,
            args.score_edge_chunk,
            device,
            args.warmup_repetitions,
            args.measured_repetitions,
        )
        generated = repeated.generated
        warmup_seconds = repeated.warmup_seconds
        repetition_seconds = repeated.measured_seconds
        component_totals_per_repetition = repeated.component_totals_per_repetition
        prediction_seconds = sum(repetition_seconds) / len(repetition_seconds)
        endpoint_failures = sum(
            prediction[-1] != trip.edges[-1]
            for trip, prediction in zip(trips, generated)
        )
        write_prediction_rows(args.predictions, trips, generated)
        output_sha256 = sha256_file(args.predictions)
        execution = {
            "mode": "legacy_full_batch",
            "route_chunk_size": 0,
        }
    if chunked_greedy:
        prediction_peak_rss_kib = chunked.peak_rss_kib
        prediction_peak_cuda_memory_bytes = chunked.peak_cuda_memory_bytes
    else:
        prediction_peak_rss_kib = peak_rss_kib()
        prediction_peak_cuda_memory_bytes = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        )
    diagnostics = {
        "schema": "ewr.neuromlr-diagnostics/v1",
        "method": f"neuromlr_{args.method}",
        "upstream_commit": UPSTREAM_COMMIT,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": checkpoint["epoch"],
        "dataset_manifest": str(dataset.manifest_path),
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "dataset_records": str(dataset.records_path),
        "dataset_records_sha256": dataset.records_sha256,
        "dataset_id": dataset.manifest.dataset_id,
        "network_id": dataset.manifest.network_id,
        "graph_identity": graph.identity,
        "coordinate_identity": graph.coordinate_identity,
        "samples": len(trips),
        "query_protocol": "fixed_true_first_edge_to_true_last_edge",
        "endpoint_failures": endpoint_failures,
        "predictions_sha256": output_sha256,
        "execution": execution,
        "timing": {
            "data_and_graph_seconds": data_seconds,
            "model_load_seconds": model_seconds,
            "prediction_seconds": prediction_seconds,
            "warmup_repetition_seconds": warmup_seconds,
            "prediction_repetition_seconds": repetition_seconds,
            "mean_seconds_per_query": prediction_seconds / len(trips),
            "queries_per_second": len(trips) / prediction_seconds,
            "component_totals": mean_timing(component_totals_per_repetition),
            "component_totals_per_repetition": component_totals_per_repetition,
            "total_process_seconds": time.perf_counter() - total_started,
        },
        "peak_rss_kib": prediction_peak_rss_kib,
        "peak_cuda_memory_bytes": prediction_peak_cuda_memory_bytes,
        "warmup_repetitions": args.warmup_repetitions,
        "measured_repetitions": args.measured_repetitions,
        "seed": args.seed,
        "traffic": False,
    }
    receipt = build_run_receipt(
        args, dataset, graph, device, checkpoint, checkpoint_sha256
    )
    write_json(args.diagnostics, diagnostics)
    write_json(args.run_receipt, receipt)
    print(json.dumps(receipt, indent=2))


def build_run_receipt(
    args: argparse.Namespace,
    dataset: DatasetArtifact,
    graph: RoadGraph,
    device: torch.device,
    checkpoint: dict,
    checkpoint_sha256: str,
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    if environment is None:
        environment = model_environment(device)
    if not environment or any(
        not isinstance(key, str)
        or not key.strip()
        or not isinstance(value, str)
        or not value.strip()
        for key, value in environment.items()
    ):
        raise RuntimeError("run receipt environment must contain nonempty string pairs")
    return {
        "schema": RUN_RECEIPT_SCHEMA,
        "method": {
            "name": f"neuromlr_{args.method}",
            "version": ADAPTER_VERSION,
        },
        "dataset_id": dataset.manifest.dataset_id,
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "prediction_records_schema": PREDICTION_RECORD_SCHEMA,
        "configuration": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_epoch": checkpoint["epoch"],
            "dataset_records_sha256": dataset.records_sha256,
            "network_id": dataset.manifest.network_id,
            "graph_identity": graph.identity,
            "coordinate_identity": graph.coordinate_identity,
            "query_protocol": "fixed_true_first_edge_to_true_last_edge",
            "seed": args.seed,
            "model": vars(model_arguments(args)),
            "score_edge_chunk": args.score_edge_chunk,
            "route_chunk_size": args.route_chunk_size,
            "prediction_execution_mode": (
                "chunked_resumable_quality_prediction"
                if args.route_chunk_size > 0
                else "legacy_full_batch"
            ),
            "warmup_repetitions": args.warmup_repetitions,
            "measured_repetitions": args.measured_repetitions,
            "upstream_commit": UPSTREAM_COMMIT,
        },
        "source_revision": args.source_revision,
        "environment": environment,
    }


def model_environment(device: torch.device) -> dict[str, str]:
    load_model_dependencies()
    return {
        "device": str(device),
        "numpy": str(np.__version__),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
    }


def prediction_environment_identity(device: torch.device) -> dict[str, str]:
    environment = model_environment(device)
    if device.type == "cuda":
        environment.update(
            {
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "cuda_runtime": str(torch.version.cuda),
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_compute_capability": ".".join(
                    str(value) for value in torch.cuda.get_device_capability(device)
                ),
            }
        )
    return environment


def route_metrics(observed: list[list[int]], predicted: list[list[int]]) -> dict[str, float | int]:
    if len(observed) != len(predicted) or not observed:
        raise RuntimeError("route metrics require equal nonempty path lists")
    exact = precision = recall = f1 = jaccard = 0.0
    for truth, prediction in zip(observed, predicted):
        truth_set, prediction_set = set(truth), set(prediction)
        intersection = len(truth_set & prediction_set)
        sample_precision = intersection / max(len(prediction_set), 1)
        sample_recall = intersection / max(len(truth_set), 1)
        sample_f1 = (
            0.0
            if sample_precision + sample_recall == 0
            else 2 * sample_precision * sample_recall / (sample_precision + sample_recall)
        )
        exact += prediction == truth
        precision += sample_precision
        recall += sample_recall
        f1 += sample_f1
        jaccard += intersection / max(len(truth_set | prediction_set), 1)
    count = len(observed)
    return {
        "samples": count,
        "edge_precision": precision / count,
        "edge_recall": recall / count,
        "edge_f1": f1 / count,
        "exact_match": exact / count,
        "edge_jaccard": jaccard / count,
    }


def write_prediction_rows(
    path: Path,
    trips: list[Trip],
    generated: list[list[int]],
) -> None:
    if not trips or len(trips) != len(generated):
        raise RuntimeError("predictions require equal nonempty sample and path lists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for trip, prediction in zip(trips, generated):
            if not prediction or any(
                isinstance(edge, bool)
                or not isinstance(edge, int)
                or edge < 0
                or edge > 0xFFFF_FFFF
                for edge in prediction
            ):
                raise RuntimeError(f"sample {trip.sample_id!r} has an invalid predicted path")
            row = {
                "sample_id": trip.sample_id,
                "predicted_edge_ids": prediction,
            }
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def sum_timing(rows: list[dict]) -> dict[str, float]:
    return {
        key: sum(float(row[key]) for row in rows)
        for key in ["embedding_seconds", "transition_scoring_seconds", "dijkstra_seconds"]
    } if rows else {}


def mean_timing(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = ["embedding_seconds", "transition_scoring_seconds", "dijkstra_seconds"]
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def peak_rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


if __name__ == "__main__":
    main()
