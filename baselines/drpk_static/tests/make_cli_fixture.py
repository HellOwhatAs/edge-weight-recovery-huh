#!/usr/bin/env python3
"""Create a six-road directed fixture for manual CLI smoke tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import shapefile


def write_manifest(root: Path, split: str, routes: list[list[int]]) -> None:
    records_name = f"{split}.jsonl"
    with (root / records_name).open("w", encoding="utf-8") as output:
        for index, route in enumerate(routes):
            output.write(
                json.dumps(
                    {
                        "sample_id": f"fixture:{split}:{index}",
                        "original_edge_ids": route,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    (root / f"{split}.manifest.json").write_text(
        json.dumps(
            {
                "schema": "ewr.dataset-manifest/v1",
                "dataset_id": f"drpk-static-fixture/{split}",
                "network_id": "drpk-static-six-road-v1",
                "records_schema": "ewr.dataset-record/v1",
                "records_file": records_name,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    map_dir = root / "map"
    map_dir.mkdir(parents=True, exist_ok=True)

    coordinates = [(0, 0), (1, 0), (2, 0), (1, 1), (3, 0), (4, 0)]
    nodes = shapefile.Writer(str(map_dir / "nodes"), shapeType=shapefile.POINT)
    nodes.field("y", "N", decimal=8)
    nodes.field("x", "N", decimal=8)
    nodes.field("osmid", "N", size=18)
    for osmid, (x, y) in enumerate(coordinates, start=100):
        nodes.point(x, y)
        nodes.record(y, x, osmid)
    nodes.close()

    endpoints = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5)]
    edges = shapefile.Writer(str(map_dir / "edges"), shapeType=shapefile.POLYLINE)
    edges.field("u", "N", size=18)
    edges.field("v", "N", size=18)
    edges.field("fid", "N", size=18)
    for fid, (source, destination) in enumerate(endpoints):
        edges.line([[coordinates[source], coordinates[destination]]])
        edges.record(100 + source, 100 + destination, fid)
    edges.close()

    write_manifest(
        root,
        "train",
        [[0, 1, 3, 5], [0, 2, 4, 5], [0, 1, 3, 5], [0, 2, 4, 5]],
    )
    write_manifest(root, "validation", [[0, 1, 3, 5], [0, 2, 4, 5]])
    write_manifest(root, "test", [[0, 1, 3, 5], [0, 2, 4, 5]])


if __name__ == "__main__":
    main()
