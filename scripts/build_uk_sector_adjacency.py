#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_GEOJSON_PATH = Path("src/py_lucidum/tools/uk_map/static/geodata/sectors_MappaR.geojson")
DEFAULT_OUTPUT_PATH = Path("src/py_lucidum/tools/uk_map/static/geodata/sector_adjacency.json")
JOIN_PROPERTY = "PostcodeSector"
COORDINATE_DECIMALS = 6


def main() -> None:
    parser = argparse.ArgumentParser(description="Build postcode-sector shared-edge adjacency from bundled GeoJSON.")
    parser.add_argument("--geojson", type=Path, default=DEFAULT_GEOJSON_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--decimals", type=int, default=COORDINATE_DECIMALS)
    args = parser.parse_args()

    payload = build_adjacency(args.geojson, decimals=args.decimals)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    neighbour_counts = [len(indexes) for indexes in payload["neighbours"]]
    print(
        f"Wrote {args.output} with {len(payload['keys'])} sectors, "
        f"{sum(neighbour_counts)} directed neighbour links, max degree {max(neighbour_counts, default=0)}."
    )


def build_adjacency(geojson_path: Path, *, decimals: int = COORDINATE_DECIMALS) -> dict[str, Any]:
    geojson = json.loads(geojson_path.read_text(encoding="utf-8"))
    features = geojson.get("features") or []
    keys = [feature_key(feature) for feature in features]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{geojson_path} contains duplicate {JOIN_PROPERTY} values")

    edge_map: dict[tuple[tuple[float, float], tuple[float, float]], list[int]] = defaultdict(list)
    for feature_index, feature in enumerate(features):
        for ring in geometry_rings(feature.get("geometry") or {}):
            for start, end in zip(ring, ring[1:]):
                owners = edge_map[edge_key(start, end, decimals)]
                if not owners or owners[-1] != feature_index:
                    owners.append(feature_index)

    neighbours = [set() for _ in features]
    for owners in edge_map.values():
        if len(owners) < 2:
            continue
        for left_pos, left in enumerate(owners):
            for right in owners[left_pos + 1:]:
                neighbours[left].add(right)
                neighbours[right].add(left)

    return {
        "version": 1,
        "level": "sector",
        "join_property": JOIN_PROPERTY,
        "source": geojson_path.name,
        "neighbour_type": "shared_edge",
        "coordinate_decimals": decimals,
        "keys": keys,
        "neighbours": [sorted(indexes) for indexes in neighbours],
    }


def feature_key(feature: dict[str, Any]) -> str:
    key = str((feature.get("properties") or {}).get(JOIN_PROPERTY) or "").strip()
    if not key:
        raise ValueError(f"Feature is missing {JOIN_PROPERTY}")
    return key


def geometry_rings(geometry: dict[str, Any]) -> list[list[list[float]]]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates") or []
    if geometry_type == "Polygon":
        return coordinates
    if geometry_type == "MultiPolygon":
        return [ring for polygon in coordinates for ring in polygon]
    return []


def edge_key(start: list[float], end: list[float], decimals: int) -> tuple[tuple[float, float], tuple[float, float]]:
    left = quantized_coordinate(start, decimals)
    right = quantized_coordinate(end, decimals)
    return (left, right) if left < right else (right, left)


def quantized_coordinate(coordinate: list[float], decimals: int) -> tuple[float, float]:
    return (round(float(coordinate[0]), decimals), round(float(coordinate[1]), decimals))


if __name__ == "__main__":
    main()
