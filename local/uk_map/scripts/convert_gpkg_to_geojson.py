#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import struct
from pathlib import Path
from typing import Any


GPKG_ENVELOPE_BYTES = {
    0: 0,
    1: 32,
    2: 48,
    3: 48,
    4: 64,
}

GEOMETRY_TYPES = {
    1: "Point",
    2: "LineString",
    3: "Polygon",
    4: "MultiPoint",
    5: "MultiLineString",
    6: "MultiPolygon",
    7: "GeometryCollection",
}


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def round_xy(coordinate: list[float], precision: int) -> list[float]:
    return [round(coordinate[0], precision), round(coordinate[1], precision)]


def wkb_type(raw_type: int) -> tuple[int, int, bool]:
    has_srid = bool(raw_type & 0x20000000)
    if raw_type & 0xE0000000:
        dimensions = 2 + int(bool(raw_type & 0x80000000)) + int(bool(raw_type & 0x40000000))
        return raw_type & 0x000000FF, dimensions, has_srid
    if raw_type >= 3000:
        return raw_type - 3000, 4, False
    if raw_type >= 2000:
        return raw_type - 2000, 3, False
    if raw_type >= 1000:
        return raw_type - 1000, 3, False
    return raw_type, 2, False


def read_uint32(data: bytes, offset: int, endian: str) -> tuple[int, int]:
    return struct.unpack_from(endian + "I", data, offset)[0], offset + 4


def read_point(data: bytes, offset: int, endian: str, dimensions: int, precision: int) -> tuple[list[float], int]:
    values = struct.unpack_from(endian + ("d" * dimensions), data, offset)
    return round_xy([values[0], values[1]], precision), offset + 8 * dimensions


def read_line_string(
    data: bytes,
    offset: int,
    endian: str,
    dimensions: int,
    precision: int,
) -> tuple[list[list[float]], int]:
    point_count, offset = read_uint32(data, offset, endian)
    coordinates = []
    for _ in range(point_count):
        point, offset = read_point(data, offset, endian, dimensions, precision)
        coordinates.append(point)
    return coordinates, offset


def read_polygon(
    data: bytes,
    offset: int,
    endian: str,
    dimensions: int,
    precision: int,
) -> tuple[list[list[list[float]]], int]:
    ring_count, offset = read_uint32(data, offset, endian)
    rings = []
    for _ in range(ring_count):
        ring, offset = read_line_string(data, offset, endian, dimensions, precision)
        rings.append(ring)
    return rings, offset


def read_wkb_geometry(data: bytes, offset: int, precision: int) -> tuple[dict[str, Any], int]:
    byte_order = data[offset]
    if byte_order == 0:
        endian = ">"
    elif byte_order == 1:
        endian = "<"
    else:
        raise ValueError(f"Unsupported WKB byte order: {byte_order}")
    raw_type, offset = read_uint32(data, offset + 1, endian)
    geometry_type, dimensions, has_srid = wkb_type(raw_type)
    if has_srid:
        offset += 4

    if geometry_type == 1:
        coordinates, offset = read_point(data, offset, endian, dimensions, precision)
    elif geometry_type == 2:
        coordinates, offset = read_line_string(data, offset, endian, dimensions, precision)
    elif geometry_type == 3:
        coordinates, offset = read_polygon(data, offset, endian, dimensions, precision)
    elif geometry_type in {4, 5, 6, 7}:
        geometry_count, offset = read_uint32(data, offset, endian)
        geometries = []
        for _ in range(geometry_count):
            geometry, offset = read_wkb_geometry(data, offset, precision)
            geometries.append(geometry)
        if geometry_type == 4:
            if any(geometry["type"] != "Point" for geometry in geometries):
                raise ValueError("Invalid MultiPoint member geometry")
            coordinates = [geometry["coordinates"] for geometry in geometries]
        elif geometry_type == 5:
            if any(geometry["type"] != "LineString" for geometry in geometries):
                raise ValueError("Invalid MultiLineString member geometry")
            coordinates = [geometry["coordinates"] for geometry in geometries]
        elif geometry_type == 6:
            if any(geometry["type"] != "Polygon" for geometry in geometries):
                raise ValueError("Invalid MultiPolygon member geometry")
            coordinates = [geometry["coordinates"] for geometry in geometries]
        else:
            return {"type": "GeometryCollection", "geometries": geometries}, offset
    else:
        raise ValueError(f"Unsupported WKB geometry type: {geometry_type}")

    return {"type": GEOMETRY_TYPES[geometry_type], "coordinates": coordinates}, offset


def read_gpkg_geometry(blob: bytes, precision: int) -> dict[str, Any]:
    if blob[:2] != b"GP":
        raise ValueError("Geometry blob does not start with GeoPackage magic bytes")
    flags = blob[3]
    envelope_indicator = (flags >> 1) & 0x07
    if envelope_indicator not in GPKG_ENVELOPE_BYTES:
        raise ValueError(f"Unsupported GeoPackage envelope indicator: {envelope_indicator}")
    wkb_offset = 8 + GPKG_ENVELOPE_BYTES[envelope_indicator]
    geometry, _ = read_wkb_geometry(blob, wkb_offset, precision)
    return geometry


def layer_metadata(connection: sqlite3.Connection) -> list[tuple[str, str, int]]:
    return [
        (str(row[0]), str(row[1]), int(row[2]))
        for row in connection.execute(
            "SELECT table_name, column_name, srs_id FROM gpkg_geometry_columns ORDER BY table_name"
        )
    ]


def table_columns(connection: sqlite3.Connection, table_name: str, geometry_column: str) -> list[str]:
    rows = connection.execute(f"PRAGMA table_info({quote_identifier(table_name)})").fetchall()
    return [str(row[1]) for row in rows if str(row[1]) != geometry_column]


def convert_layer(
    connection: sqlite3.Connection,
    table_name: str,
    geometry_column: str,
    precision: int,
) -> dict[str, Any]:
    property_columns = table_columns(connection, table_name, geometry_column)
    selected_columns = [quote_identifier(geometry_column), *[quote_identifier(column) for column in property_columns]]
    sql = f"SELECT {', '.join(selected_columns)} FROM {quote_identifier(table_name)}"

    features = []
    for row in connection.execute(sql):
        geometry_blob = row[0]
        if geometry_blob is None:
            geometry = None
        else:
            geometry = read_gpkg_geometry(bytes(geometry_blob), precision)
        properties = dict(zip(property_columns, row[1:]))
        features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": properties,
            }
        )

    return {
        "type": "FeatureCollection",
        "name": table_name,
        "features": features,
    }


def output_path_for_layer(gpkg_path: Path, table_name: str, layer_count: int, output_dir: Path) -> Path:
    if layer_count == 1:
        return output_dir / f"{gpkg_path.stem}.geojson"
    return output_dir / f"{gpkg_path.stem}_{table_name}.geojson"


def convert_gpkg(gpkg_path: Path, output_dir: Path, precision: int) -> list[Path]:
    output_paths = []
    with sqlite3.connect(gpkg_path) as connection:
        layers = layer_metadata(connection)
        if not layers:
            raise ValueError(f"No feature layers found in {gpkg_path}")
        for table_name, geometry_column, srs_id in layers:
            if srs_id != 4326:
                raise ValueError(f"{gpkg_path.name}:{table_name} has SRS {srs_id}; expected EPSG:4326")
            geojson = convert_layer(connection, table_name, geometry_column, precision)
            output_path = output_path_for_layer(gpkg_path, table_name, len(layers), output_dir)
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(geojson, handle, ensure_ascii=False, separators=(",", ":"))
            output_paths.append(output_path)
    return output_paths


def write_preview(output_paths: list[Path], output_dir: Path) -> Path:
    layer_scripts = []
    for index, path in enumerate(output_paths):
        name = path.stem
        color = "#2276d2" if index == 0 else "#b42318"
        layer_scripts.append(
            f"""
        const layer{index} = await loadLayer("{path.name}", "{name}", "{color}");
        layers["{name}"] = layer{index};
        bounds.extend(layer{index}.getBounds());
            """.rstrip()
        )

    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>UK map GeoJSON preview</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
      html, body, #map {{ height: 100%; margin: 0; }}
      .leaflet-control-layers label {{ font: 13px system-ui, sans-serif; }}
      #message {{
        position: fixed;
        top: 12px;
        left: 50%;
        z-index: 1000;
        max-width: 680px;
        transform: translateX(-50%);
        border: 1px solid #d7dde7;
        border-radius: 6px;
        background: white;
        color: #1f2937;
        box-shadow: 0 8px 24px rgb(15 23 42 / 18%);
        padding: 12px 14px;
        font: 13px system-ui, sans-serif;
      }}
      #message code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
      #message[hidden] {{ display: none; }}
    </style>
  </head>
  <body>
    <div id="map"></div>
    <div id="message" hidden></div>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
      const map = L.map("map");
      L.tileLayer("https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
        maxZoom: 19,
        attribution: "&copy; OpenStreetMap contributors"
      }}).addTo(map);
      const layers = {{}};
      const bounds = L.latLngBounds();

      async function loadLayer(url, name, color) {{
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Failed to fetch ${{url}}`);
        const data = await response.json();
        const layer = L.geoJSON(data, {{
          style: {{ color, weight: 1, fillOpacity: 0.08 }},
          onEachFeature: (feature, layer) => {{
            const props = feature.properties || {{}};
            layer.bindPopup(Object.entries(props).map(([key, value]) => `<b>${{key}}</b>: ${{value}}`).join("<br>"));
          }}
        }}).addTo(map);
        return layer;
      }}

      function showError(error) {{
        const message = document.getElementById("message");
        message.hidden = false;
        message.innerHTML = `
          <strong>Preview could not load the GeoJSON files.</strong><br>
          ${{error.message}}<br><br>
          Run this from the project root and open <code>http://127.0.0.1:8766/preview.html</code>:<br>
          <code>.venv/bin/python -m http.server 8766 --directory local/uk_map/output</code>
        `;
      }}

      async function main() {{
        if (location.protocol === "file:") {{
          throw new Error("Browsers usually block fetch() from file:// pages.");
        }}

{chr(10).join(layer_scripts)}
        if (bounds.isValid()) map.fitBounds(bounds, {{ padding: [16, 16] }});
        L.control.layers(null, layers, {{ collapsed: false }}).addTo(map);
      }}

      main().catch(showError);
    </script>
  </body>
</html>
"""
    preview_path = output_dir / "preview.html"
    preview_path.write_text(html, encoding="utf-8")
    return preview_path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Convert local UK map GeoPackages to compact GeoJSON.")
    parser.add_argument("--source-dir", type=Path, default=root / "source")
    parser.add_argument("--output-dir", type=Path, default=root / "output")
    parser.add_argument("--precision", type=int, default=6, help="Coordinate decimal places to keep.")
    parser.add_argument("--no-preview", action="store_true", help="Skip writing output/preview.html.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gpkg_paths = sorted(args.source_dir.glob("*.gpkg"))
    if not gpkg_paths:
        raise SystemExit(f"No .gpkg files found in {args.source_dir}")

    output_paths = []
    for gpkg_path in gpkg_paths:
        converted = convert_gpkg(gpkg_path, args.output_dir, args.precision)
        output_paths.extend(converted)
        for output_path in converted:
            print(f"{gpkg_path.name} -> {output_path}")

    if not args.no_preview:
        preview_path = write_preview(output_paths, args.output_dir)
        print(f"Preview: {preview_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
