"""Build ``data/reference/europe_borders.geojson`` from Natural Earth (public domain).

The source is the ``world-atlas`` npm package (Natural Earth 1:50m countries,
TopoJSON). It is only needed to *regenerate* the committed GeoJSON:

    npm pack world-atlas@2 && tar xzf world-atlas-*.tgz
    python scripts/make_borders.py package/countries-50m.json

The output is clipped to countries intersecting a Europe bounding box and
lightly simplified so the file stays small; it is used purely as a map
background.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BBOX = (-25.0, 34.0, 45.0, 72.0)  # lon_min, lat_min, lon_max, lat_max
MIN_STEP_DEG = 0.03  # drop vertices closer than this to the previous one


def decode_arcs(topo: dict) -> list[list[tuple[float, float]]]:
    sx, sy = topo["transform"]["scale"]
    tx, ty = topo["transform"]["translate"]
    arcs = []
    for arc in topo["arcs"]:
        x = y = 0
        pts = []
        for dx, dy in arc:
            x += dx
            y += dy
            pts.append((x * sx + tx, y * sy + ty))
        arcs.append(pts)
    return arcs


def ring(arc_ids: list[int], arcs) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for i in arc_ids:
        seg = arcs[i] if i >= 0 else arcs[~i][::-1]
        pts.extend(seg if not pts else seg[1:])
    return pts


def unwrap(pts):
    """Rings crossing the antimeridian (e.g. Russia) would fill a band across the
    whole map; shift their western points by +360 so the ring stays contiguous."""
    xs = [p[0] for p in pts]
    if max(xs) - min(xs) > 180:
        return [(x + 360 if x < 0 else x, y) for x, y in pts]
    return pts


def simplify(pts):
    out = [pts[0]]
    for p in pts[1:-1]:
        if abs(p[0] - out[-1][0]) + abs(p[1] - out[-1][1]) >= MIN_STEP_DEG:
            out.append(p)
    out.append(pts[-1])
    return [[round(x, 3), round(y, 3)] for x, y in out]


def intersects(pts) -> bool:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return not (max(xs) < BBOX[0] or min(xs) > BBOX[2] or max(ys) < BBOX[1] or min(ys) > BBOX[3])


def main(src: str, dst: str) -> None:
    topo = json.loads(Path(src).read_text())
    arcs = decode_arcs(topo)
    features = []
    for geom in topo["objects"]["countries"]["geometries"]:
        polys = geom["arcs"] if geom["type"] == "MultiPolygon" else [geom["arcs"]] if geom["type"] == "Polygon" else []
        keep = []
        for poly in polys:
            rings = [unwrap(ring(r, arcs)) for r in poly]
            if rings and intersects(rings[0]):
                keep.append([simplify(r) for r in rings if len(r) >= 4])
        if keep:
            features.append({
                "type": "Feature",
                "properties": {"name": geom.get("properties", {}).get("name", "")},
                "geometry": {"type": "MultiPolygon", "coordinates": keep},
            })
    Path(dst).write_text(json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":")))
    print(f"wrote {len(features)} countries -> {dst}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parents[1]
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else str(here / "data/reference/europe_borders.geojson"))
