"""End-to-end network construction: GTFS feeds -> city nodes -> rail graphs."""
from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from . import gtfs
from .cities import assign_stations, build_nodes, load_cities
from .graph import RailNetwork, build_network


@dataclass
class FeedSpec:
    name: str
    path: Path
    url: str | None = None
    route_types: frozenset[int] | None = gtfs.INTERCITY_ROUTE_TYPES
    name_regex: str | None = None
    enabled: bool = True
    description: str = ""


def load_feed_specs(config: str | Path, root: str | Path = ".", raw_dir: str | Path = "data/raw") -> list[FeedSpec]:
    cfg = tomllib.loads(Path(config).read_text())
    root = Path(root)
    specs = []
    for f in cfg.get("feed", []):
        path = root / f.get("path", Path(raw_dir) / f"{f['name']}.zip")
        specs.append(FeedSpec(
            name=f["name"], path=path, url=f.get("url"),
            route_types=frozenset(f["route_types"]) if "route_types" in f else gtfs.INTERCITY_ROUTE_TYPES,
            name_regex=f.get("name_regex"), enabled=f.get("enabled", True), description=f.get("description", ""),
        ))
    return specs


@dataclass
class BuildResult:
    network: RailNetwork
    stations: pd.DataFrame
    meta: dict = field(default_factory=dict)


def build_from_feeds(specs: list[FeedSpec], week: date | None = None, min_city_population: int = 15000) -> BuildResult:
    trip_stops, stations, meta = [], [], {"feeds": []}
    for spec in specs:
        feed = gtfs.filter_routes(gtfs.read_feed(spec.path, spec.name), spec.route_types, spec.name_regex)
        wk = week or gtfs.choose_reference_week(feed)
        ts = gtfs.trip_stop_table(feed, wk)
        st, _ = gtfs.stations_table(feed)
        stations.append(st[st["station_id"].isin(ts["station_id"])])
        trip_stops.append(ts)
        try:
            shown = str(Path(spec.path).resolve().relative_to(Path.cwd().resolve()))
        except ValueError:
            shown = Path(spec.path).name
        meta["feeds"].append({"name": spec.name, "path": shown, "reference_week": wk.isoformat(),
                              "trips": int(ts["trip_id"].nunique()), "stations": int(ts["station_id"].nunique())})
    if not trip_stops:
        raise ValueError("no feeds to build from")
    ts = pd.concat(trip_stops, ignore_index=True)
    st = pd.concat(stations, ignore_index=True)

    cities = load_cities(min_city_population)
    assigned = assign_stations(st, cities)
    nodes = build_nodes(assigned, cities)
    ts["node_id"] = ts["station_id"].map(assigned.set_index("station_id")["node_id"])
    net = build_network(ts.dropna(subset=["node_id"]), nodes)
    meta.update({
        "nodes": len(net.nodes), "city_nodes": int((net.nodes.kind == "city").sum()),
        "service_edges": len(net.service_edges), "segment_edges": len(net.segment_edges), "trips": len(net.trips),
        "stations": len(assigned), "min_city_population": min_city_population,
    })
    cols = ["station_id", "feed", "name", "lat", "lon", "node_id", "node_kind"]
    return BuildResult(net, assigned[cols], meta)


def save(result: BuildResult, outdir: str | Path) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    net = result.network
    net.nodes.to_csv(out / "nodes.csv", index=False)
    net.service_edges.to_csv(out / "edges_service.csv", index=False, float_format="%.3f")
    net.segment_edges.to_csv(out / "edges_segment.csv", index=False, float_format="%.3f")
    net.trips.to_csv(out / "trips.csv", index=False)
    net.sequences.to_csv(out / "trip_sequences.csv.gz", index=False, float_format="%.1f")
    result.stations.to_csv(out / "stations.csv", index=False)
    (out / "build_meta.json").write_text(json.dumps(result.meta, indent=2))


def load(outdir: str | Path) -> RailNetwork:
    d = Path(outdir)
    return RailNetwork(
        nodes=pd.read_csv(d / "nodes.csv", keep_default_na=False, na_values=[""]),
        service_edges=pd.read_csv(d / "edges_service.csv"),
        segment_edges=pd.read_csv(d / "edges_segment.csv"),
        trips=pd.read_csv(d / "trips.csv"),
        sequences=pd.read_csv(d / "trip_sequences.csv.gz"),
    )
