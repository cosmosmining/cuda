"""Build the rail graphs from per-trip stop sequences.

Two undirected graphs share the same nodes:

* **service graph** - an edge joins every pair of nodes served by the same
  train (a TGV Paris -> Lyon -> Marseille links all three pairs). A path of
  k edges therefore means k trains, i.e. k-1 changes.
* **segment graph** - only consecutive stops are joined; it approximates the
  physical corridors and is used for "new direct service on existing track"
  travel times.

Edge attributes: ``trains_per_week`` (both directions summed), ``trips``,
``time_min`` / ``time_median`` (minutes, weekly-weighted median), ``dist_km``.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from .geo import haversine_km


@dataclass
class RailNetwork:
    nodes: pd.DataFrame
    service_edges: pd.DataFrame
    segment_edges: pd.DataFrame
    trips: pd.DataFrame      # one row per trip: trip_id, feed, route_id, route_name, weekly, n_stops, origin, destination
    sequences: pd.DataFrame  # one row per (trip, node) call after merging: trip_id, route_id, pos, n, node_id, arr, dep, weekly

    def service_graph(self) -> nx.Graph:
        return to_networkx(self.nodes, self.service_edges)

    def segment_graph(self) -> nx.Graph:
        return to_networkx(self.nodes, self.segment_edges)


def collapse_to_nodes(trip_stops: pd.DataFrame) -> pd.DataFrame:
    """Merge consecutive calls at the same node (e.g. Berlin Spandau -> Berlin Hbf)
    and drop repeat visits later in the trip. Keeps first arrival, last departure."""
    df = trip_stops.sort_values(["trip_id", "seq"]).reset_index(drop=True)
    new = (df["node_id"] != df["node_id"].shift()) | (df["trip_id"] != df["trip_id"].shift())
    grp = new.cumsum()
    out = df.groupby(grp, sort=False).agg(
        feed=("feed", "first"), trip_id=("trip_id", "first"), route_id=("route_id", "first"), route_name=("route_name", "first"),
        node_id=("node_id", "first"), arr=("arr", "first"), dep=("dep", "last"), weekly=("weekly", "first"),
    )
    out = out.drop_duplicates(["trip_id", "node_id"], keep="first")
    out["pos"] = out.groupby("trip_id").cumcount()
    out["n"] = out.groupby("trip_id")["pos"].transform("size")
    return out[out["n"] >= 2].reset_index(drop=True)


def drop_cross_feed_duplicates(seq: pd.DataFrame) -> pd.DataFrame:
    """A cross-border train can appear in two operators' feeds. Trips with the
    same node sequence and first departure time (to 5 min) from *different*
    feeds are counted once, keeping the feed that reports more weekly runs."""
    if seq["feed"].nunique() < 2:
        return seq
    head = seq[seq["pos"] == 0].set_index("trip_id")
    key = seq.groupby("trip_id")["node_id"].agg(">".join) + "@" + (head["dep"] / 5).round().astype(int).astype(str)
    t = pd.DataFrame({"key": key, "feed": head["feed"].reindex(key.index), "weekly": head["weekly"].reindex(key.index)})
    per_feed = t.groupby(["key", "feed"])["weekly"].sum().reset_index()
    best = per_feed.sort_values("weekly", ascending=False).drop_duplicates("key")
    keep = t.reset_index().merge(best[["key", "feed"]], on=["key", "feed"])["trip_id"]
    return seq[seq["trip_id"].isin(set(keep))]


def trip_pairs(seq: pd.DataFrame, all_pairs: bool = True) -> pd.DataFrame:
    """Origin/destination pairs along each trip with in-vehicle time.

    Columns: u, v, time, weekly, trip_id, route_id, i, j (positions in the trip).
    Vectorised by grouping trips of equal length into 2-D arrays.
    """
    frames = []
    for n, grp in seq.groupby("n"):
        grp = grp.sort_values(["trip_id", "pos"])
        k = len(grp) // n
        nodes = grp["node_id"].to_numpy().reshape(k, n)
        arr = grp["arr"].to_numpy(dtype=float).reshape(k, n)
        dep = grp["dep"].to_numpy(dtype=float).reshape(k, n)
        weekly = grp["weekly"].to_numpy().reshape(k, n)[:, 0]
        trip = grp["trip_id"].to_numpy().reshape(k, n)[:, 0]
        route = grp["route_id"].to_numpy().reshape(k, n)[:, 0]
        if all_pairs:
            i, j = np.triu_indices(n, 1)
        else:
            i = np.arange(n - 1)
            j = i + 1
        frames.append(pd.DataFrame({
            "u": nodes[:, i].ravel(),
            "v": nodes[:, j].ravel(),
            "time": (arr[:, j] - dep[:, i]).ravel(),
            "weekly": np.repeat(weekly, len(i)),
            "trip_id": np.repeat(trip, len(i)),
            "route_id": np.repeat(route, len(i)),
            "i": np.tile(i, k),
            "j": np.tile(j, k),
        }))
    cols = ["u", "v", "time", "weekly", "trip_id", "route_id", "i", "j"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)


def aggregate_edges(pairs: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Undirected edges with summed weekly trains and weekly-weighted median time."""
    p = pairs[pairs["time"] > 0].copy()
    swap = p["u"] > p["v"]
    p["a"] = np.where(swap, p["v"], p["u"])
    p["b"] = np.where(swap, p["u"], p["v"])
    p = p.sort_values(["a", "b", "time"])
    g = p.groupby(["a", "b"], sort=False)
    cw = g["weekly"].cumsum()
    tot = g["weekly"].transform("sum")
    med = p[cw >= tot / 2].groupby(["a", "b"], sort=False)["time"].first()
    e = g.agg(trains_per_week=("weekly", "sum"), trips=("weekly", "size"), time_min=("time", "min"))
    e["time_median"] = med
    e = e.reset_index().rename(columns={"a": "u", "b": "v"})
    xy = nodes.set_index("node_id")[["lat", "lon"]]
    e["dist_km"] = haversine_km(xy.lat.reindex(e.u).to_numpy(), xy.lon.reindex(e.u).to_numpy(),
                                xy.lat.reindex(e.v).to_numpy(), xy.lon.reindex(e.v).to_numpy())
    e["speed_kmh"] = e["dist_km"] / (e["time_median"] / 60)
    return e.sort_values(["u", "v"]).reset_index(drop=True)


def build_network(trip_stops: pd.DataFrame, nodes: pd.DataFrame) -> RailNetwork:
    """``trip_stops`` must already carry a ``node_id`` column."""
    seq = drop_cross_feed_duplicates(collapse_to_nodes(trip_stops))
    service = aggregate_edges(trip_pairs(seq, all_pairs=True), nodes)
    segment = aggregate_edges(trip_pairs(seq, all_pairs=False), nodes)

    heads = seq[seq["pos"] == 0].set_index("trip_id")
    tails = seq[seq["pos"] == seq["n"] - 1].set_index("trip_id")
    trips = pd.DataFrame({
        "feed": heads["feed"], "route_id": heads["route_id"], "route_name": heads["route_name"], "weekly": heads["weekly"],
        "n_stops": heads["n"], "origin": heads["node_id"], "destination": tails["node_id"].reindex(heads.index),
        "duration_min": tails["arr"].reindex(heads.index) - heads["dep"],
    }).reset_index()

    calls = seq.groupby("node_id")["weekly"].sum()
    used = nodes[nodes["node_id"].isin(calls.index)].copy()
    used["weekly_calls"] = used["node_id"].map(calls).astype(int)
    sequences = seq[["trip_id", "route_id", "pos", "n", "node_id", "arr", "dep", "weekly"]].reset_index(drop=True)
    return RailNetwork(used.reset_index(drop=True), service, segment, trips, sequences)


def to_networkx(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for rec in nodes.to_dict("records"):
        G.add_node(rec["node_id"], **{k: v for k, v in rec.items() if k != "node_id"})
    for rec in edges.to_dict("records"):
        G.add_edge(rec["u"], rec["v"], **{k: v for k, v in rec.items() if k not in ("u", "v")})
    return G
