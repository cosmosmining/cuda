"""Which routes and corridors matter most? Knock-out (removal) experiments.

* **Route importance** - cancel every trip of one route (line) and measure the
  increase in demand-weighted mean generalized travel time.
* **Corridor importance** - close one physical segment (consecutive stops):
  every trip running over it loses all origin-destination pairs that span the
  segment (the train is cut in two), as with a line closure.

Only the service edges touched by the removed trips are re-aggregated, so each
experiment costs one multi-source Dijkstra.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .graph import RailNetwork, trip_pairs
from .metrics import CostModel, edge_cost


class ServiceAggregator:
    """Recompute service-edge costs after removing a subset of trip pairs."""

    def __init__(self, net: RailNetwork):
        self.edges = net.service_edges
        key = {(a, b): k for k, (a, b) in enumerate(zip(self.edges.u, self.edges.v))}
        p = trip_pairs(net.sequences, all_pairs=True)
        p = p[p["time"] > 0].reset_index(drop=True)
        a = np.where(p.u < p.v, p.u, p.v)
        b = np.where(p.u < p.v, p.v, p.u)
        p["e"] = [key[x] for x in zip(a, b)]
        self.pairs = p
        self.by_trip = p.groupby("trip_id").indices
        self.by_route = p.groupby("route_id").indices
        self.by_edge = p.groupby("e").indices
        seq = net.sequences
        self.calls = seq.assign(nxt=seq.groupby("trip_id")["node_id"].shift(-1))[["trip_id", "pos", "node_id", "nxt"]]
        self.base_costs = edge_cost(self.edges.time_median.to_numpy(), self.edges.trains_per_week.to_numpy())

    def costs_without(self, removed_rows: np.ndarray) -> np.ndarray:
        costs = self.base_costs.copy()
        removed = np.zeros(len(self.pairs), dtype=bool)
        removed[removed_rows] = True
        for e in np.unique(self.pairs["e"].to_numpy()[removed_rows]):
            rows = self.by_edge[e]
            rows = rows[~removed[rows]]
            if len(rows) == 0:
                costs[e] = np.inf
                continue
            t = self.pairs["time"].to_numpy()[rows]
            w = self.pairs["weekly"].to_numpy()[rows]
            o = np.argsort(t)
            med = t[o][np.searchsorted(np.cumsum(w[o]), w.sum() / 2)]
            costs[e] = edge_cost(med, w.sum())
        return costs

    def segment_rows(self, a: str, b: str) -> np.ndarray:
        """Pair rows whose journey runs over the consecutive segment a-b (either direction)."""
        s = self.calls
        hit = s[((s.node_id == a) & (s.nxt == b)) | ((s.node_id == b) & (s.nxt == a))]
        rows = []
        i_all = self.pairs["i"].to_numpy()
        j_all = self.pairs["j"].to_numpy()
        for trip, k in zip(hit.trip_id, hit.pos):
            r = self.by_trip.get(trip)
            if r is not None:
                rows.append(r[(i_all[r] <= k) & (j_all[r] > k)])
        return np.concatenate(rows) if rows else np.array([], dtype=int)


def _impact(model: CostModel, costs: np.ndarray, base_T: float, base_R: np.ndarray) -> dict:
    R = model.od_matrix(costs)
    J0, J1 = model.journey_minutes(base_R), model.journey_minutes(R)
    lost = (np.isfinite(J0) & ~np.isfinite(J1)) & (model.W > 0)
    E0, E1 = model.effective_minutes(base_R), model.effective_minutes(R)
    return {
        "delta_mean_min": model.mean_time(R) - base_T,
        "demand_disconnected": float(model.W[lost].sum()),
        "demand_slower": float(model.W[(E1 - E0) > 1.0].sum()),
        "rail_share_lost": model.rail_share(base_R) - model.rail_share(R),
    }


def route_importance(net: RailNetwork, model: CostModel, agg: ServiceAggregator | None = None) -> pd.DataFrame:
    agg = agg or ServiceAggregator(net)
    base_R = model.od_matrix(agg.base_costs)
    base_T = model.mean_time(base_R)
    info = net.trips.groupby("route_id").agg(route_name=("route_name", "first"), trips_per_week=("weekly", "sum"),
                                              origin=("origin", "first"), destination=("destination", "first"))
    names = net.nodes.set_index("node_id")["name"]
    rows = []
    for rid, pair_rows in agg.by_route.items():
        rec = {"route_id": rid, **_impact(model, agg.costs_without(pair_rows), base_T, base_R)}
        rows.append(rec)
    out = pd.DataFrame(rows).merge(info, left_on="route_id", right_index=True)
    out["origin"] = out["origin"].map(names)
    out["destination"] = out["destination"].map(names)
    out["delta_mean_pct"] = 100 * out["delta_mean_min"] / base_T
    return out.sort_values("delta_mean_min", ascending=False).reset_index(drop=True)


def corridor_importance(net: RailNetwork, model: CostModel, agg: ServiceAggregator | None = None,
                        top_k: int | None = None) -> pd.DataFrame:
    """Close each physical segment in turn (``top_k`` busiest only, if given)."""
    agg = agg or ServiceAggregator(net)
    base_R = model.od_matrix(agg.base_costs)
    base_T = model.mean_time(base_R)
    seg = net.segment_edges.sort_values("trains_per_week", ascending=False)
    if top_k:
        seg = seg.head(top_k)
    names = net.nodes.set_index("node_id")["name"]
    rows = []
    for a, b, tpw, km in zip(seg.u, seg.v, seg.trains_per_week, seg.dist_km):
        cut = agg.segment_rows(a, b)
        rec = {"u": a, "v": b, "segment": f"{names[a]} - {names[b]}", "trains_per_week": tpw, "dist_km": km,
               **_impact(model, agg.costs_without(cut), base_T, base_R)}
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["delta_mean_pct"] = 100 * out["delta_mean_min"] / base_T
    return out.sort_values("delta_mean_min", ascending=False).reset_index(drop=True)
