"""Graph metrics: structure, centrality, generalized travel time, accessibility.

Generalized travel time
-----------------------
Riding a service-graph edge costs

    in-vehicle time (weekly-weighted median)
    + expected wait = min(headway / 2, MAX_WAIT)   (headway over a 16 h day)
    + TRANSFER_PENALTY                              (charged per boarding)

and the cost of a journey is the shortest-path sum minus one penalty (the first
boarding is not a change). Frequency therefore matters, as it does for real
travellers, and every change costs extra.

Demand
------
Origin-destination weights follow a gravity model between "OD zones" (nodes
with population >= ``min_pop``): W_ij = P_i P_j / d_ij^gamma for
d_ij >= ``min_km`` (100 km, the usual threshold for long-distance travel;
shorter trips are commuter traffic an intercity network is not built for),
normalised to sum to 1.

The headline accessibility measure is the demand-weighted mean generalized
journey time (minutes, lower is better). Each pair's rail time is capped by a
road alternative, 30 min + 1.25 x distance at 80 km/h, so that a pair with poor
or no rail service counts as travelled by road rather than as an arbitrary
infinite penalty; improvements only count where rail becomes competitive.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .geo import pairwise_km

TRANSFER_PENALTY_MIN = 15.0
SERVICE_SPAN_MIN = 16 * 60.0
MAX_WAIT_MIN = 60.0
ROAD_ACCESS_MIN, ROAD_DETOUR, ROAD_SPEED_KMH = 30.0, 1.25, 80.0


def road_alternative_min(dist_km) -> np.ndarray:
    return ROAD_ACCESS_MIN + 60.0 * ROAD_DETOUR * np.asarray(dist_km, dtype=float) / ROAD_SPEED_KMH


def edge_cost(time_min, trains_per_week, transfer_penalty: float = TRANSFER_PENALTY_MIN,
              max_wait: float = MAX_WAIT_MIN) -> np.ndarray:
    per_dir_daily = np.asarray(trains_per_week, dtype=float) / 14.0
    wait = np.minimum(SERVICE_SPAN_MIN / np.maximum(per_dir_daily, 1e-9) / 2.0, max_wait)
    return np.asarray(time_min, dtype=float) + wait + transfer_penalty


@dataclass
class CostModel:
    """Everything needed to evaluate accessibility on the service graph quickly."""
    node_ids: np.ndarray        # all nodes, graph order
    od: np.ndarray              # indices of OD-zone nodes into node_ids
    W: np.ndarray               # |od| x |od| demand weights (sum 1, zero diagonal)
    dist_km: np.ndarray         # |od| x |od| great-circle distances
    u: np.ndarray               # edge endpoints (indices into node_ids)
    v: np.ndarray
    penalty: float = TRANSFER_PENALTY_MIN

    @property
    def n(self) -> int:
        return len(self.node_ids)

    def raw_costs(self, costs: np.ndarray, sources: np.ndarray | None = None, u=None, v=None) -> np.ndarray:
        """Shortest raw costs (penalty per boarding included) from ``sources`` (default: OD zones)
        to all nodes, for edge costs ``costs`` on edges (u, v)."""
        u = self.u if u is None else u
        v = self.v if v is None else v
        keep = np.isfinite(costs)
        g = csr_matrix((costs[keep], (u[keep], v[keep])), shape=(self.n, self.n))
        return dijkstra(g, directed=False, indices=self.od if sources is None else sources)

    def od_matrix(self, costs: np.ndarray, **kw) -> np.ndarray:
        """Raw cost matrix between OD zones."""
        return self.raw_costs(costs, **kw)[:, self.od]

    def journey_minutes(self, R_od: np.ndarray) -> np.ndarray:
        """Rail-only generalized journey times (inf if unreachable)."""
        J = R_od - self.penalty
        np.fill_diagonal(J, 0.0)
        return J

    def effective_minutes(self, R_od: np.ndarray) -> np.ndarray:
        """Rail journey time capped by the road alternative."""
        return np.minimum(self.journey_minutes(R_od), road_alternative_min(self.dist_km))

    def mean_time(self, R_od: np.ndarray) -> float:
        """Demand-weighted mean generalized journey time (minutes)."""
        return float((self.W * self.effective_minutes(R_od)).sum())

    def rail_share(self, R_od: np.ndarray) -> float:
        """Demand share for which rail beats the road alternative."""
        return float(self.W[self.journey_minutes(R_od) < road_alternative_min(self.dist_km)].sum())

    def share_within(self, R_od: np.ndarray, minutes: float) -> float:
        return float(self.W[self.journey_minutes(R_od) <= minutes].sum())


def gravity_weights(pop: np.ndarray, dist_km: np.ndarray, gamma: float = 2.0, min_km: float = 100.0) -> np.ndarray:
    P = np.asarray(pop, dtype=float)
    with np.errstate(divide="ignore"):
        W = np.outer(P, P) / np.maximum(dist_km, min_km) ** gamma
    W[dist_km < min_km] = 0.0
    np.fill_diagonal(W, 0.0)
    return W / W.sum()


def build_cost_model(nodes: pd.DataFrame, edges: pd.DataFrame, min_pop: int = 100_000,
                     gamma: float = 2.0, min_km: float = 100.0) -> CostModel:
    ids = nodes["node_id"].to_numpy()
    index = {k: i for i, k in enumerate(ids)}
    od = np.flatnonzero(nodes["population"].to_numpy() >= min_pop)
    sub = nodes.iloc[od]
    D = pairwise_km(sub.lat.to_numpy(), sub.lon.to_numpy())
    W = gravity_weights(sub.population.to_numpy(), D, gamma, min_km)
    return CostModel(ids, od, W, D, edges["u"].map(index).to_numpy(), edges["v"].map(index).to_numpy())


# --------------------------------------------------------------------- structure
def network_summary(G: nx.Graph, weight_key: str = "trains_per_week") -> dict:
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    lcc = G.subgraph(comps[0])
    deg = np.array([d for _, d in G.degree()])
    strength = np.array([d for _, d in G.degree(weight=weight_key)])
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "components": len(comps),
        "component_sizes": [len(c) for c in comps[:8]],
        "largest_component_share": len(comps[0]) / G.number_of_nodes(),
        "mean_degree": float(deg.mean()),
        "max_degree": int(deg.max()),
        "mean_strength": float(strength.mean()),
        "avg_clustering": nx.average_clustering(G),
        "transitivity": nx.transitivity(G),
        "degree_assortativity": nx.degree_assortativity_coefficient(G),
        "lcc_diameter_hops": nx.diameter(lcc),
        "lcc_mean_hops": nx.average_shortest_path_length(lcc),
    }


def flow_assignment(model: CostModel, costs: np.ndarray, u=None, v=None) -> tuple[np.ndarray, np.ndarray]:
    """All-or-nothing assignment of the gravity demand onto shortest generalized paths.

    Returns (node_through, edge_load): the demand share passing *through* each
    node as an intermediate (i.e. changing trains or staying on board) and the
    demand share carried by each edge.
    """
    u = model.u if u is None else u
    v = model.v if v is None else v
    keep = np.isfinite(costs)
    g = csr_matrix((costs[keep], (u[keep], v[keep])), shape=(model.n, model.n))
    R, pred = dijkstra(g, directed=False, indices=model.od, return_predecessors=True)
    edge_index = {}
    for k, (a, b) in enumerate(zip(u, v)):
        edge_index[(a, b)] = k
        edge_index[(b, a)] = k
    node_through = np.zeros(model.n)
    edge_load = np.zeros(len(u))
    for si in range(len(model.od)):
        f = np.zeros(model.n)
        f[model.od] = model.W[si]
        order = np.argsort(-np.where(np.isfinite(R[si]), R[si], -1.0))
        for t in order:
            p = pred[si, t]
            if p < 0 or f[t] == 0.0:
                continue
            edge_load[edge_index[(p, t)]] += f[t]
            f[p] += f[t]
            if p != model.od[si]:
                node_through[p] += f[t]
    return node_through, edge_load


def centrality_table(nodes: pd.DataFrame, service: pd.DataFrame, segment: pd.DataFrame,
                     model: CostModel, costs: np.ndarray) -> pd.DataFrame:
    Gs = nx.Graph()
    Gs.add_nodes_from(nodes["node_id"])
    for a, b, c, w in zip(service.u, service.v, costs, service.trains_per_week):
        Gs.add_edge(a, b, cost=c, trains=w)
    Gt = nx.Graph()
    Gt.add_nodes_from(nodes["node_id"])
    for a, b, t in zip(segment.u, segment.v, segment.time_min):
        Gt.add_edge(a, b, time=t)

    through, _ = flow_assignment(model, costs)
    R_all = model.raw_costs(costs, sources=np.arange(model.n))
    J = R_all - model.penalty
    np.fill_diagonal(J, np.inf)
    harmonic = (1.0 / J).sum(axis=1) / (model.n - 1) * 60.0  # mean of 1/hours

    # Accessibility of each OD zone: demand-weighted mean journey time to all others.
    R_od = R_all[np.ix_(model.od, model.od)]
    Jod = model.effective_minutes(R_od)
    rowW = model.W.sum(axis=1)
    access = np.full(model.n, np.nan)
    access[model.od] = np.where(rowW > 0, (model.W * Jod).sum(axis=1) / np.where(rowW > 0, rowW, 1), np.nan)

    out = nodes[["node_id", "name", "kind", "country", "population", "lat", "lon", "weekly_calls"]].copy()
    out["degree"] = out["node_id"].map(dict(Gs.degree()))
    out["strength"] = out["node_id"].map(dict(Gs.degree(weight="trains")))
    out["betweenness"] = out["node_id"].map(nx.betweenness_centrality(Gs, weight="cost"))
    out["segment_betweenness"] = out["node_id"].map(nx.betweenness_centrality(Gt, weight="time"))
    out["harmonic_closeness"] = harmonic
    out["pagerank"] = out["node_id"].map(nx.pagerank(Gs, weight="trains"))
    out["transfer_flow"] = through
    out["accessibility_min"] = access
    return out.sort_values("betweenness", ascending=False).reset_index(drop=True)
