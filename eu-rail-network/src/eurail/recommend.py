"""Which new routes would help most? Greedy budgeted network design.

Objective: the demand-weighted mean generalized journey time T between OD zones
(``metrics.CostModel.mean_time``). Adding a direct link (u, v) with raw cost c
changes the OD cost matrix exactly by

    R'_ij = min(R_ij, R_iu + c + R_vj, R_iv + c + R_uj)

because positive-cost shortest paths use a new edge at most once. That lets us
score every candidate on the OD sub-matrix without re-running Dijkstra, and
prune any candidate with c >= R_uv (it cannot shorten any path).

Two scenarios:

* **New high-speed lines** - a new line between two cities, run at
  ``speed_kmh`` average commercial speed over ``detour`` x the straight-line
  distance, with ``trains_per_day`` per direction. Cost = construction:
  ``meur_per_km`` per km on land plus ``meur_per_sea_km`` for the part of the
  straight line over sea (fixed links: tunnels/bridges), when a land mask is given.
* **New direct services on existing track** - a through train between two
  cities that have no direct service today, taking the fastest existing track
  path (segment graph), with ``trains_per_day`` per direction. Cost = daily
  train-km (operating cost proxy); no construction.

The greedy picks, each round, the candidate with the best gain / cost, updates
R and repeats. (T is a monotone set function of added edges; greedy ratio
selection is the standard heuristic for such budgeted problems.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .graph import RailNetwork
from .metrics import CostModel, edge_cost


@dataclass
class HSRScenario:
    speed_kmh: float = 200.0       # average commercial speed incl. stops
    detour: float = 1.2            # line length / straight-line distance
    access_min: float = 10.0       # terminal time on top of running time
    trains_per_day: int = 16       # per direction
    meur_per_km: float = 25.0      # construction cost on land, million EUR per km
    meur_per_sea_km: float = 150.0 # fixed-link (tunnel/bridge) cost over sea
    min_pop: int = 200_000
    min_km: float = 150.0
    max_km: float = 800.0


@dataclass
class DirectServiceScenario:
    trains_per_day: int = 8        # per direction
    dwell_min: float = 2.0
    min_pop: int = 200_000
    min_km: float = 100.0
    max_km: float = 1000.0


def _gain(model: CostModel, R: np.ndarray, p: int, q: int, c: float) -> tuple[float, np.ndarray]:
    via = np.minimum(R[:, [p]] + c + R[[q], :], R[:, [q]] + c + R[[p], :])
    Rn = np.minimum(R, via)
    return float((model.W * (model.effective_minutes(R) - model.effective_minutes(Rn))).sum()), Rn


def _pairs(nodes: pd.DataFrame, model: CostModel, min_pop: int, min_km: float, max_km: float):
    pop = nodes["population"].to_numpy()[model.od]
    big = np.flatnonzero(pop >= min_pop)
    for x in range(len(big)):
        for y in range(x + 1, len(big)):
            p, q = big[x], big[y]
            if min_km <= model.dist_km[p, q] <= max_km:
                yield p, q


def hsr_candidates(nodes: pd.DataFrame, model: CostModel, R: np.ndarray, sc: HSRScenario = HSRScenario(),
                   land=None) -> pd.DataFrame:
    """``land``: optional ``geo.LandMask``; cost is in billion EUR."""
    lat = nodes["lat"].to_numpy()[model.od]
    lon = nodes["lon"].to_numpy()[model.od]
    rows = []
    for p, q in _pairs(nodes, model, sc.min_pop, sc.min_km, sc.max_km):
        d = model.dist_km[p, q]
        ride = sc.detour * d / sc.speed_kmh * 60 + sc.access_min
        c = float(edge_cost(ride, sc.trains_per_day * 14))
        if c < R[p, q]:
            sea = land.sea_km(lat[p], lon[p], lat[q], lon[q]) if land is not None else 0.0
            capex = (sc.detour * (d - sea) * sc.meur_per_km + sea * sc.meur_per_sea_km) / 1000.0
            rows.append((p, q, d, sea, ride, c, capex))
    return pd.DataFrame(rows, columns=["p", "q", "dist_km", "sea_km", "ride_min", "raw_cost", "cost"])


def direct_service_candidates(net: RailNetwork, model: CostModel, R: np.ndarray,
                              sc: DirectServiceScenario = DirectServiceScenario()) -> pd.DataFrame:
    seg = net.segment_edges
    idx = {k: i for i, k in enumerate(model.node_ids)}
    g = csr_matrix((seg.time_min.to_numpy() + sc.dwell_min, (seg.u.map(idx), seg.v.map(idx))), shape=(model.n, model.n))
    gk = csr_matrix((seg.dist_km.to_numpy(), (seg.u.map(idx), seg.v.map(idx))), shape=(model.n, model.n))
    T, pred = dijkstra(g, directed=False, indices=model.od, return_predecessors=True)
    served = set(zip(net.service_edges.u, net.service_edges.v)) | set(zip(net.service_edges.v, net.service_edges.u))
    ids = model.node_ids
    rows = []
    for p, q in _pairs(net.nodes, model, sc.min_pop, sc.min_km, sc.max_km):
        a, b = model.od[p], model.od[q]
        if (ids[a], ids[b]) in served or not np.isfinite(T[p, b]):
            continue
        ride = T[p, b] - sc.dwell_min
        c = float(edge_cost(ride, sc.trains_per_day * 14))
        if c >= R[p, q]:
            continue
        km, t = 0.0, b
        while t != a:  # track length along the fastest path
            s = pred[p, t]
            km += gk[s, t] or gk[t, s]
            t = s
        rows.append((p, q, model.dist_km[p, q], ride, c, 2 * sc.trains_per_day * km / 1000.0))
    # cost = thousand train-km per day
    return pd.DataFrame(rows, columns=["p", "q", "dist_km", "ride_min", "raw_cost", "cost"])


def score_candidates(model: CostModel, R: np.ndarray, cands: pd.DataFrame) -> pd.DataFrame:
    """Independent (one-at-a-time) gain of every candidate."""
    out = cands.copy()
    out["gain_min"] = [_gain(model, R, int(p), int(q), c)[0] for p, q, c in zip(cands.p, cands.q, cands.raw_cost)]
    out["gain_per_cost"] = out.gain_min / out.cost
    return out.sort_values("gain_min", ascending=False).reset_index(drop=True)


def greedy_select(model: CostModel, R: np.ndarray, cands: pd.DataFrame, k: int = 10,
                  budget: float | None = None, by_ratio: bool = True) -> pd.DataFrame:
    R = R.copy()
    base = model.mean_time(R)
    remaining = cands.reset_index(drop=True).copy()
    picks, spent = [], 0.0
    for rnd in range(k):
        if remaining.empty:
            break
        best, best_score, best_R = None, 0.0, None
        for i, (p, q, c, cost) in enumerate(zip(remaining.p, remaining.q, remaining.raw_cost, remaining.cost)):
            if budget is not None and spent + cost > budget:
                continue
            if c >= R[p, q]:
                continue
            g, Rn = _gain(model, R, int(p), int(q), c)
            s = g / cost if by_ratio else g
            if s > best_score:
                best, best_score, best_R, best_gain = i, s, Rn, g
        if best is None:
            break
        row = remaining.iloc[best].to_dict()
        R = best_R
        spent += row["cost"]
        picks.append({**row, "round": rnd + 1, "gain_min": best_gain, "cum_cost": spent,
                      "mean_time_after": model.mean_time(R), "cum_gain_min": base - model.mean_time(R)})
        remaining = remaining.drop(index=best).reset_index(drop=True)
    return pd.DataFrame(picks)


def label(df: pd.DataFrame, nodes: pd.DataFrame, model: CostModel) -> pd.DataFrame:
    names = nodes["name"].to_numpy()[model.od]
    ctry = nodes["country"].to_numpy()[model.od]
    out = df.copy()
    out.insert(0, "city_a", names[out.p.astype(int)])
    out.insert(1, "city_b", names[out.q.astype(int)])
    out.insert(2, "countries", [f"{ctry[int(p)]}-{ctry[int(q)]}" for p, q in zip(out.p, out.q)])
    return out
