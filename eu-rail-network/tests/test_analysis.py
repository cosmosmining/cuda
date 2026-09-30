import networkx as nx
import numpy as np
import pandas as pd
import pytest

from eurail import metrics, recommend
from eurail.graph import build_network
from eurail.importance import ServiceAggregator, corridor_importance, route_importance


def _line_network():
    """Four cities on a line, 150 km apart; one train A-B-C-D and one shuttle C-D."""
    nodes = pd.DataFrame({"node_id": list("ABCD"), "name": list("ABCD"), "kind": "city",
                          "lat": [50.0] * 4, "lon": [0.0, 2.1, 4.2, 6.3], "population": [1_000_000, 500_000, 800_000, 300_000],
                          "country": ["XX", "XX", "YY", "YY"], "n_stations": 1, "stations": ""})
    rows = []
    for k in range(10):  # 10 trains per direction on the long line
        for tid, seq in ((f"l{k}", "ABCD"), (f"r{k}", "DCBA")):
            rows += [("f", tid, "LONG", "L", i, n, 60 * i, 60 * i + 1, 7) for i, n in enumerate(seq)]
    for k in range(5):
        rows += [("f", f"s{k}", "SHUT", "S", i, n, 50 * i, 50 * i, 7) for i, n in enumerate("CD")]
    ts = pd.DataFrame(rows, columns=["feed", "trip_id", "route_id", "route_name", "seq", "node_id", "arr", "dep", "weekly"])
    return build_network(ts, nodes)


def test_edge_cost_components():
    # 140 trains/week = 10 per direction per day -> headway 96 min -> wait 48
    assert metrics.edge_cost(60, 140) == pytest.approx(60 + 48 + metrics.TRANSFER_PENALTY_MIN)
    assert metrics.edge_cost(60, 7) == pytest.approx(60 + metrics.MAX_WAIT_MIN + metrics.TRANSFER_PENALTY_MIN)


def test_cost_model_matches_networkx_and_subtracts_one_penalty():
    net = _line_network()
    model = metrics.build_cost_model(net.nodes, net.service_edges, min_pop=0, min_km=0)
    costs = metrics.edge_cost(net.service_edges.time_median, net.service_edges.trains_per_week)
    G = nx.Graph()
    for (a, b), c in zip(zip(net.service_edges.u, net.service_edges.v), costs):
        G.add_edge(a, b, w=c)
    R = model.od_matrix(costs)
    ids = list(net.nodes.node_id)
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            assert R[i, j] == pytest.approx(nx.shortest_path_length(G, a, b, weight="w"))
    J = model.journey_minutes(R)
    assert J[0, 3] == pytest.approx(R[0, 3] - metrics.TRANSFER_PENALTY_MIN)
    assert np.all(model.effective_minutes(R) <= metrics.road_alternative_min(model.dist_km) + 1e-9)


def test_flow_assignment_conserves_demand():
    net = _line_network()
    model = metrics.build_cost_model(net.nodes, net.service_edges, min_pop=0, min_km=0)
    costs = metrics.edge_cost(net.service_edges.time_median, net.service_edges.trains_per_week)
    through, load = metrics.flow_assignment(model, costs)
    # every OD pair is served by a direct edge on the clique -> each unit of demand uses exactly one edge
    assert load.sum() == pytest.approx(1.0)
    assert through.sum() == pytest.approx(0.0)


def test_incremental_gain_equals_full_recompute():
    rng = np.random.default_rng(1)
    n = 9
    lat, lon = 45 + rng.random(n) * 5, rng.random(n) * 10
    nodes = pd.DataFrame({"node_id": [f"n{i}" for i in range(n)], "name": [f"n{i}" for i in range(n)],
                          "lat": lat, "lon": lon, "population": rng.integers(2e5, 2e6, n), "country": "XX"})
    edges = pd.DataFrame([(f"n{i}", f"n{i + 1}") for i in range(n - 1)] + [("n0", "n4")], columns=["u", "v"])
    model = metrics.build_cost_model(nodes, edges, min_pop=0, min_km=0)
    costs = rng.uniform(40, 200, len(edges))
    R = model.od_matrix(costs)
    p, q, c = 2, 7, 90.0
    gain, Rn = recommend._gain(model, R, p, q, c)
    u2 = np.append(model.u, model.od[p])
    v2 = np.append(model.v, model.od[q])
    R_full = model.od_matrix(np.append(costs, c), u=u2, v=v2)
    assert np.allclose(Rn, R_full)
    assert gain == pytest.approx(model.mean_time(R) - model.mean_time(R_full))


def test_greedy_first_pick_is_best_ratio_and_gains_add_up():
    nodes = pd.DataFrame({"node_id": list("ABCDE"), "name": list("ABCDE"), "lat": [50.0] * 5,
                          "lon": [0.0, 2.0, 4.0, 6.0, 8.0], "population": [10**6] * 5, "country": "XX"})
    edges = pd.DataFrame({"u": list("ABCD"), "v": list("BCDE")})
    model = metrics.build_cost_model(nodes, edges, min_pop=0, min_km=0)
    R = model.od_matrix(np.full(4, 200.0))
    cands = pd.DataFrame({"p": [0, 0, 1], "q": [4, 2, 3], "raw_cost": [150.0, 150.0, 120.0], "cost": [1.0, 3.0, 2.0]})
    scored = recommend.score_candidates(model, R, cands)
    best = scored.loc[scored.gain_per_cost.idxmax()]
    picks = recommend.greedy_select(model, R, cands, k=3)
    assert (picks.p.iloc[0], picks.q.iloc[0]) == (best.p, best.q)
    assert picks.gain_min.iloc[0] == pytest.approx(best.gain_min)
    assert picks.cum_gain_min.iloc[-1] == pytest.approx(picks.gain_min.sum())
    assert picks.cum_gain_min.is_monotonic_increasing


def test_knockouts_hurt_and_segment_cut_is_positional():
    net = _line_network()
    model = metrics.build_cost_model(net.nodes, net.service_edges, min_pop=0, min_km=0)
    agg = ServiceAggregator(net)
    # Segment B-C lies strictly inside the long trips: pairs (i<=1<j) are cut, A-B and C-D are not.
    rows = agg.segment_rows("B", "C")
    cut = agg.pairs.iloc[rows]
    assert set(zip(cut.i, cut.j)) == {(0, 2), (0, 3), (1, 2), (1, 3)}
    routes = route_importance(net, model, agg).set_index("route_id")
    assert routes.loc["LONG", "delta_mean_min"] > routes.loc["SHUT", "delta_mean_min"] >= 0
    # Cancelling only the shuttle strands nobody (the long line still serves C-D) ...
    assert routes.loc["SHUT", "demand_disconnected"] == 0
    # ... but closing the C-D track cuts every train over it, the shuttle included.
    assert set(agg.pairs.iloc[agg.segment_rows("C", "D")].route_id) == {"LONG", "SHUT"}
    corr = corridor_importance(net, model, agg).set_index("segment")
    assert (corr.demand_disconnected > 0).all() and (corr.delta_mean_min > 0).all()
