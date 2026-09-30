import pandas as pd
import pytest

from eurail.graph import aggregate_edges, build_network, collapse_to_nodes, drop_cross_feed_duplicates, trip_pairs

NODES = pd.DataFrame({"node_id": list("ABCD"), "name": list("ABCD"), "kind": "city", "lat": [50, 50, 50, 50],
                      "lon": [0, 1, 2, 3], "population": 1, "country": "XX", "n_stations": 1, "stations": ""})


def _stops(rows):
    return pd.DataFrame(rows, columns=["feed", "trip_id", "route_id", "route_name", "seq", "node_id", "arr", "dep", "weekly"])


def test_collapse_merges_consecutive_calls_in_same_city():
    ts = _stops([("f", "t", "r", "R", 1, "A", 0, 0, 7), ("f", "t", "r", "R", 2, "A", 10, 12, 7),
                 ("f", "t", "r", "R", 3, "B", 60, 62, 7)])
    seq = collapse_to_nodes(ts)
    assert seq.node_id.tolist() == ["A", "B"]
    assert seq.dep.iloc[0] == 12  # last departure from the merged city


def test_service_graph_is_clique_per_trip_segment_graph_is_path():
    ts = _stops([("f", "t", "r", "R", i, n, a, a + 2, 5) for i, (n, a) in enumerate([("A", 0), ("B", 60), ("C", 120)])])
    net = build_network(ts, NODES)
    assert set(zip(net.service_edges.u, net.service_edges.v)) == {("A", "B"), ("A", "C"), ("B", "C")}
    assert set(zip(net.segment_edges.u, net.segment_edges.v)) == {("A", "B"), ("B", "C")}
    ac = net.service_edges.set_index(["u", "v"]).loc[("A", "C")]
    assert ac.time_median == 118 and ac.trains_per_week == 5  # arrival at C minus departure from A
    assert "D" not in set(net.nodes.node_id)  # unused nodes are dropped


def test_weighted_median_time_uses_weekly_counts():
    pairs = pd.DataFrame({"u": ["A", "B", "A"], "v": ["B", "A", "B"], "time": [60, 70, 300], "weekly": [11, 9, 1]})
    e = aggregate_edges(pairs, NODES)
    assert e.trains_per_week.iloc[0] == 21 and e.time_median.iloc[0] == 60 and e.time_min.iloc[0] == 60


def test_cross_feed_duplicate_trains_are_counted_once():
    rows = []
    for feed, w in (("de", 7), ("nl", 5)):
        rows += [(feed, f"{feed}:t", "r", "R", 1, "A", 0, 0, w), (feed, f"{feed}:t", "r", "R", 2, "B", 60, 60, w)]
    seq = drop_cross_feed_duplicates(collapse_to_nodes(_stops(rows)))
    assert seq.trip_id.unique().tolist() == ["de:t"]


def test_trip_pairs_positions():
    ts = _stops([("f", "t", "r", "R", i, n, 10 * i, 10 * i, 1) for i, n in enumerate("ABCD")])
    p = trip_pairs(collapse_to_nodes(ts))
    assert len(p) == 6 and set(zip(p.i, p.j)) == {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
    assert p.set_index(["u", "v"]).loc[("A", "D"), "time"] == pytest.approx(30)
