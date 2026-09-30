from pathlib import Path

import pytest

from eurail import gtfs
from eurail.build import FeedSpec, build_from_feeds
from eurail.demo import generate_demo_feed, parse_lines

ROOT = Path(__file__).resolve().parents[1]


def test_parse_lines_rejects_non_increasing_times(tmp_path):
    f = tmp_path / "lines.txt"
    f.write_text("X1;Op;IC;4;;A 0 | B 30 | C 31\n")
    with pytest.raises(ValueError):
        parse_lines(f)


def test_generated_feed_is_valid_gtfs_with_mirrored_times(tmp_path):
    lines = tmp_path / "lines.txt"
    lines.write_text("X1;Op;HSR;3;;Paris Gare de Lyon 0 | Lyon Part-Dieu 116 | Marseille Saint-Charles 200\n"
                     "X2;Op;NIGHT;1;mon,wed;Paris Gare de Lyon 0 | Marseille Saint-Charles 500\n")
    out = tmp_path / "feed.zip"
    info = generate_demo_feed(out, lines, ROOT / "data/demo/stations.csv")
    assert info["trips"] == 2 * (3 + 3) + 2 * 1  # weekday + weekend per direction, plus the night train
    feed = gtfs.read_feed(out)
    ts = gtfs.trip_stop_table(feed)
    for _, t in ts.groupby("trip_id"):
        t = t.sort_values("seq")
        assert (t.arr.diff().dropna() > 0).all() and (t.dep >= t.arr).all()
    weekly = ts.drop_duplicates("trip_id").groupby(ts.drop_duplicates("trip_id").trip_id.str.contains("X2")).weekly.sum()
    assert weekly[True] == 4  # two directions x two nights


@pytest.mark.skipif(not (ROOT / "data/demo/eu_intercity_demo_gtfs.zip").exists(), reason="demo feed not generated")
def test_demo_network_merges_city_stations():
    res = build_from_feeds([FeedSpec("demo", ROOT / "data/demo/eu_intercity_demo_gtfs.zip")])
    nodes = res.network.nodes.set_index("name")
    assert nodes.loc["Paris", "n_stations"] == 6
    assert nodes.loc["London", "n_stations"] == 7
    assert nodes.loc["Schiphol Airport", "kind"] == "station"
    assert len(res.network.service_edges) > 2500
