import numpy as np
import pandas as pd

from eurail.cities import assign_stations, merge_radius_km


def _cities():
    return pd.DataFrame({
        "geonameid": [1, 2, 3], "name": ["Bigcity", "Suburb", "Smalltown"], "country": ["FR", "FR", "FR"],
        "lat": [48.85, 48.85, 49.40], "lon": [2.35, 2.50, 2.35], "population": [2_000_000, 50_000, 20_000],
    })


def _km_east(lon0, lat, km):
    return lon0 + km / (111.32 * np.cos(np.radians(lat)))


def test_merge_radius_is_bounded_and_monotone():
    r = merge_radius_km([1_000, 100_000, 650_000, 2_000_000, 10_000_000])
    assert r[0] == 3.0 and r[-1] == 15.0 and np.all(np.diff(r) >= 0)
    assert 9 < r[2] < 10


def test_assignment_prefers_largest_city_within_radius():
    c = _cities()
    st = pd.DataFrame({
        "station_id": ["f:a", "f:b", "f:c", "f:d", "g:e", "h:e2"],
        "name": ["Bigcity Nord", "Bigcity Est", "Far Junction", "Bigcity Airport", "Remote Halt", "Remote Halt"],
        "lat": [48.87, 48.85, 48.20, 48.85, 47.00, 47.003],
        "lon": [2.35, _km_east(2.35, 48.85, 12), 2.35, _km_east(2.35, 48.85, 5), 3.00, 3.001],
    })
    a = assign_stations(st, c).set_index("station_id")
    assert a.loc["f:a", "node_id"] == "city:1"
    assert a.loc["f:b", "node_id"] == "city:1"   # nearer to Suburb, but inside Bigcity's 15 km radius
    assert a.loc["f:c", "node_kind"] == "station"
    assert a.loc["f:d", "node_kind"] == "station"  # airports stay separate
    assert a.loc["g:e", "node_id"] == a.loc["h:e2", "node_id"]  # same station in two feeds
