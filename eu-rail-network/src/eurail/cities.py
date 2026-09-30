"""Merge railway stations into city nodes.

A station is assigned to the most populous GeoNames city (population >= 15k)
whose population-dependent radius covers it. The radius grows with city size,
so all Paris termini (2-4 km from the centre) become one "Paris" node, while
out-of-town high-speed stations such as Marne-la-Vallee or Frankfurt Airport
stay separate nodes, as they are in reality.

Stations that fall inside no city radius are kept as stand-alone station nodes
(stations of different feeds within ``standalone_merge_km`` are merged).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geo import EARTH_RADIUS_KM, haversine_km

EUROPE_BBOX = (-25.0, 34.0, 45.0, 72.0)  # lon_min, lat_min, lon_max, lat_max
MAX_RADIUS_KM = 15.0
MIN_RADIUS_KM = 3.0
# Airport stations are distinct destinations; never fold them into a city.
AIRPORT_REGEX = r"(?i)airport|flughafen|a[eé]roport|lufthavn|aeroporto|aeropuerto|lotnisko|leti[sš]t[eě]|lentoasema"


def merge_radius_km(population) -> np.ndarray:
    """Catchment radius of a city: 4.5 km * (pop / 100k) ** 0.4, clipped to [3, 15] km.

    Paris (2.1M) -> 15 km, Frankfurt (0.65M) -> 9.5 km, Lille (0.23M) -> 6.6 km.
    """
    pop = np.asarray(population, dtype=float)
    return np.clip(4.5 * (pop / 1e5) ** 0.4, MIN_RADIUS_KM, MAX_RADIUS_KM)


@lru_cache(maxsize=4)
def load_cities(min_population: int = 15000, bbox: tuple = EUROPE_BBOX) -> pd.DataFrame:
    """GeoNames cities (bundled with the ``geonamescache`` package, so no download)."""
    import geonamescache

    raw = geonamescache.GeonamesCache(min_city_population=min_population).get_cities()
    df = pd.DataFrame.from_records(
        [
            (int(c["geonameid"]), c["name"], c["countrycode"], float(c["latitude"]), float(c["longitude"]), int(c["population"]))
            for c in raw.values()
        ],
        columns=["geonameid", "name", "country", "lat", "lon", "population"],
    )
    lo_x, lo_y, hi_x, hi_y = bbox
    df = df[(df.lon >= lo_x) & (df.lon <= hi_x) & (df.lat >= lo_y) & (df.lat <= hi_y) & (df.population >= min_population)]
    return df.reset_index(drop=True)


def _unit_xyz(lat, lon) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def _chord(km: float) -> float:
    return 2 * np.sin(km / (2 * EARTH_RADIUS_KM))


def assign_stations(stations: pd.DataFrame, cities: pd.DataFrame, standalone_merge_km: float = 1.0,
                    keep_separate_regex: str | None = AIRPORT_REGEX) -> pd.DataFrame:
    """Add node_id / node_kind columns to ``stations`` (station_id, name, lat, lon)."""
    st = stations.reset_index(drop=True).copy()
    tree = cKDTree(_unit_xyz(cities.lat.to_numpy(), cities.lon.to_numpy()))
    xyz = _unit_xyz(st.lat.to_numpy(), st.lon.to_numpy())
    radius = merge_radius_km(cities.population.to_numpy())
    pop = cities.population.to_numpy()
    separate = st["name"].str.contains(keep_separate_regex, regex=True) if keep_separate_regex else pd.Series(False, index=st.index)

    node_city = np.full(len(st), -1)
    for i, cand in enumerate(tree.query_ball_point(xyz, _chord(MAX_RADIUS_KM))):
        if not cand or separate.iat[i]:
            continue
        cand = np.asarray(cand)
        d = haversine_km(st.lat.iat[i], st.lon.iat[i], cities.lat.to_numpy()[cand], cities.lon.to_numpy()[cand])
        ok = cand[d <= radius[cand]]
        if len(ok):
            node_city[i] = ok[np.argmax(pop[ok])]

    st["node_kind"] = np.where(node_city >= 0, "city", "station")
    st["node_id"] = [f"city:{cities.geonameid.iat[c]}" if c >= 0 else "" for c in node_city]
    st["city_row"] = node_city

    # Stand-alone stations: union stations of different feeds that are within
    # ``standalone_merge_km`` of each other (same physical station).
    lone = np.flatnonzero(node_city < 0)
    if len(lone):
        parent = {int(i): int(i) for i in lone}

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        lt = cKDTree(xyz[lone])
        for a, b in lt.query_pairs(_chord(standalone_merge_km)):
            ra, rb = find(int(lone[a])), find(int(lone[b]))
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        for i in lone:
            st.at[i, "node_id"] = "stn:" + st.station_id.iat[find(int(i))]
    return st


def build_nodes(assigned: pd.DataFrame, cities: pd.DataFrame, local_radius_km: float = 5.0) -> pd.DataFrame:
    """One row per node with name, coordinates, population and country.

    Stand-alone station nodes take their country from the nearest place in the
    denser GeoNames >=1000 list, and as population that of the nearest such place
    within ``local_radius_km`` (0 otherwise, e.g. remote airports or junctions).
    """
    rows = []
    places = load_cities(1000)
    nearest = cKDTree(_unit_xyz(places.lat.to_numpy(), places.lon.to_numpy()))
    for node_id, grp in assigned.groupby("node_id", sort=True):
        names = sorted(set(grp["name"]))
        if grp["node_kind"].iat[0] == "city":
            c = cities.iloc[int(grp["city_row"].iat[0])]
            rows.append((node_id, c["name"], "city", c["lat"], c["lon"], int(c["population"]), c["country"], len(grp), "; ".join(names)))
        else:
            lat, lon = grp["lat"].mean(), grp["lon"].mean()
            chord, j = nearest.query(_unit_xyz([lat], [lon])[0])
            p = places.iloc[int(j)]
            local_pop = int(p["population"]) if chord <= _chord(local_radius_km) else 0
            name = min(names, key=len)
            rows.append((node_id, name, "station", lat, lon, local_pop, p["country"], len(grp), "; ".join(names)))
    return pd.DataFrame(rows, columns=["node_id", "name", "kind", "lat", "lon", "population", "country", "n_stations", "stations"])
