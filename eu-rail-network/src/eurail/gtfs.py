"""GTFS ingestion.

Reads a static GTFS feed (zip or directory), keeps rail routes, and turns the
timetable into one row per (trip, stop) with the number of times that trip runs
in a reference week. Everything downstream (station merging, graph building)
works on that table, so any operator's feed can be dropped in.
"""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# GTFS route_type values that are rail. 2 is the basic "Rail" type; 100-117 are
# the extended (HVT) rail types used by many European feeds.
RAIL_ROUTE_TYPES = frozenset({2, *range(100, 118)})
# Extended types that denote intercity-style service. 2 is included because most
# feeds only use the basic type; restrict further with a name regex if needed.
INTERCITY_ROUTE_TYPES = frozenset({2, 100, 101, 102, 103, 105, 114})

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

_COLUMNS = {
    "stops": ["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type", "parent_station"],
    "routes": ["route_id", "agency_id", "route_short_name", "route_long_name", "route_type"],
    "trips": ["route_id", "service_id", "trip_id", "trip_short_name", "trip_headsign"],
    "stop_times": ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"],
    "calendar": ["service_id", *WEEKDAYS, "start_date", "end_date"],
    "calendar_dates": ["service_id", "date", "exception_type"],
}
_REQUIRED = ("stops", "routes", "trips", "stop_times")


@dataclass
class Feed:
    name: str
    stops: pd.DataFrame
    routes: pd.DataFrame
    trips: pd.DataFrame
    stop_times: pd.DataFrame
    calendar: pd.DataFrame
    calendar_dates: pd.DataFrame


def _read_table(raw: bytes, table: str) -> pd.DataFrame:
    wanted = _COLUMNS[table]
    df = pd.read_csv(
        io.BytesIO(raw),
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        usecols=lambda c: c.strip() in wanted,
    )
    df.columns = [c.strip() for c in df.columns]
    for col in wanted:
        if col not in df.columns:
            df[col] = ""
    return df[wanted].apply(lambda s: s.str.strip())


def read_feed(path: str | Path, name: str | None = None) -> Feed:
    """Load the tables we need from a GTFS zip file or an unpacked directory."""
    path = Path(path)
    name = name or path.stem
    blobs: dict[str, bytes] = {}
    if path.is_dir():
        for table in _COLUMNS:
            f = path / f"{table}.txt"
            if f.exists():
                blobs[table] = f.read_bytes()
    else:
        with zipfile.ZipFile(path) as zf:
            # Some publishers nest the files in a folder inside the zip.
            members = {Path(m).name: m for m in zf.namelist() if m.endswith(".txt")}
            for table in _COLUMNS:
                if f"{table}.txt" in members:
                    blobs[table] = zf.read(members[f"{table}.txt"])
    missing = [t for t in _REQUIRED if t not in blobs]
    if missing:
        raise ValueError(f"{path}: missing required GTFS tables {missing}")
    if "calendar" not in blobs and "calendar_dates" not in blobs:
        raise ValueError(f"{path}: needs calendar.txt and/or calendar_dates.txt")
    tables = {t: _read_table(blobs[t], t) if t in blobs else pd.DataFrame(columns=_COLUMNS[t]) for t in _COLUMNS}
    return Feed(name=name, **tables)


def parse_gtfs_time(s: pd.Series) -> np.ndarray:
    """'HH:MM:SS' -> minutes after midnight of the service day (hours may exceed 24)."""
    parts = s.fillna("").astype(str).str.strip().str.split(":", expand=True)
    if parts.shape[1] < 2:
        return np.full(len(s), np.nan)
    h = pd.to_numeric(parts[0], errors="coerce")
    m = pd.to_numeric(parts[1], errors="coerce")
    sec = pd.to_numeric(parts[2], errors="coerce").fillna(0) if parts.shape[1] > 2 else 0
    return (h * 60 + m + sec / 60).to_numpy(dtype=float)


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%Y%m%d", errors="coerce").dt.date


def service_date_range(feed: Feed) -> tuple[date, date]:
    """First and last date on which any service is defined."""
    lo, hi = [], []
    if len(feed.calendar):
        lo.append(_to_date(feed.calendar["start_date"]).min())
        hi.append(_to_date(feed.calendar["end_date"]).max())
    if len(feed.calendar_dates):
        d = _to_date(feed.calendar_dates["date"])
        lo.append(d.min())
        hi.append(d.max())
    lo = [d for d in lo if pd.notna(d)]
    hi = [d for d in hi if pd.notna(d)]
    if not lo:
        raise ValueError(f"{feed.name}: no usable calendar dates")
    return min(lo), max(hi)


def service_days_in_week(feed: Feed, week_start: date) -> pd.Series:
    """Number of days in [week_start, week_start + 6] on which each service_id runs."""
    days = [week_start + timedelta(days=i) for i in range(7)]
    active: set[tuple[str, date]] = set()
    cal = feed.calendar
    if len(cal):
        start = _to_date(cal["start_date"])
        end = _to_date(cal["end_date"])
        for d in days:
            flag = cal[WEEKDAYS[d.weekday()]] == "1"
            on = flag & (start <= d) & (end >= d)
            active.update((sid, d) for sid in cal.loc[on, "service_id"])
    cd = feed.calendar_dates
    if len(cd):
        dd = _to_date(cd["date"])
        in_week = dd.isin(days)
        for sid, d, exc in zip(cd.loc[in_week, "service_id"], dd[in_week], cd.loc[in_week, "exception_type"]):
            if exc == "1":
                active.add((sid, d))
            elif exc == "2":
                active.discard((sid, d))
    counts = pd.Series([sid for sid, _ in active], dtype=str).value_counts()
    counts.index.name = "service_id"
    return counts.rename("days")


def choose_reference_week(feed: Feed, max_weeks: int = 8) -> date:
    """Monday of the busiest week among the first ``max_weeks`` full weeks of the feed.

    Taking the busiest week avoids holiday weeks with thinned-out service.
    """
    lo, hi = service_date_range(feed)
    monday = lo + timedelta(days=(7 - lo.weekday()) % 7)
    trips_per_service = feed.trips["service_id"].value_counts()
    best, best_score = monday, -1
    for k in range(max_weeks):
        wk = monday + timedelta(weeks=k)
        if wk + timedelta(days=6) > hi and k > 0:
            break
        days = service_days_in_week(feed, wk)
        score = int((days * trips_per_service.reindex(days.index).fillna(0)).sum())
        if score > best_score:
            best, best_score = wk, score
    return best


def filter_routes(
    feed: Feed,
    route_types: frozenset[int] | set[int] | None = INTERCITY_ROUTE_TYPES,
    name_regex: str | None = None,
) -> Feed:
    """Keep only routes of the given types whose short/long name matches ``name_regex``."""
    routes = feed.routes.copy()
    rtype = pd.to_numeric(routes["route_type"], errors="coerce")
    keep = pd.Series(True, index=routes.index)
    if route_types is not None:
        keep &= rtype.isin(list(route_types))
    if name_regex:
        pat = re.compile(name_regex)
        label = routes["route_short_name"] + " " + routes["route_long_name"]
        keep &= label.map(lambda x: bool(pat.search(x)))
    routes = routes[keep]
    trips = feed.trips[feed.trips["route_id"].isin(routes["route_id"])]
    stop_times = feed.stop_times[feed.stop_times["trip_id"].isin(trips["trip_id"])]
    return replace(feed, routes=routes, trips=trips, stop_times=stop_times)


def stations_table(feed: Feed) -> tuple[pd.DataFrame, pd.Series]:
    """Collapse platforms to their parent station.

    Returns (stations, stop_to_station) where station ids are namespaced with the
    feed name so several feeds can be combined.
    """
    stops = feed.stops.copy()
    ids = set(stops["stop_id"])
    parent = stops["parent_station"].where(stops["parent_station"].isin(ids) & (stops["parent_station"] != ""), stops["stop_id"])
    stop_to_station = pd.Series((feed.name + ":" + parent).to_numpy(), index=stops["stop_id"].to_numpy())
    st = stops.set_index("stop_id")
    lat = pd.to_numeric(st["stop_lat"], errors="coerce")
    lon = pd.to_numeric(st["stop_lon"], errors="coerce")
    parent_ids = parent.to_numpy()
    stations = pd.DataFrame({
        "station_id": feed.name + ":" + pd.Series(parent_ids),
        "name": st["stop_name"].reindex(parent_ids).to_numpy(),
        # parent coordinates when present, otherwise the child's
        "lat": lat.reindex(parent_ids).to_numpy(),
        "lon": lon.reindex(parent_ids).to_numpy(),
        "child_lat": lat.to_numpy(),
        "child_lon": lon.to_numpy(),
    })
    stations["lat"] = stations["lat"].fillna(stations["child_lat"])
    stations["lon"] = stations["lon"].fillna(stations["child_lon"])
    stations = (
        stations.groupby("station_id", as_index=False)
        .agg(name=("name", "first"), lat=("lat", "mean"), lon=("lon", "mean"))
        .dropna(subset=["lat", "lon"])
    )
    stations["feed"] = feed.name
    return stations, stop_to_station


def trip_stop_table(feed: Feed, week_start: date | None = None) -> pd.DataFrame:
    """One row per (trip, stop) with times in minutes and weekly run count.

    Columns: feed, trip_id, route_id, route_name, seq, station_id, arr, dep, weekly.
    Trips that do not run in the reference week are dropped.
    """
    week_start = week_start or choose_reference_week(feed)
    days = service_days_in_week(feed, week_start)
    trips = feed.trips.merge(feed.routes[["route_id", "route_short_name", "route_long_name"]], on="route_id", how="left")
    trips["weekly"] = trips["service_id"].map(days).fillna(0).astype(int)
    trips = trips[trips["weekly"] > 0]
    trips["route_name"] = trips["route_short_name"].where(trips["route_short_name"] != "", trips["route_long_name"])

    st = feed.stop_times[feed.stop_times["trip_id"].isin(trips["trip_id"])].copy()
    _, stop_to_station = stations_table(feed)
    st["station_id"] = st["stop_id"].map(stop_to_station)
    st["seq"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
    arr = parse_gtfs_time(st["arrival_time"])
    dep = parse_gtfs_time(st["departure_time"])
    # GTFS allows either time to be blank on one side; fill from the other.
    st["arr"] = np.where(np.isnan(arr), dep, arr)
    st["dep"] = np.where(np.isnan(dep), arr, dep)
    st = st.dropna(subset=["station_id", "seq"])
    st = st.merge(trips[["trip_id", "route_id", "route_name", "weekly"]], on="trip_id")
    st["feed"] = feed.name
    st["trip_id"] = feed.name + ":" + st["trip_id"]
    st["route_id"] = feed.name + ":" + st["route_id"]
    st = st.sort_values(["trip_id", "seq"])
    # Interpolate missing intermediate times (untimed stops) linearly within a trip.
    if st["arr"].isna().any():
        st[["arr", "dep"]] = st.groupby("trip_id")[["arr", "dep"]].transform(lambda s: s.interpolate(limit_area="inside"))
        st = st.dropna(subset=["arr", "dep"])
    return st[["feed", "trip_id", "route_id", "route_name", "seq", "station_id", "arr", "dep", "weekly"]].reset_index(drop=True)
