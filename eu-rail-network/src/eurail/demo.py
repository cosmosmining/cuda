"""Generate the offline DEMO GTFS feed from ``data/demo/lines.txt``.

The demo network is approximate and hand-compiled (see the header of
lines.txt). It is emitted as a standards-compliant GTFS zip so it flows through
exactly the same ingestion code as an official feed.
"""
from __future__ import annotations

import csv
import io
import unicodedata
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .cities import EUROPE_BBOX

ROUTE_TYPE = {"HSR": 101, "IC": 102, "NIGHT": 105}
DAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
WEEKEND_FACTOR = 0.85
DWELL_MIN = 2
DAY_WINDOW = (6 * 60, 20 * 60 + 30)    # first/last departure for day trains
NIGHT_WINDOW = (19 * 60 + 30, 23 * 60)


@dataclass
class Line:
    id: str
    operator: str
    category: str
    per_day: int
    days: tuple[str, ...]
    stops: list[tuple[str, float]]


def normalise(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    for ch in "-'.()/":
        s = s.replace(ch, " ")
    return " ".join(s.split())


def parse_lines(path: str | Path) -> list[Line]:
    lines = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        lid, op, cat, per_day, days, stops = [p.strip() for p in raw.split(";", 5)]
        seq = []
        for tok in stops.split("|"):
            name, _, minutes = tok.strip().rpartition(" ")
            seq.append((name.strip(), float(minutes)))
        if any(b[1] <= a[1] + DWELL_MIN for a, b in zip(seq, seq[1:])):
            raise ValueError(f"{lid}: stop times must increase by more than the dwell time")
        day_set = tuple(d.strip() for d in days.split(",")) if days else ()
        if any(d not in DAY_NAMES for d in day_set):
            raise ValueError(f"{lid}: bad days field {days!r}")
        lines.append(Line(lid, op, cat, int(per_day), day_set, seq))
    return lines


def _gazetteer() -> tuple[dict, dict]:
    """Normalised name -> (pop, lat, lon, cc) for European populated places (pop >= 1000)."""
    import geonamescache

    lo_x, lo_y, hi_x, hi_y = EUROPE_BBOX
    by_name: dict[str, tuple] = {}
    by_alt: dict[str, tuple] = {}
    for c in geonamescache.GeonamesCache(min_city_population=1000).get_cities().values():
        lat, lon = float(c["latitude"]), float(c["longitude"])
        if not (lo_x <= lon <= hi_x and lo_y <= lat <= hi_y):
            continue
        rec = (int(c["population"]), lat, lon, c["countrycode"])
        k = normalise(c["name"])
        if k not in by_name or rec[0] > by_name[k][0]:
            by_name[k] = rec
        for alt in c.get("alternatenames") or []:
            k = normalise(alt)
            if k and (k not in by_alt or rec[0] > by_alt[k][0]):
                by_alt[k] = rec
    return by_name, by_alt


def resolve_stations(names: list[str], registry_path: str | Path) -> pd.DataFrame:
    reg = pd.read_csv(registry_path, comment="#").set_index("station")
    by_name, by_alt = _gazetteer()
    rows, missing = [], []
    for n in sorted(set(names)):
        if n in reg.index:
            r = reg.loc[n]
            rows.append((n, float(r.lat), float(r.lon), r.country, "registry"))
            continue
        hit = by_name.get(normalise(n)) or by_alt.get(normalise(n))
        if hit is None:
            missing.append(n)
            continue
        rows.append((n, hit[1], hit[2], hit[3], "geonames"))
    if missing:
        raise ValueError(f"cannot place stations {missing}; add them to {registry_path}")
    return pd.DataFrame(rows, columns=["station", "lat", "lon", "country", "source"])


def _hhmmss(minutes: float) -> str:
    m = int(round(minutes))
    return f"{m // 60:02d}:{m % 60:02d}:00"


def _departures(n: int, window: tuple[int, int]) -> np.ndarray:
    lo, hi = window
    return lo + (hi - lo) * (np.arange(n) + 0.5) / n


def generate_demo_feed(out_zip: str | Path, lines_path: str | Path, registry_path: str | Path,
                       start: date = date(2026, 1, 5), end: date = date(2026, 12, 12)) -> dict:
    lines = parse_lines(lines_path)
    stations = resolve_stations([s for ln in lines for s, _ in ln.stops], registry_path)
    stop_id = {s: f"S{i:04d}" for i, s in enumerate(stations.station)}

    services = {
        "WD": (1, 1, 1, 1, 1, 0, 0),
        "WE": (0, 0, 0, 0, 0, 1, 1),
        "ALL": (1,) * 7,
    }
    routes, trips, stop_times = [], [], []
    for ln in lines:
        routes.append((ln.id, ln.operator, ln.id, f"{ln.operator} {ln.category}", ROUTE_TYPE[ln.category]))
        if ln.days:
            sid = "D_" + "_".join(ln.days)
            services[sid] = tuple(int(d in ln.days) for d in DAY_NAMES)
            plan = [(sid, ln.per_day)]
        elif ln.category == "NIGHT":
            plan = [("ALL", ln.per_day)]
        else:
            plan = [("WD", ln.per_day), ("WE", max(1, round(ln.per_day * WEEKEND_FACTOR)))]
        window = NIGHT_WINDOW if ln.category == "NIGHT" else DAY_WINDOW

        arr_fwd = np.array([t for _, t in ln.stops])
        dep_fwd = arr_fwd + DWELL_MIN  # dwell at intermediate stops only
        dep_fwd[0] = arr_fwd[0] = 0.0
        dep_fwd[-1] = arr_fwd[-1]
        total = arr_fwd[-1]
        names_fwd = [s for s, _ in ln.stops]
        # Reverse direction mirrors the run times.
        arr_rev = (total - dep_fwd)[::-1]
        dep_rev = (total - arr_fwd)[::-1]
        for direction, names, arr, dep in ((0, names_fwd, arr_fwd, dep_fwd), (1, names_fwd[::-1], arr_rev, dep_rev)):
            for sid, n in plan:
                for k, t0 in enumerate(_departures(n, window)):
                    tid = f"{ln.id}_{direction}_{sid}_{k:02d}"
                    trips.append((ln.id, sid, tid, direction))
                    for seq, (s, a, d) in enumerate(zip(names, arr, dep), start=1):
                        stop_times.append((tid, _hhmmss(t0 + a), _hhmmss(t0 + d), stop_id[s], seq))

    def table(header, rows) -> bytes:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
        return buf.getvalue().encode("utf-8")

    agencies = sorted({ln.operator for ln in lines})
    ymd = lambda d: d.strftime("%Y%m%d")  # noqa: E731
    files = {
        "agency.txt": table(["agency_id", "agency_name", "agency_url", "agency_timezone"],
                            [(a, a, "https://example.org/demo", "Europe/Brussels") for a in agencies]),
        "stops.txt": table(["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type"],
                           [(stop_id[r.station], r.station, f"{r.lat:.5f}", f"{r.lon:.5f}", 0) for r in stations.itertuples()]),
        "routes.txt": table(["route_id", "agency_id", "route_short_name", "route_long_name", "route_type"], routes),
        "trips.txt": table(["route_id", "service_id", "trip_id", "direction_id"], trips),
        "stop_times.txt": table(["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"], stop_times),
        "calendar.txt": table(["service_id", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "start_date", "end_date"],
                              [(sid, *flags, ymd(start), ymd(end)) for sid, flags in services.items()]),
        # Public holidays run the weekend timetable (exercises calendar_dates).
        "calendar_dates.txt": table(["service_id", "date", "exception_type"],
                                    [(s, d, e) for d in ("20260406", "20260501", "20260525") for s, e in (("WD", 2), ("WE", 1))]),
        "feed_info.txt": table(["feed_publisher_name", "feed_publisher_url", "feed_lang", "feed_version"],
                               [("eurail DEMO - approximate, hand-compiled, not an official timetable", "https://example.org/demo", "en", "demo-1")]),
    }
    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, blob in files.items():
            zf.writestr(name, blob)
    return {"lines": len(lines), "stations": len(stations), "trips": len(trips), "stop_times": len(stop_times),
            "resolved": stations}
