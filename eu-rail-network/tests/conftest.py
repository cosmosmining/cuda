import csv
from pathlib import Path

import pytest


def write_gtfs(root: Path, tables: dict[str, list[list]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        with open(root / f"{name}.txt", "w", newline="") as fh:
            csv.writer(fh).writerows(rows)
    return root


@pytest.fixture
def tiny_feed(tmp_path) -> Path:
    """Three stations on a line, one parent station with a platform, one bus route to be filtered out.

    Service WD runs Mon-Fri; calendar_dates removes Wed 2026-01-07 and adds Sat 2026-01-10,
    so in the week of 2026-01-05 it runs 5 days.
    """
    return write_gtfs(tmp_path / "feed", {
        "stops": [["stop_id", "stop_name", "stop_lat", "stop_lon", "location_type", "parent_station"],
                  ["PA", "Alpha Central", "50.0", "4.0", "1", ""],
                  ["PA1", "Alpha Central platform 1", "50.0001", "4.0001", "0", "PA"],
                  ["B", "Beta", "50.5", "5.0", "0", ""],
                  ["C", "Gamma", "51.0", "6.0", "0", ""]],
        "routes": [["route_id", "agency_id", "route_short_name", "route_long_name", "route_type"],
                   ["R1", "X", "IC 1", "Alpha - Gamma", "2"],
                   ["BUS", "X", "B9", "Bus", "3"]],
        "trips": [["route_id", "service_id", "trip_id"],
                  ["R1", "WD", "t1"], ["R1", "WD", "t2"], ["BUS", "WD", "b1"]],
        "stop_times": [["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"],
                       ["t1", "23:00:00", "23:00:00", "PA1", "1"],
                       ["t1", "23:50:00", "23:52:00", "B", "2"],
                       ["t1", "24:40:00", "24:40:00", "C", "3"],
                       ["t2", "08:00:00", "08:00:00", "C", "1"],
                       ["t2", "", "", "B", "2"],
                       ["t2", "09:40:00", "09:40:00", "PA1", "3"],
                       ["b1", "08:00:00", "08:00:00", "PA1", "1"],
                       ["b1", "09:00:00", "09:00:00", "C", "2"]],
        "calendar": [["service_id", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                      "start_date", "end_date"],
                     ["WD", "1", "1", "1", "1", "1", "0", "0", "20260105", "20261231"]],
        "calendar_dates": [["service_id", "date", "exception_type"],
                           ["WD", "20260107", "2"], ["WD", "20260110", "1"]],
    })
