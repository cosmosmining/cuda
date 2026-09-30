import zipfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from eurail import gtfs


def test_parse_gtfs_time_handles_after_midnight_and_blanks():
    t = gtfs.parse_gtfs_time(pd.Series(["08:00:00", "25:10:30", "", " 7:05:00"]))
    assert t[0] == 480 and t[1] == pytest.approx(1510.5) and np.isnan(t[2]) and t[3] == 425


def test_service_days_apply_calendar_dates(tiny_feed):
    feed = gtfs.read_feed(tiny_feed)
    days = gtfs.service_days_in_week(feed, date(2026, 1, 5))
    assert days["WD"] == 5  # 5 weekdays - Wednesday + Saturday
    assert gtfs.service_days_in_week(feed, date(2025, 1, 6)).empty  # before the calendar starts


def test_read_feed_from_zip_with_nested_folder(tiny_feed, tmp_path):
    z = tmp_path / "nested.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in tiny_feed.iterdir():
            zf.write(f, f"export/{f.name}")
    feed = gtfs.read_feed(z, "nested")
    assert feed.name == "nested" and len(feed.stop_times) == 8


def test_filter_routes_drops_non_rail(tiny_feed):
    feed = gtfs.filter_routes(gtfs.read_feed(tiny_feed))
    assert set(feed.routes.route_id) == {"R1"}
    assert set(feed.trips.trip_id) == {"t1", "t2"}
    assert set(gtfs.filter_routes(gtfs.read_feed(tiny_feed), None, r"^B").routes.route_id) == {"BUS"}


def test_trip_stop_table_maps_platforms_and_interpolates(tiny_feed):
    feed = gtfs.filter_routes(gtfs.read_feed(tiny_feed, "f"))
    ts = gtfs.trip_stop_table(feed, date(2026, 1, 5))
    assert set(ts.station_id) == {"f:PA", "f:B", "f:C"}  # platform folded into its parent
    assert (ts.weekly == 5).all()
    t2 = ts[ts.trip_id == "f:t2"].sort_values("seq")
    assert t2.arr.tolist() == [480, 530, 580]  # blank time interpolated linearly
    t1 = ts[ts.trip_id == "f:t1"].sort_values("seq")
    assert t1.arr.iloc[-1] == 24 * 60 + 40


def test_choose_reference_week_skips_thin_weeks(tiny_feed):
    feed = gtfs.read_feed(tiny_feed)
    # First week has 5 days (one removed, one added); later weeks have 5 as well -> earliest wins.
    assert gtfs.choose_reference_week(feed) == date(2026, 1, 5)
