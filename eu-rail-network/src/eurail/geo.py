"""Small geographic helpers (great-circle distances)."""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; accepts scalars or broadcastable arrays."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def pairwise_km(lat, lon) -> np.ndarray:
    """Full n x n great-circle distance matrix."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    return haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])


class LandMask:
    """Rasterised land/sea mask from a GeoJSON of country polygons (for sea-crossing checks)."""

    def __init__(self, geojson_path, bbox=(-25.0, 34.0, 45.0, 72.0), res_deg: float = 0.1):
        import json

        from matplotlib.path import Path as MplPath

        self.x0, self.y0, x1, y1 = bbox
        self.res = res_deg
        nx_, ny_ = int(round((x1 - self.x0) / res_deg)), int(round((y1 - self.y0) / res_deg))
        xs = self.x0 + (np.arange(nx_) + 0.5) * res_deg
        ys = self.y0 + (np.arange(ny_) + 0.5) * res_deg
        self.mask = np.zeros((ny_, nx_), dtype=bool)
        with open(geojson_path) as fh:
            features = json.load(fh)["features"]
        for f in features:
            for poly in f["geometry"]["coordinates"]:
                ring = np.asarray(poly[0])
                ix = np.flatnonzero((xs >= ring[:, 0].min()) & (xs <= ring[:, 0].max()))
                iy = np.flatnonzero((ys >= ring[:, 1].min()) & (ys <= ring[:, 1].max()))
                if not len(ix) or not len(iy):
                    continue
                gx, gy = np.meshgrid(xs[ix], ys[iy])
                inside = MplPath(ring).contains_points(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gy.shape)
                self.mask[np.ix_(iy, ix)] |= inside

    def is_land(self, lat, lon) -> np.ndarray:
        i = np.clip(((np.asarray(lat) - self.y0) / self.res).astype(int), 0, self.mask.shape[0] - 1)
        j = np.clip(((np.asarray(lon) - self.x0) / self.res).astype(int), 0, self.mask.shape[1] - 1)
        return self.mask[i, j]

    def sea_km(self, lat1, lon1, lat2, lon2, step_km: float = 2.0) -> float:
        """Length of the straight line between two points that lies over sea."""
        d = float(haversine_km(lat1, lon1, lat2, lon2))
        n = max(int(d / step_km), 2)
        t = (np.arange(n) + 0.5) / n
        lat = lat1 + (lat2 - lat1) * t
        lon = lon1 + (lon2 - lon1) * t
        return d * float((~self.is_land(lat, lon)).mean())
